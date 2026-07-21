"""Independent artifact-root recomputation for WM-001.

This module intentionally does not call the experiment's metric, manifest, or
checkpoint decoders.  It treats ``result.json`` and every referenced sidecar as
untrusted input, applies explicit byte limits, decodes the safe formats again,
and recomputes the claims that the current artifact actually makes auditable.

The audit distinguishes three outcomes:

``integrity_passed``
    Every check possible from the current artifact passed.

``engineering_complete``
    The producer supplied enough evidence to independently bind predictions,
    targets, and controller actions to their claimed causes.

``complete_for_claim``
    The evidence is engineering-complete and came from the formal lane.

For a normal audit, ``passed`` requires integrity and engineering completeness.
Development can therefore pass its rehearsal audit without ever becoming
claim-eligible.  This prevents a clean rehearsal from being mistaken for
confirmatory evidence of learning and retention.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import numpy.typing as npt

# Audit-local CUDA replay must request deterministic cuBLAS before its first
# Torch matrix multiplication.  The producer launcher makes the same request,
# but an external auditor cannot assume the caller exported it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
RESULT_SCHEMA_PATH = HERE / "schemas" / "raw-result.schema.json"
_AUDITOR_SOURCE_INVOCATION = __file__
_AUDITOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ASSURANCE = {
    "trust_model_id": "prospect.wm001.trust-model.v1",
    "tamper_resistant": False,
    "external_attestation": False,
    "exclusive_path_use_required": True,
}
_OUTER_COMPLETIONS_ROOT = Path.cwd() / "bench" / "world_model_lifecycle" / "results" / "outer-completions" / "v1.16"
_FORMAL_CONFIRMATION_NAME = "confirmation-v1.16.0"

_MAGIC = b"PROSPECT-WM001\0"
_PREDICTION_FORMAT = "prospect.wm001.predictive-evidence.v1"
_SAMPLING_FORMAT = "prospect.wm001.bootstrap-manifest.v1"
_MODEL_FORMAT = "prospect.wm001.probabilistic-ensemble.v1"
_OWNED_MODEL_MAGIC = b"PROSPECT-WM001-STATE\0"
_OWNED_MODEL_FORMAT = "prospect.wm001.owned-model-state.v1"
_CHECKPOINT_SCHEMA = "prospect.world-model-lifecycle.checkpoint.v1"
_CHECKPOINT_BOUNDARY = "episode_complete"
_CHECKPOINT_MANIFEST = "manifest.json"
_SHA256_EMPTY = hashlib.sha256(b"").hexdigest()
_TARGET_REWARD_SCALE = 16.2736044
_PENDULUM_OBSERVATION_ATOL = 2e-6
_PENDULUM_REWARD_ATOL = 2e-5
_PENDULUM_MAX_SPEED = 8.0
_PENDULUM_MAX_TORQUE = 2.0
_PENDULUM_TIME_STEP = 0.05
_OSCILLATOR_SOURCE = "prospect:IndependentPhaseOscillator-v1"
_OSCILLATOR_TIME_STEP = 0.05
_OSCILLATOR_OBSERVATION_ATOL = 1e-12
_OSCILLATOR_REWARD_ATOL = 1e-12
_FORMAL_OSCILLATOR_CONFORMANCE_CASES = 512
_FORMAL_OSCILLATOR_CONFORMANCE_SEED = 20260718
_CEM_PLANNING_HORIZON = 10
_CEM_OPTIM_STEPS = 3
_CEM_NUM_CANDIDATES = 64
_CEM_TOP_K = 8
_SEED_HASH_DOMAIN_VERSION = "1.16.0"
_RESTART_RUNTIME_CONFORMANCE_SCHEMA = (
    "prospect.wm001.restart-runtime-conformance.v1"
)
_RESTART_RUNTIME_SUPPORT_FILES = (
    "producer_bootstrap.py",
    "protocol.json",
    "schemas/raw-result.schema.json",
)
_EPISODE_HORIZON = 200
_GRAPH_SCHEMA = "prospect.wm001.domain-graph.v1"
_MAX_GRAPH_NODES = 250_000
_MAX_OBSERVATION_SEQUENCES = 20_000
_MAX_OBSERVATION_SEQUENCE_LENGTH = 512
_MAX_GRAPH_VALUES = 5_000_000
_MAX_GRAPH_DEPTH = 64
_MAX_GRAPH_ROOTS = 16

_GRAPH_RECORD_TYPES = frozenset(
    {
        "Action",
        "AgentSnapshot",
        "Belief",
        "BeliefUpdate",
        "CandidateAssessment",
        "DecisionRecord",
        "Distribution",
        "EpistemicEffect",
        "EpistemicTarget",
        "EpistemicTransition",
        "Evidence",
        "EvidenceLineage",
        "ExecutedAction",
        "ExperienceEvent",
        "Goal",
        "InformationSet",
        "InformationValue",
        "IntendedAction",
        "Observation",
        "Outcome",
        "Prediction",
        "ProperScore",
        "Provenance",
        "ResourceLedger",
        "ResourceUse",
        "TimePoint",
        "UncertaintyEstimate",
        "UpdateReceipt",
        "Utility",
    }
)
_GRAPH_RECORD_FIELDS: Mapping[str, frozenset[str]] = {
    "TimePoint": frozenset({"tick", "clock_id"}),
    "Provenance": frozenset({"source_id", "trust", "source_kind", "detail"}),
    "EvidenceLineage": frozenset(
        {
            "evidence_id",
            "origin",
            "provenance",
            "parent_evidence_ids",
            "producer_version",
        }
    ),
    "Evidence": frozenset({"evidence_id", "payload", "occurred_at", "available_at", "lineage"}),
    "Observation": frozenset({"observation_id", "agent_id", "modality", "evidence"}),
    "InformationSet": frozenset(
        {
            "information_set_id",
            "agent_id",
            "as_of",
            "observations",
            "memory_version",
        }
    ),
    "EpistemicTarget": frozenset({"target_id", "description", "target_kind"}),
    "Distribution": frozenset(
        {
            "distribution_id",
            "family",
            "support",
            "parameters",
            "representation_version",
            "event_shape",
        }
    ),
    "Belief": frozenset(
        {
            "belief_id",
            "agent_id",
            "target",
            "information_set",
            "distribution",
            "formed_at",
            "model_version",
            "representation_version",
        }
    ),
    "Action": frozenset({"action_id", "action_kind", "parameters"}),
    "UncertaintyEstimate": frozenset(
        {
            "estimate_id",
            "kind",
            "measure",
            "value",
            "unit",
            "target_id",
            "estimator_version",
            "assessed_at",
            "calibration_version",
        }
    ),
    "IntendedAction": frozenset({"intention_id", "agent_id", "action", "intended_at"}),
    "ExecutedAction": frozenset(
        {
            "execution_id",
            "intention",
            "status",
            "started_at",
            "ended_at",
            "realized_action",
            "deviation_reason",
        }
    ),
    "Prediction": frozenset(
        {
            "prediction_id",
            "prior_belief",
            "action",
            "target",
            "distribution",
            "issued_at",
            "horizon_end",
            "model_version",
            "representation_version",
            "calibration_version",
            "uncertainties",
        }
    ),
    "Goal": frozenset(
        {
            "goal_id",
            "task_id",
            "target",
            "description",
            "issued_at",
            "preference_version",
            "deadline",
        }
    ),
    "Utility": frozenset(
        {
            "utility_id",
            "goal_id",
            "prediction_id",
            "expected_value",
            "unit",
            "evaluator_version",
            "assessed_at",
        }
    ),
    "InformationValue": frozenset(
        {
            "information_value_id",
            "prior_belief_id",
            "action_id",
            "target_id",
            "expected_reduction",
            "expected_cost",
            "unit",
            "evaluator_version",
            "assessed_at",
        }
    ),
    "CandidateAssessment": frozenset(
        {
            "assessment_id",
            "action",
            "prediction",
            "utility",
            "information_value",
            "expected_action_cost",
            "expected_risk",
            "admissible",
            "constraint_reasons",
            "constraint_penalty",
            "total_value",
            "unit",
            "evaluator_version",
            "assessed_at",
        }
    ),
    "DecisionRecord": frozenset(
        {
            "decision_id",
            "agent_id",
            "belief",
            "goal",
            "intended_action",
            "alternatives",
            "selected_assessment",
            "policy_version",
            "decided_at",
        }
    ),
    "Outcome": frozenset({"outcome_id", "evidence", "execution_id"}),
    "ExperienceEvent": frozenset(
        {
            "experience_id",
            "agent_id",
            "run_id",
            "task_id",
            "episode_id",
            "step_index",
            "kind",
            "observation",
            "outcome",
            "terminated",
            "truncated",
            "discount",
            "behavior_policy_version",
            "closed_at",
            "decision",
            "execution",
        }
    ),
    "BeliefUpdate": frozenset(
        {
            "update_id",
            "prior",
            "experience",
            "posterior",
            "updater_version",
            "updated_at",
        }
    ),
    "ProperScore": frozenset(
        {
            "score_id",
            "prediction_id",
            "realized_evidence_id",
            "rule",
            "value",
            "unit",
            "scorer_version",
            "scored_at",
        }
    ),
    "EpistemicEffect": frozenset(
        {
            "effect_id",
            "belief_update_id",
            "target_id",
            "kind",
            "measure",
            "before",
            "after",
            "improvement",
            "higher_is_better",
            "evaluator_version",
            "evaluated_at",
            "externally_calibrated",
        }
    ),
    "EpistemicTransition": frozenset(
        {
            "transition_id",
            "experience",
            "belief_update",
            "proper_scores",
            "effects",
            "created_at",
        }
    ),
    "UpdateReceipt": frozenset(
        {
            "receipt_id",
            "agent_id",
            "transitions",
            "learner_version",
            "status",
            "previous_configuration_version",
            "new_configuration_version",
            "previous_model_version",
            "new_model_version",
            "previous_representation_version",
            "new_representation_version",
            "previous_policy_version",
            "new_policy_version",
            "started_at",
            "completed_at",
            "resulting_belief",
            "rollback_of",
            "metrics",
        }
    ),
    "ResourceUse": frozenset({"resource", "amount", "unit"}),
    "ResourceLedger": frozenset({"ledger_id", "started_at", "completed_at", "uses"}),
    "AgentSnapshot": frozenset(
        {
            "snapshot_id",
            "agent_id",
            "captured_at",
            "belief",
            "configuration_version",
            "memory_version",
            "knowledge_version",
            "model_version",
            "representation_version",
            "policy_version",
            "resources",
            "pending_intentions",
            "latest_update",
        }
    ),
}
_GRAPH_ENUM_TYPES = frozenset(
    {
        "EpistemicEffectKind",
        "EvidenceOrigin",
        "ExecutionStatus",
        "ExperienceKind",
        "TrustLevel",
        "UncertaintyKind",
        "UpdateStatus",
    }
)

# A formal result retains roughly 800k transition rows inline.  Four GiB is a
# hard safety ceiling, not a target; moving transition rows to independently
# hashed columnar sidecars is the natural follow-up if WM-001 grows.
_MAX_RESULT_BYTES = 4 << 30
_MAX_PREDICTION_BYTES = 8 << 20
_MAX_OWNED_MODEL_BYTES = 256 << 20
_MAX_MANIFEST_BYTES = 64 << 20
_MAX_PERMUTATION_BYTES = 64 << 20
_MAX_REJECTED_STATE_BYTES = 512 << 20
_MAX_RESTART_EVALUATION_BYTES = 16 << 20
_MAX_CHECKPOINT_BYTES = 4 << 30
_MAX_CONTAINER_HEADER_BYTES = 1 << 20
_MAX_CHECKPOINT_MANIFEST_BYTES = 1 << 20
_MAX_CHECKPOINT_COMPONENT_BYTES = 1 << 30
_MAX_CHECKPOINT_TOTAL_BYTES = 4 << 30
_MAX_SOURCE_FILE_BYTES = 64 << 20
_MAX_SOURCE_SNAPSHOT_BYTES = 512 << 20
_MAX_BOUND_PACKAGES = 512
_MAX_PRODUCER_MANIFEST_BYTES = 64 << 20
_MAX_PRODUCER_FILE_BYTES = 4 << 30
_MAX_PRODUCER_TOTAL_BYTES = 16 << 30
_MAX_PRODUCER_FILES = 100_000
_MAX_PRODUCER_TREE_ENTRIES = 200_000
_MAX_DEVELOPMENT_QUALIFICATION_ARCHIVE_BYTES = 40 << 30
_MAX_RETAINED_DEVELOPMENT_ARCHIVE_MEMBER_BYTES = 64 << 20
_MAX_RETAINED_DEVELOPMENT_ARCHIVE_TOTAL_BYTES = 256 << 20
_PRODUCER_CUSTODY_INDEPENDENCE_LIMITATION = (
    "The producer manifest is fully reopened and hash-checked, but it is not "
    "externally signed or transparency-log anchored; filesystem-level "
    "replacement of the entire root and manifest is outside this audit's "
    "threat model."
)
_DEVELOPMENT_EVIDENCE_INDEPENDENCE_LIMITATION = (
    "This is development evidence, not formal claim evidence. Protocol 1.4 "
    "makes K3-K6 development outcomes descriptive and permanently ineligible "
    "for capability adjudication."
)
_PENDULUM_RECONSTRUCTION_INDEPENDENCE_LIMITATION = (
    "Pendulum reset observations are reconstructed from the documented "
    "Gymnasium 0.29.1 default_rng/PCG64 algorithm without importing "
    "Gymnasium; exact replay consequently relies on NumPy's bound Generator "
    "semantics. The independent oscillator is reconstructed directly from "
    "its SHA-256 reset and autonomous analytic dynamics."
)
_OPTIMIZER_RNG_INDEPENDENCE_LIMITATION = (
    "Bootstrap, balanced-minibatch, minibatch-order, and "
    "corruption-permutation bytes are regenerated without producer sampling "
    "code, but exact replay uses the same dependency-bound Torch CPU RNG "
    "primitives rather than a separately implemented Philox engine."
)
_CEM_REPLAY_INDEPENDENCE_LIMITATION = (
    "CEM is independently reimplemented from the public algorithm and "
    "retained tensor bytes rather than calling producer planning/model code, "
    "but bit-exact action replay intentionally uses the same bound Torch "
    "device kernels and RNG implementation as the producer."
)
_MAX_PREBINDING_REQUEST_BYTES = 16 << 20
_MAX_PREBINDING_ROOT_ENTRIES = 250_000
_MAX_PREBINDING_ROOT_BYTES = 32 << 30
_PREBINDING_REQUEST_SCHEMA = "prospect.wm001.prebinding-conformance-request.v2"
_PREBINDING_REPORT_SCHEMA = "prospect.wm001.prebinding-conformance.v2"
_PREBINDING_PACKAGE_ROOT_DOMAIN = b"prospect.wm001.package-root.v2\0"
_PREBINDING_STDLIB_DOMAIN = b"prospect.wm001.standard-library.v2\0"
_PREBINDING_DISTRIBUTION_DOMAIN = b"prospect.wm001.distribution.v2\0"
_PREBINDING_SCIENTIFIC_BLOCKS = (
    "claim",
    "null_hypothesis",
    "scope",
    "causal_chain",
    "environment",
    "tasks",
    "splits",
    "representation_and_model",
    "experience_and_learning",
    "budgets",
    "controls",
    "metrics",
    "thresholds",
    "evaluation_scheduling_rule",
    "execution_sequence",
    "killing_order",
    "sources",
)
_PREBINDING_SCIENTIFIC_SOURCES = (
    "learning.py",
    "model.py",
    "planning.py",
    "runtime_lane.py",
)
_PRODUCER_MANIFEST_NAME = "producer-manifest.json"
_PRODUCER_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "lane",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "error",
        "manifest_excludes",
        "file_count",
        "files",
    }
)
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
_FORMAL_CONFORMANCE_KEYS = frozenset(
    {
        "schema",
        "environment_id",
        "gymnasium_version",
        "seed",
        "samples_per_task",
        "cases",
        "semantic_parameters",
        "semantic_parameter_absolute_errors",
        "spec_horizon",
        "max_observation_absolute_error",
        "max_reward_absolute_error",
        "planner_dtype",
        "max_planner_observation_absolute_error",
        "max_planner_reward_absolute_error",
        "terminated_or_truncated_cases",
        "observation_atol",
        "reward_atol",
        "planner_observation_atol",
        "planner_reward_atol",
        "passed",
        "report_sha256",
    }
)
_FORMAL_CONFORMANCE_PARAMETERS = {
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
    "dt": 0.05,
    "max_speed": 8.0,
    "max_torque": 2.0,
}
_FORMAL_CONFORMANCE_TOLERANCES = {
    "observation_atol": 2e-6,
    "reward_atol": 1e-9,
    "planner_observation_atol": 2e-6,
    "planner_reward_atol": 2e-5,
}

_TASK_A = "pendulum_normal_torque"
_TASK_B = "pendulum_reversed_torque"
_TASK_IRRELEVANT = "independent_phase_oscillator"
_TASK_CONTEXT = {
    _TASK_A: 0.0,
    _TASK_B: 1.0,
    _TASK_IRRELEVANT: 2.0,
}
_FORMAL_SEEDS = (
    721_000_968,
    1_733_386_057,
    1_129_257_495,
    1_461_304_433,
    345_413_014,
    76_587_833,
    404_195_464,
    3_550_251_066,
)
_DEVELOPMENT_SEEDS = (3_922_749_719, 1_847_570_536)
_COVERAGE_SEMANTICS = "wm001-mixture-pit-binary64-count-v1"
_V100_MASTER_SEEDS = (
    101,
    211,
    104_729,
    130_363,
    155_921,
    181_081,
    206_369,
    232_003,
    257_371,
    283_009,
)
_V120_MASTER_SEEDS = (
    1_905_245_264,
    3_477_142_941,
    70_359_369,
    2_936_962_664,
    976_469_083,
    1_434_863_921,
    714_423_665,
    4_202_129_964,
    2_335_380_198,
    854_314_474,
)
_V130_MASTER_SEEDS = (
    17_123_296,
    3_280_610_186,
    2_725_263_418,
    3_124_246_399,
    4_093_604_926,
    3_908_390_087,
    3_332_986_400,
    724_244_869,
    3_625_750_835,
    2_671_781_227,
)
_V140_MASTER_SEEDS = (
    2_439_054_559,
    3_246_851_043,
    339_970_590,
    474_769_515,
    550_273_937,
    438_984_650,
    2_732_731_971,
    2_253_809_848,
    2_206_960_337,
    3_506_881_479,
)
_V150_MASTER_SEEDS = (
    4_085_517_670,
    2_227_535_912,
    1_800_791_691,
    1_963_228_177,
    2_416_009_491,
    3_925_214_220,
    1_508_934_628,
    2_118_526_007,
    4_212_585_034,
    530_094_003,
)
_V160_MASTER_SEEDS = (
    2_999_896_578,
    3_783_052_994,
    3_863_790_658,
    3_900_021_454,
    1_437_244_820,
    3_175_470_977,
    228_708_147,
    3_835_462_042,
    3_342_200_973,
    1_751_060_143,
)
_V160_PROTOCOL_SHA256 = "6f5c21d6e77683c283e09c6257c35abd0e6857e17620e585f414024852d972b2"
_V170_MASTER_SEEDS = (
    3_920_043_614,
    3_703_229_797,
    2_080_036_362,
    865_871_218,
    3_636_713_390,
    2_195_564_811,
    2_000_167_339,
    329_754_669,
    4_064_290_468,
    1_911_057_116,
)
_V170_PROTOCOL_SHA256 = "bb7fe6de4fc5de231155fd555bcc0fce6e041b63d99b0c03def8daaf293a364a"
_V180_MASTER_SEEDS = (
    1_196_068_124,
    758_859_051,
    3_362_668_913,
    1_230_840_469,
    428_983_069,
    1_629_522_391,
    1_347_202_040,
    1_247_885_121,
    3_968_594_484,
    3_609_284_286,
)
_V180_PROTOCOL_SHA256 = "3aa795e1a54b7cda04b94c77afc683f79639b8f9fffc3dae8be839d53b5d89bc"
_V190_MASTER_SEEDS = (
    86_535_224,
    2_906_056_242,
    1_369_779_618,
    2_721_934_008,
    2_798_280_967,
    926_105_433,
    4_118_470_289,
    919_763_803,
    2_112_633_694,
    2_832_104_894,
)
_V190_PROTOCOL_SHA256 = (
    "3b97eaa1330066a7773345afd3445f086139d5e6090e8f86bfad87d14e93f090"
)
_V1100_MASTER_SEEDS = (
    1_647_437_737,
    1_166_509_260,
    3_363_134_750,
    2_153_178_322,
    2_277_484_641,
    572_614_265,
    3_119_775_486,
    3_121_614_244,
    3_646_941_950,
    827_253_974,
)
_V1100_PROTOCOL_SHA256 = (
    "fb2584cbbeab133692867e2396ee1ded5953ca7ceb7d68134febccd5aed3970b"
)
_V1110_MASTER_SEEDS = (
    670_819_759,
    624_845_448,
    3_391_764_770,
    20_596_598,
    999_954_271,
    2_371_040_464,
    2_073_495_343,
    962_058_337,
    2_170_781_413,
    3_523_651_983,
)
_V1110_PROTOCOL_SHA256 = (
    "757288cac9fc2935799e4500f0f0d0cf8135417eecb8d138cf8f3e38811d51c8"
)
_V1120_MASTER_SEEDS = (
    2_530_568_307,
    3_822_916_726,
    402_304_386,
    1_582_362_517,
    3_717_100_311,
    3_870_324_956,
    2_551_652_339,
    986_753_049,
    4_074_588_580,
    1_996_653_376,
)
_V1120_PROTOCOL_SHA256 = (
    "d64aede84e402d05bd587e1fdf2694381ab6742a28ca19ed88097d0480fa5b80"
)
_V1130_MASTER_SEEDS = (
    560_818_116,
    1_392_377_688,
    140_647_545,
    2_239_253_745,
    3_333_612_762,
    4_269_572_592,
    2_151_457_732,
    4_034_984_701,
    2_426_483_518,
    2_833_322_658,
)
_V1130_PROTOCOL_SHA256 = (
    "e7988e3605079b7b7830949d6fd107f26066059ac3cc3974c5bfe15af876dc0c"
)
_V1140_MASTER_SEEDS = (
    630_481_329,
    2_204_125_221,
    900_802_928,
    2_035_185_068,
    3_817_247_901,
    14_769_188,
    2_670_334_085,
    2_866_408_483,
    671_166_171,
    333_753_598,
)
_V1140_PROTOCOL_SHA256 = (
    "39f5820a91c8a504355f971449726ae0a9067cc856111a575bb038455d1fd635"
)
_V1150_MASTER_SEEDS = (
    2_388_891_654,
    3_201_418_215,
    2_465_968_807,
    3_494_485_289,
    1_615_601_571,
    2_220_840_580,
    280_448_223,
    597_199_725,
    712_207_456,
    1_727_907_751,
)
_V1150_PROTOCOL_SHA256 = (
    "8db5560044bbedfb491be12a26bd8b39c43fd6d6a314ce86d6afdc71f50486bb"
)
_DEVELOPMENT_MATRIX_CONTRACT_SHA256 = "09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"
_V130_BOUNDARY_TARGET_F32_HEX = "ac3cdebd"
_V130_BOUNDARY_MEANS_F32_HEX = (
    "8cd85cbb",
    "f032d7bb",
    "d0d5aebc",
    "fcaa09bc",
    "0086a53a",
)
_V130_BOUNDARY_LOG_VARIANCES_F32_HEX = (
    "66b8b3c0",
    "cb11b5c0",
    "d611b2c0",
    "86dcb2c0",
    "9390b2c0",
)
_EXPECTED_PHASE_SPLITS = {
    "train_a": ("collect_a",),
    "train_a_irrelevant": ("collect_irrelevant",),
    "train_a_corrupted": ("collect_a",),
    "train_b_replay": ("collect_a", "collect_b"),
    "train_b_naive": ("collect_b",),
}
_EXPECTED_SEED_COUNTS = {
    "model_initialization": 1,
    "torch_runtime": 1,
    "collection_action": 2,
    "irrelevant_collection_action": 1,
    "predictive_validation_irrelevant_action": 1,
    "predictive_validation_action": 2,
    "random_policy_action": 2,
    "ensemble_bootstrap_a": 5,
    "ensemble_bootstrap_b": 5,
    "minibatch_order_a": 1,
    "minibatch_order_b": 1,
    "corrupted_target_permutation": 1,
    "collect_a_episode": 8,
    "collect_irrelevant_episode": 8,
    "predictive_validation_irrelevant_episode": 8,
    "predictive_validation_a_episode": 8,
    "behavior_evaluation_a_episode": 32,
    "collect_b_episode": 8,
    "predictive_validation_b_episode": 8,
    "behavior_evaluation_b_episode": 32,
    "planner": 1,
}
_CANONICAL_COMPONENT_IDS = (
    "world_model",
    "optimizer",
    "model_version_ledger",
    "experience_store",
    "replay_index",
    "replay_sampling_history",
    "update_receipts",
    "agent_runtime",
    "scaling_configuration",
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_accelerator_rng",
    "collection_rng",
    "planner_rng",
)


class ArtifactAuditError(ValueError):
    """Untrusted artifact bytes violate a bounded WM-001 evidence format."""


@dataclass(frozen=True, slots=True)
class PredictionRecomputation:
    """Metrics and row identities independently recovered from one sidecar."""

    transition_ids: tuple[str, ...]
    normalized_targets: npt.NDArray[np.float32]
    member_means: npt.NDArray[np.float32]
    member_log_variances: npt.NDArray[np.float32]
    mixture_nll_nats_per_target_dimension: float
    normalized_rmse: float
    interval_90_coverage: float
    covered_target_count: int
    coverage_target_count: int


@dataclass(frozen=True, slots=True)
class SamplingManifest:
    """Decoded optimizer consumption indices and their eligible-ID binding."""

    indices: npt.NDArray[np.uint32]
    transition_ids_sha256: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class _EvaluatedCheckpoint:
    condition: str
    model_version: str
    parameter_sha256: str
    live_state_sha256: str
    model_tensors: Mapping[str, npt.NDArray[np.float32]]


class _Audit:
    def __init__(self) -> None:
        self.passed_checks = 0
        self.failed_checks = 0
        self.findings: list[dict[str, object]] = []
        self.coverage_gaps: list[dict[str, object]] = []
        self.independence_limitations: list[str] = []
        self.coverage_conformance_verified = False
        self.audit_execution_conformance_verified = False
        self.formal_runtime_binding: Mapping[str, object] | None = None
        self.formal_dependency_binding: Mapping[str, object] | None = None
        self.formal_source_binding: Mapping[str, object] | None = None
        self.custody: dict[str, object] = {
            "producer_manifest_checked": False,
            "producer_manifest_status": None,
            "producer_manifest_sha256": None,
        }

    def require(
        self,
        condition: bool,
        *,
        code: str,
        message: str,
        replicate_id: str | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> bool:
        if condition:
            self.passed_checks += 1
            return True
        self.failed_checks += 1
        finding: dict[str, object] = {
            "severity": "error",
            "code": code,
            "message": message,
        }
        if replicate_id is not None:
            finding["replicate_id"] = replicate_id
        if evidence:
            finding["evidence"] = dict(evidence)
        self.findings.append(finding)
        return False

    def error(
        self,
        code: str,
        message: str,
        *,
        replicate_id: str | None = None,
    ) -> None:
        self.require(False, code=code, message=message, replicate_id=replicate_id)

    def gap(self, code: str, message: str, *, evidence_needed: str) -> None:
        if any(row.get("code") == code for row in self.coverage_gaps):
            return
        self.coverage_gaps.append(
            {
                "severity": "blocker",
                "code": code,
                "message": message,
                "evidence_needed": evidence_needed,
            }
        )

    def limitation(self, message: str) -> None:
        if message not in self.independence_limitations:
            self.independence_limitations.append(message)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_equal(observed: object, expected: object) -> bool:
    """Compare decoded JSON without Python's bool/int or int/float aliases."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        return set(observed) == set(expected) and all(
            _strict_json_equal(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(observed, list)
        return len(observed) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ArtifactAuditError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ArtifactAuditError(f"{label} is not a real UTC timestamp") from error
    if parsed.tzinfo != UTC:
        raise ArtifactAuditError(f"{label} is not UTC")
    return parsed


def _json_without_duplicate_keys(payload: bytes, *, label: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ArtifactAuditError(f"{label} contains duplicate object key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactAuditError(f"{label} is not valid UTF-8 JSON") from error


def _read_bounded(path: Path, limit: int, *, label: str) -> bytes:
    try:
        stat = path.stat()
    except OSError as error:
        raise ArtifactAuditError(f"{label} cannot be read: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise ArtifactAuditError(f"{label} must be a regular non-symlink file")
    if stat.st_size > limit:
        raise ArtifactAuditError(f"{label} exceeds its {limit}-byte audit limit")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactAuditError(f"{label} cannot be read: {error}") from error
    if len(payload) != stat.st_size:
        raise ArtifactAuditError(f"{label} changed while it was being read")
    return payload


def _resolve_artifact_file(root: Path, filename: object, *, label: str) -> Path:
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ArtifactAuditError(f"{label} has an unsafe artifact filename")
    root_resolved = root.resolve()
    candidate = root / filename
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ArtifactAuditError(f"{label} is missing: {filename}") from error
    if resolved.parent != root_resolved:
        raise ArtifactAuditError(f"{label} escapes the artifact root")
    return candidate


def _verify_file_reference(
    root: Path,
    reference: Mapping[str, object],
    *,
    limit: int,
    label: str,
) -> tuple[Path, bytes]:
    path = _resolve_artifact_file(root, reference.get("filename"), label=label)
    payload = _read_bounded(path, limit, label=label)
    expected_bytes = reference.get("bytes")
    expected_digest = reference.get("sha256")
    if type(expected_bytes) is not int or expected_bytes != len(payload):
        raise ArtifactAuditError(f"{label} byte count does not match its file")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_digest != digest:
        raise ArtifactAuditError(f"{label} SHA-256 does not match its file")
    return path, payload


def _parse_container(
    payload: bytes,
    *,
    label: str,
) -> tuple[dict[str, object], dict[str, npt.NDArray[np.float32]]]:
    prefix_length = len(_MAGIC) + 8
    if len(payload) < prefix_length or not payload.startswith(_MAGIC):
        raise ArtifactAuditError(f"{label} has invalid container magic")
    header_length = struct.unpack(">Q", payload[len(_MAGIC) : prefix_length])[0]
    if header_length > _MAX_CONTAINER_HEADER_BYTES or header_length > len(payload) - prefix_length:
        raise ArtifactAuditError(f"{label} has an invalid header length")
    raw_header = payload[prefix_length : prefix_length + header_length]
    raw = _json_without_duplicate_keys(raw_header, label=f"{label} header")
    if not isinstance(raw, dict) or set(raw) != {"metadata", "payload_bytes", "tensors"}:
        raise ArtifactAuditError(f"{label} header has an unexpected field set")
    if _canonical_json_bytes(raw) != raw_header:
        raise ArtifactAuditError(f"{label} header is not canonical JSON")
    metadata = raw["metadata"]
    entries = raw["tensors"]
    declared_payload_bytes = raw["payload_bytes"]
    if not isinstance(metadata, dict) or not isinstance(entries, list) or type(declared_payload_bytes) is not int:
        raise ArtifactAuditError(f"{label} header has invalid field types")
    data = payload[prefix_length + header_length :]
    if declared_payload_bytes != len(data):
        raise ArtifactAuditError(f"{label} declared tensor bytes do not match its payload")

    tensors: dict[str, npt.NDArray[np.float32]] = {}
    expected_offset = 0
    previous_name = ""
    expected_fields = {"bytes", "dtype", "name", "offset", "sha256", "shape"}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected_fields:
            raise ArtifactAuditError(f"{label} tensor entry has an unexpected field set")
        name = entry["name"]
        dtype_name = entry["dtype"]
        shape = entry["shape"]
        offset = entry["offset"]
        byte_count = entry["bytes"]
        expected_sha256 = entry["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or name in tensors
            or name <= previous_name
            or dtype_name != "<f4"
            or not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
            or type(offset) is not int
            or type(byte_count) is not int
            or not isinstance(expected_sha256, str)
        ):
            raise ArtifactAuditError(f"{label} tensor metadata is invalid")
        if offset != expected_offset or byte_count < 0 or offset + byte_count > len(data):
            raise ArtifactAuditError(f"{label} tensor offsets are invalid")
        tensor_payload = data[offset : offset + byte_count]
        if hashlib.sha256(tensor_payload).hexdigest() != expected_sha256:
            raise ArtifactAuditError(f"{label} tensor {name!r} failed SHA-256 verification")
        element_count = math.prod(shape)
        if element_count * np.dtype("<f4").itemsize != byte_count:
            raise ArtifactAuditError(f"{label} tensor {name!r} shape disagrees with its bytes")
        array = np.frombuffer(tensor_payload, dtype="<f4").reshape(shape).copy()
        if not np.isfinite(array).all():
            raise ArtifactAuditError(f"{label} tensor {name!r} contains non-finite values")
        tensors[name] = array
        expected_offset += byte_count
        previous_name = name
    if expected_offset != len(data):
        raise ArtifactAuditError(f"{label} contains unclaimed tensor bytes")
    return metadata, tensors


def recompute_prediction_evidence(payload: bytes) -> PredictionRecomputation:
    """Independently decode one prediction sidecar and recompute all metrics."""

    if len(payload) > _MAX_PREDICTION_BYTES:
        raise ArtifactAuditError("prediction evidence exceeds its audit byte limit")
    metadata, tensors = _parse_container(payload, label="prediction evidence")
    if set(metadata) != {"format", "transition_ids"} or metadata.get("format") != _PREDICTION_FORMAT:
        raise ArtifactAuditError("prediction evidence metadata violates its format")
    raw_ids = metadata.get("transition_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or len(raw_ids) > 100_000
        or any(not isinstance(identity, str) or not identity or len(identity) > 4096 for identity in raw_ids)
        or len(set(raw_ids)) != len(raw_ids)
    ):
        raise ArtifactAuditError("prediction evidence transition IDs are malformed")
    if set(tensors) != {"member_log_variances", "member_means", "normalized_targets"}:
        raise ArtifactAuditError("prediction evidence tensor set violates its format")
    targets = tensors["normalized_targets"]
    means = tensors["member_means"]
    log_variances = tensors["member_log_variances"]
    row_count = len(raw_ids)
    if targets.shape != (row_count, 4):
        raise ArtifactAuditError("prediction targets must have shape [rows, 4]")
    if means.shape != (5, row_count, 4) or log_variances.shape != means.shape:
        raise ArtifactAuditError("prediction ensemble tensors must have shape [5, rows, 4]")
    if np.any(log_variances < -10.0) or np.any(log_variances > 0.5):
        raise ArtifactAuditError("prediction log variances exceed the sealed model bounds")

    # Use NumPy float64 and the standard-library erf rather than the producer's
    # Torch aggregation.  This gives an implementation-independent semantic
    # recomputation while retaining tight numerical agreement.
    target64 = targets.astype(np.float64)
    means64 = means.astype(np.float64)
    log_variances64 = log_variances.astype(np.float64)
    member_log_prob = -0.5 * (
        math.log(2.0 * math.pi) + log_variances64 + np.square(target64[None, :, :] - means64) * np.exp(-log_variances64)
    )
    maximum = np.max(member_log_prob, axis=0)
    mixture_log_prob = maximum + np.log(np.mean(np.exp(member_log_prob - maximum[None, :, :]), axis=0))
    mixture_nll = float(-np.mean(mixture_log_prob, dtype=np.float64))

    # The producer forms its ensemble mean and residuals in float32, then
    # aggregates stored squared errors in float64.  Repeat those declared
    # arithmetic semantics through NumPy, not through producer code.
    ensemble_mean = np.mean(means, axis=0, dtype=np.float32)
    squared_error = np.square(targets - ensemble_mean)
    normalized_rmse = float(math.sqrt(float(np.mean(squared_error, dtype=np.float64))))

    covered_target_count = 0
    coverage_target_count = int(targets.size)
    for row_index in range(row_count):
        for target_index in range(4):
            member_cdfs: list[float] = []
            target = float(targets[row_index, target_index])
            for member_index in range(5):
                mean = float(means[member_index, row_index, target_index])
                log_variance = float(log_variances[member_index, row_index, target_index])
                z_score = (target - mean) * math.exp(-0.5 * log_variance)
                member_cdfs.append(0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))))
            mixture_pit = math.fsum(member_cdfs) / 5
            if _binary64_mixture_pit_is_covered(mixture_pit):
                covered_target_count += 1
    coverage = covered_target_count / coverage_target_count
    if not all(math.isfinite(value) for value in (mixture_nll, normalized_rmse, coverage)):
        raise ArtifactAuditError("recomputed prediction metrics are non-finite")
    return PredictionRecomputation(
        transition_ids=tuple(raw_ids),
        normalized_targets=targets,
        member_means=means,
        member_log_variances=log_variances,
        mixture_nll_nats_per_target_dimension=mixture_nll,
        normalized_rmse=normalized_rmse,
        interval_90_coverage=coverage,
        covered_target_count=covered_target_count,
        coverage_target_count=coverage_target_count,
    )


def decode_sampling_manifest(payload: bytes) -> SamplingManifest:
    """Independently decode the fixed-endian optimizer sampling manifest."""

    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ArtifactAuditError("sampling manifest exceeds its audit byte limit")
    prefix_length = len(_MAGIC) + 8
    if len(payload) < prefix_length or not payload.startswith(_MAGIC):
        raise ArtifactAuditError("sampling manifest has invalid container magic")
    header_length = struct.unpack(">Q", payload[len(_MAGIC) : prefix_length])[0]
    if header_length > _MAX_CONTAINER_HEADER_BYTES or header_length > len(payload) - prefix_length:
        raise ArtifactAuditError("sampling manifest has an invalid header length")
    raw_header = payload[prefix_length : prefix_length + header_length]
    header = _json_without_duplicate_keys(raw_header, label="sampling manifest header")
    expected = {"dtype", "format", "payload_sha256", "shape", "transition_ids_sha256"}
    if not isinstance(header, dict) or set(header) != expected:
        raise ArtifactAuditError("sampling manifest header has an unexpected field set")
    if _canonical_json_bytes(header) != raw_header:
        raise ArtifactAuditError("sampling manifest header is not canonical JSON")
    shape = header.get("shape")
    if (
        header.get("format") != _SAMPLING_FORMAT
        or header.get("dtype") != "uint32-le"
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(type(size) is not int or size < 1 for size in shape)
        or shape[1:] != [5, 256]
        or not isinstance(header.get("payload_sha256"), str)
        or not isinstance(header.get("transition_ids_sha256"), str)
    ):
        raise ArtifactAuditError("sampling manifest header values violate the format")
    raw_indices = payload[prefix_length + header_length :]
    expected_bytes = math.prod(shape) * np.dtype("<u4").itemsize
    if len(raw_indices) != expected_bytes:
        raise ArtifactAuditError("sampling manifest shape disagrees with its bytes")
    payload_digest = hashlib.sha256(raw_indices).hexdigest()
    if payload_digest != header["payload_sha256"]:
        raise ArtifactAuditError("sampling manifest index payload failed SHA-256 verification")
    indices = np.frombuffer(raw_indices, dtype="<u4").reshape(shape).copy()
    return SamplingManifest(
        indices=indices,
        transition_ids_sha256=str(header["transition_ids_sha256"]),
        payload_sha256=payload_digest,
    )


def _decode_owned_model_state(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) > _MAX_OWNED_MODEL_BYTES or not payload.startswith(_OWNED_MODEL_MAGIC):
        raise ArtifactAuditError("owned model state is too large or has invalid magic")
    prefix_length = len(_OWNED_MODEL_MAGIC) + 8
    if len(payload) < prefix_length:
        raise ArtifactAuditError("owned model state is truncated")
    header_length = struct.unpack(
        ">Q",
        payload[len(_OWNED_MODEL_MAGIC) : prefix_length],
    )[0]
    if header_length > _MAX_CONTAINER_HEADER_BYTES or header_length > len(payload) - prefix_length:
        raise ArtifactAuditError("owned model state has an invalid header length")
    raw_header = payload[prefix_length : prefix_length + header_length]
    header = _json_without_duplicate_keys(raw_header, label="owned model state header")
    expected_fields = {
        "format",
        "model_bytes",
        "model_sha256",
        "optimizer_bytes",
        "optimizer_sha256",
    }
    if (
        not isinstance(header, dict)
        or set(header) != expected_fields
        or header.get("format") != _OWNED_MODEL_FORMAT
        or _canonical_json_bytes(header) != raw_header
    ):
        raise ArtifactAuditError("owned model state header violates its canonical format")
    model_size = header.get("model_bytes")
    optimizer_size = header.get("optimizer_bytes")
    if (
        type(model_size) is not int
        or model_size < 1
        or type(optimizer_size) is not int
        or optimizer_size < 1
        or prefix_length + header_length + model_size + optimizer_size != len(payload)
    ):
        raise ArtifactAuditError("owned model state component sizes are invalid")
    offset = prefix_length + header_length
    model_payload = payload[offset : offset + model_size]
    optimizer_payload = payload[offset + model_size :]
    if hashlib.sha256(model_payload).hexdigest() != header.get("model_sha256"):
        raise ArtifactAuditError("owned model parameter payload failed SHA-256 verification")
    if hashlib.sha256(optimizer_payload).hexdigest() != header.get("optimizer_sha256"):
        raise ArtifactAuditError("owned model optimizer payload failed SHA-256 verification")
    return model_payload, optimizer_payload


def _decode_sealed_model(
    payload: bytes,
) -> Mapping[str, npt.NDArray[np.float32]]:
    metadata, tensors = _parse_container(payload, label="evaluated world model")
    config = metadata.get("config")
    expected_config = {
        "ensemble_members": 5,
        "hidden_dimensions": [256, 256],
        "input_dimension": 5,
        "log_variance_max": 0.5,
        "log_variance_min": -10.0,
        "output_dimension": 4,
        "scaling": {
            "action": 2.0,
            "context": 1.0,
            "delta": [2.0, 2.0, 16.0],
            "observation": [1.0, 1.0, 8.0],
            "reward": _TARGET_REWARD_SCALE,
        },
    }
    if set(metadata) != {"config", "format"} or metadata.get("format") != _MODEL_FORMAT or config != expected_config:
        raise ArtifactAuditError("evaluated world model differs from the sealed architecture")
    expected_tensor_names = {
        f"members.{member}.network.{layer}.{field}"
        for member in range(5)
        for layer in (0, 2, 4)
        for field in ("bias", "weight")
    }
    if set(tensors) != expected_tensor_names:
        raise ArtifactAuditError("evaluated world model tensor names violate the architecture")
    expected_shapes = {
        0: {"weight": (256, 5), "bias": (256,)},
        2: {"weight": (256, 256), "bias": (256,)},
        4: {"weight": (8, 256), "bias": (8,)},
    }
    for member in range(5):
        for layer, fields in expected_shapes.items():
            for field, shape in fields.items():
                name = f"members.{member}.network.{layer}.{field}"
                if tensors[name].shape != shape:
                    raise ArtifactAuditError(f"evaluated world model tensor {name!r} has the wrong shape")
    return tensors


def _silu(value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # Clipping changes no meaningful SiLU result in float32 while avoiding an
    # overflow warning for adversarial but finite weights.
    exponent = np.exp(np.clip(-value, -80.0, 80.0)).astype(np.float32)
    return np.asarray(value / (np.float32(1.0) + exponent), dtype=np.float32)


def _model_forward(
    tensors: Mapping[str, npt.NDArray[np.float32]],
    transition_rows: Sequence[Mapping[str, object]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    inputs: list[list[float]] = []
    for row in transition_rows:
        observation = row.get("pre_observation")
        context = _finite_float(row.get("task_context"))
        action = _finite_float(row.get("intended_action"))
        if (
            not isinstance(observation, list)
            or len(observation) != 3
            or any(_finite_float(value) is None for value in observation)
            or context is None
            or action is None
        ):
            raise ArtifactAuditError("prediction source transition lacks numeric model inputs")
        inputs.append(
            [
                float(observation[0]),
                float(observation[1]),
                float(observation[2]) / 8.0,
                context,
                action / 2.0,
            ]
        )
    features = np.asarray(inputs, dtype=np.float32)
    means: npt.NDArray[np.float32] = np.empty(
        (5, len(inputs), 4),
        dtype=np.float32,
    )
    log_variances: npt.NDArray[np.float32] = np.empty_like(means)
    for member in range(5):
        prefix = f"members.{member}.network"
        hidden_a = _silu(features @ tensors[f"{prefix}.0.weight"].T + tensors[f"{prefix}.0.bias"])
        hidden_b = _silu(hidden_a @ tensors[f"{prefix}.2.weight"].T + tensors[f"{prefix}.2.bias"])
        output = hidden_b @ tensors[f"{prefix}.4.weight"].T + tensors[f"{prefix}.4.bias"]
        means[member] = output[:, :4]
        log_variances[member] = np.clip(output[:, 4:], -10.0, 0.5)
    return means, log_variances


def _audit_evaluated_checkpoints(
    audit: _Audit,
    root: Path,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
) -> dict[str, _EvaluatedCheckpoint]:
    result: dict[str, _EvaluatedCheckpoint] = {}
    rows = _mapping_rows(replicate.get("evaluated_checkpoints"))
    for row in rows:
        condition = row.get("condition")
        if not isinstance(condition, str) or condition in result:
            audit.error(
                "duplicate_or_invalid_evaluated_checkpoint",
                f"{replicate_id} has duplicate or invalid evaluated checkpoint {condition!r}",
                replicate_id=replicate_id,
            )
            continue
        try:
            _, payload = _verify_file_reference(
                root,
                row,
                limit=_MAX_OWNED_MODEL_BYTES,
                label=f"{replicate_id} evaluated checkpoint {condition}",
            )
            live_digest = hashlib.sha256(payload).hexdigest()
            model_payload, _ = _decode_owned_model_state(payload)
            parameter_digest = hashlib.sha256(model_payload).hexdigest()
            model_version = f"wm001-state-sha256:{live_digest}"
            audit.require(
                row.get("live_state_sha256") == live_digest
                and row.get("model_version") == model_version
                and row.get("parameter_sha256") == parameter_digest,
                code="evaluated_checkpoint_identity_mismatch",
                message=(
                    f"{replicate_id} {condition} checkpoint identity does not bind its exact compound/model bytes"
                ),
                replicate_id=replicate_id,
            )
            result[condition] = _EvaluatedCheckpoint(
                condition=condition,
                model_version=model_version,
                parameter_sha256=parameter_digest,
                live_state_sha256=live_digest,
                model_tensors=_decode_sealed_model(model_payload),
            )
        except (ArtifactAuditError, OSError, ValueError) as error:
            audit.error(
                "evaluated_checkpoint_invalid",
                f"{replicate_id} evaluated checkpoint {condition}: {error}",
                replicate_id=replicate_id,
            )
    expected_conditions = {
        "cold",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_a",
        "after_b_replay",
        "after_b_naive",
    }
    audit.require(
        set(result) == expected_conditions,
        code="evaluated_checkpoint_set_mismatch",
        message=f"{replicate_id} does not retain exactly the seven evaluated learned states",
        replicate_id=replicate_id,
    )
    shared_tensor_names = {
        f"members.{member}.network.{layer}.{field}"
        for member in range(5)
        for layer in (0, 2, 4)
        for field in ("bias", "weight")
    }
    audit.require(
        bool(result)
        and all(set(checkpoint.model_tensors) == shared_tensor_names for checkpoint in result.values())
        and not any(
            token in name
            for name in shared_tensor_names
            for token in ("task_a", "task_b", "route", "adapter", "task_head")
        ),
        code="shared_model_architecture_mismatch",
        message=(
            f"{replicate_id} evaluated states do not all use one exact shared "
            "five-member network without task-keyed routes, adapters, or heads"
        ),
        replicate_id=replicate_id,
    )
    if "cold" in result and "frozen" in result:
        audit.require(
            result["cold"].live_state_sha256 == result["frozen"].live_state_sha256,
            code="frozen_checkpoint_mutated",
            message=f"{replicate_id} frozen control differs from the cold checkpoint",
            replicate_id=replicate_id,
        )
    phase_conditions = {
        "train_a": ("cold", "after_a"),
        "train_a_corrupted": ("cold", "corrupted"),
        "train_a_irrelevant": ("cold", "irrelevant"),
        "train_b_replay": ("after_a", "after_b_replay"),
        "train_b_naive": ("after_a", "after_b_naive"),
    }
    updates = {
        str(row.get("phase")): row
        for row in _mapping_rows(replicate.get("updates"))
        if row.get("status") == "committed"
    }
    for phase, (before, after) in phase_conditions.items():
        update = updates.get(phase)
        before_state = result.get(before)
        after_state = result.get(after)
        if update is None or before_state is None or after_state is None:
            continue
        audit.require(
            update.get("predecessor_parameter_sha256") == before_state.parameter_sha256
            and update.get("live_state_before_sha256") == before_state.live_state_sha256
            and update.get("committed_parameter_sha256") == after_state.parameter_sha256
            and update.get("live_state_after_sha256") == after_state.live_state_sha256,
            code="update_checkpoint_lineage_mismatch",
            message=f"{replicate_id} {phase} update lineage differs from evaluated state bytes",
            replicate_id=replicate_id,
        )
    return result


def _close_enough(actual: float, expected: object, *, absolute: float = 2e-6) -> bool:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return False
    expected_float = float(expected)
    return math.isfinite(expected_float) and math.isclose(
        actual,
        expected_float,
        rel_tol=2e-6,
        abs_tol=absolute,
    )


def _ordered_validation_ids(
    replicate: Mapping[str, object],
    *,
    task_id: str,
    split: str,
) -> tuple[str, ...]:
    result: list[str] = []
    for episode in _mapping_rows(replicate.get("episodes")):
        if episode.get("task_id") == task_id and episode.get("split") == split:
            transition_ids = episode.get("transition_ids")
            if not isinstance(transition_ids, list) or any(
                not isinstance(identity, str) for identity in transition_ids
            ):
                raise ArtifactAuditError("validation episode transition IDs are malformed")
            result.extend(transition_ids)
    return tuple(result)


def _expected_prediction_targets_f32(
    transitions: Sequence[Mapping[str, object]],
    *,
    device: str,
) -> npt.NDArray[np.float32]:
    """Reproduce the producer's exact bound-device target operation path."""

    before_rows: list[list[float]] = []
    after_rows: list[list[float]] = []
    rewards: list[float] = []
    for transition in transitions:
        before_raw = transition.get("pre_observation")
        after_raw = transition.get("next_observation")
        reward = _finite_float(transition.get("reward"))
        if (
            not isinstance(before_raw, list)
            or len(before_raw) != 3
            or any(_finite_float(value) is None for value in before_raw)
            or not isinstance(after_raw, list)
            or len(after_raw) != 3
            or any(_finite_float(value) is None for value in after_raw)
            or reward is None
        ):
            raise ArtifactAuditError("raw transition cannot reconstruct the prediction target")
        before_rows.append([float(value) for value in before_raw])
        after_rows.append([float(value) for value in after_raw])
        rewards.append(reward)
    if not before_rows:
        raise ArtifactAuditError("prediction target reconstruction is empty")

    before_f64 = np.asarray(before_rows, dtype=np.float64)
    after_f64 = np.asarray(after_rows, dtype=np.float64)
    delta_f64 = np.subtract(after_f64, before_f64, dtype=np.float64)
    reconstructed_after_f64 = np.add(
        before_f64,
        delta_f64,
        dtype=np.float64,
    )
    try:
        import torch

        destination = torch.device(device)
        if destination.type == "cuda" and not torch.cuda.is_available():
            raise ArtifactAuditError("bound CUDA target arithmetic is unavailable to the auditor")
        before_f32 = torch.as_tensor(
            before_f64,
            dtype=torch.float32,
        ).contiguous()
        after_f32 = torch.as_tensor(
            reconstructed_after_f64,
            dtype=torch.float32,
        ).contiguous()
        reward_f32 = (
            torch.as_tensor(
                rewards,
                dtype=torch.float32,
            )
            .reshape(-1, 1)
            .contiguous()
        )
        before_device = before_f32.to(destination)
        after_device = after_f32.to(destination)
        reward_device = reward_f32.to(destination)
        delta_scale = before_device.new_tensor((2.0, 2.0, 16.0))
        expected = torch.cat(
            (
                (after_device - before_device) / delta_scale,
                reward_device / _TARGET_REWARD_SCALE,
            ),
            dim=-1,
        )
        result = expected.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    except ImportError as error:
        raise ArtifactAuditError("PyTorch is required for bound-device target reconstruction") from error
    except (RuntimeError, ValueError) as error:
        raise ArtifactAuditError(f"bound-device target reconstruction failed: {error}") from error
    if result.shape != (len(transitions), 4) or not np.isfinite(result).all():
        raise ArtifactAuditError("reconstructed prediction targets have invalid shape or values")
    return result


def _expected_prediction_target_f32(
    transition: Mapping[str, object],
    *,
    device: str = "cpu",
) -> npt.NDArray[np.float32]:
    return cast(
        npt.NDArray[np.float32],
        _expected_prediction_targets_f32(
            (transition,),
            device=device,
        )[0],
    )


def _binary64_mixture_pit_is_covered(mixture_pit: float) -> bool:
    """Apply the sealed inclusive interval to an exact binary64 ratio."""

    if not math.isfinite(mixture_pit):
        raise ArtifactAuditError("mixture PIT is non-finite")
    numerator, denominator = mixture_pit.as_integer_ratio()
    return 20 * numerator >= denominator and 20 * numerator <= 19 * denominator


def _audit_prediction_coverage(
    audit: _Audit,
    recomputed: PredictionRecomputation,
    row: Mapping[str, object],
    *,
    label: str,
    replicate_id: str,
) -> None:
    """Require the complete v1.4 count contract and exact recomputation."""

    audit.require(
        row.get("coverage_semantics") == _COVERAGE_SEMANTICS,
        code="prediction_coverage_semantics_mismatch",
        message=f"{label} does not declare the exact sealed coverage semantics",
        replicate_id=replicate_id,
        evidence={
            "expected": _COVERAGE_SEMANTICS,
            "stored": row.get("coverage_semantics"),
        },
    )
    transition_count = row.get("transition_count")
    stored_count = row.get("interval_90_covered_target_count")
    stored_target_count = row.get("coverage_target_count")
    expected_target_count = 4 * transition_count if type(transition_count) is int and transition_count > 0 else None
    count_contract_valid = (
        type(stored_count) is int
        and type(stored_target_count) is int
        and expected_target_count is not None
        and stored_target_count == expected_target_count
        and recomputed.coverage_target_count == expected_target_count
        and 0 <= stored_count <= stored_target_count
    )
    audit.require(
        count_contract_valid,
        code="prediction_coverage_count_contract_mismatch",
        message=f"{label} coverage counts do not match four targets per transition",
        replicate_id=replicate_id,
        evidence={
            "transition_count": transition_count,
            "stored_covered_target_count": stored_count,
            "stored_coverage_target_count": stored_target_count,
            "recomputed_coverage_target_count": recomputed.coverage_target_count,
        },
    )
    if not count_contract_valid:
        return
    assert type(stored_count) is int
    assert type(stored_target_count) is int
    stored_fraction = row.get("interval_90_coverage")
    expected_fraction = stored_count / stored_target_count
    audit.require(
        (
            not isinstance(stored_fraction, bool)
            and isinstance(stored_fraction, (int, float))
            and math.isfinite(float(stored_fraction))
            and float(stored_fraction) == expected_fraction
        ),
        code="prediction_coverage_fraction_mismatch",
        message=f"{label} stored coverage is not the exact covered/total fraction",
        replicate_id=replicate_id,
        evidence={
            "stored": stored_fraction,
            "expected": expected_fraction,
            "stored_covered_target_count": stored_count,
            "stored_coverage_target_count": stored_target_count,
        },
    )
    audit.require(
        stored_count == recomputed.covered_target_count,
        code="prediction_coverage_mismatch",
        message=f"{label} stored covered-target count differs from independent recomputation",
        replicate_id=replicate_id,
        evidence={
            "recomputed": recomputed.interval_90_coverage,
            "stored": stored_fraction,
            "recomputed_covered_target_count": recomputed.covered_target_count,
            "stored_covered_target_count": stored_count,
            "coverage_target_count": recomputed.coverage_target_count,
            "covered_target_count_difference": abs(recomputed.covered_target_count - stored_count),
        },
    )


def _audit_predictions(
    audit: _Audit,
    root: Path,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
    device: str,
    transitions_by_id: Mapping[str, Mapping[str, object]],
    evaluated_checkpoints: Mapping[str, _EvaluatedCheckpoint],
) -> None:
    verified: dict[
        tuple[str, str, str],
        PredictionRecomputation,
    ] = {}
    target_cache: dict[tuple[str, ...], bytes] = {}
    for index, row in enumerate(_mapping_rows(replicate.get("predictive_metrics"))):
        label = f"{replicate_id} predictive metric {index}"
        try:
            filename = row.get("prediction_evidence_file")
            path = _resolve_artifact_file(root, filename, label=label)
            payload = _read_bounded(path, _MAX_PREDICTION_BYTES, label=label)
            audit.require(
                row.get("prediction_evidence_bytes") == len(payload),
                code="prediction_file_size_mismatch",
                message=f"{label} byte count does not match {path.name}",
                replicate_id=replicate_id,
            )
            digest = hashlib.sha256(payload).hexdigest()
            audit.require(
                row.get("prediction_rows_sha256") == digest,
                code="prediction_file_digest_mismatch",
                message=f"{label} SHA-256 does not match {path.name}",
                replicate_id=replicate_id,
            )
            recomputed = recompute_prediction_evidence(payload)
            split = row.get("split")
            task_id = row.get("task_id")
            condition = row.get("condition")
            if not isinstance(split, str) or not isinstance(task_id, str):
                raise ArtifactAuditError(f"{label} lacks a valid task and split")
            if isinstance(condition, str):
                key = (task_id, split, condition)
                if key in verified:
                    raise ArtifactAuditError(f"{label} duplicates predictive condition {key!r}")
                verified[key] = recomputed
            expected_ids = _ordered_validation_ids(replicate, task_id=task_id, split=split)
            audit.require(
                recomputed.transition_ids == expected_ids,
                code="prediction_transition_binding_mismatch",
                message=f"{label} IDs do not exactly match the ordered validation split",
                replicate_id=replicate_id,
            )
            audit.require(
                row.get("transition_count") == len(recomputed.transition_ids),
                code="prediction_transition_count_mismatch",
                message=f"{label} transition count disagrees with its sidecar",
                replicate_id=replicate_id,
            )
            source_target_rows = [
                transitions_by_id[transition_id]
                for transition_id in recomputed.transition_ids
                if transition_id in transitions_by_id
            ]
            target_rows_complete = len(source_target_rows) == len(recomputed.transition_ids)
            expected_target_bytes = b""
            if target_rows_complete:
                expected_target_bytes = target_cache.get(
                    recomputed.transition_ids,
                    b"",
                )
                if not expected_target_bytes:
                    expected_target_bytes = (
                        _expected_prediction_targets_f32(
                            source_target_rows,
                            device=device,
                        )
                        .astype("<f4", copy=False)
                        .tobytes(order="C")
                    )
                    target_cache[recomputed.transition_ids] = expected_target_bytes
            actual_target_bytes = recomputed.normalized_targets.astype("<f4", copy=False).tobytes(order="C")
            targets_match = target_rows_complete and actual_target_bytes == expected_target_bytes
            audit.require(
                targets_match,
                code="prediction_target_binding_mismatch",
                message=(
                    f"{label} normalized target bytes do not exactly match the "
                    "binary32 targets reconstructed from raw transitions"
                ),
                replicate_id=replicate_id,
            )
            evaluated = evaluated_checkpoints.get(condition) if isinstance(condition, str) else None
            audit.require(
                evaluated is not None
                and row.get("checkpoint_id") == condition
                and row.get("model_version") == evaluated.model_version
                and row.get("parameter_sha256") == evaluated.parameter_sha256
                and row.get("live_state_sha256") == evaluated.live_state_sha256,
                code="prediction_checkpoint_binding_mismatch",
                message=f"{label} is not bound to the retained evaluated checkpoint bytes",
                replicate_id=replicate_id,
            )
            if evaluated is not None:
                source_rows = [
                    transitions_by_id[identity]
                    for identity in recomputed.transition_ids
                    if identity in transitions_by_id
                ]
                if len(source_rows) != len(recomputed.transition_ids):
                    raise ArtifactAuditError(f"{label} cannot recover every transition required for model replay")
                replayed_means, replayed_log_variances = _model_forward(
                    evaluated.model_tensors,
                    source_rows,
                )
                audit.require(
                    bool(
                        np.allclose(
                            replayed_means,
                            recomputed.member_means,
                            rtol=2e-5,
                            atol=2e-5,
                        )
                        and np.allclose(
                            replayed_log_variances,
                            recomputed.member_log_variances,
                            rtol=2e-5,
                            atol=2e-5,
                        )
                    ),
                    code="prediction_model_replay_mismatch",
                    message=(
                        f"{label} tensors do not match an independent NumPy forward pass "
                        "through the retained checkpoint"
                    ),
                    replicate_id=replicate_id,
                )
            audit.require(
                _close_enough(
                    recomputed.mixture_nll_nats_per_target_dimension,
                    row.get("mixture_nll_nats_per_target_dimension"),
                ),
                code="prediction_nll_mismatch",
                message=f"{label} stored NLL differs from independent recomputation",
                replicate_id=replicate_id,
                evidence={
                    "recomputed": recomputed.mixture_nll_nats_per_target_dimension,
                    "stored": row.get("mixture_nll_nats_per_target_dimension"),
                },
            )
            audit.require(
                _close_enough(recomputed.normalized_rmse, row.get("normalized_rmse")),
                code="prediction_rmse_mismatch",
                message=f"{label} stored RMSE differs from independent recomputation",
                replicate_id=replicate_id,
                evidence={
                    "recomputed": recomputed.normalized_rmse,
                    "stored": row.get("normalized_rmse"),
                },
            )
            _audit_prediction_coverage(
                audit,
                recomputed,
                row,
                label=label,
                replicate_id=replicate_id,
            )
        except (ArtifactAuditError, OSError, ValueError) as error:
            audit.error("prediction_evidence_invalid", f"{label}: {error}", replicate_id=replicate_id)

    _audit_irrelevant_prediction_manipulation(
        audit,
        replicate,
        replicate_id=replicate_id,
        transitions_by_id=transitions_by_id,
        verified=verified,
    )


def _audit_irrelevant_prediction_manipulation(
    audit: _Audit,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
    transitions_by_id: Mapping[str, Mapping[str, object]],
    verified: Mapping[tuple[str, str, str], PredictionRecomputation],
) -> None:
    """Cross-bind both v1.4 oscillator sidecars to one analytic held-out set."""

    split = "predictive_validation_irrelevant"
    expected_ids = _ordered_validation_ids(
        replicate,
        task_id=_TASK_IRRELEVANT,
        split=split,
    )
    predictive_rows = [
        row
        for row in _mapping_rows(replicate.get("predictive_metrics"))
        if row.get("task_id") == _TASK_IRRELEVANT or row.get("split") == split
    ]
    if not expected_ids and not predictive_rows:
        audit.gap(
            "irrelevant_prediction_manipulation_absent",
            (f"{replicate_id} has no held-out oscillator prediction pair for the protocol-v1.4 manipulation check."),
            evidence_needed=(
                "Cold and after-irrelevant prediction sidecars over the same "
                "disjoint predictive_validation_irrelevant transitions."
            ),
        )
        return

    expected_keys = {
        (_TASK_IRRELEVANT, split, "cold"),
        (_TASK_IRRELEVANT, split, "irrelevant"),
    }
    actual_contracts = {
        (
            row.get("task_id"),
            row.get("split"),
            row.get("condition"),
            row.get("checkpoint_id"),
        )
        for row in predictive_rows
    }
    expected_contracts = {
        (_TASK_IRRELEVANT, split, "cold", "cold"),
        (_TASK_IRRELEVANT, split, "irrelevant", "irrelevant"),
    }
    pair = {key: verified[key] for key in expected_keys if key in verified}
    audit.require(
        actual_contracts == expected_contracts and len(predictive_rows) == 2 and set(pair) == expected_keys,
        code="irrelevant_prediction_pair_incomplete",
        message=(
            f"{replicate_id} does not contain exactly the independently "
            "decoded cold and after-irrelevant oscillator sidecars"
        ),
        replicate_id=replicate_id,
    )
    if set(pair) != expected_keys:
        return

    cold = pair[(_TASK_IRRELEVANT, split, "cold")]
    irrelevant = pair[(_TASK_IRRELEVANT, split, "irrelevant")]
    audit.require(
        bool(expected_ids)
        and cold.transition_ids == expected_ids
        and irrelevant.transition_ids == expected_ids
        and np.array_equal(
            cold.normalized_targets,
            irrelevant.normalized_targets,
        ),
        code="irrelevant_prediction_pair_binding_mismatch",
        message=(
            f"{replicate_id} oscillator checkpoints were not scored on the "
            "same ordered held-out transition IDs and byte-identical targets"
        ),
        replicate_id=replicate_id,
    )

    analytic_targets: list[list[float]] = []
    analytic_valid = True
    for transition_id in expected_ids:
        row = transitions_by_id.get(transition_id)
        before = None if row is None else row.get("pre_observation")
        if (
            row is None
            or row.get("task_id") != _TASK_IRRELEVANT
            or row.get("split") != split
            or not isinstance(before, list)
            or len(before) != 3
            or any(_finite_float(value) is None for value in before)
        ):
            analytic_valid = False
            break
        try:
            expected_next, expected_reward, expected_applied = _independent_oscillator_step(before)
        except ArtifactAuditError:
            analytic_valid = False
            break
        if expected_applied != 0.0:
            analytic_valid = False
            break
        analytic_targets.append(
            [
                (float(expected_next[0]) - float(before[0])) / 2.0,
                (float(expected_next[1]) - float(before[1])) / 2.0,
                (float(expected_next[2]) - float(before[2])) / 16.0,
                expected_reward / _TARGET_REWARD_SCALE,
            ]
        )
    analytic_array = np.asarray(analytic_targets, dtype=np.float32)
    analytic_match = (
        analytic_valid
        and analytic_array.shape == cold.normalized_targets.shape
        and np.allclose(
            cold.normalized_targets,
            analytic_array,
            rtol=2e-6,
            atol=2e-7,
        )
        and np.allclose(
            irrelevant.normalized_targets,
            analytic_array,
            rtol=2e-6,
            atol=2e-7,
        )
    )
    audit.require(
        bool(analytic_match),
        code="irrelevant_prediction_analytic_target_mismatch",
        message=(
            f"{replicate_id} oscillator prediction targets are not independently "
            "derivable from the bound autonomous analytic dynamics"
        ),
        replicate_id=replicate_id,
    )

    heldout_ids = set(expected_ids)
    update_ids: set[str] = set()
    for update in _mapping_rows(replicate.get("updates")):
        eligible = update.get("eligible_transition_ids")
        if isinstance(eligible, list):
            update_ids.update(identity for identity in eligible if isinstance(identity, str))
    audit.require(
        bool(heldout_ids) and heldout_ids.isdisjoint(update_ids),
        code="irrelevant_validation_training_contamination",
        message=(
            f"{replicate_id} held-out oscillator validation IDs occur in a "
            "candidate update's eligible training identities"
        ),
        replicate_id=replicate_id,
    )


def _consumption_sha256(
    indices: npt.NDArray[np.uint32],
    transition_ids: Sequence[str],
) -> str:
    encoded = tuple(identity.encode("utf-8") + b"\n" for identity in transition_ids)
    digest = hashlib.sha256()
    flat = indices.reshape(-1)
    chunk_size = 65_536
    for start in range(0, flat.size, chunk_size):
        chunk = flat[start : start + chunk_size]
        digest.update(b"".join(encoded[int(index)] for index in chunk))
    return digest.hexdigest()


def _encode_reconstructed_sampling_manifest(
    indices: npt.NDArray[np.uint32],
    transition_ids: Sequence[str],
) -> bytes:
    raw_indices = np.asarray(indices, dtype="<u4").tobytes(order="C")
    identity_payload = json.dumps(
        list(transition_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    header = _canonical_json_bytes(
        {
            "dtype": "uint32-le",
            "format": _SAMPLING_FORMAT,
            "payload_sha256": hashlib.sha256(raw_indices).hexdigest(),
            "shape": list(indices.shape),
            "transition_ids_sha256": hashlib.sha256(identity_payload).hexdigest(),
        }
    )
    return bytes(_MAGIC + struct.pack(">Q", len(header)) + header + raw_indices)


def _reconstruct_optimizer_sampling(
    *,
    phase: str,
    master_seed: int,
    optimizer_steps: int,
    eligible_ids: Sequence[str],
    transitions_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[npt.NDArray[np.uint32], bytes]:
    """Regenerate Torch CPU bootstrap/minibatch draws from sealed namespaces.

    This is deliberately local audit code.  It uses only Torch's public CPU
    generator primitives and does not import the producer model/sampling
    module.
    """

    try:
        import torch
    except ImportError as error:
        raise ArtifactAuditError("Torch is unavailable for optimizer RNG replay") from error

    if phase in {"train_a", "train_a_corrupted", "train_a_irrelevant"}:
        bootstrap_namespace = "ensemble_bootstrap_a"
        order_namespace = "minibatch_order_a"
        balanced = False
    elif phase in {"train_b_replay", "train_b_naive"}:
        bootstrap_namespace = "ensemble_bootstrap_b"
        order_namespace = "minibatch_order_b"
        balanced = phase == "train_b_replay"
    else:
        raise ArtifactAuditError(f"unsupported optimizer phase {phase!r}")
    if optimizer_steps < 1 or not eligible_ids:
        raise ArtifactAuditError("optimizer RNG replay requires positive steps and rows")

    row_count = len(eligible_ids)
    member_generators: list[Any] = []
    for member in range(5):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_derive_seed(bootstrap_namespace, master_seed, member))
        member_generators.append(generator)

    task_a: Any | None = None
    task_b: Any | None = None
    if balanced:
        task_a = torch.as_tensor(
            [
                index
                for index, identity in enumerate(eligible_ids)
                if transitions_by_id[identity].get("task_id") == _TASK_A
            ],
            dtype=torch.long,
        )
        task_b = torch.as_tensor(
            [
                index
                for index, identity in enumerate(eligible_ids)
                if transitions_by_id[identity].get("task_id") == _TASK_B
            ],
            dtype=torch.long,
        )
        if task_a.numel() == 0 or task_b.numel() == 0:
            raise ArtifactAuditError("balanced replay has no eligible transition from one task")

    indices: npt.NDArray[np.uint32] = np.empty(
        (optimizer_steps, 5, 256),
        dtype="<u4",
    )
    for step in range(optimizer_steps):
        for member, untyped_generator in enumerate(member_generators):
            generator = untyped_generator
            if not balanced:
                selected = torch.randint(
                    row_count,
                    (256,),
                    generator=generator,
                    dtype=torch.long,
                )
            else:
                assert task_a is not None and task_b is not None
                selected_a = task_a[
                    torch.randint(
                        task_a.numel(),
                        (128,),
                        generator=generator,
                    )
                ]
                selected_b = task_b[
                    torch.randint(
                        task_b.numel(),
                        (128,),
                        generator=generator,
                    )
                ]
                combined = torch.cat((selected_a, selected_b))
                selected = combined[torch.randperm(256, generator=generator)]
            indices[step, member] = selected.detach().cpu().numpy().astype("<u4", copy=False)

    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(_derive_seed(order_namespace, master_seed))
    order = (
        torch.randperm(
            optimizer_steps,
            generator=order_generator,
        )
        .detach()
        .cpu()
        .numpy()
    )
    indices = indices[order].copy()
    return indices, _encode_reconstructed_sampling_manifest(indices, eligible_ids)


def _reconstruct_target_permutation(
    *,
    master_seed: int,
    eligible_count: int,
) -> bytes:
    """Regenerate the corrupted-control joint-target permutation exactly."""

    try:
        import torch
    except ImportError as error:
        raise ArtifactAuditError("Torch is unavailable for target RNG replay") from error
    if eligible_count < 1:
        raise ArtifactAuditError("target permutation requires eligible transitions")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_derive_seed("corrupted_target_permutation", master_seed))
    return bytes(
        torch.randperm(eligible_count, generator=generator)
        .detach()
        .cpu()
        .numpy()
        .astype("<u4", copy=False)
        .tobytes(order="C")
    )


def _decode_strict_base64(value: object, *, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ArtifactAuditError(f"{label} is not nonempty base64")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ArtifactAuditError(f"{label} is not canonical base64") from error
    if not decoded or base64.b64encode(decoded).decode("ascii") != value:
        raise ArtifactAuditError(f"{label} is empty or non-canonical base64")
    return decoded


def _decode_rejected_probe_state(
    payload: bytes,
    *,
    update: Mapping[str, object],
    train_a_update: Mapping[str, object] | None,
) -> Mapping[str, object]:
    raw = _json_without_duplicate_keys(payload, label="rejected-probe full state")
    expected_fields = {
        "schema",
        "captured_at",
        "model_state",
        "domain_graph",
        "source_replay_rows",
        "probe_replay_rows",
        "source_identity_base64",
        "probe_identity_base64",
        "collection_rng_state",
        "process_rng",
        "retained_learning_evidence",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != expected_fields
        or raw.get("schema") != "prospect.wm001.rejected-probe-full-state.v1"
        or _canonical_json_bytes(raw) != payload
    ):
        raise ArtifactAuditError("rejected-probe full state is not canonical or has wrong fields")
    captured_at = raw.get("captured_at")
    if (
        not isinstance(captured_at, list)
        or len(captured_at) != 2
        or not isinstance(captured_at[0], str)
        or not captured_at[0]
        or isinstance(captured_at[1], bool)
        or not isinstance(captured_at[1], int)
        or captured_at[1] < 0
    ):
        raise ArtifactAuditError("rejected-probe capture time is malformed")

    model_state = raw.get("model_state")
    if not isinstance(model_state, Mapping) or set(model_state) != {
        "version",
        "digest",
        "payload_base64",
    }:
        raise ArtifactAuditError("rejected-probe model state is malformed")
    model_payload = _decode_strict_base64(
        model_state.get("payload_base64"),
        label="rejected-probe model payload",
    )
    live_digest = hashlib.sha256(model_payload).hexdigest()
    model_bytes, _ = _decode_owned_model_state(model_payload)
    parameter_digest = hashlib.sha256(model_bytes).hexdigest()
    if (
        model_state.get("digest") != live_digest
        or model_state.get("version") != f"wm001-state-sha256:{live_digest}"
        or update.get("live_state_before_sha256") != live_digest
        or update.get("predecessor_model_version") != f"wm001-state-sha256:{live_digest}"
        or update.get("predecessor_parameter_sha256") != parameter_digest
    ):
        raise ArtifactAuditError("rejected-probe model bytes do not bind the rejected update row")
    _validate_domain_graph_structure(
        raw.get("domain_graph"),
        component_id="rejected_probe",
    )
    if not isinstance(raw.get("source_replay_rows"), list) or not isinstance(raw.get("probe_replay_rows"), list):
        raise ArtifactAuditError("rejected-probe replay rows are malformed")
    for name in ("source_identity_base64", "probe_identity_base64"):
        _decode_strict_base64(raw.get(name), label=f"rejected-probe {name}")
    collection_rng = raw.get("collection_rng_state")
    if not isinstance(collection_rng, Mapping) or not collection_rng:
        raise ArtifactAuditError("rejected-probe collection RNG is malformed")
    process_rng = raw.get("process_rng")
    if not isinstance(process_rng, Mapping) or set(process_rng) != {
        "python_base64",
        "numpy_base64",
        "torch_cpu_base64",
        "torch_accelerator_base64",
    }:
        raise ArtifactAuditError("rejected-probe process RNG block is malformed")
    for name, encoded in process_rng.items():
        _decode_strict_base64(encoded, label=f"rejected-probe process RNG {name}")

    evidence = raw.get("retained_learning_evidence")
    evidence_fields = {
        "phase",
        "consumed_transition_ids",
        "consumed_multiset_sha256",
        "predecessor_parameter_sha256",
        "candidate_parameter_sha256",
        "predecessor_live_state_sha256",
        "candidate_live_state_sha256",
        "optimizer_steps",
        "sampling_manifest_base64",
        "sampling_manifest_sha256",
        "sampled_id_counts",
        "target_permutation_sha256",
        "target_permutation_base64",
        "loss_history",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        raise ArtifactAuditError("rejected-probe retained-learning evidence is malformed")
    sampling_payload = _decode_strict_base64(
        evidence.get("sampling_manifest_base64"),
        label="rejected-probe sampling manifest",
    )
    sampling_digest = hashlib.sha256(sampling_payload).hexdigest()
    decode_sampling_manifest(sampling_payload)
    target_encoded = evidence.get("target_permutation_base64")
    target_digest = evidence.get("target_permutation_sha256")
    if target_encoded is None:
        target_valid = target_digest is None
    else:
        target_payload = _decode_strict_base64(
            target_encoded,
            label="rejected-probe target permutation",
        )
        target_valid = hashlib.sha256(target_payload).hexdigest() == target_digest
    consumed_ids = evidence.get("consumed_transition_ids")
    sampled_counts = evidence.get("sampled_id_counts")
    losses = evidence.get("loss_history")
    evidence_valid = (
        evidence.get("phase") == "train_a"
        and isinstance(consumed_ids, list)
        and all(isinstance(identity, str) and identity for identity in consumed_ids)
        and len(consumed_ids) == len(set(consumed_ids))
        and isinstance(sampled_counts, list)
        and isinstance(losses, list)
        and all(_finite_float(loss) is not None for loss in losses)
        and type(evidence.get("optimizer_steps")) is int
        and len(losses) == evidence.get("optimizer_steps")
        and evidence.get("sampling_manifest_sha256") == sampling_digest
        and evidence.get("candidate_parameter_sha256") == parameter_digest
        and evidence.get("candidate_live_state_sha256") == live_digest
        and target_valid
    )
    if train_a_update is not None:
        evidence_valid = evidence_valid and all(
            (
                evidence.get("consumed_transition_ids") == train_a_update.get("eligible_transition_ids"),
                evidence.get("consumed_multiset_sha256") == train_a_update.get("consumed_multiset_sha256"),
                evidence.get("predecessor_parameter_sha256") == train_a_update.get("predecessor_parameter_sha256"),
                evidence.get("candidate_parameter_sha256") == train_a_update.get("committed_parameter_sha256"),
                evidence.get("predecessor_live_state_sha256") == train_a_update.get("live_state_before_sha256"),
                evidence.get("candidate_live_state_sha256") == train_a_update.get("live_state_after_sha256"),
                evidence.get("optimizer_steps") == train_a_update.get("optimizer_steps"),
                evidence.get("sampling_manifest_sha256") == train_a_update.get("sampling_manifest_sha256"),
            )
        )
    if not evidence_valid:
        raise ArtifactAuditError("rejected-probe retained-learning evidence is not bound to train_a")
    return raw


def _audit_rejected_probe_full_state(
    audit: _Audit,
    root: Path,
    update: Mapping[str, object],
    *,
    replicate_id: str,
    train_a_update: Mapping[str, object] | None,
) -> None:
    before_reference = update.get("full_state_before_file")
    after_reference = update.get("full_state_after_file")
    if not isinstance(before_reference, Mapping) or not isinstance(after_reference, Mapping):
        audit.gap(
            "rejected_probe_full_state_unavailable",
            f"{replicate_id} rejected probe has no before/after full-state sidecars.",
            evidence_needed=(
                "Content-addressed canonical before/after full-live-state JSON "
                "covering model/optimizer, domain store/ledgers/agent, replay, "
                "identity sources, and every process/collection RNG."
            ),
        )
        return
    try:
        _, before_payload = _verify_file_reference(
            root,
            before_reference,
            limit=_MAX_REJECTED_STATE_BYTES,
            label=f"{replicate_id} rejected-probe state before",
        )
        _, after_payload = _verify_file_reference(
            root,
            after_reference,
            limit=_MAX_REJECTED_STATE_BYTES,
            label=f"{replicate_id} rejected-probe state after",
        )
        media_type = "application/vnd.prospect.wm001.rejected-probe-state+json"
        before_digest = hashlib.sha256(before_payload).hexdigest()
        after_digest = hashlib.sha256(after_payload).hexdigest()
        audit.require(
            before_reference.get("media_type") == media_type
            and after_reference.get("media_type") == media_type
            and update.get("full_state_before_sha256") == before_digest
            and update.get("full_state_after_sha256") == after_digest
            and before_payload == after_payload
            and before_digest == after_digest,
            code="rejected_probe_full_state_changed",
            message=(f"{replicate_id} rejected update did not preserve exact complete live-state bytes"),
            replicate_id=replicate_id,
        )
        _decode_rejected_probe_state(
            before_payload,
            update=update,
            train_a_update=train_a_update,
        )
        _decode_rejected_probe_state(
            after_payload,
            update=update,
            train_a_update=train_a_update,
        )
    except (ArtifactAuditError, OSError, ValueError) as error:
        audit.error(
            "rejected_probe_full_state_invalid",
            f"{replicate_id} rejected-probe full-state evidence: {error}",
            replicate_id=replicate_id,
        )


def _audit_target_permutation(
    audit: _Audit,
    root: Path,
    update: Mapping[str, object],
    *,
    replicate_id: str,
    eligible_count: int,
    master_seed: int,
) -> None:
    phase = str(update.get("phase"))
    reference = update.get("target_permutation_file")
    digest = update.get("target_permutation_sha256")
    if phase != "train_a_corrupted":
        audit.require(
            reference is None and digest is None,
            code="unexpected_target_permutation",
            message=f"{replicate_id} {phase} unexpectedly declares a target permutation",
            replicate_id=replicate_id,
        )
        return
    if not isinstance(reference, Mapping):
        audit.error(
            "missing_target_permutation",
            f"{replicate_id} corrupted control has no target-permutation file",
            replicate_id=replicate_id,
        )
        return
    try:
        _, payload = _verify_file_reference(
            root,
            reference,
            limit=_MAX_PERMUTATION_BYTES,
            label=f"{replicate_id} target permutation",
        )
        audit.require(
            digest == hashlib.sha256(payload).hexdigest(),
            code="target_permutation_digest_mismatch",
            message=f"{replicate_id} update does not bind its target-permutation file",
            replicate_id=replicate_id,
        )
        if len(payload) != eligible_count * 4:
            raise ArtifactAuditError("target permutation byte count does not match eligible transitions")
        permutation = np.frombuffer(payload, dtype="<u4")
        valid = bool(
            len(permutation) == eligible_count
            and np.array_equal(np.sort(permutation), np.arange(eligible_count, dtype=np.uint32))
        )
        audit.require(
            valid,
            code="target_permutation_not_bijective",
            message=f"{replicate_id} corrupted target mapping is not a complete permutation",
            replicate_id=replicate_id,
        )
        audit.require(
            eligible_count <= 1 or not np.array_equal(permutation, np.arange(eligible_count)),
            code="target_permutation_identity",
            message=f"{replicate_id} corrupted target control uses the identity mapping",
            replicate_id=replicate_id,
        )
        expected_payload = _reconstruct_target_permutation(
            master_seed=master_seed,
            eligible_count=eligible_count,
        )
        audit.require(
            payload == expected_payload,
            code="target_permutation_seed_replay_mismatch",
            message=(
                f"{replicate_id} corrupted target permutation does not byte-for-byte "
                "replay from corrupted_target_permutation[0]"
            ),
            replicate_id=replicate_id,
        )
    except (ArtifactAuditError, OSError, ValueError) as error:
        audit.error(
            "target_permutation_invalid",
            f"{replicate_id} corrupted target permutation: {error}",
            replicate_id=replicate_id,
        )


def _audit_optimizer_manifests(
    audit: _Audit,
    root: Path,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
    transitions_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    raw_master_seed = replicate.get("master_seed")
    if isinstance(raw_master_seed, bool) or not isinstance(raw_master_seed, int):
        audit.error(
            "optimizer_master_seed_invalid",
            f"{replicate_id} has no integer master seed for optimizer RNG replay",
            replicate_id=replicate_id,
        )
        return
    master_seed = raw_master_seed
    manifest_rows = _mapping_rows(replicate.get("optimizer_batch_manifests"))
    manifests_by_phase: dict[str, Mapping[str, object]] = {}
    for row in manifest_rows:
        phase = row.get("phase")
        if isinstance(phase, str) and phase not in manifests_by_phase:
            manifests_by_phase[phase] = row
        else:
            audit.error(
                "duplicate_or_invalid_manifest_phase",
                f"{replicate_id} has duplicate or invalid optimizer manifest phase {phase!r}",
                replicate_id=replicate_id,
            )

    update_rows = _mapping_rows(replicate.get("updates"))
    train_a_update = next(
        (row for row in update_rows if row.get("phase") == "train_a"),
        None,
    )
    decoded_indices: dict[str, npt.NDArray[np.uint32]] = {}
    committed_phases: set[str] = set()
    for update in update_rows:
        phase = update.get("phase")
        status = update.get("status")
        if phase == "rejected_update_probe":
            audit.require(
                status == "rejected"
                and update.get("optimizer_steps") == 0
                and update.get("consumed_sample_count") == 0
                and update.get("consumed_multiset_sha256") == _SHA256_EMPTY
                and update.get("sampling_manifest_sha256") is None,
                code="rejected_probe_consumed_evidence",
                message=f"{replicate_id} rejected update probe reports optimizer consumption",
                replicate_id=replicate_id,
            )
            _audit_rejected_probe_full_state(
                audit,
                root,
                update,
                replicate_id=replicate_id,
                train_a_update=train_a_update,
            )
            continue
        if not isinstance(phase, str) or phase not in _EXPECTED_PHASE_SPLITS or status != "committed":
            audit.error(
                "unexpected_update_phase",
                f"{replicate_id} has unexpected update phase/status {phase!r}/{status!r}",
                replicate_id=replicate_id,
            )
            continue
        committed_phases.add(phase)
        expected_splits = _EXPECTED_PHASE_SPLITS[phase]
        eligible_ids_raw = update.get("eligible_transition_ids")
        if not isinstance(eligible_ids_raw, list) or any(
            not isinstance(identity, str) for identity in eligible_ids_raw
        ):
            audit.error(
                "eligible_transition_ids_invalid",
                f"{replicate_id} {phase} eligible transition IDs are malformed",
                replicate_id=replicate_id,
            )
            continue
        eligible_ids = tuple(eligible_ids_raw)
        expected_ids = tuple(
            str(row["transition_id"])
            for row in _mapping_rows(replicate.get("transitions"))
            if row.get("split") in expected_splits and isinstance(row.get("transition_id"), str)
        )
        raw_eligible_splits = update.get("eligible_splits")
        audit.require(
            isinstance(raw_eligible_splits, list) and tuple(raw_eligible_splits) == expected_splits,
            code="eligible_splits_mismatch",
            message=f"{replicate_id} {phase} declares the wrong eligible splits",
            replicate_id=replicate_id,
        )
        audit.require(
            eligible_ids == expected_ids and len(set(eligible_ids)) == len(eligible_ids),
            code="eligible_transition_binding_mismatch",
            message=f"{replicate_id} {phase} eligible IDs do not exactly bind its collection rows",
            replicate_id=replicate_id,
        )
        audit.require(
            update.get("eligible_transition_count") == len(eligible_ids),
            code="eligible_transition_count_mismatch",
            message=f"{replicate_id} {phase} eligible count disagrees with its ID list",
            replicate_id=replicate_id,
        )
        manifest_ref = manifests_by_phase.get(phase)
        if manifest_ref is None:
            audit.error(
                "missing_optimizer_manifest",
                f"{replicate_id} {phase} has no optimizer sampling manifest",
                replicate_id=replicate_id,
            )
            _audit_target_permutation(
                audit,
                root,
                update,
                replicate_id=replicate_id,
                eligible_count=len(eligible_ids),
                master_seed=master_seed,
            )
            continue
        try:
            _, payload = _verify_file_reference(
                root,
                manifest_ref,
                limit=_MAX_MANIFEST_BYTES,
                label=f"{replicate_id} {phase} optimizer manifest",
            )
            payload_digest = hashlib.sha256(payload).hexdigest()
            audit.require(
                update.get("sampling_manifest_sha256") == payload_digest,
                code="update_manifest_digest_mismatch",
                message=f"{replicate_id} {phase} update does not bind its manifest file",
                replicate_id=replicate_id,
            )
            manifest = decode_sampling_manifest(payload)
            decoded_indices[phase] = manifest.indices
            expected_identity_digest = hashlib.sha256(
                json.dumps(
                    list(eligible_ids),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            audit.require(
                manifest.transition_ids_sha256 == expected_identity_digest,
                code="manifest_eligible_ids_digest_mismatch",
                message=f"{replicate_id} {phase} manifest does not bind its eligible ID order",
                replicate_id=replicate_id,
            )
            indices_in_range = bool(
                len(eligible_ids) > 0
                and manifest.indices.size > 0
                and int(np.max(manifest.indices)) < len(eligible_ids)
            )
            audit.require(
                indices_in_range,
                code="manifest_index_out_of_range",
                message=f"{replicate_id} {phase} manifest indexes outside its eligible rows",
                replicate_id=replicate_id,
            )
            audit.require(
                update.get("optimizer_steps") == manifest.indices.shape[0],
                code="manifest_optimizer_steps_mismatch",
                message=f"{replicate_id} {phase} manifest shape disagrees with optimizer steps",
                replicate_id=replicate_id,
            )
            audit.require(
                update.get("consumed_sample_count") == manifest.indices.size,
                code="consumed_sample_count_mismatch",
                message=f"{replicate_id} {phase} consumed count disagrees with exact manifest entries",
                replicate_id=replicate_id,
            )
            expected_indices, expected_payload = _reconstruct_optimizer_sampling(
                phase=phase,
                master_seed=master_seed,
                optimizer_steps=manifest.indices.shape[0],
                eligible_ids=eligible_ids,
                transitions_by_id=transitions_by_id,
            )
            audit.require(
                payload == expected_payload and np.array_equal(manifest.indices, expected_indices),
                code="optimizer_sampling_seed_replay_mismatch",
                message=(
                    f"{replicate_id} {phase} sampling manifest does not "
                    "byte-for-byte replay from its five bootstrap seeds, "
                    "minibatch-order seed, and declared balance rule"
                ),
                replicate_id=replicate_id,
            )
            if indices_in_range:
                consumed_digest = _consumption_sha256(manifest.indices, eligible_ids)
                audit.require(
                    update.get("consumed_multiset_sha256") == consumed_digest,
                    code="consumed_sequence_digest_mismatch",
                    message=f"{replicate_id} {phase} consumed-sequence SHA-256 is not reproducible",
                    replicate_id=replicate_id,
                )
                if phase == "train_b_replay":
                    is_task_a = np.asarray(
                        [transitions_by_id[identity].get("task_id") == _TASK_A for identity in eligible_ids],
                        dtype=np.int16,
                    )
                    task_a_counts = np.sum(is_task_a[manifest.indices], axis=-1)
                    audit.require(
                        bool(np.all(task_a_counts == 128)),
                        code="replay_batch_not_balanced",
                        message=(
                            f"{replicate_id} replay manifest is not 128/128 task-balanced "
                            "for every optimizer step and ensemble member"
                        ),
                        replicate_id=replicate_id,
                    )
        except (ArtifactAuditError, KeyError, OSError, ValueError) as error:
            audit.error(
                "optimizer_manifest_invalid",
                f"{replicate_id} {phase} optimizer manifest: {error}",
                replicate_id=replicate_id,
            )
        _audit_target_permutation(
            audit,
            root,
            update,
            replicate_id=replicate_id,
            eligible_count=len(eligible_ids),
            master_seed=master_seed,
        )
    audit.require(
        set(manifests_by_phase) == committed_phases,
        code="optimizer_manifest_phase_set_mismatch",
        message=f"{replicate_id} optimizer manifest phases do not exactly match committed updates",
        replicate_id=replicate_id,
    )
    train_a_indices = decoded_indices.get("train_a")
    irrelevant_indices = decoded_indices.get("train_a_irrelevant")
    audit.require(
        train_a_indices is not None
        and irrelevant_indices is not None
        and np.array_equal(train_a_indices, irrelevant_indices),
        code="irrelevant_control_sampling_schedule_mismatch",
        message=(
            f"{replicate_id} irrelevant-control and task-A updates do not use "
            "the same bootstrap/minibatch sample-index schedule"
        ),
        replicate_id=replicate_id,
    )
    audit.limitation(_OPTIMIZER_RNG_INDEPENDENCE_LIMITATION)


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _derive_seed(namespace: str, master_seed: int, index: int = 0) -> int:
    """Reconstruct one sealed seed without trusting producer seed rows."""

    return int.from_bytes(
        hashlib.sha256((f"WM-001|{_SEED_HASH_DOMAIN_VERSION}|{namespace}|{master_seed}|{index}").encode()).digest()[:4],
        "big",
    )


def _derive_master_seed(lane: str, index: int) -> int:
    if lane not in {"development", "formal"} or index < 0:
        raise ArtifactAuditError("invalid master-seed lane or index")
    return int.from_bytes(
        hashlib.sha256(f"WM-001|{_SEED_HASH_DOMAIN_VERSION}|{lane}-master|{index}".encode()).digest()[:4],
        "big",
        signed=False,
    )


def _audit_protocol_seed_contract(
    audit: _Audit,
    protocol: Mapping[str, object],
) -> None:
    schedule = protocol.get("seed_schedule")
    if not isinstance(schedule, Mapping):
        audit.error("protocol_seed_contract_mismatch", "protocol seed schedule is missing")
        return
    master_derivation = schedule.get("master_seed_derivation")
    collision_audit = master_derivation.get("collision_audit") if isinstance(master_derivation, Mapping) else None
    namespace_rows = schedule.get("namespaces")
    declared_counts = (
        {
            str(namespace): declaration.get("count")
            for namespace, declaration in namespace_rows.items()
            if isinstance(namespace, str) and isinstance(declaration, Mapping)
        }
        if isinstance(namespace_rows, Mapping)
        else {}
    )
    current_masters = set((*_DEVELOPMENT_SEEDS, *_FORMAL_SEEDS))
    current_stream_values = [
        _derive_seed(namespace, master_seed, index)
        for master_seed in (*_DEVELOPMENT_SEEDS, *_FORMAL_SEEDS)
        for namespace, count in _EXPECTED_SEED_COUNTS.items()
        for index in range(count)
    ]
    current_streams = set(current_stream_values)
    prior_domains = (
        ("1.0.0", _V100_MASTER_SEEDS),
        ("1.2.0", _V120_MASTER_SEEDS),
        ("1.3.0", _V130_MASTER_SEEDS),
        ("1.4.0", _V140_MASTER_SEEDS),
        ("1.5.0", _V150_MASTER_SEEDS),
        ("1.6.0", _V160_MASTER_SEEDS),
        ("1.7.0", _V170_MASTER_SEEDS),
        ("1.8.0", _V180_MASTER_SEEDS),
        ("1.9.0", _V190_MASTER_SEEDS),
        ("1.10.0", _V1100_MASTER_SEEDS),
        ("1.11.0", _V1110_MASTER_SEEDS),
        ("1.12.0", _V1120_MASTER_SEEDS),
        ("1.13.0", _V1130_MASTER_SEEDS),
        ("1.14.0", _V1140_MASTER_SEEDS),
        ("1.15.0", _V1150_MASTER_SEEDS),
    )
    prior_masters = {master_seed for _, version_masters in prior_domains for master_seed in version_masters}
    prior_stream_values = [
        int.from_bytes(
            hashlib.sha256(f"WM-001|{version}|{namespace}|{master_seed}|{index}".encode()).digest()[:4],
            "big",
            signed=False,
        )
        for version, version_masters in prior_domains
        for master_seed in version_masters
        for namespace, count in _EXPECTED_SEED_COUNTS.items()
        for index in range(count)
    ]
    prior_streams = set(prior_stream_values)
    bindings = protocol.get("bindings")
    development_qualification = (
        bindings.get("development_qualification")
        if isinstance(bindings, Mapping)
        else None
    )
    valid = (
        protocol.get("schema") == "prospect.world-model-lifecycle.protocol.v9"
        and schedule.get("derivation_domain_version") == _SEED_HASH_DOMAIN_VERSION
        and tuple(schedule.get("development_replicate_master_seeds", ()))
        == tuple(_derive_master_seed("development", index) for index in range(2))
        == _DEVELOPMENT_SEEDS
        and tuple(schedule.get("formal_replicate_master_seeds", ()))
        == tuple(_derive_master_seed("formal", index) for index in range(8))
        == _FORMAL_SEEDS
        and isinstance(master_derivation, Mapping)
        and master_derivation.get("lane_index_domains") == {"development": [0, 1], "formal": [0, 7]}
        and declared_counts == _EXPECTED_SEED_COUNTS
        and isinstance(collision_audit, Mapping)
        and collision_audit.get("current_master_seed_count") == 10
        and collision_audit.get("current_derived_stream_count") == 1360
        and collision_audit.get("unique_current_derived_stream_count")
        == len(current_streams)
        == len(current_stream_values)
        == 1360
        and collision_audit.get("current_internal_collision_count") == 0
        and collision_audit.get("current_master_stream_overlap_count") == 0
        and current_masters.isdisjoint(current_streams)
        and collision_audit.get("prior_master_seed_count") == len(prior_masters) == 150
        and collision_audit.get("unique_prior_derived_stream_count")
        == len(prior_streams)
        == len(prior_stream_values)
        == 20400
        and collision_audit.get("current_prior_master_master_overlap_count") == 0
        and collision_audit.get("current_prior_stream_stream_overlap_count") == 0
        and collision_audit.get("current_master_prior_stream_overlap_count") == 0
        and collision_audit.get("prior_master_current_stream_overlap_count") == 0
        and current_masters.isdisjoint(prior_masters)
        and current_streams.isdisjoint(prior_streams)
        and current_masters.isdisjoint(prior_streams)
        and prior_masters.isdisjoint(current_streams)
        and isinstance(development_qualification, Mapping)
        and development_qualification.get("matrix_contract_sha256")
        == _DEVELOPMENT_MATRIX_CONTRACT_SHA256
    )
    audit.require(
        valid,
        code="protocol_seed_contract_mismatch",
        message=(
            "protocol master derivation, schedule parity, prior-domain collision audit, "
            "or matrix-contract binding differs from v1.16"
        ),
    )


def _independent_pendulum_reset(seed: int) -> npt.NDArray[np.float32]:
    """Reconstruct Gymnasium Pendulum-v1's default seeded reset observation.

    Gymnasium 0.29.1 seeds a PCG64 ``Generator`` through ``default_rng``, draws
    ``(theta, theta_dot)`` in one vectorized uniform call over
    ``[-pi, -1]..[pi, 1]``, and projects the result to a float32 observation.
    No environment or producer reset helper is imported here.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactAuditError("Pendulum reset seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    draw: npt.NDArray[np.float64] = np.asarray(
        rng.uniform(
            low=np.asarray([-math.pi, -1.0], dtype=np.float64),
            high=np.asarray([math.pi, 1.0], dtype=np.float64),
        ),
        dtype=np.float64,
    )
    theta = float(draw[0])
    angular_velocity = float(draw[1])
    return np.asarray(
        [math.cos(theta), math.sin(theta), angular_velocity],
        dtype=np.float32,
    )


def _independent_oscillator_reset(seed: int) -> npt.NDArray[np.float64]:
    """Reconstruct the sealed distractor reset without producer runtime code."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactAuditError("independent oscillator reset seed must be a non-negative integer")
    digest = hashlib.sha256(f"{_OSCILLATOR_SOURCE}:{seed}".encode("ascii")).digest()
    phase_unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    velocity_unit = int.from_bytes(digest[8:16], "big") / float(1 << 64)
    phase = (2.0 * phase_unit - 1.0) * math.pi
    velocity = 0.5 + velocity_unit
    return np.asarray(
        [math.cos(phase), math.sin(phase), velocity],
        dtype=np.float64,
    )


def _independent_oscillator_step(
    observation: Sequence[object],
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Reconstruct one autonomous oscillator transition.

    The action is absent by construction: the environment advances only from
    phase and constant velocity and reports an applied action of exactly zero.
    """

    if len(observation) != 3:
        raise ArtifactAuditError("independent oscillator observation must contain three values")
    parsed = tuple(_finite_float(value) for value in observation)
    if any(value is None for value in parsed):
        raise ArtifactAuditError("independent oscillator observation contains a non-finite value")
    cosine, sine, velocity = (float(value) for value in parsed if value is not None)
    phase = math.atan2(sine, cosine)
    next_phase = math.remainder(
        phase + _OSCILLATOR_TIME_STEP * velocity,
        2.0 * math.pi,
    )
    next_observation = np.asarray(
        [math.cos(next_phase), math.sin(next_phase), velocity],
        dtype=np.float64,
    )
    return next_observation, float(math.cos(next_phase)), 0.0


def _independent_pendulum_step(
    observation: Sequence[object],
    *,
    context: float,
    intended_action: float,
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Reconstruct one Pendulum-v1 step without importing producer dynamics.

    Raw observations are Gymnasium's float32 projection of its hidden
    ``(theta, theta_dot)`` state.  The reconstruction therefore uses the
    protocol's float32-planner conformance tolerances rather than pretending
    that the hidden float64 angle can be recovered exactly.
    """

    if len(observation) != 3:
        raise ArtifactAuditError("Pendulum source observation must contain three values")
    parsed = tuple(_finite_float(value) for value in observation)
    if any(value is None for value in parsed):
        raise ArtifactAuditError("Pendulum source observation contains a non-finite value")
    cosine, sine, angular_velocity = (float(value) for value in parsed if value is not None)
    if context not in (0.0, 1.0):
        raise ArtifactAuditError("Pendulum context must be exactly zero or one")

    theta = math.atan2(sine, cosine)
    normalized_theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    clipped_intended = min(
        _PENDULUM_MAX_TORQUE,
        max(-_PENDULUM_MAX_TORQUE, intended_action),
    )
    applied_action = clipped_intended if context == 0.0 else -clipped_intended
    reward = -(
        normalized_theta * normalized_theta
        + 0.1 * angular_velocity * angular_velocity
        + 0.001 * applied_action * applied_action
    )
    acceleration = 15.0 * math.sin(theta) + 3.0 * applied_action
    next_angular_velocity = min(
        _PENDULUM_MAX_SPEED,
        max(
            -_PENDULUM_MAX_SPEED,
            angular_velocity + acceleration * _PENDULUM_TIME_STEP,
        ),
    )
    next_theta = theta + next_angular_velocity * _PENDULUM_TIME_STEP
    next_observation = np.asarray(
        [
            math.cos(next_theta),
            math.sin(next_theta),
            next_angular_velocity,
        ],
        dtype=np.float64,
    )
    return next_observation, reward, applied_action


def _audit_analytic_transition_dynamics(
    audit: _Audit,
    row: Mapping[str, object],
    *,
    replicate_id: str,
    transition_id: str,
) -> None:
    """Check one raw row against its independently implemented dynamics."""

    before = row.get("pre_observation")
    after = row.get("next_observation")
    task_id = row.get("task_id")
    context = _finite_float(row.get("task_context"))
    intended = _finite_float(row.get("intended_action"))
    applied = _finite_float(row.get("applied_action"))
    reward = _finite_float(row.get("reward"))
    if (
        not isinstance(before, list)
        or not isinstance(after, list)
        or len(after) != 3
        or any(_finite_float(value) is None for value in after)
        or context is None
        or intended is None
        or applied is None
        or reward is None
    ):
        audit.error(
            "transition_dynamics_source_invalid",
            f"{replicate_id} transition {transition_id} lacks numeric dynamics evidence",
            replicate_id=replicate_id,
        )
        return
    try:
        if task_id == _TASK_IRRELEVANT:
            if context != _TASK_CONTEXT[_TASK_IRRELEVANT]:
                raise ArtifactAuditError("independent oscillator context must be exactly two")
            expected_observation, expected_reward, expected_applied = _independent_oscillator_step(before)
            observation_tolerance = _OSCILLATOR_OBSERVATION_ATOL
            reward_tolerance = _OSCILLATOR_REWARD_ATOL
        else:
            expected_observation, expected_reward, expected_applied = _independent_pendulum_step(
                before,
                context=context,
                intended_action=intended,
            )
            observation_tolerance = _PENDULUM_OBSERVATION_ATOL
            reward_tolerance = _PENDULUM_REWARD_ATOL
    except ArtifactAuditError as error:
        audit.error(
            "transition_dynamics_source_invalid",
            f"{replicate_id} transition {transition_id}: {error}",
            replicate_id=replicate_id,
        )
        return

    actual_observation = np.asarray(after, dtype=np.float64)
    observation_error = float(np.max(np.abs(actual_observation - expected_observation)))
    reward_error = abs(reward - expected_reward)
    applied_error = abs(applied - expected_applied)
    audit.require(
        observation_error <= observation_tolerance,
        code="transition_dynamics_observation_mismatch",
        message=(
            f"{replicate_id} transition {transition_id} next observation is not "
            "consistent with the sealed analytic task dynamics"
        ),
        replicate_id=replicate_id,
        evidence={
            "max_absolute_error": observation_error,
            "absolute_tolerance": observation_tolerance,
        },
    )
    audit.require(
        reward_error <= reward_tolerance,
        code="transition_dynamics_reward_mismatch",
        message=(
            f"{replicate_id} transition {transition_id} reward is not consistent with the sealed analytic task dynamics"
        ),
        replicate_id=replicate_id,
        evidence={
            "absolute_error": reward_error,
            "absolute_tolerance": reward_tolerance,
        },
    )
    audit.require(
        applied_error <= 1e-7,
        code="transition_dynamics_applied_action_mismatch",
        message=(
            f"{replicate_id} transition {transition_id} applied action is not "
            "consistent with the analytic task actuation"
        ),
        replicate_id=replicate_id,
        evidence={"absolute_error": applied_error, "absolute_tolerance": 1e-7},
    )


def _audit_transition_and_episode_rows(
    audit: _Audit,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
) -> dict[str, Mapping[str, object]]:
    transition_rows = _mapping_rows(replicate.get("transitions"))
    transition_ids = [row.get("transition_id") for row in transition_rows]
    valid_transition_ids = all(isinstance(identity, str) and identity for identity in transition_ids)
    audit.require(
        valid_transition_ids and len(set(transition_ids)) == len(transition_ids),
        code="duplicate_or_invalid_transition_id",
        message=f"{replicate_id} transition IDs are not globally unique nonempty strings",
        replicate_id=replicate_id,
    )
    transitions_by_id = {
        str(row["transition_id"]): row for row in transition_rows if isinstance(row.get("transition_id"), str)
    }
    referenced_ids: list[str] = []
    episode_ids: set[str] = set()
    for episode in _mapping_rows(replicate.get("episodes")):
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in episode_ids:
            audit.error(
                "duplicate_or_invalid_episode_id",
                f"{replicate_id} has a duplicate or invalid episode ID",
                replicate_id=replicate_id,
            )
            continue
        episode_ids.add(episode_id)
        raw_ids = episode.get("transition_ids")
        if not isinstance(raw_ids, list) or any(not isinstance(value, str) for value in raw_ids):
            audit.error(
                "episode_transition_ids_invalid",
                f"{replicate_id} episode {episode_id} transition IDs are malformed",
                replicate_id=replicate_id,
            )
            continue
        ids = [str(value) for value in raw_ids]
        referenced_ids.extend(ids)
        rows = [transitions_by_id.get(identity) for identity in ids]
        complete = all(row is not None for row in rows)
        audit.require(
            complete,
            code="episode_transition_missing",
            message=f"{replicate_id} episode {episode_id} references a missing transition",
            replicate_id=replicate_id,
        )
        if not complete:
            continue
        present_rows = [row for row in rows if row is not None]
        if not present_rows:
            audit.error(
                "episode_step_sequence_mismatch",
                f"{replicate_id} episode {episode_id} has no transition rows",
                replicate_id=replicate_id,
            )
            continue
        audit.require(
            episode.get("environment_steps") == _EPISODE_HORIZON
            and len(present_rows) == _EPISODE_HORIZON
            and [row.get("step_index") for row in present_rows] == list(range(_EPISODE_HORIZON)),
            code="episode_step_sequence_mismatch",
            message=(f"{replicate_id} episode {episode_id} is not one complete {_EPISODE_HORIZON}-step trajectory"),
            replicate_id=replicate_id,
        )
        audit.require(
            all(row.get("terminated") is False for row in present_rows)
            and all(row.get("truncated") is (step == _EPISODE_HORIZON - 1) for step, row in enumerate(present_rows)),
            code="episode_horizon_flags_mismatch",
            message=(
                f"{replicate_id} episode {episode_id} does not terminate only "
                "through the exact 200-step TimeLimit truncation"
            ),
            replicate_id=replicate_id,
        )
        reset_seed = episode.get("reset_seed")
        first_observation = present_rows[0].get("pre_observation")
        try:
            if episode.get("task_id") == _TASK_IRRELEVANT:
                expected_reset = _independent_oscillator_reset(reset_seed)  # type: ignore[arg-type]
                reset_matches = (
                    isinstance(first_observation, list)
                    and len(first_observation) == 3
                    and np.array_equal(
                        np.asarray(first_observation, dtype=np.float64),
                        expected_reset,
                    )
                )
            else:
                expected_pendulum_reset = _independent_pendulum_reset(reset_seed)  # type: ignore[arg-type]
                reset_matches = (
                    isinstance(first_observation, list)
                    and len(first_observation) == 3
                    and np.array_equal(
                        np.asarray(first_observation, dtype=np.float32),
                        expected_pendulum_reset,
                    )
                )
        except (ArtifactAuditError, TypeError, ValueError):
            reset_matches = False
        audit.require(
            reset_matches,
            code="episode_reset_observation_mismatch",
            message=(
                f"{replicate_id} episode {episode_id} first pre-observation "
                "does not exactly replay from its sealed task reset seed"
            ),
            replicate_id=replicate_id,
        )
        trajectory_contiguous = True
        for previous, current in zip(
            present_rows,
            present_rows[1:],
            strict=False,
        ):
            previous_after = previous.get("next_observation")
            current_before = current.get("pre_observation")
            if (
                not isinstance(previous_after, list)
                or not isinstance(current_before, list)
                or len(previous_after) != 3
                or len(current_before) != 3
                or not np.array_equal(
                    np.asarray(previous_after, dtype=np.float32),
                    np.asarray(current_before, dtype=np.float32),
                )
            ):
                trajectory_contiguous = False
                break
        audit.require(
            trajectory_contiguous,
            code="episode_observation_chain_mismatch",
            message=(
                f"{replicate_id} episode {episode_id} next/pre observations do not form one contiguous real trajectory"
            ),
            replicate_id=replicate_id,
        )
        invariant_fields = ("episode_id", "run_id", "task_id", "split")
        audit.require(
            all(all(row.get(field) == episode.get(field) for field in invariant_fields) for row in present_rows),
            code="episode_transition_metadata_mismatch",
            message=f"{replicate_id} episode {episode_id} metadata differs from its transitions",
            replicate_id=replicate_id,
        )
        audit.require(
            all(
                row.get("model_version_at_action") == episode.get("model_version")
                and row.get("parameter_sha256_at_action") == episode.get("parameter_sha256")
                for row in present_rows
            ),
            code="episode_model_lineage_mismatch",
            message=f"{replicate_id} episode {episode_id} model lineage changes within the episode",
            replicate_id=replicate_id,
        )
        rewards = [_finite_float(row.get("reward")) for row in present_rows]
        episode_return = _finite_float(episode.get("return"))
        if episode_return is not None and all(value is not None for value in rewards):
            summed = math.fsum(float(value) for value in rewards if value is not None)
            reward_sum_valid = math.isclose(
                summed,
                episode_return,
                rel_tol=1e-10,
                abs_tol=1e-8 * max(1.0, abs(episode_return)),
            )
        else:
            reward_sum_valid = False
        audit.require(
            reward_sum_valid,
            code="episode_return_mismatch",
            message=f"{replicate_id} episode {episode_id} return is not the sum of transition rewards",
            replicate_id=replicate_id,
        )
        action_trace = {
            "applied": [row.get("applied_action") for row in present_rows],
            "intended": [row.get("intended_action") for row in present_rows],
        }
        try:
            action_digest = hashlib.sha256(_canonical_json_bytes(action_trace)).hexdigest()
        except (TypeError, ValueError):
            action_digest = ""
        audit.require(
            episode.get("action_trace_sha256") == action_digest,
            code="episode_action_trace_digest_mismatch",
            message=f"{replicate_id} episode {episode_id} action trace SHA-256 is not reproducible",
            replicate_id=replicate_id,
        )
        held_out = str(episode.get("split", "")).startswith(("predictive_validation_", "behavior_evaluation_"))
        audit.require(
            not held_out
            or (episode.get("learning_allowed") is False and episode.get("replay_writes_allowed") is False),
            code="held_out_episode_write_enabled",
            message=f"{replicate_id} held-out episode {episode_id} permits learning or replay writes",
            replicate_id=replicate_id,
        )

    audit.require(
        len(referenced_ids) == len(set(referenced_ids)) and set(referenced_ids) == set(transitions_by_id),
        code="episode_transition_coverage_mismatch",
        message=f"{replicate_id} episodes do not partition all transition rows exactly once",
        replicate_id=replicate_id,
    )
    for row in transition_rows:
        transition_id = str(row.get("transition_id", "<missing>"))
        task_id = row.get("task_id")
        intended = _finite_float(row.get("intended_action"))
        applied = _finite_float(row.get("applied_action"))
        expected_context = _TASK_CONTEXT.get(str(task_id))
        audit.require(
            expected_context is not None and row.get("task_context") == expected_context,
            code="transition_task_context_mismatch",
            message=f"{replicate_id} transition {transition_id} task context is inconsistent",
            replicate_id=replicate_id,
        )
        if intended is not None and applied is not None and task_id == _TASK_A:
            action_valid = math.isclose(applied, intended, rel_tol=0.0, abs_tol=1e-7)
        elif intended is not None and applied is not None and task_id == _TASK_B:
            action_valid = math.isclose(applied, -intended, rel_tol=0.0, abs_tol=1e-7)
        elif intended is not None and applied is not None and task_id == _TASK_IRRELEVANT:
            action_valid = applied == 0.0
        else:
            action_valid = False
        audit.require(
            action_valid,
            code="transition_applied_action_mismatch",
            message=f"{replicate_id} transition {transition_id} applied action violates task semantics",
            replicate_id=replicate_id,
        )
        raw_target = row.get("scaled_target")
        if (
            isinstance(raw_target, list)
            and len(raw_target) == 4
            and all(_finite_float(value) is not None for value in raw_target)
        ):
            valid_target = True
            target_array = np.asarray(raw_target, dtype="<f8")
            target_digest = hashlib.sha256(target_array.tobytes(order="C")).hexdigest()
            reward = _finite_float(row.get("reward"))
            reward_target_valid = reward is not None and struct.pack("<d", float(raw_target[3])) == struct.pack(
                "<d", reward / _TARGET_REWARD_SCALE
            )
            before = row.get("pre_observation")
            after = row.get("next_observation")
            if (
                isinstance(before, list)
                and len(before) == 3
                and all(_finite_float(value) is not None for value in before)
                and isinstance(after, list)
                and len(after) == 3
                and all(_finite_float(value) is not None for value in after)
                and reward is not None
            ):
                independently_scaled = np.asarray(
                    [
                        (float(after[0]) - float(before[0])) / 2.0,
                        (float(after[1]) - float(before[1])) / 2.0,
                        (float(after[2]) - float(before[2])) / 16.0,
                        reward / _TARGET_REWARD_SCALE,
                    ],
                    dtype=np.float64,
                )
                source_target_valid = target_array.tobytes(order="C") == independently_scaled.astype(
                    "<f8",
                    copy=False,
                ).tobytes(order="C")
            else:
                source_target_valid = False
        else:
            valid_target = False
            target_digest = ""
            reward_target_valid = False
            source_target_valid = False
        audit.require(
            valid_target and row.get("target_sha256") == target_digest,
            code="transition_target_digest_mismatch",
            message=f"{replicate_id} transition {transition_id} target SHA-256 is not reproducible",
            replicate_id=replicate_id,
        )
        audit.require(
            reward_target_valid,
            code="transition_scaled_reward_mismatch",
            message=f"{replicate_id} transition {transition_id} scaled reward is inconsistent",
            replicate_id=replicate_id,
        )
        audit.require(
            source_target_valid,
            code="transition_target_source_mismatch",
            message=(
                f"{replicate_id} transition {transition_id} scaled target is not "
                "derivable from pre/next observations and reward"
            ),
            replicate_id=replicate_id,
        )
        _audit_analytic_transition_dynamics(
            audit,
            row,
            replicate_id=replicate_id,
            transition_id=transition_id,
        )
    audit.limitation(_PENDULUM_RECONSTRUCTION_INDEPENDENCE_LIMITATION)
    return transitions_by_id


def _expected_policy_contract(
    *,
    task_id: object,
    split: object,
    condition: object,
) -> tuple[tuple[str, int], str, Mapping[str, int] | None] | None:
    if task_id == _TASK_IRRELEVANT and split == "collect_irrelevant" and condition == "collection_random":
        return ("irrelevant_collection_action", 0), "uniform_random", None
    if task_id == _TASK_IRRELEVANT and split == "predictive_validation_irrelevant" and condition == "validation_random":
        return (
            ("predictive_validation_irrelevant_action", 0),
            "uniform_random",
            None,
        )
    task_index = {_TASK_A: 0, _TASK_B: 1}.get(str(task_id))
    if task_index is None:
        return None
    task_suffix = "a" if task_index == 0 else "b"
    if split == f"collect_{task_suffix}" and condition == "collection_random":
        return ("collection_action", task_index), "uniform_random", None
    if split == f"predictive_validation_{task_suffix}" and condition == "validation_random":
        return (
            ("predictive_validation_action", task_index),
            "uniform_random",
            None,
        )
    if split != f"behavior_evaluation_{task_suffix}":
        return None
    if condition == "random":
        return ("random_policy_action", task_index), "uniform_random", None
    if condition == "oracle":
        controller_kind = "cem_oracle"
    elif condition in {
        "cold",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_a",
        "after_b_replay",
        "after_b_naive",
    }:
        controller_kind = "cem_learned"
    else:
        return None
    return (
        ("planner", 0),
        controller_kind,
        {
            "planning_horizon": _CEM_PLANNING_HORIZON,
            "optim_steps": _CEM_OPTIM_STEPS,
            "num_candidates": _CEM_NUM_CANDIDATES,
            "top_k": _CEM_TOP_K,
        },
    )


def _expected_reset_namespace(*, task_id: object, split: object) -> str | None:
    if task_id == _TASK_IRRELEVANT and split == "collect_irrelevant":
        return "collect_irrelevant_episode"
    if task_id == _TASK_IRRELEVANT and split == "predictive_validation_irrelevant":
        return "predictive_validation_irrelevant_episode"
    suffix = {_TASK_A: "a", _TASK_B: "b"}.get(str(task_id))
    if suffix is None:
        return None
    return {
        f"collect_{suffix}": f"collect_{suffix}_episode",
        f"predictive_validation_{suffix}": (f"predictive_validation_{suffix}_episode"),
        f"behavior_evaluation_{suffix}": (f"behavior_evaluation_{suffix}_episode"),
    }.get(str(split))


_CEM_RNG_DIGEST_CACHE: dict[tuple[str, int, int, int], tuple[str, str]] = {}


def _recompute_cem_rng_digests(
    *,
    seed: int,
    device: str,
    batch_size: int,
    planning_calls: int,
) -> tuple[str, str]:
    """Advance the exact TorchRL 0.13.3 CEM Gaussian draw schedule.

    CEMPlanner makes one ``torch.randn`` call with shape
    ``[batch, 64, 10, 1]`` on each of three optimization iterations.  The
    experiment batches paired episodes and invokes the planner once per real
    environment step.  Model rollout and elite selection consume no RNG.
    """

    if not 1 <= batch_size <= 32 or planning_calls != 200:
        raise ArtifactAuditError("CEM execution dimensions violate the sealed budget")
    key = (device, seed, batch_size, planning_calls)
    cached = _CEM_RNG_DIGEST_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import torch

        destination = torch.device(device)
        if destination.type not in {"cpu", "cuda"}:
            raise ArtifactAuditError(f"independent CEM RNG replay does not support device {device!r}")
        generator = torch.Generator(device=destination)
        generator.manual_seed(seed)

        def digest() -> str:
            state = generator.get_state().detach().cpu().numpy().tobytes()
            return hashlib.sha256(state).hexdigest()

        start_digest = digest()
        shape = (
            batch_size,
            _CEM_NUM_CANDIDATES,
            _CEM_PLANNING_HORIZON,
            1,
        )
        for _ in range(planning_calls):
            for _ in range(_CEM_OPTIM_STEPS):
                torch.randn(
                    shape,
                    device=destination,
                    dtype=torch.float32,
                    generator=generator,
                )
        result = (start_digest, digest())
    except ArtifactAuditError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        raise ArtifactAuditError(f"independent CEM RNG replay failed on {device!r}: {error}") from error
    _CEM_RNG_DIGEST_CACHE[key] = result
    return result


def _replay_cem_action_trace(
    *,
    observed_states: npt.NDArray[np.float32],
    context: float,
    seed: int,
    device: str,
    model_tensors: Mapping[str, npt.NDArray[np.float32]] | None,
) -> tuple[npt.NDArray[np.float32], str, str]:
    """Replay complete CEM decisions with an audit-local Torch implementation.

    ``observed_states`` is step-major ``[steps, paired episodes, 3]``.  A
    ``None`` model selects the analytic oracle; otherwise the seven retained
    snapshot's sealed five-member tensor map is used.  The implementation
    mirrors the public CEM algorithm (Gaussian sample, action projection,
    rollout, top-k, elite mean/std) but imports neither producer planning nor
    producer model code.
    """

    try:
        import torch
        from torch.nn import functional as torch_functional
    except ImportError as error:
        raise ArtifactAuditError("Torch is unavailable for CEM action replay") from error

    if (
        observed_states.ndim != 3
        or observed_states.shape[2] != 3
        or observed_states.shape[0] < 1
        or not 1 <= observed_states.shape[1] <= 32
        or not np.isfinite(observed_states).all()
        or context not in (0.0, 1.0)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ArtifactAuditError("CEM replay states, context, batch, or seed are invalid")
    destination = torch.device(device)
    if destination.type not in {"cpu", "cuda"}:
        raise ArtifactAuditError(f"independent CEM action replay does not support {device!r}")
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise ArtifactAuditError("declared CUDA CEM run cannot be replayed without CUDA")

    weights: tuple[list[Any], list[Any]] | None = None
    if model_tensors is not None:
        expected_names = {
            f"members.{member}.network.{layer}.{field}"
            for member in range(5)
            for layer in (0, 2, 4)
            for field in ("bias", "weight")
        }
        if set(model_tensors) != expected_names:
            raise ArtifactAuditError("CEM snapshot tensor set violates the sealed model")
        stacked_weights: list[Any] = []
        stacked_biases: list[Any] = []
        for layer in (0, 2, 4):
            stacked_weights.append(
                torch.stack(
                    [
                        torch.from_numpy(
                            np.asarray(
                                model_tensors[f"members.{member}.network.{layer}.weight"],
                                dtype=np.float32,
                            ).copy()
                        )
                        for member in range(5)
                    ],
                    dim=0,
                ).to(destination)
            )
            stacked_biases.append(
                torch.stack(
                    [
                        torch.from_numpy(
                            np.asarray(
                                model_tensors[f"members.{member}.network.{layer}.bias"],
                                dtype=np.float32,
                            ).copy()
                        )
                        for member in range(5)
                    ],
                    dim=0,
                ).to(destination)
            )
        weights = (stacked_weights, stacked_biases)

    generator = torch.Generator(device=destination)
    generator.manual_seed(seed)

    def generator_digest() -> str:
        return hashlib.sha256(generator.get_state().detach().cpu().numpy().tobytes()).hexdigest()

    def learned_step(state: Any, action: Any) -> tuple[Any, Any]:
        assert weights is not None
        stacked_weights, stacked_biases = weights
        observation = state[..., :3]
        task_context = state[..., 3:4]
        inputs = torch.cat(
            (
                observation / observation.new_tensor((1.0, 1.0, 8.0)),
                task_context,
                action / 2.0,
            ),
            dim=-1,
        )
        flattened = inputs.reshape(-1, 5)
        hidden = torch.matmul(
            flattened.unsqueeze(0),
            stacked_weights[0].transpose(1, 2),
        )
        hidden = torch_functional.silu(hidden + stacked_biases[0].unsqueeze(1))
        hidden = torch.bmm(hidden, stacked_weights[1].transpose(1, 2))
        hidden = torch_functional.silu(hidden + stacked_biases[1].unsqueeze(1))
        output = torch.bmm(hidden, stacked_weights[2].transpose(1, 2))
        output = output + stacked_biases[2].unsqueeze(1)
        output = output.reshape(5, *state.shape[:-1], 8)
        normalized_mean = output[..., :4].mean(dim=0)
        physical_target = normalized_mean * normalized_mean.new_tensor((2.0, 2.0, 16.0, _TARGET_REWARD_SCALE))
        proposed = observation + physical_target[..., :3]
        direction = proposed[..., :2]
        direction = direction / torch.linalg.vector_norm(
            direction,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        velocity = proposed[..., 2:3].clamp(
            -_PENDULUM_MAX_SPEED,
            _PENDULUM_MAX_SPEED,
        )
        next_state = torch.cat(
            (direction, velocity, task_context),
            dim=-1,
        )
        return next_state, physical_target[..., 3:4]

    def oracle_step(state: Any, action: Any) -> tuple[Any, Any]:
        observation = state[..., :3]
        task_context = state[..., 3:4]
        cosine = observation[..., 0]
        sine = observation[..., 1]
        angular_velocity = observation[..., 2]
        intended = action[..., 0]
        encoded_context = task_context[..., 0]
        theta = torch.atan2(sine, cosine)
        normalized_theta = torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi
        clipped = intended.clamp(-_PENDULUM_MAX_TORQUE, _PENDULUM_MAX_TORQUE)
        direction = torch.where(
            encoded_context >= 0.5,
            clipped.new_tensor(-1.0),
            clipped.new_tensor(1.0),
        )
        applied = clipped * direction
        cost = normalized_theta.square() + 0.1 * angular_velocity.square() + 0.001 * applied.square()
        acceleration = 15.0 * torch.sin(theta) + 3.0 * applied
        next_velocity = (angular_velocity + acceleration * _PENDULUM_TIME_STEP).clamp(
            -_PENDULUM_MAX_SPEED, _PENDULUM_MAX_SPEED
        )
        next_theta = theta + next_velocity * _PENDULUM_TIME_STEP
        next_observation = torch.stack(
            (
                torch.cos(next_theta),
                torch.sin(next_theta),
                next_velocity,
            ),
            dim=-1,
        )
        return (
            torch.cat((next_observation, task_context), dim=-1),
            -cost.unsqueeze(-1),
        )

    start_digest = generator_digest()
    actions_by_step: list[Any] = []
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cuda_matmul_precision = (
        str(torch.backends.cuda.matmul.fp32_precision)
        if destination.type == "cuda"
        else None
    )
    previous_cudnn_conv_precision = (
        str(torch.backends.cudnn.conv.fp32_precision)
        if destination.type == "cuda"
        else None
    )
    previous_cudnn_rnn_precision = (
        str(torch.backends.cudnn.rnn.fp32_precision)
        if destination.type == "cuda"
        else None
    )
    try:
        torch.use_deterministic_algorithms(True)
        if destination.type == "cuda":
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
        with torch.inference_mode():
            for raw_observation in observed_states:
                observation = torch.as_tensor(
                    raw_observation,
                    device=destination,
                    dtype=torch.float32,
                )
                task_context = torch.full(
                    (observation.shape[0], 1),
                    context,
                    device=destination,
                    dtype=torch.float32,
                )
                initial_state = torch.cat((observation, task_context), dim=-1)
                action_means = torch.zeros(
                    (
                        observation.shape[0],
                        1,
                        _CEM_PLANNING_HORIZON,
                        1,
                    ),
                    device=destination,
                    dtype=torch.float32,
                )
                action_stds = torch.ones_like(action_means)
                for _ in range(_CEM_OPTIM_STEPS):
                    candidate_actions = action_means + action_stds * torch.randn(
                        (
                            observation.shape[0],
                            _CEM_NUM_CANDIDATES,
                            _CEM_PLANNING_HORIZON,
                            1,
                        ),
                        device=destination,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    candidate_actions = candidate_actions.clamp(
                        -_PENDULUM_MAX_TORQUE,
                        _PENDULUM_MAX_TORQUE,
                    )
                    imagined_state = initial_state.unsqueeze(1).expand(
                        observation.shape[0],
                        _CEM_NUM_CANDIDATES,
                        4,
                    )
                    imagined_rewards: list[Any] = []
                    for horizon_index in range(_CEM_PLANNING_HORIZON):
                        if weights is None:
                            imagined_state, reward = oracle_step(
                                imagined_state,
                                candidate_actions[:, :, horizon_index],
                            )
                        else:
                            imagined_state, reward = learned_step(
                                imagined_state,
                                candidate_actions[:, :, horizon_index],
                            )
                        imagined_rewards.append(reward)
                    cumulative_reward = torch.stack(
                        imagined_rewards,
                        dim=2,
                    ).sum(dim=2, keepdim=True)
                    top_indices = cumulative_reward.topk(
                        _CEM_TOP_K,
                        dim=1,
                    ).indices
                    expanded_indices = top_indices.expand(
                        observation.shape[0],
                        _CEM_TOP_K,
                        _CEM_PLANNING_HORIZON,
                        1,
                    )
                    elite_actions = candidate_actions.gather(
                        1,
                        expanded_indices,
                    )
                    action_means = elite_actions.mean(dim=1, keepdim=True)
                    action_stds = elite_actions.std(dim=1, keepdim=True)
                actions_by_step.append(action_means[:, 0, 0, :].detach().cpu())
    except (RuntimeError, TypeError, ValueError) as error:
        raise ArtifactAuditError(f"CEM action replay failed: {error}") from error
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        if destination.type == "cuda":
            torch.backends.cuda.matmul.fp32_precision = previous_cuda_matmul_precision
            torch.backends.cudnn.conv.fp32_precision = previous_cudnn_conv_precision
            torch.backends.cudnn.rnn.fp32_precision = previous_cudnn_rnn_precision
    return (
        torch.stack(actions_by_step, dim=0).numpy().astype(np.float32, copy=False),
        start_digest,
        generator_digest(),
    )


def _audit_policy_runs(
    audit: _Audit,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
    transitions_by_id: Mapping[str, Mapping[str, object]],
    evaluated_checkpoints: Mapping[str, _EvaluatedCheckpoint],
    device: str,
) -> None:
    episode_rows = _mapping_rows(replicate.get("episodes"))
    episodes_by_id = {str(row.get("episode_id")): row for row in episode_rows if isinstance(row.get("episode_id"), str)}
    master_seed = replicate.get("master_seed")
    declared_seeds: dict[tuple[str, int], int] = {}
    declared_seed_counts: dict[str, int] = {}
    for namespace_row in _mapping_rows(replicate.get("derived_seeds")):
        namespace = namespace_row.get("namespace")
        values = namespace_row.get("values")
        if not isinstance(namespace, str) or namespace in declared_seed_counts or not isinstance(values, list):
            audit.error(
                "derived_seed_namespace_invalid",
                f"{replicate_id} has a malformed or duplicate derived-seed namespace",
                replicate_id=replicate_id,
            )
            continue
        declared_seed_counts[namespace] = len(values)
        for index, seed in enumerate(values):
            valid = (
                type(master_seed) is int and type(seed) is int and seed == _derive_seed(namespace, master_seed, index)
            )
            audit.require(
                valid,
                code="derived_seed_value_mismatch",
                message=f"{replicate_id} seed {namespace}[{index}] is not derivable",
                replicate_id=replicate_id,
            )
            if type(seed) is int:
                declared_seeds[(namespace, index)] = seed
    if declared_seed_counts != _EXPECTED_SEED_COUNTS:
        audit.gap(
            "derived_seed_schedule_incomplete",
            f"{replicate_id} does not retain the exact full seed namespace/count schedule.",
            evidence_needed=(
                "Every sealed namespace row, in full declared counts, with each value "
                "regenerated from the v1.16 experimental seed hash domain."
            ),
        )

    covered_episodes: list[str] = []
    policy_rows = _mapping_rows(replicate.get("policy_runs"))
    cem_rng_pairs: list[tuple[object, object]] = []
    cem_policy_count = 0
    cem_replay_count = 0
    cem_policy_keys: list[tuple[str, str]] = []
    for index, policy in enumerate(policy_rows):
        label = f"{replicate_id} policy run {index}"
        episode_ids = policy.get("episode_ids")
        reset_seeds = policy.get("reset_seeds")
        if (
            not isinstance(episode_ids, list)
            or any(not isinstance(value, str) for value in episode_ids)
            or not isinstance(reset_seeds, list)
        ):
            audit.error(
                "policy_run_episode_binding_invalid",
                f"{label} has malformed episode/reset-seed lists",
                replicate_id=replicate_id,
            )
            continue
        episodes = [episodes_by_id.get(str(episode_id)) for episode_id in episode_ids]
        complete = all(episode is not None for episode in episodes)
        audit.require(
            complete and len(set(episode_ids)) == len(episode_ids),
            code="policy_run_episode_binding_invalid",
            message=f"{label} references missing or duplicate episodes",
            replicate_id=replicate_id,
        )
        if not complete:
            continue
        present_episodes = [episode for episode in episodes if episode is not None]
        covered_episodes.extend(str(value) for value in episode_ids)
        invariant_fields = ("run_id", "task_id", "split", "condition", "checkpoint_id")
        audit.require(
            all(
                all(episode.get(field) == policy.get(field) for field in invariant_fields)
                for episode in present_episodes
            )
            and reset_seeds == [episode.get("reset_seed") for episode in present_episodes],
            code="policy_run_metadata_mismatch",
            message=f"{label} metadata/reset seeds differ from its episode rows",
            replicate_id=replicate_id,
        )
        reset_namespace = _expected_reset_namespace(
            task_id=policy.get("task_id"),
            split=policy.get("split"),
        )
        expected_reset_seeds = (
            [_derive_seed(reset_namespace, master_seed, reset_index) for reset_index in range(len(present_episodes))]
            if reset_namespace is not None and type(master_seed) is int
            else []
        )
        audit.require(
            bool(expected_reset_seeds) and reset_seeds == expected_reset_seeds,
            code="policy_reset_seed_schedule_mismatch",
            message=(
                f"{label} reset seeds do not exactly match the ordered "
                f"{reset_namespace or '<invalid>'} namespace schedule"
            ),
            replicate_id=replicate_id,
        )
        transition_rows: list[Mapping[str, object]] = []
        for episode in present_episodes:
            raw_ids = episode.get("transition_ids")
            if isinstance(raw_ids, list):
                transition_rows.extend(
                    transitions_by_id[str(identity)] for identity in raw_ids if str(identity) in transitions_by_id
                )
        intended = [row.get("intended_action") for row in transition_rows]
        applied = [row.get("applied_action") for row in transition_rows]
        trace = {
            "episode_ids": list(episode_ids),
            "intended": intended,
            "applied": applied,
        }
        trace_digest = hashlib.sha256(_canonical_json_bytes(trace)).hexdigest()
        audit.require(
            policy.get("action_count") == len(transition_rows) and policy.get("action_trace_sha256") == trace_digest,
            code="policy_run_action_trace_mismatch",
            message=f"{label} action count/hash is not reproducible from transitions",
            replicate_id=replicate_id,
        )
        namespace = policy.get("seed_namespace")
        seed_index = policy.get("seed_index")
        seed = policy.get("seed")
        audit.require(
            isinstance(namespace, str)
            and type(seed_index) is int
            and declared_seeds.get((namespace, seed_index)) == seed,
            code="policy_run_seed_binding_mismatch",
            message=f"{label} does not use its declared derived seed",
            replicate_id=replicate_id,
        )
        controller_kind = policy.get("controller_kind")
        budget = policy.get("planner_budget")
        expected_contract = _expected_policy_contract(
            task_id=policy.get("task_id"),
            split=policy.get("split"),
            condition=policy.get("condition"),
        )
        audit.require(
            expected_contract is not None
            and (namespace, seed_index) == expected_contract[0]
            and controller_kind == expected_contract[1]
            and budget == expected_contract[2],
            code="policy_run_contract_mismatch",
            message=(
                f"{label} split/condition, seed namespace, controller kind, or "
                "exact planner budget violates the sealed policy contract"
            ),
            replicate_id=replicate_id,
        )
        if controller_kind == "uniform_random" and type(seed) is int:
            rng = np.random.default_rng(seed)
            start_digest = hashlib.sha256(_canonical_json_bytes(rng.bit_generator.state)).hexdigest()
            expected_actions = [float(rng.uniform(-2.0, 2.0)) for _ in range(len(transition_rows))]
            end_digest = hashlib.sha256(_canonical_json_bytes(rng.bit_generator.state)).hexdigest()
            numeric_intended = [_finite_float(value) for value in intended]
            audit.require(
                policy.get("rng_start_sha256") == start_digest
                and policy.get("rng_end_sha256") == end_digest
                and all(value is not None for value in numeric_intended)
                and np.array_equal(
                    np.asarray(
                        [value for value in numeric_intended if value is not None],
                        dtype=np.float64,
                    ),
                    np.asarray(expected_actions, dtype=np.float64),
                ),
                code="random_policy_replay_mismatch",
                message=f"{label} actions/RNG states do not replay from the declared seed",
                replicate_id=replicate_id,
            )
        elif controller_kind in {"cem_learned", "cem_oracle"} and type(seed) is int:
            cem_policy_count += 1
            cem_policy_keys.append((str(policy.get("task_id")), str(policy.get("condition"))))
            step_counts = [
                len(raw_ids)
                for episode in present_episodes
                if isinstance((raw_ids := episode.get("transition_ids")), list)
            ]
            try:
                if (
                    len(step_counts) != len(present_episodes)
                    or any(count != _EPISODE_HORIZON for count in step_counts)
                    or policy.get("action_count") != _EPISODE_HORIZON * len(present_episodes)
                ):
                    raise ArtifactAuditError("CEM episodes do not encode 200 paired planning steps")
                episode_transitions: list[list[Mapping[str, object]]] = []
                for episode in present_episodes:
                    raw_ids = episode.get("transition_ids")
                    if not isinstance(raw_ids, list):
                        raise ArtifactAuditError("CEM episode has malformed transition IDs")
                    episode_rows = [
                        transitions_by_id[str(identity)] for identity in raw_ids if str(identity) in transitions_by_id
                    ]
                    if len(episode_rows) != _EPISODE_HORIZON:
                        raise ArtifactAuditError("CEM episode transition rows are incomplete")
                    episode_transitions.append(episode_rows)
                observed_states = np.asarray(
                    [
                        [
                            episode_transitions[episode_index][step]["pre_observation"]
                            for episode_index in range(len(episode_transitions))
                        ]
                        for step in range(_EPISODE_HORIZON)
                    ],
                    dtype=np.float32,
                )
                expected_cem_actions = np.asarray(
                    [
                        [
                            episode_transitions[episode_index][step]["intended_action"]
                            for episode_index in range(len(episode_transitions))
                        ]
                        for step in range(_EPISODE_HORIZON)
                    ],
                    dtype=np.float32,
                )[..., None]
                task_id = policy.get("task_id")
                if task_id not in _TASK_CONTEXT:
                    raise ArtifactAuditError("CEM policy task has no sealed context")
                model_tensors = None
                if controller_kind == "cem_learned":
                    condition = policy.get("condition")
                    checkpoint = evaluated_checkpoints.get(condition) if isinstance(condition, str) else None
                    if checkpoint is None:
                        raise ArtifactAuditError("learned CEM policy has no retained checkpoint bytes")
                    model_tensors = checkpoint.model_tensors
                    audit.require(
                        policy.get("checkpoint_id") == checkpoint.condition
                        and policy.get("controller_version") == f"wm001-sha256:{checkpoint.parameter_sha256}",
                        code="cem_policy_checkpoint_binding_mismatch",
                        message=(
                            f"{label} controller identity does not bind the retained learned model parameter bytes"
                        ),
                        replicate_id=replicate_id,
                    )
                else:
                    audit.require(
                        policy.get("controller_version") == "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1",
                        code="cem_oracle_identity_mismatch",
                        message=f"{label} does not identify the sealed analytic oracle",
                        replicate_id=replicate_id,
                    )
                replayed_actions, start_digest, end_digest = _replay_cem_action_trace(
                    observed_states=observed_states,
                    context=_TASK_CONTEXT[str(task_id)],
                    seed=seed,
                    device=device,
                    model_tensors=model_tensors,
                )
                rng_valid = (
                    policy.get("rng_start_sha256") == start_digest and policy.get("rng_end_sha256") == end_digest
                )
                audit.require(
                    np.array_equal(replayed_actions, expected_cem_actions),
                    code="cem_policy_action_replay_mismatch",
                    message=(
                        f"{label} complete action trace does not exactly replay "
                        "from observed states, retained checkpoint, sealed CEM "
                        "budget, and planner seed"
                    ),
                    replicate_id=replicate_id,
                )
                cem_replay_count += 1
            except (
                ArtifactAuditError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                rng_valid = False
                audit.error(
                    "cem_policy_action_replay_unavailable",
                    f"{label} cannot be independently replayed: {error}",
                    replicate_id=replicate_id,
                )
                audit.gap(
                    "cem_action_trace_replay_incomplete",
                    "At least one declared CEM trace could not be independently replayed.",
                    evidence_needed=(
                        "A runnable bound Torch/device environment plus complete 200-step "
                        "observations and retained model snapshot bytes for every learned "
                        "and oracle policy run."
                    ),
                )
            audit.require(
                rng_valid,
                code="cem_policy_rng_replay_mismatch",
                message=(
                    f"{label} initial/final CEM RNG state is not reproducible "
                    "from its seed and exact Torch draw schedule"
                ),
                replicate_id=replicate_id,
            )
            cem_rng_pairs.append(
                (
                    policy.get("rng_start_sha256"),
                    policy.get("rng_end_sha256"),
                )
            )
    expected_cem_policy_keys = {
        *(
            (_TASK_A, condition)
            for condition in (
                "cold",
                "frozen",
                "corrupted",
                "irrelevant",
                "after_a",
                "after_b_replay",
                "after_b_naive",
                "oracle",
            )
        ),
        *(
            (_TASK_B, condition)
            for condition in (
                "after_a",
                "after_b_replay",
                "after_b_naive",
                "oracle",
            )
        ),
    }
    cem_set_complete = (
        len(cem_policy_keys) == len(set(cem_policy_keys)) and set(cem_policy_keys) == expected_cem_policy_keys
    )
    if cem_policy_count == 0:
        audit.gap(
            "cem_action_trace_replay_absent",
            "The artifact contains no learned/oracle CEM policy run to replay.",
            evidence_needed=(
                "Every paired learned and oracle behavior run with 200-step observed "
                "states, exact actions, retained learned checkpoint bytes, and planner RNG."
            ),
        )
    elif cem_replay_count != cem_policy_count or not cem_set_complete:
        audit.gap(
            "cem_action_trace_replay_incomplete",
            "Not every declared CEM action trace was independently regenerated.",
            evidence_needed=("Successful exact replay for every learned and oracle CEM policy run."),
        )
    if cem_replay_count:
        audit.limitation(_CEM_REPLAY_INDEPENDENCE_LIMITATION)
    audit.require(
        not cem_rng_pairs or len(set(cem_rng_pairs)) == 1,
        code="paired_cem_rng_mismatch",
        message=(
            f"{replicate_id} paired learned/oracle CEM conditions do not share the same initial and final RNG states"
        ),
        replicate_id=replicate_id,
    )
    audit.require(
        len(covered_episodes) == len(set(covered_episodes)) and set(covered_episodes) == set(episodes_by_id),
        code="policy_run_episode_coverage_mismatch",
        message=f"{replicate_id} policy runs do not partition every real episode exactly once",
        replicate_id=replicate_id,
    )


def _decode_restart_evaluation(
    payload: bytes,
    *,
    label: str,
) -> Mapping[str, object]:
    """Decode one retained K7 continuation without producer parity helpers."""

    if not payload.endswith(b"\n"):
        raise ArtifactAuditError(f"{label} lacks its canonical trailing newline")
    raw = _json_without_duplicate_keys(payload[:-1], label=label)
    expected_fields = {
        "schema",
        "process_id",
        "checkpoint_manifest_sha256",
        "component_hashes",
        "model_version",
        "parameter_sha256",
        "boundary_state",
        "post_evaluation_custody",
        "tasks",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != expected_fields
        or raw.get("schema") != "prospect.wm001.restart-evaluation.v1"
        or payload != _canonical_json_bytes(raw) + b"\n"
    ):
        raise ArtifactAuditError(f"{label} is not a canonical restart evaluation")
    if (
        type(raw.get("process_id")) is not int
        or int(raw["process_id"]) < 1
        or not _is_sha256(raw.get("checkpoint_manifest_sha256"))
        or not isinstance(raw.get("model_version"), str)
        or not raw.get("model_version")
        or not _is_sha256(raw.get("parameter_sha256"))
    ):
        raise ArtifactAuditError(f"{label} has invalid process or model identity")

    component_hashes = raw.get("component_hashes")
    if (
        not isinstance(component_hashes, dict)
        or set(component_hashes) != set(_CANONICAL_COMPONENT_IDS)
        or any(not _is_sha256(component_hashes.get(component_id)) for component_id in _CANONICAL_COMPONENT_IDS)
    ):
        raise ArtifactAuditError(f"{label} has an invalid canonical component-hash set")

    custody_fields = {"experiences", "transitions", "updates", "replay_events"}

    def validate_custody(value: object, *, field: str) -> None:
        if (
            not isinstance(value, dict)
            or set(value) != custody_fields
            or any(type(value.get(name)) is not int or int(value[name]) < 0 for name in custody_fields)
        ):
            raise ArtifactAuditError(f"{label} {field} is invalid")

    boundary = raw.get("boundary_state")
    boundary_fields = {
        "snapshot_id",
        "agent_id",
        "captured_at",
        "belief_id",
        "latest_update_id",
        "configuration_version",
        "memory_version",
        "knowledge_version",
        "model_version",
        "representation_version",
        "policy_version",
        "custody",
    }
    if not isinstance(boundary, dict) or set(boundary) != boundary_fields:
        raise ArtifactAuditError(f"{label} boundary state has an unexpected field set")
    if any(
        not isinstance(boundary.get(field), str) or not boundary.get(field)
        for field in boundary_fields - {"captured_at", "custody"}
    ):
        raise ArtifactAuditError(f"{label} boundary state has an invalid identity")
    captured_at = boundary.get("captured_at")
    if (
        not isinstance(captured_at, list)
        or len(captured_at) != 2
        or not isinstance(captured_at[0], str)
        or not captured_at[0]
        or type(captured_at[1]) is not int
        or int(captured_at[1]) < 0
    ):
        raise ArtifactAuditError(f"{label} boundary timestamp is invalid")
    validate_custody(boundary.get("custody"), field="boundary custody")
    validate_custody(
        raw.get("post_evaluation_custody"),
        field="post-evaluation custody",
    )

    tasks = raw.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 2
        or [task.get("task_id") for task in tasks if isinstance(task, Mapping)] != [_TASK_A, _TASK_B]
    ):
        raise ArtifactAuditError(f"{label} must contain task A then task B")
    for task_index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != {
            "task_id",
            "reset_seed",
            "return",
            "actions",
            "predictions",
            "identities",
        }:
            raise ArtifactAuditError(f"{label} task {task_index} has an unexpected field set")
        actions = task.get("actions")
        predictions = task.get("predictions")
        identities = task.get("identities")
        if (
            type(task.get("reset_seed")) is not int
            or int(task["reset_seed"]) < 0
            or _finite_float(task.get("return")) is None
            or not isinstance(actions, list)
            or len(actions) != _EPISODE_HORIZON
            or any(
                _finite_float(action) is None
                or float(action) < -_PENDULUM_MAX_TORQUE
                or float(action) > _PENDULUM_MAX_TORQUE
                for action in actions
            )
            or not isinstance(predictions, list)
            or len(predictions) != _EPISODE_HORIZON
            or not isinstance(identities, list)
            or len(identities) != _EPISODE_HORIZON
        ):
            raise ArtifactAuditError(f"{label} task {task_index} has invalid scalar or trace dimensions")
        for step_index, prediction in enumerate(predictions):
            if not isinstance(prediction, dict) or set(prediction) != {
                "member_means",
                "member_variances",
            }:
                raise ArtifactAuditError(f"{label} task {task_index} prediction {step_index} is malformed")
            for field, require_nonnegative in (
                ("member_means", False),
                ("member_variances", True),
            ):
                matrix = prediction.get(field)
                if (
                    not isinstance(matrix, list)
                    or len(matrix) != 5
                    or any(not isinstance(member, list) or len(member) != 3 for member in matrix)
                    or any(
                        _finite_float(item) is None or (require_nonnegative and float(item) < 0.0)
                        for member in matrix
                        for item in member
                    )
                ):
                    raise ArtifactAuditError(f"{label} task {task_index} prediction {step_index} {field} is invalid")
        for step_index, identity_row in enumerate(identities):
            if (
                not isinstance(identity_row, list)
                or len(identity_row) != 4
                or any(not isinstance(identity, str) or not identity for identity in identity_row)
            ):
                raise ArtifactAuditError(f"{label} task {task_index} identity row {step_index} is invalid")
    return raw


def _restart_trace_differences(
    live: Mapping[str, object],
    restored: Mapping[str, object],
) -> dict[str, object]:
    """Recompute K7 differences from both retained traces."""

    live_components = cast(Mapping[str, object], live["component_hashes"])
    restored_components = cast(Mapping[str, object], restored["component_hashes"])
    component_mismatches = [
        component_id
        for component_id in _CANONICAL_COMPONENT_IDS
        if live_components.get(component_id) != restored_components.get(component_id)
    ]
    identity_mismatches = int(live.get("boundary_state") != restored.get("boundary_state"))
    identity_mismatches += int(live.get("post_evaluation_custody") != restored.get("post_evaluation_custody"))
    prediction_difference = 0.0
    action_difference = 0.0
    return_difference = 0.0
    live_tasks = cast(list[Mapping[str, object]], live["tasks"])
    restored_tasks = cast(list[Mapping[str, object]], restored["tasks"])
    for live_task, restored_task in zip(live_tasks, restored_tasks, strict=True):
        if (
            live_task.get("task_id"),
            live_task.get("reset_seed"),
        ) != (
            restored_task.get("task_id"),
            restored_task.get("reset_seed"),
        ):
            raise ArtifactAuditError("restart traces use different task/reset assignments")
        live_actions = np.asarray(live_task["actions"], dtype=np.float64)
        restored_actions = np.asarray(restored_task["actions"], dtype=np.float64)
        action_difference = max(
            action_difference,
            float(
                np.max(
                    np.abs(live_actions - restored_actions),
                    initial=0.0,
                )
            ),
        )
        live_predictions = np.asarray(
            [
                [prediction["member_means"], prediction["member_variances"]]
                for prediction in cast(
                    list[Mapping[str, object]],
                    live_task["predictions"],
                )
            ],
            dtype=np.float64,
        )
        restored_predictions = np.asarray(
            [
                [prediction["member_means"], prediction["member_variances"]]
                for prediction in cast(
                    list[Mapping[str, object]],
                    restored_task["predictions"],
                )
            ],
            dtype=np.float64,
        )
        prediction_difference = max(
            prediction_difference,
            float(
                np.max(
                    np.abs(live_predictions - restored_predictions),
                    initial=0.0,
                )
            ),
        )
        return_difference = max(
            return_difference,
            abs(float(cast(int | float, live_task["return"])) - float(cast(int | float, restored_task["return"]))),
        )
        live_identities = cast(list[list[str]], live_task["identities"])
        restored_identities = cast(list[list[str]], restored_task["identities"])
        identity_mismatches += sum(
            int(before != after)
            for before, after in zip(
                live_identities,
                restored_identities,
                strict=True,
            )
        )
    live_pid = cast(int, live["process_id"])
    restored_pid = cast(int, restored["process_id"])
    return {
        "checkpoint_manifest_sha256": live["checkpoint_manifest_sha256"],
        "original_process_id": live_pid,
        "restored_process_id": restored_pid,
        "fresh_process": live_pid != restored_pid,
        "component_hash_mismatches": component_mismatches,
        "identity_or_lineage_mismatches": identity_mismatches,
        "prediction_max_abs_difference": prediction_difference,
        "action_max_abs_difference": action_difference,
        "episode_return_max_abs_difference": return_difference,
    }


def _validate_restart_restore_runtime(
    *,
    root: Path,
    reference: object,
    replicate_id: str,
    execution: Mapping[str, object],
    producer_bootstrap_sha256: str,
    expected_branch: str,
    binding_runtime: Mapping[str, object] | None,
    dependencies: Mapping[str, object] | None,
    source: Mapping[str, object] | None,
) -> None:
    """Reopen and bind the fresh restore interpreter to its parent closure."""

    _require_restart_runtime_branch(
        expected_branch,
        binding_runtime=binding_runtime,
        dependencies=dependencies,
        source=source,
    )
    runtime_fields = {
        "schema",
        "python_executable",
        "python_executable_sha256",
        "python_version",
        "python_flags",
        "process_environment",
        "package_root",
        "package_root_inventory",
        "package_ownership",
        "standard_library",
        "runtime_seal_sha256",
        "runtime_seal_descriptor_custody",
        "bootstrap_source_sha256",
        "bootstrap_descriptor_custody",
        "deterministic_algorithms",
    }
    outer_fields = {*runtime_fields, "bytes", "sha256", "filename"}
    expected_filename = f"{replicate_id}-restore-runtime.json"
    if (
        not isinstance(reference, Mapping)
        or set(reference) != outer_fields
        or reference.get("filename") != expected_filename
        or type(reference.get("bytes")) is not int
        or cast(int, reference.get("bytes")) < 1
        or not _is_sha256(reference.get("sha256"))
    ):
        raise ArtifactAuditError("restart restore-runtime reference is malformed")
    path = _resolve_artifact_file(
        root,
        reference.get("filename"),
        label=f"{replicate_id} restore runtime",
    )
    payload, observed_bytes, observed_digest, _ = _read_stable_regular_file(
        path,
        4 << 20,
        label=f"{replicate_id} restore runtime",
        capture_payload=True,
    )
    assert payload is not None
    if (
        path.is_symlink()
        or path.lstat().st_nlink != 1
        or observed_bytes != reference.get("bytes")
        or observed_digest != reference.get("sha256")
    ):
        raise ArtifactAuditError("restart restore-runtime bytes differ from their reference")
    body = {field: reference.get(field) for field in runtime_fields}
    if payload != _canonical_json_bytes(body) + b"\n":
        raise ArtifactAuditError("restart restore-runtime payload is not the exact canonical body")

    expected_executable = sys.executable
    expected_executable_digest = _sha256_file(Path(sys.executable).resolve(strict=True))
    execution_package_roots = execution.get("package_roots")
    execution_standard_library = execution.get("standard_library")
    execution_package_ownership = execution.get("package_ownership")
    if (
        not isinstance(execution_package_roots, list)
        or len(execution_package_roots) != 1
        or not isinstance(execution_package_roots[0], Mapping)
        or not isinstance(execution_standard_library, Mapping)
        or not isinstance(execution_package_ownership, Mapping)
    ):
        raise ArtifactAuditError("restart restore runtime parent root closure is malformed")
    expected_package_root_inventory: Mapping[str, object] = execution_package_roots[0]
    expected_standard_library: Mapping[str, object] = execution_standard_library
    expected_package_root: object = expected_package_root_inventory.get("path")
    if dependencies is not None:
        package_roots = dependencies.get("package_roots")
        standard_library = dependencies.get("standard_library")
        if (
            not isinstance(package_roots, list)
            or len(package_roots) != 1
            or not isinstance(package_roots[0], Mapping)
            or not isinstance(standard_library, Mapping)
        ):
            raise ArtifactAuditError("restart restore runtime has no singular bound root closure")
        expected_package_root = package_roots[0].get("path")
        expected_package_root_inventory = package_roots[0]
        expected_standard_library = standard_library
        expected_executable = cast(
            str,
            dependencies.get("python_executable"),
        )
        expected_executable_digest = cast(
            str,
            dependencies.get("python_executable_sha256"),
        )

    if not _is_sha256(producer_bootstrap_sha256):
        raise ArtifactAuditError("captured producer bootstrap identity is malformed")
    expected_bootstrap_digest = producer_bootstrap_sha256
    if source is not None:
        implementation = source.get("implementation_files")
        bootstrap_rows = (
            [
                row
                for row in implementation
                if isinstance(row, Mapping) and row.get("path") == "bench/world_model_lifecycle/producer_bootstrap.py"
            ]
            if isinstance(implementation, list)
            else []
        )
        if len(bootstrap_rows) != 1:
            raise ArtifactAuditError("restart restore runtime lacks one bound bootstrap source")
        bootstrap_snapshot = root / "source" / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"
        bootstrap_payload = _read_bounded(
            bootstrap_snapshot,
            _MAX_SOURCE_FILE_BYTES,
            label="bound producer bootstrap source",
        )
        if (
            bootstrap_rows[0].get("bytes") != len(bootstrap_payload)
            or bootstrap_rows[0].get("sha256") != hashlib.sha256(bootstrap_payload).hexdigest()
            or bootstrap_rows[0].get("sha256") != producer_bootstrap_sha256
        ):
            raise ArtifactAuditError(
                "captured producer bootstrap differs from the formal source snapshot"
            )
        expected_bootstrap_digest = cast(
            str,
            bootstrap_rows[0].get("sha256"),
        )

    expected_flags = execution.get("python_flags")
    expected_environment = execution.get("process_environment")
    expected_runtime_seal_digest = execution.get("runtime_seal_sha256")

    def valid_inventory(row: Mapping[str, object]) -> bool:
        path = row.get("path")
        canonical_directory = False
        if isinstance(path, str) and path:
            candidate = Path(path)
            try:
                canonical_directory = (
                    candidate.is_absolute()
                    and candidate.resolve(strict=True) == candidate
                    and candidate.is_dir()
                    and not candidate.is_symlink()
                )
            except OSError:
                canonical_directory = False
        return (
            set(row)
            == {
                "path",
                "semantics_id",
                "file_count",
                "directory_count",
                "total_bytes",
                "inventory_sha256",
            }
            and isinstance(path, str)
            and bool(path)
            and os.path.abspath(path) == path
            and canonical_directory
            and row.get("semantics_id")
            in {
                "prospect.wm001.package-root.v2",
                "prospect.wm001.standard-library.v2",
            }
            and type(row.get("file_count")) is int
            and cast(int, row.get("file_count")) >= 1
            and type(row.get("directory_count")) is int
            and cast(int, row.get("directory_count")) >= 0
            and type(row.get("total_bytes")) is int
            and cast(int, row.get("total_bytes")) >= 1
            and _is_sha256(row.get("inventory_sha256"))
        )

    if (
        not isinstance(expected_flags, Mapping)
        or dict(expected_flags) != dict(_PREBINDING_PRODUCER_FLAGS)
        or not isinstance(expected_environment, Mapping)
        or not isinstance(expected_executable, str)
        or not _is_sha256(expected_executable_digest)
        or execution.get("python_executable") != expected_executable
        or execution.get("python_executable_sha256") != expected_executable_digest
        or execution.get("python_version") != platform.python_version()
        or not _is_sha256(expected_runtime_seal_digest)
        or not isinstance(expected_package_root, str)
        or not Path(expected_package_root).is_absolute()
        or not valid_inventory(expected_package_root_inventory)
        or not valid_inventory(expected_standard_library)
        or expected_package_root_inventory.get("path") != expected_package_root
        or expected_package_root_inventory.get("semantics_id") != "prospect.wm001.package-root.v2"
        or expected_standard_library.get("semantics_id") != "prospect.wm001.standard-library.v2"
        or dict(execution_package_roots[0]) != dict(expected_package_root_inventory)
        or dict(execution_standard_library) != dict(expected_standard_library)
        or execution.get("runtime_seal_descriptor_custody") is not True
        or execution.get("producer_bootstrap_sha256") != expected_bootstrap_digest
        or execution.get("bootstrap_descriptor_custody") is not True
        or execution.get("deterministic_algorithms") is not True
        or (
            binding_runtime is not None
            and (
                binding_runtime.get("python_flags") != expected_flags
                or binding_runtime.get("process_environment") != expected_environment
                or binding_runtime.get("deterministic_algorithms") is not True
            )
        )
    ):
        raise ArtifactAuditError("restart restore runtime parent/binding closure is invalid")
    expected_body = {
        "schema": "prospect.wm001.restart-runtime.v2",
        "python_executable": expected_executable,
        "python_executable_sha256": expected_executable_digest,
        "python_version": platform.python_version(),
        "python_flags": dict(expected_flags),
        "process_environment": dict(expected_environment),
        "package_root": expected_package_root,
        "package_root_inventory": dict(expected_package_root_inventory),
        "package_ownership": dict(execution_package_ownership),
        "standard_library": dict(expected_standard_library),
        "runtime_seal_sha256": expected_runtime_seal_digest,
        "runtime_seal_descriptor_custody": True,
        "bootstrap_source_sha256": expected_bootstrap_digest,
        "bootstrap_descriptor_custody": True,
        "deterministic_algorithms": True,
    }
    if body != expected_body:
        raise ArtifactAuditError("restart restore runtime differs from the parent and formal binding")


def _restart_runtime_conformance_support_files() -> list[str]:
    """Return every private-capture support, excluding the auditor source."""

    observed: list[str] = []
    try:
        for directory, directory_names, filenames in os.walk(
            HERE,
            topdown=True,
            followlinks=False,
        ):
            directory_names.sort()
            filenames.sort()
            current = Path(directory)
            for name in directory_names:
                path = current / name
                if path.is_symlink() or not path.is_dir():
                    raise ArtifactAuditError(
                        "restart-runtime conformance capture contains a non-directory"
                    )
            for name in filenames:
                path = current / name
                relative = path.relative_to(HERE).as_posix()
                if path.is_symlink() or not path.is_file():
                    raise ArtifactAuditError(
                        "restart-runtime conformance capture contains a non-regular file"
                    )
                if relative != "artifact_audit.py":
                    observed.append(relative)
    except OSError as error:
        raise ArtifactAuditError(
            "restart-runtime conformance capture cannot be enumerated"
        ) from error
    return sorted(observed)


def _require_restart_runtime_support_set(observed: Sequence[str]) -> None:
    if tuple(observed) != _RESTART_RUNTIME_SUPPORT_FILES:
        raise ArtifactAuditError(
            "restart-runtime conformance support set differs from the exact outcome support"
        )


def _require_restart_runtime_branch(
    branch: str,
    *,
    binding_runtime: Mapping[str, object] | None,
    dependencies: Mapping[str, object] | None,
    source: Mapping[str, object] | None,
) -> None:
    if branch == "development":
        if any(value is not None for value in (binding_runtime, dependencies, source)):
            raise ArtifactAuditError(
                "development restart-runtime conformance received formal source custody"
            )
        return
    if branch == "formal":
        if not all(
            isinstance(value, Mapping)
            for value in (binding_runtime, dependencies, source)
        ):
            raise ArtifactAuditError(
                "formal restart-runtime conformance lacks bound source custody"
            )
        return
    raise ArtifactAuditError("restart-runtime conformance branch is invalid")


def _restart_runtime_conformance_inventory(
    path: Path,
    *,
    semantics_id: str,
) -> dict[str, object]:
    canonical = path.resolve(strict=True)
    if canonical != path or not path.is_dir() or path.is_symlink():
        raise ArtifactAuditError(
            "restart-runtime conformance inventory root is absent or aliased"
        )
    identity = {
        "semantics_id": semantics_id,
        "path_kind": "result-free-conformance",
    }
    return {
        "path": str(path),
        "semantics_id": semantics_id,
        "file_count": 1,
        "directory_count": 0,
        "total_bytes": 1,
        "inventory_sha256": hashlib.sha256(
            _canonical_json_bytes(identity)
        ).hexdigest(),
    }


def audit_restart_runtime_conformance(
    producer_bootstrap: str | Path,
    *,
    expected_producer_bootstrap_sha256: str,
) -> dict[str, object]:
    """Exercise both restart-runtime custody branches without outcome evidence."""

    bootstrap_path = Path(producer_bootstrap)
    try:
        bootstrap_payload, bootstrap_bytes, bootstrap_sha256, _ = (
            _read_stable_regular_file(
                bootstrap_path,
                _MAX_SOURCE_FILE_BYTES,
                label="captured producer bootstrap support",
                capture_payload=True,
            )
        )
        assert bootstrap_payload is not None
        if (
            not _is_sha256(expected_producer_bootstrap_sha256)
            or bootstrap_sha256
            != expected_producer_bootstrap_sha256
        ):
            raise ArtifactAuditError(
                "captured producer bootstrap differs from its independently bound identity"
            )
        if (
            bootstrap_path.resolve(strict=True)
            != (HERE / "producer_bootstrap.py").resolve(strict=True)
        ):
            raise ArtifactAuditError(
                "restart-runtime conformance did not receive its declared captured bootstrap"
            )
        observed_support = _restart_runtime_conformance_support_files()
        _require_restart_runtime_support_set(observed_support)
        protocol_payload = _read_bounded(
            HERE / "protocol.json",
            16 << 20,
            label="captured protocol support",
        )
        schema_payload = _read_bounded(
            RESULT_SCHEMA_PATH,
            16 << 20,
            label="captured raw-result schema support",
        )
    except (ArtifactAuditError, OSError):
        return {
            "schema": _RESTART_RUNTIME_CONFORMANCE_SCHEMA,
            "protocol_version": "1.16.0",
            "support_files": list(_RESTART_RUNTIME_SUPPORT_FILES),
            "branches": {
                "development": {"passed": False},
                "formal": {"passed": False},
            },
            "negative_cases": [],
            "failure_code": "captured_support_invalid",
            "passed": False,
        }

    package_roots = [
        Path(value)
        for value in sys.path
        if value
        and Path(value).is_absolute()
        and Path(value).is_dir()
        and not Path(value).is_symlink()
    ]
    if not package_roots:
        raise ArtifactAuditError(
            "restart-runtime conformance has no explicit import root"
        )
    package_root = package_roots[0].resolve(strict=True)
    standard_library_root = Path(os.__file__).resolve(strict=True).parent
    package_inventory = _restart_runtime_conformance_inventory(
        package_root,
        semantics_id="prospect.wm001.package-root.v2",
    )
    standard_library = _restart_runtime_conformance_inventory(
        standard_library_root,
        semantics_id="prospect.wm001.standard-library.v2",
    )
    executable = Path(sys.executable).resolve(strict=True)
    executable_sha256 = _sha256_file(executable)
    runtime_seal_sha256 = hashlib.sha256(
        b"prospect.wm001.restart-runtime-conformance.v1\0"
        + bootstrap_payload
    ).hexdigest()
    producer_environment = dict(sorted({**os.environ, "PATH": "/usr/bin:/bin"}.items()))
    package_ownership = {
        "schema": "prospect.wm001.result-free-package-ownership.v1",
        "identity_sha256": hashlib.sha256(
            _canonical_json_bytes(
                {
                    "package_root_semantics": package_inventory["semantics_id"],
                    "standard_library_semantics": standard_library["semantics_id"],
                }
            )
        ).hexdigest(),
    }
    execution: dict[str, object] = {
        "python_executable": sys.executable,
        "python_executable_sha256": executable_sha256,
        "python_version": platform.python_version(),
        "python_flags": dict(_PREBINDING_PRODUCER_FLAGS),
        "process_environment": producer_environment,
        "package_roots": [package_inventory],
        "package_ownership": package_ownership,
        "standard_library": standard_library,
        "runtime_seal_sha256": runtime_seal_sha256,
        "runtime_seal_descriptor_custody": True,
        "producer_bootstrap_sha256": bootstrap_sha256,
        "bootstrap_descriptor_custody": True,
        "deterministic_algorithms": True,
    }
    binding_runtime: dict[str, object] = {
        "python_flags": dict(_PREBINDING_PRODUCER_FLAGS),
        "process_environment": producer_environment,
        "deterministic_algorithms": True,
    }
    dependencies: dict[str, object] = {
        "python_executable": sys.executable,
        "python_executable_sha256": executable_sha256,
        "package_roots": [package_inventory],
        "standard_library": standard_library,
    }
    source: dict[str, object] = {
        "implementation_files": [
            {
                "path": (
                    "bench/world_model_lifecycle/producer_bootstrap.py"
                ),
                "bytes": bootstrap_bytes,
                "sha256": bootstrap_sha256,
            }
        ],
    }

    with tempfile.TemporaryDirectory(
        prefix="prospect-wm001-restart-conformance-",
    ) as temporary:
        root = Path(temporary).resolve(strict=True)
        replicate_id = "wm001-result-free-conformance"
        snapshot = (
            root
            / "source"
            / "bench"
            / "world_model_lifecycle"
            / "producer_bootstrap.py"
        )
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(bootstrap_payload)
        body = {
            "schema": "prospect.wm001.restart-runtime.v2",
            "python_executable": sys.executable,
            "python_executable_sha256": executable_sha256,
            "python_version": platform.python_version(),
            "python_flags": dict(_PREBINDING_PRODUCER_FLAGS),
            "process_environment": producer_environment,
            "package_root": str(package_root),
            "package_root_inventory": package_inventory,
            "package_ownership": package_ownership,
            "standard_library": standard_library,
            "runtime_seal_sha256": runtime_seal_sha256,
            "runtime_seal_descriptor_custody": True,
            "bootstrap_source_sha256": bootstrap_sha256,
            "bootstrap_descriptor_custody": True,
            "deterministic_algorithms": True,
        }
        payload = _canonical_json_bytes(body) + b"\n"
        filename = f"{replicate_id}-restore-runtime.json"
        (root / filename).write_bytes(payload)
        reference = {
            **body,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": filename,
        }

        _validate_restart_restore_runtime(
            root=root,
            reference=reference,
            replicate_id=replicate_id,
            execution=execution,
            producer_bootstrap_sha256=bootstrap_sha256,
            expected_branch="development",
            binding_runtime=None,
            dependencies=None,
            source=None,
        )
        _validate_restart_restore_runtime(
            root=root,
            reference=reference,
            replicate_id=replicate_id,
            execution=execution,
            producer_bootstrap_sha256=bootstrap_sha256,
            expected_branch="formal",
            binding_runtime=binding_runtime,
            dependencies=dependencies,
            source=source,
        )

        negative_cases: list[dict[str, object]] = []

        def require_rejection(case_id: str, callback: Any) -> None:
            try:
                callback()
            except (ArtifactAuditError, OSError):
                negative_cases.append(
                    {
                        "case_id": case_id,
                        "rejected": True,
                    }
                )
            else:
                negative_cases.append(
                    {
                        "case_id": case_id,
                        "rejected": False,
                    }
                )

        require_rejection(
            "missing-bootstrap-support",
            lambda: _require_restart_runtime_support_set(
                [
                    "protocol.json",
                    "schemas/raw-result.schema.json",
                ]
            ),
        )
        require_rejection(
            "extra-bootstrap-support",
            lambda: _require_restart_runtime_support_set(
                [
                    *_RESTART_RUNTIME_SUPPORT_FILES,
                    "unexpected.py",
                ]
            ),
        )
        require_rejection(
            "mutated-bootstrap-identity",
            lambda: _validate_restart_restore_runtime(
                root=root,
                reference=reference,
                replicate_id=replicate_id,
                execution=execution,
                producer_bootstrap_sha256=("f" * 64),
                expected_branch="development",
                binding_runtime=None,
                dependencies=None,
                source=None,
            ),
        )
        require_rejection(
            "development-formal-branch-substitution",
            lambda: _validate_restart_restore_runtime(
                root=root,
                reference=reference,
                replicate_id=replicate_id,
                execution=execution,
                producer_bootstrap_sha256=bootstrap_sha256,
                expected_branch="development",
                binding_runtime=binding_runtime,
                dependencies=dependencies,
                source=source,
            ),
        )
        require_rejection(
            "formal-development-branch-substitution",
            lambda: _validate_restart_restore_runtime(
                root=root,
                reference=reference,
                replicate_id=replicate_id,
                execution=execution,
                producer_bootstrap_sha256=bootstrap_sha256,
                expected_branch="formal",
                binding_runtime=None,
                dependencies=None,
                source=None,
            ),
        )

    report: dict[str, object] = {
        "schema": _RESTART_RUNTIME_CONFORMANCE_SCHEMA,
        "protocol_version": "1.16.0",
        "support_files": [
            {
                "path": "producer_bootstrap.py",
                "bytes": bootstrap_bytes,
                "sha256": bootstrap_sha256,
            },
            {
                "path": "protocol.json",
                "bytes": len(protocol_payload),
                "sha256": hashlib.sha256(protocol_payload).hexdigest(),
            },
            {
                "path": "schemas/raw-result.schema.json",
                "bytes": len(schema_payload),
                "sha256": hashlib.sha256(schema_payload).hexdigest(),
            },
        ],
        "branches": {
            "development": {
                "source_block_present": False,
                "captured_bootstrap_bound": True,
                "passed": True,
            },
            "formal": {
                "source_block_present": True,
                "source_snapshot_bound": True,
                "captured_bootstrap_bound": True,
                "passed": True,
            },
        },
        "negative_cases": negative_cases,
        "failure_code": None,
        "passed": all(
            row.get("rejected") is True
            for row in negative_cases
        ),
    }
    return report


def _audit_restart_parity_evidence(
    audit: _Audit,
    root: Path,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
    launcher_process_id: object,
    execution: Mapping[str, object],
    producer_bootstrap_sha256: str,
    binding_runtime: Mapping[str, object] | None,
    dependencies: Mapping[str, object] | None,
    source: Mapping[str, object] | None,
) -> None:
    """Reopen both K7 paths and independently reproduce the parity row."""

    parity = replicate.get("restart_parity")
    if not isinstance(parity, Mapping):
        audit.gap(
            "restart_original_and_restored_trace_unavailable",
            "K7 has no restart-parity row or retained live/fresh-process trace pair.",
            evidence_needed=(
                "Content-addressed live-continuation and restored-continuation "
                "files whose exact differences can be independently recomputed."
            ),
        )
        return
    live_reference = parity.get("live_evaluation")
    restored_reference = parity.get("restored_evaluation")
    if not isinstance(live_reference, Mapping) or not isinstance(
        restored_reference,
        Mapping,
    ):
        audit.gap(
            "restart_original_and_restored_trace_unavailable",
            (
                "K7 reports derived parity values but does not retain both the "
                "original live continuation and the fresh-process continuation."
            ),
            evidence_needed=(
                "Content-addressed live-continuation and restored-continuation "
                "files whose exact differences can be independently recomputed."
            ),
        )
        return
    try:
        _validate_restart_restore_runtime(
            root=root,
            reference=parity.get("restore_runtime"),
            replicate_id=replicate_id,
            execution=execution,
            producer_bootstrap_sha256=producer_bootstrap_sha256,
            expected_branch=(
                "formal"
                if source is not None
                else "development"
            ),
            binding_runtime=binding_runtime,
            dependencies=dependencies,
            source=source,
        )
        audit.passed_checks += 1
        expected_media_type = "application/vnd.prospect.wm001.restart-evaluation+json"
        if (
            live_reference.get("media_type") != expected_media_type
            or restored_reference.get("media_type") != expected_media_type
        ):
            raise ArtifactAuditError("restart evaluation media type differs from the sealed format")
        _, live_payload = _verify_file_reference(
            root,
            live_reference,
            limit=_MAX_RESTART_EVALUATION_BYTES,
            label=f"{replicate_id} live restart evaluation",
        )
        _, restored_payload = _verify_file_reference(
            root,
            restored_reference,
            limit=_MAX_RESTART_EVALUATION_BYTES,
            label=f"{replicate_id} restored restart evaluation",
        )
        live = _decode_restart_evaluation(
            live_payload,
            label=f"{replicate_id} live restart evaluation",
        )
        restored = _decode_restart_evaluation(
            restored_payload,
            label=f"{replicate_id} restored restart evaluation",
        )
        audit.require(
            all(
                live.get(field) == restored.get(field)
                for field in (
                    "checkpoint_manifest_sha256",
                    "model_version",
                    "parameter_sha256",
                )
            ),
            code="restart_static_identity_mismatch",
            message=(f"{replicate_id} restart traces bind different checkpoint/model identities"),
            replicate_id=replicate_id,
        )
        audit.require(
            type(launcher_process_id) is int
            and live.get("process_id") == launcher_process_id
            and restored.get("process_id") != launcher_process_id,
            code="restart_process_identity_mismatch",
            message=(
                f"{replicate_id} retained traces do not bind the live "
                "continuation to the launcher and the restored continuation "
                "to a distinct process"
            ),
            replicate_id=replicate_id,
        )

        master_seed = replicate.get("master_seed")
        expected_reset_seeds = (
            _derive_seed(
                "behavior_evaluation_a_episode",
                int(master_seed),
                0,
            )
            if type(master_seed) is int
            else None,
            _derive_seed(
                "behavior_evaluation_b_episode",
                int(master_seed),
                0,
            )
            if type(master_seed) is int
            else None,
        )
        live_tasks = cast(list[Mapping[str, object]], live["tasks"])
        audit.require(
            tuple(task.get("reset_seed") for task in live_tasks) == expected_reset_seeds,
            code="restart_reset_seed_mismatch",
            message=(f"{replicate_id} restart traces do not use the sealed task reset seeds"),
            replicate_id=replicate_id,
        )

        checkpoint_components = {
            row.get("component_id"): row.get("sha256") for row in _mapping_rows(replicate.get("checkpoint_components"))
        }
        audit.require(
            checkpoint_components == live.get("component_hashes")
            and checkpoint_components == restored.get("component_hashes"),
            code="restart_component_binding_mismatch",
            message=(f"{replicate_id} restart traces do not bind the checkpoint component rows"),
            replicate_id=replicate_id,
        )
        retained_updates = [
            row
            for row in _mapping_rows(replicate.get("updates"))
            if row.get("phase") == "train_b_replay" and row.get("status") == "committed"
        ]
        retained_update: Mapping[str, object] = retained_updates[0] if len(retained_updates) == 1 else {}
        live_boundary = cast(Mapping[str, object], live["boundary_state"])
        audit.require(
            len(retained_updates) == 1
            and live.get("parameter_sha256") == retained_update.get("committed_parameter_sha256")
            and live.get("model_version") == retained_update.get("committed_model_version")
            and live_boundary.get("latest_update_id") == retained_update.get("receipt_id")
            and live_boundary.get("model_version") == retained_update.get("committed_model_version"),
            code="restart_retained_state_lineage_mismatch",
            message=(f"{replicate_id} restart traces do not continue the retained B-replay state"),
            replicate_id=replicate_id,
        )

        recomputed = _restart_trace_differences(live, restored)
        stored_summary = {
            key: value
            for key, value in parity.items()
            if key
            not in {
                "live_evaluation",
                "restored_evaluation",
                "restore_runtime",
            }
        }
        audit.require(
            _canonical_json_bytes(stored_summary) == _canonical_json_bytes(recomputed),
            code="restart_parity_recomputation_mismatch",
            message=(
                f"{replicate_id} stored K7 differences differ from the independently reopened live/restored traces"
            ),
            replicate_id=replicate_id,
        )
    except (ArtifactAuditError, KeyError, TypeError, ValueError) as error:
        audit.error(
            "restart_evidence_invalid",
            f"{replicate_id} restart evidence: {error}",
            replicate_id=replicate_id,
        )


def _stream_zip_member_digest(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> str:
    digest = hashlib.sha256()
    with archive.open(info, mode="r") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_domain_graph_structure(
    value: object,
    *,
    component_id: str,
) -> None:
    """Validate the non-executable WM-001 graph grammar and stable refs."""

    if not isinstance(value, dict) or set(value) != {
        "schema",
        "roots",
        "nodes",
        "observation_sequences",
    }:
        raise ArtifactAuditError(f"{component_id} domain graph has invalid root fields")
    roots = value.get("roots")
    nodes = value.get("nodes")
    sequences = value.get("observation_sequences")
    if (
        value.get("schema") != _GRAPH_SCHEMA
        or not isinstance(roots, dict)
        or not isinstance(nodes, list)
        or not isinstance(sequences, list)
        or len(roots) > _MAX_GRAPH_ROOTS
        or len(nodes) > _MAX_GRAPH_NODES
        or len(sequences) > _MAX_OBSERVATION_SEQUENCES
    ):
        raise ArtifactAuditError(f"{component_id} domain graph violates its bounds")
    expected_roots = {
        "experience_store": {"events", "transitions"},
        "update_receipts": {"receipts"},
        "agent_runtime": {"snapshot"},
        "rejected_probe": {
            "agent_snapshot",
            "source_events",
            "source_transitions",
            "source_updates",
            "probe_transitions",
            "probe_updates",
        },
    }[component_id]
    if set(roots) != expected_roots:
        raise ArtifactAuditError(f"{component_id} domain graph root set differs")

    node_types: dict[str, str] = {}
    raw_fields: list[Mapping[str, object]] = []
    for index, raw_node in enumerate(nodes):
        expected_ref = f"n{index:08d}"
        if (
            not isinstance(raw_node, Mapping)
            or set(raw_node) != {"ref", "type", "fields"}
            or raw_node.get("ref") != expected_ref
            or raw_node.get("type") not in _GRAPH_RECORD_TYPES
            or not isinstance(raw_node.get("fields"), Mapping)
            or set(raw_node.get("fields", {})) != _GRAPH_RECORD_FIELDS.get(str(raw_node.get("type")), frozenset())
        ):
            raise ArtifactAuditError(f"{component_id} domain graph node {index} is not allowlisted")
        node_types[expected_ref] = str(raw_node["type"])
        raw_fields.append(raw_node["fields"])  # type: ignore[arg-type]

    root_type_contract = {
        "experience_store": {
            "events": "ExperienceEvent",
            "transitions": "EpistemicTransition",
        },
        "update_receipts": {"receipts": "UpdateReceipt"},
        "agent_runtime": {"snapshot": "AgentSnapshot"},
        "rejected_probe": {
            "agent_snapshot": "AgentSnapshot",
            "source_events": "ExperienceEvent",
            "source_transitions": "EpistemicTransition",
            "source_updates": "UpdateReceipt",
            "probe_transitions": "EpistemicTransition",
            "probe_updates": "UpdateReceipt",
        },
    }[component_id]
    for root_name, expected_type in root_type_contract.items():
        encoded_root = roots[root_name]
        root_items = (
            encoded_root.get("$tuple")
            if isinstance(encoded_root, Mapping) and set(encoded_root) == {"$tuple"}
            else [encoded_root]
        )
        if not isinstance(root_items, list) or any(
            not isinstance(item, Mapping)
            or set(item) != {"$ref"}
            or node_types.get(str(item.get("$ref"))) != expected_type
            for item in root_items
        ):
            raise ArtifactAuditError(f"{component_id} root {root_name!r} does not reference only {expected_type} nodes")

    sequence_refs: set[str] = set()
    referenced_sequences: set[str] = set()
    sequence_lengths: dict[str, int] = {}
    sequence_items: list[object] = []
    for index, raw_sequence in enumerate(sequences):
        expected_ref = f"s{index:08d}"
        if (
            not isinstance(raw_sequence, Mapping)
            or set(raw_sequence) != {"ref", "prefix", "item"}
            or raw_sequence.get("ref") != expected_ref
        ):
            raise ArtifactAuditError(f"{component_id} observation-sequence row {index} is malformed")
        prefix = raw_sequence.get("prefix")
        if prefix is not None and prefix not in sequence_refs:
            raise ArtifactAuditError(f"{component_id} observation sequence does not use a prior prefix")
        if isinstance(prefix, str):
            referenced_sequences.add(prefix)
            sequence_length = sequence_lengths[prefix] + 1
        else:
            sequence_length = 1
        if sequence_length > _MAX_OBSERVATION_SEQUENCE_LENGTH:
            raise ArtifactAuditError(f"{component_id} observation sequence exceeds its prefix-chain bound")
        sequence_refs.add(expected_ref)
        sequence_lengths[expected_ref] = sequence_length
        sequence_items.append(raw_sequence.get("item"))

    referenced_nodes: set[str] = set()
    encoded_values = 0

    def validate_encoded(encoded: object, depth: int) -> None:
        nonlocal encoded_values
        encoded_values += 1
        if encoded_values > _MAX_GRAPH_VALUES or depth > _MAX_GRAPH_DEPTH:
            raise ArtifactAuditError(f"{component_id} domain graph exceeds encoded-value/depth bounds")
        if encoded is None or isinstance(encoded, (bool, str, int)):
            return
        if isinstance(encoded, float):
            if not math.isfinite(encoded):
                raise ArtifactAuditError(f"{component_id} domain graph contains a non-finite number")
            return
        if isinstance(encoded, list):
            for item in encoded:
                validate_encoded(item, depth + 1)
            return
        if not isinstance(encoded, Mapping):
            raise ArtifactAuditError(f"{component_id} domain graph contains a non-JSON value")
        fields = set(encoded)
        if fields == {"$ref"}:
            reference = encoded.get("$ref")
            if not isinstance(reference, str) or reference not in node_types:
                raise ArtifactAuditError(f"{component_id} domain graph contains a dangling node ref")
            referenced_nodes.add(reference)
            return
        if fields == {"$external"}:
            reference = encoded.get("$external")
            allowed_prefixes = (
                ()
                if component_id in {"experience_store", "rejected_probe"}
                else (("transition:",) if component_id == "update_receipts" else ("update:", "belief:"))
            )
            if (
                not isinstance(reference, str)
                or not reference
                or not reference.startswith(allowed_prefixes)
                or reference.endswith(":")
            ):
                raise ArtifactAuditError(f"{component_id} domain graph external ref is not allowlisted")
            return
        if fields == {"$tuple"}:
            items = encoded.get("$tuple")
            if not isinstance(items, list):
                raise ArtifactAuditError(f"{component_id} domain graph tuple tag is malformed")
            validate_encoded(items, depth + 1)
            return
        if fields == {"$mapping"}:
            items = encoded.get("$mapping")
            if not isinstance(items, list):
                raise ArtifactAuditError(f"{component_id} domain graph mapping tag is malformed")
            prior = ""
            for pair in items:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str) or pair[0] <= prior:
                    raise ArtifactAuditError(f"{component_id} domain graph mapping keys are non-canonical")
                prior = pair[0]
                validate_encoded(pair[1], depth + 1)
            return
        if fields == {"$enum", "value"}:
            if (
                encoded.get("$enum") not in _GRAPH_ENUM_TYPES
                or isinstance(encoded.get("value"), bool)
                or not isinstance(encoded.get("value"), (str, int))
            ):
                raise ArtifactAuditError(f"{component_id} domain graph enum is not allowlisted")
            return
        if fields == {"$observation_sequence"}:
            reference = encoded.get("$observation_sequence")
            if not isinstance(reference, str) or reference not in sequence_refs:
                raise ArtifactAuditError(f"{component_id} domain graph observation sequence is dangling")
            referenced_sequences.add(reference)
            return
        raise ArtifactAuditError(f"{component_id} domain graph uses an unknown encoded tag")

    for encoded_root in roots.values():
        validate_encoded(encoded_root, 0)
    for fields in raw_fields:
        for encoded_field in fields.values():
            validate_encoded(encoded_field, 0)
    for item in sequence_items:
        validate_encoded(item, 0)
        if (
            not isinstance(item, Mapping)
            or set(item) != {"$ref"}
            or node_types.get(str(item.get("$ref"))) != "Observation"
        ):
            raise ArtifactAuditError(f"{component_id} observation sequence item is not an Observation ref")
    if referenced_nodes != set(node_types):
        raise ArtifactAuditError(f"{component_id} domain graph contains unreachable record nodes")
    if referenced_sequences != sequence_refs:
        raise ArtifactAuditError(f"{component_id} domain graph contains unreachable observation sequences")


def _validate_checkpoint_domain_component(
    payload: bytes,
    *,
    component_id: str,
) -> Mapping[str, object]:
    raw = _json_without_duplicate_keys(
        payload,
        label=f"checkpoint {component_id}",
    )
    if not isinstance(raw, dict) or _canonical_json_bytes(raw) != payload:
        raise ArtifactAuditError(f"checkpoint {component_id} is not canonical JSON")
    expected_fields = {
        "experience_store": {"schema", "transition_rows", "domain_graph"},
        "update_receipts": {"schema", "updates", "domain_graph"},
        "agent_runtime": {
            "schema",
            "identity_namespace",
            "identity_checkpoint_base64",
            "next_tick",
            "agent_id",
            "configuration_version",
            "memory_version",
            "knowledge_version",
            "model_version",
            "representation_version",
            "policy_version",
            "belief_id",
            "domain_graph",
        },
    }[component_id]
    expected_schema = {
        "experience_store": "prospect.wm001.experience-custody.v1",
        "update_receipts": "prospect.wm001.update-receipts.v1",
        "agent_runtime": "prospect.wm001.agent-runtime.v1",
    }[component_id]
    if set(raw) != expected_fields or raw.get("schema") != expected_schema:
        raise ArtifactAuditError(f"checkpoint {component_id} summary/domain-graph fields differ")
    _validate_domain_graph_structure(
        raw.get("domain_graph"),
        component_id=component_id,
    )
    graph = raw["domain_graph"]
    assert isinstance(graph, Mapping)
    nodes = graph["nodes"]
    roots = graph["roots"]
    assert isinstance(nodes, list) and isinstance(roots, Mapping)
    nodes_by_ref = {
        str(node["ref"]): node for node in nodes if isinstance(node, Mapping) and isinstance(node.get("ref"), str)
    }

    def root_refs(name: str) -> list[str]:
        encoded = roots[name]
        if isinstance(encoded, Mapping) and set(encoded) == {"$tuple"} and isinstance(encoded.get("$tuple"), list):
            items = encoded["$tuple"]
        else:
            items = [encoded]
        return [str(item["$ref"]) for item in items if isinstance(item, Mapping) and set(item) == {"$ref"}]

    def fields(reference: str) -> Mapping[str, object]:
        node = nodes_by_ref[reference]
        raw_fields = node["fields"]
        assert isinstance(raw_fields, Mapping)
        return raw_fields

    if component_id == "experience_store":
        rows = raw.get("transition_rows")
        transition_refs = root_refs("transitions")
        event_refs = root_refs("events")
        if not isinstance(rows, list) or len(rows) != len(transition_refs) or len(rows) != len(event_refs):
            raise ArtifactAuditError("experience domain graph count differs from transition custody")
        for index, (row, transition_ref, event_ref) in enumerate(zip(rows, transition_refs, event_refs, strict=True)):
            if not isinstance(row, Mapping):
                raise ArtifactAuditError("experience transition custody row is malformed")
            transition_fields = fields(transition_ref)
            event_fields = fields(event_ref)
            experience_ref = transition_fields.get("experience")
            if (
                transition_fields.get("transition_id") != row.get("transition_id")
                or experience_ref != {"$ref": event_ref}
                or any(
                    event_fields.get(field_name) != row.get(field_name)
                    for field_name in (
                        "run_id",
                        "task_id",
                        "episode_id",
                        "step_index",
                        "terminated",
                        "truncated",
                    )
                )
            ):
                raise ArtifactAuditError(f"experience domain graph row {index} differs from custody")
            for graph_field, row_field, expected_type in (
                ("decision", "decision_id", "DecisionRecord"),
                ("execution", "executed_action_id", "ExecutedAction"),
                ("observation", "next_observation_id", "Observation"),
            ):
                encoded_ref = event_fields.get(graph_field)
                if (
                    not isinstance(encoded_ref, Mapping)
                    or set(encoded_ref) != {"$ref"}
                    or str(encoded_ref.get("$ref")) not in nodes_by_ref
                ):
                    raise ArtifactAuditError(f"experience graph {graph_field} lineage is malformed")
                lineage_ref = str(encoded_ref["$ref"])
                lineage = nodes_by_ref[lineage_ref]
                expected_id_field = {
                    "DecisionRecord": "decision_id",
                    "ExecutedAction": "execution_id",
                    "Observation": "observation_id",
                }[expected_type]
                if lineage.get("type") != expected_type or fields(lineage_ref).get(expected_id_field) != row.get(
                    row_field
                ):
                    raise ArtifactAuditError(f"experience graph {graph_field} lineage differs from custody")
    elif component_id == "update_receipts":
        rows = raw.get("updates")
        receipt_refs = root_refs("receipts")
        if not isinstance(rows, list) or len(rows) != len(receipt_refs):
            raise ArtifactAuditError("update domain graph count differs from receipt custody")
        for index, (row, receipt_ref) in enumerate(zip(rows, receipt_refs, strict=True)):
            if not isinstance(row, Mapping):
                raise ArtifactAuditError("update custody row is malformed")
            receipt = fields(receipt_ref)
            encoded_transitions = receipt.get("transitions")
            if (
                not isinstance(encoded_transitions, Mapping)
                or set(encoded_transitions) != {"$tuple"}
                or not isinstance(encoded_transitions.get("$tuple"), list)
            ):
                raise ArtifactAuditError("update receipt transition references are malformed")
            transition_ids = [
                str(item.get("$external", "")).removeprefix("transition:")
                for item in encoded_transitions["$tuple"]
                if isinstance(item, Mapping) and set(item) == {"$external"}
            ]
            if (
                receipt.get("receipt_id") != row.get("receipt_id")
                or transition_ids != row.get("eligible_transition_ids")
                or receipt.get("previous_model_version") != row.get("predecessor_model_version")
                or receipt.get("new_model_version") != row.get("committed_model_version")
            ):
                raise ArtifactAuditError(f"update domain graph receipt {index} differs from custody")
    elif component_id == "agent_runtime":
        snapshot_refs = root_refs("snapshot")
        if len(snapshot_refs) != 1:
            raise ArtifactAuditError("agent graph has no unique snapshot")
        snapshot = fields(snapshot_refs[0])
        scalar_fields = (
            "agent_id",
            "configuration_version",
            "memory_version",
            "knowledge_version",
            "model_version",
            "representation_version",
            "policy_version",
        )
        belief = snapshot.get("belief")
        latest_update = snapshot.get("latest_update")
        if (
            any(snapshot.get(field) != raw.get(field) for field in scalar_fields)
            or belief != {"$external": f"belief:{raw.get('belief_id')}"}
            or not isinstance(latest_update, Mapping)
            or set(latest_update) != {"$external"}
            or not str(latest_update.get("$external", "")).startswith("update:")
            or snapshot.get("pending_intentions") != {"$tuple": []}
        ):
            raise ArtifactAuditError("agent domain graph snapshot differs from runtime custody")
    return raw


def _decode_checkpoint_replay_component(
    payload: bytes,
    *,
    component_id: str,
) -> Mapping[str, object]:
    """Decode one canonical replay component without producer serializers."""

    raw = _json_without_duplicate_keys(
        payload,
        label=f"checkpoint {component_id}",
    )
    expected_fields = (
        {
            "schema",
            "canonical_experience_rows",
            "collect_a",
            "collect_b",
        }
        if component_id == "replay_index"
        else {"schema", "manifests"}
        if component_id == "replay_sampling_history"
        else None
    )
    expected_schema = {
        "replay_index": "prospect.wm001.replay-index.v1",
        "replay_sampling_history": "prospect.wm001.replay-sampling-history.v1",
    }.get(component_id)
    if (
        expected_fields is None
        or expected_schema is None
        or not isinstance(raw, dict)
        or set(raw) != expected_fields
        or raw.get("schema") != expected_schema
        or payload != _canonical_json_bytes(raw)
    ):
        raise ArtifactAuditError(f"checkpoint {component_id} violates its canonical format")
    return raw


def _expected_checkpoint_transition_dataset(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "transition_ids": [row.get("transition_id") for row in rows],
        "observations": [row.get("pre_observation") for row in rows],
        "contexts": [row.get("task_context") for row in rows],
        "actions": [row.get("intended_action") for row in rows],
        "next_observations": [row.get("next_observation") for row in rows],
        "rewards": [row.get("reward") for row in rows],
    }


def _audit_retained_replay_components(
    audit: _Audit,
    root: Path,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
    replay_index: Mapping[str, object],
    replay_sampling_history: Mapping[str, object],
) -> None:
    """Bind retained replay bytes to only the persistent A/B agent evidence."""

    transitions = _mapping_rows(replicate.get("transitions"))
    heldout_ids = {
        str(row["transition_id"])
        for row in transitions
        if str(row.get("split", "")).startswith(("predictive_validation_", "behavior_evaluation_"))
        and isinstance(row.get("transition_id"), str)
    }
    expected_by_split = {
        split: [row for row in transitions if row.get("split") == split] for split in ("collect_a", "collect_b")
    }
    expected_rows = [
        *expected_by_split["collect_a"],
        *expected_by_split["collect_b"],
    ]
    canonical_experiences = replay_index.get("canonical_experience_rows")
    experience_linkage_valid = False
    if isinstance(canonical_experiences, list):
        expected_links = [
            (
                row.get("run_id"),
                row.get("task_id"),
                row.get("episode_id"),
                row.get("step_index"),
            )
            for row in expected_rows
        ]
        actual_links: list[tuple[object, object, object, object]] = []
        experience_ids: list[object] = []
        experience_rows_valid = True
        for row in canonical_experiences:
            if not isinstance(row, Mapping) or set(row) != {
                "experience_id",
                "run_id",
                "task_id",
                "episode_id",
                "step_index",
                "closed_at",
            }:
                experience_rows_valid = False
                continue
            closed_at = row.get("closed_at")
            if (
                not isinstance(row.get("experience_id"), str)
                or not row.get("experience_id")
                or row.get("task_id") not in {_TASK_A, _TASK_B}
                or not isinstance(closed_at, list)
                or len(closed_at) != 2
                or not isinstance(closed_at[0], str)
                or not closed_at[0]
                or type(closed_at[1]) is not int
                or int(closed_at[1]) < 0
            ):
                experience_rows_valid = False
            experience_ids.append(row.get("experience_id"))
            actual_links.append(
                (
                    row.get("run_id"),
                    row.get("task_id"),
                    row.get("episode_id"),
                    row.get("step_index"),
                )
            )
        experience_linkage_valid = (
            experience_rows_valid and len(experience_ids) == len(set(experience_ids)) and actual_links == expected_links
        )
    replay_dataset_valid = (
        replay_index.get("collect_a") == _expected_checkpoint_transition_dataset(expected_by_split["collect_a"])
        and replay_index.get("collect_b") == _expected_checkpoint_transition_dataset(expected_by_split["collect_b"])
        and all(
            row.get("task_id") in {_TASK_A, _TASK_B} and row.get("split") in {"collect_a", "collect_b"}
            for row in expected_rows
        )
    )
    audit.require(
        experience_linkage_valid and replay_dataset_valid,
        code="checkpoint_replay_index_isolation_mismatch",
        message=(
            f"{replicate_id} retained replay index is not exactly the ordered "
            "collect-A/collect-B evidence or contains irrelevant-control custody"
        ),
        replicate_id=replicate_id,
    )
    retained_transition_ids: set[str] = set()
    for split in ("collect_a", "collect_b"):
        dataset = replay_index.get(split)
        if not isinstance(dataset, Mapping):
            continue
        identities = dataset.get("transition_ids")
        if isinstance(identities, list):
            retained_transition_ids.update(identity for identity in identities if isinstance(identity, str))
    audit.require(
        retained_transition_ids.isdisjoint(heldout_ids),
        code="checkpoint_replay_heldout_contamination",
        message=(
            f"{replicate_id} retained checkpoint replay contains a "
            "prediction-validation or behavior-evaluation transition ID"
        ),
        replicate_id=replicate_id,
    )

    expected_updates = [
        row
        for row in _mapping_rows(replicate.get("updates"))
        if row.get("phase") in {"train_a", "train_b_replay"} and row.get("status") == "committed"
    ]
    expected_phases = ("train_a", "train_b_replay")
    result_manifests = {
        str(row.get("phase")): row
        for row in _mapping_rows(replicate.get("optimizer_batch_manifests"))
        if row.get("phase") in expected_phases
    }
    retained_manifests = replay_sampling_history.get("manifests")
    sampling_valid = (
        isinstance(retained_manifests, list)
        and tuple(row.get("phase") for row in retained_manifests if isinstance(row, Mapping)) == expected_phases
        and tuple(row.get("phase") for row in expected_updates) == expected_phases
        and set(result_manifests) == set(expected_phases)
    )
    if isinstance(retained_manifests, list):
        updates_by_phase = {str(row.get("phase")): row for row in expected_updates}
        for row in retained_manifests:
            if not isinstance(row, Mapping) or set(row) != {
                "phase",
                "sha256",
                "bytes",
                "payload_base64",
            }:
                sampling_valid = False
                continue
            phase = row.get("phase")
            if phase not in expected_phases:
                sampling_valid = False
                continue
            try:
                payload = _decode_strict_base64(
                    row.get("payload_base64"),
                    label=f"{replicate_id} checkpoint {phase} replay manifest",
                )
                reference = result_manifests[str(phase)]
                _, external_payload = _verify_file_reference(
                    root,
                    reference,
                    limit=_MAX_MANIFEST_BYTES,
                    label=f"{replicate_id} retained {phase} optimizer manifest",
                )
                update = updates_by_phase[str(phase)]
                sampling_valid = sampling_valid and (
                    row.get("bytes") == len(payload)
                    and row.get("sha256") == hashlib.sha256(payload).hexdigest()
                    and payload == external_payload
                    and update.get("sampling_manifest_sha256") == row.get("sha256")
                )
            except (
                ArtifactAuditError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ):
                sampling_valid = False
    audit.require(
        sampling_valid,
        code="checkpoint_replay_sampling_isolation_mismatch",
        message=(
            f"{replicate_id} retained replay sampling history is not exactly "
            "train-A then train-B-replay or contains an irrelevant-control manifest"
        ),
        replicate_id=replicate_id,
    )


def _audit_checkpoint(
    audit: _Audit,
    root: Path,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
) -> None:
    reference = replicate.get("checkpoint_archive")
    if not isinstance(reference, Mapping):
        audit.error(
            "checkpoint_archive_missing",
            f"{replicate_id} has no checkpoint archive reference",
            replicate_id=replicate_id,
        )
        return
    try:
        path = _resolve_artifact_file(root, reference.get("filename"), label=f"{replicate_id} checkpoint")
        if path.is_symlink() or not path.is_file():
            raise ArtifactAuditError("checkpoint must be a regular non-symlink file")
        stat = path.stat()
        if stat.st_size > _MAX_CHECKPOINT_BYTES:
            raise ArtifactAuditError("checkpoint exceeds its audit byte limit")
        expected_bytes = reference.get("bytes")
        if type(expected_bytes) is not int or expected_bytes != stat.st_size:
            raise ArtifactAuditError("checkpoint outer byte count mismatch")
        outer_digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1 << 20):
                outer_digest.update(chunk)
        if outer_digest.hexdigest() != reference.get("sha256"):
            raise ArtifactAuditError("checkpoint outer SHA-256 mismatch")

        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or _CHECKPOINT_MANIFEST not in names:
                raise ArtifactAuditError("checkpoint ZIP members are duplicate or lack manifest.json")
            if any(
                info.flag_bits & 0x1
                or info.compress_type != zipfile.ZIP_STORED
                or info.is_dir()
                or Path(info.filename).is_absolute()
                or ".." in Path(info.filename).parts
                for info in infos
            ):
                raise ArtifactAuditError("checkpoint ZIP uses unsafe or non-canonical members")
            manifest_info = archive.getinfo(_CHECKPOINT_MANIFEST)
            if manifest_info.file_size > _MAX_CHECKPOINT_MANIFEST_BYTES:
                raise ArtifactAuditError("checkpoint manifest exceeds its audit byte limit")
            raw_manifest = archive.read(manifest_info)
            manifest = _json_without_duplicate_keys(raw_manifest, label="checkpoint manifest")
            if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != raw_manifest:
                raise ArtifactAuditError("checkpoint manifest is not canonical JSON")
            expected_manifest_fields = {
                "agent_id",
                "checkpoint_id",
                "components",
                "created_at",
                "format",
                "metadata",
                "schema_version",
                "versions",
            }
            if set(manifest) != expected_manifest_fields:
                raise ArtifactAuditError("checkpoint manifest has an unexpected field set")
            component_entries = manifest.get("components")
            metadata = manifest.get("metadata")
            created_at = manifest.get("created_at")
            versions = manifest.get("versions")
            if (
                manifest.get("format") != "prospect-checkpoint"
                or manifest.get("schema_version") != 1
                or not isinstance(component_entries, list)
                or not isinstance(metadata, dict)
                or not isinstance(created_at, dict)
                or not isinstance(versions, dict)
                or versions.get("wm001_checkpoint") != _CHECKPOINT_SCHEMA
            ):
                raise ArtifactAuditError("checkpoint manifest has invalid component or metadata fields")
            expected_paths = {_CHECKPOINT_MANIFEST}
            archive_components: dict[str, Mapping[str, object]] = {}
            verified_domain_components: dict[str, Mapping[str, object]] = {}
            verified_replay_components: dict[str, Mapping[str, object]] = {}
            total_bytes = 0
            for entry in component_entries:
                if not isinstance(entry, Mapping) or set(entry) != {
                    "media_type",
                    "name",
                    "path",
                    "sha256",
                    "size_bytes",
                    "version",
                }:
                    raise ArtifactAuditError("checkpoint component manifest entry is not an object")
                component_id = entry.get("name")
                member_path = entry.get("path")
                size = entry.get("size_bytes")
                if (
                    not isinstance(component_id, str)
                    or component_id in archive_components
                    or member_path != f"components/{component_id}.bin"
                    or type(size) is not int
                    or size < 0
                    or size > _MAX_CHECKPOINT_COMPONENT_BYTES
                ):
                    raise ArtifactAuditError("checkpoint component entry is malformed")
                info = archive.getinfo(str(member_path))
                if info.file_size != size:
                    raise ArtifactAuditError(f"checkpoint component {component_id} size mismatch")
                total_bytes += size
                if total_bytes > _MAX_CHECKPOINT_TOTAL_BYTES:
                    raise ArtifactAuditError("checkpoint components exceed the total audit byte limit")
                digest = _stream_zip_member_digest(archive, info)
                if digest != entry.get("sha256"):
                    raise ArtifactAuditError(f"checkpoint component {component_id} digest mismatch")
                if component_id in {
                    "experience_store",
                    "update_receipts",
                    "agent_runtime",
                }:
                    if entry.get("media_type") == "application/json" and size <= 512 << 20:
                        verified_domain_components[component_id] = _validate_checkpoint_domain_component(
                            archive.read(info),
                            component_id=component_id,
                        )
                if component_id in {
                    "replay_index",
                    "replay_sampling_history",
                }:
                    if entry.get("media_type") == "application/json" and size <= 512 << 20:
                        verified_replay_components[component_id] = _decode_checkpoint_replay_component(
                            archive.read(info),
                            component_id=component_id,
                        )
                expected_paths.add(str(member_path))
                archive_components[component_id] = entry
            if set(names) != expected_paths:
                raise ArtifactAuditError("checkpoint ZIP member set differs from its manifest")
            if set(archive_components) != set(_CANONICAL_COMPONENT_IDS):
                raise ArtifactAuditError("checkpoint does not contain all and only 15 canonical components")
            if set(verified_domain_components) != {
                "experience_store",
                "update_receipts",
                "agent_runtime",
            }:
                audit.gap(
                    "checkpoint_domain_graph_semantics_unverified",
                    (
                        f"{replicate_id} checkpoint does not expose all three "
                        "canonical JSON domain-graph components for independent review."
                    ),
                    evidence_needed=(
                        "Canonical bounded experience_store, update_receipts, and "
                        "agent_runtime JSON components using only the allowlisted graph grammar."
                    ),
                )
            else:
                heldout_transition_ids = {
                    str(row["transition_id"])
                    for row in _mapping_rows(replicate.get("transitions"))
                    if str(row.get("split", "")).startswith(
                        (
                            "predictive_validation_",
                            "behavior_evaluation_",
                        )
                    )
                    and isinstance(row.get("transition_id"), str)
                }
                expected_transition_rows = [
                    row
                    for row in _mapping_rows(replicate.get("transitions"))
                    if row.get("split") in {"collect_a", "collect_b"}
                ]
                expected_update_rows = [
                    row
                    for row in _mapping_rows(replicate.get("updates"))
                    if row.get("phase") in {"train_a", "train_b_replay"} and row.get("status") == "committed"
                ]
                experience_component = verified_domain_components["experience_store"]
                receipts_component = verified_domain_components["update_receipts"]
                audit.require(
                    experience_component.get("transition_rows") == expected_transition_rows
                    and all(row.get("task_id") in {_TASK_A, _TASK_B} for row in expected_transition_rows)
                    and not any(row.get("split") == "collect_irrelevant" for row in expected_transition_rows),
                    code="checkpoint_irrelevant_experience_isolation_mismatch",
                    message=(
                        f"{replicate_id} retained checkpoint experience is not "
                        "exactly the ordered collect-A/collect-B evidence"
                    ),
                    replicate_id=replicate_id,
                )
                checkpoint_transition_ids = {
                    str(row["transition_id"])
                    for row in _mapping_rows(experience_component.get("transition_rows"))
                    if isinstance(row.get("transition_id"), str)
                }
                audit.require(
                    checkpoint_transition_ids.isdisjoint(heldout_transition_ids),
                    code="checkpoint_heldout_experience_contamination",
                    message=(
                        f"{replicate_id} retained checkpoint experience contains "
                        "a prediction-validation or behavior-evaluation transition ID"
                    ),
                    replicate_id=replicate_id,
                )
                audit.require(
                    receipts_component.get("updates") == expected_update_rows
                    and tuple(row.get("phase") for row in expected_update_rows) == ("train_a", "train_b_replay"),
                    code="checkpoint_irrelevant_update_isolation_mismatch",
                    message=(
                        f"{replicate_id} retained checkpoint receipts are not exactly train-A then train-B-replay"
                    ),
                    replicate_id=replicate_id,
                )
            if set(verified_replay_components) != {
                "replay_index",
                "replay_sampling_history",
            }:
                audit.gap(
                    "checkpoint_replay_semantics_unverified",
                    (
                        f"{replicate_id} checkpoint does not expose canonical "
                        "JSON replay_index and replay_sampling_history components."
                    ),
                    evidence_needed=(
                        "Canonical bounded replay-index and sampling-history "
                        "JSON that can be cross-bound to collect-A/collect-B "
                        "rows and retained optimizer manifests."
                    ),
                )
            else:
                _audit_retained_replay_components(
                    audit,
                    root,
                    replicate,
                    replicate_id=replicate_id,
                    replay_index=verified_replay_components["replay_index"],
                    replay_sampling_history=verified_replay_components["replay_sampling_history"],
                )

            result_rows = _mapping_rows(replicate.get("checkpoint_components"))
            rows_by_id = {
                str(row.get("component_id")): row for row in result_rows if isinstance(row.get("component_id"), str)
            }
            audit.require(
                tuple(row.get("component_id") for row in result_rows) == _CANONICAL_COMPONENT_IDS,
                code="checkpoint_component_order_mismatch",
                message=f"{replicate_id} result does not list all 15 components in canonical order",
                replicate_id=replicate_id,
            )
            checkpoint_id = manifest.get("checkpoint_id")
            component_binding_valid = set(rows_by_id) == set(_CANONICAL_COMPONENT_IDS)
            for component_id in _CANONICAL_COMPONENT_IDS:
                result_row = rows_by_id.get(component_id)
                archive_row = archive_components[component_id]
                if result_row is None:
                    component_binding_valid = False
                    continue
                predecessor = metadata.get(f"wm001.predecessor.{component_id}")
                predecessor = None if predecessor == "none" else predecessor
                component_binding_valid = component_binding_valid and all(
                    (
                        result_row.get("checkpoint_id") == checkpoint_id,
                        result_row.get("logical_version") == archive_row.get("version"),
                        result_row.get("media_type") == archive_row.get("media_type"),
                        result_row.get("bytes") == archive_row.get("size_bytes"),
                        result_row.get("sha256") == archive_row.get("sha256"),
                        result_row.get("predecessor_sha256") == predecessor,
                    )
                )
            audit.require(
                component_binding_valid,
                code="checkpoint_component_binding_mismatch",
                message=f"{replicate_id} result component rows differ from checkpoint bytes",
                replicate_id=replicate_id,
            )
            body = {
                "agent_id": manifest.get("agent_id"),
                "boundary": _CHECKPOINT_BOUNDARY,
                "checkpoint_id": checkpoint_id,
                "components": result_rows,
                "created_at": created_at,
                "schema": _CHECKPOINT_SCHEMA,
            }
            logical_digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
            audit.require(
                metadata.get("wm001.schema") == _CHECKPOINT_SCHEMA
                and metadata.get("wm001.boundary") == _CHECKPOINT_BOUNDARY
                and metadata.get("wm001.aggregate_manifest_sha256") == logical_digest,
                code="checkpoint_logical_manifest_mismatch",
                message=f"{replicate_id} checkpoint logical manifest digest is not reproducible",
                replicate_id=replicate_id,
            )
            restart_parity = replicate.get("restart_parity")
            if isinstance(restart_parity, Mapping):
                audit.require(
                    restart_parity.get("checkpoint_manifest_sha256") == logical_digest,
                    code="restart_checkpoint_manifest_mismatch",
                    message=f"{replicate_id} restart evidence binds a different checkpoint manifest",
                    replicate_id=replicate_id,
                )
    except (ArtifactAuditError, KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        audit.error(
            "checkpoint_evidence_invalid",
            f"{replicate_id} checkpoint: {error}",
            replicate_id=replicate_id,
        )


_StableFileIdentity = tuple[int, int, int, int, int, int, int, int, int]


def _stable_file_identity(value: os.stat_result) -> _StableFileIdentity:
    """Return metadata whose change makes a custody read non-atomic."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_regular_file(
    path: Path,
    limit: int,
    *,
    label: str,
    capture_payload: bool,
) -> tuple[bytes | None, int, str, _StableFileIdentity]:
    """Hash one bounded regular file while rejecting replacement or mutation."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactAuditError(f"{label} cannot be opened safely: {error}") from error
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactAuditError(f"{label} must be a regular non-symlink file")
        if before.st_size > limit:
            raise ArtifactAuditError(f"{label} exceeds its {limit}-byte audit limit")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > limit:
                raise ArtifactAuditError(f"{label} exceeds its {limit}-byte audit limit")
            digest.update(chunk)
            if capture_payload:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ArtifactAuditError(f"{label} cannot be read safely: {error}") from error
    finally:
        os.close(descriptor)

    try:
        current = path.lstat()
    except OSError as error:
        raise ArtifactAuditError(f"{label} changed while it was being read: {error}") from error
    before_identity = _stable_file_identity(before)
    after_identity = _stable_file_identity(after)
    current_identity = _stable_file_identity(current)
    if (
        before_identity != after_identity
        or after_identity != current_identity
        or not stat.S_ISREG(current.st_mode)
        or observed_bytes != after.st_size
    ):
        raise ArtifactAuditError(f"{label} changed while it was being read")
    payload = b"".join(chunks) if capture_payload else None
    return payload, observed_bytes, digest.hexdigest(), after_identity


def _scan_producer_tree(root: Path) -> dict[str, _StableFileIdentity]:
    """Enumerate one symlink-free producer tree without following aliases."""

    try:
        root_stat = root.lstat()
    except OSError as error:
        raise ArtifactAuditError(f"producer artifact root cannot be inspected: {error}") from error
    if root.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
        raise ArtifactAuditError("producer artifact root must be a regular non-symlink directory")

    files: dict[str, _StableFileIdentity] = {}
    pending = [(root, Path())]
    entry_count = 0
    while pending:
        directory, relative_directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise ArtifactAuditError(f"producer artifact tree cannot be scanned: {error}") from error
        for entry in entries:
            entry_count += 1
            if entry_count > _MAX_PRODUCER_TREE_ENTRIES:
                raise ArtifactAuditError("producer artifact tree exceeds its audit entry limit")
            relative_path = relative_directory / entry.name
            relative = relative_path.as_posix()
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ArtifactAuditError(f"producer artifact entry cannot be inspected: {relative}") from error
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ArtifactAuditError(f"producer artifact tree contains a symbolic link: {relative}")
            if stat.S_ISDIR(entry_stat.st_mode):
                pending.append((Path(entry.path), relative_path))
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ArtifactAuditError(f"producer artifact tree contains a non-regular file: {relative}")
            if relative != _PRODUCER_MANIFEST_NAME:
                files[relative] = _stable_file_identity(entry_stat)
                if len(files) > _MAX_PRODUCER_FILES:
                    raise ArtifactAuditError("producer manifest exceeds its audit file-count limit")
    return files


def _safe_producer_relative_path(root: Path, value: object, *, index: int) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ArtifactAuditError(f"producer manifest files[{index}].path is invalid")
    relative = Path(value)
    if (
        relative.is_absolute()
        or value == _PRODUCER_MANIFEST_NAME
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != value
    ):
        raise ArtifactAuditError(f"unsafe producer manifest path: {value}")
    candidate = root / relative
    try:
        root_resolved = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ArtifactAuditError(f"manifested producer file is missing: {value}") from error
    if not resolved.is_relative_to(root_resolved):
        raise ArtifactAuditError(f"manifested producer file escapes the artifact root: {value}")
    return value, candidate


def _verify_producer_manifest_locally(root: Path) -> tuple[dict[str, object], str]:
    """Independently verify finalized producer custody from untrusted bytes."""

    root_before = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(root_before.st_mode):
        raise ArtifactAuditError("producer artifact root must be a regular non-symlink directory")
    manifest_path = root / _PRODUCER_MANIFEST_NAME
    raw_manifest, _, manifest_digest, manifest_identity = _read_stable_regular_file(
        manifest_path,
        _MAX_PRODUCER_MANIFEST_BYTES,
        label="producer manifest",
        capture_payload=True,
    )
    if raw_manifest is None or not raw_manifest.endswith(b"\n"):
        raise ArtifactAuditError("producer manifest lacks its canonical trailing newline")
    manifest_raw = _json_without_duplicate_keys(raw_manifest[:-1], label="producer manifest")
    if (
        not isinstance(manifest_raw, dict)
        or set(manifest_raw) != _PRODUCER_MANIFEST_FIELDS
        or raw_manifest != _canonical_json_bytes(manifest_raw) + b"\n"
    ):
        raise ArtifactAuditError("producer manifest is not canonical JSON plus one LF")
    manifest: dict[str, object] = manifest_raw
    if manifest.get("schema") != "prospect.wm001.producer-manifest.v1":
        raise ArtifactAuditError("producer manifest schema identity is invalid")
    if manifest.get("experiment_id") != "WM-001":
        raise ArtifactAuditError("producer manifest experiment identity is invalid")
    if manifest.get("lane") not in {"development", "formal"}:
        raise ArtifactAuditError("producer manifest lane identity is invalid")
    status = manifest.get("status")
    error = manifest.get("error")
    if status not in {"completed", "failed"}:
        raise ArtifactAuditError("producer manifest status is invalid")
    if status == "completed" and error is not None:
        raise ArtifactAuditError("completed producer manifest must not contain an error")
    if status == "failed" and (
        not isinstance(error, dict)
        or set(error) != {"type", "message"}
        or not isinstance(error.get("type"), str)
        or not error.get("type")
        or not isinstance(error.get("message"), str)
    ):
        raise ArtifactAuditError("failed producer manifest has an invalid error identity")
    started_at = _parse_utc_timestamp(
        manifest.get("started_at_utc"),
        label="producer manifest started_at_utc",
    )
    completed_at = _parse_utc_timestamp(
        manifest.get("completed_at_utc"),
        label="producer manifest completed_at_utc",
    )
    if completed_at < started_at:
        raise ArtifactAuditError("producer manifest completed before it started")
    if manifest.get("manifest_excludes") != [_PRODUCER_MANIFEST_NAME]:
        raise ArtifactAuditError("producer manifest exclusion contract is invalid")

    rows = manifest.get("files")
    file_count = manifest.get("file_count")
    if not isinstance(rows, list):
        raise ArtifactAuditError("producer manifest files must be an array")
    if type(file_count) is not int or file_count != len(rows):
        raise ArtifactAuditError("producer manifest file_count is invalid")
    if len(rows) > _MAX_PRODUCER_FILES:
        raise ArtifactAuditError("producer manifest exceeds its audit file-count limit")

    references: dict[str, tuple[Path, int, str]] = {}
    total_bytes = 0
    previous_path = ""
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ArtifactAuditError(f"producer manifest files[{index}] has an invalid field set")
        relative, path = _safe_producer_relative_path(root, row.get("path"), index=index)
        byte_count = row.get("bytes")
        expected_digest = row.get("sha256")
        if (
            relative in references
            or relative <= previous_path
            or type(byte_count) is not int
            or byte_count < 0
            or byte_count > _MAX_PRODUCER_FILE_BYTES
            or not _is_sha256(expected_digest)
        ):
            raise ArtifactAuditError(f"producer manifest files[{index}] has invalid identity metadata")
        total_bytes += byte_count
        if total_bytes > _MAX_PRODUCER_TOTAL_BYTES:
            raise ArtifactAuditError("producer manifest exceeds its aggregate byte limit")
        references[relative] = (path, byte_count, str(expected_digest))
        previous_path = relative

    initial_tree = _scan_producer_tree(root)
    if set(references) != set(initial_tree):
        unmanifested = sorted(set(initial_tree) - set(references))
        missing = sorted(set(references) - set(initial_tree))
        raise ArtifactAuditError(f"producer manifest file set changed; unmanifested={unmanifested}, missing={missing}")
    for relative, (path, expected_bytes, expected_digest) in references.items():
        _, actual_bytes, actual_digest, identity = _read_stable_regular_file(
            path,
            _MAX_PRODUCER_FILE_BYTES,
            label=f"manifested producer file {relative}",
            capture_payload=False,
        )
        if identity != initial_tree[relative]:
            raise ArtifactAuditError(f"manifested producer file changed before reading: {relative}")
        expected_links = 2 if manifest.get("lane") == "formal" and relative == "formal-launch.json" else 1
        if path.lstat().st_nlink != expected_links:
            raise ArtifactAuditError(f"manifested producer file has invalid hard-link custody: {relative}")
        if actual_bytes != expected_bytes:
            raise ArtifactAuditError(f"manifested producer file size changed: {relative}")
        if actual_digest != expected_digest:
            raise ArtifactAuditError(f"manifested producer file digest changed: {relative}")

    final_tree = _scan_producer_tree(root)
    if initial_tree != final_tree:
        raise ArtifactAuditError("producer artifact tree changed while custody was being verified")
    final_manifest, _, final_digest, final_manifest_identity = _read_stable_regular_file(
        manifest_path,
        _MAX_PRODUCER_MANIFEST_BYTES,
        label="producer manifest",
        capture_payload=True,
    )
    if (
        final_manifest != raw_manifest
        or final_digest != manifest_digest
        or final_manifest_identity != manifest_identity
        or _stable_file_identity(root.lstat()) != _stable_file_identity(root_before)
    ):
        raise ArtifactAuditError("producer manifest or artifact root changed during verification")
    if manifest_path.lstat().st_nlink != 2:
        raise ArtifactAuditError("producer manifest lacks its outer-completion hardlink")
    completion_marker = _OUTER_COMPLETIONS_ROOT / (
        hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest() + ".json"
    )
    completion_payload, _, completion_digest, completion_identity = _read_stable_regular_file(
        completion_marker,
        _MAX_PRODUCER_MANIFEST_BYTES,
        label="producer outer completion marker",
        capture_payload=True,
    )
    if (
        completion_payload != raw_manifest
        or completion_digest != manifest_digest
        or completion_marker.lstat().st_nlink != 2
        or completion_identity != manifest_identity
        or not os.path.samefile(manifest_path, completion_marker)
    ):
        raise ArtifactAuditError("producer outer completion is not the terminal-manifest inode")
    return manifest, manifest_digest


def _verify_finalized_custody(audit: _Audit, root: Path) -> None:
    try:
        manifest, manifest_digest = _verify_producer_manifest_locally(root)
        audit.custody = {
            "producer_manifest_checked": True,
            "producer_manifest_status": manifest.get("status"),
            "producer_manifest_sha256": manifest_digest,
        }
        audit.require(
            manifest.get("status") == "completed",
            code="producer_attempt_not_completed",
            message="producer manifest does not identify a completed attempt",
        )
        audit.independence_limitations.append(
            _PRODUCER_CUSTODY_INDEPENDENCE_LIMITATION
        )
    except (ArtifactAuditError, OSError, ValueError) as error:
        audit.error(
            "producer_manifest_verification_failed",
            f"finalized producer custody failed verification: {error}",
        )


def _validate_result_schema(
    audit: _Audit,
    result: Mapping[str, object],
    *,
    schema_path: Path = RESULT_SCHEMA_PATH,
) -> None:
    try:
        import jsonschema  # type: ignore[import-untyped]

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(result),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except (ImportError, OSError, json.JSONDecodeError) as error:
        audit.error("schema_validation_unavailable", f"cannot validate raw-result schema: {error}")
        return
    if errors:
        for validation_error in errors[:20]:
            location = "/".join(str(part) for part in validation_error.absolute_path) or "<root>"
            audit.error(
                "raw_result_schema_error",
                f"raw-result schema violation at {location}: {validation_error.message}",
            )
        if len(errors) > 20:
            audit.error(
                "raw_result_schema_error_limit",
                f"{len(errors) - 20} additional raw-result schema violations were suppressed",
            )
    else:
        audit.passed_checks += 1


def _validate_formal_conformance_report(report: object) -> None:
    """Validate the fixed formal corpus without producer/verifier helpers."""

    if not isinstance(report, dict):
        raise ArtifactAuditError("Pendulum conformance report is not an object")
    if set(report) != _FORMAL_CONFORMANCE_KEYS:
        raise ArtifactAuditError("Pendulum conformance report fields differ from the fixed contract")
    if (
        report.get("schema") != "prospect.wm001.pendulum-conformance.v1"
        or report.get("environment_id") != "Pendulum-v1"
        or not isinstance(report.get("gymnasium_version"), str)
        or not report["gymnasium_version"]
    ):
        raise ArtifactAuditError("Pendulum conformance identity is invalid")
    if report.get("seed") != 20260717 or report.get("samples_per_task") != 512 or report.get("cases") != 1024:
        raise ArtifactAuditError("Pendulum conformance must contain exactly 512 cases per task from seed 20260717")
    if report.get("semantic_parameters") != _FORMAL_CONFORMANCE_PARAMETERS:
        raise ArtifactAuditError("Pendulum semantic parameters changed")
    if report.get("semantic_parameter_absolute_errors") != {name: 0.0 for name in _FORMAL_CONFORMANCE_PARAMETERS}:
        raise ArtifactAuditError("Pendulum semantic parameters differ from the bound environment")
    if report.get("spec_horizon") != 200:
        raise ArtifactAuditError("Pendulum episode horizon changed")
    if report.get("terminated_or_truncated_cases") != 0:
        raise ArtifactAuditError("Pendulum conformance cases terminated or truncated")
    if any(report.get(field) != expected for field, expected in _FORMAL_CONFORMANCE_TOLERANCES.items()):
        raise ArtifactAuditError("Pendulum conformance tolerances changed")
    if report.get("planner_dtype") != "float32":
        raise ArtifactAuditError("Pendulum planner conformance dtype changed")
    error_limits = {
        "max_observation_absolute_error": _FORMAL_CONFORMANCE_TOLERANCES["observation_atol"],
        "max_reward_absolute_error": _FORMAL_CONFORMANCE_TOLERANCES["reward_atol"],
        "max_planner_observation_absolute_error": _FORMAL_CONFORMANCE_TOLERANCES["planner_observation_atol"],
        "max_planner_reward_absolute_error": _FORMAL_CONFORMANCE_TOLERANCES["planner_reward_atol"],
    }
    for field, limit in error_limits.items():
        value = report.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= limit
        ):
            raise ArtifactAuditError(f"Pendulum conformance {field} exceeds its fixed tolerance")
    if report.get("passed") is not True:
        raise ArtifactAuditError("Pendulum conformance report did not pass")
    body = dict(report)
    report_sha256 = body.pop("report_sha256", None)
    expected_sha256 = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    if report_sha256 != expected_sha256:
        raise ArtifactAuditError("Pendulum conformance self-hash changed")


def _expected_formal_oscillator_conformance() -> dict[str, object]:
    """Recompute the sealed distractor conformance report from its equations."""

    trajectory = hashlib.sha256()
    for case_index in range(_FORMAL_OSCILLATOR_CONFORMANCE_CASES):
        reset_seed = _FORMAL_OSCILLATOR_CONFORMANCE_SEED + case_index
        digest = hashlib.sha256(f"{_OSCILLATOR_SOURCE}:{reset_seed}".encode("ascii")).digest()
        phase_unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        velocity_unit = int.from_bytes(digest[8:16], "big") / float(1 << 64)
        phase = (2.0 * phase_unit - 1.0) * math.pi
        velocity = 0.5 + velocity_unit
        trajectory.update(reset_seed.to_bytes(8, "big", signed=False))
        trajectory.update(
            struct.pack(
                "<3d",
                math.cos(phase),
                math.sin(phase),
                velocity,
            )
        )
        for step_index in range(_EPISODE_HORIZON):
            phase = math.remainder(
                phase + _OSCILLATOR_TIME_STEP * velocity,
                2.0 * math.pi,
            )
            trajectory.update(
                struct.pack(
                    "<3d",
                    math.cos(phase),
                    math.sin(phase),
                    velocity,
                )
            )
            trajectory.update(struct.pack("<d", math.cos(phase)))
            trajectory.update(
                bytes(
                    (
                        0,
                        int(step_index == _EPISODE_HORIZON - 1),
                    )
                )
            )
    report: dict[str, object] = {
        "schema": ("prospect.wm001.independent-phase-oscillator-conformance.v1"),
        "source_id": _OSCILLATOR_SOURCE,
        "cases": _FORMAL_OSCILLATOR_CONFORMANCE_CASES,
        "steps_per_case": _EPISODE_HORIZON,
        "seed": _FORMAL_OSCILLATOR_CONFORMANCE_SEED,
        "max_reset_absolute_difference": 0.0,
        "max_action_pair_observation_absolute_difference": 0.0,
        "max_action_pair_reward_absolute_difference": 0.0,
        "unexpected_terminations": 0,
        "premature_or_missing_truncations": 0,
        "trajectory_sha256": trajectory.hexdigest(),
        "passed": True,
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json_bytes(report)).hexdigest()
    return report


def _validate_formal_oscillator_conformance_report(
    report: object,
) -> None:
    """Reject a self-consistent report unless every semantic value replays."""

    if report != _expected_formal_oscillator_conformance():
        raise ArtifactAuditError(
            "independent oscillator conformance differs from the independently recomputed sealed trajectories"
        )


def _validate_formal_coverage_conformance_report(report: object) -> None:
    """Recompute the bound endpoint corpus without producer coverage code."""

    if not isinstance(report, dict):
        raise ArtifactAuditError("coverage conformance report is not an object")
    if (
        report.get("schema") != "prospect.wm001.coverage-conformance.v1"
        or report.get("semantics_id") != _COVERAGE_SEMANTICS
        or report.get("python_executable") != sys.executable
        or report.get("python_implementation") != platform.python_implementation()
        or report.get("python_implementation") != "CPython"
        or report.get("python_version") != platform.python_version()
        or report.get("platform") != platform.platform()
        or report.get("machine") != platform.machine()
    ):
        raise ArtifactAuditError("coverage conformance runtime identity differs")
    rows = report.get("cases")
    direct_expected = (
        ("lower-binary64", 0.05, True),
        ("lower-predecessor", math.nextafter(0.05, -math.inf), False),
        ("lower-successor", math.nextafter(0.05, math.inf), True),
        ("upper-binary64", 0.95, True),
        ("upper-predecessor", math.nextafter(0.95, -math.inf), True),
        ("upper-successor", math.nextafter(0.95, math.inf), False),
        ("central", 0.5, True),
        ("zero-tail", 0.0, False),
        ("one-tail", 1.0, False),
    )
    if not isinstance(rows, list) or len(rows) != len(direct_expected) + 1:
        raise ArtifactAuditError("coverage conformance case matrix changed")
    for row, (case_id, pit, expected) in zip(rows[: len(direct_expected)], direct_expected, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("case_id") != case_id
            or row.get("kind") != "binary64_pit"
            or row.get("pit_hex") != pit.hex()
            or row.get("expected_covered") is not expected
            or row.get("observed_covered") is not expected
            or row.get("passed") is not True
            or _binary64_mixture_pit_is_covered(float.fromhex(str(row.get("pit_hex")))) is not expected
        ):
            raise ArtifactAuditError(f"coverage conformance direct case {case_id} changed or failed")
    regression = rows[-1]
    if (
        not isinstance(regression, Mapping)
        or regression.get("case_id") != "v130-disclosed-boundary-coordinate"
        or regression.get("kind") != "float32_mixture_inputs"
        or regression.get("target_little_endian_f32_hex") != _V130_BOUNDARY_TARGET_F32_HEX
        or tuple(regression.get("member_means_little_endian_f32_hex", ())) != _V130_BOUNDARY_MEANS_F32_HEX
        or tuple(regression.get("member_log_variances_little_endian_f32_hex", ()))
        != _V130_BOUNDARY_LOG_VARIANCES_F32_HEX
    ):
        raise ArtifactAuditError("coverage v1.3 boundary regression inputs changed")
    target = float(struct.unpack("<f", bytes.fromhex(_V130_BOUNDARY_TARGET_F32_HEX))[0])
    means = tuple(float(struct.unpack("<f", bytes.fromhex(value))[0]) for value in _V130_BOUNDARY_MEANS_F32_HEX)
    log_variances = tuple(
        float(struct.unpack("<f", bytes.fromhex(value))[0]) for value in _V130_BOUNDARY_LOG_VARIANCES_F32_HEX
    )
    member_cdfs = []
    for mean, log_variance in zip(means, log_variances, strict=True):
        z_score = (target - mean) * math.exp(-0.5 * log_variance)
        member_cdfs.append(0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))))
    pit = math.fsum(member_cdfs) / 5
    if (
        pit.hex() != "0x1.999998b3745adp-5"
        or regression.get("expected_pit_hex") != pit.hex()
        or regression.get("observed_pit_hex") != pit.hex()
        or regression.get("expected_covered") is not False
        or regression.get("observed_covered") is not False
        or regression.get("passed") is not True
        or _binary64_mixture_pit_is_covered(pit)
    ):
        raise ArtifactAuditError("coverage v1.3 boundary regression did not reproduce exactly")
    corpus = {"semantics_id": _COVERAGE_SEMANTICS, "cases": rows}
    if report.get("corpus_sha256") != hashlib.sha256(_canonical_json_bytes(corpus)).hexdigest():
        raise ArtifactAuditError("coverage conformance corpus digest changed")
    body = dict(report)
    report_sha256 = body.pop("report_sha256", None)
    if report.get("passed") is not True or report_sha256 != hashlib.sha256(_canonical_json_bytes(body)).hexdigest():
        raise ArtifactAuditError("coverage conformance self-hash or status changed")


def _audit_bound_irrelevant_control(
    audit: _Audit,
    root: Path,
    block: Mapping[str, object],
) -> None:
    """Bind the distractor implementation and conformance independently."""

    try:
        source_path = root / "source" / "bench" / "world_model_lifecycle" / "runtime_lane.py"
        source_payload = _read_bounded(
            source_path,
            _MAX_SOURCE_FILE_BYTES,
            label="bound independent oscillator source",
        )
        audit.require(
            block.get("id") == _TASK_IRRELEVANT
            and block.get("source_id") == _OSCILLATOR_SOURCE
            and block.get("source_sha256") == hashlib.sha256(source_payload).hexdigest(),
            code="formal_irrelevant_control_source_mismatch",
            message=("formal irrelevant-control identity/source digest differs from the bound runtime source snapshot"),
        )
        report_path = _resolve_artifact_file(
            root,
            block.get("conformance_report_file"),
            label="independent oscillator conformance report",
        )
        report_payload = _read_bounded(
            report_path,
            64 << 20,
            label="independent oscillator conformance report",
        )
        audit.require(
            block.get("conformance_report_bytes") == len(report_payload)
            and block.get("conformance_report_sha256") == hashlib.sha256(report_payload).hexdigest(),
            code="formal_irrelevant_conformance_file_mismatch",
            message=("copied independent oscillator conformance report differs from its formal binding"),
        )
        report = _json_without_duplicate_keys(
            report_payload,
            label="independent oscillator conformance report",
        )
        if not isinstance(report, dict) or report_payload != _canonical_json_bytes(report) + b"\n":
            raise ArtifactAuditError("independent oscillator conformance report is not canonical JSON")
        _validate_formal_oscillator_conformance_report(report)
    except (ArtifactAuditError, OSError, TypeError, ValueError) as error:
        audit.error(
            "formal_irrelevant_control_verification_failed",
            str(error),
        )
    else:
        audit.passed_checks += 1


def _is_bound_implementation_path(path: Path) -> bool:
    relative = path.as_posix()
    if relative in {
        "Makefile",
        "pyproject.toml",
        "requirements-wm001.lock",
        "bench/world_model_lifecycle/SEALED_PROTOCOL.sha256",
        "bench/world_model_lifecycle/protocol.json",
        "bench/world_model_lifecycle/schemas/formal-binding.schema.json",
        "bench/world_model_lifecycle/schemas/raw-result.schema.json",
        "docs/wm001-v1160-confirmation-plan.md",
        "docs/wm001-v1160-operator-runbook.md",
        "docs/wm001-v1160-prospective-harness-review.json",
    }:
        return True
    return (
        path.suffix == ".py"
        and len(path.parts) >= 2
        and (path.parts[:2] == ("src", "prospect") or path.parts[0] == "bench" or path.parts[0] == "tests")
    )


def _audit_bound_source_snapshot(
    audit: _Audit,
    root: Path,
    source: Mapping[str, object],
) -> None:
    """Require an exact regular-file snapshot of every bound source row."""

    manifest = source.get("implementation_files")
    if not isinstance(manifest, list) or not manifest:
        raise ArtifactAuditError("formal binding has no implementation manifest")
    source_root = root / "source"
    if source_root.is_symlink() or not source_root.is_dir():
        raise ArtifactAuditError("formal artifact has no regular source snapshot")
    source_root_resolved = source_root.resolve()
    expected_paths: list[str] = []
    total_bytes = 0
    for index, raw_row in enumerate(manifest):
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ArtifactAuditError(f"implementation_files[{index}] is not an exact file-digest row")
        relative = raw_row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ArtifactAuditError(f"implementation_files[{index}].path is invalid")
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
            or not _is_bound_implementation_path(candidate)
        ):
            raise ArtifactAuditError(f"implementation_files[{index}].path is outside the source contract")
        expected_paths.append(relative)
        snapshot = source_root / candidate
        if (
            snapshot.is_symlink()
            or not snapshot.is_file()
            or not snapshot.resolve().is_relative_to(source_root_resolved)
        ):
            raise ArtifactAuditError(f"bound source snapshot file is missing or aliased: {relative}")
        payload = _read_bounded(
            snapshot,
            _MAX_SOURCE_FILE_BYTES,
            label=f"bound source {relative}",
        )
        total_bytes += len(payload)
        if total_bytes > _MAX_SOURCE_SNAPSHOT_BYTES:
            raise ArtifactAuditError("bound source snapshot exceeds its total byte limit")
        digest = raw_row.get("sha256")
        if (
            raw_row.get("bytes") != len(payload)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or digest != hashlib.sha256(payload).hexdigest()
        ):
            raise ArtifactAuditError(f"bound source snapshot size/digest changed: {relative}")
    if expected_paths != sorted(set(expected_paths)):
        raise ArtifactAuditError("formal implementation manifest is not unique and ordered by path")
    for candidate in source_root.rglob("*"):
        if candidate.is_symlink():
            raise ArtifactAuditError("formal source snapshot contains a symbolic link")
    actual_paths = sorted(
        candidate.relative_to(source_root).as_posix() for candidate in source_root.rglob("*") if candidate.is_file()
    )
    if actual_paths != expected_paths:
        raise ArtifactAuditError("formal source snapshot file set differs from its implementation manifest")
    audit.passed_checks += 1


_FORMAL_EXECUTION_SOURCE_FILES = (
    "adjudication.py",
    "artifact_audit.py",
    "audit_runner.py",
    "binding.py",
    "launch_bootstrap.py",
    "operator.py",
    "preformal.py",
    "producer_bootstrap.py",
    "run.py",
    "verify.py",
)


def _validate_bound_execution_source_manifest(
    root: Path,
    source: Mapping[str, object],
) -> Mapping[str, object]:
    """Tie every formal execution entry source to its preserved snapshot."""

    execution_sources = source.get("execution_source_sha256")
    implementation = source.get("implementation_files")
    if (
        not isinstance(execution_sources, Mapping)
        or set(execution_sources) != set(_FORMAL_EXECUTION_SOURCE_FILES)
        or not isinstance(implementation, list)
    ):
        raise ArtifactAuditError("formal execution-source manifest has the wrong field set")
    expected: dict[str, str] = {}
    for filename in _FORMAL_EXECUTION_SOURCE_FILES:
        relative = f"bench/world_model_lifecycle/{filename}"
        rows = [row for row in implementation if isinstance(row, Mapping) and row.get("path") == relative]
        if len(rows) != 1:
            raise ArtifactAuditError(f"formal execution-source manifest lacks {filename}")
        path = root / "source" / relative
        payload = _read_bounded(
            path,
            _MAX_SOURCE_FILE_BYTES,
            label=f"bound execution source {filename}",
        )
        digest = hashlib.sha256(payload).hexdigest()
        if path.is_symlink() or rows[0].get("bytes") != len(payload) or rows[0].get("sha256") != digest:
            raise ArtifactAuditError(f"bound execution source changed: {filename}")
        expected[filename] = digest
    if dict(execution_sources) != expected:
        raise ArtifactAuditError("formal execution-source digests differ from the source snapshot")
    return execution_sources


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_distribution_sha256(name: str) -> str:
    """Independently reproduce the binding's installed-distribution digest."""

    try:
        distribution = importlib.metadata.distribution(name)
        raw_name = distribution.metadata["Name"]
        if not isinstance(raw_name, str) or not raw_name:
            raise ArtifactAuditError("installed distribution has no canonical name")
        canonical_name = re.sub(r"[-_.]+", "-", raw_name).lower()
        digest, _, _, editable = _prebinding_distribution_digest(
            distribution,
            canonical_name=canonical_name,
        )
    except _PrebindingConformanceError as error:
        raise ArtifactAuditError(f"installed distribution bytes failed v2 verification: {error.code}") from error
    if editable:
        raise ArtifactAuditError("installed distribution is editable")
    return digest


def _live_bound_package_rows(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_BOUND_PACKAGES:
        raise ArtifactAuditError("formal dependency package rows are malformed")
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "name",
            "version",
            "distribution_sha256",
            "declared_file_count",
            "editable",
        }:
            raise ArtifactAuditError(f"formal dependency package row {index} is malformed")
    try:
        return _prebinding_live_package_rows()
    except _PrebindingConformanceError as error:
        raise ArtifactAuditError(f"live package closure failed v2 verification: {error.code}") from error


def _live_record_hash_identity(value: object) -> tuple[str, str] | None:
    """Decode ``FileHash`` fields without depending on version-specific repr."""

    if value is None:
        return None
    mode = getattr(value, "mode", None)
    encoded = getattr(value, "value", None)
    if not isinstance(mode, str) or not mode or not isinstance(encoded, str) or not encoded:
        raise ArtifactAuditError("installed ownership distribution has malformed RECORD hash")
    return mode, encoded


def _live_record_sha256_hex(identity: tuple[str, str]) -> str:
    algorithm, encoded = identity
    if algorithm != "sha256":
        raise ArtifactAuditError("shared package file has non-SHA256 RECORD")
    try:
        decoded = base64.b64decode(
            encoded.encode("ascii") + b"=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise ArtifactAuditError("shared package file has malformed RECORD hash") from error
    if len(decoded) != hashlib.sha256().digest_size:
        raise ArtifactAuditError("shared package file has malformed SHA256 RECORD")
    return decoded.hex()


def _live_package_ownership(root_value: object) -> dict[str, object]:
    """Independently prove exact RECORD ownership of the live import root."""

    if not isinstance(root_value, str) or not root_value:
        raise ArtifactAuditError("formal package-ownership root is invalid")
    root = Path(root_value)
    try:
        if not root.is_absolute() or root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
            raise ArtifactAuditError("formal package-ownership root is absent or aliased")
    except OSError as error:
        raise ArtifactAuditError("formal package-ownership root is unavailable") from error

    distributions: dict[str, importlib.metadata.Distribution] = {}
    owners: dict[str, list[tuple[str, tuple[str, str] | None]]] = {}
    for distribution in importlib.metadata.distributions(path=[str(root)]):
        raw_name = distribution.metadata["Name"]
        if not isinstance(raw_name, str) or not raw_name:
            raise ArtifactAuditError("installed ownership distribution has no canonical name")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if name in distributions:
            raise ArtifactAuditError("installed ownership distribution identity is duplicated")
        distributions[name] = distribution
        declared = tuple(distribution.files or ())
        if not declared:
            raise ArtifactAuditError(f"installed ownership distribution has no RECORD files: {name}")
        for entry in declared:
            located = Path(os.path.abspath(str(distribution.locate_file(entry))))
            if not located.is_relative_to(root):
                continue
            relative = located.relative_to(root).as_posix()
            owners.setdefault(relative, []).append(
                (name, _live_record_hash_identity(entry.hash))
            )

    def discover() -> tuple[
        dict[str, tuple[Path, tuple[int, ...]]],
        dict[str, tuple[int, ...]],
    ]:
        files: dict[str, tuple[Path, tuple[int, ...]]] = {}
        directories: dict[str, tuple[int, ...]] = {}
        for directory, directory_names, filenames in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            current = Path(directory)
            directory_names.sort()
            filenames.sort()
            for name in directory_names:
                path = current / name
                relative = path.relative_to(root).as_posix()
                try:
                    metadata = path.lstat()
                except OSError as error:
                    raise ArtifactAuditError(f"package ownership directory is unavailable: {relative}") from error
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise ArtifactAuditError(f"package ownership found non-directory: {relative}")
                if name == "__pycache__":
                    raise ArtifactAuditError(f"package ownership found bytecode cache: {relative}")
                directories[relative] = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_uid,
                    metadata.st_gid,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            for name in filenames:
                path = current / name
                relative = path.relative_to(root).as_posix()
                try:
                    metadata = path.lstat()
                except OSError as error:
                    raise ArtifactAuditError(f"package ownership file is unavailable: {relative}") from error
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise ArtifactAuditError(f"package ownership found non-regular file: {relative}")
                if path.suffix == ".pyc":
                    raise ArtifactAuditError(f"package ownership found bytecode: {relative}")
                files[relative] = (
                    path,
                    (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mode,
                        metadata.st_nlink,
                        metadata.st_uid,
                        metadata.st_gid,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    ),
                )
                if len(files) > _MAX_PREBINDING_ROOT_ENTRIES:
                    raise ArtifactAuditError("package ownership file limit exceeded")
        return files, directories

    initial_files, initial_directories = discover()
    if set(owners) != set(initial_files):
        raise ArtifactAuditError("live package-root RECORD ownership is not exact")
    implied_directories = {
        parent.as_posix() for relative in initial_files for parent in Path(relative).parents if parent != Path(".")
    }
    if set(initial_directories) != implied_directories:
        raise ArtifactAuditError("live package-root directory ownership is not exact")

    ownership_rows: list[dict[str, object]] = []
    shared_file_count = 0
    for relative in sorted(initial_files):
        file_owners = sorted(owners[relative])
        owner_names = [name for name, _ in file_owners]
        if len(owner_names) != len(set(owner_names)):
            raise ArtifactAuditError(f"package file has duplicate RECORD owner: {relative}")
        if len(file_owners) > 1:
            declared_hashes = {value for _, value in file_owners}
            if None in declared_hashes or len(declared_hashes) != 1:
                raise ArtifactAuditError(f"package file has conflicting RECORD owners: {relative}")
            record_identity = cast(
                tuple[str, str],
                next(iter(declared_hashes)),
            )
            expected = _live_record_sha256_hex(record_identity)
            path, file_identity = initial_files[relative]
            observed_digest = _sha256_file(path)
            metadata = path.lstat()
            observed_identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            if observed_digest != expected or observed_identity != file_identity:
                raise ArtifactAuditError(f"shared package file differs from its RECORD: {relative}")
            shared_file_count += 1
        ownership_rows.append(
            {
                "path": relative,
                "owners": owner_names,
            }
        )
    final_files, final_directories = discover()
    if {relative: identity for relative, (_, identity) in initial_files.items()} != {
        relative: identity for relative, (_, identity) in final_files.items()
    } or initial_directories != final_directories:
        raise ArtifactAuditError("package-root namespace changed during ownership verification")
    identity = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(root),
        "files": ownership_rows,
        "directories": sorted(initial_directories),
    }
    return {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(root),
        "file_count": len(initial_files),
        "directory_count": len(initial_directories),
        "shared_file_count": shared_file_count,
        "identity_sha256": hashlib.sha256(_canonical_json_bytes(identity)).hexdigest(),
    }


def _cuda_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    versions = sorted({row.strip() for row in completed.stdout.splitlines() if row.strip()})
    return ",".join(versions) if versions else None


def _live_runtime_identity(device: str) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        raise ArtifactAuditError("PyTorch is required to verify the bound formal runtime") from error
    if device not in {"cpu", "cuda", "mps"}:
        raise ArtifactAuditError("formal binding has an invalid runtime device")
    if device == "cuda" and not torch.cuda.is_available():
        raise ArtifactAuditError("bound CUDA runtime is unavailable to the independent auditor")
    if device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise ArtifactAuditError("bound MPS runtime is unavailable to the independent auditor")
    torch.use_deterministic_algorithms(True)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "accelerator": (torch.cuda.get_device_name(0) if device == "cuda" else None),
        "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
        "thread_count": torch.get_num_threads(),
        "interop_thread_count": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": (_cuda_driver_version() if device == "cuda" else None),
        "cublas_workspace_config": (os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None),
    }


def _audit_formal_runtime_binding(
    audit: _Audit,
    *,
    runtime: Mapping[str, object],
    dependencies: Mapping[str, object],
    execution: Mapping[str, object],
) -> str:
    """Bind result arithmetic and the live auditor to the pre-run runtime."""

    device = runtime.get("device")
    if not isinstance(device, str) or device not in {"cpu", "cuda", "mps"}:
        raise ArtifactAuditError("formal binding runtime device is invalid")
    runtime_fields = (
        "platform",
        "machine",
        "device",
        "python_flags",
        "process_environment",
        "accelerator",
        "thread_count",
        "interop_thread_count",
        "cuda_runtime",
        "cuda_driver",
        "cublas_workspace_config",
        "deterministic_algorithms",
    )
    audit.require(
        {field: execution.get(field) for field in runtime_fields}
        == {field: runtime.get(field) for field in runtime_fields}
        and execution.get("deterministic_algorithms") is True,
        code="formal_result_runtime_binding_mismatch",
        message=(
            "formal result platform, machine, interpreter isolation, process "
            "environment, accelerator, thread, CUDA, or deterministic state "
            "differs from its complete pre-run runtime binding"
        ),
    )
    producer_flags = runtime.get("python_flags")
    producer_environment = runtime.get("process_environment")
    shared_runtime = {
        key: value for key, value in runtime.items() if key not in {"python_flags", "process_environment"}
    }
    live_runtime = _live_runtime_identity(device)
    audit.require(
        isinstance(producer_flags, Mapping)
        and dict(producer_flags) == dict(_PREBINDING_PRODUCER_FLAGS)
        and isinstance(producer_environment, Mapping)
        and {
            "CUBLAS_WORKSPACE_CONFIG",
            "LAZY_LEGACY_OP",
            "LC_ALL",
            "PATH",
            "PYGAME_HIDE_SUPPORT_PROMPT",
            "SDL_AUDIODRIVER",
            "TZ",
        }.issubset(producer_environment)
        and set(producer_environment).issubset(_PREBINDING_PROCESS_ENVIRONMENT_KEYS)
        and producer_environment.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and producer_environment.get("LAZY_LEGACY_OP") == "False"
        and producer_environment.get("LC_ALL") == "C.UTF-8"
        and producer_environment.get("PATH") == "/usr/bin:/bin"
        and producer_environment.get("PYGAME_HIDE_SUPPORT_PROMPT") == "hide"
        and producer_environment.get("SDL_AUDIODRIVER") == "dsp"
        and producer_environment.get("TZ") == "UTC"
        and all(
            os.environ.get(key) == value
            for key, value in producer_environment.items()
            if key in _PREBINDING_SHARED_ENVIRONMENT_KEYS
        )
        and _prebinding_live_python_flags() == dict(_PREBINDING_AUDITOR_FLAGS)
        and live_runtime == shared_runtime,
        code="formal_auditor_runtime_binding_mismatch",
        message=(
            "independent auditor isolation, shared process environment, "
            "platform, accelerator, CUDA, or Torch runtime differs from the "
            "pre-run producer binding"
        ),
    )
    executable = dependencies.get("python_executable")
    executable_path = Path(executable) if isinstance(executable, str) else None
    audit.require(
        executable_path is not None
        and executable_path.is_absolute()
        and executable == sys.executable
        and dependencies.get("python_executable_sha256") == _sha256_file(Path(sys.executable).resolve()),
        code="formal_auditor_python_executable_mismatch",
        message=("independent auditor executable path or bytes differ from the pre-run dependency binding"),
    )
    package_roots = dependencies.get("package_roots")
    try:
        standard_library = dependencies.get("standard_library")
        if not isinstance(standard_library, Mapping):
            raise ArtifactAuditError("formal dependency standard-library row is malformed")
        standard_actual = _prebinding_root_inventory(
            "standard-library",
            standard_library.get("path"),
            kind="standard_library",
        )
        standard_public = {
            "path": standard_library.get("path"),
            **{key: value for key, value in standard_actual.items() if key not in {"id", "kind"}},
        }
        if not isinstance(package_roots, list) or not package_roots:
            raise ArtifactAuditError("formal dependency package-root rows are malformed")
        package_actual = []
        for index, row in enumerate(package_roots):
            if not isinstance(row, Mapping):
                raise ArtifactAuditError("formal dependency package-root row is malformed")
            actual = _prebinding_root_inventory(
                f"package-root-{index:04d}",
                row.get("path"),
                kind="package_root",
            )
            package_actual.append(
                {
                    "path": row.get("path"),
                    **{key: value for key, value in actual.items() if key not in {"id", "kind"}},
                }
            )
    except (_PrebindingConformanceError, ArtifactAuditError):
        inventory_matches = False
    else:
        inventory_matches = standard_public == dict(standard_library) and package_actual == [
            dict(row) for row in package_roots
        ]
    audit.require(
        inventory_matches,
        code="formal_auditor_root_inventory_mismatch",
        message=(
            "independent auditor standard-library or complete package-root "
            "bytes differ from the pre-run dependency binding"
        ),
    )
    bound_ownership = dependencies.get("package_ownership")
    ownership_matches = False
    if (
        isinstance(bound_ownership, Mapping)
        and isinstance(package_roots, list)
        and len(package_roots) == 1
        and isinstance(package_roots[0], Mapping)
    ):
        try:
            live_ownership = _live_package_ownership(package_roots[0].get("path"))
        except ArtifactAuditError:
            ownership_matches = False
        else:
            ownership_matches = live_ownership == dict(bound_ownership)
    audit.require(
        ownership_matches,
        code="formal_auditor_package_ownership_mismatch",
        message=("independent auditor package-root RECORD ownership differs from the pre-run binding"),
    )
    package_rows = dependencies.get("packages")
    audit.require(
        _live_bound_package_rows(package_rows) == package_rows,
        code="formal_auditor_dependency_binding_mismatch",
        message=("independent auditor dependency bytes differ from the pre-run binding"),
    )
    return device


def _canonical_json_object_payload(
    payload: bytes,
    *,
    label: str,
) -> Mapping[str, object]:
    value = _json_without_duplicate_keys(payload, label=label)
    if not isinstance(value, Mapping) or payload != _canonical_json_bytes(value) + b"\n":
        raise ArtifactAuditError(f"{label} is not one canonical JSON object followed by LF")
    return value


def _preformal_result_free_inventory(
    value: object,
    *,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate the recorded runtime inventory without using ambient QA state."""

    if not isinstance(value, Mapping) or set(value) != {
        "packages",
        "package_roots",
        "standard_library",
        "package_ownership",
    }:
        raise ArtifactAuditError(
            "preformal runtime inventory has wrong fields"
        )
    packages = value.get("packages")
    package_roots = value.get("package_roots")
    standard_library = value.get("standard_library")
    package_ownership = value.get("package_ownership")
    package_fields = {
        "name",
        "version",
        "distribution_sha256",
        "declared_file_count",
        "editable",
    }
    if not isinstance(packages, list) or not packages:
        raise ArtifactAuditError(
            "preformal runtime package inventory is absent"
        )
    names: list[str] = []
    for row in packages:
        if (
            not isinstance(row, Mapping)
            or set(row) != package_fields
            or not isinstance(row.get("name"), str)
            or re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*",
                cast(str, row.get("name")),
            )
            is None
            or not isinstance(row.get("version"), str)
            or not row.get("version")
            or "\0" in cast(str, row.get("version"))
            or not _is_sha256(row.get("distribution_sha256"))
            or type(row.get("declared_file_count")) is not int
            or cast(int, row.get("declared_file_count")) <= 0
            or row.get("editable") is not False
        ):
            raise ArtifactAuditError(
                "preformal runtime package row is malformed"
            )
        names.append(cast(str, row["name"]))
    if (
        names[0] != "python"
        or names[1:] != sorted(names[1:])
        or len(names) != len(set(names))
        or re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+",
            cast(str, packages[0]["version"]),
        )
        is None
    ):
        raise ArtifactAuditError(
            "preformal runtime packages are duplicated or unordered"
        )

    root_fields = {
        "semantics_id",
        "path",
        "file_count",
        "directory_count",
        "total_bytes",
        "inventory_sha256",
    }

    def valid_root(row: object, *, semantics_id: str) -> bool:
        if not isinstance(row, Mapping) or set(row) != root_fields:
            return False
        path = row.get("path")
        return bool(
            row.get("semantics_id") == semantics_id
            and isinstance(path, str)
            and "\0" not in path
            and Path(path).is_absolute()
            and os.path.abspath(path) == path
            and type(row.get("file_count")) is int
            and cast(int, row["file_count"]) > 0
            and type(row.get("directory_count")) is int
            and cast(int, row["directory_count"]) >= 0
            and type(row.get("total_bytes")) is int
            and cast(int, row["total_bytes"]) >= 0
            and _is_sha256(row.get("inventory_sha256"))
        )

    if (
        not isinstance(package_roots, list)
        or len(package_roots) != 1
        or not valid_root(
            package_roots[0],
            semantics_id="prospect.wm001.package-root.v2",
        )
        or not valid_root(
            standard_library,
            semantics_id="prospect.wm001.standard-library.v2",
        )
        or not isinstance(package_ownership, Mapping)
        or set(package_ownership)
        != {
            "semantics_id",
            "root",
            "file_count",
            "directory_count",
            "shared_file_count",
            "identity_sha256",
        }
        or package_ownership.get("semantics_id")
        != "prospect.wm001.package-ownership.v1"
        or package_ownership.get("root")
        != cast(Mapping[str, object], package_roots[0]).get("path")
        or package_ownership.get("file_count")
        != cast(Mapping[str, object], package_roots[0]).get("file_count")
        or package_ownership.get("directory_count")
        != cast(Mapping[str, object], package_roots[0]).get(
            "directory_count"
        )
        or type(package_ownership.get("shared_file_count")) is not int
        or cast(int, package_ownership["shared_file_count"]) < 0
        or cast(int, package_ownership["shared_file_count"])
        > cast(int, package_ownership["file_count"])
        or not _is_sha256(package_ownership.get("identity_sha256"))
        or packages != dependencies.get("packages")
        or package_roots != dependencies.get("package_roots")
        or standard_library != dependencies.get("standard_library")
        or package_ownership != dependencies.get("package_ownership")
    ):
        raise ArtifactAuditError(
            "preformal runtime inventory differs from bound dependencies"
        )
    return value


def _preformal_fresh_identity_conformance(
    value: object,
) -> Mapping[str, object]:
    fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "mode",
        "challenge",
        "requesting_process_id",
        "verifier_process_id",
        "matrix_contract_sha256",
        "passed",
    }
    requesting_process_id = (
        value.get("requesting_process_id")
        if isinstance(value, Mapping)
        else None
    )
    verifier_process_id = (
        value.get("verifier_process_id")
        if isinstance(value, Mapping)
        else None
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema")
        != "prospect.wm001.fresh-runtime-identity-conformance.v1"
        or value.get("experiment_id") != "WM-001"
        or value.get("protocol_version") != "1.16.0"
        or value.get("mode") != "fresh-identity-conformance"
        or not _is_sha256(value.get("challenge"))
        or type(requesting_process_id) is not int
        or cast(int, requesting_process_id) <= 0
        or type(verifier_process_id) is not int
        or cast(int, verifier_process_id) <= 0
        or requesting_process_id == verifier_process_id
        or value.get("matrix_contract_sha256")
        != _DEVELOPMENT_MATRIX_CONTRACT_SHA256
        or value.get("passed") is not True
    ):
        raise ArtifactAuditError(
            "preformal fresh runtime identity conformance is malformed"
        )
    return value


def _preformal_runtime_conformance(
    value: object,
    *,
    dependencies: Mapping[str, object],
    device: object,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactAuditError(
            "preformal runtime bootstrap conformance is incomplete"
        )
    inventory = _preformal_result_free_inventory(
        value.get("inventory"),
        dependencies=dependencies,
    )
    fresh = _preformal_fresh_identity_conformance(
        value.get("fresh_runtime_identity_conformance")
    )
    if (
        set(value)
        != {
            "schema",
            "mode",
            "device",
            "passed",
            "inventory",
            "inventory_sha256",
            "conformance_sha256",
            "fresh_runtime_identity_conformance",
            "fresh_runtime_identity_conformance_sha256",
            "restart_runtime_conformance_report_sha256",
            "restart_runtime_execution_receipt_sha256",
            "restart_runtime_support_files",
            "restart_runtime_repeat_count",
            "restart_runtime_path_descriptor_equal",
            "repeat_count",
            "path_descriptor_equal",
        }
        or value.get("schema")
        != "prospect.wm001.preformal-runtime-check.v1"
        or value.get("mode") != "bootstrap-inventory-conformance"
        or value.get("device") != device
        or value.get("passed") is not True
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "inventory_sha256",
                "conformance_sha256",
                "fresh_runtime_identity_conformance_sha256",
                "restart_runtime_conformance_report_sha256",
                "restart_runtime_execution_receipt_sha256",
            )
        )
        or value.get("inventory_sha256")
        != hashlib.sha256(_canonical_json_bytes(inventory)).hexdigest()
        or value.get("fresh_runtime_identity_conformance_sha256")
        != hashlib.sha256(_canonical_json_bytes(fresh)).hexdigest()
        or value.get("restart_runtime_support_files")
        != list(_RESTART_RUNTIME_SUPPORT_FILES)
        or value.get("restart_runtime_repeat_count") != 3
        or value.get("restart_runtime_path_descriptor_equal") is not True
        or value.get("repeat_count") != 3
        or value.get("path_descriptor_equal") is not True
    ):
        raise ArtifactAuditError(
            "preformal runtime bootstrap conformance is incomplete"
        )
    return value


_PREFORMAL_REPORT_NAME = "preformal-test-report-v1.16.0.json"
_PREFORMAL_LOG_PREFIX = "preformal-v1.16.0-command-"
_PREFORMAL_REVIEW_PATH = "docs/wm001-v1160-prospective-harness-review.json"
_PREFORMAL_LIVE_RELATIVE_DIRECTORY = Path("v1.16.0") / "preformal"
_FORMAL_INPUT_PREFLIGHT_NAME = "formal-input-preflight.json"
_DEVELOPMENT_RESULT_QUALIFICATION_NAME = (
    "development-result-qualification.json"
)
_PREFORMAL_COMMAND_NAMES = (
    "protocol-seal-continuity",
    "ruff",
    "mypy-core",
    "mypy-wm001",
    "pytest-epistemic",
    "pytest-wm001",
    "audit-runner-adversarial",
    "prospective-harness-review",
    "runtime-accepted-closure-evidence",
    "runtime-bootstrap-inventory-conformance",
)
_PREFORMAL_AUTHORIZATION_CONTRACT: Mapping[str, object] = {
    "report_schema": "prospect.wm001.preformal-test-report.v2",
    "canonical_directory": (
        "bench/world_model_lifecycle/results/development/"
        "v1.16.0/preformal"
    ),
    "report_file": _PREFORMAL_REPORT_NAME,
    "ordered_commands": list(_PREFORMAL_COMMAND_NAMES),
    "all_command_stderr_bytes": 0,
    "input_link_counts": {
        "closure_attempt_terminal": 2,
        "closure_outer_completion": 2,
        "development_closure": 1,
        "launch_bootstrap": 1,
        "producer_bootstrap": 1,
        "prospective_review": 1,
        "runtime_seal": 2,
    },
    "accepted_closure_output_fields": [
        "schema",
        "mode",
        "passed",
        "development_closure_sha256",
        "producer_manifest_sha256",
        "raw_result_sha256",
        "closure_attempt_manifest_sha256",
        "closure_outer_completion_sha256",
    ],
    "accepted_closure_stderr_bytes": 0,
    "runtime_conformance_output_fields": [
        "schema",
        "mode",
        "device",
        "passed",
        "inventory",
        "inventory_sha256",
        "conformance_sha256",
        "fresh_runtime_identity_conformance",
        "fresh_runtime_identity_conformance_sha256",
        "restart_runtime_conformance_report_sha256",
        "restart_runtime_execution_receipt_sha256",
        "restart_runtime_support_files",
        "restart_runtime_repeat_count",
        "restart_runtime_path_descriptor_equal",
        "repeat_count",
        "path_descriptor_equal",
    ],
    "runtime_conformance_stderr_bytes": 0,
    "claim_rule": (
        "Generation exclusively creates the deterministic hidden sibling "
        ".preformal.staging and fsyncs its parent before the first command. "
        "Either that hidden claim or the final bundle consumes the one-shot "
        "attempt and forbids same-version retry."
    ),
    "publication_rule": (
        "Run all ten commands while the canonical bundle remains absent, "
        "stage exactly 20 content-addressed logs and one report under the "
        "hidden claim, then atomically rename that complete directory with "
        "no replacement. No other bundle member is permitted."
    ),
    "failure_rule": (
        "The report preserves every command result. all_pass exactly equals "
        "the conjunction of the ten zero exit statuses, ten exactly empty "
        "stderr streams, a clean post-command Git worktree, and stable "
        "pre/post Git, source, executable, package, "
        "runtime-seal, review, and input identities, plus strict semantic "
        "validation of accepted-closure and runtime-conformance stdout. "
        "Generation returns nonzero with passed=false, exact failed-command "
        "ordinals, names, and exit codes, and exact non-command failure "
        "identifiers for nonempty command stderr, post-run worktree drift, "
        "pre/post identity drift, accepted-closure semantics, or runtime-"
        "conformance semantics. Failed evidence can never authorize binding."
    ),
    "public_runtime_modes": [
        "accepted-closure-evidence",
        "bootstrap-inventory-conformance",
        "fresh-closure-reopen",
        "fresh-identity-conformance",
    ],
    "formal_input_preflight": {
        "required": True,
        "schema": "prospect.wm001.formal-input-preflight.v1",
        "receipt_file": _FORMAL_INPUT_PREFLIGHT_NAME,
        "validated_before_binding_acceptance": True,
        "required_by_outer_launcher": True,
        "outer_launcher_runtime_output_digests": True,
        "preserved_in_formal_artifact": True,
    },
}
_PREFORMAL_INPUT_FIELDS = frozenset(
    {
        "closure_attempt_terminal",
        "closure_outer_completion",
        "development_closure",
        "launch_bootstrap",
        "producer_bootstrap",
        "prospective_review",
        "runtime_seal",
    }
)
_PREFORMAL_MYPY_FILES = (
    "bench/world_model_lifecycle/audit_runner.py",
    "bench/world_model_lifecycle/artifact.py",
    "bench/world_model_lifecycle/artifact_audit.py",
    "bench/world_model_lifecycle/adjudication.py",
    "bench/world_model_lifecycle/binding.py",
    "bench/world_model_lifecycle/experiment.py",
    "bench/world_model_lifecycle/launch_bootstrap.py",
    "bench/world_model_lifecycle/operator.py",
    "bench/world_model_lifecycle/preformal.py",
    "bench/world_model_lifecycle/producer_bootstrap.py",
    "bench/world_model_lifecycle/restore_eval.py",
    "bench/world_model_lifecycle/run.py",
)
_PREFORMAL_FIXED_ENVIRONMENT: Mapping[str, str] = {
    "COLUMNS": "120",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LAZY_LEGACY_OP": "False",
    "NO_COLOR": "1",
    "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "SDL_AUDIODRIVER": "dsp",
    "TERM": "dumb",
    "TZ": "UTC",
}
_PREFORMAL_OPTIONAL_ENVIRONMENT = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "ROCR_VISIBLE_DEVICES",
    }
)
_PREFORMAL_RUNTIME_ENVIRONMENT = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "LAZY_LEGACY_OP",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYGAME_HIDE_SUPPORT_PROMPT",
        "ROCR_VISIBLE_DEVICES",
        "SDL_AUDIODRIVER",
        "TZ",
    }
)
_PREFORMAL_RUNTIME_FLAGS: Mapping[str, object] = {
    "dont_write_bytecode": 1,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "safe_path": True,
}
_LIVE_REPOSITORY_ROOT = Path.cwd().resolve(strict=True)


def _preformal_environment(
    identity: object,
    *,
    role: str,
    runtime: Mapping[str, object],
) -> tuple[dict[str, str], str]:
    if not isinstance(identity, Mapping) or set(identity) != {
        "variables",
        "sha256",
    }:
        raise ArtifactAuditError("formal test report environment identity has wrong fields")
    raw_variables = identity.get("variables")
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ArtifactAuditError("formal test report environment variables are absent")
    variables: list[tuple[str, str]] = []
    for row in raw_variables:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"name", "value"}
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("value"), str)
        ):
            raise ArtifactAuditError("formal test report environment row is malformed")
        name = cast(str, row.get("name"))
        value = cast(str, row.get("value"))
        if "\0" in name or "\0" in value:
            raise ArtifactAuditError("formal test report environment contains NUL")
        variables.append((name, value))
    if variables != sorted(variables) or len({name for name, _ in variables}) != len(variables):
        raise ArtifactAuditError("formal test report environment is duplicated or unordered")
    environment = dict(variables)
    expected_digest = hashlib.sha256(
        _canonical_json_bytes([{"name": name, "value": value} for name, value in variables])
    ).hexdigest()
    if role == "qa":
        allowed = {
            *_PREFORMAL_FIXED_ENVIRONMENT,
            *_PREFORMAL_OPTIONAL_ENVIRONMENT,
            "PATH",
        }
        valid = (
            not set(environment) - allowed
            and environment.get("PATH") not in {None, ""}
            and all(environment.get(name) == value for name, value in _PREFORMAL_FIXED_ENVIRONMENT.items())
        )
    elif role == "runtime":
        process_environment = runtime.get("process_environment")
        valid = (
            isinstance(process_environment, Mapping)
            and environment == dict(process_environment)
            and not set(environment) - _PREFORMAL_RUNTIME_ENVIRONMENT
            and environment.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and environment.get("LAZY_LEGACY_OP") == "False"
            and environment.get("LC_ALL") == "C.UTF-8"
            and environment.get("PATH") == "/usr/bin:/bin"
            and environment.get("PYGAME_HIDE_SUPPORT_PROMPT") == "hide"
            and environment.get("SDL_AUDIODRIVER") == "dsp"
            and environment.get("TZ") == "UTC"
        )
    else:
        raise ArtifactAuditError("formal test report has an unknown environment role")
    if not valid or identity.get("sha256") != expected_digest:
        raise ArtifactAuditError(f"formal test report {role} environment differs from its contract")
    return environment, expected_digest


def _preformal_git_identity(
    identity: object,
    *,
    source: Mapping[str, object],
    label: str,
) -> Mapping[str, object]:
    def is_sha1(value: object) -> bool:
        return (
            isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)
        )

    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"commit", "tree", "worktree_clean"}
        or identity.get("commit") != source.get("git_commit")
        or identity.get("tree") != source.get("git_tree")
        or identity.get("worktree_clean") is not True
        or not is_sha1(identity.get("commit"))
        or not is_sha1(identity.get("tree"))
    ):
        raise ArtifactAuditError(f"formal test report {label} Git identity is invalid")
    return identity


def _preformal_qa_executable_identity(
    identity: object,
) -> Mapping[str, object]:
    fields = {
        "invocation_path",
        "invocation_symlink_target",
        "resolved_path",
        "bytes",
        "sha256",
        "implementation",
        "version",
    }
    if not isinstance(identity, Mapping) or set(identity) != fields:
        raise ArtifactAuditError("formal test report QA executable identity has wrong fields")
    invocation = identity.get("invocation_path")
    resolved = identity.get("resolved_path")
    link_target = identity.get("invocation_symlink_target")
    if (
        not isinstance(invocation, str)
        or not isinstance(resolved, str)
        or not Path(invocation).is_absolute()
        or not Path(resolved).is_absolute()
        or os.path.abspath(invocation) != invocation
        or os.path.abspath(resolved) != resolved
        or (link_target is not None and not isinstance(link_target, str))
        or type(identity.get("bytes")) is not int
        or cast(int, identity["bytes"]) <= 0
        or not _is_sha256(identity.get("sha256"))
        or identity.get("implementation") != "CPython"
        or not isinstance(identity.get("version"), str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", cast(str, identity["version"])) is None
    ):
        raise ArtifactAuditError("formal test report QA executable identity is malformed")
    return identity


def _preformal_runtime_executable_identity(
    identity: object,
    *,
    dependencies: Mapping[str, object],
    runtime_seal: object,
) -> Mapping[str, object]:
    fields = {
        "invocation_path",
        "invocation_symlink_target",
        "resolved_path",
        "bytes",
        "sha256",
        "implementation",
        "version",
    }
    if not isinstance(identity, Mapping) or set(identity) != fields:
        raise ArtifactAuditError("formal test report executable identity has wrong fields")
    invocation_value = identity.get("invocation_path")
    resolved_value = identity.get("resolved_path")
    seal_python = (
        runtime_seal.get("python")
        if isinstance(runtime_seal, Mapping)
        else None
    )
    packages = dependencies.get("packages")
    python_rows = (
        [
            row
            for row in packages
            if isinstance(row, Mapping) and row.get("name") == "python"
        ]
        if isinstance(packages, list)
        else []
    )
    expected_symlink_target = (
        os.readlink(invocation_value)
        if isinstance(invocation_value, str) and Path(invocation_value).is_symlink()
        else None
    )
    if (
        not isinstance(invocation_value, str)
        or not isinstance(resolved_value, str)
        or not Path(invocation_value).is_absolute()
        or not Path(resolved_value).is_absolute()
        or os.path.abspath(invocation_value) != invocation_value
        or os.path.abspath(resolved_value) != resolved_value
        or identity.get("invocation_symlink_target") != expected_symlink_target
        or identity.get("implementation") != "CPython"
        or not isinstance(seal_python, Mapping)
        or set(seal_python)
        != {"executable", "resolved_executable", "sha256", "version"}
        or len(python_rows) != 1
        or not isinstance(seal_python.get("version"), list)
        or len(cast(list[object], seal_python["version"])) != 3
        or any(
            type(part) is not int or cast(int, part) < 0
            for part in cast(list[object], seal_python["version"])
        )
        or identity.get("version")
        != ".".join(
            str(part)
            for part in cast(list[object], seal_python["version"])
        )
        or identity.get("version") != python_rows[0].get("version")
        or identity.get("invocation_path") != dependencies.get("python_executable")
        or identity.get("invocation_path") != seal_python.get("executable")
        or identity.get("resolved_path")
        != seal_python.get("resolved_executable")
        or identity.get("sha256") != dependencies.get("python_executable_sha256")
        or identity.get("sha256") != seal_python.get("sha256")
        or identity.get("sha256")
        != python_rows[0].get("distribution_sha256")
    ):
        raise ArtifactAuditError("formal test report executable differs from the bound interpreter")
    try:
        resolved = Path(invocation_value).resolve(strict=True)
    except OSError as error:
        raise ArtifactAuditError("formal test report executable cannot be reopened") from error
    executable_payload, executable_bytes, executable_digest, _ = _read_stable_regular_file(
        resolved,
        512 << 20,
        label="bound preformal Python executable",
        capture_payload=False,
    )
    assert executable_payload is None
    if (
        str(resolved) != resolved_value
        or type(identity.get("bytes")) is not int
        or identity.get("bytes") != executable_bytes
        or identity.get("sha256") != executable_digest
    ):
        raise ArtifactAuditError("formal test report executable bytes changed")
    return identity


def _preformal_expected_commands(
    *,
    qa_executable: str,
    runtime_executable: str,
    source: Mapping[str, object],
    repository_cwd: str,
    runtime_seal_path: str,
    development_closure_path: str,
    closure_attempt_path: str,
    prospective_review_path: str,
    device: str,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    implementation = source.get("implementation_files")
    if not isinstance(implementation, list):
        raise ArtifactAuditError("formal test report has no bound implementation manifest")
    paths: list[str] = []
    for row in implementation:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
        ):
            raise ArtifactAuditError("formal test report implementation manifest is malformed")
        paths.append(cast(str, row.get("path")))
    epistemic_tests = tuple(
        sorted(
            path
            for path in paths
            if path.startswith("tests/test_epistemic_") and path.endswith(".py") and len(Path(path).parts) == 2
        )
    )
    wm001_tests = tuple(
        sorted(
            path
            for path in paths
            if path.startswith("tests/test_world_model_") and path.endswith(".py") and len(Path(path).parts) == 2
        )
    )
    if not epistemic_tests or not wm001_tests or "tests/test_world_model_audit_runner.py" not in wm001_tests:
        raise ArtifactAuditError("formal test report source snapshot lacks a required test set")
    launch = f"{repository_cwd}/bench/world_model_lifecycle/launch_bootstrap.py"
    bootstrap = f"{repository_cwd}/bench/world_model_lifecycle/producer_bootstrap.py"
    runtime_prefix = (
        runtime_executable,
        "-I",
        "-S",
        "-B",
        launch,
        "--bootstrap",
        bootstrap,
        "--runtime-seal",
        runtime_seal_path,
        "preformal-runtime",
    )
    return (
        (
            "protocol-seal-continuity",
            "qa",
            (
                qa_executable,
                "-m",
                "bench.world_model_lifecycle.verify",
                "protocol",
            ),
        ),
        (
            "ruff",
            "qa",
            (
                qa_executable,
                "-m",
                "ruff",
                "check",
                "src/prospect",
                "bench",
                "tests",
            ),
        ),
        ("mypy-core", "qa", (qa_executable, "-m", "mypy")),
        (
            "mypy-wm001",
            "qa",
            (
                qa_executable,
                "-m",
                "mypy",
                "--follow-imports=skip",
                *_PREFORMAL_MYPY_FILES,
            ),
        ),
        (
            "pytest-epistemic",
            "qa",
            (qa_executable, "-m", "pytest", "-q", *epistemic_tests),
        ),
        (
            "pytest-wm001",
            "qa",
            (qa_executable, "-m", "pytest", "-q", *wm001_tests),
        ),
        (
            "audit-runner-adversarial",
            "qa",
            (
                qa_executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_world_model_audit_runner.py",
                "tests/test_world_model_prebinding_audit.py",
            ),
        ),
        (
            "prospective-harness-review",
            "qa",
            (
                qa_executable,
                "-I",
                "-B",
                "-m",
                "bench.world_model_lifecycle.preformal",
                "verify-prospective-review",
                "--review",
                prospective_review_path,
            ),
        ),
        (
            "runtime-accepted-closure-evidence",
            "runtime",
            (
                *runtime_prefix,
                "accepted-closure-evidence",
                "--development-closure",
                development_closure_path,
                "--closure-attempt",
                closure_attempt_path,
            ),
        ),
        (
            "runtime-bootstrap-inventory-conformance",
            "runtime",
            (
                *runtime_prefix,
                "bootstrap-inventory-conformance",
                "--device",
                device,
            ),
        ),
    )


def _preformal_file_identity(
    identity: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"path", "bytes", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not Path(cast(str, identity["path"])).is_absolute()
        or os.path.abspath(cast(str, identity["path"])) != identity["path"]
        or type(identity.get("bytes")) is not int
        or cast(int, identity["bytes"]) < 0
        or not _is_sha256(identity.get("sha256"))
    ):
        raise ArtifactAuditError(f"formal test report {label} file identity is malformed")
    return identity


def _preformal_qa_closure(closure: object) -> Mapping[str, object]:
    if not isinstance(closure, Mapping) or set(closure) != {
        "schema",
        "sys_path",
        "distributions",
        "inventory_sha256",
    }:
        raise ArtifactAuditError("formal test report QA closure has wrong fields")
    paths = closure.get("sys_path")
    rows = closure.get("distributions")
    if (
        closure.get("schema") != "prospect.wm001.qa-closure.v1"
        or not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or "\0" in path for path in paths)
        or len(paths) != len(set(paths))
        or not isinstance(rows, list)
        or not rows
    ):
        raise ArtifactAuditError("formal test report QA closure is malformed")
    names: list[str] = []
    fields = {
        "name",
        "version",
        "editable",
        "declared_file_count",
        "total_bytes",
        "distribution_sha256",
    }
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or not isinstance(row.get("name"), str)
            or re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*",
                cast(str, row["name"]),
            )
            is None
            or not isinstance(row.get("version"), str)
            or not row.get("version")
            or row.get("editable") is not False
            or type(row.get("declared_file_count")) is not int
            or cast(int, row["declared_file_count"]) <= 0
            or type(row.get("total_bytes")) is not int
            or cast(int, row["total_bytes"]) < 0
            or not _is_sha256(row.get("distribution_sha256"))
        ):
            raise ArtifactAuditError("formal test report QA distribution identity is malformed")
        names.append(cast(str, row["name"]))
    unsigned = {key: value for key, value in closure.items() if key != "inventory_sha256"}
    if (
        names != sorted(names)
        or len(names) != len(set(names))
        or "prospect" not in names
        or closure.get("inventory_sha256") != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise ArtifactAuditError("formal test report QA closure is duplicated, unordered, or misbound")
    return closure


def _preformal_prospective_review(
    review: object,
    *,
    source: Mapping[str, object],
) -> Mapping[str, object]:
    fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "implementation_files",
        "implementation_manifest_sha256",
        "reviewer",
        "disposition",
        "unresolved_blockers",
        "findings",
    }
    implementation = source.get("implementation_files")
    if not isinstance(implementation, list):
        raise ArtifactAuditError("formal test report source snapshot has no implementation manifest")
    expected_rows = [
        dict(row) for row in implementation if isinstance(row, Mapping) and row.get("path") != _PREFORMAL_REVIEW_PATH
    ]
    reviewer = review.get("reviewer") if isinstance(review, Mapping) else None
    findings = review.get("findings") if isinstance(review, Mapping) else None
    if (
        not isinstance(review, Mapping)
        or set(review) != fields
        or review.get("schema") != "prospect.wm001.prospective-harness-review.v1"
        or review.get("experiment_id") != "WM-001"
        or review.get("protocol_version") != "1.16.0"
        or review.get("implementation_files") != expected_rows
        or review.get("implementation_manifest_sha256")
        != hashlib.sha256(_canonical_json_bytes(expected_rows)).hexdigest()
        or not isinstance(reviewer, Mapping)
        or set(reviewer) != {"kind", "identifier"}
        or reviewer.get("kind") != "independent-adversarial-referee"
        or not isinstance(reviewer.get("identifier"), str)
        or not reviewer.get("identifier")
        or cast(str, reviewer["identifier"]).strip() != reviewer["identifier"]
        or review.get("disposition") != "accepted"
        or review.get("unresolved_blockers") != []
        or not isinstance(findings, list)
    ):
        raise ArtifactAuditError("formal test report prospective review is not an accepted exact-source review")
    finding_ids: list[str] = []
    for finding in findings:
        if (
            not isinstance(finding, Mapping)
            or set(finding) != {"id", "severity", "summary", "resolution"}
            or not isinstance(finding.get("id"), str)
            or not finding.get("id")
            or finding.get("severity") not in {"blocker", "major", "minor", "note"}
            or not isinstance(finding.get("summary"), str)
            or not finding.get("summary")
            or finding.get("resolution") not in {"resolved", "informational"}
        ):
            raise ArtifactAuditError("formal test report prospective review finding is malformed")
        finding_ids.append(cast(str, finding["id"]))
    if finding_ids != sorted(finding_ids) or len(finding_ids) != len(set(finding_ids)):
        raise ArtifactAuditError("formal test report prospective review findings are duplicated or unordered")
    return review


def _preformal_runtime_seal(
    seal: object,
    *,
    source: Mapping[str, object],
    dependencies: Mapping[str, object],
    runtime: Mapping[str, object],
    runtime_executable: Mapping[str, object],
) -> Mapping[str, object]:
    fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "assurance",
        "git_commit",
        "git_tree",
        "worktree_clean",
        "python",
        "required_flags",
        "process_environment",
        "bootstrap_source_sha256",
        "standard_library",
        "package_roots",
        "package_ownership",
    }
    python = seal.get("python") if isinstance(seal, Mapping) else None
    execution_sources = source.get("execution_source_sha256")
    if (
        not isinstance(seal, Mapping)
        or set(seal) != fields
        or seal.get("schema") != "prospect.wm001.runtime-seal.v1"
        or seal.get("experiment_id") != "WM-001"
        or seal.get("protocol_version") != "1.16.0"
        or not _strict_json_equal(seal.get("assurance"), _ASSURANCE)
        or seal.get("git_commit") != source.get("git_commit")
        or seal.get("git_tree") != source.get("git_tree")
        or seal.get("worktree_clean") is not True
        or not isinstance(python, Mapping)
        or set(python) != {"executable", "resolved_executable", "sha256", "version"}
        or python.get("executable") != dependencies.get("python_executable")
        or python.get("sha256") != dependencies.get("python_executable_sha256")
        or not isinstance(python.get("resolved_executable"), str)
        or not Path(cast(str, python["resolved_executable"])).is_absolute()
        or not isinstance(python.get("version"), list)
        or len(cast(list[object], python["version"])) != 3
        or any(
            type(part) is not int or cast(int, part) < 0
            for part in cast(list[object], python["version"])
        )
        or ".".join(
            str(part) for part in cast(list[object], python["version"])
        )
        != runtime_executable.get("version")
        or python.get("executable")
        != runtime_executable.get("invocation_path")
        or python.get("resolved_executable")
        != runtime_executable.get("resolved_path")
        or python.get("sha256") != runtime_executable.get("sha256")
        or not _strict_json_equal(
            seal.get("required_flags"),
            _PREFORMAL_RUNTIME_FLAGS,
        )
        or not _strict_json_equal(
            seal.get("required_flags"),
            runtime.get("python_flags"),
        )
        or not _strict_json_equal(
            seal.get("process_environment"),
            runtime.get("process_environment"),
        )
        or not isinstance(execution_sources, Mapping)
        or seal.get("bootstrap_source_sha256") != execution_sources.get("producer_bootstrap.py")
        or not _strict_json_equal(
            seal.get("standard_library"),
            dependencies.get("standard_library"),
        )
        or not _strict_json_equal(
            seal.get("package_roots"),
            dependencies.get("package_roots"),
        )
        or not _strict_json_equal(
            seal.get("package_ownership"),
            dependencies.get("package_ownership"),
        )
    ):
        raise ArtifactAuditError("formal test report runtime seal differs from the formal binding")
    return seal


def _preformal_log_reference(
    reference: object,
    *,
    ordinal: int,
    command_name: str,
    stream: str,
) -> dict[str, object]:
    if not isinstance(reference, Mapping) or set(reference) != {
        "file",
        "bytes",
        "sha256",
    }:
        raise ArtifactAuditError(f"formal test report {command_name} {stream} reference is malformed")
    size = reference.get("bytes")
    digest = reference.get("sha256")
    expected_name = f"{_PREFORMAL_LOG_PREFIX}{ordinal:02d}-{command_name}.{stream}.{digest}.log"
    if (
        type(size) is not int
        or cast(int, size) < 0
        or not _is_sha256(digest)
        or reference.get("file") != expected_name
        or Path(expected_name).name != expected_name
    ):
        raise ArtifactAuditError(f"formal test report {command_name} {stream} identity is invalid")
    if stream == "stderr" and (
        size != 0 or digest != _SHA256_EMPTY
    ):
        raise ArtifactAuditError(
            f"formal test report command {ordinal} stderr is not exactly empty"
        )
    return {
        "path": expected_name,
        "bytes": size,
        "sha256": digest,
    }


def _preformal_accepted_closure_evidence(
    *,
    root: Path,
    commands: Sequence[object],
    inputs: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    """Independently reopen and bind the sealed command-9 authorization."""

    closure_identity = inputs.get("development_closure")
    if not isinstance(closure_identity, Mapping):
        raise ArtifactAuditError(
            "formal test report has no development-closure identity"
        )
    closure_path_value = closure_identity.get("path")
    if not isinstance(closure_path_value, str):
        raise ArtifactAuditError(
            "formal test report development-closure path is malformed"
        )
    closure_path = Path(closure_path_value)
    (
        closure_payload,
        closure_bytes,
        closure_sha256,
        closure_file_identity,
    ) = _read_stable_regular_file(
        closure_path,
        64 << 20,
        label="preformal development closure",
        capture_payload=True,
    )
    if (
        closure_payload is None
        or closure_path.is_symlink()
        or closure_file_identity[3] != 1
        or closure_identity.get("bytes") != closure_bytes
        or closure_identity.get("sha256") != closure_sha256
    ):
        raise ArtifactAuditError(
            "preformal development-closure identity changed"
        )
    closure = _canonical_json_object_payload(
        closure_payload,
        label="preformal development closure",
    )
    archive = closure.get("qualification_archive")
    members = archive.get("members") if isinstance(archive, Mapping) else None
    if (
        closure.get("schema")
        != "prospect.wm001.development-closure.v2"
        or closure.get("experiment_id") != "WM-001"
        or closure.get("protocol_version") != "1.16.0"
        or closure.get("producer_manifest_member")
        != "producer/producer-manifest.json"
        or closure.get("raw_result_member") != "producer/result.json"
        or not isinstance(members, list)
        or not members
    ):
        raise ArtifactAuditError(
            "preformal development closure has no exact archived roles"
        )
    member_digests: dict[str, str] = {}
    previous = ""
    for index, member in enumerate(members):
        if (
            not isinstance(member, Mapping)
            or set(member) != {"path", "bytes", "sha256"}
            or not isinstance(member.get("path"), str)
            or cast(str, member["path"]) <= previous
            or type(member.get("bytes")) is not int
            or cast(int, member["bytes"]) < 0
            or not _is_sha256(member.get("sha256"))
        ):
            raise ArtifactAuditError(
                f"preformal development closure member {index} is invalid"
            )
        previous = cast(str, member["path"])
        member_digests[previous] = cast(str, member["sha256"])
    producer_manifest_sha256 = member_digests.get(
        "producer/producer-manifest.json"
    )
    raw_result_sha256 = member_digests.get("producer/result.json")
    if (
        not _is_sha256(producer_manifest_sha256)
        or not _is_sha256(raw_result_sha256)
    ):
        raise ArtifactAuditError(
            "preformal development closure omits a result-bearing member"
        )

    matches = [
        row
        for row in commands
        if isinstance(row, Mapping)
        and row.get("name") == "runtime-accepted-closure-evidence"
    ]
    if len(matches) != 1:
        raise ArtifactAuditError(
            "formal test report lacks one accepted-closure command"
        )
    row = matches[0]
    stdout = row.get("stdout")
    stderr = row.get("stderr")
    if not isinstance(stdout, Mapping) or not isinstance(stderr, Mapping):
        raise ArtifactAuditError(
            "formal test report accepted-closure streams are malformed"
        )
    stdout_path = _resolve_artifact_file(
        root,
        stdout.get("file"),
        label="preformal accepted-closure stdout",
    )
    stderr_path = _resolve_artifact_file(
        root,
        stderr.get("file"),
        label="preformal accepted-closure stderr",
    )
    stdout_payload = _read_bounded(
        stdout_path,
        4 << 20,
        label="preformal accepted-closure stdout",
    )
    stderr_payload = _read_bounded(
        stderr_path,
        4 << 20,
        label="preformal accepted-closure stderr",
    )
    value = _canonical_json_object_payload(
        stdout_payload,
        label="preformal accepted-closure stdout",
    )
    expected_fields = {
        "schema",
        "mode",
        "passed",
        "development_closure_sha256",
        "producer_manifest_sha256",
        "raw_result_sha256",
        "closure_attempt_manifest_sha256",
        "closure_outer_completion_sha256",
    }
    if (
        stdout.get("bytes") != len(stdout_payload)
        or stdout.get("sha256")
        != hashlib.sha256(stdout_payload).hexdigest()
        or stderr_payload != b""
        or stderr.get("bytes") != 0
        or stderr.get("sha256") != _SHA256_EMPTY
        or set(value) != expected_fields
        or value.get("schema")
        != "prospect.wm001.preformal-runtime-check.v1"
        or value.get("mode") != "accepted-closure-evidence"
        or value.get("passed") is not True
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "development_closure_sha256",
                "producer_manifest_sha256",
                "raw_result_sha256",
                "closure_attempt_manifest_sha256",
                "closure_outer_completion_sha256",
            )
        )
        or value.get("development_closure_sha256")
        != inputs["development_closure"].get("sha256")
        or value.get("producer_manifest_sha256")
        != producer_manifest_sha256
        or value.get("raw_result_sha256") != raw_result_sha256
        or value.get("closure_attempt_manifest_sha256")
        != inputs["closure_attempt_terminal"].get("sha256")
        or value.get("closure_outer_completion_sha256")
        != inputs["closure_outer_completion"].get("sha256")
    ):
        raise ArtifactAuditError(
            "preformal accepted-closure stdout is not one complete sealed pass"
        )
    return value


def _validate_preformal_test_report_v2(
    payload: bytes,
    *,
    root: Path,
    source: Mapping[str, object],
    dependencies: Mapping[str, object],
    runtime: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    """Independently validate and reopen the complete preformal v2 custody set."""

    report = _canonical_json_object_payload(
        payload,
        label="formal test report",
    )
    expected_report_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "repository_cwd",
        "device",
        "qa_environment",
        "runtime_environment",
        "git_before",
        "git_after",
        "qa_executable_before",
        "qa_executable_after",
        "runtime_executable_before",
        "runtime_executable_after",
        "qa_closure_before",
        "qa_closure_after",
        "runtime_seal",
        "prospective_review",
        "input_files_before",
        "input_files_after",
        "generator_source_before",
        "generator_source_after",
        "identities_stable",
        "commands",
        "all_pass",
    }
    repository_cwd = report.get("repository_cwd")
    if (
        set(report) != expected_report_fields
        or report.get("schema") != "prospect.wm001.preformal-test-report.v2"
        or report.get("experiment_id") != "WM-001"
        or report.get("protocol_version") != "1.16.0"
        or not isinstance(repository_cwd, str)
        or not repository_cwd
        or "\0" in repository_cwd
        or not Path(repository_cwd).is_absolute()
        or os.path.abspath(repository_cwd) != repository_cwd
        or Path(repository_cwd) != _LIVE_REPOSITORY_ROOT
        or Path(repository_cwd).resolve(strict=True)
        != _LIVE_REPOSITORY_ROOT
        or report.get("device") not in {"cpu", "cuda"}
        or report.get("device") != runtime.get("device")
        or report.get("identities_stable") is not True
        or report.get("all_pass") is not True
    ):
        raise ArtifactAuditError("formal test report identity, repository, or status is invalid")
    _, qa_environment_digest = _preformal_environment(
        report.get("qa_environment"),
        role="qa",
        runtime=runtime,
    )
    _, runtime_environment_digest = _preformal_environment(
        report.get("runtime_environment"),
        role="runtime",
        runtime=runtime,
    )
    git_before = _preformal_git_identity(
        report.get("git_before"),
        source=source,
        label="before",
    )
    git_after = _preformal_git_identity(
        report.get("git_after"),
        source=source,
        label="after",
    )
    qa_executable_before = _preformal_qa_executable_identity(
        report.get("qa_executable_before"),
    )
    qa_executable_after = _preformal_qa_executable_identity(
        report.get("qa_executable_after"),
    )
    runtime_executable_before = _preformal_runtime_executable_identity(
        report.get("runtime_executable_before"),
        dependencies=dependencies,
        runtime_seal=report.get("runtime_seal"),
    )
    runtime_executable_after = _preformal_runtime_executable_identity(
        report.get("runtime_executable_after"),
        dependencies=dependencies,
        runtime_seal=report.get("runtime_seal"),
    )
    qa_closure_before = _preformal_qa_closure(report.get("qa_closure_before"))
    qa_closure_after = _preformal_qa_closure(report.get("qa_closure_after"))
    runtime_seal = _preformal_runtime_seal(
        report.get("runtime_seal"),
        source=source,
        dependencies=dependencies,
        runtime=runtime,
        runtime_executable=runtime_executable_before,
    )
    prospective_review = _preformal_prospective_review(
        report.get("prospective_review"),
        source=source,
    )
    generator_before = report.get("generator_source_before")
    generator_after = report.get("generator_source_after")
    implementation = source.get("implementation_files")
    preformal_rows = (
        [
            row
            for row in implementation
            if isinstance(row, Mapping) and row.get("path") == "bench/world_model_lifecycle/preformal.py"
        ]
        if isinstance(implementation, list)
        else []
    )
    review_rows = (
        [row for row in implementation if isinstance(row, Mapping) and row.get("path") == _PREFORMAL_REVIEW_PATH]
        if isinstance(implementation, list)
        else []
    )
    inputs_before = report.get("input_files_before")
    inputs_after = report.get("input_files_after")
    if (
        not isinstance(inputs_before, Mapping)
        or set(inputs_before) != _PREFORMAL_INPUT_FIELDS
        or not isinstance(inputs_after, Mapping)
        or set(inputs_after) != _PREFORMAL_INPUT_FIELDS
    ):
        raise ArtifactAuditError("formal test report input-file identities are incomplete or changed")
    inputs: dict[str, Mapping[str, object]] = {}
    for name, identity in inputs_before.items():
        inputs[name] = _preformal_file_identity(identity, label=name)
        after = _preformal_file_identity(
            inputs_after[name],
            label=f"{name} after",
        )
        if not _strict_json_equal(inputs[name], after):
            raise ArtifactAuditError(
                "formal test report input-file identities are incomplete "
                "or changed"
            )
    development_root = (
        Path(repository_cwd)
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
    )
    expected_development_closure_path = (
        development_root / "development-closure-v1.16.0.json"
    )
    expected_runtime_seal_path = (
        development_root / "runtime-seal-v1.16.0.json"
    )
    runtime_seal_path = Path(
        cast(str, inputs["runtime_seal"]["path"])
    )
    if (
        inputs["development_closure"].get("path")
        != str(expected_development_closure_path)
        or runtime_seal_path != expected_runtime_seal_path
    ):
        raise ArtifactAuditError(
            "formal test report runtime seal or development closure is not "
            "canonical"
        )
    runtime_seal_completion_path = (
        Path(repository_cwd)
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "outer-completions"
        / "v1.16"
        / (
            hashlib.sha256(
                str(expected_runtime_seal_path).encode("utf-8")
            ).hexdigest()
            + ".json"
        )
    )
    (
        live_runtime_seal_payload,
        live_runtime_seal_bytes,
        live_runtime_seal_sha256,
        live_runtime_seal_identity,
    ) = _read_stable_regular_file(
        runtime_seal_path,
        64 << 20,
        label="preformal runtime seal",
        capture_payload=True,
    )
    (
        runtime_seal_completion_payload,
        _,
        _,
        runtime_seal_completion_identity,
    ) = _read_stable_regular_file(
        runtime_seal_completion_path,
        64 << 20,
        label="preformal runtime-seal outer completion",
        capture_payload=True,
    )
    if (
        inputs["runtime_seal"].get("bytes")
        != live_runtime_seal_bytes
        or inputs["runtime_seal"].get("sha256")
        != live_runtime_seal_sha256
        or live_runtime_seal_payload
        != _canonical_json_bytes(runtime_seal) + b"\n"
        or live_runtime_seal_payload
        != runtime_seal_completion_payload
        or live_runtime_seal_identity
        != runtime_seal_completion_identity
        or live_runtime_seal_identity[3] != 2
        or not os.path.samefile(
            runtime_seal_path,
            runtime_seal_completion_path,
        )
    ):
        raise ArtifactAuditError(
            "formal test report runtime seal or development closure is not "
            "one canonical outer-finalized input"
        )
    closure_terminal_path = (
        Path(repository_cwd)
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.16"
        / "closures"
        / "development-closure-v1.16.0"
        / "operator-attempt.json"
    )
    closure_completion_path = (
        Path(repository_cwd)
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "outer-completions"
        / "v1.16"
        / (
            hashlib.sha256(
                str(closure_terminal_path).encode("utf-8")
            ).hexdigest()
            + ".json"
        )
    )
    terminal_payload, terminal_bytes, terminal_sha256, terminal_identity = (
        _read_stable_regular_file(
            closure_terminal_path,
            16 << 20,
            label="preformal closure-attempt terminal",
            capture_payload=True,
        )
    )
    completion_payload, completion_bytes, completion_sha256, completion_identity = (
        _read_stable_regular_file(
            closure_completion_path,
            16 << 20,
            label="preformal closure outer completion",
            capture_payload=True,
        )
    )
    if (
        inputs["closure_attempt_terminal"].get("path")
        != str(closure_terminal_path)
        or inputs["closure_outer_completion"].get("path")
        != str(closure_completion_path)
        or inputs["closure_attempt_terminal"].get("bytes")
        != terminal_bytes
        or inputs["closure_attempt_terminal"].get("sha256")
        != terminal_sha256
        or inputs["closure_outer_completion"].get("bytes")
        != completion_bytes
        or inputs["closure_outer_completion"].get("sha256")
        != completion_sha256
        or terminal_payload != completion_payload
        or terminal_identity != completion_identity
        or terminal_identity[3] != 2
        or not os.path.samefile(
            closure_terminal_path,
            closure_completion_path,
        )
    ):
        raise ArtifactAuditError(
            "formal test report closure attempt is not one same-inode "
            "outer-finalized input"
        )
    expected_repository_inputs = {
        "launch_bootstrap": "bench/world_model_lifecycle/launch_bootstrap.py",
        "producer_bootstrap": "bench/world_model_lifecycle/producer_bootstrap.py",
        "prospective_review": _PREFORMAL_REVIEW_PATH,
    }
    implementation_by_path = (
        {
            cast(str, row["path"]): row
            for row in implementation
            if isinstance(row, Mapping) and isinstance(row.get("path"), str)
        }
        if isinstance(implementation, list)
        else {}
    )
    for name, relative in expected_repository_inputs.items():
        recorded = inputs[name]
        bound = implementation_by_path.get(relative)
        if (
            recorded.get("path") != str(Path(repository_cwd) / relative)
            or not isinstance(bound, Mapping)
            or not _strict_json_equal(
                recorded.get("bytes"),
                bound.get("bytes"),
            )
            or recorded.get("sha256") != bound.get("sha256")
        ):
            raise ArtifactAuditError(f"formal test report {name} differs from its source binding")
    review_payload = _canonical_json_bytes(prospective_review) + b"\n"
    seal_payload = _canonical_json_bytes(runtime_seal) + b"\n"
    if (
        inputs["prospective_review"].get("bytes") != len(review_payload)
        or inputs["prospective_review"].get("sha256") != hashlib.sha256(review_payload).hexdigest()
        or inputs["runtime_seal"].get("bytes") != len(seal_payload)
        or inputs["runtime_seal"].get("sha256") != hashlib.sha256(seal_payload).hexdigest()
        or len(review_rows) != 1
    ):
        raise ArtifactAuditError("formal test report embedded review or runtime seal bytes are misbound")
    if (
        not _strict_json_equal(git_before, git_after)
        or not _strict_json_equal(
            qa_executable_before,
            qa_executable_after,
        )
        or not _strict_json_equal(
            runtime_executable_before,
            runtime_executable_after,
        )
        or not _strict_json_equal(
            qa_closure_before,
            qa_closure_after,
        )
        or not isinstance(generator_before, Mapping)
        or set(generator_before) != {"path", "bytes", "sha256"}
        or not _strict_json_equal(generator_before, generator_after)
        or len(preformal_rows) != 1
        or not _strict_json_equal(
            dict(generator_before),
            dict(preformal_rows[0]),
        )
    ):
        raise ArtifactAuditError("formal test report source, Git, or executable identity changed")
    qa_executable = qa_executable_before.get("invocation_path")
    runtime_executable = runtime_executable_before.get("invocation_path")
    assert isinstance(qa_executable, str)
    assert isinstance(runtime_executable, str)
    runtime_seal_input_path = inputs["runtime_seal"].get("path")
    development_closure_path = inputs["development_closure"].get("path")
    closure_attempt_terminal_path = inputs[
        "closure_attempt_terminal"
    ].get("path")
    prospective_review_path = inputs["prospective_review"].get("path")
    assert isinstance(runtime_seal_input_path, str)
    assert isinstance(development_closure_path, str)
    assert isinstance(closure_attempt_terminal_path, str)
    assert isinstance(prospective_review_path, str)
    closure_attempt_path = str(Path(closure_attempt_terminal_path).parent)
    expected_commands = _preformal_expected_commands(
        qa_executable=qa_executable,
        runtime_executable=runtime_executable,
        source=source,
        repository_cwd=repository_cwd,
        runtime_seal_path=runtime_seal_input_path,
        development_closure_path=development_closure_path,
        closure_attempt_path=closure_attempt_path,
        prospective_review_path=prospective_review_path,
        device=cast(str, report["device"]),
    )
    commands = report.get("commands")
    if (
        not isinstance(commands, list)
        or len(commands) != len(expected_commands)
        or tuple(name for name, _, _ in expected_commands) != _PREFORMAL_COMMAND_NAMES
    ):
        raise ArtifactAuditError("formal test report command set is incomplete")
    bound_log_rows: list[dict[str, object]] = []
    for ordinal, (row, expected) in enumerate(
        zip(commands, expected_commands, strict=True),
        start=1,
    ):
        expected_name, expected_role, expected_argv = expected
        environment_digest = qa_environment_digest if expected_role == "qa" else runtime_environment_digest
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "ordinal",
                "name",
                "role",
                "argv",
                "cwd",
                "environment_sha256",
                "exit_code",
                "passed",
                "stdout",
                "stderr",
            }
            or type(row.get("ordinal")) is not int
            or row.get("ordinal") != ordinal
            or row.get("name") != expected_name
            or row.get("role") != expected_role
            or row.get("argv") != list(expected_argv)
            or row.get("cwd") != repository_cwd
            or row.get("environment_sha256") != environment_digest
            or type(row.get("exit_code")) is not int
            or row.get("exit_code") != 0
            or row.get("passed") is not True
        ):
            raise ArtifactAuditError(f"formal test report command {ordinal} differs from its fixed contract")
        for stream in ("stdout", "stderr"):
            bound_log_rows.append(
                _preformal_log_reference(
                    row.get(stream),
                    ordinal=ordinal,
                    command_name=expected_name,
                    stream=stream,
                )
            )
    source_logs = source.get("test_log_files")
    if (
        not isinstance(source_logs, list)
        or source_logs != bound_log_rows
        or len({cast(str, row["path"]) for row in bound_log_rows}) != len(bound_log_rows)
    ):
        raise ArtifactAuditError("formal test report log references differ from their binding")
    if (
        source.get("test_report_file") != _PREFORMAL_REPORT_NAME
        or source.get("test_report_bytes") != len(payload)
        or source.get("test_report_sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise ArtifactAuditError("formal test report bytes differ from their binding")
    root_resolved = root.resolve(strict=True)
    expected_evidence = {_PREFORMAL_REPORT_NAME}
    report_path = root / _PREFORMAL_REPORT_NAME
    report_reopened, report_bytes, report_digest, _ = _read_stable_regular_file(
        report_path,
        64 << 20,
        label="bound preformal report",
        capture_payload=True,
    )
    if (
        report_path.is_symlink()
        or report_path.lstat().st_nlink != 1
        or report_path.resolve(strict=True).parent != root_resolved
        or report_reopened != payload
        or report_bytes != len(payload)
        or report_digest != hashlib.sha256(payload).hexdigest()
    ):
        raise ArtifactAuditError("formal test report changed when independently reopened")
    total_log_bytes = 0
    for row in bound_log_rows:
        filename = cast(str, row["path"])
        if Path(filename).is_absolute() or len(Path(filename).parts) != 1 or Path(filename).name != filename:
            raise ArtifactAuditError("formal test report log path escapes the artifact root")
        path = root / filename
        _, observed_bytes, observed_digest, _ = _read_stable_regular_file(
            path,
            64 << 20,
            label=f"bound preformal log {filename}",
            capture_payload=False,
        )
        total_log_bytes += observed_bytes
        if (
            total_log_bytes > 512 << 20
            or path.is_symlink()
            or path.lstat().st_nlink != 1
            or path.resolve(strict=True).parent != root_resolved
            or observed_bytes != row["bytes"]
            or observed_digest != row["sha256"]
        ):
            raise ArtifactAuditError(f"formal test report log bytes changed: {filename}")
        expected_evidence.add(filename)
    actual_evidence = {
        candidate.name
        for candidate in root.iterdir()
        if candidate.name.startswith("preformal-") or candidate.name.startswith(".preformal-")
    }
    if actual_evidence != expected_evidence:
        raise ArtifactAuditError("formal preformal evidence file set has missing or extra members")
    accepted_closure_evidence = _preformal_accepted_closure_evidence(
        root=root,
        commands=commands,
        inputs=inputs,
    )
    runtime_commands = [
        row
        for row in commands
        if isinstance(row, Mapping)
        and row.get("name")
        == "runtime-bootstrap-inventory-conformance"
    ]
    if len(runtime_commands) != 1:
        raise ArtifactAuditError(
            "formal test report lacks one runtime bootstrap conformance"
        )
    runtime_command = runtime_commands[0]
    runtime_stdout = runtime_command.get("stdout")
    runtime_stderr = runtime_command.get("stderr")
    runtime_filename = (
        runtime_stdout.get("file")
        if isinstance(runtime_stdout, Mapping)
        else None
    )
    runtime_path = _resolve_artifact_file(
        root,
        runtime_filename,
        label="preformal runtime bootstrap conformance stdout",
    )
    runtime_payload = _read_bounded(
        runtime_path,
        4 << 20,
        label="preformal runtime bootstrap conformance stdout",
    )
    runtime_conformance = _canonical_json_object_payload(
        runtime_payload,
        label="preformal runtime bootstrap conformance stdout",
    )
    runtime_stderr_path = _resolve_artifact_file(
        root,
        (
            runtime_stderr.get("file")
            if isinstance(runtime_stderr, Mapping)
            else None
        ),
        label="preformal runtime bootstrap conformance stderr",
    )
    runtime_stderr_payload = _read_bounded(
        runtime_stderr_path,
        4 << 20,
        label="preformal runtime bootstrap conformance stderr",
    )
    if (
        runtime_command.get("exit_code") != 0
        or runtime_command.get("passed") is not True
        or not isinstance(runtime_stdout, Mapping)
        or runtime_stdout.get("bytes") != len(runtime_payload)
        or runtime_stdout.get("sha256")
        != hashlib.sha256(runtime_payload).hexdigest()
        or not isinstance(runtime_stderr, Mapping)
        or set(runtime_stderr) != {"file", "bytes", "sha256"}
        or runtime_stderr.get("bytes") != 0
        or runtime_stderr.get("sha256") != _SHA256_EMPTY
        or runtime_stderr_payload != b""
    ):
        raise ArtifactAuditError(
            "preformal runtime bootstrap conformance is incomplete"
        )
    runtime_conformance = _preformal_runtime_conformance(
        runtime_conformance,
        dependencies=dependencies,
        device=report.get("device"),
    )
    return (
        report,
        runtime_conformance,
        accepted_closure_evidence,
    )


_DEVELOPMENT_SOURCE_FIELDS = {
    "git_commit",
    "git_tree",
    "worktree_clean",
    "dependency_lock_sha256",
    "producer_bootstrap_sha256",
    "launch_bootstrap_sha256",
    "runner_source_sha256",
    "auditor_source_sha256",
}
_DEVELOPMENT_EXECUTION_FIELDS = {
    "git_commit",
    "git_tree",
    "worktree_clean",
    "dependency_lock_sha256",
    "python_executable",
    "python_executable_sha256",
    "python_version",
    "platform",
    "machine",
    "device",
    "python_flags",
    "process_environment",
    "accelerator",
    "thread_count",
    "interop_thread_count",
    "cuda_runtime",
    "cuda_driver",
    "cublas_workspace_config",
    "deterministic_algorithms",
    "runtime_seal_sha256",
    "runtime_seal_descriptor_custody",
    "producer_bootstrap_sha256",
    "bootstrap_descriptor_custody",
    "package_roots",
    "standard_library",
}
_DEVELOPMENT_CUSTODY_FIELDS = {
    "runtime_seal_member",
    "runtime_seal_sha256",
    "producer_bootstrap_member",
    "producer_bootstrap_sha256",
    "launch_bootstrap_member",
    "launch_bootstrap_sha256",
    "package_ownership",
}
_DEVELOPMENT_AUDIT_EXECUTION_FIELDS = {
    "receipt_sha256",
    "runtime_manifest_sha256",
    "invocation_manifest_sha256",
    "stderr_sha256",
    "bootstrap_sha256",
    "runner_source_sha256",
    "auditor_source_sha256",
    "support_files",
    "source_mode",
}


def _validate_archived_development_producer(
    retained: Mapping[str, bytes],
    members: Sequence[object],
) -> None:
    """Bind the archived producer namespace to its own terminal manifest."""

    manifest_member = "producer/producer-manifest.json"
    manifest_payload = retained.get(manifest_member)
    if not isinstance(manifest_payload, bytes):
        raise ArtifactAuditError(
            "development archive omits its producer manifest payload"
        )
    manifest = _canonical_json_object_payload(
        manifest_payload,
        label="archived development producer manifest",
    )
    rows = manifest.get("files")
    if (
        set(manifest) != _PRODUCER_MANIFEST_FIELDS
        or manifest.get("schema") != "prospect.wm001.producer-manifest.v1"
        or manifest.get("experiment_id") != "WM-001"
        or manifest.get("lane") != "development"
        or manifest.get("status") != "completed"
        or manifest.get("error") is not None
        or manifest.get("manifest_excludes") != [_PRODUCER_MANIFEST_NAME]
        or not isinstance(rows, list)
        or type(manifest.get("file_count")) is not int
        or manifest.get("file_count") != len(rows)
    ):
        raise ArtifactAuditError(
            "archived development producer manifest is incomplete"
        )
    started_at = _parse_utc_timestamp(
        manifest.get("started_at_utc"),
        label="archived development producer started_at_utc",
    )
    completed_at = _parse_utc_timestamp(
        manifest.get("completed_at_utc"),
        label="archived development producer completed_at_utc",
    )
    if completed_at < started_at:
        raise ArtifactAuditError(
            "archived development producer completed before it started"
        )
    manifest_paths: list[str] = []
    expected_producer: dict[str, dict[str, object]] = {}
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ArtifactAuditError(
                f"archived development producer manifest row {index} is malformed"
            )
        relative = raw_row.get("path")
        byte_count = raw_row.get("bytes")
        digest = raw_row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or "." in Path(relative).parts
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or type(byte_count) is not int
            or cast(int, byte_count) < 0
            or cast(int, byte_count) > (4 << 30)
            or not _is_sha256(digest)
        ):
            raise ArtifactAuditError(
                f"archived development producer manifest row {index} is malformed"
            )
        manifest_paths.append(relative)
        expected_producer[f"producer/{relative}"] = {
            "bytes": byte_count,
            "sha256": digest,
        }
    if manifest_paths != sorted(set(manifest_paths)):
        raise ArtifactAuditError(
            "archived development producer manifest rows are not exact and ordered"
        )
    expected_producer[manifest_member] = {
        "bytes": len(manifest_payload),
        "sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    observed_producer = {
        str(row["path"]): {
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in members
        if isinstance(row, Mapping)
        and isinstance(row.get("path"), str)
        and cast(str, row["path"]).startswith("producer/")
    }
    if not _strict_json_equal(observed_producer, expected_producer):
        raise ArtifactAuditError(
            "archived producer member set differs from its manifest"
        )


def _validate_archived_development_semantics(
    retained: Mapping[str, bytes],
    *,
    closure: Mapping[str, object],
    member_digests: Mapping[str, str],
    protocol_sha256: object,
    source: Mapping[str, object],
) -> None:
    """Independently bind performance-free qualification and audit evidence."""

    def role_payload(field: str) -> bytes:
        member = closure.get(field)
        payload = retained.get(member) if isinstance(member, str) else None
        if not isinstance(payload, bytes):
            raise ArtifactAuditError(
                f"development archive omits retained role {field}"
            )
        return payload

    producer_execution = closure.get("producer_execution")
    qualification = _canonical_json_object_payload(
        role_payload("result_qualification_member"),
        label="archived development result qualification",
    )
    replicates = qualification.get("replicates")
    expected_counts = {
        "episodes": 496,
        "transitions": 99_200,
        "predictive_metrics": 12,
        "policy_runs": 20,
        "updates": 6,
        "optimizer_batch_manifests": 5,
    }
    if (
        set(qualification)
        != {
            "schema",
            "experiment_id",
            "protocol_version",
            "protocol_sha256",
            "raw_result_sha256",
            "lane",
            "claim_eligible",
            "replicates",
            "matrix_contract_sha256",
            "producer_execution",
        }
        or qualification.get("schema")
        != "prospect.wm001.development-result-qualification.v1"
        or qualification.get("experiment_id") != "WM-001"
        or qualification.get("protocol_version") != "1.16.0"
        or qualification.get("protocol_sha256") != protocol_sha256
        or qualification.get("raw_result_sha256")
        != member_digests.get("producer/result.json")
        or qualification.get("lane") != "development"
        or qualification.get("claim_eligible") is not False
        or qualification.get("matrix_contract_sha256")
        != _DEVELOPMENT_MATRIX_CONTRACT_SHA256
        or not _strict_json_equal(
            qualification.get("producer_execution"),
            producer_execution,
        )
        or not isinstance(replicates, list)
        or len(replicates) != 2
        or tuple(
            row.get("master_seed")
            if isinstance(row, Mapping)
            else None
            for row in replicates
        )
        != _DEVELOPMENT_SEEDS
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "replicate_id",
                "master_seed",
                *expected_counts,
            }
            or not isinstance(row.get("replicate_id"), str)
            or not row.get("replicate_id")
            or type(row.get("master_seed")) is not int
            or any(
                type(row.get(field)) is not int
                or row.get(field) != expected
                for field, expected in expected_counts.items()
            )
            for row in replicates
        )
    ):
        raise ArtifactAuditError(
            "archived development result qualification is not exact"
        )

    audit_payload = role_payload("independent_audit_member")
    audit = _canonical_json_object_payload(
        audit_payload,
        label="archived development independent audit",
    )
    counts = audit.get("check_counts")
    audit_implementation = audit.get("audit_implementation")
    expected_audit_fields = {
        "schema",
        "artifact_root",
        "result_file",
        "result_sha256",
        "lane",
        "integrity_passed",
        "engineering_complete",
        "complete_for_claim",
        "passed",
        "check_counts",
        "audit_implementation",
        "audit_execution_conformance_verified",
        "resource_limits_bytes",
        "custody",
        "findings",
        "coverage_gaps",
        "independence_limitations",
    }
    expected_audit_implementation = {
        "auditor_source_sha256": cast(
            Mapping[str, object],
            closure["source"],
        ).get("auditor_source_sha256"),
        "bound_auditor_source_sha256": None,
        "formal_test_report_sha256": None,
        "coverage_conformance_report_sha256": None,
        "auditor_source_matches_binding": False,
        "coverage_conformance_verified": False,
        "audit_execution_conformance_verified": False,
    }
    expected_resource_limits = {
        "result": _MAX_RESULT_BYTES,
        "prediction_sidecar": _MAX_PREDICTION_BYTES,
        "owned_model_state": _MAX_OWNED_MODEL_BYTES,
        "optimizer_manifest": _MAX_MANIFEST_BYTES,
        "target_permutation": _MAX_PERMUTATION_BYTES,
        "restart_evaluation": _MAX_RESTART_EVALUATION_BYTES,
        "checkpoint_archive": _MAX_CHECKPOINT_BYTES,
        "source_file": _MAX_SOURCE_FILE_BYTES,
        "source_snapshot": _MAX_SOURCE_SNAPSHOT_BYTES,
    }
    expected_custody = {
        "producer_manifest_checked": True,
        "producer_manifest_status": "completed",
        "producer_manifest_sha256": member_digests.get(
            "producer/producer-manifest.json"
        ),
    }
    expected_limitations = [
        _PRODUCER_CUSTODY_INDEPENDENCE_LIMITATION,
        _PENDULUM_RECONSTRUCTION_INDEPENDENCE_LIMITATION,
        _OPTIMIZER_RNG_INDEPENDENCE_LIMITATION,
        _CEM_REPLAY_INDEPENDENCE_LIMITATION,
        _DEVELOPMENT_EVIDENCE_INDEPENDENCE_LIMITATION,
    ]
    if (
        set(audit) != expected_audit_fields
        or audit.get("schema")
        != "prospect.world-model-lifecycle.artifact-audit.v2"
        or audit.get("artifact_root") != closure.get("producer_root")
        or audit.get("result_file") != "result.json"
        or audit.get("result_sha256")
        != member_digests.get("producer/result.json")
        or audit.get("lane") != "development"
        or audit.get("integrity_passed") is not True
        or audit.get("engineering_complete") is not True
        or audit.get("complete_for_claim") is not False
        or audit.get("passed") is not True
        or not isinstance(counts, Mapping)
        or set(counts) != {"passed", "failed", "coverage_gaps"}
        or type(counts.get("passed")) is not int
        or cast(int, counts["passed"]) <= 0
        or type(counts.get("failed")) is not int
        or counts.get("failed") != 0
        or type(counts.get("coverage_gaps")) is not int
        or counts.get("coverage_gaps") != 0
        or audit.get("coverage_gaps") != []
        or audit.get("findings") != []
        or audit.get("audit_execution_conformance_verified") is not False
        or not isinstance(audit_implementation, Mapping)
        or not _strict_json_equal(
            audit_implementation,
            expected_audit_implementation,
        )
        or not _strict_json_equal(
            audit.get("resource_limits_bytes"),
            expected_resource_limits,
        )
        or not _strict_json_equal(
            audit.get("custody"),
            expected_custody,
        )
        or not _strict_json_equal(
            audit.get("independence_limitations"),
            expected_limitations,
        )
    ):
        raise ArtifactAuditError(
            "archived development independent audit is not passing"
        )

    receipt_payload = role_payload("audit_reproduction_member")
    receipt = _canonical_json_object_payload(
        receipt_payload,
        label="archived development audit reproduction",
    )
    audit_execution = closure.get("audit_execution")
    if not isinstance(audit_execution, Mapping):
        raise ArtifactAuditError(
            "development audit execution identity is absent"
        )
    sidecars = {
        "stderr": role_payload("audit_stderr_member"),
        "runtime_manifest": role_payload(
            "audit_runtime_manifest_member"
        ),
        "invocation_manifest": role_payload(
            "audit_invocation_manifest_member"
        ),
    }
    runtime_manifest = _canonical_json_object_payload(
        sidecars["runtime_manifest"],
        label="archived development audit runtime manifest",
    )
    invocation_manifest = _canonical_json_object_payload(
        sidecars["invocation_manifest"],
        label="archived development audit invocation manifest",
    )
    receipt_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "supplied_audit_sha256",
        "reproduced_audit_sha256",
        "byte_identical",
        "returncode",
        "source_mode",
        "stdout_bytes",
        "stderr_file",
        "stderr_bytes",
        "stderr_sha256",
        "runtime_manifest_file",
        "runtime_manifest_bytes",
        "runtime_manifest_sha256",
        "invocation_manifest_file",
        "invocation_manifest_bytes",
        "invocation_manifest_sha256",
        "bootstrap_sha256",
        "runner_source_sha256",
        "auditor_source_sha256",
        "support_files",
        "passed",
    }
    if (
        set(receipt) != receipt_fields
        or receipt.get("schema")
        != "prospect.wm001.audit-reproduction.v2"
        or receipt.get("experiment_id") != "WM-001"
        or receipt.get("protocol_version") != "1.16.0"
        or receipt.get("supplied_audit_sha256")
        != hashlib.sha256(audit_payload).hexdigest()
        or receipt.get("reproduced_audit_sha256")
        != hashlib.sha256(audit_payload).hexdigest()
        or receipt.get("byte_identical") is not True
        or type(receipt.get("returncode")) is not int
        or receipt.get("returncode") != 0
        or receipt.get("source_mode") != "descriptor"
        or type(receipt.get("stdout_bytes")) is not int
        or receipt.get("stdout_bytes") != len(audit_payload)
        or receipt.get("passed") is not True
        or receipt.get("bootstrap_sha256")
        != audit_execution.get("bootstrap_sha256")
        or receipt.get("runner_source_sha256")
        != audit_execution.get("runner_source_sha256")
        or receipt.get("auditor_source_sha256")
        != audit_execution.get("auditor_source_sha256")
        or not _strict_json_equal(
            receipt.get("support_files"),
            audit_execution.get("support_files"),
        )
        or hashlib.sha256(receipt_payload).hexdigest()
        != audit_execution.get("receipt_sha256")
    ):
        raise ArtifactAuditError(
            "archived development audit reproduction is not exact"
        )
    for prefix, payload in sidecars.items():
        member = closure.get(
            {
                "stderr": "audit_stderr_member",
                "runtime_manifest": "audit_runtime_manifest_member",
                "invocation_manifest": "audit_invocation_manifest_member",
            }[prefix]
        )
        filename = (
            member.removeprefix("evidence/")
            if isinstance(member, str)
            else None
        )
        if (
            type(receipt.get(f"{prefix}_bytes")) is not int
            or receipt.get(f"{prefix}_bytes") != len(payload)
            or receipt.get(f"{prefix}_sha256")
            != hashlib.sha256(payload).hexdigest()
            or receipt.get(f"{prefix}_file") != filename
            or receipt.get(f"{prefix}_sha256")
            != audit_execution.get(f"{prefix}_sha256")
        ):
            raise ArtifactAuditError(
                f"archived development audit {prefix} is not exact"
            )
    if not isinstance(producer_execution, Mapping):
        raise ArtifactAuditError(
            "development producer execution identity is absent"
        )
    implementation_rows = source.get("implementation_files")
    auditor_row = next(
        (
            row
            for row in implementation_rows
            if isinstance(row, Mapping)
            and row.get("path")
            == "bench/world_model_lifecycle/artifact_audit.py"
        ),
        None,
    ) if isinstance(implementation_rows, list) else None
    python_executable = producer_execution.get("python_executable")
    python_version = producer_execution.get("python_version")
    process_environment = producer_execution.get("process_environment")
    if (
        not isinstance(auditor_row, Mapping)
        or type(auditor_row.get("bytes")) is not int
        or not _is_sha256(auditor_row.get("sha256"))
        or not isinstance(python_executable, str)
        or not isinstance(python_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", python_version)
        is None
        or not isinstance(process_environment, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in process_environment.items()
        )
    ):
        raise ArtifactAuditError(
            "development audit runtime binding is incomplete"
        )
    expected_runtime_manifest = {
        "schema": "prospect.wm001.audit-runtime-manifest.v1",
        "assurance": _ASSURANCE,
        "bootstrap_sha256": audit_execution.get("bootstrap_sha256"),
        "python": {
            "executable": python_executable,
            "resolved_executable": str(
                Path(python_executable).resolve(strict=True)
            ),
            "sha256": producer_execution.get(
                "python_executable_sha256"
            ),
            "version": [
                int(part) for part in python_version.split(".")
            ],
        },
        "required_flags": dict(_PREBINDING_AUDITOR_FLAGS),
        "source": {
            "mode": "descriptor",
            "path": "artifact_audit.py",
            "bytes": auditor_row.get("bytes"),
            "sha256": auditor_row.get("sha256"),
        },
        "support_files": audit_execution.get("support_files"),
        "closure_import_roots": producer_execution.get("package_roots"),
        "standard_library": producer_execution.get("standard_library"),
        "environment": {
            key: value
            for key, value in process_environment.items()
            if key in _PREBINDING_SHARED_ENVIRONMENT_KEYS
        },
        "limits": {
            "timeout_seconds": 600,
            "stdout_bytes": 64 << 20,
            "stderr_bytes": 16 << 20,
        },
    }
    if not _strict_json_equal(
        runtime_manifest,
        expected_runtime_manifest,
    ):
        raise ArtifactAuditError(
            "archived development audit runtime differs from its closure"
        )
    expected_invocation = {
        "schema": "prospect.wm001.audit-invocation-manifest.v1",
        "runtime_manifest_sha256": hashlib.sha256(
            sidecars["runtime_manifest"]
        ).hexdigest(),
        "working_directory": str(Path.cwd()),
        "auditor_argv": [
            cast(str, closure["producer_root"]),
            "--producer-bootstrap",
            "@captured/producer_bootstrap.py",
        ],
    }
    if not _strict_json_equal(
        invocation_manifest,
        expected_invocation,
    ):
        raise ArtifactAuditError(
            "archived development audit invocation differs from its closure"
        )


def _validate_development_qualification(
    payload: bytes,
    *,
    block: Mapping[str, object],
    source: Mapping[str, object],
    dependencies: Mapping[str, object],
    runtime: Mapping[str, object],
    bound_audit_execution: Mapping[str, object],
    preformal_runtime_seal: Mapping[str, object],
) -> bytes:
    closure = _canonical_json_object_payload(
        payload,
        label="development qualification closure",
    )
    closure_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "source",
        "producer_root",
        "producer_manifest_member",
        "raw_result_member",
        "result_qualification_member",
        "independent_audit_member",
        "audit_reproduction_member",
        "audit_runtime_manifest_member",
        "audit_invocation_manifest_member",
        "audit_stderr_member",
        "producer_execution",
        "producer_custody",
        "audit_execution",
        "qualification_archive",
        "engineering_verified",
        "audit_reproduced",
        "performance_values_bound",
    }
    block_fields = {
        "closure_schema",
        "closure_file",
        "closure_bytes",
        "closure_sha256",
        "qualification_archive_file",
        "qualification_archive_path",
        "qualification_archive_bytes",
        "qualification_archive_sha256",
        "qualification_archive_members_sha256",
        "producer_manifest_sha256",
        "raw_result_sha256",
        "result_qualification_sha256",
        "independent_audit_sha256",
        "audit_reproduction_sha256",
        "audit_runtime_manifest_sha256",
        "audit_invocation_manifest_sha256",
        "audit_stderr_sha256",
        "source_identity_sha256",
        "producer_execution_identity_sha256",
        "producer_custody_identity_sha256",
        "audit_execution_identity_sha256",
        "git_commit",
        "git_tree",
        "engineering_verified",
        "audit_reproduced",
        "performance_values_bound",
    }
    closure_source = closure.get("source")
    producer_execution = closure.get("producer_execution")
    producer_custody = closure.get("producer_custody")
    audit_execution = closure.get("audit_execution")
    archive = closure.get("qualification_archive")
    producer_root = closure.get("producer_root")
    execution_sources = source.get("execution_source_sha256")
    implementation_rows = source.get("implementation_files")
    if (
        set(closure) != closure_fields
        or set(block) != block_fields
        or closure.get("schema") != "prospect.wm001.development-closure.v2"
        or closure.get("experiment_id") != "WM-001"
        or closure.get("protocol_version") != "1.16.0"
        or not isinstance(closure_source, Mapping)
        or not isinstance(producer_execution, Mapping)
        or not isinstance(producer_custody, Mapping)
        or not isinstance(audit_execution, Mapping)
        or not isinstance(archive, Mapping)
        or not isinstance(producer_root, str)
        or Path(producer_root)
        != (
            Path.cwd()
            / "bench"
            / "world_model_lifecycle"
            / "results"
            / "development"
            / "qualification-v1.16.0"
        )
        or not Path(producer_root).is_absolute()
        or Path(producer_root).resolve(strict=False)
        != Path(producer_root)
        or not isinstance(execution_sources, Mapping)
        or not isinstance(implementation_rows, list)
        or closure.get("engineering_verified") is not True
        or closure.get("audit_reproduced") is not True
        or closure.get("performance_values_bound") is not False
        or block.get("engineering_verified") is not True
        or block.get("audit_reproduced") is not True
        or block.get("performance_values_bound") is not False
    ):
        raise ArtifactAuditError("development qualification closure differs from its binding")

    expected_source = {
        "git_commit": source.get("git_commit"),
        "git_tree": source.get("git_tree"),
        "worktree_clean": True,
        "dependency_lock_sha256": dependencies.get("lockfile_sha256"),
        "producer_bootstrap_sha256": execution_sources.get("producer_bootstrap.py"),
        "launch_bootstrap_sha256": execution_sources.get("launch_bootstrap.py"),
        "runner_source_sha256": execution_sources.get("audit_runner.py"),
        "auditor_source_sha256": _AUDITOR_SOURCE_SHA256,
    }
    if (
        set(closure_source) != _DEVELOPMENT_SOURCE_FIELDS
        or not _strict_json_equal(closure_source, expected_source)
    ):
        raise ArtifactAuditError("development qualification source identity is not exact")

    execution_binding = {
        "git_commit": source.get("git_commit"),
        "git_tree": source.get("git_tree"),
        "worktree_clean": True,
        "dependency_lock_sha256": dependencies.get("lockfile_sha256"),
        "python_executable": dependencies.get("python_executable"),
        "python_executable_sha256": dependencies.get("python_executable_sha256"),
        "platform": runtime.get("platform"),
        "machine": runtime.get("machine"),
        "device": runtime.get("device"),
        "python_flags": runtime.get("python_flags"),
        "process_environment": runtime.get("process_environment"),
        "accelerator": runtime.get("accelerator"),
        "thread_count": runtime.get("thread_count"),
        "interop_thread_count": runtime.get("interop_thread_count"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "cuda_driver": runtime.get("cuda_driver"),
        "cublas_workspace_config": runtime.get("cublas_workspace_config"),
        "deterministic_algorithms": True,
        "runtime_seal_descriptor_custody": True,
        "producer_bootstrap_sha256": execution_sources.get("producer_bootstrap.py"),
        "bootstrap_descriptor_custody": True,
        "package_roots": dependencies.get("package_roots"),
        "standard_library": dependencies.get("standard_library"),
    }
    python_version = producer_execution.get("python_version")
    if (
        set(producer_execution) != _DEVELOPMENT_EXECUTION_FIELDS
        or any(
            not _strict_json_equal(producer_execution.get(field), expected)
            for field, expected in execution_binding.items()
        )
        or not isinstance(python_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", python_version) is None
        or not _is_sha256(producer_execution.get("runtime_seal_sha256"))
    ):
        raise ArtifactAuditError("development producer execution identity is not exact")

    expected_custody = {
        "runtime_seal_member": "evidence/producer-runtime-seal.json",
        "runtime_seal_sha256": producer_execution.get("runtime_seal_sha256"),
        "producer_bootstrap_member": "evidence/producer-bootstrap.py",
        "producer_bootstrap_sha256": execution_sources.get("producer_bootstrap.py"),
        "launch_bootstrap_member": "evidence/launch-bootstrap.py",
        "launch_bootstrap_sha256": execution_sources.get("launch_bootstrap.py"),
        "package_ownership": dependencies.get("package_ownership"),
    }
    if (
        set(producer_custody) != _DEVELOPMENT_CUSTODY_FIELDS
        or not _strict_json_equal(producer_custody, expected_custody)
    ):
        raise ArtifactAuditError("development producer custody identity is not exact")

    implementation_by_path = {row.get("path"): row for row in implementation_rows if isinstance(row, Mapping)}
    expected_support_files = []
    for captured_path, source_path in (
        (
            "producer_bootstrap.py",
            "bench/world_model_lifecycle/producer_bootstrap.py",
        ),
        ("protocol.json", "bench/world_model_lifecycle/protocol.json"),
        (
            "schemas/raw-result.schema.json",
            "bench/world_model_lifecycle/schemas/raw-result.schema.json",
        ),
    ):
        source_row = implementation_by_path.get(source_path)
        if not isinstance(source_row, Mapping):
            raise ArtifactAuditError("development audit support source is absent from the binding")
        expected_support_files.append(
            {
                "path": captured_path,
                "bytes": source_row.get("bytes"),
                "sha256": source_row.get("sha256"),
            }
        )
    if (
        set(audit_execution) != _DEVELOPMENT_AUDIT_EXECUTION_FIELDS
        or audit_execution.get("source_mode") != "descriptor"
        or any(
            not _is_sha256(audit_execution.get(field))
            for field in (
                "receipt_sha256",
                "runtime_manifest_sha256",
                "invocation_manifest_sha256",
                "stderr_sha256",
                "bootstrap_sha256",
                "runner_source_sha256",
                "auditor_source_sha256",
            )
        )
        or audit_execution.get("bootstrap_sha256") != bound_audit_execution.get("bootstrap_source_sha256")
        or audit_execution.get("runner_source_sha256") != execution_sources.get("audit_runner.py")
        or audit_execution.get("auditor_source_sha256") != _AUDITOR_SOURCE_SHA256
        or not _strict_json_equal(
            audit_execution.get("support_files"),
            expected_support_files,
        )
    ):
        raise ArtifactAuditError("development audit execution identity is not exact")

    fixed_roles = {
        "producer_manifest_member": "producer/producer-manifest.json",
        "raw_result_member": "producer/result.json",
        "result_qualification_member": ("evidence/development-result-qualification.json"),
        "independent_audit_member": "evidence/independent-audit.json",
        "audit_reproduction_member": "evidence/audit-reproduction.json",
    }
    if any(closure.get(field) != expected for field, expected in fixed_roles.items()):
        raise ArtifactAuditError("development qualification fixed archive roles changed")
    sidecar_roles = {
        "audit_runtime_manifest_member": (
            "development-audit-runtime",
            audit_execution["runtime_manifest_sha256"],
            ".json",
        ),
        "audit_invocation_manifest_member": (
            "development-audit-invocation",
            audit_execution["invocation_manifest_sha256"],
            ".json",
        ),
        "audit_stderr_member": (
            "development-audit-stderr",
            audit_execution["stderr_sha256"],
            ".log",
        ),
    }
    for field, (stem, digest, suffix) in sidecar_roles.items():
        if closure.get(field) != f"evidence/{stem}-{cast(str, digest)[:16]}{suffix}":
            raise ArtifactAuditError(f"development qualification {field} is not content addressed")

    members = archive.get("members")
    if (
        set(archive)
        != {
            "format",
            "file",
            "canonical_path",
            "bytes",
            "sha256",
            "members",
        }
        or archive.get("format") != "ustar-uncompressed-v1"
        or type(archive.get("bytes")) is not int
        or cast(int, archive.get("bytes")) < 0
        or not isinstance(members, list)
        or not members
        or len(members) > 100_000
    ):
        raise ArtifactAuditError("development qualification archive identity is malformed")
    member_digests: dict[str, str] = {}
    previous = ""
    for index, row in enumerate(members):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row.get("path")
            or cast(str, row.get("path")) <= previous
            or "\\" in cast(str, row.get("path"))
            or Path(cast(str, row.get("path"))).is_absolute()
            or "." in Path(cast(str, row.get("path"))).parts
            or ".." in Path(cast(str, row.get("path"))).parts
            or Path(cast(str, row.get("path"))).as_posix() != row.get("path")
            or type(row.get("bytes")) is not int
            or cast(int, row.get("bytes")) < 0
            or cast(int, row.get("bytes")) > (4 << 30)
            or not _is_sha256(row.get("sha256"))
        ):
            raise ArtifactAuditError(f"development qualification archive member {index} is invalid")
        previous = cast(str, row["path"])
        member_digests[previous] = cast(str, row["sha256"])

    def member_digest(role: str) -> str:
        member = closure.get(role)
        if not isinstance(member, str) or member not in member_digests:
            raise ArtifactAuditError(f"development qualification archive role {role} is invalid")
        return member_digests[member]

    expected_block = {
        "closure_schema": closure["schema"],
        "closure_file": (
            "development-closure-"
            f"{hashlib.sha256(payload).hexdigest()[:16]}.json"
        ),
        "closure_bytes": len(payload),
        "closure_sha256": hashlib.sha256(payload).hexdigest(),
        "qualification_archive_file": archive["file"],
        "qualification_archive_path": archive["canonical_path"],
        "qualification_archive_bytes": archive["bytes"],
        "qualification_archive_sha256": archive["sha256"],
        "qualification_archive_members_sha256": hashlib.sha256(_canonical_json_bytes({"members": members})).hexdigest(),
        "producer_manifest_sha256": member_digest("producer_manifest_member"),
        "raw_result_sha256": member_digest("raw_result_member"),
        "result_qualification_sha256": member_digest("result_qualification_member"),
        "independent_audit_sha256": member_digest("independent_audit_member"),
        "audit_reproduction_sha256": member_digest("audit_reproduction_member"),
        "audit_runtime_manifest_sha256": member_digest("audit_runtime_manifest_member"),
        "audit_invocation_manifest_sha256": member_digest("audit_invocation_manifest_member"),
        "audit_stderr_sha256": member_digest("audit_stderr_member"),
        "source_identity_sha256": hashlib.sha256(_canonical_json_bytes(closure_source)).hexdigest(),
        "producer_execution_identity_sha256": hashlib.sha256(_canonical_json_bytes(producer_execution)).hexdigest(),
        "producer_custody_identity_sha256": hashlib.sha256(_canonical_json_bytes(producer_custody)).hexdigest(),
        "audit_execution_identity_sha256": hashlib.sha256(_canonical_json_bytes(audit_execution)).hexdigest(),
        "git_commit": closure_source["git_commit"],
        "git_tree": closure_source["git_tree"],
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }
    if not _strict_json_equal(block, expected_block):
        raise ArtifactAuditError("development qualification projection differs from its binding")
    runtime_seal_member = producer_custody.get("runtime_seal_member")
    producer_bootstrap_member = producer_custody.get(
        "producer_bootstrap_member"
    )
    launch_bootstrap_member = producer_custody.get(
        "launch_bootstrap_member"
    )
    if (
        not isinstance(runtime_seal_member, str)
        or member_digests.get(runtime_seal_member)
        != producer_execution.get("runtime_seal_sha256")
        or not isinstance(producer_bootstrap_member, str)
        or member_digests.get(producer_bootstrap_member)
        != producer_custody.get("producer_bootstrap_sha256")
        or not isinstance(launch_bootstrap_member, str)
        or member_digests.get(launch_bootstrap_member)
        != producer_custody.get("launch_bootstrap_sha256")
    ):
        raise ArtifactAuditError(
            "development producer custody member is absent from its archive"
        )
    expected_evidence_members = {
        str(closure["result_qualification_member"]),
        str(closure["independent_audit_member"]),
        str(closure["audit_reproduction_member"]),
        str(closure["audit_runtime_manifest_member"]),
        str(closure["audit_invocation_manifest_member"]),
        str(closure["audit_stderr_member"]),
        runtime_seal_member,
        producer_bootstrap_member,
        launch_bootstrap_member,
    }
    observed_evidence_members = {
        path for path in member_digests if path.startswith("evidence/")
    }
    if observed_evidence_members != expected_evidence_members:
        raise ArtifactAuditError(
            "development qualification evidence member set is not exact"
        )
    retained_members = frozenset(
        {
            "producer/producer-manifest.json",
            *expected_evidence_members,
        }
    )
    retained = _verify_development_qualification_archive(
        archive,
        members=members,
        retain_members=retained_members,
    )
    _validate_archived_development_producer(retained, members)
    protocol_source_row = implementation_by_path.get(
        "bench/world_model_lifecycle/protocol.json"
    )
    if not isinstance(protocol_source_row, Mapping):
        raise ArtifactAuditError(
            "development protocol source is absent from its binding"
        )
    _validate_archived_development_semantics(
        retained,
        closure=closure,
        member_digests=member_digests,
        protocol_sha256=protocol_source_row.get("sha256"),
        source=source,
    )
    if (
        hashlib.sha256(retained[producer_bootstrap_member]).hexdigest()
        != producer_custody.get("producer_bootstrap_sha256")
        or hashlib.sha256(retained[launch_bootstrap_member]).hexdigest()
        != producer_custody.get("launch_bootstrap_sha256")
    ):
        raise ArtifactAuditError(
            "development producer bootstrap custody differs from its archive"
        )
    for member, relative in (
        (
            producer_bootstrap_member,
            "bench/world_model_lifecycle/producer_bootstrap.py",
        ),
        (
            launch_bootstrap_member,
            "bench/world_model_lifecycle/launch_bootstrap.py",
        ),
    ):
        source_row = implementation_by_path.get(relative)
        live_path = Path.cwd() / relative
        try:
            if (
                live_path.resolve(strict=True) != live_path
                or live_path.is_symlink()
            ):
                raise ArtifactAuditError(
                    "live development bootstrap path is aliased"
                )
            live_payload, live_bytes, live_digest, live_identity = (
                _read_stable_regular_file(
                    live_path,
                    _MAX_SOURCE_FILE_BYTES,
                    label=f"live development bootstrap {relative}",
                    capture_payload=True,
                )
            )
        except OSError as error:
            raise ArtifactAuditError(
                "live development bootstrap cannot be resolved"
            ) from error
        if (
            live_payload is None
            or live_identity[3] != 1
            or not isinstance(source_row, Mapping)
            or type(source_row.get("bytes")) is not int
            or source_row.get("bytes") != live_bytes
            or source_row.get("sha256") != live_digest
            or retained[member] != live_payload
        ):
            raise ArtifactAuditError(
                "live development bootstrap differs from its binding "
                "or archived custody role"
            )
    archived_runtime_seal_payload = retained.get(runtime_seal_member)
    expected_runtime_seal_payload = (
        _canonical_json_bytes(preformal_runtime_seal) + b"\n"
    )
    archived_runtime_seal = (
        _canonical_json_object_payload(
            archived_runtime_seal_payload,
            label="archived development producer runtime seal",
        )
        if isinstance(archived_runtime_seal_payload, bytes)
        else None
    )
    archived_runtime_python = (
        archived_runtime_seal.get("python")
        if isinstance(archived_runtime_seal, Mapping)
        else None
    )
    if (
        archived_runtime_seal is None
        or set(archived_runtime_seal)
        != {
            "schema",
            "experiment_id",
            "protocol_version",
            "assurance",
            "git_commit",
            "git_tree",
            "worktree_clean",
            "python",
            "required_flags",
            "process_environment",
            "bootstrap_source_sha256",
            "standard_library",
            "package_roots",
            "package_ownership",
        }
        or archived_runtime_seal.get("schema")
        != "prospect.wm001.runtime-seal.v1"
        or archived_runtime_seal.get("experiment_id") != "WM-001"
        or archived_runtime_seal.get("protocol_version") != "1.16.0"
        or not _strict_json_equal(
            archived_runtime_seal.get("assurance"),
            _ASSURANCE,
        )
        or not isinstance(archived_runtime_python, Mapping)
        or archived_runtime_python.get("executable")
        != producer_execution.get("python_executable")
        or archived_runtime_python.get("sha256")
        != producer_execution.get("python_executable_sha256")
        or not _strict_json_equal(
            archived_runtime_python.get("version"),
            [
                int(part)
                for part in cast(str, producer_execution["python_version"])
                .split(".")
            ],
        )
        or archived_runtime_seal_payload != expected_runtime_seal_payload
        or hashlib.sha256(archived_runtime_seal_payload).hexdigest()
        != producer_execution.get("runtime_seal_sha256")
    ):
        raise ArtifactAuditError(
            "development producer runtime seal differs from the exact "
            "preformal assured seal"
        )
    result_qualification_member = closure.get(
        "result_qualification_member"
    )
    if not isinstance(result_qualification_member, str):
        raise ArtifactAuditError(
            "development result qualification archive role is invalid"
        )
    return retained[result_qualification_member]


def preflight_formal_input_package(
    binding_path: Path,
) -> dict[str, object]:
    """Exercise the exact independent pre-outcome consumer before formal use."""

    binding_payload, binding_bytes, binding_sha256, binding_identity = (
        _read_stable_regular_file(
            binding_path,
            64 << 20,
            label="prospective formal binding",
            capture_payload=True,
        )
    )
    if (
        binding_payload is None
        or binding_identity[3] != 1
        or binding_path.is_symlink()
        or binding_path.name != "formal-binding.json"
    ):
        raise ArtifactAuditError(
            "prospective formal binding lacks single-link custody"
        )
    binding = _canonical_json_object_payload(
        binding_payload,
        label="prospective formal binding",
    )
    source = binding.get("source")
    dependencies = binding.get("dependencies")
    runtime = binding.get("runtime")
    development = binding.get("development_qualification")
    audit_execution = binding.get("audit_execution")
    protocol = binding.get("protocol")
    if (
        binding.get("schema")
        != "prospect.world-model-lifecycle.formal-binding.v10"
        or binding.get("experiment_id") != "WM-001"
        or not _strict_json_equal(binding.get("assurance"), _ASSURANCE)
        or not isinstance(protocol, Mapping)
        or protocol.get("version") != "1.16.0"
        or not all(
            isinstance(value, Mapping)
            for value in (
                source,
                dependencies,
                runtime,
                development,
                audit_execution,
            )
        )
    ):
        raise ArtifactAuditError(
            "prospective formal binding has no complete v1.16 input package"
        )
    assert isinstance(source, Mapping)
    assert isinstance(dependencies, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(development, Mapping)
    assert isinstance(audit_execution, Mapping)
    root = binding_path.parent
    report_path = _resolve_artifact_file(
        root,
        source.get("test_report_file"),
        label="prospective preformal report",
    )
    report_payload_raw, _, _, report_identity = (
        _read_stable_regular_file(
            report_path,
            64 << 20,
            label="prospective preformal report",
            capture_payload=True,
        )
    )
    if report_payload_raw is None or report_identity[3] != 1:
        raise ArtifactAuditError(
            "prospective preformal report lacks single-link custody"
        )
    report_payload = report_payload_raw
    (
        report,
        runtime_conformance,
        accepted_closure,
    ) = _validate_preformal_test_report_v2(
        report_payload,
        root=root,
        source=source,
        dependencies=dependencies,
        runtime=runtime,
    )
    closure_path = _resolve_artifact_file(
        root,
        development.get("closure_file"),
        label="prospective development closure",
    )
    closure_payload_raw, _, _, closure_identity = (
        _read_stable_regular_file(
            closure_path,
            64 << 20,
            label="prospective development closure",
            capture_payload=True,
        )
    )
    if closure_payload_raw is None or closure_identity[3] != 1:
        raise ArtifactAuditError(
            "prospective development closure lacks single-link custody"
        )
    closure_payload = closure_payload_raw
    archived_result_qualification = _validate_development_qualification(
        closure_payload,
        block=development,
        source=source,
        dependencies=dependencies,
        runtime=runtime,
        bound_audit_execution=audit_execution,
        preformal_runtime_seal=cast(
            Mapping[str, object],
            report["runtime_seal"],
        ),
    )
    result_qualification_path = _resolve_artifact_file(
        root,
        _DEVELOPMENT_RESULT_QUALIFICATION_NAME,
        label="prospective development result qualification",
    )
    (
        result_qualification_payload,
        _,
        result_qualification_sha256,
        result_qualification_identity,
    ) = _read_stable_regular_file(
        result_qualification_path,
        64 << 20,
        label="prospective development result qualification",
        capture_payload=True,
    )
    if (
        result_qualification_payload is None
        or result_qualification_identity[3] != 1
        or result_qualification_payload
        != archived_result_qualification
        or result_qualification_sha256
        != development.get("result_qualification_sha256")
    ):
        raise ArtifactAuditError(
            "prospective development result qualification differs from "
            "its archive or formal binding"
        )
    if (
        accepted_closure.get("development_closure_sha256")
        != development.get("closure_sha256")
        or accepted_closure.get("producer_manifest_sha256")
        != development.get("producer_manifest_sha256")
        or accepted_closure.get("raw_result_sha256")
        != development.get("raw_result_sha256")
        or runtime_conformance.get("conformance_sha256")
        != hashlib.sha256(
            _canonical_json_bytes(dict(audit_execution))
        ).hexdigest()
        or runtime_conformance.get(
            "restart_runtime_conformance_report_sha256"
        )
        != audit_execution.get(
            "restart_runtime_conformance_report_sha256"
        )
        or runtime_conformance.get(
            "restart_runtime_execution_receipt_sha256"
        )
        != audit_execution.get(
            "restart_runtime_execution_receipt_sha256"
        )
        or runtime_conformance.get("restart_runtime_support_files")
        != audit_execution.get("restart_runtime_support_files")
        or runtime_conformance.get("restart_runtime_repeat_count")
        != audit_execution.get("restart_runtime_repeat_count")
        or runtime_conformance.get(
            "restart_runtime_path_descriptor_equal"
        )
        != audit_execution.get(
            "restart_runtime_path_descriptor_equal"
        )
    ):
        raise ArtifactAuditError(
            "prospective formal inputs differ from their sealed preformal "
            "rehearsal"
        )
    return {
        "schema": "prospect.wm001.formal-input-preflight.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "binding_bytes": binding_bytes,
        "binding_sha256": binding_sha256,
        "preformal_report_sha256": hashlib.sha256(
            report_payload
        ).hexdigest(),
        "development_closure_sha256": hashlib.sha256(
            closure_payload
        ).hexdigest(),
        "accepted_closure_evidence_sha256": hashlib.sha256(
            _canonical_json_bytes(dict(accepted_closure))
        ).hexdigest(),
        "runtime_conformance_sha256": hashlib.sha256(
            _canonical_json_bytes(dict(runtime_conformance))
        ).hexdigest(),
        "auditor_source_sha256": _AUDITOR_SOURCE_SHA256,
        "passed": True,
    }


def _formal_input_preflight_receipt(
    payload: bytes,
    *,
    binding_payload: bytes,
) -> Mapping[str, object]:
    receipt = _canonical_json_object_payload(
        payload,
        label="formal input preflight receipt",
    )
    fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "binding_bytes",
        "binding_sha256",
        "preformal_report_sha256",
        "development_closure_sha256",
        "accepted_closure_evidence_sha256",
        "runtime_conformance_sha256",
        "auditor_source_sha256",
        "passed",
    }
    if (
        set(receipt) != fields
        or receipt.get("schema")
        != "prospect.wm001.formal-input-preflight.v1"
        or receipt.get("experiment_id") != "WM-001"
        or receipt.get("protocol_version") != "1.16.0"
        or type(receipt.get("binding_bytes")) is not int
        or receipt.get("binding_bytes") != len(binding_payload)
        or receipt.get("binding_sha256")
        != hashlib.sha256(binding_payload).hexdigest()
        or any(
            not _is_sha256(receipt.get(field))
            for field in (
                "preformal_report_sha256",
                "development_closure_sha256",
                "accepted_closure_evidence_sha256",
                "runtime_conformance_sha256",
                "auditor_source_sha256",
            )
        )
        or receipt.get("auditor_source_sha256") != _AUDITOR_SOURCE_SHA256
        or receipt.get("passed") is not True
    ):
        raise ArtifactAuditError(
            "formal input preflight receipt is malformed or misbound"
        )
    return receipt


def _verify_canonical_development_ustar(
    descriptor: int,
    *,
    archive_bytes: int,
    members: list[object],
) -> None:
    """Require exact USTAR headers, layout, padding, and terminal zero blocks."""

    offset = 0
    for index, raw_member in enumerate(members):
        if not isinstance(raw_member, Mapping):
            raise ArtifactAuditError(f"development archive member {index} is malformed")
        name = raw_member.get("path")
        size = raw_member.get("bytes")
        if not isinstance(name, str) or type(size) is not int:
            raise ArtifactAuditError(f"development archive member {index} has no exact layout")
        header = os.pread(descriptor, tarfile.BLOCKSIZE, offset)
        if len(header) != tarfile.BLOCKSIZE:
            raise ArtifactAuditError("development archive ended before a canonical USTAR header")
        expected = tarfile.TarInfo(name)
        expected.size = size
        expected.mode = 0o444
        expected.uid = 0
        expected.gid = 0
        expected.uname = ""
        expected.gname = ""
        expected.mtime = 0
        expected.type = tarfile.REGTYPE
        expected.linkname = ""
        expected.pax_headers = {}
        try:
            expected_header = expected.tobuf(
                format=tarfile.USTAR_FORMAT,
                encoding="utf-8",
                errors="strict",
            )
        except (UnicodeError, ValueError) as error:
            raise ArtifactAuditError("development archive member cannot use canonical USTAR") from error
        if header != expected_header:
            raise ArtifactAuditError("development archive contains a noncanonical or hidden header")
        offset += tarfile.BLOCKSIZE + size
        padding = (-size) % tarfile.BLOCKSIZE
        if padding:
            padding_payload = os.pread(descriptor, padding, offset)
            if len(padding_payload) != padding or padding_payload != bytes(padding):
                raise ArtifactAuditError("development archive member padding is not canonical zero fill")
            offset += padding

    minimum_terminal = offset + 2 * tarfile.BLOCKSIZE
    expected_archive_bytes = (minimum_terminal + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE * tarfile.RECORDSIZE
    if archive_bytes != expected_archive_bytes:
        raise ArtifactAuditError("development archive length is not the canonical USTAR record length")
    while offset < archive_bytes:
        chunk = os.pread(
            descriptor,
            min(1 << 20, archive_bytes - offset),
            offset,
        )
        if not chunk or chunk != bytes(len(chunk)):
            raise ArtifactAuditError("development archive terminal records are not all zero")
        offset += len(chunk)


def _verify_development_qualification_archive(
    archive: Mapping[str, object],
    *,
    members: list[object],
    retain_members: frozenset[str] = frozenset(),
) -> dict[str, bytes]:
    """Independently stream every external USTAR member without extraction."""

    relative = archive.get("canonical_path")
    filename = archive.get("file")
    if (
        not isinstance(relative, str)
        or not isinstance(filename, str)
        or relative != ("bench/world_model_lifecycle/results/development/" + filename)
        or re.fullmatch(
            r"development-qualification-[0-9a-f]{16}\.tar",
            filename,
        )
        is None
    ):
        raise ArtifactAuditError("development qualification archive path is not canonical")
    archive_path = Path.cwd() / relative
    try:
        if (
            not archive_path.is_absolute()
            or archive_path.resolve(strict=True) != archive_path
            or archive_path.is_symlink()
            or not archive_path.is_file()
        ):
            raise ArtifactAuditError("development qualification archive is missing or aliased")
    except OSError as error:
        raise ArtifactAuditError("development qualification archive cannot be resolved") from error
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(archive_path, flags)
    except OSError as error:
        raise ArtifactAuditError("development qualification archive cannot be opened") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 10_240
            or before.st_size
            > _MAX_DEVELOPMENT_QUALIFICATION_ARCHIVE_BYTES
            or archive.get("bytes") != before.st_size
        ):
            raise ArtifactAuditError("development qualification archive custody is invalid")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise ArtifactAuditError("development qualification archive ended while hashed")
            digest.update(chunk)
            offset += len(chunk)
        if (
            digest.hexdigest() != archive.get("sha256")
            or filename != f"development-qualification-{digest.hexdigest()[:16]}.tar"
        ):
            raise ArtifactAuditError("development qualification archive digest changed")
        _verify_canonical_development_ustar(
            descriptor,
            archive_bytes=before.st_size,
            members=members,
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as raw:
            with tarfile.open(fileobj=raw, mode="r|") as stream:
                observed_count = 0
                observed_total = 0
                retained: dict[str, bytes] = {}
                retained_total = 0
                stream_iterator = iter(stream)
                for expected, member in zip(
                    members,
                    stream_iterator,
                    strict=False,
                ):
                    if not isinstance(expected, Mapping):
                        raise ArtifactAuditError("development archive expected member is malformed")
                    if (
                        member.name != expected.get("path")
                        or not member.isreg()
                        or member.size != expected.get("bytes")
                        or member.mode != 0o444
                        or member.uid != 0
                        or member.gid != 0
                        or member.uname != ""
                        or member.gname != ""
                        or member.mtime != 0
                        or member.linkname != ""
                        or bool(member.pax_headers)
                    ):
                        raise ArtifactAuditError("development archive member metadata changed")
                    extracted = stream.extractfile(member)
                    if extracted is None:
                        raise ArtifactAuditError("development archive regular member is unreadable")
                    member_digest = hashlib.sha256()
                    member_bytes = 0
                    retained_payload = (
                        bytearray()
                        if member.name in retain_members
                        else None
                    )
                    while True:
                        chunk = extracted.read(1 << 20)
                        if not chunk:
                            break
                        member_digest.update(chunk)
                        member_bytes += len(chunk)
                        if retained_payload is not None:
                            if (
                                member_bytes
                                > _MAX_RETAINED_DEVELOPMENT_ARCHIVE_MEMBER_BYTES
                            ):
                                raise ArtifactAuditError(
                                    "retained development archive member "
                                    "exceeds its byte limit"
                                )
                            retained_payload.extend(chunk)
                    if member_bytes != expected.get("bytes") or member_digest.hexdigest() != expected.get("sha256"):
                        raise ArtifactAuditError("development archive member bytes changed")
                    if retained_payload is not None:
                        retained[member.name] = bytes(retained_payload)
                        retained_total += len(retained_payload)
                        if (
                            retained_total
                            > _MAX_RETAINED_DEVELOPMENT_ARCHIVE_TOTAL_BYTES
                        ):
                            raise ArtifactAuditError(
                                "retained development archive evidence "
                                "exceeds its total byte limit"
                            )
                    observed_count += 1
                    observed_total += member_bytes
                    if observed_count > 100_000 or observed_total > (32 << 30):
                        raise ArtifactAuditError("development archive exceeds evidence limits")
                if observed_count != len(members):
                    raise ArtifactAuditError("development archive member count changed")
                try:
                    next(stream_iterator)
                except StopIteration:
                    pass
                else:
                    raise ArtifactAuditError("development archive contains an extra member")
        if set(retained) != set(retain_members):
            raise ArtifactAuditError(
                "development archive omits a retained custody member"
            )
        after = os.fstat(descriptor)
        if offset != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ArtifactAuditError("development qualification archive changed during audit")
    finally:
        os.close(descriptor)
    return retained


def _validate_audit_execution_conformance(
    *,
    block: Mapping[str, object],
    bootstrap_payload: bytes,
    request_payload: bytes,
    path_runtime_manifest_payload: bytes,
    descriptor_runtime_manifest_payload: bytes,
    path_invocation_manifest_payload: bytes,
    descriptor_invocation_manifest_payload: bytes,
    report_payload: bytes,
    execution_receipt_payload: bytes,
    outcome_runtime_manifest_payload: bytes,
    restart_runtime_report_payload: bytes,
    restart_runtime_receipt_payload: bytes,
    dependencies: Mapping[str, object],
    runtime: Mapping[str, object],
    source: Mapping[str, object],
    root: Path,
    preformal_repository_cwd: object,
    verify_live_outcome_runtime: bool = True,
) -> None:
    expected_fields = {
        "runner_source_sha256",
        "auditor_source_sha256",
        "adjudicator_source_sha256",
        "bootstrap_source_file",
        "bootstrap_source_bytes",
        "bootstrap_source_sha256",
        "prebinding_request_file",
        "prebinding_request_bytes",
        "prebinding_request_sha256",
        "prebinding_request_identity_sha256",
        "prebinding_path_runtime_manifest_file",
        "prebinding_path_runtime_manifest_bytes",
        "prebinding_path_runtime_manifest_sha256",
        "prebinding_descriptor_runtime_manifest_file",
        "prebinding_descriptor_runtime_manifest_bytes",
        "prebinding_descriptor_runtime_manifest_sha256",
        "prebinding_path_invocation_manifest_file",
        "prebinding_path_invocation_manifest_bytes",
        "prebinding_path_invocation_manifest_sha256",
        "prebinding_descriptor_invocation_manifest_file",
        "prebinding_descriptor_invocation_manifest_bytes",
        "prebinding_descriptor_invocation_manifest_sha256",
        "prebinding_conformance_report_file",
        "prebinding_conformance_report_bytes",
        "prebinding_conformance_report_sha256",
        "prebinding_execution_receipt_file",
        "prebinding_execution_receipt_bytes",
        "prebinding_execution_receipt_sha256",
        "outcome_runtime_manifest_file",
        "outcome_runtime_manifest_bytes",
        "outcome_runtime_manifest_sha256",
        "restart_runtime_conformance_report_file",
        "restart_runtime_conformance_report_bytes",
        "restart_runtime_conformance_report_sha256",
        "restart_runtime_execution_receipt_file",
        "restart_runtime_execution_receipt_bytes",
        "restart_runtime_execution_receipt_sha256",
        "restart_runtime_support_files",
        "restart_runtime_repeat_count",
        "restart_runtime_path_descriptor_equal",
        "outcome_source_mode",
        "outcome_support_files",
        "outcome_argv_role",
        "outcome_working_directory",
        "interpreter_flags",
        "repeat_count",
        "path_descriptor_equal",
        "passed",
    }
    if (
        set(block) != expected_fields
        or block.get("interpreter_flags") != ["-I", "-S", "-B"]
        or type(block.get("repeat_count")) is not int
        or cast(int, block.get("repeat_count")) < 3
        or block.get("path_descriptor_equal") is not True
        or type(block.get("restart_runtime_repeat_count")) is not int
        or cast(int, block.get("restart_runtime_repeat_count")) < 3
        or block.get("restart_runtime_repeat_count")
        != block.get("repeat_count")
        or block.get("restart_runtime_path_descriptor_equal") is not True
        or block.get("restart_runtime_support_files")
        != list(_RESTART_RUNTIME_SUPPORT_FILES)
        or block.get("outcome_source_mode") != "descriptor"
        or block.get("outcome_support_files")
        != [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ]
        or block.get("outcome_argv_role")
        != [
            "<canonical-producer-root>",
            "--producer-bootstrap",
            "<captured-producer-bootstrap>",
        ]
        or block.get("outcome_working_directory") != preformal_repository_cwd
        or block.get("passed") is not True
    ):
        raise ArtifactAuditError("audit-execution conformance declaration is invalid")
    for payload, prefix in (
        (bootstrap_payload, "bootstrap_source"),
        (request_payload, "prebinding_request"),
        (
            path_runtime_manifest_payload,
            "prebinding_path_runtime_manifest",
        ),
        (
            descriptor_runtime_manifest_payload,
            "prebinding_descriptor_runtime_manifest",
        ),
        (
            path_invocation_manifest_payload,
            "prebinding_path_invocation_manifest",
        ),
        (
            descriptor_invocation_manifest_payload,
            "prebinding_descriptor_invocation_manifest",
        ),
        (report_payload, "prebinding_conformance_report"),
        (
            execution_receipt_payload,
            "prebinding_execution_receipt",
        ),
        (outcome_runtime_manifest_payload, "outcome_runtime_manifest"),
        (
            restart_runtime_report_payload,
            "restart_runtime_conformance_report",
        ),
        (
            restart_runtime_receipt_payload,
            "restart_runtime_execution_receipt",
        ),
    ):
        if (
            block.get(f"{prefix}_bytes") != len(payload)
            or block.get(f"{prefix}_sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ArtifactAuditError("audit-execution support bytes differ from their binding")
    for prefix, suffix in (
        ("audit-bootstrap", ".py"),
        ("audit-prebinding-request", ".json"),
        ("audit-prebinding-path-runtime", ".json"),
        ("audit-prebinding-descriptor-runtime", ".json"),
        ("audit-prebinding-path-invocation", ".json"),
        ("audit-prebinding-descriptor-invocation", ".json"),
        ("audit-prebinding-conformance", ".json"),
        ("audit-prebinding-execution-receipt", ".json"),
        ("audit-outcome-runtime", ".json"),
        ("audit-restart-runtime-conformance", ".json"),
        ("audit-restart-runtime-execution-receipt", ".json"),
    ):
        field_prefix = {
            "audit-bootstrap": "bootstrap_source",
            "audit-prebinding-request": "prebinding_request",
            "audit-prebinding-path-runtime": ("prebinding_path_runtime_manifest"),
            "audit-prebinding-descriptor-runtime": ("prebinding_descriptor_runtime_manifest"),
            "audit-prebinding-path-invocation": ("prebinding_path_invocation_manifest"),
            "audit-prebinding-descriptor-invocation": ("prebinding_descriptor_invocation_manifest"),
            "audit-prebinding-conformance": ("prebinding_conformance_report"),
            "audit-prebinding-execution-receipt": ("prebinding_execution_receipt"),
            "audit-outcome-runtime": "outcome_runtime_manifest",
            "audit-restart-runtime-conformance": (
                "restart_runtime_conformance_report"
            ),
            "audit-restart-runtime-execution-receipt": (
                "restart_runtime_execution_receipt"
            ),
        }[prefix]
        digest = block.get(f"{field_prefix}_sha256")
        if not _is_sha256(digest) or block.get(f"{field_prefix}_file") != f"{prefix}-{cast(str, digest)[:16]}{suffix}":
            raise ArtifactAuditError("audit-execution evidence filename is not content-addressed")

    path_runtime_manifest = _canonical_json_object_payload(
        path_runtime_manifest_payload,
        label="prebinding path runtime manifest",
    )
    descriptor_runtime_manifest = _canonical_json_object_payload(
        descriptor_runtime_manifest_payload,
        label="prebinding descriptor runtime manifest",
    )
    path_invocation_manifest = _canonical_json_object_payload(
        path_invocation_manifest_payload,
        label="prebinding path invocation manifest",
    )
    descriptor_invocation_manifest = _canonical_json_object_payload(
        descriptor_invocation_manifest_payload,
        label="prebinding descriptor invocation manifest",
    )
    outcome_runtime_manifest = _canonical_json_object_payload(
        outcome_runtime_manifest_payload,
        label="outcome descriptor runtime manifest",
    )
    request = _canonical_json_object_payload(
        request_payload,
        label="prebinding conformance request",
    )
    report = _canonical_json_object_payload(
        report_payload,
        label="prebinding conformance report",
    )
    execution_receipt = _canonical_json_object_payload(
        execution_receipt_payload,
        label="prebinding execution receipt",
    )
    restart_runtime_report = _canonical_json_object_payload(
        restart_runtime_report_payload,
        label="restart-runtime conformance report",
    )
    restart_runtime_receipt = _canonical_json_object_payload(
        restart_runtime_receipt_payload,
        label="restart-runtime execution receipt",
    )
    try:
        validated_request = _prebinding_validate_request(request)
    except _PrebindingConformanceError as error:
        raise ArtifactAuditError(f"bound prebinding request is invalid: {error.code}") from error

    implementation = source.get("implementation_files")
    if not isinstance(implementation, list):
        raise ArtifactAuditError("audit-execution has no implementation source binding")

    def bound_source(relative: str) -> tuple[bytes, str]:
        matches = [row for row in implementation if isinstance(row, Mapping) and row.get("path") == relative]
        if len(matches) != 1:
            raise ArtifactAuditError(f"audit-execution source binding is missing {relative}")
        path = root / "source" / Path(relative)
        payload = _read_bounded(
            path,
            _MAX_SOURCE_FILE_BYTES,
            label=f"bound audit source {relative}",
        )
        digest = hashlib.sha256(payload).hexdigest()
        if matches[0].get("bytes") != len(payload) or matches[0].get("sha256") != digest:
            raise ArtifactAuditError(f"audit-execution source bytes changed: {relative}")
        return payload, digest

    auditor_payload, auditor_digest = bound_source("bench/world_model_lifecycle/artifact_audit.py")
    runner_payload, runner_digest = bound_source("bench/world_model_lifecycle/audit_runner.py")
    _, adjudicator_digest = bound_source("bench/world_model_lifecycle/adjudication.py")
    if (
        block.get("runner_source_sha256") != runner_digest
        or block.get("auditor_source_sha256") != auditor_digest
        or block.get("adjudicator_source_sha256") != adjudicator_digest
        or auditor_digest != _AUDITOR_SOURCE_SHA256
    ):
        raise ArtifactAuditError("audit-execution source identities differ from the source snapshot")
    try:
        runner_tree = ast.parse(
            runner_payload.decode("utf-8"),
            filename="bound audit_runner.py",
        )
        bootstrap_candidates = [
            ast.literal_eval(node.value)
            for node in runner_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_BOOTSTRAP_SOURCE" for target in node.targets)
        ]
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise ArtifactAuditError("bound audit-runner bootstrap cannot be independently decoded") from error
    if (
        len(bootstrap_candidates) != 1
        or not isinstance(bootstrap_candidates[0], bytes)
        or bootstrap_candidates[0] != bootstrap_payload
    ):
        raise ArtifactAuditError("audit bootstrap bytes differ from the bound runner source")

    package_roots = dependencies.get("package_roots")
    producer_environment = runtime.get("process_environment")
    python_executable = dependencies.get("python_executable")
    if (
        not isinstance(package_roots, list)
        or not package_roots
        or any(not isinstance(row, Mapping) for row in package_roots)
        or not isinstance(producer_environment, Mapping)
        or not isinstance(python_executable, str)
    ):
        raise ArtifactAuditError("audit-execution dependency or runtime closure is malformed")
    closure_root_rows = [dict(row) for row in package_roots if isinstance(row, Mapping)]
    closure_root_paths = [cast(str, row.get("path")) for row in package_roots if isinstance(row, Mapping)]
    standard_library = dependencies.get("standard_library")
    if not isinstance(standard_library, Mapping):
        raise ArtifactAuditError("audit-execution standard-library closure is malformed")
    safe_environment = {
        key: value for key, value in producer_environment.items() if key in _PREBINDING_SHARED_ENVIRONMENT_KEYS
    }
    expected_python = {
        "executable": python_executable,
        "resolved_executable": str(Path(python_executable).resolve(strict=True)),
        "sha256": dependencies.get("python_executable_sha256"),
        "version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ],
    }

    protocol_payload = _read_bounded(
        root / "protocol.json",
        4 << 20,
        label="bound protocol support",
    )
    raw_schema_payload = _read_bounded(
        root / "schemas" / "raw-result.schema.json",
        4 << 20,
        label="bound raw-result schema support",
    )
    producer_bootstrap_payload, _ = bound_source(
        "bench/world_model_lifecycle/producer_bootstrap.py"
    )
    prebinding_support_payloads: dict[str, bytes] = {
        "prebinding-request.json": request_payload,
        "protocol.json": protocol_payload,
    }
    for name in (
        "learning.py",
        "model.py",
        "planning.py",
        "runtime_lane.py",
    ):
        payload, _ = bound_source(f"bench/world_model_lifecycle/{name}")
        prebinding_support_payloads[name] = payload
    outcome_support_payloads = {
        "producer_bootstrap.py": producer_bootstrap_payload,
        "protocol.json": protocol_payload,
        "schemas/raw-result.schema.json": raw_schema_payload,
    }

    def expected_support_rows(
        payloads: Mapping[str, bytes],
    ) -> list[dict[str, object]]:
        return [
            {
                "path": path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(payloads.items())
        ]

    expected_common = {
        "schema": "prospect.wm001.audit-runtime-manifest.v1",
        "assurance": _ASSURANCE,
        "bootstrap_sha256": block.get("bootstrap_source_sha256"),
        "python": expected_python,
        "required_flags": dict(_PREBINDING_AUDITOR_FLAGS),
        "closure_import_roots": closure_root_rows,
        "standard_library": dict(standard_library),
        "environment": safe_environment,
        "limits": {
            "timeout_seconds": 600,
            "stdout_bytes": 64 << 20,
            "stderr_bytes": 16 << 20,
        },
    }

    def validate_manifest(
        manifest: Mapping[str, object],
        *,
        mode: str,
        support_payloads: Mapping[str, bytes],
        label: str,
    ) -> None:
        if (
            set(manifest)
            != {
                "schema",
                "assurance",
                "bootstrap_sha256",
                "python",
                "required_flags",
                "source",
                "support_files",
                "closure_import_roots",
                "standard_library",
                "environment",
                "limits",
            }
            or any(
                not _strict_json_equal(
                    manifest.get(field),
                    value,
                )
                for field, value in expected_common.items()
            )
            or not _strict_json_equal(
                manifest.get("source"),
                {
                    "mode": mode,
                    "path": "artifact_audit.py",
                    "bytes": len(auditor_payload),
                    "sha256": auditor_digest,
                },
            )
            or not _strict_json_equal(
                manifest.get("support_files"),
                expected_support_rows(support_payloads),
            )
        ):
            raise ArtifactAuditError(f"{label} differs from the exact bound audit closure")

    validate_manifest(
        path_runtime_manifest,
        mode="path",
        support_payloads=prebinding_support_payloads,
        label="prebinding path runtime manifest",
    )
    validate_manifest(
        descriptor_runtime_manifest,
        mode="descriptor",
        support_payloads=prebinding_support_payloads,
        label="prebinding descriptor runtime manifest",
    )
    path_as_descriptor = json.loads(_canonical_json_bytes(path_runtime_manifest))
    if not isinstance(path_as_descriptor, dict):
        raise ArtifactAuditError("prebinding runtime manifest normalization failed")
    path_source = path_as_descriptor.get("source")
    if not isinstance(path_source, dict):
        raise ArtifactAuditError("prebinding path runtime source is malformed")
    path_source["mode"] = "descriptor"
    if not _strict_json_equal(
        path_as_descriptor,
        dict(descriptor_runtime_manifest),
    ):
        raise ArtifactAuditError("prebinding path and descriptor manifests differ beyond source mode")

    expected_auditor_argv = [
        "--prebinding-conformance",
        "@captured/prebinding-request.json",
    ]

    def validate_invocation_manifest(
        invocation: Mapping[str, object],
        *,
        runtime_payload: bytes,
        label: str,
    ) -> None:
        expected = {
            "schema": ("prospect.wm001.audit-invocation-manifest.v1"),
            "runtime_manifest_sha256": hashlib.sha256(runtime_payload).hexdigest(),
            "working_directory": preformal_repository_cwd,
            "auditor_argv": expected_auditor_argv,
        }
        if not _strict_json_equal(dict(invocation), expected):
            raise ArtifactAuditError(f"{label} differs from the exact prebinding invocation")

    validate_invocation_manifest(
        path_invocation_manifest,
        runtime_payload=path_runtime_manifest_payload,
        label="prebinding path invocation manifest",
    )
    validate_invocation_manifest(
        descriptor_invocation_manifest,
        runtime_payload=descriptor_runtime_manifest_payload,
        label="prebinding descriptor invocation manifest",
    )
    validate_manifest(
        outcome_runtime_manifest,
        mode="descriptor",
        support_payloads=outcome_support_payloads,
        label="outcome descriptor runtime manifest",
    )

    if block.get("prebinding_request_identity_sha256") != _prebinding_request_identity(validated_request):
        raise ArtifactAuditError("audit-execution semantic request identity changed")
    components = report.get("components")
    identities = report.get("identities")
    protocol_request = validated_request.get("protocol")
    if (
        set(report)
        != {
            "schema",
            "request_sha256",
            "components",
            "identities",
            "passed",
        }
        or report.get("schema") != "prospect.wm001.prebinding-conformance.v2"
        or report.get("request_sha256") != _prebinding_request_identity(validated_request)
        or report.get("request_sha256") != block.get("prebinding_request_identity_sha256")
        or report.get("passed") is not True
        or not isinstance(components, Mapping)
        or set(components) != set(_PREBINDING_COMPONENT_NAMES)
        or any(
            not isinstance(component, Mapping) or component.get("passed") is not True
            for component in components.values()
        )
        or not isinstance(identities, Mapping)
        or not isinstance(protocol_request, Mapping)
        or identities.get("protocol_sha256") != protocol_request.get("sha256")
        or identities.get("scientific_kernel_sha256") != protocol_request.get("scientific_kernel_sha256")
    ):
        raise ArtifactAuditError("bound prebinding conformance report is not a complete pass")

    repeat_count = cast(int, block.get("repeat_count"))
    expected_modes = [
        *(["path"] * repeat_count),
        *(["descriptor"] * repeat_count),
    ]
    receipt_rows = execution_receipt.get("executions")
    report_identity = {
        "bytes": len(report_payload),
        "sha256": hashlib.sha256(report_payload).hexdigest(),
    }
    first_stderr = (
        receipt_rows[0].get("stderr")
        if isinstance(receipt_rows, list) and receipt_rows and isinstance(receipt_rows[0], Mapping)
        else None
    )
    stderr_identity_valid = (
        isinstance(first_stderr, Mapping)
        and set(first_stderr) == {"bytes", "sha256"}
        and first_stderr.get("bytes") == 0
        and first_stderr.get("sha256") == _SHA256_EMPTY
    )
    runtime_identities = {
        "path": {
            "bytes": len(path_runtime_manifest_payload),
            "sha256": hashlib.sha256(path_runtime_manifest_payload).hexdigest(),
        },
        "descriptor": {
            "bytes": len(descriptor_runtime_manifest_payload),
            "sha256": hashlib.sha256(descriptor_runtime_manifest_payload).hexdigest(),
        },
    }
    invocation_identities = {
        "path": {
            "bytes": len(path_invocation_manifest_payload),
            "sha256": hashlib.sha256(path_invocation_manifest_payload).hexdigest(),
        },
        "descriptor": {
            "bytes": len(descriptor_invocation_manifest_payload),
            "sha256": hashlib.sha256(descriptor_invocation_manifest_payload).hexdigest(),
        },
    }
    expected_prebinding_support = expected_support_rows(prebinding_support_payloads)
    receipt_valid = (
        set(execution_receipt)
        == {
            "schema",
            "repeat_count",
            "execution_count",
            "executions",
            "report_sha256",
            "path_descriptor_byte_identical",
            "execution_conformance_passed",
        }
        and execution_receipt.get("schema") == "prospect.wm001.audit-conformance-receipt.v1"
        and _strict_json_equal(
            execution_receipt.get("repeat_count"),
            repeat_count,
        )
        and _strict_json_equal(
            execution_receipt.get("execution_count"),
            2 * repeat_count,
        )
        and execution_receipt.get("report_sha256") == report_identity["sha256"]
        and execution_receipt.get("path_descriptor_byte_identical") is True
        and execution_receipt.get("execution_conformance_passed") is True
        and isinstance(receipt_rows, list)
        and len(receipt_rows) == len(expected_modes)
        and stderr_identity_valid
    )
    if receipt_valid:
        assert isinstance(receipt_rows, list)
        for ordinal, (row, mode) in enumerate(
            zip(receipt_rows, expected_modes, strict=True),
            start=1,
        ):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "ordinal",
                    "source_mode",
                    "returncode",
                    "stdout",
                    "stderr",
                    "runtime_manifest",
                    "invocation_manifest",
                    "bootstrap_sha256",
                    "auditor_source_sha256",
                    "support_files",
                    "auditor_report_passed",
                }
                or type(row.get("ordinal")) is not int
                or row.get("ordinal") != ordinal
                or row.get("source_mode") != mode
                or type(row.get("returncode")) is not int
                or row.get("returncode") != 0
                or not _strict_json_equal(
                    row.get("stdout"),
                    report_identity,
                )
                or not _strict_json_equal(
                    row.get("stderr"),
                    first_stderr,
                )
                or not _strict_json_equal(
                    row.get("runtime_manifest"),
                    runtime_identities[mode],
                )
                or not _strict_json_equal(
                    row.get("invocation_manifest"),
                    invocation_identities[mode],
                )
                or row.get("bootstrap_sha256") != block.get("bootstrap_source_sha256")
                or row.get("auditor_source_sha256") != block.get("auditor_source_sha256")
                or not _strict_json_equal(
                    row.get("support_files"),
                    expected_prebinding_support,
                )
                or row.get("auditor_report_passed") is not True
            ):
                receipt_valid = False
                break
    if not receipt_valid:
        raise ArtifactAuditError(
            "prebinding execution receipt does not prove every exact path and descriptor execution"
        )
    expected_restart_support = expected_support_rows(
        outcome_support_payloads
    )
    restart_negative_case_ids = [
        "missing-bootstrap-support",
        "extra-bootstrap-support",
        "mutated-bootstrap-identity",
        "development-formal-branch-substitution",
        "formal-development-branch-substitution",
    ]
    if (
        set(restart_runtime_report)
        != {
            "schema",
            "protocol_version",
            "support_files",
            "branches",
            "negative_cases",
            "failure_code",
            "passed",
        }
        or restart_runtime_report.get("schema")
        != _RESTART_RUNTIME_CONFORMANCE_SCHEMA
        or restart_runtime_report.get("protocol_version") != "1.16.0"
        or not _strict_json_equal(
            restart_runtime_report.get("support_files"),
            expected_restart_support,
        )
        or not _strict_json_equal(
            restart_runtime_report.get("branches"),
            {
                "development": {
                    "source_block_present": False,
                    "captured_bootstrap_bound": True,
                    "passed": True,
                },
                "formal": {
                    "source_block_present": True,
                    "source_snapshot_bound": True,
                    "captured_bootstrap_bound": True,
                    "passed": True,
                },
            },
        )
        or not _strict_json_equal(
            restart_runtime_report.get("negative_cases"),
            [
                {"case_id": case_id, "rejected": True}
                for case_id in restart_negative_case_ids
            ],
        )
        or restart_runtime_report.get("failure_code") is not None
        or restart_runtime_report.get("passed") is not True
    ):
        raise ArtifactAuditError(
            "restart-runtime conformance report is not one complete result-free pass"
        )

    path_restart_runtime = json.loads(
        _canonical_json_bytes(outcome_runtime_manifest)
    )
    if (
        not isinstance(path_restart_runtime, dict)
        or not isinstance(path_restart_runtime.get("source"), dict)
    ):
        raise ArtifactAuditError(
            "restart-runtime path manifest normalization failed"
        )
    path_restart_runtime["source"]["mode"] = "path"
    restart_runtime_payloads = {
        "path": _canonical_json_bytes(path_restart_runtime) + b"\n",
        "descriptor": outcome_runtime_manifest_payload,
    }
    restart_arguments = [
        "--restart-runtime-conformance",
        "--producer-bootstrap",
        "@captured/producer_bootstrap.py",
        "--expected-producer-bootstrap-sha256",
        hashlib.sha256(producer_bootstrap_payload).hexdigest(),
    ]
    restart_invocation_payloads = {
        mode: (
            _canonical_json_bytes(
                {
                    "schema": (
                        "prospect.wm001.audit-invocation-manifest.v1"
                    ),
                    "runtime_manifest_sha256": hashlib.sha256(
                        runtime_payload
                    ).hexdigest(),
                    "working_directory": preformal_repository_cwd,
                    "auditor_argv": restart_arguments,
                }
            )
            + b"\n"
        )
        for mode, runtime_payload in restart_runtime_payloads.items()
    }
    restart_repeat_count = cast(
        int,
        block.get("restart_runtime_repeat_count"),
    )
    restart_expected_modes = [
        *(["path"] * restart_repeat_count),
        *(["descriptor"] * restart_repeat_count),
    ]
    restart_receipt_rows = restart_runtime_receipt.get("executions")
    restart_stderr = (
        restart_receipt_rows[0].get("stderr")
        if isinstance(restart_receipt_rows, list)
        and restart_receipt_rows
        and isinstance(restart_receipt_rows[0], Mapping)
        else None
    )
    restart_stderr_valid = (
        isinstance(restart_stderr, Mapping)
        and set(restart_stderr) == {"bytes", "sha256"}
        and restart_stderr.get("bytes") == 0
        and restart_stderr.get("sha256") == _SHA256_EMPTY
    )
    restart_report_identity = {
        "bytes": len(restart_runtime_report_payload),
        "sha256": hashlib.sha256(
            restart_runtime_report_payload
        ).hexdigest(),
    }
    restart_receipt_valid = (
        set(restart_runtime_receipt)
        == {
            "schema",
            "repeat_count",
            "execution_count",
            "executions",
            "report_sha256",
            "path_descriptor_byte_identical",
            "execution_conformance_passed",
        }
        and restart_runtime_receipt.get("schema")
        == "prospect.wm001.audit-conformance-receipt.v1"
        and _strict_json_equal(
            restart_runtime_receipt.get("repeat_count"),
            restart_repeat_count,
        )
        and _strict_json_equal(
            restart_runtime_receipt.get("execution_count"),
            len(restart_expected_modes),
        )
        and restart_runtime_receipt.get("report_sha256")
        == restart_report_identity["sha256"]
        and restart_runtime_receipt.get(
            "path_descriptor_byte_identical"
        )
        is True
        and restart_runtime_receipt.get(
            "execution_conformance_passed"
        )
        is True
        and isinstance(restart_receipt_rows, list)
        and len(restart_receipt_rows) == len(restart_expected_modes)
        and restart_stderr_valid
    )
    if restart_receipt_valid:
        assert isinstance(restart_receipt_rows, list)
        for ordinal, (row, mode) in enumerate(
            zip(
                restart_receipt_rows,
                restart_expected_modes,
                strict=True,
            ),
            start=1,
        ):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "ordinal",
                    "source_mode",
                    "returncode",
                    "stdout",
                    "stderr",
                    "runtime_manifest",
                    "invocation_manifest",
                    "bootstrap_sha256",
                    "auditor_source_sha256",
                    "support_files",
                    "auditor_report_passed",
                }
                or not _strict_json_equal(
                    row.get("ordinal"),
                    ordinal,
                )
                or row.get("source_mode") != mode
                or not _strict_json_equal(row.get("returncode"), 0)
                or not _strict_json_equal(
                    row.get("stdout"),
                    restart_report_identity,
                )
                or not _strict_json_equal(
                    row.get("stderr"),
                    restart_stderr,
                )
                or not _strict_json_equal(
                    row.get("runtime_manifest"),
                    {
                        "bytes": len(restart_runtime_payloads[mode]),
                        "sha256": hashlib.sha256(
                            restart_runtime_payloads[mode]
                        ).hexdigest(),
                    },
                )
                or not _strict_json_equal(
                    row.get("invocation_manifest"),
                    {
                        "bytes": len(
                            restart_invocation_payloads[mode]
                        ),
                        "sha256": hashlib.sha256(
                            restart_invocation_payloads[mode]
                        ).hexdigest(),
                    },
                )
                or row.get("bootstrap_sha256")
                != block.get("bootstrap_source_sha256")
                or row.get("auditor_source_sha256")
                != block.get("auditor_source_sha256")
                or not _strict_json_equal(
                    row.get("support_files"),
                    expected_restart_support,
                )
                or row.get("auditor_report_passed") is not True
            ):
                restart_receipt_valid = False
                break
    if not restart_receipt_valid:
        raise ArtifactAuditError(
            "restart-runtime execution receipt does not prove every exact path and descriptor execution"
        )
    working_directory = block.get("outcome_working_directory")
    if (
        not isinstance(working_directory, str)
        or not Path(working_directory).is_absolute()
        or os.path.abspath(working_directory) != working_directory
    ):
        raise ArtifactAuditError("outcome audit working directory is not canonical")
    if verify_live_outcome_runtime:
        invocation_path = str(_AUDITOR_SOURCE_INVOCATION)
        descriptor_invocation = invocation_path.startswith("/proc/self/fd/") or invocation_path.startswith("/dev/fd/")
        try:
            current_working_directory = str(Path.cwd().resolve(strict=True))
            canonical_root = str(root.resolve(strict=True))
            captured_bootstrap_argument = (
                Path(sys.argv[3]).resolve(strict=True)
                if len(sys.argv) == 4
                else None
            )
        except OSError as error:
            raise ArtifactAuditError("live outcome auditor path identity cannot be resolved") from error
        if (
            not descriptor_invocation
            or current_working_directory != working_directory
            or len(sys.argv) != 4
            or sys.argv[1] != canonical_root
            or sys.argv[2] != "--producer-bootstrap"
            or captured_bootstrap_argument
            != (HERE / "producer_bootstrap.py").resolve(strict=True)
            or dict(os.environ) != safe_environment
            or sys.path[: len(closure_root_paths)] != closure_root_paths
            or _read_bounded(
                cast(Path, captured_bootstrap_argument),
                _MAX_SOURCE_FILE_BYTES,
                label="live captured producer bootstrap support",
            )
            != producer_bootstrap_payload
            or _read_bounded(
                HERE / "protocol.json",
                4 << 20,
                label="live captured protocol support",
            )
            != protocol_payload
            or _read_bounded(
                RESULT_SCHEMA_PATH,
                4 << 20,
                label="live captured raw-result schema support",
            )
            != raw_schema_payload
        ):
            raise ArtifactAuditError(
                "live outcome auditor did not use the bound descriptor "
                "source, working directory, or canonical captured-support argv"
            )


_AUTHORIZATION_ATTEMPT_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "assurance",
    "kind",
    "lane",
    "status",
    "inputs",
    "primary",
    "error",
    "files",
    "file_count",
    "manifest_excludes",
}
_AUTHORIZATION_INPUT_FIELDS = {"path", "bytes", "sha256"}
_AUTHORIZATION_TERMINAL = "operator-attempt.json"
_AUTHORIZATION_MAX_FILES = 1_000
_AUTHORIZATION_MAX_FILE_BYTES = 64 << 20
_AUTHORIZATION_CLOSURE_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "source",
    "producer_root",
    "producer_manifest_member",
    "raw_result_member",
    "result_qualification_member",
    "independent_audit_member",
    "audit_reproduction_member",
    "audit_runtime_manifest_member",
    "audit_invocation_manifest_member",
    "audit_stderr_member",
    "producer_execution",
    "producer_custody",
    "audit_execution",
    "qualification_archive",
    "engineering_verified",
    "audit_reproduced",
    "performance_values_bound",
}
_AUTHORIZATION_AUDIT_PRIMARY_FIELDS = {
    "producer_root",
    "audit_file",
    "executions",
    "execution_failures",
    "reproduction_file",
    "reproduction_runtime_file",
    "claim_file",
}
_AUTHORIZATION_EXECUTION_FIELDS = {
    "schema",
    "returncode",
    "passed",
    "source_mode",
    "command",
    "stdout_file",
    "stderr_file",
    "runtime_manifest_file",
    "invocation_manifest_file",
    "stdout_bytes",
    "stdout_sha256",
    "stderr_bytes",
    "stderr_sha256",
    "runtime_manifest_bytes",
    "runtime_manifest_sha256",
    "invocation_manifest_bytes",
    "invocation_manifest_sha256",
    "bootstrap_sha256",
    "auditor_source_sha256",
    "support_files",
}
_AUTHORIZATION_REPRODUCTION_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "supplied_audit_sha256",
    "reproduced_audit_sha256",
    "byte_identical",
    "returncode",
    "source_mode",
    "stdout_bytes",
    "stderr_file",
    "stderr_bytes",
    "stderr_sha256",
    "runtime_manifest_file",
    "runtime_manifest_bytes",
    "runtime_manifest_sha256",
    "invocation_manifest_file",
    "invocation_manifest_bytes",
    "invocation_manifest_sha256",
    "bootstrap_sha256",
    "runner_source_sha256",
    "auditor_source_sha256",
    "support_files",
    "passed",
}


@dataclass(frozen=True)
class _AuthorizationAttempt:
    root: Path
    manifest: Mapping[str, object]
    terminal: Path
    terminal_payload: bytes
    terminal_identity: _StableFileIdentity
    member_rows: tuple[dict[str, object], ...]
    payloads: Mapping[str, bytes]
    completion: Path
    completion_row: Mapping[str, object]


def _authorization_directory(path: Path, *, label: str) -> None:
    """Require one absolute, lexical, symlink-free canonical directory."""

    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ArtifactAuditError(f"{label} cannot be resolved") from error
    if (
        not path.is_absolute()
        or Path(os.path.abspath(path)) != path
        or resolved != path
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ArtifactAuditError(f"{label} is not one canonical directory")


def _authorization_file_row(
    path: Path,
    *,
    label: str,
    expected_nlink: int,
    capture_payload: bool = True,
    limit: int = _AUTHORIZATION_MAX_FILE_BYTES,
) -> tuple[dict[str, object], bytes | None, _StableFileIdentity]:
    """Reopen one exact authorization input with typed hard-link custody."""

    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ArtifactAuditError(f"{label} path is not canonical and absolute")
    try:
        if path.resolve(strict=True) != path:
            raise ArtifactAuditError(f"{label} path is aliased")
    except OSError as error:
        raise ArtifactAuditError(f"{label} path cannot be resolved") from error
    payload, observed_bytes, digest, identity = _read_stable_regular_file(
        path,
        limit,
        label=label,
        capture_payload=capture_payload,
    )
    if identity[3] != expected_nlink:
        raise ArtifactAuditError(
            f"{label} must have exactly {expected_nlink} hard link(s)"
        )
    return (
        {
            "path": str(path),
            "bytes": observed_bytes,
            "sha256": digest,
        },
        payload,
        identity,
    )


def _authorization_completion(
    terminal: Path,
    *,
    terminal_payload: bytes,
    terminal_identity: _StableFileIdentity,
    label: str,
) -> tuple[Path, dict[str, object]]:
    marker = _OUTER_COMPLETIONS_ROOT / (
        hashlib.sha256(str(terminal).encode("utf-8")).hexdigest() + ".json"
    )
    marker_row, marker_payload, marker_identity = _authorization_file_row(
        marker,
        label=f"{label} outer completion",
        expected_nlink=2,
    )
    try:
        same_inode = os.path.samefile(terminal, marker)
    except OSError as error:
        raise ArtifactAuditError(
            f"{label} outer completion cannot be compared"
        ) from error
    if (
        marker_payload != terminal_payload
        or marker_identity != terminal_identity
        or not same_inode
    ):
        raise ArtifactAuditError(
            f"{label} outer completion is not the terminal-manifest inode"
        )
    return marker, marker_row


def _authorization_input_rows(
    value: object,
    *,
    label: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ArtifactAuditError(f"{label} inputs are not an array")
    rows: list[dict[str, object]] = []
    paths: list[str] = []
    for index, row in enumerate(value):
        if (
            not isinstance(row, dict)
            or set(row) != _AUTHORIZATION_INPUT_FIELDS
            or not isinstance(row.get("path"), str)
            or not Path(cast(str, row["path"])).is_absolute()
            or Path(os.path.abspath(cast(str, row["path"])))
            != Path(cast(str, row["path"]))
            or type(row.get("bytes")) is not int
            or cast(int, row["bytes"]) < 0
            or not _is_sha256(row.get("sha256"))
        ):
            raise ArtifactAuditError(
                f"{label} input identity {index} is malformed"
            )
        rows.append(row)
        paths.append(cast(str, row["path"]))
    if len(paths) != len(set(paths)):
        raise ArtifactAuditError(f"{label} input paths repeat")
    return rows


def _authorization_attempt(
    path: Path,
    *,
    kind: str,
    lane: str | None,
    primary: Mapping[str, object] | None,
    label: str,
) -> _AuthorizationAttempt:
    """Independently reopen an accepted, outer-finalized operator package."""

    _authorization_directory(path, label=label)
    terminal = path / _AUTHORIZATION_TERMINAL
    terminal_row, terminal_payload, terminal_identity = (
        _authorization_file_row(
            terminal,
            label=f"{label} terminal",
            expected_nlink=2,
        )
    )
    assert terminal_payload is not None
    manifest = _canonical_json_object_payload(
        terminal_payload,
        label=f"{label} terminal",
    )
    rows = manifest.get("files")
    if (
        set(manifest) != _AUTHORIZATION_ATTEMPT_FIELDS
        or manifest.get("schema") != "prospect.wm001.operator-attempt.v1"
        or manifest.get("experiment_id") != "WM-001"
        or manifest.get("protocol_version") != "1.16.0"
        or not _strict_json_equal(manifest.get("assurance"), _ASSURANCE)
        or manifest.get("kind") != kind
        or manifest.get("lane") != lane
        or manifest.get("status") != "accepted"
        or manifest.get("error") is not None
        or (
            primary is not None
            and not _strict_json_equal(manifest.get("primary"), primary)
        )
        or not isinstance(rows, list)
        or type(manifest.get("file_count")) is not int
        or manifest.get("file_count") != len(rows)
        or manifest.get("manifest_excludes")
        != [_AUTHORIZATION_TERMINAL]
    ):
        raise ArtifactAuditError(f"{label} terminal is not exactly accepted")
    _authorization_input_rows(manifest.get("inputs"), label=label)

    try:
        entries = sorted(os.scandir(path), key=lambda entry: entry.name)
    except OSError as error:
        raise ArtifactAuditError(f"{label} namespace cannot be scanned") from error
    if not entries or len(entries) > _AUTHORIZATION_MAX_FILES:
        raise ArtifactAuditError(f"{label} file count is outside its bound")
    member_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    for entry in entries:
        if (
            entry.name.startswith(".")
            or Path(entry.name).name != entry.name
            or entry.is_symlink()
            or not entry.is_file(follow_symlinks=False)
        ):
            raise ArtifactAuditError(
                f"{label} contains unsafe entry {entry.name!r}"
            )
        member_path = path / entry.name
        member_row, member_payload, _ = _authorization_file_row(
            member_path,
            label=f"{label} file {entry.name}",
            expected_nlink=(
                2 if entry.name == _AUTHORIZATION_TERMINAL else 1
            ),
        )
        assert member_payload is not None
        member_rows.append(member_row)
        payloads[entry.name] = member_payload
        if entry.name != _AUTHORIZATION_TERMINAL:
            manifest_rows.append(
                {
                    "path": entry.name,
                    "bytes": member_row["bytes"],
                    "sha256": member_row["sha256"],
                }
            )
    if (
        not _strict_json_equal(rows, manifest_rows)
        or terminal_row
        != next(
            (
                row
                for row in member_rows
                if row["path"] == str(terminal)
            ),
            None,
        )
    ):
        raise ArtifactAuditError(
            f"{label} files differ from its terminal manifest"
        )
    completion, completion_row = _authorization_completion(
        terminal,
        terminal_payload=terminal_payload,
        terminal_identity=terminal_identity,
        label=label,
    )
    return _AuthorizationAttempt(
        root=path,
        manifest=manifest,
        terminal=terminal,
        terminal_payload=terminal_payload,
        terminal_identity=terminal_identity,
        member_rows=tuple(member_rows),
        payloads=payloads,
        completion=completion,
        completion_row=completion_row,
    )


def _authorization_referenced_payload(
    attempt: _AuthorizationAttempt,
    reference: Mapping[str, object],
    *,
    prefix: str,
    label: str,
) -> bytes:
    filename = reference.get(f"{prefix}_file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename.startswith(".")
        or filename not in attempt.payloads
    ):
        raise ArtifactAuditError(f"{label} file reference is unsafe")
    payload = attempt.payloads[filename]
    if (
        type(reference.get(f"{prefix}_bytes")) is not int
        or reference.get(f"{prefix}_bytes") != len(payload)
        or reference.get(f"{prefix}_sha256")
        != hashlib.sha256(payload).hexdigest()
    ):
        raise ArtifactAuditError(f"{label} differs from its identity")
    return payload


def _authorization_development_producer(
    repository: Path,
) -> tuple[Path, list[dict[str, object]]]:
    producer = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "qualification-v1.16.0"
    )
    _authorization_directory(
        producer,
        label="canonical development producer",
    )
    manifest, manifest_digest = _verify_producer_manifest_locally(producer)
    if (
        manifest.get("lane") != "development"
        or manifest.get("status") != "completed"
        or manifest.get("error") is not None
    ):
        raise ArtifactAuditError(
            "canonical development producer is not one completed rehearsal"
        )
    manifest_row, _, _ = _authorization_file_row(
        producer / _PRODUCER_MANIFEST_NAME,
        label="canonical development producer manifest",
        expected_nlink=2,
        limit=_MAX_PRODUCER_MANIFEST_BYTES,
    )
    result_row, _, _ = _authorization_file_row(
        producer / "result.json",
        label="canonical development raw result",
        expected_nlink=1,
        capture_payload=False,
        limit=_MAX_PRODUCER_FILE_BYTES,
    )
    if manifest_row["sha256"] != manifest_digest:
        raise ArtifactAuditError(
            "canonical development producer manifest digest changed"
        )
    return producer, [manifest_row, result_row]


def _authorization_development_audit(
    repository: Path,
    *,
    producer: Path,
    producer_rows: list[dict[str, object]],
) -> _AuthorizationAttempt:
    audit_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.16"
        / "audits"
        / "development-audit-v1.16.0"
    )
    attempt = _authorization_attempt(
        audit_path,
        kind="audit",
        lane="development",
        primary=None,
        label="canonical development audit attempt",
    )
    primary = attempt.manifest.get("primary")
    if (
        not isinstance(primary, Mapping)
        or set(primary) != _AUTHORIZATION_AUDIT_PRIMARY_FIELDS
        or primary.get("producer_root") != str(producer)
        or primary.get("audit_file") != "independent-audit.json"
        or primary.get("executions")
        != [
            "audit-execution-01.execution.json",
            "audit-execution-02.execution.json",
        ]
        or primary.get("execution_failures") != []
        or primary.get("reproduction_file") != "audit-reproduction.json"
        or not isinstance(primary.get("reproduction_runtime_file"), str)
        or primary.get("claim_file") is not None
        or _authorization_input_rows(
            attempt.manifest.get("inputs"),
            label="canonical development audit attempt",
        )
        != producer_rows
    ):
        raise ArtifactAuditError(
            "canonical development audit authorization is not exact"
        )

    audit_payload = attempt.payloads.get("independent-audit.json")
    if audit_payload is None:
        raise ArtifactAuditError(
            "canonical development audit omits its independent report"
        )
    audit_report = _canonical_json_object_payload(
        audit_payload,
        label="canonical development independent audit",
    )
    execution_evidence: list[dict[str, bytes]] = []
    for filename in cast(list[object], primary["executions"]):
        if not isinstance(filename, str) or filename not in attempt.payloads:
            raise ArtifactAuditError(
                "canonical development audit execution reference is absent"
            )
        receipt = _canonical_json_object_payload(
            attempt.payloads[filename],
            label=f"canonical development audit {filename}",
        )
        command = receipt.get("command")
        if (
            set(receipt) != _AUTHORIZATION_EXECUTION_FIELDS
            or receipt.get("schema")
            != "prospect.wm001.captured-audit-execution.v1"
            or type(receipt.get("returncode")) is not int
            or receipt.get("returncode") != 0
            or receipt.get("passed") is not True
            or receipt.get("source_mode") != "descriptor"
            or not isinstance(command, list)
            or not command
            or any(not isinstance(argument, str) for argument in command)
            or not _is_sha256(receipt.get("bootstrap_sha256"))
            or not _is_sha256(receipt.get("auditor_source_sha256"))
            or not isinstance(receipt.get("support_files"), list)
        ):
            raise ArtifactAuditError(
                "canonical development audit execution receipt is malformed"
            )
        execution_evidence.append(
            {
                prefix: _authorization_referenced_payload(
                    attempt,
                    receipt,
                    prefix=prefix,
                    label=f"canonical development audit {filename} {prefix}",
                )
                for prefix in (
                    "stdout",
                    "stderr",
                    "runtime_manifest",
                    "invocation_manifest",
                )
            }
        )
    if (
        audit_report.get("passed") is not True
        or execution_evidence[0]["stdout"] != audit_payload
        or execution_evidence[1]["stdout"] != audit_payload
        or execution_evidence[1]["runtime_manifest"]
        != execution_evidence[0]["runtime_manifest"]
        or execution_evidence[1]["invocation_manifest"]
        != execution_evidence[0]["invocation_manifest"]
    ):
        raise ArtifactAuditError(
            "canonical development audit replay is not byte-identical"
        )

    reproduction_payload = attempt.payloads.get("audit-reproduction.json")
    if reproduction_payload is None:
        raise ArtifactAuditError(
            "canonical development audit omits its reproduction receipt"
        )
    reproduction = _canonical_json_object_payload(
        reproduction_payload,
        label="canonical development audit reproduction",
    )
    second_receipt = _canonical_json_object_payload(
        attempt.payloads["audit-execution-02.execution.json"],
        label="canonical development second execution",
    )
    audit_digest = hashlib.sha256(audit_payload).hexdigest()
    if (
        set(reproduction) != _AUTHORIZATION_REPRODUCTION_FIELDS
        or reproduction.get("schema")
        != "prospect.wm001.audit-reproduction.v2"
        or reproduction.get("experiment_id") != "WM-001"
        or reproduction.get("protocol_version") != "1.16.0"
        or reproduction.get("supplied_audit_sha256") != audit_digest
        or reproduction.get("reproduced_audit_sha256") != audit_digest
        or reproduction.get("byte_identical") is not True
        or type(reproduction.get("returncode")) is not int
        or reproduction.get("returncode") != 0
        or reproduction.get("source_mode") != "descriptor"
        or type(reproduction.get("stdout_bytes")) is not int
        or reproduction.get("stdout_bytes") != len(audit_payload)
        or reproduction.get("passed") is not True
        or primary.get("reproduction_runtime_file")
        != reproduction.get("runtime_manifest_file")
        or reproduction.get("bootstrap_sha256")
        != second_receipt.get("bootstrap_sha256")
        or reproduction.get("auditor_source_sha256")
        != second_receipt.get("auditor_source_sha256")
        or not _is_sha256(reproduction.get("runner_source_sha256"))
        or not _strict_json_equal(
            reproduction.get("support_files"),
            second_receipt.get("support_files"),
        )
        or _authorization_referenced_payload(
            attempt,
            reproduction,
            prefix="stderr",
            label="canonical development reproduction stderr",
        )
        != execution_evidence[1]["stderr"]
        or _authorization_referenced_payload(
            attempt,
            reproduction,
            prefix="runtime_manifest",
            label="canonical development reproduction runtime",
        )
        != execution_evidence[1]["runtime_manifest"]
        or _authorization_referenced_payload(
            attempt,
            reproduction,
            prefix="invocation_manifest",
            label="canonical development reproduction invocation",
        )
        != execution_evidence[1]["invocation_manifest"]
    ):
        raise ArtifactAuditError(
            "canonical development audit reproduction is malformed"
        )
    return attempt


def _authorization_development_closure(
    repository: Path,
    *,
    binding: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    _AuthorizationAttempt,
]:
    results = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
    )
    closure_path = (
        results
        / "development"
        / "development-closure-v1.16.0.json"
    )
    closure_row, closure_payload, _ = _authorization_file_row(
        closure_path,
        label="canonical development closure",
        expected_nlink=1,
    )
    assert closure_payload is not None
    closure = _canonical_json_object_payload(
        closure_payload,
        label="canonical development closure",
    )
    development = binding.get("development_qualification")
    producer, producer_rows = _authorization_development_producer(
        repository
    )
    if (
        set(closure) != _AUTHORIZATION_CLOSURE_FIELDS
        or closure.get("schema")
        != "prospect.wm001.development-closure.v2"
        or closure.get("experiment_id") != "WM-001"
        or closure.get("protocol_version") != "1.16.0"
        or closure.get("producer_root") != str(producer)
        or closure.get("engineering_verified") is not True
        or closure.get("audit_reproduced") is not True
        or closure.get("performance_values_bound") is not False
        or not isinstance(development, Mapping)
        or not _strict_json_equal(
            development.get("closure_bytes"),
            closure_row["bytes"],
        )
        or development.get("closure_sha256") != closure_row["sha256"]
    ):
        raise ArtifactAuditError(
            "canonical development closure differs from the formal binding"
        )
    development_audit = _authorization_development_audit(
        repository,
        producer=producer,
        producer_rows=producer_rows,
    )
    closure_attempt_path = (
        results
        / "operator-v1.16"
        / "closures"
        / "development-closure-v1.16.0"
    )
    closure_attempt = _authorization_attempt(
        closure_attempt_path,
        kind="closure",
        lane="development",
        primary={"closure_reference_file": "closure-reference.json"},
        label="canonical development closure attempt",
    )
    expected_closure_inputs = [
        *producer_rows,
        *development_audit.member_rows,
        dict(development_audit.completion_row),
    ]
    if (
        _authorization_input_rows(
            closure_attempt.manifest.get("inputs"),
            label="canonical development closure attempt",
        )
        != expected_closure_inputs
    ):
        raise ArtifactAuditError(
            "canonical development closure inputs differ from the exact "
            "producer/audit evidence"
        )
    reference_payload = closure_attempt.payloads.get(
        "closure-reference.json"
    )
    if reference_payload is None:
        raise ArtifactAuditError(
            "canonical development closure reference is absent"
        )
    reference = _canonical_json_object_payload(
        reference_payload,
        label="canonical development closure reference",
    )
    expected_reference_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "closure_marker",
        "closure_sha256",
        "qualification_archive",
        "producer_root",
        "audit_attempt",
        "audit_attempt_manifest_sha256",
        "fresh_reopen_file",
        "fresh_reopen_sha256",
    }
    development_audit_terminal = next(
        row
        for row in development_audit.member_rows
        if row["path"] == str(development_audit.terminal)
    )
    if (
        set(reference) != expected_reference_fields
        or reference.get("schema")
        != "prospect.wm001.closure-reference.v1"
        or reference.get("experiment_id") != "WM-001"
        or reference.get("protocol_version") != "1.16.0"
        or reference.get("closure_marker") != str(closure_path)
        or reference.get("closure_sha256") != closure_row["sha256"]
        or not _strict_json_equal(
            reference.get("qualification_archive"),
            closure.get("qualification_archive"),
        )
        or reference.get("producer_root") != str(producer)
        or reference.get("audit_attempt")
        != str(development_audit.root)
        or reference.get("audit_attempt_manifest_sha256")
        != development_audit_terminal["sha256"]
        or reference.get("fresh_reopen_file")
        != "fresh-runtime-reopen.json"
        or not _is_sha256(reference.get("fresh_reopen_sha256"))
    ):
        raise ArtifactAuditError(
            "canonical development closure reference differs from live "
            "producer/audit evidence"
        )
    fresh_payload = closure_attempt.payloads.get(
        "fresh-runtime-reopen.json"
    )
    if fresh_payload is None:
        raise ArtifactAuditError(
            "canonical fresh closure-reopen report is absent"
        )
    fresh = _canonical_json_object_payload(
        fresh_payload,
        label="canonical fresh closure-reopen report",
    )
    archive = closure.get("qualification_archive")
    members = archive.get("members") if isinstance(archive, Mapping) else None
    member_digests = (
        {
            row.get("path"): row.get("sha256")
            for row in members
            if isinstance(row, Mapping)
        }
        if isinstance(members, list)
        else {}
    )
    producer_digest_by_path = {
        cast(str, row["path"]): cast(str, row["sha256"])
        for row in producer_rows
    }

    def archived_role_digest(field: str) -> object:
        return member_digests.get(closure.get(field))

    def audit_role_payload(field: str) -> bytes | None:
        member = closure.get(field)
        if (
            not isinstance(member, str)
            or not member.startswith("evidence/")
        ):
            return None
        return development_audit.payloads.get(
            member.removeprefix("evidence/")
        )

    audit_payload = development_audit.payloads.get("independent-audit.json")
    reproduction_payload = development_audit.payloads.get(
        "audit-reproduction.json"
    )
    runtime_payload = audit_role_payload("audit_runtime_manifest_member")
    invocation_payload = audit_role_payload(
        "audit_invocation_manifest_member"
    )
    stderr_payload = audit_role_payload("audit_stderr_member")
    live_role_digests = {
        "producer_manifest_member": producer_digest_by_path.get(
            str(producer / _PRODUCER_MANIFEST_NAME)
        ),
        "raw_result_member": producer_digest_by_path.get(
            str(producer / "result.json")
        ),
        "independent_audit_member": (
            hashlib.sha256(audit_payload).hexdigest()
            if audit_payload is not None
            else None
        ),
        "audit_reproduction_member": (
            hashlib.sha256(reproduction_payload).hexdigest()
            if reproduction_payload is not None
            else None
        ),
        "audit_runtime_manifest_member": (
            hashlib.sha256(runtime_payload).hexdigest()
            if runtime_payload is not None
            else None
        ),
        "audit_invocation_manifest_member": (
            hashlib.sha256(invocation_payload).hexdigest()
            if invocation_payload is not None
            else None
        ),
        "audit_stderr_member": (
            hashlib.sha256(stderr_payload).hexdigest()
            if stderr_payload is not None
            else None
        ),
    }
    if any(
        digest is None or archived_role_digest(field) != digest
        for field, digest in live_role_digests.items()
    ):
        raise ArtifactAuditError(
            "development archive roles differ from live closure inputs"
        )
    if (
        set(fresh)
        != {
            "schema",
            "experiment_id",
            "protocol_version",
            "mode",
            "challenge",
            "requesting_process_id",
            "verifier_process_id",
            "matrix_contract_sha256",
            "development_closure_sha256",
            "producer_manifest_sha256",
            "raw_result_sha256",
            "passed",
        }
        or fresh.get("schema")
        != "prospect.wm001.development-closure-fresh-reopen.v1"
        or fresh.get("experiment_id") != "WM-001"
        or fresh.get("protocol_version") != "1.16.0"
        or fresh.get("mode") != "fresh-closure-reopen"
        or not _is_sha256(fresh.get("challenge"))
        or type(fresh.get("requesting_process_id")) is not int
        or cast(int, fresh["requesting_process_id"]) <= 0
        or type(fresh.get("verifier_process_id")) is not int
        or cast(int, fresh["verifier_process_id"]) <= 0
        or fresh.get("requesting_process_id")
        == fresh.get("verifier_process_id")
        or fresh.get("matrix_contract_sha256")
        != _DEVELOPMENT_MATRIX_CONTRACT_SHA256
        or fresh.get("development_closure_sha256")
        != closure_row["sha256"]
        or fresh.get("producer_manifest_sha256")
        != member_digests.get("producer/producer-manifest.json")
        or fresh.get("raw_result_sha256")
        != member_digests.get("producer/result.json")
        or fresh.get("passed") is not True
        or hashlib.sha256(fresh_payload).hexdigest()
        != reference.get("fresh_reopen_sha256")
    ):
        raise ArtifactAuditError(
            "canonical fresh closure-reopen report differs from closure "
            "evidence"
        )
    return (
        [
            closure_row,
            next(
                row
                for row in closure_attempt.member_rows
                if row["path"] == str(closure_attempt.terminal)
            ),
            dict(closure_attempt.completion_row),
        ],
        closure_attempt,
    )


def _authorization_preformal_rows(
    repository: Path,
    *,
    artifact_root: Path,
    binding: Mapping[str, object],
) -> list[dict[str, object]]:
    development_root = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
    )
    preformal_root = development_root / _PREFORMAL_LIVE_RELATIVE_DIRECTORY
    report_path = preformal_root / _PREFORMAL_REPORT_NAME
    report_row, report_payload, _ = _authorization_file_row(
        report_path,
        label="canonical preformal report",
        expected_nlink=1,
    )
    assert report_payload is not None
    report = _canonical_json_object_payload(
        report_payload,
        label="canonical preformal report",
    )
    commands = report.get("commands")
    source = binding.get("source")
    if (
        report.get("schema")
        != "prospect.wm001.preformal-test-report.v2"
        or report.get("experiment_id") != "WM-001"
        or report.get("protocol_version") != "1.16.0"
        or report.get("all_pass") is not True
        or not isinstance(commands, list)
        or len(commands) != len(_PREFORMAL_COMMAND_NAMES)
        or not isinstance(source, Mapping)
    ):
        raise ArtifactAuditError(
            "canonical preformal report is not an accepted v1.16 report"
        )
    log_rows: list[dict[str, object]] = []
    relative_log_rows: list[dict[str, object]] = []
    for ordinal, (command, command_name) in enumerate(
        zip(commands, _PREFORMAL_COMMAND_NAMES, strict=True),
        start=1,
    ):
        if (
            not isinstance(command, Mapping)
            or type(command.get("ordinal")) is not int
            or command.get("ordinal") != ordinal
            or command.get("name") != command_name
            or command.get("passed") is not True
            or type(command.get("exit_code")) is not int
            or command.get("exit_code") != 0
        ):
            raise ArtifactAuditError(
                "canonical preformal commands are not exact and ordered"
            )
        for stream in ("stdout", "stderr"):
            relative = _preformal_log_reference(
                command.get(stream),
                ordinal=ordinal,
                command_name=command_name,
                stream=stream,
            )
            log_path = preformal_root / cast(str, relative["path"])
            live_row, live_payload, _ = _authorization_file_row(
                log_path,
                label=f"canonical preformal {command_name} {stream}",
                expected_nlink=1,
            )
            assert live_payload is not None
            if (
                live_row["bytes"] != relative["bytes"]
                or live_row["sha256"] != relative["sha256"]
            ):
                raise ArtifactAuditError(
                    "canonical preformal log differs from its report"
                )
            copied_log = artifact_root / cast(str, relative["path"])
            _, copied_payload, _ = _authorization_file_row(
                copied_log,
                label=f"copied preformal {command_name} {stream}",
                expected_nlink=1,
            )
            if copied_payload != live_payload:
                raise ArtifactAuditError(
                    "copied preformal log differs from its live "
                    "authorization input"
                )
            log_rows.append(live_row)
            relative_log_rows.append(relative)
    _, copied_report_payload, _ = _authorization_file_row(
        artifact_root / _PREFORMAL_REPORT_NAME,
        label="copied canonical preformal report",
        expected_nlink=1,
    )
    if (
        copied_report_payload != report_payload
        or source.get("test_report_file") != _PREFORMAL_REPORT_NAME
        or not _strict_json_equal(
            source.get("test_report_bytes"),
            report_row["bytes"],
        )
        or source.get("test_report_sha256") != report_row["sha256"]
        or not _strict_json_equal(
            source.get("test_log_files"),
            relative_log_rows,
        )
    ):
        raise ArtifactAuditError(
            "formal binding preformal projection differs from exact live "
            "evidence"
        )
    expected_live_names = {
        _PREFORMAL_REPORT_NAME,
        *(cast(str, row["path"]) for row in relative_log_rows),
    }
    if {path.name for path in preformal_root.iterdir()} != expected_live_names:
        raise ArtifactAuditError(
            "canonical preformal bundle has missing or extra members"
        )
    return [report_row, *log_rows]


def _validate_formal_authorization_lineage(
    *,
    repository: Path,
    artifact_root: Path,
    binding: Mapping[str, object],
    binding_payload: bytes,
) -> _AuthorizationAttempt:
    """Reconstruct every live input that authorized the formal binding."""

    _authorization_directory(repository, label="formal audit repository")
    binding_attempt_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.16"
        / "bindings"
        / "formal-binding-v1.16.0"
    )
    binding_attempt = _authorization_attempt(
        binding_attempt_path,
        kind="binding",
        lane=None,
        primary={"binding_file": "formal-binding.json"},
        label="canonical formal binding attempt",
    )
    if (
        binding_attempt.payloads.get("formal-binding.json")
        != binding_payload
    ):
        raise ArtifactAuditError(
            "canonical formal binding attempt contains different binding "
            "bytes"
        )
    preflight_payload = binding_attempt.payloads.get(
        _FORMAL_INPUT_PREFLIGHT_NAME
    )
    if preflight_payload is None:
        raise ArtifactAuditError(
            "canonical formal binding attempt omits its independent preflight"
        )
    _formal_input_preflight_receipt(
        preflight_payload,
        binding_payload=binding_payload,
    )
    _, copied_preflight_payload, _ = _authorization_file_row(
        artifact_root / _FORMAL_INPUT_PREFLIGHT_NAME,
        label="copied formal input preflight receipt",
        expected_nlink=1,
    )
    if copied_preflight_payload != preflight_payload:
        raise ArtifactAuditError(
            "copied formal input preflight differs from the terminal-bound "
            "binding receipt"
        )
    development = binding.get("development_qualification")
    live_result_qualification = binding_attempt.payloads.get(
        _DEVELOPMENT_RESULT_QUALIFICATION_NAME
    )
    if (
        not isinstance(development, Mapping)
        or live_result_qualification is None
        or hashlib.sha256(live_result_qualification).hexdigest()
        != development.get("result_qualification_sha256")
    ):
        raise ArtifactAuditError(
            "terminal-bound development result qualification differs "
            "from the formal binding"
        )
    _, copied_result_qualification, _ = _authorization_file_row(
        artifact_root / _DEVELOPMENT_RESULT_QUALIFICATION_NAME,
        label="copied development result qualification",
        expected_nlink=1,
    )
    if copied_result_qualification != live_result_qualification:
        raise ArtifactAuditError(
            "copied development result qualification differs from the "
            "terminal-bound sidecar"
        )
    preformal_rows = _authorization_preformal_rows(
        repository,
        artifact_root=artifact_root,
        binding=binding,
    )
    closure_rows, _ = _authorization_development_closure(
        repository,
        binding=binding,
    )
    expected_inputs = [*preformal_rows, *closure_rows]
    observed_inputs = _authorization_input_rows(
        binding_attempt.manifest.get("inputs"),
        label="canonical formal binding attempt",
    )
    if observed_inputs != expected_inputs:
        raise ArtifactAuditError(
            "formal binding authorization inputs differ from exact live "
            "preformal/closure evidence"
        )
    return binding_attempt


def _formal_launch_namespace_is_canonical(
    root: Path,
    *,
    binding_digest: str,
    launch: Mapping[str, object],
) -> bool:
    """Require the one exact v1.16 formal producer namespace."""

    expected_root = (
        Path.cwd()
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "formal"
        / binding_digest
        / _FORMAL_CONFIRMATION_NAME
    )
    return (
        root == expected_root
        and launch.get("attempt_directory") == _FORMAL_CONFIRMATION_NAME
    )


def _audit_formal_input_package(
    audit: _Audit,
    root: Path,
    result: Mapping[str, object],
    *,
    producer_bootstrap_sha256: str,
) -> str | None:
    if result.get("lane") != "formal":
        return None
    bound_device: str | None = None
    fixed_files = {
        "protocol.json": root / "protocol.json",
        "schemas/formal-binding.schema.json": root / "schemas" / "formal-binding.schema.json",
        "schemas/raw-result.schema.json": root / "schemas" / "raw-result.schema.json",
    }
    binding_path = root / "formal-binding.json"
    launch_path = root / "formal-launch.json"
    binding_attempt_copy = root / "formal-binding-operator-attempt.json"
    binding_completion_copy = root / "formal-binding-outer-completion.json"
    result_qualification_copy = (
        root / _DEVELOPMENT_RESULT_QUALIFICATION_NAME
    )
    lock_path = root / "requirements-wm001.lock"
    seal_path = root / "SEALED_PROTOCOL.sha256"
    required = [
        binding_path,
        launch_path,
        binding_attempt_copy,
        binding_completion_copy,
        result_qualification_copy,
        lock_path,
        seal_path,
        *fixed_files.values(),
    ]
    try:
        root_resolved = root.resolve()
        if any(
            path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root_resolved)
            for path in required
        ):
            raise ArtifactAuditError("formal root is missing a required fixed input file")
        payloads = {relative: _read_bounded(path, 64 << 20, label=relative) for relative, path in fixed_files.items()}
        binding_payload = _read_bounded(
            binding_path,
            64 << 20,
            label="formal binding",
        )
        binding_raw = _json_without_duplicate_keys(
            binding_payload,
            label="formal binding",
        )
        if not isinstance(binding_raw, dict):
            raise ArtifactAuditError("formal binding root is not an object")
        binding: Mapping[str, object] = binding_raw
        binding_digest = hashlib.sha256(binding_payload).hexdigest()
        audit.require(
            binding.get("schema") == "prospect.world-model-lifecycle.formal-binding.v10"
            and binding.get("experiment_id") == "WM-001",
            code="formal_binding_identity_mismatch",
            message="formal binding is not the active WM-001 v1.16 identity",
        )
        audit.require(
            _strict_json_equal(binding.get("assurance"), _ASSURANCE),
            code="formal_binding_assurance_mismatch",
            message=("formal binding omits or overstates the fixed WM-001 trust boundary"),
        )
        audit.require(
            result.get("formal_binding_sha256") == binding_digest,
            code="formal_binding_result_digest_mismatch",
            message="result does not bind the copied pre-outcome formal binding",
        )
        launch_payload = _read_bounded(
            launch_path,
            1 << 20,
            label="formal launch record",
        )
        launch_raw = _json_without_duplicate_keys(
            launch_payload,
            label="formal launch record",
        )
        if not isinstance(launch_raw, dict):
            raise ArtifactAuditError("formal launch record is not an object")
        launch_body = dict(launch_raw)
        launch_record_sha256 = launch_body.pop("record_sha256", None)
        execution_for_launch = result.get("execution")
        protocol_wide_marker = root.parent.parent / "formal-launch-v1.16.0.json"
        repository = Path.cwd()
        binding_attempt = (
            repository
            / "bench"
            / "world_model_lifecycle"
            / "results"
            / "operator-v1.16"
            / "bindings"
            / "formal-binding-v1.16.0"
        )
        binding_attempt_terminal = binding_attempt / "operator-attempt.json"
        binding_attempt_completion = (
            repository
            / "bench"
            / "world_model_lifecycle"
            / "results"
            / "outer-completions"
            / "v1.16"
            / (hashlib.sha256(str(binding_attempt_terminal).encode("utf-8")).hexdigest() + ".json")
        )
        binding_attempt_payload = _read_bounded(
            binding_attempt_copy,
            16 << 20,
            label="copied formal binding attempt manifest",
        )
        binding_completion_payload = _read_bounded(
            binding_completion_copy,
            16 << 20,
            label="copied formal binding outer completion",
        )
        binding_attempt_raw = _json_without_duplicate_keys(
            binding_attempt_payload,
            label="copied formal binding attempt manifest",
        )
        binding_attempt_primary = binding_attempt_raw.get("primary") if isinstance(binding_attempt_raw, dict) else None
        binding_attempt_rows = binding_attempt_raw.get("files") if isinstance(binding_attempt_raw, dict) else None
        binding_row = (
            next(
                (
                    row
                    for row in binding_attempt_rows
                    if isinstance(row, Mapping) and row.get("path") == "formal-binding.json"
                ),
                None,
            )
            if isinstance(binding_attempt_rows, list)
            else None
        )
        canonical_binding_attempt_payload = _read_bounded(
            binding_attempt_terminal,
            16 << 20,
            label="canonical formal binding attempt manifest",
        )
        canonical_binding_completion_payload = _read_bounded(
            binding_attempt_completion,
            16 << 20,
            label="canonical formal binding outer completion",
        )
        binding_attempt_sha256 = hashlib.sha256(binding_attempt_payload).hexdigest()
        authorization_attempt = _validate_formal_authorization_lineage(
            repository=repository,
            artifact_root=root,
            binding=binding,
            binding_payload=binding_payload,
        )
        formal_input_preflight_receipt = (
            _formal_input_preflight_receipt(
                authorization_attempt.payloads[
                    _FORMAL_INPUT_PREFLIGHT_NAME
                ],
                binding_payload=binding_payload,
            )
        )
        if authorization_attempt.terminal_payload != binding_attempt_payload:
            raise ArtifactAuditError(
                "copied formal binding attempt differs from the independently "
                "reconstructed authorization terminal"
            )
        audit.require(
            _formal_launch_namespace_is_canonical(
                root,
                binding_digest=binding_digest,
                launch=launch_raw,
            )
            and set(launch_raw)
            == {
                "schema",
                "experiment_id",
                "protocol_version",
                "formal_binding_sha256",
                "formal_binding_attempt_path",
                "formal_binding_attempt_manifest_file",
                "formal_binding_attempt_manifest_sha256",
                "formal_binding_outer_completion_file",
                "formal_binding_outer_completion_marker",
                "formal_binding_outer_completion_sha256",
                "attempt_directory",
                "global_marker_file",
                "claimed_at_utc",
                "git_commit",
                "git_tree",
                "record_sha256",
            }
            and not protocol_wide_marker.is_symlink()
            and protocol_wide_marker.is_file()
            and not launch_path.is_symlink()
            and launch_path.is_file()
            and os.path.samefile(protocol_wide_marker, launch_path)
            and _read_bounded(
                protocol_wide_marker,
                1 << 20,
                label="protocol-wide formal launch marker",
            )
            == launch_payload
            and launch_raw.get("schema") == "prospect.wm001.formal-launch.v2"
            and launch_raw.get("experiment_id") == "WM-001"
            and launch_raw.get("protocol_version") == "1.16.0"
            and launch_raw.get("formal_binding_sha256") == binding_digest
            and launch_raw.get("formal_binding_attempt_path") == str(binding_attempt)
            and launch_raw.get("formal_binding_attempt_manifest_file") == "formal-binding-operator-attempt.json"
            and launch_raw.get("formal_binding_attempt_manifest_sha256") == binding_attempt_sha256
            and launch_raw.get("formal_binding_outer_completion_file") == "formal-binding-outer-completion.json"
            and launch_raw.get("formal_binding_outer_completion_marker") == str(binding_attempt_completion)
            and launch_raw.get("formal_binding_outer_completion_sha256") == binding_attempt_sha256
            and binding_attempt_payload
            == binding_completion_payload
            == canonical_binding_attempt_payload
            == canonical_binding_completion_payload
            and binding_attempt_terminal.lstat().st_nlink == 2
            and binding_attempt_completion.lstat().st_nlink == 2
            and os.path.samefile(
                binding_attempt_terminal,
                binding_attempt_completion,
            )
            and isinstance(binding_attempt_raw, dict)
            and binding_attempt_payload == _canonical_json_bytes(binding_attempt_raw) + b"\n"
            and binding_attempt_raw.get("schema") == "prospect.wm001.operator-attempt.v1"
            and binding_attempt_raw.get("experiment_id") == "WM-001"
            and binding_attempt_raw.get("protocol_version") == "1.16.0"
            and _strict_json_equal(
                binding_attempt_raw.get("assurance"),
                _ASSURANCE,
            )
            and binding_attempt_raw.get("kind") == "binding"
            and binding_attempt_raw.get("lane") is None
            and binding_attempt_raw.get("status") == "accepted"
            and isinstance(binding_attempt_primary, Mapping)
            and binding_attempt_primary.get("binding_file") == "formal-binding.json"
            and isinstance(binding_row, Mapping)
            and type(binding_row.get("bytes")) is int
            and binding_row.get("bytes") == len(binding_payload)
            and binding_row.get("sha256") == binding_digest
            and (binding_attempt / "formal-binding.json").read_bytes() == binding_payload
            and launch_raw.get("global_marker_file") == "formal-launch-v1.16.0.json"
            and isinstance(execution_for_launch, Mapping)
            and launch_raw.get("git_commit") == execution_for_launch.get("git_commit")
            and launch_raw.get("git_tree") == execution_for_launch.get("git_tree")
            and launch_record_sha256 == hashlib.sha256(_canonical_json_bytes(launch_body)).hexdigest()
            and launch_payload == _canonical_json_bytes(launch_raw) + b"\n"
            and execution_for_launch.get("formal_launch_file") == "formal-launch.json"
            and execution_for_launch.get("formal_launch_sha256") == hashlib.sha256(launch_payload).hexdigest(),
            code="formal_single_launch_binding_mismatch",
            message=("formal result does not bind the same-inode, version-scoped protocol-wide v1.16 launch claim"),
        )
        seal_payload = _read_bounded(
            seal_path,
            1 << 20,
            label="protocol seal",
        )
        try:
            seal_rows = [line.split() for line in seal_payload.decode("utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError as error:
            raise ArtifactAuditError("protocol seal is not UTF-8") from error
        valid_seal = (
            all(len(row) == 2 for row in seal_rows)
            and {row[1] for row in seal_rows} == set(fixed_files)
            and all(
                hashlib.sha256(payloads[relative]).hexdigest()
                == next(row[0] for row in seal_rows if row[1] == relative)
                for relative in fixed_files
            )
        )
        audit.require(
            valid_seal,
            code="formal_protocol_seal_mismatch",
            message="copied protocol/schema bytes do not match the copied pre-outcome seal",
        )
        protocol_block = binding.get("protocol")
        dependencies = binding.get("dependencies")
        source = binding.get("source")
        runtime = binding.get("runtime")
        environment = binding.get("environment")
        irrelevant_control = binding.get("irrelevant_control")
        coverage_arithmetic = binding.get("coverage_arithmetic")
        audit_execution = binding.get("audit_execution")
        development_qualification = binding.get("development_qualification")
        execution = result.get("execution")
        if not all(
            isinstance(value, Mapping)
            for value in (
                protocol_block,
                dependencies,
                source,
                runtime,
                environment,
                irrelevant_control,
                coverage_arithmetic,
                audit_execution,
                development_qualification,
                execution,
            )
        ):
            raise ArtifactAuditError("formal binding/result blocks are malformed")
        assert isinstance(protocol_block, Mapping)
        assert isinstance(dependencies, Mapping)
        assert isinstance(source, Mapping)
        assert isinstance(runtime, Mapping)
        assert isinstance(environment, Mapping)
        assert isinstance(irrelevant_control, Mapping)
        assert isinstance(coverage_arithmetic, Mapping)
        assert isinstance(audit_execution, Mapping)
        assert isinstance(development_qualification, Mapping)
        assert isinstance(execution, Mapping)
        audit.formal_runtime_binding = runtime
        audit.formal_dependency_binding = dependencies
        audit.formal_source_binding = source
        lock_payload = _read_bounded(
            lock_path,
            64 << 20,
            label="formal dependency lock",
        )
        lock_digest = hashlib.sha256(lock_payload).hexdigest()
        audit.require(
            protocol_block.get("version") == result.get("protocol_version")
            and protocol_block.get("sha256")
            == hashlib.sha256(payloads["protocol.json"]).hexdigest()
            == result.get("protocol_sha256")
            and protocol_block.get("raw_result_schema_sha256")
            == hashlib.sha256(payloads["schemas/raw-result.schema.json"]).hexdigest()
            and protocol_block.get("binding_schema_sha256")
            == hashlib.sha256(payloads["schemas/formal-binding.schema.json"]).hexdigest(),
            code="formal_protocol_binding_mismatch",
            message="formal binding/result do not agree with copied protocol/schema bytes",
        )
        audit.require(
            dependencies.get("lockfile_sha256") == lock_digest
            and execution.get("dependency_lock_sha256") == lock_digest,
            code="formal_lockfile_binding_mismatch",
            message="binding/result do not agree with the copied dependency lock",
        )
        audit.require(
            source.get("git_commit") == execution.get("git_commit")
            and source.get("git_tree") == execution.get("git_tree")
            and source.get("worktree_clean") is True
            and execution.get("worktree_clean") is True,
            code="formal_source_binding_mismatch",
            message="formal result source identity differs from its pre-run binding",
        )
        execution_sources = _validate_bound_execution_source_manifest(
            root,
            source,
        )
        audit.require(
            isinstance(execution_sources, Mapping)
            and execution.get("runtime_seal_sha256") == binding_digest
            and execution.get("runtime_seal_descriptor_custody") is True
            and execution.get("producer_bootstrap_sha256") == execution_sources.get("producer_bootstrap.py")
            and execution_sources.get("producer_bootstrap.py")
            == producer_bootstrap_sha256
            and execution.get("bootstrap_descriptor_custody") is True
            and execution.get("package_roots") == dependencies.get("package_roots")
            and execution.get("package_ownership") == dependencies.get("package_ownership")
            and execution.get("standard_library") == dependencies.get("standard_library")
            and execution.get("python_executable") == dependencies.get("python_executable")
            and execution.get("python_executable_sha256") == dependencies.get("python_executable_sha256"),
            code="formal_preimport_runtime_custody_mismatch",
            message=(
                "formal result does not exactly bind the pre-import seal, "
                "bootstrap descriptor, package roots, and standard library"
            ),
        )
        bound_device = _audit_formal_runtime_binding(
            audit,
            runtime=runtime,
            dependencies=dependencies,
            execution=execution,
        )
        _audit_bound_source_snapshot(audit, root, source)
        _audit_bound_irrelevant_control(
            audit,
            root,
            irrelevant_control,
        )
        auditor_snapshot = root / "source" / "bench" / "world_model_lifecycle" / "artifact_audit.py"
        model_snapshot = root / "source" / "bench" / "world_model_lifecycle" / "model.py"
        audit.require(
            coverage_arithmetic.get("semantics_id") == _COVERAGE_SEMANTICS
            and coverage_arithmetic.get("python_executable") == sys.executable
            and coverage_arithmetic.get("python_implementation") == platform.python_implementation() == "CPython"
            and coverage_arithmetic.get("python_version") == platform.python_version()
            and coverage_arithmetic.get("platform") == platform.platform()
            and coverage_arithmetic.get("machine") == platform.machine()
            and coverage_arithmetic.get("producer_source_sha256")
            == hashlib.sha256(
                _read_bounded(
                    model_snapshot,
                    _MAX_SOURCE_FILE_BYTES,
                    label="bound model source",
                )
            ).hexdigest()
            and coverage_arithmetic.get("auditor_source_sha256")
            == hashlib.sha256(
                _read_bounded(auditor_snapshot, _MAX_SOURCE_FILE_BYTES, label="bound auditor source")
            ).hexdigest()
            == _AUDITOR_SOURCE_SHA256
            and coverage_arithmetic.get("formal_test_report_sha256") == source.get("test_report_sha256"),
            code="formal_coverage_binding_mismatch",
            message="coverage arithmetic runtime, source, or test-report binding changed",
        )
        audit.require(
            result.get("claim_eligible") is True,
            code="formal_claim_eligibility_mismatch",
            message="formal result is not marked claim eligible",
        )
        for block, filename_field, bytes_field, digest_field, label in (
            (
                source,
                "test_report_file",
                "test_report_bytes",
                "test_report_sha256",
                "formal test report",
            ),
            (
                environment,
                "conformance_report_file",
                "conformance_report_bytes",
                "conformance_report_sha256",
                "Pendulum conformance report",
            ),
            (
                coverage_arithmetic,
                "conformance_report_file",
                "conformance_report_bytes",
                "conformance_report_sha256",
                "coverage conformance report",
            ),
            (
                development_qualification,
                "closure_file",
                "closure_bytes",
                "closure_sha256",
                "development qualification closure",
            ),
            (
                audit_execution,
                "bootstrap_source_file",
                "bootstrap_source_bytes",
                "bootstrap_source_sha256",
                "audit bootstrap source",
            ),
            (
                audit_execution,
                "prebinding_request_file",
                "prebinding_request_bytes",
                "prebinding_request_sha256",
                "prebinding conformance request",
            ),
            (
                audit_execution,
                "prebinding_path_runtime_manifest_file",
                "prebinding_path_runtime_manifest_bytes",
                "prebinding_path_runtime_manifest_sha256",
                "prebinding path runtime manifest",
            ),
            (
                audit_execution,
                "prebinding_descriptor_runtime_manifest_file",
                "prebinding_descriptor_runtime_manifest_bytes",
                "prebinding_descriptor_runtime_manifest_sha256",
                "prebinding descriptor runtime manifest",
            ),
            (
                audit_execution,
                "prebinding_path_invocation_manifest_file",
                "prebinding_path_invocation_manifest_bytes",
                "prebinding_path_invocation_manifest_sha256",
                "prebinding path invocation manifest",
            ),
            (
                audit_execution,
                "prebinding_descriptor_invocation_manifest_file",
                "prebinding_descriptor_invocation_manifest_bytes",
                "prebinding_descriptor_invocation_manifest_sha256",
                "prebinding descriptor invocation manifest",
            ),
            (
                audit_execution,
                "prebinding_conformance_report_file",
                "prebinding_conformance_report_bytes",
                "prebinding_conformance_report_sha256",
                "prebinding conformance report",
            ),
            (
                audit_execution,
                "prebinding_execution_receipt_file",
                "prebinding_execution_receipt_bytes",
                "prebinding_execution_receipt_sha256",
                "prebinding execution receipt",
            ),
            (
                audit_execution,
                "outcome_runtime_manifest_file",
                "outcome_runtime_manifest_bytes",
                "outcome_runtime_manifest_sha256",
                "outcome descriptor runtime manifest",
            ),
            (
                audit_execution,
                "restart_runtime_conformance_report_file",
                "restart_runtime_conformance_report_bytes",
                "restart_runtime_conformance_report_sha256",
                "restart-runtime conformance report",
            ),
            (
                audit_execution,
                "restart_runtime_execution_receipt_file",
                "restart_runtime_execution_receipt_bytes",
                "restart_runtime_execution_receipt_sha256",
                "restart-runtime execution receipt",
            ),
        ):
            filename = block.get(filename_field)
            path = _resolve_artifact_file(root, filename, label=label)
            payload = _read_bounded(path, 64 << 20, label=label)
            audit.require(
                block.get(bytes_field) == len(payload)
                and block.get(digest_field) == hashlib.sha256(payload).hexdigest(),
                code="formal_supporting_file_mismatch",
                message=f"copied {label} differs from its formal binding",
            )
        formal_test_path = _resolve_artifact_file(
            root,
            source.get("test_report_file"),
            label="formal test report",
        )
        formal_test_payload = _read_bounded(
            formal_test_path,
            64 << 20,
            label="formal test report",
        )
        (
            preformal_report,
            preformal_runtime_conformance,
            preformal_accepted_closure_evidence,
        ) = _validate_preformal_test_report_v2(
            formal_test_payload,
            root=root,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )
        audit.passed_checks += 1
        development_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                development_qualification.get("closure_file"),
                label="development qualification closure",
            ),
            64 << 20,
            label="development qualification closure",
        )
        archived_result_qualification = _validate_development_qualification(
            development_payload,
            block=development_qualification,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
            bound_audit_execution=audit_execution,
            preformal_runtime_seal=cast(
                Mapping[str, object],
                preformal_report["runtime_seal"],
            ),
        )
        live_result_qualification = authorization_attempt.payloads.get(
            _DEVELOPMENT_RESULT_QUALIFICATION_NAME
        )
        copied_result_qualification = _read_bounded(
            result_qualification_copy,
            64 << 20,
            label="copied development result qualification",
        )
        if (
            live_result_qualification is None
            or copied_result_qualification
            != live_result_qualification
            or copied_result_qualification
            != archived_result_qualification
            or hashlib.sha256(copied_result_qualification).hexdigest()
            != development_qualification.get(
                "result_qualification_sha256"
            )
        ):
            raise ArtifactAuditError(
                "formal development result qualification differs across "
                "the live attempt, copied artifact, archive, or binding"
            )
        if (
            preformal_accepted_closure_evidence.get(
                "development_closure_sha256"
            )
            != development_qualification.get("closure_sha256")
            or preformal_accepted_closure_evidence.get(
                "producer_manifest_sha256"
            )
            != development_qualification.get(
                "producer_manifest_sha256"
            )
            or preformal_accepted_closure_evidence.get(
                "raw_result_sha256"
            )
            != development_qualification.get("raw_result_sha256")
        ):
            raise ArtifactAuditError(
                "formal development qualification differs from the sealed "
                "preformal accepted-closure evidence"
            )
        audit.passed_checks += 1
        audit_bootstrap_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("bootstrap_source_file"),
                label="audit bootstrap source",
            ),
            _MAX_SOURCE_FILE_BYTES,
            label="audit bootstrap source",
        )
        audit_request_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("prebinding_request_file"),
                label="prebinding conformance request",
            ),
            _MAX_PREBINDING_REQUEST_BYTES,
            label="prebinding conformance request",
        )
        audit_path_runtime_manifest_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("prebinding_path_runtime_manifest_file"),
                label="prebinding path runtime manifest",
            ),
            4 << 20,
            label="prebinding path runtime manifest",
        )
        audit_descriptor_runtime_manifest_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("prebinding_descriptor_runtime_manifest_file"),
                label="prebinding descriptor runtime manifest",
            ),
            4 << 20,
            label="prebinding descriptor runtime manifest",
        )
        audit_path_invocation_manifest_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("prebinding_path_invocation_manifest_file"),
                label="prebinding path invocation manifest",
            ),
            1 << 20,
            label="prebinding path invocation manifest",
        )
        audit_descriptor_invocation_manifest_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("prebinding_descriptor_invocation_manifest_file"),
                label="prebinding descriptor invocation manifest",
            ),
            1 << 20,
            label="prebinding descriptor invocation manifest",
        )
        audit_conformance_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("prebinding_conformance_report_file"),
                label="prebinding conformance report",
            ),
            64 << 20,
            label="prebinding conformance report",
        )
        audit_execution_receipt_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("prebinding_execution_receipt_file"),
                label="prebinding execution receipt",
            ),
            4 << 20,
            label="prebinding execution receipt",
        )
        outcome_runtime_manifest_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get("outcome_runtime_manifest_file"),
                label="outcome descriptor runtime manifest",
            ),
            4 << 20,
            label="outcome descriptor runtime manifest",
        )
        restart_runtime_report_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get(
                    "restart_runtime_conformance_report_file"
                ),
                label="restart-runtime conformance report",
            ),
            4 << 20,
            label="restart-runtime conformance report",
        )
        restart_runtime_receipt_payload = _read_bounded(
            _resolve_artifact_file(
                root,
                audit_execution.get(
                    "restart_runtime_execution_receipt_file"
                ),
                label="restart-runtime execution receipt",
            ),
            4 << 20,
            label="restart-runtime execution receipt",
        )
        if (
            preformal_runtime_conformance.get(
                "conformance_sha256"
            )
            != hashlib.sha256(
                _canonical_json_bytes(dict(audit_execution))
            ).hexdigest()
            or preformal_runtime_conformance.get(
                "restart_runtime_conformance_report_sha256"
            )
            != hashlib.sha256(
                restart_runtime_report_payload
            ).hexdigest()
            or preformal_runtime_conformance.get(
                "restart_runtime_execution_receipt_sha256"
            )
            != hashlib.sha256(
                restart_runtime_receipt_payload
            ).hexdigest()
            or preformal_runtime_conformance.get(
                "restart_runtime_support_files"
            )
            != audit_execution.get("restart_runtime_support_files")
            or preformal_runtime_conformance.get(
                "restart_runtime_repeat_count"
            )
            != audit_execution.get("restart_runtime_repeat_count")
            or preformal_runtime_conformance.get(
                "restart_runtime_path_descriptor_equal"
            )
            != audit_execution.get(
                "restart_runtime_path_descriptor_equal"
            )
        ):
            raise ArtifactAuditError(
                "formal audit execution differs from the sealed preformal rehearsal"
            )
        if (
            formal_input_preflight_receipt.get(
                "preformal_report_sha256"
            )
            != hashlib.sha256(formal_test_payload).hexdigest()
            or formal_input_preflight_receipt.get(
                "development_closure_sha256"
            )
            != hashlib.sha256(development_payload).hexdigest()
            or formal_input_preflight_receipt.get(
                "accepted_closure_evidence_sha256"
            )
            != hashlib.sha256(
                _canonical_json_bytes(
                    dict(
                        preformal_accepted_closure_evidence
                    )
                )
            ).hexdigest()
            or formal_input_preflight_receipt.get(
                "runtime_conformance_sha256"
            )
            != hashlib.sha256(
                _canonical_json_bytes(
                    dict(preformal_runtime_conformance)
                )
            ).hexdigest()
        ):
            raise ArtifactAuditError(
                "formal input preflight receipt differs from the exact "
                "independently reopened inputs"
            )
        _validate_audit_execution_conformance(
            block=audit_execution,
            bootstrap_payload=audit_bootstrap_payload,
            request_payload=audit_request_payload,
            path_runtime_manifest_payload=(audit_path_runtime_manifest_payload),
            descriptor_runtime_manifest_payload=(audit_descriptor_runtime_manifest_payload),
            path_invocation_manifest_payload=(audit_path_invocation_manifest_payload),
            descriptor_invocation_manifest_payload=(audit_descriptor_invocation_manifest_payload),
            report_payload=audit_conformance_payload,
            execution_receipt_payload=(audit_execution_receipt_payload),
            outcome_runtime_manifest_payload=(outcome_runtime_manifest_payload),
            restart_runtime_report_payload=(
                restart_runtime_report_payload
            ),
            restart_runtime_receipt_payload=(
                restart_runtime_receipt_payload
            ),
            dependencies=dependencies,
            runtime=runtime,
            source=source,
            root=root,
            preformal_repository_cwd=preformal_report.get("repository_cwd"),
        )
        audit.audit_execution_conformance_verified = True
        audit.passed_checks += 1
        conformance_name = environment.get("conformance_report_file")
        conformance_path = _resolve_artifact_file(
            root,
            conformance_name,
            label="Pendulum conformance report",
        )
        conformance_raw = _json_without_duplicate_keys(
            _read_bounded(
                conformance_path,
                64 << 20,
                label="Pendulum conformance report",
            ),
            label="Pendulum conformance report",
        )
        try:
            _validate_formal_conformance_report(conformance_raw)
            canonical_conformance = (
                _canonical_json_bytes(conformance_raw) + b"\n" if isinstance(conformance_raw, dict) else b""
            )
            if conformance_path.read_bytes() != canonical_conformance:
                raise ArtifactAuditError("Pendulum conformance report is not canonical JSON")
        except (ArtifactAuditError, OSError) as error:
            audit.error("formal_conformance_report_failed", str(error))
        else:
            audit.passed_checks += 1
        coverage_conformance_path = _resolve_artifact_file(
            root,
            coverage_arithmetic.get("conformance_report_file"),
            label="coverage conformance report",
        )
        coverage_conformance_payload = _read_bounded(
            coverage_conformance_path,
            64 << 20,
            label="coverage conformance report",
        )
        coverage_conformance_raw = _json_without_duplicate_keys(
            coverage_conformance_payload,
            label="coverage conformance report",
        )
        try:
            _validate_formal_coverage_conformance_report(coverage_conformance_raw)
            canonical_coverage_conformance = (
                _canonical_json_bytes(coverage_conformance_raw) + b"\n"
                if isinstance(coverage_conformance_raw, dict)
                else b""
            )
            if coverage_conformance_payload != canonical_coverage_conformance:
                raise ArtifactAuditError("coverage conformance report is not canonical JSON")
        except (ArtifactAuditError, OSError, TypeError, ValueError) as error:
            audit.error("formal_coverage_conformance_failed", str(error))
        else:
            audit.coverage_conformance_verified = True
            audit.passed_checks += 1
        try:
            import jsonschema

            binding_schema = json.loads(payloads["schemas/formal-binding.schema.json"].decode("utf-8"))
            binding_errors = list(jsonschema.Draft202012Validator(binding_schema).iter_errors(binding))
        except (ImportError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactAuditError(f"cannot validate copied formal-binding schema: {error}") from error
        audit.require(
            not binding_errors,
            code="formal_binding_schema_error",
            message=(
                "copied formal binding violates its copied schema"
                if binding_errors
                else "formal binding schema validation"
            ),
        )
    except (ArtifactAuditError, OSError, StopIteration, ValueError) as error:
        audit.error("formal_input_package_invalid", str(error))
    return bound_device


def _audit_formal_schedule(
    audit: _Audit,
    root: Path,
    result: Mapping[str, object],
) -> None:
    """Independently enforce the complete v1.16 formal replicate schedule."""

    if result.get("lane") != "formal":
        return
    replicates = _mapping_rows(result.get("replicates"))
    seeds = tuple(replicate.get("master_seed") for replicate in replicates)
    replicate_ids = tuple(replicate.get("replicate_id") for replicate in replicates)
    expected_ids = tuple(f"wm001-formal-{seed}" for seed in _FORMAL_SEEDS)
    audit.require(
        seeds == _FORMAL_SEEDS and replicate_ids == expected_ids,
        code="formal_replicate_schedule_mismatch",
        message=(
            "formal result does not contain the exact ordered eight v1.16 master seeds and seed-bound replicate IDs"
        ),
    )

    behavior_conditions = {
        _TASK_A: (
            "cold",
            "after_a",
            "frozen",
            "corrupted",
            "irrelevant",
            "after_b_replay",
            "after_b_naive",
            "random",
            "oracle",
        ),
        _TASK_B: (
            "after_a",
            "after_b_replay",
            "after_b_naive",
            "random",
            "oracle",
        ),
    }
    predictive_conditions = {
        _TASK_A: (
            "cold",
            "after_a",
            "frozen",
            "corrupted",
            "irrelevant",
            "after_b_replay",
            "after_b_naive",
        ),
        _TASK_B: ("after_a", "after_b_replay", "after_b_naive"),
        _TASK_IRRELEVANT: ("cold", "irrelevant"),
    }
    episode_contracts: dict[tuple[str, str, str, str], int] = {
        ("collect_a", _TASK_A, "collection_random", "cold"): 8,
        ("collect_b", _TASK_B, "collection_random", "after_a"): 8,
        (
            "collect_irrelevant",
            _TASK_IRRELEVANT,
            "collection_random",
            "cold",
        ): 8,
        (
            "predictive_validation_a",
            _TASK_A,
            "validation_random",
            "after_a",
        ): 8,
        (
            "predictive_validation_b",
            _TASK_B,
            "validation_random",
            "after_a",
        ): 8,
        (
            "predictive_validation_irrelevant",
            _TASK_IRRELEVANT,
            "validation_random",
            "irrelevant",
        ): 8,
    }
    for task_id, conditions in behavior_conditions.items():
        split = "behavior_evaluation_a" if task_id == _TASK_A else "behavior_evaluation_b"
        for condition in conditions:
            episode_contracts[(split, task_id, condition, condition)] = 32

    for replicate in replicates:
        replicate_id = str(replicate.get("replicate_id", "<missing>"))
        episodes = _mapping_rows(replicate.get("episodes"))
        actual_episode_contracts: dict[tuple[object, object, object, object], int] = {}
        for episode in episodes:
            key = (
                episode.get("split"),
                episode.get("task_id"),
                episode.get("condition"),
                episode.get("checkpoint_id"),
            )
            actual_episode_contracts[key] = actual_episode_contracts.get(key, 0) + 1
        audit.require(
            actual_episode_contracts == episode_contracts
            and len(episodes) == 496
            and all(episode.get("environment_steps") == _EPISODE_HORIZON for episode in episodes),
            code="formal_episode_budget_mismatch",
            message=(
                f"{replicate_id} does not contain exactly the 496 sealed "
                "whole episodes across A, B, and the independent distractor"
            ),
            replicate_id=replicate_id,
        )

        transitions = _mapping_rows(replicate.get("transitions"))
        split_counts: dict[object, int] = {}
        for transition in transitions:
            transition_split = transition.get("split")
            split_counts[transition_split] = split_counts.get(transition_split, 0) + 1
        expected_split_counts: dict[str, int] = {}
        for (split, _task, _condition, _checkpoint), episode_count in episode_contracts.items():
            expected_split_counts[split] = expected_split_counts.get(split, 0) + episode_count * _EPISODE_HORIZON
        audit.require(
            len(transitions) == 99_200 and split_counts == expected_split_counts,
            code="formal_transition_budget_mismatch",
            message=(
                f"{replicate_id} does not retain exactly 99,200 real "
                "transitions partitioned by the sealed episode schedule"
            ),
            replicate_id=replicate_id,
        )

        predictive = _mapping_rows(replicate.get("predictive_metrics"))
        actual_predictive: dict[tuple[object, object, object], int] = {}
        predictive_counts_valid = True
        for row in predictive:
            predictive_key = (
                row.get("task_id"),
                row.get("condition"),
                row.get("checkpoint_id"),
            )
            actual_predictive[predictive_key] = actual_predictive.get(predictive_key, 0) + 1
            predictive_counts_valid = predictive_counts_valid and row.get("transition_count") == 1_600
        expected_predictive = {
            (task_id, condition, condition): 1
            for task_id, conditions in predictive_conditions.items()
            for condition in conditions
        }
        audit.require(
            actual_predictive == expected_predictive and len(predictive) == 12 and predictive_counts_valid,
            code="formal_predictive_budget_mismatch",
            message=(
                f"{replicate_id} predictive matrix is not exactly twelve "
                "condition rows over 1,600 paired validation transitions each"
            ),
            replicate_id=replicate_id,
        )

        policy_runs = _mapping_rows(replicate.get("policy_runs"))
        audit.require(
            len(policy_runs) == 20,
            code="formal_policy_run_budget_mismatch",
            message=(
                f"{replicate_id} does not contain exactly 20 policy runs "
                "covering every sealed collection, validation, and behavior condition"
            ),
            replicate_id=replicate_id,
        )

        updates = _mapping_rows(replicate.get("updates"))
        committed = [row for row in updates if row.get("status") == "committed"]
        rejected = [row for row in updates if row.get("status") == "rejected"]
        expected_committed = tuple(_EXPECTED_PHASE_SPLITS)
        audit.require(
            tuple(row.get("phase") for row in committed) == expected_committed
            and all(row.get("optimizer_steps") == 2_000 for row in committed)
            and len(rejected) == 1
            and rejected[0].get("phase") == "rejected_update_probe"
            and rejected[0].get("optimizer_steps") == 0,
            code="formal_update_budget_mismatch",
            message=(
                f"{replicate_id} does not contain the exact ordered five "
                "matched 2,000-step updates plus the zero-step rejection probe"
            ),
            replicate_id=replicate_id,
        )

        sidecar_name = f"{replicate_id}.json"
        try:
            sidecar = _read_bounded(
                _resolve_artifact_file(root, sidecar_name, label=f"{replicate_id} sidecar"),
                _MAX_RESULT_BYTES,
                label=f"{replicate_id} sidecar",
            )
            expected_sidecar = _canonical_json_bytes(replicate) + b"\n"
            sidecar_matches = sidecar == expected_sidecar
        except (ArtifactAuditError, OSError, TypeError, ValueError):
            sidecar_matches = False
        audit.require(
            sidecar_matches,
            code="formal_replicate_sidecar_mismatch",
            message=(
                f"{replicate_id} embedded result row is not byte-for-byte "
                "cross-bound to its canonical per-replicate sidecar"
            ),
            replicate_id=replicate_id,
        )


_ANALYSIS_TASK_A = "pendulum_normal_torque"
_ANALYSIS_TASK_B = "pendulum_reversed_torque"
_ANALYSIS_TASK_IRRELEVANT = "independent_phase_oscillator"
_ANALYSIS_BEHAVIOR_CONDITIONS: Mapping[str, tuple[str, ...]] = {
    _ANALYSIS_TASK_A: (
        "cold",
        "after_a",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_b_replay",
        "after_b_naive",
        "random",
        "oracle",
    ),
    _ANALYSIS_TASK_B: (
        "after_a",
        "after_b_replay",
        "after_b_naive",
        "random",
        "oracle",
    ),
}
_ANALYSIS_PREDICTIVE_CONDITIONS: Mapping[str, tuple[str, ...]] = {
    _ANALYSIS_TASK_A: (
        "cold",
        "after_a",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_b_replay",
        "after_b_naive",
    ),
    _ANALYSIS_TASK_B: ("after_a", "after_b_replay", "after_b_naive"),
    _ANALYSIS_TASK_IRRELEVANT: ("cold", "irrelevant"),
}
_ANALYSIS_BEHAVIOR_SPLITS = {
    _ANALYSIS_TASK_A: "behavior_evaluation_a",
    _ANALYSIS_TASK_B: "behavior_evaluation_b",
}
_ANALYSIS_PREDICTIVE_SPLITS = {
    _ANALYSIS_TASK_A: "predictive_validation_a",
    _ANALYSIS_TASK_B: "predictive_validation_b",
    _ANALYSIS_TASK_IRRELEVANT: "predictive_validation_irrelevant",
}
_ANALYSIS_METRIC_UNITS: Mapping[str, str] = {
    "irrelevant_source_nll_improvement_after_irrelevant_vs_cold": "nats/target-dimension",
    "a_nll_improvement_after_a_vs_frozen": "nats/target-dimension",
    "a_nll_improvement_after_a_vs_corrupted": "nats/target-dimension",
    "a_nll_improvement_after_a_vs_irrelevant": "nats/target-dimension",
    "a_irrelevant_nll_improvement_vs_frozen": "nats/target-dimension",
    "a_after_a_interval_90_coverage": "fraction",
    "a_return_improvement_after_a_vs_cold": "return",
    "a_return_improvement_after_a_vs_frozen": "return",
    "a_return_improvement_after_a_vs_irrelevant": "return",
    "a_irrelevant_return_improvement_vs_frozen": "return",
    "a_after_a_oracle_normalized_score": "fraction",
    "a_oracle_vs_random_return_gap": "return",
    "b_nll_improvement_after_b_replay_vs_before_b": "nats/target-dimension",
    "b_return_improvement_after_b_replay_vs_before_b": "return",
    "b_return_improvement_after_b_naive_vs_before_b": "return",
    "a_naive_forgetting_return_drop": "return",
    "retained_a_gain_fraction": "fraction",
    "a_replay_vs_naive_return_advantage": "return",
    "b_replay_minus_naive_return": "return",
    "shared_parameter_violations": "count",
    "restart_component_hash_mismatches": "count",
    "restart_identity_or_lineage_mismatches": "count",
    "restart_prediction_max_abs_difference": "absolute-difference",
    "restart_action_max_abs_difference": "absolute-difference",
    "restart_episode_return_max_abs_difference": "return",
}
_ANALYSIS_T_CRITICAL_BY_N: Mapping[int, float] = {
    2: 12.706204736,
    3: 4.30265273,
    4: 3.182446305,
    5: 2.776445105,
    6: 2.570581836,
    7: 2.446911851,
    8: 2.364624251,
}
_ANALYSIS_STRUCTURAL_CHECK_NAMES: Mapping[str, tuple[str, ...]] = {
    "K0": (
        "result_envelope_matches_lane",
        "replicate_schedule_complete",
        "formal_or_paired_development_budgets",
        "derived_seed_schedule_exact",
        "raw_numeric_sources_complete_and_finite",
    ),
    "K1": (
        "real_identity_uniqueness",
        "heldout_and_replay_split_isolation",
        "episode_update_and_action_digest_lineage",
    ),
    "K2": (
        "committed_updates_change_exact_candidate_bytes",
        "update_branch_ancestry_exact",
        "rejected_probe_is_byte_stable",
        "committed_digest_used_downstream",
    ),
    "K7": (
        "checkpoint_component_set_complete",
        "fresh_process_state_and_behavior_parity_exact",
    ),
}
_ANALYSIS_NUMERIC_GATE_DECLARATIONS: Mapping[
    str,
    tuple[tuple[str, str, str, float, str], ...],
] = {
    "K3": (
        (
            "irrelevant_source_nll_improvement_after_irrelevant_vs_cold",
            "mean",
            "ge",
            0.05,
            "irrelevant_source_vs_cold_mean_nll_improvement",
        ),
        (
            "irrelevant_source_nll_improvement_after_irrelevant_vs_cold",
            "ci_95_lower",
            "gt",
            0.0,
            "irrelevant_source_vs_cold_nll_improvement_ci_lower",
        ),
        (
            "a_nll_improvement_after_a_vs_frozen",
            "mean",
            "ge",
            0.05,
            "a_vs_frozen_mean_nll_improvement",
        ),
        (
            "a_nll_improvement_after_a_vs_frozen",
            "ci_95_lower",
            "gt",
            0.0,
            "a_vs_frozen_nll_improvement_ci_lower",
        ),
        (
            "a_nll_improvement_after_a_vs_corrupted",
            "mean",
            "ge",
            0.05,
            "a_vs_corrupted_mean_nll_improvement",
        ),
        (
            "a_nll_improvement_after_a_vs_corrupted",
            "ci_95_lower",
            "gt",
            0.0,
            "a_vs_corrupted_nll_improvement_ci_lower",
        ),
        (
            "a_nll_improvement_after_a_vs_irrelevant",
            "mean",
            "ge",
            0.05,
            "a_vs_irrelevant_mean_nll_improvement",
        ),
        (
            "a_nll_improvement_after_a_vs_irrelevant",
            "ci_95_lower",
            "gt",
            0.0,
            "a_vs_irrelevant_nll_improvement_ci_lower",
        ),
    ),
    "K4": (
        (
            "a_return_improvement_after_a_vs_cold",
            "mean",
            "ge",
            100.0,
            "after_a_vs_cold_mean_return_improvement",
        ),
        (
            "a_return_improvement_after_a_vs_cold",
            "ci_95_lower",
            "gt",
            0.0,
            "after_a_vs_cold_return_improvement_ci_lower",
        ),
        (
            "a_return_improvement_after_a_vs_frozen",
            "mean",
            "ge",
            100.0,
            "after_a_vs_frozen_mean_return_improvement",
        ),
        (
            "a_return_improvement_after_a_vs_frozen",
            "ci_95_lower",
            "gt",
            0.0,
            "after_a_vs_frozen_return_improvement_ci_lower",
        ),
        (
            "a_return_improvement_after_a_vs_irrelevant",
            "mean",
            "ge",
            100.0,
            "after_a_vs_irrelevant_mean_return_improvement",
        ),
        (
            "a_return_improvement_after_a_vs_irrelevant",
            "ci_95_lower",
            "gt",
            0.0,
            "after_a_vs_irrelevant_return_improvement_ci_lower",
        ),
        (
            "a_after_a_oracle_normalized_score",
            "mean",
            "ge",
            0.2,
            "after_a_oracle_normalized_score",
        ),
        (
            "a_oracle_vs_random_return_gap",
            "mean",
            "ge",
            100.0,
            "oracle_vs_random_mean_return_gap",
        ),
    ),
    "K5": (
        (
            "b_nll_improvement_after_b_replay_vs_before_b",
            "mean",
            "ge",
            0.05,
            "after_b_replay_vs_before_b_mean_b_nll_improvement",
        ),
        (
            "b_nll_improvement_after_b_replay_vs_before_b",
            "ci_95_lower",
            "gt",
            0.0,
            "after_b_replay_vs_before_b_b_nll_improvement_ci_lower",
        ),
        (
            "b_return_improvement_after_b_replay_vs_before_b",
            "mean",
            "ge",
            100.0,
            "after_b_replay_vs_before_b_mean_b_return_improvement",
        ),
        (
            "b_return_improvement_after_b_replay_vs_before_b",
            "ci_95_lower",
            "gt",
            0.0,
            "after_b_replay_vs_before_b_b_return_improvement_ci_lower",
        ),
        (
            "b_return_improvement_after_b_naive_vs_before_b",
            "mean",
            "ge",
            100.0,
            "after_b_naive_vs_before_b_mean_b_return_improvement",
        ),
        (
            "b_return_improvement_after_b_naive_vs_before_b",
            "ci_95_lower",
            "gt",
            0.0,
            "after_b_naive_vs_before_b_b_return_improvement_ci_lower",
        ),
        (
            "a_naive_forgetting_return_drop",
            "mean",
            "ge",
            50.0,
            "naive_a_forgetting_mean_return_drop",
        ),
        (
            "a_naive_forgetting_return_drop",
            "ci_95_lower",
            "gt",
            0.0,
            "naive_a_forgetting_return_drop_ci_lower",
        ),
        (
            "shared_parameter_violations",
            "mean",
            "eq",
            0.0,
            "shared_parameter_violations",
        ),
    ),
    "K6": (
        (
            "retained_a_gain_fraction",
            "mean",
            "ge",
            0.8,
            "mean_seed_level_retained_a_gain_fraction",
        ),
        (
            "retained_a_gain_fraction",
            "ci_95_lower",
            "ge",
            0.65,
            "retained_a_gain_fraction_ci_lower",
        ),
        (
            "a_replay_vs_naive_return_advantage",
            "mean",
            "ge",
            50.0,
            "replay_vs_naive_mean_a_return_advantage",
        ),
        (
            "a_replay_vs_naive_return_advantage",
            "ci_95_lower",
            "gt",
            0.0,
            "replay_vs_naive_a_return_advantage_ci_lower",
        ),
        (
            "b_replay_minus_naive_return",
            "mean",
            "ge",
            -25.0,
            "replay_minus_naive_mean_b_return",
        ),
        (
            "b_replay_minus_naive_return",
            "ci_95_lower",
            "ge",
            -75.0,
            "replay_minus_naive_b_return_ci_lower",
        ),
    ),
}


def _analysis_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _analysis_phase_index(
    replicate: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    rows = _mapping_rows(replicate.get("updates"))
    counts: dict[str, int] = {}
    for row in rows:
        phase = row.get("phase")
        if isinstance(phase, str):
            counts[phase] = counts.get(phase, 0) + 1
    return {phase: row for row in rows if isinstance((phase := row.get("phase")), str) and counts.get(phase) == 1}


def _analysis_mean_return(
    replicate: Mapping[str, object],
    task_id: str,
    condition: str,
) -> float | None:
    rows = [
        row
        for row in _mapping_rows(replicate.get("episodes"))
        if row.get("task_id") == task_id
        and row.get("split") == _ANALYSIS_BEHAVIOR_SPLITS[task_id]
        and row.get("condition") == condition
    ]
    values: list[float] = []
    for row in rows:
        value = row.get("return")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    if not rows or len(values) != len(rows):
        return None
    return statistics.fmean(values)


def _analysis_predictive_row(
    replicate: Mapping[str, object],
    task_id: str,
    condition: str,
) -> Mapping[str, object] | None:
    rows = [
        row
        for row in _mapping_rows(replicate.get("predictive_metrics"))
        if row.get("task_id") == task_id
        and row.get("split") == _ANALYSIS_PREDICTIVE_SPLITS[task_id]
        and row.get("condition") == condition
        and row.get("checkpoint_id") == condition
    ]
    return rows[0] if len(rows) == 1 else None


def _analysis_replicate_values(
    replicate: Mapping[str, object],
) -> dict[str, float]:
    predictive = {
        (task_id, condition): _analysis_predictive_row(
            replicate,
            task_id,
            condition,
        )
        for task_id, conditions in _ANALYSIS_PREDICTIVE_CONDITIONS.items()
        for condition in conditions
    }
    returns = {
        (task_id, condition): _analysis_mean_return(
            replicate,
            task_id,
            condition,
        )
        for task_id, conditions in _ANALYSIS_BEHAVIOR_CONDITIONS.items()
        for condition in conditions
    }

    def predictive_value(
        task_id: str,
        condition: str,
        field: str,
    ) -> float | None:
        row = predictive[(task_id, condition)]
        if row is None:
            return None
        value = row.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            return None
        return float(value)

    def coverage_value(
        task_id: str,
        condition: str,
    ) -> float | None:
        row = predictive[(task_id, condition)]
        if row is None or row.get("coverage_semantics") != _COVERAGE_SEMANTICS:
            return None
        covered = row.get("interval_90_covered_target_count")
        total = row.get("coverage_target_count")
        if type(covered) is not int or type(total) is not int or total <= 0 or covered < 0 or covered > total:
            return None
        return covered / total

    nll = lambda task_id, condition: predictive_value(  # noqa: E731
        task_id,
        condition,
        "mixture_nll_nats_per_target_dimension",
    )
    operands: Mapping[str, tuple[float | None, float | None]] = {
        "irrelevant_source_nll_improvement_after_irrelevant_vs_cold": (
            nll(_ANALYSIS_TASK_IRRELEVANT, "cold"),
            nll(_ANALYSIS_TASK_IRRELEVANT, "irrelevant"),
        ),
        "a_nll_improvement_after_a_vs_frozen": (
            nll(_ANALYSIS_TASK_A, "frozen"),
            nll(_ANALYSIS_TASK_A, "after_a"),
        ),
        "a_nll_improvement_after_a_vs_corrupted": (
            nll(_ANALYSIS_TASK_A, "corrupted"),
            nll(_ANALYSIS_TASK_A, "after_a"),
        ),
        "a_nll_improvement_after_a_vs_irrelevant": (
            nll(_ANALYSIS_TASK_A, "irrelevant"),
            nll(_ANALYSIS_TASK_A, "after_a"),
        ),
        "a_irrelevant_nll_improvement_vs_frozen": (
            nll(_ANALYSIS_TASK_A, "frozen"),
            nll(_ANALYSIS_TASK_A, "irrelevant"),
        ),
        "a_return_improvement_after_a_vs_cold": (
            returns[(_ANALYSIS_TASK_A, "after_a")],
            returns[(_ANALYSIS_TASK_A, "cold")],
        ),
        "a_return_improvement_after_a_vs_frozen": (
            returns[(_ANALYSIS_TASK_A, "after_a")],
            returns[(_ANALYSIS_TASK_A, "frozen")],
        ),
        "a_return_improvement_after_a_vs_irrelevant": (
            returns[(_ANALYSIS_TASK_A, "after_a")],
            returns[(_ANALYSIS_TASK_A, "irrelevant")],
        ),
        "a_irrelevant_return_improvement_vs_frozen": (
            returns[(_ANALYSIS_TASK_A, "irrelevant")],
            returns[(_ANALYSIS_TASK_A, "frozen")],
        ),
        "a_oracle_vs_random_return_gap": (
            returns[(_ANALYSIS_TASK_A, "oracle")],
            returns[(_ANALYSIS_TASK_A, "random")],
        ),
        "b_nll_improvement_after_b_replay_vs_before_b": (
            nll(_ANALYSIS_TASK_B, "after_a"),
            nll(_ANALYSIS_TASK_B, "after_b_replay"),
        ),
        "b_return_improvement_after_b_replay_vs_before_b": (
            returns[(_ANALYSIS_TASK_B, "after_b_replay")],
            returns[(_ANALYSIS_TASK_B, "after_a")],
        ),
        "b_return_improvement_after_b_naive_vs_before_b": (
            returns[(_ANALYSIS_TASK_B, "after_b_naive")],
            returns[(_ANALYSIS_TASK_B, "after_a")],
        ),
        "a_naive_forgetting_return_drop": (
            returns[(_ANALYSIS_TASK_A, "after_a")],
            returns[(_ANALYSIS_TASK_A, "after_b_naive")],
        ),
        "a_replay_vs_naive_return_advantage": (
            returns[(_ANALYSIS_TASK_A, "after_b_replay")],
            returns[(_ANALYSIS_TASK_A, "after_b_naive")],
        ),
        "b_replay_minus_naive_return": (
            returns[(_ANALYSIS_TASK_B, "after_b_replay")],
            returns[(_ANALYSIS_TASK_B, "after_b_naive")],
        ),
    }
    values = {
        name: float(left - right)
        for name, (left, right) in operands.items()
        if left is not None and right is not None and math.isfinite(left) and math.isfinite(right)
    }
    coverage = coverage_value(
        _ANALYSIS_TASK_A,
        "after_a",
    )
    if coverage is not None and math.isfinite(coverage):
        values["a_after_a_interval_90_coverage"] = coverage
        after_a_prediction = predictive[(_ANALYSIS_TASK_A, "after_a")]
        assert after_a_prediction is not None
        covered_target_count = after_a_prediction.get("interval_90_covered_target_count")
        coverage_target_count = after_a_prediction.get("coverage_target_count")
        assert type(covered_target_count) is int
        assert type(coverage_target_count) is int
        values["_a_after_a_interval_90_covered_target_count"] = float(covered_target_count)
        values["_a_after_a_coverage_target_count"] = float(coverage_target_count)

    oracle = returns[(_ANALYSIS_TASK_A, "oracle")]
    random_return = returns[(_ANALYSIS_TASK_A, "random")]
    after_a = returns[(_ANALYSIS_TASK_A, "after_a")]
    if oracle is not None and random_return is not None and after_a is not None and oracle != random_return:
        values["a_after_a_oracle_normalized_score"] = (after_a - random_return) / (oracle - random_return)

    cold = returns[(_ANALYSIS_TASK_A, "cold")]
    after_b_replay_a = returns[(_ANALYSIS_TASK_A, "after_b_replay")]
    if cold is not None and after_a is not None and after_b_replay_a is not None and after_a > cold:
        values["retained_a_gain_fraction"] = (after_b_replay_a - cold) / (after_a - cold)

    updates = _analysis_phase_index(replicate)
    shared_violations = 0
    train_a = updates.get("train_a")
    for phase in ("train_b_replay", "train_b_naive"):
        update = updates.get(phase)
        if train_a is None or update is None:
            shared_violations += 1
            continue
        if update.get("predecessor_parameter_sha256") != train_a.get("committed_parameter_sha256"):
            shared_violations += 1
        if update.get("predecessor_model_version") != train_a.get("committed_model_version"):
            shared_violations += 1
    for component in _mapping_rows(replicate.get("checkpoint_components")):
        component_id = str(component.get("component_id", "")).lower()
        if "task_a" in component_id or "task_b" in component_id:
            shared_violations += 1
    values["shared_parameter_violations"] = float(shared_violations)

    parity = replicate.get("restart_parity")
    if isinstance(parity, Mapping):
        mismatches = parity.get("component_hash_mismatches")
        if isinstance(mismatches, list):
            values["restart_component_hash_mismatches"] = float(len(mismatches))
        for source, name in (
            (
                "identity_or_lineage_mismatches",
                "restart_identity_or_lineage_mismatches",
            ),
            (
                "prediction_max_abs_difference",
                "restart_prediction_max_abs_difference",
            ),
            ("action_max_abs_difference", "restart_action_max_abs_difference"),
            (
                "episode_return_max_abs_difference",
                "restart_episode_return_max_abs_difference",
            ),
        ):
            item = parity.get(source)
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
                values[name] = float(item)
    return values


def _independent_recompute_aggregate_metrics(
    result: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, float]]]:
    """Recompute every declared contrast from raw rows without producer code."""

    replicate_values = [_analysis_replicate_values(replicate) for replicate in _mapping_rows(result.get("replicates"))]
    rows: list[dict[str, object]] = []
    for name, unit in _ANALYSIS_METRIC_UNITS.items():
        values = [replicate[name] for replicate in replicate_values if name in replicate]
        if not values or len(values) != len(replicate_values):
            continue
        mean = statistics.fmean(values)
        if len(values) == 1:
            lower = upper = mean
        else:
            critical = _ANALYSIS_T_CRITICAL_BY_N.get(
                len(values),
                1.959963985,
            )
            # The sealed producer computes the standard error first, then
            # multiplies by the critical value.  Preserve that association:
            # the mathematically equivalent ``critical * stdev / sqrt(n)``
            # can differ by one ULP and would make exact evidence hashes drift.
            standard_error = statistics.stdev(values) / math.sqrt(len(values))
            margin = critical * standard_error
            lower, upper = mean - margin, mean + margin
        rows.append(
            {
                "name": name,
                "unit": unit,
                "replicate_values": [float(value) for value in values],
                "mean": float(mean),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
            }
        )
    return rows, replicate_values


def _analysis_evidence_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _analysis_check(
    name: str,
    observed: object,
    comparator: str,
    threshold: object,
    passed: bool,
    evidence: object,
) -> dict[str, object]:
    return {
        "name": name,
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
        "passed": bool(passed),
        "raw_evidence_sha256": _analysis_evidence_sha256(evidence),
    }


def _analysis_metric_check(
    metrics: Mapping[str, Mapping[str, object]],
    metric_name: str,
    field: str,
    comparator: str,
    threshold: float,
    check_name: str,
) -> dict[str, object]:
    metric = metrics.get(metric_name)
    if metric is None:
        return _analysis_check(
            check_name,
            "missing",
            "eq",
            "present",
            False,
            {"metric": metric_name, "field": field},
        )
    value = metric.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return _analysis_check(
            check_name,
            "missing",
            "eq",
            "present",
            False,
            {"metric": metric_name, "field": field},
        )
    observed = float(value)
    passed = {
        "eq": observed == threshold,
        "gt": observed > threshold,
        "ge": observed >= threshold,
        "lt": observed < threshold,
        "le": observed <= threshold,
    }[comparator]
    return _analysis_check(
        check_name,
        observed,
        comparator,
        threshold,
        passed,
        metric,
    )


def _independent_coverage_count_gate_checks(
    replicate_values: Sequence[Mapping[str, float]],
) -> tuple[dict[str, object], dict[str, object]]:
    count_pairs: list[tuple[int, int]] = []
    for row in replicate_values:
        covered_value = row.get("_a_after_a_interval_90_covered_target_count")
        total_value = row.get("_a_after_a_coverage_target_count")
        if (
            not isinstance(covered_value, (int, float))
            or isinstance(covered_value, bool)
            or not math.isfinite(float(covered_value))
            or not float(covered_value).is_integer()
            or not isinstance(total_value, (int, float))
            or isinstance(total_value, bool)
            or not math.isfinite(float(total_value))
            or not float(total_value).is_integer()
        ):
            count_pairs = []
            break
        covered = int(covered_value)
        total = int(total_value)
        if total <= 0 or covered < 0 or covered > total:
            count_pairs = []
            break
        count_pairs.append((covered, total))

    complete = bool(replicate_values) and len(count_pairs) == len(replicate_values)
    covered_sum = sum(covered for covered, _ in count_pairs)
    target_sum = sum(total for _, total in count_pairs)
    evidence = {
        "replicate_counts": [
            {"covered_target_count": covered, "coverage_target_count": total} for covered, total in count_pairs
        ],
        "covered_target_count_sum": covered_sum,
        "coverage_target_count_sum": target_sum,
    }
    observed: int | float | bool | str = f"{covered_sum}/{target_sum}" if complete and target_sum > 0 else "missing"
    return (
        _analysis_check(
            "after_a_interval_coverage_lower_bound",
            observed,
            "10*C >= 7*T",
            "0.70 inclusive",
            complete and target_sum > 0 and 10 * covered_sum >= 7 * target_sum,
            {**evidence, "left": 10 * covered_sum, "right": 7 * target_sum},
        ),
        _analysis_check(
            "after_a_interval_coverage_upper_bound",
            observed,
            "100*C <= 99*T",
            "0.99 inclusive",
            complete and target_sum > 0 and 100 * covered_sum <= 99 * target_sum,
            {**evidence, "left": 100 * covered_sum, "right": 99 * target_sum},
        ),
    )


def _independent_recompute_gate_results(
    aggregates: Sequence[Mapping[str, object]],
    replicate_values: Sequence[Mapping[str, float]],
    *,
    protocol: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Apply the sealed K0--K7 order and thresholds without producer code.

    K0/K1/K2/K7 are zero-violation gates.  Their raw invariants are audited
    independently elsewhere in this module before this decision-table check;
    a claim-facing artifact therefore must declare the canonical passing rows.
    K3--K6 are recalculated directly from the raw-row aggregates above.
    """

    structural = {
        gate: [_analysis_check(name, 0, "eq", 0, True, []) for name in names]
        for gate, names in _ANALYSIS_STRUCTURAL_CHECK_NAMES.items()
    }
    metric_index = {str(row["name"]): row for row in aggregates if isinstance(row.get("name"), str)}
    numeric: dict[str, list[dict[str, object]]] = {}
    for gate, declarations in _ANALYSIS_NUMERIC_GATE_DECLARATIONS.items():
        gate_checks = [
            _analysis_metric_check(
                metric_index,
                metric_name,
                field,
                comparator,
                threshold,
                check_name,
            )
            for metric_name, field, comparator, threshold, check_name in declarations
        ]
        if gate == "K3":
            gate_checks.extend(_independent_coverage_count_gate_checks(replicate_values))
        if gate == "K6":
            denominator_passed = bool(replicate_values) and all(
                "retained_a_gain_fraction" in row for row in replicate_values
            )
            gate_checks.insert(
                0,
                _analysis_check(
                    "retention_denominators_positive",
                    denominator_passed,
                    "eq",
                    True,
                    denominator_passed,
                    [row.get("retained_a_gain_fraction", "undefined") for row in replicate_values],
                ),
            )
        numeric[gate] = gate_checks

    if protocol is None:
        protocol_raw = _json_without_duplicate_keys(
            _read_bounded(
                HERE / "protocol.json",
                16 << 20,
                label="WM-001 protocol",
            ),
            label="WM-001 protocol",
        )
        if not isinstance(protocol_raw, Mapping):
            raise ArtifactAuditError("WM-001 protocol root is not an object")
        protocol = protocol_raw
    killing_order = _mapping_rows(protocol.get("killing_order"))
    gates: list[dict[str, object]] = []
    for declaration in killing_order:
        gate_value = declaration.get("gate")
        if not isinstance(gate_value, str):
            raise ArtifactAuditError("WM-001 protocol has a non-string gate ID")
        gate = gate_value
        checks = [*structural.get(gate, ()), *numeric.get(gate, ())]
        passed = bool(checks) and all(row["passed"] is True for row in checks)
        gates.append(
            {
                "gate": gate,
                "name": str(declaration.get("name")),
                "checks": checks,
                "passed": passed,
                "claim_supported": False,
                "stop_reason": (None if passed else str(declaration.get("on_failure"))),
            }
        )
        if not passed:
            break
    return gates


def _audit_recomputed_analysis(
    audit: _Audit,
    result: Mapping[str, object],
    *,
    protocol: Mapping[str, object] | None = None,
) -> None:
    if not isinstance(result.get("aggregate_metrics"), list) or not isinstance(
        result.get("gate_results"),
        list,
    ):
        return
    try:
        recomputed_metrics, replicate_values = _independent_recompute_aggregate_metrics(result)
        recomputed_gates = _independent_recompute_gate_results(
            recomputed_metrics,
            replicate_values,
            protocol=protocol,
        )
        audit.require(
            _canonical_json_bytes(result.get("aggregate_metrics")) == _canonical_json_bytes(recomputed_metrics),
            code="aggregate_metrics_recomputation_mismatch",
            message="stored aggregate metrics differ from independent raw-row recomputation",
        )
        audit.require(
            _canonical_json_bytes(result.get("gate_results")) == _canonical_json_bytes(recomputed_gates),
            code="gate_results_recomputation_mismatch",
            message=("stored K0-K7 rows differ from independently recomputed thresholds, order, or decisions"),
        )
    except (ArtifactAuditError, KeyError, TypeError, ValueError) as error:
        audit.error(
            "analysis_recomputation_failed",
            f"cannot recompute aggregate metrics and gates: {error}",
        )


def _declare_current_coverage_gaps(
    audit: _Audit,
    result: Mapping[str, object],
    root: Path,
    *,
    verify_custody: bool,
) -> None:
    if result.get("lane") != "formal":
        audit.limitation(_DEVELOPMENT_EVIDENCE_INDEPENDENCE_LIMITATION)
    if not verify_custody:
        audit.gap(
            "producer_custody_not_verified",
            ("The producer manifest and its complete file inventory were not reopened in this audit invocation."),
            evidence_needed=("Run the default claim-facing audit with producer custody verification enabled."),
        )
    replicates = _mapping_rows(result.get("replicates"))
    has_evaluated_checkpoints = bool(replicates) and all(
        isinstance(replicate.get("evaluated_checkpoints"), list) and bool(replicate.get("evaluated_checkpoints"))
        for replicate in replicates
    )
    if not has_evaluated_checkpoints:
        audit.gap(
            "prediction_model_snapshot_unbound",
            (
                "Prediction tensors are recomputable, but the artifact has no immutable model-state "
                "snapshot for each evaluated condition, so the tensors cannot be proven to come from "
                "the claimed checkpoint."
            ),
            evidence_needed=(
                "Per-condition compound model/optimizer snapshot file refs, parameter and live-state "
                "digests, with each predictive sidecar header bound to that snapshot."
            ),
        )
    transition_rows = [
        transition for replicate in replicates for transition in _mapping_rows(replicate.get("transitions"))
    ]

    def has_observations(transition: Mapping[str, object]) -> bool:
        before = transition.get("pre_observation")
        after = transition.get("next_observation")
        return isinstance(before, list) and len(before) == 3 and isinstance(after, list) and len(after) == 3

    has_transition_sources = bool(transition_rows) and all(
        has_observations(transition) for transition in transition_rows
    )
    if not has_transition_sources:
        audit.gap(
            "transition_delta_source_unavailable",
            (
                "Raw transition rows retain scaled targets but omit pre- and next-observation values. "
                "Their delta target components therefore cannot be independently derived from reality."
            ),
            evidence_needed=(
                "Canonical pre-observation and next-observation vectors (or an independently bound raw "
                "environment trace) for every transition."
            ),
        )
    has_policy_runs = bool(replicates) and all(
        isinstance(replicate.get("policy_runs"), list) and bool(replicate.get("policy_runs"))
        for replicate in replicates
    )
    if not has_policy_runs:
        audit.gap(
            "controller_rng_consumption_unproven",
            (
                "Executed action hashes can be reconstructed, but no policy-run record binds the "
                "declared planner/random seeds, initial/final RNG states, draw counts, and action trace."
            ),
            evidence_needed=(
                "Per condition/task policy-run records with controller kind, derived seed, reset seeds, "
                "RNG start/end digests, draw/action counts, checkpoint binding, and action-trace digest."
            ),
        )
    has_rejected_full_state = bool(replicates) and all(
        len(
            [
                update
                for update in _mapping_rows(replicate.get("updates"))
                if update.get("phase") == "rejected_update_probe"
                and isinstance(update.get("full_state_before_file"), Mapping)
                and isinstance(update.get("full_state_after_file"), Mapping)
            ]
        )
        == 1
        for replicate in replicates
    )
    if not has_rejected_full_state:
        audit.gap(
            "rejected_probe_full_state_unavailable",
            (
                "The rejected-update control does not retain independently "
                "reopenable before/after snapshots of every live component."
            ),
            evidence_needed=(
                "Per replicate, exact before/after content-addressed rejected-probe "
                "state files covering model/optimizer, domain graph, replay, "
                "identities, and all RNGs."
            ),
        )
    formal = result.get("lane") == "formal"
    if formal:
        binding_digest = result.get("formal_binding_sha256")
        root_files = [path for path in root.iterdir() if path.is_file()]
        binding_present = isinstance(binding_digest, str) and any(
            hashlib.sha256(_read_bounded(path, 16 << 20, label="formal binding candidate")).hexdigest()
            == binding_digest
            for path in root_files
            if "binding" in path.name
        )
        if not binding_present:
            audit.gap(
                "formal_binding_not_self_contained",
                "The formal result names only a binding digest; the bound pre-run document is absent.",
                evidence_needed=(
                    "The exact formal binding, protocol, schemas, dependency lock, conformance report, "
                    "and test logs copied into the immutable artifact root before outcomes exist."
                ),
            )


def audit_artifact(
    artifact: str | Path,
    *,
    producer_bootstrap: str | Path,
    validate_schema: bool = True,
    require_claim_completeness: bool = True,
    verify_custody: bool = True,
) -> dict[str, object]:
    """Audit a WM-001 artifact directory or its ``result.json``.

    ``validate_schema=False`` exists for focused format tests and forensic
    recovery of partial attempts.  The CLI and all claim-facing callers use
    schema validation by default.
    """

    supplied = Path(artifact)
    result_path = supplied / "result.json" if supplied.is_dir() else supplied
    root = supplied if supplied.is_dir() else supplied.parent
    audit = _Audit()
    audit.require(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == _AUDITOR_SOURCE_SHA256,
        code="auditor_source_changed_during_execution",
        message="independent auditor source changed after module import",
    )
    try:
        _, _, producer_bootstrap_sha256, _ = _read_stable_regular_file(
            Path(producer_bootstrap),
            _MAX_SOURCE_FILE_BYTES,
            label="captured producer bootstrap support",
            capture_payload=False,
        )
    except (ArtifactAuditError, OSError) as error:
        audit.error("producer_bootstrap_support_invalid", str(error))
        return _audit_report(
            audit,
            root=root,
            result_path=result_path,
            result_sha256=None,
            lane=None,
            require_claim_completeness=require_claim_completeness,
        )
    if verify_custody:
        _verify_finalized_custody(audit, root)
    try:
        payload = _read_bounded(result_path, _MAX_RESULT_BYTES, label="WM-001 result")
        raw_result = _json_without_duplicate_keys(payload, label="WM-001 result")
        if not isinstance(raw_result, dict):
            raise ArtifactAuditError("WM-001 result root must be an object")
        result: Mapping[str, object] = raw_result
    except (ArtifactAuditError, OSError) as error:
        audit.error("result_unreadable", str(error))
        return _audit_report(
            audit,
            root=root,
            result_path=result_path,
            result_sha256=None,
            lane=None,
            require_claim_completeness=require_claim_completeness,
        )

    result_digest = hashlib.sha256(payload).hexdigest()
    if validate_schema:
        copied_schema = root / "schemas" / "raw-result.schema.json"
        schema_path = (
            copied_schema if result.get("lane") == "formal" and copied_schema.is_file() else RESULT_SCHEMA_PATH
        )
        _validate_result_schema(audit, result, schema_path=schema_path)
    audit.require(
        result.get("schema") == "prospect.world-model-lifecycle.raw-result.v9"
        and result.get("experiment_id") == "WM-001",
        code="result_identity_mismatch",
        message="artifact is not an active WM-001 raw-result v9 document",
    )
    protocol_source = root / "protocol.json" if (root / "protocol.json").is_file() else HERE / "protocol.json"
    protocol_value: Mapping[str, object] | None = None
    try:
        protocol_payload = _read_bounded(
            protocol_source,
            16 << 20,
            label="WM-001 protocol",
        )
        protocol_digest = hashlib.sha256(protocol_payload).hexdigest()
        protocol_raw = _json_without_duplicate_keys(
            protocol_payload,
            label="WM-001 protocol",
        )
        if not isinstance(protocol_raw, Mapping):
            raise ArtifactAuditError("WM-001 protocol root is not an object")
        protocol_value = protocol_raw
        _audit_protocol_seed_contract(audit, protocol_value)
    except ArtifactAuditError:
        protocol_digest = ""
    audit.require(
        result.get("protocol_version") == "1.16.0" and result.get("protocol_sha256") == protocol_digest,
        code="result_protocol_binding_mismatch",
        message="result does not bind the exact WM-001 protocol 1.16.0 bytes",
    )
    replicates = _mapping_rows(result.get("replicates"))
    audit.require(
        bool(replicates),
        code="replicates_missing",
        message="result has no auditable replicate rows",
    )
    bound_formal_device = _audit_formal_input_package(
        audit,
        root,
        result,
        producer_bootstrap_sha256=producer_bootstrap_sha256,
    )
    _audit_formal_schedule(audit, root, result)
    seen_replicates: set[str] = set()
    execution = result.get("execution")
    if validate_schema or (
        isinstance(execution, Mapping)
        and "producer_bootstrap_sha256" in execution
    ):
        audit.require(
            isinstance(execution, Mapping)
            and execution.get("producer_bootstrap_sha256")
            == producer_bootstrap_sha256,
            code="producer_bootstrap_support_mismatch",
            message=(
                "captured producer bootstrap support differs from the "
                "producer execution identity"
            ),
        )
    if result.get("lane") == "formal":
        # Formal arithmetic is selected only by the pre-outcome binding.  A
        # missing or malformed binding cannot silently fall back to a mutable
        # result field or to CPU.
        device = bound_formal_device if bound_formal_device is not None else "invalid-formal-binding-device"
    else:
        device = (
            str(execution.get("device"))
            if isinstance(execution, Mapping) and execution.get("device") in {"cpu", "cuda", "mps"}
            else "cpu"
        )
    for index, replicate in enumerate(replicates):
        raw_replicate_id = replicate.get("replicate_id")
        replicate_id = (
            raw_replicate_id if isinstance(raw_replicate_id, str) and raw_replicate_id else f"<replicate-{index}>"
        )
        audit.require(
            replicate_id not in seen_replicates,
            code="duplicate_replicate_id",
            message=f"duplicate replicate ID {replicate_id!r}",
            replicate_id=replicate_id,
        )
        seen_replicates.add(replicate_id)
        transitions_by_id = _audit_transition_and_episode_rows(
            audit,
            replicate,
            replicate_id=replicate_id,
        )
        evaluated_checkpoints = _audit_evaluated_checkpoints(
            audit,
            root,
            replicate,
            replicate_id=replicate_id,
        )
        _audit_predictions(
            audit,
            root,
            replicate,
            replicate_id=replicate_id,
            device=device,
            transitions_by_id=transitions_by_id,
            evaluated_checkpoints=evaluated_checkpoints,
        )
        _audit_optimizer_manifests(
            audit,
            root,
            replicate,
            replicate_id=replicate_id,
            transitions_by_id=transitions_by_id,
        )
        _audit_policy_runs(
            audit,
            replicate,
            replicate_id=replicate_id,
            transitions_by_id=transitions_by_id,
            evaluated_checkpoints=evaluated_checkpoints,
            device=device,
        )
        _audit_checkpoint(
            audit,
            root,
            replicate,
            replicate_id=replicate_id,
        )
        _audit_restart_parity_evidence(
            audit,
            root,
            replicate,
            replicate_id=replicate_id,
            launcher_process_id=(execution.get("launcher_process_id") if isinstance(execution, Mapping) else None),
            execution=(execution if isinstance(execution, Mapping) else {}),
            producer_bootstrap_sha256=producer_bootstrap_sha256,
            binding_runtime=audit.formal_runtime_binding,
            dependencies=audit.formal_dependency_binding,
            source=audit.formal_source_binding,
        )
    _audit_recomputed_analysis(
        audit,
        result,
        protocol=protocol_value,
    )
    _declare_current_coverage_gaps(
        audit,
        result,
        root,
        verify_custody=verify_custody,
    )
    return _audit_report(
        audit,
        root=root,
        result_path=result_path,
        result_sha256=result_digest,
        lane=(str(result["lane"]) if result.get("lane") in {"development", "formal"} else None),
        require_claim_completeness=require_claim_completeness,
    )


def _audit_report(
    audit: _Audit,
    *,
    root: Path,
    result_path: Path,
    result_sha256: str | None,
    lane: str | None,
    require_claim_completeness: bool,
) -> dict[str, object]:
    integrity_passed = audit.failed_checks == 0
    engineering_complete = not audit.coverage_gaps
    complete_for_claim = lane == "formal" and engineering_complete
    live_auditor_sha256 = _AUDITOR_SOURCE_SHA256
    audit_implementation: dict[str, object] = {
        "auditor_source_sha256": live_auditor_sha256,
        "bound_auditor_source_sha256": None,
        "formal_test_report_sha256": None,
        "coverage_conformance_report_sha256": None,
        "auditor_source_matches_binding": False,
        "coverage_conformance_verified": audit.coverage_conformance_verified,
        "audit_execution_conformance_verified": (audit.audit_execution_conformance_verified),
    }
    binding_path = root / "formal-binding.json"
    if binding_path.is_file() and not binding_path.is_symlink():
        try:
            binding_value = _json_without_duplicate_keys(
                _read_bounded(binding_path, 64 << 20, label="formal binding"),
                label="formal binding",
            )
            coverage_block = binding_value.get("coverage_arithmetic") if isinstance(binding_value, Mapping) else None
            if isinstance(coverage_block, Mapping):
                bound_auditor_sha256 = coverage_block.get("auditor_source_sha256")
                audit_implementation.update(
                    {
                        "bound_auditor_source_sha256": bound_auditor_sha256,
                        "formal_test_report_sha256": coverage_block.get("formal_test_report_sha256"),
                        "coverage_conformance_report_sha256": coverage_block.get("conformance_report_sha256"),
                        "auditor_source_matches_binding": (bound_auditor_sha256 == live_auditor_sha256),
                    }
                )
        except (ArtifactAuditError, OSError):
            pass
    return {
        "schema": "prospect.world-model-lifecycle.artifact-audit.v2",
        "artifact_root": str(root.resolve()),
        "result_file": result_path.name,
        "result_sha256": result_sha256,
        "lane": lane,
        "integrity_passed": integrity_passed,
        "engineering_complete": engineering_complete,
        "complete_for_claim": complete_for_claim,
        "passed": integrity_passed and (engineering_complete or not require_claim_completeness),
        "check_counts": {
            "passed": audit.passed_checks,
            "failed": audit.failed_checks,
            "coverage_gaps": len(audit.coverage_gaps),
        },
        "audit_implementation": audit_implementation,
        "audit_execution_conformance_verified": (audit.audit_execution_conformance_verified),
        "resource_limits_bytes": {
            "result": _MAX_RESULT_BYTES,
            "prediction_sidecar": _MAX_PREDICTION_BYTES,
            "owned_model_state": _MAX_OWNED_MODEL_BYTES,
            "optimizer_manifest": _MAX_MANIFEST_BYTES,
            "target_permutation": _MAX_PERMUTATION_BYTES,
            "restart_evaluation": _MAX_RESTART_EVALUATION_BYTES,
            "checkpoint_archive": _MAX_CHECKPOINT_BYTES,
            "source_file": _MAX_SOURCE_FILE_BYTES,
            "source_snapshot": _MAX_SOURCE_SNAPSHOT_BYTES,
        },
        "custody": audit.custody,
        "findings": audit.findings,
        "coverage_gaps": audit.coverage_gaps,
        "independence_limitations": audit.independence_limitations,
    }


class _PrebindingConformanceError(ValueError):
    """One stable, non-location-bearing prebinding failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _prebinding_fail(code: str) -> NoReturn:
    raise _PrebindingConformanceError(code)


def _prebinding_is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _prebinding_exact_keys(
    value: object,
    expected: set[str],
    *,
    code: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        _prebinding_fail(code)
    return cast(Mapping[str, object], value)


def _prebinding_canonical_name(value: str) -> str:
    canonical = re.sub(r"[-_.]+", "-", value).lower()
    if not canonical or canonical != value or len(canonical) > 256:
        _prebinding_fail("package_name_not_canonical")
    return canonical


def _prebinding_stable_regular_payload(
    raw_path: object,
    *,
    limit: int,
    code: str,
    locator_root: Path | None = None,
) -> bytes:
    if not isinstance(raw_path, str) or not raw_path or "\0" in raw_path:
        _prebinding_fail(code)
    path = Path(raw_path)
    if not path.is_absolute():
        if locator_root is None or ".." in path.parts or "." in path.parts or not path.parts:
            _prebinding_fail(code)
        path = locator_root / path
    try:
        if path.resolve(strict=True) != path:
            _prebinding_fail(code)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
    except (OSError, RuntimeError):
        _prebinding_fail(code)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1 or before.st_size < 0 or before.st_size > limit:
            _prebinding_fail(code)
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    except OSError:
        _prebinding_fail(code)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if len(payload) != before.st_size or identity != after_identity:
        _prebinding_fail(code)
    return payload


def _prebinding_request_identity(request: Mapping[str, object]) -> str:
    """Hash semantic request fields while redacting filesystem locators."""

    identity = json.loads(_canonical_json_bytes(request))
    if isinstance(identity, dict):
        protocol = identity.get("protocol")
        if isinstance(protocol, dict):
            protocol.pop("path", None)
            sources = protocol.get("scientific_source_files")
            if isinstance(sources, list):
                for row in sources:
                    if isinstance(row, dict):
                        row.pop("path", None)
        inventories = identity.get("root_inventories")
        if isinstance(inventories, list):
            for row in inventories:
                if isinstance(row, dict):
                    row.pop("path", None)
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


_PREBINDING_PRODUCER_FLAGS: Mapping[str, object] = {
    "dont_write_bytecode": 1,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "safe_path": True,
}
_PREBINDING_AUDITOR_FLAGS: Mapping[str, object] = {
    **_PREBINDING_PRODUCER_FLAGS,
}
_PREBINDING_PROCESS_ENVIRONMENT_KEYS = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "LAZY_LEGACY_OP",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYGAME_HIDE_SUPPORT_PROMPT",
        "ROCR_VISIBLE_DEVICES",
        "SDL_AUDIODRIVER",
        "TZ",
    }
)
_PREBINDING_SHARED_ENVIRONMENT_KEYS = _PREBINDING_PROCESS_ENVIRONMENT_KEYS - {"PATH"}
_PREBINDING_RUNTIME_SHARED_FIELDS = frozenset(
    {
        "python_implementation",
        "python_version",
        "python_cache_tag",
        "python_byteorder",
        "python_executable_sha256",
        "platform",
        "machine",
        "device",
        "torch_version",
        "accelerator",
        "cuda_available",
        "cuda_device_count",
        "cuda_capability",
        "cuda_runtime",
        "cuda_driver",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "deterministic_debug_mode",
        "thread_count",
        "interop_thread_count",
        "cudnn_version",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "global_fp32_precision",
        "cudnn_fp32_precision",
        "cudnn_conv_fp32_precision",
        "cudnn_rnn_fp32_precision",
        "cuda_matmul_fp32_precision",
    }
)
_PREBINDING_RUNTIME_REQUEST_FIELDS = _PREBINDING_RUNTIME_SHARED_FIELDS | {
    "producer_python_flags",
    "producer_process_environment",
}


def _prebinding_validate_request(value: object) -> Mapping[str, object]:
    request = _prebinding_exact_keys(
        value,
        {
            "schema",
            "protocol",
            "runtime",
            "packages",
            "root_inventories",
            "representative_tensor",
        },
        code="request_fields_invalid",
    )
    if request.get("schema") != _PREBINDING_REQUEST_SCHEMA:
        _prebinding_fail("request_schema_invalid")
    _prebinding_exact_keys(
        request.get("protocol"),
        {
            "path",
            "value",
            "sha256",
            "scientific_block_names",
            "scientific_kernel_sha256",
            "scientific_source_files",
        },
        code="protocol_request_fields_invalid",
    )
    _prebinding_exact_keys(
        request.get("runtime"),
        set(_PREBINDING_RUNTIME_REQUEST_FIELDS),
        code="runtime_request_fields_invalid",
    )
    tensor = _prebinding_exact_keys(
        request.get("representative_tensor"),
        {"repeat_count", "cpu_sha256", "cuda_sha256"},
        code="tensor_request_fields_invalid",
    )
    if tensor.get("repeat_count") != 3:
        _prebinding_fail("tensor_repeat_count_invalid")
    packages = request.get("packages")
    if not isinstance(packages, list) or not packages or len(packages) > _MAX_BOUND_PACKAGES:
        _prebinding_fail("package_request_invalid")
    inventories = request.get("root_inventories")
    if not isinstance(inventories, list) or not inventories or len(inventories) > 32:
        _prebinding_fail("root_inventory_request_invalid")
    return request


def _prebinding_protocol_component(
    block: object,
    *,
    locator_root: Path | None = None,
) -> dict[str, object]:
    protocol_request = _prebinding_exact_keys(
        block,
        {
            "path",
            "value",
            "sha256",
            "scientific_block_names",
            "scientific_kernel_sha256",
            "scientific_source_files",
        },
        code="protocol_request_fields_invalid",
    )
    payload = _prebinding_stable_regular_payload(
        protocol_request.get("path"),
        limit=_MAX_PREBINDING_REQUEST_BYTES,
        code="protocol_file_unavailable",
        locator_root=locator_root,
    )
    try:
        decoded = _json_without_duplicate_keys(payload, label="prebinding protocol")
    except (ArtifactAuditError, TypeError, ValueError):
        _prebinding_fail("protocol_json_invalid")
    if not isinstance(decoded, Mapping) or decoded != protocol_request.get("value"):
        _prebinding_fail("protocol_value_mismatch")
    protocol_sha256 = hashlib.sha256(payload).hexdigest()
    if not _prebinding_is_sha256(protocol_request.get("sha256")) or protocol_request.get("sha256") != protocol_sha256:
        _prebinding_fail("protocol_sha256_mismatch")
    experiment = decoded.get("experiment")
    if (
        decoded.get("schema") != "prospect.world-model-lifecycle.protocol.v9"
        or not isinstance(experiment, Mapping)
        or experiment.get("id") != "WM-001"
        or experiment.get("protocol_version") != "1.16.0"
    ):
        _prebinding_fail("protocol_identity_mismatch")
    revision = experiment.get("revision")
    if (
        not isinstance(revision, Mapping)
        or revision.get("supersedes") != "1.15.0"
        or revision.get("superseded_protocol_sha256")
        != _V1150_PROTOCOL_SHA256
    ):
        _prebinding_fail("protocol_lineage_mismatch")
    bindings = decoded.get("bindings")
    development_qualification = (
        bindings.get("development_qualification")
        if isinstance(bindings, Mapping)
        else None
    )
    if (
        not isinstance(development_qualification, Mapping)
        or development_qualification.get("matrix_contract_sha256")
        != _DEVELOPMENT_MATRIX_CONTRACT_SHA256
    ):
        _prebinding_fail("development_matrix_contract_mismatch")
    if (
        not isinstance(bindings, Mapping)
        or bindings.get("preformal_authorization")
        != _PREFORMAL_AUTHORIZATION_CONTRACT
    ):
        _prebinding_fail("preformal_authorization_contract_mismatch")
    continuity = revision.get("scientific_continuity") if isinstance(revision, Mapping) else None
    if not isinstance(continuity, Mapping):
        _prebinding_fail("scientific_continuity_missing")
    block_names = protocol_request.get("scientific_block_names")
    if (
        not isinstance(block_names, list)
        or tuple(block_names) != _PREBINDING_SCIENTIFIC_BLOCKS
        or tuple(continuity.get("unchanged_top_level_blocks", ())) != _PREBINDING_SCIENTIFIC_BLOCKS
    ):
        _prebinding_fail("scientific_block_contract_mismatch")
    try:
        scientific_value = {name: decoded[name] for name in _PREBINDING_SCIENTIFIC_BLOCKS}
    except KeyError:
        _prebinding_fail("scientific_block_missing")
    scientific_sha256 = hashlib.sha256(_canonical_json_bytes(scientific_value)).hexdigest()
    if (
        not _prebinding_is_sha256(protocol_request.get("scientific_kernel_sha256"))
        or protocol_request.get("scientific_kernel_sha256") != scientific_sha256
        or continuity.get("v1_4_scientific_blocks_sha256") != scientific_sha256
    ):
        _prebinding_fail("scientific_kernel_sha256_mismatch")
    expected_source_hashes = continuity.get("kernel_source_sha256")
    if not isinstance(expected_source_hashes, Mapping):
        _prebinding_fail("scientific_source_contract_missing")
    rows = protocol_request.get("scientific_source_files")
    if not isinstance(rows, list) or len(rows) != len(_PREBINDING_SCIENTIFIC_SOURCES):
        _prebinding_fail("scientific_source_rows_invalid")
    actual_source_hashes: dict[str, str] = {}
    for index, row_value in enumerate(rows):
        row = _prebinding_exact_keys(
            row_value,
            {"name", "path", "sha256"},
            code="scientific_source_row_invalid",
        )
        name = row.get("name")
        if name != _PREBINDING_SCIENTIFIC_SOURCES[index]:
            _prebinding_fail("scientific_source_order_invalid")
        source_payload = _prebinding_stable_regular_payload(
            row.get("path"),
            limit=_MAX_SOURCE_FILE_BYTES,
            code="scientific_source_unavailable",
            locator_root=locator_root,
        )
        digest = hashlib.sha256(source_payload).hexdigest()
        if row.get("sha256") != digest or expected_source_hashes.get(name) != digest:
            _prebinding_fail("scientific_source_sha256_mismatch")
        actual_source_hashes[cast(str, name)] = digest
    if set(expected_source_hashes) != set(actual_source_hashes):
        _prebinding_fail("scientific_source_set_mismatch")
    identity = {
        "protocol_sha256": protocol_sha256,
        "scientific_kernel_sha256": scientific_sha256,
        "scientific_source_sha256": actual_source_hashes,
    }
    return {
        "passed": True,
        **identity,
        "identity_sha256": hashlib.sha256(_canonical_json_bytes(identity)).hexdigest(),
    }


def _prebinding_pendulum_component() -> dict[str, object]:
    """Exercise the fixed 1,024-case corpus without producer dynamics code."""

    try:
        import gymnasium as gym
        import torch
    except ImportError:
        _prebinding_fail("pendulum_dependency_unavailable")
    torch.use_deterministic_algorithms(True)
    env = gym.make("Pendulum-v1")
    unwrapped: Any = env.unwrapped
    parameter_errors = {
        name: abs(float(getattr(unwrapped, name)) - expected)
        for name, expected in _FORMAL_CONFORMANCE_PARAMETERS.items()
    }
    rng = np.random.default_rng(20260717)
    max_observation_error = 0.0
    max_reward_error = 0.0
    max_planner_observation_error = 0.0
    max_planner_reward_error = 0.0
    terminated_or_truncated = 0
    trajectory = hashlib.sha256()
    try:
        env.reset(seed=20260717)
        for context in (0.0, 1.0):
            for _ in range(512):
                theta = float(rng.uniform(-math.pi, math.pi))
                angular_velocity = float(rng.uniform(-10.0, 10.0))
                intended = float(np.float32(rng.uniform(-3.0, 3.0)))
                unwrapped.state = np.asarray(
                    [theta, angular_velocity],
                    dtype=np.float64,
                )
                applied = intended if context == 0.0 else -intended
                actual_observation, actual_reward, terminated, truncated, _ = unwrapped.step(
                    np.asarray([applied], dtype=np.float32)
                )

                clipped = min(2.0, max(-2.0, applied))
                normalized_theta = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
                expected_reward = -(
                    normalized_theta * normalized_theta
                    + 0.1 * angular_velocity * angular_velocity
                    + 0.001 * clipped * clipped
                )
                expected_velocity = min(
                    8.0,
                    max(
                        -8.0,
                        angular_velocity + (15.0 * math.sin(theta) + 3.0 * clipped) * 0.05,
                    ),
                )
                expected_theta = theta + expected_velocity * 0.05
                expected_observation = np.asarray(
                    [
                        math.cos(expected_theta),
                        math.sin(expected_theta),
                        expected_velocity,
                    ],
                    dtype=np.float64,
                )

                physical = torch.tensor(
                    [
                        math.cos(theta),
                        math.sin(theta),
                        angular_velocity,
                    ],
                    dtype=torch.float32,
                )
                action = torch.tensor([intended], dtype=torch.float32)
                encoded_context = torch.tensor(context, dtype=torch.float32)
                p_theta = torch.atan2(physical[1], physical[0])
                p_normalized = torch.remainder(p_theta + torch.pi, 2.0 * torch.pi) - torch.pi
                p_intended = action[0].clamp(-2.0, 2.0)
                direction = torch.where(
                    encoded_context >= 0.5,
                    p_intended.new_tensor(-1.0),
                    p_intended.new_tensor(1.0),
                )
                p_applied = p_intended * direction
                p_reward = -(p_normalized.square() + 0.1 * physical[2].square() + 0.001 * p_applied.square())
                p_acceleration = 15.0 * torch.sin(p_theta) + 3.0 * p_applied
                p_velocity = (physical[2] + p_acceleration * 0.05).clamp(-8.0, 8.0)
                p_next_theta = p_theta + p_velocity * 0.05
                planner_observation = torch.stack(
                    (
                        torch.cos(p_next_theta),
                        torch.sin(p_next_theta),
                        p_velocity,
                    )
                )

                actual64 = np.asarray(actual_observation, dtype=np.float64)
                max_observation_error = max(
                    max_observation_error,
                    float(np.max(np.abs(actual64 - expected_observation))),
                )
                max_reward_error = max(
                    max_reward_error,
                    abs(float(actual_reward) - expected_reward),
                )
                max_planner_observation_error = max(
                    max_planner_observation_error,
                    float(np.max(np.abs(actual64 - planner_observation.cpu().numpy().astype(np.float64)))),
                )
                max_planner_reward_error = max(
                    max_planner_reward_error,
                    abs(float(actual_reward) - float(p_reward.item())),
                )
                terminated_or_truncated += int(bool(terminated) or bool(truncated))
                trajectory.update(
                    struct.pack(
                        "<3fdi",
                        *(
                            float(item)
                            for item in np.asarray(
                                actual_observation,
                                dtype=np.float32,
                            )
                        ),
                        float(actual_reward),
                        int(context),
                    )
                )
    except Exception:
        _prebinding_fail("pendulum_execution_failed")
    finally:
        env.close()
    body: dict[str, object] = {
        "environment_id": "Pendulum-v1",
        "gymnasium_version": gym.__version__,
        "seed": 20260717,
        "samples_per_task": 512,
        "cases": 1024,
        "semantic_parameters": dict(_FORMAL_CONFORMANCE_PARAMETERS),
        "semantic_parameter_absolute_errors": parameter_errors,
        "spec_horizon": getattr(env.spec, "max_episode_steps", None),
        "max_observation_absolute_error": max_observation_error,
        "max_reward_absolute_error": max_reward_error,
        "planner_dtype": "float32",
        "max_planner_observation_absolute_error": (max_planner_observation_error),
        "max_planner_reward_absolute_error": max_planner_reward_error,
        "terminated_or_truncated_cases": terminated_or_truncated,
        **_FORMAL_CONFORMANCE_TOLERANCES,
        "trajectory_sha256": trajectory.hexdigest(),
    }
    passed = (
        all(value == 0.0 for value in parameter_errors.values())
        and body["spec_horizon"] == 200
        and terminated_or_truncated == 0
        and max_observation_error <= _FORMAL_CONFORMANCE_TOLERANCES["observation_atol"]
        and max_reward_error <= _FORMAL_CONFORMANCE_TOLERANCES["reward_atol"]
        and max_planner_observation_error <= _FORMAL_CONFORMANCE_TOLERANCES["planner_observation_atol"]
        and max_planner_reward_error <= _FORMAL_CONFORMANCE_TOLERANCES["planner_reward_atol"]
    )
    return {
        **body,
        "identity_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
        "passed": passed,
    }


def _prebinding_oscillator_component() -> dict[str, object]:
    report = _expected_formal_oscillator_conformance()
    body = dict(report)
    report_sha256 = body.pop("report_sha256")
    if report_sha256 != hashlib.sha256(_canonical_json_bytes(body)).hexdigest():
        _prebinding_fail("oscillator_self_hash_failed")
    return {
        "source_id": body["source_id"],
        "cases": body["cases"],
        "steps_per_case": body["steps_per_case"],
        "seed": body["seed"],
        "trajectory_sha256": body["trajectory_sha256"],
        "identity_sha256": cast(str, report_sha256),
        "passed": body["passed"] is True,
    }


def _prebinding_coverage_component() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    direct_cases = (
        ("lower-binary64", 0.05, True),
        ("lower-predecessor", math.nextafter(0.05, -math.inf), False),
        ("lower-successor", math.nextafter(0.05, math.inf), True),
        ("upper-binary64", 0.95, True),
        ("upper-predecessor", math.nextafter(0.95, -math.inf), True),
        ("upper-successor", math.nextafter(0.95, math.inf), False),
        ("central", 0.5, True),
        ("zero-tail", 0.0, False),
        ("one-tail", 1.0, False),
    )
    passed = True
    for case_id, pit, expected in direct_cases:
        observed = 0.05 <= pit <= 0.95
        row_passed = observed is expected
        rows.append(
            {
                "case_id": case_id,
                "pit_hex": pit.hex(),
                "expected_covered": expected,
                "observed_covered": observed,
                "passed": row_passed,
            }
        )
        passed = passed and row_passed
    target = float(struct.unpack("<f", bytes.fromhex(_V130_BOUNDARY_TARGET_F32_HEX))[0])
    means = tuple(float(struct.unpack("<f", bytes.fromhex(value))[0]) for value in _V130_BOUNDARY_MEANS_F32_HEX)
    log_variances = tuple(
        float(struct.unpack("<f", bytes.fromhex(value))[0]) for value in _V130_BOUNDARY_LOG_VARIANCES_F32_HEX
    )
    cdfs = [
        0.5 * (1.0 + math.erf(((target - mean) * math.exp(-0.5 * log_variance)) / math.sqrt(2.0)))
        for mean, log_variance in zip(
            means,
            log_variances,
            strict=True,
        )
    ]
    boundary_pit = math.fsum(cdfs) / len(cdfs)
    boundary_covered = 0.05 <= boundary_pit <= 0.95
    boundary_passed = boundary_pit.hex() == "0x1.999998b3745adp-5" and boundary_covered is False
    rows.append(
        {
            "case_id": "v130-disclosed-boundary-coordinate",
            "pit_hex": boundary_pit.hex(),
            "expected_pit_hex": "0x1.999998b3745adp-5",
            "expected_covered": False,
            "observed_covered": boundary_covered,
            "passed": boundary_passed,
        }
    )
    passed = passed and boundary_passed
    corpus = {
        "semantics_id": _COVERAGE_SEMANTICS,
        "cases": rows,
    }
    corpus_sha256 = hashlib.sha256(_canonical_json_bytes(corpus)).hexdigest()
    return {
        "semantics_id": _COVERAGE_SEMANTICS,
        "cases": len(rows),
        "corpus_sha256": corpus_sha256,
        "identity_sha256": corpus_sha256,
        "passed": passed,
    }


def _prebinding_live_runtime_identity(device: str) -> dict[str, object]:
    try:
        import torch
    except ImportError:
        _prebinding_fail("torch_unavailable")
    if device not in {"cpu", "cuda"}:
        _prebinding_fail("runtime_device_invalid")
    if device == "cuda" and not torch.cuda.is_available():
        _prebinding_fail("cuda_unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    cuda_available = bool(torch.cuda.is_available())
    cudnn = torch.backends.cudnn
    cuda_matmul = torch.backends.cuda.matmul
    executable = Path(sys.executable).resolve()
    try:
        executable_sha256 = _sha256_file(executable)
    except OSError:
        _prebinding_fail("python_executable_unavailable")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_byteorder": sys.byteorder,
        "python_executable_sha256": executable_sha256,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "torch_version": str(torch.__version__),
        "accelerator": (torch.cuda.get_device_name(0) if device == "cuda" else None),
        "cuda_available": cuda_available,
        "cuda_device_count": (int(torch.cuda.device_count()) if cuda_available else 0),
        "cuda_capability": (list(torch.cuda.get_device_capability(0)) if device == "cuda" else None),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": (_cuda_driver_version() if cuda_available else None),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "thread_count": int(torch.get_num_threads()),
        "interop_thread_count": int(torch.get_num_interop_threads()),
        "cudnn_version": (int(cudnn.version()) if cudnn.is_available() else None),
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "global_fp32_precision": str(torch.backends.fp32_precision),
        "cudnn_fp32_precision": str(cudnn.fp32_precision),
        "cudnn_conv_fp32_precision": str(cudnn.conv.fp32_precision),
        "cudnn_rnn_fp32_precision": str(cudnn.rnn.fp32_precision),
        "cuda_matmul_fp32_precision": str(cuda_matmul.fp32_precision),
    }


def _prebinding_live_python_flags() -> dict[str, object]:
    return {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    }


def _prebinding_runtime_component(expected: object) -> dict[str, object]:
    expected_runtime = _prebinding_exact_keys(
        expected,
        set(_PREBINDING_RUNTIME_REQUEST_FIELDS),
        code="runtime_request_fields_invalid",
    )
    device = expected_runtime.get("device")
    if not isinstance(device, str):
        _prebinding_fail("runtime_device_invalid")
    actual_shared = _prebinding_live_runtime_identity(device)
    producer_flags = expected_runtime.get("producer_python_flags")
    producer_environment = expected_runtime.get("producer_process_environment")
    if not isinstance(producer_flags, Mapping) or dict(producer_flags) != dict(_PREBINDING_PRODUCER_FLAGS):
        _prebinding_fail("producer_python_flags_invalid")
    if not isinstance(producer_environment, Mapping):
        _prebinding_fail("producer_process_environment_invalid")
    if (
        not {
            "CUBLAS_WORKSPACE_CONFIG",
            "LAZY_LEGACY_OP",
            "LC_ALL",
            "PATH",
            "PYGAME_HIDE_SUPPORT_PROMPT",
            "SDL_AUDIODRIVER",
            "TZ",
        }.issubset(producer_environment)
        or not set(producer_environment).issubset(_PREBINDING_PROCESS_ENVIRONMENT_KEYS)
        or producer_environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or producer_environment.get("LAZY_LEGACY_OP") != "False"
        or producer_environment.get("LC_ALL") != "C.UTF-8"
        or producer_environment.get("PATH") != "/usr/bin:/bin"
        or producer_environment.get("PYGAME_HIDE_SUPPORT_PROMPT") != "hide"
        or producer_environment.get("SDL_AUDIODRIVER") != "dsp"
        or producer_environment.get("TZ") != "UTC"
        or any(
            not isinstance(key, str) or not isinstance(value, str) or "\0" in key or "\0" in value
            for key, value in producer_environment.items()
        )
    ):
        _prebinding_fail("producer_process_environment_invalid")
    expected_shared = {key: expected_runtime[key] for key in _PREBINDING_RUNTIME_SHARED_FIELDS}
    auditor_flags = _prebinding_live_python_flags()
    shared_environment_matches = all(
        os.environ.get(key) == value
        for key, value in producer_environment.items()
        if key in _PREBINDING_SHARED_ENVIRONMENT_KEYS
    )
    identity = {
        "producer_python_flags": dict(producer_flags),
        "producer_process_environment": dict(producer_environment),
        "shared_runtime": actual_shared,
        "auditor_python_flags": auditor_flags,
    }
    identity_sha256 = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return {
        "identity_sha256": identity_sha256,
        "python_executable_sha256": actual_shared["python_executable_sha256"],
        "device": actual_shared["device"],
        "torch_version": actual_shared["torch_version"],
        "cuda_runtime": actual_shared["cuda_runtime"],
        "cuda_driver": actual_shared["cuda_driver"],
        "producer_python_flags": dict(producer_flags),
        "auditor_python_flags": auditor_flags,
        "shared_environment_matches": shared_environment_matches,
        "passed": (
            actual_shared == expected_shared
            and auditor_flags == dict(_PREBINDING_AUDITOR_FLAGS)
            and shared_environment_matches
        ),
    }


def _prebinding_distribution_digest(
    distribution: importlib.metadata.Distribution,
    *,
    canonical_name: str,
) -> tuple[str, int, int, bool]:
    declared = list(distribution.files or ())
    selected = [entry for entry in declared if "__pycache__" not in entry.parts and entry.suffix != ".pyc"]
    if not selected:
        _prebinding_fail("distribution_files_missing")
    if len(selected) > _MAX_PREBINDING_ROOT_ENTRIES:
        _prebinding_fail("distribution_file_limit_exceeded")
    digest = hashlib.sha256(_PREBINDING_DISTRIBUTION_DOMAIN)
    digest.update(canonical_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(distribution.version.encode("utf-8"))
    total_bytes = 0
    editable = False
    seen: set[str] = set()
    for entry in sorted(selected, key=lambda item: str(item)):
        relative = str(entry).replace("\\", "/")
        if not relative or relative in seen:
            _prebinding_fail("distribution_file_identity_invalid")
        seen.add(relative)
        try:
            path = Path(str(distribution.locate_file(entry)))
            if path.is_symlink():
                _prebinding_fail("distribution_file_aliased")
            before = path.lstat()
        except (OSError, RuntimeError):
            _prebinding_fail("distribution_file_unavailable")
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_size < 0:
            _prebinding_fail("distribution_file_not_regular")
        captured_direct_url = bytearray()
        try:
            with path.open("rb") as stream:
                digest.update(b"\0")
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(before.st_size.to_bytes(8, "big"))
                digest.update(b"\0")
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
                    if relative.endswith(".dist-info/direct_url.json"):
                        captured_direct_url.extend(chunk)
            after = path.lstat()
        except OSError:
            _prebinding_fail("distribution_file_unavailable")
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _prebinding_fail("distribution_file_changed")
        total_bytes += before.st_size
        if total_bytes > _MAX_PREBINDING_ROOT_BYTES:
            _prebinding_fail("distribution_byte_limit_exceeded")
        if captured_direct_url:
            try:
                direct_url = json.loads(bytes(captured_direct_url).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                _prebinding_fail("distribution_direct_url_invalid")
            directory_info = direct_url.get("dir_info") if isinstance(direct_url, Mapping) else None
            editable = editable or (isinstance(directory_info, Mapping) and directory_info.get("editable") is True)
    return digest.hexdigest(), len(selected), total_bytes, editable


def _prebinding_live_package_rows() -> list[dict[str, object]]:
    distributions: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not isinstance(raw_name, str) or not raw_name:
            _prebinding_fail("distribution_name_missing")
        canonical = re.sub(r"[-_.]+", "-", raw_name).lower()
        distributions.setdefault(canonical, []).append(distribution)
    if any(len(rows) != 1 for rows in distributions.values()):
        _prebinding_fail("duplicate_distribution_identity")
    try:
        python_digest = _sha256_file(Path(sys.executable).resolve())
    except OSError:
        _prebinding_fail("python_executable_unavailable")
    result: list[dict[str, object]] = [
        {
            "name": "python",
            "version": platform.python_version(),
            "distribution_sha256": python_digest,
            "declared_file_count": 1,
            "editable": False,
        }
    ]
    for name in sorted(distributions):
        distribution = distributions[name][0]
        digest, declared_file_count, _, editable = _prebinding_distribution_digest(
            distribution,
            canonical_name=name,
        )
        if editable:
            _prebinding_fail("editable_distribution_forbidden")
        result.append(
            {
                "name": name,
                "version": distribution.version,
                "distribution_sha256": digest,
                "declared_file_count": declared_file_count,
                "editable": False,
            }
        )
    return result


def _prebinding_packages_component(expected: object) -> dict[str, object]:
    if not isinstance(expected, list) or not expected:
        _prebinding_fail("package_request_invalid")
    expected_rows: list[dict[str, object]] = []
    names: list[str] = []
    for row_value in expected:
        row = _prebinding_exact_keys(
            row_value,
            {
                "name",
                "version",
                "distribution_sha256",
                "declared_file_count",
                "editable",
            },
            code="package_row_invalid",
        )
        name = row.get("name")
        if not isinstance(name, str):
            _prebinding_fail("package_name_invalid")
        _prebinding_canonical_name(name)
        if (
            not isinstance(row.get("version"), str)
            or not row.get("version")
            or not _prebinding_is_sha256(row.get("distribution_sha256"))
            or type(row.get("declared_file_count")) is not int
            or cast(int, row.get("declared_file_count")) < 1
            or row.get("editable") is not False
        ):
            _prebinding_fail("package_row_invalid")
        names.append(name)
        expected_rows.append(dict(row))
    if names != ["python", *sorted(name for name in names if name != "python")] or len(names) != len(set(names)):
        _prebinding_fail("package_order_invalid")
    actual_rows = _prebinding_live_package_rows()
    identity = {
        "semantics_id": "prospect.wm001.distribution.v2",
        "packages": actual_rows,
    }
    digest = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return {
        "semantics_id": "prospect.wm001.distribution.v2",
        "package_count": len(actual_rows),
        "packages_sha256": digest,
        "identity_sha256": digest,
        "passed": actual_rows == expected_rows,
    }


def _prebinding_root_inventory(
    identifier: str,
    raw_path: object,
    *,
    kind: str,
) -> dict[str, object]:
    if not identifier or len(identifier) > 128 or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", identifier) is None:
        _prebinding_fail("root_inventory_id_invalid")
    if not isinstance(raw_path, str) or not raw_path or "\0" in raw_path:
        _prebinding_fail("root_inventory_path_invalid")
    if kind not in {"package_root", "standard_library"}:
        _prebinding_fail("root_inventory_kind_invalid")
    root = Path(raw_path)
    try:
        if not root.is_absolute() or root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
            _prebinding_fail("root_inventory_path_invalid")
    except OSError:
        _prebinding_fail("root_inventory_path_invalid")

    def discover() -> tuple[
        list[tuple[str, Path, tuple[int, ...]]],
        list[tuple[str, tuple[int, ...]]],
    ]:
        files: list[tuple[str, Path, tuple[int, ...]]] = []
        directories: list[tuple[str, tuple[int, ...]]] = []
        try:
            root_metadata = root.lstat()
        except OSError:
            _prebinding_fail("root_inventory_changed")
        directories.append(
            (
                "",
                (
                    root_metadata.st_dev,
                    root_metadata.st_ino,
                    root_metadata.st_mode,
                    root_metadata.st_size,
                    root_metadata.st_mtime_ns,
                    root_metadata.st_ctime_ns,
                ),
            )
        )
        for current, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            current_path = Path(current)
            directory_names.sort()
            file_names.sort()
            retained_directories: list[str] = []
            for name in directory_names:
                candidate = current_path / name
                try:
                    metadata = candidate.lstat()
                except OSError:
                    _prebinding_fail("root_inventory_entry_unavailable")
                if stat.S_ISLNK(metadata.st_mode):
                    _prebinding_fail("root_inventory_non_regular_entry")
                if not stat.S_ISDIR(metadata.st_mode):
                    _prebinding_fail("root_inventory_non_regular_entry")
                if kind == "standard_library" and name == "site-packages":
                    continue
                relative = candidate.relative_to(root).as_posix()
                directories.append(
                    (
                        relative,
                        (
                            metadata.st_dev,
                            metadata.st_ino,
                            metadata.st_mode,
                            metadata.st_size,
                            metadata.st_mtime_ns,
                            metadata.st_ctime_ns,
                        ),
                    )
                )
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in file_names:
                candidate = current_path / name
                try:
                    metadata = candidate.lstat()
                except OSError:
                    _prebinding_fail("root_inventory_entry_unavailable")
                if stat.S_ISLNK(metadata.st_mode):
                    _prebinding_fail("root_inventory_non_regular_entry")
                if not stat.S_ISREG(metadata.st_mode):
                    _prebinding_fail("root_inventory_non_regular_entry")
                relative = candidate.relative_to(root).as_posix()
                files.append(
                    (
                        relative,
                        candidate,
                        (
                            metadata.st_dev,
                            metadata.st_ino,
                            metadata.st_mode,
                            metadata.st_size,
                            metadata.st_mtime_ns,
                            metadata.st_ctime_ns,
                        ),
                    )
                )
                if len(files) > _MAX_PREBINDING_ROOT_ENTRIES:
                    _prebinding_fail("root_inventory_limit_exceeded")
        return (
            sorted(files, key=lambda row: row[0]),
            sorted(directories, key=lambda row: row[0]),
        )

    initial_files, initial_directories = discover()
    digest = hashlib.sha256(_PREBINDING_PACKAGE_ROOT_DOMAIN if kind == "package_root" else _PREBINDING_STDLIB_DOMAIN)
    file_count = 0
    directory_count = 0
    total_bytes = 0
    file_entries = {relative: (candidate, identity) for relative, candidate, identity in initial_files}
    directory_entries = {relative: identity for relative, identity in initial_directories if relative}
    for relative in sorted({*file_entries, *directory_entries}):
        if relative in directory_entries:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0directory\0")
            directory_count += 1
            continue
        candidate, expected_identity = file_entries[relative]
        expected_size = expected_identity[3]
        try:
            before = candidate.lstat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != expected_identity:
                _prebinding_fail("root_inventory_entry_changed")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(b"file")
            digest.update(b"\0")
            digest.update(expected_size.to_bytes(8, "big"))
            digest.update(b"\0")
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            digest.update(b"\0")
            after = candidate.lstat()
        except OSError:
            _prebinding_fail("root_inventory_entry_unavailable")
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != expected_identity:
            _prebinding_fail("root_inventory_entry_changed")
        file_count += 1
        total_bytes += expected_size
        if total_bytes > _MAX_PREBINDING_ROOT_BYTES:
            _prebinding_fail("root_inventory_limit_exceeded")
    final_files, final_directories = discover()
    if [(relative, identity) for relative, _, identity in initial_files] != [
        (relative, identity) for relative, _, identity in final_files
    ] or initial_directories != final_directories:
        _prebinding_fail("root_inventory_changed")
    if file_count == 0:
        _prebinding_fail("root_inventory_empty")
    return {
        "id": identifier,
        "kind": kind,
        "semantics_id": (
            "prospect.wm001.package-root.v2" if kind == "package_root" else "prospect.wm001.standard-library.v2"
        ),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def _prebinding_root_inventories_component(
    expected: object,
) -> dict[str, object]:
    if not isinstance(expected, list) or not expected:
        _prebinding_fail("root_inventory_request_invalid")
    actual_rows: list[dict[str, object]] = []
    public_expected: list[dict[str, object]] = []
    identifiers: list[str] = []
    for row_value in expected:
        row = _prebinding_exact_keys(
            row_value,
            {
                "id",
                "kind",
                "semantics_id",
                "path",
                "file_count",
                "directory_count",
                "total_bytes",
                "inventory_sha256",
            },
            code="root_inventory_row_invalid",
        )
        identifier = row.get("id")
        kind = row.get("kind")
        if not isinstance(identifier, str):
            _prebinding_fail("root_inventory_id_invalid")
        if (
            kind not in {"package_root", "standard_library"}
            or row.get("semantics_id")
            != ("prospect.wm001.package-root.v2" if kind == "package_root" else "prospect.wm001.standard-library.v2")
            or type(row.get("file_count")) is not int
            or type(row.get("directory_count")) is not int
            or type(row.get("total_bytes")) is not int
            or not _prebinding_is_sha256(row.get("inventory_sha256"))
        ):
            _prebinding_fail("root_inventory_row_invalid")
        identifiers.append(identifier)
        actual_rows.append(
            _prebinding_root_inventory(
                identifier,
                row.get("path"),
                kind=cast(str, kind),
            )
        )
        public_expected.append({key: value for key, value in row.items() if key != "path"})
    if identifiers != sorted(set(identifiers)):
        _prebinding_fail("root_inventory_order_invalid")
    identity = {
        "roots": actual_rows,
    }
    digest = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return {
        "root_count": len(actual_rows),
        "roots": actual_rows,
        "root_inventories_sha256": digest,
        "identity_sha256": digest,
        "passed": actual_rows == public_expected,
    }


def _prebinding_tensor_digest(
    tensor: Any,
) -> str:
    value = tensor.detach().contiguous().cpu()
    header = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    digest = hashlib.sha256(_canonical_json_bytes(header))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _prebinding_representative_tensor_identity(
    device: str,
) -> dict[str, object]:
    try:
        import torch
    except ImportError:
        _prebinding_fail("torch_unavailable")
    torch.use_deterministic_algorithms(True)

    def execute(target: str) -> str:
        selected = torch.device(target)
        dtype = torch.float64 if target == "cpu" else torch.float32
        left = torch.linspace(
            -1.75,
            1.75,
            steps=1024,
            dtype=dtype,
            device=selected,
        ).reshape(32, 32)
        right = torch.linspace(
            0.875,
            -0.625,
            steps=1024,
            dtype=dtype,
            device=selected,
        ).reshape(32, 32)
        value = torch.tanh((left @ right.mT) / 32.0)
        value = value + 0.125 * torch.sin(left)
        if target == "cuda":
            torch.cuda.synchronize()
        return _prebinding_tensor_digest(value)

    cpu_digests = [execute("cpu") for _ in range(3)]
    if len(set(cpu_digests)) != 1:
        _prebinding_fail("cpu_tensor_nondeterministic")
    cuda_digest: str | None = None
    if device == "cuda":
        cuda_digests = [execute("cuda") for _ in range(3)]
        if len(set(cuda_digests)) != 1:
            _prebinding_fail("cuda_tensor_nondeterministic")
        cuda_digest = cuda_digests[0]
    return {
        "repeat_count": 3,
        "cpu_sha256": cpu_digests[0],
        "cuda_sha256": cuda_digest,
    }


def _prebinding_tensor_component(
    expected: object,
    *,
    device: str,
) -> dict[str, object]:
    expected_tensor = _prebinding_exact_keys(
        expected,
        {"repeat_count", "cpu_sha256", "cuda_sha256"},
        code="tensor_request_fields_invalid",
    )
    actual = _prebinding_representative_tensor_identity(device)
    identity_sha256 = hashlib.sha256(_canonical_json_bytes(actual)).hexdigest()
    return {
        **actual,
        "identity_sha256": identity_sha256,
        "passed": actual == dict(expected_tensor),
    }


def build_prebinding_conformance_request(
    protocol_path: Path,
    *,
    scientific_source_paths: Mapping[str, Path],
    root_paths: Mapping[str, Path],
    device: str,
    support_locator_root: Path | None = None,
    runtime: Mapping[str, object] | None = None,
    package_rows: Sequence[Mapping[str, object]] | None = None,
    representative_tensor: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the exact canonical-value request consumed by prebinding mode.

    The optional identity arguments let a separately implemented binder supply
    its own expected values.  When omitted, this helper computes the live
    values using the same public contract, which is useful for development and
    for constructing the first seal before isolated subprocess replay.
    """

    payload = _prebinding_stable_regular_payload(
        str(protocol_path.resolve()),
        limit=_MAX_PREBINDING_REQUEST_BYTES,
        code="protocol_file_unavailable",
    )
    try:
        protocol_value = _json_without_duplicate_keys(
            payload,
            label="prebinding protocol",
        )
    except (ArtifactAuditError, TypeError, ValueError):
        _prebinding_fail("protocol_json_invalid")
    if not isinstance(protocol_value, Mapping):
        _prebinding_fail("protocol_value_mismatch")
    scientific_value: dict[str, object] = {}
    try:
        for name in _PREBINDING_SCIENTIFIC_BLOCKS:
            scientific_value[name] = protocol_value[name]
    except KeyError:
        _prebinding_fail("scientific_block_missing")
    if set(scientific_source_paths) != set(_PREBINDING_SCIENTIFIC_SOURCES):
        _prebinding_fail("scientific_source_set_mismatch")

    def support_locator(path: Path) -> str:
        resolved = path.resolve()
        if support_locator_root is None:
            return str(resolved)
        root = support_locator_root.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            _prebinding_fail("support_locator_outside_root")
        if not relative.parts or ".." in relative.parts:
            _prebinding_fail("support_locator_invalid")
        return relative.as_posix()

    scientific_rows = []
    for name in _PREBINDING_SCIENTIFIC_SOURCES:
        path = scientific_source_paths[name].resolve()
        source_payload = _prebinding_stable_regular_payload(
            str(path),
            limit=_MAX_SOURCE_FILE_BYTES,
            code="scientific_source_unavailable",
        )
        scientific_rows.append(
            {
                "name": name,
                "path": support_locator(path),
                "sha256": hashlib.sha256(source_payload).hexdigest(),
            }
        )
    if runtime is not None:
        live_runtime = dict(runtime)
    else:
        producer_environment = {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LAZY_LEGACY_OP": "False",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
            "SDL_AUDIODRIVER": "dsp",
            "TZ": "UTC",
        }
        producer_environment.update(
            {
                key: os.environ[key]
                for key in sorted(_PREBINDING_PROCESS_ENVIRONMENT_KEYS - set(producer_environment))
                if key in os.environ
            }
        )
        live_runtime = {
            **_prebinding_live_runtime_identity(device),
            "producer_python_flags": dict(_PREBINDING_PRODUCER_FLAGS),
            "producer_process_environment": producer_environment,
        }
    live_packages = [dict(row) for row in package_rows] if package_rows is not None else _prebinding_live_package_rows()
    root_rows: list[dict[str, object]] = []
    for identifier in sorted(root_paths):
        path = root_paths[identifier].resolve()
        kind = "standard_library" if identifier == "standard-library" else "package_root"
        row = _prebinding_root_inventory(
            identifier,
            str(path),
            kind=kind,
        )
        root_rows.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "semantics_id": row["semantics_id"],
                "path": str(path),
                "file_count": row["file_count"],
                "directory_count": row["directory_count"],
                "total_bytes": row["total_bytes"],
                "inventory_sha256": row["inventory_sha256"],
            }
        )
    tensor_identity = (
        dict(representative_tensor)
        if representative_tensor is not None
        else _prebinding_representative_tensor_identity(device)
    )
    request: dict[str, object] = {
        "schema": _PREBINDING_REQUEST_SCHEMA,
        "protocol": {
            "path": support_locator(protocol_path),
            "value": dict(protocol_value),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "scientific_block_names": list(_PREBINDING_SCIENTIFIC_BLOCKS),
            "scientific_kernel_sha256": hashlib.sha256(_canonical_json_bytes(scientific_value)).hexdigest(),
            "scientific_source_files": scientific_rows,
        },
        "runtime": live_runtime,
        "packages": live_packages,
        "root_inventories": root_rows,
        "representative_tensor": tensor_identity,
    }
    _prebinding_validate_request(request)
    return request


def canonical_prebinding_request_bytes(
    request: Mapping[str, object],
) -> bytes:
    """Return the sole accepted on-disk/stdin request encoding."""

    _prebinding_validate_request(request)
    return _canonical_json_bytes(request) + b"\n"


_PREBINDING_COMPONENT_NAMES = (
    "request",
    "protocol",
    "pendulum",
    "oscillator",
    "coverage",
    "runtime",
    "packages",
    "root_inventories",
    "representative_tensor",
)


def _prebinding_failed_component(code: str) -> dict[str, object]:
    return {
        "code": code,
        "identity_sha256": None,
        "passed": False,
    }


def audit_prebinding_conformance(
    request_value: object,
    *,
    raw_request_sha256: str | None = None,
    locator_root: Path | None = None,
) -> dict[str, object]:
    """Run all no-outcome WM-001 v1.16 prebinding checks.

    The report contains semantic identities only.  Filesystem locations,
    descriptor numbers, process IDs, clocks, and outcome/result paths are
    deliberately absent.
    """

    try:
        request = _prebinding_validate_request(request_value)
        request_sha256 = _prebinding_request_identity(request)
    except _PrebindingConformanceError as error:
        digest = raw_request_sha256 or hashlib.sha256(_canonical_json_bytes(request_value)).hexdigest()
        invalid_components = {
            name: _prebinding_failed_component(error.code if name == "request" else "request_invalid")
            for name in _PREBINDING_COMPONENT_NAMES
        }
        return {
            "schema": _PREBINDING_REPORT_SCHEMA,
            "request_sha256": digest,
            "components": invalid_components,
            "identities": {
                "protocol_sha256": None,
                "scientific_kernel_sha256": None,
                "scientific_source_sha256": None,
                "runtime_sha256": None,
                "packages_sha256": None,
                "root_inventories_sha256": None,
                "representative_tensor_sha256": None,
            },
            "passed": False,
        }

    components: dict[str, dict[str, object]] = {
        "request": {
            "code": "ok",
            "identity_sha256": request_sha256,
            "passed": True,
        }
    }

    def run_component(
        name: str,
        callback: Any,
    ) -> None:
        try:
            result = callback()
        except _PrebindingConformanceError as error:
            components[name] = _prebinding_failed_component(error.code)
        except Exception:
            components[name] = _prebinding_failed_component(f"{name}_execution_failed")
        else:
            components[name] = result

    run_component(
        "protocol",
        lambda: _prebinding_protocol_component(
            request["protocol"],
            locator_root=locator_root,
        ),
    )
    run_component(
        "runtime",
        lambda: _prebinding_runtime_component(request["runtime"]),
    )
    run_component("pendulum", _prebinding_pendulum_component)
    run_component("oscillator", _prebinding_oscillator_component)
    run_component("coverage", _prebinding_coverage_component)
    run_component(
        "packages",
        lambda: _prebinding_packages_component(request["packages"]),
    )
    run_component(
        "root_inventories",
        lambda: _prebinding_root_inventories_component(request["root_inventories"]),
    )
    runtime_request = cast(Mapping[str, object], request["runtime"])
    device = runtime_request.get("device")
    run_component(
        "representative_tensor",
        lambda: _prebinding_tensor_component(
            request["representative_tensor"],
            device=cast(str, device),
        ),
    )
    identities = {
        "protocol_sha256": components["protocol"].get("protocol_sha256"),
        "scientific_kernel_sha256": components["protocol"].get("scientific_kernel_sha256"),
        "scientific_source_sha256": components["protocol"].get("scientific_source_sha256"),
        "runtime_sha256": components["runtime"].get("identity_sha256"),
        "packages_sha256": components["packages"].get("packages_sha256"),
        "root_inventories_sha256": components["root_inventories"].get("root_inventories_sha256"),
        "representative_tensor_sha256": components["representative_tensor"].get("identity_sha256"),
    }
    passed = all(components[name].get("passed") is True for name in _PREBINDING_COMPONENT_NAMES)
    return {
        "schema": _PREBINDING_REPORT_SCHEMA,
        "request_sha256": request_sha256,
        "components": components,
        "identities": identities,
        "passed": passed,
    }


def _load_prebinding_request(
    path: str,
) -> tuple[object, str, Path | None]:
    if path == "-":
        payload = sys.stdin.buffer.read(_MAX_PREBINDING_REQUEST_BYTES + 1)
        if len(payload) > _MAX_PREBINDING_REQUEST_BYTES:
            _prebinding_fail("request_byte_limit_exceeded")
        locator_root = None
    else:
        request_path = Path(path)
        if not request_path.is_absolute():
            request_path = Path.cwd() / request_path
        payload = _prebinding_stable_regular_payload(
            str(request_path),
            limit=_MAX_PREBINDING_REQUEST_BYTES,
            code="request_file_unavailable",
        )
        locator_root = request_path.parent
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        value = _json_without_duplicate_keys(
            payload,
            label="prebinding request",
        )
        canonical_payload = _canonical_json_bytes(value) + b"\n"
    except (ArtifactAuditError, TypeError, ValueError):
        _prebinding_fail("request_json_invalid")
    if payload != canonical_payload:
        _prebinding_fail("request_not_canonical")
    return value, raw_sha256, locator_root


def audit_prebinding_conformance_file(path: str) -> dict[str, object]:
    try:
        value, raw_sha256, locator_root = _load_prebinding_request(path)
    except _PrebindingConformanceError as error:
        components = {
            name: _prebinding_failed_component(error.code if name == "request" else "request_invalid")
            for name in _PREBINDING_COMPONENT_NAMES
        }
        return {
            "schema": _PREBINDING_REPORT_SCHEMA,
            "request_sha256": None,
            "components": components,
            "identities": {
                "protocol_sha256": None,
                "scientific_kernel_sha256": None,
                "scientific_source_sha256": None,
                "runtime_sha256": None,
                "packages_sha256": None,
                "root_inventories_sha256": None,
                "representative_tensor_sha256": None,
            },
            "passed": False,
        }
    return audit_prebinding_conformance(
        value,
        raw_request_sha256=raw_sha256,
        locator_root=locator_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        type=Path,
        nargs="?",
        help="artifact directory or result.json",
    )
    parser.add_argument(
        "--prebinding-conformance",
        metavar="REQUEST_JSON",
        help=("run the result-free WM-001 v1.16 conformance request; use '-' for canonical JSON on stdin"),
    )
    parser.add_argument(
        "--restart-runtime-conformance",
        action="store_true",
        help=(
            "run the result-free WM-001 v1.16 development/formal "
            "restart-runtime branch conformance"
        ),
    )
    parser.add_argument(
        "--producer-bootstrap",
        type=Path,
        help=(
            "explicit captured producer_bootstrap.py support used to bind "
            "restart-runtime evidence"
        ),
    )
    parser.add_argument(
        "--expected-producer-bootstrap-sha256",
        help=(
            "independently bound producer_bootstrap.py SHA-256 used by "
            "restart-runtime conformance"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical JSON report here in addition to stdout",
    )
    parser.add_argument(
        "--forensic-partial",
        action="store_true",
        help="skip raw-result schema validation and do not fail only for known coverage gaps",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.prebinding_conformance is not None:
        if (
            arguments.artifact is not None
            or arguments.forensic_partial
            or arguments.producer_bootstrap is not None
            or arguments.expected_producer_bootstrap_sha256 is not None
            or arguments.restart_runtime_conformance
        ):
            raise SystemExit(
                "--prebinding-conformance is mutually exclusive with artifact, "
                "--producer-bootstrap, "
                "--expected-producer-bootstrap-sha256, "
                "--restart-runtime-conformance, and --forensic-partial"
            )
        report = audit_prebinding_conformance_file(arguments.prebinding_conformance)
        encoded = _canonical_json_bytes(report) + b"\n"
        if arguments.output is not None:
            with arguments.output.resolve().open("xb") as stream:
                stream.write(encoded)
        sys.stdout.buffer.write(encoded)
        return 0 if report["passed"] is True else 1
    if arguments.restart_runtime_conformance:
        if arguments.artifact is not None or arguments.forensic_partial:
            raise SystemExit(
                "--restart-runtime-conformance is mutually exclusive with "
                "artifact and --forensic-partial"
            )
        if arguments.producer_bootstrap is None:
            raise SystemExit(
                "--producer-bootstrap is required for restart-runtime conformance"
            )
        if arguments.expected_producer_bootstrap_sha256 is None:
            raise SystemExit(
                "--expected-producer-bootstrap-sha256 is required for "
                "restart-runtime conformance"
            )
        report = audit_restart_runtime_conformance(
            arguments.producer_bootstrap,
            expected_producer_bootstrap_sha256=(
                arguments.expected_producer_bootstrap_sha256
            ),
        )
        encoded = _canonical_json_bytes(report) + b"\n"
        if arguments.output is not None:
            with arguments.output.resolve().open("xb") as stream:
                stream.write(encoded)
        sys.stdout.buffer.write(encoded)
        return 0 if report["passed"] is True else 1
    if arguments.artifact is None:
        raise SystemExit(
            "artifact is required unless a conformance mode is used"
        )
    if arguments.expected_producer_bootstrap_sha256 is not None:
        raise SystemExit(
            "--expected-producer-bootstrap-sha256 is only valid for "
            "restart-runtime conformance"
        )
    if arguments.producer_bootstrap is None:
        raise SystemExit("--producer-bootstrap is required for artifact audits")
    report = audit_artifact(
        arguments.artifact,
        producer_bootstrap=arguments.producer_bootstrap,
        validate_schema=not arguments.forensic_partial,
        require_claim_completeness=not arguments.forensic_partial,
        verify_custody=not arguments.forensic_partial,
    )
    encoded = _canonical_json_bytes(report) + b"\n"
    if arguments.output is not None:
        artifact_root = (arguments.artifact if arguments.artifact.is_dir() else arguments.artifact.parent).resolve()
        output_path = arguments.output.resolve()
        if output_path == artifact_root or output_path.is_relative_to(artifact_root):
            raise SystemExit("audit output must be outside the immutable producer artifact root")
        with output_path.open("xb") as stream:
            stream.write(encoded)
    sys.stdout.buffer.write(encoded)
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
