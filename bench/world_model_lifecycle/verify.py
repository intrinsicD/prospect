#!/usr/bin/env python3
"""Dependency-free integrity checks for the sealed WM-001 evidence package.

This is deliberately not the experiment runner.  It verifies the immutable
scientific protocol now, an implementation binding before a formal launch, and
the causal/evidence envelope of a result after execution.  Metric recomputation
from tensors and episode traces belongs to the later independent audit.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import stat
import struct
import subprocess
import sys
import tarfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from bench.world_model_lifecycle.assurance import ASSURANCE, TRUST_MODEL_STATEMENT

HERE = Path(__file__).resolve().parent


def _repository_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        candidate = Path(completed.stdout.strip())
        if (
            candidate.is_absolute()
            and candidate.resolve(strict=True) == candidate
            and (candidate / ".git").exists()
            and (candidate / "bench" / "world_model_lifecycle" / "protocol.json").is_file()
        ):
            return candidate
    source_candidate = HERE.parents[1]
    if (source_candidate / ".git").exists():
        return source_candidate
    raise RuntimeError("WM-001 verifier requires a canonical Prospect Git worktree")


REPO = _repository_root()
PROTOCOL_PATH = HERE / "protocol.json"
SEAL_PATH = HERE / "SEALED_PROTOCOL.sha256"
RESULT_SCHEMA_PATH = HERE / "schemas" / "raw-result.schema.json"
BINDING_SCHEMA_PATH = HERE / "schemas" / "formal-binding.schema.json"

FORMAL_SEEDS = (
    3772418031,
    1586188972,
    155797552,
    2704051827,
    818738828,
    4077496645,
    1566512625,
    2151461680,
)
DEVELOPMENT_SEEDS = (3626676950, 2572962267)
COVERAGE_SEMANTICS = "wm001-mixture-pit-binary64-count-v1"
DEVELOPMENT_MATRIX_CONTRACT_SHA256 = "09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"
_V100_MASTER_SEEDS = (
    101,
    211,
    104729,
    130363,
    155921,
    181081,
    206369,
    232003,
    257371,
    283009,
)
_V120_MASTER_SEEDS = (
    1905245264,
    3477142941,
    70359369,
    2936962664,
    976469083,
    1434863921,
    714423665,
    4202129964,
    2335380198,
    854314474,
)
_V130_MASTER_SEEDS = (
    17123296,
    3280610186,
    2725263418,
    3124246399,
    4093604926,
    3908390087,
    3332986400,
    724244869,
    3625750835,
    2671781227,
)
_V140_MASTER_SEEDS = (
    2439054559,
    3246851043,
    339970590,
    474769515,
    550273937,
    438984650,
    2732731971,
    2253809848,
    2206960337,
    3506881479,
)
_V150_MASTER_SEEDS = (
    4085517670,
    2227535912,
    1800791691,
    1963228177,
    2416009491,
    3925214220,
    1508934628,
    2118526007,
    4212585034,
    530094003,
)
_V160_MASTER_SEEDS = (
    2999896578,
    3783052994,
    3863790658,
    3900021454,
    1437244820,
    3175470977,
    228708147,
    3835462042,
    3342200973,
    1751060143,
)
_V160_PROTOCOL_SHA256 = "6f5c21d6e77683c283e09c6257c35abd0e6857e17620e585f414024852d972b2"
_V170_MASTER_SEEDS = (
    3920043614,
    3703229797,
    2080036362,
    865871218,
    3636713390,
    2195564811,
    2000167339,
    329754669,
    4064290468,
    1911057116,
)
_V170_PROTOCOL_SHA256 = "bb7fe6de4fc5de231155fd555bcc0fce6e041b63d99b0c03def8daaf293a364a"
_V180_MASTER_SEEDS = (
    1196068124,
    758859051,
    3362668913,
    1230840469,
    428983069,
    1629522391,
    1347202040,
    1247885121,
    3968594484,
    3609284286,
)
_V180_PROTOCOL_SHA256 = "3aa795e1a54b7cda04b94c77afc683f79639b8f9fffc3dae8be839d53b5d89bc"
_V190_MASTER_SEEDS = (
    86535224,
    2906056242,
    1369779618,
    2721934008,
    2798280967,
    926105433,
    4118470289,
    919763803,
    2112633694,
    2832104894,
)
_V190_PROTOCOL_SHA256 = (
    "3b97eaa1330066a7773345afd3445f086139d5e6090e8f86bfad87d14e93f090"
)
_V1100_MASTER_SEEDS = (
    1647437737,
    1156509260,
    3363134750,
    2153178322,
    2277484641,
    572614265,
    3119775486,
    3121614244,
    3646941950,
    827253974,
)
_V1100_PROTOCOL_SHA256 = (
    "fb2584cbbeab133692867e2396ee1ded5953ca7ceb7d68134febccd5aed3970b"
)
_V1110_MASTER_SEEDS = (
    670819759,
    624845448,
    3391764770,
    20596598,
    999954271,
    2371040464,
    2073495343,
    962058337,
    2170781413,
    3523651983,
)
_V1110_PROTOCOL_SHA256 = (
    "757288cac9fc2935799e4500f0f0d0cf8135417eecb8d138cf8f3e38811d51c8"
)
_V1120_MASTER_SEEDS = (
    2530568307,
    3822916726,
    402304386,
    1582362517,
    3717100311,
    3870324956,
    2551652339,
    986753049,
    4074588580,
    1996653376,
)
_V1120_PROTOCOL_SHA256 = (
    "d64aede84e402d05bd587e1fdf2694381ab6742a28ca19ed88097d0480fa5b80"
)
_V1130_MASTER_SEEDS = (
    560818116,
    1392377688,
    140647545,
    2239253745,
    3333612762,
    4269572592,
    2151457732,
    4034984701,
    2426483518,
    2833322658,
)
_V1130_PROTOCOL_SHA256 = (
    "e7988e3605079b7b7830949d6fd107f26066059ac3cc3974c5bfe15af876dc0c"
)
_V1140_MASTER_SEEDS = (
    630481329,
    2204125221,
    900802928,
    2035185068,
    3817247901,
    14769188,
    2670334085,
    2866408483,
    671156171,
    333753598,
)
_V1140_PROTOCOL_SHA256 = (
    "39f5820a91c8a504355f971449726ae0a9067cc856111a575bb038455d1fd635"
)
_V1150_MASTER_SEEDS = (
    2388891654,
    3201418215,
    2465968807,
    3494485289,
    1615601571,
    2220840580,
    280448223,
    597199725,
    712207456,
    1727907751,
)
_V1150_PROTOCOL_SHA256 = (
    "8db5560044bbedfb491be12a26bd8b39c43fd6d6a314ce86d6afdc71f50486bb"
)
_V1160_MASTER_SEEDS = (
    3922749719,
    1847570536,
    721000968,
    1733386057,
    1129257495,
    1461304433,
    345413014,
    76587833,
    404195464,
    3550251066,
)
_V1160_PROTOCOL_SHA256 = (
    "ac7a8aa331f15412c80a1dad6af9b30c154db33b6d313940e8d2ee546b57dc00"
)
_V1170_MASTER_SEEDS = (
    3454397035,
    2131905789,
    3651766805,
    1960341898,
    785042759,
    1752824577,
    3284431163,
    2694043685,
    2970882769,
    386448916,
)
_V1170_PROTOCOL_SHA256 = (
    "b915d70eef0b09c7562b04f7c9f2e416cd12249c1b512108e11759d008473905"
)
_V1180_MASTER_SEEDS = (
    1787261725,
    1697528199,
    952286440,
    4273641788,
    1748047518,
    2531734648,
    2611012043,
    1851041586,
    1135019273,
    1670867274,
)
_V1180_PROTOCOL_SHA256 = (
    "5def6aaa0fc474675483049dd0b8661abb8819bab459f8e42d4d33b919145cb1"
)
_V1190_MASTER_SEEDS = (
    2548769521,
    799442746,
    3714505505,
    79878112,
    795255854,
    1251505627,
    1184933223,
    3676873506,
    286726369,
    2337061326,
)
_V1190_PROTOCOL_SHA256 = (
    "07c6fe364aeddbd5689fa4f638a6f9a38506b16e8845a947fffa87e01eb3854a"
)
_SCIENTIFIC_BLOCKS = (
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
_V140_SCIENTIFIC_BLOCKS_SHA256 = "fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3"
_SCIENTIFIC_KERNEL_SHA256 = {
    "model.py": "51e61c719dcdd0ebe2f993f16d06eeec03adf6c0fa82b7476356f4b51b3634f1",
    "learning.py": "ed53fcaac32d77e7ed8d9d1af2a91df66682f590de3d60cb6db198670948eaa4",
    "planning.py": "5ebbc3083ff2b49dbee287bf7efbeaec488419bfbbde033a064ad23c356b1d51",
    "runtime_lane.py": "265e65dfd11af62a96d8ea6471f18c664e0e0860febff1a10945642d246b3fdb",
}
_PREFORMAL_AUTHORIZATION_CONTRACT: dict[str, object] = {
    "report_schema": "prospect.wm001.preformal-test-report.v2",
    "canonical_directory": (
        "bench/world_model_lifecycle/results/development/"
        "v1.20.0/preformal"
    ),
    "report_file": "preformal-test-report-v1.20.0.json",
    "ordered_commands": [
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
    ],
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
        "receipt_file": "formal-input-preflight.json",
        "validated_before_binding_acceptance": True,
        "required_by_outer_launcher": True,
        "outer_launcher_runtime_output_digests": True,
        "preserved_in_formal_artifact": True,
    },
}
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
TASK_A = "pendulum_normal_torque"
TASK_B = "pendulum_reversed_torque"
TASK_IRRELEVANT = "independent_phase_oscillator"
INDEPENDENT_OSCILLATOR_SOURCE = "prospect:IndependentPhaseOscillator-v1"
EPISODE_CONTRACTS = frozenset(
    {
        ("collect_a", TASK_A, "collection_random", "cold"),
        ("collect_b", TASK_B, "collection_random", "after_a"),
        (
            "collect_irrelevant",
            TASK_IRRELEVANT,
            "collection_random",
            "cold",
        ),
        (
            "predictive_validation_a",
            TASK_A,
            "validation_random",
            "after_a",
        ),
        (
            "predictive_validation_b",
            TASK_B,
            "validation_random",
            "after_a",
        ),
        (
            "predictive_validation_irrelevant",
            TASK_IRRELEVANT,
            "validation_random",
            "irrelevant",
        ),
        *(
            ("behavior_evaluation_a", TASK_A, condition, condition)
            for condition in (
                "cold",
                "after_a",
                "frozen",
                "corrupted",
                "irrelevant",
                "after_b_replay",
                "after_b_naive",
                "random",
                "oracle",
            )
        ),
        *(
            ("behavior_evaluation_b", TASK_B, condition, condition)
            for condition in (
                "after_a",
                "after_b_replay",
                "after_b_naive",
                "random",
                "oracle",
            )
        ),
    }
)
PREDICTIVE_CONTRACTS = frozenset(
    {
        *(
            ("predictive_validation_a", TASK_A, condition, condition)
            for condition in (
                "cold",
                "after_a",
                "frozen",
                "corrupted",
                "irrelevant",
                "after_b_replay",
                "after_b_naive",
            )
        ),
        *(
            ("predictive_validation_b", TASK_B, condition, condition)
            for condition in (
                "after_a",
                "after_b_replay",
                "after_b_naive",
            )
        ),
        *(
            (
                "predictive_validation_irrelevant",
                TASK_IRRELEVANT,
                condition,
                condition,
            )
            for condition in ("cold", "irrelevant")
        ),
    }
)
FORMAL_EPISODE_CONTRACT_COUNTS = {
    contract: 32 if str(contract[0]).startswith("behavior_evaluation_") else 8 for contract in EPISODE_CONTRACTS
}
FORMAL_PREDICTIVE_CONTRACT_COUNTS = {contract: 1 for contract in PREDICTIVE_CONTRACTS}
COMMITTED_PHASE_SPLITS = {
    "train_a": ("collect_a",),
    "train_a_corrupted": ("collect_a",),
    "train_a_irrelevant": ("collect_irrelevant",),
    "train_b_replay": ("collect_a", "collect_b"),
    "train_b_naive": ("collect_b",),
}
EXPECTED_SEED_COUNTS = {
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
REQUIRED_PACKAGES = frozenset(
    {
        "python",
        "gymnasium",
        "jsonschema",
        "numpy",
        "torch",
        "torchrl",
        "tensordict",
        "prospect",
    }
)
REQUIRED_COMPONENTS = (
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
GATES = tuple(f"K{index}" for index in range(8))
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DEVELOPMENT_CLOSURE_FIELDS = frozenset(
    {
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
)
_DEVELOPMENT_SOURCE_FIELDS = frozenset(
    {
        "git_commit",
        "git_tree",
        "worktree_clean",
        "dependency_lock_sha256",
        "producer_bootstrap_sha256",
        "launch_bootstrap_sha256",
        "runner_source_sha256",
        "auditor_source_sha256",
    }
)
_DEVELOPMENT_EXECUTION_FIELDS = frozenset(
    {
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
)
_DEVELOPMENT_CUSTODY_FIELDS = frozenset(
    {
        "runtime_seal_member",
        "runtime_seal_sha256",
        "producer_bootstrap_member",
        "producer_bootstrap_sha256",
        "launch_bootstrap_member",
        "launch_bootstrap_sha256",
        "package_ownership",
    }
)
_DEVELOPMENT_AUDIT_EXECUTION_FIELDS = frozenset(
    {
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
)
_DEVELOPMENT_FIXED_ROLE_MEMBERS = {
    "producer_manifest_member": "producer/producer-manifest.json",
    "raw_result_member": "producer/result.json",
    "result_qualification_member": "evidence/development-result-qualification.json",
    "independent_audit_member": "evidence/independent-audit.json",
    "audit_reproduction_member": "evidence/audit-reproduction.json",
}
_DEVELOPMENT_CUSTODY_ROLE_MEMBERS = {
    "runtime_seal_member": "evidence/producer-runtime-seal.json",
    "producer_bootstrap_member": "evidence/producer-bootstrap.py",
    "launch_bootstrap_member": "evidence/launch-bootstrap.py",
}
_MAX_DEVELOPMENT_QUALIFICATION_MEMBERS = 100_000
_MAX_DEVELOPMENT_QUALIFICATION_MEMBER_BYTES = 4 << 30
_MAX_DEVELOPMENT_QUALIFICATION_TOTAL_BYTES = 32 << 30
FORMAL_CONFORMANCE_KEYS = frozenset(
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
FORMAL_CONFORMANCE_PARAMETERS = {
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
    "dt": 0.05,
    "max_speed": 8.0,
    "max_torque": 2.0,
}
FORMAL_CONFORMANCE_TOLERANCES = {
    "observation_atol": 2e-6,
    "reward_atol": 1e-9,
    "planner_observation_atol": 2e-6,
    "planner_reward_atol": 2e-5,
}


class Violation(ValueError):
    """A protocol, binding, or result integrity violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Violation(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Violation(f"cannot load JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def _validate_json_schema(
    value: dict[str, Any],
    schema: dict[str, Any],
    *,
    label: str,
) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise Violation("jsonschema 4.25.1 is required for binding/result verification") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise Violation(f"{label} schema is invalid: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise Violation(f"{label} violates JSON Schema at {location}: {first.message}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


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


def _binding_sibling(path: Path, value: object, field: str) -> Path:
    _require(isinstance(value, str) and value, f"{field} is missing")
    relative = Path(value)
    _require(
        not relative.is_absolute() and len(relative.parts) == 1 and relative.name == value,
        f"{field} must be a safe sibling filename",
    )
    return path.parent / relative


def _bound_evidence_payload(
    binding_path: Path,
    block: Mapping[str, object],
    *,
    prefix: str,
) -> tuple[Path, bytes]:
    evidence_path = _binding_sibling(
        binding_path,
        block.get(f"{prefix}_file"),
        f"audit_execution.{prefix}_file",
    )
    _require(
        evidence_path.is_file() and not evidence_path.is_symlink(),
        f"bound {prefix} is missing or aliased",
    )
    payload = evidence_path.read_bytes()
    _require(
        block.get(f"{prefix}_bytes") == len(payload),
        f"bound {prefix} byte size changed",
    )
    _require(
        block.get(f"{prefix}_sha256") == sha256(payload).hexdigest(),
        f"bound {prefix} digest changed",
    )
    return evidence_path, payload


def _canonical_object_payload(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Violation(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not one JSON object")
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(payload == canonical, f"{label} is not canonical JSON followed by LF")
    return value


def _require_sha256(value: object, field: str) -> str:
    _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None, f"{field} is not SHA-256")
    return value


def _safe_development_archive_member(value: object) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    candidate = Path(value)
    return (
        not candidate.is_absolute()
        and candidate.as_posix() == value
        and len(candidate.parts) >= 2
        and "." not in candidate.parts
        and ".." not in candidate.parts
        and candidate.parts[0] in {"evidence", "producer"}
    )


def _recorded_development_closure_identity(
    path: Path,
) -> tuple[dict[str, Any], dict[str, object], dict[str, str]]:
    """Authenticate a closure receipt without replaying its runtime inventory.

    The sealed-runtime closure verifier already streamed the qualification
    archive and compared live package inventories before the closure and its
    post-finalization receipt were committed.  A downstream QA interpreter is
    intentionally a different environment.  It therefore reopens the exact
    canonical closure bytes, validates the complete typed projection used by
    the binding, and cross-links every named archive role, but does not claim
    to reproduce the earlier live-runtime check.
    """

    _require(path.is_file() and not path.is_symlink(), "recorded development closure is missing or aliased")
    payload = path.read_bytes()
    closure = _canonical_object_payload(
        payload,
        label="recorded development closure",
    )
    _require(
        set(closure) == _DEVELOPMENT_CLOSURE_FIELDS
        and closure.get("schema") == "prospect.wm001.development-closure.v2"
        and closure.get("experiment_id") == "WM-001"
        and closure.get("protocol_version") == "1.20.0"
        and closure.get("engineering_verified") is True
        and closure.get("audit_reproduced") is True
        and closure.get("performance_values_bound") is False,
        "recorded development closure identity or status is invalid",
    )
    source = closure.get("source")
    execution = closure.get("producer_execution")
    custody = closure.get("producer_custody")
    audit_execution = closure.get("audit_execution")
    archive = closure.get("qualification_archive")
    producer_root_raw = closure.get("producer_root")
    _require(
        isinstance(source, dict)
        and set(source) == _DEVELOPMENT_SOURCE_FIELDS
        and isinstance(execution, dict)
        and set(execution) == _DEVELOPMENT_EXECUTION_FIELDS
        and isinstance(custody, dict)
        and set(custody) == _DEVELOPMENT_CUSTODY_FIELDS
        and isinstance(audit_execution, dict)
        and set(audit_execution) == _DEVELOPMENT_AUDIT_EXECUTION_FIELDS
        and isinstance(archive, dict)
        and set(archive)
        == {"format", "file", "canonical_path", "bytes", "sha256", "members"}
        and isinstance(producer_root_raw, str)
        and bool(producer_root_raw),
        "recorded development closure structure is malformed",
    )
    assert isinstance(source, dict)
    assert isinstance(execution, dict)
    assert isinstance(custody, dict)
    assert isinstance(audit_execution, dict)
    assert isinstance(archive, dict)

    producer_root = Path(cast(str, producer_root_raw))
    _require(
        producer_root.is_absolute()
        and Path(str(producer_root)) == producer_root
        and "\x00" not in cast(str, producer_root_raw),
        "recorded development producer root is unsafe",
    )
    _require(
        GIT_SHA_PATTERN.fullmatch(cast(str, source.get("git_commit", ""))) is not None
        and GIT_SHA_PATTERN.fullmatch(cast(str, source.get("git_tree", ""))) is not None
        and source.get("worktree_clean") is True
        and all(
            isinstance(source.get(field), str)
            and SHA256_PATTERN.fullmatch(cast(str, source[field])) is not None
            for field in _DEVELOPMENT_SOURCE_FIELDS
            - {"git_commit", "git_tree", "worktree_clean"}
        ),
        "recorded development source identity is malformed",
    )
    python_executable = execution.get("python_executable")
    process_environment = execution.get("process_environment")
    package_roots = execution.get("package_roots")
    _require(
        execution.get("git_commit") == source.get("git_commit")
        and execution.get("git_tree") == source.get("git_tree")
        and execution.get("worktree_clean") is True
        and execution.get("dependency_lock_sha256") == source.get("dependency_lock_sha256")
        and execution.get("producer_bootstrap_sha256") == source.get("producer_bootstrap_sha256")
        and all(
            isinstance(execution.get(field), str)
            and SHA256_PATTERN.fullmatch(cast(str, execution[field])) is not None
            for field in (
                "dependency_lock_sha256",
                "python_executable_sha256",
                "runtime_seal_sha256",
                "producer_bootstrap_sha256",
            )
        )
        and isinstance(python_executable, str)
        and bool(python_executable)
        and "\x00" not in python_executable
        and Path(python_executable).is_absolute()
        and isinstance(execution.get("python_version"), str)
        and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", cast(str, execution["python_version"])) is not None
        and isinstance(execution.get("platform"), str)
        and bool(execution["platform"])
        and isinstance(execution.get("machine"), str)
        and bool(execution["machine"])
        and execution.get("device") in {"cpu", "cuda"}
        and isinstance(execution.get("python_flags"), dict)
        and isinstance(process_environment, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in process_environment.items())
        and type(execution.get("thread_count")) is int
        and cast(int, execution["thread_count"]) > 0
        and type(execution.get("interop_thread_count")) is int
        and cast(int, execution["interop_thread_count"]) > 0
        and execution.get("deterministic_algorithms") is True
        and execution.get("runtime_seal_descriptor_custody") is True
        and execution.get("bootstrap_descriptor_custody") is True
        and isinstance(package_roots, list)
        and bool(package_roots)
        and all(isinstance(row, dict) for row in package_roots)
        and isinstance(execution.get("standard_library"), dict),
        "recorded development execution identity is malformed",
    )
    if execution.get("device") == "cpu":
        _require(
            execution.get("accelerator") is None
            and (execution.get("cuda_runtime") is None or isinstance(execution.get("cuda_runtime"), str))
            and execution.get("cuda_driver") is None
            and execution.get("cublas_workspace_config") is None,
            "recorded development CPU identity is malformed",
        )
    else:
        _require(
            isinstance(execution.get("accelerator"), str)
            and bool(execution["accelerator"])
            and isinstance(execution.get("cuda_runtime"), str)
            and bool(execution["cuda_runtime"])
            and isinstance(execution.get("cuda_driver"), str)
            and bool(execution["cuda_driver"])
            and execution.get("cublas_workspace_config") == ":4096:8",
            "recorded development CUDA identity is malformed",
        )

    _require(
        all(
            closure.get(field) == expected
            for field, expected in _DEVELOPMENT_FIXED_ROLE_MEMBERS.items()
        )
        and all(
            custody.get(field) == expected
            for field, expected in _DEVELOPMENT_CUSTODY_ROLE_MEMBERS.items()
        )
        and isinstance(custody.get("package_ownership"), dict)
        and custody.get("runtime_seal_sha256") == execution.get("runtime_seal_sha256")
        and custody.get("producer_bootstrap_sha256") == source.get("producer_bootstrap_sha256")
        and custody.get("launch_bootstrap_sha256") == source.get("launch_bootstrap_sha256")
        and all(
            isinstance(custody.get(field), str)
            and SHA256_PATTERN.fullmatch(cast(str, custody[field])) is not None
            for field in (
                "runtime_seal_sha256",
                "producer_bootstrap_sha256",
                "launch_bootstrap_sha256",
            )
        ),
        "recorded development custody identity is malformed",
    )
    support_rows = audit_execution.get("support_files")
    _require(
        audit_execution.get("source_mode") == "descriptor"
        and isinstance(support_rows, list)
        and all(
            isinstance(row, dict)
            and set(row) == {"path", "bytes", "sha256"}
            and isinstance(row.get("path"), str)
            and bool(row["path"])
            and not Path(cast(str, row["path"])).is_absolute()
            and Path(cast(str, row["path"])).as_posix() == row["path"]
            and "." not in Path(cast(str, row["path"])).parts
            and ".." not in Path(cast(str, row["path"])).parts
            and type(row.get("bytes")) is int
            and cast(int, row["bytes"]) >= 0
            and isinstance(row.get("sha256"), str)
            and SHA256_PATTERN.fullmatch(cast(str, row["sha256"])) is not None
            for row in support_rows
        )
        and [cast(str, row["path"]) for row in support_rows]
        == sorted({cast(str, row["path"]) for row in support_rows})
        and all(
            isinstance(audit_execution.get(field), str)
            and SHA256_PATTERN.fullmatch(cast(str, audit_execution[field])) is not None
            for field in _DEVELOPMENT_AUDIT_EXECUTION_FIELDS
            - {"support_files", "source_mode"}
        )
        and audit_execution.get("runner_source_sha256") == source.get("runner_source_sha256")
        and audit_execution.get("auditor_source_sha256") == source.get("auditor_source_sha256"),
        "recorded development audit identity is malformed",
    )

    archive_digest = archive.get("sha256")
    archive_file = archive.get("file")
    archive_path = archive.get("canonical_path")
    members = archive.get("members")
    _require(
        archive.get("format") == "ustar-uncompressed-v1"
        and isinstance(archive_digest, str)
        and SHA256_PATTERN.fullmatch(archive_digest) is not None
        and isinstance(archive_file, str)
        and archive_file == f"development-qualification-{archive_digest[:16]}.tar"
        and Path(archive_file).name == archive_file
        and isinstance(archive_path, str)
        and not Path(archive_path).is_absolute()
        and Path(archive_path).as_posix() == archive_path
        and "." not in Path(archive_path).parts
        and ".." not in Path(archive_path).parts
        and Path(archive_path).name == archive_file
        and type(archive.get("bytes")) is int
        and cast(int, archive["bytes"]) >= 0
        and isinstance(members, list)
        and 1 <= len(members) <= _MAX_DEVELOPMENT_QUALIFICATION_MEMBERS,
        "recorded development archive identity is malformed",
    )
    member_paths: list[str] = []
    member_digests: dict[str, str] = {}
    total_member_bytes = 0
    for index, row in enumerate(cast(list[object], members)):
        _require(
            isinstance(row, dict)
            and set(row) == {"path", "bytes", "sha256"}
            and _safe_development_archive_member(row.get("path"))
            and type(row.get("bytes")) is int
            and 0 <= cast(int, row["bytes"]) <= _MAX_DEVELOPMENT_QUALIFICATION_MEMBER_BYTES
            and isinstance(row.get("sha256"), str)
            and SHA256_PATTERN.fullmatch(cast(str, row["sha256"])) is not None,
            f"recorded development archive member {index} is malformed",
        )
        assert isinstance(row, dict)
        member_path = cast(str, row["path"])
        member_paths.append(member_path)
        member_digests[member_path] = cast(str, row["sha256"])
        total_member_bytes += cast(int, row["bytes"])
    _require(
        member_paths == sorted(member_paths)
        and len(member_paths) == len(set(member_paths))
        and total_member_bytes <= _MAX_DEVELOPMENT_QUALIFICATION_TOTAL_BYTES,
        "recorded development archive members are unordered, duplicated, or oversized",
    )
    sidecar_fields = (
        "audit_runtime_manifest_member",
        "audit_invocation_manifest_member",
        "audit_stderr_member",
    )
    _require(
        all(
            isinstance(closure.get(field), str)
            and cast(str, closure[field]).startswith("evidence/")
            and _safe_development_archive_member(closure[field])
            and Path(cast(str, closure[field])).name
            == cast(str, closure[field]).removeprefix("evidence/")
            for field in sidecar_fields
        ),
        "recorded development audit sidecar role is unsafe",
    )
    role_members = {
        **{field: cast(str, closure[field]) for field in _DEVELOPMENT_FIXED_ROLE_MEMBERS},
        **{field: cast(str, closure[field]) for field in sidecar_fields},
        **{field: cast(str, custody[field]) for field in _DEVELOPMENT_CUSTODY_ROLE_MEMBERS},
    }
    _require(
        len(set(role_members.values())) == len(role_members)
        and all(member_paths.count(member) == 1 for member in role_members.values()),
        "recorded development closure role has no unique archive member",
    )
    custody_digest_fields = {
        "runtime_seal_member": "runtime_seal_sha256",
        "producer_bootstrap_member": "producer_bootstrap_sha256",
        "launch_bootstrap_member": "launch_bootstrap_sha256",
    }
    audit_digest_fields = {
        "audit_reproduction_member": "receipt_sha256",
        "audit_runtime_manifest_member": "runtime_manifest_sha256",
        "audit_invocation_manifest_member": "invocation_manifest_sha256",
        "audit_stderr_member": "stderr_sha256",
    }
    _require(
        all(
            member_digests[cast(str, custody[member_field])] == custody[digest_field]
            for member_field, digest_field in custody_digest_fields.items()
        )
        and all(
            member_digests[cast(str, closure[member_field])] == audit_execution[digest_field]
            for member_field, digest_field in audit_digest_fields.items()
        ),
        "recorded development closure role digest differs from its archive member",
    )

    def role_digest(field: str) -> str:
        return member_digests[cast(str, closure[field])]

    identity: dict[str, object] = {
        "closure_schema": closure["schema"],
        "closure_file": path.name,
        "closure_bytes": len(payload),
        "closure_sha256": sha256(payload).hexdigest(),
        "qualification_archive_file": archive["file"],
        "qualification_archive_path": archive["canonical_path"],
        "qualification_archive_bytes": archive["bytes"],
        "qualification_archive_sha256": archive["sha256"],
        "qualification_archive_members_sha256": _canonical_sha256({"members": members}),
        "producer_manifest_sha256": role_digest("producer_manifest_member"),
        "raw_result_sha256": role_digest("raw_result_member"),
        "result_qualification_sha256": role_digest("result_qualification_member"),
        "independent_audit_sha256": role_digest("independent_audit_member"),
        "audit_reproduction_sha256": role_digest("audit_reproduction_member"),
        "audit_runtime_manifest_sha256": role_digest("audit_runtime_manifest_member"),
        "audit_invocation_manifest_sha256": role_digest("audit_invocation_manifest_member"),
        "audit_stderr_sha256": role_digest("audit_stderr_member"),
        "source_identity_sha256": _canonical_sha256(source),
        "producer_execution_identity_sha256": _canonical_sha256(execution),
        "producer_custody_identity_sha256": _canonical_sha256(custody),
        "audit_execution_identity_sha256": _canonical_sha256(audit_execution),
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }
    return closure, identity, member_digests


def _recorded_accepted_closure_receipt(
    report_path: Path,
    report: Mapping[str, object],
    *,
    closure_sha256: str,
    producer_manifest_sha256: str,
    raw_result_sha256: str,
) -> dict[str, Any]:
    """Cross-link command 9's sealed-runtime receipt to recorded closure bytes."""

    commands = report.get("commands")
    matches = (
        [
            row
            for row in commands
            if isinstance(row, dict)
            and row.get("name") == "runtime-accepted-closure-evidence"
        ]
        if isinstance(commands, list)
        else []
    )
    _require(len(matches) == 1, "preformal report lacks one accepted-closure receipt")
    row = cast(dict[str, object], matches[0])
    stdout = row.get("stdout")
    stderr = row.get("stderr")
    _require(
        isinstance(stdout, dict)
        and set(stdout) == {"file", "bytes", "sha256"}
        and isinstance(stderr, dict)
        and set(stderr) == {"file", "bytes", "sha256"}
        and type(stdout.get("bytes")) is int
        and cast(int, stdout["bytes"]) > 0
        and type(stderr.get("bytes")) is int
        and stderr.get("bytes") == 0
        and stderr.get("sha256") == sha256(b"").hexdigest(),
        "accepted-closure receipt stream identity is malformed",
    )
    assert isinstance(stdout, dict)
    assert isinstance(stderr, dict)
    stdout_path = _binding_sibling(
        report_path,
        stdout.get("file"),
        "accepted-closure receipt stdout.file",
    )
    stderr_path = _binding_sibling(
        report_path,
        stderr.get("file"),
        "accepted-closure receipt stderr.file",
    )
    stdout_payload = stdout_path.read_bytes()
    stderr_payload = stderr_path.read_bytes()
    _require(
        stdout_path.is_file()
        and not stdout_path.is_symlink()
        and stderr_path.is_file()
        and not stderr_path.is_symlink()
        and len(stdout_payload) == stdout.get("bytes")
        and sha256(stdout_payload).hexdigest() == stdout.get("sha256")
        and not stderr_payload,
        "accepted-closure receipt stream bytes changed",
    )
    receipt = _canonical_object_payload(
        stdout_payload,
        label="accepted-closure runtime receipt",
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
    _require(
        set(receipt) == expected_fields
        and receipt.get("schema") == "prospect.wm001.preformal-runtime-check.v1"
        and receipt.get("mode") == "accepted-closure-evidence"
        and receipt.get("passed") is True
        and all(
            isinstance(receipt.get(field), str)
            and SHA256_PATTERN.fullmatch(cast(str, receipt[field])) is not None
            for field in expected_fields
            - {"schema", "mode", "passed"}
        )
        and receipt.get("development_closure_sha256") == closure_sha256
        and receipt.get("producer_manifest_sha256") == producer_manifest_sha256
        and receipt.get("raw_result_sha256") == raw_result_sha256,
        "accepted-closure runtime receipt differs from the recorded closure",
    )
    return receipt


def _verify_archived_audit_capacity(
    closure: Mapping[str, object],
) -> dict[str, object]:
    """Reopen the durable calibration receipts and derive capacity independently."""

    archive = closure.get("qualification_archive")
    _require(
        isinstance(archive, dict)
        and set(archive)
        == {"format", "file", "canonical_path", "bytes", "sha256", "members"}
        and archive.get("format") == "ustar-uncompressed-v1"
        and isinstance(archive.get("sha256"), str)
        and SHA256_PATTERN.fullmatch(cast(str, archive["sha256"])) is not None,
        "development capacity archive identity is absent",
    )
    assert isinstance(archive, dict)
    relative = archive.get("canonical_path")
    rows = archive.get("members")
    _require(
        isinstance(relative, str)
        and not Path(relative).is_absolute()
        and Path(relative).as_posix() == relative
        and "." not in Path(relative).parts
        and ".." not in Path(relative).parts
        and isinstance(rows, list),
        "development capacity archive identity is unsafe",
    )
    assert isinstance(relative, str)
    assert isinstance(rows, list)
    archive_path = REPO / relative
    try:
        metadata = archive_path.lstat()
        resolved = archive_path.resolve(strict=True)
    except OSError as error:
        raise Violation("development capacity archive is missing") from error
    _require(
        resolved == archive_path
        and stat.S_ISREG(metadata.st_mode)
        and not archive_path.is_symlink()
        and metadata.st_nlink == 1
        and archive.get("bytes") == metadata.st_size
        and metadata.st_size <= (40 << 30),
        "development capacity archive file identity changed",
    )
    archive_digest = sha256()
    try:
        with archive_path.open("rb") as archive_stream:
            for chunk in iter(lambda: archive_stream.read(1 << 20), b""):
                archive_digest.update(chunk)
        final_metadata = archive_path.lstat()
    except OSError as error:
        raise Violation("development capacity archive cannot be authenticated") from error
    _require(
        archive_digest.hexdigest() == archive.get("sha256")
        and (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        == (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_mode,
            final_metadata.st_nlink,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        ),
        "development capacity archive digest or stable identity changed",
    )
    row_by_path = {
        cast(str, row["path"]): cast(dict[str, object], row)
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
    }
    role_targets = {
        closure.get("independent_audit_member"),
        closure.get("audit_runtime_manifest_member"),
        closure.get("audit_invocation_manifest_member"),
        closure.get("audit_stderr_member"),
    }
    _require(
        all(isinstance(target, str) for target in role_targets),
        "development capacity archive sidecar roles are malformed",
    )
    targets = {
        "producer/producer-manifest.json",
        "evidence/audit-reproduction.json",
        "evidence/audit-execution-01.execution.json",
        "evidence/audit-execution-02.execution.json",
        *cast(set[str], role_targets),
    }
    _require(
        len(row_by_path) == len(rows) and targets <= set(row_by_path),
        "development capacity archive omits durable calibration evidence",
    )
    retained: dict[str, bytes] = {}
    observed_names: list[str] = []
    try:
        with tarfile.open(archive_path, mode="r:") as tar:
            for member in tar:
                observed_names.append(member.name)
                expected = row_by_path.get(member.name)
                _require(
                    expected is not None
                    and member.isreg()
                    and member.type == tarfile.REGTYPE
                    and member.size == expected.get("bytes"),
                    "development capacity archive member metadata changed",
                )
                assert expected is not None
                if member.name in targets:
                    _require(
                        member.size <= (64 << 20),
                        "development capacity evidence exceeds its read bound",
                    )
                    member_stream = tar.extractfile(member)
                    _require(
                        member_stream is not None,
                        "development capacity evidence is unreadable",
                    )
                    assert member_stream is not None
                    payload = member_stream.read((64 << 20) + 1)
                    _require(
                        len(payload) == member.size
                        and len(payload) <= (64 << 20)
                        and sha256(payload).hexdigest() == expected.get("sha256"),
                        "development capacity evidence bytes changed",
                    )
                    retained[member.name] = payload
    except (OSError, tarfile.TarError) as error:
        raise Violation("development capacity archive cannot be parsed") from error
    _require(
        observed_names == [cast(str, row["path"]) for row in rows]
        and set(retained) == targets,
        "development capacity archive member order or set changed",
    )
    manifest_payload = retained["producer/producer-manifest.json"]
    manifest = _canonical_object_payload(
        manifest_payload,
        label="archived development producer manifest for capacity",
    )
    manifest_rows = manifest.get("files")
    _require(
        isinstance(manifest_rows, list)
        and manifest.get("schema") == "prospect.wm001.producer-manifest.v1"
        and manifest.get("experiment_id") == "WM-001"
        and manifest.get("lane") == "development"
        and manifest.get("status") == "completed"
        and manifest.get("error") is None
        and manifest.get("manifest_excludes") == ["producer-manifest.json"]
        and manifest.get("file_count") == len(manifest_rows),
        "archived development producer manifest is invalid for capacity",
    )
    assert isinstance(manifest_rows, list)
    manifest_paths: list[str] = []
    total = 0
    result_bytes: int | None = None
    for row in manifest_rows:
        _require(
            isinstance(row, dict)
            and set(row) == {"path", "bytes", "sha256"}
            and isinstance(row.get("path"), str)
            and type(row.get("bytes")) is int
            and 0 <= cast(int, row["bytes"]) <= (4 << 30)
            and isinstance(row.get("sha256"), str)
            and SHA256_PATTERN.fullmatch(cast(str, row["sha256"])) is not None,
            "archived development producer capacity row is malformed",
        )
        assert isinstance(row, dict)
        path = cast(str, row["path"])
        manifest_paths.append(path)
        total += cast(int, row["bytes"])
        if path == "result.json":
            _require(result_bytes is None, "archived capacity has duplicate results")
            result_bytes = cast(int, row["bytes"])
    _require(
        manifest_paths == sorted(set(manifest_paths))
        and 0 < total <= (8 << 30)
        and result_bytes is not None
        and 0 < result_bytes <= (2 << 30),
        "archived development producer capacity sizes are invalid",
    )
    execution_fields = {
        "schema", "returncode", "passed", "source_mode", "command",
        "stdout_file", "stderr_file", "runtime_manifest_file",
        "invocation_manifest_file", "stdout_bytes", "stdout_sha256",
        "stderr_bytes", "stderr_sha256", "runtime_manifest_bytes",
        "runtime_manifest_sha256", "invocation_manifest_bytes",
        "invocation_manifest_sha256", "bootstrap_sha256",
        "auditor_source_sha256", "support_files", "subprocess_elapsed_ns",
    }
    execution_payloads = [
        retained[f"evidence/audit-execution-{ordinal:02d}.execution.json"]
        for ordinal in (1, 2)
    ]
    executions = [
        _canonical_object_payload(payload, label=f"archived capacity execution {ordinal}")
        for ordinal, payload in enumerate(execution_payloads, start=1)
    ]
    for execution in executions:
        _require(
            set(execution) == execution_fields
            and execution.get("schema") == "prospect.wm001.captured-audit-execution.v2"
            and execution.get("returncode") == 0
            and execution.get("passed") is True
            and execution.get("source_mode") == "descriptor"
            and type(execution.get("subprocess_elapsed_ns")) is int
            and cast(int, execution["subprocess_elapsed_ns"]) > 0,
            "archived capacity execution receipt is invalid",
        )
    audit_member = cast(str, closure["independent_audit_member"])
    runtime_member = cast(str, closure["audit_runtime_manifest_member"])
    invocation_member = cast(str, closure["audit_invocation_manifest_member"])
    stderr_member = cast(str, closure["audit_stderr_member"])
    audit_payload = retained[audit_member]
    runtime_payload = retained[runtime_member]
    invocation_payload = retained[invocation_member]
    stderr_payload = retained[stderr_member]
    audit_report = _canonical_object_payload(
        audit_payload,
        label="archived outcome audit report for capacity",
    )
    runtime = _canonical_object_payload(
        runtime_payload,
        label="archived outcome runtime manifest for capacity",
    )
    invocation = _canonical_object_payload(
        invocation_payload,
        label="archived outcome invocation manifest for capacity",
    )
    runtime_source = runtime.get("source")
    runtime_fields = {
        "schema",
        "execution_role",
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
    invocation_fields = {
        "schema",
        "runtime_manifest_sha256",
        "working_directory",
        "auditor_argv",
    }
    _require(
        audit_report.get("passed") is True
        and set(runtime) == runtime_fields
        and runtime.get("schema") == "prospect.wm001.audit-runtime-manifest.v2"
        and runtime.get("execution_role") == "outcome_audit"
        and isinstance(runtime_source, dict)
        and runtime_source.get("mode") == "descriptor"
        and runtime.get("limits")
        == {
            "timeout_seconds": 10_800,
            "stdout_bytes": 64 << 20,
            "stderr_bytes": 16 << 20,
        }
        and set(invocation) == invocation_fields
        and invocation.get("schema")
        == "prospect.wm001.audit-invocation-manifest.v1"
        and invocation.get("runtime_manifest_sha256")
        == sha256(runtime_payload).hexdigest()
        and invocation.get("working_directory") == str(REPO)
        and invocation.get("auditor_argv")
        == [
            closure.get("producer_root"),
            "--producer-bootstrap",
            "@captured/producer_bootstrap.py",
        ],
        "archived capacity execution was not the bound full outcome audit",
    )
    _require(
        all(
            executions[0].get(field) == executions[1].get(field)
            for field in (
                "stdout_bytes", "stdout_sha256", "stderr_bytes", "stderr_sha256",
                "runtime_manifest_bytes", "runtime_manifest_sha256",
                "invocation_manifest_bytes", "invocation_manifest_sha256",
                "bootstrap_sha256", "auditor_source_sha256", "support_files",
            )
        ),
        "archived capacity executions are not byte-identical",
    )
    for execution in executions:
        _require(
            execution.get("stdout_bytes") == len(audit_payload)
            and execution.get("stdout_sha256") == sha256(audit_payload).hexdigest()
            and execution.get("stderr_bytes") == len(stderr_payload)
            and execution.get("stderr_sha256") == sha256(stderr_payload).hexdigest()
            and execution.get("runtime_manifest_bytes") == len(runtime_payload)
            and execution.get("runtime_manifest_sha256")
            == sha256(runtime_payload).hexdigest()
            and execution.get("invocation_manifest_bytes")
            == len(invocation_payload)
            and execution.get("invocation_manifest_sha256")
            == sha256(invocation_payload).hexdigest()
            and execution.get("support_files") == runtime.get("support_files")
            and execution.get("bootstrap_sha256") == runtime.get("bootstrap_sha256")
            and execution.get("auditor_source_sha256")
            == cast(dict[str, object], runtime_source).get("sha256"),
            "archived capacity execution receipt differs from captured full-audit bytes",
        )
    reproduction_payload = retained["evidence/audit-reproduction.json"]
    reproduction = _canonical_object_payload(
        reproduction_payload,
        label="archived audit reproduction for capacity",
    )
    reproduction_fields = {
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
        "first_execution_receipt_file",
        "first_execution_receipt_bytes",
        "first_execution_receipt_sha256",
        "replay_execution_receipt_file",
        "replay_execution_receipt_bytes",
        "replay_execution_receipt_sha256",
        "capacity",
        "passed",
    }
    _require(
        set(reproduction) == reproduction_fields
        and reproduction.get("schema") == "prospect.wm001.audit-reproduction.v3"
        and reproduction.get("experiment_id") == "WM-001"
        and reproduction.get("protocol_version") == "1.20.0"
        and reproduction.get("supplied_audit_sha256")
        == sha256(audit_payload).hexdigest()
        and reproduction.get("reproduced_audit_sha256")
        == sha256(audit_payload).hexdigest()
        and reproduction.get("byte_identical") is True
        and reproduction.get("returncode") == 0
        and reproduction.get("source_mode") == "descriptor"
        and reproduction.get("stdout_bytes") == len(audit_payload)
        and reproduction.get("runtime_manifest_bytes") == len(runtime_payload)
        and reproduction.get("runtime_manifest_sha256")
        == sha256(runtime_payload).hexdigest()
        and reproduction.get("runtime_manifest_file") == Path(runtime_member).name
        and reproduction.get("invocation_manifest_bytes")
        == len(invocation_payload)
        and reproduction.get("invocation_manifest_sha256")
        == sha256(invocation_payload).hexdigest()
        and reproduction.get("invocation_manifest_file")
        == Path(invocation_member).name
        and reproduction.get("stderr_bytes") == len(stderr_payload)
        and reproduction.get("stderr_sha256")
        == sha256(stderr_payload).hexdigest()
        and reproduction.get("stderr_file") == Path(stderr_member).name
        and reproduction.get("bootstrap_sha256")
        == executions[1].get("bootstrap_sha256")
        and reproduction.get("auditor_source_sha256")
        == executions[1].get("auditor_source_sha256")
        and reproduction.get("support_files") == executions[1].get("support_files")
        and reproduction.get("passed") is True
        and reproduction.get("first_execution_receipt_file")
        == "audit-execution-01.execution.json"
        and reproduction.get("first_execution_receipt_bytes") == len(execution_payloads[0])
        and reproduction.get("first_execution_receipt_sha256")
        == sha256(execution_payloads[0]).hexdigest()
        and reproduction.get("replay_execution_receipt_file")
        == "audit-execution-02.execution.json"
        and reproduction.get("replay_execution_receipt_bytes") == len(execution_payloads[1])
        and reproduction.get("replay_execution_receipt_sha256")
        == sha256(execution_payloads[1]).hexdigest(),
        "archived audit reproduction does not bind its execution receipts",
    )
    capacity = reproduction.get("capacity")
    _verify_audit_capacity_projection(
        capacity,
        producer_manifest_sha256=sha256(manifest_payload).hexdigest(),
    )
    assert isinstance(capacity, dict)
    _require(
        capacity.get("development_total_bytes") == total
        and capacity.get("development_result_bytes") == result_bytes
        and capacity.get("first_elapsed_ns")
        == executions[0].get("subprocess_elapsed_ns")
        and capacity.get("replay_elapsed_ns")
        == executions[1].get("subprocess_elapsed_ns"),
        "archived audit capacity inputs differ from durable evidence",
    )
    return capacity


def _verify_audit_capacity_projection(
    value: object,
    *,
    producer_manifest_sha256: object,
) -> None:
    """Independently recompute the sealed v1.20 liveness arithmetic."""

    fields = {
        "schema", "producer_manifest_sha256", "development_total_bytes",
        "development_result_bytes", "first_elapsed_ns", "replay_elapsed_ns",
        "calibration_elapsed_ns", "producer_limit_bytes", "result_limit_bytes",
        "safety_numerator", "safety_denominator", "aggregate_required_ns",
        "result_required_ns", "combined_required_ns", "available_timeout_ns",
        "passed",
    }
    integer_fields = fields - {"schema", "producer_manifest_sha256", "passed"}
    _require(
        isinstance(value, dict)
        and set(value) == fields
        and value.get("schema") == "prospect.wm001.audit-capacity.v1"
        and value.get("producer_manifest_sha256") == producer_manifest_sha256
        and value.get("passed") is True
        and all(type(value.get(field)) is int for field in integer_fields),
        "accepted-closure audit capacity has an invalid shape",
    )
    assert isinstance(value, dict)
    total = cast(int, value["development_total_bytes"])
    result = cast(int, value["development_result_bytes"])
    first = cast(int, value["first_elapsed_ns"])
    replay = cast(int, value["replay_elapsed_ns"])
    _require(
        0 < total <= (8 << 30)
        and 0 < result <= (2 << 30)
        and result <= total
        and first > 0
        and replay > 0,
        "accepted-closure audit capacity inputs are outside the sealed bounds",
    )
    calibration = max(first, replay)

    def ceil_div(numerator: int, denominator: int) -> int:
        return (numerator + denominator - 1) // denominator

    aggregate = ceil_div(2 * calibration * (8 << 30), total)
    result_required = ceil_div(2 * calibration * (2 << 30), result)
    combined = aggregate + result_required
    available = 10_800 * 1_000_000_000
    expected = {
        "calibration_elapsed_ns": calibration,
        "producer_limit_bytes": 8 << 30,
        "result_limit_bytes": 2 << 30,
        "safety_numerator": 2,
        "safety_denominator": 1,
        "aggregate_required_ns": aggregate,
        "result_required_ns": result_required,
        "combined_required_ns": combined,
        "available_timeout_ns": available,
    }
    _require(
        all(value.get(field) == expected_value for field, expected_value in expected.items())
        and combined <= available,
        "accepted-closure audit capacity arithmetic or headroom is invalid",
    )


def _verify_bound_coverage_runtime_identity(
    coverage: Mapping[str, object],
    *,
    dependencies: Mapping[str, object],
    runtime: Mapping[str, object],
    bound_python: object,
) -> None:
    """Compare coverage arithmetic to the bound runtime, never QA ambient state."""

    _require(isinstance(bound_python, dict), "bound audit Python identity is malformed")
    assert isinstance(bound_python, dict)
    version = bound_python.get("version")
    _require(
        set(bound_python) == {"executable", "resolved_executable", "sha256", "version"}
        and bound_python.get("executable") == dependencies.get("python_executable")
        and bound_python.get("sha256") == dependencies.get("python_executable_sha256")
        and isinstance(version, list)
        and len(version) == 3
        and all(type(part) is int and cast(int, part) >= 0 for part in version),
        "bound audit Python identity differs from dependency custody",
    )
    expected_version = ".".join(str(part) for part in cast(list[int], version))
    _require(
        coverage.get("semantics_id") == COVERAGE_SEMANTICS
        and coverage.get("python_executable") == dependencies.get("python_executable")
        and coverage.get("python_implementation") == "CPython"
        and coverage.get("python_version") == expected_version
        and coverage.get("platform") == runtime.get("platform")
        and coverage.get("machine") == runtime.get("machine"),
        "bound coverage arithmetic differs from its recorded runtime identity",
    )


def _parse_timestamp(value: object, field: str) -> datetime:
    _require(isinstance(value, str), f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Violation(f"{field} is not an ISO-8601 timestamp") from exc
    _require(parsed.tzinfo is not None, f"{field} must include a UTC offset")
    return parsed


def _verify_implementation_manifest(value: object) -> None:
    """Require the exact complete ordered implementation manifest."""

    from .binding import implementation_files

    expected = implementation_files()
    _require(
        value == expected,
        "binding implementation_files differs from the exact complete ordered live implementation manifest",
    )


def _verify_pendulum_conformance_report(report: object) -> None:
    """Independently enforce the fixed formal Pendulum conformance contract."""

    _require(isinstance(report, dict), "bound Pendulum conformance report is not an object")
    assert isinstance(report, dict)
    _require(
        set(report) == FORMAL_CONFORMANCE_KEYS,
        "bound Pendulum conformance report fields differ from the formal contract",
    )
    _require(
        report.get("schema") == "prospect.wm001.pendulum-conformance.v1",
        "bound Pendulum conformance report has the wrong schema",
    )
    _require(
        report.get("environment_id") == "Pendulum-v1",
        "bound Pendulum conformance report has the wrong environment",
    )
    _require(
        isinstance(report.get("gymnasium_version"), str) and bool(report["gymnasium_version"]),
        "bound Pendulum conformance report has no Gymnasium version",
    )
    _require(
        report.get("seed") == 20260717 and report.get("samples_per_task") == 512 and report.get("cases") == 1024,
        "bound Pendulum conformance report does not contain exactly 512 cases per task from seed 20260717",
    )
    _require(
        report.get("semantic_parameters") == FORMAL_CONFORMANCE_PARAMETERS,
        "bound Pendulum semantic parameters changed",
    )
    _require(
        report.get("semantic_parameter_absolute_errors") == {name: 0.0 for name in FORMAL_CONFORMANCE_PARAMETERS},
        "bound Pendulum semantic parameters differ from Gymnasium",
    )
    _require(
        report.get("spec_horizon") == 200,
        "bound Pendulum horizon changed",
    )
    _require(
        report.get("terminated_or_truncated_cases") == 0,
        "bound Pendulum conformance cases terminated or truncated",
    )
    _require(
        all(report.get(field) == expected for field, expected in FORMAL_CONFORMANCE_TOLERANCES.items()),
        "bound Pendulum conformance tolerances changed",
    )
    _require(
        report.get("planner_dtype") == "float32",
        "bound Pendulum planner conformance dtype changed",
    )
    error_limits = {
        "max_observation_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["observation_atol"],
        "max_reward_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["reward_atol"],
        "max_planner_observation_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["planner_observation_atol"],
        "max_planner_reward_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["planner_reward_atol"],
    }
    for field, limit in error_limits.items():
        value = report.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= limit,
            f"bound Pendulum conformance {field} exceeds its fixed tolerance",
        )
    _require(
        report.get("passed") is True,
        "bound Pendulum conformance report did not pass",
    )
    body = dict(report)
    report_sha256 = body.pop("report_sha256", None)
    _require_sha256(report_sha256, "Pendulum conformance report_sha256")
    _require(
        report_sha256 == _canonical_sha256(body),
        "bound Pendulum conformance self-hash changed",
    )


def _binary64_pit_covered(pit: float) -> bool:
    _require(math.isfinite(pit), "coverage conformance PIT is non-finite")
    numerator, denominator = pit.as_integer_ratio()
    return 20 * numerator >= denominator and 20 * numerator <= 19 * denominator


def _verify_coverage_conformance_report(
    report: object,
    *,
    recorded_runtime: Mapping[str, object] | None = None,
) -> None:
    """Verify coverage semantics against live or explicitly recorded identity."""

    _require(isinstance(report, dict), "bound coverage conformance report is not an object")
    assert isinstance(report, dict)
    expected_runtime: Mapping[str, object]
    if recorded_runtime is None:
        expected_runtime = {
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
    else:
        expected_runtime = recorded_runtime
    _require(
        report.get("schema") == "prospect.wm001.coverage-conformance.v1"
        and report.get("semantics_id") == COVERAGE_SEMANTICS
        and report.get("python_executable") == expected_runtime.get("python_executable")
        and report.get("python_implementation") == expected_runtime.get("python_implementation") == "CPython"
        and report.get("python_version") == expected_runtime.get("python_version")
        and report.get("platform") == expected_runtime.get("platform")
        and report.get("machine") == expected_runtime.get("machine"),
        "bound coverage conformance runtime identity changed",
    )
    rows = report.get("cases")
    _require(isinstance(rows, list), "bound coverage conformance cases are missing")
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
    _require(len(rows) == len(direct_expected) + 1, "coverage conformance case count changed")
    for row, (case_id, pit, expected) in zip(rows[: len(direct_expected)], direct_expected, strict=True):
        _require(
            isinstance(row, dict)
            and row.get("case_id") == case_id
            and row.get("kind") == "binary64_pit"
            and row.get("pit_hex") == pit.hex()
            and row.get("expected_covered") is expected
            and row.get("observed_covered") is expected
            and row.get("passed") is True
            and _binary64_pit_covered(float.fromhex(str(row.get("pit_hex")))) is expected,
            f"coverage conformance direct case {case_id} changed or failed",
        )
    regression = rows[-1]
    _require(isinstance(regression, dict), "coverage regression case is malformed")
    _require(
        regression.get("case_id") == "v130-disclosed-boundary-coordinate"
        and regression.get("kind") == "float32_mixture_inputs"
        and regression.get("target_little_endian_f32_hex") == _V130_BOUNDARY_TARGET_F32_HEX
        and tuple(regression.get("member_means_little_endian_f32_hex", ())) == _V130_BOUNDARY_MEANS_F32_HEX
        and tuple(regression.get("member_log_variances_little_endian_f32_hex", ()))
        == _V130_BOUNDARY_LOG_VARIANCES_F32_HEX,
        "coverage v1.3 boundary regression inputs changed",
    )
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
    _require(
        pit.hex() == "0x1.999998b3745adp-5"
        and regression.get("expected_pit_hex") == pit.hex()
        and regression.get("observed_pit_hex") == pit.hex()
        and regression.get("expected_covered") is False
        and regression.get("observed_covered") is False
        and regression.get("passed") is True
        and not _binary64_pit_covered(pit),
        "coverage v1.3 boundary regression did not reproduce exactly",
    )
    corpus = {"semantics_id": COVERAGE_SEMANTICS, "cases": rows}
    _require(
        report.get("corpus_sha256") == _canonical_sha256(corpus),
        "coverage conformance corpus digest changed",
    )
    body = dict(report)
    report_sha256 = body.pop("report_sha256", None)
    _require_sha256(report_sha256, "coverage conformance report_sha256")
    _require(
        report_sha256 == _canonical_sha256(body) and report.get("passed") is True,
        "coverage conformance report self-hash or pass status changed",
    )


def derive_seed(namespace: str, master_seed: int, index: int) -> int:
    """Derive the exact protocol-1.20.0 uint32 seed."""

    payload = f"WM-001|1.20.0|{namespace}|{master_seed}|{index}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)


def derive_master_seed(lane: str, index: int) -> int:
    """Derive one protocol-1.20.0 lane master from its prospective index."""

    if lane not in {"development", "formal"} or index < 0:
        raise ValueError("invalid WM-001 master-seed lane or index")
    payload = f"WM-001|1.20.0|{lane}-master|{index}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)


def _expected_oscillator_conformance() -> dict[str, object]:
    """Recompute the bound oscillator report without producer runtime imports."""

    cases = 512
    steps = 200
    seed = 20260718
    trajectory = sha256()
    for case_index in range(cases):
        reset_seed = seed + case_index
        digest = sha256(f"{INDEPENDENT_OSCILLATOR_SOURCE}:{reset_seed}".encode("ascii")).digest()
        phase_unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        velocity_unit = int.from_bytes(digest[8:16], "big") / float(1 << 64)
        phase = (2.0 * phase_unit - 1.0) * math.pi
        velocity = 0.5 + velocity_unit
        reset_observation = (math.cos(phase), math.sin(phase), velocity)
        trajectory.update(reset_seed.to_bytes(8, "big", signed=False))
        trajectory.update(struct.pack("<3d", *reset_observation))
        for step_index in range(steps):
            phase = math.remainder(phase + 0.05 * velocity, 2.0 * math.pi)
            observation = (math.cos(phase), math.sin(phase), velocity)
            reward = math.cos(phase)
            trajectory.update(struct.pack("<3d", *observation))
            trajectory.update(struct.pack("<d", reward))
            trajectory.update(bytes((0, int(step_index == steps - 1))))
    report: dict[str, object] = {
        "schema": "prospect.wm001.independent-phase-oscillator-conformance.v1",
        "source_id": INDEPENDENT_OSCILLATOR_SOURCE,
        "cases": cases,
        "steps_per_case": steps,
        "seed": seed,
        "max_reset_absolute_difference": 0.0,
        "max_action_pair_observation_absolute_difference": 0.0,
        "max_action_pair_reward_absolute_difference": 0.0,
        "unexpected_terminations": 0,
        "premature_or_missing_truncations": 0,
        "trajectory_sha256": trajectory.hexdigest(),
        "passed": True,
    }
    report["report_sha256"] = sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return report


def verify_protocol() -> dict[str, Any]:
    """Verify the raw-byte seal and non-negotiable internal invariants."""

    protocol = _load_json(PROTOCOL_PATH)
    result_schema = _load_json(RESULT_SCHEMA_PATH)
    binding_schema = _load_json(BINDING_SCHEMA_PATH)

    try:
        seal_rows = [line.split() for line in SEAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise Violation(f"cannot load protocol seal: {exc}") from exc
    _require(all(len(row) == 2 for row in seal_rows), "invalid protocol seal format")
    seals = {relative: digest for digest, relative in seal_rows}
    expected_sealed_files = {
        "protocol.json": PROTOCOL_PATH,
        "schemas/formal-binding.schema.json": BINDING_SCHEMA_PATH,
        "schemas/raw-result.schema.json": RESULT_SCHEMA_PATH,
    }
    _require(set(seals) == set(expected_sealed_files), "sealed evidence file set changed")
    for relative, path in expected_sealed_files.items():
        _require_sha256(seals[relative], f"{relative} seal")
        _require(_file_sha256(path) == seals[relative], f"{relative} bytes do not match SEALED_PROTOCOL.sha256")

    experiment = protocol.get("experiment", {})
    trust_model = protocol.get("trust_model", {})
    _require(protocol.get("schema") == "prospect.world-model-lifecycle.protocol.v9", "wrong protocol schema")
    _require(
        trust_model
        == {
            "id": ASSURANCE["trust_model_id"],
            "tamper_resistant": ASSURANCE["tamper_resistant"],
            "external_attestation": ASSURANCE["external_attestation"],
            "exclusive_path_use_required": ASSURANCE["exclusive_path_use_required"],
            "statement": TRUST_MODEL_STATEMENT,
        },
        "protocol trust model is missing or overstated",
    )
    _require(experiment.get("id") == "WM-001", "wrong experiment ID")
    _require(experiment.get("protocol_version") == "1.20.0", "wrong protocol version")
    _require(experiment.get("status") == "sealed_before_formal_outcomes", "protocol is not marked sealed")
    _require(experiment.get("thresholds_sealed_before_outcomes") is True, "experiment thresholds are not sealed")
    _require(protocol.get("thresholds", {}).get("sealed_before_outcomes") is True, "threshold block is not sealed")
    scientific_continuity = experiment.get("revision", {}).get("scientific_continuity", {})
    _require(
        experiment.get("revision", {}).get("supersedes") == "1.19.0"
        and experiment.get("revision", {}).get("superseded_protocol_sha256")
        == _V1190_PROTOCOL_SHA256,
        "v1.20 protocol does not directly and exactly supersede sealed v1.19",
    )
    scientific_payload = {name: protocol.get(name) for name in _SCIENTIFIC_BLOCKS}
    _require(
        tuple(scientific_continuity.get("unchanged_top_level_blocks", ())) == _SCIENTIFIC_BLOCKS
        and scientific_continuity.get("v1_4_scientific_blocks_sha256") == _V140_SCIENTIFIC_BLOCKS_SHA256
        and _canonical_sha256(scientific_payload) == _V140_SCIENTIFIC_BLOCKS_SHA256,
        "v1.20 scientific blocks differ from the sealed v1.4 system",
    )
    _require(
        scientific_continuity.get("kernel_source_sha256") == _SCIENTIFIC_KERNEL_SHA256
        and all(_file_sha256(HERE / name) == digest for name, digest in _SCIENTIFIC_KERNEL_SHA256.items()),
        "v1.20 scientific kernel source differs from the sealed v1.4 system",
    )
    audit_contract = protocol.get("bindings", {}).get("audit_execution", {})
    _require(
        isinstance(audit_contract, dict)
        and audit_contract.get("runtime_manifest_schema")
        == "prospect.wm001.audit-runtime-manifest.v2"
        and audit_contract.get("execution_roles")
        == {
            "conformance": {"timeout_seconds": 600},
            "outcome_audit": {"timeout_seconds": 10_800},
        }
        and audit_contract.get("capacity")
        == {
            "schema": "prospect.wm001.audit-capacity.v1",
            "captured_execution_schema": "prospect.wm001.captured-audit-execution.v2",
            "audit_reproduction_schema": "prospect.wm001.audit-reproduction.v3",
            "adjudication_execution_schema": (
                "prospect.wm001.adjudication-audit-execution.v3"
            ),
            "producer_limit_bytes": 8 << 30,
            "result_limit_bytes": 2 << 30,
            "safety_numerator": 2,
            "safety_denominator": 1,
            "calibration_rule": (
                "Use max(first development outcome-audit subprocess_elapsed_ns, "
                "byte-identical development replay subprocess_elapsed_ns); report, "
                "stderr, runtime, and invocation bytes must all match, and the closure "
                "retains both exact captured execution receipts."
            ),
            "aggregate_required_ns": (
                "ceil(safety_numerator * calibration_elapsed_ns * producer_limit_bytes / "
                "(safety_denominator * development_total_bytes))"
            ),
            "result_required_ns": (
                "ceil(safety_numerator * calibration_elapsed_ns * result_limit_bytes / "
                "(safety_denominator * development_result_bytes))"
            ),
            "acceptance_rule": (
                "aggregate_required_ns + result_required_ns <= "
                "outcome_audit.timeout_seconds * 1000000000"
            ),
            "enforcement_rule": (
                "The runner and embedded bootstrap enforce the role-selected timeout "
                "and the runner captures elapsed time. Operator and binding stages "
                "preserve and join the receipts. Launcher, central verifier, independent "
                "auditor, and adjudicator reopen the exact runtime and canonical "
                "producer-target invocation plus both durable receipts and independently "
                "recompute capacity."
            ),
        },
        "v1.20 audit role/capacity contract changed",
    )

    _require(protocol.get("splits", {}).get("unit") == "whole_episode", "splits are not whole-episode")
    formal_splits = protocol.get("splits", {}).get("formal", {})
    for split in (
        "collect_a",
        "collect_b",
        "collect_irrelevant",
        "predictive_validation_a",
        "predictive_validation_b",
        "predictive_validation_irrelevant",
    ):
        _require(formal_splits.get(split, {}).get("episodes_per_replicate") == 8, f"{split} episode budget changed")
        _require(formal_splits.get(split, {}).get("transitions_per_replicate") == 1600, f"{split} step budget changed")
    _require(
        formal_splits.get("collect_irrelevant", {}).get("task") == "task_irrelevant",
        "irrelevant collection split is not bound to task_irrelevant",
    )
    _require(
        formal_splits.get("predictive_validation_irrelevant", {}).get("task") == "task_irrelevant",
        "irrelevant validation split is not bound to task_irrelevant",
    )
    for split in ("behavior_evaluation_a", "behavior_evaluation_b"):
        _require(
            formal_splits.get(split, {}).get("episodes_per_replicate") == 32,
            f"{split} must use 32 reset seeds",
        )

    seed_schedule = protocol.get("seed_schedule", {})
    _require(
        seed_schedule.get("derivation_domain_version") == "1.20.0",
        "seed derivation domain differs from protocol 1.20.0",
    )
    formal_seeds = tuple(seed_schedule.get("formal_replicate_master_seeds", ()))
    development_seeds = tuple(seed_schedule.get("development_replicate_master_seeds", ()))
    _require(formal_seeds == FORMAL_SEEDS, "formal master seeds changed")
    _require(development_seeds == DEVELOPMENT_SEEDS, "development master seeds changed")
    _require(set(formal_seeds).isdisjoint(development_seeds), "development and formal seeds overlap")
    _require(protocol.get("budgets", {}).get("formal_replicates") == 8, "formal replicate count changed")
    namespaces = seed_schedule.get("namespaces", {})
    actual_seed_counts = {
        namespace: declaration.get("count")
        for namespace, declaration in namespaces.items()
        if isinstance(declaration, dict)
    }
    _require(
        actual_seed_counts == EXPECTED_SEED_COUNTS,
        "seed namespace/count schedule differs from protocol 1.20.0",
    )
    master_derivation = seed_schedule.get("master_seed_derivation", {})
    _require(
        isinstance(master_derivation, dict)
        and master_derivation.get("lane_index_domains")
        == {
            "development": [0, 1],
            "formal": [0, 7],
        },
        "master-seed derivation index domains changed",
    )
    _require(
        DEVELOPMENT_SEEDS == tuple(derive_master_seed("development", index) for index in range(2))
        and FORMAL_SEEDS == tuple(derive_master_seed("formal", index) for index in range(8)),
        "master seeds do not match their prospective SHA-256 derivation",
    )
    current_masters = set((*DEVELOPMENT_SEEDS, *FORMAL_SEEDS))
    current_stream_values = [
        derive_seed(namespace, master_seed, index)
        for master_seed in (*DEVELOPMENT_SEEDS, *FORMAL_SEEDS)
        for namespace, count in EXPECTED_SEED_COUNTS.items()
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
        ("1.16.0", _V1160_MASTER_SEEDS),
        ("1.17.0", _V1170_MASTER_SEEDS),
        ("1.18.0", _V1180_MASTER_SEEDS),
        ("1.19.0", _V1190_MASTER_SEEDS),
    )
    prior_masters = {master_seed for _, version_masters in prior_domains for master_seed in version_masters}
    prior_stream_values = [
        int.from_bytes(
            sha256(f"WM-001|{version}|{namespace}|{master_seed}|{index}".encode()).digest()[:4],
            "big",
            signed=False,
        )
        for version, version_masters in prior_domains
        for master_seed in version_masters
        for namespace, count in EXPECTED_SEED_COUNTS.items()
        for index in range(count)
    ]
    prior_streams = set(prior_stream_values)
    collision_audit = master_derivation.get("collision_audit", {})
    _require(
        isinstance(collision_audit, dict)
        and collision_audit.get("current_master_seed_count") == 10
        and collision_audit.get("current_derived_stream_count") == 1360
        and collision_audit.get("unique_current_derived_stream_count") == len(current_streams) == 1360
        and collision_audit.get("current_internal_collision_count") == 0
        and collision_audit.get("current_master_stream_overlap_count") == 0
        and current_masters.isdisjoint(current_streams)
        and collision_audit.get("prior_master_seed_count") == len(prior_masters) == 190
        and collision_audit.get("unique_prior_derived_stream_count") == len(prior_streams) == 25840
        and len(prior_stream_values) == 25840
        and collision_audit.get("current_prior_master_master_overlap_count") == 0
        and collision_audit.get("current_prior_stream_stream_overlap_count") == 0
        and collision_audit.get("current_master_prior_stream_overlap_count") == 0
        and collision_audit.get("prior_master_current_stream_overlap_count") == 0
        and current_masters.isdisjoint(prior_masters)
        and current_streams.isdisjoint(prior_streams)
        and current_masters.isdisjoint(prior_streams)
        and prior_masters.isdisjoint(current_streams),
        "master or derived seed collision audit failed",
    )
    development_qualification = protocol.get("bindings", {}).get(
        "development_qualification",
        {},
    )
    _require(
        isinstance(development_qualification, dict)
        and development_qualification.get("matrix_contract_sha256")
        == DEVELOPMENT_MATRIX_CONTRACT_SHA256,
        "development matrix contract digest differs from the protocol-bound v1.20 identity",
    )
    _require(
        protocol.get("bindings", {}).get("preformal_authorization")
        == _PREFORMAL_AUTHORIZATION_CONTRACT,
        "preformal authorization contract differs from the sealed v1.20 "
        "gate",
    )

    tasks = protocol.get("tasks", {})
    _require(
        tasks.get("context_encoding", {}).get("task_irrelevant") == 2.0,
        "irrelevant task context encoding changed",
    )
    irrelevant_task = tasks.get("task_irrelevant", {})
    _require(
        irrelevant_task.get("id") == TASK_IRRELEVANT and irrelevant_task.get("context") == 2.0,
        "irrelevant task identity or context changed",
    )

    model = protocol.get("representation_and_model", {}).get("world_model", {})
    planner = protocol.get("representation_and_model", {}).get("planner", {})
    _require(model.get("ensemble_members") == 5, "world-model ensemble size changed")
    _require(planner.get("learning_during_evaluation") is False, "evaluation learning must be disabled")
    _require(planner.get("replay_writes_during_evaluation") is False, "evaluation replay writes must be disabled")
    predictive_secondary = protocol.get("metrics", {}).get("prediction", {}).get("secondary", ())
    coverage_metric = next(
        (
            row
            for row in predictive_secondary
            if isinstance(row, dict) and row.get("name") == "heldout_90_percent_interval_coverage"
        ),
        None,
    )
    _require(
        isinstance(coverage_metric, dict)
        and coverage_metric.get("semantics_id") == COVERAGE_SEMANTICS
        and coverage_metric.get("raw_fields")
        == [
            "interval_90_covered_target_count",
            "coverage_target_count",
            "interval_90_coverage",
            "coverage_semantics",
        ],
        "coverage metric semantics or raw fields changed",
    )
    k3_thresholds = protocol.get("thresholds", {}).get("k3_predictive_learning", {})
    _require(
        k3_thresholds.get("after_a_90_percent_interval_coverage_bounds_inclusive") == [0.7, 0.99]
        and "10*C >= 7*T" in str(k3_thresholds.get("after_a_coverage_decision_arithmetic"))
        and "100*C <= 99*T" in str(k3_thresholds.get("after_a_coverage_decision_arithmetic")),
        "coverage thresholds or count-space decision arithmetic changed",
    )
    _require(
        protocol.get("experience_and_learning", {}).get("task_a_update", {}).get("optimizer_steps") == 2000,
        "task-A update budget changed",
    )
    _require(
        protocol.get("experience_and_learning", {}).get("task_a_irrelevant_update", {}).get("optimizer_steps") == 2000,
        "irrelevant-evidence update budget changed",
    )
    _require(
        tuple(
            protocol.get("experience_and_learning", {}).get("task_a_irrelevant_update", {}).get("eligible_replay", ())
        )
        == ("collect_irrelevant",),
        "irrelevant-evidence update eligibility changed",
    )
    _require(
        protocol.get("experience_and_learning", {}).get("task_b_replay_update", {}).get("optimizer_steps") == 2000,
        "task-B update budget changed",
    )

    _require(
        set(protocol.get("controls", {}))
        == {
            "frozen",
            "corrupted_target",
            "irrelevant_evidence",
            "naive_sequential",
            "random_policy",
            "true_dynamics_mpc",
        },
        "control set changed",
    )
    _require(tuple(item.get("gate") for item in protocol.get("killing_order", ())) == GATES, "killing order changed")
    _require(
        tuple(protocol.get("bindings", {}).get("checkpoint", {}).get("canonical_component_ids", ()))
        == REQUIRED_COMPONENTS,
        "checkpoint component contract changed",
    )
    development_lane = protocol.get("lanes", {}).get("development", {})
    formal_lane = protocol.get("lanes", {}).get("formal", {})
    _require(
        development_lane.get("tuning_allowed") is False
        and "K3-K6 development performance values are descriptive" in str(development_lane.get("rule"))
        and tuple(development_lane.get("master_seeds", ())) == DEVELOPMENT_SEEDS,
        "development rehearsal governance changed",
    )
    _require(
        tuple(formal_lane.get("master_seeds", ())) == FORMAL_SEEDS
        and "sole permitted formal attempt" in str(formal_lane.get("launch_rule"))
        and "no same-version formal rerun" in str(formal_lane.get("launch_rule"))
        and "rehearsal-claim.json" in str(formal_lane.get("launch_rule"))
        and "rehearsal-terminal.json" in str(formal_lane.get("launch_rule"))
        and "Only the exact live accepted package authorizes formal execution"
        in str(formal_lane.get("launch_rule"))
        and "prospect.wm001.formal-launch.v3"
        in str(formal_lane.get("launch_rule")),
        "single-attempt formal governance changed",
    )
    _require(
        set(protocol.get("bindings", {}).get("coverage_arithmetic", {}).get("required", ()))
        >= {
            "semantics identifier",
            "exact CPython executable and platform identity",
            "content-addressed conformance corpus and report",
            "producer-reference and independent-auditor agreement",
        },
        "coverage arithmetic binding contract is incomplete",
    )
    _require(
        result_schema.get("$id") == "https://prospect.local/schemas/wm-001-raw-result-v9.json",
        "wrong raw-result schema",
    )
    _require(
        binding_schema.get("$id") == "https://prospect.local/schemas/wm-001-formal-binding-v10.json",
        "wrong formal-binding schema",
    )
    return protocol


def verify_binding(path: Path) -> dict[str, Any]:
    """Verify a completed implementation binding before formal execution."""

    protocol = verify_protocol()
    binding = _load_json(path)
    _validate_json_schema(binding, _load_json(BINDING_SCHEMA_PATH), label="formal binding")
    _require(binding.get("schema") == "prospect.world-model-lifecycle.formal-binding.v10", "wrong binding schema")
    _require(binding.get("experiment_id") == "WM-001", "binding has wrong experiment")
    _require(
        binding.get("assurance") == ASSURANCE,
        "binding assurance boundary is missing or overstated",
    )
    _parse_timestamp(binding.get("sealed_at_utc"), "sealed_at_utc")

    bound_protocol = binding.get("protocol", {})
    _require(bound_protocol.get("version") == "1.20.0", "binding has wrong protocol version")
    _require(bound_protocol.get("sha256") == _file_sha256(PROTOCOL_PATH), "binding has wrong protocol digest")
    _require(
        bound_protocol.get("raw_result_schema_sha256") == _file_sha256(RESULT_SCHEMA_PATH),
        "binding has wrong raw-result schema digest",
    )
    _require(
        bound_protocol.get("binding_schema_sha256") == _file_sha256(BINDING_SCHEMA_PATH),
        "binding has wrong binding-schema digest",
    )

    source = binding.get("source", {})
    _require(
        isinstance(source.get("git_commit"), str) and GIT_SHA_PATTERN.fullmatch(source["git_commit"]) is not None,
        "binding source commit is not a full Git SHA",
    )
    _require(
        isinstance(source.get("git_tree"), str) and GIT_SHA_PATTERN.fullmatch(source["git_tree"]) is not None,
        "binding source tree is not a full Git SHA",
    )
    _require(source.get("worktree_clean") is True, "formal binding requires a clean worktree")
    bound_implementation_files = source.get("implementation_files")
    _require(
        isinstance(bound_implementation_files, list) and bound_implementation_files,
        "binding has no implementation files",
    )
    seen_paths: set[str] = set()
    for index, entry in enumerate(bound_implementation_files):
        _require(isinstance(entry, dict), f"implementation_files[{index}] is not an object")
        relative = entry.get("path")
        _require(isinstance(relative, str) and relative, f"implementation_files[{index}].path is invalid")
        candidate = Path(relative)
        _require(not candidate.is_absolute() and ".." not in candidate.parts, f"unsafe implementation path: {relative}")
        _require(relative not in seen_paths, f"duplicate implementation path: {relative}")
        seen_paths.add(relative)
        actual = REPO / candidate
        _require(actual.is_file(), f"bound implementation file is missing: {relative}")
        _require(entry.get("bytes") == actual.stat().st_size, f"bound byte size changed: {relative}")
        _require(entry.get("sha256") == _file_sha256(actual), f"bound file digest changed: {relative}")
    _verify_implementation_manifest(bound_implementation_files)
    execution_source_sha256 = source.get("execution_source_sha256")
    expected_execution_sources = {
        filename: _file_sha256(HERE / filename)
        for filename in (
            "audit_runner.py",
            "artifact_audit.py",
            "adjudication.py",
            "binding.py",
            "launch_bootstrap.py",
            "operator.py",
            "preformal.py",
            "producer_bootstrap.py",
            "run.py",
            "verify.py",
        )
    }
    _require(
        execution_source_sha256 == expected_execution_sources,
        "binding execution-source identities differ from the exact live sources",
    )
    test_report_path = _binding_sibling(
        path,
        source.get("test_report_file"),
        "source.test_report_file",
    )
    _require(test_report_path.is_file(), "bound test report is missing")
    _require(
        source.get("test_report_bytes") == test_report_path.stat().st_size,
        "bound test report byte size changed",
    )
    _require(
        source.get("test_report_sha256") == _file_sha256(test_report_path),
        "bound test report digest changed",
    )
    from .binding import preformal_log_rows, verify_bound_machine_test_report

    test_report = verify_bound_machine_test_report(test_report_path, binding)
    _require(
        source.get("test_log_files") == preformal_log_rows(test_report_path, test_report),
        "bound preformal command-log custody differs from report v2",
    )

    dependencies = binding.get("dependencies", {})
    lockfile = dependencies.get("lockfile")
    _require(isinstance(lockfile, str) and lockfile, "binding lockfile path is missing")
    lock_path = REPO / lockfile
    _require(lock_path.is_file(), "bound dependency lockfile is missing")
    _require(dependencies.get("lockfile_sha256") == _file_sha256(lock_path), "dependency lockfile digest changed")
    python_executable = dependencies.get("python_executable")
    _require(
        isinstance(python_executable, str)
        and Path(python_executable).is_absolute()
        and Path(python_executable).is_file()
        and dependencies.get("python_executable_sha256") == _file_sha256(Path(python_executable)),
        "bound CPython executable is missing or differs",
    )
    standard_library = dependencies.get("standard_library")
    _require(
        isinstance(standard_library, dict)
        and standard_library.get("semantics_id") == "prospect.wm001.standard-library.v2"
        and isinstance(standard_library.get("path"), str)
        and Path(standard_library["path"]).is_absolute()
        and isinstance(standard_library.get("file_count"), int)
        and standard_library["file_count"] > 0
        and isinstance(standard_library.get("directory_count"), int)
        and standard_library["directory_count"] >= 0
        and isinstance(standard_library.get("total_bytes"), int)
        and standard_library["total_bytes"] > 0,
        "binding standard-library inventory is malformed",
    )
    _require_sha256(
        standard_library.get("inventory_sha256"),
        "dependencies.standard_library.inventory_sha256",
    )
    package_roots = dependencies.get("package_roots")
    _require(
        isinstance(package_roots, list) and package_roots,
        "binding package-root inventory is missing",
    )
    root_paths: list[str] = []
    for index, root in enumerate(package_roots):
        _require(isinstance(root, dict), f"package_roots[{index}] is malformed")
        root_path = root.get("path")
        _require(
            isinstance(root_path, str)
            and root.get("semantics_id") == "prospect.wm001.package-root.v2"
            and Path(root_path).is_absolute()
            and isinstance(root.get("file_count"), int)
            and root["file_count"] > 0
            and isinstance(root.get("directory_count"), int)
            and root["directory_count"] >= 0
            and isinstance(root.get("total_bytes"), int)
            and root["total_bytes"] > 0,
            f"package_roots[{index}] identity is malformed",
        )
        _require_sha256(root.get("inventory_sha256"), f"package_roots[{index}].inventory_sha256")
        root_paths.append(root_path)
    _require(root_paths == sorted(set(root_paths)), "binding package roots are duplicated or unordered")
    ownership = dependencies.get("package_ownership")
    _require(
        isinstance(ownership, dict)
        and set(ownership)
        == {
            "semantics_id",
            "root",
            "file_count",
            "directory_count",
            "shared_file_count",
            "identity_sha256",
        }
        and ownership.get("semantics_id") == "prospect.wm001.package-ownership.v1"
        and len(root_paths) == 1
        and ownership.get("root") == root_paths[0]
        and ownership.get("file_count") == package_roots[0].get("file_count")
        and ownership.get("directory_count") == package_roots[0].get("directory_count")
        and isinstance(ownership.get("shared_file_count"), int)
        and ownership["shared_file_count"] >= 0,
        "binding package-root ownership identity is malformed",
    )
    _require_sha256(
        ownership.get("identity_sha256"),
        "dependencies.package_ownership.identity_sha256",
    )
    packages = dependencies.get("packages")
    _require(isinstance(packages, list), "binding packages must be an array")
    package_names = {entry.get("name") for entry in packages if isinstance(entry, dict)}
    _require(REQUIRED_PACKAGES <= package_names, "binding is missing a required root package identity")
    _require(len(package_names) == len(packages), "binding package identities are duplicated")
    for package in packages:
        _require(
            isinstance(package.get("name"), str) and package["name"] == re.sub(r"[-_.]+", "-", package["name"]).lower(),
            "package name is not PEP-503 canonical",
        )
        _require(isinstance(package.get("version"), str) and package["version"], "package version is missing")
        _require_sha256(package.get("distribution_sha256"), f"{package.get('name')} distribution_sha256")
        _require(
            isinstance(package.get("declared_file_count"), int)
            and package["declared_file_count"] > 0
            and package.get("editable") is False,
            f"{package.get('name')} package file count or editable status is invalid",
        )

    runtime = binding.get("runtime", {})
    _require(runtime.get("deterministic_algorithms") is True, "formal runtime must enable deterministic algorithms")
    _require(
        runtime.get("python_flags")
        == {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
            "safe_path": True,
        },
        "formal producer runtime does not bind exact CPython -I -S -B flags",
    )
    process_environment = runtime.get("process_environment")
    _require(
        isinstance(process_environment, dict)
        and process_environment.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and process_environment.get("LAZY_LEGACY_OP") == "False"
        and process_environment.get("LC_ALL") == "C.UTF-8"
        and process_environment.get("PATH") == "/usr/bin:/bin"
        and process_environment.get("PYGAME_HIDE_SUPPORT_PROMPT") == "hide"
        and process_environment.get("SDL_AUDIODRIVER") == "dsp"
        and process_environment.get("TZ") == "UTC"
        and set(process_environment)
        <= {
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
        },
        "formal producer process environment is not the exact safe contract",
    )
    if runtime.get("device") == "cuda":
        _require(
            runtime.get("cublas_workspace_config") == ":4096:8",
            "formal CUDA runtime must bind CUBLAS_WORKSPACE_CONFIG=:4096:8",
        )
    else:
        _require(
            runtime.get("cublas_workspace_config") is None,
            "non-CUDA formal runtime must not bind a cuBLAS workspace",
        )

    audit_execution = binding.get("audit_execution")
    _require(
        isinstance(audit_execution, dict),
        "binding audit_execution block is missing",
    )
    _require(
        audit_execution.get("runner_source_sha256") == expected_execution_sources["audit_runner.py"]
        and audit_execution.get("auditor_source_sha256") == expected_execution_sources["artifact_audit.py"]
        and audit_execution.get("adjudicator_source_sha256") == expected_execution_sources["adjudication.py"],
        "bound audit runner, auditor, or adjudicator source identity changed",
    )
    _, bootstrap_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="bootstrap_source",
    )
    from .audit_runner import bootstrap_source_bytes

    _require(
        bootstrap_payload == bootstrap_source_bytes(),
        "bound audit bootstrap bytes differ from the executable bootstrap",
    )
    _, request_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="prebinding_request",
    )
    _, path_runtime_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="prebinding_path_runtime_manifest",
    )
    _, descriptor_runtime_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="prebinding_descriptor_runtime_manifest",
    )
    _, path_invocation_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="prebinding_path_invocation_manifest",
    )
    _, descriptor_invocation_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="prebinding_descriptor_invocation_manifest",
    )
    _, conformance_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="prebinding_conformance_report",
    )
    _, execution_receipt_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="prebinding_execution_receipt",
    )
    _, outcome_runtime_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="outcome_runtime_manifest",
    )
    _, restart_runtime_report_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="restart_runtime_conformance_report",
    )
    _, restart_runtime_receipt_payload = _bound_evidence_payload(
        path,
        audit_execution,
        prefix="restart_runtime_execution_receipt",
    )
    request_value = _canonical_object_payload(
        request_payload,
        label="prebinding request",
    )
    path_runtime_value = _canonical_object_payload(
        path_runtime_payload,
        label="prebinding path runtime manifest",
    )
    descriptor_runtime_value = _canonical_object_payload(
        descriptor_runtime_payload,
        label="prebinding descriptor runtime manifest",
    )
    path_invocation_value = _canonical_object_payload(
        path_invocation_payload,
        label="prebinding path invocation manifest",
    )
    descriptor_invocation_value = _canonical_object_payload(
        descriptor_invocation_payload,
        label="prebinding descriptor invocation manifest",
    )
    conformance_value = _canonical_object_payload(
        conformance_payload,
        label="prebinding conformance report",
    )
    execution_receipt_value = _canonical_object_payload(
        execution_receipt_payload,
        label="prebinding execution receipt",
    )
    outcome_runtime_value = _canonical_object_payload(
        outcome_runtime_payload,
        label="outcome audit runtime manifest",
    )
    restart_runtime_report_value = _canonical_object_payload(
        restart_runtime_report_payload,
        label="restart-runtime conformance report",
    )
    restart_runtime_receipt_value = _canonical_object_payload(
        restart_runtime_receipt_payload,
        label="restart-runtime execution receipt",
    )
    _require(
        request_value.get("schema") == "prospect.wm001.prebinding-conformance-request.v2"
        and conformance_value.get("schema") == "prospect.wm001.prebinding-conformance.v2"
        and conformance_value.get("passed") is True
        and conformance_value.get("request_sha256") == audit_execution.get("prebinding_request_identity_sha256"),
        "bound prebinding request or conformance report identity is invalid",
    )
    normalized_path_runtime = json.loads(json.dumps(path_runtime_value))
    normalized_descriptor_runtime = json.loads(json.dumps(descriptor_runtime_value))
    normalized_path_runtime["source"]["mode"] = "normalized"
    normalized_descriptor_runtime["source"]["mode"] = "normalized"
    _require(
        path_runtime_value.get("source", {}).get("mode") == "path"
        and descriptor_runtime_value.get("source", {}).get("mode") == "descriptor"
        and _strict_json_equal(
            normalized_path_runtime,
            normalized_descriptor_runtime,
        ),
        "prebinding path and descriptor runtime identities differ beyond source mode",
    )
    repeat_count = audit_execution.get("repeat_count")
    receipt_rows = execution_receipt_value.get("executions")
    expected_modes = ["path"] * repeat_count + ["descriptor"] * repeat_count if isinstance(repeat_count, int) else []
    stderr_identity = (
        receipt_rows[0].get("stderr")
        if isinstance(receipt_rows, list) and receipt_rows and isinstance(receipt_rows[0], dict)
        else None
    )
    valid_stderr_identity = (
        isinstance(stderr_identity, dict)
        and set(stderr_identity) == {"bytes", "sha256"}
        and stderr_identity.get("bytes") == 0
        and stderr_identity.get("sha256") == sha256(b"").hexdigest()
    )
    _require(
        execution_receipt_value.get("schema") == "prospect.wm001.audit-conformance-receipt.v1"
        and _strict_json_equal(
            execution_receipt_value.get("repeat_count"),
            repeat_count,
        )
        and _strict_json_equal(
            execution_receipt_value.get("execution_count"),
            len(expected_modes),
        )
        and execution_receipt_value.get("report_sha256")
        == _file_sha256(path.parent / str(audit_execution["prebinding_conformance_report_file"]))
        and execution_receipt_value.get("path_descriptor_byte_identical") is True
        and execution_receipt_value.get("execution_conformance_passed") is True
        and isinstance(receipt_rows, list)
        and len(receipt_rows) == len(expected_modes)
        and valid_stderr_identity
        and all(
            isinstance(row, dict)
            and set(row)
            == {
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
            and _strict_json_equal(row.get("ordinal"), ordinal)
            and row.get("source_mode") == mode
            and _strict_json_equal(row.get("returncode"), 0)
            and row.get("auditor_report_passed") is True
            and _strict_json_equal(
                row.get("stdout"),
                {
                    "bytes": len(conformance_payload),
                    "sha256": _file_sha256(
                        path.parent
                        / str(
                            audit_execution[
                                "prebinding_conformance_report_file"
                            ]
                        )
                    ),
                },
            )
            and _strict_json_equal(
                row.get("stderr"),
                stderr_identity,
            )
            and _strict_json_equal(
                row.get("runtime_manifest"),
                {
                    "bytes": len(
                        path_runtime_payload
                        if mode == "path"
                        else descriptor_runtime_payload
                    ),
                    "sha256": sha256(
                        path_runtime_payload
                        if mode == "path"
                        else descriptor_runtime_payload
                    ).hexdigest(),
                },
            )
            and _strict_json_equal(
                row.get("invocation_manifest"),
                {
                    "bytes": len(
                        path_invocation_payload
                        if mode == "path"
                        else descriptor_invocation_payload
                    ),
                    "sha256": sha256(
                        path_invocation_payload
                        if mode == "path"
                        else descriptor_invocation_payload
                    ).hexdigest(),
                },
            )
            and row.get("bootstrap_sha256") == audit_execution.get("bootstrap_source_sha256")
            and row.get("auditor_source_sha256") == audit_execution.get("auditor_source_sha256")
            and _strict_json_equal(
                row.get("support_files"),
                path_runtime_value.get("support_files"),
            )
            for ordinal, (row, mode) in enumerate(
                zip(receipt_rows, expected_modes, strict=True),
                start=1,
            )
        ),
        "bound prebinding receipt does not preserve every execution identity",
    )
    _require(
        path_invocation_value.get("runtime_manifest_sha256") == sha256(path_runtime_payload).hexdigest()
        and descriptor_invocation_value.get("runtime_manifest_sha256")
        == sha256(descriptor_runtime_payload).hexdigest(),
        "prebinding invocation manifests identify different runtime manifests",
    )
    expected_audit_environment = {key: value for key, value in process_environment.items() if key != "PATH"}
    expected_outcome_support = [
        "producer_bootstrap.py",
        "protocol.json",
        "schemas/raw-result.schema.json",
    ]
    expected_outcome_support_rows = [
        {
            "path": relative,
            "bytes": source_path.stat().st_size,
            "sha256": _file_sha256(source_path),
        }
        for relative, source_path in (
            (
                "producer_bootstrap.py",
                HERE / "producer_bootstrap.py",
            ),
            ("protocol.json", PROTOCOL_PATH),
            (
                "schemas/raw-result.schema.json",
                RESULT_SCHEMA_PATH,
            ),
        )
    ]
    outcome_support_rows = outcome_runtime_value.get("support_files")
    bound_audit_python = path_runtime_value.get("python")
    _require(
        set(outcome_runtime_value)
        == {
            "schema",
            "execution_role",
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
        and outcome_runtime_value.get("schema")
        == "prospect.wm001.audit-runtime-manifest.v2"
        and outcome_runtime_value.get("execution_role") == "outcome_audit"
        and outcome_runtime_value.get("assurance") == ASSURANCE
        and outcome_runtime_value.get("bootstrap_sha256")
        == audit_execution.get("bootstrap_source_sha256")
        and isinstance(bound_audit_python, dict)
        and bound_audit_python.get("executable") == python_executable
        and bound_audit_python.get("resolved_executable")
        == str(Path(cast(str, python_executable)).resolve(strict=True))
        and bound_audit_python.get("sha256")
        == dependencies.get("python_executable_sha256")
        and outcome_runtime_value.get("python") == bound_audit_python
        and outcome_runtime_value.get("required_flags")
        == {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
            "safe_path": True,
        }
        and outcome_runtime_value.get("source")
        == {
            "mode": "descriptor",
            "path": "artifact_audit.py",
            "bytes": (HERE / "artifact_audit.py").stat().st_size,
            "sha256": _file_sha256(
                HERE / "artifact_audit.py"
            ),
        }
        and outcome_runtime_value.get("closure_import_roots") == package_roots
        and outcome_runtime_value.get("standard_library") == standard_library
        and outcome_runtime_value.get("environment") == expected_audit_environment
        and outcome_runtime_value.get("limits")
        == {
            "timeout_seconds": 10_800,
            "stdout_bytes": 64 << 20,
            "stderr_bytes": 16 << 20,
        }
        and isinstance(outcome_support_rows, list)
        and outcome_support_rows == expected_outcome_support_rows
        and audit_execution.get("outcome_source_mode") == "descriptor"
        and audit_execution.get("outcome_support_files") == expected_outcome_support
        and audit_execution.get("outcome_argv_role")
        == [
            "<canonical-producer-root>",
            "--producer-bootstrap",
            "<captured-producer-bootstrap>",
        ]
        and audit_execution.get("outcome_working_directory") == str(REPO)
        and audit_execution.get("interpreter_flags") == ["-I", "-S", "-B"]
        and isinstance(audit_execution.get("repeat_count"), int)
        and audit_execution["repeat_count"] >= 3
        and audit_execution.get("path_descriptor_equal") is True
        and audit_execution.get("passed") is True,
        "bound outcome-audit execution role differs from the exact contract",
    )
    from .preformal import _runtime_bootstrap_conformance_from_report

    rehearsed_audit_execution = (
        _runtime_bootstrap_conformance_from_report(
            test_report_path.parent,
            test_report,
        )
    )
    rehearsed_inventory = {
        "packages": dependencies.get("packages"),
        "package_roots": dependencies.get("package_roots"),
        "standard_library": dependencies.get("standard_library"),
        "package_ownership": dependencies.get("package_ownership"),
    }
    _require(
        rehearsed_audit_execution.get("inventory")
        == rehearsed_inventory
        and rehearsed_audit_execution.get("inventory_sha256")
        == sha256(
            json.dumps(
                rehearsed_inventory,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        and rehearsed_audit_execution.get("conformance_sha256")
        == sha256(
            (
                json.dumps(
                    audit_execution,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            ).encode("utf-8")
        ).hexdigest()
        and rehearsed_audit_execution.get(
            "restart_runtime_conformance_report_sha256"
        )
        == sha256(restart_runtime_report_payload).hexdigest()
        and rehearsed_audit_execution.get(
            "restart_runtime_execution_receipt_sha256"
        )
        == sha256(restart_runtime_receipt_payload).hexdigest()
        and rehearsed_audit_execution.get(
            "restart_runtime_support_files"
        )
        == audit_execution.get("restart_runtime_support_files")
        and rehearsed_audit_execution.get(
            "restart_runtime_repeat_count"
        )
        == audit_execution.get("restart_runtime_repeat_count")
        and rehearsed_audit_execution.get(
            "restart_runtime_path_descriptor_equal"
        )
        == audit_execution.get(
            "restart_runtime_path_descriptor_equal"
        ),
        "bound audit execution differs from the sealed preformal rehearsal",
    )
    restart_repeat_count = audit_execution.get(
        "restart_runtime_repeat_count"
    )
    restart_support_files = [
        "producer_bootstrap.py",
        "protocol.json",
        "schemas/raw-result.schema.json",
    ]
    restart_negative_case_ids = [
        "missing-bootstrap-support",
        "extra-bootstrap-support",
        "mutated-bootstrap-identity",
        "development-formal-branch-substitution",
        "formal-development-branch-substitution",
    ]
    _require(
        set(restart_runtime_report_value)
        == {
            "schema",
            "protocol_version",
            "support_files",
            "branches",
            "negative_cases",
            "failure_code",
            "passed",
        }
        and restart_runtime_report_value.get("schema")
        == "prospect.wm001.restart-runtime-conformance.v1"
        and restart_runtime_report_value.get("protocol_version") == "1.20.0"
        and _strict_json_equal(
            restart_runtime_report_value.get("support_files"),
            outcome_support_rows,
        )
        and _strict_json_equal(
            restart_runtime_report_value.get("branches"),
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
        and _strict_json_equal(
            restart_runtime_report_value.get("negative_cases"),
            [
                {"case_id": case_id, "rejected": True}
                for case_id in restart_negative_case_ids
            ],
        )
        and restart_runtime_report_value.get("failure_code") is None
        and restart_runtime_report_value.get("passed") is True
        and audit_execution.get("restart_runtime_support_files")
        == restart_support_files
        and type(restart_repeat_count) is int
        and cast(int, restart_repeat_count) >= 3
        and restart_repeat_count == repeat_count
        and audit_execution.get("restart_runtime_path_descriptor_equal")
        is True,
        "bound restart-runtime conformance report is not one complete result-free pass",
    )

    def canonical_payload(value: object) -> bytes:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    descriptor_restart_runtime_value = json.loads(
        json.dumps(outcome_runtime_value)
    )
    descriptor_restart_runtime_value["execution_role"] = "conformance"
    descriptor_restart_runtime_value["limits"]["timeout_seconds"] = 600
    path_restart_runtime_value = json.loads(
        json.dumps(descriptor_restart_runtime_value)
    )
    path_restart_runtime_value["source"]["mode"] = "path"
    path_restart_runtime_payload = canonical_payload(
        path_restart_runtime_value
    )
    restart_arguments = [
        "--restart-runtime-conformance",
        "--producer-bootstrap",
        "@captured/producer_bootstrap.py",
        "--expected-producer-bootstrap-sha256",
        _file_sha256(HERE / "producer_bootstrap.py"),
    ]
    restart_runtime_payloads = {
        "path": path_restart_runtime_payload,
        "descriptor": canonical_payload(
            descriptor_restart_runtime_value
        ),
    }
    restart_invocation_payloads = {
        mode: canonical_payload(
            {
                "schema": (
                    "prospect.wm001.audit-invocation-manifest.v1"
                ),
                "runtime_manifest_sha256": sha256(
                    runtime_payload
                ).hexdigest(),
                "working_directory": str(REPO),
                "auditor_argv": restart_arguments,
            }
        )
        for mode, runtime_payload in restart_runtime_payloads.items()
    }
    restart_receipt_rows = restart_runtime_receipt_value.get(
        "executions"
    )
    restart_expected_modes = (
        ["path"] * cast(int, restart_repeat_count)
        + ["descriptor"] * cast(int, restart_repeat_count)
    )
    restart_stderr_identity = (
        restart_receipt_rows[0].get("stderr")
        if isinstance(restart_receipt_rows, list)
        and restart_receipt_rows
        and isinstance(restart_receipt_rows[0], dict)
        else None
    )
    valid_restart_stderr_identity = (
        isinstance(restart_stderr_identity, dict)
        and set(restart_stderr_identity) == {"bytes", "sha256"}
        and restart_stderr_identity.get("bytes") == 0
        and restart_stderr_identity.get("sha256")
        == sha256(b"").hexdigest()
    )
    restart_report_identity = {
        "bytes": len(restart_runtime_report_payload),
        "sha256": sha256(
            restart_runtime_report_payload
        ).hexdigest(),
    }
    _require(
        set(restart_runtime_receipt_value)
        == {
            "schema",
            "repeat_count",
            "execution_count",
            "executions",
            "report_sha256",
            "path_descriptor_byte_identical",
            "execution_conformance_passed",
        }
        and restart_runtime_receipt_value.get("schema")
        == "prospect.wm001.audit-conformance-receipt.v1"
        and _strict_json_equal(
            restart_runtime_receipt_value.get("repeat_count"),
            restart_repeat_count,
        )
        and _strict_json_equal(
            restart_runtime_receipt_value.get("execution_count"),
            len(restart_expected_modes),
        )
        and restart_runtime_receipt_value.get("report_sha256")
        == restart_report_identity["sha256"]
        and restart_runtime_receipt_value.get(
            "path_descriptor_byte_identical"
        )
        is True
        and restart_runtime_receipt_value.get(
            "execution_conformance_passed"
        )
        is True
        and isinstance(restart_receipt_rows, list)
        and len(restart_receipt_rows) == len(restart_expected_modes)
        and valid_restart_stderr_identity
        and all(
            isinstance(row, dict)
            and set(row)
            == {
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
            and _strict_json_equal(row.get("ordinal"), ordinal)
            and row.get("source_mode") == mode
            and _strict_json_equal(row.get("returncode"), 0)
            and _strict_json_equal(
                row.get("stdout"),
                restart_report_identity,
            )
            and _strict_json_equal(
                row.get("stderr"),
                restart_stderr_identity,
            )
            and _strict_json_equal(
                row.get("runtime_manifest"),
                {
                    "bytes": len(restart_runtime_payloads[mode]),
                    "sha256": sha256(
                        restart_runtime_payloads[mode]
                    ).hexdigest(),
                },
            )
            and _strict_json_equal(
                row.get("invocation_manifest"),
                {
                    "bytes": len(restart_invocation_payloads[mode]),
                    "sha256": sha256(
                        restart_invocation_payloads[mode]
                    ).hexdigest(),
                },
            )
            and row.get("bootstrap_sha256")
            == audit_execution.get("bootstrap_source_sha256")
            and row.get("auditor_source_sha256")
            == audit_execution.get("auditor_source_sha256")
            and _strict_json_equal(
                row.get("support_files"),
                outcome_support_rows,
            )
            and row.get("auditor_report_passed") is True
            for ordinal, (row, mode) in enumerate(
                zip(
                    restart_receipt_rows,
                    restart_expected_modes,
                    strict=True,
                ),
                start=1,
            )
        ),
        "bound restart-runtime receipt does not preserve every exact path and descriptor execution",
    )
    environment = binding.get("environment", {})
    _require(environment.get("id") == "Pendulum-v1", "binding has wrong environment")
    for field in ("wrapper_source_sha256", "installed_distribution_sha256", "conformance_report_sha256"):
        _require_sha256(environment.get(field), f"environment.{field}")
    conformance_path = _binding_sibling(
        path,
        environment.get("conformance_report_file"),
        "environment.conformance_report_file",
    )
    _require(conformance_path.is_file(), "bound Pendulum conformance report is missing")
    _require(
        environment.get("conformance_report_bytes") == conformance_path.stat().st_size,
        "bound Pendulum conformance report byte size changed",
    )
    _require(
        environment.get("conformance_report_sha256") == _file_sha256(conformance_path),
        "bound Pendulum conformance report digest changed",
    )
    conformance_report = _load_json(conformance_path)
    _verify_pendulum_conformance_report(conformance_report)
    expected_conformance_bytes = (
        json.dumps(
            conformance_report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(
        conformance_path.read_bytes() == expected_conformance_bytes,
        "bound Pendulum conformance report is not canonical JSON",
    )

    irrelevant_control = binding.get("irrelevant_control", {})
    _require(
        irrelevant_control.get("id") == TASK_IRRELEVANT
        and irrelevant_control.get("source_id") == INDEPENDENT_OSCILLATOR_SOURCE,
        "binding has the wrong irrelevant-control identity",
    )
    oscillator_source = HERE / "runtime_lane.py"
    _require(
        irrelevant_control.get("source_sha256") == _file_sha256(oscillator_source),
        "bound irrelevant-control source digest changed",
    )
    oscillator_conformance_path = _binding_sibling(
        path,
        irrelevant_control.get("conformance_report_file"),
        "irrelevant_control.conformance_report_file",
    )
    _require(
        oscillator_conformance_path.is_file(),
        "bound oscillator conformance report is missing",
    )
    _require(
        irrelevant_control.get("conformance_report_bytes") == oscillator_conformance_path.stat().st_size,
        "bound oscillator conformance report byte size changed",
    )
    _require(
        irrelevant_control.get("conformance_report_sha256") == _file_sha256(oscillator_conformance_path),
        "bound oscillator conformance report digest changed",
    )
    oscillator_conformance = _load_json(oscillator_conformance_path)
    _require(
        oscillator_conformance == _expected_oscillator_conformance(),
        "bound oscillator conformance report differs from independent semantics",
    )
    expected_oscillator_bytes = (
        json.dumps(
            oscillator_conformance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(
        oscillator_conformance_path.read_bytes() == expected_oscillator_bytes,
        "bound oscillator conformance report is not canonical JSON",
    )

    coverage_arithmetic = binding.get("coverage_arithmetic", {})
    _require(
        isinstance(coverage_arithmetic, dict),
        "bound coverage arithmetic block is malformed",
    )
    _verify_bound_coverage_runtime_identity(
        coverage_arithmetic,
        dependencies=dependencies,
        runtime=runtime,
        bound_python=bound_audit_python,
    )
    _require(
        coverage_arithmetic.get("producer_source_sha256") == _file_sha256(HERE / "model.py")
        and coverage_arithmetic.get("auditor_source_sha256") == _file_sha256(HERE / "artifact_audit.py")
        and coverage_arithmetic.get("formal_test_report_sha256") == source.get("test_report_sha256"),
        "bound coverage source or test-report digest changed",
    )
    coverage_conformance_path = _binding_sibling(
        path,
        coverage_arithmetic.get("conformance_report_file"),
        "coverage_arithmetic.conformance_report_file",
    )
    _require(coverage_conformance_path.is_file(), "bound coverage conformance report is missing")
    _require(
        coverage_arithmetic.get("conformance_report_bytes") == coverage_conformance_path.stat().st_size
        and coverage_arithmetic.get("conformance_report_sha256") == _file_sha256(coverage_conformance_path),
        "bound coverage conformance report size or digest changed",
    )
    coverage_conformance = _load_json(coverage_conformance_path)
    _verify_coverage_conformance_report(
        coverage_conformance,
        recorded_runtime=coverage_arithmetic,
    )
    expected_coverage_bytes = (
        json.dumps(
            coverage_conformance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(
        coverage_conformance_path.read_bytes() == expected_coverage_bytes,
        "bound coverage conformance report is not canonical JSON",
    )

    development = binding.get("development_qualification", {})
    development_closure_path = _binding_sibling(
        path,
        development.get("closure_file"),
        "development_qualification.closure_file",
    )
    _require(
        development_closure_path.is_file()
        and development.get("closure_bytes") == development_closure_path.stat().st_size
        and development.get("closure_sha256") == _file_sha256(development_closure_path),
        "bound development closure bytes changed",
    )
    (
        development_closure,
        expected_development,
        development_member_digests,
    ) = _recorded_development_closure_identity(
        development_closure_path,
    )
    _verify_archived_audit_capacity(
        development_closure,
    )
    accepted_closure_receipt = _recorded_accepted_closure_receipt(
        test_report_path,
        test_report,
        closure_sha256=cast(str, expected_development["closure_sha256"]),
        producer_manifest_sha256=development_member_digests[
            cast(str, development_closure["producer_manifest_member"])
        ],
        raw_result_sha256=development_member_digests[
            cast(str, development_closure["raw_result_member"])
        ],
    )
    _require(
        development == expected_development
        and development.get("git_commit")
        == source.get("git_commit")
        == cast(dict[str, object], development_closure["source"]).get("git_commit")
        and development.get("git_tree")
        == source.get("git_tree")
        == cast(dict[str, object], development_closure["source"]).get("git_tree")
        and development.get("engineering_verified") is True
        and development.get("audit_reproduced") is True
        and development.get("performance_values_bound") is False,
        "bound development qualification identity or clean status changed",
    )
    _require(
        accepted_closure_receipt.get("development_closure_sha256")
        == development.get("closure_sha256")
        and accepted_closure_receipt.get("producer_manifest_sha256")
        == development.get("producer_manifest_sha256")
        and accepted_closure_receipt.get("raw_result_sha256")
        == development.get("raw_result_sha256")
        == development.get("raw_result_sha256"),
        "binding differs from its accepted-closure runtime receipt",
    )

    checkpoint = binding.get("checkpoint_implementation", {})
    _require(tuple(checkpoint.get("component_ids", ())) == REQUIRED_COMPONENTS, "checkpoint components are incomplete")
    _require(
        tuple(binding.get("formal_replicate_master_seeds", ())) == FORMAL_SEEDS,
        "binding formal seeds do not match protocol",
    )
    _require(
        tuple(protocol["lanes"]["formal"]["master_seeds"]) == FORMAL_SEEDS,
        "protocol formal lane seed disagreement",
    )
    return binding


def _verify_formal_launch_record(
    path: Path,
    *,
    binding_sha256: str,
    execution: Mapping[str, Any],
) -> None:
    from .artifact import (
        FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
        FORMAL_BINDING_OUTER_COMPLETION_NAME,
        FORMAL_CONFIRMATION_NAME,
        FORMAL_RESULTS_ROOT,
    )
    from .operator import (
        FORMAL_BINDING_ATTEMPT_PATH,
        outer_completion_marker,
        verify_operator_attempt,
        verify_outer_completion,
    )
    try:
        from .rehearsal import accepted_binding_rehearsal_identity
    except ImportError as error:
        raise Violation(
            "formal launch rehearsal verifier cannot be imported"
        ) from error

    _require(path.name == "formal-launch.json", "formal launch record has the wrong filename")
    _require(path.is_file() and not path.is_symlink(), "formal launch record is missing or aliased")
    payload = path.read_bytes()
    record = _load_json(path)
    expected_fields = {
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
        "accepted_binding_rehearsal",
        "attempt_directory",
        "global_marker_file",
        "claimed_at_utc",
        "git_commit",
        "git_tree",
        "record_sha256",
    }
    body = dict(record)
    record_sha256 = body.pop("record_sha256", None)
    _require_sha256(record_sha256, "formal launch record_sha256")
    binding_attempt_terminal = FORMAL_BINDING_ATTEMPT_PATH / "operator-attempt.json"
    binding_attempt_completion = outer_completion_marker(binding_attempt_terminal)
    copied_attempt = path.parent / FORMAL_BINDING_ATTEMPT_MANIFEST_NAME
    copied_completion = path.parent / FORMAL_BINDING_OUTER_COMPLETION_NAME
    attempt_manifest = verify_operator_attempt(FORMAL_BINDING_ATTEMPT_PATH)
    attempt_primary = attempt_manifest.get("primary")
    completion = verify_outer_completion(binding_attempt_terminal)
    try:
        rehearsal_identity = accepted_binding_rehearsal_identity(
            FORMAL_BINDING_ATTEMPT_PATH / "formal-binding.json"
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise Violation(
            "formal launch has no accepted outer-finalized binding rehearsal"
        ) from error
    _require(
        set(record) == expected_fields
        and record.get("schema") == "prospect.wm001.formal-launch.v3"
        and record.get("experiment_id") == "WM-001"
        and record.get("protocol_version") == "1.20.0"
        and record.get("formal_binding_sha256") == binding_sha256
        and _file_sha256(FORMAL_BINDING_ATTEMPT_PATH / "formal-binding.json") == binding_sha256
        and record.get("formal_binding_attempt_path") == str(FORMAL_BINDING_ATTEMPT_PATH)
        and record.get("formal_binding_attempt_manifest_file") == FORMAL_BINDING_ATTEMPT_MANIFEST_NAME
        and record.get("formal_binding_attempt_manifest_sha256")
        == _file_sha256(copied_attempt)
        == _file_sha256(binding_attempt_terminal)
        and record.get("formal_binding_outer_completion_file") == FORMAL_BINDING_OUTER_COMPLETION_NAME
        and record.get("formal_binding_outer_completion_marker") == str(binding_attempt_completion)
        and record.get("formal_binding_outer_completion_sha256")
        == _file_sha256(copied_completion)
        == _file_sha256(binding_attempt_completion)
        and copied_attempt.read_bytes()
        == copied_completion.read_bytes()
        == binding_attempt_terminal.read_bytes()
        == binding_attempt_completion.read_bytes()
        and attempt_manifest.get("kind") == "binding"
        and attempt_manifest.get("lane") is None
        and attempt_manifest.get("status") == "accepted"
        and isinstance(attempt_primary, dict)
        and attempt_primary.get("binding_file") == "formal-binding.json"
        and completion.get("terminal_sha256") == record.get("formal_binding_attempt_manifest_sha256")
        and _strict_json_equal(
            record.get("accepted_binding_rehearsal"),
            rehearsal_identity,
        )
        and path.parent
        == FORMAL_RESULTS_ROOT / binding_sha256 / FORMAL_CONFIRMATION_NAME
        and record.get("attempt_directory") == FORMAL_CONFIRMATION_NAME
        and record.get("global_marker_file") == "formal-launch-v1.20.0.json"
        and record.get("git_commit") == execution.get("git_commit")
        and record.get("git_tree") == execution.get("git_tree")
        and record_sha256 == _canonical_sha256(body),
        "formal launch record identity or self-hash changed",
    )
    expected_payload = (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(payload == expected_payload, "formal launch record is not canonical JSON")


def _verify_result_runtime_binding(
    execution: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    fields = (
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
    _require(
        all(execution.get(field) == runtime.get(field) for field in fields)
        and execution.get("deterministic_algorithms") is True,
        "result runtime differs from binding",
    )


def verify_result(path: Path, binding_path: Path | None) -> dict[str, Any]:
    """Verify result-envelope invariants and causal custody."""

    protocol = verify_protocol()
    result = _load_json(path)
    _validate_json_schema(result, _load_json(RESULT_SCHEMA_PATH), label="raw result")
    _require(result.get("schema") == "prospect.world-model-lifecycle.raw-result.v9", "wrong result schema")
    _require(result.get("experiment_id") == "WM-001", "result has wrong experiment")
    _require(result.get("protocol_version") == "1.20.0", "result has wrong protocol version")
    _require(result.get("protocol_sha256") == _file_sha256(PROTOCOL_PATH), "result protocol digest mismatch")

    lane = result.get("lane")
    _require(lane in {"development", "formal"}, "result lane must be development or formal")
    if lane == "development":
        _require(result.get("claim_eligible") is False, "development result cannot be claim eligible")
    else:
        _require(result.get("claim_eligible") is True, "formal result must identify its claim-eligible lane")
        _require(binding_path is not None, "formal result verification requires --binding")
        assert binding_path is not None
        binding = verify_binding(binding_path)
        _require(result.get("formal_binding_sha256") == _file_sha256(binding_path), "formal binding digest mismatch")
        _require(
            _parse_timestamp(result.get("started_at_utc"), "started_at_utc")
            >= _parse_timestamp(binding.get("sealed_at_utc"), "binding.sealed_at_utc"),
            "formal result started before its implementation binding was sealed",
        )
        execution = result.get("execution", {})
        _require(
            execution.get("git_commit") == binding["source"]["git_commit"],
            "result source commit differs from binding",
        )
        _require(execution.get("git_tree") == binding["source"]["git_tree"], "result source tree differs from binding")
        _require(execution.get("worktree_clean") is True, "formal result did not run from a clean worktree")
        _require(
            execution.get("dependency_lock_sha256") == binding["dependencies"]["lockfile_sha256"],
            "result dependency lock differs from binding",
        )
        runtime = binding.get("runtime", {})
        _require(
            isinstance(runtime, dict),
            "formal binding runtime block is malformed",
        )
        _verify_result_runtime_binding(execution, runtime)
        _require(execution.get("deterministic_algorithms") is True, "formal result was nondeterministic")
        dependencies = binding.get("dependencies", {})
        execution_sources = binding.get("source", {}).get(
            "execution_source_sha256",
            {},
        )
        _require(
            isinstance(dependencies, dict)
            and isinstance(execution_sources, dict)
            and execution.get("runtime_seal_sha256") == _file_sha256(binding_path)
            and execution.get("runtime_seal_descriptor_custody") is True
            and execution.get("producer_bootstrap_sha256") == execution_sources.get("producer_bootstrap.py")
            and execution.get("bootstrap_descriptor_custody") is True
            and execution.get("package_roots") == dependencies.get("package_roots")
            and execution.get("package_ownership") == dependencies.get("package_ownership")
            and execution.get("standard_library") == dependencies.get("standard_library")
            and execution.get("python_executable") == dependencies.get("python_executable")
            and execution.get("python_executable_sha256") == dependencies.get("python_executable_sha256"),
            "result runtime closure differs from the bound pre-import custody",
        )
        launch_path = path.parent / "formal-launch.json"
        _require(
            execution.get("formal_launch_file") == "formal-launch.json"
            and execution.get("formal_launch_sha256") == _file_sha256(launch_path),
            "formal result does not bind its sole-launch record",
        )
        _verify_formal_launch_record(
            launch_path,
            binding_sha256=_file_sha256(binding_path),
            execution=execution,
        )
    if lane == "development":
        execution = result.get("execution", {})
        _require(
            isinstance(execution, dict)
            and execution.get("formal_launch_file") is None
            and execution.get("formal_launch_sha256") is None,
            "development result unexpectedly binds a formal launch",
        )

    started = _parse_timestamp(result.get("started_at_utc"), "started_at_utc")
    completed = _parse_timestamp(result.get("completed_at_utc"), "completed_at_utc")
    _require(completed >= started, "result completed before it started")

    replicates = result.get("replicates")
    _require(isinstance(replicates, list) and replicates, "result has no replicates")
    expected_masters = FORMAL_SEEDS if lane == "formal" else DEVELOPMENT_SEEDS
    if lane == "formal":
        _require(len(replicates) == 8, "formal result must contain exactly 8 replicates")
        _require(
            tuple(item.get("master_seed") for item in replicates) == FORMAL_SEEDS,
            "formal replicate order/seeds changed",
        )
    else:
        _require(all(item.get("master_seed") in expected_masters for item in replicates), "undeclared development seed")

    transition_owner: dict[str, tuple[str, str]] = {}
    for replicate in replicates:
        _verify_replicate(replicate, protocol, transition_owner, lane=lane)

    metrics = result.get("aggregate_metrics")
    _require(isinstance(metrics, list), "aggregate_metrics must be an array")
    for metric in metrics:
        values = metric.get("replicate_values")
        _require(isinstance(values, list), "aggregate metric replicate_values must be an array")
        if lane == "formal":
            _require(len(values) == 8, f"formal aggregate {metric.get('name')} must have 8 values")

    gates = result.get("gate_results")
    _require(isinstance(gates, list) and gates, "result has no gate results")
    gate_ids = tuple(item.get("gate") for item in gates)
    _require(gate_ids == GATES[: len(gate_ids)], "gate results are not a K0-K7 prefix")
    prior_passed = True
    for index, gate in enumerate(gates):
        _require(
            prior_passed,
            "gate results continue after the first failed killing gate",
        )
        checks = gate.get("checks")
        _require(isinstance(checks, list), f"{gate.get('gate')} checks must be an array")
        computed_pass = bool(checks) and all(check.get("passed") is True for check in checks)
        _require(gate.get("passed") is computed_pass, f"{gate.get('gate')} passed value disagrees with checks")
        _require(
            gate.get("claim_supported") is False,
            "producer result claimed support before independent artifact audit",
        )
        if not computed_pass:
            _require(
                index == len(gates) - 1,
                "failed killing gate is not the final reported gate",
            )
        prior_passed = computed_pass
    _require(
        gate_ids[-1] == "K7" or gates[-1].get("passed") is False,
        "passing gate prefix ends before K7",
    )
    if "K7" in gate_ids:
        for replicate in replicates:
            _verify_restart_parity(
                replicate,
                artifact_root=path.parent,
                execution=execution,
            )
    return result


def _row_contract(row: dict[str, Any]) -> tuple[object, object, object, object]:
    return (
        row.get("split"),
        row.get("task_id"),
        row.get("condition"),
        row.get("checkpoint_id"),
    )


def _verify_formal_matrix(
    replicate: dict[str, Any],
    *,
    replicate_id: str,
) -> None:
    """Require the exact sealed v1.20 formal evidence matrix."""

    episodes = replicate["episodes"]
    actual_episode_counts = Counter(_row_contract(row) for row in episodes)
    _require(
        len(episodes) == 496 and actual_episode_counts == Counter(FORMAL_EPISODE_CONTRACT_COUNTS),
        f"{replicate_id}: formal episode matrix differs from the exact 496-episode contract",
    )

    expected_transition_counts: Counter[object] = Counter()
    for contract, episode_count in FORMAL_EPISODE_CONTRACT_COUNTS.items():
        expected_transition_counts[contract[0]] += episode_count * 200
    transitions = replicate["transitions"]
    actual_transition_counts = Counter(row.get("split") for row in transitions)
    _require(
        len(transitions) == 99_200 and actual_transition_counts == expected_transition_counts,
        f"{replicate_id}: formal transition matrix differs from the exact 99,200-transition contract",
    )

    predictive = replicate.get("predictive_metrics")
    _require(
        isinstance(predictive, list)
        and len(predictive) == 12
        and Counter(_row_contract(row) for row in predictive) == Counter(FORMAL_PREDICTIVE_CONTRACT_COUNTS)
        and all(row.get("transition_count") == 1_600 for row in predictive)
        and all(row.get("coverage_target_count") == 6_400 for row in predictive),
        f"{replicate_id}: formal predictive matrix differs from the exact twelve-row contract",
    )

    policy_runs = replicate.get("policy_runs")
    _require(
        isinstance(policy_runs, list)
        and len(policy_runs) == 20
        and Counter(_row_contract(row) for row in policy_runs)
        == Counter({contract: 1 for contract in EPISODE_CONTRACTS}),
        f"{replicate_id}: formal policy-run matrix differs from the exact twenty-run contract",
    )

    updates = replicate["updates"]
    expected_updates = Counter(
        {(phase, "committed", 2_000): 1 for phase in COMMITTED_PHASE_SPLITS}
        | {("rejected_update_probe", "rejected", 0): 1}
    )
    actual_updates = Counter(
        (
            row.get("phase"),
            row.get("status"),
            row.get("optimizer_steps"),
        )
        for row in updates
    )
    _require(
        len(updates) == 6 and actual_updates == expected_updates,
        f"{replicate_id}: formal update matrix differs from five matched updates plus the rejection probe",
    )

    manifests = replicate.get("optimizer_batch_manifests")
    _require(
        isinstance(manifests, list)
        and len(manifests) == 5
        and Counter(row.get("phase") for row in manifests) == Counter({phase: 1 for phase in COMMITTED_PHASE_SPLITS}),
        f"{replicate_id}: formal optimizer-manifest matrix differs",
    )


def _verify_update_eligibility(
    update: dict[str, Any],
    *,
    local_transitions: dict[str, dict[str, Any]],
    replicate_id: str,
) -> None:
    """Bind one update to its exact collection-only canonical input set."""

    expected_phase_splits = {
        **COMMITTED_PHASE_SPLITS,
        "rejected_update_probe": ("collect_a",),
    }
    phase = update.get("phase")
    consumed = update.get("eligible_transition_ids")
    _require(isinstance(consumed, list), f"{replicate_id}: update eligible IDs must be an array")
    _require(update.get("eligible_transition_count") == len(consumed), f"{replicate_id}: eligible count mismatch")
    _require(
        len(consumed) == len(set(consumed)),
        f"{replicate_id}: {phase} repeats an eligible transition ID",
    )
    expected_splits = expected_phase_splits.get(phase)
    _require(
        expected_splits is not None and tuple(update.get("eligible_splits", ())) == expected_splits,
        f"{replicate_id}: {phase} eligible split declaration differs",
    )
    eligible = set(expected_splits or ())
    for transition_id in consumed:
        _require(transition_id in local_transitions, f"{replicate_id}: update consumes unknown {transition_id}")
        _require(
            local_transitions[transition_id].get("split") in eligible,
            f"{replicate_id}: update consumes held-out or phase-ineligible {transition_id}",
        )
    expected_ids = {
        transition_id for transition_id, transition in local_transitions.items() if transition.get("split") in eligible
    }
    if phase == "rejected_update_probe":
        _require(not consumed, f"{replicate_id}: rejected probe consumed real experience")
    else:
        _require(
            set(consumed) == expected_ids,
            f"{replicate_id}: {phase} does not bind the exact eligible collection set",
        )


def _verify_replicate(
    replicate: dict[str, Any],
    protocol: dict[str, Any],
    global_transition_owner: dict[str, tuple[str, str]],
    *,
    lane: str,
) -> None:
    replicate_id = replicate.get("replicate_id")
    master = replicate.get("master_seed")
    _require(isinstance(replicate_id, str) and replicate_id, "replicate_id is missing")
    _require(isinstance(master, int), f"{replicate_id}: master_seed is missing")

    seed_rows = replicate.get("derived_seeds")
    _require(isinstance(seed_rows, list), f"{replicate_id}: derived_seeds must be an array")
    declared_namespaces = protocol["seed_schedule"]["namespaces"]
    _require(
        len(seed_rows) == len(declared_namespaces)
        and all(isinstance(row, dict) for row in seed_rows)
        and {row.get("namespace") for row in seed_rows} == set(declared_namespaces),
        f"{replicate_id}: derived seed namespace matrix is incomplete, duplicated, or extended",
    )
    seeds_by_namespace = {row.get("namespace"): row.get("values") for row in seed_rows if isinstance(row, dict)}
    for namespace, declaration in declared_namespaces.items():
        values = seeds_by_namespace.get(namespace)
        _require(isinstance(values, list), f"{replicate_id}: missing seed namespace {namespace}")
        expected = [derive_seed(namespace, master, index) for index in range(declaration["count"])]
        _require(values == expected, f"{replicate_id}: derived seeds changed for {namespace}")

    episodes = replicate.get("episodes")
    transitions = replicate.get("transitions")
    updates = replicate.get("updates")
    _require(isinstance(episodes, list), f"{replicate_id}: episodes must be an array")
    _require(isinstance(transitions, list), f"{replicate_id}: transitions must be an array")
    _require(isinstance(updates, list), f"{replicate_id}: updates must be an array")
    if lane == "formal":
        _verify_formal_matrix(replicate, replicate_id=replicate_id)

    local_transitions: dict[str, dict[str, Any]] = {}
    seed_split: dict[tuple[str, int], str] = {}
    for transition in transitions:
        transition_id = transition.get("transition_id")
        _require(isinstance(transition_id, str) and transition_id, f"{replicate_id}: transition ID missing")
        _require(transition_id not in global_transition_owner, f"duplicate real transition ID: {transition_id}")
        _require(
            transition.get("real_or_imagined") == "real",
            f"imagined transition in real namespace: {transition_id}",
        )
        _require(
            isinstance(transition.get("run_id"), str) and transition["run_id"],
            f"{transition_id}: run ID missing",
        )
        expected_context = {
            TASK_A: 0.0,
            TASK_B: 1.0,
            TASK_IRRELEVANT: 2.0,
        }.get(transition.get("task_id"))
        _require(
            expected_context is not None and transition.get("task_context") == expected_context,
            f"{transition_id}: invalid task identity or context",
        )
        scaled_target = transition.get("scaled_target")
        _require(
            isinstance(scaled_target, list)
            and len(scaled_target) == 4
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in scaled_target),
            f"{transition_id}: scaled target is malformed",
        )
        expected_target_sha256 = sha256(struct.pack("<4d", *(float(value) for value in scaled_target))).hexdigest()
        _require(
            transition.get("target_sha256") == expected_target_sha256,
            f"{transition_id}: target digest differs from scaled target",
        )
        pre_observation = transition.get("pre_observation")
        next_observation = transition.get("next_observation")
        _require(
            isinstance(pre_observation, list)
            and len(pre_observation) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in pre_observation),
            f"{transition_id}: pre-observation is malformed",
        )
        _require(
            isinstance(next_observation, list)
            and len(next_observation) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in next_observation),
            f"{transition_id}: next observation is malformed",
        )
        reward = transition.get("reward")
        _require(
            isinstance(reward, (int, float)) and math.isfinite(float(reward)),
            f"{transition_id}: reward is malformed",
        )
        expected_scaled = (
            (float(next_observation[0]) - float(pre_observation[0])) / 2.0,
            (float(next_observation[1]) - float(pre_observation[1])) / 2.0,
            (float(next_observation[2]) - float(pre_observation[2])) / 16.0,
            float(reward) / 16.2736044,
        )
        _require(
            all(
                math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
                for actual, expected in zip(scaled_target, expected_scaled, strict=True)
            ),
            f"{transition_id}: scaled target differs from raw transition values",
        )
        intended = float(transition.get("intended_action"))
        applied = float(transition.get("applied_action"))
        task_id = transition.get("task_id")
        expected_applied = -intended if task_id == TASK_B else 0.0 if task_id == TASK_IRRELEVANT else intended
        _require(
            math.isclose(applied, expected_applied, rel_tol=0.0, abs_tol=1e-12),
            f"{transition_id}: applied action differs from task semantics",
        )
        global_transition_owner[transition_id] = (replicate_id, transition.get("split"))
        local_transitions[transition_id] = transition

    episode_ids: set[str] = set()
    referenced_transition_ids: Counter[str] = Counter()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        _require(isinstance(episode_id, str) and episode_id, f"{replicate_id}: episode ID missing")
        _require(episode_id not in episode_ids, f"{replicate_id}: duplicate episode ID {episode_id}")
        episode_ids.add(episode_id)
        _require(episode.get("environment_steps") == 200, f"{episode_id}: incomplete episode")
        run_id = episode.get("run_id")
        _require(isinstance(run_id, str) and run_id, f"{episode_id}: run ID missing")
        split = episode.get("split")
        task = episode.get("task_id")
        contract = (
            split,
            task,
            episode.get("condition"),
            episode.get("checkpoint_id"),
        )
        _require(
            contract in EPISODE_CONTRACTS,
            f"{episode_id}: split/task/condition/checkpoint is outside the sealed matrix",
        )
        reset_seed = episode.get("reset_seed")
        _require(isinstance(reset_seed, int), f"{episode_id}: reset seed missing")
        assignment_key = (task, reset_seed)
        previous_split = seed_split.setdefault(assignment_key, split)
        _require(previous_split == split, f"{replicate_id}: reset seed crosses episode splits")
        if str(split).startswith(("predictive_validation_", "behavior_evaluation_")):
            _require(episode.get("learning_allowed") is False, f"{episode_id}: held-out learning was allowed")
            _require(episode.get("replay_writes_allowed") is False, f"{episode_id}: held-out replay write was allowed")
        transition_ids = episode.get("transition_ids")
        _require(isinstance(transition_ids, list) and len(transition_ids) == 200, f"{episode_id}: bad transition count")
        intended_actions: list[float] = []
        applied_actions: list[float] = []
        rewards: list[float] = []
        previous_next: list[float] | None = None
        for expected_step, transition_id in enumerate(transition_ids):
            referenced_transition_ids[str(transition_id)] += 1
            _require(transition_id in local_transitions, f"{episode_id}: unresolved transition {transition_id}")
            transition = local_transitions[transition_id]
            _require(transition.get("episode_id") == episode_id, f"{transition_id}: episode linkage mismatch")
            _require(transition.get("run_id") == run_id, f"{transition_id}: run linkage mismatch")
            _require(transition.get("split") == split, f"{transition_id}: split linkage mismatch")
            _require(transition.get("task_id") == task, f"{transition_id}: task linkage mismatch")
            _require(
                transition.get("model_version_at_action") == episode.get("model_version")
                and transition.get("parameter_sha256_at_action") == episode.get("parameter_sha256"),
                f"{transition_id}: action-time model lineage differs from its episode",
            )
            _require(
                transition.get("step_index") == expected_step,
                f"{transition_id}: step index differs from episode order",
            )
            _require(
                transition.get("terminated") is False,
                f"{transition_id}: Pendulum must not terminate",
            )
            _require(
                transition.get("truncated") is (expected_step == 199),
                f"{transition_id}: TimeLimit truncation flag differs",
            )
            current_pre = transition["pre_observation"]
            if previous_next is not None:
                _require(
                    all(
                        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
                        for left, right in zip(previous_next, current_pre, strict=True)
                    ),
                    f"{transition_id}: physical observation chain is discontinuous",
                )
            previous_next = transition["next_observation"]
            intended_actions.append(float(transition["intended_action"]))
            applied_actions.append(float(transition["applied_action"]))
            rewards.append(float(transition["reward"]))
        _require(
            math.isclose(
                float(episode.get("return")),
                math.fsum(rewards),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            f"{episode_id}: return differs from raw rewards",
        )
        _require(
            episode.get("action_trace_sha256")
            == _canonical_sha256(
                {
                    "intended": intended_actions,
                    "applied": applied_actions,
                }
            ),
            f"{episode_id}: action trace digest differs from raw transitions",
        )

    _require(
        set(referenced_transition_ids) == set(local_transitions)
        and all(count == 1 for count in referenced_transition_ids.values()),
        f"{replicate_id}: every real transition must be referenced by exactly one episode",
    )

    split_reset_namespaces = {
        "collect_a": "collect_a_episode",
        "collect_b": "collect_b_episode",
        "collect_irrelevant": "collect_irrelevant_episode",
        "predictive_validation_a": "predictive_validation_a_episode",
        "predictive_validation_b": "predictive_validation_b_episode",
        "predictive_validation_irrelevant": "predictive_validation_irrelevant_episode",
    }
    for split, namespace in split_reset_namespaces.items():
        grouped = [episode for episode in episodes if episode.get("split") == split]
        actual = [episode.get("reset_seed") for episode in grouped]
        expected = [derive_seed(namespace, master, index) for index in range(len(grouped))]
        _require(
            actual == expected,
            f"{replicate_id}: {split} reset seeds differ from {namespace}",
        )
    behavior_reset_namespaces = {
        "behavior_evaluation_a": "behavior_evaluation_a_episode",
        "behavior_evaluation_b": "behavior_evaluation_b_episode",
    }
    for split, namespace in behavior_reset_namespaces.items():
        conditions = {str(episode.get("condition")) for episode in episodes if episode.get("split") == split}
        for condition in sorted(conditions):
            grouped = [
                episode
                for episode in episodes
                if episode.get("split") == split and episode.get("condition") == condition
            ]
            actual = [episode.get("reset_seed") for episode in grouped]
            expected = [derive_seed(namespace, master, index) for index in range(len(grouped))]
            _require(
                actual == expected,
                f"{replicate_id}: {split}/{condition} reset seeds differ from {namespace}",
            )

    policy_runs = replicate.get("policy_runs")
    _require(isinstance(policy_runs, list), f"{replicate_id}: policy_runs must be an array")
    runs_by_id: dict[str, list[dict[str, Any]]] = {}
    for run in policy_runs:
        run_id = run.get("run_id")
        _require(isinstance(run_id, str) and run_id, f"{replicate_id}: policy run ID missing")
        runs_by_id.setdefault(run_id, []).append(run)
    episode_runs = {str(episode.get("run_id")) for episode in episodes}
    _require(set(runs_by_id) == episode_runs, f"{replicate_id}: policy-run set differs from real executions")
    for run_id, run_rows in runs_by_id.items():
        _require(len(run_rows) == 1, f"{replicate_id}: duplicate policy run {run_id}")
        run = run_rows[0]
        grouped = [episode for episode in episodes if episode.get("run_id") == run_id]
        ordered_ids = [str(episode["episode_id"]) for episode in grouped]
        reset_seeds = [int(episode["reset_seed"]) for episode in grouped]
        _require(run.get("episode_ids") == ordered_ids, f"{run_id}: episode order differs")
        _require(run.get("reset_seeds") == reset_seeds, f"{run_id}: reset seeds differ")
        _require(run.get("action_count") == 200 * len(grouped), f"{run_id}: action count differs")
        for field in ("task_id", "split", "condition", "checkpoint_id"):
            values = {episode.get(field) for episode in grouped}
            _require(len(values) == 1 and run.get(field) in values, f"{run_id}: {field} differs")
        namespace = run.get("seed_namespace")
        seed_index = run.get("seed_index")
        _require(
            isinstance(namespace, str)
            and namespace in protocol["seed_schedule"]["namespaces"]
            and isinstance(seed_index, int)
            and 0 <= seed_index < int(protocol["seed_schedule"]["namespaces"][namespace]["count"]),
            f"{run_id}: seed reference is invalid",
        )
        _require(
            run.get("seed") == derive_seed(namespace, master, seed_index),
            f"{run_id}: seed differs from declared namespace",
        )
        task_id = run.get("task_id")
        task_index = 0 if task_id == TASK_A else 1
        split = str(run.get("split"))
        condition = str(run.get("condition"))
        controller = run.get("controller_kind")
        expected_controller = (
            "uniform_random"
            if condition in {"collection_random", "validation_random", "random"}
            else "cem_oracle"
            if condition == "oracle"
            else "cem_learned"
        )
        _require(
            (
                run.get("split"),
                run.get("task_id"),
                condition,
                run.get("checkpoint_id"),
            )
            in EPISODE_CONTRACTS
            and controller == expected_controller,
            f"{run_id}: execution/controller contract is outside the sealed matrix",
        )
        expected_seed_ref = (
            ("irrelevant_collection_action", 0)
            if split == "collect_irrelevant"
            else ("predictive_validation_irrelevant_action", 0)
            if split == "predictive_validation_irrelevant"
            else ("collection_action", task_index)
            if split in {"collect_a", "collect_b"}
            else ("predictive_validation_action", task_index)
            if split in {"predictive_validation_a", "predictive_validation_b"}
            else ("random_policy_action", task_index)
            if condition == "random"
            else ("planner", 0)
        )
        _require((namespace, seed_index) == expected_seed_ref, f"{run_id}: wrong policy seed namespace")
        intended_actions = [
            float(local_transitions[transition_id]["intended_action"])
            for episode in grouped
            for transition_id in episode["transition_ids"]
        ]
        applied_actions = [
            float(local_transitions[transition_id]["applied_action"])
            for episode in grouped
            for transition_id in episode["transition_ids"]
        ]
        _require(
            run.get("action_trace_sha256")
            == _canonical_sha256(
                {
                    "episode_ids": ordered_ids,
                    "intended": intended_actions,
                    "applied": applied_actions,
                }
            ),
            f"{run_id}: run action trace differs",
        )

    snapshots = replicate.get("evaluated_checkpoints")
    _require(isinstance(snapshots, list), f"{replicate_id}: evaluated_checkpoints must be an array")
    snapshots_by_condition = {
        snapshot.get("condition"): snapshot for snapshot in snapshots if isinstance(snapshot, dict)
    }
    _require(
        len(snapshots) == 7
        and set(snapshots_by_condition)
        == {
            "cold",
            "frozen",
            "corrupted",
            "irrelevant",
            "after_a",
            "after_b_replay",
            "after_b_naive",
        },
        f"{replicate_id}: evaluated checkpoint set differs",
    )
    for condition, snapshot in snapshots_by_condition.items():
        _require(
            snapshot.get("sha256") == snapshot.get("live_state_sha256"),
            f"{replicate_id}: {condition} model artifact does not bind live state",
        )
    for run in policy_runs:
        controller_kind = run.get("controller_kind")
        condition = str(run.get("condition"))
        if controller_kind == "cem_learned":
            snapshot = snapshots_by_condition.get(condition)
            _require(
                isinstance(snapshot, dict)
                and run.get("controller_version") == f"wm001-sha256:{snapshot.get('parameter_sha256')}",
                f"{replicate_id}: learned CEM {condition} is not bound to its evaluated parameter snapshot",
            )
        elif controller_kind == "cem_oracle":
            _require(
                run.get("controller_version") == "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1",
                f"{replicate_id}: oracle CEM controller version differs",
            )
    predictive_metrics = replicate.get("predictive_metrics")
    _require(isinstance(predictive_metrics, list), f"{replicate_id}: predictive_metrics must be an array")
    for metric in predictive_metrics:
        _require(
            (
                metric.get("split"),
                metric.get("task_id"),
                metric.get("condition"),
                metric.get("checkpoint_id"),
            )
            in PREDICTIVE_CONTRACTS,
            f"{replicate_id}: predictive row is outside the sealed matrix",
        )
        snapshot = snapshots_by_condition.get(metric.get("condition"))
        _require(isinstance(snapshot, dict), f"{replicate_id}: predictive model snapshot missing")
        transition_count = metric.get("transition_count")
        covered_target_count = metric.get("interval_90_covered_target_count")
        coverage_target_count = metric.get("coverage_target_count")
        coverage = metric.get("interval_90_coverage")
        _require(
            metric.get("coverage_semantics") == COVERAGE_SEMANTICS,
            f"{replicate_id}: predictive coverage semantics differ from v1.20",
        )
        _require(
            isinstance(transition_count, int)
            and not isinstance(transition_count, bool)
            and isinstance(covered_target_count, int)
            and not isinstance(covered_target_count, bool)
            and isinstance(coverage_target_count, int)
            and not isinstance(coverage_target_count, bool)
            and coverage_target_count == 4 * transition_count
            and 0 <= covered_target_count <= coverage_target_count,
            f"{replicate_id}: predictive coverage counts are invalid",
        )
        _require(
            isinstance(coverage, (int, float))
            and not isinstance(coverage, bool)
            and math.isfinite(float(coverage))
            and float(coverage) == covered_target_count / coverage_target_count,
            f"{replicate_id}: predictive coverage fraction differs from exact counts",
        )
        for field in ("model_version", "parameter_sha256", "live_state_sha256"):
            _require(
                metric.get(field) == snapshot.get(field),
                f"{replicate_id}: predictive {metric.get('condition')} {field} differs from artifact",
            )

    for update in updates:
        _verify_update_eligibility(
            update,
            local_transitions=local_transitions,
            replicate_id=replicate_id,
        )
        steps = update.get("optimizer_steps")
        expected_samples = (
            0 if update.get("status") == "rejected" else int(steps) * 5 * 256 if isinstance(steps, int) else -1
        )
        _require(
            update.get("consumed_sample_count") == expected_samples,
            f"{replicate_id}: consumed sample count differs from optimizer budget",
        )

    updates_by_phase = {update.get("phase"): update for update in updates if isinstance(update, dict)}
    _require(
        len(updates_by_phase) == len(updates)
        and set(updates_by_phase) == {*COMMITTED_PHASE_SPLITS, "rejected_update_probe"},
        f"{replicate_id}: update phase matrix is incomplete, duplicated, or extended",
    )
    train_a = updates_by_phase.get("train_a")
    corrupted = updates_by_phase.get("train_a_corrupted")
    irrelevant = updates_by_phase.get("train_a_irrelevant")
    replay = updates_by_phase.get("train_b_replay")
    naive = updates_by_phase.get("train_b_naive")
    rejected_probe = updates_by_phase.get("rejected_update_probe")
    _require(
        all(
            isinstance(update, dict)
            for update in (
                train_a,
                corrupted,
                irrelevant,
                replay,
                naive,
                rejected_probe,
            )
        ),
        f"{replicate_id}: update and rejected-probe set is incomplete",
    )
    assert isinstance(train_a, dict)
    assert isinstance(corrupted, dict)
    assert isinstance(irrelevant, dict)
    assert isinstance(replay, dict)
    assert isinstance(naive, dict)
    assert isinstance(rejected_probe, dict)
    for phase, update in updates_by_phase.items():
        expected_status = "rejected" if phase == "rejected_update_probe" else "committed"
        _require(
            update.get("status") == expected_status,
            f"{replicate_id}: {phase} has the wrong transaction status",
        )
    _require(
        rejected_probe.get("status") == "rejected"
        and rejected_probe.get("full_state_before_sha256") == rejected_probe.get("full_state_after_sha256"),
        f"{replicate_id}: rejected probe did not preserve full-state bytes",
    )
    for label in ("before", "after"):
        digest = rejected_probe.get(f"full_state_{label}_sha256")
        reference = rejected_probe.get(f"full_state_{label}_file")
        _require(
            isinstance(reference, dict)
            and reference.get("sha256") == digest
            and reference.get("media_type") == "application/vnd.prospect.wm001.rejected-probe-state+json",
            f"{replicate_id}: rejected probe {label} full-state reference differs",
        )
    for field in (
        "predecessor_parameter_sha256",
        "predecessor_model_version",
        "live_state_before_sha256",
    ):
        _require(
            corrupted.get(field) == train_a.get(field),
            f"{replicate_id}: corrupted control cold {field} ancestry differs",
        )
        _require(
            irrelevant.get(field) == train_a.get(field),
            f"{replicate_id}: irrelevant control cold {field} ancestry differs",
        )
    for label, update in (("train_b_replay", replay), ("train_b_naive", naive)):
        for previous, resulting in (
            ("predecessor_parameter_sha256", "committed_parameter_sha256"),
            ("predecessor_model_version", "committed_model_version"),
            ("live_state_before_sha256", "live_state_after_sha256"),
        ):
            _require(
                update.get(previous) == train_a.get(resulting),
                f"{replicate_id}: {label} post-A {previous} ancestry differs",
            )
    snapshot_ancestry = {
        "cold": train_a.get("live_state_before_sha256"),
        "frozen": train_a.get("live_state_before_sha256"),
        "corrupted": corrupted.get("live_state_after_sha256"),
        "irrelevant": irrelevant.get("live_state_after_sha256"),
        "after_a": train_a.get("live_state_after_sha256"),
        "after_b_replay": replay.get("live_state_after_sha256"),
        "after_b_naive": naive.get("live_state_after_sha256"),
    }
    for condition, expected_digest in snapshot_ancestry.items():
        _require(
            snapshots_by_condition[condition].get("live_state_sha256") == expected_digest,
            f"{replicate_id}: {condition} compound snapshot ancestry differs",
        )

    manifests = replicate.get("optimizer_batch_manifests")
    _require(isinstance(manifests, list), f"{replicate_id}: optimizer manifests must be an array")
    manifests_by_phase = {manifest.get("phase"): manifest for manifest in manifests if isinstance(manifest, dict)}
    _require(
        len(manifests) == len(COMMITTED_PHASE_SPLITS)
        and len(manifests_by_phase) == len(manifests)
        and set(manifests_by_phase) == set(COMMITTED_PHASE_SPLITS),
        f"{replicate_id}: optimizer manifest matrix is incomplete, duplicated, or extended",
    )
    for phase, manifest in manifests_by_phase.items():
        _require(
            manifest.get("sha256") == updates_by_phase[phase].get("sampling_manifest_sha256"),
            f"{replicate_id}: {phase} optimizer manifest differs from its update receipt",
        )


def _verify_restart_parity(
    replicate: dict[str, Any],
    *,
    artifact_root: Path,
    execution: Mapping[str, Any],
) -> None:
    replicate_id = replicate.get("replicate_id")
    parity = replicate.get("restart_parity")
    _require(isinstance(parity, dict), f"{replicate_id}: restart parity is missing")
    _require(parity.get("fresh_process") is True, f"{replicate_id}: restore was not fresh-process")
    _require(
        parity.get("original_process_id") != parity.get("restored_process_id"),
        f"{replicate_id}: restore reused the original process",
    )
    runtime = parity.get("restore_runtime")
    _require(
        isinstance(runtime, dict),
        f"{replicate_id}: restore runtime identity is missing",
    )
    expected_filename = f"{replicate_id}-restore-runtime.json"
    _require(
        runtime.get("filename") == expected_filename,
        f"{replicate_id}: restore runtime filename is not exact",
    )
    runtime_path = artifact_root / expected_filename
    _require(
        runtime_path.is_file() and not runtime_path.is_symlink(),
        f"{replicate_id}: restore runtime file is missing or aliased",
    )
    payload = runtime_path.read_bytes()
    _require(
        runtime.get("bytes") == len(payload) and runtime.get("sha256") == sha256(payload).hexdigest(),
        f"{replicate_id}: restore runtime file identity changed",
    )
    recorded = dict(runtime)
    for field in ("bytes", "sha256", "filename"):
        recorded.pop(field, None)
    _require(
        payload
        == json.dumps(
            recorded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
        f"{replicate_id}: restore runtime file is not canonical",
    )
    _require(
        recorded.get("schema") == "prospect.wm001.restart-runtime.v2"
        and recorded.get("python_executable") == execution.get("python_executable")
        and recorded.get("python_executable_sha256") == execution.get("python_executable_sha256")
        and recorded.get("python_version") == execution.get("python_version")
        and recorded.get("python_flags")
        == {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
            "safe_path": True,
        }
        and recorded.get("process_environment") == execution.get("process_environment")
        and recorded.get("deterministic_algorithms") is True
        and recorded.get("bootstrap_source_sha256") == execution.get("producer_bootstrap_sha256")
        and recorded.get("bootstrap_descriptor_custody") is True
        and recorded.get("runtime_seal_sha256") == execution.get("runtime_seal_sha256")
        and recorded.get("runtime_seal_descriptor_custody") is True
        and recorded.get("package_root_inventory")
        == (
            execution.get("package_roots", [None])[0]
            if isinstance(execution.get("package_roots"), list) and len(execution["package_roots"]) == 1
            else None
        )
        and recorded.get("standard_library") == execution.get("standard_library")
        and recorded.get("package_ownership") == execution.get("package_ownership")
        and isinstance(recorded.get("package_root"), str)
        and Path(recorded["package_root"]).is_absolute()
        and recorded.get("package_root")
        == (
            recorded.get("package_root_inventory", {}).get("path")
            if isinstance(recorded.get("package_root_inventory"), dict)
            else None
        ),
        f"{replicate_id}: restore child differs from the sealed -I -S -B closure",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol", help="verify the sealed scientific protocol")
    binding = subparsers.add_parser("binding", help="verify a pre-run formal implementation binding")
    binding.add_argument("path", type=Path)
    result = subparsers.add_parser("result", help="verify a development or formal raw result")
    result.add_argument("path", type=Path)
    result.add_argument("--binding", type=Path, help="required for a formal result")
    seed = subparsers.add_parser("seed", help="print one declared derived seed")
    seed.add_argument("namespace")
    seed.add_argument("master_seed", type=int)
    seed.add_argument("index", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested verifier."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "protocol":
            verify_protocol()
            print(f"WM-001 protocol valid: {_file_sha256(PROTOCOL_PATH)}")
        elif arguments.command == "binding":
            verify_binding(arguments.path)
            print(f"WM-001 formal binding valid: {_file_sha256(arguments.path)}")
        elif arguments.command == "result":
            verify_result(arguments.path, arguments.binding)
            print(f"WM-001 result envelope valid: {_file_sha256(arguments.path)}")
        else:
            print(derive_seed(arguments.namespace, arguments.master_seed, arguments.index))
    except Violation as exc:
        print(f"WM-001 verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
