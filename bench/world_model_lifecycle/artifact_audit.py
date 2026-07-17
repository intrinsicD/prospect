"""Independent artifact-root recomputation for WM-001.

This module intentionally does not call the experiment's metric, manifest, or
checkpoint decoders.  It treats ``result.json`` and every referenced sidecar as
untrusted input, applies explicit byte limits, decodes the safe formats again,
and recomputes the claims that the current artifact actually makes auditable.

The audit distinguishes two outcomes:

``integrity_passed``
    Every check possible from the current artifact passed.

``complete_for_claim``
    The producer supplied enough evidence to independently bind predictions,
    targets, controller actions, and formal execution to their claimed causes.

``passed`` is true only when both are true.  This prevents a byte-consistent but
causally incomplete artifact from being mistaken for independent evidence of
learning and retention.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib
import json
import math
import statistics
import struct
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
RESULT_SCHEMA_PATH = HERE / "schemas" / "raw-result.schema.json"

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
_CEM_PLANNING_HORIZON = 10
_CEM_OPTIM_STEPS = 3
_CEM_NUM_CANDIDATES = 64
_CEM_TOP_K = 8
# Protocol 1.1 is an evidence-only repair.  Its experimental random streams
# intentionally retain the v1.0 seed-domain tag so no task outcome changes.
_SEED_HASH_DOMAIN_VERSION = "1.0.0"
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

# A formal result retains roughly 700k transition rows inline.  Four GiB is a
# hard safety ceiling, not a target; moving transition rows to independently
# hashed columnar sidecars is the natural follow-up if WM-001 grows.
_MAX_RESULT_BYTES = 4 << 30
_MAX_PREDICTION_BYTES = 8 << 20
_MAX_OWNED_MODEL_BYTES = 256 << 20
_MAX_MANIFEST_BYTES = 64 << 20
_MAX_PERMUTATION_BYTES = 64 << 20
_MAX_REJECTED_STATE_BYTES = 512 << 20
_MAX_CHECKPOINT_BYTES = 4 << 30
_MAX_CONTAINER_HEADER_BYTES = 1 << 20
_MAX_CHECKPOINT_MANIFEST_BYTES = 1 << 20
_MAX_CHECKPOINT_COMPONENT_BYTES = 1 << 30
_MAX_CHECKPOINT_TOTAL_BYTES = 4 << 30
_MAX_SOURCE_FILE_BYTES = 64 << 20
_MAX_SOURCE_SNAPSHOT_BYTES = 512 << 20
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
_TASK_CONTEXT = {_TASK_A: 0.0, _TASK_B: 1.0}
_EXPECTED_PHASE_SPLITS = {
    "train_a": ("collect_a",),
    "train_a_corrupted": ("collect_a",),
    "train_b_replay": ("collect_a", "collect_b"),
    "train_b_naive": ("collect_b",),
}
_EXPECTED_SEED_COUNTS = {
    "model_initialization": 1,
    "torch_runtime": 1,
    "collection_action": 2,
    "predictive_validation_action": 2,
    "random_policy_action": 2,
    "ensemble_bootstrap_a": 5,
    "ensemble_bootstrap_b": 5,
    "minibatch_order_a": 1,
    "minibatch_order_b": 1,
    "corrupted_target_permutation": 1,
    "collect_a_episode": 8,
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

    z_score = (target64[None, :, :] - means64) * np.exp(-0.5 * log_variances64)
    erf_values = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in z_score.flat),
        dtype=np.float64,
        count=z_score.size,
    ).reshape(z_score.shape)
    mixture_pit = np.mean(0.5 * (1.0 + erf_values), axis=0)
    coverage = float(np.mean((mixture_pit >= 0.05) & (mixture_pit <= 0.95)))
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
    means = np.empty((5, len(inputs), 4), dtype=np.float32)
    log_variances = np.empty_like(means)
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
        "after_a",
        "after_b_replay",
        "after_b_naive",
    }
    audit.require(
        set(result) == expected_conditions,
        code="evaluated_checkpoint_set_mismatch",
        message=f"{replicate_id} does not retain exactly the six evaluated learned states",
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


def _audit_predictions(
    audit: _Audit,
    root: Path,
    replicate: Mapping[str, object],
    *,
    replicate_id: str,
    transitions_by_id: Mapping[str, Mapping[str, object]],
    evaluated_checkpoints: Mapping[str, _EvaluatedCheckpoint],
) -> None:
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
            expected_targets: list[list[float]] = []
            target_rows_complete = True
            for transition_id in recomputed.transition_ids:
                transition = transitions_by_id.get(transition_id)
                raw_target = None if transition is None else transition.get("scaled_target")
                if (
                    not isinstance(raw_target, list)
                    or len(raw_target) != 4
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in raw_target
                    )
                ):
                    target_rows_complete = False
                    break
                expected_targets.append([float(value) for value in raw_target])
            targets_match = target_rows_complete and np.allclose(
                recomputed.normalized_targets,
                np.asarray(expected_targets, dtype=np.float32),
                rtol=2e-6,
                atol=2e-7,
            )
            audit.require(
                targets_match,
                code="prediction_target_binding_mismatch",
                message=f"{label} normalized targets do not match raw transition targets",
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
            coverage_tolerance = 1.0 / max(1, 4 * len(recomputed.transition_ids))
            audit.require(
                _close_enough(
                    recomputed.interval_90_coverage,
                    row.get("interval_90_coverage"),
                    absolute=coverage_tolerance,
                ),
                code="prediction_coverage_mismatch",
                message=f"{label} stored coverage differs from independent recomputation",
                replicate_id=replicate_id,
                evidence={
                    "recomputed": recomputed.interval_90_coverage,
                    "stored": row.get("interval_90_coverage"),
                },
            )
        except (ArtifactAuditError, OSError, ValueError) as error:
            audit.error("prediction_evidence_invalid", f"{label}: {error}", replicate_id=replicate_id)


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

    if phase in {"train_a", "train_a_corrupted"}:
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

    indices = np.empty((optimizer_steps, 5, 256), dtype="<u4")
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
    audit.limitation(
        "Bootstrap, balanced-minibatch, minibatch-order, and corruption-permutation "
        "bytes are regenerated without producer sampling code, but exact replay uses "
        "the same dependency-bound Torch CPU RNG primitives rather than a separately "
        "implemented Philox engine."
    )


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _derive_seed(namespace: str, master_seed: int, index: int = 0) -> int:
    """Reconstruct one sealed seed without trusting producer seed rows."""

    return int.from_bytes(
        hashlib.sha256((f"WM-001|{_SEED_HASH_DOMAIN_VERSION}|{namespace}|{master_seed}|{index}").encode()).digest()[:4],
        "big",
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
    """Check one raw row against independently implemented Pendulum dynamics."""

    before = row.get("pre_observation")
    after = row.get("next_observation")
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
        expected_observation, expected_reward, expected_applied = _independent_pendulum_step(
            before,
            context=context,
            intended_action=intended,
        )
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
        observation_error <= _PENDULUM_OBSERVATION_ATOL,
        code="transition_dynamics_observation_mismatch",
        message=(
            f"{replicate_id} transition {transition_id} next observation is not "
            "consistent with analytic Pendulum-v1 dynamics"
        ),
        replicate_id=replicate_id,
        evidence={
            "max_absolute_error": observation_error,
            "absolute_tolerance": _PENDULUM_OBSERVATION_ATOL,
        },
    )
    audit.require(
        reward_error <= _PENDULUM_REWARD_ATOL,
        code="transition_dynamics_reward_mismatch",
        message=(
            f"{replicate_id} transition {transition_id} reward is not consistent with analytic Pendulum-v1 dynamics"
        ),
        replicate_id=replicate_id,
        evidence={
            "absolute_error": reward_error,
            "absolute_tolerance": _PENDULUM_REWARD_ATOL,
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
            expected_reset = _independent_pendulum_reset(reset_seed)  # type: ignore[arg-type]
            reset_matches = (
                isinstance(first_observation, list)
                and len(first_observation) == 3
                and np.array_equal(
                    np.asarray(first_observation, dtype=np.float32),
                    expected_reset,
                )
            )
        except (ArtifactAuditError, TypeError, ValueError):
            reset_matches = False
        audit.require(
            reset_matches,
            code="episode_reset_observation_mismatch",
            message=(
                f"{replicate_id} episode {episode_id} first pre-observation "
                "does not exactly replay from its Gymnasium reset seed"
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
        else:
            action_valid = False
        audit.require(
            action_valid,
            code="transition_applied_action_mismatch",
            message=f"{replicate_id} transition {transition_id} applied torque violates task semantics",
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
            reward_target_valid = reward is not None and math.isclose(
                float(raw_target[3]),
                reward / _TARGET_REWARD_SCALE,
                rel_tol=1e-12,
                abs_tol=1e-12,
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
                source_target_valid = bool(
                    np.allclose(
                        target_array,
                        independently_scaled,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )
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
    audit.limitation(
        "Episode reset observations are reconstructed from the documented "
        "Gymnasium 0.29.1 default_rng/PCG64 Pendulum reset algorithm without "
        "importing Gymnasium; exact replay consequently relies on NumPy's bound "
        "Generator semantics."
    )
    return transitions_by_id


def _expected_policy_contract(
    *,
    task_id: object,
    split: object,
    condition: object,
) -> tuple[tuple[str, int], str, Mapping[str, int] | None] | None:
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
    ``None`` model selects the analytic oracle; otherwise the six retained
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
    previous_cuda_tf32 = bool(torch.backends.cuda.matmul.allow_tf32) if destination.type == "cuda" else False
    previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32) if destination.type == "cuda" else False
    try:
        torch.use_deterministic_algorithms(True)
        if destination.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
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
            torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
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
                "All 17 sealed namespace rows, in full declared counts, with each value "
                "regenerated from the v1.0 experimental seed hash domain."
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
                expected_actions = np.asarray(
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
                    np.array_equal(replayed_actions, expected_actions),
                    code="cem_policy_action_replay_mismatch",
                    message=(
                        f"{label} complete action trace does not exactly replay "
                        "from observed states, retained checkpoint, sealed CEM "
                        "budget, and planner seed"
                    ),
                    replicate_id=replicate_id,
                )
                cem_replay_count += 1
            except (ArtifactAuditError, KeyError, TypeError, ValueError) as error:
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
        audit.limitation(
            "CEM is independently reimplemented from the public algorithm and "
            "retained tensor bytes rather than calling producer planning/model code, "
            "but bit-exact action replay intentionally uses the same bound Torch "
            "device kernels and RNG implementation as the producer."
        )
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
) -> None:
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
            domain_components_verified: set[str] = set()
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
                        _validate_checkpoint_domain_component(
                            archive.read(info),
                            component_id=component_id,
                        )
                        domain_components_verified.add(component_id)
                expected_paths.add(str(member_path))
                archive_components[component_id] = entry
            if set(names) != expected_paths:
                raise ArtifactAuditError("checkpoint ZIP member set differs from its manifest")
            if set(archive_components) != set(_CANONICAL_COMPONENT_IDS):
                raise ArtifactAuditError("checkpoint does not contain all and only 15 canonical components")
            if domain_components_verified != {
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


def _verify_finalized_custody(audit: _Audit, root: Path) -> None:
    try:
        artifact_module = importlib.import_module("bench.world_model_lifecycle.artifact")
        manifest_raw: object = artifact_module.verify_producer_manifest(root)
        if not isinstance(manifest_raw, dict):
            raise ArtifactAuditError("producer manifest verifier returned a non-object")
        manifest: Mapping[str, object] = manifest_raw
        manifest_path = root / "producer-manifest.json"
        audit.custody = {
            "producer_manifest_checked": True,
            "producer_manifest_status": manifest.get("status"),
            "producer_manifest_sha256": hashlib.sha256(
                _read_bounded(
                    manifest_path,
                    64 << 20,
                    label="producer manifest",
                )
            ).hexdigest(),
        }
        audit.require(
            manifest.get("status") == "completed",
            code="producer_attempt_not_completed",
            message="producer manifest does not identify a completed attempt",
        )
        audit.independence_limitations.append(
            "The producer manifest is fully reopened and hash-checked, but it is not "
            "externally signed or transparency-log anchored; filesystem-level replacement "
            "of the entire root and manifest is outside this audit's threat model."
        )
    except (ArtifactAuditError, ImportError, OSError, ValueError) as error:
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


def _audit_formal_input_package(
    audit: _Audit,
    root: Path,
    result: Mapping[str, object],
) -> None:
    if result.get("lane") != "formal":
        return
    fixed_files = {
        "protocol.json": root / "protocol.json",
        "schemas/formal-binding.schema.json": root / "schemas" / "formal-binding.schema.json",
        "schemas/raw-result.schema.json": root / "schemas" / "raw-result.schema.json",
    }
    binding_path = root / "formal-binding.json"
    lock_path = root / "requirements-wm001.lock"
    seal_path = root / "SEALED_PROTOCOL.sha256"
    required = [binding_path, lock_path, seal_path, *fixed_files.values()]
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
            result.get("formal_binding_sha256") == binding_digest,
            code="formal_binding_result_digest_mismatch",
            message="result does not bind the copied pre-outcome formal binding",
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
        environment = binding.get("environment")
        execution = result.get("execution")
        if not all(
            isinstance(value, Mapping)
            for value in (
                protocol_block,
                dependencies,
                source,
                environment,
                execution,
            )
        ):
            raise ArtifactAuditError("formal binding/result blocks are malformed")
        assert isinstance(protocol_block, Mapping)
        assert isinstance(dependencies, Mapping)
        assert isinstance(source, Mapping)
        assert isinstance(environment, Mapping)
        assert isinstance(execution, Mapping)
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
        _audit_bound_source_snapshot(audit, root, source)
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
        audit.limitation(
            "The preserved formal test report is content-addressed but has no "
            "machine-verifiable result schema; its pass/fail semantics still "
            "require human review."
        )
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


_ANALYSIS_TASK_A = "pendulum_normal_torque"
_ANALYSIS_TASK_B = "pendulum_reversed_torque"
_ANALYSIS_BEHAVIOR_CONDITIONS: Mapping[str, tuple[str, ...]] = {
    _ANALYSIS_TASK_A: (
        "cold",
        "after_a",
        "frozen",
        "corrupted",
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
        "after_b_replay",
        "after_b_naive",
    ),
    _ANALYSIS_TASK_B: ("after_a", "after_b_replay", "after_b_naive"),
}
_ANALYSIS_BEHAVIOR_SPLITS = {
    _ANALYSIS_TASK_A: "behavior_evaluation_a",
    _ANALYSIS_TASK_B: "behavior_evaluation_b",
}
_ANALYSIS_PREDICTIVE_SPLITS = {
    _ANALYSIS_TASK_A: "predictive_validation_a",
    _ANALYSIS_TASK_B: "predictive_validation_b",
}
_ANALYSIS_METRIC_UNITS: Mapping[str, str] = {
    "a_nll_improvement_after_a_vs_frozen": "nats/target-dimension",
    "a_nll_improvement_after_a_vs_corrupted": "nats/target-dimension",
    "a_after_a_interval_90_coverage": "fraction",
    "a_return_improvement_after_a_vs_cold": "return",
    "a_return_improvement_after_a_vs_frozen": "return",
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
            "a_after_a_interval_90_coverage",
            "mean",
            "ge",
            0.7,
            "after_a_interval_coverage_lower_bound",
        ),
        (
            "a_after_a_interval_90_coverage",
            "mean",
            "le",
            0.99,
            "after_a_interval_coverage_upper_bound",
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

    nll = lambda task_id, condition: predictive_value(  # noqa: E731
        task_id,
        condition,
        "mixture_nll_nats_per_target_dimension",
    )
    operands: Mapping[str, tuple[float | None, float | None]] = {
        "a_nll_improvement_after_a_vs_frozen": (
            nll(_ANALYSIS_TASK_A, "frozen"),
            nll(_ANALYSIS_TASK_A, "after_a"),
        ),
        "a_nll_improvement_after_a_vs_corrupted": (
            nll(_ANALYSIS_TASK_A, "corrupted"),
            nll(_ANALYSIS_TASK_A, "after_a"),
        ),
        "a_return_improvement_after_a_vs_cold": (
            returns[(_ANALYSIS_TASK_A, "after_a")],
            returns[(_ANALYSIS_TASK_A, "cold")],
        ),
        "a_return_improvement_after_a_vs_frozen": (
            returns[(_ANALYSIS_TASK_A, "after_a")],
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
    coverage = predictive_value(
        _ANALYSIS_TASK_A,
        "after_a",
        "interval_90_coverage",
    )
    if coverage is not None and math.isfinite(coverage):
        values["a_after_a_interval_90_coverage"] = coverage

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


def _independent_recompute_gate_results(
    aggregates: Sequence[Mapping[str, object]],
    replicate_values: Sequence[Mapping[str, float]],
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

    protocol = _json_without_duplicate_keys(
        _read_bounded(HERE / "protocol.json", 16 << 20, label="WM-001 protocol"),
        label="WM-001 protocol",
    )
    if not isinstance(protocol, Mapping):
        raise ArtifactAuditError("WM-001 protocol root is not an object")
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
        audit.gap(
            "formal_execution_not_present",
            (
                "This is a development artifact. Its measurements may be "
                "structurally valid, but the sealed protocol explicitly makes "
                "development runs ineligible for capability-claim adjudication."
            ),
            evidence_needed=(
                "A completed formal-lane artifact with all eight predeclared "
                "replicates, the sealed budgets, and a valid pre-run binding."
            ),
        )
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
        result.get("schema") == "prospect.world-model-lifecycle.raw-result.v2"
        and result.get("experiment_id") == "WM-001",
        code="result_identity_mismatch",
        message="artifact is not a WM-001 raw-result v2 document",
    )
    protocol_source = root / "protocol.json" if (root / "protocol.json").is_file() else HERE / "protocol.json"
    try:
        protocol_digest = hashlib.sha256(
            _read_bounded(
                protocol_source,
                16 << 20,
                label="WM-001 protocol",
            )
        ).hexdigest()
    except ArtifactAuditError:
        protocol_digest = ""
    audit.require(
        result.get("protocol_version") == "1.1.1" and result.get("protocol_sha256") == protocol_digest,
        code="result_protocol_binding_mismatch",
        message="result does not bind the exact WM-001 protocol 1.1.1 bytes",
    )
    replicates = _mapping_rows(result.get("replicates"))
    audit.require(
        bool(replicates),
        code="replicates_missing",
        message="result has no auditable replicate rows",
    )
    _audit_formal_input_package(audit, root, result)
    seen_replicates: set[str] = set()
    execution = result.get("execution")
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
    _audit_recomputed_analysis(audit, result)
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
        require_claim_completeness=require_claim_completeness,
    )


def _audit_report(
    audit: _Audit,
    *,
    root: Path,
    result_path: Path,
    result_sha256: str | None,
    require_claim_completeness: bool,
) -> dict[str, object]:
    integrity_passed = audit.failed_checks == 0
    complete_for_claim = not audit.coverage_gaps
    return {
        "schema": "prospect.world-model-lifecycle.artifact-audit.v1",
        "artifact_root": str(root.resolve()),
        "result_file": result_path.name,
        "result_sha256": result_sha256,
        "integrity_passed": integrity_passed,
        "complete_for_claim": complete_for_claim,
        "passed": integrity_passed and (complete_for_claim or not require_claim_completeness),
        "check_counts": {
            "passed": audit.passed_checks,
            "failed": audit.failed_checks,
            "coverage_gaps": len(audit.coverage_gaps),
        },
        "resource_limits_bytes": {
            "result": _MAX_RESULT_BYTES,
            "prediction_sidecar": _MAX_PREDICTION_BYTES,
            "owned_model_state": _MAX_OWNED_MODEL_BYTES,
            "optimizer_manifest": _MAX_MANIFEST_BYTES,
            "target_permutation": _MAX_PERMUTATION_BYTES,
            "checkpoint_archive": _MAX_CHECKPOINT_BYTES,
            "source_file": _MAX_SOURCE_FILE_BYTES,
            "source_snapshot": _MAX_SOURCE_SNAPSHOT_BYTES,
        },
        "custody": audit.custody,
        "findings": audit.findings,
        "coverage_gaps": audit.coverage_gaps,
        "independence_limitations": audit.independence_limitations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="artifact directory or result.json")
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
    report = audit_artifact(
        arguments.artifact,
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
