"""Independent, fail-closed auditor for the WM-002 Q1 runtime qualification.

The auditor deliberately does not import :mod:`bench.active_acquisition.q1`.
It reopens the canonical producer artifacts, reconstructs private random
variables from a separately supplied salt, verifies every framed checkpoint,
and recomputes episode semantics and master-level statistics from primitive
rows.  Producer aggregates are independently cross-checked for discrepancies,
but never used as evidence.

Q1 is permanently claim-ineligible and formally unauthorized.  A passing
audit qualifies only the exact known-model runtime chain frozen by protocol
``0.3.0-q1``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import stat
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from importlib.metadata import version as package_version
from pathlib import Path
from typing import BinaryIO, Final, Protocol, cast

from bench.active_acquisition.checkpoint import (
    QualificationBinding,
    RestoredQ1Checkpoint,
    load_q1_checkpoint,
)
from bench.active_acquisition.contracts import (
    ACTION_ORDER,
    ARM_ORDER,
    MACHINE_GENERATED_REVIEWER_MARK,
    Q0_IMPLEMENTATION_SHA256,
    Q0_PROTOCOL_SHA256,
    Q0_REPORT_SHA256,
    Q1_PROTOCOL_PATH,
    Q1_PROTOCOL_VERSION,
    Q1_SCHEMA_PATHS,
    canonical_json_bytes,
    canonical_sha256,
    implementation_manifest,
    sha256_bytes,
    validate_artifact,
)
from bench.active_acquisition.q1_audit_privacy import PrivatePrefixScanner
from prospect.runtime import ModelState

MASTER_COUNT: Final = 4
EPISODES_PER_MASTER: Final = 1024

RAW_TRACE_FILENAME: Final = "raw-trace.jsonl"
PRIVATE_AUDIT_FILENAME: Final = "private-audit.jsonl"
CHECKPOINT_INDEX_FILENAME: Final = "checkpoint-index.jsonl"
CHECKPOINT_FRAMES_FILENAME: Final = "checkpoint-frames.bin"
RESTORED_TRACE_FILENAME: Final = "restored-trace.jsonl"
Q1_AUDIT_IMPLEMENTATION_PATHS: Final = (
    "bench/__init__.py",
    "bench/active_acquisition/__init__.py",
    "bench/active_acquisition/attempt.py",
    "bench/active_acquisition/checkpoint.py",
    "bench/active_acquisition/contracts.py",
    "bench/active_acquisition/oracle.py",
    "bench/active_acquisition/policies.py",
    "bench/active_acquisition/problem.py",
    "bench/active_acquisition/q1.py",
    "bench/active_acquisition/q1_audit.py",
    "bench/active_acquisition/q1_audit_privacy.py",
    "bench/active_acquisition/q1_protocol.json",
    "bench/active_acquisition/q1_qualification.py",
    "bench/active_acquisition/restore_worker.py",
    "bench/active_acquisition/runtime_lane.py",
    "bench/active_acquisition/worker_capability.py",
    "bench/active_acquisition/schemas/q1-aggregate.schema.json",
    "bench/active_acquisition/schemas/q1-audit-output.schema.json",
    "bench/active_acquisition/schemas/q1-checkpoint-frame.schema.json",
    "bench/active_acquisition/schemas/q1-entry-qualification.schema.json",
    "bench/active_acquisition/schemas/q1-private-audit.schema.json",
    "bench/active_acquisition/schemas/q1-prospective-review.schema.json",
    "bench/active_acquisition/schemas/q1-raw-trace.schema.json",
    "bench/active_acquisition/schemas/q1-restored-trace.schema.json",
    "bench/active_acquisition/schemas/q1-worker-capability.schema.json",
    "bench/active_acquisition/seeding.py",
    "pyproject.toml",
    "src/prospect/__init__.py",
    "src/prospect/decision/__init__.py",
    "src/prospect/decision/policy.py",
    "src/prospect/domain/__init__.py",
    "src/prospect/domain/protocols.py",
    "src/prospect/domain/records.py",
    "src/prospect/epistemics/__init__.py",
    "src/prospect/epistemics/assessments.py",
    "src/prospect/epistemics/information.py",
    "src/prospect/epistemics/scoring.py",
    "src/prospect/runtime/__init__.py",
    "src/prospect/runtime/agent.py",
    "src/prospect/runtime/journal.py",
    "src/prospect/runtime/learning.py",
    "src/prospect/runtime/state.py",
    "src/prospect/storage/__init__.py",
    "src/prospect/storage/checkpoint.py",
    "src/prospect/storage/domain_graph.py",
    "src/prospect/storage/ledger.py",
    "src/prospect/storage/memory.py",
    "src/prospect/storage/torchrl_replay.py",
)
PRODUCER_AGGREGATE_FILENAME: Final = "aggregate.json"

AUDIT_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-independent-audit.v2"
GATE_ORDER: Final = ("Q1-K0", "Q1-K1", "Q1-K2", "Q1-K3", "Q1-K4", "Q1-K5")
CONTROL_ARMS: Final = (
    "goal_only",
    "raw_observation_entropy",
    "eig_only",
    "shuffled_information",
    "uniform_random",
)
T_CRITICAL_DF3: Final = 3.182446305284263
EXPECTED_EPISODES: Final = MASTER_COUNT * len(ARM_ORDER) * EPISODES_PER_MASTER
_ZERO_SHA256: Final = "0" * 64
_FLOAT_ABS_TOL: Final = 1e-12
_MAX_REPORTED_VIOLATIONS: Final = 128
_MAX_JSONL_ROW_BYTES: Final = 32 * 1024 * 1024
_MAX_CHECKPOINT_FRAME_BYTES: Final = 32 * 1024 * 1024
_MAX_SMALL_DOCUMENT_BYTES: Final = 8 * 1024 * 1024
_NORMALIZED_Q1_PROTOCOL_SHA256: Final = "e015c8b519a10eeab19ecda1384071b17c450484c06ebbd4142d95143c92353b"
_RUN_IDENTITY_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-run-identity.v1"
_ATTEMPT_MARKER_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-attempt-marker.v2"
_ATTEMPT_MARKER_FILENAME: Final = "wm002-q1.attempt.json"
_ATTEMPT_ARTIFACT_NAMES: Final = (
    "aggregate",
    "checkpoint_frames",
    "checkpoint_index",
    "private_audit",
    "raw_trace",
    "restored_trace",
)
_ARTIFACT_FILENAMES: Final = frozenset(
    {
        PRODUCER_AGGREGATE_FILENAME,
        CHECKPOINT_FRAMES_FILENAME,
        CHECKPOINT_INDEX_FILENAME,
        PRIVATE_AUDIT_FILENAME,
        RAW_TRACE_FILENAME,
        RESTORED_TRACE_FILENAME,
    }
)
_ENTRY_SCHEMA_PATH: Final = Path(__file__).with_name("schemas") / "q1-entry-qualification.schema.json"
_PROSPECTIVE_REVIEW_SCHEMA_PATH: Final = Path(__file__).with_name("schemas") / "q1-prospective-review.schema.json"
_PROSPECTIVE_REVIEW_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-prospective-review.v1"
_PROSPECTIVE_REVIEW_METHOD: Final = "adversarial_result_free_selected_source_review"
_PROSPECTIVE_REVIEW_ASSURANCE_BOUNDARY: Final = "local_procedural_review_without_external_signature"
_PROSPECTIVE_REVIEW_SCOPE: Final = (
    "accepted_q0_and_successor_authority",
    "runtime_semantics_and_transactional_causality",
    "private_seed_exactness_and_noninterference",
    "checkpoint_and_fresh_process_restore",
    "artifact_schemas_attempt_integrity_and_resources",
    "independent_auditor_recomputation_and_scale",
    "evidence_and_claim_boundary",
)
_ENTRY_CHECK_ORDER: Final = (
    "accepted_q0_binding",
    "successor_protocol_boundary",
    "protocol_code_parity",
    "artifact_schema_contract",
    "selected_source_implementation_binding",
    "private_salt_custody",
    "authoritative_runtime_checkpoint_privacy_probe",
    "execution_resource_and_single_attempt_preflight",
    "emitted_artifact_schema_and_mutation_probe",
    "independent_prospective_review",
)
_JSONSCHEMA_VERSION: Final = "4.25.1"
_ATTEMPT_EXPECTED_COUNTS: Final = {
    "acquisition_updates": EXPECTED_EPISODES,
    "checkpoints": EXPECTED_EPISODES,
    "environment_steps": EXPECTED_EPISODES * 2,
    "episodes": EXPECTED_EPISODES,
    "restores": EXPECTED_EPISODES,
    "terminal_updates": 0,
    "transitions": EXPECTED_EPISODES * 2,
}

_AGENT_ID: Final = "wm002-agent"
_INITIAL_IDENTITY_COUNTER: Final = 0
# Five acquisition candidates allocate five IDs each, followed by 18 IDs for
# the selected decision, environment step, experience/transition, assimilation,
# score/effect, and committed learning record.
_PRETERMINAL_IDENTITY_COUNTER: Final = 43
_MODEL_VERSION_PREFIX: Final = "wm002-model-sha256:"
_CONFIGURATION_VERSION_PREFIX: Final = "wm002-config-sha256:"
_POSTERIOR_MODEL_SCHEMA: Final = "prospect.wm002.posterior-model.v1"
_LIKELIHOOD_VERSION: Final = "wm002-hidden-actuator-true-v1"
_PRIVATE_KEY_PREFIX: Final = "WM-002|0.3.0-q1|q1v3|"
_THETA_NAMESPACE: Final = "q1v3-theta-balanced-order"
_PULSE_NAMESPACE: Final = "q1v3-pulse-outcome"
_NUISANCE_NAMESPACE: Final = "q1v3-nuisance-outcome"
_TERMINAL_NAMESPACE: Final = "q1v3-terminal-outcome"
_UNIFORM_NAMESPACE: Final = "q1v3-uniform"
_DIGEST_DENOMINATOR: Final = 1 << 256

_TARGET_ID: Final = "wm002-acquisition-observation-and-terminal-success"
_TARGET_KIND: Final = "composite_observation_regime_and_outcome"
_TARGET_DESCRIPTION: Final = (
    "phase-qualified acquisition observation, inferred hidden actuator regime, "
    "and executed terminal success; the belief distribution represents the regime "
    "while action-conditional predictions represent the phase-qualified observable"
)
_ASSIMILATOR_VERSION: Final = "wm002-no-op-observation-assimilator-v1"
_LEARNER_VERSION: Final = "wm002-exact-posterior-transactional-learner-v1"
_SCORER_VERSION: Final = "wm002-acquisition-terminal-log-scorer-v1"
_EFFECT_VERSION: Final = "wm002-no-op-assimilation-effect-v1"
_REPRESENTATION_VERSION: Final = "wm002-categorical-regime-v1"
_CALIBRATION_VERSION: Final = "wm002-known-likelihood-v1"
_UTILITY_EVALUATOR_VERSION: Final = "wm002-known-return-utility-v1"
_INFORMATION_EVALUATOR_VERSION: Final = "wm002-known-decision-value-v1"
_CANDIDATE_EVALUATOR_VERSION: Final = "wm002-truthful-candidate-assessor-v1"

_INITIAL_MODEL_PAYLOAD: Final = canonical_json_bytes(
    {
        "evidence_count": 0,
        "last_experience_id": None,
        "last_transition_id": None,
        "likelihood_version": _LIKELIHOOD_VERSION,
        "posterior_direct": {"denominator": 2, "numerator": 1},
        "schema": "prospect.wm002.posterior-model.v1",
    }
)
_INITIAL_MODEL_SHA256: Final = sha256_bytes(_INITIAL_MODEL_PAYLOAD)
_INITIAL_MODEL_VERSION: Final = f"{_MODEL_VERSION_PREFIX}{_INITIAL_MODEL_SHA256}"
_INITIAL_CONFIGURATION_VERSION: Final = f"{_CONFIGURATION_VERSION_PREFIX}{_INITIAL_MODEL_SHA256}"

_EXPECTED_ACTION_BY_ARM: Final = {
    "prospect_expected_return": "strong",
    "independent_fraction_oracle": "strong",
    "goal_only": "skip",
    "raw_observation_entropy": "nuisance",
    "eig_only": "overpowered",
    "shuffled_information": "weak",
}
_ACTION_ACCURACY: Final = {
    "weak": Fraction(7, 10),
    "strong": Fraction(9, 10),
    "overpowered": Fraction(1),
}
_ACTION_COST: Final = {
    "skip": Fraction(0),
    "weak": Fraction(53, 100),
    "strong": Fraction(58, 100),
    "overpowered": Fraction(95, 100),
    "nuisance": Fraction(0),
}
_INFORMATION_COST: Final = {
    "skip": Fraction(0),
    "weak": Fraction(0),
    "strong": Fraction(0),
    "overpowered": Fraction(0),
    "nuisance": Fraction(1, 100),
}
_EXPECTED_EPISODE_VALUE: Final = {
    "skip": Fraction(1, 2),
    "weak": Fraction(63, 100),
    "strong": Fraction(37, 50),
    "overpowered": Fraction(9, 20),
    "nuisance": Fraction(49, 100),
}
_EXPECTED_IMMEDIATE: Final = {
    "skip": Fraction(0),
    "weak": Fraction(1, 2),
    "strong": Fraction(1, 2),
    "overpowered": Fraction(1, 2),
    "nuisance": Fraction(0),
}
_EXPECTED_DECISION_VALUE: Final = {
    "skip": Fraction(0),
    "weak": Fraction(4, 25),
    "strong": Fraction(8, 25),
    "overpowered": Fraction(2, 5),
    "nuisance": Fraction(0),
}
_SHUFFLED_SOURCE: Final = {
    "skip": "skip",
    "weak": "overpowered",
    "strong": "weak",
    "overpowered": "strong",
    "nuisance": "nuisance",
}


class _IndependentSeedSchedule:
    """Local HMAC implementation; no producer seeding helper is trusted."""

    def __init__(self, secret_salt: bytes) -> None:
        if type(secret_salt) is not bytes or len(secret_salt) < 32:
            raise Q1AuditError("secret salt must be immutable bytes with at least 32 bytes")
        self._salt = secret_salt
        self._theta_cache: dict[int, tuple[int, ...]] = {}

    @property
    def salt_commitment_sha256(self) -> str:
        return hashlib.sha256(self._salt).hexdigest()

    def theta_schedule(self, master: int) -> tuple[int, ...]:
        _require_master_episode(master, 0)
        cached = self._theta_cache.get(master)
        if cached is not None:
            return cached
        ranked = sorted(
            (self._digest(self._theta_key(master, episode)), episode) for episode in range(EPISODES_PER_MASTER)
        )
        reversed_episodes = {episode for _, episode in ranked[: EPISODES_PER_MASTER // 2]}
        schedule = tuple(-1 if episode in reversed_episodes else 1 for episode in range(EPISODES_PER_MASTER))
        self._theta_cache[master] = schedule
        return schedule

    def theta(self, master: int, episode: int) -> int:
        _require_master_episode(master, episode)
        return self.theta_schedule(master)[episode]

    def theta_semantic_sha256(self, master: int, episode: int) -> str:
        _require_master_episode(master, episode)
        return hashlib.sha256(self._theta_key(master, episode)).hexdigest()

    def pulse_observation(self, master: int, episode: int, action: str, theta: int) -> int:
        _require_master_episode(master, episode)
        if action not in _ACTION_ACCURACY or theta not in {-1, 1}:
            raise Q1AuditError("pulse reconstruction received an invalid action or theta")
        draw = int.from_bytes(self._digest(self._pulse_key(master, episode, action)), "big")
        reliability = _ACTION_ACCURACY[action]
        agrees = draw * reliability.denominator < reliability.numerator * _DIGEST_DENOMINATOR
        return theta if agrees else -theta

    def nuisance_observation(self, master: int, episode: int) -> int:
        _require_master_episode(master, episode)
        draw = int.from_bytes(self._digest(self._nuisance_key(master, episode)), "big")
        return draw * 4 // _DIGEST_DENOMINATOR

    def terminal_success(self, master: int, episode: int, decision: int, theta: int) -> bool:
        _require_master_episode(master, episode)
        if decision not in {-1, 1} or theta not in {-1, 1}:
            raise Q1AuditError("terminal reconstruction received an invalid decision or theta")
        draw = int.from_bytes(self._digest(self._terminal_key(master, episode, decision)), "big")
        probability = Fraction(9, 10) if decision == theta else Fraction(1, 10)
        return draw * probability.denominator < probability.numerator * _DIGEST_DENOMINATOR

    def uniform_selection(self, master: int, episode: int) -> tuple[str, str]:
        _require_master_episode(master, episode)
        key = f"WM-002|{Q1_PROTOCOL_VERSION}|{_UNIFORM_NAMESPACE}|{master}|{episode}".encode("ascii")
        digest = hashlib.sha256(key).hexdigest()
        return ACTION_ORDER[int(digest, 16) % len(ACTION_ORDER)], digest

    def private_row(self, master: int, episode: int, arm: str) -> dict[str, object]:
        _require_master_episode(master, episode)
        if arm not in ARM_ORDER:
            raise Q1AuditError("private row uses an undeclared arm")
        return {
            "arm_id": arm,
            "episode": episode,
            "hmac_sha256": {
                "nuisance": self._digest(self._nuisance_key(master, episode)).hex(),
                "pulse": {
                    action: self._digest(self._pulse_key(master, episode, action)).hex()
                    for action in ("weak", "strong", "overpowered")
                },
                "terminal": {
                    "+1": self._digest(self._terminal_key(master, episode, 1)).hex(),
                    "-1": self._digest(self._terminal_key(master, episode, -1)).hex(),
                },
                "theta_order": self._digest(self._theta_key(master, episode)).hex(),
            },
            "master": master,
            "salt_commitment_sha256": self.salt_commitment_sha256,
            "schema": "prospect.wm002.q1-private-seed-material.v1",
            "theta": self.theta(master, episode),
        }

    def private_hmac_digests(self, master: int, episode: int, arm: str) -> tuple[bytes, ...]:
        row = self.private_row(master, episode, arm)
        hmacs = cast(Mapping[str, object], row["hmac_sha256"])
        pulse = cast(Mapping[str, object], hmacs["pulse"])
        terminal = cast(Mapping[str, object], hmacs["terminal"])
        values = (
            hmacs["theta_order"],
            pulse["weak"],
            pulse["strong"],
            pulse["overpowered"],
            hmacs["nuisance"],
            terminal["+1"],
            terminal["-1"],
        )
        return tuple(bytes.fromhex(cast(str, value)) for value in values)

    def _digest(self, key: bytes) -> bytes:
        return hmac.digest(self._salt, key, "sha256")

    @staticmethod
    def _theta_key(master: int, episode: int) -> bytes:
        return _private_key(
            _THETA_NAMESPACE,
            ("master", master),
            ("episode", episode),
            ("role", "order"),
        )

    @staticmethod
    def _pulse_key(master: int, episode: int, action: str) -> bytes:
        return _private_key(
            _PULSE_NAMESPACE,
            ("master", master),
            ("episode", episode),
            ("action", action),
            ("role", "observation"),
        )

    @staticmethod
    def _nuisance_key(master: int, episode: int) -> bytes:
        return _private_key(
            _NUISANCE_NAMESPACE,
            ("master", master),
            ("episode", episode),
            ("action", "nuisance"),
            ("role", "observation"),
        )

    @staticmethod
    def _terminal_key(master: int, episode: int, decision: int) -> bytes:
        rendered = f"+{decision}" if decision > 0 else str(decision)
        return _private_key(
            _TERMINAL_NAMESPACE,
            ("master", master),
            ("episode", episode),
            ("decision", rendered),
            ("role", "success"),
        )


@dataclass(frozen=True, slots=True)
class _DecodedPosteriorModel:
    evidence_count: int
    last_experience_id: str | None
    last_transition_id: str | None
    posterior_direct: Fraction


class _HashSink(Protocol):
    def update(self, data: bytes, /) -> object: ...


class Q1AuditError(ValueError):
    """The auditor could not safely interpret an input artifact."""


_AUDIT_DIRECTORY_IDENTITY_FIELDS: Final = frozenset(
    {"canonical_path", "st_dev", "st_ino", "st_uid", "st_gid", "file_type", "mode"}
)


@dataclass(frozen=True, slots=True)
class _AuditDirectoryIdentity:
    """Independent reconstruction of one entry-bound private directory."""

    canonical_path: str
    st_dev: int
    st_ino: int
    st_uid: int
    st_gid: int
    file_type: str = "directory"
    mode: str = "0700"

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_path": self.canonical_path,
            "file_type": self.file_type,
            "mode": self.mode,
            "st_dev": self.st_dev,
            "st_gid": self.st_gid,
            "st_ino": self.st_ino,
            "st_uid": self.st_uid,
        }


@dataclass(frozen=True, slots=True)
class _AuditBindings:
    """Exact external identities that producer artifacts cannot self-attest."""

    protocol_sha256: str
    implementation_sha256: str
    q0_report_sha256: str
    entry_qualification_sha256: str
    prospective_review_sha256: str
    salt_commitment_sha256: str
    run_sha256: str
    run_id: str
    attempt_id: str
    execution_root: _AuditDirectoryIdentity | None
    attempt_registry_directory: _AuditDirectoryIdentity | None

    def __post_init__(self) -> None:
        for label, value in (
            ("protocol_sha256", self.protocol_sha256),
            ("implementation_sha256", self.implementation_sha256),
            ("q0_report_sha256", self.q0_report_sha256),
            ("entry_qualification_sha256", self.entry_qualification_sha256),
            ("prospective_review_sha256", self.prospective_review_sha256),
            ("salt_commitment_sha256", self.salt_commitment_sha256),
            ("run_sha256", self.run_sha256),
        ):
            if not _is_sha256(value):
                raise Q1AuditError(f"{label} must be lowercase SHA-256 hexadecimal")
        if self.q0_report_sha256 != Q0_REPORT_SHA256:
            raise Q1AuditError("q0_report_sha256 differs from the accepted canonical Q0 report")
        if self.run_id != f"wm002-q1-{self.run_sha256}":
            raise Q1AuditError("run_id does not bind the independently recomputed run digest")
        if self.attempt_id != f"{self.run_id}-attempt-0001":
            raise Q1AuditError("attempt_id differs from the sole frozen attempt")
        for label, identity in (
            ("execution_root", self.execution_root),
            ("attempt_registry_directory", self.attempt_registry_directory),
        ):
            if identity is not None and not Path(identity.canonical_path).is_absolute():
                raise Q1AuditError(f"entry-bound {label} must contain an absolute canonical path")


@dataclass(frozen=True, slots=True)
class _StreamingAuditResult:
    returns: Mapping[tuple[int, str], tuple[float, ...]]
    counts: Mapping[str, int]
    aggregate: Mapping[str, object] | None


class _Violations:
    """Bound diagnostic output while retaining the exact total failure count."""

    def __init__(self) -> None:
        self._rows: dict[str, list[str]] = {gate: [] for gate in GATE_ORDER}
        self._totals: dict[str, int] = {gate: 0 for gate in GATE_ORDER}

    def add(self, gate: str, message: str) -> None:
        self._totals[gate] += 1
        rows = self._rows[gate]
        if len(rows) < _MAX_REPORTED_VIOLATIONS:
            rows.append(message)

    def extend(self, gate: str, messages: Iterable[str]) -> None:
        for message in messages:
            self.add(gate, message)

    def rows(self, gate: str) -> tuple[str, ...]:
        rows = list(self._rows[gate])
        omitted = self._totals[gate] - len(rows)
        if omitted:
            rows.append(f"{omitted} additional violation(s) omitted")
        return tuple(rows)


def _validate_schema_without_instance_echo(
    value: Mapping[str, object],
    *,
    schema_path: Path,
    label: str,
) -> None:
    """Validate one mapping while never reproducing instance values in diagnostics."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise Q1AuditError(f"{label} validation requires jsonschema") from error
    try:
        schema_payload = _read_regular_file(schema_path, label=f"{label} schema")
        schema = json.loads(
            schema_payload.decode("utf-8"),
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
        if not isinstance(schema, dict):
            raise Q1AuditError(f"{label} schema is not an object")
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda row: list(row.path),
        )
    except Q1AuditError:
        raise
    except Exception as error:
        raise Q1AuditError(f"{label} schema validation setup failed ({type(error).__name__})") from error
    if errors:
        first = errors[0]
        keyword = first.validator if isinstance(first.validator, str) else "unknown"
        depth = len(tuple(first.absolute_path))
        raise Q1AuditError(f"{label} schema validation failed (keyword={keyword},depth={depth})")


def _validate_entry_schema(entry: Mapping[str, object]) -> None:
    """Validate the external entry report without importing its producer."""

    _validate_schema_without_instance_echo(
        entry,
        schema_path=_ENTRY_SCHEMA_PATH,
        label="entry report",
    )


def _load_validated_prospective_review(
    path: Path,
    *,
    protocol_sha256: str,
    implementation_sha256: str,
    reviewed_source_count: int,
) -> str:
    """Reopen and independently validate the exact result-free prospective review."""

    payload = _read_regular_file(path, label="Q1 prospective review")
    digest = sha256_bytes(payload)
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except Exception as error:
        raise Q1AuditError(f"prospective review JSON decoding failed ({type(error).__name__})") from error
    if not isinstance(decoded, dict):
        raise Q1AuditError("prospective review is not an object")
    review = cast(Mapping[str, object], decoded)
    try:
        canonical = canonical_json_bytes(review, newline=True)
    except Exception as error:
        raise Q1AuditError(f"prospective review canonicalization failed ({type(error).__name__})") from error
    if payload != canonical:
        raise Q1AuditError("prospective review is not one canonical JSON document")
    _validate_schema_without_instance_echo(
        review,
        schema_path=_PROSPECTIVE_REVIEW_SCHEMA_PATH,
        label="prospective review",
    )

    expected = {
        "schema": _PROSPECTIVE_REVIEW_SCHEMA,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "review_method": _PROSPECTIVE_REVIEW_METHOD,
        "assurance_boundary": _PROSPECTIVE_REVIEW_ASSURANCE_BOUNDARY,
        "reviewed_source_count": reviewed_source_count,
        "review_scope": list(_PROSPECTIVE_REVIEW_SCOPE),
        "q1_environment_interactions": 0,
        "q1_private_draws": 0,
        "claim_eligible": False,
        "formal_authorized": False,
        "passed": True,
    }
    for field, expected_value in expected.items():
        if review.get(field) != expected_value or type(review.get(field)) is not type(expected_value):
            raise Q1AuditError(f"prospective review field mismatch:{field}")
    if review.get("blocking_findings") != []:
        raise Q1AuditError("prospective review has blocking findings")
    nonblocking = review.get("nonblocking_findings")
    if not isinstance(nonblocking, list) or any(type(row) is not str or not row for row in nonblocking):
        raise Q1AuditError("prospective review nonblocking findings are not nonempty strings")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer:
        raise Q1AuditError("prospective review reviewer is not a nonempty string")
    if MACHINE_GENERATED_REVIEWER_MARK in reviewer:
        raise Q1AuditError("prospective review is the machine-generated rehearsal review")
    if not isinstance(review.get("statement"), str) or not cast(str, review.get("statement")):
        raise Q1AuditError("prospective review statement is not a nonempty string")
    return digest


def _require_entry_prospective_review_binding(
    entry: Mapping[str, object],
    prospective_review_sha256: str,
) -> None:
    declared = entry.get("prospective_review_sha256")
    if not _is_sha256(declared) or declared == _ZERO_SHA256:
        raise Q1AuditError("entry qualification lacks a nonzero prospective-review digest")
    if declared != prospective_review_sha256:
        raise Q1AuditError("entry qualification prospective-review digest differs from reopened review bytes")


def _canonical_absolute_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise Q1AuditError(f"{label} must be a nonempty absolute path")
    resolved = str(Path(value).resolve())
    if resolved != value:
        raise Q1AuditError(f"{label} is not canonical/resolved")
    return resolved


def _parse_audit_directory_identity(value: object, *, label: str) -> _AuditDirectoryIdentity:
    if not isinstance(value, Mapping) or set(value) != _AUDIT_DIRECTORY_IDENTITY_FIELDS:
        raise Q1AuditError(f"{label} fields differ from the exact directory identity contract")
    canonical_path = _canonical_absolute_path(value.get("canonical_path"), f"{label} canonical_path")
    integers: dict[str, int] = {}
    for field in ("st_dev", "st_ino", "st_uid", "st_gid"):
        candidate = value.get(field)
        if type(candidate) is not int or candidate < 0:
            raise Q1AuditError(f"{label} {field} must be a nonnegative integer")
        integers[field] = candidate
    if value.get("file_type") != "directory" or type(value.get("file_type")) is not str:
        raise Q1AuditError(f"{label} file_type must be exactly directory")
    if value.get("mode") != "0700" or type(value.get("mode")) is not str:
        raise Q1AuditError(f"{label} mode must be exactly 0700")
    return _AuditDirectoryIdentity(
        canonical_path=canonical_path,
        st_dev=integers["st_dev"],
        st_ino=integers["st_ino"],
        st_uid=integers["st_uid"],
        st_gid=integers["st_gid"],
    )


def _capture_audit_directory_identity(path: Path, *, label: str) -> _AuditDirectoryIdentity:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Q1AuditError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o700:
            raise Q1AuditError(f"{label} must be a non-symlink directory with exact mode 0700")
        if before.st_uid != os.geteuid():
            raise Q1AuditError(f"{label} must be owned by the effective user")
        canonical_path = path.resolve(strict=True)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        canonical_after = path.resolve(strict=True)
        canonical_metadata = os.stat(canonical_after, follow_symlinks=False)
    except OSError as error:
        raise Q1AuditError(f"{label} identity cannot be verified safely") from error
    finally:
        os.close(descriptor)
    signature = _directory_identity_signature(before)
    if (
        signature != _directory_identity_signature(after)
        or signature != _directory_identity_signature(path_after)
        or signature != _directory_identity_signature(canonical_metadata)
        or canonical_path != canonical_after
    ):
        raise Q1AuditError(f"{label} identity changed during descriptor verification")
    return _AuditDirectoryIdentity(
        canonical_path=str(canonical_path),
        st_dev=before.st_dev,
        st_ino=before.st_ino,
        st_uid=before.st_uid,
        st_gid=before.st_gid,
    )


def _require_entry_bound_directory_identity(
    declared: _AuditDirectoryIdentity,
    *,
    label: str,
) -> _AuditDirectoryIdentity:
    actual = _capture_audit_directory_identity(Path(declared.canonical_path), label=label)
    if actual != declared:
        raise Q1AuditError(f"{label} differs from its entry-bound directory identity")
    return actual


def _require_audit_binding_directories(
    bindings: _AuditBindings,
) -> tuple[_AuditDirectoryIdentity, _AuditDirectoryIdentity]:
    if bindings.execution_root is None or bindings.attempt_registry_directory is None:
        raise Q1AuditError("entry qualification lacks complete directory identities")
    execution_root = _require_entry_bound_directory_identity(
        bindings.execution_root,
        label="entry execution root",
    )
    registry = _require_entry_bound_directory_identity(
        bindings.attempt_registry_directory,
        label="entry attempt registry",
    )
    return execution_root, registry


def _validate_loaded_source_origins() -> None:
    """Bind every loaded selected module to this hashed repository closure."""

    if sys.flags.optimize != 0:
        raise Q1AuditError("Q1 auditor requires an unoptimized Python interpreter")

    repository_root = Q1_PROTOCOL_PATH.resolve().parents[2]
    selected_modules: dict[str, str] = {}
    for relative in Q1_AUDIT_IMPLEMENTATION_PATHS:
        path = Path(relative)
        if path.suffix != ".py":
            continue
        parts = path.parts[1:] if path.parts[0] == "src" else path.parts
        if parts[-1] == "__init__.py":
            module_name = ".".join(parts[:-1])
        else:
            module_name = ".".join((*parts[:-1], Path(parts[-1]).stem))
        selected_modules[module_name] = relative

    required_modules = {
        "bench",
        "bench.active_acquisition",
        "bench.active_acquisition.checkpoint",
        "bench.active_acquisition.contracts",
        "bench.active_acquisition.q1_audit",
        "bench.active_acquisition.q1_audit_privacy",
        "prospect",
        "prospect.runtime",
    }
    main_module = sys.modules.get("__main__")
    main_name = getattr(getattr(main_module, "__spec__", None), "name", None)
    loaded_required: set[str] = set()
    for name, expected_relative in selected_modules.items():
        module = sys.modules.get(name)
        if module is None and name == main_name:
            # `python -S -m bench.active_acquisition.q1_audit` registers this
            # module as __main__, not under its package name. Bind the running
            # entrypoint by origin instead of declaring it absent.
            module = main_module
        if module is None:
            continue
        if name in required_modules:
            loaded_required.add(name)
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            raise Q1AuditError(f"loaded selected-source module lacks a filesystem origin:{name}")
        resolved = Path(origin).resolve()
        expected = (repository_root / expected_relative).resolve()
        if resolved != expected:
            raise Q1AuditError(f"loaded selected module origin differs from hashed source:{name}:{resolved}")
    if loaded_required != required_modules:
        missing = sorted(required_modules - loaded_required)
        raise Q1AuditError(f"auditor essential selected modules were not loaded:{missing}")


def _normalized_protocol_contract_sha256(protocol: Mapping[str, object]) -> str:
    """Independently bind every semantic field except the authorization-bit value."""

    if not isinstance(protocol, Mapping):
        raise Q1AuditError("Q1 protocol contract must be an object")
    experiment = protocol.get("experiment")
    if not isinstance(experiment, Mapping):
        raise Q1AuditError("Q1 protocol contract has no experiment object")
    execution_authorized = experiment.get("execution_authorized")
    if type(execution_authorized) is not bool:
        raise Q1AuditError("Q1 protocol execution_authorized must be exactly boolean")
    normalized_protocol = dict(protocol)
    normalized_experiment = dict(experiment)
    normalized_experiment["execution_authorized"] = False
    normalized_protocol["experiment"] = normalized_experiment
    payload = canonical_json_bytes(normalized_protocol)
    return hashlib.sha256(payload).hexdigest()


def _require_exact_protocol_fields(
    value: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for field, expected_value in expected.items():
        actual = value.get(field)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise Q1AuditError(f"canonical protocol {label} mismatch:{field}")


def _resolve_external_bindings(
    *,
    protocol_path: Path,
    q0_report_path: Path,
    entry_report_path: Path,
    prospective_review_path: Path,
    salt_commitment: str,
    violations: _Violations,
) -> _AuditBindings:
    protocol_digest = _ZERO_SHA256
    implementation_digest = _ZERO_SHA256
    entry_digest = _ZERO_SHA256
    prospective_review_digest = _ZERO_SHA256
    manifest_rows: list[dict[str, object]] = []
    execution_root: _AuditDirectoryIdentity | None = None
    attempt_registry_directory: _AuditDirectoryIdentity | None = None

    try:
        if protocol_path.resolve() != Q1_PROTOCOL_PATH.resolve():
            raise Q1AuditError("auditor requires the canonical Q1 protocol path")
        protocol_payload = _read_regular_file(protocol_path, label="canonical Q1 protocol")
        protocol_digest = sha256_bytes(protocol_payload)
        protocol = json.loads(
            protocol_payload.decode("utf-8"),
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
        if not isinstance(protocol, dict):
            raise Q1AuditError("canonical Q1 protocol is not an object")
        if protocol.get("schema") != "prospect.wm002.active-acquisition.q1-protocol.v1":
            raise Q1AuditError("canonical Q1 protocol schema differs")
        experiment = _required_mapping(protocol, "experiment")
        expected_experiment = {
            "protocol_version": Q1_PROTOCOL_VERSION,
            "claim_eligible": False,
            "formal_authorized": False,
            "execution_authorized": True,
        }
        for field, expected in expected_experiment.items():
            actual = experiment.get(field)
            if type(actual) is not type(expected) or actual != expected:
                raise Q1AuditError(f"canonical protocol execution boundary mismatch:{field}")
        formal = _required_mapping(protocol, "formal_boundary")
        if formal.get("authorized") is not False:
            raise Q1AuditError("canonical protocol formal boundary is not disabled")
        budget = _required_mapping(protocol, "budget")
        expected_budget = {
            "episodes_total": EXPECTED_EPISODES,
            "environment_steps_total": EXPECTED_EPISODES * 2,
            "transitions_total": EXPECTED_EPISODES * 2,
            "acquisition_updates_total": EXPECTED_EPISODES,
            "terminal_updates_total": 0,
        }
        for field, expected in expected_budget.items():
            actual = budget.get(field)
            if type(actual) is not type(expected) or actual != expected:
                raise Q1AuditError(f"canonical protocol budget mismatch:{field}")
        runtime = _required_mapping(protocol, "runtime")
        process_watchdogs = _required_mapping(runtime, "process_watchdogs")
        expected_process_watchdogs: dict[str, float | str] = {
            "producer_stage_timeout_seconds": 3600.0,
            "restore_child_timeout_seconds": 900.0,
            "restore_stage_timeout_seconds": 7200.0,
            "process_terminate_grace_seconds": 10.0,
            "rule": (
                "Any timeout or parent-side exception terminates, then kills if needed, and reaps "
                "every started child before a failed marker may be finalized; if quiescence cannot "
                "be proven, the marker remains started."
            ),
        }
        if set(process_watchdogs) != set(expected_process_watchdogs):
            raise Q1AuditError("canonical protocol process watchdog key set differs")
        for field, expected in expected_process_watchdogs.items():
            actual = process_watchdogs.get(field)
            if type(actual) is not type(expected) or actual != expected:
                raise Q1AuditError(f"canonical protocol process watchdog mismatch:{field}")
        process_launch = _required_mapping(runtime, "process_launch")
        if set(process_launch) != {
            "parent_cli_prefix",
            "parent_cli_no_site_required",
            "worker_base_command_token_count",
            "producer_base_command",
            "restore_base_command",
            "capability_argument",
            "working_directory",
            "stdin",
            "inherited_descriptor_count",
            "child_environment",
        }:
            raise Q1AuditError("canonical protocol process launch key set differs")
        _require_exact_protocol_fields(
            process_launch,
            {
                "parent_cli_prefix": [
                    "<sys.executable>",
                    "-S",
                    "-m",
                    "bench.active_acquisition.q1",
                    "run",
                ],
                "parent_cli_no_site_required": True,
                "worker_base_command_token_count": 5,
                "producer_base_command": [
                    "<sys.executable>",
                    "-S",
                    "-m",
                    "bench.active_acquisition.q1",
                    "_producer-master",
                ],
                "restore_base_command": [
                    "<sys.executable>",
                    "-S",
                    "-m",
                    "bench.active_acquisition.restore_worker",
                    "q1",
                ],
                "capability_argument": ["--capability-fd", "<inherited-fd>"],
                "working_directory": "canonical repository root",
                "stdin": "subprocess.DEVNULL",
                "inherited_descriptor_count": 1,
            },
            label="process launch",
        )
        child_environment = _required_mapping(process_launch, "child_environment")
        if set(child_environment) != {
            "allowlisted_keys",
            "fixed_values",
            "pythonpath_rule",
            "site_processing",
        }:
            raise Q1AuditError("canonical protocol child environment key set differs")
        _require_exact_protocol_fields(
            child_environment,
            {
                "allowlisted_keys": [
                    "LANG",
                    "LC_ALL",
                    "PATH",
                    "TZ",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONHASHSEED",
                    "PYTHONIOENCODING",
                    "PYTHONNOUSERSITE",
                    "PYTHONPATH",
                    "PYTHONSAFEPATH",
                    "PYTHONUTF8",
                ],
                "fixed_values": {
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "os.defpath",
                    "TZ": "UTC",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONSAFEPATH": "1",
                    "PYTHONUTF8": "1",
                },
                "site_processing": False,
            },
            label="child environment",
        )
        worker_capability = _required_mapping(runtime, "worker_capability")
        if set(worker_capability) != {
            "transport_family",
            "transport_type",
            "wire_format",
            "secret_bytes",
            "payload_max_bytes",
            "authenticator_bytes",
            "authentication_algorithm",
            "authentication_domain_hex",
            "payload_encoding",
            "payload_schema",
            "payload_fields",
            "path_fields",
            "binding_rule",
            "marker_commitment",
            "parent_half_close",
            "acknowledgement_bytes",
            "acknowledgement_algorithm",
            "acknowledgement_domain_hex",
            "acknowledgement_rule",
            "shared_exchange_timeout_seconds",
            "deadline_origin",
            "local_assurance_boundary",
        }:
            raise Q1AuditError("canonical protocol worker capability key set differs")
        _require_exact_protocol_fields(
            worker_capability,
            {
                "transport_family": "AF_UNIX",
                "transport_type": "SOCK_STREAM",
                "wire_format": (
                    "4-byte unsigned big-endian payload length || 32-byte secret || canonical payload || "
                    "32-byte HMAC-SHA256 authenticator"
                ),
                "secret_bytes": 32,
                "payload_max_bytes": 65_536,
                "authenticator_bytes": 32,
                "authentication_algorithm": "HMAC-SHA256",
                "authentication_domain_hex": (
                    "70726f73706563742e776d3030322e6163746976652d6163717569736974696f6e2e71312d776f"
                    "726b65722d6361706162696c6974792e763100"
                ),
                "payload_encoding": "finite sorted compact canonical ASCII JSON",
                "payload_schema": "prospect.wm002.active-acquisition.q1-worker-capability.v1",
                "payload_fields": [
                    "arm",
                    "child_pid",
                    "master",
                    "parent_pid",
                    "paths",
                    "role",
                    "run_identity",
                    "schema",
                ],
                "path_fields": [
                    "attempt_registry_directory",
                    "entry_report_path",
                    "execution_root",
                    "frame_path",
                    "incomplete_directory",
                    "index_path",
                    "master_directory",
                    "output_path",
                    "prospective_review_path",
                    "q0_report_path",
                    "raw_trace_path",
                    "secret_salt_path",
                ],
                "parent_half_close": "socket.SHUT_WR after the complete authenticated wire",
                "acknowledgement_bytes": 32,
                "acknowledgement_algorithm": "HMAC-SHA256",
                "acknowledgement_domain_hex": (
                    "70726f73706563742e776d3030322e6163746976652d6163717569736974696f6e2e71312d776f"
                    "726b65722d6361706162696c6974792d61636b2e763100"
                ),
                "shared_exchange_timeout_seconds": 10.0,
            },
            label="worker capability",
        )
        worker_capture = _required_mapping(runtime, "worker_capture")
        if set(worker_capture) != {
            "stdout_max_bytes",
            "stderr_max_bytes",
            "bounded_memory_tail_bytes",
            "capture_finish_timeout_seconds",
            "rule",
        }:
            raise Q1AuditError("canonical protocol worker capture key set differs")
        _require_exact_protocol_fields(
            worker_capture,
            {
                "stdout_max_bytes": 0,
                "stderr_max_bytes": 65_536,
                "bounded_memory_tail_bytes": 2_000,
                "capture_finish_timeout_seconds": 2.0,
            },
            label="worker capture",
        )
        filesystem_custody = _required_mapping(runtime, "filesystem_custody")
        if set(filesystem_custody) != {
            "directory_mode",
            "private_artifact_file_mode",
            "public_report_file_mode",
            "regular_file_link_count",
            "directory_identity_fields",
            "authority_roots",
            "authority_root_rule",
            "publication_rule",
        }:
            raise Q1AuditError("canonical protocol filesystem custody key set differs")
        _require_exact_protocol_fields(
            filesystem_custody,
            {
                "directory_mode": "0700",
                "private_artifact_file_mode": "0600",
                "public_report_file_mode": "0644",
                "regular_file_link_count": 1,
                "directory_identity_fields": [
                    "canonical_path",
                    "file_type",
                    "mode",
                    "st_dev",
                    "st_gid",
                    "st_ino",
                    "st_uid",
                ],
                "authority_roots": ["execution_root", "attempt_registry_directory"],
            },
            label="filesystem custody",
        )
        external_inputs = _required_mapping(runtime, "external_inputs")
        if set(external_inputs) != {
            "document_max_bytes",
            "secret_salt_min_bytes",
            "secret_salt_max_bytes",
            "secret_salt_mode",
            "regular_file_link_count",
            "read_rule",
            "json_rule",
        }:
            raise Q1AuditError("canonical protocol external input key set differs")
        _require_exact_protocol_fields(
            external_inputs,
            {
                "document_max_bytes": _MAX_SMALL_DOCUMENT_BYTES,
                "secret_salt_min_bytes": 32,
                "secret_salt_max_bytes": _MAX_SMALL_DOCUMENT_BYTES,
                "secret_salt_mode": "0600",
                "regular_file_link_count": 1,
            },
            label="external input",
        )
        identity_counter = _required_mapping(runtime, "identity_counter")
        if set(identity_counter) != {
            "initial_value",
            "checkpoint_preterminal_value",
            "checkpoint_decode_max_next_counter",
            "rule",
        }:
            raise Q1AuditError("canonical protocol identity counter key set differs")
        _require_exact_protocol_fields(
            identity_counter,
            {
                "initial_value": _INITIAL_IDENTITY_COUNTER,
                "checkpoint_preterminal_value": _PRETERMINAL_IDENTITY_COUNTER,
                "checkpoint_decode_max_next_counter": 64,
            },
            label="identity counter",
        )
        analysis = _required_mapping(protocol, "analysis")
        if not _same_number(analysis.get("student_t_critical_df3"), T_CRITICAL_DF3):
            raise Q1AuditError("canonical protocol df=3 critical value differs")
        q0_binding = _required_mapping(protocol, "q0_binding")
        if q0_binding.get("report_sha256") != Q0_REPORT_SHA256:
            raise Q1AuditError("canonical protocol Q0 report binding differs")
        if _normalized_protocol_contract_sha256(protocol) != _NORMALIZED_Q1_PROTOCOL_SHA256:
            raise Q1AuditError("canonical normalized whole-document protocol contract differs")
    except Exception as error:
        violations.add("Q1-K0", f"canonical protocol rejected: {type(error).__name__}:{error}")

    try:
        _validate_loaded_source_origins()
        manifest, implementation_digest = implementation_manifest(Q1_AUDIT_IMPLEMENTATION_PATHS)
        manifest_rows = [row.as_dict() for row in manifest]
    except Exception as error:
        violations.add("Q1-K0", f"selected-source closure rejected: {type(error).__name__}:{error}")

    try:
        q0_payload = _read_regular_file(q0_report_path, label="canonical Q0 report")
        if sha256_bytes(q0_payload) != Q0_REPORT_SHA256:
            raise Q1AuditError("Q0 report digest differs from the accepted canonical report")
        q0 = json.loads(
            q0_payload.decode("utf-8"),
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
        if not isinstance(q0, Mapping):
            raise Q1AuditError("Q0 report is not an object")
        if q0_payload != canonical_json_bytes(q0, newline=True):
            raise Q1AuditError("Q0 report is not one canonical JSON document")
        expected_q0 = {
            "schema": "prospect.wm002.active-acquisition.q0-qualification.v1",
            "passed": True,
            "claim_eligible": False,
            "formal_authorized": False,
            "environment_interactions": 0,
        }
        for field, expected in expected_q0.items():
            if q0.get(field) != expected:
                raise Q1AuditError(f"canonical Q0 report boundary mismatch:{field}")
    except Exception as error:
        violations.add("Q1-K0", f"canonical Q0 report rejected: {type(error).__name__}:{error}")

    try:
        prospective_review_digest = _load_validated_prospective_review(
            prospective_review_path,
            protocol_sha256=protocol_digest,
            implementation_sha256=implementation_digest,
            reviewed_source_count=len(manifest_rows),
        )
    except Exception as error:
        violations.add("Q1-K0", f"prospective review rejected: {type(error).__name__}:{error}")

    try:
        entry_payload = _read_regular_file(entry_report_path, label="Q1 entry qualification")
        entry_digest = sha256_bytes(entry_payload)
        entry = json.loads(
            entry_payload.decode("utf-8"),
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
        if not isinstance(entry, Mapping):
            raise Q1AuditError("entry qualification report is not an object")
        if entry_payload != canonical_json_bytes(entry, newline=True):
            raise Q1AuditError("entry qualification is not one canonical JSON document")
        _validate_entry_schema(entry)
        expected_entry = {
            "schema": "prospect.wm002.active-acquisition.q1-entry-qualification.v1",
            "dependency_versions": {"jsonschema": _JSONSCHEMA_VERSION},
            "protocol_version": Q1_PROTOCOL_VERSION,
            "protocol_sha256": protocol_digest,
            "implementation_sha256": implementation_digest,
            "q0_report_sha256": Q0_REPORT_SHA256,
            "q0_protocol_sha256": Q0_PROTOCOL_SHA256,
            "q0_implementation_sha256": Q0_IMPLEMENTATION_SHA256,
            "salt_commitment_sha256": salt_commitment,
            "passed": True,
            "claim_eligible": False,
            "formal_authorized": False,
            "q1_environment_interactions": 0,
            "q1_private_draws": 0,
            "synthetic_development_interactions": 19,
        }
        for field, expected in expected_entry.items():
            if entry.get(field) != expected:
                raise Q1AuditError(f"entry qualification boundary mismatch:{field}")
        if package_version("jsonschema") != _JSONSCHEMA_VERSION:
            raise Q1AuditError("auditor jsonschema runtime differs from the entry-pinned version")
        if entry.get("implementation_manifest") != manifest_rows:
            raise Q1AuditError("entry qualification selected-source manifest differs from recomputation")
        actual_schema_sha256 = {
            name: _stream_file_sha256(path, label=f"selected {name} schema")
            for name, path in sorted(Q1_SCHEMA_PATHS.items())
        }
        if entry.get("schema_sha256") != actual_schema_sha256:
            raise Q1AuditError("entry qualification schema digests differ from selected schema bytes")
        _require_entry_prospective_review_binding(entry, prospective_review_digest)
        checks = entry.get("checks")
        if (
            not isinstance(checks, list)
            or tuple(row.get("name") if isinstance(row, Mapping) else None for row in checks) != _ENTRY_CHECK_ORDER
        ):
            raise Q1AuditError("entry qualification check names/order differ from the frozen ten checks")
        for row in checks:
            if not isinstance(row, Mapping):
                raise Q1AuditError("entry qualification check is not an object")
            check_violations = row.get("violations")
            if not isinstance(check_violations, list):
                raise Q1AuditError("entry qualification check violations are not an array")
            if row.get("passed") is not (not check_violations):
                raise Q1AuditError(f"entry qualification check coherence differs:{row.get('name')}")
            if row.get("passed") is not True:
                raise Q1AuditError(f"entry qualification check did not pass:{row.get('name')}")
        if entry.get("passed") is not all(row.get("passed") is True for row in checks):
            raise Q1AuditError("entry qualification overall/check coherence differs")
        resources = _required_mapping(entry, "resource_preflight")
        if resources.get("passed") is not True or resources.get("passed") is not checks[7].get("passed"):
            raise Q1AuditError("entry resource preflight is not coherently passing")
        positive_samples = (
            "raw_trace_max_bytes",
            "private_audit_max_bytes",
            "checkpoint_index_max_bytes",
            "checkpoint_frame_max_bytes",
            "restored_trace_max_bytes",
        )
        for field in positive_samples:
            value = resources.get(field)
            if type(value) is not int or not 0 < value <= _MAX_CHECKPOINT_FRAME_BYTES:
                raise Q1AuditError(f"entry resource sample is absent or beyond auditor bound:{field}")
        if (
            resources.get("max_restore_concurrency") != 4
            or resources.get("sampled_arms") != len(ARM_ORDER)
            or resources.get("probe_duration_under_30_seconds") is not True
            or type(resources.get("estimated_canonical_bytes")) is not int
            or cast(int, resources.get("estimated_canonical_bytes")) <= 0
            or type(resources.get("required_free_bytes")) is not int
            or cast(int, resources.get("required_free_bytes")) <= 0
        ):
            raise Q1AuditError("entry resource preflight differs from the frozen complete contract")
        declared_execution_root = _parse_audit_directory_identity(
            resources.get("execution_root"),
            label="entry execution_root",
        )
        declared_attempt_registry = _parse_audit_directory_identity(
            resources.get("attempt_registry_directory"),
            label="entry attempt_registry_directory",
        )
        execution_root = _require_entry_bound_directory_identity(
            declared_execution_root,
            label="entry execution root",
        )
        attempt_registry_directory = _require_entry_bound_directory_identity(
            declared_attempt_registry,
            label="entry attempt registry",
        )
        same_directory = execution_root.canonical_path == attempt_registry_directory.canonical_path or (
            execution_root.st_dev,
            execution_root.st_ino,
        ) == (attempt_registry_directory.st_dev, attempt_registry_directory.st_ino)
        execution_path = Path(execution_root.canonical_path)
        registry_path = Path(attempt_registry_directory.canonical_path)
        if same_directory:
            raise Q1AuditError("entry-bound execution root and attempt registry are not distinct")
        if execution_path in registry_path.parents or registry_path in execution_path.parents:
            raise Q1AuditError("entry-bound execution root and attempt registry are nested")
    except Exception as error:
        violations.add("Q1-K0", f"entry qualification rejected: {type(error).__name__}:{error}")

    run_sha256 = canonical_sha256(
        {
            "entry_qualification_sha256": entry_digest,
            "implementation_sha256": implementation_digest,
            "protocol_sha256": protocol_digest,
            "protocol_version": Q1_PROTOCOL_VERSION,
            "q0_report_sha256": Q0_REPORT_SHA256,
            "salt_commitment_sha256": salt_commitment,
            "schema": _RUN_IDENTITY_SCHEMA,
        }
    )
    return _AuditBindings(
        protocol_sha256=protocol_digest,
        implementation_sha256=implementation_digest,
        q0_report_sha256=Q0_REPORT_SHA256,
        entry_qualification_sha256=entry_digest,
        prospective_review_sha256=prospective_review_digest,
        salt_commitment_sha256=salt_commitment,
        run_sha256=run_sha256,
        run_id=f"wm002-q1-{run_sha256}",
        attempt_id=f"wm002-q1-{run_sha256}-attempt-0001",
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
    )


def audit_q1_directory(
    output_directory: Path,
    *,
    secret_salt_path: Path,
    q0_report_path: Path,
    entry_report_path: Path,
    prospective_review_path: Path,
    protocol_path: Path = Q1_PROTOCOL_PATH,
    attempt_marker_path: Path,
) -> dict[str, object]:
    """Independently audit one complete canonical Q1 output directory.

    External bindings are accepted only as paths and reopened by the auditor;
    caller-supplied digests are never trusted.  Malformed or missing evidence
    yields a schema-valid failed audit artifact instead of a partial success.
    """

    directory = Path(output_directory)
    violations = _Violations()
    secret_salt = b""
    schedule: _IndependentSeedSchedule | None = None
    privacy_scanner: PrivatePrefixScanner | None = None
    try:
        secret_salt = _read_regular_file(
            secret_salt_path,
            label="private Q1 salt",
            private=True,
        )
        schedule = _IndependentSeedSchedule(secret_salt)
    except Exception as error:
        violations.add("Q1-K0", f"secret salt rejected: {type(error).__name__}:{error}")
    if schedule is not None:
        try:
            privacy_scanner = PrivatePrefixScanner.from_private_values(_global_private_values(schedule, secret_salt))
        except Exception as error:
            violations.add("Q1-K1", f"global private-prefix index rejected: {type(error).__name__}")
    expected_salt_commitment = schedule.salt_commitment_sha256 if schedule is not None else _ZERO_SHA256

    bindings = _resolve_external_bindings(
        protocol_path=protocol_path,
        q0_report_path=q0_report_path,
        entry_report_path=entry_report_path,
        prospective_review_path=prospective_review_path,
        salt_commitment=expected_salt_commitment,
        violations=violations,
    )

    try:
        execution_identity, registry_identity = _require_audit_binding_directories(bindings)
        if directory.parent.resolve() != Path(execution_identity.canonical_path):
            raise Q1AuditError("Q1 output directory is outside the entry-bound execution root")
        if attempt_marker_path.parent.resolve() != Path(registry_identity.canonical_path):
            raise Q1AuditError("attempt marker is outside the entry-bound attempt registry")
        _validate_artifact_directory(directory)
    except Exception as error:
        violations.add("Q1-K0", f"artifact directory rejected: {type(error).__name__}:{error}")
    paths = _artifact_paths(directory)
    digests = _hash_artifacts(paths, violations)
    attempt_marker_sha256, worker_capability_sha256 = _validate_attempt_marker(
        attempt_marker_path,
        bindings,
        digests,
        violations,
    )

    expected_binding: QualificationBinding | None = None
    try:
        expected_binding = QualificationBinding(
            protocol_version=Q1_PROTOCOL_VERSION,
            protocol_sha256=bindings.protocol_sha256,
            implementation_sha256=bindings.implementation_sha256,
            q0_report_sha256=bindings.q0_report_sha256,
            entry_qualification_sha256=bindings.entry_qualification_sha256,
            run_id=bindings.run_id,
            attempt_id=bindings.attempt_id,
            salt_commitment_sha256=expected_salt_commitment,
        )
    except Exception as error:
        violations.add("Q1-K0", f"checkpoint binding rejected: {type(error).__name__}:{error}")

    streamed = _stream_audit_evidence(
        paths=paths,
        committed_digests=digests,
        bindings=bindings,
        schedule=schedule,
        privacy_scanner=privacy_scanner,
        expected_binding=expected_binding,
        violations=violations,
    )
    arm_mean_rows, arm_mean_values = _recompute_arm_means(streamed.returns, violations)
    comparisons = _comparison_rows(arm_mean_values)
    counts = dict(streamed.counts)
    _validate_producer_aggregate(
        streamed.aggregate,
        counts=counts,
        arm_means=arm_mean_rows,
        comparisons=comparisons,
        violations=violations,
    )
    try:
        _require_audit_binding_directories(bindings)
    except Exception as error:
        violations.add("Q1-K0", f"entry-bound directory closing check rejected: {type(error).__name__}:{error}")
    if not all(bool(row["passed"]) for row in comparisons):
        violations.add("Q1-K4", "one or more conjunctive paired comparisons failed")
    gates = []
    accepted_prefix: list[str] = []
    prefix_open = True
    for gate in GATE_ORDER:
        gate_violations = violations.rows(gate)
        passed = not gate_violations
        gates.append(
            {
                "gate": gate,
                "passed": passed,
                "violations": list(gate_violations),
            }
        )
        if prefix_open and passed:
            accepted_prefix.append(gate)
        else:
            prefix_open = False

    passed = len(accepted_prefix) == len(GATE_ORDER)
    artifact = {
        "attempt_id": bindings.attempt_id,
        "attempt_marker_sha256": attempt_marker_sha256,
        "worker_capability_sha256": worker_capability_sha256,
        "run_id": bindings.run_id,
        "accepted_prefix": accepted_prefix,
        "checkpoint_frames_sha256": digests["checkpoint_frames"],
        "checkpoint_index_sha256": digests["checkpoint_index"],
        "claim_eligible": False,
        "comparisons": comparisons,
        "counts": counts,
        "entry_qualification_sha256": bindings.entry_qualification_sha256,
        "formal_authorized": False,
        "gates": gates,
        "implementation_sha256": bindings.implementation_sha256,
        "independent_recomputation": True,
        "passed": passed,
        "private_audit_sha256": digests["private_audit"],
        "producer_aggregate_sha256": digests["producer_aggregate"],
        "prospective_review_sha256": bindings.prospective_review_sha256,
        "protocol_sha256": bindings.protocol_sha256,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "q0_report_sha256": bindings.q0_report_sha256,
        "raw_trace_sha256": digests["raw_trace"],
        "restored_trace_sha256": digests["restored_trace"],
        "salt_commitment_sha256": expected_salt_commitment,
        "schema": AUDIT_SCHEMA,
        "scope_limitations": [
            (
                "Q1 is permanently claim-ineligible and qualifies only this exact "
                "known-likelihood, two-step hidden-actuator runtime chain."
            ),
            (
                "This audit is not evidence of learned uncertainty, general active "
                "learning, transfer, continual improvement, or formal capability."
            ),
            (
                "The producer aggregate is a non-authoritative schema, digest, and semantic "
                "cross-check; every reported statistic and gate is recomputed from primitive traces."
            ),
            (
                "Checkpoint acquisition graph, model, and counter semantics plus terminal candidates, "
                "actions, outcomes, and exposed IDs are independently decoded or derived. The terminal "
                "domain graph is not serialized, so fresh-process execution is source-bound procedural "
                "evidence with row, counter, and PID parity—not external terminal re-execution attestation."
            ),
            (
                "Producer/restorer PID separation is local process evidence and not cryptographic "
                "process-origin attestation; one restorer PID may legally serve multiple lanes."
            ),
            (
                "Entry-bound canonical path, device, inode, owner, group, type, and mode checks are "
                "local filesystem snapshot evidence, not cryptographic host identity or protection "
                "against a privileged namespace adversary."
            ),
        ],
        "verdict": (
            "PASS: exact claim-ineligible Q1 runtime qualification gates K0-K5 passed."
            if passed
            else f"FAIL: accepted ordered prefix is {accepted_prefix!r}; later gates cannot rescue it."
        ),
    }
    _require_completed_audit_private_clean(artifact, privacy_scanner)
    try:
        validate_artifact("audit_output", artifact)
        _validate_audit_semantics(artifact)
    except Exception as error:
        raise Q1AuditError(
            f"constructed audit output violates its frozen schema or semantics ({type(error).__name__})"
        ) from error
    return artifact


def development_audit_artifact_sample() -> dict[str, object]:
    """Return a result-free schema sample for entry qualification mutation tests."""

    comparisons = [
        {
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
            "control_arm": control,
            "master_differences": [0.0, 0.0, 0.0, 0.0],
            "mean_difference": 0.0,
            "passed": False,
        }
        for control in CONTROL_ARMS
    ]
    return {
        "accepted_prefix": [],
        "checkpoint_frames_sha256": _ZERO_SHA256,
        "checkpoint_index_sha256": _ZERO_SHA256,
        "claim_eligible": False,
        "comparisons": comparisons,
        "counts": {
            "acquisition_updates": 0,
            "arms": 0,
            "checkpoint_frames": 0,
            "environment_steps": 0,
            "episodes": 0,
            "masters": 0,
            "private_rows": 0,
            "raw_rows": 0,
            "restored_rows": 0,
            "terminal_updates": 0,
            "transitions": 0,
        },
        "entry_qualification_sha256": _ZERO_SHA256,
        "attempt_id": f"wm002-q1-{_ZERO_SHA256}-attempt-0001",
        "attempt_marker_sha256": _ZERO_SHA256,
        "worker_capability_sha256": _ZERO_SHA256,
        "run_id": f"wm002-q1-{_ZERO_SHA256}",
        "formal_authorized": False,
        "gates": [
            {
                "gate": gate,
                "passed": False,
                "violations": ["synthetic result-free schema sample; no Q1 audit executed"],
            }
            for gate in GATE_ORDER
        ],
        "implementation_sha256": _ZERO_SHA256,
        "independent_recomputation": True,
        "passed": False,
        "private_audit_sha256": _ZERO_SHA256,
        "producer_aggregate_sha256": _ZERO_SHA256,
        "prospective_review_sha256": _ZERO_SHA256,
        "protocol_sha256": _ZERO_SHA256,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "q0_report_sha256": Q0_REPORT_SHA256,
        "raw_trace_sha256": _ZERO_SHA256,
        "restored_trace_sha256": _ZERO_SHA256,
        "salt_commitment_sha256": _ZERO_SHA256,
        "schema": AUDIT_SCHEMA,
        "scope_limitations": ["Synthetic schema sample only; it contains no Q1 outcome."],
        "verdict": "NOT RUN: synthetic result-free audit-output schema sample.",
    }


def _validate_audit_semantics(artifact: Mapping[str, object]) -> None:
    """Reject schema-valid but internally contradictory audit conclusions."""

    if artifact.get("claim_eligible") is not False or artifact.get("formal_authorized") is not False:
        raise Q1AuditError("audit output must remain claim-ineligible and formally unauthorized")
    if artifact.get("independent_recomputation") is not True:
        raise Q1AuditError("audit output must identify independent recomputation")
    if not _is_sha256(artifact.get("prospective_review_sha256")):
        raise Q1AuditError("audit output lacks its prospective-review digest binding")
    worker_capability_sha256 = artifact.get("worker_capability_sha256")
    if not _is_sha256(worker_capability_sha256):
        raise Q1AuditError("audit output lacks its worker capability commitment")
    if artifact.get("passed") is True and worker_capability_sha256 == _ZERO_SHA256:
        raise Q1AuditError("passing audit output has a zero worker capability commitment")
    run_id = artifact.get("run_id")
    if not isinstance(run_id, str) or artifact.get("attempt_id") != f"{run_id}-attempt-0001":
        raise Q1AuditError("audit output attempt identity does not continue its run identity")

    gates = artifact.get("gates")
    if (
        not isinstance(gates, list)
        or tuple(row.get("gate") if isinstance(row, Mapping) else None for row in gates) != GATE_ORDER
    ):
        raise Q1AuditError("audit gates differ from the exact unique frozen order")
    expected_prefix: list[str] = []
    prefix_open = True
    for gate, row in zip(GATE_ORDER, gates, strict=True):
        if not isinstance(row, Mapping):
            raise Q1AuditError(f"audit gate is not an object:{gate}")
        gate_violations = row.get("violations")
        if not isinstance(gate_violations, list):
            raise Q1AuditError(f"audit gate violations are not an array:{gate}")
        expected_passed = not gate_violations
        if row.get("passed") is not expected_passed:
            raise Q1AuditError(f"audit gate pass/violation coherence differs:{gate}")
        if prefix_open and expected_passed:
            expected_prefix.append(gate)
        else:
            prefix_open = False
    if artifact.get("accepted_prefix") != expected_prefix:
        raise Q1AuditError("accepted_prefix is not the exact leading sequence of passing gates")
    expected_overall = all(row.get("passed") is True for row in gates if isinstance(row, Mapping))
    if artifact.get("passed") is not expected_overall:
        raise Q1AuditError("audit overall result differs from the conjunction of ordered gates")

    comparisons = artifact.get("comparisons")
    if (
        not isinstance(comparisons, list)
        or tuple(row.get("control_arm") if isinstance(row, Mapping) else None for row in comparisons) != CONTROL_ARMS
    ):
        raise Q1AuditError("audit comparisons differ from the exact unique control-arm order")
    for control, row in zip(CONTROL_ARMS, comparisons, strict=True):
        if not isinstance(row, Mapping):
            raise Q1AuditError(f"audit comparison is not an object:{control}")
        differences = row.get("master_differences")
        if not isinstance(differences, list) or len(differences) != MASTER_COUNT:
            raise Q1AuditError(f"audit comparison lacks four master differences:{control}")
        values: list[float] = []
        for value in differences:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Q1AuditError(f"audit comparison contains a nonfinite/noncanonical difference:{control}")
            values.append(float(value))
        mean = math.fsum(values) / MASTER_COUNT
        variance = math.fsum((value - mean) ** 2 for value in values) / (MASTER_COUNT - 1)
        margin = T_CRITICAL_DF3 * math.sqrt(variance) / math.sqrt(MASTER_COUNT)
        expected_numbers = {
            "mean_difference": mean,
            "ci95_lower": mean - margin,
            "ci95_upper": mean + margin,
        }
        for field, expected in expected_numbers.items():
            if not _same_number(row.get(field), expected):
                raise Q1AuditError(f"audit comparison statistic differs from its master differences:{control}:{field}")
        expected_comparison_pass = mean > 0.0 and mean - margin > 0.0
        if row.get("passed") is not expected_comparison_pass:
            raise Q1AuditError(f"audit comparison pass value differs from mean/lower-bound conjunction:{control}")

    if gates[4].get("passed") is True and not all(
        isinstance(row, Mapping) and row.get("passed") is True for row in comparisons
    ):
        raise Q1AuditError("passing Q1-K4 gate contradicts one or more failed comparisons")

    if expected_overall:
        expected_counts = {
            "acquisition_updates": EXPECTED_EPISODES,
            "arms": len(ARM_ORDER),
            "checkpoint_frames": EXPECTED_EPISODES,
            "environment_steps": EXPECTED_EPISODES * 2,
            "episodes": EXPECTED_EPISODES,
            "masters": MASTER_COUNT,
            "private_rows": EXPECTED_EPISODES,
            "raw_rows": EXPECTED_EPISODES,
            "restored_rows": EXPECTED_EPISODES,
            "terminal_updates": 0,
            "transitions": EXPECTED_EPISODES * 2,
        }
        if artifact.get("counts") != expected_counts:
            raise Q1AuditError("passing audit counts differ from the exact frozen budget")


def _require_completed_audit_private_clean(
    artifact: Mapping[str, object],
    privacy_scanner: PrivatePrefixScanner | None,
) -> None:
    """Refuse to return or publish a completed artifact containing private material."""

    if privacy_scanner is None:
        return
    try:
        leaks = privacy_scanner.scan(artifact)
    except Exception as error:
        raise Q1AuditError(f"completed audit private-prefix scan failed ({type(error).__name__})") from error
    if leaks:
        raise Q1AuditError("completed audit artifact contains private-prefix material")


def _directory_identity_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
    )


def _open_verified_audit_directory(path: Path, *, label: str) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Q1AuditError(f"{label} cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise Q1AuditError(f"{label} must be a non-symlink directory with exact mode 0700")
        if metadata.st_uid != os.geteuid():
            raise Q1AuditError(f"{label} must be owned by the effective user")
        path_metadata = os.stat(path, follow_symlinks=False)
        identity = _directory_identity_signature(metadata)
        if _directory_identity_signature(path_metadata) != identity:
            raise Q1AuditError(f"{label} path differs from its opened descriptor")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _require_audit_directory_identity(
    path: Path,
    descriptor: int,
    expected_identity: tuple[int, ...],
    *,
    label: str,
) -> None:
    metadata = os.fstat(descriptor)
    path_metadata = os.stat(path, follow_symlinks=False)
    if (
        _directory_identity_signature(metadata) != expected_identity
        or _directory_identity_signature(path_metadata) != expected_identity
    ):
        raise Q1AuditError(f"{label} identity changed")


def _write_exclusive_durable(
    path: Path,
    payload: bytes,
    *,
    expected_parent_identity: tuple[int, ...],
) -> None:
    """Publish new audit evidence atomically through one identity-bound directory FD."""

    target = Path(path)
    directory_descriptor, parent_identity = _open_verified_audit_directory(
        target.parent,
        label="audit output parent",
    )
    if parent_identity != expected_parent_identity:
        os.close(directory_descriptor)
        raise Q1AuditError("audit output parent identity changed after disjointness validation")
    descriptor: int | None = None
    target_descriptor: int | None = None
    temporary_name = f".{target.name}.tmp-{os.getpid()}"
    created = False
    try:
        try:
            os.stat(target.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise Q1AuditError("audit output path already exists (including a symlink)")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary_name, flags, 0o644, dir_fd=directory_descriptor)
        created = True
        os.fchmod(descriptor, 0o644)
        created_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(created_metadata.st_mode)
            or created_metadata.st_nlink != 1
            or stat.S_IMODE(created_metadata.st_mode) != 0o644
        ):
            raise Q1AuditError("temporary audit output file custody differs")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Q1AuditError("audit output write made no progress")
            view = view[written:]
        if os.fstat(descriptor).st_size != len(payload):
            raise Q1AuditError("audit output size differs after writing")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _require_audit_directory_identity(
            target.parent,
            directory_descriptor,
            expected_parent_identity,
            label="audit output parent",
        )
        os.link(
            temporary_name,
            target.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        created = False
        read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        target_descriptor = os.open(target.name, read_flags, dir_fd=directory_descriptor)
        target_metadata = os.fstat(target_descriptor)
        target_path_metadata = os.stat(target.name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(target_metadata.st_mode)
            or target_metadata.st_nlink != 1
            or stat.S_IMODE(target_metadata.st_mode) != 0o644
            or target_metadata.st_size != len(payload)
            or _regular_metadata_signature(target_metadata) != _regular_metadata_signature(target_path_metadata)
        ):
            raise Q1AuditError("published audit output file custody differs")
        os.fsync(directory_descriptor)
        _require_audit_directory_identity(
            target.parent,
            directory_descriptor,
            expected_parent_identity,
            label="audit output parent",
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if target_descriptor is not None:
            os.close(target_descriptor)
        if created:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        os.close(directory_descriptor)


def _require_disjoint_audit_output_path(
    path: Path,
    *,
    audited_directory: Path,
    protected_input_paths: Sequence[Path],
) -> tuple[int, ...]:
    """Reject output paths inside or aliasing evidence and bind its parent."""

    target = Path(path)
    try:
        audited = Path(audited_directory).resolve(strict=False)
        target_parent = target.parent.resolve(strict=True)
        target_resolved = target.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise Q1AuditError("audit output path cannot be resolved safely") from error
    for candidate in (target_parent, target_resolved):
        try:
            candidate.relative_to(audited)
        except ValueError:
            pass
        else:
            raise Q1AuditError("audit output must be outside the audited artifact directory")

    for protected_path in protected_input_paths:
        try:
            protected_resolved = Path(protected_path).resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise Q1AuditError("protected audit input path cannot be resolved safely") from error
        if target_resolved == protected_resolved:
            raise Q1AuditError("audit output aliases a protected audit input")
        try:
            aliases = os.path.samefile(target, protected_path)
        except OSError:
            aliases = False
        if aliases:
            raise Q1AuditError("audit output aliases a protected audit input")

    descriptor, identity = _open_verified_audit_directory(
        target_parent,
        label="audit output parent",
    )
    os.close(descriptor)
    return identity


def write_audit_artifact(
    path: Path,
    artifact: Mapping[str, object],
    *,
    secret_salt_path: Path,
    audited_directory: Path,
    protected_input_paths: Sequence[Path] = (),
) -> None:
    """Private-scan and write one prevalidated canonical audit document."""

    parent_identity = _require_disjoint_audit_output_path(
        path,
        audited_directory=audited_directory,
        protected_input_paths=(secret_salt_path, *protected_input_paths),
    )

    try:
        secret_salt = _read_regular_file(
            secret_salt_path,
            label="private Q1 salt for audit publication",
            private=True,
        )
        schedule = _IndependentSeedSchedule(secret_salt)
        privacy_scanner = PrivatePrefixScanner.from_private_values(_global_private_values(schedule, secret_salt))
    except Exception as error:
        raise Q1AuditError(f"audit publication private-prefix setup failed ({type(error).__name__})") from error
    _require_completed_audit_private_clean(artifact, privacy_scanner)

    try:
        validate_artifact("audit_output", artifact)
        _validate_audit_semantics(artifact)
    except Exception as error:
        raise Q1AuditError(f"audit output validation failed ({type(error).__name__})") from error
    _write_exclusive_durable(
        path,
        canonical_json_bytes(artifact, newline=True),
        expected_parent_identity=parent_identity,
    )


def _regular_metadata_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_path_matches_open_file(
    path: Path,
    metadata: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        path_metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise Q1AuditError(f"{label} path cannot be revalidated") from error
    if _regular_metadata_signature(path_metadata) != _regular_metadata_signature(metadata):
        raise Q1AuditError(f"{label} path differs from its opened descriptor")


def _open_regular_descriptor(path: Path, *, label: str, private: bool = False) -> int:
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise Q1AuditError(f"{label} is not a regular file")
        if metadata.st_nlink != 1:
            raise Q1AuditError(f"{label} must have exactly one hard link")
        if private and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise Q1AuditError(f"{label} must have exact private mode 0600")
        _require_path_matches_open_file(candidate, metadata, label=label)
        return descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise Q1AuditError(f"cannot open {label}: {type(error).__name__}:{error}") from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _read_regular_file(
    path: Path,
    *,
    label: str,
    private: bool = False,
    max_bytes: int | None = _MAX_SMALL_DOCUMENT_BYTES,
) -> bytes:
    """Read one stable opened regular file without following a final symlink."""

    descriptor: int | None = None
    try:
        descriptor = _open_regular_descriptor(path, label=label, private=private)
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise Q1AuditError(f"{label} exceeds the bounded {max_bytes}-byte document limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _regular_metadata_signature(before) != _regular_metadata_signature(after) or total != after.st_size:
            raise Q1AuditError(f"{label} changed during its descriptor read")
        _require_path_matches_open_file(path, after, label=label)
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stream_file_sha256(path: Path, *, label: str, private: bool = False) -> str:
    descriptor = _open_regular_descriptor(path, label=label, private=private)
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if _regular_metadata_signature(before) != _regular_metadata_signature(after) or total != after.st_size:
            raise Q1AuditError(f"{label} changed while hashing")
        _require_path_matches_open_file(path, after, label=label)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _validate_artifact_directory(directory: Path) -> None:
    try:
        metadata = directory.lstat()
    except OSError as error:
        raise Q1AuditError(f"cannot inspect Q1 artifact directory: {error}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise Q1AuditError("Q1 artifact directory is not a non-symlink directory")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise Q1AuditError("Q1 artifact directory must have exact private mode 0700")
    with os.scandir(directory) as iterator:
        entries = tuple(iterator)
    names = {entry.name for entry in entries}
    if names != _ARTIFACT_FILENAMES or len(entries) != len(_ARTIFACT_FILENAMES):
        missing = sorted(_ARTIFACT_FILENAMES - names)
        extra = sorted(names - _ARTIFACT_FILENAMES)
        raise Q1AuditError(f"Q1 publication set differs from exact six files (missing={missing},extra={extra})")
    for entry in entries:
        entry_metadata = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(entry_metadata.st_mode):
            raise Q1AuditError(f"Q1 publication entry is not a non-symlink regular file:{entry.name}")
        if entry_metadata.st_nlink != 1:
            raise Q1AuditError(f"Q1 publication entry must have exactly one hard link:{entry.name}")
        if stat.S_IMODE(entry_metadata.st_mode) != 0o600:
            # The protocol's publication rule covers all six artifacts, not only
            # the private sidecar.
            raise Q1AuditError(f"Q1 publication entry must have exact private mode 0600:{entry.name}")


def _artifact_paths(directory: Path) -> dict[str, Path]:
    return {
        "raw_trace": directory / RAW_TRACE_FILENAME,
        "private_audit": directory / PRIVATE_AUDIT_FILENAME,
        "checkpoint_index": directory / CHECKPOINT_INDEX_FILENAME,
        "checkpoint_frames": directory / CHECKPOINT_FRAMES_FILENAME,
        "restored_trace": directory / RESTORED_TRACE_FILENAME,
        "producer_aggregate": directory / PRODUCER_AGGREGATE_FILENAME,
    }


def _hash_artifacts(paths: Mapping[str, Path], violations: _Violations) -> dict[str, str]:
    digests: dict[str, str] = {}
    for name, path in paths.items():
        try:
            # Every published artifact is exact-0600 under the protocol's
            # publication rule, so hash them all through the private reader.
            digests[name] = _stream_file_sha256(path, label=name, private=True)
        except Exception as error:
            violations.add("Q1-K0", f"cannot hash {name}: {type(error).__name__}:{error}")
            digests[name] = _ZERO_SHA256
    return digests


def _validate_attempt_marker(
    path: Path,
    bindings: _AuditBindings,
    digests: Mapping[str, str],
    violations: _Violations,
) -> tuple[str, str]:
    """Strictly reopen the external marker without producer registry helpers."""
    marker_sha256 = _ZERO_SHA256
    worker_capability_sha256 = _ZERO_SHA256

    try:
        if path.name != _ATTEMPT_MARKER_FILENAME:
            raise Q1AuditError(f"attempt marker filename differs from {_ATTEMPT_MARKER_FILENAME!r}")
        if bindings.attempt_registry_directory is None:
            raise Q1AuditError("entry qualification lacks an attempt-registry identity")
        registry_identity = _require_entry_bound_directory_identity(
            bindings.attempt_registry_directory,
            label="attempt registry",
        )
        if path.parent.resolve() != Path(registry_identity.canonical_path):
            raise Q1AuditError("attempt marker is outside the entry-bound attempt registry")
        payload = _read_regular_file(
            path,
            label="completed attempt marker",
            private=True,
        )
        marker_sha256 = sha256_bytes(payload)
        value = json.loads(
            payload,
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
        if not isinstance(value, dict) or payload != canonical_json_bytes(value, newline=True):
            raise Q1AuditError("attempt marker is not one canonical JSON document")
        expected_fields = {
            "artifact_sha256",
            "attempt_id",
            "entry_qualification_sha256",
            "expected_counts",
            "implementation_sha256",
            "protocol_sha256",
            "protocol_version",
            "q0_report_sha256",
            "run_id",
            "run_sha256",
            "salt_commitment_sha256",
            "schema",
            "status",
            "worker_capability_sha256",
        }
        if set(value) != expected_fields:
            raise Q1AuditError("attempt marker fields differ from the frozen exact set")
        expected_identity = {
            "attempt_id": bindings.attempt_id,
            "entry_qualification_sha256": bindings.entry_qualification_sha256,
            "implementation_sha256": bindings.implementation_sha256,
            "protocol_sha256": bindings.protocol_sha256,
            "protocol_version": Q1_PROTOCOL_VERSION,
            "q0_report_sha256": bindings.q0_report_sha256,
            "run_id": bindings.run_id,
            "run_sha256": bindings.run_sha256,
        }
        for field, expected in expected_identity.items():
            if value.get(field) != expected:
                raise Q1AuditError(f"attempt marker identity mismatch:{field}")
        if value.get("salt_commitment_sha256") != bindings.salt_commitment_sha256:
            raise Q1AuditError("attempt marker salt commitment differs from artifact binding")
        candidate_worker_capability_sha256 = value.get("worker_capability_sha256")
        if (
            not isinstance(candidate_worker_capability_sha256, str)
            or not _is_sha256(candidate_worker_capability_sha256)
            or candidate_worker_capability_sha256 == _ZERO_SHA256
        ):
            raise Q1AuditError("attempt marker worker capability commitment is invalid")
        worker_capability_sha256 = candidate_worker_capability_sha256
        if value.get("schema") != _ATTEMPT_MARKER_SCHEMA:
            raise Q1AuditError("attempt marker schema differs")
        if value.get("status") != "completed":
            raise Q1AuditError("attempt marker is not durably completed")
        if value.get("expected_counts") != _ATTEMPT_EXPECTED_COUNTS:
            raise Q1AuditError("attempt marker frozen budget differs")
        expected_artifacts = {
            "aggregate": digests["producer_aggregate"],
            "checkpoint_frames": digests["checkpoint_frames"],
            "checkpoint_index": digests["checkpoint_index"],
            "private_audit": digests["private_audit"],
            "raw_trace": digests["raw_trace"],
            "restored_trace": digests["restored_trace"],
        }
        artifacts = value.get("artifact_sha256")
        if not isinstance(artifacts, dict) or tuple(sorted(artifacts)) != _ATTEMPT_ARTIFACT_NAMES:
            raise Q1AuditError("attempt marker does not bind exactly six artifacts")
        if artifacts != expected_artifacts:
            raise Q1AuditError("attempt marker artifact digests differ from independently reopened bytes")
    except Exception as error:
        violations.add("Q1-K0", f"attempt marker rejected: {type(error).__name__}:{error}")

    return marker_sha256, worker_capability_sha256


def _decode_canonical_document(
    payload: bytes,
    label: str,
    violations: _Violations,
) -> Mapping[str, object] | None:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
        if not isinstance(value, dict):
            raise Q1AuditError("document is not an object")
        if payload != canonical_json_bytes(value, newline=True):
            raise Q1AuditError("document is not canonical JSON with one final newline")
        return cast(Mapping[str, object], value)
    except Exception as error:
        violations.add("Q1-K0", f"{label} rejected: {type(error).__name__}:{error}")
        return None


def _read_hashed_jsonl_row(
    stream: BinaryIO,
    digest: _HashSink,
    *,
    label: str,
    ordinal: int,
    violations: _Violations,
) -> tuple[Mapping[str, object] | None, bool]:
    line = stream.readline(_MAX_JSONL_ROW_BYTES + 1)
    if not line:
        return None, False
    digest.update(line)
    oversized = len(line) > _MAX_JSONL_ROW_BYTES
    if oversized and not line.endswith(b"\n"):
        while True:
            continuation = stream.readline(_MAX_JSONL_ROW_BYTES + 1)
            if not continuation:
                break
            digest.update(continuation)
            if continuation.endswith(b"\n"):
                break
    if oversized:
        violations.add("Q1-K0", f"{label}[{ordinal}] exceeds the bounded row limit")
        return None, True
    try:
        value = json.loads(
            line,
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
        if not isinstance(value, dict):
            raise Q1AuditError("row is not an object")
        if line != canonical_json_bytes(value, newline=True):
            raise Q1AuditError("row is not canonical JSON with one final newline")
        return cast(Mapping[str, object], value), True
    except Exception as error:
        violations.add(
            "Q1-K0",
            f"{label}[{ordinal}] rejected: {type(error).__name__}:{error}",
        )
        return None, True


def _consume_hashed_remainder(stream: BinaryIO, digest: _HashSink) -> tuple[int, int]:
    byte_count = 0
    newline_count = 0
    last_byte = b""
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
        newline_count += chunk.count(b"\n")
        last_byte = chunk[-1:]
    row_count = newline_count + (1 if byte_count and last_byte != b"\n" else 0)
    return byte_count, row_count


def _validate_stream_row(
    *,
    row: Mapping[str, object],
    schema_name: str,
    label: str,
    expected_key: tuple[int, str, int],
    arm_field: str,
    bindings: _AuditBindings,
    violations: _Violations,
) -> None:
    try:
        validate_artifact(schema_name, row)
    except Exception as error:
        violations.add("Q1-K0", f"{label} schema validation failed ({type(error).__name__})")
    if (row.get("master"), row.get(arm_field), row.get("episode")) != expected_key:
        violations.add("Q1-K0", f"{label} differs from frozen lockstep key {_render_key(expected_key)}")
    expected_bindings = {
        "attempt_id": bindings.attempt_id,
        "entry_qualification_sha256": bindings.entry_qualification_sha256,
        "implementation_sha256": bindings.implementation_sha256,
        "protocol_sha256": bindings.protocol_sha256,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "q0_report_sha256": bindings.q0_report_sha256,
        "run_id": bindings.run_id,
        "salt_commitment_sha256": bindings.salt_commitment_sha256,
    }
    for field_name, expected in expected_bindings.items():
        if row.get(field_name) != expected:
            violations.add("Q1-K0", f"{label} binding mismatch:{field_name}")


def _load_streaming_aggregate(
    *,
    path: Path,
    expected_digest: str,
    bindings: _AuditBindings,
    privacy_scanner: PrivatePrefixScanner | None,
    violations: _Violations,
) -> Mapping[str, object] | None:
    try:
        payload = _read_regular_file(path, label="producer aggregate")
        if sha256_bytes(payload) != expected_digest:
            raise Q1AuditError("producer aggregate changed after the committed hash pass")
        aggregate = _decode_canonical_document(payload, "producer aggregate", violations)
        if aggregate is None:
            return None
        try:
            validate_artifact("aggregate", aggregate)
        except Exception as error:
            raise Q1AuditError(f"producer aggregate schema validation failed ({type(error).__name__})") from error
        expected = {
            "attempt_id": bindings.attempt_id,
            "entry_qualification_sha256": bindings.entry_qualification_sha256,
            "implementation_sha256": bindings.implementation_sha256,
            "protocol_sha256": bindings.protocol_sha256,
            "protocol_version": Q1_PROTOCOL_VERSION,
            "q0_report_sha256": bindings.q0_report_sha256,
            "run_id": bindings.run_id,
            "salt_commitment_sha256": bindings.salt_commitment_sha256,
        }
        for field_name, expected_value in expected.items():
            if aggregate.get(field_name) != expected_value:
                violations.add("Q1-K0", f"producer aggregate binding mismatch:{field_name}")
        if aggregate.get("claim_eligible") is not False:
            violations.add("Q1-K0", "producer aggregate must remain claim-ineligible")
        if aggregate.get("formal_authorized") is not False:
            violations.add("Q1-K0", "producer aggregate must remain formally unauthorized")
        if aggregate.get("producer_analysis_authoritative") is not False:
            violations.add("Q1-K0", "producer aggregate incorrectly claims analytical authority")
        if privacy_scanner is not None:
            leaks = privacy_scanner.scan(aggregate)
            if leaks:
                violations.add("Q1-K1", "producer aggregate contains private-prefix material")
        return aggregate
    except Exception as error:
        violations.add("Q1-K0", f"producer aggregate rejected: {type(error).__name__}:{error}")
        return None


def _read_hashed_bytes(
    stream: BinaryIO,
    digest: _HashSink,
    size: int,
    *,
    retain: bool,
) -> tuple[bytes, int]:
    remaining = size
    chunks: list[bytes] = []
    consumed = 0
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            break
        digest.update(chunk)
        consumed += len(chunk)
        remaining -= len(chunk)
        if retain:
            chunks.append(chunk)
    return (b"".join(chunks) if retain else b""), consumed


def _read_streamed_checkpoint(
    *,
    stream: BinaryIO,
    digest: _HashSink,
    index: Mapping[str, object] | None,
    expected_offset: int,
    key: tuple[int, str, int],
    violations: _Violations,
) -> tuple[bytes | None, int]:
    header = stream.read(8)
    digest.update(header)
    if len(header) != 8:
        violations.add("Q1-K0", f"{_render_key(key)} checkpoint frame header is truncated/missing")
        return None, expected_offset + len(header)
    actual_length = int.from_bytes(header, "big")
    if index is None:
        violations.add("Q1-K0", f"{_render_key(key)} lacks a decodable checkpoint index row")
    else:
        if index.get("frame_offset") != expected_offset:
            violations.add("Q1-K0", f"{_render_key(key)} frame offset is not contiguous")
        if index.get("frame_header_bytes") != 8:
            violations.add("Q1-K0", f"{_render_key(key)} frame header width differs from eight")
        if index.get("frame_length") != actual_length:
            violations.add("Q1-K0", f"{_render_key(key)} frame header differs from indexed length")
    retain = 0 < actual_length <= _MAX_CHECKPOINT_FRAME_BYTES
    payload, consumed = _read_hashed_bytes(
        stream,
        digest,
        actual_length,
        retain=retain,
    )
    next_offset = expected_offset + 8 + consumed
    if consumed != actual_length:
        violations.add("Q1-K0", f"{_render_key(key)} checkpoint frame payload is truncated")
        return None, next_offset
    if not retain:
        violations.add(
            "Q1-K0",
            f"{_render_key(key)} checkpoint frame exceeds the {_MAX_CHECKPOINT_FRAME_BYTES}-byte bound",
        )
        return None, next_offset
    return payload, next_offset


def _validate_stream_process_partition(
    *,
    producer_by_master: Mapping[int, set[int]],
    restorer_by_lane: Mapping[tuple[int, str], set[int]],
    violations: _Violations,
) -> None:
    for master in range(MASTER_COUNT):
        count = len(producer_by_master.get(master, set()))
        if count != 1:
            violations.add("Q1-K0", f"master {master} has {count} producer PIDs; expected one constant PID")
        for arm in ARM_ORDER:
            restore_count = len(restorer_by_lane.get((master, arm), set()))
            if restore_count != 1:
                violations.add(
                    "Q1-K5",
                    f"master={master},arm={arm} has {restore_count} restorer PIDs; expected one lane PID",
                )


def _stream_audit_evidence(
    *,
    paths: Mapping[str, Path],
    committed_digests: Mapping[str, str],
    bindings: _AuditBindings,
    schedule: _IndependentSeedSchedule | None,
    privacy_scanner: PrivatePrefixScanner | None,
    expected_binding: QualificationBinding | None,
    violations: _Violations,
) -> _StreamingAuditResult:
    aggregate = _load_streaming_aggregate(
        path=paths["producer_aggregate"],
        expected_digest=committed_digests["producer_aggregate"],
        bindings=bindings,
        privacy_scanner=privacy_scanner,
        violations=violations,
    )
    stream_names = ("raw_trace", "private_audit", "checkpoint_index", "restored_trace")
    streams: dict[str, BinaryIO] = {}
    stream_metadata: dict[str, os.stat_result] = {}
    try:
        for name in (*stream_names, "checkpoint_frames"):
            descriptor = _open_regular_descriptor(
                paths[name],
                label=name,
                private=name == "private_audit",
            )
            stream_metadata[name] = os.fstat(descriptor)
            streams[name] = os.fdopen(descriptor, "rb")
    except Exception as error:
        for stream in streams.values():
            stream.close()
        violations.add("Q1-K0", f"streaming evidence open failed: {type(error).__name__}:{error}")
        return _StreamingAuditResult(
            returns={},
            counts=_audit_counts_from_line_counts({}, set(), set()),
            aggregate=aggregate,
        )

    hashes = {name: hashlib.sha256() for name in (*stream_names, "checkpoint_frames")}
    line_counts = {name: 0 for name in stream_names}
    returns: dict[tuple[int, str], list[float]] = defaultdict(list)
    masters_seen: set[int] = set()
    arms_seen: set[str] = set()
    producer_by_master: dict[int, set[int]] = defaultdict(set)
    restorer_by_lane: dict[tuple[int, str], set[int]] = defaultdict(set)
    frame_offset = 0
    ordinal = 0
    schemas = {
        "raw_trace": ("raw_trace", "raw trace", "arm"),
        "private_audit": ("private_audit", "private audit", "arm_id"),
        "checkpoint_index": ("checkpoint_frame", "checkpoint index", "arm"),
        "restored_trace": ("restored_trace", "restored trace", "arm"),
    }
    try:
        if schedule is not None:
            for master in range(MASTER_COUNT):
                theta = schedule.theta_schedule(master)
                if theta.count(-1) != EPISODES_PER_MASTER // 2 or theta.count(1) != EPISODES_PER_MASTER // 2:
                    violations.add("Q1-K1", f"master {master} private theta schedule is not exactly balanced")
        for master in range(MASTER_COUNT):
            for arm in ARM_ORDER:
                for episode in range(EPISODES_PER_MASTER):
                    key = (master, arm, episode)
                    rows: dict[str, Mapping[str, object] | None] = {}
                    for name in stream_names:
                        row, present = _read_hashed_jsonl_row(
                            streams[name],
                            hashes[name],
                            label=schemas[name][1],
                            ordinal=ordinal,
                            violations=violations,
                        )
                        if present:
                            line_counts[name] += 1
                        else:
                            violations.add("Q1-K0", f"{schemas[name][1]} misses row {ordinal}")
                        rows[name] = row
                        if row is not None:
                            _validate_stream_row(
                                row=row,
                                schema_name=schemas[name][0],
                                label=f"{schemas[name][1]}[{ordinal}]",
                                expected_key=key,
                                arm_field=schemas[name][2],
                                bindings=bindings,
                                violations=violations,
                            )
                            if privacy_scanner is not None and name != "private_audit":
                                leaks = privacy_scanner.scan(row)
                                if leaks:
                                    violations.add(
                                        "Q1-K1",
                                        f"{schemas[name][1]}[{ordinal}] contains private-prefix material",
                                    )

                    raw = rows["raw_trace"]
                    private = rows["private_audit"]
                    index = rows["checkpoint_index"]
                    restored_row = rows["restored_trace"]
                    if raw is not None:
                        masters_seen.add(master)
                        arms_seen.add(arm)
                        producer_pid = raw.get("producer_pid")
                        if type(producer_pid) is int:
                            producer_by_master[master].add(producer_pid)
                        _record_identifier_violations(raw, key, violations)
                    if private is not None and schedule is not None:
                        try:
                            expected_private = schedule.private_row(master, episode, arm)
                            expected_private["attempt_id"] = bindings.attempt_id
                            expected_private["run_id"] = bindings.run_id
                            if private != expected_private:
                                violations.add(
                                    "Q1-K1", f"{_render_key(key)} private row differs from HMAC recomputation"
                                )
                        except Exception as error:
                            violations.add("Q1-K1", f"{_render_key(key)} private row cannot be reconstructed:{error}")
                    if raw is not None and private is not None:
                        episode_violations, episode_return = _episode_semantic_violations(raw, private, schedule)
                        for gate, message in episode_violations:
                            violations.add(gate, f"{_render_key(key)}:{message}")
                        if episode_return is not None:
                            returns[(master, arm)].append(episode_return)

                    payload, frame_offset = _read_streamed_checkpoint(
                        stream=streams["checkpoint_frames"],
                        digest=hashes["checkpoint_frames"],
                        index=index,
                        expected_offset=frame_offset,
                        key=key,
                        violations=violations,
                    )
                    if payload is not None and raw is not None and index is not None and expected_binding is not None:
                        _audit_checkpoint_payload(
                            key=key,
                            raw=raw,
                            index=index,
                            payload=payload,
                            expected_binding=expected_binding,
                            privacy_scanner=privacy_scanner,
                            violations=violations,
                        )
                    if restored_row is not None:
                        restorer_pid = restored_row.get("restorer_pid")
                        if type(restorer_pid) is int:
                            restorer_by_lane[(master, arm)].add(restorer_pid)
                    if raw is not None and restored_row is not None:
                        if restored_row.get("restorer_pid") == raw.get("producer_pid"):
                            violations.add("Q1-K5", f"{_render_key(key)} restorer PID equals producer PID")
                        _audit_restored_parity(
                            key=key,
                            raw=raw,
                            restored=restored_row,
                            violations=violations,
                        )
                    ordinal += 1

        for name in stream_names:
            extra_bytes, extra_rows = _consume_hashed_remainder(streams[name], hashes[name])
            if extra_bytes:
                line_counts[name] += extra_rows
                violations.add("Q1-K0", f"{schemas[name][1]} has {extra_rows} trailing out-of-budget row(s)")
        trailing_bytes, _ = _consume_hashed_remainder(
            streams["checkpoint_frames"],
            hashes["checkpoint_frames"],
        )
        if trailing_bytes:
            violations.add("Q1-K0", f"checkpoint frames contain {trailing_bytes} unindexed trailing byte(s)")
        for name, digest in hashes.items():
            if digest.hexdigest() != committed_digests[name]:
                violations.add("Q1-K0", f"{name} changed between committed hash and semantic pass")
            try:
                stream = streams[name]
                after = os.fstat(stream.fileno())
                if (
                    _regular_metadata_signature(stream_metadata[name]) != _regular_metadata_signature(after)
                    or stream.tell() != after.st_size
                ):
                    raise Q1AuditError(f"{name} changed during the semantic pass")
                _require_path_matches_open_file(paths[name], after, label=name)
            except Exception as error:
                violations.add(
                    "Q1-K0",
                    f"{name} identity changed during semantic pass: {type(error).__name__}:{error}",
                )
    except Exception as error:
        violations.add("Q1-K0", f"streaming audit aborted safely: {type(error).__name__}:{error}")
        for name, stream in streams.items():
            try:
                _consume_hashed_remainder(stream, hashes[name])
            except Exception:
                pass
    finally:
        for stream in streams.values():
            stream.close()

    _validate_stream_process_partition(
        producer_by_master=producer_by_master,
        restorer_by_lane=restorer_by_lane,
        violations=violations,
    )
    counts = _audit_counts_from_line_counts(line_counts, masters_seen, arms_seen)
    return _StreamingAuditResult(
        returns={key: tuple(values) for key, values in returns.items()},
        counts=counts,
        aggregate=aggregate,
    )


def _audit_counts_from_line_counts(
    line_counts: Mapping[str, int],
    masters_seen: set[int],
    arms_seen: set[str],
) -> dict[str, int]:
    raw_rows = line_counts.get("raw_trace", 0)
    return {
        "acquisition_updates": raw_rows,
        "arms": len(arms_seen),
        "checkpoint_frames": line_counts.get("checkpoint_index", 0),
        "environment_steps": raw_rows * 2,
        "episodes": raw_rows,
        "masters": len(masters_seen),
        "private_rows": line_counts.get("private_audit", 0),
        "raw_rows": raw_rows,
        "restored_rows": line_counts.get("restored_trace", 0),
        "terminal_updates": 0,
        "transitions": raw_rows * 2,
    }


def _episode_semantic_violations(
    raw: Mapping[str, object],
    private: Mapping[str, object],
    schedule: _IndependentSeedSchedule | None,
) -> tuple[list[tuple[str, str]], float | None]:
    violations: list[tuple[str, str]] = []
    try:
        master = _required_int(raw, "master")
        arm = _required_str(raw, "arm")
        episode = _required_int(raw, "episode")
        acquisition = _required_mapping(raw, "acquisition")
        terminal = _required_mapping(raw, "terminal")
        counts = _required_mapping(raw, "counts")
        action = _required_str(acquisition, "selected_action")
        observed_symbol = _required_int(acquisition, "observed_symbol")
    except Exception as error:
        return [("Q1-K1", f"cannot decode primitive episode row: {error}")], None

    if arm not in ARM_ORDER or action not in ACTION_ORDER:
        violations.append(("Q1-K2", "arm or selected action is outside the frozen order"))
        return violations, None
    expected_action = _EXPECTED_ACTION_BY_ARM.get(arm)
    uniform = None
    if arm == "uniform_random":
        try:
            expected_action, uniform_digest = schedule.uniform_selection(master, episode) if schedule else (None, None)
            uniform = uniform_digest
        except Exception as error:
            violations.append(("Q1-K2", f"uniform selector reconstruction failed: {error}"))
    if action != expected_action:
        violations.append(("Q1-K2", f"selector chose {action!r}; expected {expected_action!r}"))
    declared_uniform = acquisition.get("uniform_selector_sha256")
    expected_uniform_digest = uniform
    if declared_uniform != expected_uniform_digest:
        violations.append(("Q1-K2", "uniform selector digest differs from the public derivation"))

    candidate_rows = raw.get("candidate_rows")
    if not isinstance(candidate_rows, list):
        violations.append(("Q1-K2", "candidate_rows is not an array"))
    else:
        violations.extend(_candidate_row_violations(candidate_rows, arm))
        if raw.get("candidate_rows_sha256") != canonical_sha256(candidate_rows):
            violations.append(("Q1-K2", "candidate row digest does not bind the emitted ordered rows"))

    if raw.get("semantic_key_sha256") is not None and schedule is not None:
        expected_semantic_digest = schedule.theta_semantic_sha256(master, episode)
        if raw.get("semantic_key_sha256") != expected_semantic_digest:
            violations.append(("Q1-K1", "public semantic key does not bind the hidden schedule member"))

    theta_value = private.get("theta")
    try:
        if type(theta_value) is not int:
            raise ValueError("theta is not a canonical integer")
        theta = cast(int, theta_value)
        if theta not in {-1, 1}:
            raise ValueError("theta is outside {-1,+1}")
    except Exception:
        violations.append(("Q1-K1", "private theta is not exactly -1 or +1"))
        theta = None
    if schedule is not None and theta is not None:
        if theta != schedule.theta(master, episode):
            violations.append(("Q1-K1", "private theta differs from reconstructed balanced schedule"))
        if action == "skip":
            expected_symbol = 0
        elif action == "nuisance":
            expected_symbol = schedule.nuisance_observation(master, episode)
        else:
            expected_symbol = schedule.pulse_observation(master, episode, action, theta)
        if observed_symbol != expected_symbol:
            violations.append(("Q1-K1", "observed symbol differs from the action-keyed private outcome"))

    expected_payoff = 1.0 if action in _ACTION_ACCURACY and observed_symbol == 1 else 0.0
    expected_action_cost = float(_ACTION_COST[action])
    expected_information_cost = float(_INFORMATION_COST[action])
    expected_net = expected_payoff - expected_action_cost - expected_information_cost
    for field, expected in (
        ("task_payoff", expected_payoff),
        ("physical_action_cost", expected_action_cost),
        ("information_acquisition_cost", expected_information_cost),
        ("net_reward", expected_net),
    ):
        if not _same_number(acquisition.get(field), expected):
            violations.append(("Q1-K1", f"acquisition primitive {field} differs from exact reconstruction"))

    posterior_before = _parse_fraction(acquisition.get("posterior_before"))
    posterior_after = _parse_fraction(acquisition.get("posterior_after"))
    expected_posterior = _posterior_direct(action, observed_symbol)
    if posterior_before != Fraction(1, 2):
        violations.append(("Q1-K3", "acquisition posterior_before is not the independent prior 1/2"))
    if posterior_after != expected_posterior:
        violations.append(("Q1-K3", "posterior_after differs from true executed-action likelihood"))
    if acquisition.get("evidence_count_before") != 0 or acquisition.get("evidence_count_after") != 1:
        violations.append(("Q1-K3", "evidence count did not advance exactly 0 -> 1"))
    before_digest = acquisition.get("model_before_sha256")
    after_digest = acquisition.get("model_after_sha256")
    if before_digest == after_digest:
        violations.append(("Q1-K3", "model bytes did not change after acquisition learning"))
    if before_digest != _INITIAL_MODEL_SHA256:
        violations.append(("Q1-K3", "model_before_sha256 differs from the canonical zero-evidence prior"))
    if acquisition.get("model_before_version") != _INITIAL_MODEL_VERSION:
        violations.append(("Q1-K3", "model_before_version differs from the canonical zero-evidence prior"))
    if acquisition.get("model_before_version") != f"{_MODEL_VERSION_PREFIX}{before_digest}":
        violations.append(("Q1-K3", "model_before_version does not bind predecessor bytes"))
    if acquisition.get("model_after_version") != f"{_MODEL_VERSION_PREFIX}{after_digest}":
        violations.append(("Q1-K3", "model_after_version does not bind resulting bytes"))

    selected_terminal = terminal.get("selected_action")
    violations.extend(_terminal_candidate_violations(terminal, expected_posterior))

    expected_terminal = 1 if expected_posterior >= Fraction(1, 2) else -1
    if selected_terminal != expected_terminal:
        violations.append(("Q1-K2", "terminal selector differs from posterior optimum or +1 tie rule"))
    success = terminal.get("success")
    if type(success) is not bool:
        violations.append(("Q1-K1", "terminal success is not boolean"))
        success_value = 0.0
    else:
        success_value = float(success)
        if schedule is not None and theta is not None:
            expected_success = schedule.terminal_success(
                master,
                episode,
                expected_terminal,
                theta,
            )
            if success is not expected_success:
                violations.append(("Q1-K1", "terminal success differs from decision-keyed private outcome"))
    if not _same_number(terminal.get("terminal_reward"), success_value):
        violations.append(("Q1-K1", "terminal reward differs from primitive success"))
    reconstructed_return = expected_payoff - expected_action_cost - expected_information_cost + success_value
    if not _same_number(terminal.get("episode_return"), reconstructed_return):
        violations.append(("Q1-K1", "episode return differs from primitive executed reconstruction"))

    expected_counts = {
        "decisions": 2,
        "executions": 2,
        "experiences": 2,
        "transitions": 2,
        "acquisition_updates": 1,
        "terminal_updates": 0,
    }
    if counts != expected_counts:
        violations.append(("Q1-K0", "per-episode runtime counts differ from the exact two-step budget"))
    return violations, reconstructed_return


def _candidate_row_violations(
    rows: Sequence[object],
    arm: str,
) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    if len(rows) != len(ACTION_ORDER):
        return [("Q1-K2", "candidate row count differs from five")]
    expected_unit = {
        "raw_observation_entropy": "nats",
        "eig_only": "nats",
        "uniform_random": None,
    }.get(arm, "return")
    for ordinal, (action, row_value) in enumerate(zip(ACTION_ORDER, rows, strict=True)):
        if not isinstance(row_value, Mapping):
            violations.append(("Q1-K2", f"candidate {ordinal} is not an object"))
            continue
        row = row_value
        if row.get("ordinal") != ordinal or row.get("semantic_action") != action:
            violations.append(("Q1-K2", f"candidate {ordinal} identity/order differs"))
        decision_value = _EXPECTED_DECISION_VALUE[action]
        expected_terminal = Fraction(1, 2) + decision_value
        eig = _binary_information_gain(action)
        raw_entropy = _raw_observation_entropy(action)
        expected_numeric = {
            "expected_episode_value": float(_EXPECTED_EPISODE_VALUE[action]),
            "expected_immediate_payoff": float(_EXPECTED_IMMEDIATE[action]),
            "expected_terminal_value": float(expected_terminal),
            "expected_decision_value": float(decision_value),
            "expected_information_gain_nats": eig,
            "raw_observation_entropy_nats": raw_entropy,
            "physical_action_cost": float(_ACTION_COST[action]),
            "information_acquisition_cost": float(_INFORMATION_COST[action]),
        }
        for field, expected in expected_numeric.items():
            if not _same_number(row.get(field), expected):
                violations.append(("Q1-K2", f"candidate {ordinal} {field} differs"))
        expected_score = _arm_selection_score(arm, action)
        if expected_score is None:
            if row.get("arm_selection_score") is not None:
                violations.append(("Q1-K2", f"candidate {ordinal} uniform score must be null"))
        elif not _same_number(row.get("arm_selection_score"), expected_score):
            violations.append(("Q1-K2", f"candidate {ordinal} arm selection score differs"))
        if row.get("selection_unit") != expected_unit:
            violations.append(("Q1-K2", f"candidate {ordinal} selection unit differs"))
    return violations


def _arm_selection_score(arm: str, action: str) -> float | None:
    if arm in {"prospect_expected_return", "independent_fraction_oracle"}:
        return float(_EXPECTED_EPISODE_VALUE[action])
    if arm == "goal_only":
        return float(Fraction(1, 2) + _EXPECTED_IMMEDIATE[action] - _ACTION_COST[action] - _INFORMATION_COST[action])
    if arm == "raw_observation_entropy":
        return _raw_observation_entropy(action)
    if arm == "eig_only":
        return _binary_information_gain(action)
    if arm == "shuffled_information":
        source = _SHUFFLED_SOURCE[action]
        return float(
            Fraction(1, 2)
            + _EXPECTED_IMMEDIATE[action]
            + _EXPECTED_DECISION_VALUE[source]
            - _ACTION_COST[action]
            - _INFORMATION_COST[action]
        )
    return None


def _binary_information_gain(action: str) -> float:
    if action in {"skip", "nuisance"}:
        return 0.0
    accuracy = float(_ACTION_ACCURACY[action])
    return math.log(2.0) + (
        0.0 if accuracy in {0.0, 1.0} else accuracy * math.log(accuracy) + (1.0 - accuracy) * math.log(1.0 - accuracy)
    )


def _terminal_candidate_violations(
    terminal: Mapping[str, object],
    posterior_direct: Fraction,
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    rows = terminal.get("candidate_rows")
    if not isinstance(rows, list) or len(rows) != 2:
        return [("Q1-K2", "terminal decision lacks exactly two auditable candidate rows")]
    if terminal.get("candidate_rows_sha256") != canonical_sha256(rows):
        found.append(("Q1-K2", "terminal candidate digest does not bind the emitted two rows"))
    expected_values = (
        Fraction(1, 10) + Fraction(4, 5) * posterior_direct,
        Fraction(9, 10) - Fraction(4, 5) * posterior_direct,
    )
    assessment_ids: set[str] = set()
    prediction_ids: set[str] = set()
    for ordinal, (action, expected, row) in enumerate(zip((1, -1), expected_values, rows, strict=True)):
        if not isinstance(row, Mapping):
            found.append(("Q1-K2", f"terminal candidate {ordinal} is not an object"))
            continue
        if row.get("ordinal") != ordinal or row.get("terminal_action") != action:
            found.append(("Q1-K2", f"terminal candidate {ordinal} identity/order differs"))
        if row.get("unit") != "return" or not _same_number(row.get("expected_success"), float(expected)):
            found.append(("Q1-K2", f"terminal candidate {ordinal} expected success/unit differs"))
        assessment_id = row.get("assessment_id")
        prediction_id = row.get("prediction_id")
        if not isinstance(assessment_id, str) or not assessment_id or assessment_id in assessment_ids:
            found.append(("Q1-K2", f"terminal candidate {ordinal} assessment identity is invalid/duplicate"))
        else:
            assessment_ids.add(assessment_id)
        if not isinstance(prediction_id, str) or not prediction_id or prediction_id in prediction_ids:
            found.append(("Q1-K2", f"terminal candidate {ordinal} prediction identity is invalid/duplicate"))
        else:
            prediction_ids.add(prediction_id)
    expected_selected = 1 if expected_values[0] >= expected_values[1] else -1
    if terminal.get("selected_action") != expected_selected:
        found.append(("Q1-K2", "terminal selection differs from recomputed candidate values/+1 tie rule"))
    return found


def _raw_observation_entropy(action: str) -> float:
    if action == "skip":
        return 0.0
    if action == "nuisance":
        return math.log(4.0)
    return math.log(2.0)


def _posterior_direct(action: str, observed_symbol: int) -> Fraction:
    if action in {"skip", "nuisance"}:
        return Fraction(1, 2)
    accuracy = _ACTION_ACCURACY[action]
    if observed_symbol == 1:
        return accuracy
    if observed_symbol == -1:
        return 1 - accuracy
    raise Q1AuditError(f"observed symbol {observed_symbol} is outside {action} support")


def _record_identifier_violations(
    raw: Mapping[str, object],
    key: tuple[int, str, int],
    violations: _Violations,
) -> None:
    """Prove global uniqueness structurally while retaining only one row of IDs."""

    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        violations.add("Q1-K1", f"{_render_key(key)} lacks a run-scoped identity namespace")
        return
    namespace = f"{run_id}:m{key[0]}:a{key[1]}:e{key[2]}"
    identities: list[tuple[str, object]] = []
    sections = (
        (
            "acquisition",
            (
                "decision_id",
                "intention_id",
                "execution_id",
                "observation_id",
                "outcome_id",
                "experience_id",
                "transition_id",
                "receipt_id",
            ),
        ),
        (
            "terminal",
            (
                "decision_id",
                "intention_id",
                "execution_id",
                "observation_id",
                "outcome_id",
                "experience_id",
                "transition_id",
            ),
        ),
    )
    for section_name, field_names in sections:
        section = raw.get(section_name)
        if not isinstance(section, Mapping):
            continue
        identities.extend((f"{section_name}.{field_name}", section.get(field_name)) for field_name in field_names)
    candidate_rows = raw.get("candidate_rows")
    if isinstance(candidate_rows, Sequence) and not isinstance(candidate_rows, (str, bytes, bytearray)):
        for ordinal, row in enumerate(candidate_rows):
            if isinstance(row, Mapping):
                identities.append((f"candidate_rows[{ordinal}].assessment_id", row.get("assessment_id")))
    terminal = raw.get("terminal")
    terminal_rows = terminal.get("candidate_rows") if isinstance(terminal, Mapping) else None
    if isinstance(terminal_rows, Sequence) and not isinstance(terminal_rows, (str, bytes, bytearray)):
        for ordinal, row in enumerate(terminal_rows):
            if isinstance(row, Mapping):
                identities.append((f"terminal.candidate_rows[{ordinal}].assessment_id", row.get("assessment_id")))
                identities.append((f"terminal.candidate_rows[{ordinal}].prediction_id", row.get("prediction_id")))

    local_seen: set[str] = set()
    for label, value in identities:
        if not isinstance(value, str) or not value.startswith(f"{namespace}:"):
            violations.add("Q1-K1", f"{_render_key(key)} {label} is outside its exact episode namespace")
            continue
        if value in local_seen:
            violations.add("Q1-K1", f"{_render_key(key)} duplicates record identity {value!r} at {label}")
        local_seen.add(value)
    # Lockstep keys are exact and each embeds a distinct namespace, so namespace
    # membership plus within-row uniqueness proves cross-row uniqueness without
    # retaining O(total-records) identifier strings.


def _audit_checkpoint_payload(
    *,
    key: tuple[int, str, int],
    raw: Mapping[str, object],
    index: Mapping[str, object],
    payload: bytes,
    expected_binding: QualificationBinding,
    privacy_scanner: PrivatePrefixScanner | None,
    violations: _Violations,
) -> None:
    try:
        length = _required_int(index, "frame_length")
        declared_sha = _required_str(index, "checkpoint_sha256")
        if len(payload) != length:
            raise Q1AuditError("streamed checkpoint payload length differs from index")
        if sha256_bytes(payload) != declared_sha:
            raise Q1AuditError("streamed checkpoint digest differs from index")
        checkpoint = _required_mapping(raw, "checkpoint")
        if checkpoint.get("sha256") != declared_sha:
            raise Q1AuditError("raw trace and frame index checkpoint digests differ")
        if checkpoint.get("size_bytes") != length:
            raise Q1AuditError("raw trace checkpoint size differs from frame length")
        if checkpoint.get("component_sha256") != index.get("component_sha256"):
            raise Q1AuditError("raw trace and frame index component digests differ")
        restored = load_q1_checkpoint(
            payload,
            expected_agent_id=_AGENT_ID,
            expected_aggregate_sha256=declared_sha,
            expected_binding=expected_binding,
            model_validator=_auditor_model_validator,
        )
        report_components = {row.component_id: row.sha256 for row in restored.report.component_digests}
        if report_components != index.get("component_sha256"):
            raise Q1AuditError("decoded component digests differ from the frame index")
        acquisition = _required_mapping(raw, "acquisition")
        model_state = restored.model_owner.snapshot_state()
        decoded_model = _decode_posterior_model(model_state.payload)
        if model_state.digest != acquisition.get("model_after_sha256"):
            raise Q1AuditError("checkpoint model digest differs from raw acquisition result")
        if restored.model_predecessor_sha256 != acquisition.get("model_before_sha256"):
            raise Q1AuditError("checkpoint predecessor digest differs from raw acquisition model")
        if str(decoded_model.posterior_direct) != acquisition.get("posterior_after"):
            raise Q1AuditError("checkpoint posterior differs from raw acquisition posterior")
        if decoded_model.evidence_count != 1:
            raise Q1AuditError("checkpoint model does not contain exactly one acquisition update")
        if decoded_model.last_experience_id != acquisition.get("experience_id"):
            raise Q1AuditError("checkpoint model last_experience_id differs from raw lineage")
        if decoded_model.last_transition_id != acquisition.get("transition_id"):
            raise Q1AuditError("checkpoint model last_transition_id differs from raw lineage")
        if restored.experience.experience_id != acquisition.get("experience_id"):
            raise Q1AuditError("checkpoint experience differs from raw lineage")
        if restored.transition.transition_id != acquisition.get("transition_id"):
            raise Q1AuditError("checkpoint transition differs from raw lineage")
        if restored.receipt.receipt_id != acquisition.get("receipt_id"):
            raise Q1AuditError("checkpoint receipt differs from raw lineage")
        if privacy_scanner is not None:
            public_checkpoint = {
                "accumulator": restored.accumulator.as_dict(),
                "binding": restored.binding.as_dict(),
                "domain_roots": [restored.snapshot, restored.experience, restored.transition, restored.receipt],
                "identity_counter": json.loads(
                    restored.identity_source.checkpoint_bytes(),
                    object_pairs_hook=_reject_duplicate_object_pairs,
                ),
                "model": json.loads(
                    model_state.payload,
                    object_pairs_hook=_reject_duplicate_object_pairs,
                ),
            }
            leaks = privacy_scanner.scan(_domain_json_value(public_checkpoint))
            if leaks:
                violations.add("Q1-K1", f"{_render_key(key)} checkpoint contains private-prefix material")
        for gate, message in _checkpoint_semantic_violations(restored, raw):
            violations.add(gate, f"{_render_key(key)}:{message}")
    except Q1AuditError as error:
        violations.add("Q1-K3", f"{_render_key(key)} checkpoint rejected: Q1AuditError:{error}")
    except Exception as error:
        violations.add("Q1-K3", f"{_render_key(key)} checkpoint rejected ({type(error).__name__})")


def _decode_posterior_model(payload: bytes) -> _DecodedPosteriorModel:
    """Decode the exact six-field posterior model without producer semantics."""

    if type(payload) is not bytes:
        raise ValueError("posterior model payload must be immutable bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: _reject_nonfinite(token),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("posterior model is not finite UTF-8 JSON") from error
    expected_fields = {
        "evidence_count",
        "last_experience_id",
        "last_transition_id",
        "likelihood_version",
        "posterior_direct",
        "schema",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("posterior model differs from the exact six-field schema")
    if payload != canonical_json_bytes(value):
        raise ValueError("posterior model is not canonical JSON")

    evidence_count = value["evidence_count"]
    if type(evidence_count) is not int or evidence_count < 0:
        raise ValueError("posterior model evidence_count must be a nonnegative integer")
    fraction_value = value["posterior_direct"]
    if not isinstance(fraction_value, dict) or set(fraction_value) != {"denominator", "numerator"}:
        raise ValueError("posterior_direct differs from the exact rational schema")
    numerator = fraction_value["numerator"]
    denominator = fraction_value["denominator"]
    if type(numerator) is not int or type(denominator) is not int:
        raise ValueError("posterior rational fields must be canonical integers")
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("posterior rational must lie in [0,1] with positive denominator")
    if math.gcd(numerator, denominator) != 1:
        raise ValueError("posterior rational must be reduced")

    last_experience_id = _nullable_model_identifier(value["last_experience_id"], "last_experience_id")
    last_transition_id = _nullable_model_identifier(value["last_transition_id"], "last_transition_id")
    if value["schema"] != _POSTERIOR_MODEL_SCHEMA:
        raise ValueError("posterior model schema is not the frozen Q1 schema")
    if value["likelihood_version"] != _LIKELIHOOD_VERSION:
        raise ValueError("posterior model likelihood version is not the frozen known model")
    if evidence_count == 0:
        if last_experience_id is not None or last_transition_id is not None:
            raise ValueError("zero-evidence posterior model claims consumed lineage")
    elif last_experience_id is None or last_transition_id is None:
        raise ValueError("nonzero-evidence posterior model lacks consumed lineage")
    return _DecodedPosteriorModel(
        evidence_count=evidence_count,
        last_experience_id=last_experience_id,
        last_transition_id=last_transition_id,
        posterior_direct=Fraction(numerator, denominator),
    )


def _nullable_model_identifier(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"posterior model {label} must be null or a nonempty string")
    return value


def _auditor_model_validator(state: ModelState) -> None:
    _decode_posterior_model(state.payload)
    if state.digest != sha256_bytes(state.payload):
        raise ValueError("model state digest differs from its payload")
    if state.version != f"{_MODEL_VERSION_PREFIX}{state.digest}":
        raise ValueError("model version differs from its canonical payload digest")


def _identity_counter_violations(
    restored: RestoredQ1Checkpoint,
    raw: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Prove public-zero initialization and checkpoint-to-terminal continuity."""

    found: list[tuple[str, str]] = []
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return [("Q1-K0", "raw trace lacks its independently bound run identity")]
    namespace = f"{run_id}:m{raw.get('master')}:a{raw.get('arm')}:e{raw.get('episode')}"
    source = restored.identity_source
    if source.namespace != namespace:
        found.append(("Q1-K1", "checkpoint identity-counter namespace differs from the frozen episode namespace"))
        return found
    next_counter = source.next_counter
    if type(next_counter) is not int:
        found.append(("Q1-K1", "checkpoint identity next_counter is not a nonnegative canonical integer"))
        return found
    if next_counter != _PRETERMINAL_IDENTITY_COUNTER:
        found.append(
            (
                "Q1-K1",
                f"checkpoint identity next_counter must equal the exact preterminal value "
                f"{_PRETERMINAL_IDENTITY_COUNTER}",
            )
        )
        return found

    try:
        checkpoint = _required_mapping(raw, "checkpoint")
        component_sha256 = _required_mapping(checkpoint, "component_sha256")
        expected_digest = _required_str(component_sha256, "identity_counter")
        if sha256_bytes(source.checkpoint_bytes()) != expected_digest:
            found.append(("Q1-K1", "decoded identity counter does not reproduce its bound component digest"))
    except Exception as error:
        found.append(("Q1-K1", f"identity-counter component binding cannot be decoded: {error}"))

    counters, malformed = _namespace_identity_counters(
        (
            restored.snapshot,
            restored.experience,
            restored.transition,
            restored.receipt,
        ),
        namespace,
    )
    if malformed:
        found.append(("Q1-K1", "checkpoint graph contains a malformed episode-scoped record identity"))
    expected_counters = set(range(_INITIAL_IDENTITY_COUNTER, _PRETERMINAL_IDENTITY_COUNTER))
    if counters != expected_counters:
        missing = len(expected_counters - counters)
        extra = len(counters - expected_counters)
        found.append(
            (
                "Q1-K1",
                "checkpoint identity sequence does not prove contiguous public-zero allocation "
                f"(missing={missing},extra={extra})",
            )
        )

    terminal = raw.get("terminal")
    if not isinstance(terminal, Mapping):
        found.append(("Q1-K5", "terminal trace is unavailable for exact counter continuation"))
        return found
    candidate_rows = terminal.get("candidate_rows")
    if not isinstance(candidate_rows, Sequence) or isinstance(candidate_rows, (str, bytes, bytearray)):
        found.append(("Q1-K5", "terminal candidates are unavailable for exact counter continuation"))
        return found
    candidate_values = tuple(candidate_rows)
    if len(candidate_values) != 2 or any(not isinstance(row, Mapping) for row in candidate_values):
        found.append(("Q1-K5", "terminal candidates differ from the exact two-row counter sequence"))
        return found
    direct = cast(Mapping[str, object], candidate_values[0])
    reversed_row = cast(Mapping[str, object], candidate_values[1])
    expected_identities = {
        "terminal.candidate_rows[0].prediction_id": (
            direct.get("prediction_id"),
            f"{namespace}:prediction-terminal-direct:{next_counter}",
        ),
        "terminal.candidate_rows[0].assessment_id": (
            direct.get("assessment_id"),
            f"{namespace}:assessment-terminal-direct:{next_counter + 4}",
        ),
        "terminal.candidate_rows[1].prediction_id": (
            reversed_row.get("prediction_id"),
            f"{namespace}:prediction-terminal-reversed:{next_counter + 5}",
        ),
        "terminal.candidate_rows[1].assessment_id": (
            reversed_row.get("assessment_id"),
            f"{namespace}:assessment-terminal-reversed:{next_counter + 9}",
        ),
        "terminal.intention_id": (terminal.get("intention_id"), f"{namespace}:intention:{next_counter + 10}"),
        "terminal.decision_id": (terminal.get("decision_id"), f"{namespace}:decision:{next_counter + 11}"),
        "terminal.execution_id": (terminal.get("execution_id"), f"{namespace}:execution-terminal:{next_counter + 12}"),
        "terminal.observation_id": (
            terminal.get("observation_id"),
            f"{namespace}:observation-terminal:{next_counter + 13}",
        ),
        "terminal.outcome_id": (terminal.get("outcome_id"), f"{namespace}:outcome-terminal:{next_counter + 15}"),
        "terminal.experience_id": (terminal.get("experience_id"), f"{namespace}:experience:{next_counter + 16}"),
        "terminal.transition_id": (terminal.get("transition_id"), f"{namespace}:transition:{next_counter + 24}"),
    }
    for label, (actual, expected) in expected_identities.items():
        if actual != expected:
            found.append(("Q1-K5", f"{label} differs from exact checkpoint-counter continuation"))
    return found


def _domain_json_value(value: object, active: set[int] | None = None) -> object:
    """Convert decoded immutable domain records to scanner-safe JSON values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Q1AuditError("decoded checkpoint contains a nonfinite number")
        return value
    if isinstance(value, Enum):
        return _domain_json_value(value.value, active)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    containers = active if active is not None else set()
    marker = id(value)
    if marker in containers:
        raise Q1AuditError("decoded checkpoint contains a recursive domain graph")
    containers.add(marker)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: _domain_json_value(getattr(value, field.name), containers) for field in fields(value)}
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise Q1AuditError("decoded checkpoint mapping has a non-string key")
            return {cast(str, key): _domain_json_value(nested, containers) for key, nested in value.items()}
        if isinstance(value, Sequence):
            return [_domain_json_value(nested, containers) for nested in value]
        if isinstance(value, (set, frozenset)):
            return [_domain_json_value(nested, containers) for nested in value]
    finally:
        containers.remove(marker)
    raise Q1AuditError(f"decoded checkpoint contains unsupported {type(value).__name__}")


def _namespace_identity_counters(value: object, namespace: str) -> tuple[set[int], bool]:
    counters: set[int] = set()
    malformed = False
    visited: set[int] = set()

    def visit(item: object) -> None:
        nonlocal malformed
        if isinstance(item, str):
            if item.startswith(f"{namespace}:"):
                suffix = item.rpartition(":")[2]
                counter = _identity_counter(item, namespace)
                if suffix.isdigit() and counter is None:
                    malformed = True
                elif counter is not None:
                    counters.add(counter)
            return
        if item is None or isinstance(item, (bool, int, float, bytes, bytearray)):
            return
        marker = id(item)
        if marker in visited:
            return
        visited.add(marker)
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif isinstance(item, Sequence):
            for nested in item:
                visit(nested)

    visit(value)
    return counters, malformed


def _identity_counter(value: object, namespace: str) -> int | None:
    if not isinstance(value, str) or not value.startswith(f"{namespace}:"):
        return None
    suffix = value.rpartition(":")[2]
    if not suffix.isdigit() or str(int(suffix)) != suffix:
        return None
    return int(suffix)


def _checkpoint_semantic_violations(
    restored: RestoredQ1Checkpoint,
    raw: Mapping[str, object],
) -> list[tuple[str, str]]:
    """Inspect full canonical acquisition records, not producer summaries."""

    found: list[tuple[str, str]] = []
    acquisition = _required_mapping(raw, "acquisition")
    candidate_rows = raw.get("candidate_rows")
    if not isinstance(candidate_rows, list):
        return [("Q1-K2", "checkpoint comparison lacks emitted candidate rows")]
    experience = restored.experience
    decision = experience.decision
    if decision is None or experience.execution is None:
        return [("Q1-K3", "checkpoint acquisition lacks canonical decision/execution")]
    found.extend(_identity_counter_violations(restored, raw))
    if decision.decision_id != acquisition.get("decision_id"):
        found.append(("Q1-K2", "checkpoint acquisition decision ID differs from raw trace"))
    if decision.intended_action.intention_id != acquisition.get("intention_id"):
        found.append(("Q1-K2", "checkpoint acquisition intention ID differs from raw trace"))
    if experience.execution.execution_id != acquisition.get("execution_id"):
        found.append(("Q1-K2", "checkpoint acquisition execution ID differs from raw trace"))
    run_id = raw.get("run_id")
    expected_episode_id = f"{run_id}:m{raw.get('master')}:a{raw.get('arm')}:e{raw.get('episode')}"
    expected_policy_version = f"wm002-q1:{raw.get('arm')}:policy-v1"
    if (
        experience.run_id != run_id
        or experience.task_id != "wm002-hidden-actuator"
        or experience.episode_id != expected_episode_id
        or experience.step_index != 0
        or experience.terminated
        or experience.truncated
        or experience.behavior_policy_version != expected_policy_version
        or decision.policy_version != expected_policy_version
    ):
        found.append(("Q1-K1", "checkpoint acquisition episode/policy ancestry differs from the frozen lane"))
    if experience.observation.observation_id != acquisition.get("observation_id"):
        found.append(("Q1-K1", "checkpoint acquisition observation ID differs from raw trace"))
    if experience.outcome.outcome_id != acquisition.get("outcome_id"):
        found.append(("Q1-K1", "checkpoint acquisition outcome ID differs from raw trace"))
    expected_observation = {
        "observed_symbol": acquisition.get("observed_symbol"),
        "phase": "acquisition",
        "semantic_action": acquisition.get("selected_action"),
    }
    if (
        experience.observation.modality != "wm002_acquisition_symbol"
        or experience.observation.evidence.payload != expected_observation
    ):
        found.append(("Q1-K1", "checkpoint acquisition observation payload differs from primitive raw trace"))
    outcome_payload = experience.outcome.evidence.payload
    expected_outcome = {
        "information_acquisition_cost": acquisition.get("information_acquisition_cost"),
        "net_reward": acquisition.get("net_reward"),
        "physical_action_cost": acquisition.get("physical_action_cost"),
        "task_payoff": acquisition.get("task_payoff"),
    }
    if outcome_payload != expected_outcome:
        found.append(("Q1-K1", "checkpoint acquisition outcome payload differs from primitive raw trace"))
    execution = experience.execution
    if (
        execution.intention is not decision.intended_action
        or execution.realized_action is not decision.intended_action.action
        or execution.status.value != "succeeded"
        or execution.deviation_reason
    ):
        found.append(("Q1-K1", "checkpoint acquisition execution is not the exact successful intended action"))
    accumulator_values = (
        (restored.accumulator.task_payoff, acquisition.get("task_payoff")),
        (restored.accumulator.physical_action_cost, acquisition.get("physical_action_cost")),
        (restored.accumulator.information_acquisition_cost, acquisition.get("information_acquisition_cost")),
        (restored.accumulator.net_acquisition_return, acquisition.get("net_reward")),
    )
    if any(not _same_number(expected, actual) for actual, expected in accumulator_values):
        found.append(("Q1-K1", "checkpoint accumulator differs from raw primitive acquisition values"))
    target = decision.goal.target
    if (
        target.target_id != _TARGET_ID
        or target.target_kind != _TARGET_KIND
        or target.description != _TARGET_DESCRIPTION
        or decision.belief.target is not target
    ):
        found.append(("Q1-K2", "acquisition decision does not reuse the exact frozen composite target"))
    if decision.belief.model_version != _INITIAL_MODEL_VERSION:
        found.append(("Q1-K3", "acquisition decision belief does not bind canonical prior model bytes"))
    if not _same_numeric_sequence(decision.belief.distribution.parameters, (0.5, 0.5)):
        found.append(("Q1-K3", "acquisition decision belief is not the exact symmetric prior"))

    alternatives = decision.alternatives
    if len(alternatives) != len(ACTION_ORDER) or len(candidate_rows) != len(ACTION_ORDER):
        found.append(("Q1-K2", "checkpoint acquisition decision does not contain exactly five candidates"))
    else:
        for ordinal, (action, assessment, row_value) in enumerate(
            zip(ACTION_ORDER, alternatives, candidate_rows, strict=True)
        ):
            if not isinstance(row_value, Mapping):
                found.append(("Q1-K2", f"candidate {ordinal} diagnostic row is not an object"))
                continue
            parameters = assessment.action.parameters
            if not isinstance(parameters, Mapping):
                found.append(("Q1-K2", f"core candidate {ordinal} action parameters are not an object"))
                continue
            expected_parameters = {
                "ordinal": ordinal,
                "phase": "acquisition",
                "semantic_action": action,
            }
            if (
                dict(parameters) != expected_parameters
                or assessment.action.action_id != f"acquisition:{ordinal:02d}:{action}"
                or assessment.action.action_kind != "wm002_acquisition"
            ):
                found.append(("Q1-K2", f"core candidate {ordinal} action identity/order differs"))
            if assessment.assessment_id != row_value.get("assessment_id"):
                found.append(("Q1-K2", f"core candidate {ordinal} is not linked to its diagnostic row"))
            expected_utility = Fraction(1, 2) + _EXPECTED_IMMEDIATE[action]
            expected_decision = _EXPECTED_DECISION_VALUE[action]
            numeric_checks = (
                (assessment.utility.expected_value, float(expected_utility), "utility"),
                (assessment.information_value.expected_reduction, float(expected_decision), "decision value"),
                (
                    assessment.information_value.expected_cost,
                    float(_INFORMATION_COST[action]),
                    "information cost",
                ),
                (assessment.expected_action_cost, float(_ACTION_COST[action]), "physical action cost"),
                (assessment.expected_risk, 0.0, "risk"),
                (assessment.constraint_penalty, 0.0, "constraint penalty"),
                (assessment.total_value, float(_EXPECTED_EPISODE_VALUE[action]), "truthful total"),
            )
            for actual, expected, label in numeric_checks:
                if not _same_number(actual, expected):
                    found.append(("Q1-K2", f"core candidate {ordinal} {label} differs"))
            if (
                assessment.unit != "return"
                or assessment.utility.unit != "return"
                or assessment.information_value.unit != "return"
                or not assessment.admissible
                or assessment.constraint_reasons
            ):
                found.append(("Q1-K2", f"core candidate {ordinal} unit/admissibility differs"))
            if (
                assessment.evaluator_version != _CANDIDATE_EVALUATOR_VERSION
                or assessment.utility.evaluator_version != _UTILITY_EVALUATOR_VERSION
                or assessment.information_value.evaluator_version != _INFORMATION_EVALUATOR_VERSION
                or assessment.utility.goal_id != decision.goal.goal_id
                or assessment.information_value.prior_belief_id != decision.belief.belief_id
                or assessment.information_value.action_id != assessment.action.action_id
                or assessment.information_value.target_id != target.target_id
            ):
                found.append(("Q1-K2", f"core candidate {ordinal} value provenance/linkage differs"))
            prediction = assessment.prediction
            expected_prediction: tuple[float, ...] = (
                (1.0,) if action == "skip" else ((0.25, 0.25, 0.25, 0.25) if action == "nuisance" else (0.5, 0.5))
            )
            if not _same_numeric_sequence(prediction.distribution.parameters, expected_prediction):
                found.append(("Q1-K2", f"core candidate {ordinal} prediction distribution differs"))
            if prediction.target is not target or prediction.prior_belief is not decision.belief:
                found.append(("Q1-K2", f"core candidate {ordinal} prediction linkage differs"))
            if (
                prediction.action is not assessment.action
                or prediction.model_version != _INITIAL_MODEL_VERSION
                or prediction.representation_version != _REPRESENTATION_VERSION
                or prediction.calibration_version != _CALIBRATION_VERSION
                or prediction.distribution.family != "categorical"
                or prediction.distribution.representation_version != _REPRESENTATION_VERSION
                or prediction.distribution.event_shape != (len(expected_prediction),)
                or prediction.uncertainties
            ):
                found.append(("Q1-K2", f"core candidate {ordinal} prediction provenance/shape differs"))

        selected_parameters = decision.selected_assessment.action.parameters
        if not isinstance(selected_parameters, Mapping):
            found.append(("Q1-K2", "core selected assessment parameters are not an object"))
        elif selected_parameters.get("semantic_action") != acquisition.get("selected_action"):
            found.append(("Q1-K2", "core selected assessment differs from executed acquisition action"))
        if decision.intended_action.action is not decision.selected_assessment.action:
            found.append(("Q1-K2", "core selected assessment is not the canonical intention action"))

    transition = restored.transition
    update = transition.belief_update
    if update.prior is not decision.belief or update.experience is not experience:
        found.append(("Q1-K3", "no-op assimilation does not share canonical prior/experience"))
    if update.updater_version != _ASSIMILATOR_VERSION:
        found.append(("Q1-K3", "acquisition used a non-frozen observation assimilator"))
    posterior_parameters = update.posterior.distribution.parameters
    if not isinstance(posterior_parameters, Sequence) or isinstance(posterior_parameters, (str, bytes, bytearray)):
        found.append(("Q1-K3", "no-op posterior parameters are not a numeric sequence"))
    elif not _same_numeric_sequence(update.prior.distribution.parameters, cast(Sequence[float], posterior_parameters)):
        found.append(("Q1-K3", "observation assimilation changed posterior probabilities"))
    if update.posterior.model_version != _INITIAL_MODEL_VERSION:
        found.append(("Q1-K3", "no-op assimilation changed the canonical prior model version"))
    prior_observation_ids = tuple(row.observation_id for row in update.prior.information_set.observations)
    observation_ids = tuple(row.observation_id for row in update.posterior.information_set.observations)
    if prior_observation_ids or observation_ids != (experience.observation.observation_id,):
        found.append(("Q1-K3", "no-op assimilation did not append exactly the executed observation to an empty prior"))
    if (
        update.posterior.target is not update.prior.target
        or update.posterior.distribution.family != "categorical"
        or update.posterior.distribution.support != "actuator_regime={-1,+1}"
        or update.posterior.distribution.event_shape != (2,)
        or update.posterior.representation_version != _REPRESENTATION_VERSION
        or update.posterior.distribution.representation_version != _REPRESENTATION_VERSION
    ):
        found.append(("Q1-K3", "no-op assimilation changed target or categorical representation semantics"))
    if len(transition.proper_scores) != 1:
        found.append(("Q1-K3", "acquisition transition does not contain exactly one proper score"))
    else:
        score = transition.proper_scores[0]
        selected = decision.selected_assessment.prediction.distribution.parameters
        if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes, bytearray)):
            found.append(("Q1-K3", "selected prediction parameters are not a numeric sequence"))
            selected_values: tuple[object, ...] = ()
        else:
            selected_values = tuple(selected)
        symbol = acquisition.get("observed_symbol")
        if acquisition.get("selected_action") == "nuisance":
            observed_index = symbol
        elif acquisition.get("selected_action") == "skip":
            observed_index = 0
        else:
            observed_index = 0 if symbol == -1 else 1
        expected_score = None
        if type(observed_index) is int and 0 <= observed_index < len(selected_values):
            expected_score = -math.log(cast(float, selected_values[observed_index]))
        if (
            score.scorer_version != _SCORER_VERSION
            or score.rule != "categorical_log_score"
            or score.unit != "nats"
            or expected_score is None
            or score.realized_evidence_id != experience.observation.evidence.evidence_id
            or not _same_number(score.value, expected_score)
        ):
            found.append(("Q1-K3", "acquisition proper score differs from independent recomputation"))
    if len(transition.effects) != 1:
        found.append(("Q1-K3", "acquisition transition does not contain exactly one no-op effect"))
    else:
        effect = transition.effects[0]
        if (
            effect.evaluator_version != _EFFECT_VERSION
            or effect.measure != "assimilation_only_categorical_entropy"
            or not _same_number(effect.before, math.log(2.0))
            or not _same_number(effect.after, math.log(2.0))
            or not _same_number(effect.improvement, 0.0)
            or effect.kind.value != "information_gain"
            or effect.higher_is_better is not False
            or effect.externally_calibrated is not False
            or effect.belief_update_id != update.update_id
            or effect.target_id != target.target_id
        ):
            found.append(("Q1-K3", "assimilation effect is not the independently verified zero effect"))

    try:
        if len(restored.experience_store) != 1:
            found.append(("Q1-K3", "restored experience store does not contain exactly the acquisition event"))
        elif restored.experience_store.get(experience.experience_id) is not experience:
            found.append(("Q1-K3", "restored store does not retain the canonical acquisition experience"))
        if restored.ledger.transition_count != 1 or restored.ledger.update_count != 1:
            found.append(("Q1-K3", "restored ledger is not exactly one acquisition transition and one update"))
        elif (
            restored.ledger.get_transition(transition.transition_id) is not transition
            or restored.ledger.get_update(restored.receipt.receipt_id) is not restored.receipt
        ):
            found.append(("Q1-K3", "restored ledger continuity does not share canonical acquisition records"))
        if restored.snapshot.latest_update is not restored.receipt:
            found.append(("Q1-K3", "restored pre-terminal snapshot does not bind the sole acquisition receipt"))
    except Exception as error:
        found.append(("Q1-K3", f"restored acquisition-only ledger continuity rejected: {error}"))

    receipt = restored.receipt
    resulting = receipt.resulting_belief
    expected_posterior = _parse_fraction(acquisition.get("posterior_after"))
    if (
        receipt.learner_version != _LEARNER_VERSION
        or receipt.status.value != "applied"
        or len(receipt.transitions) != 1
        or receipt.transitions[0] is not transition
    ):
        found.append(("Q1-K3", "receipt is not one sole applied acquisition learner update"))
    if (
        receipt.previous_model_version != _INITIAL_MODEL_VERSION
        or receipt.previous_configuration_version != _INITIAL_CONFIGURATION_VERSION
        or receipt.new_model_version != acquisition.get("model_after_version")
        or receipt.new_configuration_version
        != f"{_CONFIGURATION_VERSION_PREFIX}{acquisition.get('model_after_sha256')}"
        or receipt.previous_policy_version != receipt.new_policy_version
        or receipt.previous_representation_version != receipt.new_representation_version
    ):
        found.append(("Q1-K3", "receipt predecessor/resulting version lineage differs"))
    if resulting is None or expected_posterior is None:
        found.append(("Q1-K3", "applied receipt lacks a decodable resulting posterior belief"))
    elif resulting.model_version != receipt.new_model_version or not _same_numeric_sequence(
        resulting.distribution.parameters,
        (float(1 - expected_posterior), float(expected_posterior)),
    ):
        found.append(("Q1-K3", "receipt resulting belief differs from the exact learned posterior"))
    if expected_posterior is not None:
        before_entropy = math.log(2.0)
        after_entropy = _binary_entropy(float(expected_posterior))
        expected_metrics = {
            "posterior_direct_before": 0.5,
            "posterior_direct_after": float(expected_posterior),
            "entropy_before_nats": before_entropy,
            "entropy_after_nats": after_entropy,
            "entropy_reduction_nats": before_entropy - after_entropy,
            "consumed_transition_count": 1.0,
            "evidence_count_before": 0.0,
            "evidence_count_after": 1.0,
        }
        metrics = dict(receipt.metrics)
        if set(metrics) != set(expected_metrics) or any(
            not _same_number(metrics.get(name), value) for name, value in expected_metrics.items()
        ):
            found.append(("Q1-K3", "receipt metrics differ from independent exact recomputation"))
    if restored.model_predecessor_sha256 != _INITIAL_MODEL_SHA256:
        found.append(("Q1-K3", "checkpoint predecessor does not bind canonical prior model bytes"))
    return found


def _audit_restored_parity(
    *,
    key: tuple[int, str, int],
    raw: Mapping[str, object],
    restored: Mapping[str, object],
    violations: _Violations,
) -> None:
    try:
        checkpoint = _required_mapping(raw, "checkpoint")
        acquisition = _required_mapping(raw, "acquisition")
        terminal = _required_mapping(raw, "terminal")
        for field in (
            "protocol_version",
            "run_id",
            "attempt_id",
            "protocol_sha256",
            "implementation_sha256",
            "q0_report_sha256",
            "entry_qualification_sha256",
            "salt_commitment_sha256",
            "master",
            "arm",
            "episode",
            "producer_pid",
        ):
            if restored.get(field) != raw.get(field):
                raise Q1AuditError(f"restored/live binding mismatch:{field}")
        if restored.get("restorer_pid") == raw.get("producer_pid"):
            raise Q1AuditError("restorer PID equals producer PID")
        expected = {
            "checkpoint_sha256": checkpoint.get("sha256"),
            "component_sha256": checkpoint.get("component_sha256"),
            "model_sha256": acquisition.get("model_after_sha256"),
            "model_version": acquisition.get("model_after_version"),
            "posterior_direct": acquisition.get("posterior_after"),
            "acquisition_experience_id": acquisition.get("experience_id"),
            "acquisition_transition_id": acquisition.get("transition_id"),
            "acquisition_receipt_id": acquisition.get("receipt_id"),
            "identity_counter_sha256": cast(Mapping[str, object], checkpoint.get("component_sha256", {})).get(
                "identity_counter"
            ),
            "terminal_candidate_rows": terminal.get("candidate_rows"),
            "terminal_candidate_rows_sha256": terminal.get("candidate_rows_sha256"),
            "selected_terminal_action": terminal.get("selected_action"),
            "terminal_success": terminal.get("success"),
            "episode_return": terminal.get("episode_return"),
        }
        for restored_field, raw_field in (
            ("terminal_decision_id", "decision_id"),
            ("terminal_intention_id", "intention_id"),
            ("terminal_execution_id", "execution_id"),
            ("terminal_observation_id", "observation_id"),
            ("terminal_outcome_id", "outcome_id"),
            ("terminal_experience_id", "experience_id"),
            ("terminal_transition_id", "transition_id"),
        ):
            expected[restored_field] = terminal.get(raw_field)
        for field, value in expected.items():
            if restored.get(field) != value:
                raise Q1AuditError(f"restored/live parity mismatch:{field}")
        model_digest = acquisition.get("model_after_sha256")
        expected_configuration = f"wm002-config-sha256:{model_digest}"
        if restored.get("configuration_version") != expected_configuration:
            raise Q1AuditError("restored configuration version does not bind model digest")
    except Exception as error:
        violations.add("Q1-K5", f"{_render_key(key)} restored parity rejected: {type(error).__name__}:{error}")


def _recompute_arm_means(
    returns: Mapping[tuple[int, str], Sequence[float]],
    violations: _Violations,
) -> tuple[list[dict[str, object]], dict[tuple[int, str], float]]:
    rows: list[dict[str, object]] = []
    means: dict[tuple[int, str], float] = {}
    for master in range(MASTER_COUNT):
        for arm in ARM_ORDER:
            values = tuple(returns.get((master, arm), ()))
            if len(values) != EPISODES_PER_MASTER:
                violations.add(
                    "Q1-K4",
                    f"master={master},arm={arm} has {len(values)} reconstructible returns; "
                    f"expected {EPISODES_PER_MASTER}",
                )
                mean = 0.0
            else:
                mean = math.fsum(values) / EPISODES_PER_MASTER
            means[(master, arm)] = mean
            rows.append(
                {
                    "arm": arm,
                    "episode_count": len(values),
                    "master": master,
                    "mean_return": mean,
                }
            )
    return rows, means


def _comparison_rows(arm_means: Mapping[tuple[int, str], float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control in CONTROL_ARMS:
        differences = [
            arm_means[(master, "prospect_expected_return")] - arm_means[(master, control)]
            for master in range(MASTER_COUNT)
        ]
        mean = math.fsum(differences) / MASTER_COUNT
        sum_squares = math.fsum((value - mean) ** 2 for value in differences)
        sample_variance = sum_squares / (MASTER_COUNT - 1)
        standard_error = math.sqrt(sample_variance) / math.sqrt(MASTER_COUNT)
        margin = T_CRITICAL_DF3 * standard_error
        lower = mean - margin
        upper = mean + margin
        rows.append(
            {
                "ci95_lower": lower,
                "ci95_upper": upper,
                "control_arm": control,
                "master_differences": differences,
                "mean_difference": mean,
                "passed": mean > 0.0 and lower > 0.0,
            }
        )
    return rows


def _recompute_comparisons(
    returns: Mapping[tuple[int, str], Sequence[float]],
    violations: _Violations,
) -> list[dict[str, object]]:
    _, means = _recompute_arm_means(returns, violations)
    return _comparison_rows(means)


def _validate_producer_aggregate(
    aggregate: Mapping[str, object] | None,
    *,
    counts: Mapping[str, int],
    arm_means: Sequence[Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
    violations: _Violations,
) -> None:
    if aggregate is None:
        return
    expected_counts = {
        "acquisition_updates": counts["acquisition_updates"],
        "arms": counts["arms"],
        "checkpoints": counts["checkpoint_frames"],
        "environment_steps": counts["environment_steps"],
        "episodes": counts["episodes"],
        "masters": counts["masters"],
        "restores": counts["restored_rows"],
        "terminal_updates": counts["terminal_updates"],
        "transitions": counts["transitions"],
    }
    if aggregate.get("counts") != expected_counts:
        violations.add("Q1-K0", "producer aggregate counts differ from independent primitive accounting")

    actual_arm_means = aggregate.get("arm_means")
    if not isinstance(actual_arm_means, Sequence) or isinstance(actual_arm_means, (str, bytes, bytearray)):
        violations.add("Q1-K4", "producer aggregate arm_means is not an array")
    else:
        actual_rows = tuple(actual_arm_means)
        if len(actual_rows) != len(arm_means):
            violations.add("Q1-K4", "producer aggregate arm mean count differs")
        else:
            for ordinal, (actual, expected) in enumerate(zip(actual_rows, arm_means, strict=True)):
                if not isinstance(actual, Mapping) or (
                    actual.get("master") != expected.get("master")
                    or actual.get("arm") != expected.get("arm")
                    or actual.get("episode_count") != expected.get("episode_count")
                    or not _same_number(actual.get("mean_return"), cast(float, expected.get("mean_return")))
                ):
                    violations.add("Q1-K4", f"producer aggregate arm mean {ordinal} differs from recomputation")

    actual_comparisons = aggregate.get("comparisons")
    if not isinstance(actual_comparisons, Sequence) or isinstance(actual_comparisons, (str, bytes, bytearray)):
        violations.add("Q1-K4", "producer aggregate comparisons is not an array")
    else:
        actual_rows = tuple(actual_comparisons)
        if len(actual_rows) != len(comparisons):
            violations.add("Q1-K4", "producer aggregate comparison count differs")
        else:
            for ordinal, (actual, expected) in enumerate(zip(actual_rows, comparisons, strict=True)):
                if not isinstance(actual, Mapping) or (
                    actual.get("control_arm") != expected.get("control_arm")
                    or not _same_numeric_sequence(
                        actual.get("master_differences"), cast(Sequence[float], expected["master_differences"])
                    )
                    or not _same_number(actual.get("mean_difference"), cast(float, expected["mean_difference"]))
                    or not _same_number(actual.get("ci95_lower"), cast(float, expected["ci95_lower"]))
                    or not _same_number(actual.get("ci95_upper"), cast(float, expected["ci95_upper"]))
                ):
                    violations.add("Q1-K4", f"producer aggregate comparison {ordinal} differs from recomputation")


def _private_key(namespace: str, *fields: tuple[str, int | str]) -> bytes:
    allowed = {
        _THETA_NAMESPACE,
        _PULSE_NAMESPACE,
        _NUISANCE_NAMESPACE,
        _TERMINAL_NAMESPACE,
    }
    if namespace not in allowed:
        raise Q1AuditError("private semantic key uses an undeclared namespace")
    segments = [namespace]
    for name, value in fields:
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz_" for character in name):
            raise Q1AuditError("private semantic key field name is not canonical")
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise Q1AuditError("private semantic key field value is not canonical")
        rendered = str(value)
        if not rendered or any(character in rendered for character in "|=\r\n\t "):
            raise Q1AuditError("private semantic key field cannot be delimited")
        segments.append(f"{name}={rendered}")
    return (_PRIVATE_KEY_PREFIX + "|".join(segments)).encode("ascii")


def _require_master_episode(master: object, episode: object) -> None:
    if type(master) is not int or not 0 <= master < MASTER_COUNT:
        raise Q1AuditError("master is outside [0,3]")
    if type(episode) is not int or not 0 <= episode < EPISODES_PER_MASTER:
        raise Q1AuditError("episode is outside [0,1023]")


def _global_private_values(
    schedule: _IndependentSeedSchedule,
    secret_salt: bytes,
) -> tuple[bytes, ...]:
    values: set[bytes] = {secret_salt}
    reference_arm = ARM_ORDER[0]
    for master in range(MASTER_COUNT):
        for episode in range(EPISODES_PER_MASTER):
            values.update(schedule.private_hmac_digests(master, episode, reference_arm))
    return tuple(sorted(values))


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise Q1AuditError(f"{key} must be an object")
    return cast(Mapping[str, object], nested)


def _required_str(value: Mapping[str, object], key: str) -> str:
    nested = value.get(key)
    if not isinstance(nested, str) or not nested:
        raise Q1AuditError(f"{key} must be a nonempty string")
    return nested


def _required_int(value: Mapping[str, object], key: str) -> int:
    nested = value.get(key)
    if type(nested) is not int:
        raise Q1AuditError(f"{key} must be a canonical integer")
    return nested


def _parse_fraction(value: object) -> Fraction | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if str(parsed) != value:
        return None
    return parsed


def _same_number(value: object, expected: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and math.isclose(numeric, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)


def _same_numeric_sequence(actual: object, expected: Sequence[float]) -> bool:
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
        return False
    values = tuple(actual)
    return len(values) == len(expected) and all(
        _same_number(value, target) for value, target in zip(values, expected, strict=True)
    )


def _binary_entropy(probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise Q1AuditError("binary entropy probability is outside [0,1]")
    terms = (probability, 1.0 - probability)
    return -math.fsum(value * math.log(value) for value in terms if value > 0.0)


def _integer_or_zero(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _render_key(key: tuple[int, str, int] | None) -> str:
    if key is None:
        return "<unknown-episode>"
    return f"master={key[0]},arm={key[1]},episode={key[2]}"


def _reject_duplicate_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise Q1AuditError("duplicate JSON object keys are forbidden")
        value[key] = item
    return value


def _reject_nonfinite(token: str) -> object:
    raise Q1AuditError(f"non-finite JSON token {token!r} is forbidden")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--secret-salt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Q1_PROTOCOL_PATH)
    parser.add_argument("--attempt-marker", type=Path, required=True)
    parser.add_argument("--q0-report", type=Path, required=True)
    parser.add_argument("--entry-report", type=Path, required=True)
    parser.add_argument("--prospective-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _require_no_site_cli() -> None:
    if sys.flags.no_site != 1:
        raise SystemExit("Q1 independent audit requires invocation with Python -S")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the independent auditor from separately supplied bindings."""

    _require_no_site_cli()
    arguments = _build_parser().parse_args(argv)
    artifact = audit_q1_directory(
        arguments.output_directory,
        secret_salt_path=arguments.secret_salt,
        attempt_marker_path=arguments.attempt_marker,
        protocol_path=arguments.protocol,
        q0_report_path=arguments.q0_report,
        entry_report_path=arguments.entry_report,
        prospective_review_path=arguments.prospective_review,
    )
    write_audit_artifact(
        arguments.output,
        artifact,
        secret_salt_path=arguments.secret_salt,
        audited_directory=arguments.output_directory,
        protected_input_paths=(
            arguments.protocol,
            arguments.attempt_marker,
            arguments.q0_report,
            arguments.entry_report,
            arguments.prospective_review,
        ),
    )
    return 0 if artifact["passed"] is True else 1


if __name__ == "__main__":  # pragma: no cover - exercised by operator workflow
    raise SystemExit(main())


__all__ = (
    "AUDIT_SCHEMA",
    "CHECKPOINT_FRAMES_FILENAME",
    "CHECKPOINT_INDEX_FILENAME",
    "PRIVATE_AUDIT_FILENAME",
    "PRODUCER_AGGREGATE_FILENAME",
    "Q1AuditError",
    "RAW_TRACE_FILENAME",
    "RESTORED_TRACE_FILENAME",
    "Q1_AUDIT_IMPLEMENTATION_PATHS",
    "audit_q1_directory",
    "development_audit_artifact_sample",
    "main",
    "write_audit_artifact",
)
