"""Result-free entry qualification for the WM-002 Q1 runtime.

The report produced here is an execution precondition, not a Q1 result.  Its
synthetic probes use no Q1 master/episode schedule and do not count toward, or
authorize reinterpretation of, the permanently claim-ineligible Q1 budget.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import shutil
import stat
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Final

from bench.active_acquisition.contracts import (
    ACTION_ORDER,
    ARM_ORDER,
    CHECKPOINT_COMPONENTS,
    MACHINE_GENERATED_REVIEWER_MARK,
    Q0_IMPLEMENTATION_SHA256,
    Q0_PROTOCOL_SHA256,
    Q0_REPORT_SHA256,
    Q1_PROTOCOL_PATH,
    Q1_PROTOCOL_VERSION,
    Q1_SCHEMA_PATHS,
    TERMINAL_ORDER,
    ManifestEntry,
    canonical_json_bytes,
    implementation_manifest,
    sha256_bytes,
)
from bench.active_acquisition.seeding import Q1ExecutionMode

ENTRY_REPORT_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-entry-qualification.v1"
ENTRY_REPORT_SCHEMA_PATH: Final = Path(__file__).with_name("schemas") / "q1-entry-qualification.schema.json"
PROSPECTIVE_REVIEW_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-prospective-review.v1"
PROSPECTIVE_REVIEW_SCHEMA_PATH: Final = Path(__file__).with_name("schemas") / "q1-prospective-review.schema.json"
Q1_JSONSCHEMA_VERSION: Final = "4.25.1"
_Q1_NORMALIZED_PROTOCOL_SHA256: Final = "e015c8b519a10eeab19ecda1384071b17c450484c06ebbd4142d95143c92353b"
_Q1_PRETERMINAL_IDENTITY_COUNTER: Final = 43
_MAX_QUALIFICATION_INPUT_BYTES: Final = 8 * 1024 * 1024
_QUALIFICATION_READ_CHUNK_BYTES: Final = 1024 * 1024
_UNDECLARED_HIDDEN_STATE_SENTINEL_KEY: Final = "__wm002_undeclared_hidden_state_sentinel__"
ENTRY_CHECK_ORDER: Final = (
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
PROSPECTIVE_REVIEW_SCOPE: Final = (
    "accepted_q0_and_successor_authority",
    "runtime_semantics_and_transactional_causality",
    "private_seed_exactness_and_noninterference",
    "checkpoint_and_fresh_process_restore",
    "artifact_schemas_attempt_integrity_and_resources",
    "independent_auditor_recomputation_and_scale",
    "evidence_and_claim_boundary",
)

Q1_IMPLEMENTATION_PATHS: Final = (
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


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    """One fail-closed entry predicate."""

    name: str
    passed: bool
    violations: tuple[str, ...]
    summary: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "summary": self.summary,
            "violations": list(self.violations),
        }


@dataclass(frozen=True, slots=True)
class Q1DirectoryIdentity:
    """Stable identity of one private Q1 directory."""

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
class Q1ResourcePreflight:
    """Deterministic report of the result-free execution-capacity gate."""

    execution_root: Q1DirectoryIdentity | None
    attempt_registry_directory: Q1DirectoryIdentity | None
    raw_trace_max_bytes: int
    private_audit_max_bytes: int
    checkpoint_index_max_bytes: int
    checkpoint_frame_max_bytes: int
    restored_trace_max_bytes: int
    estimated_canonical_bytes: int
    required_free_bytes: int
    max_restore_concurrency: int
    sampled_arms: int
    probe_duration_under_30_seconds: bool
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_registry_directory": (
                None if self.attempt_registry_directory is None else self.attempt_registry_directory.as_dict()
            ),
            "checkpoint_frame_max_bytes": self.checkpoint_frame_max_bytes,
            "checkpoint_index_max_bytes": self.checkpoint_index_max_bytes,
            "estimated_canonical_bytes": self.estimated_canonical_bytes,
            "execution_root": None if self.execution_root is None else self.execution_root.as_dict(),
            "max_restore_concurrency": self.max_restore_concurrency,
            "passed": self.passed,
            "private_audit_max_bytes": self.private_audit_max_bytes,
            "probe_duration_under_30_seconds": self.probe_duration_under_30_seconds,
            "raw_trace_max_bytes": self.raw_trace_max_bytes,
            "required_free_bytes": self.required_free_bytes,
            "restored_trace_max_bytes": self.restored_trace_max_bytes,
            "sampled_arms": self.sampled_arms,
        }


@dataclass(frozen=True, slots=True)
class Q1EntryQualificationReport:
    """Canonical immutable authorization precondition."""

    protocol_version: str
    protocol_sha256: str
    q0_report_sha256: str
    q0_protocol_sha256: str
    q0_implementation_sha256: str
    implementation_sha256: str
    implementation_manifest: tuple[ManifestEntry, ...]
    dependency_versions: tuple[tuple[str, str], ...]
    schema_sha256: tuple[tuple[str, str], ...]
    salt_commitment_sha256: str
    prospective_review_sha256: str
    interpreter_identity: str
    q1_environment_interactions: int
    q1_private_draws: int
    synthetic_development_interactions: int
    resource_preflight: Q1ResourcePreflight
    claim_eligible: bool
    formal_authorized: bool
    checks: tuple[QualificationCheck, ...]
    passed: bool
    schema: str = ENTRY_REPORT_SCHEMA

    def as_dict(self) -> dict[str, object]:
        return {
            "checks": [row.as_dict() for row in self.checks],
            "claim_eligible": self.claim_eligible,
            "formal_authorized": self.formal_authorized,
            "dependency_versions": dict(self.dependency_versions),
            "implementation_manifest": [row.as_dict() for row in self.implementation_manifest],
            "implementation_sha256": self.implementation_sha256,
            "interpreter_identity": self.interpreter_identity,
            "passed": self.passed,
            "prospective_review_sha256": self.prospective_review_sha256,
            "protocol_sha256": self.protocol_sha256,
            "protocol_version": self.protocol_version,
            "q0_implementation_sha256": self.q0_implementation_sha256,
            "q0_protocol_sha256": self.q0_protocol_sha256,
            "q0_report_sha256": self.q0_report_sha256,
            "q1_environment_interactions": self.q1_environment_interactions,
            "q1_private_draws": self.q1_private_draws,
            "resource_preflight": self.resource_preflight.as_dict(),
            "salt_commitment_sha256": self.salt_commitment_sha256,
            "schema": self.schema,
            "schema_sha256": dict(self.schema_sha256),
            "synthetic_development_interactions": self.synthetic_development_interactions,
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.as_dict()).decode("ascii")


def _metadata_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_regular_bytes(
    path: Path,
    *,
    label: str,
    private: bool = False,
    max_bytes: int = _MAX_QUALIFICATION_INPUT_BYTES,
) -> bytes:
    """Read one bounded, stable, one-link regular-file descriptor."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("qualification input byte bound must be positive")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be a regular non-symlink file")
            if before.st_nlink != 1:
                raise ValueError(f"{label} must have exactly one hard link")
            if private and stat.S_IMODE(before.st_mode) != 0o600:
                raise ValueError(f"{label} permissions must be exactly 0600")
            if before.st_size > max_bytes:
                raise ValueError(f"{label} exceeds the bounded document limit")

            chunks: list[bytes] = []
            total = 0
            while True:
                remaining = max_bytes + 1 - total
                chunk = os.read(
                    descriptor,
                    min(_QUALIFICATION_READ_CHUNK_BYTES, remaining),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"{label} exceeds the bounded document limit")
            after = os.fstat(descriptor)
        except OSError as error:
            raise ValueError(f"{label} descriptor read failed") from error
    finally:
        os.close(descriptor)

    if after.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    if _metadata_signature(before) != _metadata_signature(after) or total != after.st_size:
        raise ValueError(f"{label} changed during its descriptor read")
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} path cannot be verified after its descriptor read") from error
    if _metadata_signature(after) != _metadata_signature(path_after):
        raise ValueError(f"{label} path changed during its descriptor read")
    return b"".join(chunks)


def _reject_nonfinite_json(_token: str) -> object:
    raise ValueError("non-finite JSON value is forbidden")


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key is forbidden")
        value[key] = item
    return value


def _read_stable_json_document(
    path: Path,
    *,
    label: str,
    require_canonical: bool,
    max_bytes: int = _MAX_QUALIFICATION_INPUT_BYTES,
) -> tuple[str, object]:
    """Digest and decode the same bounded stable descriptor payload."""

    payload = _read_stable_regular_bytes(
        path,
        label=label,
        max_bytes=max_bytes,
    )
    digest = sha256_bytes(payload)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid finite UTF-8 JSON") from error
    if require_canonical and payload != canonical_json_bytes(value, newline=True):
        raise ValueError(f"{label} is not one canonical JSON document")
    return digest, value


def _normalized_protocol_contract_sha256(protocol: Mapping[str, object]) -> str:
    """Hash every semantic protocol field except the authorization-bit value."""

    if not isinstance(protocol, Mapping):
        raise ValueError("Q1 protocol contract must be an object")
    experiment = protocol.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("Q1 protocol contract has no experiment object")
    execution_authorized = experiment.get("execution_authorized")
    if type(execution_authorized) is not bool:
        raise ValueError("Q1 protocol execution_authorized must be exactly boolean")
    normalized_protocol = dict(protocol)
    normalized_experiment = dict(experiment)
    normalized_experiment["execution_authorized"] = False
    normalized_protocol["experiment"] = normalized_experiment
    return sha256_bytes(canonical_json_bytes(normalized_protocol))


def _protocol_snapshot(path: Path) -> tuple[str, Mapping[str, object]]:
    digest, value = _read_stable_json_document(
        path,
        label="Q1 protocol",
        require_canonical=False,
    )
    if not isinstance(value, dict):
        raise ValueError("Q1 protocol must be a JSON object")
    experiment = value.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Q1 protocol has no experiment object")
    if experiment.get("protocol_version") != Q1_PROTOCOL_VERSION:
        raise ValueError("Q1 protocol version mismatch")
    if value.get("schema") != "prospect.wm002.active-acquisition.q1-protocol.v1":
        raise ValueError("Q1 protocol schema mismatch")
    return digest, value


def _load_schema_snapshot(path: Path, *, label: str) -> Mapping[str, object]:
    _digest, value = _read_stable_json_document(
        path,
        label=label,
        require_canonical=False,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(value)
    return value


def _artifact_schema_snapshots() -> tuple[dict[str, Mapping[str, object]], tuple[tuple[str, str], ...]]:
    schemas: dict[str, Mapping[str, object]] = {}
    digests: list[tuple[str, str]] = []
    from jsonschema import Draft202012Validator

    for name, path in sorted(Q1_SCHEMA_PATHS.items()):
        digest, value = _read_stable_json_document(
            path,
            label=f"{name} artifact schema",
            require_canonical=False,
        )
        if not isinstance(value, dict):
            raise ValueError(f"{name} artifact schema must be a JSON object")
        Draft202012Validator.check_schema(value)
        schemas[name] = value
        digests.append((name, digest))
    return schemas, tuple(digests)


def _schema_accepts(schema: Mapping[str, object], value: object) -> bool:
    from jsonschema import Draft202012Validator

    return next(Draft202012Validator(schema).iter_errors(value), None) is None


def _read_private_salt(path: Path) -> bytes:
    """Read one bounded private salt through the descriptor that was checked."""

    salt = _read_stable_regular_bytes(
        path,
        label="secret salt",
        private=True,
    )
    if len(salt) < 32:
        raise ValueError("secret salt must contain at least 32 bytes")
    return salt


def run_entry_qualification(
    *,
    q0_report_path: Path,
    secret_salt_path: Path,
    prospective_review_path: Path,
    execution_root: Path,
    attempt_registry_directory: Path,
    protocol_path: Path = Q1_PROTOCOL_PATH,
    execution_mode: Q1ExecutionMode = Q1ExecutionMode.PRODUCTION,
) -> Q1EntryQualificationReport:
    """Run all result-free entry checks against exact external inputs.

    ``execution_mode`` selects the required protocol authorization bit and is
    never inferred from the document: production demands
    ``execution_authorized: true`` and rehearsal demands ``false``, so one
    protocol document can qualify for exactly one mode. The emitted report
    carries no separate mode field because ``protocol_sha256`` already binds it.
    """

    checks: list[QualificationCheck] = []
    protocol_sha256, protocol = _protocol_snapshot(protocol_path)
    artifact_schemas, artifact_schema_sha256 = _artifact_schema_snapshots()

    q0_violations = _q0_binding_violations(q0_report_path, protocol)
    checks.append(
        _check(
            "accepted_q0_binding",
            q0_violations,
            "The canonical independently accepted Q0 report and its protocol/implementation identities match.",
        )
    )

    protocol_violations = _protocol_boundary_violations(protocol, execution_mode=execution_mode)
    checks.append(
        _check(
            "successor_protocol_boundary",
            protocol_violations,
            "The successor protocol explicitly authorizes only claim-ineligible Q1 after this entry report passes.",
        )
    )

    parity_violations = _protocol_code_parity_violations(protocol)
    checks.append(
        _check(
            "protocol_code_parity",
            parity_violations,
            "Candidate, target, arm, seed, checkpoint, and exact budget constants match executable code.",
        )
    )

    schema_violations = _schema_violations(artifact_schemas)
    checks.append(
        _check(
            "artifact_schema_contract",
            schema_violations,
            "All six strict schemas pass metaschema checks and reject undeclared top-level private fields.",
        )
    )

    manifest: tuple[ManifestEntry, ...] = ()
    implementation_sha256 = ""
    manifest_violations: list[str] = []
    try:
        manifest, implementation_sha256 = implementation_manifest(Q1_IMPLEMENTATION_PATHS)
    except Exception as error:
        manifest_violations.append(f"{type(error).__name__}:{error}")
    checks.append(
        _check(
            "selected_source_implementation_binding",
            manifest_violations,
            "Every executable Q1 and imported Prospect source is content-bound in a sorted manifest.",
        )
    )

    salt_commitment = ""
    salt_violations: list[str] = []
    try:
        salt = _read_private_salt(secret_salt_path)
        salt_commitment = hashlib.sha256(salt).hexdigest()
    except Exception as error:
        salt_violations.append(f"{type(error).__name__}:{error}")
    checks.append(
        _check(
            "private_salt_custody",
            salt_violations,
            "A >=256-bit private salt exists outside artifacts and is represented publicly only by its commitment.",
        )
    )

    probe_violations: list[str] = []
    synthetic_interactions = 0
    samples: Mapping[str, object] = {}
    resource_samples: Mapping[str, object] = {}
    if not manifest_violations and not salt_violations:
        try:
            from bench.active_acquisition.q1 import run_development_qualification_probe

            probe = run_development_qualification_probe(
                protocol_sha256=protocol_sha256,
                implementation_sha256=implementation_sha256,
                q0_report_sha256=Q0_REPORT_SHA256,
                salt_commitment_sha256=salt_commitment,
            )
            synthetic_interactions = probe.synthetic_interactions
            samples = probe.artifact_samples
            resource_samples = probe.resource_samples
            probe_violations.extend(probe.violations)
        except Exception as error:
            probe_violations.append(f"{type(error).__name__}:{error}")
    else:
        probe_violations.append("probe skipped because implementation or salt binding failed")
    checks.append(
        _check(
            "authoritative_runtime_checkpoint_privacy_probe",
            probe_violations,
            (
                "Synthetic development paths cover all arms, exact learning, privacy noninterference, "
                "checkpoint restore, and fresh-process replay without a Q1 draw."
            ),
        )
    )

    resource_preflight, resource_violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
        resource_samples=resource_samples,
    )
    checks.append(
        _check(
            "execution_resource_and_single_attempt_preflight",
            resource_violations,
            (
                "The private execution root, external attempt registry, actual-filesystem atomic "
                "no-replace publication support, bounded concurrency, sampled artifact sizes, "
                "disk headroom, descriptor limit, and result-free probe capacity are sufficient."
            ),
        )
    )

    emitted_schema_violations = _emitted_artifact_schema_violations(
        samples,
        schemas=artifact_schemas,
    )
    checks.append(
        _check(
            "emitted_artifact_schema_and_mutation_probe",
            emitted_schema_violations,
            "Representative emitted values satisfy every frozen schema and reject an injected hidden-state field.",
        )
    )

    review_sha256 = ""
    review_violations: list[str] = []
    if not manifest_violations:
        review_sha256, review_violations = _prospective_review_violations(
            prospective_review_path,
            protocol_sha256=protocol_sha256,
            implementation_sha256=implementation_sha256,
            reviewed_source_count=len(manifest),
            execution_mode=execution_mode,
        )
    else:
        review_violations.append("review cannot bind a missing implementation digest")
    checks.append(
        _check(
            "independent_prospective_review",
            review_violations,
            "A separate reviewer found no blocking implementation/protocol discrepancy on this exact source binding.",
        )
    )

    emitted_check_order = tuple(row.name for row in checks)
    if emitted_check_order != ENTRY_CHECK_ORDER:
        raise RuntimeError(f"emitted entry check order differs from canonical contract: {emitted_check_order!r}")
    passed = all(row.passed for row in checks)
    report = Q1EntryQualificationReport(
        protocol_version=Q1_PROTOCOL_VERSION,
        protocol_sha256=protocol_sha256,
        q0_report_sha256=Q0_REPORT_SHA256,
        q0_protocol_sha256=Q0_PROTOCOL_SHA256,
        q0_implementation_sha256=Q0_IMPLEMENTATION_SHA256,
        implementation_sha256=implementation_sha256,
        implementation_manifest=manifest,
        dependency_versions=(("jsonschema", package_version("jsonschema")),),
        schema_sha256=artifact_schema_sha256,
        salt_commitment_sha256=salt_commitment,
        prospective_review_sha256=review_sha256,
        interpreter_identity=f"{platform.python_implementation()} {platform.python_version()} ({sys.executable})",
        q1_environment_interactions=0,
        q1_private_draws=0,
        synthetic_development_interactions=synthetic_interactions,
        resource_preflight=resource_preflight,
        claim_eligible=False,
        formal_authorized=False,
        checks=tuple(checks),
        passed=passed,
    )
    _validate_entry_report_against_snapshots(
        report.as_dict(),
        protocol_sha256=protocol_sha256,
        artifact_schemas=artifact_schemas,
        artifact_schema_sha256=dict(artifact_schema_sha256),
    )
    return report


def _q0_binding_violations(
    path: Path,
    protocol: Mapping[str, object],
) -> list[str]:
    violations: list[str] = []
    try:
        digest, report = _read_stable_json_document(
            path,
            label="Q0 report",
            require_canonical=True,
        )
        if digest != Q0_REPORT_SHA256:
            violations.append("Q0 report digest mismatch")
        if not isinstance(report, dict):
            raise TypeError("Q0 report must be an object")
        expected = {
            "schema": "prospect.wm002.active-acquisition.q0-qualification.v1",
            "protocol_sha256": Q0_PROTOCOL_SHA256,
            "implementation_sha256": Q0_IMPLEMENTATION_SHA256,
            "passed": True,
            "claim_eligible": False,
            "formal_authorized": False,
            "environment_interactions": 0,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                violations.append(f"Q0 field mismatch:{key}")
        binding = protocol.get("q0_binding")
        if not isinstance(binding, dict):
            violations.append("protocol q0_binding missing")
        else:
            for key, value in (
                ("report_sha256", Q0_REPORT_SHA256),
                ("protocol_sha256", Q0_PROTOCOL_SHA256),
                ("implementation_sha256", Q0_IMPLEMENTATION_SHA256),
            ):
                if binding.get(key) != value:
                    violations.append(f"protocol Q0 binding mismatch:{key}")
    except Exception as error:
        violations.append(f"Q0 report validation failed:{type(error).__name__}")
    return violations


def _protocol_boundary_violations(
    protocol: Mapping[str, object],
    *,
    execution_mode: Q1ExecutionMode,
) -> list[str]:
    violations: list[str] = []
    if sys.flags.optimize != 0:
        violations.append("optimized Python interpreter is forbidden")
    experiment = protocol.get("experiment")
    formal = protocol.get("formal_boundary")
    entry = protocol.get("entry_qualification")
    if not isinstance(experiment, dict):
        return ["experiment object missing"]
    expected = {
        "protocol_version": Q1_PROTOCOL_VERSION,
        "claim_eligible": False,
        "formal_authorized": False,
        "execution_authorized": execution_mode is Q1ExecutionMode.PRODUCTION,
    }
    for key, value in expected.items():
        actual = experiment.get(key)
        if type(actual) is not type(value) or actual != value:
            violations.append(f"experiment boundary mismatch:{key}")
    if not isinstance(formal, dict) or formal.get("authorized") is not False:
        violations.append("formal boundary is not explicitly disabled")
    if not isinstance(entry, dict) or entry.get("must_pass_before_first_q1_draw") is not True:
        violations.append("passing entry report is not a hard pre-draw condition")
    return violations


def _protocol_code_parity_violations(protocol: Mapping[str, object]) -> list[str]:
    violations: list[str] = []
    try:
        if _normalized_protocol_contract_sha256(protocol) != _Q1_NORMALIZED_PROTOCOL_SHA256:
            violations.append("normalized whole-document protocol contract mismatch")
    except (TypeError, ValueError) as error:
        violations.append(f"normalized whole-document protocol contract rejected:{type(error).__name__}:{error}")
    try:
        from bench.active_acquisition.checkpoint import Q1_COMPONENT_IDS, Q1_MAX_IDENTITY_NEXT_COUNTER
        from bench.active_acquisition.oracle import FractionOracle
        from bench.active_acquisition.policies import SHUFFLED_INFORMATION_SOURCE_BY_ACTION
        from bench.active_acquisition.problem import ACQUISITION_ACTIONS, HiddenActuatorProblem
        from bench.active_acquisition.q1 import (
            _EXPECTED_ACTION_BY_ARM,
            _WORKER_CAPTURE_FINISH_TIMEOUT_SECONDS,
            _WORKER_CAPTURE_TAIL_BYTES,
            ENVIRONMENT_STEPS_TOTAL,
            EPISODES_TOTAL,
            MAX_RESTORE_CONCURRENCY,
            PROCESS_TERMINATE_GRACE_SECONDS,
            PRODUCER_STAGE_TIMEOUT_SECONDS,
            RESTORE_CHILD_TIMEOUT_SECONDS,
            RESTORE_STAGE_TIMEOUT_SECONDS,
            WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS,
            WORKER_STDERR_CAPTURE_MAX_BYTES,
            WORKER_STDOUT_CAPTURE_MAX_BYTES,
            _q1_child_environment,
        )
        from bench.active_acquisition.runtime_lane import (
            _SELECTION_KIND,
            TARGET_DESCRIPTION,
            TARGET_ID,
            TARGET_KIND,
            ArmMode,
            initial_posterior_model_state,
        )
        from bench.active_acquisition.seeding import (
            EPISODES_PER_MASTER,
            MASTER_COUNT,
            PRIVATE_NAMESPACES,
            PROTOCOL_VERSION,
            Q1_ARM_IDS,
            SEMANTIC_ACTION_IDS,
        )
        from bench.active_acquisition.worker_capability import (
            _ACKNOWLEDGEMENT_DOMAIN,
            _AUTHENTICATION_DOMAIN,
            MAX_WORKER_CAPABILITY_PAYLOAD_BYTES,
            WORKER_CAPABILITY_ACK_BYTES,
            WORKER_CAPABILITY_SCHEMA,
            WORKER_CAPABILITY_SECRET_BYTES,
        )

        candidates = protocol["candidates"]
        runtime = protocol["runtime"]
        budget = protocol["budget"]
        seeding = protocol["seeding"]
        target = protocol["target"]
        fixture = protocol["fixture"]
        arm_specs = protocol["arms"]
        schemas = protocol["schemas"]
        analysis = protocol["analysis"]
        attempt_integrity = protocol["attempt_integrity"]
        entry_qualification = protocol["entry_qualification"]
        if (
            not isinstance(candidates, Mapping)
            or not isinstance(runtime, Mapping)
            or not isinstance(budget, Mapping)
            or not isinstance(seeding, Mapping)
            or not isinstance(target, Mapping)
            or not isinstance(fixture, Mapping)
            or not isinstance(arm_specs, list)
            or not isinstance(schemas, Mapping)
            or not isinstance(analysis, Mapping)
            or not isinstance(attempt_integrity, Mapping)
            or not isinstance(entry_qualification, Mapping)
        ):
            raise TypeError("protocol parity sections must be objects")
        if entry_qualification.get("check_order") != list(ENTRY_CHECK_ORDER):
            violations.append("entry qualification check order mismatch")
        candidate_rows = candidates.get("acquisition_order")
        terminal_rows = candidates.get("terminal_order")
        namespaces = seeding.get("private_namespaces")
        masters = budget.get("masters")
        if not isinstance(candidate_rows, list) or not all(isinstance(row, Mapping) for row in candidate_rows):
            raise TypeError("protocol acquisition_order must be an array of objects")
        if not isinstance(terminal_rows, list) or not isinstance(namespaces, list) or not isinstance(masters, list):
            raise TypeError("protocol order and namespace values must be arrays")
        if PROTOCOL_VERSION != Q1_PROTOCOL_VERSION:
            violations.append("seeding protocol version mismatch")
        if tuple(SEMANTIC_ACTION_IDS) != ACTION_ORDER:
            violations.append("seeding action order mismatch")
        if tuple(Q1_ARM_IDS) != ARM_ORDER or tuple(mode.value for mode in ArmMode) != ARM_ORDER:
            violations.append("arm order mismatch")
        if tuple(Q1_COMPONENT_IDS) != CHECKPOINT_COMPONENTS:
            violations.append("checkpoint component order mismatch")
        if tuple(row.get("semantic_id") for row in candidate_rows) != ACTION_ORDER:
            violations.append("protocol action order mismatch")
        expected_candidate_rows = [
            {
                "ordinal": ordinal,
                "semantic_id": action_id,
                "domain_action_id": f"acquisition:{ordinal:02d}:{action_id}",
            }
            for ordinal, action_id in enumerate(ACTION_ORDER)
        ]
        if candidate_rows != expected_candidate_rows:
            violations.append("protocol action ordinals or domain IDs mismatch")
        expected_units = {
            ArmMode.PROSPECT: "return",
            ArmMode.ORACLE: "return",
            ArmMode.GOAL_ONLY: "return",
            ArmMode.RAW_ENTROPY: "nats",
            ArmMode.EIG_ONLY: "nats",
            ArmMode.SHUFFLED_INFORMATION: "return",
            ArmMode.UNIFORM_RANDOM: None,
        }
        expected_arm_rows: list[dict[str, object]] = []
        for mode in ArmMode:
            row: dict[str, object] = {
                "id": mode.value,
                "selection_kind": _SELECTION_KIND[mode],
                "selection_unit": expected_units[mode],
                "required_prior_action": (
                    _EXPECTED_ACTION_BY_ARM[mode] if mode is not ArmMode.UNIFORM_RANDOM else "seed_dependent"
                ),
            }
            if mode is ArmMode.SHUFFLED_INFORMATION:
                row["selection_information_source_by_action"] = dict(SHUFFLED_INFORMATION_SOURCE_BY_ACTION)
                row["learner_likelihood"] = "true likelihood of the executed action"
            expected_arm_rows.append(row)
        if arm_specs != expected_arm_rows:
            violations.append("arm kind, unit, prior action, or shuffled-likelihood contract mismatch")
        if tuple(terminal_rows) != TERMINAL_ORDER:
            violations.append("terminal order mismatch")
        if candidates.get("terminal_tie_break") != TERMINAL_ORDER[0]:
            violations.append("terminal tie break mismatch")
        initial_model = initial_posterior_model_state()
        if (
            runtime.get("initial_model_payload_utf8") != initial_model.payload.decode("ascii")
            or runtime.get("initial_model_sha256") != initial_model.digest
            or runtime.get("initial_model_version") != initial_model.version
        ):
            violations.append("canonical zero-evidence prior model mismatch")
        expected_runtime = {
            "steps_per_episode": 2,
            "acquisition_updates_per_episode": 1,
            "terminal_updates_per_episode": 0,
            "fresh_component_graph_per_episode": True,
            "checkpoint_components": list(CHECKPOINT_COMPONENTS),
            "max_restore_concurrency": MAX_RESTORE_CONCURRENCY,
            "process_watchdogs": {
                "producer_stage_timeout_seconds": PRODUCER_STAGE_TIMEOUT_SECONDS,
                "restore_child_timeout_seconds": RESTORE_CHILD_TIMEOUT_SECONDS,
                "restore_stage_timeout_seconds": RESTORE_STAGE_TIMEOUT_SECONDS,
                "process_terminate_grace_seconds": PROCESS_TERMINATE_GRACE_SECONDS,
                "rule": (
                    "Any timeout or parent-side exception terminates, then kills if needed, and reaps "
                    "every started child before a failed marker may be finalized; if quiescence cannot "
                    "be proven, the marker remains started."
                ),
            },
            "process_launch": {
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
                "child_environment": {
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
                    "pythonpath_rule": (
                        "Exact ordered os.pathsep join of the resolved canonical repository root, its src "
                        "directory, then resolved sysconfig purelib and platlib directories with duplicate "
                        "paths omitted; every dependency path must be a non-symlink directory."
                    ),
                    "site_processing": False,
                },
            },
            "worker_capability": {
                "transport_family": "AF_UNIX",
                "transport_type": "SOCK_STREAM",
                "wire_format": (
                    "4-byte unsigned big-endian payload length || 32-byte secret || canonical payload || "
                    "32-byte HMAC-SHA256 authenticator"
                ),
                "secret_bytes": WORKER_CAPABILITY_SECRET_BYTES,
                "payload_max_bytes": MAX_WORKER_CAPABILITY_PAYLOAD_BYTES,
                "authenticator_bytes": WORKER_CAPABILITY_ACK_BYTES,
                "authentication_algorithm": "HMAC-SHA256",
                "authentication_domain_hex": _AUTHENTICATION_DOMAIN.hex(),
                "payload_encoding": "finite sorted compact canonical ASCII JSON",
                "payload_schema": WORKER_CAPABILITY_SCHEMA,
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
                "binding_rule": (
                    "The authenticated canonical payload binds the exact positive parent PID sampled before "
                    "Popen, the exact positive child PID returned by Popen, producer or restore role, master "
                    "and arm, complete Q1 run identity, and the exact normalized absolute path field set."
                ),
                "marker_commitment": "SHA256 of the exact 32-byte operational capability secret",
                "parent_half_close": "socket.SHUT_WR after the complete authenticated wire",
                "acknowledgement_bytes": WORKER_CAPABILITY_ACK_BYTES,
                "acknowledgement_algorithm": "HMAC-SHA256",
                "acknowledgement_domain_hex": _ACKNOWLEDGEMENT_DOMAIN.hex(),
                "acknowledgement_rule": (
                    "HMAC-SHA256(secret, acknowledgement_domain || SHA256(canonical payload) || wire "
                    "authenticator), followed by exact peer EOF with no trailing byte"
                ),
                "shared_exchange_timeout_seconds": WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS,
                "deadline_origin": (
                    "One monotonic sample immediately after Popen returns is shared by complete wire delivery, "
                    "parent half-close, exact acknowledgement, and peer EOF."
                ),
                "local_assurance_boundary": (
                    "PID bindings and inherited-descriptor custody are local same-account process checks. They "
                    "do not claim PID-reuse resistance, kernel-backed peer identity, hostile same-account "
                    "process exclusion, or external attestation."
                ),
            },
            "worker_capture": {
                "stdout_max_bytes": WORKER_STDOUT_CAPTURE_MAX_BYTES,
                "stderr_max_bytes": WORKER_STDERR_CAPTURE_MAX_BYTES,
                "bounded_memory_tail_bytes": _WORKER_CAPTURE_TAIL_BYTES,
                "capture_finish_timeout_seconds": _WORKER_CAPTURE_FINISH_TIMEOUT_SECONDS,
                "rule": (
                    "Parent-owned CLOEXEC pipes are continuously drained after Popen. Any stdout byte, stderr "
                    "overflow, drain failure, or capture that cannot finish inside the bound terminates and "
                    "fails the worker; only the bounded tail may appear in diagnostics."
                ),
            },
            "filesystem_custody": {
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
                "authority_root_rule": (
                    "Both roots are exact-0700 non-symlink directories owned by the effective user, are bound "
                    "by all seven directory identity fields, and must be distinct and nonnested."
                ),
                "publication_rule": (
                    "All six Q1 artifacts are one-link regular files with exact mode 0600 inside an exact-0700 "
                    "output directory; entry-qualification and independent-audit reports are one-link regular "
                    "files with exact mode 0644."
                ),
            },
            "external_inputs": {
                "document_max_bytes": _MAX_QUALIFICATION_INPUT_BYTES,
                "secret_salt_min_bytes": 32,
                "secret_salt_max_bytes": _MAX_QUALIFICATION_INPUT_BYTES,
                "secret_salt_mode": "0600",
                "regular_file_link_count": 1,
                "read_rule": (
                    "Every protocol, accepted-Q0, prospective-review, entry-report, schema, and private-salt "
                    "input is read once from a bounded no-follow regular descriptor and rejected if its "
                    "descriptor metadata or path identity changes before completion."
                ),
                "json_rule": (
                    "JSON documents are finite UTF-8 values with duplicate object keys rejected; documents "
                    "required to be canonical must exactly equal their canonical newline-terminated bytes."
                ),
            },
            "identity_counter": {
                "initial_value": 0,
                "checkpoint_preterminal_value": _Q1_PRETERMINAL_IDENTITY_COUNTER,
                "checkpoint_decode_max_next_counter": Q1_MAX_IDENTITY_NEXT_COUNTER,
                "rule": (
                    "Every acquisition checkpoint must decode to exact next_counter 43 before any range or set "
                    "construction, and all checkpoint decoders reject values above 64 before allocation or "
                    "iteration."
                ),
            },
            "final_output_files": [
                "raw-trace.jsonl",
                "private-audit.jsonl",
                "checkpoint-index.jsonl",
                "checkpoint-frames.bin",
                "restored-trace.jsonl",
                "aggregate.json",
            ],
        }
        for key, value in expected_runtime.items():
            if runtime.get(key) != value:
                violations.append(f"runtime contract mismatch:{key}")
        repository_root = Path(__file__).resolve().parents[2]
        expected_import_roots = [repository_root, repository_root / "src"]
        for dependency_kind in ("purelib", "platlib"):
            dependency_path = sysconfig.get_path(dependency_kind)
            if not dependency_path:
                raise ValueError(f"Python {dependency_kind} dependency path is unavailable")
            resolved_dependency = Path(dependency_path).resolve()
            if resolved_dependency not in expected_import_roots:
                expected_import_roots.append(resolved_dependency)
        expected_child_environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(str(path) for path in expected_import_roots),
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
        }
        if _q1_child_environment() != expected_child_environment:
            violations.append("runtime child environment differs from the exact allowlist")
        expected_schema_paths = {name: f"schemas/{path.name}" for name, path in Q1_SCHEMA_PATHS.items()}
        if dict(schemas) != expected_schema_paths:
            violations.append("protocol artifact schema inventory mismatch")
        if not all(
            isinstance(attempt_integrity.get(key), str) and attempt_integrity.get(key)
            for key in (
                "run_identity",
                "attempt_identity",
                "registry_rule",
                "failure_rule",
                "completion_rule",
                "artifact_binding",
                "assurance_boundary",
            )
        ):
            violations.append("single-attempt integrity contract is incomplete")
        if tuple(namespaces) != tuple(PRIVATE_NAMESPACES):
            violations.append("private namespace mismatch")
        if masters != list(range(MASTER_COUNT)):
            violations.append("master set mismatch")
        episodes = MASTER_COUNT * len(ARM_ORDER) * EPISODES_PER_MASTER
        if budget.get("arms") != len(ARM_ORDER):
            violations.append("arm budget mismatch")
        if budget.get("episodes_per_master_arm") != EPISODES_PER_MASTER:
            violations.append("per-master-arm episode budget mismatch")
        if budget.get("inference_unit") != "master":
            violations.append("inference unit mismatch")
        if budget.get("episodes_total") != episodes or episodes != EPISODES_TOTAL:
            violations.append("episode budget arithmetic mismatch")
        if budget.get("environment_steps_total") != 2 * episodes or 2 * episodes != ENVIRONMENT_STEPS_TOTAL:
            violations.append("environment-step budget arithmetic mismatch")
        if budget.get("transitions_total") != 2 * episodes:
            violations.append("transition budget arithmetic mismatch")
        if budget.get("acquisition_updates_total") != episodes:
            violations.append("update budget arithmetic mismatch")
        if budget.get("terminal_updates_total") != 0:
            violations.append("terminal update budget mismatch")
        if (
            analysis.get("conjunctive_non_oracle_controls")
            != ["goal_only", "raw_observation_entropy", "eig_only", "shuffled_information", "uniform_random"]
            or analysis.get("student_t_critical_df3") != 3.182446305284263
            or analysis.get("producer_aggregates_authoritative") is not False
        ):
            violations.append("analysis controls, critical value, or authority boundary mismatch")
        if (
            target.get("id") != TARGET_ID
            or target.get("description") != TARGET_DESCRIPTION
            or target.get("kind") != TARGET_KIND
        ):
            violations.append("composite target identity, description, or kind mismatch")
        action_specs = fixture.get("actions")
        if not isinstance(action_specs, Mapping) or fixture.get("prior_direct") != "1/2":
            violations.append("exact fixture prior or action map mismatch")
        else:
            exact = FractionOracle()
            for action in ACQUISITION_ACTIONS:
                spec = action_specs.get(action.action_id)
                if not isinstance(spec, Mapping):
                    violations.append(f"missing fixture action:{action.action_id}")
                    continue
                evaluation = exact.evaluate(action)
                expected = {
                    "outcomes": list(exact.outcomes(action)),
                    "physical_action_cost": str(evaluation.action_cost),
                    "information_acquisition_cost": str(evaluation.acquisition_cost),
                    "expected_prior_return": str(evaluation.expected_episode_value),
                }
                for key, value in expected.items():
                    if spec.get(key) != value:
                        violations.append(f"fixture action mismatch:{action.action_id}:{key}")
                if action.accuracy is not None and spec.get("reliability") != str(Fraction(str(action.accuracy))):
                    violations.append(f"fixture reliability mismatch:{action.action_id}")
            terminal = fixture.get("terminal")
            if not isinstance(terminal, Mapping):
                violations.append("fixture terminal map missing")
            else:
                reliability = Fraction(str(HiddenActuatorProblem().exploit_reliability))
                if terminal.get("match_success_probability") != str(reliability):
                    violations.append("terminal match probability mismatch")
                if terminal.get("mismatch_success_probability") != str(1 - reliability):
                    violations.append("terminal mismatch probability mismatch")
    except Exception as error:
        violations.append(f"{type(error).__name__}:{error}")
    return violations


def _non_strict_object_paths(value: object, path: str = "$") -> tuple[str, ...]:
    """Find every declared JSON object that permits undeclared properties."""

    found: list[str] = []
    if isinstance(value, Mapping):
        if value.get("type") == "object" and value.get("additionalProperties") is not False:
            found.append(path)
        for key, nested in value.items():
            found.extend(_non_strict_object_paths(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_non_strict_object_paths(nested, f"{path}[{index}]"))
    return tuple(found)


def _schema_violations(schemas: Mapping[str, Mapping[str, object]]) -> list[str]:
    violations: list[str] = []
    try:
        if set(schemas) != set(Q1_SCHEMA_PATHS):
            violations.append("schema inventory mismatch")
        for name, schema in schemas.items():
            for path in _non_strict_object_paths(schema):
                violations.append(f"schema object permits undeclared properties:{name}:{path}")
        resolved_jsonschema = package_version("jsonschema")
        if resolved_jsonschema != Q1_JSONSCHEMA_VERSION:
            violations.append(
                f"resolved jsonschema version differs from the pinned runtime dependency:{resolved_jsonschema}"
            )
    except Exception as error:
        violations.append(f"artifact schema contract failed:{type(error).__name__}")
    return violations


def _emitted_artifact_schema_violations(
    samples: Mapping[str, object],
    *,
    schemas: Mapping[str, Mapping[str, object]] | None = None,
) -> list[str]:
    """Validate real samples and prove every schema rejects one absent sentinel key."""

    violations: list[str] = []
    if schemas is None:
        try:
            schemas, _digests = _artifact_schema_snapshots()
        except Exception as error:
            return [f"artifact schema snapshot failed:{type(error).__name__}"]
    for name in Q1_SCHEMA_PATHS:
        sample = samples.get(name)
        schema = schemas.get(name)
        if sample is None:
            violations.append(f"missing_sample:{name}")
            continue
        if schema is None:
            violations.append(f"missing_schema:{name}")
            continue
        try:
            if not _schema_accepts(schema, sample):
                violations.append(f"emitted artifact schema violation:{name}")
                continue
            mutated = copy.deepcopy(sample)
            if not isinstance(mutated, dict):
                raise TypeError("artifact sample must be an object")
            if _UNDECLARED_HIDDEN_STATE_SENTINEL_KEY in mutated:
                raise AssertionError(f"undeclared hidden-state sentinel unexpectedly exists before mutation:{name}")
            mutated[_UNDECLARED_HIDDEN_STATE_SENTINEL_KEY] = 1
            if _schema_accepts(schema, mutated):
                violations.append(f"undeclared_hidden_state_mutation_accepted:{name}")
        except Exception as error:
            violations.append(f"{name}:artifact validation failed:{type(error).__name__}")
    return violations


def _directory_metadata_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
    )


def _capture_private_directory_identity(path: Path, *, label: str) -> Q1DirectoryIdentity:
    """Capture one owner-held 0700 directory through a stable no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely as a directory") from error
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"{label} must be an existing non-symlink directory")
            if stat.S_IMODE(before.st_mode) != 0o700:
                raise ValueError(f"{label} permissions must be exactly 0700")
            if before.st_uid != os.geteuid():
                raise ValueError(f"{label} must be owned by the effective user")
            canonical_path = path.resolve(strict=True)
            after = os.fstat(descriptor)
            path_after = os.stat(path, follow_symlinks=False)
            canonical_after = path.resolve(strict=True)
            canonical_metadata = os.stat(canonical_after, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{label} identity could not be verified safely") from error
    finally:
        os.close(descriptor)

    signature = _directory_metadata_signature(before)
    if (
        signature != _directory_metadata_signature(after)
        or signature != _directory_metadata_signature(path_after)
        or signature != _directory_metadata_signature(canonical_metadata)
        or canonical_path != canonical_after
    ):
        raise ValueError(f"{label} identity changed during its descriptor verification")
    return Q1DirectoryIdentity(
        canonical_path=str(canonical_path),
        st_dev=before.st_dev,
        st_ino=before.st_ino,
        st_uid=before.st_uid,
        st_gid=before.st_gid,
    )


def _directory_identities_are_nested(
    left: Q1DirectoryIdentity,
    right: Q1DirectoryIdentity,
) -> bool:
    left_path = Path(left.canonical_path)
    right_path = Path(right.canonical_path)
    return left_path in right_path.parents or right_path in left_path.parents


def _resource_preflight(
    *,
    execution_root: Path,
    attempt_registry_directory: Path,
    resource_samples: Mapping[str, object],
) -> tuple[Q1ResourcePreflight, list[str]]:
    violations: list[str] = []
    expected_size_keys = (
        "raw_trace_max_bytes",
        "private_audit_max_bytes",
        "checkpoint_index_max_bytes",
        "checkpoint_frame_max_bytes",
        "restored_trace_max_bytes",
    )
    sizes: dict[str, int] = {}
    for key in expected_size_keys:
        value = resource_samples.get(key)
        if type(value) is not int or value <= 0:
            violations.append(f"invalid or missing resource sample:{key}")
            sizes[key] = 0
        else:
            sizes[key] = value
    duration_ok = resource_samples.get("probe_duration_under_30_seconds") is True
    if not duration_ok:
        violations.append("result-free development probe exceeded its 30-second capacity bound")
    if sizes["checkpoint_frame_max_bytes"] > 32 * 1024 * 1024:
        violations.append("sample checkpoint frame exceeds the component coordinator total limit")

    canonical_bytes = 28_672 * sum(sizes.values()) + 1024 * 1024
    required_free_bytes = 3 * canonical_bytes + 1024 * 1024 * 1024
    execution_root_identity: Q1DirectoryIdentity | None = None
    attempt_registry_identity: Q1DirectoryIdentity | None = None
    for path, label in (
        (execution_root, "execution root"),
        (attempt_registry_directory, "attempt registry"),
    ):
        try:
            identity = _capture_private_directory_identity(path, label=label)
        except ValueError as error:
            violations.append(str(error))
        else:
            if label == "execution root":
                execution_root_identity = identity
            else:
                attempt_registry_identity = identity

    if execution_root_identity is not None and attempt_registry_identity is not None:
        same_object = execution_root_identity.canonical_path == attempt_registry_identity.canonical_path or (
            execution_root_identity.st_dev,
            execution_root_identity.st_ino,
        ) == (attempt_registry_identity.st_dev, attempt_registry_identity.st_ino)
        if same_object:
            violations.append("execution root and attempt registry must be distinct directories")
        elif _directory_identities_are_nested(execution_root_identity, attempt_registry_identity):
            violations.append("execution root and attempt registry must not be nested")

    disk_usage_path = (
        execution_root if execution_root_identity is None else Path(execution_root_identity.canonical_path)
    )
    try:
        if shutil.disk_usage(disk_usage_path).free < required_free_bytes:
            violations.append("execution root lacks the conservative full-budget disk headroom")
    except OSError as error:
        violations.append(f"execution root disk query failed:{error}")
    soft_nofile, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_nofile != resource.RLIM_INFINITY and soft_nofile < 64:
        violations.append("soft file-descriptor limit is below 64")
    if (os.cpu_count() or 0) < 1:
        violations.append("no execution CPU is available")

    if execution_root_identity is not None:
        try:
            from bench.active_acquisition.q1 import probe_atomic_noreplace_publication

            probe_atomic_noreplace_publication(Path(execution_root_identity.canonical_path))
        except Exception as error:
            violations.append(f"atomic no-replace publication probe failed:{type(error).__name__}:{error}")

    for path, label, before in (
        (execution_root, "execution root", execution_root_identity),
        (attempt_registry_directory, "attempt registry", attempt_registry_identity),
    ):
        if before is None:
            continue
        try:
            after = _capture_private_directory_identity(path, label=label)
        except ValueError as error:
            violations.append(str(error))
        else:
            if after != before:
                violations.append(f"{label} identity changed during atomic publication probe")

    try:
        from bench.active_acquisition.q1 import MAX_RESTORE_CONCURRENCY
    except Exception as error:
        violations.append(f"restore concurrency import failed:{type(error).__name__}:{error}")
        max_restore_concurrency = 0
    else:
        max_restore_concurrency = MAX_RESTORE_CONCURRENCY
        if max_restore_concurrency != 4:
            violations.append("restore concurrency differs from the frozen bound of four")

    report = Q1ResourcePreflight(
        execution_root=execution_root_identity,
        attempt_registry_directory=attempt_registry_identity,
        raw_trace_max_bytes=sizes["raw_trace_max_bytes"],
        private_audit_max_bytes=sizes["private_audit_max_bytes"],
        checkpoint_index_max_bytes=sizes["checkpoint_index_max_bytes"],
        checkpoint_frame_max_bytes=sizes["checkpoint_frame_max_bytes"],
        restored_trace_max_bytes=sizes["restored_trace_max_bytes"],
        estimated_canonical_bytes=canonical_bytes,
        required_free_bytes=required_free_bytes,
        max_restore_concurrency=max_restore_concurrency,
        sampled_arms=7,
        probe_duration_under_30_seconds=duration_ok,
        passed=not violations,
    )
    return report, violations


def _prospective_review_violations(
    path: Path,
    *,
    protocol_sha256: str,
    implementation_sha256: str,
    reviewed_source_count: int,
    execution_mode: Q1ExecutionMode = Q1ExecutionMode.PRODUCTION,
) -> tuple[str, list[str]]:
    violations: list[str] = []
    digest = ""
    try:
        digest, review = _read_stable_json_document(
            path,
            label="prospective review",
            require_canonical=True,
        )
        if not isinstance(review, dict):
            raise TypeError("prospective review must be an object")
        schema_value = _load_schema_snapshot(
            PROSPECTIVE_REVIEW_SCHEMA_PATH,
            label="prospective review schema",
        )
        if not _schema_accepts(schema_value, review):
            violations.append("review schema violation")
        expected = {
            "schema": PROSPECTIVE_REVIEW_SCHEMA,
            "protocol_version": Q1_PROTOCOL_VERSION,
            "protocol_sha256": protocol_sha256,
            "implementation_sha256": implementation_sha256,
            "claim_eligible": False,
            "formal_authorized": False,
            "passed": True,
        }
        for key, value in expected.items():
            if review.get(key) != value:
                violations.append(f"review field mismatch:{key}")
        if review.get("blocking_findings") != []:
            violations.append("review has blocking findings")
        if review.get("reviewed_source_count") != reviewed_source_count:
            violations.append("reviewed source count differs from the selected-source closure")
        if review.get("review_scope") != list(PROSPECTIVE_REVIEW_SCOPE):
            violations.append("review scope differs from the exact required scope and order")
        if review.get("q1_environment_interactions") != 0 or review.get("q1_private_draws") != 0:
            violations.append("prospective review is not result-free")
        reviewer = review.get("reviewer")
        machine_generated = isinstance(reviewer, str) and MACHINE_GENERATED_REVIEWER_MARK in reviewer
        if execution_mode is Q1ExecutionMode.PRODUCTION and machine_generated:
            violations.append("prospective review is the machine-generated rehearsal review")
        if execution_mode is Q1ExecutionMode.REHEARSAL and not machine_generated:
            violations.append("rehearsal entry must not consume an independent prospective review")
    except Exception as error:
        violations.append(f"prospective review validation failed:{type(error).__name__}")
    return digest, violations


def _check(name: str, violations: Sequence[str], summary: str) -> QualificationCheck:
    normalized = tuple(sorted(set(violations)))
    return QualificationCheck(
        name=name,
        passed=not normalized,
        violations=normalized,
        summary=summary,
    )


def validate_entry_report(value: Mapping[str, object]) -> None:
    """Validate an entry report against stable selected-source snapshots."""

    protocol_sha256, _protocol = _protocol_snapshot(Q1_PROTOCOL_PATH)
    artifact_schemas, artifact_schema_sha256 = _artifact_schema_snapshots()
    _validate_entry_report_against_snapshots(
        value,
        protocol_sha256=protocol_sha256,
        artifact_schemas=artifact_schemas,
        artifact_schema_sha256=dict(artifact_schema_sha256),
    )


def _validate_entry_report_against_snapshots(
    value: Mapping[str, object],
    *,
    protocol_sha256: str,
    artifact_schemas: Mapping[str, Mapping[str, object]],
    artifact_schema_sha256: Mapping[str, str],
) -> None:
    """Validate the entry report without reopening its bound selected sources."""

    preliminary_checks = value.get("checks")
    if not isinstance(preliminary_checks, list):
        raise TypeError("entry report checks must be an array")
    preliminary_names: list[str] = []
    for check in preliminary_checks:
        if not isinstance(check, Mapping):
            raise TypeError("entry report check must be an object")
        name = check.get("name")
        if not isinstance(name, str):
            raise TypeError("entry report check name must be a string")
        preliminary_names.append(name)
    if tuple(preliminary_names) != ENTRY_CHECK_ORDER:
        raise ValueError("entry report check names/order differ from canonical contract")

    schema_value = _load_schema_snapshot(
        ENTRY_REPORT_SCHEMA_PATH,
        label="entry report schema",
    )
    if not _schema_accepts(schema_value, value):
        raise ValueError("entry report schema violation")

    checks = value["checks"]
    if not isinstance(checks, list):
        raise TypeError("entry report checks must be an array")
    for check in checks:
        if not isinstance(check, dict):
            raise TypeError("entry report check must be an object")
        name = check.get("name")
        if not isinstance(name, str):
            raise TypeError("entry report check name must be a string")
        violations = check["violations"]
        if not isinstance(violations, list):
            raise TypeError("entry report violations must be an array")
        if check["passed"] is not (not violations):
            raise ValueError(f"entry check pass/violation mismatch: {name}")

    checks_by_name = dict(zip(ENTRY_CHECK_ORDER, checks, strict=True))

    expected_passed = all(bool(check["passed"]) for check in checks)
    if value["passed"] is not expected_passed:
        raise ValueError("entry report overall pass value differs from its checks")

    resources = value["resource_preflight"]
    if not isinstance(resources, dict):
        raise TypeError("entry report resource_preflight must be an object")
    resource_check = checks_by_name[ENTRY_CHECK_ORDER[7]]
    if resources["passed"] is not resource_check["passed"]:
        raise ValueError("resource preflight pass value differs from its entry check")

    expected_bindings = {
        "protocol_version": Q1_PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "q0_report_sha256": Q0_REPORT_SHA256,
        "q0_protocol_sha256": Q0_PROTOCOL_SHA256,
        "q0_implementation_sha256": Q0_IMPLEMENTATION_SHA256,
    }
    for key, expected in expected_bindings.items():
        if value[key] != expected:
            raise ValueError(f"entry report selected-source binding mismatch: {key}")

    if set(artifact_schemas) != set(Q1_SCHEMA_PATHS):
        raise ValueError("entry report artifact schema inventory differs")
    if value["schema_sha256"] != dict(artifact_schema_sha256):
        raise ValueError("entry report schema digests differ from the selected schemas")

    manifest = value["implementation_manifest"]
    if not isinstance(manifest, list):
        raise TypeError("entry report implementation_manifest must be an array")
    manifest_paths = [row["relative_path"] for row in manifest if isinstance(row, dict)]
    if len(manifest_paths) != len(manifest):
        raise TypeError("entry report manifest rows must be objects")
    if manifest_paths != sorted(set(manifest_paths)):
        raise ValueError("entry report manifest paths must be unique and sorted")

    if value["passed"]:
        if value["dependency_versions"] != {"jsonschema": Q1_JSONSCHEMA_VERSION}:
            raise ValueError("passed entry report dependency identity differs from the pinned runtime")
        if value["synthetic_development_interactions"] != 19:
            raise ValueError("passed entry report lacks all 19 synthetic interactions")
        positive_resource_fields = (
            "raw_trace_max_bytes",
            "private_audit_max_bytes",
            "checkpoint_index_max_bytes",
            "checkpoint_frame_max_bytes",
            "restored_trace_max_bytes",
        )
        if any(type(resources[field]) is not int or resources[field] <= 0 for field in positive_resource_fields):
            raise ValueError("passed entry report lacks positive resource samples")
        execution_identity = resources["execution_root"]
        registry_identity = resources["attempt_registry_directory"]
        if not isinstance(execution_identity, dict) or not isinstance(registry_identity, dict):
            raise ValueError("passed entry report lacks bound directory identities")
        if execution_identity["st_uid"] != os.geteuid() or registry_identity["st_uid"] != os.geteuid():
            raise ValueError("passed entry report directory identity owner differs")
        execution_path = Path(execution_identity["canonical_path"])
        registry_path = Path(registry_identity["canonical_path"])
        if execution_path == registry_path or (execution_identity["st_dev"], execution_identity["st_ino"]) == (
            registry_identity["st_dev"],
            registry_identity["st_ino"],
        ):
            raise ValueError("passed entry report directory identities are not distinct")
        if execution_path in registry_path.parents or registry_path in execution_path.parents:
            raise ValueError("passed entry report directory identities are nested")
        if (
            resources["max_restore_concurrency"] != 4
            or resources["sampled_arms"] != 7
            or resources["probe_duration_under_30_seconds"] is not True
            or resources["passed"] is not True
        ):
            raise ValueError("passed entry report resource contract is incomplete")
        for key in (
            "implementation_sha256",
            "salt_commitment_sha256",
            "prospective_review_sha256",
        ):
            field = value[key]
            if not isinstance(field, str) or len(field) != 64:
                raise ValueError(f"passed entry report lacks a bound digest: {key}")
        if not manifest:
            raise ValueError("passed entry report lacks its selected-source manifest")
        expected_manifest, expected_implementation_sha256 = implementation_manifest(Q1_IMPLEMENTATION_PATHS)
        if manifest != [row.as_dict() for row in expected_manifest]:
            raise ValueError("passed entry report manifest differs from selected source")
        if value["implementation_sha256"] != expected_implementation_sha256:
            raise ValueError("passed entry report implementation digest is not derived from its manifest")


def _write_exclusive_durable(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise ValueError("entry report output path already exists") from error
    try:
        os.fchmod(descriptor, 0o644)
        created = os.fstat(descriptor)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or stat.S_IMODE(created.st_mode) != 0o644
        ):
            raise ValueError("entry report output file custody differs")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("entry report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        path_metadata = os.stat(path, follow_symlinks=False)
        if (
            final.st_dev != path_metadata.st_dev
            or final.st_ino != path_metadata.st_ino
            or final.st_size != len(payload)
            or path_metadata.st_size != len(payload)
            or final.st_nlink != 1
            or path_metadata.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o644
            or stat.S_IMODE(path_metadata.st_mode) != 0o644
        ):
            raise ValueError("entry report output path differs from its durable descriptor")
    finally:
        os.close(descriptor)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(path.parent, parent_flags)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def write_report(report: Q1EntryQualificationReport, path: Path) -> None:
    """Write one canonical report after the caller chooses its destination."""

    validate_entry_report(report.as_dict())
    _write_exclusive_durable(path, canonical_json_bytes(report.as_dict(), newline=True))


def _require_no_site_cli() -> None:
    if sys.flags.no_site != 1:
        raise SystemExit("Q1 qualification requires invocation with Python -S")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the entry gate and optionally persist its canonical report."""

    _require_no_site_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0-report", required=True, type=Path)
    parser.add_argument("--salt-file", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--execution-root", required=True, type=Path)
    parser.add_argument("--attempt-registry", required=True, type=Path)
    parser.add_argument("--protocol", default=Q1_PROTOCOL_PATH, type=Path)
    parser.add_argument(
        "--execution-mode",
        default=Q1ExecutionMode.PRODUCTION.value,
        choices=[mode.value for mode in Q1ExecutionMode],
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = run_entry_qualification(
        q0_report_path=arguments.q0_report,
        secret_salt_path=arguments.salt_file,
        prospective_review_path=arguments.review,
        execution_root=arguments.execution_root,
        attempt_registry_directory=arguments.attempt_registry,
        protocol_path=arguments.protocol,
        execution_mode=Q1ExecutionMode(arguments.execution_mode),
    )
    if arguments.output is not None:
        write_report(report, arguments.output)
    print(report.to_json())
    return 0 if report.passed else 1


__all__ = (
    "ENTRY_REPORT_SCHEMA",
    "ENTRY_REPORT_SCHEMA_PATH",
    "ENTRY_CHECK_ORDER",
    "PROSPECTIVE_REVIEW_SCHEMA",
    "Q1_JSONSCHEMA_VERSION",
    "PROSPECTIVE_REVIEW_SCHEMA_PATH",
    "PROSPECTIVE_REVIEW_SCOPE",
    "Q1_IMPLEMENTATION_PATHS",
    "Q1DirectoryIdentity",
    "Q1EntryQualificationReport",
    "Q1ResourcePreflight",
    "QualificationCheck",
    "run_entry_qualification",
    "validate_entry_report",
    "write_report",
)


if __name__ == "__main__":
    raise SystemExit(main())
