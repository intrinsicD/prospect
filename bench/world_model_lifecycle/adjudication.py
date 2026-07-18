"""External, immutable adjudication packages for finalized WM-001 evidence.

The producer artifact is the first custody level.  This module creates a
separate second level after an independent audit and semantic review have
completed.  Before accepting an audit report, adjudication reruns the current
pre-bound auditor and requires byte-identical canonical output.  The package
then copies both reports verbatim and binds their bytes to the finalized
producer manifest, raw result, auditor source, and pre-run binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from .artifact import (
    MANIFEST_NAME as PRODUCER_MANIFEST_NAME,
)
from .artifact import (
    atomic_write_exclusive,
    verify_producer_manifest,
)

HERE = Path(__file__).resolve().parent
AUDITOR_SOURCE_PATH = HERE / "artifact_audit.py"
AUDITOR_SOURCE_NAME = "bench/world_model_lifecycle/artifact_audit.py"
ADJUDICATION_MANIFEST_NAME = "adjudication-manifest.json"
COPIED_AUDIT_NAME = "independent-audit-report.json"
COPIED_SEMANTIC_REVIEW_NAME = "semantic-review.json"

_MAX_PRODUCER_MANIFEST_BYTES = 64 << 20
_MAX_RESULT_BYTES = 4 << 30
_MAX_AUDIT_BYTES = 64 << 20
_MAX_SEMANTIC_REVIEW_BYTES = 16 << 20
_MAX_AUDITOR_SOURCE_BYTES = 64 << 20
_SHA256_LENGTH = 64
_GATES = tuple(f"K{index}" for index in range(8))

Disposition = Literal["pending", "accepted", "rejected"]


class AdjudicationError(ValueError):
    """The supplied evidence cannot enter an adjudication package."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _rerun_independent_audit(
    producer_root: Path,
    *,
    auditor_source_payload: bytes,
    protocol_payload: bytes,
    result_schema_payload: bytes,
) -> bytes:
    """Execute the already-verified auditor bytes from an exclusive copy."""

    try:
        with tempfile.TemporaryDirectory(
            prefix="prospect-wm001-auditor-",
        ) as temporary:
            private_root = Path(temporary)
            private_source = private_root / "artifact_audit.py"
            private_schema = private_root / "schemas" / "raw-result.schema.json"
            atomic_write_exclusive(private_source, auditor_source_payload)
            private_schema.parent.mkdir(mode=0o700)
            atomic_write_exclusive(
                private_schema,
                result_schema_payload,
            )
            atomic_write_exclusive(
                private_root / "protocol.json",
                protocol_payload,
            )
            os.chmod(private_source, 0o400)
            source_descriptor = os.open(
                private_source,
                os.O_RDONLY | os.O_NOFOLLOW,
            )
            try:
                descriptor_paths = (
                    Path(f"/proc/self/fd/{source_descriptor}"),
                    Path(f"/dev/fd/{source_descriptor}"),
                )
                inherited_source = next(
                    (candidate for candidate in descriptor_paths if candidate.exists()),
                    None,
                )
                if inherited_source is None:
                    raise AdjudicationError("platform cannot execute a descriptor-bound auditor copy")
                command = [
                    sys.executable,
                    "-I",
                    "-B",
                    str(inherited_source),
                    str(producer_root),
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    env=dict(os.environ),
                    pass_fds=(source_descriptor,),
                )
            finally:
                os.close(source_descriptor)
            if (
                _read_regular_file(
                    private_source,
                    limit=_MAX_AUDITOR_SOURCE_BYTES,
                    label="private independent auditor source",
                )
                != auditor_source_payload
            ):
                raise AdjudicationError("private independent auditor source changed during execution")
    except OSError as error:
        raise AdjudicationError(f"fresh independent-audit recomputation failed: {error}") from error
    if completed.returncode not in {0, 1}:
        diagnostic = completed.stderr[:4096].decode("utf-8", errors="replace")
        raise AdjudicationError(
            f"fresh independent-audit recomputation failed with exit {completed.returncode}: {diagnostic}"
        )
    if len(completed.stdout) > _MAX_AUDIT_BYTES:
        raise AdjudicationError(f"fresh independent-audit report exceeds its {_MAX_AUDIT_BYTES}-byte limit")
    _parse_canonical_json_object(
        completed.stdout,
        label="fresh independent audit report",
    )
    return completed.stdout


def _read_regular_file(path: Path, *, limit: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise AdjudicationError(f"{label} must be a regular non-symbolic-link file")
    before = path.stat()
    if before.st_size > limit:
        raise AdjudicationError(f"{label} exceeds its {limit}-byte limit")
    payload = path.read_bytes()
    after = path.stat()
    if len(payload) != before.st_size or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise AdjudicationError(f"{label} changed while it was being read")
    return payload


def _parse_canonical_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AdjudicationError(f"{label} contains duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise AdjudicationError(f"{label} contains non-finite JSON value {value}")

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdjudicationError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise AdjudicationError(f"{label} must contain one JSON object")
    if payload != _canonical_json_bytes(decoded) + b"\n":
        raise AdjudicationError(f"{label} is not canonical JSON followed by one newline")
    return cast(dict[str, Any], decoded)


def _resolve_producer_root(path: Path) -> Path:
    if path.is_symlink():
        raise AdjudicationError("producer root must not be a symbolic link")
    try:
        root = path.resolve(strict=True)
    except OSError as error:
        raise AdjudicationError(f"producer root cannot be resolved: {error}") from error
    if not root.is_dir():
        raise AdjudicationError("producer root must be a directory")
    return root


def _resolve_external_audit(path: Path, *, producer_root: Path) -> Path:
    if path.is_symlink():
        raise AdjudicationError("independent audit report must not be a symbolic link")
    try:
        report = path.resolve(strict=True)
    except OSError as error:
        raise AdjudicationError(f"independent audit report cannot be resolved: {error}") from error
    if report == producer_root or report.is_relative_to(producer_root):
        raise AdjudicationError("independent audit report must be outside the producer root")
    return report


def _resolve_new_package(path: Path, *, producer_root: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace adjudication package: {path}")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise AdjudicationError(f"adjudication package parent cannot be resolved: {error}") from error
    if not parent.is_dir():
        raise AdjudicationError("adjudication package parent must be a directory")
    package = parent / path.name
    if not path.name or path.name in {".", ".."}:
        raise AdjudicationError("adjudication package has an unsafe directory name")
    if package == producer_root or package.is_relative_to(producer_root):
        raise AdjudicationError("adjudication package must be outside the producer root")
    return package


def _manifested_digest(
    manifest: Mapping[str, object],
    *,
    filename: str,
    expected_payload: bytes,
) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise AdjudicationError("producer manifest files block is invalid")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("path") == filename]
    if len(matches) != 1:
        raise AdjudicationError(f"producer manifest must contain exactly one {filename!r} row")
    row = matches[0]
    if row.get("bytes") != len(expected_payload) or row.get("sha256") != _sha256(expected_payload):
        raise AdjudicationError(f"producer manifest does not bind the exact {filename} bytes")


def _verify_result_identity(
    result: Mapping[str, object],
    producer_manifest: Mapping[str, object],
    *,
    producer_root: Path,
    result_payload: bytes,
) -> tuple[str, str | None]:
    if (
        result.get("schema") != "prospect.world-model-lifecycle.raw-result.v4"
        or result.get("experiment_id") != "WM-001"
        or result.get("protocol_version") != "1.4.0"
    ):
        raise AdjudicationError("raw result is not a WM-001 protocol-1.4 raw-result v4 document")
    lane = result.get("lane")
    if lane not in {"development", "formal"}:
        raise AdjudicationError("raw result lane is invalid")
    if (
        producer_manifest.get("schema") != "prospect.wm001.producer-manifest.v1"
        or producer_manifest.get("experiment_id") != "WM-001"
        or producer_manifest.get("status") != "completed"
        or producer_manifest.get("lane") != lane
    ):
        raise AdjudicationError("producer manifest and raw-result identities do not agree")
    _manifested_digest(
        producer_manifest,
        filename="result.json",
        expected_payload=result_payload,
    )

    binding_digest = result.get("formal_binding_sha256")
    if lane == "development":
        if result.get("claim_eligible") is not False or binding_digest is not None:
            raise AdjudicationError("development result must be claim-ineligible and have no formal binding")
        return lane, None

    if (
        result.get("claim_eligible") is not True
        or not isinstance(binding_digest, str)
        or len(binding_digest) != _SHA256_LENGTH
    ):
        raise AdjudicationError("formal result must be claim-eligible and identify its formal binding")
    binding_path = producer_root / "formal-binding.json"
    binding_payload = _read_regular_file(
        binding_path,
        limit=_MAX_PRODUCER_MANIFEST_BYTES,
        label="copied formal binding",
    )
    if _sha256(binding_payload) != binding_digest:
        raise AdjudicationError("formal result does not bind the copied formal-binding bytes")
    _manifested_digest(
        producer_manifest,
        filename="formal-binding.json",
        expected_payload=binding_payload,
    )
    binding = _parse_canonical_json_object(binding_payload, label="copied formal binding")
    binding_protocol = binding.get("protocol")
    if (
        binding.get("schema") != "prospect.world-model-lifecycle.formal-binding.v4"
        or binding.get("experiment_id") != "WM-001"
        or not isinstance(binding_protocol, Mapping)
        or binding_protocol.get("version") != "1.4.0"
        or binding_protocol.get("sha256") != result.get("protocol_sha256")
    ):
        raise AdjudicationError("formal binding identity and protocol do not agree with the raw result")
    execution = result.get("execution")
    binding_runtime = binding.get("runtime")
    if (
        not isinstance(execution, Mapping)
        or not isinstance(binding_runtime, Mapping)
        or execution.get("platform") != binding_runtime.get("platform")
        or execution.get("device") != binding_runtime.get("device")
        or execution.get("deterministic_algorithms") != binding_runtime.get("deterministic_algorithms")
        or execution.get("deterministic_algorithms") is not True
    ):
        raise AdjudicationError("formal result runtime differs from its pre-run binding")
    launch_payload = _read_regular_file(
        producer_root / "formal-launch.json",
        limit=_MAX_PRODUCER_MANIFEST_BYTES,
        label="copied formal launch record",
    )
    launch = _parse_canonical_json_object(
        launch_payload,
        label="copied formal launch record",
    )
    launch_body = dict(launch)
    launch_record_sha256 = launch_body.pop("record_sha256", None)
    if (
        launch.get("schema") != "prospect.wm001.formal-launch.v1"
        or launch.get("experiment_id") != "WM-001"
        or launch.get("protocol_version") != "1.4.0"
        or launch.get("formal_binding_sha256") != binding_digest
        or launch.get("attempt_directory") != producer_root.name
        or not isinstance(execution, Mapping)
        or execution.get("formal_launch_file") != "formal-launch.json"
        or execution.get("formal_launch_sha256") != _sha256(launch_payload)
        or launch.get("git_commit") != execution.get("git_commit")
        or launch.get("git_tree") != execution.get("git_tree")
        or launch_record_sha256 != _sha256(_canonical_json_bytes(launch_body))
    ):
        raise AdjudicationError("formal result does not bind its unique v1.4 launch record")
    _manifested_digest(
        producer_manifest,
        filename="formal-launch.json",
        expected_payload=launch_payload,
    )
    return lane, binding_digest


def _verify_acceptance_gates(result: Mapping[str, object]) -> None:
    gates = result.get("gate_results")
    if not isinstance(gates, list) or len(gates) != len(_GATES) or any(not isinstance(row, Mapping) for row in gates):
        raise AdjudicationError("accepted adjudication requires the complete WM-001 gate sequence")
    observed = [
        (cast(Mapping[str, object], row).get("gate"), cast(Mapping[str, object], row).get("passed")) for row in gates
    ]
    if observed != [(gate, True) for gate in _GATES]:
        raise AdjudicationError("accepted adjudication requires K0 through K7, in order, all passing")


def _verify_audit_identity(
    audit: Mapping[str, object],
    *,
    lane: str,
    producer_root: Path,
    producer_manifest_sha256: str,
    result_sha256: str,
) -> bool:
    """Verify upstream audit identity/custody and return claim cleanliness."""

    if audit.get("schema") != "prospect.world-model-lifecycle.artifact-audit.v2" or audit.get("lane") != lane:
        raise AdjudicationError("independent audit report has the wrong semantic identity")
    raw_artifact_root = audit.get("artifact_root")
    if not isinstance(raw_artifact_root, str) or not raw_artifact_root:
        raise AdjudicationError("independent audit artifact_root is invalid")
    try:
        audited_root = Path(raw_artifact_root).resolve(strict=True)
    except OSError as error:
        raise AdjudicationError("independent audit artifact_root is invalid") from error
    if (
        audited_root != producer_root
        or raw_artifact_root != str(producer_root)
        or audit.get("result_file") != "result.json"
        or audit.get("result_sha256") != result_sha256
    ):
        raise AdjudicationError("independent audit report identifies a different producer root or result")

    counts = audit.get("check_counts")
    findings = audit.get("findings")
    gaps = audit.get("coverage_gaps")
    if (
        type(audit.get("integrity_passed")) is not bool
        or type(audit.get("engineering_complete")) is not bool
        or type(audit.get("complete_for_claim")) is not bool
        or type(audit.get("passed")) is not bool
        or not isinstance(counts, Mapping)
        or type(counts.get("passed")) is not int
        or cast(int, counts.get("passed")) < 1
        or type(counts.get("failed")) is not int
        or cast(int, counts.get("failed")) < 0
        or type(counts.get("coverage_gaps")) is not int
        or cast(int, counts.get("coverage_gaps")) < 0
        or not isinstance(findings, list)
        or not isinstance(gaps, list)
        or any(not isinstance(row, Mapping) for row in findings)
        or any(not isinstance(row, Mapping) for row in gaps)
    ):
        raise AdjudicationError("independent audit status block is invalid")
    failed_count = cast(int, counts["failed"])
    gap_count = cast(int, counts["coverage_gaps"])
    integrity_passed = cast(bool, audit["integrity_passed"])
    engineering_complete = cast(bool, audit["engineering_complete"])
    complete_for_claim = cast(bool, audit["complete_for_claim"])
    passed = cast(bool, audit["passed"])
    if (
        failed_count != len(findings)
        or gap_count != len(gaps)
        or integrity_passed != (failed_count == 0)
        or engineering_complete != (gap_count == 0)
        or complete_for_claim != (lane == "formal" and engineering_complete)
        or passed != (integrity_passed and engineering_complete)
    ):
        raise AdjudicationError("independent audit status block is internally inconsistent")

    custody = audit.get("custody")
    if (
        not isinstance(custody, Mapping)
        or custody.get("producer_manifest_checked") is not True
        or custody.get("producer_manifest_status") != "completed"
        or custody.get("producer_manifest_sha256") != producer_manifest_sha256
    ):
        raise AdjudicationError("independent audit custody does not bind the completed producer manifest")

    audit_implementation = audit.get("audit_implementation")
    if not isinstance(audit_implementation, Mapping):
        raise AdjudicationError("independent audit implementation identity is missing")
    auditor_digest = audit_implementation.get("auditor_source_sha256")
    if not isinstance(auditor_digest, str) or len(auditor_digest) != _SHA256_LENGTH:
        raise AdjudicationError("independent audit source digest is invalid")
    binding_path = producer_root / "formal-binding.json"
    if binding_path.is_file():
        binding_payload = _read_regular_file(
            binding_path,
            limit=_MAX_PRODUCER_MANIFEST_BYTES,
            label="copied formal binding",
        )
        binding = _parse_canonical_json_object(binding_payload, label="copied formal binding")
        coverage_arithmetic = binding.get("coverage_arithmetic")
        if not isinstance(coverage_arithmetic, Mapping):
            raise AdjudicationError("formal binding has no coverage arithmetic identity")
        snapshot_payload = _read_regular_file(
            producer_root / "source" / AUDITOR_SOURCE_NAME,
            limit=_MAX_AUDITOR_SOURCE_BYTES,
            label="formal auditor source snapshot",
        )
        expected_auditor_digest = _sha256(snapshot_payload)
        if (
            audit_implementation.get("bound_auditor_source_sha256") != expected_auditor_digest
            or auditor_digest != expected_auditor_digest
            or coverage_arithmetic.get("auditor_source_sha256") != expected_auditor_digest
            or audit_implementation.get("formal_test_report_sha256")
            != coverage_arithmetic.get("formal_test_report_sha256")
            or audit_implementation.get("coverage_conformance_report_sha256")
            != coverage_arithmetic.get("conformance_report_sha256")
            or audit_implementation.get("auditor_source_matches_binding") is not True
            or audit_implementation.get("coverage_conformance_verified") is not True
        ):
            raise AdjudicationError(
                "independent audit source, tests, or coverage conformance do not match the pre-outcome binding"
            )

    return passed and (lane != "formal" or complete_for_claim)


def _verify_formal_auditor_snapshot(
    producer_root: Path,
    *,
    auditor_source_sha256: str,
) -> None:
    snapshot = producer_root / "source" / AUDITOR_SOURCE_NAME
    try:
        resolved_snapshot = snapshot.resolve(strict=True)
    except OSError as error:
        raise AdjudicationError(f"formal auditor source snapshot cannot be resolved: {error}") from error
    if resolved_snapshot != snapshot:
        raise AdjudicationError("formal auditor source snapshot has an unsafe aliased path")
    payload = _read_regular_file(
        snapshot,
        limit=_MAX_AUDITOR_SOURCE_BYTES,
        label="formal auditor source snapshot",
    )
    binding_payload = _read_regular_file(
        producer_root / "formal-binding.json",
        limit=_MAX_PRODUCER_MANIFEST_BYTES,
        label="copied formal binding",
    )
    binding = _parse_canonical_json_object(
        binding_payload,
        label="copied formal binding",
    )
    coverage_arithmetic = binding.get("coverage_arithmetic")
    if (
        _sha256(payload) != auditor_source_sha256
        or not isinstance(coverage_arithmetic, Mapping)
        or coverage_arithmetic.get("auditor_source_sha256") != auditor_source_sha256
    ):
        raise AdjudicationError("formal binding, source snapshot, and adjudication auditor source do not agree")


def _verify_semantic_review(
    review: Mapping[str, object],
    *,
    producer_root: Path,
    result_sha256: str,
    audit_sha256: str,
    disposition: Disposition,
    audit_clean_for_claim: bool,
) -> None:
    if review.get("schema") != "prospect.wm001.semantic-review.v1":
        raise AdjudicationError("semantic review has the wrong schema")
    if (
        review.get("artifact_root") != str(producer_root)
        or review.get("result_sha256") != result_sha256
        or review.get("independent_audit_sha256") != audit_sha256
    ):
        raise AdjudicationError("semantic review identifies different upstream evidence")
    verdict = review.get("verdict")
    if verdict not in {"accepted", "rejected"} or verdict != disposition:
        raise AdjudicationError("semantic review verdict differs from adjudication disposition")
    if not isinstance(review.get("reviewer"), str) or not review.get("reviewer"):
        raise AdjudicationError("semantic review has no reviewer identity")
    if not isinstance(review.get("conclusion"), str) or not review.get("conclusion"):
        raise AdjudicationError("semantic review has no conclusion")
    reviewed_gates = review.get("reviewed_gates")
    if reviewed_gates != list(_GATES):
        raise AdjudicationError("semantic review did not inspect K0 through K7 in order")
    fatal_findings = review.get("fatal_findings")
    if not isinstance(fatal_findings, list):
        raise AdjudicationError("semantic review fatal_findings must be an array")
    if disposition == "accepted" and fatal_findings:
        raise AdjudicationError("accepted semantic review contains unresolved fatal findings")
    if disposition == "rejected" and not audit_clean_for_claim and not fatal_findings:
        raise AdjudicationError(
            "rejected adjudication of a non-clean audit requires at least one fatal semantic finding"
        )


def create_adjudication_package(
    *,
    producer_root: Path,
    audit_report: Path,
    output_directory: Path,
    disposition: Disposition,
    semantic_review: Path | None = None,
) -> dict[str, object]:
    """Create one external package without replacing any existing path."""

    if disposition not in {"pending", "accepted", "rejected"}:
        raise AdjudicationError(f"unsupported adjudication disposition: {disposition!r}")
    root = _resolve_producer_root(producer_root)
    report_path = _resolve_external_audit(audit_report, producer_root=root)
    package = _resolve_new_package(output_directory, producer_root=root)

    verified_manifest = verify_producer_manifest(root)
    producer_manifest_path = root / PRODUCER_MANIFEST_NAME
    producer_manifest_payload = _read_regular_file(
        producer_manifest_path,
        limit=_MAX_PRODUCER_MANIFEST_BYTES,
        label="producer manifest",
    )
    producer_manifest = _parse_canonical_json_object(
        producer_manifest_payload,
        label="producer manifest",
    )
    if producer_manifest != verified_manifest:
        raise AdjudicationError("producer manifest changed across independent verification")
    producer_manifest_sha256 = _sha256(producer_manifest_payload)

    result_path = root / "result.json"
    result_payload = _read_regular_file(
        result_path,
        limit=_MAX_RESULT_BYTES,
        label="raw result",
    )
    result = _parse_canonical_json_object(result_payload, label="raw result")
    result_sha256 = _sha256(result_payload)
    lane, formal_binding_sha256 = _verify_result_identity(
        result,
        producer_manifest,
        producer_root=root,
        result_payload=result_payload,
    )

    if disposition == "accepted":
        if lane != "formal":
            raise AdjudicationError("development evidence cannot receive an accepted disposition")
        _verify_acceptance_gates(result)

    auditor_source_payload = _read_regular_file(
        AUDITOR_SOURCE_PATH,
        limit=_MAX_AUDITOR_SOURCE_BYTES,
        label="independent auditor source",
    )
    auditor_source_sha256 = _sha256(auditor_source_payload)
    if lane == "formal":
        _verify_formal_auditor_snapshot(
            root,
            auditor_source_sha256=auditor_source_sha256,
        )
        protocol_support_path = root / "protocol.json"
        result_schema_support_path = root / "schemas" / "raw-result.schema.json"
    else:
        protocol_support_path = HERE / "protocol.json"
        result_schema_support_path = HERE / "schemas" / "raw-result.schema.json"
    protocol_support_payload = _read_regular_file(
        protocol_support_path,
        limit=_MAX_AUDITOR_SOURCE_BYTES,
        label="auditor protocol",
    )
    result_schema_support_payload = _read_regular_file(
        result_schema_support_path,
        limit=_MAX_AUDITOR_SOURCE_BYTES,
        label="auditor raw-result schema",
    )
    if lane == "formal":
        _manifested_digest(
            producer_manifest,
            filename="protocol.json",
            expected_payload=protocol_support_payload,
        )
        _manifested_digest(
            producer_manifest,
            filename="schemas/raw-result.schema.json",
            expected_payload=result_schema_support_payload,
        )
        if _sha256(protocol_support_payload) != result.get("protocol_sha256"):
            raise AdjudicationError("formal auditor support protocol differs from the raw result")

    audit_payload = _read_regular_file(
        report_path,
        limit=_MAX_AUDIT_BYTES,
        label="independent audit report",
    )
    audit = _parse_canonical_json_object(
        audit_payload,
        label="independent audit report",
    )
    recomputed_audit_payload = _rerun_independent_audit(
        root,
        auditor_source_payload=auditor_source_payload,
        protocol_payload=protocol_support_payload,
        result_schema_payload=result_schema_support_payload,
    )
    if audit_payload != recomputed_audit_payload:
        raise AdjudicationError(
            "supplied independent audit report does not exactly match a fresh canonical run of the pre-bound auditor"
        )
    audit_sha256 = _sha256(audit_payload)
    audit_clean_for_claim = _verify_audit_identity(
        audit,
        lane=lane,
        producer_root=root,
        producer_manifest_sha256=producer_manifest_sha256,
        result_sha256=result_sha256,
    )
    if disposition in {"pending", "accepted"} and not audit_clean_for_claim:
        raise AdjudicationError(
            f"{disposition} adjudication requires an independent audit that passes "
            "with no failures or claim-completeness gaps"
        )

    review_payload: bytes | None = None
    review_sha256: str | None = None
    review_path: Path | None = None
    if semantic_review is not None:
        review_path = _resolve_external_audit(semantic_review, producer_root=root)
        review_payload = _read_regular_file(
            review_path,
            limit=_MAX_SEMANTIC_REVIEW_BYTES,
            label="semantic review",
        )
        review = _parse_canonical_json_object(
            review_payload,
            label="semantic review",
        )
        review_sha256 = _sha256(review_payload)
        _verify_semantic_review(
            review,
            producer_root=root,
            result_sha256=result_sha256,
            audit_sha256=audit_sha256,
            disposition=disposition,
            audit_clean_for_claim=audit_clean_for_claim,
        )
    elif disposition in {"accepted", "rejected"}:
        raise AdjudicationError(f"{disposition} adjudication requires a separate semantic review")

    files = [
        {
            "path": COPIED_AUDIT_NAME,
            "bytes": len(audit_payload),
            "sha256": audit_sha256,
        }
    ]
    if review_payload is not None and review_sha256 is not None:
        files.append(
            {
                "path": COPIED_SEMANTIC_REVIEW_NAME,
                "bytes": len(review_payload),
                "sha256": review_sha256,
            }
        )
    manifest: dict[str, object] = {
        "schema": "prospect.wm001.adjudication-package.v3",
        "experiment_id": "WM-001",
        "lane": lane,
        "disposition": disposition,
        "producer_root": str(root),
        "producer_manifest_file": PRODUCER_MANIFEST_NAME,
        "producer_manifest_sha256": producer_manifest_sha256,
        "result_file": "result.json",
        "result_sha256": result_sha256,
        "audit_file": COPIED_AUDIT_NAME,
        "audit_sha256": audit_sha256,
        "audit_clean_for_claim": audit_clean_for_claim,
        "auditor_source_file": AUDITOR_SOURCE_NAME,
        "auditor_source_sha256": auditor_source_sha256,
        "semantic_review_file": (COPIED_SEMANTIC_REVIEW_NAME if review_sha256 is not None else None),
        "semantic_review_sha256": review_sha256,
        "formal_binding_file": ("formal-binding.json" if formal_binding_sha256 is not None else None),
        "formal_binding_sha256": formal_binding_sha256,
        "files": files,
        "file_count": len(files),
        "manifest_excludes": [ADJUDICATION_MANIFEST_NAME],
    }

    # Reopen every upstream identity immediately before publication.  A
    # concurrent mutation therefore yields no package rather than a stale bind.
    if (
        verify_producer_manifest(root) != producer_manifest
        or _read_regular_file(
            producer_manifest_path,
            limit=_MAX_PRODUCER_MANIFEST_BYTES,
            label="producer manifest",
        )
        != producer_manifest_payload
        or _read_regular_file(result_path, limit=_MAX_RESULT_BYTES, label="raw result") != result_payload
        or _read_regular_file(
            report_path,
            limit=_MAX_AUDIT_BYTES,
            label="independent audit report",
        )
        != audit_payload
        or _read_regular_file(
            AUDITOR_SOURCE_PATH,
            limit=_MAX_AUDITOR_SOURCE_BYTES,
            label="independent auditor source",
        )
        != auditor_source_payload
        or (
            review_path is not None
            and review_payload is not None
            and _read_regular_file(
                review_path,
                limit=_MAX_SEMANTIC_REVIEW_BYTES,
                label="semantic review",
            )
            != review_payload
        )
    ):
        raise AdjudicationError("upstream evidence changed before package publication")

    package.mkdir(mode=0o700, exist_ok=False)
    atomic_write_exclusive(package / COPIED_AUDIT_NAME, audit_payload)
    if review_payload is not None:
        atomic_write_exclusive(
            package / COPIED_SEMANTIC_REVIEW_NAME,
            review_payload,
        )
    atomic_write_exclusive(
        package / ADJUDICATION_MANIFEST_NAME,
        _canonical_json_bytes(manifest) + b"\n",
    )
    directory_descriptor = os.open(package, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--disposition",
        required=True,
        choices=("pending", "accepted", "rejected"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = create_adjudication_package(
            producer_root=arguments.producer,
            audit_report=arguments.audit,
            output_directory=arguments.output,
            disposition=arguments.disposition,
            semantic_review=arguments.semantic_review,
        )
    except (ValueError, OSError) as error:
        print(f"adjudication package refused: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(manifest) + b"\n")
    return 0


__all__ = (
    "ADJUDICATION_MANIFEST_NAME",
    "COPIED_AUDIT_NAME",
    "COPIED_SEMANTIC_REVIEW_NAME",
    "AdjudicationError",
    "Disposition",
    "create_adjudication_package",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
