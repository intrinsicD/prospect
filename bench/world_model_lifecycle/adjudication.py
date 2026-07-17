"""External, immutable adjudication packages for finalized WM-001 evidence.

The producer artifact is the first custody level.  This module creates a
separate second level after an independent audit has completed.  The package
copies the audit report verbatim and binds its bytes to the finalized producer
manifest, raw result, auditor source, and (for formal runs) pre-run binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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

_MAX_PRODUCER_MANIFEST_BYTES = 64 << 20
_MAX_RESULT_BYTES = 4 << 30
_MAX_AUDIT_BYTES = 64 << 20
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
        result.get("schema") != "prospect.world-model-lifecycle.raw-result.v2"
        or result.get("experiment_id") != "WM-001"
    ):
        raise AdjudicationError("raw result is not a WM-001 raw-result v2 document")
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
    producer_root: Path,
    producer_manifest_sha256: str,
    result_sha256: str,
) -> None:
    if audit.get("schema") != "prospect.world-model-lifecycle.artifact-audit.v1":
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
        audit.get("integrity_passed") is not True
        or audit.get("complete_for_claim") is not True
        or audit.get("passed") is not True
        or not isinstance(counts, Mapping)
        or type(counts.get("passed")) is not int
        or cast(int, counts.get("passed")) < 1
        or type(counts.get("failed")) is not int
        or counts.get("failed") != 0
        or type(counts.get("coverage_gaps")) is not int
        or counts.get("coverage_gaps") != 0
        or findings != []
        or gaps != []
    ):
        raise AdjudicationError("independent audit must pass with no failures or claim-completeness gaps")

    custody = audit.get("custody")
    if (
        not isinstance(custody, Mapping)
        or custody.get("producer_manifest_checked") is not True
        or custody.get("producer_manifest_status") != "completed"
        or custody.get("producer_manifest_sha256") != producer_manifest_sha256
    ):
        raise AdjudicationError("independent audit custody does not bind the completed producer manifest")


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
    if _sha256(payload) != auditor_source_sha256:
        raise AdjudicationError("formal source snapshot and adjudication auditor source do not agree")


def create_adjudication_package(
    *,
    producer_root: Path,
    audit_report: Path,
    output_directory: Path,
    disposition: Disposition,
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

    audit_payload = _read_regular_file(
        report_path,
        limit=_MAX_AUDIT_BYTES,
        label="independent audit report",
    )
    audit = _parse_canonical_json_object(
        audit_payload,
        label="independent audit report",
    )
    audit_sha256 = _sha256(audit_payload)
    _verify_audit_identity(
        audit,
        producer_root=root,
        producer_manifest_sha256=producer_manifest_sha256,
        result_sha256=result_sha256,
    )

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

    manifest: dict[str, object] = {
        "schema": "prospect.wm001.adjudication-package.v1",
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
        "auditor_source_file": AUDITOR_SOURCE_NAME,
        "auditor_source_sha256": auditor_source_sha256,
        "formal_binding_file": ("formal-binding.json" if formal_binding_sha256 is not None else None),
        "formal_binding_sha256": formal_binding_sha256,
        "files": [
            {
                "path": COPIED_AUDIT_NAME,
                "bytes": len(audit_payload),
                "sha256": audit_sha256,
            }
        ],
        "file_count": 1,
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
    ):
        raise AdjudicationError("upstream evidence changed before package publication")

    package.mkdir(mode=0o700, exist_ok=False)
    atomic_write_exclusive(package / COPIED_AUDIT_NAME, audit_payload)
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
        )
    except (ValueError, OSError) as error:
        print(f"adjudication package refused: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(manifest) + b"\n")
    return 0


__all__ = (
    "ADJUDICATION_MANIFEST_NAME",
    "COPIED_AUDIT_NAME",
    "AdjudicationError",
    "Disposition",
    "create_adjudication_package",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
