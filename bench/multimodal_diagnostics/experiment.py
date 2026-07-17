"""Formal lifecycle and evidence verification for MM-002.

MM-002 is deliberately isolated from MM-001's sealed source globs.  It consumes a
byte-identical copy of MM-001's feature package and never invokes a neural frontend.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

from bench.multimodal_preflight import dataset as parent_dataset
from bench.multimodal_preflight import experiment as parent

from . import method

SCHEMA_VERSION = "mm002-formal-v1"
EXPERIMENT_ID = "MM-002"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/multimodal_diagnostics/results/MM-002")
EXPECTED_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT
PROTOCOL_DOC = Path(
    "docs/research/2026-07-15-mm002-feature-only-failure-isolation-protocol.md"
)
PARENT_OUTPUT = parent.DEFAULT_OUTPUT
PARENT_COPY_ROOT = Path("inputs/MM-001")

PROTOCOL_COPY_FILE = Path("MM-002-protocol.md")
INPUT_MANIFEST_FILE = Path("input-manifest.json")
STARTED_FILE = Path("formal-start.json")
EVIDENCE_FILE = Path("MM-002-evidence.json")
RESULT_FILE = Path("MM-002-results.json")
REPORT_FILE = Path("MM-002-report.md")
ARTIFACT_MANIFEST_FILE = Path("artifact-manifest.json")

PARENT_ARTIFACT_MANIFEST_SHA256 = (
    "a394104a6e9bcdb6c18b206d090e4afb9a540b9e3a2a2875985980e23ecaf52c"
)
PARENT_RESULT_SHA256 = "16504f4bfb36e5252aea9aa6604bc88d64233e256d184bf0e3b2889f5fd76fb7"
PARENT_FEATURE_SHA256 = "3fdf0c988cf0bdb428432b67c71fc7a18404080b6e12bfe8b6226d2276330755"
PARENT_CLASSIFICATION = "real_visual_temporal_prediction_not_supported"

PREPARED_ROOT_FILES = (PROTOCOL_COPY_FILE, INPUT_MANIFEST_FILE)
OUTCOME_FILES = (STARTED_FILE, EVIDENCE_FILE, RESULT_FILE, REPORT_FILE)
COMPLETED_ROOT_FILES = (*PREPARED_ROOT_FILES, *OUTCOME_FILES, ARTIFACT_MANIFEST_FILE)
PARENT_PACKAGE_FILES = tuple(Path(path) for path in parent.PACKAGE_FILES)
PREPARED_FILES = (
    *PREPARED_ROOT_FILES,
    *(PARENT_COPY_ROOT / path for path in PARENT_PACKAGE_FILES),
)
ARTIFACT_FILES = (*PREPARED_FILES, *OUTCOME_FILES)
COMPLETED_FILES = (*ARTIFACT_FILES, ARTIFACT_MANIFEST_FILE)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InvalidMM002Package(ValueError):
    """Stable fail-closed classification for any MM-002 integrity defect."""

    classification = "invalid_MM002_package"


def _integrity_boundary(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return function(*args, **kwargs)
        except InvalidMM002Package:
            raise
        except Exception as error:
            raise InvalidMM002Package(str(error)) from error

    return guarded


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_value(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _canonical_json_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _file_hash(path: Path) -> str:
    return parent_dataset.sha256_file(path)


def _file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or a symlink: {path}")
    return {
        "sha256": _file_hash(path),
        "bytes": path.stat().st_size,
        "mode": stat.S_IMODE(path.stat().st_mode),
    }


def _source_paths() -> tuple[Path, ...]:
    own = {
        PROTOCOL_DOC,
        Path("tests/test_mm002_method.py"),
        Path("tests/test_mm002_experiment.py"),
        *(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "bench/multimodal_diagnostics").glob("*.py")),
    }
    return tuple(sorted({*parent._source_paths(), *own}, key=str))


def _source_hashes() -> dict[str, str]:
    records: dict[str, str] = {}
    for path in _source_paths():
        full = REPO_ROOT / path
        if full.is_symlink() or not full.is_file():
            raise ValueError(f"MM-002 bound source is missing or a symlink: {path}")
        records[str(path)] = _file_hash(full)
    return records


def _dependency_versions() -> dict[str, object]:
    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for package in ("prospect",):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _assert_expected_output(output: Path) -> None:
    if output.resolve() != EXPECTED_OUTPUT.resolve():
        raise ValueError(f"MM-002 output must be the canonical path {DEFAULT_OUTPUT}")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("MM-002 output must be a real directory path")
    parent_resolved = (REPO_ROOT / PARENT_OUTPUT).resolve()
    output_resolved = output.resolve()
    if (
        output_resolved == parent_resolved
        or output_resolved in parent_resolved.parents
        or parent_resolved in output_resolved.parents
    ):
        raise ValueError("MM-002 output overlaps its protected MM-001 parent")


def _tree_members(output: Path) -> tuple[set[Path], set[Path]]:
    if output.is_symlink() or not output.is_dir():
        raise ValueError("MM-002 output must be a real directory")
    files: set[Path] = set()
    directories: set[Path] = {Path(".")}
    for root, dirnames, filenames in os.walk(output, followlinks=False):
        root_path = Path(root)
        for dirname in dirnames:
            path = root_path / dirname
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"MM-002 package contains a non-regular directory: {path}")
            directories.add(path.relative_to(output))
        for filename in filenames:
            path = root_path / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"MM-002 package contains a non-regular file: {path}")
            files.add(path.relative_to(output))
    return files, directories


def _expected_directories(files: Sequence[Path]) -> set[Path]:
    directories = {Path(".")}
    for path in files:
        directories.update(path.parents)
    return directories


def _require_membership(output: Path, expected: Sequence[Path]) -> None:
    files, directories = _tree_members(output)
    expected_files = set(expected)
    if files != expected_files or directories != _expected_directories(expected):
        missing = sorted(str(path) for path in expected_files - files)
        extra = sorted(str(path) for path in files - expected_files)
        raise ValueError(f"MM-002 package membership mismatch: missing={missing}, extra={extra}")


def _verify_live_parent() -> dict[str, object]:
    verification = parent.verify(parent.DEFAULT_OUTPUT)
    if verification.get("outcomes") != "verified_results":
        raise ValueError("MM-002 requires completed verified MM-001 results")
    if verification.get("classification") != PARENT_CLASSIFICATION:
        raise ValueError("MM-001 parent classification drifted")
    root = REPO_ROOT / PARENT_OUTPUT
    if _file_hash(root / parent.ARTIFACT_MANIFEST_FILE) != PARENT_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("MM-001 artifact-manifest identity drifted")
    if _file_hash(root / parent.RESULT_FILE) != PARENT_RESULT_SHA256:
        raise ValueError("MM-001 result identity drifted")
    if _file_hash(root / parent.FEATURE_FILE) != PARENT_FEATURE_SHA256:
        raise ValueError("MM-001 feature-package identity drifted")
    actual = {path.relative_to(root) for path in root.iterdir()}
    if actual != set(PARENT_PACKAGE_FILES):
        raise ValueError("MM-001 parent package membership drifted")
    return cast(dict[str, object], _canonical_json_value(verification))


def _parent_records(root: Path) -> dict[str, dict[str, object]]:
    return {str(path): _file_record(root / path) for path in PARENT_PACKAGE_FILES}


def _parent_snapshot() -> dict[str, object]:
    verification = _verify_live_parent()
    root = REPO_ROOT / PARENT_OUTPUT
    records = _parent_records(root)
    return {
        "experiment_id": "MM-001",
        "classification": PARENT_CLASSIFICATION,
        "verification": verification,
        "live_path": str(PARENT_OUTPUT),
        "copied_path": str(PARENT_COPY_ROOT),
        "files": records,
        "pinned": {
            "artifact_manifest_sha256": PARENT_ARTIFACT_MANIFEST_SHA256,
            "result_sha256": PARENT_RESULT_SHA256,
            "feature_sha256": PARENT_FEATURE_SHA256,
        },
        "scientific_relationship": (
            "outcome-informed mechanism diagnostic reusing MM-001 rows; not independent evidence"
        ),
    }


def _validate_parent_copy(output: Path, snapshot: Mapping[str, object]) -> None:
    copied_root = output / PARENT_COPY_ROOT
    copied_files, copied_dirs = _tree_members(copied_root)
    if copied_files != set(PARENT_PACKAGE_FILES) or copied_dirs != {Path(".")}:
        raise ValueError("copied MM-001 package has unexpected membership")
    expected = snapshot.get("files")
    if not isinstance(expected, Mapping):
        raise ValueError("MM-002 parent snapshot has no file records")
    copied = _parent_records(copied_root)
    if copied != expected:
        raise ValueError("copied MM-001 package differs from the live sealed parent")


def _fold_record() -> list[dict[str, object]]:
    return [
        {
            "index": fold.index,
            "train_ids": list(fold.train_ids),
            "test_ids": list(fold.test_ids),
        }
        for fold in parent_dataset.formal_folds()
    ]


def _expected_input_manifest(output: Path) -> dict[str, object]:
    snapshot = _parent_snapshot()
    _validate_parent_copy(output, snapshot)
    table, feature_schema = parent._load_feature_table(output / PARENT_COPY_ROOT / parent.FEATURE_FILE)
    table.validate()
    protocol_bytes = (REPO_ROOT / PROTOCOL_DOC).read_bytes()
    if (output / PROTOCOL_COPY_FILE).read_bytes() != protocol_bytes:
        raise ValueError("MM-002 protocol copy differs from its source")
    config = method.config_record()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared_before_formal_execution",
        "protocol": {
            "source": str(PROTOCOL_DOC),
            "copy": str(PROTOCOL_COPY_FILE),
            "sha256": sha256(protocol_bytes).hexdigest(),
        },
        "source": _source_hashes(),
        "dependencies": _dependency_versions(),
        "parent": snapshot,
        "feature_schema": feature_schema,
        "folds": _fold_record(),
        "config": config,
        "config_sha256": _canonical_json_sha256(config),
        "expected_prepared_files": [str(path) for path in PREPARED_FILES],
        "expected_completed_files": [str(path) for path in COMPLETED_FILES],
    }


def _validate_prepared(output: Path) -> dict[str, object]:
    _require_membership(output, PREPARED_FILES)
    saved = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(saved, dict):
        raise ValueError("MM-002 input manifest must be an object")
    expected = _expected_input_manifest(output)
    if saved != expected:
        raise ValueError("MM-002 input manifest does not recompute from frozen inputs")
    return cast(dict[str, object], saved)


def _formal_start_record(manifest: Mapping[str, object]) -> dict[str, object]:
    parent_value = manifest.get("parent")
    if not isinstance(parent_value, Mapping):
        raise ValueError("MM-002 input manifest parent record is invalid")
    pinned = parent_value.get("pinned")
    if not isinstance(pinned, Mapping):
        raise ValueError("MM-002 input manifest pinned parent record is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "input_manifest_sha256": _file_hash(EXPECTED_OUTPUT / INPUT_MANIFEST_FILE),
        "protocol_sha256": cast(Mapping[str, object], manifest["protocol"])["sha256"],
        "source_sha256": _canonical_json_sha256(manifest["source"]),
        "config_sha256": manifest["config_sha256"],
        "parent_artifact_manifest_sha256": pinned["artifact_manifest_sha256"],
    }


def _mark_formal_started(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    record = _formal_start_record(manifest)
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path = output / STARTED_FILE
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return record


def _load_parent_rows(output: Path) -> list[dict[str, Any]]:
    value = _read_json(output / PARENT_COPY_ROOT / parent.INTEGRATION_ROWS_FILE)
    return parent._validate_integration_rows(value)


def _result_record(
    evidence: Mapping[str, object],
    summary: Mapping[str, object],
    formal_start: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "epistemic_role": "outcome-informed diagnostic; not independent confirmation",
        "formal_start": dict(formal_start),
        "evidence_sha256": _canonical_json_sha256(evidence),
        "summary": dict(summary),
    }


def _artifact_manifest(output: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": {str(path): _file_record(output / path) for path in ARTIFACT_FILES},
    }


def _write_artifact_manifest(output: Path) -> None:
    _require_membership(output, ARTIFACT_FILES)
    _write_json(output / ARTIFACT_MANIFEST_FILE, _artifact_manifest(output))


def _assert_nested_close(saved: object, regenerated: object, *, path: str = "evidence") -> None:
    if isinstance(saved, Mapping) and isinstance(regenerated, Mapping):
        if set(saved) != set(regenerated):
            raise ValueError(f"semantic {path} keys differ")
        for key in saved:
            _assert_nested_close(saved[key], regenerated[key], path=f"{path}.{key}")
        return
    if isinstance(saved, Sequence) and not isinstance(saved, (str, bytes)):
        if not isinstance(regenerated, Sequence) or isinstance(regenerated, (str, bytes)):
            raise ValueError(f"semantic {path} type differs")
        if len(saved) != len(regenerated):
            raise ValueError(f"semantic {path} length differs")
        for index, (left, right) in enumerate(zip(saved, regenerated, strict=True)):
            _assert_nested_close(left, right, path=f"{path}[{index}]")
        return
    if isinstance(saved, float) or isinstance(regenerated, float):
        if isinstance(saved, bool) or isinstance(regenerated, bool):
            raise ValueError(f"semantic {path} boolean/float type differs")
        if not math.isclose(float(cast(float, saved)), float(cast(float, regenerated)), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"semantic {path} differs")
        return
    if saved != regenerated:
        raise ValueError(f"semantic {path} differs")


@_integrity_boundary
def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Copy and bind the sealed MM-001 package without creating outcomes."""

    _assert_expected_output(output)
    snapshot = _parent_snapshot()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("MM-002 output must be absent or empty before preparation")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / PARENT_COPY_ROOT
    destination.mkdir(parents=True, exist_ok=False)
    live_root = REPO_ROOT / PARENT_OUTPUT
    for relative in PARENT_PACKAGE_FILES:
        shutil.copy2(live_root / relative, destination / relative)
    _validate_parent_copy(output, snapshot)
    (output / PROTOCOL_COPY_FILE).write_bytes((REPO_ROOT / PROTOCOL_DOC).read_bytes())
    _write_json(output / INPUT_MANIFEST_FILE, _expected_input_manifest(output))
    result = verify(output)
    return {**result, "status": "prepared_only"}


@_integrity_boundary
def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Consume MM-002, execute every frozen trajectory, and seal the package."""

    _assert_expected_output(output)
    manifest = _validate_prepared(output)
    _verify_live_parent()
    formal_start = _mark_formal_started(output, manifest)
    table, _ = parent._load_feature_table(output / PARENT_COPY_ROOT / parent.FEATURE_FILE)
    parent_rows = _load_parent_rows(output)
    evidence = method.validate_evidence(method.execute(table))
    method.assert_parent_parity(evidence, parent_rows)
    summary = method.summarize(evidence, parent_rows)
    results = _result_record(evidence, summary, formal_start)
    _write_json(output / EVIDENCE_FILE, evidence)
    _write_json(output / RESULT_FILE, results)
    (output / REPORT_FILE).write_text(method.report_text(summary), encoding="utf-8")
    _write_artifact_manifest(output)
    _verify_live_parent()
    verify(output)
    return results


@_integrity_boundary
def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Verify exact membership, sources, parent copy, raw rows, and decisions."""

    _assert_expected_output(output)
    files, _ = _tree_members(output)
    if files == set(PREPARED_FILES):
        _validate_prepared(output)
        return {
            "status": "verified",
            "outcomes": "prepared_only",
            "classification": "no_outcomes_before_formal_marker",
            "artifact_count": len(PREPARED_FILES),
        }
    _require_membership(output, COMPLETED_FILES)
    saved_artifact_manifest = _read_json(output / ARTIFACT_MANIFEST_FILE)
    if saved_artifact_manifest != _artifact_manifest(output):
        raise ValueError("MM-002 artifact manifest or artifact bytes/modes differ")

    manifest = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(manifest, dict) or manifest != _expected_input_manifest(output):
        raise ValueError("MM-002 input manifest no longer recomputes")
    formal_start = _read_json(output / STARTED_FILE)
    if formal_start != _formal_start_record(manifest):
        raise ValueError("MM-002 formal-start record does not match frozen inputs")
    if stat.S_IMODE((output / STARTED_FILE).stat().st_mode) != 0o444:
        raise ValueError("MM-002 formal-start marker is not read-only")

    evidence_value = _read_json(output / EVIDENCE_FILE)
    evidence = method.validate_evidence(evidence_value)
    parent_rows = _load_parent_rows(output)
    method.assert_parent_parity(evidence, parent_rows)
    summary = method.summarize(evidence, parent_rows)
    results = _result_record(evidence, summary, formal_start)
    if _read_json(output / RESULT_FILE) != results:
        raise ValueError("MM-002 saved results do not recompute from raw evidence")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != method.report_text(summary):
        raise ValueError("MM-002 report is not canonical")
    classification = cast(Mapping[str, object], summary.get("decision", {})).get(
        "classification", "diagnostic_complete"
    )
    return {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": classification,
        "artifact_count": len(ARTIFACT_FILES),
    }


@_integrity_boundary
def verify_semantic(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Retrain every feature-only trajectory in memory and compare exact evidence."""

    verification = verify(output)
    if verification["outcomes"] != "verified_results":
        raise ValueError("semantic verification requires completed MM-002 results")
    table, _ = parent._load_feature_table(output / PARENT_COPY_ROOT / parent.FEATURE_FILE)
    saved = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    regenerated = method.validate_evidence(method.execute(table))
    _assert_nested_close(saved, regenerated)
    parent_rows = _load_parent_rows(output)
    method.assert_parent_parity(regenerated, parent_rows)
    if method.summarize(regenerated, parent_rows) != method.summarize(saved, parent_rows):
        raise ValueError("MM-002 semantic summary differs")
    return {
        **verification,
        "outcomes": "verified_semantic_results",
        "semantic_regeneration": "all world, raw-ridge, integrity, and codec evidence reproduced",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-002 feature-only failure-isolation experiment")
    parser.add_argument("command", choices=("prepare", "run", "verify", "verify-semantic"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run":
        result = run(args.output)
    elif args.command == "verify-semantic":
        result = verify_semantic(args.output)
    else:
        result = verify(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "ARTIFACT_MANIFEST_FILE",
    "DEFAULT_OUTPUT",
    "EVIDENCE_FILE",
    "EXPERIMENT_ID",
    "INPUT_MANIFEST_FILE",
    "InvalidMM002Package",
    "PROTOCOL_COPY_FILE",
    "REPORT_FILE",
    "RESULT_FILE",
    "SCHEMA_VERSION",
    "STARTED_FILE",
    "main",
    "prepare",
    "run",
    "verify",
    "verify_semantic",
]
