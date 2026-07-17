"""Sealed lifecycle and verification for MM-003."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
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

from bench.multimodal_diagnostics import experiment as mm002_experiment
from bench.multimodal_diagnostics import method as mm002_method
from bench.multimodal_preflight import experiment as mm001_experiment

from . import method

SCHEMA_VERSION = "mm003-formal-v1"
EXPERIMENT_ID = "MM-003"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/multimodal_transform_diagnostics/results/MM-003")
EXPECTED_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT
PROTOCOL_DOC = Path("docs/research/2026-07-15-mm003-taesd-projection-scale-isolation-protocol.md")

PROTOCOL_COPY_FILE = Path("MM-003-protocol.md")
INPUT_MANIFEST_FILE = Path("input-manifest.json")
STARTED_FILE = Path("formal-start.json")
TRANSFORMS_FILE = Path("MM-003-transforms.npz")
TRANSFORM_RECORDS_FILE = Path("MM-003-transform-records.json")
EVIDENCE_FILE = Path("MM-003-evidence.json")
RESULT_FILE = Path("MM-003-results.json")
REPORT_FILE = Path("MM-003-report.md")
ARTIFACT_MANIFEST_FILE = Path("artifact-manifest.json")

MM001_ROOT = Path("bench/multimodal_preflight/results/MM-001")
MM002_ROOT = Path("bench/multimodal_diagnostics/results/MM-002")
MM001_COPY_ROOT = Path("inputs/MM-001")
MM002_COPY_ROOT = Path("inputs/MM-002")

MM001_SELECTED = (
    Path("artifact-manifest.json"),
    Path("input-manifest.json"),
    Path("MM-001-features.npz"),
    Path("MM-001-component-audit.npz"),
    Path("MM-001-projections.npz"),
    Path("MM-001-results.json"),
)
MM002_SELECTED = (
    Path("artifact-manifest.json"),
    Path("input-manifest.json"),
    Path("MM-002-evidence.json"),
    Path("MM-002-results.json"),
)

MM001_ARTIFACT_MANIFEST_SHA256 = "a394104a6e9bcdb6c18b206d090e4afb9a540b9e3a2a2875985980e23ecaf52c"
MM001_FEATURE_SHA256 = "3fdf0c988cf0bdb428432b67c71fc7a18404080b6e12bfe8b6226d2276330755"
MM001_COMPONENT_SHA256 = "476da8f2192c6bd57ecab6f861e975fc0827977fa8081462423fa4644e0c89e4"
MM001_PROJECTION_SHA256 = "b131039b540735b0942f9608ab0ebda5a3ccc2018ec9126fdcbaa3b44f9aaaea"
MM001_RESULT_SHA256 = "16504f4bfb36e5252aea9aa6604bc88d64233e256d184bf0e3b2889f5fd76fb7"
MM002_ARTIFACT_MANIFEST_SHA256 = "3e119c35f4a6731df88e68bd16fc7b4e8d44c37776ad72ef660b60681628c139"
MM002_EVIDENCE_SHA256 = "093da00fbc8ef8a68cc8922f463d9febd92fb02f3d9db2b5f5bdb8660f0dbaaa"
MM002_RESULT_SHA256 = "5bf8cb1e37847cced02e304dac07b1b816cb5453d6342b6e08e353354d2953fb"

PREPARED_ROOT_FILES = (PROTOCOL_COPY_FILE, INPUT_MANIFEST_FILE)
PARENT_COPY_FILES = (
    *(MM001_COPY_ROOT / path for path in MM001_SELECTED),
    *(MM002_COPY_ROOT / path for path in MM002_SELECTED),
)
PREPARED_FILES = (*PREPARED_ROOT_FILES, *PARENT_COPY_FILES)
OUTCOME_FILES = (
    STARTED_FILE,
    TRANSFORMS_FILE,
    TRANSFORM_RECORDS_FILE,
    EVIDENCE_FILE,
    RESULT_FILE,
    REPORT_FILE,
)
ARTIFACT_FILES = (*PREPARED_FILES, *OUTCOME_FILES)
COMPLETED_FILES = (*ARTIFACT_FILES, ARTIFACT_MANIFEST_FILE)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InvalidMM003Package(ValueError):
    """Stable fail-closed classification for MM-003 integrity defects."""

    classification = "invalid_MM003_package"


class InvalidMM003ParentParity(ValueError):
    """Stable classification for failure to reproduce inherited evidence."""

    classification = "invalid_MM003_parent_parity"


def _integrity_boundary(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return function(*args, **kwargs)
        except (InvalidMM003Package, InvalidMM003ParentParity):
            raise
        except Exception as error:
            raise InvalidMM003Package(str(error)) from error

    return guarded


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _write_bytes_exclusive(path: Path, payload: bytes, mode: int = 0o644) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_json_exclusive(path: Path, value: object, mode: int = 0o644) -> None:
    _write_bytes_exclusive(path, _json_bytes(value), mode)


def _file_hash(path: Path) -> str:
    return mm001_experiment._file_hash(path)


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
        Path("tests/test_mm003_method.py"),
        Path("tests/test_mm003_experiment.py"),
        *(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "bench/multimodal_transform_diagnostics").glob("*.py")),
    }
    return tuple(sorted({*mm002_experiment._source_paths(), *own}, key=str))


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _source_paths():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bound MM-003 source is missing or a symlink: {relative}")
        output[str(relative)] = _file_hash(path)
    return output


def _dependency_versions() -> dict[str, object]:
    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        versions["prospect"] = importlib.metadata.version("prospect")
    except importlib.metadata.PackageNotFoundError:
        versions["prospect"] = None
    return versions


def _assert_expected_output(output: Path) -> None:
    if output.resolve() != EXPECTED_OUTPUT.resolve():
        raise ValueError(f"MM-003 output must be canonical path {DEFAULT_OUTPUT}")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("MM-003 output must be a real directory")
    resolved = output.resolve()
    for parent_root in (REPO_ROOT / MM001_ROOT, REPO_ROOT / MM002_ROOT):
        parent_resolved = parent_root.resolve()
        if resolved == parent_resolved or resolved in parent_resolved.parents or parent_resolved in resolved.parents:
            raise ValueError("MM-003 output overlaps a protected parent")


def _tree_members(root: Path) -> tuple[set[Path], set[Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"package root must be a real directory: {root}")
    files: set[Path] = set()
    directories: set[Path] = {Path(".")}
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            path = current_path / dirname
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"package contains a non-regular directory: {path}")
            directories.add(path.relative_to(root))
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"package contains a non-regular file: {path}")
            files.add(path.relative_to(root))
    return files, directories


def _expected_directories(files: Sequence[Path]) -> set[Path]:
    directories = {Path(".")}
    for path in files:
        directories.update(path.parents)
    return directories


def _require_membership(output: Path, expected: Sequence[Path]) -> None:
    files, directories = _tree_members(output)
    expected_files = set(expected)
    expected_dirs = _expected_directories(expected)
    if files != expected_files or directories != expected_dirs:
        missing = sorted(str(path) for path in expected_files - files)
        extra = sorted(str(path) for path in files - expected_files)
        missing_dirs = sorted(str(path) for path in expected_dirs - directories)
        extra_dirs = sorted(str(path) for path in directories - expected_dirs)
        raise ValueError(
            "MM-003 membership mismatch: "
            f"missing={missing}, extra={extra}, missing_dirs={missing_dirs}, extra_dirs={extra_dirs}"
        )


def _records(root: Path, files: Sequence[Path]) -> dict[str, dict[str, object]]:
    return {str(relative): _file_record(root / relative) for relative in files}


def _verify_selected_against_manifest(root: Path, selected: Sequence[Path]) -> None:
    manifest = _read_json(root / ARTIFACT_MANIFEST_FILE)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), dict):
        raise ValueError(f"parent artifact manifest is invalid: {root}")
    artifacts = cast(Mapping[str, object], manifest["artifacts"])
    for relative in selected:
        if relative == ARTIFACT_MANIFEST_FILE:
            continue
        expected = artifacts.get(str(relative))
        if not isinstance(expected, Mapping):
            raise ValueError(f"selected parent file is absent from manifest: {root / relative}")
        actual = _file_record(root / relative)
        if any(actual.get(name) != expected_value for name, expected_value in expected.items()):
            raise ValueError(f"selected parent file differs from manifest: {root / relative}")


def _verify_live_parents() -> dict[str, object]:
    mm001 = mm001_experiment.verify(mm001_experiment.DEFAULT_OUTPUT)
    mm002 = mm002_experiment.verify(mm002_experiment.DEFAULT_OUTPUT)
    if mm001.get("outcomes") != "verified_results" or mm002.get("outcomes") != "verified_results":
        raise ValueError("MM-003 requires completed verified MM-001 and MM-002 parents")
    mm001_root = REPO_ROOT / MM001_ROOT
    mm002_root = REPO_ROOT / MM002_ROOT
    pins = {
        mm001_root / "artifact-manifest.json": MM001_ARTIFACT_MANIFEST_SHA256,
        mm001_root / "MM-001-features.npz": MM001_FEATURE_SHA256,
        mm001_root / "MM-001-component-audit.npz": MM001_COMPONENT_SHA256,
        mm001_root / "MM-001-projections.npz": MM001_PROJECTION_SHA256,
        mm001_root / "MM-001-results.json": MM001_RESULT_SHA256,
        mm002_root / "artifact-manifest.json": MM002_ARTIFACT_MANIFEST_SHA256,
        mm002_root / "MM-002-evidence.json": MM002_EVIDENCE_SHA256,
        mm002_root / "MM-002-results.json": MM002_RESULT_SHA256,
    }
    for path, expected in pins.items():
        if _file_hash(path) != expected:
            raise ValueError(f"pinned parent hash differs: {path}")
    _verify_selected_against_manifest(mm001_root, MM001_SELECTED)
    _verify_selected_against_manifest(mm002_root, MM002_SELECTED)
    mm002_input = _read_json(mm002_root / "input-manifest.json")
    embedded = cast(Mapping[str, object], cast(Mapping[str, object], mm002_input["parent"])["pinned"])
    if (
        embedded.get("artifact_manifest_sha256") != MM001_ARTIFACT_MANIFEST_SHA256
        or embedded.get("feature_sha256") != MM001_FEATURE_SHA256
        or embedded.get("result_sha256") != MM001_RESULT_SHA256
    ):
        raise ValueError("MM-002 embedded MM-001 receipt differs from live pins")
    return {
        "MM-001": {
            "verification": mm001,
            "live_path": str(MM001_ROOT),
            "copy_path": str(MM001_COPY_ROOT),
            "files": _records(mm001_root, mm001_experiment.PACKAGE_FILES),
            "selected_files": [str(path) for path in MM001_SELECTED],
            "pinned": {
                "artifact_manifest_sha256": MM001_ARTIFACT_MANIFEST_SHA256,
                "feature_sha256": MM001_FEATURE_SHA256,
                "component_sha256": MM001_COMPONENT_SHA256,
                "projection_sha256": MM001_PROJECTION_SHA256,
                "result_sha256": MM001_RESULT_SHA256,
            },
        },
        "MM-002": {
            "verification": mm002,
            "live_path": str(MM002_ROOT),
            "copy_path": str(MM002_COPY_ROOT),
            "files": _records(mm002_root, mm002_experiment.COMPLETED_FILES),
            "selected_files": [str(path) for path in MM002_SELECTED],
            "pinned": {
                "artifact_manifest_sha256": MM002_ARTIFACT_MANIFEST_SHA256,
                "evidence_sha256": MM002_EVIDENCE_SHA256,
                "result_sha256": MM002_RESULT_SHA256,
            },
        },
        "scientific_relationship": (
            "outcome-informed mechanism diagnostic reusing the same eight videos; not independent evidence"
        ),
    }


def _validate_parent_copies(output: Path, snapshot: Mapping[str, object]) -> None:
    for parent_id, copy_root, selected in (
        ("MM-001", MM001_COPY_ROOT, MM001_SELECTED),
        ("MM-002", MM002_COPY_ROOT, MM002_SELECTED),
    ):
        root = output / copy_root
        files, directories = _tree_members(root)
        if files != set(selected) or directories != {Path(".")}:
            raise ValueError(f"copied {parent_id} receipt has unexpected membership")
        parent_record = snapshot.get(parent_id)
        if not isinstance(parent_record, Mapping) or not isinstance(parent_record.get("files"), Mapping):
            raise ValueError(f"live {parent_id} snapshot is invalid")
        live_records = cast(Mapping[str, object], parent_record["files"])
        for relative in selected:
            if _file_record(root / relative) != live_records[str(relative)]:
                raise ValueError(f"copied {parent_id} file differs from live sealed parent: {relative}")
        _verify_selected_against_manifest(root, selected)


def _load_npz(path: Path, expected: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]]) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"array package is missing or a symlink: {path}")
    with np.load(path, allow_pickle=False) as package:
        if set(package.files) != set(expected):
            raise ValueError(f"array package keys differ: {path}")
        arrays = {name: np.asarray(package[name]).copy() for name in package.files}
    for name, (dtype, shape) in expected.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ValueError(f"array schema differs for {path}:{name}")
        if arrays[name].dtype.hasobject:
            raise ValueError("object arrays are forbidden")
    return arrays


def _load_analysis_inputs(
    output: Path,
) -> tuple[method.VisualTable, np.ndarray, dict[str, object], dict[str, object]]:
    mm001_root = output / MM001_COPY_ROOT
    feature_table, feature_schema = mm001_experiment._load_feature_table(mm001_root / mm001_experiment.FEATURE_FILE)
    components = _load_npz(
        mm001_root / mm001_experiment.COMPONENT_AUDIT_FILE,
        {
            "taesd_latents": (np.dtype("<f4"), (477, 4, 8, 8)),
            "target_taesd_latents": (np.dtype("<f4"), (477, 4, 8, 8)),
            "snac_code_ids": (np.dtype("<i8"), (477, 84)),
            "t5_pooled_states": (np.dtype("<f4"), (477, 256)),
            "t5_masked_input_ids": (np.dtype("<i8"), (477, 27)),
            "t5_target_ids": (np.dtype("<i8"), (477, 8)),
            "t5_generated_ids": (np.dtype("<i8"), (477, 33)),
        },
    )
    projections = _load_npz(
        mm001_root / mm001_experiment.PROJECTION_FILE,
        {
            "vision_projection_matrix": (np.dtype("<f8"), (256, 32)),
            "audio_projection_matrix": (np.dtype("<f8"), (84, 32)),
            "text_projection_matrix": (np.dtype("<f8"), (256, 32)),
        },
    )
    raw = method.VisualTable(
        video_ids=np.asarray(feature_table.video_ids, dtype=str).copy(),
        timestamps=np.asarray(feature_table.timestamps, dtype=float).copy(),
        current=np.asarray(components["taesd_latents"], dtype=float).reshape(477, 256),
        target=np.asarray(components["target_taesd_latents"], dtype=float).reshape(477, 256),
    )
    projection = np.asarray(projections["vision_projection_matrix"], dtype=float)
    parent_evidence = mm002_method.validate_evidence(
        _read_json(output / MM002_COPY_ROOT / mm002_experiment.EVIDENCE_FILE)
    )
    try:
        alignment = method.alignment_record(
            raw,
            projection,
            np.asarray(feature_table.vision, dtype=float),
            np.asarray(feature_table.target_vision, dtype=float),
        )
        parent_preflight = method.parent_preflight_record(raw, projection, parent_evidence)
    except ValueError as error:
        raise InvalidMM003ParentParity(str(error)) from error
    return (
        raw,
        projection,
        cast(dict[str, object], parent_evidence),
        {
            "feature_schema": feature_schema,
            "alignment": alignment,
            "parent_preflight": parent_preflight,
        },
    )


def _expected_input_manifest(output: Path) -> dict[str, object]:
    snapshot = _verify_live_parents()
    _validate_parent_copies(output, snapshot)
    source_protocol = REPO_ROOT / PROTOCOL_DOC
    if (output / PROTOCOL_COPY_FILE).read_bytes() != source_protocol.read_bytes():
        raise ValueError("MM-003 protocol copy differs from bound source")
    _, _, _, input_record = _load_analysis_inputs(output)
    config = method.config_record()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared_before_formal_execution",
        "protocol": {
            "source": str(PROTOCOL_DOC),
            "copy": str(PROTOCOL_COPY_FILE),
            "sha256": _file_hash(source_protocol),
        },
        "source": _source_hashes(),
        "dependencies": _dependency_versions(),
        "parents": snapshot,
        "input_validation": input_record,
        "config": config,
        "config_sha256": _canonical_json_sha256(config),
        "prepared_membership_sha256": _canonical_json_sha256([str(path) for path in PREPARED_FILES]),
        "expected_prepared_files": [str(path) for path in PREPARED_FILES],
        "expected_completed_files": [str(path) for path in COMPLETED_FILES],
    }


def _validate_prepared(output: Path) -> dict[str, object]:
    _require_membership(output, PREPARED_FILES)
    saved = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(saved, dict) or saved != _expected_input_manifest(output):
        raise ValueError("MM-003 input manifest no longer recomputes")
    return cast(dict[str, object], saved)


def _formal_start_record(manifest: Mapping[str, object]) -> dict[str, object]:
    parents = cast(Mapping[str, Mapping[str, Any]], manifest["parents"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "input_manifest_sha256": _file_hash(EXPECTED_OUTPUT / INPUT_MANIFEST_FILE),
        "protocol_sha256": cast(Mapping[str, object], manifest["protocol"])["sha256"],
        "source_sha256": _canonical_json_sha256(manifest["source"]),
        "config_sha256": manifest["config_sha256"],
        "prepared_membership_sha256": manifest["prepared_membership_sha256"],
        "mm001_artifact_manifest_sha256": parents["MM-001"]["pinned"]["artifact_manifest_sha256"],
        "mm002_artifact_manifest_sha256": parents["MM-002"]["pinned"]["artifact_manifest_sha256"],
    }


def _mark_formal_started(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    record = _formal_start_record(manifest)
    _write_json_exclusive(output / STARTED_FILE, record, 0o444)
    return record


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    cast(Any, np.savez_compressed)(
        buffer,
        **{name: np.asarray(value) for name, value in sorted(arrays.items())},
    )
    return buffer.getvalue()


def _load_transform_arrays(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    expected: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}
    for record in records:
        prefix = f"fold_{record['fold']}__{record['representation_id']}__"
        for name, metadata in cast(Mapping[str, Mapping[str, Any]], record["parameter_arrays"]).items():
            expected[prefix + name] = (
                np.dtype(cast(str, metadata["dtype"])),
                tuple(cast(Sequence[int], metadata["shape"])),
            )
    return _load_npz(path, expected)


def _assert_arrays_equal(saved: Mapping[str, np.ndarray], regenerated: Mapping[str, np.ndarray]) -> None:
    if set(saved) != set(regenerated):
        raise ValueError("transform array keys differ")
    for name in saved:
        if saved[name].dtype != regenerated[name].dtype or saved[name].shape != regenerated[name].shape:
            raise ValueError(f"transform array schema differs: {name}")
        if not np.array_equal(saved[name], regenerated[name]):
            raise ValueError(f"transform array values differ: {name}")


def _assert_nested_close(saved: object, regenerated: object, *, path: str = "value") -> None:
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


def _result_record(
    formal_start: Mapping[str, object],
    alignment: Mapping[str, object],
    transform_records: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "epistemic_role": "outcome-informed feature-only diagnostic; not independent confirmation",
        "formal_start": dict(formal_start),
        "alignment": dict(alignment),
        "transform_records_sha256": _canonical_json_sha256(transform_records),
        "evidence_sha256": _canonical_json_sha256(evidence),
        "summary": dict(summary),
    }


def _assert_parent_parity(
    evidence: object,
    parent_evidence: Mapping[str, object],
) -> dict[str, object]:
    try:
        return method.assert_parent_parity(evidence, parent_evidence)
    except ValueError as error:
        raise InvalidMM003ParentParity(str(error)) from error


def _artifact_manifest(output: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": {str(path): _file_record(output / path) for path in ARTIFACT_FILES},
    }


@_integrity_boundary
def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Copy and bind selected parent receipts without creating outcomes."""

    _assert_expected_output(output)
    snapshot = _verify_live_parents()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("MM-003 output must be absent or empty before preparation")
    output.mkdir(parents=True, exist_ok=True)
    for source_root, copy_root, selected in (
        (REPO_ROOT / MM001_ROOT, MM001_COPY_ROOT, MM001_SELECTED),
        (REPO_ROOT / MM002_ROOT, MM002_COPY_ROOT, MM002_SELECTED),
    ):
        destination = output / copy_root
        destination.mkdir(parents=True, exist_ok=False)
        for relative in selected:
            shutil.copy2(source_root / relative, destination / relative)
    _validate_parent_copies(output, snapshot)
    shutil.copy2(REPO_ROOT / PROTOCOL_DOC, output / PROTOCOL_COPY_FILE)
    manifest = _expected_input_manifest(output)
    (output / INPUT_MANIFEST_FILE).write_bytes(_json_bytes(manifest))
    result = verify(output)
    return {**result, "status": "prepared_only"}


@_integrity_boundary
def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Consume MM-003 once, execute the frozen analysis, and seal it."""

    _assert_expected_output(output)
    manifest = _validate_prepared(output)
    _verify_live_parents()
    formal_start = _mark_formal_started(output, manifest)
    raw, projection, parent_evidence, input_record = _load_analysis_inputs(output)
    arrays, transform_records, evidence = method.execute(raw, projection, parent_evidence)
    _assert_parent_parity(evidence, parent_evidence)
    summary = method.summarize(evidence, transform_records, parent_evidence)
    result = _result_record(
        formal_start,
        cast(Mapping[str, object], input_record["alignment"]),
        transform_records,
        evidence,
        summary,
    )
    _write_bytes_exclusive(output / TRANSFORMS_FILE, _npz_bytes(arrays))
    _write_json_exclusive(output / TRANSFORM_RECORDS_FILE, transform_records)
    _write_json_exclusive(output / EVIDENCE_FILE, evidence)
    _write_json_exclusive(output / RESULT_FILE, result)
    _write_bytes_exclusive(output / REPORT_FILE, method.report_text(summary).encode("utf-8"))
    _require_membership(output, ARTIFACT_FILES)
    _write_json_exclusive(output / ARTIFACT_MANIFEST_FILE, _artifact_manifest(output))
    _verify_live_parents()
    verify(output)
    return result


@_integrity_boundary
def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Fast structural, transform, parity, and decision verification."""

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
    if _read_json(output / ARTIFACT_MANIFEST_FILE) != _artifact_manifest(output):
        raise ValueError("MM-003 artifact manifest or artifact bytes/modes differ")
    manifest = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(manifest, dict) or manifest != _expected_input_manifest(output):
        raise ValueError("MM-003 input manifest no longer recomputes")
    formal_start = _read_json(output / STARTED_FILE)
    if formal_start != _formal_start_record(manifest):
        raise ValueError("MM-003 formal marker differs from frozen inputs")
    if stat.S_IMODE((output / STARTED_FILE).stat().st_mode) != 0o444:
        raise ValueError("MM-003 formal marker is not read-only")

    raw, projection, parent_evidence, input_record = _load_analysis_inputs(output)
    saved_records = method.validate_transform_records(_read_json(output / TRANSFORM_RECORDS_FILE))
    saved_arrays = _load_transform_arrays(output / TRANSFORMS_FILE, saved_records)
    _, regenerated_records, regenerated_arrays = method.fit_all_transforms(raw, projection)
    if saved_records != regenerated_records:
        raise ValueError("MM-003 transform records do not recompute")
    _assert_arrays_equal(saved_arrays, regenerated_arrays)

    evidence = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    _assert_parent_parity(evidence, parent_evidence)
    summary = method.summarize(evidence, saved_records, parent_evidence)
    expected_result = _result_record(
        formal_start,
        cast(Mapping[str, object], input_record["alignment"]),
        saved_records,
        evidence,
        summary,
    )
    if _read_json(output / RESULT_FILE) != expected_result:
        raise ValueError("MM-003 result does not recompute from evidence")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != method.report_text(summary):
        raise ValueError("MM-003 report is not canonical")
    _verify_live_parents()
    return {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": cast(Mapping[str, object], summary["decision"])["classification"],
        "artifact_count": len(ARTIFACT_FILES),
    }


@_integrity_boundary
def verify_semantic(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Retrain new trajectories in memory and reproduce all saved evidence."""

    verification = verify(output)
    if verification["outcomes"] != "verified_results":
        raise ValueError("semantic verification requires completed MM-003 outcomes")
    raw, projection, parent_evidence, _ = _load_analysis_inputs(output)
    regenerated_arrays, regenerated_records, regenerated_evidence = method.execute(raw, projection, parent_evidence)
    saved_records = method.validate_transform_records(_read_json(output / TRANSFORM_RECORDS_FILE))
    saved_arrays = _load_transform_arrays(output / TRANSFORMS_FILE, saved_records)
    saved_evidence = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    if saved_records != regenerated_records:
        raise ValueError("semantic transform records differ")
    _assert_arrays_equal(saved_arrays, regenerated_arrays)
    _assert_nested_close(saved_evidence, regenerated_evidence, path="evidence")
    saved_summary = method.summarize(saved_evidence, saved_records, parent_evidence)
    regenerated_summary = method.summarize(regenerated_evidence, regenerated_records, parent_evidence)
    _assert_nested_close(saved_summary, regenerated_summary, path="summary")
    return {
        **verification,
        "outcomes": "verified_semantic_results",
        "semantic_regeneration": (
            "all transforms, probes, new world trajectories, integrity rows, and decisions reproduced"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-003 TAESD projection/scale isolation")
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
    "InvalidMM003Package",
    "InvalidMM003ParentParity",
    "PROTOCOL_COPY_FILE",
    "REPORT_FILE",
    "RESULT_FILE",
    "SCHEMA_VERSION",
    "STARTED_FILE",
    "TRANSFORM_RECORDS_FILE",
    "TRANSFORMS_FILE",
    "main",
    "prepare",
    "run",
    "verify",
    "verify_semantic",
]
