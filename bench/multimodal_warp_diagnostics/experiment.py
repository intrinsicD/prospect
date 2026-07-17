"""Sealed preparation, execution, and verification lifecycle for MM-006."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

from bench.multimodal_horizon_diagnostics import experiment as mm005_experiment
from bench.multimodal_horizon_diagnostics import method as mm005_method
from bench.multimodal_preflight import experiment as mm001_experiment
from bench.multimodal_spatial_diagnostics import experiment as mm004_experiment

from . import method

SCHEMA_VERSION = "mm006-formal-v1"
EXPERIMENT_ID = "MM-006"
PARENT_CLASSIFICATION = "half_second_tested_spatial_local_linear_objective_failure_supported"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/multimodal_warp_diagnostics/results/MM-006")
EXPECTED_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT
PROTOCOL_DOC = Path("docs/research/2026-07-15-mm006-causal-warp-ceiling-protocol.md")

PROTOCOL_COPY_FILE = Path("MM-006-protocol.md")
INPUT_MANIFEST_FILE = Path("input-manifest.json")
STARTED_FILE = Path("formal-start.json")
EVIDENCE_FILE = Path("MM-006-evidence.json")
RESULT_FILE = Path("MM-006-results.json")
REPORT_FILE = Path("MM-006-report.md")
ARTIFACT_MANIFEST_FILE = Path("artifact-manifest.json")

MM005_ROOT = mm005_experiment.DEFAULT_OUTPUT
MM005_COPY_ROOT = Path("inputs/MM-005")
MM005_SELECTED = (
    Path("artifact-manifest.json"),
    Path("input-manifest.json"),
    Path("MM-005-evidence.json"),
    Path("MM-005-results.json"),
    Path("inputs/MM-004/MM-004-pixel-grids.npz"),
    Path("inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-features.npz"),
    Path("inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz"),
)
MM005_PINS: dict[Path, str] = {
    Path("artifact-manifest.json"): "c0e8fc7772799631b1b9e57167d4b8d70b71dc14f1fbd8d21847a9695d9c3e66",
    Path("input-manifest.json"): "cadf01c1398d15a5ae9dbad92967f256601670b52176de60f462d3618c2040c0",
    Path("MM-005-evidence.json"): "c02bfd0d0dd5389270a8e126a69188d6e88ddf2cc6cedd1c666964562917883b",
    Path("MM-005-results.json"): "8475f3f93acd8933644b344f0ef72cf78ccfd637269c8b532acb0eb907bef0f9",
    Path("inputs/MM-004/MM-004-pixel-grids.npz"): ("cca261a941e68a7ddc510eee3a3af958d33b6abaf958cb5562ed6b66c22f47c8"),
    Path("inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-features.npz"): (
        "3fdf0c988cf0bdb428432b67c71fc7a18404080b6e12bfe8b6226d2276330755"
    ),
    Path("inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz"): (
        "476da8f2192c6bd57ecab6f861e975fc0827977fa8081462423fa4644e0c89e4"
    ),
}

PIXEL_RELATIVE = Path("inputs/MM-004/MM-004-pixel-grids.npz")
FEATURE_RELATIVE = Path("inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-features.npz")
COMPONENT_RELATIVE = Path("inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz")

PREPARED_ROOT_FILES = (PROTOCOL_COPY_FILE, INPUT_MANIFEST_FILE)
PARENT_COPY_FILES = tuple(MM005_COPY_ROOT / path for path in MM005_SELECTED)
PREPARED_FILES = (*PREPARED_ROOT_FILES, *PARENT_COPY_FILES)
OUTCOME_FILES = (STARTED_FILE, EVIDENCE_FILE, RESULT_FILE, REPORT_FILE)
ARTIFACT_FILES = (*PREPARED_FILES, *OUTCOME_FILES)
COMPLETED_FILES = (*ARTIFACT_FILES, ARTIFACT_MANIFEST_FILE)
GENERATED_0644_FILES = (
    PROTOCOL_COPY_FILE,
    INPUT_MANIFEST_FILE,
    EVIDENCE_FILE,
    RESULT_FILE,
    REPORT_FILE,
    ARTIFACT_MANIFEST_FILE,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InvalidMM006Package(ValueError):
    """Stable fail-closed classification for MM-006 package defects."""

    classification = "invalid_MM006_package"


class InvalidMM006ParentAlignment(ValueError):
    """Stable pre-marker classification for parent/alignment defects."""

    classification = "invalid_MM006_parent_alignment"


def _integrity_boundary(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return function(*args, **kwargs)
        except (InvalidMM006Package, InvalidMM006ParentAlignment):
            raise
        except Exception as error:
            raise InvalidMM006Package(str(error)) from error

    return guarded


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input is missing or a symlink: {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or a symlink: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or a symlink: {path}")
    metadata = path.stat()
    return {"sha256": _file_hash(path), "bytes": metadata.st_size, "mode": stat.S_IMODE(metadata.st_mode)}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_fsynced(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"directory path is a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"directory path is not a directory: {path}")
        return
    _mkdir_fsynced(path.parent)
    path.mkdir()
    _fsync_directory(path.parent)


def _write_bytes_exclusive(path: Path, payload: bytes, mode: int = 0o644) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("exclusive artifact write made no progress")
            written += count
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_json_exclusive(path: Path, value: object, mode: int = 0o644) -> None:
    _write_bytes_exclusive(path, _json_bytes(value), mode)


def _copy_file_exclusive(source: Path, destination: Path, *, mode: int | None = None) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"copy source is missing or a symlink: {source}")
    _mkdir_fsynced(destination.parent)
    selected_mode = stat.S_IMODE(source.stat().st_mode) if mode is None else mode
    _write_bytes_exclusive(destination, source.read_bytes(), selected_mode)


def _source_paths() -> tuple[Path, ...]:
    package_root = REPO_ROOT / "bench/multimodal_warp_diagnostics"
    expected_package = {
        Path("bench/multimodal_warp_diagnostics/__init__.py"),
        Path("bench/multimodal_warp_diagnostics/__main__.py"),
        Path("bench/multimodal_warp_diagnostics/experiment.py"),
        Path("bench/multimodal_warp_diagnostics/method.py"),
    }
    actual_package = {path.relative_to(REPO_ROOT) for path in package_root.glob("*.py")}
    if actual_package != expected_package:
        raise ValueError("MM-006 package source membership differs from the frozen four-file set")
    own = {
        PROTOCOL_DOC,
        Path("tests/test_mm006_method.py"),
        Path("tests/test_mm006_experiment.py"),
        *actual_package,
    }
    paths = tuple(sorted({*mm005_experiment._source_paths(), *own}, key=str))
    if len(paths) != 61:
        raise ValueError(f"MM-006 source membership must contain exactly 61 files, got {len(paths)}")
    return paths


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _source_paths():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bound MM-006 source is missing or a symlink: {relative}")
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
        raise ValueError(f"MM-006 output must be canonical path {DEFAULT_OUTPUT}")
    current = output
    while True:
        if current.is_symlink():
            raise ValueError(f"MM-006 output path contains a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent
    if output.exists() and not output.is_dir():
        raise ValueError("MM-006 output must be a real directory")
    resolved = output.resolve()
    protected = (REPO_ROOT / MM005_ROOT,)
    for parent_root in protected:
        parent_resolved = parent_root.resolve()
        if resolved == parent_resolved or resolved in parent_resolved.parents or parent_resolved in resolved.parents:
            raise ValueError("MM-006 output overlaps a protected parent")


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
    expected_directories = _expected_directories(expected)
    if files != expected_files or directories != expected_directories:
        raise ValueError(
            "MM-006 membership mismatch: "
            f"missing={sorted(str(path) for path in expected_files - files)}, "
            f"extra={sorted(str(path) for path in files - expected_files)}"
        )


def _assert_generated_modes(output: Path) -> None:
    for relative in GENERATED_0644_FILES:
        path = output / relative
        if path.exists() and stat.S_IMODE(path.stat().st_mode) != 0o644:
            raise ValueError(f"MM-006 generated file mode differs from 0644: {relative}")
    marker = output / STARTED_FILE
    if marker.exists() and stat.S_IMODE(marker.stat().st_mode) != 0o444:
        raise ValueError("MM-006 formal marker is not mode 0444")


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
        if not isinstance(expected, Mapping) or _file_record(root / relative) != dict(expected):
            raise ValueError(f"selected parent file differs from manifest: {relative}")


def _replay_parent_unchecked(root: Path) -> dict[str, object]:
    evidence = mm005_method.validate_evidence(_read_json(root / mm005_experiment.EVIDENCE_FILE))
    summary = mm005_method.summarize(evidence)
    result = _read_json(root / mm005_experiment.RESULT_FILE)
    if not isinstance(result, Mapping) or result.get("summary") != summary:
        raise ValueError("MM-005 evidence/result summary relationship does not replay")
    if result.get("evidence_sha256") != _canonical_json_sha256(evidence):
        raise ValueError("MM-005 result does not bind canonical evidence")
    synthetic = summary.get("synthetic_control")
    decision = summary.get("decision")
    if (
        not isinstance(synthetic, Mapping)
        or synthetic.get("positive_passes") is not True
        or synthetic.get("negative_passes") is not True
    ):
        raise ValueError("MM-005 parent synthetic controls did not both pass")
    if not isinstance(decision, Mapping) or decision.get("classification") != PARENT_CLASSIFICATION:
        raise ValueError("MM-005 copied decision differs from frozen parent branch")
    return {
        "classification": PARENT_CLASSIFICATION,
        "synthetic_positive_passes": True,
        "synthetic_negative_passes": True,
        "evidence_canonical_sha256": _canonical_json_sha256(evidence),
        "summary_canonical_sha256": _canonical_json_sha256(summary),
    }


def _replay_parent(root: Path) -> dict[str, object]:
    try:
        return _replay_parent_unchecked(root)
    except InvalidMM006ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM006ParentAlignment(str(error)) from error


def _verify_live_parent_unchecked() -> dict[str, object]:
    verification = mm005_experiment.verify(MM005_ROOT)
    if (
        verification.get("outcomes") != "verified_results"
        or verification.get("classification") != PARENT_CLASSIFICATION
    ):
        raise ValueError("MM-006 requires completed verified MM-005 results")
    root = REPO_ROOT / MM005_ROOT
    for relative, expected in MM005_PINS.items():
        if _file_hash(root / relative) != expected:
            raise ValueError(f"pinned MM-005 parent hash differs: {relative}")
    _verify_selected_against_manifest(root, MM005_SELECTED)
    replay = _replay_parent(root)
    return {
        "experiment_id": "MM-005",
        "classification": PARENT_CLASSIFICATION,
        "verification": verification,
        "live_path": str(MM005_ROOT),
        "copy_path": str(MM005_COPY_ROOT),
        "files": _records(root, mm005_experiment.COMPLETED_FILES),
        "selected_files": [str(path) for path in MM005_SELECTED],
        "pinned": {str(path): digest for path, digest in MM005_PINS.items()},
        "replay": replay,
        "scientific_relationship": (
            "outcome-informed direct child reusing the same eight videos; not independent evidence"
        ),
    }


def _verify_live_parent() -> dict[str, object]:
    try:
        return _verify_live_parent_unchecked()
    except InvalidMM006ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM006ParentAlignment(str(error)) from error


def _validate_parent_copy_unchecked(output: Path, snapshot: Mapping[str, object]) -> dict[str, object]:
    root = output / MM005_COPY_ROOT
    files, directories = _tree_members(root)
    if files != set(MM005_SELECTED) or directories != _expected_directories(MM005_SELECTED):
        raise ValueError("copied MM-005 receipt has unexpected membership")
    records = snapshot.get("files")
    if not isinstance(records, Mapping):
        raise ValueError("live MM-005 snapshot has no file records")
    for relative in MM005_SELECTED:
        expected = records.get(str(relative))
        if not isinstance(expected, Mapping) or _file_record(root / relative) != dict(expected):
            raise ValueError(f"copied MM-005 file differs from live parent: {relative}")
    _verify_selected_against_manifest(root, MM005_SELECTED)
    replay = _replay_parent(root)
    if replay != snapshot.get("replay"):
        raise ValueError("copied MM-005 replay differs from live parent")
    return replay


def _validate_parent_copy(output: Path, snapshot: Mapping[str, object]) -> dict[str, object]:
    try:
        return _validate_parent_copy_unchecked(output, snapshot)
    except InvalidMM006ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM006ParentAlignment(str(error)) from error


def _strict_component_arrays(path: Path) -> dict[str, np.ndarray]:
    return cast(dict[str, np.ndarray], mm004_experiment._load_component_arrays(path))


def _strict_pixel_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays = mm004_experiment._load_npz(path, mm004_experiment.PIXEL_SCHEMA)
    for name in ("pixel_current", "pixel_target"):
        if float(np.min(arrays[name])) < 0.0 or float(np.max(arrays[name])) > 1.0:
            raise ValueError(f"copied {name} values must remain in [0,1]")
    return cast(dict[str, np.ndarray], arrays)


def _load_analysis_inputs_unchecked(
    output: Path,
) -> tuple[mm005_method.MatchedPanel, mm005_method.MatchedPanel, dict[str, object]]:
    parent_root = output / MM005_COPY_ROOT
    feature_path = parent_root / FEATURE_RELATIVE
    component_path = parent_root / COMPONENT_RELATIVE
    pixel_path = parent_root / PIXEL_RELATIVE
    feature_table, feature_schema = mm001_experiment._load_feature_table(feature_path)
    components = _strict_component_arrays(component_path)
    pixels = _strict_pixel_arrays(pixel_path)
    video_ids = np.asarray(feature_table.video_ids, dtype="<U11")
    timestamps = np.asarray(feature_table.timestamps, dtype="<f8")
    taesd_raw = mm005_method.raw_grid_table(
        video_ids,
        timestamps,
        components["taesd_latents"],
        components["target_taesd_latents"],
        expected_channels=4,
    )
    pixel_raw = mm005_method.raw_grid_table(
        video_ids,
        timestamps,
        pixels["pixel_current"],
        pixels["pixel_target"],
        expected_channels=3,
    )
    taesd = mm005_method.matched_panel(taesd_raw)
    pixel = mm005_method.matched_panel(pixel_raw)
    alignment = mm005_method.alignment_record(taesd, pixel)
    copied_parent_evidence = mm005_method.validate_evidence(_read_json(parent_root / mm005_experiment.EVIDENCE_FILE))
    if alignment != copied_parent_evidence["alignment"]:
        raise ValueError("MM-006 reconstructed alignment differs from MM-005 evidence")
    return (
        taesd,
        pixel,
        {
            "feature_file": _file_record(feature_path),
            "feature_schema": feature_schema,
            "component_file": _file_record(component_path),
            "component_schema": {
                name: {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _array_sha256(value)}
                for name, value in sorted(components.items())
            },
            "pixel_file": _file_record(pixel_path),
            "pixel_schema": {
                name: {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _array_sha256(value)}
                for name, value in sorted(pixels.items())
            },
            "alignment": alignment,
            "alignment_sha256": _canonical_json_sha256(alignment),
        },
    )


def _load_analysis_inputs(
    output: Path,
) -> tuple[mm005_method.MatchedPanel, mm005_method.MatchedPanel, dict[str, object]]:
    try:
        return _load_analysis_inputs_unchecked(output)
    except InvalidMM006ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM006ParentAlignment(str(error)) from error


def _config_record() -> dict[str, object]:
    return {
        "method": method.config_record(),
        "lifecycle": {
            "prepared_files": [str(path) for path in PREPARED_FILES],
            "completed_files": [str(path) for path in COMPLETED_FILES],
            "prepared_file_count": 9,
            "artifact_file_count": 13,
            "completed_file_count": 14,
            "generated_file_mode": "0644",
            "formal_marker_mode": "0444",
            "semantic_tolerance": {"rtol": 1e-12, "atol": 1e-12},
            "media_used": False,
            "model_inference_used": False,
            "oracle_diagnostic_only": True,
        },
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    snapshot = _verify_live_parent()
    replay = _validate_parent_copy(output, snapshot)
    source_protocol = REPO_ROOT / PROTOCOL_DOC
    protocol_copy = output / PROTOCOL_COPY_FILE
    if (
        protocol_copy.read_bytes() != source_protocol.read_bytes()
        or stat.S_IMODE(protocol_copy.stat().st_mode) != 0o644
    ):
        raise ValueError("MM-006 protocol copy differs from bound source or mode")
    _, _, input_validation = _load_analysis_inputs(output)
    config = _config_record()
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
        "source_count": 61,
        "dependencies": _dependency_versions(),
        "parent": snapshot,
        "parent_replay": replay,
        "input_validation": input_validation,
        "config": config,
        "config_sha256": _canonical_json_sha256(config),
        "prepared_membership_sha256": _canonical_json_sha256([str(path) for path in PREPARED_FILES]),
        "expected_prepared_files": [str(path) for path in PREPARED_FILES],
        "expected_completed_files": [str(path) for path in COMPLETED_FILES],
    }


def _validate_prepared(output: Path) -> dict[str, object]:
    _require_membership(output, PREPARED_FILES)
    _assert_generated_modes(output)
    saved = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(saved, dict) or saved != _expected_input_manifest(output):
        raise ValueError("MM-006 input manifest no longer recomputes")
    return cast(dict[str, object], saved)


def _formal_start_record(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    parent = cast(Mapping[str, Any], manifest["parent"])
    validation = cast(Mapping[str, Any], manifest["input_validation"])
    pins = cast(Mapping[str, str], parent["pinned"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "input_manifest_sha256": _file_hash(output / INPUT_MANIFEST_FILE),
        "protocol_sha256": cast(Mapping[str, object], manifest["protocol"])["sha256"],
        "source_sha256": _canonical_json_sha256(manifest["source"]),
        "config_sha256": manifest["config_sha256"],
        "prepared_membership_sha256": manifest["prepared_membership_sha256"],
        "alignment_sha256": validation["alignment_sha256"],
        "mm005_receipt_sha256": {str(path): pins[str(path)] for path in MM005_SELECTED},
    }


def _mark_formal_started(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    record = _formal_start_record(output, manifest)
    _write_json_exclusive(output / STARTED_FILE, record, 0o444)
    return record


def _result_record(
    formal_start: Mapping[str, object],
    input_validation: Mapping[str, object],
    evidence: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "epistemic_role": "outcome-informed causal-warp mechanism diagnostic; not independent confirmation",
        "formal_start": dict(formal_start),
        "parent_classification": PARENT_CLASSIFICATION,
        "alignment_sha256": input_validation["alignment_sha256"],
        "evidence_sha256": _canonical_json_sha256(evidence),
        "summary": dict(summary),
    }


def _artifact_manifest(output: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": {str(path): _file_record(output / path) for path in ARTIFACT_FILES},
    }


def _validate_evidence_provenance(
    taesd: mm005_method.MatchedPanel,
    pixel: mm005_method.MatchedPanel,
    evidence: Mapping[str, object],
) -> None:
    if evidence.get("alignment") != mm005_method.alignment_record(taesd, pixel):
        raise ValueError("MM-006 evidence alignment differs from copied arrays")
    expected_normalizers: list[dict[str, Any]] = []
    for domain, panel in (("pixel", method.warp_panel(pixel)), ("taesd", method.warp_panel(taesd))):
        rows, _ = method._normalizer_rows(panel, domain)
        expected_normalizers.extend(rows)
    if evidence.get("normalizer_rows") != expected_normalizers:
        raise ValueError("MM-006 normalizer provenance differs from copied arrays")
    expected_panels: list[dict[str, Any]] = []
    template = method.warp_panel(pixel)
    for seed in method.SYNTHETIC_SEEDS:
        for channels in method.SYNTHETIC_CHANNELS:
            for scenario in method.SYNTHETIC_SCENARIOS:
                panel = method.synthetic_panel(template, seed, channels, scenario)
                expected_panels.append(
                    {
                        "panel_seed": seed,
                        "channels": channels,
                        "scenario": scenario,
                        "previous_sha256": method._array_sha256(panel.previous),
                        "current_sha256": method._array_sha256(panel.current),
                        "target_sha256": method._array_sha256(panel.target),
                    }
                )
    if evidence.get("synthetic_panel_rows") != expected_panels:
        raise ValueError("MM-006 synthetic panel provenance does not regenerate")


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


@_integrity_boundary
def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Copy frozen receipts and validate the exact panel without warp search."""

    _assert_expected_output(output)
    snapshot = _verify_live_parent()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("MM-006 output must be absent or empty before preparation")
    _mkdir_fsynced(output)
    for relative in sorted(
        _expected_directories(PREPARED_FILES) - {Path(".")}, key=lambda path: (len(path.parts), str(path))
    ):
        _mkdir_fsynced(output / relative)
    parent_root = REPO_ROOT / MM005_ROOT
    for relative in MM005_SELECTED:
        _copy_file_exclusive(parent_root / relative, output / MM005_COPY_ROOT / relative)
    _validate_parent_copy(output, snapshot)
    _copy_file_exclusive(REPO_ROOT / PROTOCOL_DOC, output / PROTOCOL_COPY_FILE, mode=0o644)
    _load_analysis_inputs(output)
    manifest = _expected_input_manifest(output)
    _write_json_exclusive(output / INPUT_MANIFEST_FILE, manifest, 0o644)
    result = verify(output)
    return {**result, "status": "prepared_only"}


@_integrity_boundary
def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Consume MM-006 exactly once and seal outcomes."""

    _assert_expected_output(output)
    manifest = _validate_prepared(output)
    _verify_live_parent()
    formal_start = _mark_formal_started(output, manifest)
    taesd, pixel, input_validation = _load_analysis_inputs(output)
    evidence = method.validate_evidence(method.execute(taesd, pixel))
    _validate_evidence_provenance(taesd, pixel, evidence)
    summary = method.summarize(evidence)
    result = _result_record(formal_start, input_validation, evidence, summary)
    _write_json_exclusive(output / EVIDENCE_FILE, evidence, 0o644)
    _write_json_exclusive(output / RESULT_FILE, result, 0o644)
    _write_bytes_exclusive(output / REPORT_FILE, method.report_text(summary).encode("utf-8"), 0o644)
    _require_membership(output, ARTIFACT_FILES)
    _assert_generated_modes(output)
    _write_json_exclusive(output / ARTIFACT_MANIFEST_FILE, _artifact_manifest(output), 0o644)
    _verify_live_parent()
    verify(output)
    return result


@_integrity_boundary
def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Fast structural, provenance, evidence, and decision verification."""

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
    _assert_generated_modes(output)
    if _read_json(output / ARTIFACT_MANIFEST_FILE) != _artifact_manifest(output):
        raise ValueError("MM-006 artifact manifest or artifact bytes/modes differ")
    manifest = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(manifest, dict) or manifest != _expected_input_manifest(output):
        raise ValueError("MM-006 input manifest no longer recomputes")
    formal_start = _read_json(output / STARTED_FILE)
    if formal_start != _formal_start_record(output, manifest):
        raise ValueError("MM-006 formal marker differs from frozen inputs")
    taesd, pixel, input_validation = _load_analysis_inputs(output)
    evidence = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    _validate_evidence_provenance(taesd, pixel, evidence)
    summary = method.summarize(evidence)
    expected_result = _result_record(formal_start, input_validation, evidence, summary)
    if _read_json(output / RESULT_FILE) != expected_result:
        raise ValueError("MM-006 result does not recompute from primitive evidence")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != method.report_text(summary):
        raise ValueError("MM-006 report is not canonical")
    _verify_live_parent()
    decision = cast(Mapping[str, object], summary["decision"])
    return {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": decision["classification"],
        "artifact_count": len(ARTIFACT_FILES),
    }


@_integrity_boundary
def verify_semantic(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Regenerate every MM-006 synthetic and real warp search."""

    verification = verify(output)
    if verification["outcomes"] != "verified_results":
        raise ValueError("semantic verification requires completed MM-006 outcomes")
    taesd, pixel, _ = _load_analysis_inputs(output)
    regenerated = method.validate_evidence(method.execute(taesd, pixel))
    saved = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    _assert_nested_close(saved, regenerated, path="evidence")
    saved_summary = method.summarize(saved)
    regenerated_summary = method.summarize(regenerated)
    _assert_nested_close(saved_summary, regenerated_summary, path="summary")
    return {
        **verification,
        "outcomes": "verified_semantic_results",
        "semantic_regeneration": "all synthetic and real causal/oracle searches reproduced from copied arrays",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-006 causal-warp ceiling diagnostic")
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
    "InvalidMM006Package",
    "InvalidMM006ParentAlignment",
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
