"""Sealed preparation, execution, and verification lifecycle for MM-007."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
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

from bench.multimodal_horizon_diagnostics import method as mm005_method
from bench.multimodal_preflight import backends as mm001_backends
from bench.multimodal_preflight import dataset as mm001_dataset
from bench.multimodal_spatial_diagnostics import experiment as mm004_experiment
from bench.multimodal_warp_diagnostics import experiment as mm006_experiment
from bench.multimodal_warp_diagnostics import method as mm006_method

from . import method

SCHEMA_VERSION = "mm007-formal-v1"
EXPERIMENT_ID = "MM-007"
PARENT_CLASSIFICATION = "tested_pixel_warp_ceiling_failure_supported"
MM004_CLASSIFICATION = "tested_local_objective_or_horizon_failure_supported"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/multimodal_resolution_diagnostics/results/MM-007")
EXPECTED_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT
PROTOCOL_DOC = Path("docs/research/2026-07-15-mm007-physically-matched-resolution-protocol.md")

PROTOCOL_COPY_FILE = Path("MM-007-protocol.md")
INPUT_MANIFEST_FILE = Path("input-manifest.json")
FRAME_FILE = Path("MM-007-frames-64x64.npz")
STARTED_FILE = Path("formal-start.json")
EVIDENCE_FILE = Path("MM-007-evidence.json")
RESULT_FILE = Path("MM-007-results.json")
REPORT_FILE = Path("MM-007-report.md")
ARTIFACT_MANIFEST_FILE = Path("artifact-manifest.json")

MM006_ROOT = mm006_experiment.DEFAULT_OUTPUT
MM006_COPY_ROOT = Path("inputs/MM-006")
MM006_PIXEL_RELATIVE = Path("inputs/MM-005/inputs/MM-004/MM-004-pixel-grids.npz")
MM006_SELECTED = (
    Path("artifact-manifest.json"),
    Path("input-manifest.json"),
    Path("MM-006-evidence.json"),
    Path("MM-006-results.json"),
    MM006_PIXEL_RELATIVE,
)
MM006_PINS: dict[Path, str] = {
    Path("artifact-manifest.json"): "9727eefc6c5665b5eb8cc65ae9cfab57bb4c8b3e353b747363bec5e3c2f573b0",
    Path("input-manifest.json"): "badd7676f1e4a60c56b59af12a1d7f82ef134e797febc86a5adcb4c33dda5cd1",
    Path("MM-006-evidence.json"): "5c5ffa514ab0f0c06c8588e69b54d6c4f2f6be3a4471a0fe7a31aa1e1dd3dac2",
    Path("MM-006-results.json"): "c5e0737acf6030315a77b497f5d5ea78693eb8a5879399e37bdfb702e2b9f648",
    MM006_PIXEL_RELATIVE: "cca261a941e68a7ddc510eee3a3af958d33b6abaf958cb5562ed6b66c22f47c8",
}

MM004_ROOT = mm004_experiment.DEFAULT_OUTPUT
MM004_COPY_ROOT = Path("inputs/MM-004")
MM004_SELECTED = (Path("input-manifest.json"),)
MM004_PINS = {
    Path("input-manifest.json"): "597a8bfc9f6ae1f6ff1f0d3be456f57d768f1866a1fac59cd981dd260076dc90"
}
FFMPEG_SHA256 = "ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41"
FFMPEG_SIZE_BYTES = 342_488
MATCHED_IDENTITY_SHA256 = "d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b"
VIDEO_IDS_SHA256 = "06e75502f8c9ab7883ba6a44d9e0f250bd5f678ac8b5989b2b7b5349b69e4c50"
TIMESTAMPS_SHA256 = "128c725db3361bf55c89017c02a4bd08f54622f09018d10c4c83b4467c4d3d55"
FRAMES_UINT8_SHA256 = "46d21d8c5b7d3a88abd96500ab07c3d54606a8f74b1500ddedeefb45e2d13eb9"
FRAME_ARRAY_SHA256 = {
    "video_ids": VIDEO_IDS_SHA256,
    "timestamps": TIMESTAMPS_SHA256,
    "frames_uint8": FRAMES_UINT8_SHA256,
}
FRAME_PACKAGE_SHA256 = "fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f"
EXPECTED_FRAME_COUNTS = {
    "video_10993": 67,
    "video_1580": 68,
    "video_2564": 64,
    "video_3501": 70,
    "video_6860": 70,
    "video_8241": 52,
    "video_874": 70,
    "video_9253": 51,
}

FRAME_KEYS = {"video_ids", "timestamps", "frames_uint8"}
FRAME_SCHEMA: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "video_ids": (np.dtype("<U11"), (477,)),
    "timestamps": (np.dtype("<f8"), (477,)),
    "frames_uint8": (np.dtype("|u1"), (477, 64, 64, 3)),
}

PREPARED_ROOT_FILES = (PROTOCOL_COPY_FILE, INPUT_MANIFEST_FILE, FRAME_FILE)
MM006_COPY_FILES = tuple(MM006_COPY_ROOT / path for path in MM006_SELECTED)
MM004_COPY_FILES = tuple(MM004_COPY_ROOT / path for path in MM004_SELECTED)
PARENT_COPY_FILES = (*MM006_COPY_FILES, *MM004_COPY_FILES)
PREPARED_FILES = (*PREPARED_ROOT_FILES, *PARENT_COPY_FILES)
OUTCOME_FILES = (STARTED_FILE, EVIDENCE_FILE, RESULT_FILE, REPORT_FILE)
ARTIFACT_FILES = (*PREPARED_FILES, *OUTCOME_FILES)
COMPLETED_FILES = (*ARTIFACT_FILES, ARTIFACT_MANIFEST_FILE)
GENERATED_0644_FILES = (
    PROTOCOL_COPY_FILE,
    INPUT_MANIFEST_FILE,
    FRAME_FILE,
    EVIDENCE_FILE,
    RESULT_FILE,
    REPORT_FILE,
    ARTIFACT_MANIFEST_FILE,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InvalidMM007Package(ValueError):
    """Stable fail-closed classification for MM-007 package defects."""

    classification = "invalid_MM007_package"


class InvalidMM007ParentParity(ValueError):
    """Stable pre-marker classification for receipt or frame-parity defects."""

    classification = "invalid_MM007_parent_parity"


def _integrity_boundary(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return function(*args, **kwargs)
        except (InvalidMM007Package, InvalidMM007ParentParity):
            raise
        except Exception as error:
            raise InvalidMM007Package(str(error)) from error

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


def _validate_frames_uint8_digest(value: np.ndarray) -> str:
    frames = np.asarray(value)
    expected_dtype, expected_shape = FRAME_SCHEMA["frames_uint8"]
    if frames.dtype != expected_dtype or frames.shape != expected_shape:
        raise ValueError("prepared frames_uint8 schema differs from the authenticated decode")
    observed = _array_sha256(frames)
    expected = FRAME_ARRAY_SHA256["frames_uint8"]
    if observed != expected:
        raise ValueError(
            "prepared frames_uint8 digest differs from the authenticated decode: "
            f"expected {expected}, observed {observed}"
        )
    return observed


def _validate_frame_array_digests(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    if set(arrays) != FRAME_KEYS:
        raise ValueError("prepared frame array membership differs from the authenticated decode")
    observed = {
        name: _array_sha256(np.asarray(arrays[name]))
        for name in sorted(FRAME_KEYS)
    }
    _validate_frames_uint8_digest(arrays["frames_uint8"])
    for name in ("video_ids", "timestamps"):
        if observed[name] != FRAME_ARRAY_SHA256[name]:
            raise ValueError(
                f"prepared {name} digest differs from the authenticated decode: "
                f"expected {FRAME_ARRAY_SHA256[name]}, observed {observed[name]}"
            )
    return observed


def _validate_frame_package_digest(path: Path) -> str:
    observed = _file_hash(path)
    if observed != FRAME_PACKAGE_SHA256:
        raise ValueError(
            "prepared frame package digest differs from the canonical NPZ: "
            f"expected {FRAME_PACKAGE_SHA256}, observed {observed}"
        )
    return observed


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
    package_root = REPO_ROOT / "bench/multimodal_resolution_diagnostics"
    expected_package = {
        Path("bench/multimodal_resolution_diagnostics/__init__.py"),
        Path("bench/multimodal_resolution_diagnostics/__main__.py"),
        Path("bench/multimodal_resolution_diagnostics/experiment.py"),
        Path("bench/multimodal_resolution_diagnostics/method.py"),
    }
    actual_package = {path.relative_to(REPO_ROOT) for path in package_root.glob("*.py")}
    if actual_package != expected_package:
        raise ValueError("MM-007 package source membership differs from the frozen four-file set")
    own = {
        PROTOCOL_DOC,
        Path("tests/test_mm007_method.py"),
        Path("tests/test_mm007_experiment.py"),
        *actual_package,
    }
    paths = tuple(sorted({*mm006_experiment._source_paths(), *own}, key=str))
    if len(paths) != 68:
        raise ValueError(f"MM-007 source membership must contain exactly 68 files, got {len(paths)}")
    return paths


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _source_paths():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bound MM-007 source is missing or a symlink: {relative}")
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
        raise ValueError(f"MM-007 output must be canonical path {DEFAULT_OUTPUT}")
    current = output
    while True:
        if current.is_symlink():
            raise ValueError(f"MM-007 output path contains a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent
    if output.exists() and not output.is_dir():
        raise ValueError("MM-007 output must be a real directory")
    resolved = output.resolve()
    for parent_root in (REPO_ROOT / MM006_ROOT, REPO_ROOT / MM004_ROOT):
        parent_resolved = parent_root.resolve()
        if resolved == parent_resolved or resolved in parent_resolved.parents or parent_resolved in resolved.parents:
            raise ValueError("MM-007 output overlaps a protected parent")


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
            "MM-007 membership mismatch: "
            f"missing={sorted(str(path) for path in expected_files-files)}, "
            f"extra={sorted(str(path) for path in files-expected_files)}, "
            f"missing_dirs={sorted(str(path) for path in expected_directories-directories)}, "
            f"extra_dirs={sorted(str(path) for path in directories-expected_directories)}"
        )


def _assert_generated_modes(output: Path) -> None:
    for relative in GENERATED_0644_FILES:
        path = output / relative
        if path.exists() and stat.S_IMODE(path.stat().st_mode) != 0o644:
            raise ValueError(f"MM-007 generated file mode differs from 0644: {relative}")
    marker = output / STARTED_FILE
    if marker.exists() and stat.S_IMODE(marker.stat().st_mode) != 0o444:
        raise ValueError("MM-007 formal marker is not mode 0444")


def _records(root: Path, files: Sequence[Path]) -> dict[str, dict[str, object]]:
    return {str(relative): _file_record(root / relative) for relative in files}


def _verify_selected_against_manifest(root: Path, selected: Sequence[Path]) -> None:
    manifest = _read_json(root / ARTIFACT_MANIFEST_FILE)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("artifacts"), Mapping):
        raise ValueError("MM-006 artifact manifest is invalid")
    artifacts = cast(Mapping[str, object], manifest["artifacts"])
    for relative in selected:
        if relative == ARTIFACT_MANIFEST_FILE:
            continue
        expected = artifacts.get(str(relative))
        if not isinstance(expected, Mapping) or _file_record(root / relative) != dict(expected):
            raise ValueError(f"selected MM-006 file differs from its manifest: {relative}")


def _replay_parent_unchecked(root: Path) -> dict[str, object]:
    evidence = mm006_method.validate_evidence(_read_json(root / mm006_experiment.EVIDENCE_FILE))
    summary = mm006_method.summarize(evidence)
    result = _read_json(root / mm006_experiment.RESULT_FILE)
    if not isinstance(result, Mapping) or result.get("summary") != summary:
        raise ValueError("MM-006 evidence/result summary relationship does not replay")
    if result.get("evidence_sha256") != _canonical_json_sha256(evidence):
        raise ValueError("MM-006 result does not bind canonical evidence")
    controls = summary.get("synthetic_control")
    decision = summary.get("decision")
    if (
        not isinstance(controls, Mapping)
        or controls.get("positive_passes") is not True
        or controls.get("negative_passes") is not True
    ):
        raise ValueError("MM-006 synthetic controls did not both pass")
    if not isinstance(decision, Mapping) or decision.get("classification") != PARENT_CLASSIFICATION:
        raise ValueError("MM-006 decision differs from the frozen parent branch")
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
    except InvalidMM007ParentParity:
        raise
    except Exception as error:
        raise InvalidMM007ParentParity(str(error)) from error


def _verify_live_inputs_unchecked() -> dict[str, object]:
    mm006_verification = mm006_experiment.verify(MM006_ROOT)
    if (
        mm006_verification.get("outcomes") != "verified_results"
        or mm006_verification.get("classification") != PARENT_CLASSIFICATION
    ):
        raise ValueError("MM-007 requires completed verified MM-006 results")
    mm004_verification = mm004_experiment.verify(MM004_ROOT)
    if (
        mm004_verification.get("outcomes") != "verified_results"
        or mm004_verification.get("classification") != MM004_CLASSIFICATION
    ):
        raise ValueError("MM-007 requires the verified MM-004 media-lineage package")
    mm006_root = REPO_ROOT / MM006_ROOT
    mm004_root = REPO_ROOT / MM004_ROOT
    for relative, expected in MM006_PINS.items():
        if _file_hash(mm006_root / relative) != expected:
            raise ValueError(f"pinned MM-006 hash differs: {relative}")
    for relative, expected in MM004_PINS.items():
        if _file_hash(mm004_root / relative) != expected:
            raise ValueError(f"pinned MM-004 hash differs: {relative}")
    _verify_selected_against_manifest(mm006_root, MM006_SELECTED)
    mm006_manifest = _read_json(mm006_root / INPUT_MANIFEST_FILE)
    lineage = cast(Mapping[str, Any], cast(Mapping[str, Any], mm006_manifest)["parent"])["files"]
    expected_mm004 = cast(Mapping[str, object], lineage["inputs/MM-004/input-manifest.json"])
    if _file_record(mm004_root / INPUT_MANIFEST_FILE) != dict(expected_mm004):
        raise ValueError("MM-004 media manifest differs from MM-006 authenticated ancestry")
    replay = _replay_parent(mm006_root)
    return {
        "parent": {
            "experiment_id": "MM-006",
            "classification": PARENT_CLASSIFICATION,
            "verification": mm006_verification,
            "live_path": str(MM006_ROOT),
            "copy_path": str(MM006_COPY_ROOT),
            "files": _records(mm006_root, mm006_experiment.COMPLETED_FILES),
            "selected_files": [str(path) for path in MM006_SELECTED],
            "pinned": {str(path): digest for path, digest in MM006_PINS.items()},
            "replay": replay,
            "scientific_relationship": (
                "outcome-informed direct child reusing the same eight videos; not independent evidence"
            ),
        },
        "media_lineage": {
            "experiment_id": "MM-004",
            "classification": MM004_CLASSIFICATION,
            "verification": mm004_verification,
            "live_path": str(MM004_ROOT),
            "copy_path": str(MM004_COPY_ROOT),
            "files": _records(mm004_root, mm004_experiment.COMPLETED_FILES),
            "selected_files": [str(path) for path in MM004_SELECTED],
            "pinned": {str(path): digest for path, digest in MM004_PINS.items()},
            "authenticated_by_mm006_input_manifest": dict(expected_mm004),
        },
    }


def _verify_live_inputs() -> dict[str, object]:
    try:
        return _verify_live_inputs_unchecked()
    except InvalidMM007ParentParity:
        raise
    except Exception as error:
        raise InvalidMM007ParentParity(str(error)) from error


def _validate_receipts_unchecked(output: Path, snapshot: Mapping[str, object]) -> dict[str, object]:
    parent = cast(Mapping[str, Any], snapshot["parent"])
    lineage = cast(Mapping[str, Any], snapshot["media_lineage"])
    mm006_copy = output / MM006_COPY_ROOT
    files, directories = _tree_members(mm006_copy)
    if files != set(MM006_SELECTED) or directories != _expected_directories(MM006_SELECTED):
        raise ValueError("copied MM-006 receipt has unexpected membership")
    for relative in MM006_SELECTED:
        expected = cast(Mapping[str, object], parent["files"][str(relative)])
        if _file_record(mm006_copy / relative) != dict(expected):
            raise ValueError(f"copied MM-006 file differs from live parent: {relative}")
    _verify_selected_against_manifest(mm006_copy, MM006_SELECTED)
    replay = _replay_parent(mm006_copy)
    if replay != parent["replay"]:
        raise ValueError("copied MM-006 replay differs from live parent")

    mm004_copy = output / MM004_COPY_ROOT
    files, directories = _tree_members(mm004_copy)
    if files != set(MM004_SELECTED) or directories != _expected_directories(MM004_SELECTED):
        raise ValueError("copied MM-004 media receipt has unexpected membership")
    copied_manifest = mm004_copy / INPUT_MANIFEST_FILE
    expected = cast(Mapping[str, object], lineage["files"][str(INPUT_MANIFEST_FILE)])
    if _file_record(copied_manifest) != dict(expected):
        raise ValueError("copied MM-004 media manifest differs from live lineage")
    mm006_input = cast(Mapping[str, Any], _read_json(mm006_copy / INPUT_MANIFEST_FILE))
    ancestry = cast(Mapping[str, Any], mm006_input["parent"])["files"]
    if _file_record(copied_manifest) != dict(ancestry["inputs/MM-004/input-manifest.json"]):
        raise ValueError("copied MM-004 media manifest is not authenticated by copied MM-006 ancestry")
    return replay


def _validate_receipts(output: Path, snapshot: Mapping[str, object]) -> dict[str, object]:
    try:
        return _validate_receipts_unchecked(output, snapshot)
    except InvalidMM007ParentParity:
        raise
    except Exception as error:
        raise InvalidMM007ParentParity(str(error)) from error


def _load_npz(path: Path, expected: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]]) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"array package is missing or a symlink: {path}")
    with np.load(path, allow_pickle=False) as package:
        names = list(package.files)
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError(f"array package keys differ: {path}")
        arrays = {name: np.asarray(package[name]).copy() for name in names}
    for name, (dtype, shape) in expected.items():
        array = arrays[name]
        if array.dtype != dtype or array.shape != shape or array.dtype.hasobject:
            raise ValueError(f"array schema differs for {path}:{name}")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"array contains non-finite values: {path}:{name}")
    return arrays


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    cast(Any, np.savez_compressed)(buffer, **{name: np.asarray(value) for name, value in sorted(arrays.items())})
    return buffer.getvalue()


def _strict_pixel_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays = mm004_experiment._load_npz(path, mm004_experiment.PIXEL_SCHEMA)
    for name in ("pixel_current", "pixel_target"):
        if float(np.min(arrays[name])) < 0.0 or float(np.max(arrays[name])) > 1.0:
            raise ValueError(f"copied {name} must remain in [0,1]")
    return arrays


def _media_contract(mm004_manifest: object) -> dict[str, object]:
    if not isinstance(mm004_manifest, Mapping):
        raise ValueError("copied MM-004 input manifest is invalid")
    validation = mm004_manifest.get("input_validation")
    if not isinstance(validation, Mapping):
        raise ValueError("copied MM-004 manifest lacks input validation")
    preparation = validation.get("pixel_preparation")
    if not isinstance(preparation, Mapping) or not isinstance(preparation.get("media"), Mapping):
        raise ValueError("copied MM-004 manifest lacks media preparation")
    media = cast(Mapping[str, object], preparation["media"])
    cache_path = media.get("cache_path")
    video_ids = media.get("video_ids")
    hashes = media.get("media_sha256")
    sizes = media.get("media_size_bytes")
    frame_counts = media.get("frame_count_2fps")
    ffmpeg = media.get("ffmpeg")
    decode = media.get("decode")
    if (
        not isinstance(cache_path, str)
        or video_ids != list(mm001_dataset.SAMPLE_VIDEO_IDS)
        or not isinstance(hashes, Mapping)
        or dict(hashes) != dict(mm001_dataset.EXTRACTED_MP4_SHA256)
        or not isinstance(sizes, Mapping)
        or dict(sizes) != dict(mm001_dataset.EXTRACTED_MP4_SIZE_BYTES)
        or not isinstance(frame_counts, Mapping)
        or dict(frame_counts) != EXPECTED_FRAME_COUNTS
        or not isinstance(ffmpeg, Mapping)
        or ffmpeg.get("sha256") != FFMPEG_SHA256
        or ffmpeg.get("size_bytes") != FFMPEG_SIZE_BYTES
        or not isinstance(ffmpeg.get("path"), str)
        or not isinstance(decode, Mapping)
        or decode.get("frame_rate") != 2
        or decode.get("frame_size") != 64
        or decode.get("letterboxed") is not True
    ):
        raise ValueError("copied MM-004 media contract differs from the frozen source")
    return {
        "cache_path": cache_path,
        "video_ids": list(mm001_dataset.SAMPLE_VIDEO_IDS),
        "media_sha256": dict(sorted(cast(Mapping[str, str], hashes).items())),
        "media_size_bytes": dict(sorted(cast(Mapping[str, int], sizes).items())),
        "frame_count_2fps": dict(sorted(cast(Mapping[str, int], frame_counts).items())),
        "ffmpeg": {
            "path": ffmpeg["path"],
            "sha256": ffmpeg["sha256"],
            "size_bytes": ffmpeg["size_bytes"],
        },
        "decode": {
            "frame_rate": 2,
            "frame_size": 64,
            "filter": (
                "fps=2,scale=64:64:force_original_aspect_ratio=decrease:flags=bicubic,"
                "pad=64:64:(ow-iw)/2:(oh-ih)/2:color=black"
            ),
            "pixel_format": "rgb24",
            "stored_dtype": "uint8",
            "stored_shape": [477, 64, 64, 3],
        },
    }


def _authenticate_media(contract: Mapping[str, object]) -> tuple[Path, Path]:
    cache_path = Path(cast(str, contract["cache_path"]))
    hashes = cast(Mapping[str, str], contract["media_sha256"])
    sizes = cast(Mapping[str, int], contract["media_size_bytes"])
    observed = mm001_dataset.validate_media_hashes(cache_path, expected_hashes=hashes, expected_sizes=sizes)
    if observed != dict(hashes):
        raise ValueError("authenticated media hashes differ from the copied MM-004 manifest")
    ffmpeg_record = cast(Mapping[str, object], contract["ffmpeg"])
    ffmpeg = Path(cast(str, ffmpeg_record["path"]))
    if (
        ffmpeg.is_symlink()
        or not ffmpeg.is_file()
        or ffmpeg.stat().st_size != ffmpeg_record["size_bytes"]
        or _file_hash(ffmpeg) != ffmpeg_record["sha256"]
    ):
        raise ValueError("sealed FFmpeg executable identity differs")
    return cache_path, ffmpeg


def _pool_8x8(frames_uint8: np.ndarray) -> np.ndarray:
    values = np.asarray(frames_uint8)
    if values.dtype != np.dtype("|u1") or values.ndim != 4 or values.shape[1:] != (64, 64, 3):
        raise ValueError("64x64 source frames must be uint8 [N,64,64,3]")
    unit = values.astype(np.float32) / np.float32(255.0)
    blocked = unit.reshape(len(unit), 8, 8, 8, 8, 3)
    pooled = np.mean(blocked, axis=(2, 4), dtype=np.float64).astype(np.float32)
    return np.asarray(np.transpose(pooled, (0, 3, 1, 2)), dtype=np.float32)


def _decode_frame_arrays(pixel_arrays: Mapping[str, np.ndarray], mm004_manifest: object) -> dict[str, np.ndarray]:
    """Authenticate and decode media; legal only in prepare and semantic verify."""

    contract = _media_contract(mm004_manifest)
    cache_path, ffmpeg = _authenticate_media(contract)
    video_ids = np.asarray(pixel_arrays["video_ids"], dtype="<U11")
    timestamps = np.asarray(pixel_arrays["timestamps"], dtype="<f8")
    frames_uint8 = np.empty((477, 64, 64, 3), dtype=np.uint8)
    decoded_targets = np.empty((477, 3, 8, 8), dtype=np.float32)
    frame_counts = cast(Mapping[str, int], contract["frame_count_2fps"])
    for video_id in mm001_dataset.SAMPLE_VIDEO_IDS:
        decoded = mm001_backends.decode_video_frames(
            cache_path / "videos" / f"{video_id}.mp4", ffmpeg=str(ffmpeg)
        )
        if decoded.dtype != np.dtype("<f4") or decoded.shape != (frame_counts[video_id], 64, 64, 3):
            raise ValueError(f"decoded frame identity differs for {video_id}")
        if not np.all(np.isfinite(decoded)) or float(np.min(decoded)) < 0.0 or float(np.max(decoded)) > 1.0:
            raise ValueError(f"decoded frames are invalid for {video_id}")
        raw = np.rint(decoded * np.float32(255.0)).astype(np.uint8)
        restored = raw.astype(np.float32) / np.float32(255.0)
        if not np.array_equal(decoded, restored):
            raise ValueError(f"decoded frames are not losslessly representable as uint8 for {video_id}")
        rows = np.flatnonzero(video_ids == video_id)
        if len(rows) != mm001_dataset.EXPECTED_WINDOW_COUNTS[video_id]:
            raise ValueError(f"frame identity count differs for {video_id}")
        for row in rows:
            timestamp = float(timestamps[row])
            current_index = mm001_backends.frame_index_at(timestamp, len(raw))
            target_index = mm001_backends.frame_index_at(
                timestamp + mm001_dataset.VISUAL_TARGET_HORIZON_SECONDS, len(raw)
            )
            frames_uint8[row] = raw[current_index]
            decoded_targets[row] = _pool_8x8(raw[target_index : target_index + 1])[0]
    arrays = {"video_ids": video_ids, "timestamps": timestamps, "frames_uint8": frames_uint8}
    if not np.array_equal(_pool_8x8(frames_uint8), pixel_arrays["pixel_current"]):
        raise ValueError("decoded 64x64 frames do not exactly replay all 477 MM-004 current grids")
    if not np.array_equal(decoded_targets, pixel_arrays["pixel_target"]):
        raise ValueError("decoded 64x64 targets do not exactly replay all 477 MM-004 target grids")
    _validate_frame_array_digests(arrays)
    return arrays


def _frame_table(
    arrays: Mapping[str, np.ndarray], expected_pixel_current_8: np.ndarray
) -> Any:
    return method.raw_frame_table(
        np.asarray(arrays["video_ids"], dtype="<U11"),
        np.asarray(arrays["timestamps"], dtype="<f8"),
        np.asarray(arrays["frames_uint8"], dtype=np.uint8),
        expected_pixel_current_8=np.asarray(expected_pixel_current_8, dtype=np.float32),
    )


def _load_analysis_inputs_unchecked(output: Path) -> tuple[Any, dict[str, object], dict[str, object]]:
    frames = _load_npz(output / FRAME_FILE, FRAME_SCHEMA)
    frame_array_sha256 = _validate_frame_array_digests(frames)
    frame_package_sha256 = _validate_frame_package_digest(output / FRAME_FILE)
    pixels = _strict_pixel_arrays(output / MM006_COPY_ROOT / MM006_PIXEL_RELATIVE)
    if not np.array_equal(frames["video_ids"], pixels["video_ids"]):
        raise ValueError("prepared frame video identities differ from copied MM-004 pixels")
    if not np.array_equal(frames["timestamps"], pixels["timestamps"]):
        raise ValueError("prepared frame timestamps differ from copied MM-004 pixels")
    pooled = _pool_8x8(frames["frames_uint8"])
    if not np.array_equal(pooled, pixels["pixel_current"]):
        raise ValueError("prepared 64-to-8 replay differs from all 477 copied current grids")
    raw = mm005_method.raw_grid_table(
        frames["video_ids"], frames["timestamps"], pooled, pixels["pixel_target"], expected_channels=3
    )
    panel_record = mm005_method.panel_provenance(raw)
    if panel_record["rows"] != 453 or panel_record["identity_sha256"] != MATCHED_IDENTITY_SHA256:
        raise ValueError("prepared frames differ from the frozen 453-row identity")
    parent_root = output / MM006_COPY_ROOT
    parent_evidence = mm006_method.validate_evidence(_read_json(parent_root / mm006_experiment.EVIDENCE_FILE))
    parent_alignment = cast(Mapping[str, Any], parent_evidence["alignment"])
    parent_pixel = cast(Mapping[str, object], cast(Mapping[str, Any], parent_alignment["domains"])["pixel"])
    for key in ("previous_sha256", "current_sha256", "target_0p5_sha256", "target_1p0_sha256"):
        if panel_record[key] != parent_pixel[key]:
            raise ValueError(f"prepared low-resolution matched-panel parity differs: {key}")
    table = _frame_table(frames, pixels["pixel_current"])
    mm004_manifest = _read_json(output / MM004_COPY_ROOT / INPUT_MANIFEST_FILE)
    contract = _media_contract(mm004_manifest)
    validation: dict[str, object] = {
        "frame_file": _file_record(output / FRAME_FILE),
        "frame_array_sha256": frame_array_sha256,
        "frame_package_sha256": frame_package_sha256,
        "frames_uint8_sha256": frame_array_sha256["frames_uint8"],
        "frame_schema": {
            name: {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _array_sha256(value)}
            for name, value in sorted(frames.items())
        },
        "media_contract": contract,
        "media_contract_sha256": _canonical_json_sha256(contract),
        "low_resolution_parity": {
            "current_rows": 477,
            "current_exact": True,
            "derived_current_sha256": _array_sha256(pooled),
            "copied_current_sha256": _array_sha256(pixels["pixel_current"]),
            "copied_target_sha256": _array_sha256(pixels["pixel_target"]),
            "preparation_and_semantic_target_rows": 477,
            "matched_panel": panel_record,
        },
        "parent_alignment_sha256": _canonical_json_sha256(parent_alignment),
    }
    return table, parent_evidence, validation


def _load_analysis_inputs(output: Path) -> tuple[Any, dict[str, object], dict[str, object]]:
    try:
        return _load_analysis_inputs_unchecked(output)
    except InvalidMM007ParentParity:
        raise
    except Exception as error:
        raise InvalidMM007ParentParity(str(error)) from error


def _config_record() -> dict[str, object]:
    return {
        "method": method.frozen_config(),
        "lifecycle": {
            "prepared_files": [str(path) for path in PREPARED_FILES],
            "completed_files": [str(path) for path in COMPLETED_FILES],
            "prepared_file_count": 9,
            "artifact_file_count": 13,
            "completed_file_count": 14,
            "generated_file_mode": "0644",
            "formal_marker_mode": "0444",
            "fast_verification": "regenerate all scientific evidence from copied NPZ",
            "semantic_verification": "reauthenticate and redecode media, then bit-compare stored uint8 frames",
            "media_used": "preparation_and_semantic_verification_only",
            "frame_array_sha256": dict(FRAME_ARRAY_SHA256),
            "frame_package_sha256": FRAME_PACKAGE_SHA256,
            "model_inference_used": False,
            "oracle_diagnostic_only": True,
        },
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    snapshot = _verify_live_inputs()
    replay = _validate_receipts(output, snapshot)
    source_protocol = REPO_ROOT / PROTOCOL_DOC
    protocol_copy = output / PROTOCOL_COPY_FILE
    if (
        protocol_copy.read_bytes() != source_protocol.read_bytes()
        or stat.S_IMODE(protocol_copy.stat().st_mode) != 0o644
    ):
        raise ValueError("MM-007 protocol copy differs from bound source or mode")
    _, _, validation = _load_analysis_inputs(output)
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
        "source_count": 68,
        "dependencies": _dependency_versions(),
        "parent": snapshot["parent"],
        "media_lineage": snapshot["media_lineage"],
        "parent_replay": replay,
        "input_validation": validation,
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
        raise ValueError("MM-007 input manifest no longer recomputes")
    return cast(dict[str, object], saved)


def _formal_start_record(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    validation = cast(Mapping[str, Any], manifest["input_validation"])
    parent = cast(Mapping[str, Any], manifest["parent"])
    lineage = cast(Mapping[str, Any], manifest["media_lineage"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "input_manifest_sha256": _file_hash(output / INPUT_MANIFEST_FILE),
        "protocol_sha256": cast(Mapping[str, object], manifest["protocol"])["sha256"],
        "source_sha256": _canonical_json_sha256(manifest["source"]),
        "config_sha256": manifest["config_sha256"],
        "prepared_membership_sha256": manifest["prepared_membership_sha256"],
        "frame_file_sha256": cast(Mapping[str, object], validation["frame_file"])["sha256"],
        "frame_array_sha256": dict(cast(Mapping[str, str], validation["frame_array_sha256"])),
        "frame_package_sha256": validation["frame_package_sha256"],
        "frames_uint8_sha256": validation["frames_uint8_sha256"],
        "frame_schema_sha256": _canonical_json_sha256(validation["frame_schema"]),
        "media_contract_sha256": validation["media_contract_sha256"],
        "parent_alignment_sha256": validation["parent_alignment_sha256"],
        "mm006_receipt_sha256": dict(cast(Mapping[str, str], parent["pinned"])),
        "mm004_receipt_sha256": dict(cast(Mapping[str, str], lineage["pinned"])),
    }


def _mark_formal_started(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    record = _formal_start_record(output, manifest)
    _write_json_exclusive(output / STARTED_FILE, record, 0o444)
    return record


def _result_record(
    formal_start: Mapping[str, object],
    validation: Mapping[str, object],
    evidence: Mapping[str, object],
    summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "epistemic_role": "outcome-informed physically matched resolution diagnostic; not independent confirmation",
        "formal_start": dict(formal_start),
        "parent_classification": PARENT_CLASSIFICATION,
        "frame_file_sha256": cast(Mapping[str, object], validation["frame_file"])["sha256"],
        "frame_array_sha256": dict(cast(Mapping[str, str], validation["frame_array_sha256"])),
        "frame_package_sha256": validation["frame_package_sha256"],
        "frames_uint8_sha256": validation["frames_uint8_sha256"],
        "parent_alignment_sha256": validation["parent_alignment_sha256"],
        "evidence_sha256": _canonical_json_sha256(evidence),
        "summary": dict(summary),
    }


def _artifact_manifest(output: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frame_array_sha256": dict(FRAME_ARRAY_SHA256),
        "frame_package_sha256": FRAME_PACKAGE_SHA256,
        "artifacts": {str(path): _file_record(output / path) for path in ARTIFACT_FILES},
    }


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
    """Copy receipts and decode authenticated 64x64 frames without scientific search."""

    _assert_expected_output(output)
    snapshot = _verify_live_inputs()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("MM-007 output must be absent or empty before preparation")
    _mkdir_fsynced(output)
    for relative in sorted(
        _expected_directories(PREPARED_FILES) - {Path(".")},
        key=lambda path: (len(path.parts), str(path)),
    ):
        _mkdir_fsynced(output / relative)
    mm006_root = REPO_ROOT / MM006_ROOT
    for relative in MM006_SELECTED:
        _copy_file_exclusive(mm006_root / relative, output / MM006_COPY_ROOT / relative)
    mm004_root = REPO_ROOT / MM004_ROOT
    for relative in MM004_SELECTED:
        _copy_file_exclusive(mm004_root / relative, output / MM004_COPY_ROOT / relative)
    _validate_receipts(output, snapshot)
    _copy_file_exclusive(REPO_ROOT / PROTOCOL_DOC, output / PROTOCOL_COPY_FILE, mode=0o644)
    pixels = _strict_pixel_arrays(output / MM006_COPY_ROOT / MM006_PIXEL_RELATIVE)
    mm004_manifest = _read_json(output / MM004_COPY_ROOT / INPUT_MANIFEST_FILE)
    frame_arrays = _decode_frame_arrays(pixels, mm004_manifest)
    if set(frame_arrays) != FRAME_KEYS:
        raise ValueError("prepared frame arrays have unexpected keys")
    _validate_frame_array_digests(frame_arrays)
    _write_bytes_exclusive(output / FRAME_FILE, _npz_bytes(frame_arrays), 0o644)
    stored_frames = _load_npz(output / FRAME_FILE, FRAME_SCHEMA)
    _validate_frame_array_digests(stored_frames)
    _validate_frame_package_digest(output / FRAME_FILE)
    manifest = _expected_input_manifest(output)
    _write_json_exclusive(output / INPUT_MANIFEST_FILE, manifest, 0o644)
    result = verify(output)
    return {**result, "status": "prepared_only"}


@_integrity_boundary
def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Consume the prepared package exactly once and seal MM-007 outcomes."""

    _assert_expected_output(output)
    manifest = _validate_prepared(output)
    _verify_live_inputs()
    table, parent_evidence, validation = _load_analysis_inputs(output)
    if validation.get("frame_array_sha256") != FRAME_ARRAY_SHA256:
        raise ValueError("pre-marker frames_uint8 digest differs from the authenticated decode")
    if validation.get("frame_package_sha256") != FRAME_PACKAGE_SHA256:
        raise ValueError("pre-marker frame package digest differs from the canonical NPZ")
    formal_start = _mark_formal_started(output, manifest)
    # The marker write is deliberately the final operation before the first scientific call.
    evidence = method.validate_evidence(method.execute(table, parent_evidence))
    summary = method.summarize(evidence)
    result = _result_record(formal_start, validation, evidence, summary)
    _write_json_exclusive(output / EVIDENCE_FILE, evidence, 0o644)
    _write_json_exclusive(output / RESULT_FILE, result, 0o644)
    _write_bytes_exclusive(output / REPORT_FILE, method.report_text(summary).encode("utf-8"), 0o644)
    _require_membership(output, ARTIFACT_FILES)
    _assert_generated_modes(output)
    _write_json_exclusive(output / ARTIFACT_MANIFEST_FILE, _artifact_manifest(output), 0o644)
    # Seal-time verification is structural/semantic but deliberately does not rerun the scientific search.
    _verify_completed(output, regenerate=False)
    return result


def _verify_completed(output: Path, *, regenerate: bool) -> dict[str, object]:
    _require_membership(output, COMPLETED_FILES)
    _assert_generated_modes(output)
    if _read_json(output / ARTIFACT_MANIFEST_FILE) != _artifact_manifest(output):
        raise ValueError("MM-007 artifact manifest or artifact bytes/modes differ")
    manifest = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(manifest, dict) or manifest != _expected_input_manifest(output):
        raise ValueError("MM-007 input manifest no longer recomputes")
    formal_start = _read_json(output / STARTED_FILE)
    if formal_start != _formal_start_record(output, manifest):
        raise ValueError("MM-007 formal marker differs from frozen inputs")
    table, parent_evidence, validation = _load_analysis_inputs(output)
    saved = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    if regenerate:
        regenerated = method.validate_evidence(method.execute(table, parent_evidence))
        _assert_nested_close(saved, regenerated, path="evidence")
    summary = method.summarize(saved)
    expected_result = _result_record(formal_start, validation, saved, summary)
    if _read_json(output / RESULT_FILE) != expected_result:
        raise ValueError("MM-007 result does not recompute from primitive evidence")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != method.report_text(summary):
        raise ValueError("MM-007 report is not canonical")
    _verify_live_inputs()
    decision = cast(Mapping[str, object], summary["decision"])
    return {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": decision["classification"],
        "artifact_count": len(ARTIFACT_FILES),
    }


@_integrity_boundary
def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Verify structure and regenerate completed science from the copied NPZ."""

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
    return _verify_completed(output, regenerate=True)


@_integrity_boundary
def verify_semantic(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Validate, redecode every source frame, then regenerate the scientific result."""

    _assert_expected_output(output)
    files, _ = _tree_members(output)
    if files != set(COMPLETED_FILES):
        raise ValueError("semantic verification requires completed MM-007 outcomes")
    _verify_completed(output, regenerate=False)
    stored = _load_npz(output / FRAME_FILE, FRAME_SCHEMA)
    pixels = _strict_pixel_arrays(output / MM006_COPY_ROOT / MM006_PIXEL_RELATIVE)
    manifest = _read_json(output / MM004_COPY_ROOT / INPUT_MANIFEST_FILE)
    regenerated = _decode_frame_arrays(pixels, manifest)
    for name in sorted(FRAME_KEYS):
        if not np.array_equal(stored[name], regenerated[name]):
            raise ValueError(f"semantic redecode differs from prepared frames: {name}")
    verification = _verify_completed(output, regenerate=True)
    return {
        **verification,
        "outcomes": "verified_semantic_results",
        "semantic_regeneration": (
            "all scientific evidence regenerated from copied NPZ and all authenticated media frames "
            "redecoded bit-exact"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-007 physically matched resolution diagnostic")
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
    "FRAME_ARRAY_SHA256",
    "FRAME_FILE",
    "FRAME_PACKAGE_SHA256",
    "FRAMES_UINT8_SHA256",
    "INPUT_MANIFEST_FILE",
    "InvalidMM007Package",
    "InvalidMM007ParentParity",
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
