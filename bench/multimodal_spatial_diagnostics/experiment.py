"""Sealed preparation, execution, and verification lifecycle for MM-004."""

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

from bench.multimodal_preflight import backends as mm001_backends
from bench.multimodal_preflight import dataset as mm001_dataset
from bench.multimodal_preflight import experiment as mm001_experiment
from bench.multimodal_transform_diagnostics import experiment as mm003_experiment
from bench.multimodal_transform_diagnostics import method as mm003_method

from . import method

SCHEMA_VERSION = "mm004-formal-v1"
EXPERIMENT_ID = "MM-004"
PARENT_CLASSIFICATION = "no_linear_full_taesd_signal_at_frozen_margin"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/multimodal_spatial_diagnostics/results/MM-004")
EXPECTED_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT
PROTOCOL_DOC = Path("docs/research/2026-07-15-mm004-spatial-history-signal-isolation-protocol.md")

PROTOCOL_COPY_FILE = Path("MM-004-protocol.md")
INPUT_MANIFEST_FILE = Path("input-manifest.json")
PIXEL_FILE = Path("MM-004-pixel-grids.npz")
STARTED_FILE = Path("formal-start.json")
EVIDENCE_FILE = Path("MM-004-evidence.json")
RESULT_FILE = Path("MM-004-results.json")
REPORT_FILE = Path("MM-004-report.md")
ARTIFACT_MANIFEST_FILE = Path("artifact-manifest.json")

MM003_ROOT = Path("bench/multimodal_transform_diagnostics/results/MM-003")
MM003_COPY_ROOT = Path("inputs/MM-003")
MM003_SELECTED = (
    Path("artifact-manifest.json"),
    Path("input-manifest.json"),
    Path("MM-003-evidence.json"),
    Path("MM-003-results.json"),
    Path("inputs/MM-001/MM-001-features.npz"),
    Path("inputs/MM-001/MM-001-component-audit.npz"),
    Path("inputs/MM-001/input-manifest.json"),
)

MM003_ARTIFACT_MANIFEST_SHA256 = "ddb0449c29411578ced95b2b221a54e1733480d6c1e6d7b9c2203f4a011bf6f6"
MM003_INPUT_MANIFEST_SHA256 = "767411c30fe048c2d10d4946f989bee7e9e3553b72ae7a4d99e210da0226f929"
MM003_EVIDENCE_SHA256 = "ee925d348e38a77ee9860251e3980d977f9ee8f9b643acf14bdd1bbfa4f3519a"
MM003_RESULT_SHA256 = "c8d3bb4107cf6c868eed31ed6e269fa770bb734efbc9deb5b6edbd92de566324"
MM001_FEATURE_SHA256 = "3fdf0c988cf0bdb428432b67c71fc7a18404080b6e12bfe8b6226d2276330755"
MM001_COMPONENT_SHA256 = "476da8f2192c6bd57ecab6f861e975fc0827977fa8081462423fa4644e0c89e4"
MM001_INPUT_MANIFEST_SHA256 = "3aee513e4a5059e63e498382ad0b5d08bb1319a1cc30d70648490fd62b1fd539"

PREPARED_ROOT_FILES = (PROTOCOL_COPY_FILE, INPUT_MANIFEST_FILE, PIXEL_FILE)
PARENT_COPY_FILES = tuple(MM003_COPY_ROOT / path for path in MM003_SELECTED)
PREPARED_FILES = (*PREPARED_ROOT_FILES, *PARENT_COPY_FILES)
OUTCOME_FILES = (STARTED_FILE, EVIDENCE_FILE, RESULT_FILE, REPORT_FILE)
ARTIFACT_FILES = (*PREPARED_FILES, *OUTCOME_FILES)
COMPLETED_FILES = (*ARTIFACT_FILES, ARTIFACT_MANIFEST_FILE)

GENERATED_0644_FILES = (
    PROTOCOL_COPY_FILE,
    INPUT_MANIFEST_FILE,
    PIXEL_FILE,
    EVIDENCE_FILE,
    RESULT_FILE,
    REPORT_FILE,
    ARTIFACT_MANIFEST_FILE,
)
PIXEL_KEYS = {"video_ids", "timestamps", "pixel_current", "pixel_target"}
PIXEL_SCHEMA: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "video_ids": (np.dtype("<U11"), (477,)),
    "timestamps": (np.dtype("<f8"), (477,)),
    "pixel_current": (np.dtype("<f4"), (477, 3, 8, 8)),
    "pixel_target": (np.dtype("<f4"), (477, 3, 8, 8)),
}

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InvalidMM004Package(ValueError):
    """Stable fail-closed classification for MM-004 package defects."""

    classification = "invalid_MM004_package"


class InvalidMM004ParentParity(ValueError):
    """Stable classification for failure to reproduce the MM-003 comparator."""

    classification = "invalid_MM004_parent_parity"


def _integrity_boundary(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return function(*args, **kwargs)
        except (InvalidMM004Package, InvalidMM004ParentParity):
            raise
        except Exception as error:
            raise InvalidMM004Package(str(error)) from error

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
    return {
        "sha256": _file_hash(path),
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


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
    package_root = REPO_ROOT / "bench/multimodal_spatial_diagnostics"
    expected_package = {
        Path("bench/multimodal_spatial_diagnostics/__init__.py"),
        Path("bench/multimodal_spatial_diagnostics/__main__.py"),
        Path("bench/multimodal_spatial_diagnostics/experiment.py"),
        Path("bench/multimodal_spatial_diagnostics/method.py"),
    }
    actual_package = {path.relative_to(REPO_ROOT) for path in package_root.glob("*.py")}
    if actual_package != expected_package:
        raise ValueError("MM-004 package source membership differs from the frozen four-file set")
    own = {
        PROTOCOL_DOC,
        Path("tests/test_mm004_method.py"),
        Path("tests/test_mm004_experiment.py"),
        *actual_package,
    }
    paths = tuple(sorted({*mm003_experiment._source_paths(), *own}, key=str))
    if len(paths) != 47:
        raise ValueError(f"MM-004 source membership must contain exactly 47 files, got {len(paths)}")
    return paths


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _source_paths():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bound MM-004 source is missing or a symlink: {relative}")
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
        raise ValueError(f"MM-004 output must be canonical path {DEFAULT_OUTPUT}")
    current = output
    while True:
        if current.is_symlink():
            raise ValueError(f"MM-004 output path contains a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent
    if output.exists() and not output.is_dir():
        raise ValueError("MM-004 output must be a real directory")
    resolved = output.resolve()
    protected = (
        REPO_ROOT / mm001_experiment.DEFAULT_OUTPUT,
        REPO_ROOT / mm003_experiment.MM002_ROOT,
        REPO_ROOT / MM003_ROOT,
    )
    for parent_root in protected:
        parent_resolved = parent_root.resolve()
        if resolved == parent_resolved or resolved in parent_resolved.parents or parent_resolved in resolved.parents:
            raise ValueError("MM-004 output overlaps a protected parent")


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
            "MM-004 membership mismatch: "
            f"missing={sorted(str(path) for path in expected_files - files)}, "
            f"extra={sorted(str(path) for path in files - expected_files)}, "
            f"missing_dirs={sorted(str(path) for path in expected_directories - directories)}, "
            f"extra_dirs={sorted(str(path) for path in directories - expected_directories)}"
        )


def _assert_generated_modes(output: Path) -> None:
    for relative in GENERATED_0644_FILES:
        path = output / relative
        if path.exists() and stat.S_IMODE(path.stat().st_mode) != 0o644:
            raise ValueError(f"MM-004 generated file mode differs from 0644: {relative}")
    marker = output / STARTED_FILE
    if marker.exists() and stat.S_IMODE(marker.stat().st_mode) != 0o444:
        raise ValueError("MM-004 formal marker is not mode 0444")


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
            raise ValueError(f"selected parent file is absent from manifest: {relative}")
        actual = _file_record(root / relative)
        if actual != dict(expected):
            raise ValueError(f"selected parent file differs from manifest: {relative}")


def _verify_live_parent() -> dict[str, object]:
    verification = mm003_experiment.verify(mm003_experiment.DEFAULT_OUTPUT)
    if verification.get("outcomes") != "verified_results":
        raise ValueError("MM-004 requires completed verified MM-003 results")
    if verification.get("classification") != PARENT_CLASSIFICATION:
        raise ValueError("MM-003 parent classification differs from the frozen branch")
    root = REPO_ROOT / MM003_ROOT
    pins = {
        Path("artifact-manifest.json"): MM003_ARTIFACT_MANIFEST_SHA256,
        Path("input-manifest.json"): MM003_INPUT_MANIFEST_SHA256,
        Path("MM-003-evidence.json"): MM003_EVIDENCE_SHA256,
        Path("MM-003-results.json"): MM003_RESULT_SHA256,
        Path("inputs/MM-001/MM-001-features.npz"): MM001_FEATURE_SHA256,
        Path("inputs/MM-001/MM-001-component-audit.npz"): MM001_COMPONENT_SHA256,
        Path("inputs/MM-001/input-manifest.json"): MM001_INPUT_MANIFEST_SHA256,
    }
    for relative, expected in pins.items():
        if _file_hash(root / relative) != expected:
            raise ValueError(f"pinned MM-003 parent hash differs: {relative}")
    _verify_selected_against_manifest(root, MM003_SELECTED)
    return {
        "experiment_id": "MM-003",
        "classification": PARENT_CLASSIFICATION,
        "verification": verification,
        "live_path": str(MM003_ROOT),
        "copy_path": str(MM003_COPY_ROOT),
        "files": _records(root, mm003_experiment.COMPLETED_FILES),
        "selected_files": [str(path) for path in MM003_SELECTED],
        "pinned": {
            "artifact_manifest_sha256": MM003_ARTIFACT_MANIFEST_SHA256,
            "input_manifest_sha256": MM003_INPUT_MANIFEST_SHA256,
            "evidence_sha256": MM003_EVIDENCE_SHA256,
            "result_sha256": MM003_RESULT_SHA256,
            "feature_sha256": MM001_FEATURE_SHA256,
            "component_sha256": MM001_COMPONENT_SHA256,
            "mm001_input_manifest_sha256": MM001_INPUT_MANIFEST_SHA256,
        },
        "scientific_relationship": (
            "outcome-informed direct child reusing the same eight videos; not independent evidence"
        ),
    }


def _validate_parent_copy(output: Path, snapshot: Mapping[str, object]) -> None:
    root = output / MM003_COPY_ROOT
    files, directories = _tree_members(root)
    expected_directories = _expected_directories(MM003_SELECTED)
    if files != set(MM003_SELECTED) or directories != expected_directories:
        raise ValueError("copied MM-003 receipt has unexpected membership")
    records = snapshot.get("files")
    if not isinstance(records, Mapping):
        raise ValueError("live MM-003 snapshot has no file records")
    for relative in MM003_SELECTED:
        expected = records.get(str(relative))
        if not isinstance(expected, Mapping) or _file_record(root / relative) != dict(expected):
            raise ValueError(f"copied MM-003 file differs from the live sealed parent: {relative}")
    _verify_selected_against_manifest(root, MM003_SELECTED)


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
    cast(Any, np.savez_compressed)(
        buffer,
        **{name: np.asarray(value) for name, value in sorted(arrays.items())},
    )
    return buffer.getvalue()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_component_arrays(path: Path) -> dict[str, np.ndarray]:
    return _load_npz(
        path,
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


def _media_contract(mm001_manifest: object) -> dict[str, object]:
    if not isinstance(mm001_manifest, Mapping) or not isinstance(mm001_manifest.get("dataset"), Mapping):
        raise ValueError("copied MM-001 manifest has no dataset record")
    dataset_record = cast(Mapping[str, object], mm001_manifest["dataset"])
    cache_path = dataset_record.get("cache_path")
    video_ids = dataset_record.get("video_ids")
    hashes = dataset_record.get("media_sha256")
    inspection = dataset_record.get("media_inspection")
    if (
        not isinstance(cache_path, str)
        or video_ids != list(mm001_dataset.SAMPLE_VIDEO_IDS)
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(mm001_dataset.SAMPLE_VIDEO_IDS)
        or not all(isinstance(value, str) and len(value) == 64 for value in hashes.values())
        or not isinstance(inspection, Mapping)
    ):
        raise ValueError("copied MM-001 media identity is invalid")
    videos = inspection.get("videos")
    environment = inspection.get("environment")
    if not isinstance(videos, Mapping) or set(videos) != set(mm001_dataset.SAMPLE_VIDEO_IDS):
        raise ValueError("copied MM-001 media inspection video set differs")
    if not isinstance(environment, Mapping) or not isinstance(environment.get("ffmpeg"), Mapping):
        raise ValueError("copied MM-001 ffmpeg identity is missing")
    ffmpeg = cast(Mapping[str, object], environment["ffmpeg"])
    if (
        not isinstance(ffmpeg.get("path"), str)
        or not isinstance(ffmpeg.get("sha256"), str)
        or not isinstance(ffmpeg.get("size_bytes"), int)
    ):
        raise ValueError("copied MM-001 ffmpeg identity is invalid")
    sizes: dict[str, int] = {}
    frame_counts: dict[str, int] = {}
    for video_id in mm001_dataset.SAMPLE_VIDEO_IDS:
        record = videos.get(video_id)
        if not isinstance(record, Mapping):
            raise ValueError(f"copied MM-001 media record is invalid: {video_id}")
        size = record.get("file_size_bytes")
        frame_count = record.get("frame_count_2fps")
        if not isinstance(size, int) or not isinstance(frame_count, int):
            raise ValueError(f"copied MM-001 media size/frame count is invalid: {video_id}")
        sizes[video_id] = size
        frame_counts[video_id] = frame_count
    return {
        "cache_path": cache_path,
        "video_ids": list(mm001_dataset.SAMPLE_VIDEO_IDS),
        "media_sha256": {key: cast(str, hashes[key]) for key in sorted(hashes)},
        "media_size_bytes": sizes,
        "frame_count_2fps": frame_counts,
        "ffmpeg": {
            "path": ffmpeg["path"],
            "sha256": ffmpeg["sha256"],
            "size_bytes": ffmpeg["size_bytes"],
        },
        "decode": {
            "frame_rate": 2,
            "frame_size": 64,
            "letterboxed": True,
            "area_pool": "float64 mean over non-overlapping 8x8 blocks then float32",
            "output_shape": [477, 3, 8, 8],
            "target_horizon_seconds": 1.0,
        },
    }


def _area_pool_frame(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame)
    if values.dtype != np.dtype("<f4") or values.shape != (64, 64, 3):
        raise ValueError("decoded RGB frame must be float32 [64,64,3]")
    if not np.all(np.isfinite(values)) or float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
        raise ValueError("decoded RGB frame must contain finite values in [0,1]")
    blocked = values.reshape(8, 8, 8, 8, 3)
    pooled = np.mean(blocked, axis=(1, 3), dtype=np.float64).astype(np.float32)
    return np.asarray(np.transpose(pooled, (2, 0, 1)), dtype=np.float32)


def _extract_pixel_arrays(feature_table: Any, mm001_manifest: object) -> dict[str, np.ndarray]:
    """Authenticate and decode media during preparation only."""

    contract = _media_contract(mm001_manifest)
    cache_path = Path(cast(str, contract["cache_path"]))
    expected_hashes = cast(Mapping[str, str], contract["media_sha256"])
    expected_sizes = cast(Mapping[str, int], contract["media_size_bytes"])
    observed = mm001_dataset.validate_media_hashes(
        cache_path,
        expected_hashes=expected_hashes,
        expected_sizes=expected_sizes,
    )
    if observed != dict(expected_hashes):
        raise ValueError("authenticated media hashes differ from the copied MM-001 manifest")
    ffmpeg_record = cast(Mapping[str, object], contract["ffmpeg"])
    ffmpeg = Path(cast(str, ffmpeg_record["path"]))
    if (
        ffmpeg.is_symlink()
        or not ffmpeg.is_file()
        or ffmpeg.stat().st_size != ffmpeg_record["size_bytes"]
        or _file_hash(ffmpeg) != ffmpeg_record["sha256"]
    ):
        raise ValueError("sealed ffmpeg executable identity differs")

    video_ids = np.asarray(feature_table.video_ids, dtype="<U11")
    timestamps = np.asarray(feature_table.timestamps, dtype="<f8")
    if video_ids.shape != (477,) or timestamps.shape != (477,):
        raise ValueError("MM-001 feature identities must contain 477 rows")
    current = np.empty((477, 3, 8, 8), dtype=np.float32)
    target = np.empty((477, 3, 8, 8), dtype=np.float32)
    frame_counts = cast(Mapping[str, int], contract["frame_count_2fps"])
    for video_id in mm001_dataset.SAMPLE_VIDEO_IDS:
        media_path = cache_path / "videos" / f"{video_id}.mp4"
        frames = mm001_backends.decode_video_frames(media_path, ffmpeg=str(ffmpeg))
        if frames.dtype != np.dtype("<f4") or frames.shape[0] != frame_counts[video_id]:
            raise ValueError(f"decoded frame identity differs for {video_id}")
        indices = np.flatnonzero(video_ids == video_id)
        if len(indices) != mm001_dataset.EXPECTED_WINDOW_COUNTS[video_id]:
            raise ValueError(f"feature window count differs for {video_id}")
        for row in indices:
            timestamp = float(timestamps[row])
            current_index = mm001_backends.frame_index_at(timestamp, len(frames))
            target_index = mm001_backends.frame_index_at(
                timestamp + mm001_dataset.VISUAL_TARGET_HORIZON_SECONDS,
                len(frames),
            )
            current[row] = _area_pool_frame(frames[current_index])
            target[row] = _area_pool_frame(frames[target_index])
    return {
        "video_ids": video_ids,
        "timestamps": timestamps,
        "pixel_current": current,
        "pixel_target": target,
    }


def _raw_grid_table(
    video_ids: np.ndarray,
    timestamps: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
) -> Any:
    """Local adapter for the frozen MM-004 method table constructor."""

    return method.raw_grid_table(
        video_ids=np.asarray(video_ids, dtype="<U11"),
        timestamps=np.asarray(timestamps, dtype="<f8"),
        current=np.asarray(current),
        target=np.asarray(target),
        expected_channels=int(np.asarray(current).shape[1]),
    )


def _parent_preflight(taesd_raw_table: Any, parent_evidence: Mapping[str, object]) -> dict[str, object]:
    try:
        return cast(dict[str, object], method.parent_preflight_record(taesd_raw_table, parent_evidence))
    except ValueError as error:
        raise InvalidMM004ParentParity(str(error)) from error


def _execute(taesd_raw_table: Any, pixel_raw_table: Any, parent_evidence: Mapping[str, object]) -> object:
    """Local adapter for the scientific execution API."""

    return method.execute(taesd_raw_table, pixel_raw_table, parent_evidence)


def _summarize(evidence: object, parent_evidence: Mapping[str, object]) -> dict[str, object]:
    """Local adapter for the frozen decision replay API."""

    del parent_evidence
    return cast(dict[str, object], method.summarize(evidence))


def _validate_evidence_provenance(
    taesd_raw_table: Any,
    evidence: Mapping[str, object],
    input_validation: Mapping[str, object],
) -> None:
    """Bind saved provenance rows to copied inputs without regenerating real fits."""

    expected_parent = input_validation.get("parent_preflight")
    if evidence.get("parent_preflight") != expected_parent:
        raise ValueError("MM-004 evidence parent preflight differs from copied-input parity")

    history = method.history_table(taesd_raw_table)
    expected_panels: list[dict[str, Any]] = []
    for seed in method.SYNTHETIC_SEEDS:
        _, record = method.synthetic_panel(history, seed)
        expected_panels.append(record)
    if evidence.get("synthetic_panels") != expected_panels:
        raise ValueError("MM-004 synthetic panel provenance does not regenerate")


def _load_analysis_inputs(
    output: Path,
) -> tuple[Any, Any, dict[str, object], dict[str, object]]:
    parent_root = output / MM003_COPY_ROOT
    feature_path = parent_root / "inputs/MM-001/MM-001-features.npz"
    component_path = parent_root / "inputs/MM-001/MM-001-component-audit.npz"
    feature_table, feature_schema = mm001_experiment._load_feature_table(feature_path)
    components = _load_component_arrays(component_path)
    pixels = _load_npz(output / PIXEL_FILE, PIXEL_SCHEMA)
    for name in ("pixel_current", "pixel_target"):
        values = pixels[name]
        if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
            raise ValueError(f"prepared {name} values must remain in [0,1]")
    video_ids = np.asarray(feature_table.video_ids, dtype="<U11")
    timestamps = np.asarray(feature_table.timestamps, dtype="<f8")
    if not np.array_equal(pixels["video_ids"], video_ids) or not np.array_equal(pixels["timestamps"], timestamps):
        raise ValueError("prepared pixel identities differ from the copied MM-001 feature table")
    taesd_table = _raw_grid_table(
        video_ids,
        timestamps,
        components["taesd_latents"],
        components["target_taesd_latents"],
    )
    pixel_table = _raw_grid_table(video_ids, timestamps, pixels["pixel_current"], pixels["pixel_target"])
    parent_evidence = mm003_method.validate_evidence(_read_json(parent_root / "MM-003-evidence.json"))
    preflight = _parent_preflight(taesd_table, cast(Mapping[str, object], parent_evidence))
    mm001_manifest = _read_json(parent_root / "inputs/MM-001/input-manifest.json")
    media = _media_contract(mm001_manifest)
    pixel_record = {
        "file": _file_record(output / PIXEL_FILE),
        "schema": {
            name: {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": _array_sha256(array)}
            for name, array in sorted(pixels.items())
        },
        "media": media,
    }
    return (
        taesd_table,
        pixel_table,
        cast(dict[str, object], parent_evidence),
        {
            "feature_schema": feature_schema,
            "taesd_schema": {
                "current": {"dtype": components["taesd_latents"].dtype.str, "shape": [477, 4, 8, 8]},
                "target": {"dtype": components["target_taesd_latents"].dtype.str, "shape": [477, 4, 8, 8]},
                "flattening": "NumPy C order [477,4,8,8] -> [477,256]",
            },
            "pixel_preparation": pixel_record,
            "parent_preflight": preflight,
        },
    )


def _config_record() -> dict[str, object]:
    return {
        "method": method.config_record(),
        "lifecycle": {
            "prepared_files": [str(path) for path in PREPARED_FILES],
            "completed_files": [str(path) for path in COMPLETED_FILES],
            "generated_file_mode": "0644",
            "formal_marker_mode": "0444",
            "semantic_tolerance": {"rtol": 1e-12, "atol": 1e-12},
            "media_used": "preparation_only",
        },
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    snapshot = _verify_live_parent()
    _validate_parent_copy(output, snapshot)
    source_protocol = REPO_ROOT / PROTOCOL_DOC
    protocol_copy = output / PROTOCOL_COPY_FILE
    if protocol_copy.read_bytes() != source_protocol.read_bytes():
        raise ValueError("MM-004 protocol copy differs from bound source")
    if stat.S_IMODE(protocol_copy.stat().st_mode) != 0o644:
        raise ValueError("MM-004 protocol copy mode differs from 0644")
    _, _, _, input_record = _load_analysis_inputs(output)
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
        "source_count": 47,
        "dependencies": _dependency_versions(),
        "parent": snapshot,
        "input_validation": input_record,
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
        raise ValueError("MM-004 input manifest no longer recomputes")
    return cast(dict[str, object], saved)


def _formal_start_record(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    parent = cast(Mapping[str, Any], manifest["parent"])
    validation = cast(Mapping[str, Any], manifest["input_validation"])
    pixel = cast(Mapping[str, Any], validation["pixel_preparation"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "input_manifest_sha256": _file_hash(output / INPUT_MANIFEST_FILE),
        "protocol_sha256": cast(Mapping[str, object], manifest["protocol"])["sha256"],
        "source_sha256": _canonical_json_sha256(manifest["source"]),
        "config_sha256": manifest["config_sha256"],
        "prepared_membership_sha256": manifest["prepared_membership_sha256"],
        "mm003_artifact_manifest_sha256": parent["pinned"]["artifact_manifest_sha256"],
        "mm003_evidence_sha256": parent["pinned"]["evidence_sha256"],
        "mm003_result_sha256": parent["pinned"]["result_sha256"],
        "pixel_grids_sha256": pixel["file"]["sha256"],
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
        "epistemic_role": "outcome-informed spatial/history diagnostic; not independent confirmation",
        "formal_start": dict(formal_start),
        "parent_preflight": dict(cast(Mapping[str, object], input_validation["parent_preflight"])),
        "pixel_grids_sha256": cast(Mapping[str, Any], input_validation["pixel_preparation"])["file"]["sha256"],
        "evidence_sha256": _canonical_json_sha256(evidence),
        "summary": dict(summary),
    }


def _artifact_manifest(output: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
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
    """Copy frozen receipts, authenticate media, and prepare pixels without outcomes."""

    _assert_expected_output(output)
    snapshot = _verify_live_parent()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("MM-004 output must be absent or empty before preparation")
    _mkdir_fsynced(output)
    for relative in sorted(
        _expected_directories(PREPARED_FILES) - {Path(".")},
        key=lambda path: (len(path.parts), str(path)),
    ):
        _mkdir_fsynced(output / relative)
    parent_root = REPO_ROOT / MM003_ROOT
    for relative in MM003_SELECTED:
        _copy_file_exclusive(parent_root / relative, output / MM003_COPY_ROOT / relative)
    _validate_parent_copy(output, snapshot)
    _copy_file_exclusive(REPO_ROOT / PROTOCOL_DOC, output / PROTOCOL_COPY_FILE, mode=0o644)

    copied_parent = output / MM003_COPY_ROOT
    feature_table, _ = mm001_experiment._load_feature_table(copied_parent / "inputs/MM-001/MM-001-features.npz")
    mm001_manifest = _read_json(copied_parent / "inputs/MM-001/input-manifest.json")
    pixel_arrays = _extract_pixel_arrays(feature_table, mm001_manifest)
    if set(pixel_arrays) != PIXEL_KEYS:
        raise ValueError("prepared pixel arrays have unexpected keys")
    _write_bytes_exclusive(output / PIXEL_FILE, _npz_bytes(pixel_arrays), 0o644)
    _load_npz(output / PIXEL_FILE, PIXEL_SCHEMA)

    manifest = _expected_input_manifest(output)
    _write_json_exclusive(output / INPUT_MANIFEST_FILE, manifest, 0o644)
    result = verify(output)
    return {**result, "status": "prepared_only"}


@_integrity_boundary
def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Consume MM-004 once, execute from copied inputs, and seal outcomes."""

    _assert_expected_output(output)
    manifest = _validate_prepared(output)
    _verify_live_parent()
    formal_start = _mark_formal_started(output, manifest)
    taesd_table, pixel_table, parent_evidence, input_validation = _load_analysis_inputs(output)
    raw_evidence = _execute(taesd_table, pixel_table, parent_evidence)
    evidence = method.validate_evidence(raw_evidence)
    _validate_evidence_provenance(taesd_table, evidence, input_validation)
    summary = _summarize(evidence, parent_evidence)
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
    """Fast structural, receipt, parity, provenance, and decision verification."""

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
        raise ValueError("MM-004 artifact manifest or artifact bytes/modes differ")
    manifest = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(manifest, dict) or manifest != _expected_input_manifest(output):
        raise ValueError("MM-004 input manifest no longer recomputes")
    formal_start = _read_json(output / STARTED_FILE)
    if formal_start != _formal_start_record(output, manifest):
        raise ValueError("MM-004 formal marker differs from frozen inputs")

    taesd_table, _, parent_evidence, input_validation = _load_analysis_inputs(output)
    evidence = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    _validate_evidence_provenance(taesd_table, evidence, input_validation)
    summary = _summarize(evidence, parent_evidence)
    expected_result = _result_record(formal_start, input_validation, evidence, summary)
    if _read_json(output / RESULT_FILE) != expected_result:
        raise ValueError("MM-004 result does not recompute from primitive evidence")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != method.report_text(summary):
        raise ValueError("MM-004 report is not canonical")
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
    """Regenerate every MM-004 fit from copied arrays without media inference."""

    verification = verify(output)
    if verification["outcomes"] != "verified_results":
        raise ValueError("semantic verification requires completed MM-004 outcomes")
    taesd_table, pixel_table, parent_evidence, _ = _load_analysis_inputs(output)
    regenerated = method.validate_evidence(_execute(taesd_table, pixel_table, parent_evidence))
    saved = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    _assert_nested_close(saved, regenerated, path="evidence")
    saved_summary = _summarize(saved, parent_evidence)
    regenerated_summary = _summarize(regenerated, parent_evidence)
    _assert_nested_close(saved_summary, regenerated_summary, path="summary")
    return {
        **verification,
        "outcomes": "verified_semantic_results",
        "semantic_regeneration": "all synthetic, TAESD, and pixel fits reproduced from copied arrays",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-004 spatial/history signal isolation")
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
    "InvalidMM004Package",
    "InvalidMM004ParentParity",
    "PIXEL_FILE",
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
