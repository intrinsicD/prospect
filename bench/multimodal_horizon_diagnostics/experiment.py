"""Sealed preparation, execution, and verification lifecycle for MM-005."""

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

from bench.multimodal_preflight import dataset as mm001_dataset
from bench.multimodal_preflight import experiment as mm001_experiment
from bench.multimodal_spatial_diagnostics import experiment as mm004_experiment
from bench.multimodal_spatial_diagnostics import method as mm004_method
from bench.multimodal_transform_diagnostics import experiment as mm003_experiment

from . import method

SCHEMA_VERSION = "mm005-formal-v1"
EXPERIMENT_ID = "MM-005"
PARENT_CLASSIFICATION = "tested_local_objective_or_horizon_failure_supported"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/multimodal_horizon_diagnostics/results/MM-005")
EXPECTED_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT
PROTOCOL_DOC = Path("docs/research/2026-07-15-mm005-matched-half-horizon-replay-protocol.md")

PROTOCOL_COPY_FILE = Path("MM-005-protocol.md")
INPUT_MANIFEST_FILE = Path("input-manifest.json")
STARTED_FILE = Path("formal-start.json")
EVIDENCE_FILE = Path("MM-005-evidence.json")
RESULT_FILE = Path("MM-005-results.json")
REPORT_FILE = Path("MM-005-report.md")
ARTIFACT_MANIFEST_FILE = Path("artifact-manifest.json")

MM004_ROOT = mm004_experiment.DEFAULT_OUTPUT
MM004_COPY_ROOT = Path("inputs/MM-004")
MM004_SELECTED = (
    Path("artifact-manifest.json"),
    Path("input-manifest.json"),
    Path("MM-004-evidence.json"),
    Path("MM-004-results.json"),
    Path("MM-004-pixel-grids.npz"),
    Path("inputs/MM-003/inputs/MM-001/MM-001-features.npz"),
    Path("inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz"),
)

MM004_PINS: dict[Path, str] = {
    Path("artifact-manifest.json"): "eb17a9b324658de95546325f1bd7c8c4cd27c9c5914988efd65eb01bd94b794c",
    Path("input-manifest.json"): "597a8bfc9f6ae1f6ff1f0d3be456f57d768f1866a1fac59cd981dd260076dc90",
    Path("MM-004-evidence.json"): "2c77eb74e8561a974f0466ff1a47c8c9a4a1c0c6f4d7fe7b3ce7fabf576e3b56",
    Path("MM-004-results.json"): "e1477c52f35d3d8f69f54092828fd3c15550e226343915c00b59529df9248c28",
    Path("MM-004-pixel-grids.npz"): "cca261a941e68a7ddc510eee3a3af958d33b6abaf958cb5562ed6b66c22f47c8",
    Path("inputs/MM-003/inputs/MM-001/MM-001-features.npz"): (
        "3fdf0c988cf0bdb428432b67c71fc7a18404080b6e12bfe8b6226d2276330755"
    ),
    Path("inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz"): (
        "476da8f2192c6bd57ecab6f861e975fc0827977fa8081462423fa4644e0c89e4"
    ),
}

FEATURE_RELATIVE = Path("inputs/MM-003/inputs/MM-001/MM-001-features.npz")
COMPONENT_RELATIVE = Path("inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz")
PIXEL_RELATIVE = Path("MM-004-pixel-grids.npz")

PREPARED_ROOT_FILES = (PROTOCOL_COPY_FILE, INPUT_MANIFEST_FILE)
PARENT_COPY_FILES = tuple(MM004_COPY_ROOT / path for path in MM004_SELECTED)
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

RAW_ROWS = 477
MATCHED_ROWS = 453
MATCHED_IDENTITY_SHA256 = "d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b"
MATCHED_COUNTS: dict[str, int] = {
    "video_10993": 60,
    "video_1580": 61,
    "video_2564": 56,
    "video_3501": 62,
    "video_6860": 62,
    "video_8241": 45,
    "video_874": 63,
    "video_9253": 44,
}
EXPECTED_TRAIN_ROWS = (332, 335, 346, 346)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InvalidMM005Package(ValueError):
    """Stable fail-closed classification for MM-005 package defects."""

    classification = "invalid_MM005_package"


class InvalidMM005ParentAlignment(ValueError):
    """Stable pre-marker classification for parent/alignment defects."""

    classification = "invalid_MM005_parent_alignment"


def _integrity_boundary(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return function(*args, **kwargs)
        except (InvalidMM005Package, InvalidMM005ParentAlignment):
            raise
        except Exception as error:
            raise InvalidMM005Package(str(error)) from error

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
    package_root = REPO_ROOT / "bench/multimodal_horizon_diagnostics"
    expected_package = {
        Path("bench/multimodal_horizon_diagnostics/__init__.py"),
        Path("bench/multimodal_horizon_diagnostics/__main__.py"),
        Path("bench/multimodal_horizon_diagnostics/experiment.py"),
        Path("bench/multimodal_horizon_diagnostics/method.py"),
    }
    actual_package = {path.relative_to(REPO_ROOT) for path in package_root.glob("*.py")}
    if actual_package != expected_package:
        raise ValueError("MM-005 package source membership differs from the frozen four-file set")
    own = {
        PROTOCOL_DOC,
        Path("tests/test_mm005_method.py"),
        Path("tests/test_mm005_experiment.py"),
        *actual_package,
    }
    paths = tuple(sorted({*mm004_experiment._source_paths(), *own}, key=str))
    if len(paths) != 54:
        raise ValueError(f"MM-005 source membership must contain exactly 54 files, got {len(paths)}")
    return paths


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _source_paths():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bound MM-005 source is missing or a symlink: {relative}")
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
        raise ValueError(f"MM-005 output must be canonical path {DEFAULT_OUTPUT}")
    current = output
    while True:
        if current.is_symlink():
            raise ValueError(f"MM-005 output path contains a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent
    if output.exists() and not output.is_dir():
        raise ValueError("MM-005 output must be a real directory")
    resolved = output.resolve()
    protected = (
        REPO_ROOT / mm001_experiment.DEFAULT_OUTPUT,
        REPO_ROOT / mm003_experiment.MM002_ROOT,
        REPO_ROOT / mm004_experiment.MM003_ROOT,
        REPO_ROOT / MM004_ROOT,
    )
    for parent_root in protected:
        parent_resolved = parent_root.resolve()
        if resolved == parent_resolved or resolved in parent_resolved.parents or parent_resolved in resolved.parents:
            raise ValueError("MM-005 output overlaps a protected parent")


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
            "MM-005 membership mismatch: "
            f"missing={sorted(str(path) for path in expected_files - files)}, "
            f"extra={sorted(str(path) for path in files - expected_files)}, "
            f"missing_dirs={sorted(str(path) for path in expected_directories - directories)}, "
            f"extra_dirs={sorted(str(path) for path in directories - expected_directories)}"
        )


def _assert_generated_modes(output: Path) -> None:
    for relative in GENERATED_0644_FILES:
        path = output / relative
        if path.exists() and stat.S_IMODE(path.stat().st_mode) != 0o644:
            raise ValueError(f"MM-005 generated file mode differs from 0644: {relative}")
    marker = output / STARTED_FILE
    if marker.exists() and stat.S_IMODE(marker.stat().st_mode) != 0o444:
        raise ValueError("MM-005 formal marker is not mode 0444")


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
        if _file_record(root / relative) != dict(expected):
            raise ValueError(f"selected parent file differs from manifest: {relative}")


def _replay_parent_unchecked(root: Path) -> dict[str, object]:
    evidence = mm004_method.validate_evidence(_read_json(root / mm004_experiment.EVIDENCE_FILE))
    summary = mm004_method.summarize(evidence)
    result = _read_json(root / mm004_experiment.RESULT_FILE)
    if not isinstance(result, Mapping) or result.get("summary") != summary:
        raise ValueError("MM-004 evidence/result summary relationship does not replay")
    if result.get("evidence_sha256") != _canonical_json_sha256(evidence):
        raise ValueError("MM-004 result does not bind its canonical evidence")
    synthetic = summary.get("synthetic_control")
    decision = summary.get("decision")
    if (
        not isinstance(synthetic, Mapping)
        or synthetic.get("positive_passes") is not True
        or synthetic.get("negative_passes") is not True
    ):
        raise ValueError("MM-004 parent synthetic controls did not both pass")
    if not isinstance(decision, Mapping) or decision.get("classification") != PARENT_CLASSIFICATION:
        raise ValueError("MM-004 copied decision differs from the frozen parent branch")
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
    except InvalidMM005ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM005ParentAlignment(str(error)) from error


def _verify_live_parent_unchecked() -> dict[str, object]:
    verification = mm004_experiment.verify(MM004_ROOT)
    if verification.get("outcomes") != "verified_results":
        raise ValueError("MM-005 requires completed verified MM-004 results")
    if verification.get("classification") != PARENT_CLASSIFICATION:
        raise ValueError("MM-004 parent classification differs from the frozen branch")
    root = REPO_ROOT / MM004_ROOT
    for relative, expected in MM004_PINS.items():
        if _file_hash(root / relative) != expected:
            raise ValueError(f"pinned MM-004 parent hash differs: {relative}")
    _verify_selected_against_manifest(root, MM004_SELECTED)
    replay = _replay_parent(root)
    return {
        "experiment_id": "MM-004",
        "classification": PARENT_CLASSIFICATION,
        "verification": verification,
        "live_path": str(MM004_ROOT),
        "copy_path": str(MM004_COPY_ROOT),
        "files": _records(root, mm004_experiment.COMPLETED_FILES),
        "selected_files": [str(path) for path in MM004_SELECTED],
        "pinned": {str(path): digest for path, digest in MM004_PINS.items()},
        "replay": replay,
        "scientific_relationship": (
            "outcome-informed direct child reusing the same eight videos; not independent evidence"
        ),
    }


def _verify_live_parent() -> dict[str, object]:
    try:
        return _verify_live_parent_unchecked()
    except InvalidMM005ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM005ParentAlignment(str(error)) from error


def _validate_parent_copy_unchecked(output: Path, snapshot: Mapping[str, object]) -> dict[str, object]:
    root = output / MM004_COPY_ROOT
    files, directories = _tree_members(root)
    if files != set(MM004_SELECTED) or directories != _expected_directories(MM004_SELECTED):
        raise ValueError("copied MM-004 receipt has unexpected membership")
    records = snapshot.get("files")
    if not isinstance(records, Mapping):
        raise ValueError("live MM-004 snapshot has no file records")
    for relative in MM004_SELECTED:
        expected = records.get(str(relative))
        if not isinstance(expected, Mapping) or _file_record(root / relative) != dict(expected):
            raise ValueError(f"copied MM-004 file differs from the live sealed parent: {relative}")
    _verify_selected_against_manifest(root, MM004_SELECTED)
    replay = _replay_parent(root)
    if replay != snapshot.get("replay"):
        raise ValueError("copied MM-004 replay differs from the live sealed parent")
    return replay


def _validate_parent_copy(output: Path, snapshot: Mapping[str, object]) -> dict[str, object]:
    try:
        return _validate_parent_copy_unchecked(output, snapshot)
    except InvalidMM005ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM005ParentAlignment(str(error)) from error


def _strict_component_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = mm004_experiment._load_component_arrays(path)
    return arrays


def _strict_pixel_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = mm004_experiment._load_npz(path, mm004_experiment.PIXEL_SCHEMA)
    for name in ("pixel_current", "pixel_target"):
        values = arrays[name]
        if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
            raise ValueError(f"copied {name} values must remain in [0,1]")
    return arrays


def _domain_alignment_record(
    *,
    current: np.ndarray,
    saved_target: np.ndarray,
    groups: Sequence[np.ndarray],
    channels: int,
) -> dict[str, object]:
    previous_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    half_parts: list[np.ndarray] = []
    one_parts: list[np.ndarray] = []
    saved_half_parts: list[np.ndarray] = []
    saved_one_parts: list[np.ndarray] = []
    for ordered in groups:
        source = ordered[1:-2]
        previous_parts.append(current[ordered[:-3]])
        current_parts.append(current[source])
        half_parts.append(current[ordered[2:-1]])
        one_parts.append(current[ordered[3:]])
        saved_half_parts.append(saved_target[ordered[:-3]])
        saved_one_parts.append(saved_target[source])
    previous = np.concatenate(previous_parts)
    present = np.concatenate(current_parts)
    target_half = np.concatenate(half_parts)
    target_one = np.concatenate(one_parts)
    saved_half = np.concatenate(saved_half_parts)
    saved_one = np.concatenate(saved_one_parts)
    expected_shape = (MATCHED_ROWS, channels, 8, 8)
    if any(
        value.shape != expected_shape for value in (previous, present, target_half, target_one, saved_half, saved_one)
    ):
        raise ValueError("matched domain arrays have unexpected shape")
    half_error = float(np.max(np.abs(saved_half.astype(np.float64) - target_half.astype(np.float64))))
    one_error = float(np.max(np.abs(saved_one.astype(np.float64) - target_one.astype(np.float64))))
    if not np.array_equal(saved_half, target_half) or not np.array_equal(saved_one, target_one):
        raise ValueError("saved MM-004 targets do not align bit-exactly with matched current grids")
    return {
        "channels": channels,
        "rows": MATCHED_ROWS,
        "saved_half_target_rows_compared": MATCHED_ROWS,
        "saved_one_target_rows_compared": MATCHED_ROWS,
        "saved_half_target_max_absolute_error": half_error,
        "saved_one_target_max_absolute_error": one_error,
        "bit_exact_saved_target_parity": True,
        "array_sha256": {
            "previous": _array_sha256(previous),
            "current": _array_sha256(present),
            "target_0p5": _array_sha256(target_half),
            "target_1p0": _array_sha256(target_one),
        },
    }


def _alignment_record(
    feature_table: Any,
    components: Mapping[str, np.ndarray],
    pixels: Mapping[str, np.ndarray],
) -> dict[str, object]:
    try:
        video_ids = np.asarray(feature_table.video_ids, dtype="<U11")
        timestamps = np.asarray(feature_table.timestamps, dtype="<f8")
        if video_ids.shape != (RAW_ROWS,) or timestamps.shape != (RAW_ROWS,):
            raise ValueError("MM-005 feature identities must contain exactly 477 rows")
        if not np.array_equal(pixels["video_ids"], video_ids) or not np.array_equal(pixels["timestamps"], timestamps):
            raise ValueError("copied pixel identities differ from copied feature identities")
        expected_ids = tuple(mm001_dataset.SAMPLE_VIDEO_IDS)
        if set(video_ids.tolist()) != set(expected_ids):
            raise ValueError("MM-005 raw table has an unexpected video set")

        groups: list[np.ndarray] = []
        output_identities: list[list[object]] = []
        observed_counts: dict[str, int] = {}
        for video_id in expected_ids:
            indices = np.flatnonzero(video_ids == video_id)
            expected_raw = mm001_dataset.EXPECTED_WINDOW_COUNTS[video_id]
            if len(indices) != expected_raw:
                raise ValueError(f"raw row count differs for {video_id}")
            ordered = indices[np.argsort(timestamps[indices], kind="stable")]
            if not np.array_equal(indices, ordered):
                raise ValueError(f"raw row order is not strict timestamp order for {video_id}")
            if not np.array_equal(np.diff(timestamps[ordered]), np.full(len(ordered) - 1, 0.5)):
                raise ValueError(f"raw timestamp cadence differs from 0.5 seconds for {video_id}")
            groups.append(ordered)
            source = ordered[1:-2]
            observed_counts[video_id] = len(source)
            for row in source:
                output_identities.append([video_id, float(timestamps[row])])
            if not (
                np.all(video_ids[ordered[:-3]] == video_id)
                and np.all(video_ids[source] == video_id)
                and np.all(video_ids[ordered[2:-1]] == video_id)
                and np.all(video_ids[ordered[3:]] == video_id)
            ):
                raise ValueError(f"matched offsets cross a video boundary for {video_id}")

        if observed_counts != MATCHED_COUNTS or len(output_identities) != MATCHED_ROWS:
            raise ValueError("MM-005 matched row counts differ from the frozen 453-row panel")
        identity_sha = _canonical_json_sha256(output_identities)
        if identity_sha != MATCHED_IDENTITY_SHA256:
            raise ValueError("MM-005 matched identity SHA differs from the frozen value")
        train_rows = tuple(
            sum(MATCHED_COUNTS[video_id] for video_id in fold.train_ids) for fold in mm001_dataset.formal_folds()
        )
        if train_rows != EXPECTED_TRAIN_ROWS:
            raise ValueError("MM-005 fold training-row counts differ")

        taesd = _domain_alignment_record(
            current=components["taesd_latents"],
            saved_target=components["target_taesd_latents"],
            groups=groups,
            channels=4,
        )
        pixel = _domain_alignment_record(
            current=pixels["pixel_current"],
            saved_target=pixels["pixel_target"],
            groups=groups,
            channels=3,
        )
        return {
            "passed": True,
            "raw_rows": RAW_ROWS,
            "matched_rows": MATCHED_ROWS,
            "matched_counts": dict(MATCHED_COUNTS),
            "matched_identity_sha256": identity_sha,
            "source_positions": "range(1, N-2)",
            "timestamp_cadence_seconds": 0.5,
            "target_offsets_seconds": [0.5, 1.0],
            "fold_train_rows": list(train_rows),
            "taesd": taesd,
            "pixel": pixel,
        }
    except InvalidMM005ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM005ParentAlignment(str(error)) from error


def _raw_grid_table(
    video_ids: np.ndarray,
    timestamps: np.ndarray,
    current: np.ndarray,
    saved_target: np.ndarray,
) -> Any:
    return method.raw_grid_table(
        video_ids=np.asarray(video_ids, dtype="<U11"),
        timestamps=np.asarray(timestamps, dtype="<f8"),
        current=np.asarray(current),
        saved_target=np.asarray(saved_target),
        expected_channels=int(np.asarray(current).shape[1]),
    )


def _execute(taesd_raw_table: Any, pixel_raw_table: Any) -> object:
    """Local adapter for the frozen scientific execution API."""

    return method.execute(taesd_raw_table, pixel_raw_table)


def _summarize(evidence: object) -> dict[str, object]:
    """Local adapter for the frozen decision replay API."""

    return cast(dict[str, object], method.summarize(evidence))


def _validate_evidence_provenance(
    taesd_raw_table: Any,
    pixel_raw_table: Any,
    evidence: Mapping[str, object],
) -> None:
    """Replay array and synthetic-panel provenance without regenerating fits."""

    taesd_panel = method.matched_panel(taesd_raw_table)
    pixel_panel = method.matched_panel(pixel_raw_table)
    expected_alignment = method.alignment_record(taesd_panel, pixel_panel)
    if evidence.get("alignment") != expected_alignment:
        raise ValueError("MM-005 evidence alignment differs from copied arrays")
    expected_panels: list[dict[str, Any]] = []
    for seed in method.SYNTHETIC_SEEDS:
        _, record = method.synthetic_panel(taesd_panel, seed)
        expected_panels.append(record)
    if evidence.get("synthetic_panels") != expected_panels:
        raise ValueError("MM-005 synthetic panel provenance does not regenerate")


def _load_analysis_inputs_unchecked(
    output: Path,
) -> tuple[Any, Any, dict[str, object]]:
    parent_root = output / MM004_COPY_ROOT
    feature_path = parent_root / FEATURE_RELATIVE
    component_path = parent_root / COMPONENT_RELATIVE
    pixel_path = parent_root / PIXEL_RELATIVE
    feature_table, feature_schema = mm001_experiment._load_feature_table(feature_path)
    components = _strict_component_arrays(component_path)
    pixels = _strict_pixel_arrays(pixel_path)
    alignment = _alignment_record(feature_table, components, pixels)

    video_ids = np.asarray(feature_table.video_ids, dtype="<U11")
    timestamps = np.asarray(feature_table.timestamps, dtype="<f8")
    taesd_raw = _raw_grid_table(
        video_ids,
        timestamps,
        components["taesd_latents"],
        components["target_taesd_latents"],
    )
    pixel_raw = _raw_grid_table(
        video_ids,
        timestamps,
        pixels["pixel_current"],
        pixels["pixel_target"],
    )
    # This is a pure indexing/parity check. It intentionally performs no fit.
    method.matched_panel(taesd_raw)
    method.matched_panel(pixel_raw)

    parent_replay = _replay_parent(parent_root)
    component_schema = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        for name, value in sorted(components.items())
    }
    pixel_schema = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        for name, value in sorted(pixels.items())
    }
    return (
        taesd_raw,
        pixel_raw,
        {
            "feature_file": _file_record(feature_path),
            "feature_schema": feature_schema,
            "component_file": _file_record(component_path),
            "component_schema": component_schema,
            "pixel_file": _file_record(pixel_path),
            "pixel_schema": pixel_schema,
            "parent_replay": parent_replay,
            "alignment": alignment,
            "alignment_sha256": _canonical_json_sha256(alignment),
        },
    )


def _load_analysis_inputs(
    output: Path,
) -> tuple[Any, Any, dict[str, object]]:
    try:
        return _load_analysis_inputs_unchecked(output)
    except InvalidMM005ParentAlignment:
        raise
    except Exception as error:
        raise InvalidMM005ParentAlignment(str(error)) from error


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
            "derived_grid_artifact": False,
        },
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    snapshot = _verify_live_parent()
    replay = _validate_parent_copy(output, snapshot)
    source_protocol = REPO_ROOT / PROTOCOL_DOC
    protocol_copy = output / PROTOCOL_COPY_FILE
    if protocol_copy.read_bytes() != source_protocol.read_bytes():
        raise ValueError("MM-005 protocol copy differs from bound source")
    if stat.S_IMODE(protocol_copy.stat().st_mode) != 0o644:
        raise ValueError("MM-005 protocol copy mode differs from 0644")
    _, _, input_validation = _load_analysis_inputs(output)
    if input_validation["parent_replay"] != replay:
        raise ValueError("MM-005 input replay differs from the copied parent replay")
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
        "source_count": 54,
        "dependencies": _dependency_versions(),
        "parent": snapshot,
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
        raise ValueError("MM-005 input manifest no longer recomputes")
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
        "mm004_receipt_sha256": {str(path): pins[str(path)] for path in MM004_SELECTED},
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
        "epistemic_role": "outcome-informed matched-row horizon diagnostic; not independent confirmation",
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
        if not math.isclose(
            float(cast(float, saved)),
            float(cast(float, regenerated)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"semantic {path} differs")
        return
    if saved != regenerated:
        raise ValueError(f"semantic {path} differs")


@_integrity_boundary
def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Copy frozen receipts and validate matched alignment without fitting."""

    _assert_expected_output(output)
    snapshot = _verify_live_parent()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("MM-005 output must be absent or empty before preparation")
    _mkdir_fsynced(output)
    for relative in sorted(
        _expected_directories(PREPARED_FILES) - {Path(".")},
        key=lambda path: (len(path.parts), str(path)),
    ):
        _mkdir_fsynced(output / relative)
    parent_root = REPO_ROOT / MM004_ROOT
    for relative in MM004_SELECTED:
        _copy_file_exclusive(parent_root / relative, output / MM004_COPY_ROOT / relative)
    _validate_parent_copy(output, snapshot)
    _copy_file_exclusive(REPO_ROOT / PROTOCOL_DOC, output / PROTOCOL_COPY_FILE, mode=0o644)

    # All schema, identity, cadence, and bit-exact target checks happen here.
    # No real or synthetic estimator is invoked before the formal marker.
    _load_analysis_inputs(output)
    manifest = _expected_input_manifest(output)
    _write_json_exclusive(output / INPUT_MANIFEST_FILE, manifest, 0o644)
    result = verify(output)
    return {**result, "status": "prepared_only"}


@_integrity_boundary
def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Consume MM-005 once, execute from copied arrays, and seal outcomes."""

    _assert_expected_output(output)
    manifest = _validate_prepared(output)
    _verify_live_parent()
    formal_start = _mark_formal_started(output, manifest)
    taesd_raw, pixel_raw, input_validation = _load_analysis_inputs(output)
    raw_evidence = _execute(taesd_raw, pixel_raw)
    evidence = method.validate_evidence(raw_evidence)
    _validate_evidence_provenance(taesd_raw, pixel_raw, evidence)
    summary = _summarize(evidence)
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
    """Fast structural, receipt, alignment, evidence, and decision verification."""

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
        raise ValueError("MM-005 artifact manifest or artifact bytes/modes differ")
    manifest = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(manifest, dict) or manifest != _expected_input_manifest(output):
        raise ValueError("MM-005 input manifest no longer recomputes")
    formal_start = _read_json(output / STARTED_FILE)
    if formal_start != _formal_start_record(output, manifest):
        raise ValueError("MM-005 formal marker differs from frozen inputs")

    taesd_raw, pixel_raw, input_validation = _load_analysis_inputs(output)
    evidence = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    _validate_evidence_provenance(taesd_raw, pixel_raw, evidence)
    summary = _summarize(evidence)
    expected_result = _result_record(formal_start, input_validation, evidence, summary)
    if _read_json(output / RESULT_FILE) != expected_result:
        raise ValueError("MM-005 result does not recompute from primitive evidence")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != method.report_text(summary):
        raise ValueError("MM-005 report is not canonical")
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
    """Regenerate all 200 MM-005 fits from copied arrays only."""

    verification = verify(output)
    if verification["outcomes"] != "verified_results":
        raise ValueError("semantic verification requires completed MM-005 outcomes")
    taesd_raw, pixel_raw, _ = _load_analysis_inputs(output)
    regenerated = method.validate_evidence(_execute(taesd_raw, pixel_raw))
    saved = method.validate_evidence(_read_json(output / EVIDENCE_FILE))
    _assert_nested_close(saved, regenerated, path="evidence")
    saved_summary = _summarize(saved)
    regenerated_summary = _summarize(regenerated)
    _assert_nested_close(saved_summary, regenerated_summary, path="summary")
    return {
        **verification,
        "outcomes": "verified_semantic_results",
        "semantic_regeneration": "all 200 synthetic and real fits reproduced from copied arrays",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-005 matched half-horizon replay")
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
    "InvalidMM005Package",
    "InvalidMM005ParentAlignment",
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
