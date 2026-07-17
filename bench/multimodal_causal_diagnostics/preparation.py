"""Pure row preparation and source/target detachment for MM-009.

The functions in this module operate on caller-supplied arrays.  They do not know
the live MM-007 result path and never open it.  The formal custodian may later call
these functions after its marker/control authority is established; development and
tests use generated arrays only.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

from . import records

SCHEMA_VERSION: Final = "mm009-preparation-v1"
PROTOCOL_SHA256: Final[str] = records.PROTOCOL_SHA256

RAW_ROWS: Final = 477
MATCHED_ROWS: Final = 453
NATIVE_SIZE: Final = 64
CHANNELS: Final = 3
R8_SIZE: Final = 8
R8_BLOCK: Final = 8
SCALE_FLOOR: Final = 1e-6
MATCHED_IDENTITY_SHA256: Final = "d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b"

VIDEO_IDS: Final = (
    "video_10993",
    "video_1580",
    "video_2564",
    "video_3501",
    "video_6860",
    "video_8241",
    "video_874",
    "video_9253",
)
RAW_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "video_10993": 63,
        "video_1580": 64,
        "video_2564": 59,
        "video_3501": 65,
        "video_6860": 65,
        "video_8241": 48,
        "video_874": 66,
        "video_9253": 47,
    }
)
MATCHED_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {video_id: RAW_COUNTS[video_id] - 3 for video_id in VIDEO_IDS}
)


class PreparationValidationError(ValueError):
    """Raised when row preparation or target detachment fails closed."""


@dataclass(frozen=True, slots=True)
class FoldSpec:
    index: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index not in range(4):
            raise PreparationValidationError("fold index must be in range(4)")
        if len(self.train_ids) != 6 or len(self.test_ids) != 2:
            raise PreparationValidationError("each fold must have six train and two test videos")
        if set(self.train_ids).intersection(self.test_ids) or set((*self.train_ids, *self.test_ids)) != set(VIDEO_IDS):
            raise PreparationValidationError("fold membership is not an exact partition")


FOLDS: Final = tuple(
    FoldSpec(
        index=index,
        test_ids=VIDEO_IDS[2 * index : 2 * index + 2],
        train_ids=tuple(video_id for video_id in VIDEO_IDS if video_id not in VIDEO_IDS[2 * index : 2 * index + 2]),
    )
    for index in range(4)
)


@dataclass(frozen=True, slots=True)
class RowIndex:
    ordinal: int
    video_id: str
    fold_index: int
    video_row: int
    previous_index: int
    current_index: int
    future_index: int
    previous_timestamp: float
    current_timestamp: float
    future_timestamp: float

    def __post_init__(self) -> None:
        integer_fields = (
            self.ordinal,
            self.fold_index,
            self.video_row,
            self.previous_index,
            self.current_index,
            self.future_index,
        )
        if any(type(value) is not int for value in integer_fields):
            raise PreparationValidationError("row indices must be exact integers")
        if self.video_id not in VIDEO_IDS or self.fold_index != VIDEO_IDS.index(self.video_id) // 2:
            raise PreparationValidationError("row video/fold identity differs")
        if not 0 <= self.ordinal < MATCHED_ROWS or not 0 <= self.video_row < MATCHED_COUNTS[self.video_id]:
            raise PreparationValidationError("row ordinal is outside the frozen panel")
        if not (
            self.previous_index + 1 == self.current_index
            and self.current_index + 1 == self.future_index
            and 0 <= self.previous_index < self.future_index < RAW_ROWS
        ):
            raise PreparationValidationError("row raw indices are not one consecutive causal triple")
        expected_current = 1.5 + 0.5 * self.video_row
        if (
            self.previous_timestamp != expected_current - 0.5
            or self.current_timestamp != expected_current
            or self.future_timestamp != expected_current + 0.5
        ):
            raise PreparationValidationError("row timestamps differ from the frozen half-second grid")


def _readonly_float64_vector(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype="<f8")
    if array.shape != (CHANNELS,) or not np.all(np.isfinite(array)):
        raise PreparationValidationError(f"{label} must be a finite three-vector")
    return records.immutable_array(np.ascontiguousarray(array))


def mm007_normalizer_fingerprint(mean: np.ndarray, scale: np.ndarray) -> str:
    """Replay MM-007's source-only normalizer fingerprint grammar."""

    left = _readonly_float64_vector(mean, "normalizer mean")
    right = _readonly_float64_vector(scale, "normalizer scale")
    joined = np.ascontiguousarray(np.concatenate((left, right)), dtype="<f8")
    digest = hashlib.sha256(b"mm007-array-v1")
    digest.update(str(joined.shape).encode("ascii"))
    digest.update(joined.dtype.str.encode("ascii"))
    digest.update(joined.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FoldNormalizer:
    fold_index: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_rows: int
    mean: np.ndarray
    scale: np.ndarray
    fingerprint: str

    def __post_init__(self) -> None:
        if type(self.fold_index) is not int or self.fold_index not in range(4):
            raise PreparationValidationError("normalizer fold index differs")
        fold = FOLDS[self.fold_index]
        if self.train_ids != fold.train_ids or self.test_ids != fold.test_ids:
            raise PreparationValidationError("normalizer fold membership differs")
        expected_rows = sum(MATCHED_COUNTS[video_id] for video_id in fold.train_ids)
        if type(self.train_rows) is not int or self.train_rows != expected_rows:
            raise PreparationValidationError("normalizer training-row count differs")
        mean = _readonly_float64_vector(self.mean, "normalizer mean")
        scale = _readonly_float64_vector(self.scale, "normalizer scale")
        if np.any(scale < SCALE_FLOOR):
            raise PreparationValidationError("normalizer scale is below the frozen floor")
        expected = mm007_normalizer_fingerprint(mean, scale)
        if self.fingerprint != expected:
            raise PreparationValidationError("normalizer fingerprint does not replay")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def apply_frame(self, frame_uint8: np.ndarray) -> np.ndarray:
        frame = _frame_chw01(frame_uint8)
        normalized = (np.asarray(frame, dtype=np.float64) - self.mean[:, None, None]) / self.scale[:, None, None]
        return records.immutable_array(np.ascontiguousarray(normalized, dtype="<f8"))


PARENT_FRAME_SCHEMA: Final[Mapping[str, records.ArraySpec]] = MappingProxyType(
    {
        "video_ids": records.ArraySpec(np.dtype("<U11"), (RAW_ROWS,)),
        "timestamps": records.ArraySpec(np.dtype("<f8"), (RAW_ROWS,)),
        "frames_uint8": records.ArraySpec(np.dtype("u1"), (RAW_ROWS, NATIVE_SIZE, NATIVE_SIZE, CHANNELS)),
    }
)

SOURCE_ROW_SCHEMA: Final[Mapping[str, records.ArraySpec]] = MappingProxyType(
    {
        "current": records.ArraySpec(np.dtype("<f8"), (CHANNELS, NATIVE_SIZE, NATIVE_SIZE)),
        "current_timestamp": records.ArraySpec(np.dtype("<f8"), (1,)),
        "fold_index": records.ArraySpec(np.dtype("<i8"), (1,)),
        "normalizer_fingerprint": records.ArraySpec(np.dtype("<U64"), (1,)),
        "normalizer_mean": records.ArraySpec(np.dtype("<f8"), (CHANNELS,)),
        "normalizer_scale": records.ArraySpec(np.dtype("<f8"), (CHANNELS,)),
        "previous": records.ArraySpec(np.dtype("<f8"), (CHANNELS, NATIVE_SIZE, NATIVE_SIZE)),
        "previous_timestamp": records.ArraySpec(np.dtype("<f8"), (1,)),
        "row_ordinal": records.ArraySpec(np.dtype("<i8"), (1,)),
        "shuffle_row_ordinal": records.ArraySpec(np.dtype("<i8"), (1,)),
        "shuffled_previous": records.ArraySpec(np.dtype("<f8"), (CHANNELS, NATIVE_SIZE, NATIVE_SIZE)),
        "video_id": records.ArraySpec(np.dtype("<U11"), (1,)),
    }
)

TARGET_ROW_SCHEMA: Final[Mapping[str, records.ArraySpec]] = MappingProxyType(
    {
        "current_timestamp": records.ArraySpec(np.dtype("<f8"), (1,)),
        "deranged_future": records.ArraySpec(np.dtype("<f8"), (CHANNELS, NATIVE_SIZE, NATIVE_SIZE)),
        "derangement_row_ordinal": records.ArraySpec(np.dtype("<i8"), (1,)),
        "future": records.ArraySpec(np.dtype("<f8"), (CHANNELS, NATIVE_SIZE, NATIVE_SIZE)),
        "future_timestamp": records.ArraySpec(np.dtype("<f8"), (1,)),
        "row_ordinal": records.ArraySpec(np.dtype("<i8"), (1,)),
        "video_id": records.ArraySpec(np.dtype("<U11"), (1,)),
    }
)


def expected_raw_identities() -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([video_id for video_id in VIDEO_IDS for _ in range(RAW_COUNTS[video_id])], dtype="<U11")
    times = np.asarray(
        [1.0 + 0.5 * index for video_id in VIDEO_IDS for index in range(RAW_COUNTS[video_id])],
        dtype="<f8",
    )
    return records.immutable_array(ids), records.immutable_array(times)


def validate_raw_identities(video_ids: np.ndarray, timestamps: np.ndarray) -> None:
    if not isinstance(video_ids, np.ndarray) or not isinstance(timestamps, np.ndarray):
        raise PreparationValidationError("raw identities must be NumPy arrays")
    if (
        video_ids.dtype != np.dtype("<U11")
        or timestamps.dtype != np.dtype("<f8")
        or video_ids.shape != (RAW_ROWS,)
        or timestamps.shape != (RAW_ROWS,)
        or not video_ids.flags.c_contiguous
        or not timestamps.flags.c_contiguous
        or not np.all(np.isfinite(timestamps))
    ):
        raise PreparationValidationError("raw identity schema differs from the frozen 477-row archive")
    expected_ids, expected_times = expected_raw_identities()
    if not np.array_equal(video_ids, expected_ids) or not np.array_equal(timestamps, expected_times):
        raise PreparationValidationError("raw 477 identities differ from the frozen order")


def row_identity_sha256(rows: Sequence[RowIndex]) -> str:
    identities: records.JsonValue = [[row.video_id, row.current_timestamp] for row in rows]
    return hashlib.sha256(records.canonical_json_bytes(identities)).hexdigest()


def build_row_index(video_ids: np.ndarray, timestamps: np.ndarray) -> tuple[RowIndex, ...]:
    """Build the exact 453 causal triples without reading any frame value."""

    validate_raw_identities(video_ids, timestamps)
    output: list[RowIndex] = []
    for video_position, video_id in enumerate(VIDEO_IDS):
        indices = np.flatnonzero(video_ids == video_id)
        ordered = indices[np.argsort(timestamps[indices], kind="stable")]
        if len(ordered) != RAW_COUNTS[video_id]:
            raise PreparationValidationError("raw per-video row count differs")
        for video_row, position in enumerate(range(1, len(ordered) - 2)):
            previous_index = int(ordered[position - 1])
            current_index = int(ordered[position])
            future_index = int(ordered[position + 1])
            output.append(
                RowIndex(
                    ordinal=len(output),
                    video_id=video_id,
                    fold_index=video_position // 2,
                    video_row=video_row,
                    previous_index=previous_index,
                    current_index=current_index,
                    future_index=future_index,
                    previous_timestamp=float(timestamps[previous_index]),
                    current_timestamp=float(timestamps[current_index]),
                    future_timestamp=float(timestamps[future_index]),
                )
            )
    rows = tuple(output)
    counts = Counter(row.video_id for row in rows)
    if len(rows) != MATCHED_ROWS or dict(counts) != dict(MATCHED_COUNTS):
        raise PreparationValidationError("matched row count or per-video membership differs")
    if row_identity_sha256(rows) != MATCHED_IDENTITY_SHA256:
        raise PreparationValidationError("matched 453-row identity hash differs")
    return rows


def canonical_row_index() -> tuple[RowIndex, ...]:
    return build_row_index(*expected_raw_identities())


def _validate_rows(rows: Sequence[RowIndex]) -> tuple[RowIndex, ...]:
    if (
        not isinstance(rows, Sequence)
        or len(rows) != MATCHED_ROWS
        or any(not isinstance(row, RowIndex) for row in rows)
    ):
        raise PreparationValidationError("row index must contain exactly 453 RowIndex records")
    candidate = tuple(rows)
    canonical = canonical_row_index()
    if candidate != canonical:
        raise PreparationValidationError("row index differs from the frozen canonical construction")
    return candidate


def half_cycle_derangement(rows: Sequence[RowIndex]) -> np.ndarray:
    """Return within-video ``(i + ceil(n/2)) mod n`` row ordinals."""

    checked = _validate_rows(rows)
    mapping = _expected_half_cycle(checked)
    validate_derangement(mapping, checked)
    return records.immutable_array(mapping)


def _expected_half_cycle(rows: Sequence[RowIndex]) -> np.ndarray:
    mapping = np.empty(MATCHED_ROWS, dtype="<i8")
    for video_id in VIDEO_IDS:
        ordinals = np.asarray([row.ordinal for row in rows if row.video_id == video_id], dtype="<i8")
        shift = (len(ordinals) + 1) // 2
        for local_index, ordinal in enumerate(ordinals.tolist()):
            mapping[ordinal] = ordinals[(local_index + shift) % len(ordinals)]
    return mapping


def validate_derangement(mapping: np.ndarray, rows: Sequence[RowIndex]) -> None:
    checked = _validate_rows(rows)
    if (
        not isinstance(mapping, np.ndarray)
        or mapping.dtype != np.dtype("<i8")
        or mapping.shape != (MATCHED_ROWS,)
        or not mapping.flags.c_contiguous
    ):
        raise PreparationValidationError("half-cycle mapping schema differs")
    identity = np.arange(MATCHED_ROWS, dtype="<i8")
    if (
        np.any(mapping == identity)
        or np.any(mapping < 0)
        or np.any(mapping >= MATCHED_ROWS)
        or sorted(mapping.tolist()) != identity.tolist()
        or any(checked[index].video_id != checked[int(mapped)].video_id for index, mapped in enumerate(mapping))
    ):
        raise PreparationValidationError("half-cycle mapping is not a fixed-point-free within-video permutation")
    if not np.array_equal(mapping, _expected_half_cycle(checked)):
        raise PreparationValidationError("half-cycle mapping differs from the frozen ceil-shift permutation")


def inverse_derangement(mapping: np.ndarray, rows: Sequence[RowIndex]) -> np.ndarray:
    validate_derangement(mapping, rows)
    inverse = np.empty_like(mapping)
    inverse[mapping] = np.arange(MATCHED_ROWS, dtype="<i8")
    if not np.array_equal(mapping[inverse], np.arange(MATCHED_ROWS, dtype="<i8")):
        raise PreparationValidationError("half-cycle inverse does not replay")
    return records.immutable_array(inverse)


def validate_frames_uint8(frames_uint8: np.ndarray) -> None:
    if (
        not isinstance(frames_uint8, np.ndarray)
        or frames_uint8.dtype != np.dtype("u1")
        or frames_uint8.shape != (RAW_ROWS, NATIVE_SIZE, NATIVE_SIZE, CHANNELS)
        or not frames_uint8.flags.c_contiguous
    ):
        raise PreparationValidationError("frames_uint8 must be C-contiguous uint8 [477,64,64,3]")


def validate_parent_frame_arrays(arrays: Mapping[str, np.ndarray], *, require_pins: bool = True) -> None:
    checked = records.validate_arrays(arrays, PARENT_FRAME_SCHEMA, label="MM-007 frame archive")
    validate_raw_identities(checked["video_ids"], checked["timestamps"])
    validate_frames_uint8(checked["frames_uint8"])
    if require_pins:
        observed = {name: records.legacy_mm007_array_sha256(checked[name]) for name in sorted(PARENT_FRAME_SCHEMA)}
        if observed != dict(records.PARENT_FRAME_ARRAY_SHA256):
            raise PreparationValidationError("MM-007 frame-array pins differ")


def _pool_r8(frames_uint8: np.ndarray) -> np.ndarray:
    if frames_uint8.ndim != 4 or frames_uint8.shape[1:] != (NATIVE_SIZE, NATIVE_SIZE, CHANNELS):
        raise PreparationValidationError("R8 pool input shape differs")
    values = frames_uint8.astype(np.float32) / np.float32(255.0)
    values = values.reshape(len(values), R8_SIZE, R8_BLOCK, R8_SIZE, R8_BLOCK, CHANNELS)
    pooled = np.asarray(np.mean(values, axis=(2, 4), dtype=np.float64), dtype=np.float32)
    return np.ascontiguousarray(np.transpose(pooled, (0, 3, 1, 2)), dtype=np.float32)


def _frame_chw01(frame_uint8: np.ndarray) -> np.ndarray:
    if (
        not isinstance(frame_uint8, np.ndarray)
        or frame_uint8.dtype != np.dtype("u1")
        or frame_uint8.shape != (NATIVE_SIZE, NATIVE_SIZE, CHANNELS)
        or not frame_uint8.flags.c_contiguous
    ):
        raise PreparationValidationError("one frame must be C-contiguous uint8 [64,64,3]")
    values = frame_uint8.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(np.transpose(values, (2, 0, 1)), dtype=np.float32)


def fit_fold_normalizers(
    frames_uint8: np.ndarray, rows: Sequence[RowIndex] | None = None
) -> tuple[FoldNormalizer, ...]:
    """Fit four R8 normalizers from training-video current frames only."""

    validate_frames_uint8(frames_uint8)
    checked = _validate_rows(canonical_row_index() if rows is None else rows)
    output: list[FoldNormalizer] = []
    for fold in FOLDS:
        current_indices = np.asarray(
            [row.current_index for row in checked if row.video_id in fold.train_ids], dtype=np.intp
        )
        pooled = _pool_r8(frames_uint8[current_indices])
        values = np.asarray(pooled, dtype=np.float64)
        mean = np.asarray(np.mean(values, axis=(0, 2, 3), dtype=np.float64), dtype="<f8")
        scale = np.asarray(np.maximum(np.std(values, axis=(0, 2, 3), dtype=np.float64), SCALE_FLOOR), dtype="<f8")
        output.append(
            FoldNormalizer(
                fold.index,
                fold.train_ids,
                fold.test_ids,
                len(current_indices),
                mean,
                scale,
                mm007_normalizer_fingerprint(mean, scale),
            )
        )
    return tuple(output)


def _normalizer_for_fold(normalizers: Sequence[FoldNormalizer], fold_index: int) -> FoldNormalizer:
    if len(normalizers) != 4 or any(not isinstance(value, FoldNormalizer) for value in normalizers):
        raise PreparationValidationError("exactly four fold normalizers are required")
    by_fold = {value.fold_index: value for value in normalizers}
    if set(by_fold) != set(range(4)):
        raise PreparationValidationError("fold normalizer membership differs")
    return by_fold[fold_index]


def construct_source_row(
    frames_uint8: np.ndarray,
    ordinal: int,
    rows: Sequence[RowIndex],
    normalizers: Sequence[FoldNormalizer],
    derangement: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Construct one predictor-visible row with no future value or future index."""

    validate_frames_uint8(frames_uint8)
    checked = _validate_rows(rows)
    validate_derangement(derangement, checked)
    if type(ordinal) is not int or ordinal not in range(MATCHED_ROWS):
        raise PreparationValidationError("source row ordinal differs")
    row = checked[ordinal]
    shuffled = checked[int(derangement[ordinal])]
    normalizer = _normalizer_for_fold(normalizers, row.fold_index)
    arrays = {
        "current": normalizer.apply_frame(frames_uint8[row.current_index]),
        "current_timestamp": np.asarray([row.current_timestamp], dtype="<f8"),
        "fold_index": np.asarray([row.fold_index], dtype="<i8"),
        "normalizer_fingerprint": np.asarray([normalizer.fingerprint], dtype="<U64"),
        "normalizer_mean": normalizer.mean,
        "normalizer_scale": normalizer.scale,
        "previous": normalizer.apply_frame(frames_uint8[row.previous_index]),
        "previous_timestamp": np.asarray([row.previous_timestamp], dtype="<f8"),
        "row_ordinal": np.asarray([row.ordinal], dtype="<i8"),
        "shuffle_row_ordinal": np.asarray([shuffled.ordinal], dtype="<i8"),
        "shuffled_previous": normalizer.apply_frame(frames_uint8[shuffled.previous_index]),
        "video_id": np.asarray([row.video_id], dtype="<U11"),
    }
    return validate_source_row_arrays(arrays)


def construct_target_row(
    frames_uint8: np.ndarray,
    ordinal: int,
    rows: Sequence[RowIndex],
    normalizers: Sequence[FoldNormalizer],
    derangement: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Construct one scorer-only target row, detached from predictor inputs."""

    validate_frames_uint8(frames_uint8)
    checked = _validate_rows(rows)
    validate_derangement(derangement, checked)
    if type(ordinal) is not int or ordinal not in range(MATCHED_ROWS):
        raise PreparationValidationError("target row ordinal differs")
    row = checked[ordinal]
    deranged = checked[int(derangement[ordinal])]
    normalizer = _normalizer_for_fold(normalizers, row.fold_index)
    arrays = {
        "current_timestamp": np.asarray([row.current_timestamp], dtype="<f8"),
        "deranged_future": normalizer.apply_frame(frames_uint8[deranged.future_index]),
        "derangement_row_ordinal": np.asarray([deranged.ordinal], dtype="<i8"),
        "future": normalizer.apply_frame(frames_uint8[row.future_index]),
        "future_timestamp": np.asarray([row.future_timestamp], dtype="<f8"),
        "row_ordinal": np.asarray([row.ordinal], dtype="<i8"),
        "video_id": np.asarray([row.video_id], dtype="<U11"),
    }
    return validate_target_row_arrays(arrays)


def validate_source_row_arrays(arrays: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    checked = records.validate_arrays(arrays, SOURCE_ROW_SCHEMA, label="MM-009 source row")
    if any("future" in name or "target" in name for name in checked):
        raise PreparationValidationError("a predictor-visible source row names future data")
    ordinal = int(checked["row_ordinal"][0])
    shuffle = int(checked["shuffle_row_ordinal"][0])
    if ordinal not in range(MATCHED_ROWS) or shuffle not in range(MATCHED_ROWS):
        raise PreparationValidationError("source row ordinal is outside the panel")
    rows = canonical_row_index()
    row = rows[ordinal]
    mapped = rows[shuffle]
    expected_shuffle = int(_expected_half_cycle(rows)[ordinal])
    if (
        str(checked["video_id"][0]) != row.video_id
        or int(checked["fold_index"][0]) != row.fold_index
        or float(checked["previous_timestamp"][0]) != row.previous_timestamp
        or float(checked["current_timestamp"][0]) != row.current_timestamp
        or mapped.video_id != row.video_id
        or shuffle != expected_shuffle
    ):
        raise PreparationValidationError("source row identity or shuffle membership differs")
    mean = checked["normalizer_mean"]
    scale = checked["normalizer_scale"]
    if np.any(scale < SCALE_FLOOR) or str(checked["normalizer_fingerprint"][0]) != mm007_normalizer_fingerprint(
        mean, scale
    ):
        raise PreparationValidationError("source row normalizer does not replay")
    return checked


def validate_target_row_arrays(arrays: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    checked = records.validate_arrays(arrays, TARGET_ROW_SCHEMA, label="MM-009 target row")
    ordinal = int(checked["row_ordinal"][0])
    mapped_ordinal = int(checked["derangement_row_ordinal"][0])
    if ordinal not in range(MATCHED_ROWS) or mapped_ordinal not in range(MATCHED_ROWS):
        raise PreparationValidationError("target row ordinal is outside the panel")
    rows = canonical_row_index()
    row = rows[ordinal]
    mapped = rows[mapped_ordinal]
    expected_derangement = int(_expected_half_cycle(rows)[ordinal])
    if (
        str(checked["video_id"][0]) != row.video_id
        or float(checked["current_timestamp"][0]) != row.current_timestamp
        or float(checked["future_timestamp"][0]) != row.future_timestamp
        or mapped.video_id != row.video_id
        or mapped_ordinal != expected_derangement
    ):
        raise PreparationValidationError("target row identity or derangement membership differs")
    return checked


def validate_detached_pair(
    source: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    checked_source = validate_source_row_arrays(source)
    checked_target = validate_target_row_arrays(target)
    if (
        int(checked_source["row_ordinal"][0]) != int(checked_target["row_ordinal"][0])
        or str(checked_source["video_id"][0]) != str(checked_target["video_id"][0])
        or float(checked_source["current_timestamp"][0]) != float(checked_target["current_timestamp"][0])
    ):
        raise PreparationValidationError("detached source/target row identities differ")
    return checked_source, checked_target


def source_row_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    checked = validate_source_row_arrays(arrays)
    return records.canonical_npz_bytes(checked, SOURCE_ROW_SCHEMA, label="MM-009 source row")


def target_row_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    checked = validate_target_row_arrays(arrays)
    return records.canonical_npz_bytes(checked, TARGET_ROW_SCHEMA, label="MM-009 target row")


def load_source_row_npz(path: str | Path) -> Mapping[str, np.ndarray]:
    return validate_source_row_arrays(records.load_npz_file(path, SOURCE_ROW_SCHEMA, label="MM-009 source row"))


def load_target_row_npz(path: str | Path) -> Mapping[str, np.ndarray]:
    return validate_target_row_arrays(records.load_npz_file(path, TARGET_ROW_SCHEMA, label="MM-009 target row"))


def source_row_manifest(arrays: Mapping[str, np.ndarray], *, protocol_sha256: str) -> dict[str, records.JsonValue]:
    checked = validate_source_row_arrays(arrays)
    bound_protocol = records.require_sha256(protocol_sha256, "frozen protocol SHA-256")
    return {
        "arrays": {
            name: {
                "dtype": checked[name].dtype.str,
                "shape": list(checked[name].shape),
                "sha256": records.scientific_array_sha256(
                    f"source:{name}", checked[name], protocol_sha256=bound_protocol
                ),
            }
            for name in sorted(checked)
        },
        "protocol_sha256": bound_protocol,
        "row_ordinal": int(checked["row_ordinal"][0]),
        "schema_version": "mm009-source-row-manifest-v1",
    }


def target_row_manifest(arrays: Mapping[str, np.ndarray], *, protocol_sha256: str) -> dict[str, records.JsonValue]:
    checked = validate_target_row_arrays(arrays)
    bound_protocol = records.require_sha256(protocol_sha256, "frozen protocol SHA-256")
    return {
        "arrays": {
            name: {
                "dtype": checked[name].dtype.str,
                "shape": list(checked[name].shape),
                "sha256": records.scientific_array_sha256(
                    f"target:{name}", checked[name], protocol_sha256=bound_protocol
                ),
            }
            for name in sorted(checked)
        },
        "protocol_sha256": bound_protocol,
        "row_ordinal": int(checked["row_ordinal"][0]),
        "schema_version": "mm009-target-row-manifest-v1",
    }


def write_source_row_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, records.JsonValue]:
    checked = validate_source_row_arrays(arrays)
    return records.write_immutable_npz_exclusive(path, checked, SOURCE_ROW_SCHEMA, label="MM-009 source row")


def write_target_row_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, records.JsonValue]:
    checked = validate_target_row_arrays(arrays)
    return records.write_immutable_npz_exclusive(path, checked, TARGET_ROW_SCHEMA, label="MM-009 target row")


__all__ = [
    "CHANNELS",
    "FOLDS",
    "FoldNormalizer",
    "FoldSpec",
    "MATCHED_COUNTS",
    "MATCHED_IDENTITY_SHA256",
    "MATCHED_ROWS",
    "NATIVE_SIZE",
    "PARENT_FRAME_SCHEMA",
    "PROTOCOL_SHA256",
    "PreparationValidationError",
    "RAW_COUNTS",
    "RAW_ROWS",
    "RowIndex",
    "SCALE_FLOOR",
    "SCHEMA_VERSION",
    "SOURCE_ROW_SCHEMA",
    "TARGET_ROW_SCHEMA",
    "VIDEO_IDS",
    "build_row_index",
    "canonical_row_index",
    "construct_source_row",
    "construct_target_row",
    "expected_raw_identities",
    "fit_fold_normalizers",
    "half_cycle_derangement",
    "inverse_derangement",
    "load_source_row_npz",
    "load_target_row_npz",
    "mm007_normalizer_fingerprint",
    "row_identity_sha256",
    "source_row_manifest",
    "source_row_npz_bytes",
    "target_row_manifest",
    "target_row_npz_bytes",
    "validate_derangement",
    "validate_detached_pair",
    "validate_frames_uint8",
    "validate_parent_frame_arrays",
    "validate_raw_identities",
    "validate_source_row_arrays",
    "validate_target_row_arrays",
    "write_source_row_npz",
    "write_target_row_npz",
]
