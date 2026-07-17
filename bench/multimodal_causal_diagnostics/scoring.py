"""Pure target scoring for frozen MM-009 central predictions.

This module has no fitting, dataset, filesystem, lifecycle, or decision authority.
It accepts already-frozen packed central predictions and detached scoring targets,
computes one untrimmed SSE per row, and aggregates those row primitives in canonical
row order.  Prediction/target custody and hash authentication remain caller duties.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, fields, is_dataclass
from typing import Final, Literal, cast

import numpy as np

PROTOCOL_SHA256: Final[str] = "ca39f7cea6a2a5b041956b419bf3530dd54eb8403096963a044d7fcf1e2121cc"
SCHEMA_VERSION: Final = "mm009-scoring-v1"

Family = Literal["affine", "appearance", "combined"]
FAMILIES: Final[tuple[Family, ...]] = ("affine", "appearance", "combined")
CHANNELS: Final = 3
CENTRAL_SITES: Final = 48 * 48
ELEMENTS_PER_ROW: Final = CHANNELS * CENTRAL_SITES
CENTRAL_SHAPE: Final = (CHANNELS, CENTRAL_SITES)

_FLOAT64_LE: Final = np.dtype("<f8")


class ScoringError(ValueError):
    """Raised when MM-009 scoring evidence violates the frozen contract."""


def _require_family(value: object) -> Family:
    if type(value) is not str or value not in FAMILIES:
        raise ScoringError("family must be exactly affine, appearance, or combined")
    return value


def _require_nonnegative_float(value: object, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ScoringError(f"{label} must be a built-in finite number")
    try:
        converted = float(cast(float, value))
    except (OverflowError, ValueError):
        raise ScoringError(f"{label} must be finite and nonnegative") from None
    if not math.isfinite(converted) or converted < 0.0:
        raise ScoringError(f"{label} must be finite and nonnegative")
    return converted


def _immutable_central(value: object, label: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise ScoringError(f"{label} must be an exact NumPy array")
    if value.shape != CENTRAL_SHAPE:
        raise ScoringError(f"{label} must have exact shape {CENTRAL_SHAPE}")
    if value.dtype != _FLOAT64_LE:
        raise ScoringError(f"{label} must have exact little-endian float64 dtype")
    if not value.flags.c_contiguous:
        raise ScoringError(f"{label} must be C-contiguous")
    if not bool(np.all(np.isfinite(value))):
        raise ScoringError(f"{label} contains a nonfinite value")
    payload = value.tobytes(order="C")
    return np.frombuffer(payload, dtype=_FLOAT64_LE).reshape(CENTRAL_SHAPE)


@dataclass(frozen=True, slots=True)
class ErrorPrimitive:
    """One untrimmed squared-error numerator and its exact scalar count."""

    sse: float
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sse", _require_nonnegative_float(self.sse, "error SSE"))
        if type(self.count) is not int or self.count <= 0:
            raise ScoringError("error count must be a positive built-in integer")
        if not math.isfinite(self.sse / self.count):
            raise ScoringError("derived error MSE is nonfinite")

    @property
    def mse(self) -> float:
        """Return the exact numerator divided by its positive scalar count."""

        return self.sse / self.count


@dataclass(frozen=True, slots=True)
class RowScoreInputs:
    """Detached targets and already-frozen predictions for one family and row."""

    video_id: str
    fold: int
    row_index: int
    family: Family
    current_target: np.ndarray
    future_target: np.ndarray
    deranged_future_target: np.ndarray
    history_identity: np.ndarray
    history_xfit: np.ndarray
    history_shuffle_xfit: np.ndarray
    persistence: np.ndarray
    forecast: np.ndarray
    forecast_shuffle: np.ndarray
    forecast_reverse: np.ndarray
    velocity: np.ndarray
    history_bias: np.ndarray
    forecast_bias: np.ndarray

    def __post_init__(self) -> None:
        if type(self.video_id) is not str or not self.video_id or not self.video_id.isascii():
            raise ScoringError("video ID must be a nonempty ASCII built-in string")
        if type(self.fold) is not int or self.fold not in (0, 1, 2, 3):
            raise ScoringError("fold must be a built-in integer in [0,3]")
        if type(self.row_index) is not int or self.row_index < 0:
            raise ScoringError("row index must be a nonnegative built-in integer")
        family = _require_family(self.family)
        object.__setattr__(self, "family", family)
        required_arrays = (
            "current_target",
            "future_target",
            "deranged_future_target",
            "history_identity",
            "history_xfit",
            "history_shuffle_xfit",
            "persistence",
            "forecast",
            "forecast_shuffle",
            "forecast_reverse",
            "velocity",
        )
        for name in required_arrays:
            object.__setattr__(self, name, _immutable_central(getattr(self, name), name))
        object.__setattr__(self, "history_bias", _immutable_central(self.history_bias, "history_bias"))
        object.__setattr__(self, "forecast_bias", _immutable_central(self.forecast_bias, "forecast_bias"))


@dataclass(frozen=True, slots=True)
class RowScores:
    """The exact named MM-009 error primitives for one scored row."""

    video_id: str
    fold: int
    row_index: int
    family: Family
    i: ErrorPrimitive
    a: ErrorPrimitive
    q: ErrorPrimitive
    p: ErrorPrimitive
    c: ErrorPrimitive
    h: ErrorPrimitive
    r: ErrorPrimitive
    z: ErrorPrimitive
    d: ErrorPrimitive
    pd: ErrorPrimitive
    u: ErrorPrimitive
    b: ErrorPrimitive
    bd: ErrorPrimitive

    def __post_init__(self) -> None:
        if type(self.video_id) is not str or not self.video_id or not self.video_id.isascii():
            raise ScoringError("row score video ID is invalid")
        if type(self.fold) is not int or self.fold not in (0, 1, 2, 3):
            raise ScoringError("row score fold is invalid")
        if type(self.row_index) is not int or self.row_index < 0:
            raise ScoringError("row score index is invalid")
        family = _require_family(self.family)
        object.__setattr__(self, "family", family)
        for name in ("i", "a", "q", "p", "c", "h", "r", "z", "d", "pd", "u", "b", "bd"):
            error = getattr(self, name)
            if type(error) is not ErrorPrimitive or error.count != ELEMENTS_PER_ROW:
                raise ScoringError(f"row metric {name} has the wrong type or scalar count")


@dataclass(frozen=True, slots=True)
class VideoScores:
    """Canonical row-SSE aggregation for one family and one support-unit video."""

    video_id: str
    fold: int
    family: Family
    row_count: int
    i: ErrorPrimitive
    a: ErrorPrimitive
    q: ErrorPrimitive
    p: ErrorPrimitive
    c: ErrorPrimitive
    h: ErrorPrimitive
    r: ErrorPrimitive
    z: ErrorPrimitive
    d: ErrorPrimitive
    pd: ErrorPrimitive
    u: ErrorPrimitive
    b: ErrorPrimitive
    bd: ErrorPrimitive

    def __post_init__(self) -> None:
        if type(self.video_id) is not str or not self.video_id or not self.video_id.isascii():
            raise ScoringError("video score ID is invalid")
        if type(self.fold) is not int or self.fold not in (0, 1, 2, 3):
            raise ScoringError("video score fold is invalid")
        family = _require_family(self.family)
        object.__setattr__(self, "family", family)
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ScoringError("video row count must be a positive built-in integer")
        expected = self.row_count * ELEMENTS_PER_ROW
        for name in ("i", "a", "q", "p", "c", "h", "r", "z", "d", "pd", "u", "b", "bd"):
            error = getattr(self, name)
            if type(error) is not ErrorPrimitive or error.count != expected:
                raise ScoringError(f"video metric {name} has the wrong type or scalar count")


def squared_error(prediction: np.ndarray, target: np.ndarray) -> ErrorPrimitive:
    """Compute one finite untrimmed central SSE without conversion or trimming."""

    predicted = _immutable_central(prediction, "prediction")
    expected = _immutable_central(target, "target")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = predicted - expected
        sse = float(np.sum(residual * residual, dtype=np.float64))
    if not math.isfinite(sse):
        raise ScoringError("untrimmed SSE overflowed or became nonfinite")
    return ErrorPrimitive(sse=sse, count=ELEMENTS_PER_ROW)


def score_row(inputs: RowScoreInputs) -> RowScores:
    """Score one detached future without exposing any fitting or selection API."""

    if type(inputs) is not RowScoreInputs:
        raise ScoringError("row scoring input must be an exact RowScoreInputs value")
    return RowScores(
        video_id=inputs.video_id,
        fold=inputs.fold,
        row_index=inputs.row_index,
        family=inputs.family,
        i=squared_error(inputs.history_identity, inputs.current_target),
        a=squared_error(inputs.history_xfit, inputs.current_target),
        q=squared_error(inputs.history_shuffle_xfit, inputs.current_target),
        p=squared_error(inputs.persistence, inputs.future_target),
        c=squared_error(inputs.forecast, inputs.future_target),
        h=squared_error(inputs.forecast_shuffle, inputs.future_target),
        r=squared_error(inputs.forecast_reverse, inputs.future_target),
        z=squared_error(inputs.velocity, inputs.future_target),
        d=squared_error(inputs.forecast, inputs.deranged_future_target),
        pd=squared_error(inputs.persistence, inputs.deranged_future_target),
        u=squared_error(inputs.history_bias, inputs.current_target),
        b=squared_error(inputs.forecast_bias, inputs.future_target),
        bd=squared_error(inputs.forecast_bias, inputs.deranged_future_target),
    )


def _metric(row: RowScores | VideoScores, name: str) -> ErrorPrimitive:
    value = getattr(row, name)
    if type(value) is not ErrorPrimitive:
        raise ScoringError(f"metric {name} is not an ErrorPrimitive")
    return value


def _sum_metric(rows: tuple[RowScores, ...], name: str) -> ErrorPrimitive:
    total_sse = 0.0
    total_count = 0
    for row in rows:
        error = _metric(row, name)
        total_sse += error.sse
        total_count += error.count
        if not math.isfinite(total_sse):
            raise ScoringError(f"video SSE accumulation for {name} overflowed")
    return ErrorPrimitive(total_sse, total_count)


def aggregate_video(rows: tuple[RowScores, ...]) -> VideoScores:
    """Aggregate canonical per-row SSEs for exactly one video/family support unit."""

    if type(rows) is not tuple or not rows or any(type(row) is not RowScores for row in rows):
        raise ScoringError("video aggregation requires a nonempty exact tuple of RowScores")
    first = rows[0]
    if tuple(row.row_index for row in rows) != tuple(range(len(rows))):
        raise ScoringError("video rows are missing, duplicated, or out of canonical order")
    if any((row.video_id, row.fold, row.family) != (first.video_id, first.fold, first.family) for row in rows):
        raise ScoringError("video aggregation mixed identities, folds, or families")
    return VideoScores(
        video_id=first.video_id,
        fold=first.fold,
        family=first.family,
        row_count=len(rows),
        i=_sum_metric(rows, "i"),
        a=_sum_metric(rows, "a"),
        q=_sum_metric(rows, "q"),
        p=_sum_metric(rows, "p"),
        c=_sum_metric(rows, "c"),
        h=_sum_metric(rows, "h"),
        r=_sum_metric(rows, "r"),
        z=_sum_metric(rows, "z"),
        d=_sum_metric(rows, "d"),
        pd=_sum_metric(rows, "pd"),
        u=_sum_metric(rows, "u"),
        b=_sum_metric(rows, "b"),
        bd=_sum_metric(rows, "bd"),
    )


def _deep_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return struct.pack("<d", left) == struct.pack("<d", cast(float, right))
    if isinstance(left, tuple):
        other = cast(tuple[object, ...], right)
        return len(left) == len(other) and all(_deep_equal(a, b) for a, b in zip(left, other, strict=True))
    if is_dataclass(left) and not isinstance(left, type):
        return all(_deep_equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left))
    return left == right


def validate_row_scores(inputs: RowScoreInputs, claimed: RowScores) -> None:
    """Recompute and bit-compare a claimed row score record."""

    if type(claimed) is not RowScores or not _deep_equal(score_row(inputs), claimed):
        raise ScoringError("claimed row scores differ from pure target recomputation")


def validate_video_scores(rows: tuple[RowScores, ...], claimed: VideoScores) -> None:
    """Reaggregate and bit-compare a claimed video score record."""

    if type(claimed) is not VideoScores or not _deep_equal(aggregate_video(rows), claimed):
        raise ScoringError("claimed video scores differ from canonical row aggregation")


__all__ = [
    "CENTRAL_SHAPE",
    "CENTRAL_SITES",
    "ELEMENTS_PER_ROW",
    "FAMILIES",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "ErrorPrimitive",
    "Family",
    "RowScoreInputs",
    "RowScores",
    "ScoringError",
    "VideoScores",
    "aggregate_video",
    "score_row",
    "squared_error",
    "validate_row_scores",
    "validate_video_scores",
]
