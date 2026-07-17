"""Pure calibration and scoring primitives for MM-008 v2.2.

This module is intentionally free of filesystem, lifecycle, random-generator, and
model-fitting code.  In particular, it contains no v2.1 F/R/Q-envelope concepts:
affine and combined each expose the single exact-global prediction ``G``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final, Literal, cast

import numpy as np

PROTOCOL_SHA256: Final = (
    "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
)
SCHEMA_VERSION: Final = "mm008-v2.2-calibration-v1"

SYNTHETIC_ROWS: Final = 6
CHANNELS: Final = 3
NATIVE_SIZE: Final = 64
POOLED_SIZE: Final = 8
POOL_BLOCK: Final = 8
CENTRAL_START: Final = 8
CENTRAL_STOP: Final = 56
SCALE_FLOOR: Final = 1e-6
BROADBAND_RANK: Final = SYNTHETIC_ROWS * CHANNELS
MIN_SINGULAR_RATIO: Final = 0.50
MIN_CENTRAL_VARIANCE: Final = 0.50
MAX_ABS_LAG_CORRELATION: Final = 0.10

PAIRING_FACTOR: Final = 1.10
PERFORMANCE_FACTOR: Final = 1.25
STRONG_FACTOR: Final = 2.0
ENDPOINT_TOLERANCE: Final = 1e-10

SupportArm = Literal[
    "global_translation",
    "quadrant_translation",
    "affine",
    "appearance",
    "combined",
]
FactorialArm = Literal["affine", "appearance", "combined"]


class CalibrationV22Error(ValueError):
    """Raised when an MM-008 v2.2 calibration value is invalid."""


def _finite_r64(value: np.ndarray, name: str, *, rows: int | None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64, order="C")
    if array.ndim != 4 or array.shape[1:] != (CHANNELS, NATIVE_SIZE, NATIVE_SIZE):
        raise CalibrationV22Error(f"{name} must have shape [N,3,64,64]")
    if rows is not None and len(array) != rows:
        raise CalibrationV22Error(f"{name} must contain exactly {rows} rows")
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise CalibrationV22Error(f"{name} must be nonempty and finite")
    return array


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, dtype="<f8", order="C", copy=True)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8")
    return immutable.reshape(contiguous.shape)


def area_pool_r8(value: np.ndarray) -> np.ndarray:
    """Area-pool finite R64 arrays to R8 with frozen float64 reductions."""

    array = _finite_r64(value, "R64 grid", rows=None)
    blocked = array.reshape(
        len(array), CHANNELS, POOLED_SIZE, POOL_BLOCK, POOLED_SIZE, POOL_BLOCK
    )
    pooled = np.asarray(np.mean(blocked, axis=(3, 5), dtype=np.float64), dtype=np.float64)
    return _readonly_float64(pooled)


@dataclass(frozen=True, slots=True)
class SourceOnlyNormalizer:
    """Current-only R8 channel normalizer used for R64 scientific arrays."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = _readonly_float64(self.mean)
        scale = _readonly_float64(self.scale)
        if mean.shape != (1, CHANNELS, 1, 1) or scale.shape != mean.shape:
            raise CalibrationV22Error("normalizer arrays must have shape [1,3,1,1]")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise CalibrationV22Error("normalizer arrays must be finite")
        if np.any(scale < SCALE_FLOOR):
            raise CalibrationV22Error("normalizer scale is below the frozen floor")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def apply(self, value: np.ndarray) -> np.ndarray:
        array = _finite_r64(value, "normalizer input", rows=None)
        result = np.asarray((array - self.mean) / self.scale, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise CalibrationV22Error("normalizer application produced nonfinite values")
        return _readonly_float64(result)

    def invert(self, value: np.ndarray) -> np.ndarray:
        array = _finite_r64(value, "normalized input", rows=None)
        result = np.asarray(array * self.scale + self.mean, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise CalibrationV22Error("normalizer inversion produced nonfinite values")
        return _readonly_float64(result)


def fit_source_only_normalizer(raw_source: np.ndarray) -> SourceOnlyNormalizer:
    """Fit the frozen channel statistics from exactly one six-row source bank."""

    source = _finite_r64(raw_source, "raw_source", rows=SYNTHETIC_ROWS)
    pooled = area_pool_r8(source)
    mean = np.mean(pooled, axis=(0, 2, 3), keepdims=True, dtype=np.float64)
    scale = np.maximum(
        np.std(pooled, axis=(0, 2, 3), keepdims=True, dtype=np.float64),
        SCALE_FLOOR,
    )
    return SourceOnlyNormalizer(
        np.asarray(mean, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class BroadbandMetrics:
    """Complete evidence for the frozen six-row broadband validity gate."""

    row_rms: tuple[float, ...]
    matrix_rank: int
    singular_value_ratio: float
    central_variance: tuple[float, ...]
    lag_correlation: tuple[float, ...]
    lag_denominator_positive: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.row_rms) != BROADBAND_RANK:
            raise CalibrationV22Error("row_rms must contain 18 values")
        if len(self.central_variance) != BROADBAND_RANK:
            raise CalibrationV22Error("central_variance must contain 18 values")
        if len(self.lag_correlation) != 2 * BROADBAND_RANK:
            raise CalibrationV22Error("lag_correlation must contain 36 values")
        if len(self.lag_denominator_positive) != 2 * BROADBAND_RANK or not all(
            type(value) is bool for value in self.lag_denominator_positive
        ):
            raise CalibrationV22Error("lag denominator evidence must contain 36 booleans")
        if type(self.matrix_rank) is not int or not 0 <= self.matrix_rank <= BROADBAND_RANK:
            raise CalibrationV22Error("matrix_rank is invalid")
        numeric = (
            *self.row_rms,
            self.singular_value_ratio,
            *self.central_variance,
            *self.lag_correlation,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise CalibrationV22Error("broadband evidence must be finite")
        if any(value < 0.0 for value in self.row_rms + self.central_variance):
            raise CalibrationV22Error("RMS and variance evidence must be nonnegative")
        if self.singular_value_ratio < 0.0:
            raise CalibrationV22Error("singular-value ratio must be nonnegative")

    def failure_reasons(self) -> tuple[str, ...]:
        failures: list[str] = []
        if not all(value > 0.0 for value in self.row_rms):
            failures.append("nonpositive_row_rms")
        if self.matrix_rank != BROADBAND_RANK:
            failures.append("matrix_rank")
        if not self.singular_value_ratio > MIN_SINGULAR_RATIO:
            failures.append("singular_value_ratio")
        if not all(value > MIN_CENTRAL_VARIANCE for value in self.central_variance):
            failures.append("central_variance")
        if not all(self.lag_denominator_positive):
            failures.append("lag_denominator")
        if not all(abs(value) < MAX_ABS_LAG_CORRELATION for value in self.lag_correlation):
            failures.append("lag_correlation")
        return tuple(failures)

    @property
    def valid(self) -> bool:
        return not self.failure_reasons()


def broadband_validity_metrics(normalized_base: np.ndarray) -> BroadbandMetrics:
    """Compute the exact rank, variance, and flattened adjacent-lag checks."""

    base = _finite_r64(normalized_base, "normalized_base", rows=SYNTHETIC_ROWS)
    rows = base.reshape(BROADBAND_RANK, NATIVE_SIZE * NATIVE_SIZE)
    centered = rows - np.mean(rows, axis=1, keepdims=True, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        row_rms = np.sqrt(np.mean(centered * centered, axis=1, dtype=np.float64))
    if not np.all(np.isfinite(row_rms)):
        raise CalibrationV22Error("row RMS computation was nonfinite")
    scaled = np.zeros_like(centered)
    np.divide(centered, row_rms[:, None], out=scaled, where=(row_rms > 0.0)[:, None])
    try:
        matrix_rank = int(np.linalg.matrix_rank(scaled))
        singular_values = np.linalg.svd(scaled, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise CalibrationV22Error("broadband singular-value computation failed") from error
    largest = float(singular_values[0])
    ratio = float(singular_values[-1] / largest) if largest > 0.0 else 0.0

    central = base[:, :, CENTRAL_START:CENTRAL_STOP, CENTRAL_START:CENTRAL_STOP]
    with np.errstate(over="ignore", invalid="ignore"):
        central_variance = np.var(central, axis=(2, 3), dtype=np.float64).reshape(-1)
    if not math.isfinite(ratio) or not np.all(np.isfinite(central_variance)):
        raise CalibrationV22Error("rank or central-variance evidence was nonfinite")

    correlations: list[float] = []
    denominator_positive: list[bool] = []
    for grid in base.reshape(BROADBAND_RANK, NATIVE_SIZE, NATIVE_SIZE):
        for left, right in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
            a = left.reshape(-1)
            b = right.reshape(-1)
            a_centered = a - np.mean(a, dtype=np.float64)
            b_centered = b - np.mean(b, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                numerator = float(np.mean(a_centered * b_centered, dtype=np.float64))
                a_square = float(np.mean(a_centered * a_centered, dtype=np.float64))
                b_square = float(np.mean(b_centered * b_centered, dtype=np.float64))
            if not all(math.isfinite(value) for value in (numerator, a_square, b_square)):
                raise CalibrationV22Error("lag-correlation evidence was nonfinite")
            positive = a_square > 0.0 and b_square > 0.0
            denominator_positive.append(positive)
            correlation = numerator / math.sqrt(a_square * b_square) if positive else 0.0
            if not math.isfinite(correlation):
                raise CalibrationV22Error("lag correlation was nonfinite")
            correlations.append(correlation)

    return BroadbandMetrics(
        row_rms=tuple(float(value) for value in row_rms),
        matrix_rank=matrix_rank,
        singular_value_ratio=ratio,
        central_variance=tuple(float(value) for value in central_variance),
        lag_correlation=tuple(correlations),
        lag_denominator_positive=tuple(denominator_positive),
    )


def _finite_nonnegative(*values: object) -> tuple[float, ...] | None:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            return None
        result.append(number)
    return tuple(result)


def require_support_arm(arm: str) -> SupportArm:
    if arm not in {
        "global_translation",
        "quadrant_translation",
        "affine",
        "appearance",
        "combined",
    }:
        raise CalibrationV22Error("unknown MM-008 v2.2 support arm")
    return cast(SupportArm, arm)


def residual_pairing(
    ordered_mse: float,
    wrong_mse: float,
    true_bias_mse: float,
    wrong_bias_mse: float,
) -> bool:
    values = _finite_nonnegative(ordered_mse, wrong_mse, true_bias_mse, wrong_bias_mse)
    if values is None:
        return False
    ordered, wrong, true_bias, wrong_bias = values
    left = PAIRING_FACTOR * ordered * wrong_bias
    right = wrong * true_bias
    return (
        true_bias > 0.0
        and wrong_bias > 0.0
        and math.isfinite(left)
        and math.isfinite(right)
        and left <= right
    )


def pair_support(
    ordered_mse: float,
    near_mse: float,
    far_mse: float,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> bool:
    return residual_pairing(
        ordered_mse, near_mse, true_bias_mse, near_bias_mse
    ) and residual_pairing(ordered_mse, far_mse, true_bias_mse, far_bias_mse)


def performance_support(persistence_mse: float, ordered_mse: float) -> bool:
    values = _finite_nonnegative(persistence_mse, ordered_mse)
    return values is not None and values[0] > 0.0 and PERFORMANCE_FACTOR * values[1] <= values[0]


def beats_bias(ordered_mse: float, true_bias_mse: float) -> bool:
    values = _finite_nonnegative(ordered_mse, true_bias_mse)
    return values is not None and values[1] > 0.0 and PERFORMANCE_FACTOR * values[0] <= values[1]


def complete_support(
    arm: SupportArm,
    persistence_mse: float,
    ordered_mse: float,
    near_mse: float,
    far_mse: float,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> bool:
    checked = require_support_arm(arm)
    return (
        performance_support(persistence_mse, ordered_mse)
        and pair_support(
            ordered_mse,
            near_mse,
            far_mse,
            true_bias_mse,
            near_bias_mse,
            far_bias_mse,
        )
        and (checked not in {"appearance", "combined"} or beats_bias(ordered_mse, true_bias_mse))
    )


def strong_support(
    arm: SupportArm,
    persistence_mse: float,
    full_mse: float,
    ordered_mse: float,
    near_mse: float,
    far_mse: float,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
    *,
    endpoints_pass: bool,
) -> bool:
    checked = require_support_arm(arm)
    values = _finite_nonnegative(persistence_mse, full_mse, ordered_mse)
    return bool(
        values is not None
        and values[0] > 0.0
        and STRONG_FACTOR * values[1] <= values[0]
        and STRONG_FACTOR * values[2] <= values[0]
        and pair_support(
            ordered_mse,
            near_mse,
            far_mse,
            true_bias_mse,
            near_bias_mse,
            far_bias_mse,
        )
        and complete_support(
            checked,
            persistence_mse,
            ordered_mse,
            near_mse,
            far_mse,
            true_bias_mse,
            near_bias_mse,
            far_bias_mse,
        )
        and type(endpoints_pass) is bool
        and endpoints_pass
    )


def no_bias_gain(ordered_mse: float, true_bias_mse: float) -> bool:
    """Return the independent-null requirement ``1.25*o_m > b_T > 0``."""

    values = _finite_nonnegative(ordered_mse, true_bias_mse)
    return values is not None and values[1] > 0.0 and PERFORMANCE_FACTOR * values[0] > values[1]


def dominates(
    preferred_full_mse: float,
    comparator_full_mse: float,
    preferred_xfit_mse: float,
    comparator_xfit_mse: float,
) -> bool:
    values = _finite_nonnegative(
        preferred_full_mse,
        comparator_full_mse,
        preferred_xfit_mse,
        comparator_xfit_mse,
    )
    if values is None:
        return False
    preferred_full, comparator_full, preferred_xfit, comparator_xfit = values
    return (
        comparator_full > preferred_full
        and PERFORMANCE_FACTOR * preferred_full <= comparator_full
        and comparator_xfit > preferred_xfit
        and PERFORMANCE_FACTOR * preferred_xfit <= comparator_xfit
    )


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """A finite SSE/count/MSE triple with a recomputable denominator."""

    sse: float
    count: int
    mse: float

    def __post_init__(self) -> None:
        values = _finite_nonnegative(self.sse, self.mse)
        if values is None or type(self.count) is not int or self.count <= 0:
            raise CalibrationV22Error("error record is invalid")
        sse, mse = values
        if not math.isclose(mse, sse / self.count, rel_tol=1e-12, abs_tol=1e-12):
            raise CalibrationV22Error("error-record MSE does not reproduce from SSE/count")
        object.__setattr__(self, "sse", sse)
        object.__setattr__(self, "mse", mse)


@dataclass(frozen=True, slots=True)
class EndpointRecord:
    """One finite per-row, per-context endpoint error record."""

    sse: float
    count: int
    mse: float
    max_abs_error: float

    def __post_init__(self) -> None:
        score = ErrorRecord(self.sse, self.count, self.mse)
        maximum = _finite_nonnegative(self.max_abs_error)
        if maximum is None:
            raise CalibrationV22Error("endpoint maximum error is invalid")
        object.__setattr__(self, "sse", score.sse)
        object.__setattr__(self, "count", score.count)
        object.__setattr__(self, "mse", score.mse)
        object.__setattr__(self, "max_abs_error", maximum[0])

    def passes(self, tolerance: float = ENDPOINT_TOLERANCE) -> bool:
        threshold = _finite_nonnegative(tolerance)
        return threshold is not None and self.max_abs_error <= threshold[0]


def error_record(actual: np.ndarray, expected: np.ndarray) -> ErrorRecord:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise CalibrationV22Error("error arrays must have the same nonempty shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise CalibrationV22Error("error arrays must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
        sse = float(np.sum(difference * difference, dtype=np.float64))
    if not math.isfinite(sse):
        raise CalibrationV22Error("SSE is nonfinite")
    count = int(left.size)
    return ErrorRecord(sse, count, sse / count)


def endpoint_record(actual: np.ndarray, expected: np.ndarray) -> EndpointRecord:
    score = error_record(actual, expected)
    maximum = float(
        np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)))
    )
    if not math.isfinite(maximum):
        raise CalibrationV22Error("endpoint maximum error is nonfinite")
    return EndpointRecord(score.sse, score.count, score.mse, maximum)
