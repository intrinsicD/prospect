"""Frozen scalar-candidate fitting mathematics for MM-008 v2.2.

This module deliberately owns no sampling, grid enumeration, random generation,
filesystem access, or lifecycle behavior.  It consumes one already-sampled affine
candidate at a time and implements only the target extraction and reductions frozen
by the v2.2 protocol.

The central geometry is repeated here as literal protocol structure instead of
importing a predecessor implementation.  That keeps this leaf independent from the
retired v1/v2 modules and from construction order of the sibling ``geometry_v22``
module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-fitting-v1"
CHANNELS: Final = 3
NATIVE_SIZE: Final = 64
CENTRAL_START: Final = 8
CENTRAL_STOP: Final = 56
CENTRAL_SIZE: Final = CENTRAL_STOP - CENTRAL_START
CENTRAL_SITE_COUNT: Final = CENTRAL_SIZE * CENTRAL_SIZE
MACRO_SIDE: Final = 6
MACRO_PIXELS: Final = 8
MACRO_COUNT: Final = MACRO_SIDE * MACRO_SIDE
TRIM_FRACTION: Final = 0.25
VARIANCE_FLOOR: Final = 1e-6
GAIN_BOUNDS: Final = (-2.0, 4.0)
BIAS_BOUNDS: Final = (-4.0, 4.0)

MaskKind = Literal["full", "parity0", "parity1"]
CandidateArm = Literal["affine", "combined"]


class FittingV22Error(ValueError):
    """Raised when an input or derived fitting record violates the frozen schema."""


def _immutable_array(value: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    """Return a C-order array backed by immutable bytes."""

    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    return _immutable_array(np.asarray(value), np.dtype("<f8"))


def _immutable_bool(value: np.ndarray) -> np.ndarray:
    normalized = np.ascontiguousarray(value, dtype=np.uint8)
    if np.any((normalized != 0) & (normalized != 1)):
        raise FittingV22Error("boolean array contains a value other than zero or one")
    return np.frombuffer(normalized.tobytes(order="C"), dtype=np.bool_).reshape(normalized.shape)


def _build_central_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=np.int64),
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=np.int64),
        indexing="ij",
    )
    coords = np.stack((yy.reshape(-1), xx.reshape(-1)), axis=1)
    macro_y = ((coords[:, 0] - CENTRAL_START) // MACRO_PIXELS).astype(np.int64)
    macro_x = ((coords[:, 1] - CENTRAL_START) // MACRO_PIXELS).astype(np.int64)
    macro_ids = (macro_y * MACRO_SIDE + macro_x).astype(np.uint8)
    parities = ((macro_y + macro_x) % 2).astype(np.uint8)
    return (
        _immutable_array(coords, np.dtype("<i8")),
        _immutable_array(macro_ids, np.dtype("u1")),
        _immutable_array(parities, np.dtype("u1")),
    )


CENTRAL_COORDS, CENTRAL_MACRO_IDS, CENTRAL_PARITIES = _build_central_geometry()
_FULL_MASK: Final = _immutable_bool(np.ones(CENTRAL_SITE_COUNT, dtype=np.bool_))
_PARITY_MASKS: Final = (
    _immutable_bool(CENTRAL_PARITIES == 0),
    _immutable_bool(CENTRAL_PARITIES == 1),
)


def full_mask() -> np.ndarray:
    """Return the immutable frozen full-context mask."""

    return _FULL_MASK


def parity_mask(parity: int) -> np.ndarray:
    """Return the immutable mask for one physical-macrocell parity."""

    if parity not in (0, 1):
        raise FittingV22Error("parity must be exactly zero or one")
    return _PARITY_MASKS[parity]


def _require_float64_array(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FittingV22Error(f"{label} must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise FittingV22Error(f"{label} must have exact float64 dtype")
    if value.shape != shape:
        raise FittingV22Error(f"{label} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise FittingV22Error(f"{label} must be C-contiguous")
    if not bool(np.all(np.isfinite(value))):
        raise FittingV22Error(f"{label} contains a nonfinite value")
    return value


def _normalize_mask(mask: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise FittingV22Error(f"{label} must be a NumPy array")
    if mask.shape != (CENTRAL_SITE_COUNT,):
        raise FittingV22Error(f"{label} must cover exactly {CENTRAL_SITE_COUNT} central sites")
    if not mask.flags.c_contiguous:
        raise FittingV22Error(f"{label} must be C-contiguous")
    if mask.dtype == np.dtype(np.bool_):
        normalized = mask
    elif mask.dtype == np.dtype(np.uint8):
        if bool(np.any((mask != 0) & (mask != 1))):
            raise FittingV22Error(f"{label} uint8 values must be zero or one")
        normalized = mask.astype(np.bool_, copy=False)
    else:
        raise FittingV22Error(f"{label} must have bool or uint8 dtype")
    return _immutable_bool(normalized)


def _mask_kind(mask: np.ndarray, label: str = "fit mask") -> tuple[np.ndarray, MaskKind]:
    normalized = _normalize_mask(mask, label)
    if np.array_equal(normalized, _FULL_MASK):
        return normalized, "full"
    if np.array_equal(normalized, _PARITY_MASKS[0]):
        return normalized, "parity0"
    if np.array_equal(normalized, _PARITY_MASKS[1]):
        return normalized, "parity1"
    raise FittingV22Error(f"{label} must be exactly full, parity zero, or parity one")


def _validate_context_masks(fit_mask: np.ndarray, output_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fitted, kind = _mask_kind(fit_mask)
    output, output_kind = _mask_kind(output_mask, "output mask")
    if kind == "full":
        expected: MaskKind = "full"
    elif kind == "parity0":
        expected = "parity1"
    else:
        expected = "parity0"
    if output_kind != expected:
        raise FittingV22Error("output mask must be full for a full fit or the opposite parity for cross-fit")
    return fitted, output


def _fit_macro_ids(fit_mask: np.ndarray) -> np.ndarray:
    return np.asarray(CENTRAL_MACRO_IDS[np.asarray(fit_mask, dtype=np.bool_)], dtype=np.int64)


def extract_target(target: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Extract ``[channel,site]`` values in frozen central row-major order."""

    future = _require_float64_array(target, (CHANNELS, NATIVE_SIZE, NATIVE_SIZE), "target")
    mask, _ = _mask_kind(selected, "target mask")
    coords = CENTRAL_COORDS[mask]
    values = future[:, coords[:, 0], coords[:, 1]]
    return _immutable_float64(values)


def target_values(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Public integration name for exact central-site target extraction."""

    return extract_target(target, mask)


def _validate_fit_values(target: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, MaskKind]:
    mask, kind = _mask_kind(fit_mask)
    count = int(np.count_nonzero(mask))
    values = _require_float64_array(target, (CHANNELS, count), "fit target")
    return values, mask, kind


def _validate_candidate(
    sampled: np.ndarray, target: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MaskKind]:
    target_values, mask, kind = _validate_fit_values(target, fit_mask)
    count = target_values.shape[1]
    source = _require_float64_array(sampled, (1, CHANNELS, count), "sampled candidate")
    return source, target_values, mask, kind


def _validate_macro_id_tuple(values: tuple[int, ...], *, expected_count: int | None = None) -> None:
    if not isinstance(values, tuple) or any(type(value) is not int for value in values):
        raise FittingV22Error("macro IDs must be a tuple of built-in integers")
    if expected_count is not None and len(values) != expected_count:
        raise FittingV22Error(f"expected exactly {expected_count} retained macro IDs")
    if len(set(values)) != len(values) or any(value < 0 or value >= MACRO_COUNT for value in values):
        raise FittingV22Error("macro IDs must be unique members of the frozen 0..35 range")


@dataclass(frozen=True, slots=True)
class OLSFit:
    """One exact float64 OLS solution before and after independent clipping."""

    source_means: np.ndarray
    target_means: np.ndarray
    variances: np.ndarray
    covariances: np.ndarray
    raw_gains: np.ndarray
    raw_biases: np.ndarray
    gains: np.ndarray
    biases: np.ndarray

    def __post_init__(self) -> None:
        names = (
            "source_means",
            "target_means",
            "variances",
            "covariances",
            "raw_gains",
            "raw_biases",
            "gains",
            "biases",
        )
        for name in names:
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (CHANNELS,) or not bool(np.all(np.isfinite(value))):
                raise FittingV22Error(f"OLS field {name} must be a finite three-vector")
            object.__setattr__(self, name, _immutable_float64(value))
        if bool(np.any(self.variances < 0.0)):
            raise FittingV22Error("OLS variances must be nonnegative")
        expected_gains = self.covariances / np.maximum(self.variances, VARIANCE_FLOOR)
        expected_biases = self.target_means - expected_gains * self.source_means
        if not np.array_equal(self.raw_gains, expected_gains):
            raise FittingV22Error("raw OLS gains do not equal covariance over floored variance")
        if not np.array_equal(self.raw_biases, expected_biases):
            raise FittingV22Error("raw OLS biases were not computed from raw gains")
        if not np.array_equal(self.gains, np.clip(self.raw_gains, *GAIN_BOUNDS)):
            raise FittingV22Error("OLS gains are not independently clipped")
        if not np.array_equal(self.biases, np.clip(self.raw_biases, *BIAS_BOUNDS)):
            raise FittingV22Error("OLS biases are not independently clipped")


@dataclass(frozen=True, slots=True)
class MacroLosses:
    """RGB-averaged per-macro loss in ascending physical macro ID order."""

    macro_ids: tuple[int, ...]
    losses: np.ndarray

    def __post_init__(self) -> None:
        _validate_macro_id_tuple(self.macro_ids)
        if tuple(sorted(self.macro_ids)) != self.macro_ids:
            raise FittingV22Error("macro-loss IDs must be strictly ascending")
        if len(self.macro_ids) not in (18, 36):
            raise FittingV22Error("macro losses must cover exactly 18 or 36 macrocells")
        values = np.asarray(self.losses, dtype=np.float64)
        if values.shape != (len(self.macro_ids),):
            raise FittingV22Error("macro-loss vector length does not match its IDs")
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0.0)):
            raise FittingV22Error("macro losses must be finite and nonnegative")
        object.__setattr__(self, "losses", _immutable_float64(values))


@dataclass(frozen=True, slots=True)
class MacroTrim:
    """Stable 25% trim of one complete full/parity macro-loss vector."""

    macro_losses: MacroLosses
    ranked_macro_ids: tuple[int, ...]
    retained_in_rank_order: tuple[int, ...]
    retained_macro_ids: tuple[int, ...]
    objective: float

    def __post_init__(self) -> None:
        count = len(self.macro_losses.macro_ids)
        keep = count - math.floor(count * TRIM_FRACTION)
        _validate_macro_id_tuple(self.ranked_macro_ids, expected_count=count)
        _validate_macro_id_tuple(self.retained_in_rank_order, expected_count=keep)
        _validate_macro_id_tuple(self.retained_macro_ids, expected_count=keep)
        order = np.argsort(self.macro_losses.losses, kind="stable")
        expected_ranked = tuple(self.macro_losses.macro_ids[int(index)] for index in order)
        expected_retained_ranked = expected_ranked[:keep]
        expected_retained_sorted = tuple(sorted(expected_retained_ranked))
        if self.ranked_macro_ids != expected_ranked:
            raise FittingV22Error("ranked macro IDs do not follow stable loss order")
        if self.retained_in_rank_order != expected_retained_ranked:
            raise FittingV22Error("retained macro IDs are not the stable pass-one prefix")
        if self.retained_macro_ids != expected_retained_sorted:
            raise FittingV22Error("persisted retained macro IDs must be unique and sorted")
        selected = self.macro_losses.losses[order[:keep]]
        expected_objective = float(np.mean(selected, dtype=np.float64))
        if not math.isfinite(self.objective) or self.objective != expected_objective:
            raise FittingV22Error("trimmed objective does not match the retained macro losses")


@dataclass(frozen=True, slots=True)
class AffineCandidateFit:
    """Scalar affine-candidate objective and its immutable fit-site prediction."""

    fit_mask: np.ndarray
    prediction: np.ndarray
    trim: MacroTrim

    def __post_init__(self) -> None:
        mask, _ = _mask_kind(self.fit_mask)
        prediction = np.asarray(self.prediction, dtype=np.float64)
        expected_shape = (CHANNELS, int(np.count_nonzero(mask)))
        if prediction.shape != expected_shape or not bool(np.all(np.isfinite(prediction))):
            raise FittingV22Error("affine prediction has invalid shape or values")
        if self.trim.macro_losses.macro_ids != tuple(int(value) for value in np.unique(_fit_macro_ids(mask))):
            raise FittingV22Error("affine trim does not belong to its exact fit mask")
        object.__setattr__(self, "fit_mask", mask)
        object.__setattr__(self, "prediction", _immutable_float64(prediction))

    @property
    def objective(self) -> float:
        return self.trim.objective


@dataclass(frozen=True, slots=True)
class AppearanceFit:
    """Frozen two-pass appearance projection for one sampled candidate."""

    fit_mask: np.ndarray
    first_pass: OLSFit
    final_pass: OLSFit
    retention: MacroTrim
    final_macro_losses: MacroLosses
    prediction: np.ndarray
    objective: float

    def __post_init__(self) -> None:
        mask, _ = _mask_kind(self.fit_mask)
        expected_ids = tuple(int(value) for value in np.unique(_fit_macro_ids(mask)))
        if self.retention.macro_losses.macro_ids != expected_ids:
            raise FittingV22Error("appearance retention does not belong to its fit mask")
        if self.final_macro_losses.macro_ids != expected_ids:
            raise FittingV22Error("appearance final losses do not belong to its fit mask")
        prediction = np.asarray(self.prediction, dtype=np.float64)
        expected_shape = (CHANNELS, int(np.count_nonzero(mask)))
        if prediction.shape != expected_shape or not bool(np.all(np.isfinite(prediction))):
            raise FittingV22Error("appearance prediction has invalid shape or values")
        by_id = dict(zip(self.final_macro_losses.macro_ids, self.final_macro_losses.losses, strict=True))
        selected = np.asarray(
            [by_id[macro] for macro in self.retention.retained_in_rank_order], dtype=np.float64
        )
        expected_objective = float(np.mean(selected, dtype=np.float64))
        if not math.isfinite(self.objective) or self.objective != expected_objective:
            raise FittingV22Error("appearance objective did not reuse the pass-one retained set")
        object.__setattr__(self, "fit_mask", mask)
        object.__setattr__(self, "prediction", _immutable_float64(prediction))

    @property
    def gains(self) -> np.ndarray:
        return self.final_pass.gains

    @property
    def biases(self) -> np.ndarray:
        return self.final_pass.biases

    @property
    def retained_macro_ids(self) -> tuple[int, ...]:
        return self.retention.retained_macro_ids


@dataclass(frozen=True, slots=True)
class CombinedCandidateFit:
    """Combined candidate result after the exact appearance projection."""

    appearance: AppearanceFit

    @property
    def objective(self) -> float:
        return self.appearance.objective

    @property
    def prediction(self) -> np.ndarray:
        return self.appearance.prediction

    @property
    def gains(self) -> np.ndarray:
        return self.appearance.gains

    @property
    def biases(self) -> np.ndarray:
        return self.appearance.biases

    @property
    def retained_macro_ids(self) -> tuple[int, ...]:
        return self.appearance.retained_macro_ids


@dataclass(frozen=True, slots=True)
class BiasOnlyFit:
    """Frozen two-pass target-only comparator for one full or cross-fit direction."""

    fit_mask: np.ndarray
    output_mask: np.ndarray
    raw_first_biases: np.ndarray
    first_biases: np.ndarray
    raw_biases: np.ndarray
    biases: np.ndarray
    retention: MacroTrim
    final_macro_losses: MacroLosses
    prediction: np.ndarray
    objective: float

    def __post_init__(self) -> None:
        fitted, output = _validate_context_masks(self.fit_mask, self.output_mask)
        expected_ids = tuple(int(value) for value in np.unique(_fit_macro_ids(fitted)))
        if self.retention.macro_losses.macro_ids != expected_ids:
            raise FittingV22Error("bias-only retention does not belong to its fit mask")
        if self.final_macro_losses.macro_ids != expected_ids:
            raise FittingV22Error("bias-only final losses do not belong to its fit mask")
        for raw_name, clipped_name in (
            ("raw_first_biases", "first_biases"),
            ("raw_biases", "biases"),
        ):
            raw = np.asarray(getattr(self, raw_name), dtype=np.float64)
            clipped = np.asarray(getattr(self, clipped_name), dtype=np.float64)
            if raw.shape != (CHANNELS,) or not bool(np.all(np.isfinite(raw))):
                raise FittingV22Error(f"{raw_name} must be a finite three-vector")
            if clipped.shape != (CHANNELS,) or not np.array_equal(clipped, np.clip(raw, *BIAS_BOUNDS)):
                raise FittingV22Error(f"{clipped_name} is not the independent clipped raw bias")
            object.__setattr__(self, raw_name, _immutable_float64(raw))
            object.__setattr__(self, clipped_name, _immutable_float64(clipped))
        prediction = np.asarray(self.prediction, dtype=np.float64)
        expected_shape = (CHANNELS, int(np.count_nonzero(output)))
        if prediction.shape != expected_shape or not bool(np.all(np.isfinite(prediction))):
            raise FittingV22Error("bias-only prediction has invalid shape or values")
        if not np.array_equal(prediction, np.broadcast_to(self.biases[:, None], expected_shape)):
            raise FittingV22Error("bias-only prediction is not the fitted target marginal")
        by_id = dict(zip(self.final_macro_losses.macro_ids, self.final_macro_losses.losses, strict=True))
        selected = np.asarray(
            [by_id[macro] for macro in self.retention.retained_in_rank_order], dtype=np.float64
        )
        expected_objective = float(np.mean(selected, dtype=np.float64))
        if not math.isfinite(self.objective) or self.objective != expected_objective:
            raise FittingV22Error("bias-only objective did not reuse the pass-one retained set")
        object.__setattr__(self, "fit_mask", fitted)
        object.__setattr__(self, "output_mask", output)
        object.__setattr__(self, "prediction", _immutable_float64(prediction))

    @property
    def retained_macro_ids(self) -> tuple[int, ...]:
        return self.retention.retained_macro_ids


def solve_ols(source: np.ndarray, target: np.ndarray, selected: np.ndarray | None = None) -> OLSFit:
    """Solve the exact per-channel float64 OLS system on selected fit pixels."""

    if not isinstance(source, np.ndarray) or source.ndim != 2 or source.shape[0] != CHANNELS:
        raise FittingV22Error("OLS source must have shape [3,site]")
    source_values = _require_float64_array(source, cast(tuple[int, ...], source.shape), "OLS source")
    target_values = _require_float64_array(target, source_values.shape, "OLS target")
    count = source_values.shape[1]
    if count == 0:
        raise FittingV22Error("OLS received no pixels")
    if selected is None:
        pixel_mask = np.ones(count, dtype=np.bool_)
    else:
        if not isinstance(selected, np.ndarray) or selected.shape != (count,):
            raise FittingV22Error("OLS pixel mask must match the site dimension")
        if selected.dtype != np.dtype(np.bool_) or not selected.flags.c_contiguous:
            raise FittingV22Error("OLS pixel mask must be a C-contiguous bool array")
        pixel_mask = selected
    if not bool(np.any(pixel_mask)):
        raise FittingV22Error("OLS pixel mask selected no pixels")
    x = source_values[:, pixel_mask]
    y = target_values[:, pixel_mask]
    mean_x = np.asarray(np.mean(x, axis=1, dtype=np.float64), dtype=np.float64)
    mean_y = np.asarray(np.mean(y, axis=1, dtype=np.float64), dtype=np.float64)
    centered_x = x - mean_x[:, None]
    centered_y = y - mean_y[:, None]
    variance = np.asarray(
        np.mean(centered_x * centered_x, axis=1, dtype=np.float64), dtype=np.float64
    )
    covariance = np.asarray(
        np.mean(centered_x * centered_y, axis=1, dtype=np.float64), dtype=np.float64
    )
    raw_gain = covariance / np.maximum(variance, VARIANCE_FLOOR)
    raw_bias = mean_y - raw_gain * mean_x
    return OLSFit(
        source_means=mean_x,
        target_means=mean_y,
        variances=variance,
        covariances=covariance,
        raw_gains=raw_gain,
        raw_biases=raw_bias,
        gains=np.clip(raw_gain, *GAIN_BOUNDS),
        biases=np.clip(raw_bias, *BIAS_BOUNDS),
    )


def macro_losses(residual: np.ndarray, fit_mask: np.ndarray) -> MacroLosses:
    """Reduce one candidate's RGB residual to physical-macrocell MSE values."""

    mask, _ = _mask_kind(fit_mask)
    count = int(np.count_nonzero(mask))
    error = _require_float64_array(residual, (1, CHANNELS, count), "candidate residual")
    local_macro_ids = _fit_macro_ids(mask)
    ids = tuple(int(value) for value in np.unique(local_macro_ids))
    pixel_error = np.asarray(np.mean(error * error, axis=1, dtype=np.float64), dtype=np.float64)
    losses = np.asarray(
        [np.mean(pixel_error[0, local_macro_ids == macro], dtype=np.float64) for macro in ids],
        dtype=np.float64,
    )
    return MacroLosses(ids, losses)


def stable_trim(losses: MacroLosses) -> MacroTrim:
    """Apply the frozen stable 25% macrocell trim."""

    order = np.argsort(losses.losses, kind="stable")
    ranked = tuple(losses.macro_ids[int(index)] for index in order)
    keep = len(ranked) - math.floor(len(ranked) * TRIM_FRACTION)
    retained_ranked = ranked[:keep]
    retained_sorted = tuple(sorted(retained_ranked))
    objective = float(np.mean(losses.losses[order[:keep]], dtype=np.float64))
    return MacroTrim(losses, ranked, retained_ranked, retained_sorted, objective)


def affine_objective(sampled: np.ndarray, target: np.ndarray, fit_mask: np.ndarray) -> AffineCandidateFit:
    """Evaluate one affine candidate with the frozen macro-trimmed objective."""

    source, target_values, mask, _ = _validate_candidate(sampled, target, fit_mask)
    residual = np.asarray(source - target_values[None, :, :], dtype=np.float64)
    trim = stable_trim(macro_losses(residual, mask))
    return AffineCandidateFit(mask, source[0], trim)


def fit_appearance(sampled: np.ndarray, target: np.ndarray, fit_mask: np.ndarray) -> AppearanceFit:
    """Fit the exact clipped two-pass shared-macrocell appearance projection."""

    sampled_values, target_values, mask, _ = _validate_candidate(sampled, target, fit_mask)
    source = sampled_values[0]
    first = solve_ols(source, target_values)
    first_prediction = first.gains[:, None] * source + first.biases[:, None]
    first_residual = np.asarray(first_prediction - target_values, dtype=np.float64)
    retention = stable_trim(macro_losses(first_residual[None, :, :], mask))
    local_macro_ids = _fit_macro_ids(mask)
    retained_pixels = np.asarray(np.isin(local_macro_ids, retention.retained_macro_ids), dtype=np.bool_)
    final = solve_ols(source, target_values, retained_pixels)
    prediction = final.gains[:, None] * source + final.biases[:, None]
    final_residual = np.asarray(prediction - target_values, dtype=np.float64)
    final_losses = macro_losses(final_residual[None, :, :], mask)
    by_id = dict(zip(final_losses.macro_ids, final_losses.losses, strict=True))
    objective = float(
        np.mean(
            np.asarray([by_id[macro] for macro in retention.retained_in_rank_order], dtype=np.float64),
            dtype=np.float64,
        )
    )
    return AppearanceFit(mask, first, final, retention, final_losses, prediction, objective)


def combined_objective(sampled: np.ndarray, target: np.ndarray, fit_mask: np.ndarray) -> CombinedCandidateFit:
    """Evaluate one combined candidate by rerunning the complete appearance fit."""

    return CombinedCandidateFit(fit_appearance(sampled, target, fit_mask))


def fit_bias_only(fit_target: np.ndarray, fit_mask: np.ndarray, output_mask: np.ndarray) -> BiasOnlyFit:
    """Fit the exact two-pass target-only comparator without a third trim."""

    fitted, output = _validate_context_masks(fit_mask, output_mask)
    count = int(np.count_nonzero(fitted))
    target = _require_float64_array(fit_target, (CHANNELS, count), "bias-only fit target")
    raw_first = np.asarray(np.mean(target, axis=1, dtype=np.float64), dtype=np.float64)
    first = np.clip(raw_first, *BIAS_BOUNDS)
    first_residual = np.asarray(first[:, None] - target, dtype=np.float64)
    retention = stable_trim(macro_losses(first_residual[None, :, :], fitted))
    local_macro_ids = _fit_macro_ids(fitted)
    retained_pixels = np.asarray(np.isin(local_macro_ids, retention.retained_macro_ids), dtype=np.bool_)
    if not bool(np.any(retained_pixels)):
        raise FittingV22Error("bias-only pass one retained no pixels")
    raw_biases = np.asarray(np.mean(target[:, retained_pixels], axis=1, dtype=np.float64), dtype=np.float64)
    biases = np.clip(raw_biases, *BIAS_BOUNDS)
    final_residual = np.asarray(biases[:, None] - target, dtype=np.float64)
    final_losses = macro_losses(final_residual[None, :, :], fitted)
    by_id = dict(zip(final_losses.macro_ids, final_losses.losses, strict=True))
    objective = float(
        np.mean(
            np.asarray([by_id[macro] for macro in retention.retained_in_rank_order], dtype=np.float64),
            dtype=np.float64,
        )
    )
    prediction = np.broadcast_to(biases[:, None], (CHANNELS, int(np.count_nonzero(output)))).copy()
    return BiasOnlyFit(
        fit_mask=fitted,
        output_mask=output,
        raw_first_biases=raw_first,
        first_biases=first,
        raw_biases=raw_biases,
        biases=biases,
        retention=retention,
        final_macro_losses=final_losses,
        prediction=prediction,
        objective=objective,
    )


def reduce_candidate(
    arm: CandidateArm, sampled: np.ndarray, target: np.ndarray, fit_mask: np.ndarray
) -> AffineCandidateFit | CombinedCandidateFit:
    """Dispatch one exact leading-dimension-one candidate reduction."""

    if arm == "affine":
        return affine_objective(sampled, target, fit_mask)
    if arm == "combined":
        return combined_objective(sampled, target, fit_mask)
    raise FittingV22Error("candidate arm must be exactly 'affine' or 'combined'")
