"""Pure-NumPy deformation/appearance mechanism core for MM-008.

The functions in this module are deliberately limited to deterministic fitting,
prediction, and synthetic calibration.  They do not load real data, write
artifacts, or make lifecycle decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np

from bench.multimodal_resolution_diagnostics import method as mm007

SCHEMA_VERSION: Final = "mm008-method-v1"
EXPERIMENT_ID: Final = "MM-008"
CHANNELS: Final = 3
NATIVE_SIZE: Final = 64
CENTRAL_START: Final = 8
CENTRAL_STOP: Final = 56
CENTRAL_SIZE: Final = 48
MACRO_SIDE: Final = 6
MACRO_PIXELS: Final = 8
MACRO_COUNT: Final = 36
TRIM_FRACTION: Final = 0.25
FLOW_LIMIT: Final = 8.0
VARIANCE_FLOOR: Final = 1e-6
GAIN_BOUNDS: Final = (-2.0, 4.0)
BIAS_BOUNDS: Final = (-4.0, 4.0)
PERSISTENCE_FACTOR: Final = 1.25
PAIRING_FACTOR: Final = 1.10
SYNTHETIC_POSITIVE_FACTOR: Final = 0.50
SYNTHETIC_NEGATIVE_FACTOR: Final = 0.90
PARAMETER_NAMES: Final = ("ty", "tx", "ayy", "ayx", "axy", "axx")
FACTORIAL_ARMS: Final = ("affine", "appearance", "combined")
SENTINEL_ARMS: Final = ("global_translation", "quadrant_translation")
ALL_ARMS: Final = (*SENTINEL_ARMS, *FACTORIAL_ARMS)
TRANSLATION_VALUES: Final = (0.0, -4.0, 4.0, -8.0, 8.0)
GRADIENT_VALUES: Final = (0.0, -2.0, 2.0, -4.0, 4.0)
COORDINATE_VALUES: Final = (
    TRANSLATION_VALUES,
    TRANSLATION_VALUES,
    GRADIENT_VALUES,
    GRADIENT_VALUES,
    GRADIENT_VALUES,
    GRADIENT_VALUES,
)
INITIAL_TRANSLATIONS: Final = mm007.NATIVE_CANDIDATES
COORDINATE_SWEEPS: Final = 2

SYNTHETIC_SCENARIOS: Final = (
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
)
SYNTHETIC_SEED_MAP: Final = {
    scenario: 800_800 + index for index, scenario in enumerate(SYNTHETIC_SCENARIOS)
}
SYNTHETIC_POSITIVE_ARMS: Final = {
    "translation": ("global_translation", "quadrant_translation", "affine", "combined"),
    "affine": ("affine", "combined"),
    "appearance": ("appearance", "combined"),
    "combined": ("combined",),
    "stationary": (),
    "independent": (),
}

Arm = Literal["affine", "appearance", "combined"]
SentinelArm = Literal["global_translation", "quadrant_translation"]


@dataclass(frozen=True, slots=True)
class Geometry:
    """Frozen R64 scoring coordinates and physical macrocell membership."""

    coords: np.ndarray
    normalized_coords: np.ndarray
    macro_ids: np.ndarray
    parities: np.ndarray


def _geometry() -> Geometry:
    yy, xx = np.meshgrid(
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=np.float64),
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=np.float64),
        indexing="ij",
    )
    macro_y = ((yy.astype(int) - CENTRAL_START) // MACRO_PIXELS).reshape(-1)
    macro_x = ((xx.astype(int) - CENTRAL_START) // MACRO_PIXELS).reshape(-1)
    coords = np.stack((yy.reshape(-1), xx.reshape(-1)), axis=1)
    normalized = np.stack(
        ((coords[:, 0] - 31.5) / 23.5, (coords[:, 1] - 31.5) / 23.5), axis=1
    )
    return Geometry(
        coords=coords,
        normalized_coords=normalized,
        macro_ids=macro_y * MACRO_SIDE + macro_x,
        parities=(macro_y + macro_x) % 2,
    )


GEOMETRY: Final = _geometry()


def _validate_grids(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(source, dtype=np.float64)
    future = np.asarray(target, dtype=np.float64)
    if current.ndim != 4 or current.shape[1:] != (CHANNELS, NATIVE_SIZE, NATIVE_SIZE):
        raise ValueError("MM-008 source must have shape [N,3,64,64]")
    if future.shape != current.shape:
        raise ValueError("MM-008 target shape differs from source")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(future)):
        raise ValueError("MM-008 grids contain non-finite values")
    return current, future


def _target_values(target: np.ndarray, selected: np.ndarray) -> np.ndarray:
    mask = np.asarray(selected, dtype=bool)
    coords = GEOMETRY.coords[mask].astype(int)
    return np.asarray(target[:, :, coords[:, 0], coords[:, 1]], dtype=np.float64)


def _affine_flow(parameters: np.ndarray, normalized_coords: np.ndarray | None = None) -> np.ndarray:
    theta = np.asarray(parameters, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("affine parameters must have shape [N,6]")
    uv = GEOMETRY.normalized_coords if normalized_coords is None else np.asarray(normalized_coords)
    dy = theta[:, 0, None] + theta[:, 2, None] * uv[None, :, 0] + theta[:, 3, None] * uv[None, :, 1]
    dx = theta[:, 1, None] + theta[:, 4, None] * uv[None, :, 0] + theta[:, 5, None] * uv[None, :, 1]
    return np.stack((dy, dx), axis=2)


def _admissible(parameters: np.ndarray) -> np.ndarray:
    flow = _affine_flow(parameters)
    return np.asarray(np.max(np.abs(flow), axis=(1, 2)) <= FLOW_LIMIT, dtype=bool)


def _sample_affine(source: np.ndarray, parameters: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Sample source at ``(y-dy,x-dx)`` using deterministic float64 bilinear interpolation."""

    values = np.asarray(source, dtype=np.float64)
    mask = np.asarray(selected, dtype=bool)
    coords = GEOMETRY.coords[mask]
    uv = GEOMETRY.normalized_coords[mask]
    flow = _affine_flow(parameters, uv)
    source_y = coords[None, :, 0] - flow[:, :, 0]
    source_x = coords[None, :, 1] - flow[:, :, 1]
    if (
        np.min(source_y) < 0.0
        or np.min(source_x) < 0.0
        or np.max(source_y) > NATIVE_SIZE - 1
        or np.max(source_x) > NATIVE_SIZE - 1
    ):
        raise ValueError("affine warp samples out-of-bounds")
    y0 = np.floor(source_y).astype(int)
    x0 = np.floor(source_x).astype(int)
    y1 = np.minimum(y0 + 1, NATIVE_SIZE - 1)
    x1 = np.minimum(x0 + 1, NATIVE_SIZE - 1)
    wy = source_y - y0
    wx = source_x - x0
    batch = np.arange(len(values))[:, None, None]
    channels = np.arange(CHANNELS)[None, :, None]
    top = (
        values[batch, channels, y0[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :]
        + values[batch, channels, y0[:, None, :], x1[:, None, :]] * wx[:, None, :]
    )
    bottom = (
        values[batch, channels, y1[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :]
        + values[batch, channels, y1[:, None, :], x1[:, None, :]] * wx[:, None, :]
    )
    return cast(np.ndarray, top * (1.0 - wy)[:, None, :] + bottom * wy[:, None, :])


def _macro_losses(residual: np.ndarray, macro_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = np.mean(np.asarray(residual, dtype=np.float64) ** 2, axis=1)
    macros = np.unique(np.asarray(macro_ids, dtype=int))
    losses = np.stack([np.mean(error[:, macro_ids == macro], axis=1) for macro in macros], axis=1)
    return macros, losses


def _macro_trimmed_loss(residual: np.ndarray, macro_ids: np.ndarray) -> np.ndarray:
    macros, losses = _macro_losses(residual, macro_ids)
    keep = len(macros) - math.floor(len(macros) * TRIM_FRACTION)
    ordered = np.sort(losses, axis=1, kind="stable")[:, :keep]
    return cast(np.ndarray, np.mean(ordered, axis=1))


def _ols(source: np.ndarray, target: np.ndarray, pixel_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = len(source)
    gains = np.empty((rows, CHANNELS), dtype=np.float64)
    biases = np.empty_like(gains)
    for row in range(rows):
        selected = np.asarray(pixel_mask[row], dtype=bool)
        if not np.any(selected):
            raise ValueError("appearance fit received no pixels")
        x = source[row][:, selected]
        y = target[row][:, selected]
        mean_x = np.asarray(np.mean(x, axis=1, dtype=np.float64), dtype=np.float64)
        mean_y = np.asarray(np.mean(y, axis=1, dtype=np.float64), dtype=np.float64)
        centered_x = x - mean_x[:, None]
        centered_y = y - mean_y[:, None]
        variance = np.mean(centered_x * centered_x, axis=1, dtype=np.float64)
        covariance = np.mean(centered_x * centered_y, axis=1, dtype=np.float64)
        gain_raw = covariance / np.maximum(variance, VARIANCE_FLOOR)
        bias_raw = mean_y - gain_raw * mean_x
        gains[row] = np.clip(gain_raw, *GAIN_BOUNDS)
        biases[row] = np.clip(bias_raw, *BIAS_BOUNDS)
    return gains, biases


@dataclass(frozen=True, slots=True)
class AppearanceFit:
    gains: np.ndarray
    biases: np.ndarray
    retained_macros: np.ndarray
    prediction: np.ndarray
    objective: np.ndarray


def _fit_appearance(sampled: np.ndarray, target: np.ndarray, macro_ids: np.ndarray) -> AppearanceFit:
    """Run the frozen shared-macrocell two-pass gain/bias fit."""

    source_values = np.asarray(sampled, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.shape != target_values.shape or source_values.ndim != 3:
        raise ValueError("appearance source/target shapes differ")
    all_pixels = np.ones((len(source_values), source_values.shape[2]), dtype=bool)
    first_gain, first_bias = _ols(source_values, target_values, all_pixels)
    first_prediction = first_gain[:, :, None] * source_values + first_bias[:, :, None]
    macros, first_losses = _macro_losses(first_prediction - target_values, macro_ids)
    keep = len(macros) - math.floor(len(macros) * TRIM_FRACTION)
    order = np.argsort(first_losses, axis=1, kind="stable")[:, :keep]
    retained = macros[order]
    retained_pixels = np.stack(
        [np.isin(macro_ids, retained[row]) for row in range(len(source_values))], axis=0
    )
    gains, biases = _ols(source_values, target_values, retained_pixels)
    prediction = gains[:, :, None] * source_values + biases[:, :, None]
    _, final_losses = _macro_losses(prediction - target_values, macro_ids)
    objective = np.mean(np.take_along_axis(final_losses, order, axis=1), axis=1)
    return AppearanceFit(gains, biases, retained, prediction, objective)


def _optimizer_tolerance(objective: np.ndarray) -> np.ndarray:
    values = np.asarray(objective, dtype=np.float64)
    return np.maximum(1e-12, 1e-10 * np.abs(values))


def _first_minimum(losses: np.ndarray) -> np.ndarray:
    """Return the first exact minimum; optimizer tolerance is probe-only."""

    return cast(np.ndarray, np.argmin(np.asarray(losses, dtype=np.float64), axis=1))


def _candidate_objective(
    source: np.ndarray,
    target: np.ndarray,
    parameters: np.ndarray,
    fit_mask: np.ndarray,
    *,
    appearance: bool,
) -> np.ndarray:
    valid = _admissible(parameters)
    output = np.full(len(source), np.inf, dtype=np.float64)
    if not np.any(valid):
        return output
    sampled = _sample_affine(source[valid], parameters[valid], fit_mask)
    target_values = _target_values(target[valid], fit_mask)
    macro_ids = GEOMETRY.macro_ids[fit_mask]
    if appearance:
        output[valid] = _fit_appearance(sampled, target_values, macro_ids).objective
    else:
        output[valid] = _macro_trimmed_loss(sampled - target_values, macro_ids)
    return output


@dataclass(frozen=True, slots=True)
class DirectionFit:
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    objective: np.ndarray
    probe_strict_improvement: np.ndarray
    probe_best_improvement: np.ndarray


def _fit_affine_direction(
    source: np.ndarray, target: np.ndarray, fit_mask: np.ndarray, *, appearance: bool
) -> DirectionFit:
    rows = len(source)
    translations = np.asarray(INITIAL_TRANSLATIONS, dtype=np.float64)
    initial_losses = []
    for translation in translations:
        candidate = np.zeros((rows, len(PARAMETER_NAMES)), dtype=np.float64)
        candidate[:, :2] = translation
        initial_losses.append(
            _candidate_objective(source, target, candidate, fit_mask, appearance=appearance)
        )
    initial_loss_array = np.stack(initial_losses, axis=1)
    initial_index = _first_minimum(initial_loss_array)
    parameters = np.zeros((rows, len(PARAMETER_NAMES)), dtype=np.float64)
    parameters[:, :2] = translations[initial_index]
    for _ in range(COORDINATE_SWEEPS):
        for coordinate, choices in enumerate(COORDINATE_VALUES):
            losses = []
            for choice in choices:
                candidate = parameters.copy()
                candidate[:, coordinate] = choice
                losses.append(
                    _candidate_objective(source, target, candidate, fit_mask, appearance=appearance)
                )
            loss_array = np.stack(losses, axis=1)
            best_index = _first_minimum(loss_array)
            parameters[:, coordinate] = np.asarray(choices)[best_index]
    objective = _candidate_objective(source, target, parameters, fit_mask, appearance=appearance)
    best_probe = objective.copy()
    for coordinate, choices in enumerate(COORDINATE_VALUES):
        for choice in choices:
            candidate = parameters.copy()
            candidate[:, coordinate] = choice
            best_probe = np.minimum(
                best_probe,
                _candidate_objective(source, target, candidate, fit_mask, appearance=appearance),
            )
    improvement = np.maximum(objective - best_probe, 0.0)
    strict = improvement > _optimizer_tolerance(objective)
    sampled = _sample_affine(source, parameters, fit_mask)
    if appearance:
        final = _fit_appearance(sampled, _target_values(target, fit_mask), GEOMETRY.macro_ids[fit_mask])
        gains, biases, objective = final.gains, final.biases, final.objective
    else:
        gains = np.ones((rows, CHANNELS), dtype=np.float64)
        biases = np.zeros_like(gains)
    return DirectionFit(parameters, gains, biases, objective, strict, improvement)


def _fit_appearance_direction(source: np.ndarray, target: np.ndarray, fit_mask: np.ndarray) -> DirectionFit:
    rows = len(source)
    parameters = np.zeros((rows, len(PARAMETER_NAMES)), dtype=np.float64)
    sampled = _sample_affine(source, parameters, fit_mask)
    fitted = _fit_appearance(sampled, _target_values(target, fit_mask), GEOMETRY.macro_ids[fit_mask])
    return DirectionFit(
        parameters,
        fitted.gains,
        fitted.biases,
        fitted.objective,
        np.zeros(rows, dtype=bool),
        np.zeros(rows, dtype=np.float64),
    )


def _fit_direction(source: np.ndarray, target: np.ndarray, arm: Arm, fit_mask: np.ndarray) -> DirectionFit:
    if arm == "affine":
        return _fit_affine_direction(source, target, fit_mask, appearance=False)
    if arm == "appearance":
        return _fit_appearance_direction(source, target, fit_mask)
    if arm == "combined":
        return _fit_affine_direction(source, target, fit_mask, appearance=True)
    raise ValueError("unknown MM-008 factorial arm")


def _predict(
    source: np.ndarray,
    parameters: np.ndarray,
    gains: np.ndarray,
    biases: np.ndarray,
    output_mask: np.ndarray,
) -> np.ndarray:
    sampled = _sample_affine(source, parameters, output_mask)
    return np.asarray(gains[:, :, None] * sampled + biases[:, :, None], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Estimate:
    """One full or checkerboard-cross-fitted factorial estimate."""

    arm: str
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    prediction: np.ndarray
    objective: np.ndarray
    probe_strict_improvement: np.ndarray
    probe_best_improvement: np.ndarray
    site_flow_boundary_fraction: np.ndarray
    gradient_boundary_fraction: np.ndarray
    gain_boundary_fraction: np.ndarray
    bias_boundary_fraction: np.ndarray
    boundary_fraction: np.ndarray


def _boundary_fractions(
    arm: Arm, parameters: np.ndarray, gains: np.ndarray, biases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    leading = parameters.shape[:-1]
    flat_parameters = parameters.reshape(-1, len(PARAMETER_NAMES))
    flat_gains = gains.reshape(-1, CHANNELS)
    flat_biases = biases.reshape(-1, CHANNELS)
    flow = _affine_flow(flat_parameters)
    site = np.mean(np.isclose(np.abs(flow), FLOW_LIMIT, rtol=0.0, atol=1e-12), axis=(1, 2))
    gradient = np.mean(
        np.isclose(np.abs(flat_parameters[:, 2:]), 4.0, rtol=0.0, atol=1e-12), axis=1
    )
    gain = np.mean(
        np.isclose(flat_gains, GAIN_BOUNDS[0], rtol=0.0, atol=1e-12)
        | np.isclose(flat_gains, GAIN_BOUNDS[1], rtol=0.0, atol=1e-12),
        axis=1,
    )
    bias = np.mean(
        np.isclose(flat_biases, BIAS_BOUNDS[0], rtol=0.0, atol=1e-12)
        | np.isclose(flat_biases, BIAS_BOUNDS[1], rtol=0.0, atol=1e-12),
        axis=1,
    )
    if arm == "affine":
        boundary = np.maximum(site, gradient)
        gain.fill(0.0)
        bias.fill(0.0)
    elif arm == "appearance":
        boundary = np.maximum(gain, bias)
        site.fill(0.0)
        gradient.fill(0.0)
    else:
        boundary = np.maximum.reduce((site, gradient, gain, bias))
    return tuple(value.reshape(leading) for value in (site, gradient, gain, bias, boundary))  # type: ignore[return-value]


def estimate_full(source: np.ndarray, target: np.ndarray, arm: Arm) -> Estimate:
    current, future = _validate_grids(source, target)
    fit_mask = np.ones(len(GEOMETRY.coords), dtype=bool)
    fitted = _fit_direction(current, future, arm, fit_mask)
    prediction = _predict(current, fitted.parameters, fitted.gains, fitted.biases, fit_mask)
    boundaries = _boundary_fractions(arm, fitted.parameters, fitted.gains, fitted.biases)
    return Estimate(
        arm,
        fitted.parameters,
        fitted.gains,
        fitted.biases,
        prediction,
        fitted.objective,
        fitted.probe_strict_improvement,
        fitted.probe_best_improvement,
        *boundaries,
    )


def estimate_xfit(source: np.ndarray, target: np.ndarray, arm: Arm) -> Estimate:
    current, future = _validate_grids(source, target)
    rows = len(current)
    parameters = np.empty((rows, 2, len(PARAMETER_NAMES)), dtype=np.float64)
    gains = np.empty((rows, 2, CHANNELS), dtype=np.float64)
    biases = np.empty_like(gains)
    objectives = np.empty((rows, 2), dtype=np.float64)
    strict = np.empty((rows, 2), dtype=bool)
    improvement = np.empty((rows, 2), dtype=np.float64)
    prediction = np.full((rows, CHANNELS, len(GEOMETRY.coords)), np.nan, dtype=np.float64)
    for output_parity in (0, 1):
        fit_mask = GEOMETRY.parities == 1 - output_parity
        output_mask = GEOMETRY.parities == output_parity
        fitted = _fit_direction(current, future, arm, fit_mask)
        parameters[:, output_parity] = fitted.parameters
        gains[:, output_parity] = fitted.gains
        biases[:, output_parity] = fitted.biases
        objectives[:, output_parity] = fitted.objective
        strict[:, output_parity] = fitted.probe_strict_improvement
        improvement[:, output_parity] = fitted.probe_best_improvement
        prediction[:, :, output_mask] = _predict(
            current, fitted.parameters, fitted.gains, fitted.biases, output_mask
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("MM-008 cross-fit did not predict every central pixel")
    boundaries = _boundary_fractions(arm, parameters, gains, biases)
    return Estimate(
        arm,
        parameters,
        gains,
        biases,
        prediction,
        objectives,
        strict,
        improvement,
        *boundaries,
    )


@dataclass(frozen=True, slots=True)
class SentinelEstimate:
    arm: str
    flow: np.ndarray
    prediction: np.ndarray


def estimate_sentinel_full(
    source: np.ndarray, target: np.ndarray, arm: SentinelArm
) -> SentinelEstimate:
    current, future = _validate_grids(source, target)
    family = "global_translation" if arm == "global_translation" else "quadrant_flow"
    result = mm007._estimate_full(current, future, NATIVE_SIZE, family)
    return SentinelEstimate(arm, result.flow, result.prediction)


def estimate_sentinel_xfit(
    source: np.ndarray, target: np.ndarray, arm: SentinelArm
) -> SentinelEstimate:
    current, future = _validate_grids(source, target)
    family = "global_translation" if arm == "global_translation" else "quadrant_flow"
    result = mm007._estimate_xfit(current, future, NATIVE_SIZE, family)
    return SentinelEstimate(arm, result.flow, result.prediction)


def persistence_prediction(source: np.ndarray) -> np.ndarray:
    current = np.asarray(source, dtype=np.float64)
    if current.ndim != 4 or current.shape[1:] != (CHANNELS, NATIVE_SIZE, NATIVE_SIZE):
        raise ValueError("MM-008 source must have shape [N,3,64,64]")
    coords = GEOMETRY.coords.astype(int)
    return np.asarray(current[:, :, coords[:, 0], coords[:, 1]], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    scenario: str
    source: np.ndarray
    target: np.ndarray
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray


_TEXTURE_MODES: Final = np.asarray(
    ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (3, 2), (2, 3)), dtype=np.float64
)
_SYNTHETIC_TRUTH: Final = {
    "translation": (
        (4.0, -4.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ),
    "affine": (
        (0.0, 0.0, 2.0, 0.0, 0.0, -2.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ),
    "appearance": (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.25, 0.75, 1.5),
        (0.35, -0.25, 0.15),
    ),
    "combined": (
        (-4.0, 4.0, 0.0, 2.0, -2.0, 0.0),
        (1.2, 0.8, 1.4),
        (0.3, -0.2, 0.1),
    ),
    "stationary": (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ),
    "independent": (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ),
}


def _texture_batch(rng: np.random.Generator, rows: int = 6) -> np.ndarray:
    qy, qx = np.meshgrid(
        (np.arange(NATIVE_SIZE, dtype=np.float64) + 0.5) / NATIVE_SIZE,
        (np.arange(NATIVE_SIZE, dtype=np.float64) + 0.5) / NATIVE_SIZE,
        indexing="ij",
    )
    phase = 2.0 * np.pi * (
        _TEXTURE_MODES[:, 0, None, None] * qy[None]
        + _TEXTURE_MODES[:, 1, None, None] * qx[None]
    )
    sine = np.sin(phase)
    cosine = np.cos(phase)
    a = rng.normal(size=(rows, CHANNELS, len(_TEXTURE_MODES)))
    b = rng.normal(size=(rows, CHANNELS, len(_TEXTURE_MODES)))
    c = rng.normal(size=(rows, CHANNELS, 1, 1))
    texture = 0.35 * (
        np.einsum("nck,kyx->ncyx", a, sine) + np.einsum("nck,kyx->ncyx", b, cosine)
    )
    return np.asarray(texture + 0.15 * c, dtype=np.float64)


def _pool_r8(values: np.ndarray) -> np.ndarray:
    frames = np.asarray(values, dtype=np.float64)
    return np.asarray(
        np.mean(
            frames.reshape(len(frames), CHANNELS, 8, 8, 8, 8),
            axis=(3, 5),
            dtype=np.float64,
        ),
        dtype=np.float64,
    )


def synthetic_case(scenario: str) -> SyntheticCase:
    if scenario not in SYNTHETIC_SCENARIOS:
        raise ValueError("unknown MM-008 synthetic scenario")
    rng = np.random.Generator(np.random.PCG64(SYNTHETIC_SEED_MAP[scenario]))
    raw_source = _texture_batch(rng)
    pooled = _pool_r8(raw_source)
    mean = np.mean(pooled, axis=(0, 2, 3), keepdims=True, dtype=np.float64)
    scale = np.maximum(np.std(pooled, axis=(0, 2, 3), keepdims=True), VARIANCE_FLOOR)
    source = (raw_source - mean) / scale
    theta, gain, bias = _SYNTHETIC_TRUTH[scenario]
    parameters = np.broadcast_to(np.asarray(theta, dtype=np.float64), (len(source), 6)).copy()
    gains = np.broadcast_to(np.asarray(gain, dtype=np.float64), (len(source), CHANNELS)).copy()
    biases = np.broadcast_to(np.asarray(bias, dtype=np.float64), (len(source), CHANNELS)).copy()
    if scenario == "independent":
        raw_target = _texture_batch(rng)
        target_normalized = np.asarray((raw_target - mean) / scale, dtype=np.float64)
    else:
        target_normalized = source.copy()
        central = _sample_affine(source, parameters, np.ones(len(GEOMETRY.coords), dtype=bool))
        central = gains[:, :, None] * central + biases[:, :, None]
        coords = GEOMETRY.coords.astype(int)
        target_normalized[:, :, coords[:, 0], coords[:, 1]] = central
    # Exercise the frozen normalized -> raw -> current-normalized construction
    # literally.  The stationary control reuses the identical raw current so
    # its required persistence error remains bit-exactly zero.
    raw_target = raw_source if scenario == "stationary" else target_normalized * scale + mean
    target = np.asarray((raw_target - mean) / scale, dtype=np.float64)
    return SyntheticCase(scenario, source, target, parameters, gains, biases)


def _mse(prediction: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target_values, dtype=np.float64)
    return np.asarray(
        np.mean(difference * difference, axis=(1, 2), dtype=np.float64), dtype=np.float64
    )


def _derangements(rows: int) -> tuple[np.ndarray, np.ndarray]:
    if rows < 2 or rows % 2:
        raise ValueError("synthetic derangements require a positive even row count")
    near = np.arange(rows).reshape(-1, 2)[:, ::-1].reshape(-1)
    far = np.roll(np.arange(rows), rows // 2)
    return near, far


@dataclass(frozen=True, slots=True)
class SyntheticMetrics:
    arm: str
    persistence_mse: np.ndarray
    full_mse: np.ndarray
    xfit_mse: np.ndarray
    near_mse: np.ndarray
    far_mse: np.ndarray
    complete_support: np.ndarray
    full_parameter_error: float | None
    xfit_parameter_error: float | None
    full_appearance_error: float | None
    xfit_appearance_error: float | None

    def expectation_failures(self, scenario: str) -> tuple[str, ...]:
        """Return frozen pre-formal calibration failures for this arm/scenario."""

        if scenario not in SYNTHETIC_POSITIVE_ARMS:
            raise ValueError("unknown MM-008 synthetic scenario")
        failures: list[str] = []
        positive = self.arm in SYNTHETIC_POSITIVE_ARMS[scenario]
        if scenario == "stationary":
            if not np.array_equal(self.persistence_mse, np.zeros_like(self.persistence_mse)):
                failures.append("persistence_nonzero")
            if np.any(self.complete_support):
                failures.append("false_complete_support")
            return tuple(failures)
        if positive:
            if not np.all(self.full_mse <= SYNTHETIC_POSITIVE_FACTOR * self.persistence_mse):
                failures.append("full_margin")
            if not np.all(self.xfit_mse <= SYNTHETIC_POSITIVE_FACTOR * self.persistence_mse):
                failures.append("xfit_margin")
            if not np.all(self.complete_support):
                failures.append("complete_support")
            for name, value in (
                ("full_parameter_endpoint", self.full_parameter_error),
                ("xfit_parameter_endpoint", self.xfit_parameter_error),
                ("full_appearance_endpoint", self.full_appearance_error),
                ("xfit_appearance_endpoint", self.xfit_appearance_error),
            ):
                if value is not None and value > 1e-10:
                    failures.append(name)
        else:
            if not np.all(self.xfit_mse >= SYNTHETIC_NEGATIVE_FACTOR * self.persistence_mse):
                failures.append("negative_margin")
            if np.any(self.complete_support):
                failures.append("false_complete_support")
        return tuple(failures)


def synthetic_metrics(case: SyntheticCase, arm: str) -> SyntheticMetrics:
    if arm not in ALL_ARMS:
        raise ValueError("unknown MM-008 arm")
    target_values = _target_values(case.target, np.ones(len(GEOMETRY.coords), dtype=bool))
    persistence_mse = _mse(persistence_prediction(case.source), target_values)
    near, far = _derangements(len(case.source))
    parameter_full: float | None = None
    parameter_xfit: float | None = None
    appearance_full: float | None = None
    appearance_xfit: float | None = None
    if arm in SENTINEL_ARMS:
        sentinel_arm = cast(SentinelArm, arm)
        full = estimate_sentinel_full(case.source, case.target, sentinel_arm)
        xfit = estimate_sentinel_xfit(case.source, case.target, sentinel_arm)
        near_fit = estimate_sentinel_xfit(case.source, case.target[near], sentinel_arm)
        far_fit = estimate_sentinel_xfit(case.source, case.target[far], sentinel_arm)
        full_prediction, xfit_prediction = full.prediction, xfit.prediction
        near_prediction, far_prediction = near_fit.prediction, far_fit.prediction
        expected_flow = case.parameters[:, :2]
        parameter_full = float(
            np.max(np.abs(full.flow - np.broadcast_to(expected_flow[:, None], full.flow.shape)))
        )
        parameter_xfit = float(
            np.max(np.abs(xfit.flow - np.broadcast_to(expected_flow[:, None, None], xfit.flow.shape)))
        )
    else:
        factorial_arm = cast(Arm, arm)
        full_result = estimate_full(case.source, case.target, factorial_arm)
        xfit_result = estimate_xfit(case.source, case.target, factorial_arm)
        near_result = estimate_xfit(case.source, case.target[near], factorial_arm)
        far_result = estimate_xfit(case.source, case.target[far], factorial_arm)
        full_prediction, xfit_prediction = full_result.prediction, xfit_result.prediction
        near_prediction, far_prediction = near_result.prediction, far_result.prediction
        if arm in {"affine", "combined"}:
            parameter_full = float(np.max(np.abs(full_result.parameters - case.parameters)))
            parameter_xfit = float(
                np.max(
                    np.abs(
                        xfit_result.parameters
                        - np.broadcast_to(case.parameters[:, None], xfit_result.parameters.shape)
                    )
                )
            )
        if arm in {"appearance", "combined"}:
            appearance_full = float(
                max(
                    np.max(np.abs(full_result.gains - case.gains)),
                    np.max(np.abs(full_result.biases - case.biases)),
                )
            )
            appearance_xfit = float(
                max(
                    np.max(
                        np.abs(
                            xfit_result.gains
                            - np.broadcast_to(case.gains[:, None], xfit_result.gains.shape)
                        )
                    ),
                    np.max(
                        np.abs(
                            xfit_result.biases
                            - np.broadcast_to(case.biases[:, None], xfit_result.biases.shape)
                        )
                    ),
                )
            )
    full_mse = _mse(full_prediction, target_values)
    xfit_mse = _mse(xfit_prediction, target_values)
    near_mse = _mse(near_prediction, target_values)
    far_mse = _mse(far_prediction, target_values)
    complete = (
        (persistence_mse > 0.0)
        & (PERSISTENCE_FACTOR * xfit_mse <= persistence_mse)
        & (PAIRING_FACTOR * xfit_mse <= near_mse)
        & (PAIRING_FACTOR * xfit_mse <= far_mse)
    )
    return SyntheticMetrics(
        arm,
        persistence_mse,
        full_mse,
        xfit_mse,
        near_mse,
        far_mse,
        complete,
        parameter_full,
        parameter_xfit,
        appearance_full,
        appearance_xfit,
    )
