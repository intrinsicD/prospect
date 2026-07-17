"""Pure-NumPy engine for the MM-006 causal-warp ceiling diagnostic.

The causal path accepts only previous/current arrays.  The primary target-aware
oracle is checkerboard cross-fitted; the full target fit is separately labeled and
used only to diagnose target-fitting overfit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final, cast

import numpy as np

from bench.multimodal_horizon_diagnostics import method as parent
from bench.multimodal_preflight import dataset

SCHEMA_VERSION: Final = "mm006-method-v1"
EXPERIMENT_ID: Final = "MM-006"
DOMAINS: Final = ("pixel", "taesd")
FLOW_FAMILIES: Final = ("global_translation", "quadrant_flow")
PRIMARY_FAMILY: Final = "quadrant_flow"
SYNTHETIC_SEEDS: Final = (660_060, 660_061, 660_062)
SYNTHETIC_SCENARIOS: Final = (
    "translation",
    "reversal",
    "appearance",
    "source_null",
    "stationary",
    "ambiguous",
)
SYNTHETIC_CHANNELS: Final = (3, 4)
SCALE_FLOOR: Final = 1e-6
LOCAL_REGULARIZER: Final = 0.05
REQUIRED_VIDEO_SUPPORT: Final = 6
REQUIRED_ANY_IMPROVEMENT: Final = 7
PERSISTENCE_FACTOR: Final = 1.25
CONTROL_FACTOR: Final = 1.10
SOURCE_FACTOR: Final = 1.25
BOUNDARY_WARNING_FRACTION: Final = 0.25
ACTIVITY_MSE_MIN: Final = 1e-4
ACTIVITY_RATIO_MIN: Final = 0.10
ACTIVITY_RATIO_MAX: Final = 1.0 / 1.2
CENTRAL_START: Final = 1
CENTRAL_STOP: Final = 7
CENTRAL_SIZE: Final = 6
CENTRAL_CELLS: Final = CENTRAL_SIZE * CENTRAL_SIZE
TRIM_FRACTION: Final = 0.25

_candidate_values = (-1.0, -0.5, 0.0, 0.5, 1.0)
CANDIDATES: Final = tuple(
    sorted(
        ((dy, dx) for dy in _candidate_values for dx in _candidate_values),
        key=lambda item: (item[0] * item[0] + item[1] * item[1], item[0], item[1]),
    )
)

RECOMMENDATIONS: Final = {
    "invalid_MM006_synthetic_positive_control": "repair signed warp recovery before interpreting real videos",
    "invalid_MM006_synthetic_negative_control": "repair branch, leakage, or ambiguity controls",
    "invalid_MM006_real_negative_control": "reject the real assay because shuffled targets cross the null gate",
    "inconclusive_MM006_real_negative_control": "strengthen or independently replicate target-pair controls",
    "target_fitted_oracle_overfit_supported": (
        "increase spatial support or regularize correspondence before using an oracle ceiling"
    ),
    "tested_transport_range_inconclusive": "widen the frozen displacement range in a new experiment",
    "tested_pixel_warp_ceiling_failure_supported": "add an appearance/occlusion residual or higher-resolution frontend",
    "two_frame_motion_extrapolation_failure_supported": (
        "add longer context, acceleration, action, or camera conditioning"
    ),
    "low_resolution_correspondence_failure_supported": (
        "increase spatial resolution or improve causal correspondence descriptors"
    ),
    "taesd_transport_equivariance_failure_supported": "change or fine-tune the visual frontend for spatial transport",
    "taesd_two_frame_motion_extrapolation_failure_supported": (
        "add longer context, acceleration, action, or camera conditioning before latent transport"
    ),
    "taesd_causal_correspondence_failure_supported": "learn latent correspondence with longer causal context",
    "single_step_causal_warp_fix_supported": (
        "adopt warp-plus-residual prediction, then require independent-panel rollout"
    ),
    "MM006_diagnostic_inconclusive": "add a discriminating control or independent panel before selecting a fix",
}


@dataclass(frozen=True, slots=True)
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return cast(np.ndarray, np.asarray((array - self.mean) / self.scale, dtype=np.float64))

    def fingerprint(self) -> str:
        return _arrays_sha256("mm006-normalizer-v1", self.mean, self.scale)


@dataclass(frozen=True, slots=True)
class FlowEstimate:
    flow: np.ndarray
    prediction: np.ndarray
    confidence: np.ndarray
    input_sha256: str
    flow_sha256: str
    prediction_sha256: str


@dataclass(frozen=True, slots=True)
class WarpPanel:
    video_ids: np.ndarray
    timestamps: np.ndarray
    previous: np.ndarray
    current: np.ndarray
    target: np.ndarray

    @property
    def channels(self) -> int:
        return int(np.asarray(self.current).shape[1])

    def validate(self, expected_channels: int | None = None, *, formal: bool = True) -> None:
        ids = np.asarray(self.video_ids, dtype=str)
        times = np.asarray(self.timestamps, dtype=np.float64)
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in (self.previous, self.current, self.target))
        if times.shape != ids.shape or any(value.shape != arrays[0].shape for value in arrays):
            raise ValueError("warp panel identity or array shapes differ")
        if arrays[0].ndim != 4 or arrays[0].shape[0] != len(ids) or arrays[0].shape[2:] != (8, 8):
            raise ValueError("warp panel arrays must be [N,C,8,8]")
        if expected_channels is not None and arrays[0].shape[1] != expected_channels:
            raise ValueError(f"warp panel must contain {expected_channels} channels")
        if not all(np.all(np.isfinite(value)) for value in (times, *arrays)):
            raise ValueError("warp panel contains non-finite values")
        if formal:
            if ids.shape != (parent.MATCHED_ROWS,):
                raise ValueError("formal warp panel must contain exactly 453 rows")
            expected = [
                (video_id, 1.5 + 0.5 * index)
                for video_id in dataset.SAMPLE_VIDEO_IDS
                for index in range(parent.MATCHED_COUNTS[video_id])
            ]
            actual = list(zip(ids.tolist(), times.tolist(), strict=True))
            if actual != expected:
                raise ValueError("formal warp identities differ from MM-005")

    def subset(self, video_ids: Sequence[str]) -> WarpPanel:
        wanted = set(video_ids)
        mask = np.asarray([str(value) in wanted for value in self.video_ids], dtype=bool)
        if not np.any(mask):
            raise ValueError("warp subset is empty")
        output = WarpPanel(
            video_ids=np.asarray(self.video_ids, dtype=str)[mask].copy(),
            timestamps=np.asarray(self.timestamps, dtype=np.float64)[mask].copy(),
            previous=np.asarray(self.previous, dtype=np.float64)[mask].copy(),
            current=np.asarray(self.current, dtype=np.float64)[mask].copy(),
            target=np.asarray(self.target, dtype=np.float64)[mask].copy(),
        )
        output.validate(self.channels, formal=False)
        return output


def warp_panel(table: parent.MatchedPanel) -> WarpPanel:
    table.validate()
    output = WarpPanel(
        video_ids=np.asarray(table.video_ids, dtype=str).copy(),
        timestamps=np.asarray(table.timestamps, dtype=np.float64).copy(),
        previous=np.asarray(table.previous, dtype=np.float64).copy(),
        current=np.asarray(table.current, dtype=np.float64).copy(),
        target=np.asarray(table.target_0p5, dtype=np.float64).copy(),
    )
    output.validate(table.channels)
    return output


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = sha256(b"mm006-array-v1")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_sha256(prefix: str, *values: np.ndarray) -> str:
    digest = sha256(prefix.encode("ascii"))
    for value in values:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    output = float(value)
    if nonnegative and output < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return output


def _exact_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _derived_close(value: object, expected: float | None, name: str) -> None:
    if expected is None:
        if value is not None:
            raise ValueError(f"{name} must be null when persistence is zero")
        return
    actual = _finite(value, name)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"derived {name} differs")


def _fit_normalizer(current: np.ndarray) -> Normalizer:
    values = np.asarray(current, dtype=np.float64)
    if values.ndim != 4 or not np.all(np.isfinite(values)):
        raise ValueError("normalizer input must be finite [N,C,H,W]")
    mean = values.mean(axis=(0, 2, 3), keepdims=True)
    scale = np.maximum(values.std(axis=(0, 2, 3), keepdims=True), SCALE_FLOOR)
    return Normalizer(mean=np.asarray(mean), scale=np.asarray(scale))


def _central(values: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        np.asarray(values, dtype=np.float64)[:, :, CENTRAL_START:CENTRAL_STOP, CENTRAL_START:CENTRAL_STOP],
    )


def _coords() -> np.ndarray:
    return cast(
        np.ndarray,
        np.asarray(
            [(y, x) for y in range(CENTRAL_START, CENTRAL_STOP) for x in range(CENTRAL_START, CENTRAL_STOP)],
            dtype=np.float64,
        ),
    )


CENTRAL_COORDS: Final = _coords()
PARITIES: Final = np.asarray([(int(y) + int(x)) % 2 for y, x in CENTRAL_COORDS], dtype=int)
TILE_IDS: Final = np.asarray(
    [2 * int(y >= 4) + int(x >= 4) for y, x in CENTRAL_COORDS],
    dtype=int,
)


def _sample(grid: np.ndarray, coords: np.ndarray, displacement: Sequence[float]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    points = np.asarray(coords, dtype=np.float64)
    dy, dx = float(displacement[0]), float(displacement[1])
    source_y = points[:, 0] - dy
    source_x = points[:, 1] - dx
    if np.min(source_y) < 0.0 or np.max(source_y) > 7.0 or np.min(source_x) < 0.0 or np.max(source_x) > 7.0:
        raise ValueError("warp candidate needs out-of-bounds sampling")
    y0 = np.floor(source_y).astype(int)
    x0 = np.floor(source_x).astype(int)
    y1 = np.minimum(y0 + 1, 7)
    x1 = np.minimum(x0 + 1, 7)
    wy = source_y - y0
    wx = source_x - x0
    top = values[:, y0, x0] * (1.0 - wx)[None, :] + values[:, y0, x1] * wx[None, :]
    bottom = values[:, y1, x0] * (1.0 - wx)[None, :] + values[:, y1, x1] * wx[None, :]
    return cast(np.ndarray, np.asarray(top * (1.0 - wy)[None, :] + bottom * wy[None, :]))


def _sample_batch(grids: np.ndarray, coords: np.ndarray, displacements: np.ndarray) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    points = np.asarray(coords, dtype=np.float64)
    shifts = np.asarray(displacements, dtype=np.float64)
    if values.ndim != 4 or shifts.shape != (len(values), 2):
        raise ValueError("batch warp shapes differ")
    source_y = points[None, :, 0] - shifts[:, None, 0]
    source_x = points[None, :, 1] - shifts[:, None, 1]
    if np.min(source_y) < 0.0 or np.max(source_y) > 7.0 or np.min(source_x) < 0.0 or np.max(source_x) > 7.0:
        raise ValueError("batch warp candidate needs out-of-bounds sampling")
    y0 = np.floor(source_y).astype(int)
    x0 = np.floor(source_x).astype(int)
    y1 = np.minimum(y0 + 1, 7)
    x1 = np.minimum(x0 + 1, 7)
    wy = source_y - y0
    wx = source_x - x0
    batch = np.arange(len(values))[:, None, None]
    channels = np.arange(values.shape[1])[None, :, None]
    top = (
        values[batch, channels, y0[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :]
        + values[batch, channels, y0[:, None, :], x1[:, None, :]] * wx[:, None, :]
    )
    bottom = (
        values[batch, channels, y1[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :]
        + values[batch, channels, y1[:, None, :], x1[:, None, :]] * wx[:, None, :]
    )
    return cast(np.ndarray, top * (1.0 - wy)[:, None, :] + bottom * wy[:, None, :])


def _target_sites(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    points = np.asarray(coords, dtype=int)
    return cast(np.ndarray, np.asarray(grid, dtype=np.float64)[:, points[:, 0], points[:, 1]])


def _target_sites_batch(grids: np.ndarray, coords: np.ndarray) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    points = np.asarray(coords, dtype=int)
    return cast(np.ndarray, values[:, :, points[:, 0], points[:, 1]])


def _trimmed_loss(residual: np.ndarray) -> float:
    site_loss = np.mean(np.asarray(residual, dtype=np.float64) ** 2, axis=0)
    trim = int(math.floor(len(site_loss) * TRIM_FRACTION))
    retained = np.sort(site_loss, kind="stable")[: len(site_loss) - trim]
    return float(np.mean(retained))


def _trimmed_loss_batch(residual: np.ndarray) -> np.ndarray:
    site_loss = np.mean(np.asarray(residual, dtype=np.float64) ** 2, axis=1)
    trim = int(math.floor(site_loss.shape[1] * TRIM_FRACTION))
    retained = np.sort(site_loss, axis=1, kind="stable")[:, : site_loss.shape[1] - trim]
    return cast(np.ndarray, np.mean(retained, axis=1))


def _select_displacement(
    source: np.ndarray,
    target: np.ndarray,
    coords: np.ndarray,
    *,
    anchor: Sequence[float] | None = None,
) -> tuple[np.ndarray, float]:
    target_values = _target_sites(target, coords)
    identity = _trimmed_loss(_sample(source, coords, (0.0, 0.0)) - target_values)
    losses: list[float] = []
    for candidate in CANDIDATES:
        loss = _trimmed_loss(_sample(source, coords, candidate) - target_values)
        if anchor is not None:
            loss += (
                LOCAL_REGULARIZER
                * (identity + 1e-12)
                * ((candidate[0] - float(anchor[0])) ** 2 + (candidate[1] - float(anchor[1])) ** 2)
            )
        losses.append(loss)
    best_index = int(np.argmin(np.asarray(losses, dtype=np.float64)))
    best = np.asarray(CANDIDATES[best_index], dtype=np.float64)
    separated = [
        loss
        for candidate, loss in zip(CANDIDATES, losses, strict=True)
        if (candidate[0] - best[0]) ** 2 + (candidate[1] - best[1]) ** 2 >= 0.25
    ]
    confidence = min(separated) - losses[best_index] if separated else 0.0
    return best, float(confidence)


def _select_displacements(
    source: np.ndarray,
    target: np.ndarray,
    coords: np.ndarray,
    *,
    anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    before = np.asarray(source, dtype=np.float64)
    after = np.asarray(target, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 4:
        raise ValueError("batch match arrays differ")
    target_values = _target_sites_batch(after, coords)
    zero: np.ndarray = np.zeros((len(before), 2), dtype=np.float64)
    identity = _trimmed_loss_batch(_sample_batch(before, coords, zero) - target_values)
    candidate_array = np.asarray(CANDIDATES, dtype=np.float64)
    losses: list[np.ndarray] = []
    for candidate in candidate_array:
        shifts = np.broadcast_to(candidate, (len(before), 2))
        losses.append(_trimmed_loss_batch(_sample_batch(before, coords, shifts) - target_values))
    loss_array = np.stack(losses, axis=1)
    if anchor is not None:
        anchors = np.asarray(anchor, dtype=np.float64)
        if anchors.shape != (len(before), 2):
            raise ValueError("batch anchor shapes differ")
        distance = np.sum((candidate_array[None, :, :] - anchors[:, None, :]) ** 2, axis=2)
        loss_array = loss_array + LOCAL_REGULARIZER * (identity[:, None] + 1e-12) * distance
    best_index = np.argmin(loss_array, axis=1)
    best = candidate_array[best_index]
    best_loss = loss_array[np.arange(len(before)), best_index]
    separated = np.sum((candidate_array[None, :, :] - best[:, None, :]) ** 2, axis=2) >= 0.25
    alternative = np.min(np.where(separated, loss_array, np.inf), axis=1)
    return cast(np.ndarray, best), cast(np.ndarray, alternative - best_loss)


def _estimate_one(
    source: np.ndarray,
    target: np.ndarray,
    family: str,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if family not in FLOW_FAMILIES:
        raise ValueError("unknown MM-006 flow family")
    mask = np.ones(CENTRAL_CELLS, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if mask.shape != (CENTRAL_CELLS,) or not np.any(mask):
        raise ValueError("flow fit mask differs")
    global_flow, global_confidence = _select_displacement(source, target, CENTRAL_COORDS[mask])
    if family == "global_translation":
        return global_flow[None, :], np.asarray([global_confidence], dtype=np.float64)
    flows: list[np.ndarray] = []
    confidences: list[float] = []
    for tile in range(4):
        selected = mask & (TILE_IDS == tile)
        if not np.any(selected):
            raise ValueError("quadrant fit mask has an empty tile")
        flow, confidence = _select_displacement(
            source,
            target,
            CENTRAL_COORDS[selected],
            anchor=(float(global_flow[0]), float(global_flow[1])),
        )
        flows.append(flow)
        confidences.append(confidence)
    return np.stack(flows), np.asarray(confidences, dtype=np.float64)


def _estimate_batch(
    source: np.ndarray,
    target: np.ndarray,
    family: str,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if family not in FLOW_FAMILIES:
        raise ValueError("unknown MM-006 flow family")
    mask = np.ones(CENTRAL_CELLS, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if mask.shape != (CENTRAL_CELLS,) or not np.any(mask):
        raise ValueError("batch flow fit mask differs")
    global_flow, global_confidence = _select_displacements(source, target, CENTRAL_COORDS[mask])
    if family == "global_translation":
        return global_flow[:, None, :], global_confidence[:, None]
    flows: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for tile in range(4):
        selected = mask & (TILE_IDS == tile)
        if not np.any(selected):
            raise ValueError("batch quadrant fit mask has an empty tile")
        flow, confidence = _select_displacements(
            source,
            target,
            CENTRAL_COORDS[selected],
            anchor=global_flow,
        )
        flows.append(flow)
        confidences.append(confidence)
    return np.stack(flows, axis=1), np.stack(confidences, axis=1)


def _predict_one(grid: np.ndarray, flow: np.ndarray, family: str, output_mask: np.ndarray | None = None) -> np.ndarray:
    mask = np.ones(CENTRAL_CELLS, dtype=bool) if output_mask is None else np.asarray(output_mask, dtype=bool)
    prediction = np.full((grid.shape[0], CENTRAL_CELLS), np.nan, dtype=np.float64)
    if family == "global_translation":
        prediction[:, mask] = _sample(
            grid,
            CENTRAL_COORDS[mask],
            (float(flow[0, 0]), float(flow[0, 1])),
        )
    elif family == "quadrant_flow":
        for tile in range(4):
            selected = mask & (TILE_IDS == tile)
            prediction[:, selected] = _sample(
                grid,
                CENTRAL_COORDS[selected],
                (float(flow[tile, 0]), float(flow[tile, 1])),
            )
    else:
        raise ValueError("unknown MM-006 flow family")
    return cast(np.ndarray, prediction)


def _predict_batch(
    grids: np.ndarray,
    flows: np.ndarray,
    family: str,
    output_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    flow_values = np.asarray(flows, dtype=np.float64)
    mask = np.ones(CENTRAL_CELLS, dtype=bool) if output_mask is None else np.asarray(output_mask, dtype=bool)
    prediction = np.full((len(values), values.shape[1], CENTRAL_CELLS), np.nan, dtype=np.float64)
    if family == "global_translation":
        prediction[:, :, mask] = _sample_batch(values, CENTRAL_COORDS[mask], flow_values[:, 0, :])
    elif family == "quadrant_flow":
        for tile in range(4):
            selected = mask & (TILE_IDS == tile)
            prediction[:, :, selected] = _sample_batch(values, CENTRAL_COORDS[selected], flow_values[:, tile, :])
    else:
        raise ValueError("unknown MM-006 flow family")
    return cast(np.ndarray, prediction)


def _estimate_causal(previous: np.ndarray, current: np.ndarray, family: str) -> FlowEstimate:
    """Estimate and apply causal flow without accepting a future target."""

    before = np.asarray(previous, dtype=np.float64)
    present = np.asarray(current, dtype=np.float64)
    if before.shape != present.shape or before.ndim != 4 or before.shape[2:] != (8, 8):
        raise ValueError("causal arrays must be equal [N,C,8,8]")
    flow_array, confidence_array = _estimate_batch(before, present, family)
    prediction_array = _predict_batch(present, flow_array, family).reshape(
        len(before), before.shape[1], CENTRAL_SIZE, CENTRAL_SIZE
    )
    return FlowEstimate(
        flow=flow_array,
        prediction=prediction_array,
        confidence=confidence_array,
        input_sha256=_arrays_sha256("mm006-causal-input-v1", before, present),
        flow_sha256=_array_sha256(flow_array),
        prediction_sha256=_array_sha256(prediction_array),
    )


def _estimate_source_xfit(previous: np.ndarray, current: np.ndarray, family: str) -> FlowEstimate:
    """Cross-fit a past-to-current source-reconstruction diagnostic."""

    before = np.asarray(previous, dtype=np.float64)
    present = np.asarray(current, dtype=np.float64)
    if before.shape != present.shape or before.ndim != 4 or before.shape[2:] != (8, 8):
        raise ValueError("source cross-fit arrays must be equal [N,C,8,8]")
    parity_flows: list[np.ndarray] = []
    parity_confidences: list[np.ndarray] = []
    prediction = np.full((len(before), before.shape[1], CENTRAL_CELLS), np.nan, dtype=np.float64)
    for output_parity in (0, 1):
        fit_mask = PARITIES != output_parity
        output_mask = PARITIES == output_parity
        flow, confidence = _estimate_batch(before, present, family, fit_mask)
        partial = _predict_batch(before, flow, family, output_mask)
        prediction[:, :, output_mask] = partial[:, :, output_mask]
        parity_flows.append(flow)
        parity_confidences.append(confidence)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("cross-fitted source estimator did not predict every central cell")
    flow_array = np.stack(parity_flows, axis=1)
    confidence_array = np.stack(parity_confidences, axis=1)
    prediction_array = prediction.reshape(len(before), before.shape[1], CENTRAL_SIZE, CENTRAL_SIZE)
    return FlowEstimate(
        flow=flow_array,
        prediction=prediction_array,
        confidence=confidence_array,
        input_sha256=_arrays_sha256("mm006-source-xfit-input-v1", before, present),
        flow_sha256=_array_sha256(flow_array),
        prediction_sha256=_array_sha256(prediction_array),
    )


def _apply_flow(grids: np.ndarray, flows: np.ndarray, family: str) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    flow_values = np.asarray(flows, dtype=np.float64)
    return cast(
        np.ndarray,
        _predict_batch(values, flow_values, family).reshape(len(values), values.shape[1], CENTRAL_SIZE, CENTRAL_SIZE),
    )


def _apply_xfit_flow(grids: np.ndarray, flows: np.ndarray, family: str) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    flow_values = np.asarray(flows, dtype=np.float64)
    family_size = 1 if family == "global_translation" else 4
    if flow_values.shape != (len(values), 2, family_size, 2):
        raise ValueError("cross-fitted flow shape differs")
    prediction = np.full((len(values), values.shape[1], CENTRAL_CELLS), np.nan, dtype=np.float64)
    for output_parity in (0, 1):
        output_mask = PARITIES == output_parity
        partial = _predict_batch(values, flow_values[:, output_parity], family, output_mask)
        prediction[:, :, output_mask] = partial[:, :, output_mask]
    if not np.all(np.isfinite(prediction)):
        raise ValueError("cross-fitted flow application did not predict every central cell")
    return cast(np.ndarray, prediction.reshape(len(values), values.shape[1], CENTRAL_SIZE, CENTRAL_SIZE))


def _estimate_oracle_full(current: np.ndarray, target: np.ndarray, family: str) -> FlowEstimate:
    present = np.asarray(current, dtype=np.float64)
    future = np.asarray(target, dtype=np.float64)
    flow_array, confidence_array = _estimate_batch(present, future, family)
    prediction_array = _predict_batch(present, flow_array, family).reshape(
        len(present), present.shape[1], CENTRAL_SIZE, CENTRAL_SIZE
    )
    return FlowEstimate(
        flow=flow_array,
        prediction=prediction_array,
        confidence=confidence_array,
        input_sha256=_arrays_sha256("mm006-oracle-full-input-v1", present, future),
        flow_sha256=_array_sha256(flow_array),
        prediction_sha256=_array_sha256(prediction_array),
    )


def _estimate_oracle_xfit(current: np.ndarray, target: np.ndarray, family: str) -> FlowEstimate:
    present = np.asarray(current, dtype=np.float64)
    future = np.asarray(target, dtype=np.float64)
    parity_flows: list[np.ndarray] = []
    parity_confidences: list[np.ndarray] = []
    prediction = np.full((len(present), present.shape[1], CENTRAL_CELLS), np.nan, dtype=np.float64)
    for output_parity in (0, 1):
        fit_mask = PARITIES != output_parity
        output_mask = PARITIES == output_parity
        flow, confidence = _estimate_batch(present, future, family, fit_mask)
        partial = _predict_batch(present, flow, family, output_mask)
        prediction[:, :, output_mask] = partial[:, :, output_mask]
        parity_flows.append(flow)
        parity_confidences.append(confidence)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("cross-fitted oracle did not predict every central cell")
    flow_array = np.stack(parity_flows, axis=1)
    prediction_array = prediction.reshape(len(present), present.shape[1], CENTRAL_SIZE, CENTRAL_SIZE)
    confidence_array = np.stack(parity_confidences, axis=1)
    return FlowEstimate(
        flow=flow_array,
        prediction=prediction_array,
        confidence=confidence_array,
        input_sha256=_arrays_sha256("mm006-oracle-xfit-input-v1", present, future),
        flow_sha256=_array_sha256(flow_array),
        prediction_sha256=_array_sha256(prediction_array),
    )


def _derangement(panel: WarpPanel) -> np.ndarray:
    ids = np.asarray(panel.video_ids, dtype=str)
    times = np.asarray(panel.timestamps, dtype=np.float64)
    mapping: np.ndarray = np.empty(len(ids), dtype=int)
    for video_id in tuple(dict.fromkeys(ids.tolist())):
        rows = np.flatnonzero(ids == video_id)
        ordered = rows[np.argsort(times[rows], kind="stable")]
        if len(ordered) < 2:
            raise ValueError("derangement requires at least two rows")
        mapping[ordered] = np.roll(ordered, len(ordered) // 2)
    if np.any(mapping == np.arange(len(mapping))) or np.any(ids[mapping] != ids):
        raise ValueError("derangement must be fixed-point-free and within-video")
    return mapping


def _sse(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int, float]:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(difference)):
        raise ValueError("metric contains non-finite values")
    numerator = float(np.sum(difference * difference))
    elements = int(difference.size)
    return numerator, elements, numerator / elements


def _flow_stats(causal: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    causal_values = np.asarray(causal, dtype=np.float64)
    oracle_values = np.asarray(oracle, dtype=np.float64)
    if oracle_values.shape != (causal_values.shape[0], 2, *causal_values.shape[1:]):
        raise ValueError("causal/oracle flow shapes differ")
    expanded_causal = np.broadcast_to(causal_values[:, None, ...], oracle_values.shape)
    flat_causal = expanded_causal.reshape(-1, 2)
    flat_oracle = oracle_values.reshape(-1, 2)
    endpoint = float(np.mean(np.sum((flat_causal - flat_oracle) ** 2, axis=1)))
    denominator = float(np.linalg.norm(flat_causal) * np.linalg.norm(flat_oracle))
    cosine = float(np.sum(flat_causal * flat_oracle) / denominator) if denominator > 0.0 else 0.0
    return {"endpoint_mse": endpoint, "cosine": cosine}


def _metric_row(
    panel: WarpPanel,
    *,
    domain: str,
    family: str,
    fold: dataset.DatasetFold,
    normalizer: Normalizer,
    panel_seed: int | None,
    scenario: str | None,
    known_causal_flow: np.ndarray | None = None,
    known_target_flow: np.ndarray | None = None,
) -> dict[str, Any]:
    panel.validate(panel.channels, formal=False)
    previous = normalizer.apply(panel.previous)
    current = normalizer.apply(panel.current)
    target = normalizer.apply(panel.target)
    mapping = _derangement(panel)
    target_central = _central(target)
    persistence = _central(current)
    velocity = _central(2.0 * current - previous)

    causal = _estimate_causal(previous, current, family)
    history = _estimate_causal(previous[mapping], current, family)
    reverse_prediction = _apply_flow(current, -causal.flow, family)
    source_xfit = _estimate_source_xfit(previous, current, family)
    history_source_xfit = _estimate_source_xfit(previous[mapping], current, family)
    source_reconstruction = source_xfit.prediction
    history_source_reconstruction = history_source_xfit.prediction
    reverse_source = _estimate_causal(current, previous, family)
    oracle_full = _estimate_oracle_full(current, target, family)
    oracle_xfit = _estimate_oracle_xfit(current, target, family)
    target_shuffle = _estimate_oracle_xfit(current, target[mapping], family)

    metrics: dict[str, tuple[float, int, float]] = {
        "persistence": _sse(persistence, target_central),
        "velocity": _sse(velocity, target_central),
        "causal": _sse(causal.prediction, target_central),
        "history_shuffle": _sse(history.prediction, target_central),
        "reverse_sign": _sse(reverse_prediction, target_central),
        "oracle_full": _sse(oracle_full.prediction, target_central),
        "oracle_xfit": _sse(oracle_xfit.prediction, target_central),
        "target_shuffle_oracle": _sse(target_shuffle.prediction, target_central),
        "source_reconstruction": _sse(source_reconstruction, _central(current)),
        "history_source_reconstruction": _sse(history_source_reconstruction, _central(current)),
        "source_identity": _sse(_central(previous), _central(current)),
    }
    element_counts = {value[1] for value in metrics.values()}
    if len(element_counts) != 1:
        raise ValueError("MM-006 metrics do not share one denominator")
    mse = {name: value[2] for name, value in metrics.items()}
    p = mse["persistence"]
    o = mse["oracle_xfit"]
    s = mse["target_shuffle_oracle"]
    c = mse["causal"]
    h = mse["history_shuffle"]
    r = mse["reverse_sign"]
    v = mse["velocity"]
    a = mse["source_reconstruction"]
    source_null = mse["history_source_reconstruction"]
    b = mse["source_identity"]
    oracle_performance_support = bool(p > 0.0 and PERSISTENCE_FACTOR * o <= p)
    oracle_pairing_support = bool(oracle_performance_support and CONTROL_FACTOR * o <= s)
    oracle_support = oracle_pairing_support
    source_shuffle_null_hit = bool(CONTROL_FACTOR * a <= source_null)
    source_support = bool(b > 0.0 and SOURCE_FACTOR * a <= b and source_shuffle_null_hit)
    causal_performance_support = bool(oracle_performance_support and PERSISTENCE_FACTOR * c <= p and 2.0 * c <= p + o)
    causal_support = bool(
        oracle_pairing_support
        and causal_performance_support
        and CONTROL_FACTOR * c <= h
        and CONTROL_FACTOR * c <= r
        and CONTROL_FACTOR * c <= v
        and source_support
    )
    full_oracle_support = bool(p > 0.0 and PERSISTENCE_FACTOR * mse["oracle_full"] <= p)
    full_only = bool(full_oracle_support and not oracle_performance_support)
    null_hit = bool(p > 0.0 and PERSISTENCE_FACTOR * s <= p)
    oracle_gap = p - o
    capture = (p - c) / oracle_gap if oracle_gap > 0.0 else 0.0
    flow_stats = _flow_stats(causal.flow, oracle_xfit.flow)
    cycle = causal.flow + reverse_source.flow
    boundary_fraction = float(np.mean(np.isclose(np.abs(causal.flow), 1.0)))
    oracle_boundary_fraction = float(np.mean(np.isclose(np.abs(oracle_xfit.flow), 1.0)))

    def known_endpoint(flow: np.ndarray, expected: np.ndarray | None) -> float | None:
        if expected is None:
            return None
        expected_flow = np.asarray(expected, dtype=np.float64)
        if expected_flow.shape != (len(panel.video_ids), 2):
            raise ValueError("known synthetic flow must be [N,2]")
        if flow.ndim == 3:
            expanded = np.broadcast_to(expected_flow[:, None, :], flow.shape)
        elif flow.ndim == 4:
            expanded = np.broadcast_to(expected_flow[:, None, None, :], flow.shape)
        else:
            raise ValueError("known synthetic flow rank differs")
        return float(np.mean(np.sum((flow - expanded) ** 2, axis=-1)))

    known_causal_endpoint = known_endpoint(causal.flow, known_causal_flow)
    known_oracle_xfit_endpoint = known_endpoint(oracle_xfit.flow, known_target_flow)
    known_oracle_full_endpoint = known_endpoint(oracle_full.flow, known_target_flow)
    return {
        "domain": domain,
        "panel_seed": panel_seed,
        "scenario": scenario,
        "family": family,
        "fold": fold.index,
        "video_id": str(panel.video_ids[0]),
        "rows": len(panel.video_ids),
        "channels": panel.channels,
        "cells_per_row": CENTRAL_CELLS,
        "candidate_count": len(CANDIDATES),
        "elements": next(iter(element_counts)),
        "normalizer_fingerprint": normalizer.fingerprint(),
        "source_sha256": _arrays_sha256("mm006-source-v1", previous, current),
        "target_sha256": _array_sha256(target),
        "causal_input_sha256": causal.input_sha256,
        "causal_flow_sha256": causal.flow_sha256,
        "causal_prediction_sha256": causal.prediction_sha256,
        "persistence_prediction_sha256": _array_sha256(persistence),
        "velocity_prediction_sha256": _array_sha256(velocity),
        "reverse_prediction_sha256": _array_sha256(reverse_prediction),
        "history_flow_sha256": history.flow_sha256,
        "history_input_sha256": history.input_sha256,
        "history_prediction_sha256": history.prediction_sha256,
        "source_xfit_input_sha256": source_xfit.input_sha256,
        "source_xfit_flow_sha256": source_xfit.flow_sha256,
        "source_xfit_prediction_sha256": source_xfit.prediction_sha256,
        "history_source_xfit_input_sha256": history_source_xfit.input_sha256,
        "history_source_xfit_flow_sha256": history_source_xfit.flow_sha256,
        "history_source_xfit_prediction_sha256": history_source_xfit.prediction_sha256,
        "oracle_full_input_sha256": oracle_full.input_sha256,
        "oracle_full_flow_sha256": oracle_full.flow_sha256,
        "oracle_full_prediction_sha256": oracle_full.prediction_sha256,
        "oracle_xfit_input_sha256": oracle_xfit.input_sha256,
        "oracle_xfit_flow_sha256": oracle_xfit.flow_sha256,
        "oracle_xfit_prediction_sha256": oracle_xfit.prediction_sha256,
        "target_shuffle_input_sha256": target_shuffle.input_sha256,
        "target_shuffle_flow_sha256": target_shuffle.flow_sha256,
        "target_shuffle_prediction_sha256": target_shuffle.prediction_sha256,
        "source_identity_prediction_sha256": _array_sha256(_central(previous)),
        "sse": {name: value[0] for name, value in metrics.items()},
        "mse": mse,
        "causal_ratio": c / p if p > 0.0 else None,
        "oracle_ratio": o / p if p > 0.0 else None,
        "full_oracle_ratio": mse["oracle_full"] / p if p > 0.0 else None,
        "causal_capture": float(capture),
        "known_causal_flow_endpoint_mse": known_causal_endpoint,
        "known_oracle_xfit_flow_endpoint_mse": known_oracle_xfit_endpoint,
        "known_oracle_full_flow_endpoint_mse": known_oracle_full_endpoint,
        "oracle_performance_support": oracle_performance_support,
        "oracle_pairing_support": oracle_pairing_support,
        "oracle_support": oracle_support,
        "causal_performance_support": causal_performance_support,
        "causal_support": causal_support,
        "source_shuffle_null_hit": source_shuffle_null_hit,
        "source_support": source_support,
        "full_oracle_support": full_oracle_support,
        "full_oracle_only_support": full_only,
        "target_shuffle_null_hit": null_hit,
        "causal_improves_persistence": bool(c < p),
        "oracle_improves_persistence": bool(o < p),
        "causal_mean_flow_magnitude": float(np.mean(np.linalg.norm(causal.flow, axis=-1))),
        "oracle_mean_flow_magnitude": float(np.mean(np.linalg.norm(oracle_xfit.flow, axis=-1))),
        "causal_boundary_fraction": boundary_fraction,
        "oracle_boundary_fraction": oracle_boundary_fraction,
        "causal_confidence_gap": float(np.mean(causal.confidence)),
        "oracle_confidence_gap": float(np.mean(oracle_xfit.confidence)),
        "flow_endpoint_mse": flow_stats["endpoint_mse"],
        "flow_cosine": flow_stats["cosine"],
        "cycle_endpoint_mse": float(np.mean(np.sum(cycle * cycle, axis=-1))),
        "uses_target": False,
        "oracle_uses_target": True,
        "oracle_diagnostic_only": True,
        "finite": True,
    }


def _normalizer_rows(panel: WarpPanel, domain: str) -> tuple[list[dict[str, Any]], dict[int, Normalizer]]:
    rows: list[dict[str, Any]] = []
    normalizers: dict[int, Normalizer] = {}
    ids = np.asarray(panel.video_ids, dtype=str)
    for fold in dataset.formal_folds():
        mask = np.asarray([video_id in fold.train_ids for video_id in ids], dtype=bool)
        normalizer = _fit_normalizer(np.asarray(panel.current)[mask])
        normalizers[fold.index] = normalizer
        rows.append(
            {
                "domain": domain,
                "fold": fold.index,
                "train_video_ids": list(fold.train_ids),
                "test_video_ids": list(fold.test_ids),
                "train_rows": int(np.sum(mask)),
                "mean": np.asarray(normalizer.mean, dtype=np.float64).reshape(-1).tolist(),
                "scale": np.asarray(normalizer.scale, dtype=np.float64).reshape(-1).tolist(),
                "fingerprint": normalizer.fingerprint(),
            }
        )
    return rows, normalizers


def _real_rows(panel: WarpPanel, domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalizer_rows, normalizers = _normalizer_rows(panel, domain)
    metric_rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        for video_id in fold.test_ids:
            video = panel.subset([video_id])
            for family in FLOW_FAMILIES:
                metric_rows.append(
                    _metric_row(
                        video,
                        domain=domain,
                        family=family,
                        fold=fold,
                        normalizer=normalizers[fold.index],
                        panel_seed=None,
                        scenario=None,
                    )
                )
    return normalizer_rows, metric_rows


def _analytic_texture(
    rows: int,
    channels: int,
    y: np.ndarray,
    x: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    if np.broadcast_shapes(y.shape, x.shape)[-2:] != (8, 8):
        raise ValueError("analytic texture coordinates must end in [8,8]")
    output: np.ndarray = np.empty((rows, channels, 8, 8), dtype=np.float64)
    frequencies = ((0.31, 0.47), (0.53, 0.29), (0.41, 0.61), (0.67, 0.37))
    for channel in range(channels):
        fy, fx = frequencies[channel]
        base = fy * y + fx * x + phase[:, channel, None, None]
        output[:, channel] = np.sin(base) + 0.45 * np.cos(0.71 * base + 0.19 * x - 0.13 * y)
    return output


def _synthetic_motions(rows: int, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    motions = np.asarray(
        [(-1.0, 0.0), (-0.5, 0.5), (0.0, 1.0), (0.5, -0.5), (1.0, 0.0), (0.0, -1.0)],
        dtype=np.float64,
    )
    past_flow = motions[np.arange(rows) % len(motions)].copy()
    future_flow = past_flow.copy()
    if scenario == "reversal":
        future_flow = -past_flow
    elif scenario == "stationary":
        past_flow[:] = 0.0
        future_flow[:] = 0.0
    return past_flow, future_flow


def synthetic_panel(template: WarpPanel, seed: int, channels: int, scenario: str) -> WarpPanel:
    if seed not in SYNTHETIC_SEEDS or channels not in SYNTHETIC_CHANNELS or scenario not in SYNTHETIC_SCENARIOS:
        raise ValueError("unknown MM-006 synthetic panel")
    rows = len(template.video_ids)
    rng = np.random.Generator(np.random.PCG64(seed + 100 * channels + 10_000 * SYNTHETIC_SCENARIOS.index(scenario)))
    phase = rng.uniform(-math.pi, math.pi, size=(rows, channels))
    coordinate_grids: tuple[np.ndarray, np.ndarray] = np.meshgrid(
        np.arange(8, dtype=np.float64),
        np.arange(8, dtype=np.float64),
        indexing="ij",
    )
    y: np.ndarray = np.asarray(coordinate_grids[0], dtype=np.float64)
    x: np.ndarray = np.asarray(coordinate_grids[1], dtype=np.float64)
    past_flow, future_flow = _synthetic_motions(rows, scenario)
    current = _analytic_texture(rows, channels, y[None, ...], x[None, ...], phase)
    previous = np.empty_like(current)
    target = np.empty_like(current)
    for index in range(rows):
        previous[index] = _analytic_texture(
            1,
            channels,
            y[None, ...] + past_flow[index, 0],
            x[None, ...] + past_flow[index, 1],
            phase[index : index + 1],
        )[0]
        target[index] = _analytic_texture(
            1,
            channels,
            y[None, ...] - future_flow[index, 0],
            x[None, ...] - future_flow[index, 1],
            phase[index : index + 1],
        )[0]
        current[index, :, CENTRAL_START:CENTRAL_STOP, CENTRAL_START:CENTRAL_STOP] = _sample(
            previous[index], CENTRAL_COORDS, (float(past_flow[index, 0]), float(past_flow[index, 1]))
        ).reshape(channels, CENTRAL_SIZE, CENTRAL_SIZE)
        target[index, :, CENTRAL_START:CENTRAL_STOP, CENTRAL_START:CENTRAL_STOP] = _sample(
            current[index], CENTRAL_COORDS, (float(future_flow[index, 0]), float(future_flow[index, 1]))
        ).reshape(channels, CENTRAL_SIZE, CENTRAL_SIZE)
    if scenario == "appearance":
        target = rng.normal(0.0, 1.0, size=(rows, channels, 8, 8))
    elif scenario == "source_null":
        previous = rng.normal(0.0, 1.0, size=(rows, channels, 8, 8))
        current = rng.normal(0.0, 1.0, size=(rows, channels, 8, 8))
        target = rng.normal(0.0, 1.0, size=(rows, channels, 8, 8))
    elif scenario == "ambiguous":
        checker = ((np.indices((8, 8)).sum(axis=0) % 2) * 2.0 - 1.0)[None, None, :, :]
        current = np.broadcast_to(checker, (rows, channels, 8, 8)).copy()
        previous = current.copy()
        target = np.roll(current, 1, axis=3)
    output = WarpPanel(
        video_ids=np.asarray(template.video_ids, dtype=str).copy(),
        timestamps=np.asarray(template.timestamps, dtype=np.float64).copy(),
        previous=previous,
        current=current,
        target=target,
    )
    output.validate(channels)
    return output


def _synthetic_rows(template: WarpPanel) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panel_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for seed in SYNTHETIC_SEEDS:
        for channels in SYNTHETIC_CHANNELS:
            for scenario in SYNTHETIC_SCENARIOS:
                panel = synthetic_panel(template, seed, channels, scenario)
                past_flow, future_flow = _synthetic_motions(len(panel.video_ids), scenario)
                panel_rows.append(
                    {
                        "panel_seed": seed,
                        "channels": channels,
                        "scenario": scenario,
                        "previous_sha256": _array_sha256(panel.previous),
                        "current_sha256": _array_sha256(panel.current),
                        "target_sha256": _array_sha256(panel.target),
                    }
                )
                normalizer_rows, normalizers = _normalizer_rows(panel, f"synthetic_{channels}")
                del normalizer_rows
                for fold in dataset.formal_folds():
                    for video_id in fold.test_ids:
                        video = panel.subset([video_id])
                        expected_flow = None
                        expected_target_flow = None
                        if scenario not in {"ambiguous", "source_null"}:
                            expected_flow = past_flow[np.asarray(panel.video_ids, dtype=str) == video_id]
                        if scenario not in {"appearance", "ambiguous", "source_null"}:
                            expected_target_flow = future_flow[np.asarray(panel.video_ids, dtype=str) == video_id]
                        for family in FLOW_FAMILIES:
                            metric_rows.append(
                                _metric_row(
                                    video,
                                    domain=f"synthetic_{channels}",
                                    family=family,
                                    fold=fold,
                                    normalizer=normalizers[fold.index],
                                    panel_seed=seed,
                                    scenario=scenario,
                                    known_causal_flow=expected_flow,
                                    known_target_flow=expected_target_flow,
                                )
                            )
    return panel_rows, metric_rows


def _activity_rows(panel: WarpPanel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        video = panel.subset([video_id])
        mapping = _derangement(video)
        persistence = _sse(_central(video.current), _central(video.target))[2]
        shuffled = _sse(_central(video.current), _central(video.target[mapping]))[2]
        ratio = persistence / max(shuffled, 1e-15)
        rows.append(
            {
                "video_id": video_id,
                "rows": len(video.video_ids),
                "persistence_mse": persistence,
                "shuffled_mse": shuffled,
                "activity_ratio": ratio,
                "active": bool(persistence >= ACTIVITY_MSE_MIN and ACTIVITY_RATIO_MIN <= ratio <= ACTIVITY_RATIO_MAX),
            }
        )
    return rows


def execute(taesd: parent.MatchedPanel, pixel: parent.MatchedPanel) -> dict[str, object]:
    taesd.validate(4)
    pixel.validate(3)
    if not np.array_equal(taesd.video_ids, pixel.video_ids) or not np.array_equal(taesd.timestamps, pixel.timestamps):
        raise ValueError("MM-006 domain identities differ")
    taesd_panel = warp_panel(taesd)
    pixel_panel = warp_panel(pixel)
    normalizer_rows: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    for domain, panel in (("pixel", pixel_panel), ("taesd", taesd_panel)):
        norms, metrics = _real_rows(panel, domain)
        normalizer_rows.extend(norms)
        real_rows.extend(metrics)
    synthetic_panels, synthetic_metrics = _synthetic_rows(pixel_panel)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "alignment": parent.alignment_record(taesd, pixel),
        "normalizer_rows": normalizer_rows,
        "synthetic_panel_rows": synthetic_panels,
        "synthetic_metric_rows": synthetic_metrics,
        "real_metric_rows": real_rows,
        "activity_rows": _activity_rows(pixel_panel),
    }
    return validate_evidence(evidence)


def _validate_metric_row(value: object, *, synthetic: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metric row must be an object")
    row = dict(value)
    required = {
        "domain",
        "panel_seed",
        "scenario",
        "family",
        "fold",
        "video_id",
        "rows",
        "channels",
        "cells_per_row",
        "candidate_count",
        "elements",
        "normalizer_fingerprint",
        "source_sha256",
        "target_sha256",
        "causal_input_sha256",
        "causal_flow_sha256",
        "causal_prediction_sha256",
        "persistence_prediction_sha256",
        "velocity_prediction_sha256",
        "reverse_prediction_sha256",
        "history_input_sha256",
        "history_flow_sha256",
        "history_prediction_sha256",
        "source_xfit_input_sha256",
        "source_xfit_flow_sha256",
        "source_xfit_prediction_sha256",
        "history_source_xfit_input_sha256",
        "history_source_xfit_flow_sha256",
        "history_source_xfit_prediction_sha256",
        "oracle_full_input_sha256",
        "oracle_full_flow_sha256",
        "oracle_full_prediction_sha256",
        "oracle_xfit_input_sha256",
        "oracle_xfit_flow_sha256",
        "oracle_xfit_prediction_sha256",
        "target_shuffle_input_sha256",
        "target_shuffle_flow_sha256",
        "target_shuffle_prediction_sha256",
        "source_identity_prediction_sha256",
        "sse",
        "mse",
        "causal_ratio",
        "oracle_ratio",
        "full_oracle_ratio",
        "causal_capture",
        "known_causal_flow_endpoint_mse",
        "known_oracle_xfit_flow_endpoint_mse",
        "known_oracle_full_flow_endpoint_mse",
        "oracle_performance_support",
        "oracle_pairing_support",
        "oracle_support",
        "causal_performance_support",
        "causal_support",
        "source_shuffle_null_hit",
        "source_support",
        "full_oracle_support",
        "full_oracle_only_support",
        "target_shuffle_null_hit",
        "causal_improves_persistence",
        "oracle_improves_persistence",
        "causal_mean_flow_magnitude",
        "oracle_mean_flow_magnitude",
        "causal_boundary_fraction",
        "oracle_boundary_fraction",
        "causal_confidence_gap",
        "oracle_confidence_gap",
        "flow_endpoint_mse",
        "flow_cosine",
        "cycle_endpoint_mse",
        "uses_target",
        "oracle_uses_target",
        "oracle_diagnostic_only",
        "finite",
    }
    if set(row) != required:
        raise ValueError("metric row schema differs")
    fold_index = _exact_int(row["fold"], "metric fold")
    if row["family"] not in FLOW_FAMILIES or fold_index not in range(4):
        raise ValueError("metric family/fold differs")
    if row["video_id"] not in dataset.SAMPLE_VIDEO_IDS:
        raise ValueError("metric video differs")
    fold = dataset.formal_folds()[fold_index]
    if row["video_id"] not in fold.test_ids:
        raise ValueError("metric video is not in its fold test pair")
    if synthetic:
        if (
            row["domain"] not in {"synthetic_3", "synthetic_4"}
            or _exact_int(row["panel_seed"], "synthetic panel seed") not in SYNTHETIC_SEEDS
            or row["scenario"] not in SYNTHETIC_SCENARIOS
        ):
            raise ValueError("synthetic metric identity differs")
    elif row["domain"] not in DOMAINS or row["panel_seed"] is not None or row["scenario"] is not None:
        raise ValueError("real metric identity differs")
    expected_channels = (
        int(str(row["domain"]).removeprefix("synthetic_"))
        if synthetic
        else {
            "pixel": 3,
            "taesd": 4,
        }[str(row["domain"])]
    )
    rows = _exact_int(row["rows"], "metric rows")
    channels = _exact_int(row["channels"], "metric channels")
    cells = _exact_int(row["cells_per_row"], "metric cells per row")
    candidate_count = _exact_int(row["candidate_count"], "metric candidate count")
    elements = _exact_int(row["elements"], "metric elements")
    if rows != parent.MATCHED_COUNTS[str(row["video_id"])] or channels != expected_channels:
        raise ValueError("metric row/channel count differs")
    if candidate_count != len(CANDIDATES):
        raise ValueError("metric candidate count differs")
    if cells != CENTRAL_CELLS or elements != rows * channels * CENTRAL_CELLS:
        raise ValueError("metric element accounting differs")
    for name in (
        "normalizer_fingerprint",
        "source_sha256",
        "target_sha256",
        "causal_input_sha256",
        "causal_flow_sha256",
        "causal_prediction_sha256",
        "persistence_prediction_sha256",
        "velocity_prediction_sha256",
        "reverse_prediction_sha256",
        "history_input_sha256",
        "history_flow_sha256",
        "history_prediction_sha256",
        "source_xfit_input_sha256",
        "source_xfit_flow_sha256",
        "source_xfit_prediction_sha256",
        "history_source_xfit_input_sha256",
        "history_source_xfit_flow_sha256",
        "history_source_xfit_prediction_sha256",
        "oracle_full_input_sha256",
        "oracle_full_flow_sha256",
        "oracle_full_prediction_sha256",
        "oracle_xfit_input_sha256",
        "oracle_xfit_flow_sha256",
        "oracle_xfit_prediction_sha256",
        "target_shuffle_input_sha256",
        "target_shuffle_flow_sha256",
        "target_shuffle_prediction_sha256",
        "source_identity_prediction_sha256",
    ):
        _fingerprint(row[name], name)
    if (
        row["uses_target"] is not False
        or row["oracle_uses_target"] is not True
        or row["oracle_diagnostic_only"] is not True
    ):
        raise ValueError("metric leakage labels differ")
    boolean_names = (
        "oracle_performance_support",
        "oracle_pairing_support",
        "oracle_support",
        "causal_performance_support",
        "causal_support",
        "source_shuffle_null_hit",
        "source_support",
        "full_oracle_support",
        "full_oracle_only_support",
        "target_shuffle_null_hit",
        "causal_improves_persistence",
        "oracle_improves_persistence",
    )
    if any(type(row[name]) is not bool for name in boolean_names):
        raise ValueError("metric predicate type differs")
    if row["finite"] is not True:
        raise ValueError("metric finite/cell record differs")
    if not isinstance(row["sse"], dict) or not isinstance(row["mse"], dict) or set(row["sse"]) != set(row["mse"]):
        raise ValueError("metric primitive maps differ")
    expected_names = {
        "persistence",
        "velocity",
        "causal",
        "history_shuffle",
        "reverse_sign",
        "oracle_full",
        "oracle_xfit",
        "target_shuffle_oracle",
        "source_reconstruction",
        "history_source_reconstruction",
        "source_identity",
    }
    if set(row["mse"]) != expected_names:
        raise ValueError("metric primitive names differ")
    mse_values: dict[str, float] = {}
    for name, value in cast(Mapping[str, object], row["mse"]).items():
        mse = _finite(value, f"{name} MSE", nonnegative=True)
        mse_values[name] = mse
        sse = _finite(cast(Mapping[str, object], row["sse"])[name], f"{name} SSE", nonnegative=True)
        if not math.isclose(mse * elements, sse, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("metric SSE/MSE relationship differs")

    p = mse_values["persistence"]
    o = mse_values["oracle_xfit"]
    s = mse_values["target_shuffle_oracle"]
    c = mse_values["causal"]
    h = mse_values["history_shuffle"]
    r = mse_values["reverse_sign"]
    velocity = mse_values["velocity"]
    a = mse_values["source_reconstruction"]
    source_null = mse_values["history_source_reconstruction"]
    b = mse_values["source_identity"]
    oracle_performance_support = bool(p > 0.0 and PERSISTENCE_FACTOR * o <= p)
    oracle_pairing_support = bool(oracle_performance_support and CONTROL_FACTOR * o <= s)
    oracle_support = oracle_pairing_support
    source_shuffle_null_hit = bool(CONTROL_FACTOR * a <= source_null)
    source_support = bool(b > 0.0 and SOURCE_FACTOR * a <= b and source_shuffle_null_hit)
    causal_performance_support = bool(oracle_performance_support and PERSISTENCE_FACTOR * c <= p and 2.0 * c <= p + o)
    causal_support = bool(
        oracle_pairing_support
        and causal_performance_support
        and CONTROL_FACTOR * c <= h
        and CONTROL_FACTOR * c <= r
        and CONTROL_FACTOR * c <= velocity
        and source_support
    )
    full_oracle_support = bool(p > 0.0 and PERSISTENCE_FACTOR * mse_values["oracle_full"] <= p)
    expected_predicates = {
        "oracle_performance_support": oracle_performance_support,
        "oracle_pairing_support": oracle_pairing_support,
        "oracle_support": oracle_support,
        "causal_performance_support": causal_performance_support,
        "causal_support": causal_support,
        "source_shuffle_null_hit": source_shuffle_null_hit,
        "source_support": source_support,
        "full_oracle_support": full_oracle_support,
        "full_oracle_only_support": bool(full_oracle_support and not oracle_performance_support),
        "target_shuffle_null_hit": bool(p > 0.0 and PERSISTENCE_FACTOR * s <= p),
        "causal_improves_persistence": bool(c < p),
        "oracle_improves_persistence": bool(o < p),
    }
    if any(row[name] is not expected for name, expected in expected_predicates.items()):
        raise ValueError("metric predicate does not replay from primitive MSE")
    _derived_close(row["causal_ratio"], c / p if p > 0.0 else None, "causal ratio")
    _derived_close(row["oracle_ratio"], o / p if p > 0.0 else None, "oracle ratio")
    _derived_close(
        row["full_oracle_ratio"],
        mse_values["oracle_full"] / p if p > 0.0 else None,
        "full oracle ratio",
    )
    known_endpoint = row["known_causal_flow_endpoint_mse"]
    if synthetic and row["scenario"] not in {"ambiguous", "source_null"}:
        _finite(known_endpoint, "known causal flow endpoint MSE", nonnegative=True)
    elif known_endpoint is not None:
        raise ValueError("known causal flow endpoint MSE appears without a unique synthetic flow")
    for field, label in (
        ("known_oracle_xfit_flow_endpoint_mse", "known cross-fitted oracle flow endpoint MSE"),
        ("known_oracle_full_flow_endpoint_mse", "known full oracle flow endpoint MSE"),
    ):
        endpoint = row[field]
        if synthetic and row["scenario"] in {"translation", "reversal", "stationary"}:
            _finite(endpoint, label, nonnegative=True)
        elif endpoint is not None:
            raise ValueError(f"{label} appears without a unique synthetic target flow")
    oracle_gap = p - o
    _derived_close(row["causal_capture"], (p - c) / oracle_gap if oracle_gap > 0.0 else 0.0, "causal capture")
    for name in (
        "causal_mean_flow_magnitude",
        "oracle_mean_flow_magnitude",
        "causal_confidence_gap",
        "oracle_confidence_gap",
        "flow_endpoint_mse",
        "cycle_endpoint_mse",
    ):
        _finite(row[name], name, nonnegative=True)
    for name in ("causal_boundary_fraction", "oracle_boundary_fraction"):
        fraction = _finite(row[name], name, nonnegative=True)
        if fraction > 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    cosine = _finite(row["flow_cosine"], "flow cosine")
    if cosine < -1.0 - 1e-12 or cosine > 1.0 + 1e-12:
        raise ValueError("flow cosine must be in [-1,1]")
    return row


def _validate_alignment(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"rows", "counts", "identity_sha256", "domains"}:
        raise ValueError("MM-006 alignment schema differs")
    if _exact_int(value["rows"], "alignment rows") != parent.MATCHED_ROWS or value["counts"] != dict(
        parent.MATCHED_COUNTS
    ):
        raise ValueError("MM-006 alignment membership differs")
    _fingerprint(value["identity_sha256"], "alignment identity")
    domains = value["domains"]
    if not isinstance(domains, dict) or set(domains) != {"pixel", "taesd"}:
        raise ValueError("MM-006 alignment domains differ")
    for domain, channels in (("pixel", 3), ("taesd", 4)):
        record = domains[domain]
        if not isinstance(record, dict) or set(record) != {
            "channels",
            "previous_sha256",
            "current_sha256",
            "target_0p5_sha256",
            "target_1p0_sha256",
        }:
            raise ValueError("MM-006 alignment domain schema differs")
        if _exact_int(record["channels"], f"{domain} channels") != channels:
            raise ValueError("MM-006 alignment channel count differs")
        for name in ("previous_sha256", "current_sha256", "target_0p5_sha256", "target_1p0_sha256"):
            _fingerprint(record[name], f"{domain} {name}")
    return dict(value)


def _validate_normalizers(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("MM-006 normalizer membership differs")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[object, object]] = []
    for candidate in value:
        if not isinstance(candidate, dict) or set(candidate) != {
            "domain",
            "fold",
            "train_video_ids",
            "test_video_ids",
            "train_rows",
            "mean",
            "scale",
            "fingerprint",
        }:
            raise ValueError("normalizer row schema differs")
        row = dict(candidate)
        domain = row["domain"]
        fold_index = _exact_int(row["fold"], "normalizer fold")
        if domain not in DOMAINS or fold_index not in range(4):
            raise ValueError("normalizer identity differs")
        fold = dataset.formal_folds()[fold_index]
        if row["train_video_ids"] != list(fold.train_ids) or row["test_video_ids"] != list(fold.test_ids):
            raise ValueError("normalizer fold membership differs")
        expected_rows = sum(parent.MATCHED_COUNTS[video_id] for video_id in fold.train_ids)
        if _exact_int(row["train_rows"], "normalizer train rows") != expected_rows:
            raise ValueError("normalizer train row count differs")
        channels = 3 if domain == "pixel" else 4
        if not isinstance(row["mean"], list) or not isinstance(row["scale"], list):
            raise ValueError("normalizer statistics must be arrays")
        if len(row["mean"]) != channels or len(row["scale"]) != channels:
            raise ValueError("normalizer channel statistics differ")
        mean = np.asarray([_finite(item, "normalizer mean") for item in row["mean"]], dtype=np.float64)
        scale = np.asarray(
            [_finite(item, "normalizer scale", nonnegative=True) for item in row["scale"]], dtype=np.float64
        )
        if np.any(scale < SCALE_FLOOR):
            raise ValueError("normalizer scale is below the frozen floor")
        expected_fingerprint = Normalizer(
            mean=mean.reshape(1, channels, 1, 1),
            scale=scale.reshape(1, channels, 1, 1),
        ).fingerprint()
        if _fingerprint(row["fingerprint"], "normalizer fingerprint") != expected_fingerprint:
            raise ValueError("normalizer fingerprint does not replay")
        identities.append((domain, fold_index))
        rows.append(row)
    if identities != [(domain, fold.index) for domain in DOMAINS for fold in dataset.formal_folds()]:
        raise ValueError("normalizer rows are incomplete, duplicated, or reordered")
    return rows


def validate_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "alignment",
        "normalizer_rows",
        "synthetic_panel_rows",
        "synthetic_metric_rows",
        "real_metric_rows",
        "activity_rows",
    }:
        raise ValueError("MM-006 evidence schema differs")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("MM-006 evidence version differs")
    alignment = _validate_alignment(value["alignment"])
    normalizer_rows = _validate_normalizers(value["normalizer_rows"])
    panels = value["synthetic_panel_rows"]
    if not isinstance(panels, list) or len(panels) != 36:
        raise ValueError("synthetic panel membership differs")
    normalized_panels: list[dict[str, Any]] = []
    panel_identities: list[tuple[object, object, object]] = []
    for candidate in panels:
        if not isinstance(candidate, dict) or set(candidate) != {
            "panel_seed",
            "channels",
            "scenario",
            "previous_sha256",
            "current_sha256",
            "target_sha256",
        }:
            raise ValueError("synthetic panel schema differs")
        row = dict(candidate)
        seed = _exact_int(row["panel_seed"], "synthetic panel seed")
        channels = _exact_int(row["channels"], "synthetic panel channels")
        if (
            seed not in SYNTHETIC_SEEDS
            or channels not in SYNTHETIC_CHANNELS
            or row["scenario"] not in SYNTHETIC_SCENARIOS
        ):
            raise ValueError("synthetic panel identity differs")
        for name in ("previous_sha256", "current_sha256", "target_sha256"):
            _fingerprint(row[name], name)
        panel_identities.append((seed, channels, row["scenario"]))
        normalized_panels.append(row)
    expected_panel_identities = [
        (seed, channels, scenario)
        for seed in SYNTHETIC_SEEDS
        for channels in SYNTHETIC_CHANNELS
        for scenario in SYNTHETIC_SCENARIOS
    ]
    if panel_identities != expected_panel_identities:
        raise ValueError("synthetic panels are incomplete, duplicated, or reordered")
    synthetic_rows = value["synthetic_metric_rows"]
    real_rows = value["real_metric_rows"]
    if not isinstance(synthetic_rows, list) or len(synthetic_rows) != 576:
        raise ValueError("synthetic metric membership differs")
    if not isinstance(real_rows, list) or len(real_rows) != 32:
        raise ValueError("real metric membership differs")
    normalized_synthetic = [_validate_metric_row(row, synthetic=True) for row in synthetic_rows]
    normalized_real = [_validate_metric_row(row, synthetic=False) for row in real_rows]

    expected_real_identities = [
        (domain, fold.index, video_id, family)
        for domain in DOMAINS
        for fold in dataset.formal_folds()
        for video_id in fold.test_ids
        for family in FLOW_FAMILIES
    ]
    real_identities = [(row["domain"], row["fold"], row["video_id"], row["family"]) for row in normalized_real]
    if real_identities != expected_real_identities:
        raise ValueError("real metrics are incomplete, duplicated, or reordered")
    expected_synthetic_identities = [
        (seed, channels, scenario, fold.index, video_id, family)
        for seed in SYNTHETIC_SEEDS
        for channels in SYNTHETIC_CHANNELS
        for scenario in SYNTHETIC_SCENARIOS
        for fold in dataset.formal_folds()
        for video_id in fold.test_ids
        for family in FLOW_FAMILIES
    ]
    synthetic_identities = [
        (
            row["panel_seed"],
            row["channels"],
            row["scenario"],
            row["fold"],
            row["video_id"],
            row["family"],
        )
        for row in normalized_synthetic
    ]
    if synthetic_identities != expected_synthetic_identities:
        raise ValueError("synthetic metrics are incomplete, duplicated, or reordered")

    real_normalizers = {(row["domain"], row["fold"]): row["fingerprint"] for row in normalizer_rows}
    scope_bindings: dict[tuple[object, ...], tuple[object, ...]] = {}
    normalizer_bindings: dict[tuple[object, ...], object] = {}
    for synthetic, rows in ((False, normalized_real), (True, normalized_synthetic)):
        for row in rows:
            scope = (
                row["domain"],
                row["panel_seed"],
                row["scenario"],
                row["fold"],
                row["video_id"],
            )
            binding = tuple(
                row[name]
                for name in (
                    "normalizer_fingerprint",
                    "source_sha256",
                    "target_sha256",
                    "causal_input_sha256",
                    "history_input_sha256",
                    "persistence_prediction_sha256",
                    "velocity_prediction_sha256",
                    "source_identity_prediction_sha256",
                    "source_xfit_input_sha256",
                    "history_source_xfit_input_sha256",
                    "oracle_full_input_sha256",
                    "oracle_xfit_input_sha256",
                    "target_shuffle_input_sha256",
                )
            ) + tuple(
                cast(Mapping[str, object], row["mse"])[name] for name in ("persistence", "velocity", "source_identity")
            )
            if scope in scope_bindings and scope_bindings[scope] != binding:
                raise ValueError("metric families do not share source, target, normalizer, or baseline primitives")
            scope_bindings[scope] = binding
            normalizer_scope = (
                row["domain"],
                row["panel_seed"],
                row["scenario"],
                row["fold"],
            )
            if (
                normalizer_scope in normalizer_bindings
                and normalizer_bindings[normalizer_scope] != row["normalizer_fingerprint"]
            ):
                raise ValueError("metrics in one fold do not share a normalizer")
            normalizer_bindings[normalizer_scope] = row["normalizer_fingerprint"]
            if not synthetic and row["normalizer_fingerprint"] != real_normalizers[(row["domain"], row["fold"])]:
                raise ValueError("real metric normalizer differs from its fold record")

    activity = value["activity_rows"]
    if not isinstance(activity, list) or len(activity) != 8:
        raise ValueError("activity membership differs")
    normalized_activity: list[dict[str, Any]] = []
    for video_id, candidate in zip(dataset.SAMPLE_VIDEO_IDS, activity, strict=True):
        if not isinstance(candidate, dict) or set(candidate) != {
            "video_id",
            "rows",
            "persistence_mse",
            "shuffled_mse",
            "activity_ratio",
            "active",
        }:
            raise ValueError("activity row schema differs")
        row = dict(candidate)
        if row["video_id"] != video_id or _exact_int(row["rows"], "activity rows") != parent.MATCHED_COUNTS[video_id]:
            raise ValueError("activity row identity differs")
        persistence = _finite(row["persistence_mse"], "activity persistence", nonnegative=True)
        shuffled = _finite(row["shuffled_mse"], "activity shuffled target", nonnegative=True)
        ratio = persistence / max(shuffled, 1e-15)
        _derived_close(row["activity_ratio"], ratio, "activity ratio")
        expected_active = bool(persistence >= ACTIVITY_MSE_MIN and ACTIVITY_RATIO_MIN <= ratio <= ACTIVITY_RATIO_MAX)
        if type(row["active"]) is not bool or row["active"] is not expected_active:
            raise ValueError("activity predicate does not replay")
        normalized_activity.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "alignment": alignment,
        "normalizer_rows": normalizer_rows,
        "synthetic_panel_rows": normalized_panels,
        "synthetic_metric_rows": normalized_synthetic,
        "real_metric_rows": normalized_real,
        "activity_rows": normalized_activity,
    }


def _family_summary(rows: Sequence[Mapping[str, Any]], domain: str, family: str) -> dict[str, Any]:
    selected = [row for row in rows if row["domain"] == domain and row["family"] == family]
    if len(selected) != 8:
        raise ValueError("real family summary requires eight videos")

    def fold_coverage(predicate: str) -> bool:
        return all(
            any(bool(row[predicate]) for row in selected if row["fold"] == fold.index)
            for fold in dataset.formal_folds()
        )

    oracle_performance_support = sum(bool(row["oracle_performance_support"]) for row in selected)
    oracle_support = sum(bool(row["oracle_support"]) for row in selected)
    causal_performance_support = sum(bool(row["causal_performance_support"]) for row in selected)
    causal_support = sum(bool(row["causal_support"]) for row in selected)
    source_support = sum(bool(row["source_support"]) for row in selected)
    full_oracle_support = sum(bool(row["full_oracle_support"]) for row in selected)
    full_only = sum(bool(row["full_oracle_only_support"]) for row in selected)
    oracle_improvement = sum(bool(row["oracle_improves_persistence"]) for row in selected)
    causal_improvement = sum(bool(row["causal_improves_persistence"]) for row in selected)
    full_oracle_improvement = sum(
        float(cast(Mapping[str, float], row["mse"])["oracle_full"])
        < float(cast(Mapping[str, float], row["mse"])["persistence"])
        for row in selected
    )
    source_improvement = sum(
        float(cast(Mapping[str, float], row["mse"])["source_reconstruction"])
        < float(cast(Mapping[str, float], row["mse"])["source_identity"])
        for row in selected
    )
    source_null_improvement = sum(
        float(cast(Mapping[str, float], row["mse"])["source_reconstruction"])
        < float(cast(Mapping[str, float], row["mse"])["history_source_reconstruction"])
        for row in selected
    )
    oracle_performance_passes = bool(
        oracle_performance_support >= REQUIRED_VIDEO_SUPPORT
        and oracle_improvement >= REQUIRED_ANY_IMPROVEMENT
        and fold_coverage("oracle_performance_support")
    )
    oracle_passes = bool(
        oracle_support >= REQUIRED_VIDEO_SUPPORT
        and oracle_improvement >= REQUIRED_ANY_IMPROVEMENT
        and fold_coverage("oracle_support")
    )
    causal_performance_passes = bool(
        causal_performance_support >= REQUIRED_VIDEO_SUPPORT
        and causal_improvement >= REQUIRED_ANY_IMPROVEMENT
        and fold_coverage("causal_performance_support")
    )
    causal_passes = bool(
        causal_support >= REQUIRED_VIDEO_SUPPORT
        and causal_improvement >= REQUIRED_ANY_IMPROVEMENT
        and fold_coverage("causal_support")
    )
    full_oracle_fold_coverage = all(
        any(bool(row["full_oracle_support"]) for row in selected if row["fold"] == fold.index)
        for fold in dataset.formal_folds()
    )
    full_oracle_passes = bool(
        full_oracle_support >= REQUIRED_VIDEO_SUPPORT
        and full_oracle_improvement >= REQUIRED_ANY_IMPROVEMENT
        and full_oracle_fold_coverage
    )
    source_passes = bool(
        source_support >= REQUIRED_VIDEO_SUPPORT
        and source_improvement >= REQUIRED_ANY_IMPROVEMENT
        and source_null_improvement >= REQUIRED_ANY_IMPROVEMENT
        and fold_coverage("source_support")
    )
    return {
        "domain": domain,
        "family": family,
        "oracle_performance_supporting_videos": oracle_performance_support,
        "oracle_supporting_videos": oracle_support,
        "causal_performance_supporting_videos": causal_performance_support,
        "causal_supporting_videos": causal_support,
        "source_supporting_videos": source_support,
        "full_oracle_supporting_videos": full_oracle_support,
        "full_oracle_only_videos": full_only,
        "oracle_improving_videos": oracle_improvement,
        "causal_improving_videos": causal_improvement,
        "full_oracle_improving_videos": full_oracle_improvement,
        "source_improving_videos": source_improvement,
        "source_null_improving_videos": source_null_improvement,
        "oracle_performance_fold_coverage": fold_coverage("oracle_performance_support"),
        "oracle_fold_coverage": fold_coverage("oracle_support"),
        "causal_performance_fold_coverage": fold_coverage("causal_performance_support"),
        "causal_fold_coverage": fold_coverage("causal_support"),
        "full_oracle_fold_coverage": full_oracle_fold_coverage,
        "source_fold_coverage": fold_coverage("source_support"),
        "oracle_performance_passes": oracle_performance_passes,
        "oracle_passes": oracle_passes,
        "causal_performance_passes": causal_performance_passes,
        "causal_passes": causal_passes,
        "full_oracle_passes": full_oracle_passes,
        "source_passes": source_passes,
        "boundary_warning_videos": sum(
            float(row["oracle_boundary_fraction"]) >= BOUNDARY_WARNING_FRACTION for row in selected
        ),
        "target_shuffle_null_support": sum(bool(row["target_shuffle_null_hit"]) for row in selected),
        "video_rows": selected,
    }


def _synthetic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    translation = [row for row in rows if row["scenario"] == "translation"]
    reversal = [row for row in rows if row["scenario"] == "reversal"]
    appearance = [row for row in rows if row["scenario"] == "appearance"]
    source_null = [row for row in rows if row["scenario"] == "source_null"]
    stationary = [row for row in rows if row["scenario"] == "stationary"]
    ambiguous = [row for row in rows if row["scenario"] == "ambiguous"]

    def translation_passes(row: Mapping[str, Any]) -> bool:
        mse = cast(Mapping[str, float], row["mse"])
        causal_endpoint = cast(float, row["known_causal_flow_endpoint_mse"])
        xfit_endpoint = cast(float, row["known_oracle_xfit_flow_endpoint_mse"])
        full_endpoint = cast(float, row["known_oracle_full_flow_endpoint_mse"])
        return bool(
            row["full_oracle_support"]
            and row["oracle_support"]
            and row["causal_support"]
            and causal_endpoint <= 1e-12
            and xfit_endpoint <= 1e-12
            and full_endpoint <= 1e-12
            and PERSISTENCE_FACTOR * mse["history_shuffle"] > mse["persistence"]
            and PERSISTENCE_FACTOR * mse["reverse_sign"] > mse["persistence"]
        )

    positive_failures = sum(not translation_passes(row) for row in translation)
    reversal_failures = sum(
        not (
            bool(row["full_oracle_support"])
            and bool(row["oracle_support"])
            and not bool(row["causal_performance_support"])
            and cast(float, row["known_oracle_xfit_flow_endpoint_mse"]) <= 1e-12
            and cast(float, row["known_oracle_full_flow_endpoint_mse"]) <= 1e-12
        )
        for row in reversal
    )
    appearance_failures = sum(
        bool(row["oracle_performance_support"]) or bool(row["causal_performance_support"]) for row in appearance
    )
    source_null_failures = sum(bool(row["source_shuffle_null_hit"]) for row in source_null)
    stationary_failures = sum(float(cast(Mapping[str, float], row["mse"])["persistence"]) != 0.0 for row in stationary)
    ambiguity_failures = sum(
        bool(row["causal_support"]) or float(row["causal_confidence_gap"]) > 1e-12 for row in ambiguous
    )
    return {
        "translation_conditions": len(translation),
        "reversal_conditions": len(reversal),
        "appearance_conditions": len(appearance),
        "source_null_conditions": len(source_null),
        "stationary_conditions": len(stationary),
        "ambiguous_conditions": len(ambiguous),
        "positive_failures": positive_failures,
        "reversal_branch_failures": reversal_failures,
        "appearance_branch_failures": appearance_failures,
        "source_null_branch_failures": source_null_failures,
        "stationary_branch_failures": stationary_failures,
        "ambiguity_branch_failures": ambiguity_failures,
        "positive_passes": positive_failures == 0,
        "negative_passes": (
            reversal_failures == 0
            and appearance_failures == 0
            and source_null_failures == 0
            and stationary_failures == 0
            and ambiguity_failures == 0
        ),
    }


def _decision(families: Sequence[Mapping[str, Any]], synthetic: Mapping[str, Any]) -> tuple[str, list[str]]:
    if not bool(synthetic["positive_passes"]):
        return "invalid_MM006_synthetic_positive_control", []
    if not bool(synthetic["negative_passes"]):
        return "invalid_MM006_synthetic_negative_control", []
    max_null = max(int(row["target_shuffle_null_support"]) for row in families)
    if max_null >= 6:
        return "invalid_MM006_real_negative_control", []
    if max_null >= 3:
        return "inconclusive_MM006_real_negative_control", []
    lookup = {(row["domain"], row["family"]): row for row in families}
    pixel = lookup[("pixel", PRIMARY_FAMILY)]
    taesd = lookup[("taesd", PRIMARY_FAMILY)]
    global_pixel = lookup[("pixel", "global_translation")]
    global_taesd = lookup[("taesd", "global_translation")]
    labels: list[str] = []
    if bool(global_pixel["causal_passes"]):
        labels.append("global_pixel_translation_sufficient")
    if bool(global_taesd["causal_passes"]):
        labels.append("global_taesd_translation_sufficient")
    for primary in (pixel, taesd):
        for prefix in ("oracle", "causal"):
            support = int(primary[f"{prefix}_supporting_videos"])
            performance_support = int(primary[f"{prefix}_performance_supporting_videos"])
            passes = bool(primary[f"{prefix}_passes"])
            performance_passes = bool(primary[f"{prefix}_performance_passes"])
            if (
                (3 <= support <= 5)
                or (support >= REQUIRED_VIDEO_SUPPORT and not passes)
                or (performance_support >= 3 and (not performance_passes or not passes))
            ):
                labels.append(f"{primary['domain']}_{prefix}_borderline_or_control_inconsistent")
                return "MM006_diagnostic_inconclusive", labels
    if (
        (bool(global_pixel["oracle_passes"]) and not bool(pixel["oracle_passes"]))
        or (bool(global_pixel["causal_passes"]) and not bool(pixel["causal_passes"]))
        or (bool(global_taesd["oracle_passes"]) and not bool(taesd["oracle_passes"]))
        or (bool(global_taesd["causal_passes"]) and not bool(taesd["causal_passes"]))
    ):
        labels.append("global_primary_inconsistency")
        return "MM006_diagnostic_inconclusive", labels
    if not bool(pixel["oracle_passes"]):
        oracle_support = int(pixel["oracle_supporting_videos"])
        if oracle_support >= 3:
            return "MM006_diagnostic_inconclusive", labels
        if bool(pixel["full_oracle_passes"]):
            return "target_fitted_oracle_overfit_supported", labels
        if int(pixel["full_oracle_supporting_videos"]) >= 3:
            return "MM006_diagnostic_inconclusive", labels
        if int(pixel["boundary_warning_videos"]) >= 3:
            return "tested_transport_range_inconclusive", labels
        return "tested_pixel_warp_ceiling_failure_supported", labels
    if not bool(pixel["causal_passes"]):
        causal_support = int(pixel["causal_supporting_videos"])
        if causal_support >= 3:
            return "MM006_diagnostic_inconclusive", labels
        if bool(pixel["source_passes"]):
            return "two_frame_motion_extrapolation_failure_supported", labels
        if int(pixel["source_supporting_videos"]) >= 3:
            return "MM006_diagnostic_inconclusive", labels
        return "low_resolution_correspondence_failure_supported", labels
    if not bool(taesd["oracle_passes"]):
        if int(taesd["oracle_supporting_videos"]) >= 3:
            return "MM006_diagnostic_inconclusive", labels
        return "taesd_transport_equivariance_failure_supported", labels
    if not bool(taesd["causal_passes"]):
        if int(taesd["causal_supporting_videos"]) >= 3:
            return "MM006_diagnostic_inconclusive", labels
        if bool(taesd["source_passes"]):
            return "taesd_two_frame_motion_extrapolation_failure_supported", labels
        if int(taesd["source_supporting_videos"]) >= 3:
            return "MM006_diagnostic_inconclusive", labels
        return "taesd_causal_correspondence_failure_supported", labels
    if bool(pixel["causal_passes"]) and bool(taesd["causal_passes"]):
        if not bool(global_pixel["causal_passes"]) or not bool(global_taesd["causal_passes"]):
            labels.append("spatially_varying_flow_required")
        return "single_step_causal_warp_fix_supported", labels
    return "MM006_diagnostic_inconclusive", labels


def summarize(evidence: object) -> dict[str, Any]:
    normalized = validate_evidence(evidence)
    synthetic = _synthetic_summary(cast(Sequence[Mapping[str, Any]], normalized["synthetic_metric_rows"]))
    real = cast(Sequence[Mapping[str, Any]], normalized["real_metric_rows"])
    families = [_family_summary(real, domain, family) for domain in DOMAINS for family in FLOW_FAMILIES]
    classification, labels = _decision(families, synthetic)
    activity_rows = cast(Sequence[Mapping[str, Any]], normalized["activity_rows"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "alignment": normalized["alignment"],
        "synthetic_control": synthetic,
        "families": families,
        "activity": {
            "supporting_videos": sum(bool(row["active"]) for row in activity_rows),
            "video_rows": activity_rows,
        },
        "decision": {
            "classification": classification,
            "mechanism_labels": labels,
            "recommended_next_step": RECOMMENDATIONS[classification],
        },
        "claim_boundary": (
            "MM-006 is an outcome-informed eight-video, one-step warp diagnostic. Its target-aware oracle is "
            "diagnostic only; even a causal pass does not establish independent-video or teacher-free rollout "
            "capability."
        ),
    }


def report_text(summary: Mapping[str, Any]) -> str:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("summary is not MM-006")
    decision = cast(Mapping[str, Any], summary["decision"])
    synthetic = cast(Mapping[str, Any], summary["synthetic_control"])
    lines = [
        "# MM-006 causal-warp ceiling report",
        "",
        "MM-006 is outcome-informed and does not reclassify MM-001 through MM-005.",
        "",
        f"Decision classification: `{decision['classification']}`.",
        f"Recommended next step: {decision['recommended_next_step']}.",
        "",
        "## Controls",
        "",
        f"Synthetic positive control: **{'PASS' if synthetic['positive_passes'] else 'FAIL'}**.",
        f"Synthetic negative control: **{'PASS' if synthetic['negative_passes'] else 'FAIL'}**.",
        "",
        "## Real transport results",
        "",
        "| Domain | Family | Oracle support | Causal support | Source support |",
        "|---|---|---:|---:|---:|",
    ]
    for row in cast(Sequence[Mapping[str, Any]], summary["families"]):
        lines.append(
            f"| `{row['domain']}` | `{row['family']}` | {row['oracle_supporting_videos']}/8 | "
            f"{row['causal_supporting_videos']}/8 | {row['source_supporting_videos']}/8 |"
        )
    labels = cast(Sequence[str], decision["mechanism_labels"])
    lines.extend(
        [
            "",
            f"Central pixel activity: {cast(Mapping[str, Any], summary['activity'])['supporting_videos']}/8 videos.",
            "",
            "Mechanism labels: " + (", ".join(f"`{label}`" for label in labels) if labels else "none"),
            "",
            str(summary["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def config_record() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matched_rows": parent.MATCHED_ROWS,
        "matched_counts": dict(parent.MATCHED_COUNTS),
        "matched_identity_sha256": parent.MATCHED_IDENTITY_SHA256,
        "horizon_seconds": 0.5,
        "domains": list(DOMAINS),
        "flow_families": list(FLOW_FAMILIES),
        "primary_family": PRIMARY_FAMILY,
        "warp_convention": "warp(X,d)[y,x] = X[y-dy,x-dx]",
        "candidate_values": list(_candidate_values),
        "candidate_order": [list(value) for value in CANDIDATES],
        "candidate_count": len(CANDIDATES),
        "central_slice": [CENTRAL_START, CENTRAL_STOP],
        "central_cells": CENTRAL_CELLS,
        "trim_fraction": TRIM_FRACTION,
        "local_regularizer": LOCAL_REGULARIZER,
        "normalizer": "one train-current channel normalizer per domain/fold, shared across arms and controls",
        "normalizer_scale_floor": SCALE_FLOOR,
        "oracle": {
            "full": "target-fitted diagnostic overfit measurement only",
            "cross_fit": "checkerboard complementary fit/evaluation",
            "uses_target": True,
            "diagnostic_only": True,
        },
        "causal": {
            "inputs": ["previous", "current"],
            "target_isolation_required": True,
            "primary_fit": "all central cells",
            "source_observability_fit": "checkerboard complementary fit/evaluation",
        },
        "controls": [
            "persistence",
            "constant_velocity",
            "history_shuffle",
            "reverse_sign",
            "target_shuffle",
            "source_identity",
            "history_source_xfit",
        ],
        "thresholds": {
            "persistence_factor": PERSISTENCE_FACTOR,
            "control_factor": CONTROL_FACTOR,
            "source_factor": SOURCE_FACTOR,
            "required_video_support": REQUIRED_VIDEO_SUPPORT,
            "required_any_improvement": REQUIRED_ANY_IMPROVEMENT,
            "fold_coverage_required": True,
            "boundary_warning_fraction": BOUNDARY_WARNING_FRACTION,
            "boundary_warning_videos": 3,
            "known_flow_endpoint_mse_max": 1e-12,
        },
        "synthetic": {
            "seeds": list(SYNTHETIC_SEEDS),
            "channels": list(SYNTHETIC_CHANNELS),
            "scenarios": list(SYNTHETIC_SCENARIOS),
            "generator": (
                "numpy.random.Generator(PCG64), analytic low-frequency textures, and target/source-independent "
                "Gaussian nulls"
            ),
        },
        "evidence_membership": {
            "normalizer_rows": 8,
            "synthetic_panel_rows": 36,
            "synthetic_metric_rows": 576,
            "real_metric_rows": 32,
            "activity_rows": 8,
        },
    }


__all__ = [
    "CANDIDATES",
    "EXPERIMENT_ID",
    "FLOW_FAMILIES",
    "PRIMARY_FAMILY",
    "SCHEMA_VERSION",
    "WarpPanel",
    "config_record",
    "execute",
    "report_text",
    "summarize",
    "synthetic_panel",
    "validate_evidence",
    "warp_panel",
]
