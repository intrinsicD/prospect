"""Pure-NumPy scientific method for MM-007.

MM-007 changes only spatial evidence.  Every resolution sees the same native
48x48 crop, 6x6 physical macrocell split, four physical quadrants, and 25
native-pixel translations.  The future-aware oracle is diagnostic only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from bench.multimodal_horizon_diagnostics import method as mm005
from bench.multimodal_preflight import dataset
from bench.multimodal_warp_diagnostics import method as mm006

SCHEMA_VERSION: Final = "mm007-method-v1"
EXPERIMENT_ID: Final = "MM-007"
RESOLUTIONS: Final = (8, 16, 32, 64)
FLOW_FAMILIES: Final = ("global_translation", "quadrant_flow")
PRIMARY_FAMILY: Final = "quadrant_flow"
RAW_ROWS: Final = 477
MATCHED_ROWS: Final = 453
CHANNELS: Final = 3
NATIVE_SIZE: Final = 64
NATIVE_BORDER: Final = 8
NATIVE_CENTRAL_SIZE: Final = 48
MACRO_SIDE: Final = 6
MACRO_COUNT: Final = 36
LOCAL_REGULARIZER: Final = 0.05
TRIM_FRACTION: Final = 0.25
PERSISTENCE_FACTOR: Final = 1.25
PAIRING_FACTOR: Final = 1.10
RELATIVE_FACTOR: Final = 1.10
REQUIRED_VIDEO_SUPPORT: Final = 6
REQUIRED_ANY_IMPROVEMENT: Final = 7
BOUNDARY_WARNING_FRACTION: Final = 0.25
SCALE_FLOOR: Final = 1e-6

_candidate_values = (-8.0, -4.0, 0.0, 4.0, 8.0)
NATIVE_CANDIDATES: Final = tuple(
    sorted(
        ((dy, dx) for dy in _candidate_values for dx in _candidate_values),
        key=lambda value: (value[0] * value[0] + value[1] * value[1], value[0], value[1]),
    )
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = sha256(b"mm007-array-v1")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolutionTable:
    """The exact 453-row half-second panel at four nested resolutions."""

    video_ids: np.ndarray
    timestamps: np.ndarray
    current_grids: tuple[np.ndarray, ...]
    target_grids: tuple[np.ndarray, ...]

    def current(self, resolution: int) -> np.ndarray:
        return np.asarray(self.current_grids[RESOLUTIONS.index(resolution)])

    def target(self, resolution: int) -> np.ndarray:
        return np.asarray(self.target_grids[RESOLUTIONS.index(resolution)])

    def validate(self, *, formal: bool = True) -> None:
        ids = np.asarray(self.video_ids, dtype=str)
        times = np.asarray(self.timestamps, dtype=np.float64)
        if ids.ndim != 1 or times.shape != ids.shape:
            raise ValueError("resolution identities differ")
        if len(self.current_grids) != len(RESOLUTIONS) or len(self.target_grids) != len(RESOLUTIONS):
            raise ValueError("resolution grid membership differs")
        for resolution, current, target in zip(
            RESOLUTIONS, self.current_grids, self.target_grids, strict=True
        ):
            left = np.asarray(current)
            right = np.asarray(target)
            if left.shape != (len(ids), CHANNELS, resolution, resolution) or right.shape != left.shape:
                raise ValueError("resolution grid shape differs")
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise ValueError("resolution grids contain non-finite values")
        if not formal:
            return
        if ids.shape != (MATCHED_ROWS,):
            raise ValueError("formal resolution table must contain 453 rows")
        expected = [
            (video_id, 1.5 + 0.5 * index)
            for video_id in dataset.SAMPLE_VIDEO_IDS
            for index in range(mm005.MATCHED_COUNTS[video_id])
        ]
        actual = list(zip(ids.tolist(), times.tolist(), strict=True))
        if actual != expected:
            raise ValueError("formal resolution identities differ")

    def subset(self, video_ids: Sequence[str]) -> ResolutionTable:
        wanted = set(video_ids)
        mask = np.asarray([str(value) in wanted for value in self.video_ids], dtype=bool)
        if not np.any(mask):
            raise ValueError("resolution subset is empty")
        output = ResolutionTable(
            video_ids=np.asarray(self.video_ids, dtype=str)[mask].copy(),
            timestamps=np.asarray(self.timestamps, dtype=np.float64)[mask].copy(),
            current_grids=tuple(np.asarray(value)[mask].copy() for value in self.current_grids),
            target_grids=tuple(np.asarray(value)[mask].copy() for value in self.target_grids),
        )
        output.validate(formal=False)
        return output


def _validate_raw_identities(video_ids: np.ndarray, timestamps: np.ndarray) -> None:
    ids = np.asarray(video_ids, dtype=str)
    times = np.asarray(timestamps, dtype=np.float64)
    if ids.shape != (RAW_ROWS,) or times.shape != (RAW_ROWS,) or not np.all(np.isfinite(times)):
        raise ValueError("raw frame identities must contain exactly 477 finite rows")
    expected = [
        (video_id, 1.0 + 0.5 * index)
        for video_id in dataset.SAMPLE_VIDEO_IDS
        for index in range(dataset.EXPECTED_WINDOW_COUNTS[video_id])
    ]
    if list(zip(ids.tolist(), times.tolist(), strict=True)) != expected:
        raise ValueError("raw frame identities differ")


def _pool_frames(frames_uint8: np.ndarray, resolution: int) -> np.ndarray:
    frames = np.asarray(frames_uint8)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (64, 64, 3):
        raise ValueError("frames_uint8 must be uint8 [N,64,64,3]")
    values: np.ndarray = frames.astype(np.float32) / np.float32(255.0)
    block = NATIVE_SIZE // resolution
    if block > 1:
        values = values.reshape(len(values), resolution, block, resolution, block, CHANNELS)
        values = np.asarray(
            np.mean(values, axis=(2, 4), dtype=np.float64), dtype=np.float32
        )
    return np.asarray(np.transpose(values, (0, 3, 1, 2)), dtype=np.float32)


def construct_table(
    video_ids: np.ndarray,
    timestamps: np.ndarray,
    frames_uint8: np.ndarray,
    expected_pixel_current_8: np.ndarray | None = None,
) -> ResolutionTable:
    """Construct the matched ladder and optionally enforce exact MM-004 R8 parity."""

    _validate_raw_identities(video_ids, timestamps)
    frames = np.asarray(frames_uint8)
    if frames.shape != (RAW_ROWS, 64, 64, 3) or frames.dtype != np.uint8:
        raise ValueError("formal frames_uint8 must be uint8 [477,64,64,3]")
    pooled = {resolution: _pool_frames(frames, resolution) for resolution in RESOLUTIONS}
    if expected_pixel_current_8 is not None and not np.array_equal(
        pooled[8], np.asarray(expected_pixel_current_8)
    ):
        raise ValueError("64-to-8 pixel parity differs from MM-004")
    ids = np.asarray(video_ids, dtype=str)
    times = np.asarray(timestamps, dtype=np.float64)
    source_indices: list[int] = []
    target_indices: list[int] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        rows = np.flatnonzero(ids == video_id)
        ordered = rows[np.argsort(times[rows], kind="stable")]
        for position in range(1, len(ordered) - 2):
            source_indices.append(int(ordered[position]))
            target_indices.append(int(ordered[position + 1]))
    source = np.asarray(source_indices, dtype=int)
    target = np.asarray(target_indices, dtype=int)
    table = ResolutionTable(
        video_ids=ids[source].copy(),
        timestamps=times[source].copy(),
        current_grids=tuple(pooled[resolution][source].copy() for resolution in RESOLUTIONS),
        target_grids=tuple(pooled[resolution][target].copy() for resolution in RESOLUTIONS),
    )
    table.validate()
    return table


raw_frame_table = construct_table


def load_table(path: str | Path, expected_pixel_current_8: np.ndarray | None = None) -> ResolutionTable:
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {"video_ids", "timestamps", "frames_uint8"}:
            raise ValueError("MM-007 frame archive membership differs")
        return construct_table(
            archive["video_ids"], archive["timestamps"], archive["frames_uint8"], expected_pixel_current_8
        )


@dataclass(frozen=True, slots=True)
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    def apply(self, value: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(value, dtype=np.float64) - self.mean) / self.scale, dtype=np.float64)

    def fingerprint(self) -> str:
        return _array_sha256(np.concatenate((self.mean.reshape(-1), self.scale.reshape(-1))))


def _fit_normalizer(current: np.ndarray) -> Normalizer:
    values = np.asarray(current, dtype=np.float64)
    mean = np.mean(values, axis=(0, 2, 3), keepdims=True)
    scale = np.maximum(np.std(values, axis=(0, 2, 3), keepdims=True), SCALE_FLOOR)
    return Normalizer(mean=mean, scale=scale)


@dataclass(frozen=True, slots=True)
class Geometry:
    resolution: int
    coords: np.ndarray
    macro_ids: np.ndarray
    parities: np.ndarray
    tile_ids: np.ndarray


def _geometry(resolution: int) -> Geometry:
    if resolution not in RESOLUTIONS:
        raise ValueError("unknown MM-007 resolution")
    scale = resolution // 8
    border = resolution // 8
    side = 6 * scale
    yy, xx = np.meshgrid(
        np.arange(border, border + side, dtype=np.float64),
        np.arange(border, border + side, dtype=np.float64),
        indexing="ij",
    )
    macro_y = ((yy.astype(int) - border) // scale).reshape(-1)
    macro_x = ((xx.astype(int) - border) // scale).reshape(-1)
    return Geometry(
        resolution=resolution,
        coords=np.stack((yy.reshape(-1), xx.reshape(-1)), axis=1),
        macro_ids=macro_y * MACRO_SIDE + macro_x,
        parities=(macro_y + macro_x) % 2,
        tile_ids=(macro_y >= 3).astype(int) * 2 + (macro_x >= 3).astype(int),
    )


def _sample_batch(grids: np.ndarray, coords: np.ndarray, native_flow: np.ndarray) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    points = np.asarray(coords, dtype=np.float64)
    flow = np.asarray(native_flow, dtype=np.float64)
    if values.ndim != 4 or flow.shape != (len(values), 2):
        raise ValueError("batch sampling shapes differ")
    scale = values.shape[2] / float(NATIVE_SIZE)
    source_y = points[None, :, 0] - flow[:, None, 0] * scale
    source_x = points[None, :, 1] - flow[:, None, 1] * scale
    if (
        np.min(source_y) < 0.0
        or np.min(source_x) < 0.0
        or np.max(source_y) > values.shape[2] - 1
        or np.max(source_x) > values.shape[3] - 1
    ):
        raise ValueError("warp samples out-of-bounds")
    y0 = np.floor(source_y).astype(int)
    x0 = np.floor(source_x).astype(int)
    y1 = np.minimum(y0 + 1, values.shape[2] - 1)
    x1 = np.minimum(x0 + 1, values.shape[3] - 1)
    wy = source_y - y0
    wx = source_x - x0
    batch = np.arange(len(values))[:, None, None]
    channels = np.arange(values.shape[1])[None, :, None]
    top = values[batch, channels, y0[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :] + values[
        batch, channels, y0[:, None, :], x1[:, None, :]
    ] * wx[:, None, :]
    bottom = values[batch, channels, y1[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :] + values[
        batch, channels, y1[:, None, :], x1[:, None, :]
    ] * wx[:, None, :]
    return cast(np.ndarray, top * (1.0 - wy)[:, None, :] + bottom * wy[:, None, :])


def _macro_trimmed_loss(residual: np.ndarray, macro_ids: np.ndarray) -> np.ndarray:
    site_loss = np.mean(np.asarray(residual, dtype=np.float64) ** 2, axis=1)
    unique = np.unique(np.asarray(macro_ids, dtype=int))
    macro_loss = np.stack([np.mean(site_loss[:, macro_ids == macro], axis=1) for macro in unique], axis=1)
    trim = int(math.floor(len(unique) * TRIM_FRACTION))
    retained = np.sort(macro_loss, axis=1, kind="stable")[:, : len(unique) - trim]
    return cast(np.ndarray, np.mean(retained, axis=1))


def _select_displacements(
    source: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    selected: np.ndarray,
    anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(selected, dtype=bool)
    coords = geometry.coords[mask]
    macros = geometry.macro_ids[mask]
    target_values = np.asarray(target, dtype=np.float64)[:, :, coords[:, 0].astype(int), coords[:, 1].astype(int)]
    candidates = np.asarray(NATIVE_CANDIDATES, dtype=np.float64)
    losses: list[np.ndarray] = []
    identity: np.ndarray | None = None
    for candidate in candidates:
        flows = np.broadcast_to(candidate, (len(source), 2))
        loss = _macro_trimmed_loss(_sample_batch(source, coords, flows) - target_values, macros)
        if candidate[0] == 0.0 and candidate[1] == 0.0:
            identity = loss.copy()
        losses.append(loss)
    if identity is None:
        raise AssertionError("identity candidate missing")
    loss_array = np.stack(losses, axis=1)
    if anchor is not None:
        anchors = np.asarray(anchor, dtype=np.float64)
        distance = np.sum((candidates[None, :, :] - anchors[:, None, :]) ** 2, axis=2) / 64.0
        loss_array = loss_array + LOCAL_REGULARIZER * (identity[:, None] + 1e-12) * distance
    best_index = np.argmin(loss_array, axis=1)
    best = candidates[best_index]
    best_loss = loss_array[np.arange(len(source)), best_index]
    separated = np.sum((candidates[None, :, :] - best[:, None, :]) ** 2, axis=2) >= 16.0
    alternative = np.min(np.where(separated, loss_array, np.inf), axis=1)
    return cast(np.ndarray, best), cast(np.ndarray, alternative - best_loss)


def _estimate_flows(
    source: np.ndarray,
    target: np.ndarray,
    geometry: Geometry,
    family: str,
    fit_macro_parity: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if family not in FLOW_FAMILIES:
        raise ValueError("unknown MM-007 flow family")
    fit = np.ones(len(geometry.coords), dtype=bool)
    if fit_macro_parity is not None:
        fit = geometry.parities == fit_macro_parity
    global_flow, global_confidence = _select_displacements(source, target, geometry, fit)
    if family == "global_translation":
        return global_flow[:, None, :], global_confidence[:, None]
    flows: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for tile in range(4):
        selected = fit & (geometry.tile_ids == tile)
        flow, confidence = _select_displacements(source, target, geometry, selected, global_flow)
        flows.append(flow)
        confidences.append(confidence)
    return np.stack(flows, axis=1), np.stack(confidences, axis=1)


def _predict(
    grids: np.ndarray,
    flows: np.ndarray,
    geometry: Geometry,
    family: str,
    output_macro_parity: int | None,
) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    selected = np.ones(len(geometry.coords), dtype=bool)
    if output_macro_parity is not None:
        selected = geometry.parities == output_macro_parity
    output = np.full((len(values), values.shape[1], len(geometry.coords)), np.nan, dtype=np.float64)
    if family == "global_translation":
        output[:, :, selected] = _sample_batch(values, geometry.coords[selected], flows[:, 0])
    elif family == "quadrant_flow":
        for tile in range(4):
            cells = selected & (geometry.tile_ids == tile)
            output[:, :, cells] = _sample_batch(values, geometry.coords[cells], flows[:, tile])
    else:
        raise ValueError("unknown MM-007 flow family")
    return output


@dataclass(frozen=True, slots=True)
class Estimate:
    flow: np.ndarray
    prediction: np.ndarray
    confidence: np.ndarray


def _estimate_full(source: np.ndarray, target: np.ndarray, resolution: int, family: str) -> Estimate:
    geometry = _geometry(resolution)
    flow, confidence = _estimate_flows(source, target, geometry, family, None)
    prediction = _predict(source, flow, geometry, family, None)
    return Estimate(flow=flow, prediction=prediction, confidence=confidence)


def _estimate_xfit(source: np.ndarray, target: np.ndarray, resolution: int, family: str) -> Estimate:
    geometry = _geometry(resolution)
    family_size = 1 if family == "global_translation" else 4
    flows = np.empty((len(source), 2, family_size, 2), dtype=np.float64)
    confidences = np.empty((len(source), 2, family_size), dtype=np.float64)
    prediction = np.full((len(source), source.shape[1], len(geometry.coords)), np.nan, dtype=np.float64)
    for output_parity in (0, 1):
        flow, confidence = _estimate_flows(source, target, geometry, family, 1 - output_parity)
        partial = _predict(source, flow, geometry, family, output_parity)
        selected = geometry.parities == output_parity
        prediction[:, :, selected] = partial[:, :, selected]
        flows[:, output_parity] = flow
        confidences[:, output_parity] = confidence
    if not np.all(np.isfinite(prediction)):
        raise ValueError("MM-007 cross-fit did not predict every cell")
    return Estimate(flow=flows, prediction=prediction, confidence=confidences)


def _near_derangement(video_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(video_ids, dtype=str)
    mapping = np.empty(len(ids), dtype=int)
    for video_id in tuple(dict.fromkeys(ids.tolist())):
        rows = np.flatnonzero(ids == video_id)
        cursor = 0
        while len(rows) - cursor > 3:
            left, right = int(rows[cursor]), int(rows[cursor + 1])
            mapping[left], mapping[right] = right, left
            cursor += 2
        remaining = rows[cursor:]
        if len(remaining) == 2:
            mapping[remaining[0]], mapping[remaining[1]] = int(remaining[1]), int(remaining[0])
        elif len(remaining) == 3:
            mapping[remaining[0]] = int(remaining[1])
            mapping[remaining[1]] = int(remaining[2])
            mapping[remaining[2]] = int(remaining[0])
        else:
            raise ValueError("near derangement requires at least two rows")
    if np.any(mapping == np.arange(len(ids))) or not np.array_equal(ids[mapping], ids):
        raise ValueError("near target derangement differs")
    return mapping


def _far_derangement(video_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(video_ids, dtype=str)
    mapping = np.empty(len(ids), dtype=int)
    for video_id in tuple(dict.fromkeys(ids.tolist())):
        rows = np.flatnonzero(ids == video_id)
        mapping[rows] = np.roll(rows, len(rows) // 2)
    if np.any(mapping == np.arange(len(ids))) or not np.array_equal(ids[mapping], ids):
        raise ValueError("far target derangement differs")
    return mapping


def _sse(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int, float]:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if not np.all(np.isfinite(difference)):
        raise ValueError("MM-007 metric contains non-finite values")
    numerator = float(np.sum(difference * difference))
    elements = int(difference.size)
    return numerator, elements, numerator / elements


def _metric_row(
    table: ResolutionTable,
    resolution: int,
    family: str,
    fold: dataset.DatasetFold,
    normalizer: Normalizer,
    *,
    synthetic_scenario: str | None = None,
    known_native_flow: np.ndarray | None = None,
) -> dict[str, Any]:
    table.validate(formal=False)
    geometry = _geometry(resolution)
    current = normalizer.apply(table.current(resolution))
    target = normalizer.apply(table.target(resolution))
    target_central = target[:, :, geometry.coords[:, 0].astype(int), geometry.coords[:, 1].astype(int)]
    persistence = current[:, :, geometry.coords[:, 0].astype(int), geometry.coords[:, 1].astype(int)]
    full = _estimate_full(current, target, resolution, family)
    xfit = _estimate_xfit(current, target, resolution, family)
    near = _estimate_xfit(current, target[_near_derangement(table.video_ids)], resolution, family)
    far = _estimate_xfit(current, target[_far_derangement(table.video_ids)], resolution, family)
    metrics = {
        "persistence": _sse(persistence, target_central),
        "oracle_full": _sse(full.prediction, target_central),
        "oracle_xfit": _sse(xfit.prediction, target_central),
        "near_target_oracle": _sse(near.prediction, target_central),
        "far_target_oracle": _sse(far.prediction, target_central),
    }
    if len({item[1] for item in metrics.values()}) != 1:
        raise ValueError("MM-007 metrics do not share one denominator")
    mse = {name: item[2] for name, item in metrics.items()}
    p = mse["persistence"]
    o = mse["oracle_xfit"]
    near_mse = mse["near_target_oracle"]
    far_mse = mse["far_target_oracle"]
    performance = bool(p > 0.0 and PERSISTENCE_FACTOR * o <= p)
    near_pairing = bool(performance and PAIRING_FACTOR * o <= near_mse)
    far_pairing = bool(performance and PAIRING_FACTOR * o <= far_mse)
    support = bool(near_pairing and far_pairing)
    full_support = bool(p > 0.0 and PERSISTENCE_FACTOR * mse["oracle_full"] <= p)
    boundary = float(np.mean(np.isclose(np.abs(xfit.flow), 8.0)))
    endpoint: float | None = None
    full_endpoint: float | None = None
    if known_native_flow is not None:
        expected = np.asarray(known_native_flow, dtype=np.float64)
        if expected.shape != (len(current), 2):
            raise ValueError("known native flow shape differs")
        expanded = np.broadcast_to(expected[:, None, None, :], xfit.flow.shape)
        endpoint = float(np.mean(np.sum((xfit.flow - expanded) ** 2, axis=-1)))
        full_expanded = np.broadcast_to(expected[:, None, :], full.flow.shape)
        full_endpoint = float(
            np.mean(np.sum((full.flow - full_expanded) ** 2, axis=-1))
        )
    return {
        "resolution": resolution,
        "family": family,
        "fold": fold.index,
        "video_id": str(table.video_ids[0]),
        "rows": len(table.video_ids),
        "channels": CHANNELS,
        "macrocell_count": MACRO_COUNT,
        "candidate_count": len(NATIVE_CANDIDATES),
        "elements": next(iter(item[1] for item in metrics.values())),
        "normalizer_fingerprint": normalizer.fingerprint(),
        "synthetic_scenario": synthetic_scenario,
        "sse": {name: item[0] for name, item in metrics.items()},
        "mse": mse,
        "oracle_ratio": o / p if p > 0.0 else None,
        "full_oracle_ratio": mse["oracle_full"] / p if p > 0.0 else None,
        "oracle_improves_persistence": bool(o < p),
        "oracle_performance_support": performance,
        "near_pairing_support": near_pairing,
        "far_pairing_support": far_pairing,
        "oracle_support": support,
        "full_oracle_support": full_support,
        "full_oracle_only_support": bool(full_support and not performance),
        "near_target_null_hit": bool(p > 0.0 and PERSISTENCE_FACTOR * near_mse <= p),
        "far_target_null_hit": bool(p > 0.0 and PERSISTENCE_FACTOR * far_mse <= p),
        "oracle_boundary_fraction": boundary,
        "oracle_confidence_gap": float(np.mean(xfit.confidence)),
        "known_native_flow_endpoint_mse": endpoint,
        "known_native_flow_full_endpoint_mse": full_endpoint,
        "current_sha256": _array_sha256(current),
        "target_sha256": _array_sha256(target),
        "oracle_full_flow_sha256": _array_sha256(full.flow),
        "oracle_full_prediction_sha256": _array_sha256(full.prediction),
        "oracle_xfit_flow_sha256": _array_sha256(xfit.flow),
        "oracle_xfit_prediction_sha256": _array_sha256(xfit.prediction),
        "near_target_flow_sha256": _array_sha256(near.flow),
        "near_target_prediction_sha256": _array_sha256(near.prediction),
        "far_target_flow_sha256": _array_sha256(far.flow),
        "far_target_prediction_sha256": _array_sha256(far.prediction),
        "oracle_uses_target": True,
        "oracle_diagnostic_only": True,
        "finite": True,
    }


def _normalizers(table: ResolutionTable) -> tuple[list[dict[str, Any]], dict[tuple[int, int], Normalizer]]:
    records: list[dict[str, Any]] = []
    output: dict[tuple[int, int], Normalizer] = {}
    ids = np.asarray(table.video_ids, dtype=str)
    shared: dict[int, tuple[np.ndarray, Normalizer]] = {}
    for fold in dataset.formal_folds():
        train = np.asarray([video_id in fold.train_ids for video_id in ids], dtype=bool)
        shared[fold.index] = (train, _fit_normalizer(table.current(8)[train]))
    for resolution in RESOLUTIONS:
        for fold in dataset.formal_folds():
            train, normalizer = shared[fold.index]
            output[(resolution, fold.index)] = normalizer
            records.append(
                {
                    "resolution": resolution,
                    "fold": fold.index,
                    "train_video_ids": list(fold.train_ids),
                    "test_video_ids": list(fold.test_ids),
                    "train_rows": int(np.sum(train)),
                    "mean": normalizer.mean.reshape(-1).tolist(),
                    "scale": normalizer.scale.reshape(-1).tolist(),
                    "fingerprint": normalizer.fingerprint(),
                    "uses_target": False,
                }
            )
    return records, output


def _alignment(table: ResolutionTable) -> dict[str, Any]:
    return {
        "rows": len(table.video_ids),
        "counts": {
            video_id: int(np.sum(np.asarray(table.video_ids, dtype=str) == video_id))
            for video_id in dataset.SAMPLE_VIDEO_IDS
        },
        "identity_sha256": _canonical_sha256(
            list(zip(np.asarray(table.video_ids, dtype=str).tolist(), table.timestamps.tolist(), strict=True))
        ),
        "resolutions": {
            str(resolution): {
                "current_sha256": _array_sha256(table.current(resolution)),
                "target_sha256": _array_sha256(table.target(resolution)),
                "shape": list(table.current(resolution).shape[1:]),
            }
            for resolution in RESOLUTIONS
        },
    }


def _parent_pixel_rows(parent: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, Any]]:
    summary = cast(Mapping[str, Any], parent.get("summary", {}))
    families = cast(Sequence[Mapping[str, Any]], summary.get("families", []))
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for family in families:
        if family.get("domain") != "pixel":
            continue
        for row in cast(Sequence[Mapping[str, Any]], family["video_rows"]):
            rows[(cast(str, family["family"]), cast(str, row["video_id"]))] = row
    if len(rows) != 16:
        raise ValueError("MM-006 pixel parent rows differ")
    return rows


def _assert_r8_replay(row: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    mse = cast(Mapping[str, float], row["mse"])
    parent_mse = cast(Mapping[str, float], parent["mse"])
    pairs = {
        "persistence": "persistence",
        "oracle_full": "oracle_full",
        "oracle_xfit": "oracle_xfit",
        "far_target_oracle": "target_shuffle_oracle",
    }
    for current_name, parent_name in pairs.items():
        if not math.isclose(mse[current_name], parent_mse[parent_name], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"R8 MM-006 {current_name} parity differs")
    for name in (
        "oracle_performance_support",
        "oracle_support",
        "full_oracle_support",
        "full_oracle_only_support",
        "oracle_improves_persistence",
    ):
        if row[name] is not parent[name]:
            raise ValueError(f"R8 MM-006 {name} parity differs")
    if not math.isclose(
        float(row["oracle_boundary_fraction"]), float(parent["oracle_boundary_fraction"]), rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("R8 MM-006 boundary parity differs")


def execute(table: ResolutionTable, parent_evidence: object) -> dict[str, object]:
    """Execute the frozen ladder; every real support unit is one video."""

    table.validate()
    parent = mm006.validate_evidence(parent_evidence)
    parent_rows = _parent_pixel_rows({"summary": mm006.summarize(parent)})
    normalizer_rows, normalizers = _normalizers(table)
    metric_rows: list[dict[str, Any]] = []
    for resolution in RESOLUTIONS:
        for fold in dataset.formal_folds():
            for video_id in fold.test_ids:
                video = table.subset([video_id])
                for family in FLOW_FAMILIES:
                    row = _metric_row(video, resolution, family, fold, normalizers[(resolution, fold.index)])
                    if resolution == 8:
                        _assert_r8_replay(row, parent_rows[(family, video_id)])
                    metric_rows.append(row)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "alignment": _alignment(table),
        "normalizer_rows": normalizer_rows,
        "real_metric_rows": metric_rows,
        "synthetic_rows": _synthetic_rows(),
        "synthetic_seed_map": dict(SYNTHETIC_SEED_MAP),
        "synthetic_expectations": deepcopy(SYNTHETIC_EXPECTATIONS),
        "parent_classification": cast(Mapping[str, Any], mm006.summarize(parent)["decision"])["classification"],
    }
    return validate_evidence(evidence)


SYNTHETIC_SCENARIOS: Final = ("translation", "stationary", "appearance", "alias_recovery")
SYNTHETIC_SEED_MAP: Final = {
    scenario: 700_700 + index for index, scenario in enumerate(SYNTHETIC_SCENARIOS)
}


def _build_synthetic_expectations() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    expectations: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    native_flows: dict[str, list[float] | None] = {
        "translation": [0.0, 8.0],
        "stationary": [0.0, 0.0],
        "appearance": None,
        "alias_recovery": [0.0, -4.0],
    }
    for scenario in SYNTHETIC_SCENARIOS:
        by_resolution: dict[str, dict[str, dict[str, Any]]] = {}
        for resolution in RESOLUTIONS:
            recovered = scenario == "translation" or (
                scenario == "alias_recovery" and resolution > 8
            )
            persistence_zero = scenario == "stationary" or (
                scenario == "alias_recovery" and resolution == 8
            )
            endpoint: float | None
            if scenario == "appearance":
                endpoint = None
            elif scenario == "alias_recovery" and resolution == 8:
                endpoint = 16.0
            else:
                endpoint = 0.0
            by_family: dict[str, dict[str, Any]] = {}
            for family in FLOW_FAMILIES:
                by_family[family] = {
                    "expected_native_flow": deepcopy(native_flows[scenario]),
                    "predicates": {
                        "persistence_zero": persistence_zero,
                        "oracle_performance_support": recovered,
                        "oracle_support": recovered,
                        "full_oracle_support": recovered,
                    },
                    "known_native_flow_endpoint_mse": endpoint,
                    "known_native_flow_full_endpoint_mse": endpoint,
                }
            by_resolution[str(resolution)] = by_family
        expectations[scenario] = by_resolution
    return expectations


SYNTHETIC_EXPECTATIONS: Final = _build_synthetic_expectations()


def _pool_float_frames(frames: np.ndarray, resolution: int) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float64)
    block = NATIVE_SIZE // resolution
    if block > 1:
        values = values.reshape(len(values), resolution, block, resolution, block, CHANNELS)
        values = np.mean(values, axis=(2, 4))
    return np.transpose(values, (0, 3, 1, 2))


def _synthetic_table(scenario: str) -> tuple[ResolutionTable, np.ndarray | None]:
    if scenario not in SYNTHETIC_SCENARIOS:
        raise ValueError("unknown MM-007 synthetic scenario")
    rows = 6
    rng = np.random.Generator(np.random.PCG64(SYNTHETIC_SEED_MAP[scenario]))
    if scenario == "alias_recovery":
        # An exact eight-pixel-periodic, zero-mean carrier vanishes under R8
        # pooling.  R16+ retains its two half-periods and recovers the signed
        # tie-broken -4 px shift exactly.  Row-varying amplitudes keep the
        # near/far target controls discriminating.
        carrier = np.tile(np.asarray([1.0] * 4 + [-1.0] * 4), 8)[None, None, :, None]
        amplitude = rng.integers(1, 5, size=(rows, 8, 1, CHANNELS)).astype(np.float64)
        current64 = np.repeat(amplitude, 8, axis=1) * carrier
        known: np.ndarray | None = np.broadcast_to(
            np.asarray([0.0, -4.0]), (rows, 2)
        ).copy()
    else:
        coarse = rng.normal(size=(rows, 8, 8, CHANNELS))
        current64 = np.repeat(np.repeat(coarse, 8, axis=1), 8, axis=2)
        known = np.broadcast_to(np.asarray([0.0, 8.0]), (rows, 2)).copy()
    target64 = current64.copy()
    if scenario in {"translation", "alias_recovery"}:
        assert known is not None
        geometry = _geometry(64)
        sampled = _sample_batch(np.transpose(current64, (0, 3, 1, 2)), geometry.coords, known)
        y = geometry.coords[:, 0].astype(int)
        x = geometry.coords[:, 1].astype(int)
        target_chw = np.transpose(target64, (0, 3, 1, 2))
        target_chw[:, :, y, x] = sampled
        target64 = np.transpose(target_chw, (0, 2, 3, 1))
    elif scenario == "appearance":
        # A row/channel-wise photometric offset cannot be repaired by spatial
        # translation; persistence is the exact spatial baseline.
        offset = rng.choice((-3.0, 3.0), size=(rows, 1, 1, CHANNELS))
        target64 = current64 + offset
        known = None
    elif scenario == "stationary":
        known = np.zeros((rows, 2), dtype=np.float64)
    ids = np.asarray(["synthetic"] * rows)
    times = np.arange(rows, dtype=np.float64) * 0.5
    table = ResolutionTable(
        video_ids=ids,
        timestamps=times,
        current_grids=tuple(_pool_float_frames(current64, resolution) for resolution in RESOLUTIONS),
        target_grids=tuple(_pool_float_frames(target64, resolution) for resolution in RESOLUTIONS),
    )
    table.validate(formal=False)
    return table, known


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fold = dataset.formal_folds()[0]
    for scenario in SYNTHETIC_SCENARIOS:
        table, known = _synthetic_table(scenario)
        normalizer = _fit_normalizer(table.current(8))
        for resolution in RESOLUTIONS:
            for family in FLOW_FAMILIES:
                rows.append(
                    _metric_row(
                        table,
                        resolution,
                        family,
                        fold,
                        normalizer,
                        synthetic_scenario=scenario,
                        known_native_flow=known,
                    )
                )
    return rows


_METRIC_KEYS: Final = {
    "resolution",
    "family",
    "fold",
    "video_id",
    "rows",
    "channels",
    "macrocell_count",
    "candidate_count",
    "elements",
    "normalizer_fingerprint",
    "synthetic_scenario",
    "sse",
    "mse",
    "oracle_ratio",
    "full_oracle_ratio",
    "oracle_improves_persistence",
    "oracle_performance_support",
    "near_pairing_support",
    "far_pairing_support",
    "oracle_support",
    "full_oracle_support",
    "full_oracle_only_support",
    "near_target_null_hit",
    "far_target_null_hit",
    "oracle_boundary_fraction",
    "oracle_confidence_gap",
    "known_native_flow_endpoint_mse",
    "known_native_flow_full_endpoint_mse",
    "current_sha256",
    "target_sha256",
    "oracle_full_flow_sha256",
    "oracle_full_prediction_sha256",
    "oracle_xfit_flow_sha256",
    "oracle_xfit_prediction_sha256",
    "near_target_flow_sha256",
    "near_target_prediction_sha256",
    "far_target_flow_sha256",
    "far_target_prediction_sha256",
    "oracle_uses_target",
    "oracle_diagnostic_only",
    "finite",
}
_METRIC_NAMES: Final = {
    "persistence",
    "oracle_full",
    "oracle_xfit",
    "near_target_oracle",
    "far_target_oracle",
}


def _exact_int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and cast(int, value) <= 0):
        raise ValueError(f"{name} must be an exact integer")
    return cast(int, value)


def _finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    output = float(value)
    if nonnegative and output < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return output


def _validate_metric_row(value: object, *, synthetic: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _METRIC_KEYS:
        raise ValueError("MM-007 metric schema differs")
    row = dict(value)
    resolution = _exact_int(row["resolution"], "resolution", positive=True)
    if resolution not in RESOLUTIONS or row["family"] not in FLOW_FAMILIES:
        raise ValueError("MM-007 metric scope differs")
    _exact_int(row["fold"], "fold")
    if not isinstance(row["video_id"], str) or not row["video_id"]:
        raise ValueError("MM-007 metric video identity differs")
    _exact_int(row["rows"], "rows", positive=True)
    if row["channels"] != CHANNELS or row["macrocell_count"] != MACRO_COUNT:
        raise ValueError("MM-007 metric dimensions differ")
    if row["candidate_count"] != len(NATIVE_CANDIDATES):
        raise ValueError("MM-007 candidate count differs")
    elements = _exact_int(row["elements"], "elements", positive=True)
    scenario = row["synthetic_scenario"]
    if (synthetic and scenario not in SYNTHETIC_SCENARIOS) or (not synthetic and scenario is not None):
        raise ValueError("MM-007 synthetic label differs")
    if not isinstance(row["sse"], dict) or not isinstance(row["mse"], dict):
        raise ValueError("MM-007 metric numerators differ")
    sse = cast(dict[str, object], row["sse"])
    mse = cast(dict[str, object], row["mse"])
    if set(sse) != _METRIC_NAMES or set(mse) != _METRIC_NAMES:
        raise ValueError("MM-007 MSE membership differs")
    values: dict[str, float] = {}
    for name in _METRIC_NAMES:
        numerator = _finite_float(sse[name], f"{name} SSE", nonnegative=True)
        mean = _finite_float(mse[name], f"{name} MSE", nonnegative=True)
        if not math.isclose(numerator / elements, mean, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("MM-007 SSE/MSE does not replay")
        values[name] = mean
    p = values["persistence"]
    o = values["oracle_xfit"]
    near = values["near_target_oracle"]
    far = values["far_target_oracle"]
    performance = bool(p > 0.0 and PERSISTENCE_FACTOR * o <= p)
    near_pair = bool(performance and PAIRING_FACTOR * o <= near)
    far_pair = bool(performance and PAIRING_FACTOR * o <= far)
    predicates = {
        "oracle_improves_persistence": o < p,
        "oracle_performance_support": performance,
        "near_pairing_support": near_pair,
        "far_pairing_support": far_pair,
        "oracle_support": near_pair and far_pair,
        "full_oracle_support": p > 0.0 and PERSISTENCE_FACTOR * values["oracle_full"] <= p,
        "near_target_null_hit": p > 0.0 and PERSISTENCE_FACTOR * near <= p,
        "far_target_null_hit": p > 0.0 and PERSISTENCE_FACTOR * far <= p,
    }
    predicates["full_oracle_only_support"] = bool(
        predicates["full_oracle_support"] and not performance
    )
    for name, expected in predicates.items():
        if type(row[name]) is not bool or row[name] is not expected:
            raise ValueError(f"MM-007 {name} predicate does not replay")
    expected_ratio = o / p if p > 0.0 else None
    expected_full_ratio = values["oracle_full"] / p if p > 0.0 else None
    for name, ratio_expected in (
        ("oracle_ratio", expected_ratio),
        ("full_oracle_ratio", expected_full_ratio),
    ):
        if ratio_expected is None:
            if row[name] is not None:
                raise ValueError(f"MM-007 {name} zero-baseline handling differs")
        elif not math.isclose(
            _finite_float(row[name], name),
            ratio_expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"MM-007 {name} does not replay")
    boundary = _finite_float(row["oracle_boundary_fraction"], "boundary", nonnegative=True)
    if boundary > 1.0:
        raise ValueError("MM-007 boundary fraction exceeds one")
    _finite_float(row["oracle_confidence_gap"], "confidence", nonnegative=True)
    for name in (
        "known_native_flow_endpoint_mse",
        "known_native_flow_full_endpoint_mse",
    ):
        endpoint = row[name]
        if endpoint is not None:
            _finite_float(endpoint, "known flow endpoint", nonnegative=True)
    for name in _METRIC_KEYS:
        if name.endswith("_sha256"):
            candidate = row[name]
            if not isinstance(candidate, str) or len(candidate) != 64:
                raise ValueError("MM-007 fingerprint differs")
    if row["oracle_uses_target"] is not True or row["oracle_diagnostic_only"] is not True:
        raise ValueError("MM-007 oracle leakage label differs")
    if row["finite"] is not True:
        raise ValueError("MM-007 finite flag differs")
    return row


def _validate_synthetic_seed_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(SYNTHETIC_SCENARIOS):
        raise ValueError("MM-007 synthetic seed membership differs")
    for scenario, expected in SYNTHETIC_SEED_MAP.items():
        if type(value[scenario]) is not int or value[scenario] != expected:
            raise ValueError(f"MM-007 synthetic seed differs: {scenario}")
    return dict(SYNTHETIC_SEED_MAP)


def _validate_synthetic_expectations(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or _canonical_sha256(value) != _canonical_sha256(
        SYNTHETIC_EXPECTATIONS
    ):
        raise ValueError("MM-007 synthetic expectations differ")
    return deepcopy(SYNTHETIC_EXPECTATIONS)


def _validate_synthetic_normalizer_sharing(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for scenario in SYNTHETIC_SCENARIOS:
        fingerprints = {
            row["normalizer_fingerprint"]
            for row in rows
            if row["synthetic_scenario"] == scenario
        }
        if len(fingerprints) != 1:
            raise ValueError("MM-007 synthetic R8 normalizer is not shared across resolutions")


def validate_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "alignment",
        "normalizer_rows",
        "real_metric_rows",
        "synthetic_rows",
        "synthetic_seed_map",
        "synthetic_expectations",
        "parent_classification",
    }:
        raise ValueError("MM-007 evidence schema differs")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("MM-007 evidence version differs")
    alignment = value["alignment"]
    if not isinstance(alignment, dict) or set(alignment) != {
        "rows",
        "counts",
        "identity_sha256",
        "resolutions",
    }:
        raise ValueError("MM-007 alignment schema differs")
    if alignment["rows"] != MATCHED_ROWS or alignment["counts"] != dict(mm005.MATCHED_COUNTS):
        raise ValueError("MM-007 alignment membership differs")
    if not isinstance(alignment["identity_sha256"], str) or len(alignment["identity_sha256"]) != 64:
        raise ValueError("MM-007 alignment identity differs")
    resolutions = alignment["resolutions"]
    if not isinstance(resolutions, dict) or set(resolutions) != {str(value) for value in RESOLUTIONS}:
        raise ValueError("MM-007 alignment resolutions differ")
    expected_identity = _canonical_sha256(
        [
            (video_id, 1.5 + 0.5 * index)
            for video_id in dataset.SAMPLE_VIDEO_IDS
            for index in range(mm005.MATCHED_COUNTS[video_id])
        ]
    )
    if alignment["identity_sha256"] != expected_identity:
        raise ValueError("MM-007 alignment identity does not replay")
    for resolution in RESOLUTIONS:
        record = resolutions[str(resolution)]
        if not isinstance(record, dict) or set(record) != {"current_sha256", "target_sha256", "shape"}:
            raise ValueError("MM-007 alignment resolution schema differs")
        if record["shape"] != [CHANNELS, resolution, resolution]:
            raise ValueError("MM-007 alignment resolution shape differs")
        for name in ("current_sha256", "target_sha256"):
            if not isinstance(record[name], str) or len(record[name]) != 64:
                raise ValueError("MM-007 alignment resolution hash differs")
    normalizer_rows = value["normalizer_rows"]
    if not isinstance(normalizer_rows, list) or len(normalizer_rows) != len(RESOLUTIONS) * 4:
        raise ValueError("MM-007 normalizer membership differs")
    seen_normalizers: set[tuple[int, int]] = set()
    normalizer_fingerprints: dict[tuple[int, int], str] = {}
    for candidate in normalizer_rows:
        if not isinstance(candidate, dict) or set(candidate) != {
            "resolution",
            "fold",
            "train_video_ids",
            "test_video_ids",
            "train_rows",
            "mean",
            "scale",
            "fingerprint",
            "uses_target",
        }:
            raise ValueError("MM-007 normalizer schema differs")
        key = (_exact_int(candidate["resolution"], "normalizer resolution"), _exact_int(candidate["fold"], "fold"))
        if key in seen_normalizers or key[0] not in RESOLUTIONS or key[1] not in range(4):
            raise ValueError("MM-007 normalizer scope differs")
        seen_normalizers.add(key)
        fold = dataset.formal_folds()[key[1]]
        if candidate["train_video_ids"] != list(fold.train_ids) or candidate["test_video_ids"] != list(fold.test_ids):
            raise ValueError("MM-007 normalizer fold membership differs")
        expected_train_rows = sum(mm005.MATCHED_COUNTS[video_id] for video_id in fold.train_ids)
        if candidate["train_rows"] != expected_train_rows:
            raise ValueError("MM-007 normalizer train rows differ")
        if candidate["uses_target"] is not False:
            raise ValueError("MM-007 normalizer used target")
        if not isinstance(candidate["mean"], list) or len(candidate["mean"]) != CHANNELS:
            raise ValueError("MM-007 normalizer mean differs")
        if not isinstance(candidate["scale"], list) or len(candidate["scale"]) != CHANNELS:
            raise ValueError("MM-007 normalizer scale differs")
        for number in [*candidate["mean"], *candidate["scale"]]:
            _finite_float(number, "normalizer value")
        if any(float(number) < SCALE_FLOOR for number in candidate["scale"]):
            raise ValueError("MM-007 normalizer scale below floor")
        reconstructed = Normalizer(
            mean=np.asarray(candidate["mean"], dtype=np.float64).reshape(1, CHANNELS, 1, 1),
            scale=np.asarray(candidate["scale"], dtype=np.float64).reshape(1, CHANNELS, 1, 1),
        )
        if candidate["fingerprint"] != reconstructed.fingerprint():
            raise ValueError("MM-007 normalizer fingerprint does not replay")
        normalizer_fingerprints[key] = reconstructed.fingerprint()
    if seen_normalizers != {(resolution, fold) for resolution in RESOLUTIONS for fold in range(4)}:
        raise ValueError("MM-007 normalizer scopes are incomplete")
    for fold_index in range(4):
        if len(
            {
                normalizer_fingerprints[(resolution, fold_index)]
                for resolution in RESOLUTIONS
            }
        ) != 1:
            raise ValueError("MM-007 R8 normalizer is not shared across resolutions")
    real = value["real_metric_rows"]
    synthetic = value["synthetic_rows"]
    if not isinstance(real, list) or len(real) != len(RESOLUTIONS) * len(FLOW_FAMILIES) * 8:
        raise ValueError("MM-007 real metric membership differs")
    expected_synthetic_rows = (
        len(RESOLUTIONS) * len(FLOW_FAMILIES) * len(SYNTHETIC_SCENARIOS)
    )
    if not isinstance(synthetic, list) or len(synthetic) != expected_synthetic_rows:
        raise ValueError("MM-007 synthetic metric membership differs")
    normalized_real = [_validate_metric_row(row, synthetic=False) for row in real]
    normalized_synthetic = [_validate_metric_row(row, synthetic=True) for row in synthetic]
    fold_by_video = {
        video_id: fold.index for fold in dataset.formal_folds() for video_id in fold.test_ids
    }
    for row in normalized_real:
        resolution = int(row["resolution"])
        video_id = str(row["video_id"])
        if row["fold"] != fold_by_video.get(video_id):
            raise ValueError("MM-007 real fold/video membership differs")
        if row["rows"] != mm005.MATCHED_COUNTS[video_id]:
            raise ValueError("MM-007 real row count differs")
        side = 6 * (resolution // 8)
        if row["elements"] != row["rows"] * CHANNELS * side * side:
            raise ValueError("MM-007 real element count differs")
        if row["normalizer_fingerprint"] != normalizer_fingerprints[(resolution, int(row["fold"]))]:
            raise ValueError("MM-007 real normalizer binding differs")
    for row in normalized_synthetic:
        resolution = int(row["resolution"])
        side = 6 * (resolution // 8)
        if row["fold"] != 0 or row["rows"] != 6 or row["elements"] != 6 * CHANNELS * side * side:
            raise ValueError("MM-007 synthetic row dimensions differ")
    real_scopes = {(row["resolution"], row["family"], row["video_id"]) for row in normalized_real}
    expected_scopes = {
        (resolution, family, video_id)
        for resolution in RESOLUTIONS
        for family in FLOW_FAMILIES
        for video_id in dataset.SAMPLE_VIDEO_IDS
    }
    if real_scopes != expected_scopes:
        raise ValueError("MM-007 real scopes differ")
    synthetic_scopes = {
        (row["resolution"], row["family"], row["synthetic_scenario"])
        for row in normalized_synthetic
    }
    expected_synthetic = {
        (resolution, family, scenario)
        for resolution in RESOLUTIONS
        for family in FLOW_FAMILIES
        for scenario in SYNTHETIC_SCENARIOS
    }
    if synthetic_scopes != expected_synthetic:
        raise ValueError("MM-007 synthetic scopes differ")
    _validate_synthetic_normalizer_sharing(normalized_synthetic)
    synthetic_seed_map = _validate_synthetic_seed_map(value["synthetic_seed_map"])
    synthetic_expectations = _validate_synthetic_expectations(
        value["synthetic_expectations"]
    )
    if not isinstance(value["parent_classification"], str):
        raise ValueError("MM-007 parent classification differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "alignment": alignment,
        "normalizer_rows": normalizer_rows,
        "real_metric_rows": normalized_real,
        "synthetic_rows": normalized_synthetic,
        "synthetic_seed_map": synthetic_seed_map,
        "synthetic_expectations": synthetic_expectations,
        "parent_classification": value["parent_classification"],
    }


def _fold_coverage(rows: Sequence[Mapping[str, Any]], predicate: str) -> bool:
    return all(
        any(bool(row[predicate]) for row in rows if row["fold"] == fold.index)
        for fold in dataset.formal_folds()
    )


def _family_summary(
    rows: Sequence[Mapping[str, Any]], resolution: int, family: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["resolution"] == resolution and row["family"] == family]
    if len(selected) != 8:
        raise ValueError("MM-007 family requires eight videos")
    performance = sum(bool(row["oracle_performance_support"]) for row in selected)
    support = sum(bool(row["oracle_support"]) for row in selected)
    full = sum(bool(row["full_oracle_support"]) for row in selected)
    full_only = sum(bool(row["full_oracle_only_support"]) for row in selected)
    improvement = sum(bool(row["oracle_improves_persistence"]) for row in selected)
    full_improvement = sum(
        float(cast(Mapping[str, float], row["mse"])["oracle_full"])
        < float(cast(Mapping[str, float], row["mse"])["persistence"])
        for row in selected
    )
    performance_passes = bool(
        performance >= REQUIRED_VIDEO_SUPPORT
        and improvement >= REQUIRED_ANY_IMPROVEMENT
        and _fold_coverage(selected, "oracle_performance_support")
    )
    passes = bool(
        support >= REQUIRED_VIDEO_SUPPORT
        and improvement >= REQUIRED_ANY_IMPROVEMENT
        and _fold_coverage(selected, "oracle_support")
    )
    full_passes = bool(
        full >= REQUIRED_VIDEO_SUPPORT
        and full_improvement >= REQUIRED_ANY_IMPROVEMENT
        and _fold_coverage(selected, "full_oracle_support")
    )
    return {
        "resolution": resolution,
        "family": family,
        "oracle_performance_supporting_videos": performance,
        "oracle_supporting_videos": support,
        "full_oracle_supporting_videos": full,
        "full_oracle_only_videos": full_only,
        "oracle_improving_videos": improvement,
        "full_oracle_improving_videos": full_improvement,
        "oracle_performance_fold_coverage": _fold_coverage(selected, "oracle_performance_support"),
        "oracle_fold_coverage": _fold_coverage(selected, "oracle_support"),
        "full_oracle_fold_coverage": _fold_coverage(selected, "full_oracle_support"),
        "oracle_performance_passes": performance_passes,
        "oracle_passes": passes,
        "full_oracle_passes": full_passes,
        "boundary_warning_videos": sum(
            float(row["oracle_boundary_fraction"]) >= BOUNDARY_WARNING_FRACTION for row in selected
        ),
        "near_target_null_support": sum(bool(row["near_target_null_hit"]) for row in selected),
        "far_target_null_support": sum(bool(row["far_target_null_hit"]) for row in selected),
        "video_rows": selected,
    }


def _relative_summary(rows: Sequence[Mapping[str, Any]], resolution: int) -> dict[str, Any]:
    lookup = {
        (int(row["resolution"]), str(row["video_id"])): row
        for row in rows
        if row["family"] == PRIMARY_FAMILY
    }
    video_rows: list[dict[str, Any]] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        baseline = lookup[(8, video_id)]
        candidate = lookup[(resolution, video_id)]
        baseline_ratio = float(cast(float, baseline["oracle_ratio"]))
        candidate_ratio = float(cast(float, candidate["oracle_ratio"]))
        video_rows.append(
            {
                "video_id": video_id,
                "fold": candidate["fold"],
                "baseline_ratio": baseline_ratio,
                "candidate_ratio": candidate_ratio,
                "improves": candidate_ratio < baseline_ratio,
                "support": RELATIVE_FACTOR * candidate_ratio <= baseline_ratio,
            }
        )
    support = sum(bool(row["support"]) for row in video_rows)
    improvement = sum(bool(row["improves"]) for row in video_rows)
    fold_coverage = all(
        any(bool(row["support"]) for row in video_rows if row["fold"] == fold.index)
        for fold in dataset.formal_folds()
    )
    return {
        "resolution": resolution,
        "supporting_videos": support,
        "improving_videos": improvement,
        "fold_coverage": fold_coverage,
        "passes": bool(
            support >= REQUIRED_VIDEO_SUPPORT
            and improvement >= REQUIRED_ANY_IMPROVEMENT
            and fold_coverage
        ),
        "video_rows": video_rows,
    }


def _synthetic_expectation_failure(row: Mapping[str, Any]) -> bool:
    scenario = cast(str, row["synthetic_scenario"])
    resolution = int(row["resolution"])
    family = cast(str, row["family"])
    expected = SYNTHETIC_EXPECTATIONS[scenario][str(resolution)][family]
    predicates = cast(Mapping[str, bool], expected["predicates"])
    persistence = float(cast(Mapping[str, float], row["mse"])["persistence"])
    if (persistence == 0.0) is not predicates["persistence_zero"]:
        return True
    for name in (
        "oracle_performance_support",
        "oracle_support",
        "full_oracle_support",
    ):
        if bool(row[name]) is not predicates[name]:
            return True
    for name in (
        "known_native_flow_endpoint_mse",
        "known_native_flow_full_endpoint_mse",
    ):
        actual = row[name]
        expected_endpoint = cast(float | None, expected[name])
        if expected_endpoint is None:
            if actual is not None:
                return True
        elif actual is None or not math.isclose(
            float(cast(float, actual)),
            expected_endpoint,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return True
    return False


def _synthetic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scenario = {
        scenario: [row for row in rows if row["synthetic_scenario"] == scenario]
        for scenario in SYNTHETIC_SCENARIOS
    }
    failures = {
        scenario: sum(_synthetic_expectation_failure(row) for row in selected)
        for scenario, selected in by_scenario.items()
    }
    translation = by_scenario["translation"]
    stationary = by_scenario["stationary"]
    appearance = by_scenario["appearance"]
    alias = by_scenario["alias_recovery"]
    translation_failures = failures["translation"]
    stationary_failures = failures["stationary"]
    appearance_failures = failures["appearance"]
    alias_failures = failures["alias_recovery"]
    return {
        "translation_conditions": len(translation),
        "stationary_conditions": len(stationary),
        "appearance_conditions": len(appearance),
        "alias_conditions": len(alias),
        "translation_failures": translation_failures,
        "stationary_failures": stationary_failures,
        "appearance_failures": appearance_failures,
        "alias_recovery_failures": alias_failures,
        "positive_passes": translation_failures == 0,
        "negative_passes": stationary_failures == 0 and appearance_failures == 0,
        "alias_recovery_passes": alias_failures == 0,
    }


def _decision(
    families: Sequence[Mapping[str, Any]],
    relative: Sequence[Mapping[str, Any]],
    synthetic: Mapping[str, Any],
) -> tuple[str, int | None, list[str]]:
    if not bool(synthetic["positive_passes"]):
        return "invalid_MM007_synthetic_translation_control", None, []
    if not bool(synthetic["negative_passes"]):
        return "invalid_MM007_synthetic_negative_control", None, []
    if not bool(synthetic["alias_recovery_passes"]):
        return "invalid_MM007_alias_recovery_control", None, []
    lookup = {(row["resolution"], row["family"]): row for row in families}
    relative_lookup = {row["resolution"]: row for row in relative}
    labels: list[str] = []
    max_null = max(
        max(int(row["near_target_null_support"]), int(row["far_target_null_support"]))
        for row in families
    )
    if max_null >= 6:
        return "invalid_MM007_real_pairing_control", None, []
    if max_null >= 3:
        return "MM007_real_pairing_control_inconclusive", None, []
    for resolution in RESOLUTIONS:
        primary = lookup[(resolution, PRIMARY_FAMILY)]
        global_row = lookup[(resolution, "global_translation")]
        if int(primary["boundary_warning_videos"]) >= 3:
            return "MM007_transport_range_inconclusive", None, []
        for key in ("oracle_supporting_videos", "oracle_performance_supporting_videos"):
            count = int(primary[key])
            expected_pass = "oracle_passes" if key == "oracle_supporting_videos" else "oracle_performance_passes"
            if 3 <= count <= 5 or (count >= 6 and not bool(primary[expected_pass])):
                return "MM007_resolution_response_inconclusive", None, []
        if bool(primary["oracle_performance_passes"]) and not bool(primary["oracle_passes"]):
            labels.append(f"performance_pairing_disagreement_R{resolution}")
            return "MM007_resolution_response_inconclusive", None, labels
        full_count = int(primary["full_oracle_supporting_videos"])
        if (
            3 <= full_count <= 5
            or int(primary["full_oracle_only_videos"]) >= 3
            or (full_count >= 6 and not bool(primary["full_oracle_passes"]))
            or (bool(primary["oracle_passes"]) and not bool(primary["full_oracle_passes"]))
        ):
            return "MM007_resolution_response_inconclusive", None, []
        if bool(global_row["oracle_passes"]) and not bool(primary["oracle_passes"]):
            labels.append(f"global_primary_inconsistency_R{resolution}")
            return "MM007_resolution_response_inconclusive", None, labels
    passes = {
        resolution: bool(lookup[(resolution, PRIMARY_FAMILY)]["oracle_passes"])
        and bool(lookup[(resolution, PRIMARY_FAMILY)]["full_oracle_passes"])
        for resolution in RESOLUTIONS
    }
    relative_passes = {resolution: bool(relative_lookup[resolution]["passes"]) for resolution in (16, 32, 64)}
    if passes[16] and passes[32] and passes[64] and all(relative_passes.values()):
        return "resolution_recovery_at_16_supported", 16, labels
    if not passes[16] and passes[32] and passes[64] and relative_passes[32] and relative_passes[64]:
        return "resolution_recovery_at_32_supported", 32, labels
    clean_failure = all(
        int(lookup[(resolution, PRIMARY_FAMILY)]["oracle_supporting_videos"]) < 3
        and int(lookup[(resolution, PRIMARY_FAMILY)]["full_oracle_supporting_videos"]) < 3
        for resolution in (16, 32, 64)
    )
    if clean_failure:
        return "physically_matched_resolution_failure_supported", None, labels
    return "MM007_resolution_response_inconclusive", None, labels


_RECOMMENDATIONS: Final = {
    "invalid_MM007_synthetic_translation_control": "repair physically matched signed-flow recovery",
    "invalid_MM007_synthetic_negative_control": "repair stationary or appearance controls",
    "invalid_MM007_alias_recovery_control": "repair the resolution-sensitivity assay",
    "invalid_MM007_real_pairing_control": "reject the real assay because shuffled targets cross the null gate",
    "MM007_real_pairing_control_inconclusive": "strengthen true-pair discrimination",
    "MM007_transport_range_inconclusive": "test a wider physical displacement range",
    "MM007_resolution_response_inconclusive": "replicate the resolution response before selecting a mechanism",
    "resolution_recovery_at_16_supported": "retain at least a 16x16 correspondence frontend, then test causal flow",
    "resolution_recovery_at_32_supported": "retain at least a 32x32 correspondence frontend, then test causal flow",
    "physically_matched_resolution_failure_supported": "run a separate frozen deformation-versus-appearance diagnostic",
}


def summarize(evidence: object) -> dict[str, Any]:
    normalized = validate_evidence(evidence)
    real = cast(Sequence[Mapping[str, Any]], normalized["real_metric_rows"])
    families = [
        _family_summary(real, resolution, family)
        for resolution in RESOLUTIONS
        for family in FLOW_FAMILIES
    ]
    relative = [_relative_summary(real, resolution) for resolution in (16, 32, 64)]
    synthetic = _synthetic_summary(cast(Sequence[Mapping[str, Any]], normalized["synthetic_rows"]))
    classification, onset, labels = _decision(families, relative, synthetic)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "alignment": normalized["alignment"],
        "parent_classification": normalized["parent_classification"],
        "synthetic_seed_map": normalized["synthetic_seed_map"],
        "synthetic_expectations": normalized["synthetic_expectations"],
        "synthetic_control": synthetic,
        "families": families,
        "relative_to_r8": relative,
        "decision": {
            "classification": classification,
            "onset_resolution": onset,
            "mechanism_labels": labels,
            "recommended_next_step": _RECOMMENDATIONS[classification],
        },
        "claim_boundary": (
            "MM-007 is an outcome-informed target-aware pixel-oracle diagnostic. A resolution pass does not "
            "establish target-isolated correspondence, causal prediction, independent-video generalization, or rollout."
        ),
    }


def report_text(summary: Mapping[str, Any]) -> str:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("summary is not MM-007")
    decision = cast(Mapping[str, Any], summary["decision"])
    synthetic = cast(Mapping[str, Any], summary["synthetic_control"])
    lines = [
        "# MM-007 physically matched resolution report",
        "",
        f"Decision classification: `{decision['classification']}`.",
        f"Onset resolution: `{decision['onset_resolution']}`.",
        f"Recommended next step: {decision['recommended_next_step']}.",
        "",
        "## Controls",
        "",
        f"Synthetic translation: **{'PASS' if synthetic['positive_passes'] else 'FAIL'}**.",
        f"Synthetic negatives: **{'PASS' if synthetic['negative_passes'] else 'FAIL'}**.",
        f"Alias recovery: **{'PASS' if synthetic['alias_recovery_passes'] else 'FAIL'}**.",
        "",
        "## Resolution ladder",
        "",
        "| Resolution | Family | Oracle support | Full support | Near null | Far null |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for family in cast(Sequence[Mapping[str, Any]], summary["families"]):
        lines.append(
            f"| {family['resolution']} | `{family['family']}` | {family['oracle_supporting_videos']}/8 | "
            f"{family['full_oracle_supporting_videos']}/8 | {family['near_target_null_support']}/8 | "
            f"{family['far_target_null_support']}/8 |"
        )
    lines.extend(("", cast(str, summary["claim_boundary"]), ""))
    return "\n".join(lines)


def frozen_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "resolutions": list(RESOLUTIONS),
        "native_frame": {"side": 64, "central_crop": [8, 56], "central_side": 48},
        "macrocells": {
            "shape": [6, 6],
            "native_side": 8,
            "checkerboard_crossfit": True,
            "trim_fraction": TRIM_FRACTION,
        },
        "flow": {
            "families": list(FLOW_FAMILIES),
            "primary_family": PRIMARY_FAMILY,
            "native_candidates": [list(value) for value in NATIVE_CANDIDATES],
            "local_regularizer": LOCAL_REGULARIZER,
            "regularizer_distance_divisor": 64.0,
        },
        "oracle": {
            "uses_target": True,
            "diagnostic_only": True,
            "arms": ["full", "crossfit", "near_target_crossfit", "far_target_crossfit", "persistence"],
        },
        "normalizer": {
            "fit": "R8 training-video current only",
            "reference_resolution": 8,
            "shared_across_resolutions": True,
            "scale_floor": SCALE_FLOOR,
        },
        "gates": {
            "persistence_factor": PERSISTENCE_FACTOR,
            "pairing_factor": PAIRING_FACTOR,
            "relative_factor": RELATIVE_FACTOR,
            "required_video_support": REQUIRED_VIDEO_SUPPORT,
            "required_any_improvement": REQUIRED_ANY_IMPROVEMENT,
            "boundary_warning_fraction": BOUNDARY_WARNING_FRACTION,
        },
        "synthetic_scenarios": list(SYNTHETIC_SCENARIOS),
        "synthetic_seed_map": dict(SYNTHETIC_SEED_MAP),
        "synthetic_expectations": deepcopy(SYNTHETIC_EXPECTATIONS),
        "causal_arms_present": False,
        "deformation_arms_present": False,
        "residual_arms_present": False,
    }


config_record = frozen_config


__all__ = [
    "EXPERIMENT_ID",
    "FLOW_FAMILIES",
    "NATIVE_CANDIDATES",
    "PRIMARY_FAMILY",
    "RESOLUTIONS",
    "ResolutionTable",
    "SCHEMA_VERSION",
    "SYNTHETIC_EXPECTATIONS",
    "SYNTHETIC_SEED_MAP",
    "construct_table",
    "execute",
    "frozen_config",
    "load_table",
    "raw_frame_table",
    "report_text",
    "summarize",
    "validate_evidence",
]
