"""Pure NumPy scientific engine for MM-004.

The module owns no filesystem or media operations.  It consumes authenticated grid
arrays, derives causal history panels, runs deterministic local ridge probes, and
reduces primitive rows through the frozen MM-004 decision tree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final, cast

import numpy as np

from bench.multimodal_preflight import dataset
from bench.multimodal_transform_diagnostics import method as mm003

SCHEMA_VERSION: Final = "mm004-method-v1"
RIDGE_PENALTY: Final = 1e-3
SCALE_FLOOR: Final = 1e-6
LINEAR_RESIDUAL_MAX: Final = 1e-10
KERNEL_ERROR_MAX: Final = 0.05
SYNTHETIC_SEEDS: Final = (440_040, 440_041, 440_042)
SYNTHETIC_DIFFERENCE_SCALE: Final = 0.35
SYNTHETIC_SMOOTHING_PASSES: Final = 2
REQUIRED_VIDEO_SUPPORT: Final = 6
REAL_PERSISTENCE_FACTOR: Final = 1.20
REAL_SHUFFLE_FACTOR: Final = 1.10
REAL_CONTRAST_FACTOR: Final = 1.10
SYNTHETIC_PERSISTENCE_RATIO: Final = 0.10
SYNTHETIC_SEPARATION_RATIO: Final = 0.50
ACTIVITY_MSE_MIN: Final = 1e-4
ACTIVITY_RATIO_MIN: Final = 0.10
ACTIVITY_RATIO_MAX: Final = 1.0 / 1.2
CENTRAL_SIZE: Final = 6
PATCHES_PER_ROW: Final = CENTRAL_SIZE * CENTRAL_SIZE
DOMAINS: Final = ("taesd", "pixel")
CONTROL_IDS: Final = ("ordered", "target_shuffle", "history_shuffle")
RECOMMENDATIONS: Final = {
    "invalid_MM004_synthetic_positive_control": "repair the assay before interpreting real videos",
    "invalid_MM004_synthetic_negative_control": "repair ablation or temporal-control separation",
    "invalid_MM004_real_negative_control": "reject the real assay because a temporal null crosses",
    "inconclusive_MM004_real_negative_control": "strengthen or independently replicate temporal controls",
    "taesd_local_linear_signal_supported": ("implement only supported factors in a tiny adapter and test end to end"),
    "taesd_representation_failure_supported": ("replace or temporally fine-tune the visual frontend"),
    "tested_local_objective_or_horizon_failure_supported": (
        "change the local objective or horizon before changing data"
    ),
    "data_dynamics_insufficient_for_local_history_assay": (
        "curate deliberately dynamic clips before changing model size"
    ),
    "inconclusive_video_heterogeneity": "enlarge the independently sampled video panel first",
    "MM004_diagnostic_inconclusive": ("do not select a mechanism without a new discriminating control"),
}


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One frozen local-linear predictor."""

    arm_id: str
    patch_size: int
    uses_history: bool

    def feature_dim(self, channels: int) -> int:
        multiplier = 2 if self.uses_history else 1
        return channels * self.patch_size * self.patch_size * multiplier


ARMS: Final = (
    ArmSpec("current_1x1", 1, False),
    ArmSpec("current_diff_1x1", 1, True),
    ArmSpec("current_3x3", 3, False),
    ArmSpec("current_diff_3x3", 3, True),
)
ARM_BY_ID: Final = {arm.arm_id: arm for arm in ARMS}
MAIN_ARM_ID: Final = "current_diff_3x3"

HISTORY_COUNTS: Final = {
    video_id: dataset.EXPECTED_WINDOW_COUNTS[video_id] - 1 for video_id in dataset.SAMPLE_VIDEO_IDS
}
HISTORY_ROWS: Final = sum(HISTORY_COUNTS.values())


@dataclass(frozen=True, slots=True)
class RawGridTable:
    """Authenticated 477-row current/one-second-target grid panel."""

    video_ids: np.ndarray
    timestamps: np.ndarray
    current: np.ndarray
    target: np.ndarray

    @property
    def channels(self) -> int:
        return int(np.asarray(self.current).shape[1])

    def validate(self, expected_channels: int | None = None) -> None:
        ids = np.asarray(self.video_ids, dtype=str)
        times = np.asarray(self.timestamps, dtype=float)
        current = np.asarray(self.current, dtype=float)
        target = np.asarray(self.target, dtype=float)
        if ids.shape != (477,) or times.shape != (477,):
            raise ValueError("raw grid identities must contain exactly 477 rows")
        if current.ndim != 4 or current.shape != target.shape or current.shape[0] != 477:
            raise ValueError("raw current/target grids must have equal four-dimensional shapes")
        if current.shape[2:] != (8, 8):
            raise ValueError("raw grids must have spatial shape [8,8]")
        if expected_channels is not None and current.shape[1] != expected_channels:
            raise ValueError(f"raw grids must have {expected_channels} channels")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
            raise ValueError("raw grid table contains non-finite values")
        expected: list[tuple[str, float]] = []
        for video_id in dataset.SAMPLE_VIDEO_IDS:
            expected.extend((video_id, 1.0 + 0.5 * index) for index in range(dataset.EXPECTED_WINDOW_COUNTS[video_id]))
        actual = list(zip(ids.tolist(), times.tolist(), strict=True))
        if any(
            left_id != right_id or not math.isclose(left_time, right_time, rel_tol=0.0, abs_tol=1e-12)
            for (left_id, left_time), (right_id, right_time) in zip(actual, expected, strict=True)
        ):
            raise ValueError("raw grid identities differ from the frozen MM-001 grid")


@dataclass(frozen=True, slots=True)
class GridTable:
    """Generic 469-row causal history panel for TAESD or source pixels."""

    video_ids: np.ndarray
    timestamps: np.ndarray
    previous: np.ndarray
    current: np.ndarray
    target: np.ndarray

    @property
    def channels(self) -> int:
        return int(np.asarray(self.current).shape[1])

    def validate(self, expected_channels: int | None = None) -> None:
        ids = np.asarray(self.video_ids, dtype=str)
        times = np.asarray(self.timestamps, dtype=float)
        previous = np.asarray(self.previous, dtype=float)
        current = np.asarray(self.current, dtype=float)
        target = np.asarray(self.target, dtype=float)
        if ids.shape != (HISTORY_ROWS,) or times.shape != (HISTORY_ROWS,):
            raise ValueError("history identities must contain exactly 469 rows")
        if previous.ndim != 4 or previous.shape != current.shape or target.shape != current.shape:
            raise ValueError("history grids must have equal four-dimensional shapes")
        if current.shape != (HISTORY_ROWS, current.shape[1], 8, 8):
            raise ValueError("history grids must have shape [469,C,8,8]")
        if expected_channels is not None and current.shape[1] != expected_channels:
            raise ValueError(f"history grids must have {expected_channels} channels")
        if not all(np.all(np.isfinite(value)) for value in (times, previous, current, target)):
            raise ValueError("history table contains non-finite values")
        expected: list[tuple[str, float]] = []
        for video_id in dataset.SAMPLE_VIDEO_IDS:
            expected.extend((video_id, 1.5 + 0.5 * index) for index in range(HISTORY_COUNTS[video_id]))
        actual = list(zip(ids.tolist(), times.tolist(), strict=True))
        if any(
            left_id != right_id or not math.isclose(left_time, right_time, rel_tol=0.0, abs_tol=1e-12)
            for (left_id, left_time), (right_id, right_time) in zip(actual, expected, strict=True)
        ):
            raise ValueError("history identities differ from the frozen causal panel")

    def subset(self, video_ids: Sequence[str]) -> GridTable:
        wanted = set(video_ids)
        mask = np.asarray([str(value) in wanted for value in self.video_ids], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"no history rows for videos {sorted(wanted)}")
        return GridTable(
            video_ids=np.asarray(self.video_ids, dtype=str)[mask].copy(),
            timestamps=np.asarray(self.timestamps, dtype=float)[mask].copy(),
            previous=np.asarray(self.previous, dtype=float)[mask].copy(),
            current=np.asarray(self.current, dtype=float)[mask].copy(),
            target=np.asarray(self.target, dtype=float)[mask].copy(),
        )


@dataclass(frozen=True, slots=True)
class ChannelNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(values, dtype=float) - self.mean) / self.scale, dtype=float)

    def fingerprint(self) -> str:
        digest = sha256(b"mm004-channel-normalizer-v1")
        for value in (self.mean, self.scale):
            array = np.asarray(value, dtype="<f8", order="C")
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FitResult:
    weights: np.ndarray
    record: dict[str, Any]


FIT_ROW_KEYS: Final = {
    "domain",
    "panel_seed",
    "fold",
    "arm_id",
    "control_id",
    "channels",
    "feature_dim",
    "train_video_ids",
    "excluded_video_ids",
    "train_rows",
    "train_patches",
    "fit_identity_sha256",
    "fit_matrix_sha256",
    "normalizer_fingerprint",
    "weight_shape",
    "weight_fingerprint",
    "linear_system_residual",
    "weights_finite",
    "kernel_relative_error",
    "recovered_kernel",
}
METRIC_ROW_KEYS: Final = {
    "domain",
    "panel_seed",
    "fold",
    "video_id",
    "arm_id",
    "channels",
    "test_rows",
    "test_patches",
    "ordered_weight_fingerprint",
    "target_shuffle_weight_fingerprint",
    "history_shuffle_weight_fingerprint",
    "persistence_mse",
    "constant_velocity_mse",
    "ordered_mse",
    "target_shuffle_mse",
    "history_shuffle_mse",
    "ordered_ratio",
    "target_shuffle_advantage",
    "history_shuffle_advantage",
    "past_delta_energy",
    "future_delta_energy",
    "past_future_cosine",
}
PANEL_ROW_KEYS: Final = {
    "panel_seed",
    "rows",
    "shape",
    "identity_sha256",
    "previous_sha256",
    "current_sha256",
    "target_sha256",
    "smoothing_passes",
    "difference_scale",
}
ACTIVITY_ROW_KEYS: Final = {
    "video_id",
    "rows",
    "persistence_mse",
    "half_cycle_persistence_mse",
    "activity_ratio",
    "active",
}
PARENT_ROW_KEYS: Final = {
    "passed",
    "rows_compared",
    "absolute_rows_compared",
    "residual_rows_compared",
    "flattened_current_sha256",
    "flattened_target_sha256",
    "rtol",
    "atol",
    "max_absolute_error",
}


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def raw_grid_table(
    video_ids: np.ndarray,
    timestamps: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
    *,
    expected_channels: int,
) -> RawGridTable:
    """Construct and validate one immutable formal raw-grid table."""

    table = RawGridTable(
        video_ids=np.asarray(video_ids, dtype=str).copy(),
        timestamps=np.asarray(timestamps, dtype=float).copy(),
        current=np.asarray(current, dtype=float).copy(),
        target=np.asarray(target, dtype=float).copy(),
    )
    table.validate(expected_channels)
    return table


def raw_grid_table_from_mappings(
    feature_arrays: Mapping[str, np.ndarray],
    grid_arrays: Mapping[str, np.ndarray],
    *,
    current_key: str,
    target_key: str,
    expected_channels: int,
) -> RawGridTable:
    """Construct a table from already strict-loaded NPZ-like mappings."""

    required_features = {"video_ids", "timestamps"}
    if not required_features.issubset(feature_arrays):
        raise ValueError("feature mapping lacks video identities/timestamps")
    if current_key not in grid_arrays or target_key not in grid_arrays:
        raise ValueError("grid mapping lacks current/target arrays")
    return raw_grid_table(
        feature_arrays["video_ids"],
        feature_arrays["timestamps"],
        grid_arrays[current_key],
        grid_arrays[target_key],
        expected_channels=expected_channels,
    )


def taesd_raw_table_from_mappings(
    feature_arrays: Mapping[str, np.ndarray], component_arrays: Mapping[str, np.ndarray]
) -> RawGridTable:
    return raw_grid_table_from_mappings(
        feature_arrays,
        component_arrays,
        current_key="taesd_latents",
        target_key="target_taesd_latents",
        expected_channels=4,
    )


def pixel_raw_table_from_mappings(
    feature_arrays: Mapping[str, np.ndarray], pixel_arrays: Mapping[str, np.ndarray]
) -> RawGridTable:
    return raw_grid_table_from_mappings(
        feature_arrays,
        pixel_arrays,
        current_key="pixel_grids",
        target_key="target_pixel_grids",
        expected_channels=3,
    )


def history_table(table: RawGridTable) -> GridTable:
    """Derive the exact 469-row previous/current/one-second-target panel."""

    table.validate()
    ids = np.asarray(table.video_ids, dtype=str)
    times = np.asarray(table.timestamps, dtype=float)
    previous_rows: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    output_ids: list[str] = []
    output_times: list[float] = []
    present_ids = tuple(video_id for video_id in dataset.SAMPLE_VIDEO_IDS if np.any(ids == video_id))
    if set(ids.tolist()) != set(present_ids):
        raise ValueError("derangement contains an unknown video identity")
    for video_id in present_ids:
        indices = np.flatnonzero(ids == video_id)
        ordered = indices[np.argsort(times[indices], kind="stable")]
        for position in range(1, len(ordered)):
            previous_rows.append(np.asarray(table.current[ordered[position - 1]], dtype=float))
            current_rows.append(np.asarray(table.current[ordered[position]], dtype=float))
            target_rows.append(np.asarray(table.target[ordered[position]], dtype=float))
            output_ids.append(video_id)
            output_times.append(float(times[ordered[position]]))
    output = GridTable(
        video_ids=np.asarray(output_ids),
        timestamps=np.asarray(output_times, dtype=float),
        previous=np.stack(previous_rows),
        current=np.stack(current_rows),
        target=np.stack(target_rows),
    )
    output.validate(table.channels)
    return output


def half_cycle_derangement(table: GridTable) -> np.ndarray:
    """Return a deterministic no-fixed-point within-video row mapping."""

    ids = np.asarray(table.video_ids, dtype=str)
    times = np.asarray(table.timestamps, dtype=float)
    mapping = np.empty(len(ids), dtype=int)
    present_ids = tuple(video_id for video_id in dataset.SAMPLE_VIDEO_IDS if np.any(ids == video_id))
    if set(ids.tolist()) != set(present_ids):
        raise ValueError("derangement contains an unknown video identity")
    for video_id in present_ids:
        indices = np.flatnonzero(ids == video_id)
        ordered = indices[np.argsort(times[indices], kind="stable")]
        if len(ordered) < 2:
            raise ValueError("each group needs at least two rows for derangement")
        shift = len(ordered) // 2
        mapping[ordered] = np.roll(ordered, shift)
    if np.any(mapping == np.arange(len(mapping))) or np.any(ids[mapping] != ids):
        raise ValueError("temporal derangement is not within-video and fixed-point-free")
    return mapping


def _fit_normalizer(train_current: np.ndarray) -> ChannelNormalizer:
    values = np.asarray(train_current, dtype=float)
    mean = values.mean(axis=(0, 2, 3), keepdims=False)[:, None, None]
    scale = np.maximum(values.std(axis=(0, 2, 3), ddof=0), SCALE_FLOOR)[:, None, None]
    return ChannelNormalizer(mean=np.asarray(mean), scale=np.asarray(scale))


def _normalize_table(table: GridTable, normalizer: ChannelNormalizer) -> GridTable:
    return GridTable(
        video_ids=np.asarray(table.video_ids, dtype=str).copy(),
        timestamps=np.asarray(table.timestamps, dtype=float).copy(),
        previous=normalizer.apply(table.previous),
        current=normalizer.apply(table.current),
        target=normalizer.apply(table.target),
    )


def _central(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[:, :, 1:7, 1:7]


def _patches(values: np.ndarray, patch_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if patch_size == 1:
        central = _central(array)
        return np.transpose(central, (0, 2, 3, 1)).reshape(-1, array.shape[1])
    if patch_size != 3:
        raise ValueError("MM-004 patch size must be 1 or 3")
    windows = np.lib.stride_tricks.sliding_window_view(array, (3, 3), axis=(2, 3))
    return np.transpose(windows, (0, 2, 3, 1, 4, 5)).reshape(-1, array.shape[1] * 9)


def _design(table: GridTable, arm: ArmSpec) -> tuple[np.ndarray, np.ndarray]:
    current = _patches(table.current, arm.patch_size)
    features = [current]
    if arm.uses_history:
        features.append(_patches(table.current - table.previous, arm.patch_size))
    x = np.concatenate(features, axis=1)
    residual = _central(table.target - table.current)
    y = np.transpose(residual, (0, 2, 3, 1)).reshape(-1, table.channels)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _expected_main_kernel(channels: int) -> np.ndarray:
    feature_dim = ARM_BY_ID[MAIN_ARM_ID].feature_dim(channels)
    weights = np.zeros((feature_dim + 1, channels), dtype=float)
    current_block = channels * 9
    for channel in range(channels):
        weights[channel * 9 + 3, channel] = 0.5
        weights[channel * 9 + 4, channel] = -0.5
        weights[current_block + channel * 9 + 4, channel] = 1.5
    return weights


def _undo_normalization(
    weights: np.ndarray,
    arm: ArmSpec,
    normalizer: ChannelNormalizer,
) -> np.ndarray:
    """Express normalized-residual weights in the original channel coordinates."""

    normalized = np.asarray(weights, dtype=float)
    channel_mean = np.asarray(normalizer.mean, dtype=float).reshape(-1)
    channel_scale = np.asarray(normalizer.scale, dtype=float).reshape(-1)
    patch_cells = arm.patch_size * arm.patch_size
    block_scale = np.repeat(channel_scale, patch_cells)
    feature_scale = np.tile(block_scale, 2 if arm.uses_history else 1)
    physical = np.empty_like(normalized)
    physical[:-1] = normalized[:-1] * channel_scale[None, :] / feature_scale[:, None]
    current_mean = np.repeat(channel_mean, patch_cells)
    current_features = len(block_scale)
    physical[-1] = channel_scale * normalized[-1] - current_mean @ physical[:current_features]
    return np.asarray(physical, dtype=float)


def _fit_one(
    table: GridTable,
    arm: ArmSpec,
    control_id: str,
    *,
    domain: str,
    panel_seed: int | None,
    fold: dataset.DatasetFold,
    normalizer: ChannelNormalizer,
) -> FitResult:
    if control_id not in CONTROL_IDS:
        raise ValueError(f"unknown control {control_id!r}")
    if control_id == "history_shuffle" and not arm.uses_history:
        raise ValueError("history shuffle applies only to history arms")
    mapping = half_cycle_derangement(table)
    previous = np.asarray(table.previous, dtype=float)
    target = np.asarray(table.target, dtype=float)
    if control_id == "target_shuffle":
        target = target[mapping]
    elif control_id == "history_shuffle":
        previous = previous[mapping]
    fit_table = GridTable(
        video_ids=table.video_ids,
        timestamps=table.timestamps,
        previous=previous,
        current=table.current,
        target=target,
    )
    x, y = _design(fit_table, arm)
    augmented = np.c_[x, np.ones(len(x))]
    regularizer = RIDGE_PENALTY * np.eye(augmented.shape[1])
    regularizer[-1, -1] = 0.0
    system = augmented.T @ augmented + regularizer
    rhs = augmented.T @ y
    weights = np.linalg.solve(system, rhs)
    residual = float(np.linalg.norm(system @ weights - rhs) / max(float(np.linalg.norm(rhs)), 1e-12))
    kernel_error: float | None = None
    recovered_kernel: list[list[float]] | None = None
    if domain == "synthetic" and arm.arm_id == MAIN_ARM_ID and control_id == "ordered":
        expected = _expected_main_kernel(table.channels)
        physical = _undo_normalization(weights, arm, normalizer)
        kernel_error = float(np.linalg.norm(physical - expected) / np.linalg.norm(expected))
        recovered_kernel = physical.tolist()
    matrix = np.c_[x, y]
    identity = [
        [str(video_id), float(timestamp)] for video_id, timestamp in zip(table.video_ids, table.timestamps, strict=True)
    ]
    record = {
        "domain": domain,
        "panel_seed": panel_seed,
        "fold": fold.index,
        "arm_id": arm.arm_id,
        "control_id": control_id,
        "channels": table.channels,
        "feature_dim": arm.feature_dim(table.channels),
        "train_video_ids": list(fold.train_ids),
        "excluded_video_ids": list(fold.test_ids),
        "train_rows": len(table.video_ids),
        "train_patches": len(x),
        "fit_identity_sha256": _canonical_sha256(identity),
        "fit_matrix_sha256": _array_sha256(matrix),
        "normalizer_fingerprint": normalizer.fingerprint(),
        "weight_shape": list(weights.shape),
        "weight_fingerprint": _array_sha256(weights),
        "linear_system_residual": residual,
        "weights_finite": bool(np.all(np.isfinite(weights))),
        "kernel_relative_error": kernel_error,
        "recovered_kernel": recovered_kernel,
    }
    return FitResult(weights=np.asarray(weights), record=record)


def _predict(table: GridTable, arm: ArmSpec, weights: np.ndarray) -> np.ndarray:
    x, _ = _design(table, arm)
    residual = np.c_[x, np.ones(len(x))] @ np.asarray(weights, dtype=float)
    residual_grid = np.transpose(
        residual.reshape(len(table.video_ids), CENTRAL_SIZE, CENTRAL_SIZE, table.channels),
        (0, 3, 1, 2),
    )
    return np.asarray(_central(table.current) + residual_grid, dtype=float)


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))


def _delta_statistics(table: GridTable) -> tuple[float, float, float]:
    past = _central(table.current - table.previous).reshape(-1)
    future = _central(table.target - table.current).reshape(-1)
    past_energy = float(np.mean(past**2))
    future_energy = float(np.mean(future**2))
    denominator = float(np.linalg.norm(past) * np.linalg.norm(future))
    cosine = 0.0 if denominator == 0.0 else float(np.dot(past, future) / denominator)
    return past_energy, future_energy, cosine


def _metric_row(
    table: GridTable,
    arm: ArmSpec,
    fits: Mapping[str, FitResult],
    *,
    domain: str,
    panel_seed: int | None,
    fold: dataset.DatasetFold,
    video_id: str,
) -> dict[str, Any]:
    truth = _central(table.target)
    persistence = _central(table.current)
    constant_velocity = _central(table.current + 2.0 * (table.current - table.previous))
    ordered = _predict(table, arm, fits["ordered"].weights)
    target_shuffle = _predict(table, arm, fits["target_shuffle"].weights)
    history_shuffle: np.ndarray | None = None
    if arm.uses_history:
        history_shuffle = _predict(table, arm, fits["history_shuffle"].weights)
    persistence_mse = _mse(persistence, truth)
    ordered_mse = _mse(ordered, truth)
    target_shuffle_mse = _mse(target_shuffle, truth)
    history_shuffle_mse = None if history_shuffle is None else _mse(history_shuffle, truth)
    past_energy, future_energy, cosine = _delta_statistics(table)
    denominator = max(persistence_mse, 1e-15)
    row: dict[str, Any] = {
        "domain": domain,
        "panel_seed": panel_seed,
        "fold": fold.index,
        "video_id": video_id,
        "arm_id": arm.arm_id,
        "channels": table.channels,
        "test_rows": len(table.video_ids),
        "test_patches": len(table.video_ids) * PATCHES_PER_ROW,
        "ordered_weight_fingerprint": fits["ordered"].record["weight_fingerprint"],
        "target_shuffle_weight_fingerprint": fits["target_shuffle"].record["weight_fingerprint"],
        "history_shuffle_weight_fingerprint": (
            fits["history_shuffle"].record["weight_fingerprint"] if arm.uses_history else None
        ),
        "persistence_mse": persistence_mse,
        "constant_velocity_mse": _mse(constant_velocity, truth),
        "ordered_mse": ordered_mse,
        "target_shuffle_mse": target_shuffle_mse,
        "history_shuffle_mse": history_shuffle_mse,
        "ordered_ratio": ordered_mse / denominator,
        "target_shuffle_advantage": target_shuffle_mse / max(ordered_mse, 1e-15),
        "history_shuffle_advantage": (
            None if history_shuffle_mse is None else history_shuffle_mse / max(ordered_mse, 1e-15)
        ),
        "past_delta_energy": past_energy,
        "future_delta_energy": future_energy,
        "past_future_cosine": cosine,
    }
    return row


def _execute_domain(
    table: GridTable,
    *,
    domain: str,
    panel_seed: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        raw_train = table.subset(fold.train_ids)
        normalizer = _fit_normalizer(raw_train.current)
        normalized = _normalize_table(table, normalizer)
        train = normalized.subset(fold.train_ids)
        for arm in ARMS:
            controls = (
                ("ordered", "target_shuffle", "history_shuffle")
                if arm.uses_history
                else (
                    "ordered",
                    "target_shuffle",
                )
            )
            fits: dict[str, FitResult] = {}
            for control_id in controls:
                fit = _fit_one(
                    train,
                    arm,
                    control_id,
                    domain=domain,
                    panel_seed=panel_seed,
                    fold=fold,
                    normalizer=normalizer,
                )
                fits[control_id] = fit
                fit_rows.append(fit.record)
            for video_id in fold.test_ids:
                test = normalized.subset([video_id])
                metric_rows.append(
                    _metric_row(
                        test,
                        arm,
                        fits,
                        domain=domain,
                        panel_seed=panel_seed,
                        fold=fold,
                        video_id=video_id,
                    )
                )
    return fit_rows, metric_rows


def _smooth_fields(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    for _ in range(SYNTHETIC_SMOOTHING_PASSES):
        padded = np.pad(output, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
        neighborhoods = np.stack(
            [padded[:, :, row : row + 8, column : column + 8] for row in range(3) for column in range(3)]
        )
        output = np.asarray(np.mean(neighborhoods, axis=0), dtype=float)
    return np.asarray(output)


def _shift_right(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.concatenate([array[:, :, :, :1], array[:, :, :, :-1]], axis=3)


def synthetic_panel(template: GridTable, seed: int) -> tuple[GridTable, dict[str, Any]]:
    """Generate one deterministic exact-class advection/momentum panel."""

    if seed not in SYNTHETIC_SEEDS:
        raise ValueError("synthetic seed is not frozen for MM-004")
    generator = np.random.Generator(np.random.PCG64(seed))
    shape = (HISTORY_ROWS, 4, 8, 8)
    current = _smooth_fields(generator.normal(size=shape))
    difference = SYNTHETIC_DIFFERENCE_SCALE * _smooth_fields(generator.normal(size=shape))
    previous = current - difference
    target = current + 0.5 * (_shift_right(current) - current) + 1.5 * difference
    panel = GridTable(
        video_ids=np.asarray(template.video_ids, dtype=str).copy(),
        timestamps=np.asarray(template.timestamps, dtype=float).copy(),
        previous=previous,
        current=current,
        target=target,
    )
    panel.validate(expected_channels=4)
    identities = [
        [str(video_id), float(timestamp)] for video_id, timestamp in zip(panel.video_ids, panel.timestamps, strict=True)
    ]
    record = {
        "panel_seed": seed,
        "rows": HISTORY_ROWS,
        "shape": list(shape),
        "identity_sha256": _canonical_sha256(identities),
        "previous_sha256": _array_sha256(previous),
        "current_sha256": _array_sha256(current),
        "target_sha256": _array_sha256(target),
        "smoothing_passes": SYNTHETIC_SMOOTHING_PASSES,
        "difference_scale": SYNTHETIC_DIFFERENCE_SCALE,
    }
    return panel, record


def parent_preflight_record(
    taesd_raw_table: RawGridTable,
    parent_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Recompute all 16 MM-003 raw256-native comparator rows exactly."""

    taesd_raw_table.validate(expected_channels=4)
    parent = mm003.validate_evidence(parent_evidence)
    flattened = mm003.VisualTable(
        video_ids=np.asarray(taesd_raw_table.video_ids, dtype=str).copy(),
        timestamps=np.asarray(taesd_raw_table.timestamps, dtype=float).copy(),
        current=np.asarray(taesd_raw_table.current, dtype=float).reshape(477, 256),
        target=np.asarray(taesd_raw_table.target, dtype=float).reshape(477, 256),
    )
    mm003.validate_raw_table(flattened)
    matched = mm003.matched_table(flattened)
    parent_rows = {
        (int(row["fold"]), str(row["predictor_id"]), str(row["video_id"])): row
        for row in cast(Sequence[Mapping[str, Any]], parent["probe_rows"])
        if row["representation_id"] == "raw256_native"
    }
    dummy_projection = np.zeros((256, 32), dtype=float)
    max_error = 0.0
    compared = 0
    for fold in dataset.formal_folds():
        train = flattened.subset(fold.train_ids)
        transform = mm003.fit_transform("raw256_native", train.current, dummy_projection)
        for predictor_id in mm003.PREDICTOR_IDS:
            rows = mm003._probe_rows_for(
                "raw256_native",
                predictor_id,
                fold,
                matched,
                transform,
            )
            for row in rows:
                old = parent_rows[(fold.index, predictor_id, row["video_id"])]
                for name in mm003.PROBE_METRICS:
                    error = abs(float(row[name]) - float(old[name]))
                    max_error = max(max_error, error)
                    if not math.isclose(float(row[name]), float(old[name]), rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(
                            f"MM-004 parent raw256 parity failed for {predictor_id}/{row['video_id']}/{name}"
                        )
                for name in ("fold", "video_id", "train_rows", "test_rows", "transform_fingerprint"):
                    if row[name] != old[name]:
                        raise ValueError(f"MM-004 parent raw256 identity failed for {predictor_id}/{name}")
                compared += 1
    if compared != 16 or len(parent_rows) != 16:
        raise ValueError("MM-004 parent parity did not cover exactly 16 rows")
    return {
        "passed": True,
        "rows_compared": compared,
        "absolute_rows_compared": 8,
        "residual_rows_compared": 8,
        "flattened_current_sha256": _array_sha256(flattened.current),
        "flattened_target_sha256": _array_sha256(flattened.target),
        "rtol": 1e-12,
        "atol": 1e-12,
        "max_absolute_error": max_error,
    }


def _activity_rows(table: GridTable) -> list[dict[str, Any]]:
    mapping = half_cycle_derangement(table)
    rows: list[dict[str, Any]] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        indices = np.flatnonzero(np.asarray(table.video_ids, dtype=str) == video_id)
        current = np.asarray(table.current, dtype=float)[indices]
        target = np.asarray(table.target, dtype=float)[indices]
        shuffled_target = np.asarray(table.target, dtype=float)[mapping[indices]]
        persistence = _mse(current, target)
        shuffled = _mse(current, shuffled_target)
        ratio, active = _activity_predicate(persistence, shuffled)
        rows.append(
            {
                "video_id": video_id,
                "rows": len(indices),
                "persistence_mse": persistence,
                "half_cycle_persistence_mse": shuffled,
                "activity_ratio": ratio,
                "active": active,
            }
        )
    return rows


def _activity_predicate(persistence_mse: float, shuffled_mse: float) -> tuple[float, bool]:
    ratio = float(persistence_mse) / max(float(shuffled_mse), 1e-15)
    active = bool(persistence_mse >= ACTIVITY_MSE_MIN and ACTIVITY_RATIO_MIN <= ratio <= ACTIVITY_RATIO_MAX)
    return ratio, active


def execute(
    taesd_raw_table: RawGridTable,
    pixel_raw_table: RawGridTable,
    parent_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Execute MM-004 from strict in-memory parent and prepared-pixel inputs."""

    taesd_raw_table.validate(expected_channels=4)
    pixel_raw_table.validate(expected_channels=3)
    if np.min(pixel_raw_table.current) < 0.0 or np.max(pixel_raw_table.current) > 1.0:
        raise ValueError("prepared current pixels must be in [0,1]")
    if np.min(pixel_raw_table.target) < 0.0 or np.max(pixel_raw_table.target) > 1.0:
        raise ValueError("prepared target pixels must be in [0,1]")
    taesd = history_table(taesd_raw_table)
    pixels = history_table(pixel_raw_table)
    if not np.array_equal(taesd.video_ids, pixels.video_ids) or not np.array_equal(taesd.timestamps, pixels.timestamps):
        raise ValueError("TAESD and pixel history identities differ")
    parent = parent_preflight_record(taesd_raw_table, parent_evidence)
    panel_rows: list[dict[str, Any]] = []
    synthetic_fits: list[dict[str, Any]] = []
    synthetic_metrics: list[dict[str, Any]] = []
    for seed in SYNTHETIC_SEEDS:
        panel, panel_record = synthetic_panel(taesd, seed)
        panel_rows.append(panel_record)
        fits, metrics = _execute_domain(panel, domain="synthetic", panel_seed=seed)
        synthetic_fits.extend(fits)
        synthetic_metrics.extend(metrics)
    real_fits: list[dict[str, Any]] = []
    real_metrics: list[dict[str, Any]] = []
    for domain, table in (("taesd", taesd), ("pixel", pixels)):
        fits, metrics = _execute_domain(table, domain=domain, panel_seed=None)
        real_fits.extend(fits)
        real_metrics.extend(metrics)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "parent_preflight": parent,
        "synthetic_panels": panel_rows,
        "synthetic_fit_rows": synthetic_fits,
        "synthetic_metric_rows": synthetic_metrics,
        "real_fit_rows": real_fits,
        "real_metric_rows": real_metrics,
        "activity_rows": _activity_rows(pixels),
    }
    return validate_evidence(evidence)


def _validate_parent(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != PARENT_ROW_KEYS:
        raise ValueError("MM-004 parent preflight schema differs")
    row = dict(value)
    if row["passed"] is not True or row["rows_compared"] != 16:
        raise ValueError("MM-004 parent preflight did not pass all 16 rows")
    if row["absolute_rows_compared"] != 8 or row["residual_rows_compared"] != 8:
        raise ValueError("MM-004 parent predictor counts differ")
    for name in ("flattened_current_sha256", "flattened_target_sha256"):
        _fingerprint(row[name], name)
    for name in ("rtol", "atol", "max_absolute_error"):
        _finite(row[name], name, nonnegative=True)
    return cast(dict[str, object], row)


def _validate_panels(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("synthetic panels must be an array")
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != PANEL_ROW_KEYS:
            raise ValueError("synthetic panel schema differs")
        row = dict(raw)
        if (
            row["panel_seed"] not in SYNTHETIC_SEEDS
            or row["rows"] != HISTORY_ROWS
            or row["shape"] != [HISTORY_ROWS, 4, 8, 8]
            or row["smoothing_passes"] != SYNTHETIC_SMOOTHING_PASSES
            or row["difference_scale"] != SYNTHETIC_DIFFERENCE_SCALE
        ):
            raise ValueError("synthetic panel configuration differs")
        for name in ("identity_sha256", "previous_sha256", "current_sha256", "target_sha256"):
            _fingerprint(row[name], name)
        rows.append(row)
    if [row["panel_seed"] for row in rows] != list(SYNTHETIC_SEEDS):
        raise ValueError("synthetic panels are incomplete or reordered")
    return rows


def _expected_fit_identities(domains: Sequence[str], panel_seeds: Sequence[int | None]) -> list[tuple[Any, ...]]:
    output: list[tuple[Any, ...]] = []
    for domain, panel_seed in zip(domains, panel_seeds, strict=True):
        for fold in dataset.formal_folds():
            for arm in ARMS:
                controls = (
                    ("ordered", "target_shuffle", "history_shuffle")
                    if arm.uses_history
                    else (
                        "ordered",
                        "target_shuffle",
                    )
                )
                output.extend((domain, panel_seed, fold.index, arm.arm_id, control) for control in controls)
    return output


def _validate_fit_rows(
    value: object,
    *,
    synthetic: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("fit rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != FIT_ROW_KEYS:
            raise ValueError("fit row schema differs")
        row = dict(raw)
        arm_id = row["arm_id"]
        if arm_id not in ARM_BY_ID or row["control_id"] not in CONTROL_IDS:
            raise ValueError("fit arm/control differs")
        arm = ARM_BY_ID[arm_id]
        if row["control_id"] == "history_shuffle" and not arm.uses_history:
            raise ValueError("non-history arm has a history-shuffle fit")
        expected_domain = "synthetic" if synthetic else row["domain"]
        if row["domain"] != expected_domain:
            raise ValueError("fit domain differs")
        if synthetic:
            if row["panel_seed"] not in SYNTHETIC_SEEDS or row["channels"] != 4:
                raise ValueError("synthetic fit seed/channels differ")
        elif row["domain"] not in DOMAINS or row["panel_seed"] is not None:
            raise ValueError("real fit domain/panel seed differs")
        channels = 4 if row["domain"] in ("taesd", "synthetic") else 3
        if row["channels"] != channels or row["feature_dim"] != arm.feature_dim(channels):
            raise ValueError("fit feature dimension differs")
        fold = dataset.formal_folds()[row["fold"]]
        expected_rows = sum(HISTORY_COUNTS[item] for item in fold.train_ids)
        if (
            row["train_video_ids"] != list(fold.train_ids)
            or row["excluded_video_ids"] != list(fold.test_ids)
            or row["train_rows"] != expected_rows
            or row["train_patches"] != expected_rows * PATCHES_PER_ROW
            or row["weight_shape"] != [arm.feature_dim(channels) + 1, channels]
        ):
            raise ValueError("fit row conflicts with its fold/arm")
        for name in (
            "fit_identity_sha256",
            "fit_matrix_sha256",
            "normalizer_fingerprint",
            "weight_fingerprint",
        ):
            _fingerprint(row[name], name)
        _finite(row["linear_system_residual"], "linear_system_residual", nonnegative=True)
        if type(row["weights_finite"]) is not bool:
            raise ValueError("weights_finite must be boolean")
        if row["kernel_relative_error"] is not None:
            _finite(row["kernel_relative_error"], "kernel_relative_error", nonnegative=True)
        should_have_kernel = synthetic and arm_id == MAIN_ARM_ID and row["control_id"] == "ordered"
        if (row["kernel_relative_error"] is not None) is not should_have_kernel:
            raise ValueError("kernel error appears on the wrong fit")
        if should_have_kernel:
            recovered = np.asarray(row["recovered_kernel"], dtype=float)
            expected_shape = (arm.feature_dim(channels) + 1, channels)
            if recovered.shape != expected_shape or not np.all(np.isfinite(recovered)):
                raise ValueError("recovered synthetic kernel shape/values differ")
            recomputed_error = float(
                np.linalg.norm(recovered - _expected_main_kernel(channels))
                / np.linalg.norm(_expected_main_kernel(channels))
            )
            if not math.isclose(
                recomputed_error,
                float(row["kernel_relative_error"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("synthetic kernel error does not recompute")
        elif row["recovered_kernel"] is not None:
            raise ValueError("recovered kernel appears on the wrong fit")
        identities.append((row["domain"], row["panel_seed"], row["fold"], arm_id, row["control_id"]))
        rows.append(row)
    if synthetic:
        expected = _expected_fit_identities(["synthetic"] * len(SYNTHETIC_SEEDS), list(SYNTHETIC_SEEDS))
    else:
        expected = _expected_fit_identities(list(DOMAINS), [None, None])
    if identities != expected:
        raise ValueError("fit rows are incomplete, duplicated, or reordered")
    return rows


def _expected_metric_identities(domains: Sequence[str], panel_seeds: Sequence[int | None]) -> list[tuple[Any, ...]]:
    return [
        (domain, panel_seed, fold.index, arm.arm_id, video_id)
        for domain, panel_seed in zip(domains, panel_seeds, strict=True)
        for fold in dataset.formal_folds()
        for arm in ARMS
        for video_id in fold.test_ids
    ]


def _validate_metric_rows(value: object, *, synthetic: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("metric rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != METRIC_ROW_KEYS:
            raise ValueError("metric row schema differs")
        row = dict(raw)
        arm_id = row["arm_id"]
        if arm_id not in ARM_BY_ID:
            raise ValueError("metric arm differs")
        arm = ARM_BY_ID[arm_id]
        if synthetic:
            if row["domain"] != "synthetic" or row["panel_seed"] not in SYNTHETIC_SEEDS:
                raise ValueError("synthetic metric identity differs")
        elif row["domain"] not in DOMAINS or row["panel_seed"] is not None:
            raise ValueError("real metric identity differs")
        channels = 4 if row["domain"] in ("taesd", "synthetic") else 3
        fold = dataset.formal_folds()[row["fold"]]
        if (
            row["video_id"] not in fold.test_ids
            or row["channels"] != channels
            or row["test_rows"] != HISTORY_COUNTS[row["video_id"]]
            or row["test_patches"] != HISTORY_COUNTS[row["video_id"]] * PATCHES_PER_ROW
        ):
            raise ValueError("metric row conflicts with fold/video")
        for name in ("ordered_weight_fingerprint", "target_shuffle_weight_fingerprint"):
            _fingerprint(row[name], name)
        if arm.uses_history:
            _fingerprint(row["history_shuffle_weight_fingerprint"], "history_shuffle_weight_fingerprint")
        elif row["history_shuffle_weight_fingerprint"] is not None:
            raise ValueError("non-history metric has history-shuffle fingerprint")
        for name in (
            "persistence_mse",
            "constant_velocity_mse",
            "ordered_mse",
            "target_shuffle_mse",
            "ordered_ratio",
            "target_shuffle_advantage",
            "past_delta_energy",
            "future_delta_energy",
        ):
            _finite(row[name], name, nonnegative=True)
        _finite(row["past_future_cosine"], "past_future_cosine")
        if arm.uses_history:
            _finite(row["history_shuffle_mse"], "history_shuffle_mse", nonnegative=True)
            _finite(row["history_shuffle_advantage"], "history_shuffle_advantage", nonnegative=True)
        elif row["history_shuffle_mse"] is not None or row["history_shuffle_advantage"] is not None:
            raise ValueError("non-history metric has history-shuffle values")
        if not math.isclose(
            float(row["ordered_ratio"]),
            float(row["ordered_mse"]) / max(float(row["persistence_mse"]), 1e-15),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("ordered ratio does not recompute")
        if not math.isclose(
            float(row["target_shuffle_advantage"]),
            float(row["target_shuffle_mse"]) / max(float(row["ordered_mse"]), 1e-15),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("target-shuffle advantage does not recompute")
        if arm.uses_history and not math.isclose(
            float(row["history_shuffle_advantage"]),
            float(row["history_shuffle_mse"]) / max(float(row["ordered_mse"]), 1e-15),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("history-shuffle advantage does not recompute")
        identities.append((row["domain"], row["panel_seed"], row["fold"], arm_id, row["video_id"]))
        rows.append(row)
    expected = (
        _expected_metric_identities(["synthetic"] * len(SYNTHETIC_SEEDS), list(SYNTHETIC_SEEDS))
        if synthetic
        else _expected_metric_identities(list(DOMAINS), [None, None])
    )
    if identities != expected:
        raise ValueError("metric rows are incomplete, duplicated, or reordered")
    return rows


def _validate_activity(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("activity rows must be an array")
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != ACTIVITY_ROW_KEYS:
            raise ValueError("activity row schema differs")
        row = dict(raw)
        video_id = row["video_id"]
        if video_id not in HISTORY_COUNTS or row["rows"] != HISTORY_COUNTS[video_id]:
            raise ValueError("activity identity/count differs")
        for name in ("persistence_mse", "half_cycle_persistence_mse", "activity_ratio"):
            _finite(row[name], name, nonnegative=True)
        ratio = float(row["persistence_mse"]) / max(float(row["half_cycle_persistence_mse"]), 1e-15)
        if not math.isclose(float(row["activity_ratio"]), ratio, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("activity ratio does not recompute")
        expected = bool(
            float(row["persistence_mse"]) >= ACTIVITY_MSE_MIN
            and ACTIVITY_RATIO_MIN <= float(row["activity_ratio"]) <= ACTIVITY_RATIO_MAX
        )
        if row["active"] is not expected:
            raise ValueError("activity predicate does not recompute")
        rows.append(row)
    if [row["video_id"] for row in rows] != list(dataset.SAMPLE_VIDEO_IDS):
        raise ValueError("activity rows are incomplete or reordered")
    return rows


def _validate_cross_links(
    fit_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    fit_lookup = {
        (row["domain"], row["panel_seed"], row["fold"], row["arm_id"], row["control_id"]): row for row in fit_rows
    }
    group_bindings: dict[tuple[object, object, object], tuple[object, object]] = {}
    for row in fit_rows:
        key = (row["domain"], row["panel_seed"], row["fold"])
        binding = (row["fit_identity_sha256"], row["normalizer_fingerprint"])
        prior = group_bindings.setdefault(key, binding)
        if prior != binding:
            raise ValueError("fit rows in one fold do not share identities/normalization")
    repeated_statistics: dict[tuple[object, object, object], tuple[float, float, float, float, float]] = {}
    for row in metric_rows:
        prefix = (row["domain"], row["panel_seed"], row["fold"], row["arm_id"])
        for control_id, field in (
            ("ordered", "ordered_weight_fingerprint"),
            ("target_shuffle", "target_shuffle_weight_fingerprint"),
        ):
            if row[field] != fit_lookup[(*prefix, control_id)]["weight_fingerprint"]:
                raise ValueError("metric row does not bind its fitted weights")
        if ARM_BY_ID[str(row["arm_id"])].uses_history and (
            row["history_shuffle_weight_fingerprint"] != fit_lookup[(*prefix, "history_shuffle")]["weight_fingerprint"]
        ):
            raise ValueError("metric row does not bind its history-shuffle weights")
        statistic_key = (row["domain"], row["panel_seed"], row["video_id"])
        statistics = (
            float(row["persistence_mse"]),
            float(row["constant_velocity_mse"]),
            float(row["past_delta_energy"]),
            float(row["future_delta_energy"]),
            float(row["past_future_cosine"]),
        )
        prior_statistics = repeated_statistics.setdefault(statistic_key, statistics)
        if prior_statistics != statistics:
            raise ValueError("arm-invariant held-out statistics differ across arms")


def validate_evidence(value: object) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "parent_preflight",
        "synthetic_panels",
        "synthetic_fit_rows",
        "synthetic_metric_rows",
        "real_fit_rows",
        "real_metric_rows",
        "activity_rows",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("MM-004 evidence schema differs")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("MM-004 evidence version differs")
    synthetic_fits = _validate_fit_rows(value["synthetic_fit_rows"], synthetic=True)
    synthetic_metrics = _validate_metric_rows(value["synthetic_metric_rows"], synthetic=True)
    real_fits = _validate_fit_rows(value["real_fit_rows"], synthetic=False)
    real_metrics = _validate_metric_rows(value["real_metric_rows"], synthetic=False)
    _validate_cross_links(synthetic_fits, synthetic_metrics)
    _validate_cross_links(real_fits, real_metrics)
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_preflight": _validate_parent(value["parent_preflight"]),
        "synthetic_panels": _validate_panels(value["synthetic_panels"]),
        "synthetic_fit_rows": synthetic_fits,
        "synthetic_metric_rows": synthetic_metrics,
        "real_fit_rows": real_fits,
        "real_metric_rows": real_metrics,
        "activity_rows": _validate_activity(value["activity_rows"]),
    }


def _supports(row: Mapping[str, Any]) -> bool:
    ordered = float(row["ordered_mse"])
    if ordered * REAL_PERSISTENCE_FACTOR > float(row["persistence_mse"]):
        return False
    if ordered * REAL_SHUFFLE_FACTOR > float(row["target_shuffle_mse"]):
        return False
    if ARM_BY_ID[str(row["arm_id"])].uses_history:
        history = cast(float, row["history_shuffle_mse"])
        if ordered * REAL_SHUFFLE_FACTOR > float(history):
            return False
    return True


def _arm_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for arm in ARMS:
            selected = [row for row in rows if row["domain"] == domain and row["arm_id"] == arm.arm_id]
            support = sum(_supports(row) for row in selected)
            output.append(
                {
                    "domain": domain,
                    "arm_id": arm.arm_id,
                    "supporting_videos": support,
                    "passes": support >= REQUIRED_VIDEO_SUPPORT,
                    "video_rows": selected,
                }
            )
    return output


def _baseline_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        # Baselines are arm-invariant and were cross-checked during evidence validation.
        selected = [row for row in rows if row["domain"] == domain and row["arm_id"] == "current_1x1"]
        constant_velocity_support = sum(
            float(row["constant_velocity_mse"]) * REAL_PERSISTENCE_FACTOR <= float(row["persistence_mse"])
            for row in selected
        )
        output.append(
            {
                "domain": domain,
                "constant_velocity_supporting_videos": constant_velocity_support,
                "constant_velocity_passes": constant_velocity_support >= REQUIRED_VIDEO_SUPPORT,
            }
        )
    return output


def _control_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for arm in ARMS:
            selected = [row for row in rows if row["domain"] == domain and row["arm_id"] == arm.arm_id]
            target_support = sum(
                float(row["target_shuffle_mse"]) * REAL_PERSISTENCE_FACTOR <= float(row["persistence_mse"])
                for row in selected
            )
            history_support: int | None = None
            if arm.uses_history:
                history_support = sum(
                    float(cast(float, row["history_shuffle_mse"])) * REAL_PERSISTENCE_FACTOR
                    <= float(row["persistence_mse"])
                    for row in selected
                )
            output.append(
                {
                    "domain": domain,
                    "arm_id": arm.arm_id,
                    "target_shuffle_support": target_support,
                    "history_shuffle_support": history_support,
                }
            )
    return output


def _contrast_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        lookup = {(row["arm_id"], row["video_id"]): row for row in rows if row["domain"] == domain}
        for contrast_id, comparator, metric in (
            ("history_contribution", "current_3x3", "ordered_mse"),
            ("spatial_neighborhood_contribution", "current_diff_1x1", "ordered_mse"),
            ("structured_advantage", MAIN_ARM_ID, "constant_velocity_mse"),
        ):
            paired: list[dict[str, Any]] = []
            support = 0
            for video_id in dataset.SAMPLE_VIDEO_IDS:
                main = lookup[(MAIN_ARM_ID, video_id)]
                comparison = main if contrast_id == "structured_advantage" else lookup[(comparator, video_id)]
                improved = float(main["ordered_mse"]) * REAL_CONTRAST_FACTOR <= float(comparison[metric])
                support += int(improved)
                paired.append({"video_id": video_id, "material_improvement": improved})
            output.append(
                {
                    "domain": domain,
                    "contrast_id": contrast_id,
                    "supporting_videos": support,
                    "passes": support >= REQUIRED_VIDEO_SUPPORT,
                    "paired_videos": paired,
                }
            )
    return output


def _synthetic_summary(
    fit_rows: Sequence[Mapping[str, Any]], metric_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    numerical_failures = [
        row
        for row in fit_rows
        if not row["weights_finite"] or float(row["linear_system_residual"]) > LINEAR_RESIDUAL_MAX
    ]
    kernel_failures = [
        row
        for row in fit_rows
        if row["arm_id"] == MAIN_ARM_ID
        and row["control_id"] == "ordered"
        and float(cast(float, row["kernel_relative_error"])) > KERNEL_ERROR_MAX
    ]
    lookup = {(row["panel_seed"], row["video_id"], row["arm_id"]): row for row in metric_rows}
    conditions: list[dict[str, Any]] = []
    positive_failures = 0
    negative_failures = 0
    for seed in SYNTHETIC_SEEDS:
        for video_id in dataset.SAMPLE_VIDEO_IDS:
            main = lookup[(seed, video_id, MAIN_ARM_ID)]
            current = lookup[(seed, video_id, "current_3x3")]
            pointwise = lookup[(seed, video_id, "current_diff_1x1")]
            positive = float(main["ordered_mse"]) <= SYNTHETIC_PERSISTENCE_RATIO * float(main["persistence_mse"])
            negative = bool(
                float(main["ordered_mse"]) <= SYNTHETIC_SEPARATION_RATIO * float(current["ordered_mse"])
                and float(main["ordered_mse"]) <= SYNTHETIC_SEPARATION_RATIO * float(pointwise["ordered_mse"])
                and float(main["ordered_mse"]) <= SYNTHETIC_SEPARATION_RATIO * float(main["target_shuffle_mse"])
                and float(main["ordered_mse"])
                <= SYNTHETIC_SEPARATION_RATIO * float(cast(float, main["history_shuffle_mse"]))
            )
            positive_failures += int(not positive)
            negative_failures += int(not negative)
            conditions.append(
                {
                    "panel_seed": seed,
                    "video_id": video_id,
                    "positive_passes": positive,
                    "negative_passes": negative,
                }
            )
    return {
        "conditions": conditions,
        "numerical_failures": len(numerical_failures),
        "kernel_failures": len(kernel_failures),
        "positive_failures": positive_failures,
        "negative_failures": negative_failures,
        "positive_passes": not numerical_failures and not kernel_failures and positive_failures == 0,
        "negative_passes": negative_failures == 0,
        "maximum_linear_system_residual": max(float(row["linear_system_residual"]) for row in fit_rows),
        "maximum_kernel_relative_error": max(
            float(cast(float, row["kernel_relative_error"]))
            for row in fit_rows
            if row["kernel_relative_error"] is not None
        ),
    }


def _decision(
    *,
    synthetic_positive: bool,
    synthetic_negative: bool,
    control_counts: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    activity_support: int,
) -> tuple[str, list[str]]:
    if not synthetic_positive:
        return "invalid_MM004_synthetic_positive_control", []
    if not synthetic_negative:
        return "invalid_MM004_synthetic_negative_control", []
    # Only target shuffle is a global shortcut check.  History shuffle leaves the
    # current patch intact and may therefore predict from spatial/current signal.
    target_control_values = [int(row["target_shuffle_support"]) for row in control_counts]
    if max(target_control_values) >= REQUIRED_VIDEO_SUPPORT:
        return "invalid_MM004_real_negative_control", []
    if max(target_control_values) >= 3:
        return "inconclusive_MM004_real_negative_control", []
    arm_lookup = {(row["domain"], row["arm_id"]): row for row in arms}
    taesd_pass = any(bool(arm_lookup[("taesd", arm.arm_id)]["passes"]) for arm in ARMS)
    if taesd_pass:
        contrast_lookup = {row["contrast_id"]: row for row in contrasts if row["domain"] == "taesd"}
        main_pass = bool(arm_lookup[("taesd", MAIN_ARM_ID)]["passes"])
        taesd_main_control = next(
            row for row in control_counts if row["domain"] == "taesd" and row["arm_id"] == MAIN_ARM_ID
        )
        history_control_support = int(cast(int, taesd_main_control["history_shuffle_support"]))
        labels: list[str] = []
        if main_pass and contrast_lookup["history_contribution"]["passes"] and history_control_support <= 2:
            labels.append("history_contribution_supported")
        if main_pass and contrast_lookup["spatial_neighborhood_contribution"]["passes"]:
            labels.append("spatial_neighborhood_contribution_supported")
        if main_pass and contrast_lookup["structured_advantage"]["passes"]:
            labels.append("structured_advantage_supported")
        if history_control_support >= 3:
            labels.append("history_control_predictive_from_current")
        return "taesd_local_linear_signal_supported", labels
    pixel_pass = any(bool(arm_lookup[("pixel", arm.arm_id)]["passes"]) for arm in ARMS)
    if pixel_pass:
        return "taesd_representation_failure_supported", []
    if activity_support >= REQUIRED_VIDEO_SUPPORT:
        return "tested_local_objective_or_horizon_failure_supported", []
    if activity_support <= 2:
        return "data_dynamics_insufficient_for_local_history_assay", []
    if 3 <= activity_support <= 5:
        return "inconclusive_video_heterogeneity", []
    return "MM004_diagnostic_inconclusive", []


def summarize(evidence: object) -> dict[str, Any]:
    normalized = validate_evidence(evidence)
    synthetic = _synthetic_summary(
        cast(Sequence[Mapping[str, Any]], normalized["synthetic_fit_rows"]),
        cast(Sequence[Mapping[str, Any]], normalized["synthetic_metric_rows"]),
    )
    real_rows = cast(Sequence[Mapping[str, Any]], normalized["real_metric_rows"])
    arms = _arm_summaries(real_rows)
    baselines = _baseline_summaries(real_rows)
    controls = _control_summaries(real_rows)
    contrasts = _contrast_summaries(real_rows)
    activity_rows = cast(Sequence[Mapping[str, Any]], normalized["activity_rows"])
    activity_support = sum(bool(row["active"]) for row in activity_rows)
    classification, labels = _decision(
        synthetic_positive=bool(synthetic["positive_passes"]),
        synthetic_negative=bool(synthetic["negative_passes"]),
        control_counts=controls,
        arms=arms,
        contrasts=contrasts,
        activity_support=activity_support,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "MM-004",
        "parent_preflight": normalized["parent_preflight"],
        "synthetic_control": synthetic,
        "arms": arms,
        "baselines": baselines,
        "controls": controls,
        "contrasts": contrasts,
        "activity": {"supporting_videos": activity_support, "video_rows": activity_rows},
        "decision": {
            "classification": classification,
            "mechanism_labels": labels,
            "recommended_next_step": RECOMMENDATIONS[classification],
        },
        "claim_boundary": (
            "MM-004 is an outcome-informed eight-video local-linear diagnostic; it does not establish "
            "nonlinear, population, or end-to-end Prospect capability."
        ),
    }


def report_text(summary: Mapping[str, Any]) -> str:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("experiment_id") != "MM-004":
        raise ValueError("summary is not an MM-004 result")
    decision = cast(Mapping[str, Any], summary["decision"])
    synthetic = cast(Mapping[str, Any], summary["synthetic_control"])
    lines = [
        "# MM-004 spatial/history signal-isolation report",
        "",
        "MM-004 is outcome-informed and does not reclassify MM-001, MM-002, or MM-003.",
        "",
        f"Decision classification: `{decision['classification']}`.",
        f"Recommended next step: {decision['recommended_next_step']}.",
        "",
        "## Controls",
        "",
        f"Synthetic positive control: **{'PASS' if synthetic['positive_passes'] else 'FAIL'}**.",
        f"Synthetic negative control: **{'PASS' if synthetic['negative_passes'] else 'FAIL'}**.",
        "",
        "## Local predictor ladder",
        "",
        "| Domain | Arm | Supporting videos |",
        "|---|---|---:|",
    ]
    for row in cast(Sequence[Mapping[str, Any]], summary["arms"]):
        lines.append(f"| `{row['domain']}` | `{row['arm_id']}` | {row['supporting_videos']}/8 |")
    lines.extend(["", "Constant-velocity support:"])
    for row in cast(Sequence[Mapping[str, Any]], summary["baselines"]):
        lines.append(f"- `{row['domain']}`: {row['constant_velocity_supporting_videos']}/8 videos")
    activity = cast(Mapping[str, Any], summary["activity"])
    labels = cast(Sequence[str], decision["mechanism_labels"])
    lines.extend(
        [
            "",
            f"Source activity support: {activity['supporting_videos']}/8 videos.",
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
        "raw_rows": 477,
        "history_rows": HISTORY_ROWS,
        "history_counts": HISTORY_COUNTS,
        "grid_shapes": {"taesd": [4, 8, 8], "pixel": [3, 8, 8]},
        "central_size": CENTRAL_SIZE,
        "patches_per_row": PATCHES_PER_ROW,
        "arms": [
            {
                "arm_id": arm.arm_id,
                "patch_size": arm.patch_size,
                "uses_history": arm.uses_history,
                "taesd_feature_dim": arm.feature_dim(4),
                "pixel_feature_dim": arm.feature_dim(3),
            }
            for arm in ARMS
        ],
        "ridge_penalty": RIDGE_PENALTY,
        "scale_floor": SCALE_FLOOR,
        "synthetic": {
            "seeds": list(SYNTHETIC_SEEDS),
            "difference_scale": SYNTHETIC_DIFFERENCE_SCALE,
            "smoothing_passes": SYNTHETIC_SMOOTHING_PASSES,
            "persistence_ratio": SYNTHETIC_PERSISTENCE_RATIO,
            "separation_ratio": SYNTHETIC_SEPARATION_RATIO,
            "linear_residual_max": LINEAR_RESIDUAL_MAX,
            "kernel_error_max": KERNEL_ERROR_MAX,
        },
        "thresholds": {
            "real_persistence_factor": REAL_PERSISTENCE_FACTOR,
            "real_shuffle_factor": REAL_SHUFFLE_FACTOR,
            "real_contrast_factor": REAL_CONTRAST_FACTOR,
            "required_video_support": REQUIRED_VIDEO_SUPPORT,
            "activity_mse_min": ACTIVITY_MSE_MIN,
            "activity_ratio_min": ACTIVITY_RATIO_MIN,
            "activity_ratio_max": ACTIVITY_RATIO_MAX,
        },
        "controls": {
            "target_shuffle": {
                "fit_scope": "train targets only",
                "preserved": ["previous", "current", "held_out_order"],
                "derangement": "within_video_half_cycle_no_fixed_points",
                "global_shortcut_check": True,
            },
            "history_shuffle": {
                "fit_scope": "train previous grids only",
                "preserved": ["current", "target", "held_out_order"],
                "derangement": "within_video_half_cycle_no_fixed_points",
                "global_shortcut_check": False,
            },
        },
        "decision_rules": {
            "synthetic_control": {
                "scope": "all_24_panel_video_conditions",
                "positive_predicate": "main_mse <= 0.10 * persistence_mse",
                "numerical_predicates": [
                    "every_solution_weights_finite",
                    "every_solution_linear_system_residual <= 1e-10",
                ],
                "kernel_predicate": ("every_panel_fold_main_physical_kernel_relative_error <= 0.05"),
                "negative_predicates": [
                    "main_mse <= 0.50 * current_3x3_mse",
                    "main_mse <= 0.50 * current_diff_1x1_mse",
                    "main_mse <= 0.50 * target_shuffle_mse",
                    "main_mse <= 0.50 * history_shuffle_mse",
                ],
            },
            "arm_support": {
                "no_history_predicates": [
                    "ordered_mse * 1.20 <= persistence_mse",
                    "ordered_mse * 1.10 <= target_shuffle_mse",
                ],
                "history_additional_predicate": ("ordered_mse * 1.10 <= history_shuffle_mse"),
                "supporting_videos_required": REQUIRED_VIDEO_SUPPORT,
            },
            "constant_velocity_support": {
                "predicate": "constant_velocity_mse * 1.20 <= persistence_mse",
                "supporting_videos_required": REQUIRED_VIDEO_SUPPORT,
            },
            "target_shuffle_global_control": {
                "predicate": "target_shuffle_mse * 1.20 <= persistence_mse",
                "scope": "every arm in both domains",
                "invalid_support": [6, 8],
                "inconclusive_support": [3, 5],
                "clear_support": [0, 2],
            },
            "history_shuffle_attribution_control": {
                "predicate": "history_shuffle_mse * 1.20 <= persistence_mse",
                "scope": "descriptive per history arm and domain",
                "history_attribution_max_support": 2,
                "predictive_from_current_min_support": 3,
            },
            "paired_contrasts": [
                {
                    "contrast_id": "history_contribution",
                    "main": MAIN_ARM_ID,
                    "comparator": "current_3x3",
                    "predicate": "main_ordered_mse * 1.10 <= comparator_ordered_mse",
                },
                {
                    "contrast_id": "spatial_neighborhood_contribution",
                    "main": MAIN_ARM_ID,
                    "comparator": "current_diff_1x1",
                    "predicate": "main_ordered_mse * 1.10 <= comparator_ordered_mse",
                },
                {
                    "contrast_id": "structured_advantage",
                    "main": MAIN_ARM_ID,
                    "comparator": "constant_velocity",
                    "predicate": "main_ordered_mse * 1.10 <= constant_velocity_mse",
                },
            ],
            "paired_contrast_supporting_videos_required": REQUIRED_VIDEO_SUPPORT,
            "mechanism_labels": {
                "common_predicates": [
                    "taesd_current_diff_3x3_arm_passes",
                    "named_paired_contrast_passes_6_of_8",
                ],
                "history_additional_predicate": ("taesd_main_history_shuffle_control_support <= 2"),
                "history_control_descriptor_predicate": ("taesd_main_history_shuffle_control_support >= 3"),
            },
            "pixel_representation_branch": "any_pixel_arm_passes",
            "activity_predicate": {
                "all": [
                    "pixel_persistence_mse >= 1e-4",
                    "pixel_persistence_mse / half_cycle_persistence_mse >= 0.10",
                    "pixel_persistence_mse / half_cycle_persistence_mse <= 1/1.2",
                ]
            },
            "activity_support_bands": {
                "supported": [6, 8],
                "heterogeneous": [3, 5],
                "insufficient": [0, 2],
            },
        },
        "decision_order": [
            {
                "order": 1,
                "classification": "invalid_MM004_parent_parity",
                "predicate": "parent_or_index_parity_failure",
                "phase": "pre_marker",
            },
            {
                "order": 2,
                "classification": "invalid_MM004_synthetic_positive_control",
                "predicate": "synthetic_main_kernel_or_numerical_failure",
            },
            {
                "order": 3,
                "classification": "invalid_MM004_synthetic_negative_control",
                "predicate": "synthetic_ablation_or_shuffle_separation_failure",
            },
            {
                "order": 4,
                "classification": "invalid_MM004_real_negative_control",
                "predicate": "target_shuffle_support_in_6_to_8",
            },
            {
                "order": 5,
                "classification": "inconclusive_MM004_real_negative_control",
                "predicate": "target_shuffle_support_in_3_to_5",
            },
            {
                "order": 6,
                "classification": "taesd_local_linear_signal_supported",
                "predicate": "any_taesd_arm_passes",
            },
            {
                "order": 7,
                "classification": "taesd_representation_failure_supported",
                "predicate": "all_taesd_arms_fail_and_any_pixel_arm_passes",
            },
            {
                "order": 8,
                "classification": "tested_local_objective_or_horizon_failure_supported",
                "predicate": "all_taesd_and_pixel_arms_fail_and_activity_support_in_6_to_8",
            },
            {
                "order": 9,
                "classification": "data_dynamics_insufficient_for_local_history_assay",
                "predicate": "all_taesd_and_pixel_arms_fail_and_activity_support_in_0_to_2",
            },
            {
                "order": 10,
                "classification": "inconclusive_video_heterogeneity",
                "predicate": "all_taesd_and_pixel_arms_fail_and_activity_support_in_3_to_5",
            },
            {
                "order": 11,
                "classification": "MM004_diagnostic_inconclusive",
                "predicate": "uncovered_valid_combination",
            },
        ],
        "recommendations": dict(RECOMMENDATIONS),
        "folds": [
            {"fold": fold.index, "train_ids": list(fold.train_ids), "test_ids": list(fold.test_ids)}
            for fold in dataset.formal_folds()
        ],
        "parent_parity": {"rows": 16, "rtol": 1e-12, "atol": 1e-12},
        "row_schemas": {
            "fit_rows": sorted(FIT_ROW_KEYS),
            "metric_rows": sorted(METRIC_ROW_KEYS),
            "synthetic_panels": sorted(PANEL_ROW_KEYS),
            "activity_rows": sorted(ACTIVITY_ROW_KEYS),
            "parent_preflight": sorted(PARENT_ROW_KEYS),
        },
    }


__all__ = [
    "ARMS",
    "GridTable",
    "HISTORY_COUNTS",
    "HISTORY_ROWS",
    "RawGridTable",
    "SCHEMA_VERSION",
    "config_record",
    "execute",
    "half_cycle_derangement",
    "history_table",
    "parent_preflight_record",
    "pixel_raw_table_from_mappings",
    "raw_grid_table",
    "raw_grid_table_from_mappings",
    "report_text",
    "summarize",
    "synthetic_panel",
    "taesd_raw_table_from_mappings",
    "validate_evidence",
]
