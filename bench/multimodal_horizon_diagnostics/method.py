"""Pure-NumPy scientific engine for MM-005 matched half-horizon replay.

This module owns no filesystem, media, or model-inference operations.  It turns
authenticated 477-row grids into one shared 453-row causal panel, fits the frozen
dual-horizon ridge probes, and reduces primitive evidence through the preregistered
MM-005 decision ladder.
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

SCHEMA_VERSION: Final = "mm005-method-v1"
EXPERIMENT_ID: Final = "MM-005"
RAW_ROWS: Final = 477
MATCHED_COUNTS: Final = {
    video_id: dataset.EXPECTED_WINDOW_COUNTS[video_id] - 3 for video_id in dataset.SAMPLE_VIDEO_IDS
}
MATCHED_ROWS: Final = sum(MATCHED_COUNTS.values())
MATCHED_IDENTITY_SHA256: Final = "d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b"
HORIZONS: Final = (0.5, 1.0)
DOMAINS: Final = ("taesd", "pixel")
RIDGE_PENALTY: Final = 1e-3
SCALE_FLOOR: Final = 1e-6
LINEAR_RESIDUAL_MAX: Final = 1e-10
KERNEL_ERROR_MAX: Final = 0.05
SYNTHETIC_SEEDS: Final = (550_050, 550_051, 550_052)
SYNTHETIC_SMOOTHING_PASSES: Final = 2
SYNTHETIC_PERSISTENCE_RATIO: Final = 0.10
SYNTHETIC_SEPARATION_RATIO: Final = 0.50
REQUIRED_VIDEO_SUPPORT: Final = 6
REAL_PERSISTENCE_FACTOR: Final = 1.20
REAL_SHUFFLE_FACTOR: Final = 1.10
PAIRED_HORIZON_FACTOR: Final = 1.10
ACTIVITY_MSE_MIN: Final = 1e-4
ACTIVITY_RATIO_MIN: Final = 0.10
ACTIVITY_RATIO_MAX: Final = 1.0 / 1.2
CENTRAL_SIZE: Final = 6
PATCHES_PER_ROW: Final = CENTRAL_SIZE * CENTRAL_SIZE


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_id: str
    patch_size: int
    uses_history: bool

    def feature_dim(self, channels: int) -> int:
        return channels * self.patch_size * self.patch_size * (2 if self.uses_history else 1)


MAIN_ARM: Final = ArmSpec("current_diff_3x3", 3, True)
COMPARATOR_ARM: Final = ArmSpec("current_3x3", 3, False)
POINTWISE_ABLATION: Final = ArmSpec("current_diff_1x1", 1, True)
# Primary first is the frozen decision order, not an outcome-dependent ranking.
ARMS: Final = (MAIN_ARM, COMPARATOR_ARM)
SYNTHETIC_ARMS: Final = (MAIN_ARM, COMPARATOR_ARM, POINTWISE_ABLATION)
ARM_BY_ID: Final = {arm.arm_id: arm for arm in SYNTHETIC_ARMS}
MAIN_ARM_ID: Final = MAIN_ARM.arm_id

RECOMMENDATIONS: Final = {
    "invalid_MM005_synthetic_positive_control": "repair the dual-horizon estimator before interpretation",
    "invalid_MM005_synthetic_negative_control": "repair target selection, ablations, or temporal controls",
    "invalid_MM005_real_negative_control": "reject the real assay because a target-shuffle shortcut crosses",
    "inconclusive_MM005_real_negative_control": "strengthen or independently replicate the real controls",
    "matched_one_second_taesd_signal_supported": "isolate common-row or endpoint sensitivity before changing horizon",
    "inconclusive_matched_one_second_taesd_signal": "replicate the matched one-second TAESD result",
    "half_second_horizon_mismatch_supported": (
        "implement a half-second target, then require a separately frozen teacher-free two-step rollout assay"
    ),
    "inconclusive_half_second_taesd_signal": "replicate before changing the temporal objective",
    "inconclusive_half_second_taesd_borderline": "replicate the borderline half-second TAESD result",
    "matched_one_second_pixel_signal_supported": "isolate pixel endpoint or common-row sensitivity",
    "inconclusive_matched_one_second_pixel_signal": "replicate the matched one-second pixel result",
    "half_second_taesd_representation_failure_supported": (
        "replace or temporally fine-tune TAESD before changing the dynamics model"
    ),
    "inconclusive_half_second_pixel_signal": "replicate before replacing the representation",
    "inconclusive_half_second_pixel_borderline": "replicate the borderline half-second pixel result",
    "half_second_tested_spatial_local_linear_objective_failure_supported": (
        "test a nonlinear causal warp or flow objective"
    ),
    "insufficient_half_second_source_change": "curate deliberately dynamic clips",
    "inconclusive_half_second_video_heterogeneity": "enlarge the independently sampled video panel",
    "MM005_diagnostic_inconclusive": "add a discriminating control before selecting a mechanism",
}


@dataclass(frozen=True, slots=True)
class RawGridTable:
    """Authenticated MM-004 477-row current/saved-target grids."""

    video_ids: np.ndarray
    timestamps: np.ndarray
    current: np.ndarray
    saved_target: np.ndarray

    @property
    def channels(self) -> int:
        return int(np.asarray(self.current).shape[1])

    def validate(self, expected_channels: int | None = None) -> None:
        ids = np.asarray(self.video_ids, dtype=str)
        times = np.asarray(self.timestamps, dtype=float)
        current = np.asarray(self.current, dtype=float)
        target = np.asarray(self.saved_target, dtype=float)
        if ids.shape != (RAW_ROWS,) or times.shape != (RAW_ROWS,):
            raise ValueError("raw identities must contain exactly 477 rows")
        if current.ndim != 4 or current.shape != target.shape or current.shape[0] != RAW_ROWS:
            raise ValueError("raw current/saved-target grids must have equal four-dimensional shapes")
        if current.shape[2:] != (8, 8):
            raise ValueError("raw grids must have spatial shape [8,8]")
        if expected_channels is not None and current.shape[1] != expected_channels:
            raise ValueError(f"raw grids must have {expected_channels} channels")
        if not all(np.all(np.isfinite(value)) for value in (times, current, target)):
            raise ValueError("raw grid table contains non-finite values")
        expected = [
            (video_id, 1.0 + 0.5 * index)
            for video_id in dataset.SAMPLE_VIDEO_IDS
            for index in range(dataset.EXPECTED_WINDOW_COUNTS[video_id])
        ]
        actual = list(zip(ids.tolist(), times.tolist(), strict=True))
        if any(
            left_id != right_id or not math.isclose(left_time, right_time, rel_tol=0.0, abs_tol=1e-12)
            for (left_id, left_time), (right_id, right_time) in zip(actual, expected, strict=True)
        ):
            raise ValueError("raw grid identities differ from the frozen MM-004 grid")


@dataclass(frozen=True, slots=True)
class MatchedPanel:
    """One source-shared dual-horizon grid panel (formally 453 rows)."""

    video_ids: np.ndarray
    timestamps: np.ndarray
    previous: np.ndarray
    current: np.ndarray
    target_0p5: np.ndarray
    target_1p0: np.ndarray

    @property
    def channels(self) -> int:
        return int(np.asarray(self.current).shape[1])

    def target(self, horizon_seconds: float) -> np.ndarray:
        if horizon_seconds == 0.5:
            return cast(np.ndarray, np.asarray(self.target_0p5, dtype=float))
        if horizon_seconds == 1.0:
            return cast(np.ndarray, np.asarray(self.target_1p0, dtype=float))
        raise ValueError("MM-005 horizon must be exactly 0.5 or 1.0 seconds")

    def validate(self, expected_channels: int | None = None, *, formal: bool = True) -> None:
        ids = np.asarray(self.video_ids, dtype=str)
        times = np.asarray(self.timestamps, dtype=float)
        grids = tuple(
            np.asarray(value, dtype=float) for value in (self.previous, self.current, self.target_0p5, self.target_1p0)
        )
        if any(value.ndim != 4 or value.shape != grids[0].shape for value in grids):
            raise ValueError("matched panel grids must have equal four-dimensional shapes")
        if grids[0].shape[0] != len(ids) or times.shape != ids.shape or grids[0].shape[2:] != (8, 8):
            raise ValueError("matched panel identities or grid shape differ")
        if expected_channels is not None and grids[0].shape[1] != expected_channels:
            raise ValueError(f"matched grids must have {expected_channels} channels")
        if not all(np.all(np.isfinite(value)) for value in (times, *grids)):
            raise ValueError("matched panel contains non-finite values")
        if not formal:
            return
        if ids.shape != (MATCHED_ROWS,):
            raise ValueError("formal matched panel must contain exactly 453 rows")
        expected = [
            (video_id, 1.5 + 0.5 * index)
            for video_id in dataset.SAMPLE_VIDEO_IDS
            for index in range(MATCHED_COUNTS[video_id])
        ]
        actual = list(zip(ids.tolist(), times.tolist(), strict=True))
        if any(
            left_id != right_id or not math.isclose(left_time, right_time, rel_tol=0.0, abs_tol=1e-12)
            for (left_id, left_time), (right_id, right_time) in zip(actual, expected, strict=True)
        ):
            raise ValueError("matched panel identities differ from the frozen 453-row grid")
        if _identity_sha256(ids, times) != MATCHED_IDENTITY_SHA256:
            raise ValueError("matched panel identity fingerprint differs")

    def subset(self, video_ids: Sequence[str]) -> MatchedPanel:
        wanted = set(video_ids)
        mask = np.asarray([str(value) in wanted for value in self.video_ids], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"no matched rows for videos {sorted(wanted)}")
        output = MatchedPanel(
            video_ids=np.asarray(self.video_ids, dtype=str)[mask].copy(),
            timestamps=np.asarray(self.timestamps, dtype=float)[mask].copy(),
            previous=np.asarray(self.previous, dtype=float)[mask].copy(),
            current=np.asarray(self.current, dtype=float)[mask].copy(),
            target_0p5=np.asarray(self.target_0p5, dtype=float)[mask].copy(),
            target_1p0=np.asarray(self.target_1p0, dtype=float)[mask].copy(),
        )
        output.validate(self.channels, formal=False)
        return output


@dataclass(frozen=True, slots=True)
class ChannelNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return cast(
            np.ndarray,
            np.asarray((np.asarray(values, dtype=float) - self.mean) / self.scale, dtype=float),
        )

    def fingerprint(self) -> str:
        return _arrays_sha256("mm005-channel-normalizer-v1", self.mean, self.scale)


@dataclass(frozen=True, slots=True)
class FitResult:
    weights: np.ndarray
    record: dict[str, Any]


FIT_ROW_KEYS: Final = {
    "domain",
    "panel_seed",
    "horizon_seconds",
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
    "source_identity_sha256",
    "input_sha256",
    "target_sha256",
    "design_sha256",
    "normalizer_fingerprint",
    "weight_shape",
    "weight_fingerprint",
    "linear_system_residual",
    "weights_finite",
    "kernel_relative_error",
    "other_horizon_kernel_relative_error",
    "recovered_kernel",
}
METRIC_ROW_KEYS: Final = {
    "domain",
    "panel_seed",
    "horizon_seconds",
    "fold",
    "video_id",
    "arm_id",
    "channels",
    "test_rows",
    "test_patches",
    "ordered_weight_fingerprint",
    "target_shuffle_weight_fingerprint",
    "history_shuffle_weight_fingerprint",
    "raw_persistence_mse",
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
    "target_0p5_sha256",
    "target_1p0_sha256",
    "smoothing_passes",
    "generator",
}
ACTIVITY_ROW_KEYS: Final = {
    "horizon_seconds",
    "video_id",
    "rows",
    "persistence_mse",
    "shuffled_mse",
    "activity_ratio",
    "active",
}
ALIGNMENT_KEYS: Final = {"rows", "counts", "identity_sha256", "domains"}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    digest = sha256(b"mm005-array-v1")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_sha256(prefix: str, *values: np.ndarray) -> str:
    digest = sha256(prefix.encode("ascii"))
    for value in values:
        array = np.asarray(value, dtype="<f8", order="C")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _identity_sha256(video_ids: np.ndarray, timestamps: np.ndarray) -> str:
    identities = [[str(video_id), float(timestamp)] for video_id, timestamp in zip(video_ids, timestamps, strict=True)]
    return _canonical_sha256(identities)


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    output = float(value)
    if nonnegative and output < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return output


def _exact_int(value: object, name: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    output = value
    if nonnegative and output < 0:
        raise ValueError(f"{name} must be nonnegative")
    return output


def raw_grid_table(
    video_ids: np.ndarray,
    timestamps: np.ndarray,
    current: np.ndarray,
    saved_target: np.ndarray,
    *,
    expected_channels: int,
) -> RawGridTable:
    table = RawGridTable(
        video_ids=np.asarray(video_ids, dtype=str).copy(),
        timestamps=np.asarray(timestamps, dtype=float).copy(),
        current=np.asarray(current).copy(),
        saved_target=np.asarray(saved_target).copy(),
    )
    table.validate(expected_channels)
    return table


def matched_panel(table: RawGridTable) -> MatchedPanel:
    """Build and exact-parity-check the formal 453-row dual-horizon panel."""

    table.validate()
    ids = np.asarray(table.video_ids, dtype=str)
    times = np.asarray(table.timestamps, dtype=float)
    previous: list[np.ndarray] = []
    current: list[np.ndarray] = []
    target_0p5: list[np.ndarray] = []
    target_1p0: list[np.ndarray] = []
    output_ids: list[str] = []
    output_times: list[float] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        rows = np.flatnonzero(ids == video_id)
        ordered = rows[np.argsort(times[rows], kind="stable")]
        if len(ordered) != dataset.EXPECTED_WINDOW_COUNTS[video_id]:
            raise ValueError(f"raw count differs for {video_id}")
        for position in range(1, len(ordered) - 2):
            source = ordered[position]
            half = ordered[position + 1]
            one = ordered[position + 2]
            if not np.array_equal(table.saved_target[ordered[position - 1]], table.current[half]):
                raise ValueError("half-second saved-target parity differs")
            if not np.array_equal(table.saved_target[source], table.current[one]):
                raise ValueError("one-second saved-target parity differs")
            previous.append(np.asarray(table.current[ordered[position - 1]]))
            current.append(np.asarray(table.current[source]))
            target_0p5.append(np.asarray(table.current[half]))
            target_1p0.append(np.asarray(table.current[one]))
            output_ids.append(video_id)
            output_times.append(float(times[source]))
    panel = MatchedPanel(
        video_ids=np.asarray(output_ids),
        timestamps=np.asarray(output_times, dtype=float),
        previous=np.stack(previous),
        current=np.stack(current),
        target_0p5=np.stack(target_0p5),
        target_1p0=np.stack(target_1p0),
    )
    panel.validate(table.channels)
    return panel


def alignment_record(taesd: MatchedPanel, pixel: MatchedPanel) -> dict[str, object]:
    taesd.validate(4)
    pixel.validate(3)
    if not np.array_equal(taesd.video_ids, pixel.video_ids) or not np.array_equal(taesd.timestamps, pixel.timestamps):
        raise ValueError("TAESD and pixel matched identities differ")
    domains: dict[str, object] = {}
    for name, panel in (("taesd", taesd), ("pixel", pixel)):
        domains[name] = {
            "channels": panel.channels,
            "previous_sha256": _array_sha256(panel.previous),
            "current_sha256": _array_sha256(panel.current),
            "target_0p5_sha256": _array_sha256(panel.target_0p5),
            "target_1p0_sha256": _array_sha256(panel.target_1p0),
        }
    return {
        "rows": MATCHED_ROWS,
        "counts": dict(MATCHED_COUNTS),
        "identity_sha256": _identity_sha256(taesd.video_ids, taesd.timestamps),
        "domains": domains,
    }


def panel_provenance(table: RawGridTable) -> dict[str, object]:
    """Regenerate a fit-free provenance record for one authenticated raw domain."""

    panel = matched_panel(table)
    return {
        "rows": MATCHED_ROWS,
        "counts": dict(MATCHED_COUNTS),
        "channels": panel.channels,
        "identity_sha256": _identity_sha256(panel.video_ids, panel.timestamps),
        "previous_sha256": _array_sha256(panel.previous),
        "current_sha256": _array_sha256(panel.current),
        "target_0p5_sha256": _array_sha256(panel.target_0p5),
        "target_1p0_sha256": _array_sha256(panel.target_1p0),
    }


def half_cycle_derangement(table: MatchedPanel) -> np.ndarray:
    ids = np.asarray(table.video_ids, dtype=str)
    times = np.asarray(table.timestamps, dtype=float)
    mapping: np.ndarray = np.empty(len(ids), dtype=int)
    present = tuple(video_id for video_id in dataset.SAMPLE_VIDEO_IDS if np.any(ids == video_id))
    if set(ids.tolist()) != set(present):
        raise ValueError("derangement contains an unknown video identity")
    for video_id in present:
        rows = np.flatnonzero(ids == video_id)
        ordered = rows[np.argsort(times[rows], kind="stable")]
        if len(ordered) < 2:
            raise ValueError("each group needs at least two rows for derangement")
        mapping[ordered] = np.roll(ordered, len(ordered) // 2)
    if np.any(mapping == np.arange(len(mapping))) or np.any(ids[mapping] != ids):
        raise ValueError("temporal derangement is not fixed-point-free and within-video")
    return mapping


def _fit_normalizer(train_current: np.ndarray) -> ChannelNormalizer:
    values = np.asarray(train_current, dtype=float)
    if values.ndim != 4 or not np.all(np.isfinite(values)):
        raise ValueError("normalizer input must be finite [N,C,H,W]")
    mean = values.mean(axis=(0, 2, 3))[:, None, None]
    scale = np.maximum(values.std(axis=(0, 2, 3), ddof=0), SCALE_FLOOR)[:, None, None]
    return ChannelNormalizer(mean=np.asarray(mean), scale=np.asarray(scale))


def _normalize_panel(table: MatchedPanel, normalizer: ChannelNormalizer) -> MatchedPanel:
    return MatchedPanel(
        video_ids=np.asarray(table.video_ids, dtype=str).copy(),
        timestamps=np.asarray(table.timestamps, dtype=float).copy(),
        previous=normalizer.apply(table.previous),
        current=normalizer.apply(table.current),
        target_0p5=normalizer.apply(table.target_0p5),
        target_1p0=normalizer.apply(table.target_1p0),
    )


def _central(values: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.asarray(values, dtype=float)[:, :, 1:7, 1:7])


def _patches(values: np.ndarray, patch_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if patch_size == 1:
        return cast(np.ndarray, np.transpose(_central(array), (0, 2, 3, 1)).reshape(-1, array.shape[1]))
    if patch_size != 3:
        raise ValueError("MM-005 patch size must be 1 or 3")
    windows = np.lib.stride_tricks.sliding_window_view(array, (3, 3), axis=(2, 3))
    return cast(
        np.ndarray,
        np.transpose(windows, (0, 2, 3, 1, 4, 5)).reshape(-1, array.shape[1] * 9),
    )


def _design(table: MatchedPanel, horizon_seconds: float, arm: ArmSpec) -> tuple[np.ndarray, np.ndarray]:
    features = [_patches(table.current, arm.patch_size)]
    if arm.uses_history:
        features.append(_patches(table.current - table.previous, arm.patch_size))
    x = np.concatenate(features, axis=1)
    residual = _central(table.target(horizon_seconds) - table.current)
    y = np.transpose(residual, (0, 2, 3, 1)).reshape(-1, table.channels)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _expected_kernel(channels: int, horizon_seconds: float) -> np.ndarray:
    if horizon_seconds == 0.5:
        advection, history = 0.25, 0.75
    elif horizon_seconds == 1.0:
        advection, history = 0.50, 1.50
    else:
        raise ValueError("unknown kernel horizon")
    weights: np.ndarray = np.zeros((MAIN_ARM.feature_dim(channels) + 1, channels), dtype=float)
    current_block = channels * 9
    for channel in range(channels):
        weights[channel * 9 + 3, channel] = advection
        weights[channel * 9 + 4, channel] = -advection
        weights[current_block + channel * 9 + 4, channel] = history
    return weights


def _undo_normalization(weights: np.ndarray, arm: ArmSpec, normalizer: ChannelNormalizer) -> np.ndarray:
    normalized = np.asarray(weights, dtype=float)
    channel_mean = np.asarray(normalizer.mean, dtype=float).reshape(-1)
    channel_scale = np.asarray(normalizer.scale, dtype=float).reshape(-1)
    patch_cells = arm.patch_size * arm.patch_size
    block_scale: np.ndarray = np.repeat(channel_scale, patch_cells)
    feature_scale: np.ndarray = np.tile(block_scale, 2 if arm.uses_history else 1)
    physical = np.empty_like(normalized)
    physical[:-1] = normalized[:-1] * channel_scale[None, :] / feature_scale[:, None]
    current_mean: np.ndarray = np.repeat(channel_mean, patch_cells)
    physical[-1] = channel_scale * normalized[-1] - current_mean @ physical[: len(block_scale)]
    return cast(np.ndarray, np.asarray(physical, dtype=float))


def _fit_one(
    table: MatchedPanel,
    horizon_seconds: float,
    arm: ArmSpec,
    control_id: str,
    *,
    domain: str,
    panel_seed: int | None,
    fold: dataset.DatasetFold,
    normalizer: ChannelNormalizer,
) -> FitResult:
    allowed = {"ordered", "target_shuffle", "history_shuffle"}
    if control_id not in allowed or (control_id == "history_shuffle" and not arm.uses_history):
        raise ValueError("invalid MM-005 fit control")
    mapping = half_cycle_derangement(table)
    previous = np.asarray(table.previous, dtype=float)
    target_0p5 = np.asarray(table.target_0p5, dtype=float)
    target_1p0 = np.asarray(table.target_1p0, dtype=float)
    if control_id == "target_shuffle":
        target_0p5 = target_0p5[mapping]
        target_1p0 = target_1p0[mapping]
    elif control_id == "history_shuffle":
        previous = previous[mapping]
    fit_table = MatchedPanel(
        video_ids=table.video_ids,
        timestamps=table.timestamps,
        previous=previous,
        current=table.current,
        target_0p5=target_0p5,
        target_1p0=target_1p0,
    )
    x, y = _design(fit_table, horizon_seconds, arm)
    augmented = np.c_[x, np.ones(len(x))]
    regularizer = RIDGE_PENALTY * np.eye(augmented.shape[1])
    regularizer[-1, -1] = 0.0
    system = augmented.T @ augmented + regularizer
    rhs = augmented.T @ y
    weights = np.linalg.solve(system, rhs)
    residual = float(np.linalg.norm(system @ weights - rhs) / max(float(np.linalg.norm(rhs)), 1e-12))
    kernel_error: float | None = None
    other_kernel_error: float | None = None
    recovered: list[list[float]] | None = None
    if domain == "synthetic" and arm.arm_id == MAIN_ARM_ID and control_id == "ordered":
        physical = _undo_normalization(weights, arm, normalizer)
        expected = _expected_kernel(table.channels, horizon_seconds)
        other = _expected_kernel(table.channels, 1.0 if horizon_seconds == 0.5 else 0.5)
        kernel_error = float(np.linalg.norm(physical - expected) / np.linalg.norm(expected))
        other_kernel_error = float(np.linalg.norm(physical - other) / np.linalg.norm(other))
        recovered = physical.tolist()
    identity = {
        "domain": domain,
        "panel_seed": panel_seed,
        "horizon_seconds": horizon_seconds,
        "fold": fold.index,
        "arm_id": arm.arm_id,
        "control_id": control_id,
        "sources": [
            [str(video_id), float(timestamp)]
            for video_id, timestamp in zip(table.video_ids, table.timestamps, strict=True)
        ],
    }
    record = {
        "domain": domain,
        "panel_seed": panel_seed,
        "horizon_seconds": horizon_seconds,
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
        "source_identity_sha256": _identity_sha256(table.video_ids, table.timestamps),
        "input_sha256": _arrays_sha256("mm005-fit-input-v1", previous, table.current),
        "target_sha256": _array_sha256(fit_table.target(horizon_seconds)),
        "design_sha256": _array_sha256(x),
        "normalizer_fingerprint": normalizer.fingerprint(),
        "weight_shape": list(weights.shape),
        "weight_fingerprint": _array_sha256(weights),
        "linear_system_residual": residual,
        "weights_finite": bool(np.all(np.isfinite(weights))),
        "kernel_relative_error": kernel_error,
        "other_horizon_kernel_relative_error": other_kernel_error,
        "recovered_kernel": recovered,
    }
    return FitResult(weights=np.asarray(weights), record=record)


def _predict(table: MatchedPanel, horizon_seconds: float, arm: ArmSpec, weights: np.ndarray) -> np.ndarray:
    x, _ = _design(table, horizon_seconds, arm)
    residual = np.c_[x, np.ones(len(x))] @ np.asarray(weights, dtype=float)
    grid = np.transpose(
        residual.reshape(len(table.video_ids), CENTRAL_SIZE, CENTRAL_SIZE, table.channels),
        (0, 3, 1, 2),
    )
    return cast(np.ndarray, np.asarray(_central(table.current) + grid, dtype=float))


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _delta_statistics(table: MatchedPanel, horizon_seconds: float) -> tuple[float, float, float]:
    past = _central(table.current - table.previous).reshape(-1)
    future = _central(table.target(horizon_seconds) - table.current).reshape(-1)
    denominator = float(np.linalg.norm(past) * np.linalg.norm(future))
    return (
        float(np.mean(past**2)),
        float(np.mean(future**2)),
        0.0 if denominator == 0.0 else float(np.dot(past, future) / denominator),
    )


def _metric_row(
    normalized: MatchedPanel,
    raw: MatchedPanel,
    horizon_seconds: float,
    arm: ArmSpec,
    fits: Mapping[str, FitResult],
    *,
    domain: str,
    panel_seed: int | None,
    fold: dataset.DatasetFold,
    video_id: str,
) -> dict[str, Any]:
    truth = _central(normalized.target(horizon_seconds))
    persistence = _central(normalized.current)
    velocity_factor = horizon_seconds / 0.5
    velocity = _central(normalized.current + velocity_factor * (normalized.current - normalized.previous))
    ordered = _predict(normalized, horizon_seconds, arm, fits["ordered"].weights)
    target_shuffle = (
        _predict(normalized, horizon_seconds, arm, fits["target_shuffle"].weights) if "target_shuffle" in fits else None
    )
    history_shuffle = (
        _predict(normalized, horizon_seconds, arm, fits["history_shuffle"].weights)
        if "history_shuffle" in fits
        else None
    )
    persistence_mse = _mse(persistence, truth)
    ordered_mse = _mse(ordered, truth)
    target_shuffle_mse = None if target_shuffle is None else _mse(target_shuffle, truth)
    history_shuffle_mse = None if history_shuffle is None else _mse(history_shuffle, truth)
    past_energy, future_energy, cosine = _delta_statistics(normalized, horizon_seconds)
    return {
        "domain": domain,
        "panel_seed": panel_seed,
        "horizon_seconds": horizon_seconds,
        "fold": fold.index,
        "video_id": video_id,
        "arm_id": arm.arm_id,
        "channels": normalized.channels,
        "test_rows": len(normalized.video_ids),
        "test_patches": len(normalized.video_ids) * PATCHES_PER_ROW,
        "ordered_weight_fingerprint": fits["ordered"].record["weight_fingerprint"],
        "target_shuffle_weight_fingerprint": (
            fits["target_shuffle"].record["weight_fingerprint"] if "target_shuffle" in fits else None
        ),
        "history_shuffle_weight_fingerprint": (
            fits["history_shuffle"].record["weight_fingerprint"] if "history_shuffle" in fits else None
        ),
        "raw_persistence_mse": _mse(_central(raw.current), _central(raw.target(horizon_seconds))),
        "persistence_mse": persistence_mse,
        "constant_velocity_mse": _mse(velocity, truth),
        "ordered_mse": ordered_mse,
        "target_shuffle_mse": target_shuffle_mse,
        "history_shuffle_mse": history_shuffle_mse,
        "ordered_ratio": _optional_ratio(ordered_mse, persistence_mse),
        "target_shuffle_advantage": (
            None if target_shuffle_mse is None else _optional_ratio(target_shuffle_mse, ordered_mse)
        ),
        "history_shuffle_advantage": (
            None if history_shuffle_mse is None else _optional_ratio(history_shuffle_mse, ordered_mse)
        ),
        "past_delta_energy": past_energy,
        "future_delta_energy": future_energy,
        "past_future_cosine": cosine,
    }


def _real_controls(arm: ArmSpec) -> tuple[str, ...]:
    return (
        ("ordered", "target_shuffle", "history_shuffle")
        if arm.uses_history
        else (
            "ordered",
            "target_shuffle",
        )
    )


def _execute_real_domain(panel: MatchedPanel, *, domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        normalizer = _fit_normalizer(panel.subset(fold.train_ids).current)
        normalized = _normalize_panel(panel, normalizer)
        train = normalized.subset(fold.train_ids)
        for horizon_seconds in HORIZONS:
            for arm in ARMS:
                fits: dict[str, FitResult] = {}
                for control_id in _real_controls(arm):
                    fit = _fit_one(
                        train,
                        horizon_seconds,
                        arm,
                        control_id,
                        domain=domain,
                        panel_seed=None,
                        fold=fold,
                        normalizer=normalizer,
                    )
                    fits[control_id] = fit
                    fit_rows.append(fit.record)
                for video_id in fold.test_ids:
                    metric_rows.append(
                        _metric_row(
                            normalized.subset([video_id]),
                            panel.subset([video_id]),
                            horizon_seconds,
                            arm,
                            fits,
                            domain=domain,
                            panel_seed=None,
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
    return cast(np.ndarray, output)


def _shift_right(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return cast(np.ndarray, np.concatenate([array[:, :, :, :1], array[:, :, :, :-1]], axis=3))


def synthetic_panel(template: MatchedPanel, seed: int) -> tuple[MatchedPanel, dict[str, Any]]:
    if seed not in SYNTHETIC_SEEDS:
        raise ValueError("synthetic seed is not frozen for MM-005")
    generator = np.random.Generator(np.random.PCG64(seed))
    shape = (MATCHED_ROWS, 4, 8, 8)
    current = _smooth_fields(generator.normal(size=shape))
    difference = _smooth_fields(generator.normal(size=shape))
    previous = current - difference
    shifted = _shift_right(current) - current
    target_0p5 = current + 0.25 * shifted + 0.75 * difference
    target_1p0 = current + 0.50 * shifted + 1.50 * difference
    panel = MatchedPanel(
        video_ids=np.asarray(template.video_ids, dtype=str).copy(),
        timestamps=np.asarray(template.timestamps, dtype=float).copy(),
        previous=previous,
        current=current,
        target_0p5=target_0p5,
        target_1p0=target_1p0,
    )
    panel.validate(4)
    record = {
        "panel_seed": seed,
        "rows": MATCHED_ROWS,
        "shape": list(shape),
        "identity_sha256": _identity_sha256(panel.video_ids, panel.timestamps),
        "previous_sha256": _array_sha256(previous),
        "current_sha256": _array_sha256(current),
        "target_0p5_sha256": _array_sha256(target_0p5),
        "target_1p0_sha256": _array_sha256(target_1p0),
        "smoothing_passes": SYNTHETIC_SMOOTHING_PASSES,
        "generator": "numpy.random.Generator(PCG64)",
    }
    return panel, record


def _synthetic_specs() -> tuple[tuple[ArmSpec, tuple[str, ...]], ...]:
    return (
        (MAIN_ARM, ("ordered", "target_shuffle", "history_shuffle")),
        (COMPARATOR_ARM, ("ordered",)),
        (POINTWISE_ABLATION, ("ordered",)),
    )


def _execute_synthetic_panel(panel: MatchedPanel, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        normalizer = _fit_normalizer(panel.subset(fold.train_ids).current)
        normalized = _normalize_panel(panel, normalizer)
        train = normalized.subset(fold.train_ids)
        for horizon_seconds in HORIZONS:
            for arm, controls in _synthetic_specs():
                fits: dict[str, FitResult] = {}
                for control_id in controls:
                    fit = _fit_one(
                        train,
                        horizon_seconds,
                        arm,
                        control_id,
                        domain="synthetic",
                        panel_seed=seed,
                        fold=fold,
                        normalizer=normalizer,
                    )
                    fits[control_id] = fit
                    fit_rows.append(fit.record)
                for video_id in fold.test_ids:
                    metric_rows.append(
                        _metric_row(
                            normalized.subset([video_id]),
                            panel.subset([video_id]),
                            horizon_seconds,
                            arm,
                            fits,
                            domain="synthetic",
                            panel_seed=seed,
                            fold=fold,
                            video_id=video_id,
                        )
                    )
    return fit_rows, metric_rows


def _activity_rows(panel: MatchedPanel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_seconds in HORIZONS:
        for video_id in dataset.SAMPLE_VIDEO_IDS:
            table = panel.subset([video_id])
            target = np.asarray(table.target(horizon_seconds), dtype=float)
            current = np.asarray(table.current, dtype=float)
            mapping = half_cycle_derangement(table)
            persistence = _mse(current, target)
            shuffled = _mse(current, target[mapping])
            ratio, active = _activity_predicate(persistence, shuffled)
            rows.append(
                {
                    "horizon_seconds": horizon_seconds,
                    "video_id": video_id,
                    "rows": len(table.video_ids),
                    "persistence_mse": persistence,
                    "shuffled_mse": shuffled,
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
    parent_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute all 200 frozen fits from already authenticated in-memory arrays."""

    del parent_evidence
    taesd_raw_table.validate(4)
    pixel_raw_table.validate(3)
    if np.min(pixel_raw_table.current) < 0.0 or np.max(pixel_raw_table.current) > 1.0:
        raise ValueError("pixel current grids must be in [0,1]")
    if np.min(pixel_raw_table.saved_target) < 0.0 or np.max(pixel_raw_table.saved_target) > 1.0:
        raise ValueError("pixel saved-target grids must be in [0,1]")
    taesd = matched_panel(taesd_raw_table)
    pixel = matched_panel(pixel_raw_table)
    alignment = alignment_record(taesd, pixel)
    synthetic_panels: list[dict[str, Any]] = []
    synthetic_fits: list[dict[str, Any]] = []
    synthetic_metrics: list[dict[str, Any]] = []
    for seed in SYNTHETIC_SEEDS:
        panel, record = synthetic_panel(taesd, seed)
        fits, metrics = _execute_synthetic_panel(panel, seed)
        synthetic_panels.append(record)
        synthetic_fits.extend(fits)
        synthetic_metrics.extend(metrics)
    real_fits: list[dict[str, Any]] = []
    real_metrics: list[dict[str, Any]] = []
    for domain, panel in (("taesd", taesd), ("pixel", pixel)):
        fits, metrics = _execute_real_domain(panel, domain=domain)
        real_fits.extend(fits)
        real_metrics.extend(metrics)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "alignment": alignment,
        "synthetic_panels": synthetic_panels,
        "synthetic_fit_rows": synthetic_fits,
        "synthetic_metric_rows": synthetic_metrics,
        "real_fit_rows": real_fits,
        "real_metric_rows": real_metrics,
        "activity_rows": _activity_rows(pixel),
    }
    return validate_evidence(evidence)


def _expected_fit_identities(*, synthetic: bool) -> list[tuple[object, ...]]:
    output: list[tuple[object, ...]] = []
    domains: Sequence[tuple[str, int | None]] = (
        tuple(("synthetic", seed) for seed in SYNTHETIC_SEEDS) if synthetic else (("taesd", None), ("pixel", None))
    )
    for domain, seed in domains:
        for fold in dataset.formal_folds():
            for horizon in HORIZONS:
                specs = _synthetic_specs() if synthetic else tuple((arm, _real_controls(arm)) for arm in ARMS)
                for arm, controls in specs:
                    output.extend((domain, seed, horizon, fold.index, arm.arm_id, control) for control in controls)
    return output


def _fold_source_identities(fold: dataset.DatasetFold) -> list[list[object]]:
    return [
        [video_id, 1.5 + 0.5 * index]
        for video_id in dataset.SAMPLE_VIDEO_IDS
        if video_id in fold.train_ids
        for index in range(MATCHED_COUNTS[video_id])
    ]


def _expected_metric_identities(*, synthetic: bool) -> list[tuple[object, ...]]:
    output: list[tuple[object, ...]] = []
    domains: Sequence[tuple[str, int | None]] = (
        tuple(("synthetic", seed) for seed in SYNTHETIC_SEEDS) if synthetic else (("taesd", None), ("pixel", None))
    )
    arms = SYNTHETIC_ARMS if synthetic else ARMS
    for domain, seed in domains:
        for fold in dataset.formal_folds():
            for horizon in HORIZONS:
                for arm in arms:
                    output.extend(
                        (domain, seed, horizon, fold.index, arm.arm_id, video_id) for video_id in fold.test_ids
                    )
    return output


def _validate_alignment(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != ALIGNMENT_KEYS:
        raise ValueError("alignment record schema differs")
    row = dict(value)
    _exact_int(row["rows"], "alignment rows", nonnegative=True)
    if row["rows"] != MATCHED_ROWS or row["counts"] != MATCHED_COUNTS:
        raise ValueError("alignment counts differ")
    counts = row["counts"]
    if not isinstance(counts, dict) or any(
        type(key) is not str or type(count) is not int for key, count in counts.items()
    ):
        raise ValueError("alignment count primitive types differ")
    if row["identity_sha256"] != MATCHED_IDENTITY_SHA256:
        raise ValueError("alignment identity differs")
    domains = row["domains"]
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
        raise ValueError("alignment domains differ")
    for domain, channels in (("taesd", 4), ("pixel", 3)):
        record = domains[domain]
        expected_keys = {
            "channels",
            "previous_sha256",
            "current_sha256",
            "target_0p5_sha256",
            "target_1p0_sha256",
        }
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ValueError("alignment domain schema differs")
        _exact_int(record["channels"], "alignment channels", nonnegative=True)
        if record["channels"] != channels:
            raise ValueError("alignment channel count differs")
        for key in expected_keys - {"channels"}:
            _fingerprint(record[key], key)
    return cast(dict[str, object], row)


def _validate_panels(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("synthetic panels must be an array")
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != PANEL_ROW_KEYS:
            raise ValueError("synthetic panel schema differs")
        row = dict(raw)
        _exact_int(row["rows"], "synthetic panel rows", nonnegative=True)
        _exact_int(row["smoothing_passes"], "synthetic smoothing passes", nonnegative=True)
        if (
            not isinstance(row["shape"], list)
            or len(row["shape"]) != 4
            or any(type(item) is not int for item in row["shape"])
        ):
            raise ValueError("synthetic panel shape primitive types differ")
        if (
            type(row["panel_seed"]) is not int
            or row["panel_seed"] not in SYNTHETIC_SEEDS
            or row["rows"] != MATCHED_ROWS
            or row["shape"] != [MATCHED_ROWS, 4, 8, 8]
            or row["identity_sha256"] != MATCHED_IDENTITY_SHA256
            or row["smoothing_passes"] != SYNTHETIC_SMOOTHING_PASSES
            or row["generator"] != "numpy.random.Generator(PCG64)"
        ):
            raise ValueError("synthetic panel configuration differs")
        for key in (
            "previous_sha256",
            "current_sha256",
            "target_0p5_sha256",
            "target_1p0_sha256",
        ):
            _fingerprint(row[key], key)
        rows.append(row)
    if [row["panel_seed"] for row in rows] != list(SYNTHETIC_SEEDS):
        raise ValueError("synthetic panels are incomplete or reordered")
    return rows


def _validate_fit_rows(value: object, *, synthetic: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("fit rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[object, ...]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != FIT_ROW_KEYS:
            raise ValueError("fit row schema differs")
        row = dict(raw)
        if type(row["domain"]) is not str or type(row["horizon_seconds"]) is not float:
            raise ValueError("fit domain/horizon primitive type differs")
        if type(row["fold"]) is not int or type(row["arm_id"]) is not str or type(row["control_id"]) is not str:
            raise ValueError("fit fold/arm/control primitive type differs")
        domain = row["domain"]
        seed = row["panel_seed"]
        horizon = _finite(row["horizon_seconds"], "horizon")
        if horizon not in HORIZONS or row["fold"] not in range(4):
            raise ValueError("fit horizon/fold differs")
        if synthetic:
            if type(seed) is not int or domain != "synthetic" or seed not in SYNTHETIC_SEEDS or row["channels"] != 4:
                raise ValueError("synthetic fit identity differs")
        elif domain not in DOMAINS or seed is not None or row["channels"] != (4 if domain == "taesd" else 3):
            raise ValueError("real fit identity differs")
        arm_id = row["arm_id"]
        if arm_id not in ARM_BY_ID or (not synthetic and arm_id not in {arm.arm_id for arm in ARMS}):
            raise ValueError("fit arm differs")
        arm = ARM_BY_ID[arm_id]
        control = row["control_id"]
        allowed = dict(_synthetic_specs())[arm] if synthetic else _real_controls(arm)
        if control not in allowed:
            raise ValueError("fit control differs")
        fold = dataset.formal_folds()[row["fold"]]
        for key in ("channels", "feature_dim", "train_rows", "train_patches"):
            _exact_int(row[key], f"fit {key}", nonnegative=True)
        if (
            not isinstance(row["weight_shape"], list)
            or len(row["weight_shape"]) != 2
            or any(type(item) is not int for item in row["weight_shape"])
        ):
            raise ValueError("fit weight shape primitive types differ")
        if type(row["weights_finite"]) is not bool:
            raise ValueError("fit finite flag must be boolean")
        expected_rows = sum(MATCHED_COUNTS[video_id] for video_id in fold.train_ids)
        if (
            row["feature_dim"] != arm.feature_dim(int(row["channels"]))
            or row["train_video_ids"] != list(fold.train_ids)
            or row["excluded_video_ids"] != list(fold.test_ids)
            or row["train_rows"] != expected_rows
            or row["train_patches"] != expected_rows * PATCHES_PER_ROW
            or row["weight_shape"] != [arm.feature_dim(int(row["channels"])) + 1, row["channels"]]
        ):
            raise ValueError("fit dimensions or fold membership differ")
        for key in (
            "fit_identity_sha256",
            "source_identity_sha256",
            "input_sha256",
            "target_sha256",
            "design_sha256",
            "normalizer_fingerprint",
            "weight_fingerprint",
        ):
            _fingerprint(row[key], key)
        sources = _fold_source_identities(fold)
        if row["source_identity_sha256"] != _canonical_sha256(sources):
            raise ValueError("fit source identity fingerprint differs")
        expected_fit_identity = _canonical_sha256(
            {
                "domain": domain,
                "panel_seed": seed,
                "horizon_seconds": horizon,
                "fold": fold.index,
                "arm_id": arm_id,
                "control_id": control,
                "sources": sources,
            }
        )
        if row["fit_identity_sha256"] != expected_fit_identity:
            raise ValueError("fit identity fingerprint differs")
        _finite(row["linear_system_residual"], "linear residual", nonnegative=True)
        kernel_expected = synthetic and arm_id == MAIN_ARM_ID and control == "ordered"
        kernel_fields = (
            row["kernel_relative_error"],
            row["other_horizon_kernel_relative_error"],
            row["recovered_kernel"],
        )
        if kernel_expected:
            _finite(kernel_fields[0], "kernel error", nonnegative=True)
            _finite(kernel_fields[1], "other kernel error", nonnegative=True)
            recovered = np.asarray(kernel_fields[2], dtype=float)
            if recovered.shape != tuple(row["weight_shape"]) or not np.all(np.isfinite(recovered)):
                raise ValueError("recovered kernel shape differs")
            expected = _expected_kernel(int(row["channels"]), horizon)
            other = _expected_kernel(int(row["channels"]), 1.0 if horizon == 0.5 else 0.5)
            expected_error = float(np.linalg.norm(recovered - expected) / np.linalg.norm(expected))
            other_error = float(np.linalg.norm(recovered - other) / np.linalg.norm(other))
            own_matches = math.isclose(float(kernel_fields[0]), expected_error, rel_tol=1e-12, abs_tol=1e-12)
            other_matches = math.isclose(float(kernel_fields[1]), other_error, rel_tol=1e-12, abs_tol=1e-12)
            if not own_matches or not other_matches:
                raise ValueError("derived kernel error differs")
        elif any(field is not None for field in kernel_fields):
            raise ValueError("kernel recovery appears on a non-main fit")
        identity = (domain, seed, horizon, row["fold"], arm_id, control)
        identities.append(identity)
        rows.append(row)
    if identities != _expected_fit_identities(synthetic=synthetic):
        raise ValueError("fit rows are incomplete, duplicated, or reordered")
    # Every estimator in one domain/panel/fold shares exactly the same source rows
    # and train-current normalizer.
    scope_fields: dict[tuple[object, ...], tuple[object, object]] = {}
    for row in rows:
        scope_key = (row["domain"], row["panel_seed"], row["fold"])
        scope_binding = (row["source_identity_sha256"], row["normalizer_fingerprint"])
        if scope_key in scope_fields and scope_fields[scope_key] != scope_binding:
            raise ValueError("fits in one fold do not share source and normalizer identities")
        scope_fields[scope_key] = scope_binding
    # Horizon pairs must share sources, inputs, design, and normalization.
    pair_fields: dict[tuple[object, ...], tuple[object, ...]] = {}
    for row in rows:
        pair_key = (row["domain"], row["panel_seed"], row["fold"], row["arm_id"], row["control_id"])
        binding = tuple(
            row[field]
            for field in ("source_identity_sha256", "input_sha256", "design_sha256", "normalizer_fingerprint")
        )
        if pair_key in pair_fields and pair_fields[pair_key] != binding:
            raise ValueError("paired horizons do not share source/design identities")
        pair_fields[pair_key] = binding
    lookup = {
        (row["domain"], row["panel_seed"], row["horizon_seconds"], row["fold"], row["arm_id"], row["control_id"]): row
        for row in rows
    }
    for row in rows:
        if row["control_id"] != "ordered":
            continue
        prefix = (row["domain"], row["panel_seed"], row["horizon_seconds"], row["fold"], row["arm_id"])
        target_shuffle = lookup.get((*prefix, "target_shuffle"))
        if target_shuffle is not None and (
            target_shuffle["input_sha256"] != row["input_sha256"]
            or target_shuffle["design_sha256"] != row["design_sha256"]
        ):
            raise ValueError("target shuffle changed estimator inputs or design")
        history_shuffle = lookup.get((*prefix, "history_shuffle"))
        if history_shuffle is not None and history_shuffle["target_sha256"] != row["target_sha256"]:
            raise ValueError("history shuffle changed the training target")
    return rows


def _derived_optional_ratio(value: object, numerator: float, denominator: float, name: str) -> None:
    expected = _optional_ratio(numerator, denominator)
    if expected is None:
        if value is not None:
            raise ValueError(f"{name} must be null at a zero denominator")
    elif value is None:
        raise ValueError(f"derived {name} differs")
    else:
        observed = _finite(value, name, nonnegative=True)
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"derived {name} differs")


def _validate_metric_rows(
    value: object, *, synthetic: bool, fit_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("metric rows must be an array")
    fit_lookup = {
        (row["domain"], row["panel_seed"], row["horizon_seconds"], row["fold"], row["arm_id"], row["control_id"]): row
        for row in fit_rows
    }
    rows: list[dict[str, Any]] = []
    identities: list[tuple[object, ...]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != METRIC_ROW_KEYS:
            raise ValueError("metric row schema differs")
        row = dict(raw)
        if type(row["domain"]) is not str or type(row["horizon_seconds"]) is not float:
            raise ValueError("metric domain/horizon primitive type differs")
        if type(row["fold"]) is not int or type(row["video_id"]) is not str or type(row["arm_id"]) is not str:
            raise ValueError("metric fold/video/arm primitive type differs")
        domain = row["domain"]
        seed = row["panel_seed"]
        horizon = _finite(row["horizon_seconds"], "metric horizon")
        arm_id = row["arm_id"]
        if horizon not in HORIZONS or arm_id not in ARM_BY_ID:
            raise ValueError("metric identity differs")
        if synthetic:
            if (
                type(seed) is not int
                or domain != "synthetic"
                or seed not in SYNTHETIC_SEEDS
                or arm_id not in {a.arm_id for a in SYNTHETIC_ARMS}
            ):
                raise ValueError("synthetic metric identity differs")
        elif domain not in DOMAINS or seed is not None or arm_id not in {a.arm_id for a in ARMS}:
            raise ValueError("real metric identity differs")
        fold = dataset.formal_folds()[row["fold"]]
        video_id = row["video_id"]
        channels = 4 if domain in ("taesd", "synthetic") else 3
        for key in ("channels", "test_rows", "test_patches"):
            _exact_int(row[key], f"metric {key}", nonnegative=True)
        if (
            video_id not in fold.test_ids
            or row["channels"] != channels
            or row["test_rows"] != MATCHED_COUNTS[video_id]
            or row["test_patches"] != MATCHED_COUNTS[video_id] * PATCHES_PER_ROW
        ):
            raise ValueError("metric fold or dimensions differ")
        numeric = {}
        for key in (
            "raw_persistence_mse",
            "persistence_mse",
            "constant_velocity_mse",
            "ordered_mse",
            "past_delta_energy",
            "future_delta_energy",
        ):
            numeric[key] = _finite(row[key], key, nonnegative=True)
        cosine = _finite(row["past_future_cosine"], "past_future_cosine")
        if cosine < -1.000000000001 or cosine > 1.000000000001:
            raise ValueError("delta cosine lies outside [-1,1]")
        _fingerprint(row["ordered_weight_fingerprint"], "ordered weight")
        ordered_fit = fit_lookup[(domain, seed, horizon, row["fold"], arm_id, "ordered")]
        if row["ordered_weight_fingerprint"] != ordered_fit["weight_fingerprint"]:
            raise ValueError("metric does not bind ordered weights")
        controls = dict(_synthetic_specs())[ARM_BY_ID[arm_id]] if synthetic else _real_controls(ARM_BY_ID[arm_id])
        for control, mse_key, fingerprint_key, advantage_key in (
            ("target_shuffle", "target_shuffle_mse", "target_shuffle_weight_fingerprint", "target_shuffle_advantage"),
            (
                "history_shuffle",
                "history_shuffle_mse",
                "history_shuffle_weight_fingerprint",
                "history_shuffle_advantage",
            ),
        ):
            if control in controls:
                mse = _finite(row[mse_key], mse_key, nonnegative=True)
                _fingerprint(row[fingerprint_key], fingerprint_key)
                fit = fit_lookup[(domain, seed, horizon, row["fold"], arm_id, control)]
                if row[fingerprint_key] != fit["weight_fingerprint"]:
                    raise ValueError("metric does not bind control weights")
                _derived_optional_ratio(row[advantage_key], mse, numeric["ordered_mse"], advantage_key)
            elif row[mse_key] is not None or row[fingerprint_key] is not None or row[advantage_key] is not None:
                raise ValueError("metric contains an inapplicable control")
        _derived_optional_ratio(
            row["ordered_ratio"], numeric["ordered_mse"], numeric["persistence_mse"], "ordered_ratio"
        )
        identities.append((domain, seed, horizon, row["fold"], arm_id, video_id))
        rows.append(row)
    if identities != _expected_metric_identities(synthetic=synthetic):
        raise ValueError("metric rows are incomplete, duplicated, or reordered")
    # Arm-invariant baselines must agree exactly for each held-out condition.
    baseline_bindings: dict[tuple[object, ...], tuple[object, ...]] = {}
    for row in rows:
        baseline_key = (
            row["domain"],
            row["panel_seed"],
            row["horizon_seconds"],
            row["fold"],
            row["video_id"],
        )
        binding = tuple(
            row[field]
            for field in (
                "raw_persistence_mse",
                "persistence_mse",
                "constant_velocity_mse",
                "past_delta_energy",
                "future_delta_energy",
                "past_future_cosine",
            )
        )
        if baseline_key in baseline_bindings and baseline_bindings[baseline_key] != binding:
            raise ValueError("arm-invariant metric baselines differ")
        baseline_bindings[baseline_key] = binding
    return rows


def _validate_activity(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("activity rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[float, str]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != ACTIVITY_ROW_KEYS:
            raise ValueError("activity row schema differs")
        row = dict(raw)
        if type(row["horizon_seconds"]) is not float or type(row["video_id"]) is not str:
            raise ValueError("activity horizon/video primitive type differs")
        horizon = _finite(row["horizon_seconds"], "activity horizon")
        video_id = row["video_id"]
        _exact_int(row["rows"], "activity rows", nonnegative=True)
        if type(row["active"]) is not bool:
            raise ValueError("activity flag must be boolean")
        if (
            horizon not in HORIZONS
            or video_id not in dataset.SAMPLE_VIDEO_IDS
            or row["rows"] != MATCHED_COUNTS[video_id]
        ):
            raise ValueError("activity identity differs")
        persistence = _finite(row["persistence_mse"], "activity persistence", nonnegative=True)
        shuffled = _finite(row["shuffled_mse"], "activity shuffled", nonnegative=True)
        ratio, active = _activity_predicate(persistence, shuffled)
        if row["activity_ratio"] is None or not math.isclose(
            float(row["activity_ratio"]), ratio, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("derived activity ratio differs")
        if row["active"] is not active:
            raise ValueError("derived activity predicate differs")
        identities.append((horizon, video_id))
        rows.append(row)
    expected = [(horizon, video_id) for horizon in HORIZONS for video_id in dataset.SAMPLE_VIDEO_IDS]
    if identities != expected:
        raise ValueError("activity rows are incomplete or reordered")
    return rows


def validate_evidence(value: object) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "alignment",
        "synthetic_panels",
        "synthetic_fit_rows",
        "synthetic_metric_rows",
        "real_fit_rows",
        "real_metric_rows",
        "activity_rows",
    }
    if not isinstance(value, dict) or set(value) != expected_keys or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("MM-005 evidence schema differs")
    synthetic_fits = _validate_fit_rows(value["synthetic_fit_rows"], synthetic=True)
    real_fits = _validate_fit_rows(value["real_fit_rows"], synthetic=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "alignment": _validate_alignment(value["alignment"]),
        "synthetic_panels": _validate_panels(value["synthetic_panels"]),
        "synthetic_fit_rows": synthetic_fits,
        "synthetic_metric_rows": _validate_metric_rows(
            value["synthetic_metric_rows"], synthetic=True, fit_rows=synthetic_fits
        ),
        "real_fit_rows": real_fits,
        "real_metric_rows": _validate_metric_rows(value["real_metric_rows"], synthetic=False, fit_rows=real_fits),
        "activity_rows": _validate_activity(value["activity_rows"]),
    }


def _supports(row: Mapping[str, Any]) -> bool:
    ordered = float(row["ordered_mse"])
    if ordered * REAL_PERSISTENCE_FACTOR > float(row["persistence_mse"]):
        return False
    target = row["target_shuffle_mse"]
    if target is None or ordered * REAL_SHUFFLE_FACTOR > float(target):
        return False
    if ARM_BY_ID[str(row["arm_id"])].uses_history:
        history = row["history_shuffle_mse"]
        if history is None or ordered * REAL_SHUFFLE_FACTOR > float(history):
            return False
    return True


def _paired_advantage(half: Mapping[str, Any], one: Mapping[str, Any]) -> bool:
    # Zero persistence makes the pair explicitly ineligible; eligible pairs use the
    # frozen cross-product form without division.
    half_persistence = float(half["persistence_mse"])
    one_persistence = float(one["persistence_mse"])
    if half_persistence <= 0.0 or one_persistence <= 0.0:
        return False
    return bool(
        float(half["ordered_mse"]) * one_persistence * PAIRED_HORIZON_FACTOR
        <= float(one["ordered_mse"]) * half_persistence
    )


def _arm_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for horizon in HORIZONS:
            for arm in ARMS:
                selected = [
                    row
                    for row in rows
                    if row["domain"] == domain and row["horizon_seconds"] == horizon and row["arm_id"] == arm.arm_id
                ]
                support = sum(_supports(row) for row in selected)
                output.append(
                    {
                        "domain": domain,
                        "horizon_seconds": horizon,
                        "arm_id": arm.arm_id,
                        "supporting_videos": support,
                        "passes": support >= REQUIRED_VIDEO_SUPPORT,
                        "band": "pass" if support >= 6 else "borderline" if support >= 3 else "strong_fail",
                        "video_rows": selected,
                    }
                )
    return output


def _control_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for horizon in HORIZONS:
            for arm in ARMS:
                selected = [
                    row
                    for row in rows
                    if row["domain"] == domain and row["horizon_seconds"] == horizon and row["arm_id"] == arm.arm_id
                ]
                target_support = sum(
                    float(cast(float, row["target_shuffle_mse"])) * REAL_PERSISTENCE_FACTOR
                    <= float(row["persistence_mse"])
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
                        "horizon_seconds": horizon,
                        "arm_id": arm.arm_id,
                        "target_shuffle_support": target_support,
                        "history_shuffle_support": history_support,
                    }
                )
    return output


def _paired_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lookup = {(row["domain"], row["arm_id"], row["horizon_seconds"], row["video_id"]): row for row in rows}
    for domain in DOMAINS:
        for arm in ARMS:
            paired: list[dict[str, Any]] = []
            support = 0
            for video_id in dataset.SAMPLE_VIDEO_IDS:
                half = lookup[(domain, arm.arm_id, 0.5, video_id)]
                one = lookup[(domain, arm.arm_id, 1.0, video_id)]
                improved = _paired_advantage(half, one)
                support += int(improved)
                paired.append({"video_id": video_id, "half_horizon_advantage": improved})
            output.append(
                {
                    "domain": domain,
                    "arm_id": arm.arm_id,
                    "supporting_videos": support,
                    "passes": support >= REQUIRED_VIDEO_SUPPORT,
                    "paired_videos": paired,
                }
            )
    return output


def _synthetic_summary(
    fit_rows: Sequence[Mapping[str, Any]], metric_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    numerical_failures = sum(
        not bool(row["weights_finite"]) or float(row["linear_system_residual"]) > LINEAR_RESIDUAL_MAX
        for row in fit_rows
    )
    main_fits = [row for row in fit_rows if row["arm_id"] == MAIN_ARM_ID and row["control_id"] == "ordered"]
    kernel_failures = sum(float(cast(float, row["kernel_relative_error"])) > KERNEL_ERROR_MAX for row in main_fits)
    selector_failures = 0
    for row in main_fits:
        recovered = np.asarray(row["recovered_kernel"], dtype=float)
        horizon = float(row["horizon_seconds"])
        own = _expected_kernel(int(row["channels"]), horizon)
        other = _expected_kernel(int(row["channels"]), 1.0 if horizon == 0.5 else 0.5)
        selector_failures += int(np.linalg.norm(recovered - own) >= np.linalg.norm(recovered - other))
    lookup = {(row["panel_seed"], row["horizon_seconds"], row["video_id"], row["arm_id"]): row for row in metric_rows}
    conditions: list[dict[str, Any]] = []
    positive_failures = 0
    separation_failures = 0
    for seed in SYNTHETIC_SEEDS:
        for horizon in HORIZONS:
            for video_id in dataset.SAMPLE_VIDEO_IDS:
                main = lookup[(seed, horizon, video_id, MAIN_ARM_ID)]
                current = lookup[(seed, horizon, video_id, COMPARATOR_ARM.arm_id)]
                pointwise = lookup[(seed, horizon, video_id, POINTWISE_ABLATION.arm_id)]
                positive = float(main["ordered_mse"]) <= SYNTHETIC_PERSISTENCE_RATIO * float(main["persistence_mse"])
                separated = bool(
                    float(main["ordered_mse"]) <= SYNTHETIC_SEPARATION_RATIO * float(current["ordered_mse"])
                    and float(main["ordered_mse"]) <= SYNTHETIC_SEPARATION_RATIO * float(pointwise["ordered_mse"])
                    and float(main["ordered_mse"])
                    <= SYNTHETIC_SEPARATION_RATIO * float(cast(float, main["target_shuffle_mse"]))
                    and float(main["ordered_mse"])
                    <= SYNTHETIC_SEPARATION_RATIO * float(cast(float, main["history_shuffle_mse"]))
                )
                positive_failures += int(not positive)
                separation_failures += int(not separated)
                conditions.append(
                    {
                        "panel_seed": seed,
                        "horizon_seconds": horizon,
                        "video_id": video_id,
                        "positive_passes": positive,
                        "negative_passes": separated,
                    }
                )
    return {
        "conditions": conditions,
        "numerical_failures": numerical_failures,
        "kernel_failures": kernel_failures,
        "horizon_selector_failures": selector_failures,
        "positive_failures": positive_failures,
        "negative_failures": separation_failures,
        "positive_passes": numerical_failures == 0 and kernel_failures == 0 and positive_failures == 0,
        "negative_passes": selector_failures == 0 and separation_failures == 0,
        "maximum_linear_system_residual": max(float(row["linear_system_residual"]) for row in fit_rows),
        "maximum_kernel_relative_error": max(float(cast(float, row["kernel_relative_error"])) for row in main_fits),
    }


def _decision(
    *,
    synthetic_positive: bool,
    synthetic_negative: bool,
    controls: Sequence[Mapping[str, Any]],
    arms: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    half_activity_support: int,
) -> tuple[str, list[str]]:
    if not synthetic_positive:
        return "invalid_MM005_synthetic_positive_control", []
    if not synthetic_negative:
        return "invalid_MM005_synthetic_negative_control", []
    target_counts = [int(row["target_shuffle_support"]) for row in controls]
    if max(target_counts) >= 6:
        return "invalid_MM005_real_negative_control", []
    if max(target_counts) >= 3:
        return "inconclusive_MM005_real_negative_control", []
    arm_lookup = {(row["domain"], row["horizon_seconds"], row["arm_id"]): row for row in arms}
    paired_lookup = {(row["domain"], row["arm_id"]): row for row in paired}

    def support(domain: str, horizon: float, arm: ArmSpec) -> int:
        return int(arm_lookup[(domain, horizon, arm.arm_id)]["supporting_videos"])

    if any(support("taesd", 1.0, arm) >= 6 for arm in ARMS):
        return "matched_one_second_taesd_signal_supported", []
    if any(3 <= support("taesd", 1.0, arm) <= 5 for arm in ARMS):
        return "inconclusive_matched_one_second_taesd_signal", []
    for arm in ARMS:
        if (
            support("taesd", 0.5, arm) >= 6
            and support("taesd", 1.0, arm) <= 2
            and int(paired_lookup[("taesd", arm.arm_id)]["supporting_videos"]) >= 6
        ):
            return "half_second_horizon_mismatch_supported", [f"{arm.arm_id}_horizon_mismatch"]
    if any(support("taesd", 0.5, arm) >= 6 for arm in ARMS):
        return "inconclusive_half_second_taesd_signal", []
    if any(3 <= support("taesd", 0.5, arm) <= 5 for arm in ARMS):
        return "inconclusive_half_second_taesd_borderline", []
    if any(support("pixel", 1.0, arm) >= 6 for arm in ARMS):
        return "matched_one_second_pixel_signal_supported", []
    if any(3 <= support("pixel", 1.0, arm) <= 5 for arm in ARMS):
        return "inconclusive_matched_one_second_pixel_signal", []
    for arm in ARMS:
        if (
            support("pixel", 0.5, arm) >= 6
            and support("pixel", 1.0, arm) <= 2
            and int(paired_lookup[("pixel", arm.arm_id)]["supporting_videos"]) >= 6
        ):
            return "half_second_taesd_representation_failure_supported", [f"{arm.arm_id}_pixel_horizon_mismatch"]
    if any(support("pixel", 0.5, arm) >= 6 for arm in ARMS):
        return "inconclusive_half_second_pixel_signal", []
    if any(3 <= support("pixel", 0.5, arm) <= 5 for arm in ARMS):
        return "inconclusive_half_second_pixel_borderline", []
    all_half_strongly_fail = all(support(domain, 0.5, arm) <= 2 for domain in DOMAINS for arm in ARMS)
    if all_half_strongly_fail:
        if half_activity_support >= 6:
            return "half_second_tested_spatial_local_linear_objective_failure_supported", []
        if half_activity_support <= 2:
            return "insufficient_half_second_source_change", []
        if 3 <= half_activity_support <= 5:
            return "inconclusive_half_second_video_heterogeneity", []
    return "MM005_diagnostic_inconclusive", []


def summarize(evidence: object) -> dict[str, Any]:
    normalized = validate_evidence(evidence)
    synthetic = _synthetic_summary(
        cast(Sequence[Mapping[str, Any]], normalized["synthetic_fit_rows"]),
        cast(Sequence[Mapping[str, Any]], normalized["synthetic_metric_rows"]),
    )
    real = cast(Sequence[Mapping[str, Any]], normalized["real_metric_rows"])
    arms = _arm_summaries(real)
    controls = _control_summaries(real)
    paired = _paired_summaries(real)
    activity_rows = cast(Sequence[Mapping[str, Any]], normalized["activity_rows"])
    activity: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        selected = [row for row in activity_rows if row["horizon_seconds"] == horizon]
        activity.append(
            {
                "horizon_seconds": horizon,
                "supporting_videos": sum(bool(row["active"]) for row in selected),
                "video_rows": selected,
            }
        )
    half_activity = int(next(row["supporting_videos"] for row in activity if row["horizon_seconds"] == 0.5))
    classification, labels = _decision(
        synthetic_positive=bool(synthetic["positive_passes"]),
        synthetic_negative=bool(synthetic["negative_passes"]),
        controls=controls,
        arms=arms,
        paired=paired,
        half_activity_support=half_activity,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "alignment": normalized["alignment"],
        "synthetic_control": synthetic,
        "arms": arms,
        "controls": controls,
        "paired_horizon": paired,
        "activity": activity,
        "decision": {
            "classification": classification,
            "mechanism_labels": labels,
            "recommended_next_step": RECOMMENDATIONS[classification],
        },
        "claim_boundary": (
            "MM-005 is an outcome-informed eight-video local-linear horizon diagnostic; it does not establish "
            "nonlinear, population, end-to-end, or teacher-free rollout capability."
        ),
    }


def report_text(summary: Mapping[str, Any]) -> str:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("summary is not an MM-005 result")
    decision = cast(Mapping[str, Any], summary["decision"])
    synthetic = cast(Mapping[str, Any], summary["synthetic_control"])
    lines = [
        "# MM-005 matched half-horizon replay report",
        "",
        "MM-005 is outcome-informed and does not reclassify MM-001 through MM-004.",
        "",
        f"Decision classification: `{decision['classification']}`.",
        f"Recommended next step: {decision['recommended_next_step']}.",
        "",
        "## Controls",
        "",
        f"Synthetic positive control: **{'PASS' if synthetic['positive_passes'] else 'FAIL'}**.",
        f"Synthetic negative control: **{'PASS' if synthetic['negative_passes'] else 'FAIL'}**.",
        "",
        "## Matched local predictors",
        "",
        "| Domain | Horizon | Arm | Supporting videos |",
        "|---|---:|---|---:|",
    ]
    for row in cast(Sequence[Mapping[str, Any]], summary["arms"]):
        lines.append(
            f"| `{row['domain']}` | {row['horizon_seconds']:.1f} s | `{row['arm_id']}` | {row['supporting_videos']}/8 |"
        )
    lines.extend(["", "Paired half-horizon advantage:", ""])
    for row in cast(Sequence[Mapping[str, Any]], summary["paired_horizon"]):
        lines.append(f"- `{row['domain']}/{row['arm_id']}`: {row['supporting_videos']}/8 videos")
    lines.extend(["", "Pixel source activity:", ""])
    for row in cast(Sequence[Mapping[str, Any]], summary["activity"]):
        lines.append(f"- {row['horizon_seconds']:.1f} s: {row['supporting_videos']}/8 videos")
    labels = cast(Sequence[str], decision["mechanism_labels"])
    lines.extend(
        [
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
        "raw_rows": RAW_ROWS,
        "matched_rows": MATCHED_ROWS,
        "matched_counts": dict(MATCHED_COUNTS),
        "matched_identity_sha256": MATCHED_IDENTITY_SHA256,
        "horizons_seconds": list(HORIZONS),
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
        "normalizer_scope": "one train-current channel normalizer per domain/fold shared across horizons",
        "real_fit_rows": 80,
        "real_metric_rows": 64,
        "synthetic_fit_rows": 120,
        "synthetic_metric_rows": 144,
        "synthetic": {
            "seeds": list(SYNTHETIC_SEEDS),
            "generator": "numpy.random.Generator(PCG64)",
            "smoothing_passes": SYNTHETIC_SMOOTHING_PASSES,
            "target_rules": {
                "0.5": "C + 0.25 * (shift_right(C) - C) + 0.75 * D",
                "1.0": "C + 0.50 * (shift_right(C) - C) + 1.50 * D",
            },
            "arms_and_controls": {
                "current_diff_3x3": ["ordered", "target_shuffle", "history_shuffle"],
                "current_3x3": ["ordered"],
                "current_diff_1x1": ["ordered"],
            },
            "persistence_ratio": SYNTHETIC_PERSISTENCE_RATIO,
            "separation_ratio": SYNTHETIC_SEPARATION_RATIO,
            "linear_residual_max": LINEAR_RESIDUAL_MAX,
            "kernel_error_max": KERNEL_ERROR_MAX,
            "kernel_selector": "strict unnormalized Frobenius distance to own versus other horizon kernel",
        },
        "thresholds": {
            "real_persistence_factor": REAL_PERSISTENCE_FACTOR,
            "real_shuffle_factor": REAL_SHUFFLE_FACTOR,
            "paired_horizon_factor": PAIRED_HORIZON_FACTOR,
            "required_video_support": REQUIRED_VIDEO_SUPPORT,
            "activity_mse_min": ACTIVITY_MSE_MIN,
            "activity_ratio_min": ACTIVITY_RATIO_MIN,
            "activity_ratio_max": ACTIVITY_RATIO_MAX,
        },
        "target_rules": {
            "source_positions": "i=1..N-3",
            "previous": "current[i-1]",
            "current": "current[i]",
            "0.5": "saved_target[i-1] == current[i+1]",
            "1.0": "saved_target[i] == current[i+2]",
        },
        "velocity_rule": "current + (horizon_seconds / 0.5) * (current - previous)",
        "paired_rule": (
            "persistence_0p5 > 0 and persistence_1p0 > 0 and "
            "ordered_0p5 * persistence_1p0 * 1.10 <= ordered_1p0 * persistence_0p5"
        ),
        "folds": [
            {
                "fold": fold.index,
                "train_ids": list(fold.train_ids),
                "test_ids": list(fold.test_ids),
                "train_rows": sum(MATCHED_COUNTS[video_id] for video_id in fold.train_ids),
            }
            for fold in dataset.formal_folds()
        ],
        "controls": {
            "target_shuffle": {
                "scope": "training targets only",
                "derangement": "within-video half-cycle fixed-point-free",
                "shared_across_horizons": True,
                "global_null_bands": {"invalid": [6, 8], "inconclusive": [3, 5], "clear": [0, 2]},
            },
            "history_shuffle": {
                "scope": "training previous grids only",
                "derangement": "within-video half-cycle fixed-point-free",
                "shared_across_horizons": True,
                "global_shortcut_check": False,
            },
        },
        "support_rules": {
            "current_3x3": [
                "ordered_mse * 1.20 <= persistence_mse",
                "ordered_mse * 1.10 <= target_shuffle_mse",
            ],
            "current_diff_3x3": [
                "ordered_mse * 1.20 <= persistence_mse",
                "ordered_mse * 1.10 <= target_shuffle_mse",
                "ordered_mse * 1.10 <= history_shuffle_mse",
            ],
            "bands": {"pass": [6, 8], "borderline": [3, 5], "strong_fail": [0, 2]},
        },
        "activity_rule": ("P >= 1e-4 and 0.10 <= P / max(S, 1e-15) <= 1/1.2 on full 8x8 pixel grids"),
        "decision_order": [
            "invalid_MM005_parent_alignment_pre_marker",
            "invalid_MM005_synthetic_positive_control",
            "invalid_MM005_synthetic_negative_control",
            "invalid_or_inconclusive_MM005_real_negative_control",
            "matched_or_inconclusive_matched_one_second_taesd_signal",
            "half_second_horizon_mismatch_supported",
            "inconclusive_half_second_taesd_signal_or_borderline",
            "matched_or_inconclusive_matched_one_second_pixel_signal",
            "half_second_taesd_representation_failure_supported",
            "inconclusive_half_second_pixel_signal_or_borderline",
            "half_second_tested_spatial_local_linear_objective_failure_supported",
            "insufficient_or_heterogeneous_half_second_source_change",
            "MM005_diagnostic_inconclusive",
        ],
        "claim_boundary": "central 6x6 diagnostic only; no teacher-free rollout or population claim",
        "row_schemas": {
            "fit_rows": sorted(FIT_ROW_KEYS),
            "metric_rows": sorted(METRIC_ROW_KEYS),
            "synthetic_panels": sorted(PANEL_ROW_KEYS),
            "activity_rows": sorted(ACTIVITY_ROW_KEYS),
            "alignment": sorted(ALIGNMENT_KEYS),
        },
        "recommendations": dict(RECOMMENDATIONS),
    }


__all__ = [
    "ARMS",
    "HORIZONS",
    "MATCHED_COUNTS",
    "MATCHED_IDENTITY_SHA256",
    "MATCHED_ROWS",
    "MatchedPanel",
    "RawGridTable",
    "SCHEMA_VERSION",
    "SYNTHETIC_SEEDS",
    "alignment_record",
    "config_record",
    "execute",
    "half_cycle_derangement",
    "matched_panel",
    "panel_provenance",
    "raw_grid_table",
    "report_text",
    "summarize",
    "synthetic_panel",
    "validate_evidence",
]
