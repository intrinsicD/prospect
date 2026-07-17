"""Pure numpy engine for the frozen MM-002 failure-isolation experiment.

The engine consumes only MM-001's frozen 32-dimensional feature table.  It does
not import media or neural-model backends and performs no filesystem writes.  A
separate orchestration module owns preparation, formal markers, packaging, and
verification.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Final, cast

import numpy as np

from bench.multimodal_preflight import core, dataset
from prospect.codec import UniversalCodec
from prospect.types import LatentState, Modality, Observation, Transition
from prospect.world_model import FlatWorldModel

SCHEMA_VERSION: Final = "mm002-method-v1"
PARITY_RTOL: Final = 1e-12
PARITY_ATOL: Final = 1e-12
RAW_RIDGE_PENALTY: Final = 1e-3
INTEGRITY_MIN_STD: Final = 0.3
INTEGRITY_MIN_EFFECTIVE_RANK: Final = 2.0
CODEC_FINGERPRINT_VERSION: Final = "mm002-codec-encoder-v1"
MATCHED_ROWS: Final = 461
MATCHED_COUNTS: Final[dict[str, int]] = {
    "video_10993": 61,
    "video_1580": 62,
    "video_2564": 57,
    "video_3501": 63,
    "video_6860": 63,
    "video_8241": 46,
    "video_874": 64,
    "video_9253": 45,
}


@dataclass(frozen=True, slots=True)
class WorldVariantSpec:
    """One frozen world endpoint, possibly a checkpoint on a longer trajectory."""

    variant_id: str
    trajectory_id: str
    horizon_seconds: float
    updates: int
    matched_sources: bool


@dataclass(frozen=True, slots=True)
class CodecVariantSpec:
    """One frozen vision-codec endpoint."""

    variant_id: str
    included_modalities: tuple[str, ...]
    update_order: tuple[str, ...]
    cycles: int
    snapshot: str


WORLD_VARIANTS: Final = (
    WorldVariantSpec("full_1s_1500", "full_1s", 1.0, 1_500, False),
    WorldVariantSpec("matched_0p5s_1500", "matched_0p5s", 0.5, 1_500, True),
    WorldVariantSpec("matched_1s_1500", "matched_1s", 1.0, 1_500, True),
    WorldVariantSpec("matched_1s_3000", "matched_1s", 1.0, 3_000, True),
    WorldVariantSpec("matched_2s_1500", "matched_2s", 2.0, 1_500, True),
    WorldVariantSpec("matched_1s_6000", "matched_1s", 1.0, 6_000, True),
)

CODEC_VARIANTS: Final = (
    CodecVariantSpec("shared_vat_600", ("vision", "audio", "text"), ("vision", "audio", "text"), 600, "after_cycle"),
    CodecVariantSpec("shared_atv_600", ("vision", "audio", "text"), ("audio", "text", "vision"), 600, "after_cycle"),
    CodecVariantSpec("vision_only_600", ("vision",), ("vision",), 600, "after_cycle"),
    CodecVariantSpec(
        "shared_vat_2400_after_v",
        ("vision", "audio", "text"),
        ("vision", "audio", "text"),
        2_400,
        "after_final_vision_before_audio_text",
    ),
    CodecVariantSpec("shared_vat_2400", ("vision", "audio", "text"), ("vision", "audio", "text"), 2_400, "after_cycle"),
    CodecVariantSpec("vision_only_2400", ("vision",), ("vision",), 2_400, "after_cycle"),
)

RAW_PROBE_HORIZONS: Final = (("matched_0p5s", 0.5), ("matched_1s", 1.0), ("matched_2s", 2.0))
INTEGRITY_CHECKPOINTS: Final[dict[str, tuple[int, ...]]] = {
    "full_1s": (300, 600, 1_500),
    "matched_0p5s": (300, 600, 1_500),
    "matched_1s": (300, 600, 1_500, 3_000, 6_000),
    "matched_2s": (300, 600, 1_500),
}

WORLD_METRICS: Final = (
    "world_mse",
    "persistence_mse",
    "ridge_mse",
    "shuffle_model_mse",
    "shuffle_model_persistence_mse",
)
RAW_METRICS: Final = ("raw_persistence_mse", "raw_ridge_mse", "raw_shuffle_ridge_mse")
CODEC_METRICS: Final = (
    "incumbent_world_mse",
    "vision_mse",
    "vision_latent_mse",
    "vision_to_incumbent_ratio",
)
WORLD_ROW_KEYS: Final = {
    "variant_id",
    "trajectory_id",
    "horizon_seconds",
    "updates",
    "matched_sources",
    "train_rows",
    "test_rows",
    "video_id",
    "fold",
    "seed",
    "model_fingerprint",
    "shuffle_model_fingerprint",
    *WORLD_METRICS,
}
RAW_PROBE_ROW_KEYS: Final = {
    "probe_id",
    "horizon_seconds",
    "train_rows",
    "test_rows",
    "video_id",
    "fold",
    *RAW_METRICS,
}
INTEGRITY_ROW_KEYS: Final = {
    "trajectory_id",
    "horizon_seconds",
    "updates",
    "fold",
    "seed",
    "model_role",
    "encoder_role",
    "pooled_test_rows",
    "model_fingerprint",
    "latent_std_min",
    "latent_effective_rank",
    "prediction_min_variance",
    "prediction_finite",
    "prediction_variance_positive",
}
CODEC_ROW_KEYS: Final = {
    "variant_id",
    "included_modalities",
    "update_order",
    "cycles",
    "snapshot",
    "train_rows",
    "test_rows",
    "video_id",
    "fold",
    "seed",
    "model_fingerprint",
    "codec_fingerprint",
    *CODEC_METRICS,
}


def config_record() -> dict[str, Any]:
    """Return a JSON-serializable record of every frozen method choice."""

    return {
        "schema_version": SCHEMA_VERSION,
        "feature_dim": core.FEATURE_DIM,
        "latent_dim": core.LATENT_DIM,
        "folds": [
            {
                "fold": fold.index,
                "train_ids": list(fold.train_ids),
                "test_ids": list(fold.test_ids),
            }
            for fold in dataset.formal_folds()
        ],
        "seeds": list(core.SEEDS),
        "full_rows": int(sum(dataset.EXPECTED_WINDOW_COUNTS.values())),
        "matched_rows": MATCHED_ROWS,
        "matched_counts": dict(MATCHED_COUNTS),
        "matched_source_drop_per_video": 2,
        "matched_target_rules": {
            "0.5": "vision[i+1]",
            "1.0": "target_vision[i]",
            "2.0": "target_vision[i+2]",
        },
        "world_variants": [asdict(spec) for spec in WORLD_VARIANTS],
        "world_integrity_checkpoints": {key: list(value) for key, value in INTEGRITY_CHECKPOINTS.items()},
        "world_trajectory_reuse": {
            "matched_1s": "one uninterrupted trajectory checkpointed at 1500, 3000, and 6000 updates",
            "checkpoint_copy": "deepcopy_without_rng_consumption",
        },
        "world": {
            "batch": core.WORLD_BATCH,
            "hidden": core.WORLD_HIDDEN,
            "ensemble": core.WORLD_ENSEMBLE,
            "learning_rate": core.WORLD_LR,
            "ema_tau": core.WORLD_EMA_TAU,
            "reward_weight": core.WORLD_W_REWARD,
            "inverse_weight": core.WORLD_W_INVERSE,
            "variance_weight": core.WORLD_W_VAR,
            "covariance_weight": core.WORLD_W_COV,
            "sample_seed_offset": core.WORLD_SAMPLE_SEED_OFFSET,
            "temporal_control": "within_video_half_cycle_derangement",
        },
        "raw_probe_horizons": [{"probe_id": name, "horizon_seconds": horizon} for name, horizon in RAW_PROBE_HORIZONS],
        "raw_ridge_penalty": RAW_RIDGE_PENALTY,
        "codec_variants": [
            {
                **asdict(spec),
                "included_modalities": list(spec.included_modalities),
                "update_order": list(spec.update_order),
            }
            for spec in CODEC_VARIANTS
        ],
        "codec": {
            "batch": core.CODEC_BATCH,
            "token_dim": core.CODEC_TOKEN_DIM,
            "hidden": core.CODEC_HIDDEN,
            "learning_rate": core.CODEC_LR,
            "initialization_seed_offset": core.CODEC_INIT_SEED_OFFSET,
            "sample_seed_offset": core.CODEC_SAMPLE_SEED_OFFSET,
            "initial_full_fold_update": True,
            "decoder_updates": False,
            "trained_trajectories": ["shared_vat", "shared_atv", "vision_only"],
            "fingerprint_version": CODEC_FINGERPRINT_VERSION,
            "encoder_update_counts": {
                "shared_vat_600": {"vision": 601, "audio": 601, "text": 601, "shared_trunk": 1_803},
                "shared_atv_600": {"vision": 601, "audio": 601, "text": 601, "shared_trunk": 1_803},
                "vision_only_600": {"vision": 601, "shared_trunk": 601},
                "shared_vat_2400_after_v": {
                    "vision": 2_401,
                    "audio": 2_400,
                    "text": 2_400,
                    "shared_trunk": 7_201,
                },
                "shared_vat_2400": {
                    "vision": 2_401,
                    "audio": 2_401,
                    "text": 2_401,
                    "shared_trunk": 7_203,
                },
                "vision_only_2400": {"vision": 2_401, "shared_trunk": 2_401},
            },
        },
        "thresholds": {
            "visual_persistence_factor": core.VISUAL_PERSISTENCE_FACTOR,
            "visual_shuffle_margin": core.VISUAL_SHUFFLE_MARGIN,
            "raw_persistence_factor": 1.2,
            "raw_shuffle_margin": 1.1,
            "vision_codec_factor": core.VISION_CODEC_FACTOR,
            "required_video_support": core.REQUIRED_VIDEO_SUPPORT,
            "formal_video_count": core.FORMAL_VIDEO_COUNT,
            "integrity_min_std": INTEGRITY_MIN_STD,
            "integrity_min_effective_rank": INTEGRITY_MIN_EFFECTIVE_RANK,
        },
        "parity": {"rtol": PARITY_RTOL, "atol": PARITY_ATOL},
        "row_schemas": {
            "world_rows": sorted(WORLD_ROW_KEYS),
            "raw_probe_rows": sorted(RAW_PROBE_ROW_KEYS),
            "integrity_rows": sorted(INTEGRITY_ROW_KEYS),
            "codec_rows": sorted(CODEC_ROW_KEYS),
        },
    }


def _validate_formal_table(table: core.FeatureTable) -> None:
    table.validate()
    ids = np.asarray(table.video_ids, dtype=str)
    timestamps = np.asarray(table.timestamps, dtype=float)
    expected_ids = tuple(dataset.SAMPLE_VIDEO_IDS)
    if tuple(sorted(set(ids))) != expected_ids:
        raise ValueError("MM-002 requires the exact eight MM-001 video IDs")
    if len(ids) != sum(dataset.EXPECTED_WINDOW_COUNTS.values()):
        raise ValueError("MM-002 requires the exact 477-row MM-001 feature table")
    expected_identity: list[tuple[str, float]] = []
    for video_id in expected_ids:
        count = dataset.EXPECTED_WINDOW_COUNTS[video_id]
        expected_identity.extend((video_id, 1.0 + 0.5 * index) for index in range(count))
    actual_identity = [(str(video_id), float(timestamp)) for video_id, timestamp in zip(ids, timestamps, strict=True)]
    if len(actual_identity) != len(expected_identity) or any(
        actual_id != expected_id or not math.isclose(actual_t, expected_t, abs_tol=1e-12, rel_tol=0.0)
        for (actual_id, actual_t), (expected_id, expected_t) in zip(actual_identity, expected_identity, strict=True)
    ):
        raise ValueError("MM-002 feature rows are not in the frozen video/timestamp order")


def matched_horizon_table(table: core.FeatureTable, horizon_seconds: float) -> core.FeatureTable:
    """Build the common 461-source-row table for one frozen target horizon.

    The final two source rows of each video are always removed.  Only the target
    feature array varies across horizons; all source arrays and identities remain
    byte-for-byte equal.
    """

    if horizon_seconds not in (0.5, 1.0, 2.0):
        raise ValueError("matched horizon must be exactly 0.5, 1.0, or 2.0 seconds")
    table.validate()
    ids = np.asarray(table.video_ids, dtype=str)
    timestamps = np.asarray(table.timestamps, dtype=float)
    source_indices: list[int] = []
    target_rows: list[np.ndarray] = []
    for video_id in sorted(set(ids)):
        rows = np.flatnonzero(ids == video_id)
        ordered = rows[np.argsort(timestamps[rows], kind="stable")]
        if len(ordered) < 3:
            raise ValueError(f"video {video_id} needs at least three rows for the matched ladder")
        for position, source in enumerate(ordered[:-2]):
            source_indices.append(int(source))
            if horizon_seconds == 0.5:
                target_rows.append(np.asarray(table.vision[ordered[position + 1]], dtype=float))
            elif horizon_seconds == 1.0:
                target_rows.append(np.asarray(table.target_vision[source], dtype=float))
            else:
                target_rows.append(np.asarray(table.target_vision[ordered[position + 2]], dtype=float))
    selected = np.asarray(source_indices, dtype=int)
    output = core.FeatureTable(
        video_ids=np.asarray(table.video_ids)[selected].copy(),
        timestamps=np.asarray(table.timestamps, dtype=float)[selected].copy(),
        vision=np.asarray(table.vision, dtype=float)[selected].copy(),
        audio=np.asarray(table.audio, dtype=float)[selected].copy(),
        text=np.asarray(table.text, dtype=float)[selected].copy(),
        target_vision=np.stack(target_rows),
        annotation_present=np.asarray(table.annotation_present, dtype=bool)[selected].copy(),
    )
    output.validate()
    return output


def _world_model(seed: int) -> FlatWorldModel:
    return FlatWorldModel(
        obs_dim=core.FEATURE_DIM,
        action_dim=1,
        latent_dim=core.LATENT_DIM,
        hidden=core.WORLD_HIDDEN,
        ensemble=core.WORLD_ENSEMBLE,
        lr=core.WORLD_LR,
        ema_tau=core.WORLD_EMA_TAU,
        w_reward=core.WORLD_W_REWARD,
        w_inverse=core.WORLD_W_INVERSE,
        w_var=core.WORLD_W_VAR,
        w_cov=core.WORLD_W_COV,
        seed=seed,
    )


def _transitions(table: core.FeatureTable, *, shuffled: bool) -> list[Transition]:
    targets = np.asarray(table.target_vision, dtype=float)
    if shuffled:
        targets = targets[core.temporal_derangement(table)]
    return [
        Transition(
            state=LatentState(z=np.asarray(source, dtype=float)),
            action=core.NULL_ACTION,
            next_state=LatentState(z=np.asarray(target, dtype=float)),
            reward=0.0,
        )
        for source, target in zip(table.vision, targets, strict=True)
    ]


def _fit_world_trajectory(
    table: core.FeatureTable,
    seed: int,
    checkpoints: Sequence[int],
    *,
    shuffled: bool,
) -> dict[int, FlatWorldModel]:
    """Fit once and deep-copy exact checkpoints without advancing model RNG."""

    wanted = tuple(sorted(set(checkpoints)))
    if not wanted or wanted[0] <= 0:
        raise ValueError("world checkpoints must be positive")
    transitions = _transitions(table, shuffled=shuffled)
    model = _world_model(seed)
    rng = np.random.default_rng(seed + core.WORLD_SAMPLE_SEED_OFFSET)
    batch_size = min(core.WORLD_BATCH, len(transitions))
    output: dict[int, FlatWorldModel] = {}
    for completed in range(1, wanted[-1] + 1):
        indices = rng.integers(0, len(transitions), size=batch_size)
        model.update([transitions[int(index)] for index in indices])
        if completed in wanted:
            output[completed] = deepcopy(model)
    return output


def _target_latents(model: FlatWorldModel, values: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(model.encode_target(row).z, dtype=float) for row in values])


def _online_latents(model: FlatWorldModel, values: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(model.encode(row).z, dtype=float) for row in values])


def _predictions(model: FlatWorldModel, latents: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.asarray(model.predict(LatentState(z=row), core.NULL_ACTION).mean, dtype=float) for row in latents]
    )


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))


def _latent_ridge(model: FlatWorldModel, train: core.FeatureTable, test: core.FeatureTable) -> np.ndarray:
    targets = _target_latents(model, train.target_vision)
    x_train = np.c_[train.vision, np.ones(len(train.video_ids))]
    x_test = np.c_[test.vision, np.ones(len(test.video_ids))]
    penalty = core.RIDGE_PENALTY * np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ targets)
    return np.asarray(x_test @ weights, dtype=float)


def _world_metrics(
    model: FlatWorldModel,
    shuffled_model: FlatWorldModel,
    train: core.FeatureTable,
    test: core.FeatureTable,
) -> dict[str, float]:
    targets = _target_latents(model, test.target_vision)
    current = _target_latents(model, test.vision)
    prediction = _predictions(model, _online_latents(model, test.vision))
    shuffled_targets = _target_latents(shuffled_model, test.target_vision)
    shuffled_current = _target_latents(shuffled_model, test.vision)
    shuffled_prediction = _predictions(shuffled_model, _online_latents(shuffled_model, test.vision))
    return {
        "world_mse": _mse(prediction, targets),
        "persistence_mse": _mse(current, targets),
        "ridge_mse": _mse(_latent_ridge(model, train, test), targets),
        "shuffle_model_mse": _mse(shuffled_prediction, shuffled_targets),
        "shuffle_model_persistence_mse": _mse(shuffled_current, shuffled_targets),
    }


def _raw_ridge_rows(
    probe_id: str,
    horizon_seconds: float,
    fold: dataset.DatasetFold,
    table: core.FeatureTable,
) -> list[dict[str, Any]]:
    train = table.subset(fold.train_ids)
    x_train = np.c_[train.vision, np.ones(len(train.video_ids))]
    penalty = RAW_RIDGE_PENALTY * np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0
    ordered_weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train.target_vision)
    derangement = core.temporal_derangement(train)
    shuffled_weights = np.linalg.solve(
        x_train.T @ x_train + penalty,
        x_train.T @ np.asarray(train.target_vision, dtype=float)[derangement],
    )
    rows: list[dict[str, Any]] = []
    for video_id in fold.test_ids:
        test = table.subset([video_id])
        x_test = np.c_[test.vision, np.ones(len(test.video_ids))]
        rows.append(
            {
                "probe_id": probe_id,
                "horizon_seconds": horizon_seconds,
                "train_rows": len(train.video_ids),
                "test_rows": len(test.video_ids),
                "video_id": video_id,
                "fold": fold.index,
                "raw_persistence_mse": _mse(test.vision, test.target_vision),
                "raw_ridge_mse": _mse(x_test @ ordered_weights, test.target_vision),
                "raw_shuffle_ridge_mse": _mse(x_test @ shuffled_weights, test.target_vision),
            }
        )
    return rows


def _effective_rank(latents: np.ndarray) -> float:
    covariance = np.cov(np.asarray(latents, dtype=float).T) + 1e-8 * np.eye(latents.shape[1])
    eigenvalues = np.linalg.eigvalsh(covariance)
    denominator = float(np.sum(eigenvalues**2))
    return float(np.sum(eigenvalues) ** 2 / denominator)


def _integrity_rows_for_model(
    *,
    trajectory_id: str,
    horizon_seconds: float,
    updates: int,
    fold: dataset.DatasetFold,
    seed: int,
    model_role: str,
    model: FlatWorldModel,
    pooled_test: core.FeatureTable,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fingerprint = core.model_fingerprint(model)
    for encoder_role in ("online", "ema_target"):
        latent_states = (
            [model.encode(row) for row in pooled_test.vision]
            if encoder_role == "online"
            else [model.encode_target(row) for row in pooled_test.vision]
        )
        latents = np.stack([np.asarray(state.z, dtype=float) for state in latent_states])
        predictions = [model.predict(state, core.NULL_ACTION) for state in latent_states]
        variances = np.stack([np.asarray(prediction.var, dtype=float) for prediction in predictions])
        finite = all(
            np.all(np.isfinite(prediction.mean))
            and np.all(np.isfinite(prediction.var))
            and math.isfinite(float(prediction.epistemic))
            and math.isfinite(float(prediction.aleatoric))
            for prediction in predictions
        )
        positive = bool(np.all(variances > 0.0))
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "horizon_seconds": horizon_seconds,
                "updates": updates,
                "fold": fold.index,
                "seed": seed,
                "model_role": model_role,
                "encoder_role": encoder_role,
                "pooled_test_rows": len(pooled_test.video_ids),
                "model_fingerprint": fingerprint,
                "latent_std_min": float(np.std(latents, axis=0, ddof=0).min()),
                "latent_effective_rank": _effective_rank(latents),
                "prediction_min_variance": float(variances.min()),
                "prediction_finite": bool(finite),
                "prediction_variance_positive": positive,
            }
        )
    return rows


def _codec(
    model: FlatWorldModel,
    table: core.FeatureTable,
    seed: int,
    modalities: tuple[Modality, ...],
) -> tuple[UniversalCodec, dict[Modality, np.ndarray], dict[Modality, np.ndarray], np.random.Generator]:
    codec = UniversalCodec(
        {modality: core.FEATURE_DIM for modality in modalities},
        latent_dim=model.latent_dim,
        token_dim=core.CODEC_TOKEN_DIM,
        hidden=core.CODEC_HIDDEN,
        lr=core.CODEC_LR,
        seed=seed + core.CODEC_INIT_SEED_OFFSET,
    )
    targets = _online_latents(model, table.vision)
    features = table.features()
    for modality in modalities:
        codec.distill_encode(features[modality], modality, targets)
    return codec, features, {modality: targets for modality in modalities}, np.random.default_rng(
        seed + core.CODEC_SAMPLE_SEED_OFFSET
    )


def _fit_codec_snapshots(
    model: FlatWorldModel,
    table: core.FeatureTable,
    seed: int,
) -> dict[str, UniversalCodec]:
    """Fit the three frozen trajectories and copy all six requested endpoints."""

    output: dict[str, UniversalCodec] = {}
    batch_size = min(core.CODEC_BATCH, len(table.video_ids))

    vat, features, targets, rng = _codec(model, table, seed, core.MODALITIES)
    for cycle in range(1, 2_401):
        indices = rng.integers(0, len(table.video_ids), size=batch_size)
        vat.distill_encode(features[Modality.VISION][indices], Modality.VISION, targets[Modality.VISION][indices])
        if cycle == 2_400:
            output["shared_vat_2400_after_v"] = deepcopy(vat)
        vat.distill_encode(features[Modality.AUDIO][indices], Modality.AUDIO, targets[Modality.AUDIO][indices])
        vat.distill_encode(features[Modality.TEXT][indices], Modality.TEXT, targets[Modality.TEXT][indices])
        if cycle == 600:
            output["shared_vat_600"] = deepcopy(vat)
        if cycle == 2_400:
            output["shared_vat_2400"] = deepcopy(vat)

    atv, features, targets, rng = _codec(model, table, seed, core.MODALITIES)
    for _ in range(600):
        indices = rng.integers(0, len(table.video_ids), size=batch_size)
        for modality in (Modality.AUDIO, Modality.TEXT, Modality.VISION):
            atv.distill_encode(features[modality][indices], modality, targets[modality][indices])
    output["shared_atv_600"] = deepcopy(atv)

    vision_only, features, targets, rng = _codec(model, table, seed, (Modality.VISION,))
    for cycle in range(1, 2_401):
        indices = rng.integers(0, len(table.video_ids), size=batch_size)
        vision_only.distill_encode(
            features[Modality.VISION][indices],
            Modality.VISION,
            targets[Modality.VISION][indices],
        )
        if cycle == 600:
            output["vision_only_600"] = deepcopy(vision_only)
        if cycle == 2_400:
            output["vision_only_2400"] = deepcopy(vision_only)
    if set(output) != {spec.variant_id for spec in CODEC_VARIANTS}:
        raise AssertionError("codec checkpoint set does not match the frozen variants")
    return output


def codec_fingerprint(codec: UniversalCodec) -> str:
    """Hash all encoder-side parameters and frozen input statistics."""

    digest = sha256()
    digest.update(CODEC_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    modality_dims = codec._modality_dims  # noqa: SLF001 - ordered constructor metadata is bound evidence
    digest.update(f"latent_dim:{codec.latent_dim};modalities:{len(modality_dims)};".encode("ascii"))
    for modality, dimension in modality_dims.items():
        digest.update(f"{modality.value}:{dimension};".encode("ascii"))
    for modality, adapter in codec._adapters.items():  # noqa: SLF001 - experiment binds learned state
        digest.update(modality.value.encode("ascii"))
        for array in (*adapter.weights, *adapter.biases):
            value = np.asarray(array, dtype="<f8", order="C")
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        mean, std = codec._stats[modality]  # noqa: SLF001 - frozen normalization is encoder state
        for array in (mean, std):
            value = np.asarray(array, dtype="<f8", order="C")
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    for array in (*codec._trunk.weights, *codec._trunk.biases):  # noqa: SLF001
        value = np.asarray(array, dtype="<f8", order="C")
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _codec_metrics(
    model: FlatWorldModel,
    codec: UniversalCodec,
    test: core.FeatureTable,
) -> dict[str, float]:
    targets = _target_latents(model, test.target_vision)
    incumbent_latents = _online_latents(model, test.vision)
    incumbent_predictions = _predictions(model, incumbent_latents)
    codec_latents = np.stack(
        [
            np.asarray(codec.encode(Observation(Modality.VISION, row)).z, dtype=float)
            for row in test.vision
        ]
    )
    vision_predictions = _predictions(model, codec_latents)
    incumbent_mse = _mse(incumbent_predictions, targets)
    vision_mse = _mse(vision_predictions, targets)
    return {
        "incumbent_world_mse": incumbent_mse,
        "vision_mse": vision_mse,
        "vision_latent_mse": _mse(codec_latents, incumbent_latents),
        "vision_to_incumbent_ratio": float(vision_mse / incumbent_mse),
    }


def execute(table: core.FeatureTable) -> dict[str, list[dict[str, Any]]]:
    """Execute every frozen fit and return four JSON-serializable evidence arrays."""

    _validate_formal_table(table)
    matched_tables = {
        0.5: matched_horizon_table(table, 0.5),
        1.0: matched_horizon_table(table, 1.0),
        2.0: matched_horizon_table(table, 2.0),
    }
    for horizon_table in matched_tables.values():
        counts = {
            video_id: int(np.sum(np.asarray(horizon_table.video_ids, dtype=str) == video_id))
            for video_id in dataset.SAMPLE_VIDEO_IDS
        }
        if counts != MATCHED_COUNTS or len(horizon_table.video_ids) != MATCHED_ROWS:
            raise ValueError("matched source rows do not reproduce the frozen 461-row panel")

    world_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    trajectory_tables = {
        "full_1s": table,
        "matched_0p5s": matched_tables[0.5],
        "matched_1s": matched_tables[1.0],
        "matched_2s": matched_tables[2.0],
    }
    trajectory_horizons = {"full_1s": 1.0, "matched_0p5s": 0.5, "matched_1s": 1.0, "matched_2s": 2.0}

    for fold in dataset.formal_folds():
        for probe_id, horizon in RAW_PROBE_HORIZONS:
            raw_rows.extend(_raw_ridge_rows(probe_id, horizon, fold, matched_tables[horizon]))
        for seed in core.SEEDS:
            primary_checkpoints: dict[str, dict[int, FlatWorldModel]] = {}
            shuffle_checkpoints: dict[str, dict[int, FlatWorldModel]] = {}
            for trajectory_id, checkpoints in INTEGRITY_CHECKPOINTS.items():
                train = trajectory_tables[trajectory_id].subset(fold.train_ids)
                primary_checkpoints[trajectory_id] = _fit_world_trajectory(
                    train, seed, checkpoints, shuffled=False
                )
                shuffle_checkpoints[trajectory_id] = _fit_world_trajectory(
                    train, seed, checkpoints, shuffled=True
                )

            for trajectory_id, checkpoints in INTEGRITY_CHECKPOINTS.items():
                pooled_test = trajectory_tables[trajectory_id].subset(fold.test_ids)
                for updates in checkpoints:
                    for model_role, model in (
                        ("primary", primary_checkpoints[trajectory_id][updates]),
                        ("shuffle", shuffle_checkpoints[trajectory_id][updates]),
                    ):
                        integrity_rows.extend(
                            _integrity_rows_for_model(
                                trajectory_id=trajectory_id,
                                horizon_seconds=trajectory_horizons[trajectory_id],
                                updates=updates,
                                fold=fold,
                                seed=seed,
                                model_role=model_role,
                                model=model,
                                pooled_test=pooled_test,
                            )
                        )

            for world_spec in WORLD_VARIANTS:
                variant_table = (
                    table if not world_spec.matched_sources else matched_tables[world_spec.horizon_seconds]
                )
                train = variant_table.subset(fold.train_ids)
                model = primary_checkpoints[world_spec.trajectory_id][world_spec.updates]
                shuffled_model = shuffle_checkpoints[world_spec.trajectory_id][world_spec.updates]
                for video_id in fold.test_ids:
                    test = variant_table.subset([video_id])
                    world_rows.append(
                        {
                            "variant_id": world_spec.variant_id,
                            "trajectory_id": world_spec.trajectory_id,
                            "horizon_seconds": world_spec.horizon_seconds,
                            "updates": world_spec.updates,
                            "matched_sources": world_spec.matched_sources,
                            "train_rows": len(train.video_ids),
                            "test_rows": len(test.video_ids),
                            "video_id": video_id,
                            "fold": fold.index,
                            "seed": seed,
                            "model_fingerprint": core.model_fingerprint(model),
                            "shuffle_model_fingerprint": core.model_fingerprint(shuffled_model),
                            **_world_metrics(model, shuffled_model, train, test),
                        }
                    )

            full_model = primary_checkpoints["full_1s"][1_500]
            full_train = table.subset(fold.train_ids)
            codec_snapshots = _fit_codec_snapshots(full_model, full_train, seed)
            model_fingerprint = core.model_fingerprint(full_model)
            for codec_spec in CODEC_VARIANTS:
                codec = codec_snapshots[codec_spec.variant_id]
                fingerprint = codec_fingerprint(codec)
                for video_id in fold.test_ids:
                    test = table.subset([video_id])
                    codec_rows.append(
                        {
                            "variant_id": codec_spec.variant_id,
                            "included_modalities": list(codec_spec.included_modalities),
                            "update_order": list(codec_spec.update_order),
                            "cycles": codec_spec.cycles,
                            "snapshot": codec_spec.snapshot,
                            "train_rows": len(full_train.video_ids),
                            "test_rows": len(test.video_ids),
                            "video_id": video_id,
                            "fold": fold.index,
                            "seed": seed,
                            "model_fingerprint": model_fingerprint,
                            "codec_fingerprint": fingerprint,
                            **_codec_metrics(full_model, codec, test),
                        }
                    )

    return validate_evidence(
        {
            "world_rows": world_rows,
            "raw_probe_rows": raw_rows,
            "integrity_rows": integrity_rows,
            "codec_rows": codec_rows,
        }
    )


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if positive and nonnegative:
        raise ValueError("a numeric validator cannot request both positive and nonnegative")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    invalid_sign = (positive and result <= 0.0) or (nonnegative and result < 0.0)
    if not math.isfinite(result) or invalid_sign:
        qualifier = "finite and positive" if positive else "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _expected_world_identities() -> list[tuple[int, int, str, str]]:
    return [
        (fold.index, seed, spec.variant_id, video_id)
        for fold in dataset.formal_folds()
        for seed in core.SEEDS
        for spec in WORLD_VARIANTS
        for video_id in fold.test_ids
    ]


def _fold(index: int) -> dataset.DatasetFold:
    try:
        return dataset.formal_folds()[index]
    except IndexError as error:
        raise ValueError(f"unknown formal fold {index}") from error


def _expected_row_counts(
    fold_index: int,
    video_id: str,
    *,
    matched: bool,
) -> tuple[int, int]:
    fold = _fold(fold_index)
    counts: Mapping[str, int] = MATCHED_COUNTS if matched else dataset.EXPECTED_WINDOW_COUNTS
    return sum(counts[item] for item in fold.train_ids), counts[video_id]


def validate_world_rows(value: object) -> list[dict[str, Any]]:
    """Validate exact world row schema, types, values, and frozen order."""

    if not isinstance(value, list):
        raise ValueError("world_rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, int, str, str]] = []
    specs = {spec.variant_id: spec for spec in WORLD_VARIANTS}
    fingerprints: dict[tuple[int, int, str], tuple[str, str]] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != WORLD_ROW_KEYS:
            raise ValueError("world row schema does not match MM-002")
        row = dict(raw)
        if type(row["fold"]) is not int or type(row["seed"]) is not int or not isinstance(row["video_id"], str):
            raise ValueError("world identity fields have invalid types")
        if not isinstance(row["variant_id"], str) or row["variant_id"] not in specs:
            raise ValueError("unknown world variant")
        spec = specs[row["variant_id"]]
        if (
            row["trajectory_id"] != spec.trajectory_id
            or type(row["horizon_seconds"]) is not float
            or row["horizon_seconds"] != spec.horizon_seconds
            or row["updates"] != spec.updates
            or row["matched_sources"] is not spec.matched_sources
        ):
            raise ValueError("world row conflicts with its frozen variant")
        for name in ("train_rows", "test_rows", "updates"):
            if type(row[name]) is not int or row[name] <= 0:
                raise ValueError(f"world {name} must be a positive integer")
        expected_train, expected_test = _expected_row_counts(
            row["fold"], row["video_id"], matched=spec.matched_sources
        )
        if (row["train_rows"], row["test_rows"]) != (expected_train, expected_test):
            raise ValueError("world train/test row counts conflict with the frozen panel")
        primary = _fingerprint(row["model_fingerprint"], "model_fingerprint")
        shuffled = _fingerprint(row["shuffle_model_fingerprint"], "shuffle_model_fingerprint")
        key = (row["fold"], row["seed"], spec.variant_id)
        if key in fingerprints and fingerprints[key] != (primary, shuffled):
            raise ValueError("world fingerprints differ within one fold/seed/variant")
        fingerprints[key] = (primary, shuffled)
        for name in WORLD_METRICS:
            _finite_number(
                row[name],
                name,
                positive=name.endswith("persistence_mse"),
                nonnegative=not name.endswith("persistence_mse"),
            )
        identities.append((row["fold"], row["seed"], spec.variant_id, row["video_id"]))
        rows.append(row)
    if identities != _expected_world_identities():
        raise ValueError("world rows are incomplete, duplicated, or out of frozen order")
    return rows


def validate_raw_probe_rows(value: object) -> list[dict[str, Any]]:
    """Validate exact raw ridge-probe evidence."""

    if not isinstance(value, list):
        raise ValueError("raw_probe_rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, str, str]] = []
    horizons = dict(RAW_PROBE_HORIZONS)
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != RAW_PROBE_ROW_KEYS:
            raise ValueError("raw probe row schema does not match MM-002")
        row = dict(raw)
        if type(row["fold"]) is not int or not isinstance(row["video_id"], str) or row.get("probe_id") not in horizons:
            raise ValueError("raw probe identity fields have invalid types")
        if type(row["horizon_seconds"]) is not float or row["horizon_seconds"] != horizons[cast(str, row["probe_id"])]:
            raise ValueError("raw probe horizon conflicts with its ID")
        for name in ("train_rows", "test_rows"):
            if type(row[name]) is not int or row[name] <= 0:
                raise ValueError(f"raw probe {name} must be a positive integer")
        expected_train, expected_test = _expected_row_counts(row["fold"], row["video_id"], matched=True)
        if (row["train_rows"], row["test_rows"]) != (expected_train, expected_test):
            raise ValueError("raw probe train/test row counts conflict with the frozen panel")
        for name in RAW_METRICS:
            _finite_number(row[name], name, nonnegative=True)
        identities.append((row["fold"], cast(str, row["probe_id"]), row["video_id"]))
        rows.append(row)
    expected = [
        (fold.index, probe_id, video_id)
        for fold in dataset.formal_folds()
        for probe_id, _ in RAW_PROBE_HORIZONS
        for video_id in fold.test_ids
    ]
    if identities != expected:
        raise ValueError("raw probe rows are incomplete, duplicated, or out of frozen order")
    return rows


def validate_integrity_rows(value: object) -> list[dict[str, Any]]:
    """Validate every world checkpoint representation-integrity probe."""

    if not isinstance(value, list):
        raise ValueError("integrity_rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, int, str, int, str, str]] = []
    horizons = {"full_1s": 1.0, "matched_0p5s": 0.5, "matched_1s": 1.0, "matched_2s": 2.0}
    fingerprints: dict[tuple[int, int, str, int, str], str] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != INTEGRITY_ROW_KEYS:
            raise ValueError("integrity row schema does not match MM-002")
        row = dict(raw)
        trajectory_id = row["trajectory_id"]
        if not isinstance(trajectory_id, str) or trajectory_id not in INTEGRITY_CHECKPOINTS:
            raise ValueError("unknown integrity trajectory")
        if (
            type(row["fold"]) is not int
            or type(row["seed"]) is not int
            or type(row["updates"]) is not int
            or row["updates"] not in INTEGRITY_CHECKPOINTS[trajectory_id]
            or type(row["horizon_seconds"]) is not float
            or row["horizon_seconds"] != horizons[trajectory_id]
            or row["model_role"] not in ("primary", "shuffle")
            or row["encoder_role"] not in ("online", "ema_target")
        ):
            raise ValueError("integrity identity fields conflict with the frozen design")
        if type(row["pooled_test_rows"]) is not int or row["pooled_test_rows"] <= 0:
            raise ValueError("pooled_test_rows must be a positive integer")
        fold = _fold(row["fold"])
        counts: Mapping[str, int] = (
            dataset.EXPECTED_WINDOW_COUNTS if trajectory_id == "full_1s" else MATCHED_COUNTS
        )
        expected_pooled = sum(counts[video_id] for video_id in fold.test_ids)
        if row["pooled_test_rows"] != expected_pooled:
            raise ValueError("pooled_test_rows conflicts with the frozen trajectory panel")
        fingerprint = _fingerprint(row["model_fingerprint"], "model_fingerprint")
        key = (row["fold"], row["seed"], trajectory_id, row["updates"], row["model_role"])
        if key in fingerprints and fingerprints[key] != fingerprint:
            raise ValueError("online and EMA integrity rows must share one model fingerprint")
        fingerprints[key] = fingerprint
        for name in ("latent_std_min", "latent_effective_rank", "prediction_min_variance"):
            _finite_number(row[name], name, nonnegative=name != "prediction_min_variance")
        for name in ("prediction_finite", "prediction_variance_positive"):
            if type(row[name]) is not bool:
                raise ValueError(f"integrity {name} must be boolean")
        if row["prediction_variance_positive"] is not (float(row["prediction_min_variance"]) > 0.0):
            raise ValueError("prediction_variance_positive conflicts with prediction_min_variance")
        identities.append(
            (row["fold"], row["seed"], trajectory_id, row["updates"], row["model_role"], row["encoder_role"])
        )
        rows.append(row)
    expected = [
        (fold.index, seed, trajectory_id, updates, model_role, encoder_role)
        for fold in dataset.formal_folds()
        for seed in core.SEEDS
        for trajectory_id, checkpoints in INTEGRITY_CHECKPOINTS.items()
        for updates in checkpoints
        for model_role in ("primary", "shuffle")
        for encoder_role in ("online", "ema_target")
    ]
    if identities != expected:
        raise ValueError("integrity rows are incomplete, duplicated, or out of frozen order")
    return rows


def validate_codec_rows(value: object) -> list[dict[str, Any]]:
    """Validate exact vision-codec ladder evidence."""

    if not isinstance(value, list):
        raise ValueError("codec_rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, int, str, str]] = []
    specs = {spec.variant_id: spec for spec in CODEC_VARIANTS}
    fingerprints: dict[tuple[int, int, str], tuple[str, str]] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != CODEC_ROW_KEYS:
            raise ValueError("codec row schema does not match MM-002")
        row = dict(raw)
        if type(row["fold"]) is not int or type(row["seed"]) is not int or not isinstance(row["video_id"], str):
            raise ValueError("codec identity fields have invalid types")
        if not isinstance(row["variant_id"], str) or row["variant_id"] not in specs:
            raise ValueError("unknown codec variant")
        spec = specs[row["variant_id"]]
        if (
            row["included_modalities"] != list(spec.included_modalities)
            or row["update_order"] != list(spec.update_order)
            or row["cycles"] != spec.cycles
            or row["snapshot"] != spec.snapshot
        ):
            raise ValueError("codec row conflicts with its frozen variant")
        for name in ("cycles", "train_rows", "test_rows"):
            if type(row[name]) is not int or row[name] <= 0:
                raise ValueError(f"codec {name} must be a positive integer")
        expected_train, expected_test = _expected_row_counts(row["fold"], row["video_id"], matched=False)
        if (row["train_rows"], row["test_rows"]) != (expected_train, expected_test):
            raise ValueError("codec train/test row counts conflict with the frozen parent panel")
        primary = _fingerprint(row["model_fingerprint"], "model_fingerprint")
        codec = _fingerprint(row["codec_fingerprint"], "codec_fingerprint")
        key = (row["fold"], row["seed"], spec.variant_id)
        if key in fingerprints and fingerprints[key] != (primary, codec):
            raise ValueError("codec fingerprints differ within one fold/seed/variant")
        fingerprints[key] = (primary, codec)
        for name in CODEC_METRICS:
            _finite_number(
                row[name],
                name,
                positive=name == "incumbent_world_mse",
                nonnegative=name != "incumbent_world_mse",
            )
        expected_ratio = float(row["vision_mse"]) / float(row["incumbent_world_mse"])
        if not math.isclose(
            float(row["vision_to_incumbent_ratio"]),
            expected_ratio,
            rel_tol=PARITY_RTOL,
            abs_tol=PARITY_ATOL,
        ):
            raise ValueError("vision_to_incumbent_ratio does not match the saved MSE values")
        identities.append((row["fold"], row["seed"], spec.variant_id, row["video_id"]))
        rows.append(row)
    expected = [
        (fold.index, seed, spec.variant_id, video_id)
        for fold in dataset.formal_folds()
        for seed in core.SEEDS
        for spec in CODEC_VARIANTS
        for video_id in fold.test_ids
    ]
    if identities != expected:
        raise ValueError("codec rows are incomplete, duplicated, or out of frozen order")
    return rows


def validate_evidence(value: object) -> dict[str, list[dict[str, Any]]]:
    """Strictly validate and normalize the complete four-array evidence object."""

    if not isinstance(value, Mapping) or set(value) != {
        "world_rows",
        "raw_probe_rows",
        "integrity_rows",
        "codec_rows",
    }:
        raise ValueError("evidence must contain exactly four frozen row arrays")
    world_rows = validate_world_rows(value["world_rows"])
    raw_probe_rows = validate_raw_probe_rows(value["raw_probe_rows"])
    integrity_rows = validate_integrity_rows(value["integrity_rows"])
    codec_rows = validate_codec_rows(value["codec_rows"])

    integrity_fingerprints = {
        (
            row["fold"],
            row["seed"],
            row["trajectory_id"],
            row["updates"],
            row["model_role"],
        ): row["model_fingerprint"]
        for row in integrity_rows
    }
    for row in world_rows:
        base = (row["fold"], row["seed"], row["trajectory_id"], row["updates"])
        if row["model_fingerprint"] != integrity_fingerprints[(*base, "primary")]:
            raise ValueError("world primary fingerprint does not match its integrity checkpoint")
        if row["shuffle_model_fingerprint"] != integrity_fingerprints[(*base, "shuffle")]:
            raise ValueError("world shuffle fingerprint does not match its integrity checkpoint")
    full_world_fingerprints = {
        (row["fold"], row["seed"]): row["model_fingerprint"]
        for row in world_rows
        if row["variant_id"] == "full_1s_1500"
    }
    for row in codec_rows:
        if row["model_fingerprint"] != full_world_fingerprints[(row["fold"], row["seed"])]:
            raise ValueError("codec incumbent fingerprint does not match the full world baseline")
    return {
        "world_rows": world_rows,
        "raw_probe_rows": raw_probe_rows,
        "integrity_rows": integrity_rows,
        "codec_rows": codec_rows,
    }


def _parent_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("parent integration rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, int, str]] = []
    required = {
        "fold",
        "seed",
        "video_id",
        "model_fingerprint",
        "incumbent_mse",
        "persistence_mse",
        "ridge_mse",
        "shuffle_model_mse",
        "shuffle_model_persistence_mse",
        "vision_mse",
        "vision_latent_mse",
    }
    for raw in value:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError("parent row lacks an MM-002 parity field")
        row = dict(raw)
        if type(row["fold"]) is not int or type(row["seed"]) is not int or not isinstance(row["video_id"], str):
            raise ValueError("parent row identity fields have invalid types")
        _fingerprint(row["model_fingerprint"], "parent model_fingerprint")
        for name in required - {"fold", "seed", "video_id", "model_fingerprint"}:
            _finite_number(row[name], f"parent {name}")
        identities.append((row["fold"], row["seed"], row["video_id"]))
        rows.append(row)
    expected = [
        (fold.index, seed, video_id)
        for fold in dataset.formal_folds()
        for seed in core.SEEDS
        for video_id in fold.test_ids
    ]
    if identities != expected:
        raise ValueError("parent rows are incomplete, duplicated, or out of frozen order")
    return rows


def assert_parent_parity(evidence: object, parent_rows: object) -> dict[str, Any]:
    """Fail unless both MM-002 baselines reproduce all relevant MM-001 rows."""

    normalized = validate_evidence(evidence)
    parents = _parent_rows(parent_rows)
    parent_by_id = {(row["fold"], row["seed"], row["video_id"]): row for row in parents}
    full_world = [row for row in normalized["world_rows"] if row["variant_id"] == "full_1s_1500"]
    baseline_codec = [row for row in normalized["codec_rows"] if row["variant_id"] == "shared_vat_600"]
    max_abs_error = 0.0

    def same(actual: object, expected: object, label: str) -> None:
        nonlocal max_abs_error
        actual_float = float(cast(int | float, actual))
        expected_float = float(cast(int | float, expected))
        max_abs_error = max(max_abs_error, abs(actual_float - expected_float))
        if not np.isclose(actual_float, expected_float, rtol=PARITY_RTOL, atol=PARITY_ATOL):
            raise ValueError(f"MM-001 parent parity failed for {label}: {actual_float} != {expected_float}")

    for row in full_world:
        identity = (row["fold"], row["seed"], row["video_id"])
        parent = parent_by_id[identity]
        if row["model_fingerprint"] != parent["model_fingerprint"]:
            raise ValueError(f"MM-001 world fingerprint parity failed for {identity}")
        for current, old in (
            ("world_mse", "incumbent_mse"),
            ("persistence_mse", "persistence_mse"),
            ("ridge_mse", "ridge_mse"),
            ("shuffle_model_mse", "shuffle_model_mse"),
            ("shuffle_model_persistence_mse", "shuffle_model_persistence_mse"),
        ):
            same(row[current], parent[old], f"world {identity} {current}")
    for row in baseline_codec:
        identity = (row["fold"], row["seed"], row["video_id"])
        parent = parent_by_id[identity]
        if row["model_fingerprint"] != parent["model_fingerprint"]:
            raise ValueError(f"MM-001 codec world fingerprint parity failed for {identity}")
        for current, old in (
            ("incumbent_world_mse", "incumbent_mse"),
            ("vision_mse", "vision_mse"),
            ("vision_latent_mse", "vision_latent_mse"),
        ):
            same(row[current], parent[old], f"codec {identity} {current}")
    return {
        "passed": True,
        "world_rows_compared": len(full_world),
        "codec_rows_compared": len(baseline_codec),
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "max_absolute_error": max_abs_error,
    }


def _video_medians(rows: list[dict[str, Any]], variant_key: str, metrics: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant_id in dict.fromkeys(str(row[variant_key]) for row in rows):
        variant_rows = [row for row in rows if row[variant_key] == variant_id]
        for video_id in dataset.SAMPLE_VIDEO_IDS:
            group = [row for row in variant_rows if row["video_id"] == video_id]
            if not group:
                raise ValueError(f"missing {variant_id} evidence for {video_id}")
            output.append(
                {
                    variant_key: variant_id,
                    "video_id": video_id,
                    **{name: float(np.median([float(row[name]) for row in group])) for name in metrics},
                }
            )
    return output


def _world_support(row: Mapping[str, Any]) -> bool:
    world = float(row["world_mse"])
    persistence = float(row["persistence_mse"])
    return bool(
        world * core.VISUAL_PERSISTENCE_FACTOR <= persistence
        and world < float(row["ridge_mse"])
        and (world / persistence) * core.VISUAL_SHUFFLE_MARGIN
        <= float(row["shuffle_model_mse"]) / float(row["shuffle_model_persistence_mse"])
    )


def _raw_support(row: Mapping[str, Any]) -> bool:
    ridge = float(row["raw_ridge_mse"])
    return bool(
        ridge * 1.2 <= float(row["raw_persistence_mse"])
        and ridge * 1.1 <= float(row["raw_shuffle_ridge_mse"])
    )


def _codec_support(row: Mapping[str, Any]) -> bool:
    return bool(float(row["vision_mse"]) <= core.VISION_CODEC_FACTOR * float(row["incumbent_world_mse"]))


def _integrity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in WORLD_VARIANTS:
        applicable = [
            row
            for row in rows
            if row["trajectory_id"] == spec.trajectory_id and int(row["updates"]) <= spec.updates
        ]
        failed = [
            row
            for row in applicable
            if float(row["latent_std_min"]) < INTEGRITY_MIN_STD
            or float(row["latent_effective_rank"]) < INTEGRITY_MIN_EFFECTIVE_RANK
            or not row["prediction_finite"]
            or not row["prediction_variance_positive"]
        ]
        output.append(
            {
                "variant_id": spec.variant_id,
                "healthy": not failed,
                "probe_rows": len(applicable),
                "failed_probe_rows": len(failed),
                "minimum_latent_std": min(float(row["latent_std_min"]) for row in applicable),
                "minimum_effective_rank": min(float(row["latent_effective_rank"]) for row in applicable),
                "minimum_prediction_variance": min(float(row["prediction_min_variance"]) for row in applicable),
            }
        )
    return output


def summarize(evidence: object, parent_rows: object) -> dict[str, Any]:
    """Derive all frozen support counts and diagnostic labels from saved rows."""

    normalized = validate_evidence(evidence)
    parity = assert_parent_parity(normalized, parent_rows)
    world_medians = _video_medians(normalized["world_rows"], "variant_id", WORLD_METRICS)
    world_variants: list[dict[str, Any]] = []
    for world_spec in WORLD_VARIANTS:
        videos = [row for row in world_medians if row["variant_id"] == world_spec.variant_id]
        supports = sum(_world_support(row) for row in videos)
        world_variants.append(
            {
                "variant_id": world_spec.variant_id,
                "supporting_videos": supports,
                "passes": supports >= core.REQUIRED_VIDEO_SUPPORT,
                "video_medians": videos,
            }
        )
    world_by_id = {row["variant_id"]: row for row in world_variants}
    integrity = _integrity_summary(normalized["integrity_rows"])
    integrity_by_id = {row["variant_id"]: row for row in integrity}

    world_labels: list[str] = []
    crossing_unhealthy = any(
        row["passes"] and not integrity_by_id[row["variant_id"]]["healthy"] for row in world_variants
    )
    if crossing_unhealthy:
        world_labels.append("apparent_rescue_via_representation_collapse")
    if (
        integrity_by_id["matched_1s_1500"]["healthy"]
        and (
            not integrity_by_id["matched_1s_3000"]["healthy"]
            or not integrity_by_id["matched_1s_6000"]["healthy"]
        )
    ):
        world_labels.append("extended_training_representation_instability")

    attribution_allowed = True
    if not integrity_by_id["full_1s_1500"]["healthy"]:
        world_labels.append("baseline_representation_integrity_failure")
        attribution_allowed = False
    elif (
        world_by_id["matched_1s_1500"]["passes"]
        and integrity_by_id["matched_1s_1500"]["healthy"]
        and not world_by_id["full_1s_1500"]["passes"]
    ):
        world_labels.append("endpoint_truncation_sensitive")
        attribution_allowed = False
    else:
        short = bool(world_by_id["matched_0p5s_1500"]["passes"] and integrity_by_id["matched_0p5s_1500"]["healthy"])
        long = bool(world_by_id["matched_2s_1500"]["passes"] and integrity_by_id["matched_2s_1500"]["healthy"])
        if short:
            world_labels.append("short_horizon_rescue")
        if long:
            world_labels.append("long_horizon_rescue")
        if short and long:
            world_labels.append("broad_horizon_rescue")
        pass_3000 = bool(world_by_id["matched_1s_3000"]["passes"] and integrity_by_id["matched_1s_3000"]["healthy"])
        pass_6000 = bool(world_by_id["matched_1s_6000"]["passes"] and integrity_by_id["matched_1s_6000"]["healthy"])
        if pass_3000 and pass_6000:
            world_labels.append("stable_world_budget_rescue")
        elif not pass_3000 and pass_6000:
            world_labels.append("late_four_x_world_budget_rescue")
        elif pass_3000 and not world_by_id["matched_1s_6000"]["passes"]:
            world_labels.append("world_overtraining_or_nonmonotonicity")
        elif (
            not world_by_id["matched_1s_3000"]["passes"]
            and not world_by_id["matched_1s_6000"]["passes"]
        ):
            world_labels.append("not_rescued_through_four_x_budget")

    raw_medians = _video_medians(normalized["raw_probe_rows"], "probe_id", RAW_METRICS)
    raw_probes: list[dict[str, Any]] = []
    raw_labels: list[str] = []
    for probe_id, _ in RAW_PROBE_HORIZONS:
        videos = [row for row in raw_medians if row["probe_id"] == probe_id]
        support = sum(_raw_support(row) for row in videos)
        passed = support >= core.REQUIRED_VIDEO_SUPPORT
        raw_probes.append(
            {"probe_id": probe_id, "supporting_videos": support, "passes": passed, "video_medians": videos}
        )
        raw_labels.append(f"raw_linear_predictability_{probe_id}_{'supported' if passed else 'not_supported'}")

    codec_medians = _video_medians(normalized["codec_rows"], "variant_id", CODEC_METRICS)
    codec_variants: list[dict[str, Any]] = []
    for codec_spec in CODEC_VARIANTS:
        videos = [row for row in codec_medians if row["variant_id"] == codec_spec.variant_id]
        support = sum(_codec_support(row) for row in videos)
        codec_variants.append(
            {
                "variant_id": codec_spec.variant_id,
                "supporting_videos": support,
                "passes": support >= core.REQUIRED_VIDEO_SUPPORT,
                "video_medians": videos,
            }
        )
    codec_by_id = {row["variant_id"]: row for row in codec_variants}
    codec_labels: list[str] = []
    baseline = bool(codec_by_id["shared_vat_600"]["passes"])
    atv = bool(codec_by_id["shared_atv_600"]["passes"])
    vision_600 = bool(codec_by_id["vision_only_600"]["passes"])
    after_v = bool(codec_by_id["shared_vat_2400_after_v"]["passes"])
    shared_2400 = bool(codec_by_id["shared_vat_2400"]["passes"])
    vision_2400 = bool(codec_by_id["vision_only_2400"]["passes"])
    if baseline:
        codec_labels.append("baseline_codec_migration_supported")
    else:
        if atv:
            codec_labels.append("nominal_codec_update_order_rescue")
        if vision_600 and not atv:
            codec_labels.append("nominal_cross_modal_shared_training_penalty")
        if shared_2400 and vision_2400 and not atv and not vision_600:
            codec_labels.append("codec_update_budget_rescue")
        candidate_passes = {
            "shared_atv_600": atv,
            "vision_only_600": vision_600,
            "shared_vat_2400_after_v": after_v,
            "shared_vat_2400": shared_2400,
            "vision_only_2400": vision_2400,
        }
        passing_candidates = {name for name, passed in candidate_passes.items() if passed}
        if passing_candidates == {"vision_only_2400"}:
            codec_labels.append("codec_sharing_by_budget_interaction")
        if shared_2400 and not atv and not vision_600 and not vision_2400:
            codec_labels.append("shared_positive_transfer_or_isolated_instability")
        if vision_600 and not vision_2400:
            codec_labels.append("isolated_codec_overtraining_or_instability")
        if after_v and not shared_2400:
            codec_labels.append("terminal_audio_text_update_recency_sensitivity")
        if not passing_candidates:
            codec_labels.append("codec_not_rescued_by_order_isolation_or_four_x_cycles")

    world_sensitivity_labels = {
        "endpoint_truncation_sensitive",
        "short_horizon_rescue",
        "long_horizon_rescue",
        "broad_horizon_rescue",
        "stable_world_budget_rescue",
        "late_four_x_world_budget_rescue",
        "world_overtraining_or_nonmonotonicity",
    }
    codec_sensitivity_labels = {
        "nominal_codec_update_order_rescue",
        "nominal_cross_modal_shared_training_penalty",
        "codec_update_budget_rescue",
        "codec_sharing_by_budget_interaction",
        "shared_positive_transfer_or_isolated_instability",
        "isolated_codec_overtraining_or_instability",
        "terminal_audio_text_update_recency_sensitivity",
    }
    world_sensitivity = bool(world_sensitivity_labels.intersection(world_labels))
    codec_sensitivity = bool(codec_sensitivity_labels.intersection(codec_labels))
    if "baseline_representation_integrity_failure" in world_labels:
        classification = "world_diagnostic_inconclusive_representation_integrity"
    elif world_sensitivity and codec_sensitivity:
        classification = "world_and_codec_factor_sensitivity_detected"
    elif world_sensitivity:
        classification = "world_factor_sensitivity_detected"
    elif codec_sensitivity:
        classification = "codec_factor_sensitivity_detected"
    elif "extended_training_representation_instability" in world_labels:
        classification = "extended_training_representation_instability_detected"
    else:
        classification = "tested_factors_not_supported"

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "MM-002",
        "scope": "outcome_informed_non_gated_diagnostic",
        "parent_parity": parity,
        "required_video_support": core.REQUIRED_VIDEO_SUPPORT,
        "total_videos": core.FORMAL_VIDEO_COUNT,
        "world": {
            "variants": world_variants,
            "integrity": integrity,
            "diagnosis_labels": world_labels,
            "horizon_budget_attribution_allowed": attribution_allowed,
            "raw_probes": raw_probes,
            "raw_diagnosis_labels": raw_labels,
        },
        "codec": {"variants": codec_variants, "diagnosis_labels": codec_labels},
        "decision": {
            "classification": classification,
            "world_factor_sensitivity": world_sensitivity,
            "codec_factor_sensitivity": codec_sensitivity,
        },
        "claim_boundary": (
            "Gate crossings identify sensitivity on the frozen eight-video sample only; "
            "MM-001 is not reclassified and no population-level causal claim is made."
        ),
    }


def report_text(results: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable report from :func:`summarize`."""

    if "summary" in results:
        nested = results.get("summary")
        if not isinstance(nested, Mapping):
            raise ValueError("formal MM-002 results contain an invalid summary")
        results = cast(Mapping[str, Any], nested)
    if results.get("schema_version") != SCHEMA_VERSION or results.get("experiment_id") != "MM-002":
        raise ValueError("results are not an MM-002 method summary")
    world = results.get("world")
    codec = results.get("codec")
    parity = results.get("parent_parity")
    decision = results.get("decision")
    if (
        not isinstance(world, Mapping)
        or not isinstance(codec, Mapping)
        or not isinstance(parity, Mapping)
        or not isinstance(decision, Mapping)
        or not isinstance(decision.get("classification"), str)
    ):
        raise ValueError("MM-002 results are missing report sections")

    lines = [
        "# MM-002 feature-only failure-isolation report",
        "",
        "MM-002 is an outcome-informed, non-gated diagnostic of MM-001. It does not reclassify MM-001.",
        "",
        f"Decision classification: `{decision['classification']}`.",
        "",
        "## Parent parity",
        "",
        f"Parent parity: **{'PASS' if parity.get('passed') is True else 'FAIL'}** "
        f"({parity.get('world_rows_compared')} world rows and {parity.get('codec_rows_compared')} codec rows).",
        "",
        "## World-model ladder",
        "",
        "| Variant | Supporting videos | Integrity |",
        "|---|---:|---|",
    ]
    integrity_by_id = {
        row["variant_id"]: row
        for row in cast(Sequence[Mapping[str, Any]], world.get("integrity", []))
    }
    for row in cast(Sequence[Mapping[str, Any]], world.get("variants", [])):
        health = integrity_by_id.get(row["variant_id"], {}).get("healthy")
        lines.append(
            f"| `{row['variant_id']}` | {row['supporting_videos']}/8 | {'healthy' if health is True else 'unhealthy'} |"
        )
    labels = cast(Sequence[str], world.get("diagnosis_labels", []))
    lines.extend(["", "World diagnosis: " + (", ".join(f"`{label}`" for label in labels) or "none"), ""])
    lines.extend(["### Raw-feature temporal probes", "", "| Probe | Supporting videos |", "|---|---:|"])
    for row in cast(Sequence[Mapping[str, Any]], world.get("raw_probes", [])):
        lines.append(f"| `{row['probe_id']}` | {row['supporting_videos']}/8 |")
    raw_labels = cast(Sequence[str], world.get("raw_diagnosis_labels", []))
    lines.extend(["", "Raw-probe diagnosis: " + ", ".join(f"`{label}`" for label in raw_labels), ""])
    lines.extend(["## Vision-codec ladder", "", "| Variant | Supporting videos |", "|---|---:|"])
    for row in cast(Sequence[Mapping[str, Any]], codec.get("variants", [])):
        lines.append(f"| `{row['variant_id']}` | {row['supporting_videos']}/8 |")
    codec_labels = cast(Sequence[str], codec.get("diagnosis_labels", []))
    lines.extend(
        [
            "",
            "Codec diagnosis: " + (", ".join(f"`{label}`" for label in codec_labels) or "none"),
            "",
            "## Claim boundary",
            "",
            str(results.get("claim_boundary", "")),
            "",
            "The tests do not address T5, SNAC, multimodal fusion, raw-media generation, "
            "action recovery, planning, control, or the full Perception Test distribution.",
            "",
        ]
    )
    return "\n".join(lines)
