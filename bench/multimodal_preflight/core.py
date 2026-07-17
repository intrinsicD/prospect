"""Small real-feature integration probe for MM-001.

This module deliberately contains no torch, codec-model, media, or network imports.
Heavyweight frozen encoders live in :mod:`bench.multimodal_preflight.backends`; the
arrays produced there enter the existing numpy Prospect seams here.  Keeping this
layer pure makes the scientific controls unit-testable without downloading weights.

MM-001 is a passive-observation preflight.  It validates modality-by-modality
substitution into a vision-anchored latent; it does not implement multimodal fusion
or claim action-conditioned control on Perception Test.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from prospect.agent import Agent
from prospect.codec import UniversalCodec
from prospect.types import Action, LatentState, Modality, Observation, Transition
from prospect.voe import SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel

FEATURE_DIM = 32
LATENT_DIM = 8
WORLD_STEPS = 1_500
WORLD_BATCH = 64
WORLD_HIDDEN = 64
WORLD_ENSEMBLE = 5
WORLD_LR = 3e-3
WORLD_EMA_TAU = 0.995
WORLD_W_REWARD = 0.0
WORLD_W_INVERSE = 0.0
WORLD_W_VAR = 25.0
WORLD_W_COV = 1.0
WORLD_SAMPLE_SEED_OFFSET = 1
CODEC_STEPS = 600
CODEC_BATCH = 128
CODEC_TOKEN_DIM = 32
CODEC_HIDDEN = 64
CODEC_LR = 3e-3
CODEC_INIT_SEED_OFFSET = 1
CODEC_SAMPLE_SEED_OFFSET = 303
RIDGE_PENALTY = 1e-3
VISUAL_PERSISTENCE_FACTOR = 1.2
VISUAL_SHUFFLE_MARGIN = 1.1
VISION_CODEC_FACTOR = 1.5
SUBSTITUTION_MARGIN = 1.1
REQUIRED_VIDEO_SUPPORT = 6
FORMAL_VIDEO_COUNT = 8
SEEDS = (0, 1, 2)
NULL_ACTION = Action(data=np.zeros(1, dtype=float))
MODALITIES = (Modality.VISION, Modality.AUDIO, Modality.TEXT)


@dataclass(frozen=True)
class FeatureTable:
    """One row per causal Perception Test window.

    ``vision``, ``audio`` and ``text`` are fixed 32-D frozen-model features at
    timestamp ``t``. ``target_vision`` is the visual feature at ``t + 1 s``.
    Rows retain their video identity so every split/control is video-grouped.
    """

    video_ids: np.ndarray
    timestamps: np.ndarray
    vision: np.ndarray
    audio: np.ndarray
    text: np.ndarray
    target_vision: np.ndarray
    annotation_present: np.ndarray

    def validate(self) -> None:
        n = len(self.video_ids)
        one_dimensional = {
            "video_ids": self.video_ids,
            "timestamps": self.timestamps,
            "annotation_present": self.annotation_present,
        }
        for name, values in one_dimensional.items():
            if np.asarray(values).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        for name, values in self.features().items():
            if np.asarray(values).shape != (n, FEATURE_DIM):
                raise ValueError(f"{name} must have shape ({n}, {FEATURE_DIM})")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
        if not np.all(np.isfinite(self.timestamps)):
            raise ValueError("timestamps contains non-finite values")
        if len(set(np.asarray(self.video_ids, dtype=str))) < 2:
            raise ValueError("feature table must contain at least two videos")

    def features(self) -> dict[Modality, np.ndarray]:
        return {
            Modality.VISION: np.asarray(self.vision, dtype=float),
            Modality.AUDIO: np.asarray(self.audio, dtype=float),
            Modality.TEXT: np.asarray(self.text, dtype=float),
        }

    def subset(self, video_ids: tuple[str, ...] | list[str]) -> FeatureTable:
        wanted = set(video_ids)
        mask = np.array([str(value) in wanted for value in self.video_ids], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"no feature rows for videos {sorted(wanted)}")
        return FeatureTable(
            video_ids=np.asarray(self.video_ids)[mask].copy(),
            timestamps=np.asarray(self.timestamps, dtype=float)[mask].copy(),
            vision=np.asarray(self.vision, dtype=float)[mask].copy(),
            audio=np.asarray(self.audio, dtype=float)[mask].copy(),
            text=np.asarray(self.text, dtype=float)[mask].copy(),
            target_vision=np.asarray(self.target_vision, dtype=float)[mask].copy(),
            annotation_present=np.asarray(self.annotation_present, dtype=bool)[mask].copy(),
        )


def fixed_projection(values: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    """Project frozen features without fitting on any video.

    A seeded Rademacher matrix is a measurement fixture, not a learned projection;
    held-out videos therefore cannot leak into it.  Modality wrappers own any
    contract-specific scaling (for example, normalizing SNAC token IDs).
    """

    rows = np.atleast_2d(np.asarray(values, dtype=float))
    if rows.shape[1] == 0 or output_dim < 1:
        raise ValueError("projection dimensions must be positive")
    signs = np.random.default_rng(seed).integers(0, 2, size=(rows.shape[1], output_dim), dtype=np.int8)
    matrix = (2.0 * signs.astype(float) - 1.0) / np.sqrt(rows.shape[1])
    return np.asarray(rows @ matrix, dtype=float)


def _indices_for_videos(table: FeatureTable, video_ids: tuple[str, ...]) -> np.ndarray:
    wanted = set(video_ids)
    return np.flatnonzero([str(video_id) in wanted for video_id in table.video_ids])


def temporal_derangement(table: FeatureTable) -> np.ndarray:
    """Return target indices shifted within each video, with no fixed points."""

    out = np.empty(len(table.video_ids), dtype=int)
    for video_id in sorted(set(np.asarray(table.video_ids, dtype=str))):
        indices = np.flatnonzero(np.asarray(table.video_ids, dtype=str) == video_id)
        order = indices[np.argsort(np.asarray(table.timestamps)[indices])]
        if len(order) < 2:
            raise ValueError(f"video {video_id} needs at least two windows")
        shift = max(1, len(order) // 2)
        if shift == len(order):
            shift = 1
        out[order] = np.roll(order, shift)
    if np.any(out == np.arange(len(out))):
        raise AssertionError("temporal derangement contains a fixed point")
    return out


def cross_video_derangement(table: FeatureTable) -> np.ndarray:
    """Map every row to a similar fractional position in the next video.

    This preserves each modality's marginal and coarse time distribution while
    destroying video pairing.  It is deterministic and never maps within-video.
    """

    video_ids = sorted(set(np.asarray(table.video_ids, dtype=str)))
    if len(video_ids) < 2:
        raise ValueError("cross-video derangement needs at least two videos")
    groups: dict[str, np.ndarray] = {}
    for video_id in video_ids:
        indices = np.flatnonzero(np.asarray(table.video_ids, dtype=str) == video_id)
        groups[video_id] = indices[np.argsort(np.asarray(table.timestamps)[indices])]
    out = np.empty(len(table.video_ids), dtype=int)
    for position, video_id in enumerate(video_ids):
        source = groups[video_id]
        target = groups[video_ids[(position + 1) % len(video_ids)]]
        for rank, row in enumerate(source):
            fraction = rank / max(len(source) - 1, 1)
            target_rank = int(round(fraction * (len(target) - 1)))
            out[row] = target[target_rank]
    source_ids = np.asarray(table.video_ids, dtype=str)
    if np.any(source_ids == source_ids[out]):
        raise AssertionError("cross-video derangement mapped within a video")
    return out


def _transitions(table: FeatureTable, shuffled: bool = False) -> list[Transition]:
    targets = table.target_vision
    if shuffled:
        targets = targets[temporal_derangement(table)]
    return [
        Transition(
            state=LatentState(z=np.asarray(source, dtype=float)),
            action=NULL_ACTION,
            next_state=LatentState(z=np.asarray(target, dtype=float)),
            reward=0.0,
        )
        for source, target in zip(table.vision, targets, strict=True)
    ]


def fit_world_model(
    table: FeatureTable,
    seed: int,
    *,
    shuffled: bool = False,
    steps: int = WORLD_STEPS,
) -> FlatWorldModel:
    """Fit the unchanged Prospect model on real frozen visual features."""

    table.validate()
    transitions = _transitions(table, shuffled=shuffled)
    model = FlatWorldModel(
        obs_dim=FEATURE_DIM,
        action_dim=1,
        latent_dim=LATENT_DIM,
        hidden=WORLD_HIDDEN,
        ensemble=WORLD_ENSEMBLE,
        lr=WORLD_LR,
        ema_tau=WORLD_EMA_TAU,
        w_reward=WORLD_W_REWARD,
        w_inverse=WORLD_W_INVERSE,
        w_var=WORLD_W_VAR,
        w_cov=WORLD_W_COV,
        seed=seed,
    )
    rng = np.random.default_rng(seed + WORLD_SAMPLE_SEED_OFFSET)
    batch_size = min(WORLD_BATCH, len(transitions))
    for _ in range(steps):
        indices = rng.integers(0, len(transitions), size=batch_size)
        model.update([transitions[index] for index in indices])
    return model


def _incumbent_latents(model: FlatWorldModel, vision: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(model.encode(row).z, dtype=float) for row in vision])


def fit_codec(
    model: FlatWorldModel,
    table: FeatureTable,
    seed: int,
    *,
    deranged_audio_text: bool = False,
    steps: int = CODEC_STEPS,
) -> UniversalCodec:
    """Distil modality adapters into the frozen incumbent visual latent."""

    table.validate()
    codec = UniversalCodec(
        {modality: FEATURE_DIM for modality in MODALITIES},
        latent_dim=model.latent_dim,
        token_dim=CODEC_TOKEN_DIM,
        hidden=CODEC_HIDDEN,
        lr=CODEC_LR,
        seed=seed + CODEC_INIT_SEED_OFFSET,
    )
    targets = _incumbent_latents(model, table.vision)
    features = table.features()
    modality_targets = {modality: targets for modality in MODALITIES}
    if deranged_audio_text:
        permutation = cross_video_derangement(table)
        modality_targets = {
            Modality.VISION: targets,
            Modality.AUDIO: targets[permutation],
            Modality.TEXT: targets[permutation],
        }

    # The codec freezes normalization on the first batch per modality.  Use the
    # complete training fold so batch order cannot leak into those statistics.
    for modality in MODALITIES:
        codec.distill_encode(features[modality], modality, modality_targets[modality])
        if not deranged_audio_text:
            codec.fit_decode(modality_targets[modality], features[modality], modality)

    rng = np.random.default_rng(seed + CODEC_SAMPLE_SEED_OFFSET)
    batch_size = min(CODEC_BATCH, len(table.video_ids))
    for _ in range(steps):
        indices = rng.integers(0, len(table.video_ids), size=batch_size)
        for modality in MODALITIES:
            codec.distill_encode(features[modality][indices], modality, modality_targets[modality][indices])
            if not deranged_audio_text:
                codec.fit_decode(modality_targets[modality][indices], features[modality][indices], modality)
    return codec


def _target_latents(model: FlatWorldModel, values: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(model.encode_target(row).z, dtype=float) for row in values])


def _predictions_from_latents(model: FlatWorldModel, latents: list[LatentState]) -> np.ndarray:
    return np.stack([np.asarray(model.predict(latent, NULL_ACTION).mean, dtype=float) for latent in latents])


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))


def _ridge(model: FlatWorldModel, train: FeatureTable, test: FeatureTable) -> tuple[np.ndarray, np.ndarray]:
    target_train = _target_latents(model, train.target_vision)
    x_train = np.c_[train.vision, np.ones(len(train.video_ids))]
    x_test = np.c_[test.vision, np.ones(len(test.video_ids))]
    penalty = RIDGE_PENALTY * np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ target_train)
    return x_test @ weights, weights


def _codec_latents(codec: UniversalCodec, modality: Modality, values: np.ndarray) -> list[LatentState]:
    return [codec.encode(Observation(modality, row)) for row in values]


def _unstandardized_decode(codec: UniversalCodec, latents: list[LatentState], modality: Modality) -> np.ndarray:
    decoded = np.stack([np.asarray(codec.decode(latent, modality).data, dtype=float) for latent in latents])
    # UniversalCodec documents standardized readout targets but has not yet
    # exposed a public raw-unit decode helper.  MM-001 records this harness-side
    # conversion explicitly instead of pretending the current public output is raw.
    mean, std = codec._stats[modality]
    return np.asarray(decoded * std + mean, dtype=float)


class _NullPlanner:
    """Passive-observation planner seam used only to exercise ``Agent`` wiring."""

    def plan(self, state: LatentState, goal: object | None = None) -> Action:
        del state, goal
        return NULL_ACTION

    def reset(self) -> None:
        return None


def passive_agent_wiring(
    model: FlatWorldModel,
    codec: UniversalCodec,
    table: FeatureTable,
    modality: Modality,
) -> bool:
    """Exercise encode -> predict -> observe -> VoE through the composition root."""

    rows = np.argsort(np.asarray(table.timestamps, dtype=float))
    if len(rows) < 3:
        return False
    monitor = SurpriseCompetenceMonitor(min_updates=2)
    agent = Agent(
        encode=codec.encode,
        planner=_NullPlanner(),
        world_model=model,
        monitor=monitor,
    )
    features = table.features()[modality]
    # Windows are spaced by 0.5 s; stride two matches the frozen 1.0 s horizon.
    updates = 0
    for start in range(0, len(rows) - 2, 2):
        current = int(rows[start])
        following = int(rows[start + 2])
        if str(table.video_ids[current]) != str(table.video_ids[following]):
            continue
        obs = Observation(modality, features[current])
        next_obs = Observation(modality, features[following])
        action = agent.act(obs)
        agent.observe(obs, action, next_obs, 0.0)
        updates += 1
        if updates >= 3:
            break
    competence = monitor.competence(SurpriseCompetenceMonitor.DEFAULT_SKILL)
    return updates >= 2 and bool(np.isfinite(competence.epistemic))


def evaluate_video(
    model: FlatWorldModel,
    shuffled_model: FlatWorldModel,
    codec: UniversalCodec,
    deranged_codec: UniversalCodec,
    train: FeatureTable,
    test_video: FeatureTable,
) -> dict[str, float]:
    """Measure all frozen MM-001 endpoints for one held-out video."""

    targets = _target_latents(model, test_video.target_vision)
    current_targets = _target_latents(model, test_video.vision)
    incumbent_latents = [model.encode(row) for row in test_video.vision]
    incumbent_predictions = _predictions_from_latents(model, incumbent_latents)
    shuffled_predictions = _predictions_from_latents(
        shuffled_model, [shuffled_model.encode(row) for row in test_video.vision]
    )
    # A shuffled model has its own target encoder.  Compare its predictions to its
    # own correctly timed target space; otherwise encoder drift would confound the control.
    shuffled_targets = _target_latents(shuffled_model, test_video.target_vision)
    shuffled_current = _target_latents(shuffled_model, test_video.vision)
    ridge_predictions, _ = _ridge(model, train, test_video)

    metrics: dict[str, float] = {
        "incumbent_mse": _mse(incumbent_predictions, targets),
        "persistence_mse": _mse(current_targets, targets),
        "ridge_mse": _mse(ridge_predictions, targets),
        "shuffle_model_mse": _mse(shuffled_predictions, shuffled_targets),
        "shuffle_model_persistence_mse": _mse(shuffled_current, shuffled_targets),
        "annotation_coverage": float(np.mean(test_video.annotation_present)),
    }

    feature_map = test_video.features()
    incumbent_array = np.stack([np.asarray(latent.z, dtype=float) for latent in incumbent_latents])
    mean_incumbent = LatentState(z=np.mean(_incumbent_latents(model, train.vision), axis=0))
    mean_predictions = _predictions_from_latents(model, [mean_incumbent] * len(test_video.video_ids))
    for modality in MODALITIES:
        name = modality.value
        aligned_latents = _codec_latents(codec, modality, feature_map[modality])
        aligned_array = np.stack([np.asarray(latent.z, dtype=float) for latent in aligned_latents])
        aligned_predictions = _predictions_from_latents(model, aligned_latents)
        metrics[f"{name}_mse"] = _mse(aligned_predictions, targets)
        metrics[f"{name}_latent_mse"] = _mse(aligned_array, incumbent_array)
        decoded = _unstandardized_decode(codec, aligned_latents, modality)
        metrics[f"{name}_feature_decode_mse"] = _mse(decoded, feature_map[modality])
        metrics[f"{name}_agent_wiring"] = float(passive_agent_wiring(model, codec, test_video, modality))
        if modality in (Modality.AUDIO, Modality.TEXT):
            control_latents = _codec_latents(deranged_codec, modality, feature_map[modality])
            control_array = np.stack([np.asarray(latent.z, dtype=float) for latent in control_latents])
            metrics[f"{name}_deranged_mse"] = _mse(_predictions_from_latents(model, control_latents), targets)
            metrics[f"{name}_deranged_latent_mse"] = _mse(control_array, incumbent_array)
            metrics[f"{name}_constant_mse"] = _mse(mean_predictions, targets)

    predictions = [model.predict(latent, NULL_ACTION) for latent in incumbent_latents]
    actual_nll = np.array(
        [-prediction.log_prob(target) for prediction, target in zip(predictions, targets, strict=True)]
    )
    wrong_indices = temporal_derangement(test_video)
    wrong_nll = np.array(
        [-prediction.log_prob(targets[index]) for prediction, index in zip(predictions, wrong_indices, strict=True)]
    )
    metrics["actual_nll"] = float(np.mean(actual_nll))
    metrics["temporal_deranged_nll"] = float(np.mean(wrong_nll))
    metrics["prediction_finite"] = float(
        all(
            np.all(np.isfinite(prediction.mean))
            and np.all(np.isfinite(prediction.var))
            and np.all(np.asarray(prediction.var) > 0.0)
            and np.isfinite(prediction.epistemic)
            and np.isfinite(prediction.aleatoric)
            for prediction in predictions
        )
    )
    return metrics


def model_fingerprint(model: FlatWorldModel) -> str:
    """Stable digest of all learned numpy parameters used in MM-001."""

    digest = sha256()
    networks = (model.encoder, *model.members, model.reward_head, model.inverse_head)
    for network in networks:
        for array in (*network.weights, *network.biases):
            value = np.asarray(array, dtype="<f8", order="C")
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    for array in (*model._target_w, *model._target_b, model._obs_mean, model._obs_var):
        value = np.asarray(array, dtype="<f8", order="C")
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _video_medians(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for video_id in sorted({str(row["video_id"]) for row in rows}):
        group = [row for row in rows if str(row["video_id"]) == video_id]
        metric_names = sorted(set(group[0]) - {"video_id", "fold", "seed", "model_fingerprint"})
        summary: dict[str, float | str] = {"video_id": video_id}
        for name in metric_names:
            summary[name] = float(np.median([float(row[name]) for row in group]))
        output.append(summary)
    return output


def integration_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen 6-of-8 video-level robustness rules."""

    videos = _video_medians(rows)
    if len(videos) != FORMAL_VIDEO_COUNT:
        raise ValueError(f"formal decision requires exactly {FORMAL_VIDEO_COUNT} videos, got {len(videos)}")

    def count(predicate: Any) -> int:
        return sum(bool(predicate(video)) for video in videos)

    visual = count(
        lambda row: (
            float(row["incumbent_mse"]) * VISUAL_PERSISTENCE_FACTOR <= float(row["persistence_mse"])
            and float(row["incumbent_mse"]) < float(row["ridge_mse"])
            and (float(row["incumbent_mse"]) / max(float(row["persistence_mse"]), 1e-12)) * VISUAL_SHUFFLE_MARGIN
            <= (float(row["shuffle_model_mse"]) / max(float(row["shuffle_model_persistence_mse"]), 1e-12))
        )
    )
    vision_swap = count(lambda row: float(row["vision_mse"]) <= VISION_CODEC_FACTOR * float(row["incumbent_mse"]))
    audio = count(
        lambda row: (
            float(row["audio_latent_mse"]) * SUBSTITUTION_MARGIN <= float(row["audio_deranged_latent_mse"])
            and float(row["audio_mse"]) * SUBSTITUTION_MARGIN <= float(row["audio_deranged_mse"])
            and float(row["audio_mse"]) * SUBSTITUTION_MARGIN <= float(row["audio_constant_mse"])
        )
    )
    text = count(
        lambda row: (
            float(row["text_latent_mse"]) * SUBSTITUTION_MARGIN <= float(row["text_deranged_latent_mse"])
            and float(row["text_mse"]) * SUBSTITUTION_MARGIN <= float(row["text_deranged_mse"])
            and float(row["text_mse"]) * SUBSTITUTION_MARGIN <= float(row["text_constant_mse"])
        )
    )
    surprise = count(lambda row: float(row["actual_nll"]) < float(row["temporal_deranged_nll"]))
    wiring = count(
        lambda row: (
            all(float(row[f"{modality.value}_agent_wiring"]) == 1.0 for modality in MODALITIES)
            and float(row["prediction_finite"]) == 1.0
        )
    )
    counts = {
        "real_visual_dynamics": visual,
        "vision_codec_migration": vision_swap,
        "audio_substitution": audio,
        "text_substitution": text,
        "temporal_surprise": surprise,
        "passive_agent_wiring": wiring,
    }
    passes = {name: value >= REQUIRED_VIDEO_SUPPORT for name, value in counts.items()}
    protocol_endpoints = (
        "real_visual_dynamics",
        "vision_codec_migration",
        "audio_substitution",
        "text_substitution",
    )
    return {
        "required_videos": REQUIRED_VIDEO_SUPPORT,
        "total_videos": FORMAL_VIDEO_COUNT,
        "supporting_videos": counts,
        "passes": passes,
        "all_integration_endpoints_pass": all(passes[name] for name in protocol_endpoints),
        "all_diagnostics_pass": all(passes.values()),
        "video_medians": videos,
    }
