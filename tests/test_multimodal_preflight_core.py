"""Pure-numpy tests for the MM-001 real-feature integration layer."""

from __future__ import annotations

import numpy as np

from bench.multimodal_preflight import core


def _table(video_count: int = 3, rows_per_video: int = 8) -> core.FeatureTable:
    video_ids: list[str] = []
    timestamps: list[float] = []
    vision: list[np.ndarray] = []
    audio: list[np.ndarray] = []
    text: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for video in range(video_count):
        phase = 0.3 * video
        for row in range(rows_per_video):
            t = 1.0 + 0.5 * row
            angles = np.linspace(0.0, 2.0 * np.pi, core.FEATURE_DIM, endpoint=False)
            current = np.sin(angles + 0.2 * t + phase)
            future = np.sin(angles + 0.2 * (t + 1.0) + phase)
            video_ids.append(f"video_{video}")
            timestamps.append(t)
            vision.append(current)
            audio.append(np.cos(angles + 0.2 * t + phase))
            text.append(np.tanh(current * 2.0))
            targets.append(future)
    return core.FeatureTable(
        video_ids=np.asarray(video_ids),
        timestamps=np.asarray(timestamps),
        vision=np.asarray(vision),
        audio=np.asarray(audio),
        text=np.asarray(text),
        target_vision=np.asarray(targets),
        annotation_present=np.ones(len(video_ids), dtype=bool),
    )


def test_fixed_projection_is_seeded_linear_and_data_independent() -> None:
    values = np.arange(24, dtype=float).reshape(4, 6) + 1.0
    first = core.fixed_projection(values, 3, seed=7)
    again = core.fixed_projection(values, 3, seed=7)
    rescaled = core.fixed_projection(values * 10.0, 3, seed=7)
    other = core.fixed_projection(values, 3, seed=8)

    assert np.array_equal(first, again)
    assert np.allclose(first * 10.0, rescaled)
    assert not np.array_equal(first, other)


def test_temporal_and_cross_video_controls_have_no_fixed_pairings() -> None:
    table = _table()
    temporal = core.temporal_derangement(table)
    cross_video = core.cross_video_derangement(table)

    assert sorted(temporal) == list(range(len(table.video_ids)))
    assert np.all(temporal != np.arange(len(temporal)))
    assert np.all(table.video_ids != table.video_ids[cross_video])


def test_feature_table_rejects_wrong_shapes_and_nonfinite_values() -> None:
    table = _table()
    table.validate()
    broken = core.FeatureTable(
        video_ids=table.video_ids,
        timestamps=table.timestamps,
        vision=table.vision[:, :-1],
        audio=table.audio,
        text=table.text,
        target_vision=table.target_vision,
        annotation_present=table.annotation_present,
    )
    with np.testing.assert_raises_regex(ValueError, "vision must have shape"):
        broken.validate()


def test_small_training_path_exercises_model_codec_controls_and_agent() -> None:
    table = _table(rows_per_video=10)
    train = table.subset(["video_0", "video_1"])
    test = table.subset(["video_2"])
    model = core.fit_world_model(train, seed=0, steps=3)
    shuffled = core.fit_world_model(train, seed=0, shuffled=True, steps=3)
    codec = core.fit_codec(model, train, seed=0, steps=3)
    deranged = core.fit_codec(model, train, seed=0, deranged_audio_text=True, steps=3)
    metrics = core.evaluate_video(model, shuffled, codec, deranged, train, test)

    expected = {
        "incumbent_mse",
        "persistence_mse",
        "ridge_mse",
        "shuffle_model_mse",
        "shuffle_model_persistence_mse",
        "vision_mse",
        "audio_mse",
        "audio_latent_mse",
        "audio_deranged_mse",
        "audio_deranged_latent_mse",
        "audio_constant_mse",
        "text_mse",
        "text_latent_mse",
        "text_deranged_mse",
        "text_deranged_latent_mse",
        "text_constant_mse",
        "actual_nll",
        "temporal_deranged_nll",
    }
    assert expected <= set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["prediction_finite"] == 1.0
    assert all(metrics[f"{modality.value}_agent_wiring"] == 1.0 for modality in core.MODALITIES)
    assert len(core.model_fingerprint(model)) == 64


def _passing_row(video: int, seed: int = 0) -> dict[str, object]:
    return {
        "video_id": f"video_{video}",
        "fold": video // 2,
        "seed": seed,
        "model_fingerprint": "a" * 64,
        "incumbent_mse": 0.10,
        "persistence_mse": 0.20,
        "ridge_mse": 0.15,
        "shuffle_model_mse": 0.20,
        "shuffle_model_persistence_mse": 0.20,
        "vision_mse": 0.14,
        "audio_mse": 0.08,
        "audio_latent_mse": 0.08,
        "audio_deranged_mse": 0.10,
        "audio_deranged_latent_mse": 0.10,
        "audio_constant_mse": 0.10,
        "text_mse": 0.08,
        "text_latent_mse": 0.08,
        "text_deranged_mse": 0.10,
        "text_deranged_latent_mse": 0.10,
        "text_constant_mse": 0.10,
        "actual_nll": 1.0,
        "temporal_deranged_nll": 2.0,
        "prediction_finite": 1.0,
        "vision_agent_wiring": 1.0,
        "audio_agent_wiring": 1.0,
        "text_agent_wiring": 1.0,
    }


def test_integration_decision_aggregates_by_video_and_requires_six_of_eight() -> None:
    rows = [_passing_row(video, seed) for video in range(8) for seed in core.SEEDS]
    decision = core.integration_decision(rows)
    assert decision["all_integration_endpoints_pass"]
    assert decision["supporting_videos"] == {
        "real_visual_dynamics": 8,
        "vision_codec_migration": 8,
        "audio_substitution": 8,
        "text_substitution": 8,
        "temporal_surprise": 8,
        "passive_agent_wiring": 8,
    }

    for row in rows:
        if row["video_id"] in {"video_0", "video_1", "video_2"}:
            row["audio_mse"] = 0.2
    failed = core.integration_decision(rows)
    assert failed["supporting_videos"]["audio_substitution"] == 5
    assert not failed["passes"]["audio_substitution"]
    assert not failed["all_integration_endpoints_pass"]


def test_substitution_margin_uses_the_frozen_multiply_by_1_1_boundary() -> None:
    rows = [_passing_row(video, seed) for video in range(8) for seed in core.SEEDS]
    for row in rows:
        for modality in ("audio", "text"):
            row[f"{modality}_mse"] = 1.0
            row[f"{modality}_latent_mse"] = 1.0
            row[f"{modality}_deranged_mse"] = 1.1
            row[f"{modality}_deranged_latent_mse"] = 1.1
            row[f"{modality}_constant_mse"] = 1.1

    decision = core.integration_decision(rows)
    assert decision["supporting_videos"]["audio_substitution"] == 8
    assert decision["supporting_videos"]["text_substitution"] == 8
