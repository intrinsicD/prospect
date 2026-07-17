"""Pure scientific-method tests for the MM-002 feature-only diagnostic."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
import pytest

from bench.multimodal_diagnostics import method
from bench.multimodal_preflight import core, dataset


def _formal_shape_table() -> core.FeatureTable:
    video_ids: list[str] = []
    timestamps: list[float] = []
    vision: list[np.ndarray] = []
    audio: list[np.ndarray] = []
    text: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    annotations: list[bool] = []
    axes = np.arange(core.FEATURE_DIM, dtype=float) / 100.0
    for video_index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
        for row in range(dataset.EXPECTED_WINDOW_COUNTS[video_id]):
            value = 1_000.0 * video_index + float(row)
            video_ids.append(video_id)
            timestamps.append(1.0 + 0.5 * row)
            vision.append(value + axes)
            audio.append(-value + axes)
            text.append(2.0 * value + axes)
            targets.append(10_000.0 + value + axes)
            annotations.append(row % 2 == 0)
    return core.FeatureTable(
        video_ids=np.asarray(video_ids),
        timestamps=np.asarray(timestamps, dtype=float),
        vision=np.asarray(vision, dtype=float),
        audio=np.asarray(audio, dtype=float),
        text=np.asarray(text, dtype=float),
        target_vision=np.asarray(targets, dtype=float),
        annotation_present=np.asarray(annotations, dtype=bool),
    )


@pytest.mark.parametrize("horizon", [0.5, 1.0, 2.0])
def test_matched_horizon_table_uses_identical_sources_and_frozen_targets(horizon: float) -> None:
    source = _formal_shape_table()
    original_vision = source.vision.copy()
    matched = method.matched_horizon_table(source, horizon)

    assert len(matched.video_ids) == 461
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        source_indices = np.flatnonzero(source.video_ids == video_id)
        matched_indices = np.flatnonzero(matched.video_ids == video_id)
        kept = source_indices[:-2]
        np.testing.assert_array_equal(matched.video_ids[matched_indices], source.video_ids[kept])
        np.testing.assert_array_equal(matched.timestamps[matched_indices], source.timestamps[kept])
        np.testing.assert_array_equal(matched.vision[matched_indices], source.vision[kept])
        np.testing.assert_array_equal(matched.audio[matched_indices], source.audio[kept])
        np.testing.assert_array_equal(matched.text[matched_indices], source.text[kept])
        np.testing.assert_array_equal(
            matched.annotation_present[matched_indices], source.annotation_present[kept]
        )
        if horizon == 0.5:
            expected_target = source.vision[source_indices[1:-1]]
        elif horizon == 1.0:
            expected_target = source.target_vision[kept]
        else:
            expected_target = source.target_vision[source_indices[2:]]
        np.testing.assert_array_equal(matched.target_vision[matched_indices], expected_target)

    np.testing.assert_array_equal(source.vision, original_vision)


def test_matched_horizon_table_rejects_unfrozen_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        method.matched_horizon_table(_formal_shape_table(), 1.5)


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))


def test_raw_ridge_probe_matches_direct_unpenalized_intercept_solution() -> None:
    table = method.matched_horizon_table(_formal_shape_table(), 1.0)
    fold = dataset.formal_folds()[0]
    rows = method._raw_ridge_rows("matched_1s", 1.0, fold, table)
    train = table.subset(fold.train_ids)
    test = table.subset([fold.test_ids[0]])
    x_train = np.c_[train.vision, np.ones(len(train.video_ids))]
    x_test = np.c_[test.vision, np.ones(len(test.video_ids))]
    penalty = method.RAW_RIDGE_PENALTY * np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0
    ordered_weights = np.linalg.solve(
        x_train.T @ x_train + penalty, x_train.T @ train.target_vision
    )
    derangement = core.temporal_derangement(train)
    shuffled_weights = np.linalg.solve(
        x_train.T @ x_train + penalty,
        x_train.T @ train.target_vision[derangement],
    )

    assert len(rows) == 2
    assert rows[0]["video_id"] == fold.test_ids[0]
    assert rows[0]["raw_persistence_mse"] == pytest.approx(
        _mse(test.vision, test.target_vision), rel=1e-12, abs=1e-12
    )
    assert rows[0]["raw_ridge_mse"] == pytest.approx(
        _mse(x_test @ ordered_weights, test.target_vision), rel=1e-12, abs=1e-12
    )
    assert rows[0]["raw_shuffle_ridge_mse"] == pytest.approx(
        _mse(x_test @ shuffled_weights, test.target_vision), rel=1e-12, abs=1e-12
    )
    assert float(rows[0]["raw_ridge_mse"]) < float(rows[0]["raw_persistence_mse"])
    assert float(rows[0]["raw_ridge_mse"]) < float(rows[0]["raw_shuffle_ridge_mse"])


def _fingerprint(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _fabricated_evidence(
    *,
    world_passes: set[str] | None = None,
    raw_passes: set[str] | None = None,
    codec_passes: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    world_passes = set() if world_passes is None else world_passes
    raw_passes = set() if raw_passes is None else raw_passes
    codec_passes = set() if codec_passes is None else codec_passes
    world_rows: list[dict[str, object]] = []
    codec_rows: list[dict[str, object]] = []
    parent_rows: list[dict[str, object]] = []

    for fold in dataset.formal_folds():
        full_train_rows = sum(dataset.EXPECTED_WINDOW_COUNTS[value] for value in fold.train_ids)
        matched_train_rows = sum(method.MATCHED_COUNTS[value] for value in fold.train_ids)
        for seed in core.SEEDS:
            parent_fingerprint = _fingerprint(
                "world", fold.index, seed, "full_1s", 1_500, "primary"
            )
            for world_spec in method.WORLD_VARIANTS:
                passed = world_spec.variant_id in world_passes
                world_mse = 1.0 if passed else 2.0
                persistence_mse = 2.0 if passed else 1.0
                ridge_mse = 1.5 if passed else 1.0
                for video_id in fold.test_ids:
                    world_rows.append(
                        {
                            "variant_id": world_spec.variant_id,
                            "trajectory_id": world_spec.trajectory_id,
                            "horizon_seconds": world_spec.horizon_seconds,
                            "updates": world_spec.updates,
                            "matched_sources": world_spec.matched_sources,
                            "train_rows": (
                                matched_train_rows if world_spec.matched_sources else full_train_rows
                            ),
                            "test_rows": (
                                method.MATCHED_COUNTS[video_id]
                                if world_spec.matched_sources
                                else dataset.EXPECTED_WINDOW_COUNTS[video_id]
                            ),
                            "video_id": video_id,
                            "fold": fold.index,
                            "seed": seed,
                            "model_fingerprint": _fingerprint(
                                "world",
                                fold.index,
                                seed,
                                world_spec.trajectory_id,
                                world_spec.updates,
                                "primary",
                            ),
                            "shuffle_model_fingerprint": _fingerprint(
                                "world",
                                fold.index,
                                seed,
                                world_spec.trajectory_id,
                                world_spec.updates,
                                "shuffle",
                            ),
                            "world_mse": world_mse,
                            "persistence_mse": persistence_mse,
                            "ridge_mse": ridge_mse,
                            "shuffle_model_mse": 2.0 if passed else 1.0,
                            "shuffle_model_persistence_mse": 2.0 if passed else 1.0,
                        }
                    )

            for codec_spec in method.CODEC_VARIANTS:
                passed = codec_spec.variant_id in codec_passes
                vision_mse = 2.0 if passed else 4.0
                for video_id in fold.test_ids:
                    codec_rows.append(
                        {
                            "variant_id": codec_spec.variant_id,
                            "included_modalities": list(codec_spec.included_modalities),
                            "update_order": list(codec_spec.update_order),
                            "cycles": codec_spec.cycles,
                            "snapshot": codec_spec.snapshot,
                            "train_rows": full_train_rows,
                            "test_rows": dataset.EXPECTED_WINDOW_COUNTS[video_id],
                            "video_id": video_id,
                            "fold": fold.index,
                            "seed": seed,
                            "model_fingerprint": parent_fingerprint,
                            "codec_fingerprint": _fingerprint(
                                "codec", fold.index, seed, codec_spec.variant_id
                            ),
                            "incumbent_world_mse": 2.0,
                            "vision_mse": vision_mse,
                            "vision_latent_mse": 0.25,
                            "vision_to_incumbent_ratio": vision_mse / 2.0,
                        }
                    )

            for video_id in fold.test_ids:
                parent_rows.append(
                    {
                        "fold": fold.index,
                        "seed": seed,
                        "video_id": video_id,
                        "model_fingerprint": parent_fingerprint,
                        "incumbent_mse": 2.0,
                        "persistence_mse": 1.0,
                        "ridge_mse": 1.0,
                        "shuffle_model_mse": 1.0,
                        "shuffle_model_persistence_mse": 1.0,
                        "vision_mse": 4.0,
                        "vision_latent_mse": 0.25,
                    }
                )

    raw_rows: list[dict[str, object]] = []
    for fold in dataset.formal_folds():
        train_rows = sum(method.MATCHED_COUNTS[value] for value in fold.train_ids)
        for probe_id, horizon in method.RAW_PROBE_HORIZONS:
            passed = probe_id in raw_passes
            for video_id in fold.test_ids:
                raw_rows.append(
                    {
                        "probe_id": probe_id,
                        "horizon_seconds": horizon,
                        "train_rows": train_rows,
                        "test_rows": method.MATCHED_COUNTS[video_id],
                        "video_id": video_id,
                        "fold": fold.index,
                        "raw_persistence_mse": 2.0 if passed else 1.0,
                        "raw_ridge_mse": 1.0 if passed else 2.0,
                        "raw_shuffle_ridge_mse": 2.0 if passed else 1.0,
                    }
                )

    integrity_rows: list[dict[str, object]] = []
    horizons = {"full_1s": 1.0, "matched_0p5s": 0.5, "matched_1s": 1.0, "matched_2s": 2.0}
    for fold in dataset.formal_folds():
        for seed in core.SEEDS:
            for trajectory_id, checkpoints in method.INTEGRITY_CHECKPOINTS.items():
                counts = (
                    dataset.EXPECTED_WINDOW_COUNTS
                    if trajectory_id == "full_1s"
                    else method.MATCHED_COUNTS
                )
                pooled_rows = sum(counts[value] for value in fold.test_ids)
                for updates in checkpoints:
                    for model_role in ("primary", "shuffle"):
                        fingerprint = _fingerprint(
                            "world", fold.index, seed, trajectory_id, updates, model_role
                        )
                        for encoder_role in ("online", "ema_target"):
                            integrity_rows.append(
                                {
                                    "trajectory_id": trajectory_id,
                                    "horizon_seconds": horizons[trajectory_id],
                                    "updates": updates,
                                    "fold": fold.index,
                                    "seed": seed,
                                    "model_role": model_role,
                                    "encoder_role": encoder_role,
                                    "pooled_test_rows": pooled_rows,
                                    "model_fingerprint": fingerprint,
                                    "latent_std_min": 0.5,
                                    "latent_effective_rank": 3.0,
                                    "prediction_min_variance": 0.1,
                                    "prediction_finite": True,
                                    "prediction_variance_positive": True,
                                }
                            )

    evidence = {
        "world_rows": world_rows,
        "raw_probe_rows": raw_rows,
        "integrity_rows": integrity_rows,
        "codec_rows": codec_rows,
    }
    return evidence, parent_rows


def test_exact_evidence_validators_reject_schema_type_order_and_finiteness_drift() -> None:
    evidence, _ = _fabricated_evidence()
    assert method.validate_evidence(evidence) == evidence

    mutations: list[tuple[object, str]] = []
    extra_top = deepcopy(evidence)
    extra_top["extra"] = []
    mutations.append((extra_top, "exactly four"))
    extra_row = deepcopy(evidence)
    extra_row["world_rows"][0]["extra"] = 1
    mutations.append((extra_row, "world row schema"))
    boolean_seed = deepcopy(evidence)
    boolean_seed["world_rows"][0]["seed"] = True
    mutations.append((boolean_seed, "identity fields"))
    nonfinite = deepcopy(evidence)
    nonfinite["world_rows"][0]["world_mse"] = float("nan")
    mutations.append((nonfinite, "finite"))
    reordered = deepcopy(evidence)
    reordered["raw_probe_rows"][0], reordered["raw_probe_rows"][1] = (
        reordered["raw_probe_rows"][1],
        reordered["raw_probe_rows"][0],
    )
    mutations.append((reordered, "frozen order"))
    wrong_count = deepcopy(evidence)
    wrong_count["raw_probe_rows"][0]["test_rows"] = 1
    mutations.append((wrong_count, "row counts"))
    wrong_codec_ratio = deepcopy(evidence)
    wrong_codec_ratio["codec_rows"][0]["vision_to_incumbent_ratio"] = 99.0
    mutations.append((wrong_codec_ratio, "does not match"))
    inconsistent_variance = deepcopy(evidence)
    inconsistent_variance["integrity_rows"][0]["prediction_min_variance"] = 0.0
    mutations.append((inconsistent_variance, "conflicts"))
    wrong_cross_link = deepcopy(evidence)
    for row in wrong_cross_link["codec_rows"]:
        if row["fold"] == 0 and row["seed"] == 0 and row["variant_id"] == "shared_atv_600":
            row["model_fingerprint"] = _fingerprint("wrong incumbent")
    mutations.append((wrong_cross_link, "incumbent fingerprint"))

    for mutation, message in mutations:
        with pytest.raises(ValueError, match=message):
            method.validate_evidence(mutation)


def test_summary_mechanically_reports_distinct_world_raw_and_codec_branches() -> None:
    evidence, parent_rows = _fabricated_evidence(
        world_passes={"matched_0p5s_1500", "matched_1s_3000", "matched_1s_6000"},
        raw_passes={"matched_0p5s"},
        codec_passes={"vision_only_2400"},
    )
    summary = method.summarize(evidence, parent_rows)

    assert summary["parent_parity"]["passed"] is True
    assert summary["world"]["diagnosis_labels"] == [
        "short_horizon_rescue",
        "stable_world_budget_rescue",
    ]
    assert summary["world"]["raw_diagnosis_labels"] == [
        "raw_linear_predictability_matched_0p5s_supported",
        "raw_linear_predictability_matched_1s_not_supported",
        "raw_linear_predictability_matched_2s_not_supported",
    ]
    assert summary["codec"]["diagnosis_labels"] == ["codec_sharing_by_budget_interaction"]
    assert summary["decision"] == {
        "classification": "world_and_codec_factor_sensitivity_detected",
        "world_factor_sensitivity": True,
        "codec_factor_sensitivity": True,
    }


def test_shared_high_budget_label_allows_its_pre_audio_text_snapshot_to_pass() -> None:
    evidence, parent_rows = _fabricated_evidence(
        codec_passes={"shared_vat_2400_after_v", "shared_vat_2400"},
    )

    summary = method.summarize(evidence, parent_rows)

    assert summary["codec"]["diagnosis_labels"] == [
        "shared_positive_transfer_or_isolated_instability"
    ]
    assert summary["decision"]["classification"] == "codec_factor_sensitivity_detected"
