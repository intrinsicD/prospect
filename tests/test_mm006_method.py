"""Scientific, leakage, control, and branch tests for MM-006."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_horizon_diagnostics import method as mm005_method
from bench.multimodal_preflight import dataset
from bench.multimodal_warp_diagnostics import method


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def template() -> method.WarpPanel:
    ids: list[str] = []
    times: list[float] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        for index in range(mm005_method.MATCHED_COUNTS[video_id]):
            ids.append(video_id)
            times.append(1.5 + 0.5 * index)
    zeros: np.ndarray = np.zeros((len(ids), 3, 8, 8), dtype=np.float64)
    panel = method.WarpPanel(
        video_ids=np.asarray(ids),
        timestamps=np.asarray(times),
        previous=zeros.copy(),
        current=zeros.copy(),
        target=zeros.copy(),
    )
    panel.validate(3)
    return panel


def _synthetic_metric(
    template: method.WarpPanel,
    scenario: str,
    family: str = "quadrant_flow",
) -> dict[str, Any]:
    panel = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0], 3, scenario)
    known_flow, known_target_flow = method._synthetic_motions(len(panel.video_ids), scenario)
    fold = dataset.formal_folds()[0]
    train = np.asarray([value in fold.train_ids for value in panel.video_ids], dtype=bool)
    normalizer = method._fit_normalizer(panel.current[train])
    return method._metric_row(
        panel.subset([fold.test_ids[0]]),
        domain="synthetic_3",
        family=family,
        fold=fold,
        normalizer=normalizer,
        panel_seed=method.SYNTHETIC_SEEDS[0],
        scenario=scenario,
        known_causal_flow=(
            None
            if scenario in {"ambiguous", "source_null"}
            else known_flow[np.asarray(panel.video_ids) == fold.test_ids[0]]
        ),
        known_target_flow=(
            known_target_flow[np.asarray(panel.video_ids) == fold.test_ids[0]]
            if scenario in {"translation", "reversal", "stationary"}
            else None
        ),
    )


def test_candidate_order_and_common_support_are_frozen() -> None:
    assert len(method.CANDIDATES) == 25
    assert method.CANDIDATES[0] == (0.0, 0.0)
    assert method.CANDIDATES == tuple(
        sorted(
            method.CANDIDATES,
            key=lambda item: (item[0] ** 2 + item[1] ** 2, item[0], item[1]),
        )
    )
    assert method.CENTRAL_COORDS.shape == (36, 2)
    assert set(method.PARITIES.tolist()) == {0, 1}
    assert [int(np.sum(method.TILE_IDS == tile)) for tile in range(4)] == [9, 9, 9, 9]


def test_signed_bilinear_warp_convention() -> None:
    y: np.ndarray
    x: np.ndarray
    y, x = np.indices((8, 8), dtype=np.float64)
    grid = (10.0 * y + x)[None, :, :]
    coords = np.asarray([(3.0, 3.0), (4.0, 4.0)])
    shifted = method._sample(grid, coords, (0.5, 1.0))
    expected = np.asarray([[10.0 * 2.5 + 2.0, 10.0 * 3.5 + 3.0]])
    assert np.array_equal(shifted, expected)
    with pytest.raises(ValueError, match="out-of-bounds"):
        method._sample(grid, np.asarray([(0.0, 0.0)]), (1.0, 0.0))


def test_batch_search_is_bit_exact_with_reference() -> None:
    rng = np.random.Generator(np.random.PCG64(6006))
    previous = rng.normal(size=(4, 3, 8, 8))
    current = rng.normal(size=(4, 3, 8, 8))
    for family in method.FLOW_FAMILIES:
        batch_flow, batch_confidence = method._estimate_batch(previous, current, family)
        for index in range(len(previous)):
            flow, confidence = method._estimate_one(previous[index], current[index], family)
            assert np.array_equal(batch_flow[index], flow)
            assert np.allclose(batch_confidence[index], confidence, rtol=0.0, atol=1e-15)


def test_causal_api_and_outputs_are_target_isolated(template: method.WarpPanel) -> None:
    panel = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0], 3, "translation")
    fold = dataset.formal_folds()[0]
    train = np.asarray([value in fold.train_ids for value in panel.video_ids], dtype=bool)
    normalizer = method._fit_normalizer(panel.current[train])
    video = panel.subset([fold.test_ids[0]])
    known, known_target = method._synthetic_motions(len(panel.video_ids), "translation")
    known_video = known[np.asarray(panel.video_ids) == fold.test_ids[0]]
    known_target_video = known_target[np.asarray(panel.video_ids) == fold.test_ids[0]]
    before = method._metric_row(
        video,
        domain="synthetic_3",
        family="quadrant_flow",
        fold=fold,
        normalizer=normalizer,
        panel_seed=method.SYNTHETIC_SEEDS[0],
        scenario="translation",
        known_causal_flow=known_video,
        known_target_flow=known_target_video,
    )
    mutated = method.WarpPanel(
        video_ids=video.video_ids.copy(),
        timestamps=video.timestamps.copy(),
        previous=video.previous.copy(),
        current=video.current.copy(),
        target=video.target + 1_000.0,
    )
    after = method._metric_row(
        mutated,
        domain="synthetic_3",
        family="quadrant_flow",
        fold=fold,
        normalizer=normalizer,
        panel_seed=method.SYNTHETIC_SEEDS[0],
        scenario="translation",
        known_causal_flow=known_video,
        known_target_flow=known_target_video,
    )
    for name in (
        "causal_input_sha256",
        "causal_flow_sha256",
        "causal_prediction_sha256",
        "source_xfit_input_sha256",
        "source_xfit_flow_sha256",
        "source_xfit_prediction_sha256",
    ):
        assert before[name] == after[name]
    assert before["oracle_xfit_input_sha256"] != after["oracle_xfit_input_sha256"]
    assert before["target_sha256"] != after["target_sha256"]


def test_oracle_is_target_sensitive_and_checkerboard_complete(template: method.WarpPanel) -> None:
    panel = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0], 3, "translation")
    first = method._estimate_oracle_xfit(panel.current[:6], panel.target[:6], "quadrant_flow")
    second = method._estimate_oracle_xfit(panel.current[:6], panel.target[:6] + 4.0, "quadrant_flow")
    assert first.input_sha256 != second.input_sha256
    assert first.flow.shape == (6, 2, 4, 2)
    assert first.prediction.shape == (6, 3, 6, 6)
    assert np.all(np.isfinite(first.prediction))


def test_source_crossfit_held_cells_cannot_change_their_own_prediction() -> None:
    rng = np.random.Generator(np.random.PCG64(606_006))
    previous = rng.normal(size=(5, 3, 8, 8))
    current = rng.normal(size=(5, 3, 8, 8))
    before = method._estimate_source_xfit(previous, current, "quadrant_flow")
    mutated = current.copy()
    parity_zero = method.CENTRAL_COORDS[method.PARITIES == 0].astype(int)
    mutated[:, :, parity_zero[:, 0], parity_zero[:, 1]] += 1_000.0
    after = method._estimate_source_xfit(previous, mutated, "quadrant_flow")
    before_flat = before.prediction.reshape(5, 3, method.CENTRAL_CELLS)
    after_flat = after.prediction.reshape(5, 3, method.CENTRAL_CELLS)
    assert np.array_equal(before.flow[:, 0], after.flow[:, 0])
    assert np.array_equal(before_flat[:, :, method.PARITIES == 0], after_flat[:, :, method.PARITIES == 0])


@pytest.mark.parametrize("family", method.FLOW_FAMILIES)  # type: ignore[untyped-decorator]
def test_causal_channel_permutation_and_prediction_replay(
    template: method.WarpPanel,
    family: str,
) -> None:
    panel = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0], 3, "translation")
    video = panel.subset([dataset.formal_folds()[0].test_ids[0]])
    forward = method._estimate_causal(video.previous, video.current, family)
    permutation = np.asarray([2, 0, 1])
    permuted = method._estimate_causal(video.previous[:, permutation], video.current[:, permutation], family)
    replay = method._apply_flow(video.current, forward.flow, family)
    assert np.array_equal(permuted.flow, forward.flow)
    assert np.array_equal(permuted.prediction, forward.prediction[:, permutation])
    assert np.allclose(permuted.confidence, forward.confidence, rtol=0.0, atol=1e-15)
    assert np.array_equal(replay, forward.prediction)
    assert method._array_sha256(replay) == forward.prediction_sha256


@pytest.mark.parametrize("family", method.FLOW_FAMILIES)  # type: ignore[untyped-decorator]
def test_known_translation_recovers_and_rejects_controls(
    template: method.WarpPanel,
    family: str,
) -> None:
    row = _synthetic_metric(template, "translation", family)
    mse = cast(dict[str, float], row["mse"])
    assert row["oracle_support"] is True
    assert row["causal_support"] is True
    assert row["source_support"] is True
    assert mse["causal"] < 0.01 * mse["persistence"]
    assert mse["causal"] < 0.10 * mse["history_shuffle"]
    assert mse["causal"] < 0.10 * mse["reverse_sign"]


def test_reversal_validates_oracle_only_branch(template: method.WarpPanel) -> None:
    row = _synthetic_metric(template, "reversal")
    assert row["oracle_support"] is True
    assert row["causal_support"] is False
    assert row["source_support"] is True


def test_appearance_stationary_and_ambiguity_do_not_false_pass(template: method.WarpPanel) -> None:
    appearance = _synthetic_metric(template, "appearance")
    source_null = _synthetic_metric(template, "source_null")
    stationary = _synthetic_metric(template, "stationary")
    ambiguous = _synthetic_metric(template, "ambiguous")
    assert appearance["oracle_support"] is False
    assert source_null["source_support"] is False
    assert source_null["source_shuffle_null_hit"] is False
    assert (
        source_null["mse"]["source_reconstruction"] * method.CONTROL_FACTOR
        > source_null["mse"]["history_source_reconstruction"]
    )
    assert stationary["mse"]["persistence"] == 0.0
    assert stationary["oracle_support"] is False
    assert stationary["causal_support"] is False
    assert stationary["source_support"] is False
    assert ambiguous["causal_support"] is False
    assert ambiguous["causal_confidence_gap"] <= 1e-12


def test_derangement_is_fixed_point_free_and_within_video(template: method.WarpPanel) -> None:
    mapping = method._derangement(template)
    assert np.all(mapping != np.arange(len(mapping)))
    assert np.array_equal(template.video_ids[mapping], template.video_ids)


def test_flow_endpoint_compares_each_oracle_parity_without_cancellation() -> None:
    causal = np.asarray([[[1.0, 0.0]]])
    oracle = np.asarray([[[[1.0, 0.0]], [[-1.0, 0.0]]]])
    stats = method._flow_stats(causal, oracle)
    assert stats["endpoint_mse"] == 2.0
    assert stats["cosine"] == 0.0


def test_metric_schema_recomputes_sse_and_rejects_leakage_label(template: method.WarpPanel) -> None:
    row = _synthetic_metric(template, "translation")
    validated = method._validate_metric_row(row, synthetic=True)
    assert validated == row
    tampered = deepcopy(row)
    tampered["uses_target"] = True
    with pytest.raises(ValueError, match="leakage labels"):
        method._validate_metric_row(tampered, synthetic=True)
    tampered = deepcopy(row)
    cast(dict[str, float], tampered["mse"])["causal"] += 0.01
    with pytest.raises(ValueError, match="SSE/MSE"):
        method._validate_metric_row(tampered, synthetic=True)
    tampered = deepcopy(row)
    tampered["causal_support"] = not bool(tampered["causal_support"])
    with pytest.raises(ValueError, match="predicate does not replay"):
        method._validate_metric_row(tampered, synthetic=True)
    tampered = deepcopy(row)
    tampered["causal_ratio"] = float(tampered["causal_ratio"]) + 0.01
    with pytest.raises(ValueError, match="derived causal ratio"):
        method._validate_metric_row(tampered, synthetic=True)
    tampered = deepcopy(row)
    tampered["candidate_count"] = 24
    with pytest.raises(ValueError, match="candidate count"):
        method._validate_metric_row(tampered, synthetic=True)


def test_source_null_gate_is_independent_of_identity_baseline() -> None:
    row: dict[str, Any] = {
        "scenario": "source_null",
        "source_support": False,
        "source_shuffle_null_hit": True,
        "mse": {
            "source_reconstruction": 0.89,
            "history_source_reconstruction": 1.0,
            "source_identity": 0.90,
        },
    }
    summary = method._synthetic_summary([row])
    assert summary["source_null_branch_failures"] == 1
    assert summary["negative_passes"] is False


def _family(
    domain: str,
    *,
    oracle: int,
    causal: int,
    source: int,
    full_only: int = 0,
    boundary: int = 0,
    null: int = 0,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "family": method.PRIMARY_FAMILY,
        "oracle_performance_supporting_videos": oracle,
        "oracle_supporting_videos": oracle,
        "causal_performance_supporting_videos": causal,
        "causal_supporting_videos": causal,
        "source_supporting_videos": source,
        "full_oracle_supporting_videos": full_only,
        "full_oracle_only_videos": full_only,
        "oracle_improving_videos": oracle,
        "causal_improving_videos": causal,
        "full_oracle_improving_videos": 7 if full_only >= 6 else full_only,
        "source_improving_videos": 7 if source >= 6 else source,
        "source_null_improving_videos": 7 if source >= 6 else source,
        "oracle_performance_fold_coverage": oracle >= 4,
        "oracle_fold_coverage": oracle >= 4,
        "causal_performance_fold_coverage": causal >= 4,
        "causal_fold_coverage": causal >= 4,
        "full_oracle_fold_coverage": full_only >= 4,
        "source_fold_coverage": source >= 4,
        "oracle_performance_passes": oracle >= 6,
        "oracle_passes": oracle >= 6,
        "causal_performance_passes": causal >= 6,
        "causal_passes": causal >= 6,
        "full_oracle_passes": full_only >= 6,
        "source_passes": source >= 6,
        "boundary_warning_videos": boundary,
        "target_shuffle_null_support": null,
        "video_rows": [],
    }


def _decision_families(pixel: dict[str, Any], taesd: dict[str, Any]) -> list[dict[str, Any]]:
    global_pixel = {**_family("pixel", oracle=0, causal=0, source=0), "family": "global_translation"}
    global_taesd = {**_family("taesd", oracle=0, causal=0, source=0), "family": "global_translation"}
    return [global_pixel, pixel, global_taesd, taesd]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "pixel,taesd,expected",
    [
        (
            _family("pixel", oracle=0, causal=0, source=0, full_only=6),
            _family("taesd", oracle=0, causal=0, source=0),
            "target_fitted_oracle_overfit_supported",
        ),
        (
            _family("pixel", oracle=0, causal=0, source=0, boundary=3),
            _family("taesd", oracle=0, causal=0, source=0),
            "tested_transport_range_inconclusive",
        ),
        (
            _family("pixel", oracle=0, causal=0, source=0),
            _family("taesd", oracle=0, causal=0, source=0),
            "tested_pixel_warp_ceiling_failure_supported",
        ),
        (
            _family("pixel", oracle=8, causal=0, source=8),
            _family("taesd", oracle=0, causal=0, source=0),
            "two_frame_motion_extrapolation_failure_supported",
        ),
        (
            _family("pixel", oracle=8, causal=0, source=0),
            _family("taesd", oracle=0, causal=0, source=0),
            "low_resolution_correspondence_failure_supported",
        ),
        (
            _family("pixel", oracle=8, causal=8, source=8),
            _family("taesd", oracle=0, causal=0, source=0),
            "taesd_transport_equivariance_failure_supported",
        ),
        (
            _family("pixel", oracle=8, causal=8, source=8),
            _family("taesd", oracle=8, causal=0, source=8),
            "taesd_two_frame_motion_extrapolation_failure_supported",
        ),
        (
            _family("pixel", oracle=8, causal=8, source=8),
            _family("taesd", oracle=8, causal=0, source=0),
            "taesd_causal_correspondence_failure_supported",
        ),
        (
            _family("pixel", oracle=8, causal=8, source=8),
            _family("taesd", oracle=8, causal=8, source=8),
            "single_step_causal_warp_fix_supported",
        ),
    ],
)
def test_decision_ladder(
    pixel: dict[str, Any],
    taesd: dict[str, Any],
    expected: str,
) -> None:
    synthetic = {"positive_passes": True, "negative_passes": True}
    classification, _ = method._decision(_decision_families(pixel, taesd), synthetic)
    assert classification == expected


def test_synthetic_and_null_precedence() -> None:
    families = _decision_families(
        _family("pixel", oracle=8, causal=8, source=8),
        _family("taesd", oracle=8, causal=8, source=8),
    )
    assert method._decision(families, {"positive_passes": False, "negative_passes": True})[0] == (
        "invalid_MM006_synthetic_positive_control"
    )
    assert method._decision(families, {"positive_passes": True, "negative_passes": False})[0] == (
        "invalid_MM006_synthetic_negative_control"
    )
    families[0]["target_shuffle_null_support"] = 6
    assert method._decision(families, {"positive_passes": True, "negative_passes": True})[0] == (
        "invalid_MM006_real_negative_control"
    )
    families[0]["target_shuffle_null_support"] = 3
    assert method._decision(families, {"positive_passes": True, "negative_passes": True})[0] == (
        "inconclusive_MM006_real_negative_control"
    )


def test_global_pass_primary_failure_is_inconclusive() -> None:
    pixel = _family("pixel", oracle=0, causal=0, source=0)
    taesd = _family("taesd", oracle=0, causal=0, source=0)
    families = _decision_families(pixel, taesd)
    families[0].update(
        {
            "oracle_passes": True,
            "causal_passes": True,
            "oracle_supporting_videos": 8,
            "causal_supporting_videos": 8,
        }
    )
    classification, labels = method._decision(
        families,
        {"positive_passes": True, "negative_passes": True},
    )
    assert classification == "MM006_diagnostic_inconclusive"
    assert "global_primary_inconsistency" in labels


def test_raw_six_without_complete_family_gate_is_inconclusive() -> None:
    pixel = _family("pixel", oracle=6, causal=0, source=0)
    pixel["oracle_passes"] = False
    pixel["oracle_fold_coverage"] = False
    families = _decision_families(pixel, _family("taesd", oracle=0, causal=0, source=0))
    assert method._decision(families, {"positive_passes": True, "negative_passes": True})[0] == (
        "MM006_diagnostic_inconclusive"
    )


def test_source_branch_requires_complete_source_gate() -> None:
    pixel = _family("pixel", oracle=8, causal=0, source=6)
    pixel["source_passes"] = False
    pixel["source_fold_coverage"] = False
    families = _decision_families(pixel, _family("taesd", oracle=0, causal=0, source=0))
    assert method._decision(families, {"positive_passes": True, "negative_passes": True})[0] == (
        "MM006_diagnostic_inconclusive"
    )


def test_downstream_primary_borderline_preempts_strong_label() -> None:
    families = _decision_families(
        _family("pixel", oracle=0, causal=0, source=0),
        _family("taesd", oracle=4, causal=0, source=0),
    )
    assert method._decision(families, {"positive_passes": True, "negative_passes": True})[0] == (
        "MM006_diagnostic_inconclusive"
    )


def test_oracle_pairing_failure_cannot_be_called_transport_overfit() -> None:
    pixel = _family("pixel", oracle=0, causal=0, source=0, full_only=8)
    pixel.update(
        {
            "oracle_performance_supporting_videos": 8,
            "oracle_performance_fold_coverage": True,
            "oracle_performance_passes": True,
        }
    )
    families = _decision_families(pixel, _family("taesd", oracle=0, causal=0, source=0))
    assert method._decision(families, {"positive_passes": True, "negative_passes": True})[0] == (
        "MM006_diagnostic_inconclusive"
    )


def test_config_and_report_keep_oracle_diagnostic_only() -> None:
    config = method.config_record()
    assert config["oracle"]["uses_target"] is True
    assert config["oracle"]["diagnostic_only"] is True
    assert config["causal"]["target_isolation_required"] is True
    summary: dict[str, Any] = {
        "schema_version": method.SCHEMA_VERSION,
        "experiment_id": method.EXPERIMENT_ID,
        "synthetic_control": {"positive_passes": True, "negative_passes": True},
        "families": [],
        "activity": {"supporting_videos": 7},
        "decision": {
            "classification": "MM006_diagnostic_inconclusive",
            "recommended_next_step": "replicate",
            "mechanism_labels": [],
        },
        "claim_boundary": "oracle is diagnostic only",
    }
    report = method.report_text(summary)
    assert "oracle is diagnostic only" in report
    assert "MM006_diagnostic_inconclusive" in report
