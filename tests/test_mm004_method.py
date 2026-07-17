"""Scientific-method tests for MM-004 spatial/history signal isolation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from bench.multimodal_preflight import dataset
from bench.multimodal_spatial_diagnostics import method


def _formal_raw_table(channels: int, *, pixels: bool = False) -> method.RawGridTable:
    video_ids: list[str] = []
    timestamps: list[float] = []
    current: list[np.ndarray] = []
    target: list[np.ndarray] = []
    row_axis = np.linspace(-0.8, 0.8, 8)[:, None]
    column_axis = np.linspace(-1.0, 1.0, 8)[None, :]
    for video_index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
        for row in range(dataset.EXPECTED_WINDOW_COUNTS[video_id]):
            channel_grids = []
            target_grids = []
            for channel in range(channels):
                phase = 0.09 * row + 0.13 * video_index + 0.17 * channel
                field = np.sin(row_axis + column_axis + phase)
                future = np.sin(row_axis + column_axis + phase + 0.18)
                if pixels:
                    field = 0.5 + 0.18 * field
                    future = 0.5 + 0.18 * future
                channel_grids.append(field)
                target_grids.append(future)
            video_ids.append(video_id)
            timestamps.append(1.0 + 0.5 * row)
            current.append(np.stack(channel_grids))
            target.append(np.stack(target_grids))
    return method.raw_grid_table(
        np.asarray(video_ids),
        np.asarray(timestamps),
        np.asarray(current),
        np.asarray(target),
        expected_channels=channels,
    )


def _parent_record() -> dict[str, object]:
    return {
        "passed": True,
        "rows_compared": 16,
        "absolute_rows_compared": 8,
        "residual_rows_compared": 8,
        "flattened_current_sha256": "0" * 64,
        "flattened_target_sha256": "1" * 64,
        "rtol": 1e-12,
        "atol": 1e-12,
        "max_absolute_error": 0.0,
    }


@pytest.fixture(scope="module")
def complete_evidence() -> dict[str, object]:
    taesd = _formal_raw_table(4)
    pixels = _formal_raw_table(3, pixels=True)
    with patch.object(method, "parent_preflight_record", return_value=_parent_record()):
        return method.execute(taesd, pixels, {})


def test_raw_and_history_tables_have_exact_causal_identities() -> None:
    raw = _formal_raw_table(4)
    history = method.history_table(raw)

    assert len(raw.video_ids) == 477
    assert len(history.video_ids) == method.HISTORY_ROWS == 469
    assert {
        video_id: int(np.sum(history.video_ids == video_id)) for video_id in dataset.SAMPLE_VIDEO_IDS
    } == method.HISTORY_COUNTS
    np.testing.assert_array_equal(history.previous[0], raw.current[0])
    np.testing.assert_array_equal(history.current[0], raw.current[1])
    np.testing.assert_array_equal(history.target[0], raw.target[1])
    assert history.timestamps[0] == 1.5


def test_raw_table_rejects_identity_or_channel_drift() -> None:
    raw = _formal_raw_table(4)
    bad_times = raw.timestamps.copy()
    bad_times[0] += 0.5
    with pytest.raises(ValueError, match="identities"):
        method.raw_grid_table(
            raw.video_ids,
            bad_times,
            raw.current,
            raw.target,
            expected_channels=4,
        )
    with pytest.raises(ValueError, match="4 channels"):
        method.raw_grid_table(
            raw.video_ids,
            raw.timestamps,
            raw.current[:, :3],
            raw.target[:, :3],
            expected_channels=4,
        )


def test_half_cycle_derangement_is_deterministic_within_video_and_fixed_point_free() -> None:
    history = method.history_table(_formal_raw_table(4))
    train = history.subset(dataset.formal_folds()[0].train_ids)
    first = method.half_cycle_derangement(train)
    second = method.half_cycle_derangement(train)

    np.testing.assert_array_equal(first, second)
    assert np.all(first != np.arange(len(first)))
    np.testing.assert_array_equal(train.video_ids[first], train.video_ids)
    assert sorted(first.tolist()) == list(range(len(first)))


def test_train_channel_normalization_cannot_see_held_out_values() -> None:
    history = method.history_table(_formal_raw_table(4))
    fold = dataset.formal_folds()[0]
    normalizer = method._fit_normalizer(history.subset(fold.train_ids).current)
    held_out = np.isin(history.video_ids, fold.test_ids)
    mutated_previous = history.previous.copy()
    mutated_current = history.current.copy()
    mutated_target = history.target.copy()
    mutated_previous[held_out] -= 1e9
    mutated_current[held_out] += 1e9
    mutated_target[held_out] *= -1e9
    mutated = method.GridTable(
        history.video_ids,
        history.timestamps,
        mutated_previous,
        mutated_current,
        mutated_target,
    )
    refit = method._fit_normalizer(mutated.subset(fold.train_ids).current)
    arm = method.ARM_BY_ID[method.MAIN_ARM_ID]
    reference_fit = method._fit_one(
        method._normalize_table(history, normalizer).subset(fold.train_ids),
        arm,
        "ordered",
        domain="taesd",
        panel_seed=None,
        fold=fold,
        normalizer=normalizer,
    )
    mutated_fit = method._fit_one(
        method._normalize_table(mutated, refit).subset(fold.train_ids),
        arm,
        "ordered",
        domain="taesd",
        panel_seed=None,
        fold=fold,
        normalizer=refit,
    )

    assert normalizer.fingerprint() == refit.fingerprint()
    assert reference_fit.record["fit_matrix_sha256"] == mutated_fit.record["fit_matrix_sha256"]
    assert reference_fit.record["weight_fingerprint"] == mutated_fit.record["weight_fingerprint"]
    np.testing.assert_array_equal(reference_fit.weights, mutated_fit.weights)
    assert normalizer.mean.shape == (4, 1, 1)
    assert normalizer.scale.shape == (4, 1, 1)


def test_predictor_ladder_uses_valid_six_by_six_patches_and_frozen_dimensions() -> None:
    table = method.history_table(_formal_raw_table(4)).subset([dataset.SAMPLE_VIDEO_IDS[0]])
    expected_dimensions = {
        "current_1x1": 4,
        "current_diff_1x1": 8,
        "current_3x3": 36,
        "current_diff_3x3": 72,
    }
    for arm in method.ARMS:
        design, target = method._design(table, arm)
        assert design.shape == (len(table.video_ids) * 36, expected_dimensions[arm.arm_id])
        assert target.shape == (len(table.video_ids) * 36, 4)


def test_synthetic_panel_is_deterministic_and_obeys_the_exact_rule() -> None:
    template = method.history_table(_formal_raw_table(4))
    first, first_record = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0])
    second, second_record = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0])

    assert first_record == second_record
    np.testing.assert_array_equal(first.previous, second.previous)
    np.testing.assert_array_equal(first.current, second.current)
    difference = first.current - first.previous
    expected_target = first.current + 0.5 * (method._shift_right(first.current) - first.current) + 1.5 * difference
    np.testing.assert_allclose(first.target, expected_target, rtol=0.0, atol=5e-16)


def _synthetic_threshold_summary(
    *,
    main_mse: float = 0.1,
    persistence_mse: float = 1.0,
    current_mse: float = 0.2,
    pointwise_mse: float = 0.2,
    target_shuffle_mse: float = 0.2,
    history_shuffle_mse: float = 0.2,
    residual: float = method.LINEAR_RESIDUAL_MAX,
    kernel_error: float = method.KERNEL_ERROR_MAX,
) -> dict[str, Any]:
    fits = [
        {
            "weights_finite": True,
            "linear_system_residual": residual,
            "arm_id": method.MAIN_ARM_ID,
            "control_id": "ordered",
            "kernel_relative_error": kernel_error,
        }
    ]
    metrics: list[dict[str, Any]] = []
    for seed in method.SYNTHETIC_SEEDS:
        for video_id in dataset.SAMPLE_VIDEO_IDS:
            metrics.extend(
                [
                    {
                        "panel_seed": seed,
                        "video_id": video_id,
                        "arm_id": method.MAIN_ARM_ID,
                        "ordered_mse": main_mse,
                        "persistence_mse": persistence_mse,
                        "target_shuffle_mse": target_shuffle_mse,
                        "history_shuffle_mse": history_shuffle_mse,
                    },
                    {
                        "panel_seed": seed,
                        "video_id": video_id,
                        "arm_id": "current_3x3",
                        "ordered_mse": current_mse,
                    },
                    {
                        "panel_seed": seed,
                        "video_id": video_id,
                        "arm_id": "current_diff_1x1",
                        "ordered_mse": pointwise_mse,
                    },
                ]
            )
    return method._synthetic_summary(fits, metrics)


def test_synthetic_thresholds_are_inclusive_and_nextafter_strict() -> None:
    exact = _synthetic_threshold_summary()
    assert exact["positive_passes"] is True
    assert exact["negative_passes"] is True

    persistence_fail = _synthetic_threshold_summary(
        main_mse=np.nextafter(0.1, np.inf),
        current_mse=1.0,
        pointwise_mse=1.0,
        target_shuffle_mse=1.0,
        history_shuffle_mse=1.0,
    )
    assert persistence_fail["positive_passes"] is False
    assert persistence_fail["negative_passes"] is True

    for field in (
        "current_mse",
        "pointwise_mse",
        "target_shuffle_mse",
        "history_shuffle_mse",
    ):
        separated = _synthetic_threshold_summary(**{field: np.nextafter(0.2, 0.0)})
        assert separated["positive_passes"] is True
        assert separated["negative_passes"] is False

    residual_fail = _synthetic_threshold_summary(residual=np.nextafter(method.LINEAR_RESIDUAL_MAX, np.inf))
    kernel_fail = _synthetic_threshold_summary(kernel_error=np.nextafter(method.KERNEL_ERROR_MAX, np.inf))
    assert residual_fail["positive_passes"] is False
    assert kernel_fail["positive_passes"] is False


def test_complete_execution_passes_all_synthetic_numerical_and_separation_gates(
    complete_evidence: dict[str, object],
) -> None:
    summary = method.summarize(complete_evidence)
    synthetic = cast(dict[str, Any], summary["synthetic_control"])

    assert len(cast(list[object], complete_evidence["synthetic_fit_rows"])) == 120
    assert len(cast(list[object], complete_evidence["synthetic_metric_rows"])) == 96
    assert len(cast(list[object], complete_evidence["real_fit_rows"])) == 80
    assert len(cast(list[object], complete_evidence["real_metric_rows"])) == 64
    assert synthetic["positive_passes"] is True
    assert synthetic["negative_passes"] is True
    assert synthetic["positive_failures"] == 0
    assert synthetic["negative_failures"] == 0
    assert synthetic["maximum_linear_system_residual"] <= method.LINEAR_RESIDUAL_MAX
    assert synthetic["maximum_kernel_relative_error"] <= method.KERNEL_ERROR_MAX
    assert len(synthetic["conditions"]) == 24


@pytest.mark.parametrize(
    "mutation",
    ["weight_binding", "derived_ratio", "recovered_kernel"],
)
def test_evidence_validator_rejects_cross_link_or_derived_value_tampering(
    complete_evidence: dict[str, object],
    mutation: str,
) -> None:
    bad = deepcopy(complete_evidence)
    if mutation == "weight_binding":
        cast(list[dict[str, Any]], bad["real_metric_rows"])[0]["ordered_weight_fingerprint"] = "f" * 64
    elif mutation == "derived_ratio":
        cast(list[dict[str, Any]], bad["real_metric_rows"])[0]["ordered_ratio"] += 0.01
    else:
        fit = next(
            row for row in cast(list[dict[str, Any]], bad["synthetic_fit_rows"]) if row["recovered_kernel"] is not None
        )
        cast(list[list[float]], fit["recovered_kernel"])[0][0] += 0.01
    with pytest.raises(ValueError):
        method.validate_evidence(bad)


def _decision_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    controls = [
        {
            "domain": domain,
            "arm_id": arm.arm_id,
            "target_shuffle_support": 0,
            "history_shuffle_support": 0 if arm.uses_history else None,
        }
        for domain in method.DOMAINS
        for arm in method.ARMS
    ]
    arms = [
        {"domain": domain, "arm_id": arm.arm_id, "passes": False} for domain in method.DOMAINS for arm in method.ARMS
    ]
    contrasts = [
        {"domain": domain, "contrast_id": contrast_id, "passes": False}
        for domain in method.DOMAINS
        for contrast_id in (
            "history_contribution",
            "spatial_neighborhood_contribution",
            "structured_advantage",
        )
    ]
    return controls, arms, contrasts


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("synthetic_positive", "invalid_MM004_synthetic_positive_control"),
        ("synthetic_negative", "invalid_MM004_synthetic_negative_control"),
        ("target_null_fail", "invalid_MM004_real_negative_control"),
        ("target_null_uncertain", "inconclusive_MM004_real_negative_control"),
        ("taesd", "taesd_local_linear_signal_supported"),
        ("pixel", "taesd_representation_failure_supported"),
        ("pixel_non_main", "taesd_representation_failure_supported"),
        ("objective", "tested_local_objective_or_horizon_failure_supported"),
        ("inactive", "data_dynamics_insufficient_for_local_history_assay"),
        ("heterogeneous", "inconclusive_video_heterogeneity"),
        ("fallback", "MM004_diagnostic_inconclusive"),
    ],
)
def test_frozen_decision_order_covers_every_terminal_branch(case: str, expected: str) -> None:
    controls, arms, contrasts = _decision_inputs()
    synthetic_positive = case != "synthetic_positive"
    synthetic_negative = case != "synthetic_negative"
    activity_support: int = 0
    if case == "target_null_fail":
        controls[0]["target_shuffle_support"] = 6
    elif case == "target_null_uncertain":
        controls[0]["target_shuffle_support"] = 3
    elif case == "taesd":
        arms[0]["passes"] = True
    elif case == "pixel":
        next(row for row in arms if row["domain"] == "pixel" and row["arm_id"] == method.MAIN_ARM_ID)["passes"] = True
    elif case == "pixel_non_main":
        next(row for row in arms if row["domain"] == "pixel" and row["arm_id"] == "current_1x1")["passes"] = True
    elif case == "objective":
        activity_support = 6
    elif case == "heterogeneous":
        activity_support = 3
    elif case == "fallback":
        activity_support = cast(int, 2.5)
    classification, _ = method._decision(
        synthetic_positive=synthetic_positive,
        synthetic_negative=synthetic_negative,
        control_counts=controls,
        arms=arms,
        contrasts=contrasts,
        activity_support=activity_support,
    )
    assert classification == expected


@pytest.mark.parametrize(("history_support", "expects_history"), [(2, True), (3, False), (6, False)])
def test_history_shuffle_is_scoped_to_attribution_not_global_validity(
    history_support: int,
    expects_history: bool,
) -> None:
    controls, arms, contrasts = _decision_inputs()
    next(row for row in controls if row["domain"] == "taesd" and row["arm_id"] == method.MAIN_ARM_ID)[
        "history_shuffle_support"
    ] = history_support
    next(row for row in arms if row["domain"] == "taesd" and row["arm_id"] == method.MAIN_ARM_ID)["passes"] = True
    for row in contrasts:
        if row["domain"] == "taesd":
            row["passes"] = True

    classification, labels = method._decision(
        synthetic_positive=True,
        synthetic_negative=True,
        control_counts=controls,
        arms=arms,
        contrasts=contrasts,
        activity_support=0,
    )

    assert classification == "taesd_local_linear_signal_supported"
    assert ("history_contribution_supported" in labels) is expects_history
    assert ("history_control_predictive_from_current" in labels) is (history_support >= 3)
    assert "spatial_neighborhood_contribution_supported" in labels
    assert "structured_advantage_supported" in labels


@pytest.mark.parametrize(
    ("field", "boundary"),
    [
        ("persistence_mse", 1.2),
        ("target_shuffle_mse", 1.1),
        ("history_shuffle_mse", 1.1),
    ],
)
def test_real_arm_value_thresholds_are_inclusive_and_nextafter_strict(
    field: str,
    boundary: float,
) -> None:
    exact = {
        "arm_id": method.MAIN_ARM_ID,
        "ordered_mse": 1.0,
        "persistence_mse": 1.2,
        "target_shuffle_mse": 1.1,
        "history_shuffle_mse": 1.1,
    }
    assert method._supports(exact)
    below = dict(exact)
    below[field] = np.nextafter(boundary, 0.0)
    assert not method._supports(below)


def _count_threshold_rows(
    *,
    arm_support: int,
    target_control_support: int,
    history_control_support: int,
    constant_velocity_support: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in method.DOMAINS:
        for arm in method.ARMS:
            for index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
                requires_exact_persistence = index < max(
                    arm_support,
                    target_control_support,
                    history_control_support,
                    constant_velocity_support,
                )
                persistence = 1.2 if requires_exact_persistence else np.nextafter(1.2, 0.0)
                target_shuffle = 1.0 if index < target_control_support else 1.1
                history_shuffle = 1.0 if index < history_control_support else 1.1
                rows.append(
                    {
                        "domain": domain,
                        "arm_id": arm.arm_id,
                        "video_id": video_id,
                        "ordered_mse": 1.0,
                        "persistence_mse": persistence,
                        "target_shuffle_mse": target_shuffle,
                        "history_shuffle_mse": history_shuffle if arm.uses_history else None,
                        "constant_velocity_mse": (1.0 if index < constant_velocity_support else 1.1),
                    }
                )
    return rows


@pytest.mark.parametrize(("support", "passes"), [(5, False), (6, True)])
def test_real_arm_and_constant_velocity_count_boundaries(support: int, passes: bool) -> None:
    rows = _count_threshold_rows(
        arm_support=support,
        target_control_support=0,
        history_control_support=0,
        constant_velocity_support=support,
    )
    arm = next(
        row for row in method._arm_summaries(rows) if row["domain"] == "taesd" and row["arm_id"] == "current_1x1"
    )
    baseline = next(row for row in method._baseline_summaries(rows) if row["domain"] == "taesd")
    assert arm["supporting_videos"] == support
    assert arm["passes"] is passes
    assert baseline["constant_velocity_supporting_videos"] == support
    assert baseline["constant_velocity_passes"] is passes


@pytest.mark.parametrize(
    ("support", "classification"), [(2, None), (3, "inconclusive"), (5, "inconclusive"), (6, "invalid")]
)
def test_target_null_count_bands_and_twenty_percent_value_boundary(
    support: int,
    classification: str | None,
) -> None:
    rows = _count_threshold_rows(
        arm_support=0,
        target_control_support=support,
        history_control_support=0,
        constant_velocity_support=0,
    )
    control = next(
        row for row in method._control_summaries(rows) if row["domain"] == "taesd" and row["arm_id"] == "current_1x1"
    )
    assert control["target_shuffle_support"] == support
    controls, arms, contrasts = _decision_inputs()
    controls[0]["target_shuffle_support"] = support
    decision, _ = method._decision(
        synthetic_positive=True,
        synthetic_negative=True,
        control_counts=controls,
        arms=arms,
        contrasts=contrasts,
        activity_support=0,
    )
    if classification == "invalid":
        assert decision == "invalid_MM004_real_negative_control"
    elif classification == "inconclusive":
        assert decision == "inconclusive_MM004_real_negative_control"
    else:
        assert decision == "data_dynamics_insufficient_for_local_history_assay"


@pytest.mark.parametrize(
    ("gate", "arm_id", "summary_field"),
    [
        ("target", "current_1x1", "target_shuffle_support"),
        ("history", method.MAIN_ARM_ID, "history_shuffle_support"),
        ("constant_velocity", "current_1x1", "constant_velocity_supporting_videos"),
    ],
)
def test_null_and_constant_velocity_value_boundaries_are_isolated_nextafter_strict(
    gate: str,
    arm_id: str,
    summary_field: str,
) -> None:
    exact = _count_threshold_rows(
        arm_support=0,
        target_control_support=8 if gate == "target" else 0,
        history_control_support=8 if gate == "history" else 0,
        constant_velocity_support=8 if gate == "constant_velocity" else 0,
    )

    def support(rows: list[dict[str, Any]]) -> int:
        if gate == "constant_velocity":
            summary = next(row for row in method._baseline_summaries(rows) if row["domain"] == "taesd")
        else:
            summary = next(
                row for row in method._control_summaries(rows) if row["domain"] == "taesd" and row["arm_id"] == arm_id
            )
        return int(summary[summary_field])

    assert support(exact) == 8
    above = deepcopy(exact)
    for row in above:
        if gate == "target":
            row["target_shuffle_mse"] = np.nextafter(1.0, np.inf)
        elif gate == "history" and row["history_shuffle_mse"] is not None:
            row["history_shuffle_mse"] = np.nextafter(1.0, np.inf)
        elif gate == "constant_velocity":
            row["constant_velocity_mse"] = np.nextafter(1.0, np.inf)
    assert support(above) == 0


@pytest.mark.parametrize("support", [2, 3])
def test_history_null_count_boundary_is_computed_by_the_reducer(support: int) -> None:
    rows = _count_threshold_rows(
        arm_support=0,
        target_control_support=0,
        history_control_support=support,
        constant_velocity_support=0,
    )
    main = next(
        row
        for row in method._control_summaries(rows)
        if row["domain"] == "taesd" and row["arm_id"] == method.MAIN_ARM_ID
    )
    assert main["history_shuffle_support"] == support


@pytest.mark.parametrize(
    ("activity_support", "classification"),
    [
        (2, "data_dynamics_insufficient_for_local_history_assay"),
        (3, "inconclusive_video_heterogeneity"),
        (5, "inconclusive_video_heterogeneity"),
        (6, "tested_local_objective_or_horizon_failure_supported"),
    ],
)
def test_activity_support_band_boundaries(
    activity_support: int,
    classification: str,
) -> None:
    controls, arms, contrasts = _decision_inputs()
    decision, _ = method._decision(
        synthetic_positive=True,
        synthetic_negative=True,
        control_counts=controls,
        arms=arms,
        contrasts=contrasts,
        activity_support=activity_support,
    )
    assert decision == classification


def test_activity_thresholds_include_exact_boundaries() -> None:

    ratio, active = method._activity_predicate(1e-4, 1e-3)
    assert ratio == pytest.approx(0.1)
    assert active
    _, active_at_upper = method._activity_predicate(1e-4, 1.2e-4)
    assert active_at_upper
    assert not method._activity_predicate(np.nextafter(1e-4, 0.0), 1e-3)[1]
    assert not method._activity_predicate(1e-4, np.nextafter(1e-3, np.inf))[1]
    assert not method._activity_predicate(1e-4, np.nextafter(1.2e-4, 0.0))[1]


def _contrast_rows(comparator_mse: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in method.DOMAINS:
        for arm in method.ARMS:
            for video_id in dataset.SAMPLE_VIDEO_IDS:
                rows.append(
                    {
                        "domain": domain,
                        "arm_id": arm.arm_id,
                        "video_id": video_id,
                        "ordered_mse": 1.0 if arm.arm_id == method.MAIN_ARM_ID else comparator_mse,
                        "constant_velocity_mse": comparator_mse,
                    }
                )
    return rows


def test_paired_contrast_ten_percent_boundary_is_inclusive() -> None:
    exact = method._contrast_summaries(_contrast_rows(1.1))
    below = method._contrast_summaries(_contrast_rows(np.nextafter(1.1, 0.0)))

    assert all(row["supporting_videos"] == 8 and row["passes"] for row in exact)
    assert all(row["supporting_videos"] == 0 and not row["passes"] for row in below)


@pytest.mark.parametrize(("support", "passes"), [(5, False), (6, True)])
def test_paired_contrast_count_boundary(support: int, passes: bool) -> None:
    rows: list[dict[str, Any]] = []
    for domain in method.DOMAINS:
        for arm in method.ARMS:
            for index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
                comparator = 1.1 if index < support else np.nextafter(1.1, 0.0)
                rows.append(
                    {
                        "domain": domain,
                        "arm_id": arm.arm_id,
                        "video_id": video_id,
                        "ordered_mse": (1.0 if arm.arm_id == method.MAIN_ARM_ID else comparator),
                        "constant_velocity_mse": comparator,
                    }
                )
    summaries = method._contrast_summaries(rows)
    assert all(row["supporting_videos"] == support for row in summaries)
    assert all(row["passes"] is passes for row in summaries)


def test_parent_raw256_preflight_recomputes_all_sixteen_rows_and_rejects_drift() -> None:
    raw = _formal_raw_table(4)
    flattened = method.mm003.VisualTable(
        raw.video_ids,
        raw.timestamps,
        raw.current.reshape(477, 256),
        raw.target.reshape(477, 256),
    )
    matched = method.mm003.matched_table(flattened)
    projection = np.zeros((256, 32), dtype=float)
    rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        transform = method.mm003.fit_transform(
            "raw256_native",
            flattened.subset(fold.train_ids).current,
            projection,
        )
        for predictor_id in method.mm003.PREDICTOR_IDS:
            rows.extend(
                method.mm003._probe_rows_for(
                    "raw256_native",
                    predictor_id,
                    fold,
                    matched,
                    transform,
                )
            )
    parent = {"probe_rows": rows}
    with patch.object(method.mm003, "validate_evidence", return_value=parent):
        record = method.parent_preflight_record(raw, parent)
    assert record["passed"] is True
    assert record["rows_compared"] == 16
    assert record["max_absolute_error"] == 0.0

    tampered = deepcopy(parent)
    cast(list[dict[str, Any]], tampered["probe_rows"])[0]["ridge_mse"] += 1e-3
    with (
        patch.object(method.mm003, "validate_evidence", return_value=tampered),
        pytest.raises(ValueError, match="parity failed"),
    ):
        method.parent_preflight_record(raw, tampered)


def test_pixel_bounds_are_checked_before_formal_execution() -> None:
    taesd = _formal_raw_table(4)
    pixels = _formal_raw_table(3, pixels=True)
    invalid = method.RawGridTable(
        pixels.video_ids,
        pixels.timestamps,
        pixels.current + 1.0,
        pixels.target,
    )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        method.execute(taesd, invalid, {})


def test_config_and_report_expose_frozen_dimensions_and_claim_boundary(
    complete_evidence: dict[str, object],
) -> None:
    config = method.config_record()
    dimensions = {row["arm_id"]: row["taesd_feature_dim"] for row in config["arms"]}
    summary = method.summarize(complete_evidence)
    report = method.report_text(summary)

    assert dimensions == {
        "current_1x1": 4,
        "current_diff_1x1": 8,
        "current_3x3": 36,
        "current_diff_3x3": 72,
    }
    assert config["history_rows"] == 469
    assert config["decision_rules"]["pixel_representation_branch"] == "any_pixel_arm_passes"
    assert len(config["decision_rules"]["synthetic_control"]["negative_predicates"]) == 4
    assert config["decision_rules"]["activity_predicate"]["all"] == [
        "pixel_persistence_mse >= 1e-4",
        "pixel_persistence_mse / half_cycle_persistence_mse >= 0.10",
        "pixel_persistence_mse / half_cycle_persistence_mse <= 1/1.2",
    ]
    assert config["decision_rules"]["mechanism_labels"]["common_predicates"] == [
        "taesd_current_diff_3x3_arm_passes",
        "named_paired_contrast_passes_6_of_8",
    ]
    assert (
        config["decision_rules"]["mechanism_labels"]["history_additional_predicate"]
        == "taesd_main_history_shuffle_control_support <= 2"
    )
    assert [row["order"] for row in config["decision_order"]] == list(range(1, 12))
    assert config["decision_order"][6]["predicate"] == ("all_taesd_arms_fail_and_any_pixel_arm_passes")
    assert "outcome-informed" in report
    assert "Source activity support" in report
    assert "Constant-velocity support" in report
