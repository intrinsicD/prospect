"""Scientific-engine tests for MM-005 matched half-horizon replay."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_horizon_diagnostics import method
from bench.multimodal_preflight import dataset


def _formal_raw_table(channels: int, *, pixels: bool = False) -> method.RawGridTable:
    video_ids: list[str] = []
    timestamps: list[float] = []
    current: list[np.ndarray] = []
    target: list[np.ndarray] = []
    rows = np.linspace(-0.8, 0.8, 8)[:, None]
    columns = np.linspace(-1.0, 1.0, 8)[None, :]
    for video_index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
        count = dataset.EXPECTED_WINDOW_COUNTS[video_id]
        sequence: list[np.ndarray] = []
        for position in range(count + 2):
            values = []
            for channel in range(channels):
                field = np.sin(rows + columns + 0.07 * position + 0.11 * video_index + 0.13 * channel)
                values.append(0.5 + 0.15 * field if pixels else field)
            sequence.append(np.stack(values))
        for position in range(count):
            video_ids.append(video_id)
            timestamps.append(1.0 + 0.5 * position)
            current.append(sequence[position])
            target.append(sequence[position + 2])
    return method.raw_grid_table(
        video_ids=np.asarray(video_ids),
        timestamps=np.asarray(timestamps),
        current=np.asarray(current, dtype=np.float32),
        saved_target=np.asarray(target, dtype=np.float32),
        expected_channels=channels,
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def complete_evidence() -> dict[str, object]:
    return method.execute(_formal_raw_table(4), _formal_raw_table(3, pixels=True))


def test_matched_panel_has_exact_common_causal_rows_and_parent_parity() -> None:
    raw = _formal_raw_table(4)
    panel = method.matched_panel(raw)

    assert len(panel.video_ids) == method.MATCHED_ROWS == 453
    assert {
        video_id: int(np.sum(panel.video_ids == video_id)) for video_id in dataset.SAMPLE_VIDEO_IDS
    } == method.MATCHED_COUNTS
    assert method._identity_sha256(panel.video_ids, panel.timestamps) == method.MATCHED_IDENTITY_SHA256
    np.testing.assert_array_equal(panel.previous[0], raw.current[0])
    np.testing.assert_array_equal(panel.current[0], raw.current[1])
    np.testing.assert_array_equal(panel.target_0p5[0], raw.current[2])
    np.testing.assert_array_equal(panel.target_1p0[0], raw.current[3])
    np.testing.assert_array_equal(panel.target_0p5[0], raw.saved_target[0])
    np.testing.assert_array_equal(panel.target_1p0[0], raw.saved_target[1])
    assert panel.timestamps[0] == 1.5

    provenance = method.panel_provenance(raw)
    assert provenance["rows"] == 453
    assert provenance["counts"] == method.MATCHED_COUNTS
    assert provenance["identity_sha256"] == method.MATCHED_IDENTITY_SHA256
    assert set(provenance) == {
        "rows",
        "counts",
        "channels",
        "identity_sha256",
        "previous_sha256",
        "current_sha256",
        "target_0p5_sha256",
        "target_1p0_sha256",
    }


def test_matched_panel_rejects_saved_target_or_identity_drift() -> None:
    raw = _formal_raw_table(4)
    target = raw.saved_target.copy()
    target[0, 0, 0, 0] += 0.125
    tampered = method.RawGridTable(raw.video_ids, raw.timestamps, raw.current, target)
    with pytest.raises(ValueError, match="saved-target parity"):
        method.matched_panel(tampered)

    times = raw.timestamps.copy()
    times[1] += 0.125
    with pytest.raises(ValueError, match="identities"):
        method.raw_grid_table(raw.video_ids, times, raw.current, raw.saved_target, expected_channels=4)


def test_half_cycle_derangement_is_shared_deterministic_and_within_video() -> None:
    panel = method.matched_panel(_formal_raw_table(4)).subset(dataset.formal_folds()[0].train_ids)
    first = method.half_cycle_derangement(panel)
    second = method.half_cycle_derangement(panel)

    np.testing.assert_array_equal(first, second)
    assert np.all(first != np.arange(len(first)))
    np.testing.assert_array_equal(panel.video_ids[first], panel.video_ids)
    assert sorted(first.tolist()) == list(range(len(first)))


def test_normalizer_is_train_current_only_shared_across_horizons_and_scale_floored() -> None:
    panel = method.matched_panel(_formal_raw_table(4))
    fold = dataset.formal_folds()[0]
    normalizer = method._fit_normalizer(panel.subset(fold.train_ids).current)
    held_out = np.isin(panel.video_ids, fold.test_ids)
    mutated = method.MatchedPanel(
        panel.video_ids,
        panel.timestamps,
        panel.previous.copy(),
        panel.current.copy(),
        panel.target_0p5.copy(),
        panel.target_1p0.copy(),
    )
    mutated.current[held_out] += 1e9
    mutated.target_0p5[held_out] -= 1e9
    mutated.target_1p0[held_out] *= -1e9
    refit = method._fit_normalizer(mutated.subset(fold.train_ids).current)
    assert normalizer.fingerprint() == refit.fingerprint()
    np.testing.assert_array_equal(normalizer.mean, refit.mean)
    np.testing.assert_array_equal(normalizer.scale, refit.scale)
    for horizon in method.HORIZONS:
        reference_fit = method._fit_one(
            method._normalize_panel(panel, normalizer).subset(fold.train_ids),
            horizon,
            method.MAIN_ARM,
            "ordered",
            domain="taesd",
            panel_seed=None,
            fold=fold,
            normalizer=normalizer,
        )
        mutated_fit = method._fit_one(
            method._normalize_panel(mutated, refit).subset(fold.train_ids),
            horizon,
            method.MAIN_ARM,
            "ordered",
            domain="taesd",
            panel_seed=None,
            fold=fold,
            normalizer=refit,
        )
        np.testing.assert_array_equal(reference_fit.weights, mutated_fit.weights)
        assert reference_fit.record["design_sha256"] == mutated_fit.record["design_sha256"]
        assert reference_fit.record["weight_fingerprint"] == mutated_fit.record["weight_fingerprint"]

    constant: np.ndarray = np.ones((10, 4, 8, 8), dtype=float)
    floored = method._fit_normalizer(constant)
    np.testing.assert_array_equal(floored.scale, np.full((4, 1, 1), method.SCALE_FLOOR))


def test_patch_order_dimensions_and_residual_targets_are_frozen() -> None:
    panel = method.matched_panel(_formal_raw_table(4)).subset([dataset.SAMPLE_VIDEO_IDS[0]])
    current_design, half_target = method._design(panel, 0.5, method.COMPARATOR_ARM)
    history_design, one_target = method._design(panel, 1.0, method.MAIN_ARM)
    assert current_design.shape == (len(panel.video_ids) * 36, 36)
    assert history_design.shape == (len(panel.video_ids) * 36, 72)
    assert half_target.shape == one_target.shape == (len(panel.video_ids) * 36, 4)

    coded: np.ndarray = np.arange(2 * 8 * 8, dtype=float).reshape(1, 2, 8, 8)
    first_patch = method._patches(coded, 3)[0]
    np.testing.assert_array_equal(first_patch[:9], coded[0, 0, :3, :3].reshape(-1))
    np.testing.assert_array_equal(first_patch[9:], coded[0, 1, :3, :3].reshape(-1))


def test_ridge_matches_direct_solution_and_horizon_is_in_fit_identity() -> None:
    panel = method.matched_panel(_formal_raw_table(4))
    fold = dataset.formal_folds()[0]
    normalizer = method._fit_normalizer(panel.subset(fold.train_ids).current)
    train = method._normalize_panel(panel, normalizer).subset(fold.train_ids)
    fit = method._fit_one(
        train,
        0.5,
        method.COMPARATOR_ARM,
        "ordered",
        domain="taesd",
        panel_seed=None,
        fold=fold,
        normalizer=normalizer,
    )
    x, y = method._design(train, 0.5, method.COMPARATOR_ARM)
    augmented = np.c_[x, np.ones(len(x))]
    penalty = method.RIDGE_PENALTY * np.eye(augmented.shape[1])
    penalty[-1, -1] = 0.0
    expected = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
    np.testing.assert_allclose(fit.weights, expected, rtol=1e-12, atol=1e-12)
    assert fit.record["horizon_seconds"] == 0.5

    one = method._fit_one(
        train,
        1.0,
        method.COMPARATOR_ARM,
        "ordered",
        domain="taesd",
        panel_seed=None,
        fold=fold,
        normalizer=normalizer,
    )
    assert fit.record["fit_identity_sha256"] != one.record["fit_identity_sha256"]
    assert fit.record["source_identity_sha256"] == one.record["source_identity_sha256"]
    assert fit.record["design_sha256"] == one.record["design_sha256"]
    assert fit.record["normalizer_fingerprint"] == one.record["normalizer_fingerprint"]


def test_velocity_baseline_scales_by_horizon() -> None:
    shape = (1, 1, 8, 8)
    panel = method.MatchedPanel(
        np.asarray(["video_10993"]),
        np.asarray([1.5]),
        np.ones(shape),
        np.full(shape, 2.0),
        np.full(shape, 3.0),
        np.full(shape, 4.0),
    )
    arm = method.COMPARATOR_ARM
    weights = np.zeros((arm.feature_dim(1) + 1, 1))
    fit = method.FitResult(weights, {"weight_fingerprint": "0" * 64})
    fits = {"ordered": fit, "target_shuffle": fit}
    fold = dataset.formal_folds()[0]

    half = method._metric_row(
        panel, panel, 0.5, arm, fits, domain="taesd", panel_seed=None, fold=fold, video_id="video_10993"
    )
    one = method._metric_row(
        panel, panel, 1.0, arm, fits, domain="taesd", panel_seed=None, fold=fold, video_id="video_10993"
    )
    assert half["constant_velocity_mse"] == 0.0
    assert one["constant_velocity_mse"] == 0.0


def test_synthetic_panel_is_deterministic_and_obeys_both_exact_target_rules() -> None:
    template = method.matched_panel(_formal_raw_table(4))
    first, first_record = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0])
    second, second_record = method.synthetic_panel(template, method.SYNTHETIC_SEEDS[0])
    assert first_record == second_record
    np.testing.assert_array_equal(first.previous, second.previous)
    np.testing.assert_array_equal(first.current, second.current)
    difference = first.current - first.previous
    shift = method._shift_right(first.current) - first.current
    np.testing.assert_allclose(first.target_0p5, first.current + 0.25 * shift + 0.75 * difference)
    np.testing.assert_allclose(first.target_1p0, first.current + 0.50 * shift + 1.50 * difference)
    assert not np.array_equal(method._expected_kernel(4, 0.5), method._expected_kernel(4, 1.0))


def test_complete_execution_has_exact_membership_and_passes_synthetic_controls(
    complete_evidence: dict[str, object],
) -> None:
    assert len(cast(list[object], complete_evidence["real_fit_rows"])) == 80
    assert len(cast(list[object], complete_evidence["real_metric_rows"])) == 64
    assert len(cast(list[object], complete_evidence["synthetic_fit_rows"])) == 120
    assert len(cast(list[object], complete_evidence["synthetic_metric_rows"])) == 144
    assert len(cast(list[object], complete_evidence["activity_rows"])) == 16
    summary = method.summarize(complete_evidence)
    synthetic = cast(dict[str, Any], summary["synthetic_control"])
    assert synthetic["positive_passes"] is True
    assert synthetic["negative_passes"] is True
    assert synthetic["numerical_failures"] == 0
    assert synthetic["kernel_failures"] == 0
    assert synthetic["horizon_selector_failures"] == 0
    assert len(synthetic["conditions"]) == 48


def test_fit_membership_shares_normalizers_sources_and_only_permitted_control_fields(
    complete_evidence: dict[str, object],
) -> None:
    fits = cast(list[dict[str, Any]], complete_evidence["real_fit_rows"])
    for domain in method.DOMAINS:
        for fold in range(4):
            selected = [row for row in fits if row["domain"] == domain and row["fold"] == fold]
            assert len({row["source_identity_sha256"] for row in selected}) == 1
            assert len({row["normalizer_fingerprint"] for row in selected}) == 1
            for arm in method.ARMS:
                for horizon in method.HORIZONS:
                    group = {
                        row["control_id"]: row
                        for row in selected
                        if row["arm_id"] == arm.arm_id and row["horizon_seconds"] == horizon
                    }
                    assert group["ordered"]["input_sha256"] == group["target_shuffle"]["input_sha256"]
                    assert group["ordered"]["design_sha256"] == group["target_shuffle"]["design_sha256"]
                    if arm.uses_history:
                        assert group["ordered"]["target_sha256"] == group["history_shuffle"]["target_sha256"]
                half = next(
                    row
                    for row in selected
                    if row["arm_id"] == arm.arm_id and row["control_id"] == "ordered" and row["horizon_seconds"] == 0.5
                )
                one = next(
                    row
                    for row in selected
                    if row["arm_id"] == arm.arm_id and row["control_id"] == "ordered" and row["horizon_seconds"] == 1.0
                )
                for key in ("source_identity_sha256", "input_sha256", "design_sha256", "normalizer_fingerprint"):
                    assert half[key] == one[key]


def test_synthetic_thresholds_are_inclusive_and_nextafter_strict(
    complete_evidence: dict[str, object],
) -> None:
    fits = deepcopy(cast(list[dict[str, Any]], complete_evidence["synthetic_fit_rows"]))
    metrics = deepcopy(cast(list[dict[str, Any]], complete_evidence["synthetic_metric_rows"]))
    main = next(row for row in metrics if row["arm_id"] == method.MAIN_ARM_ID)
    identity = (main["panel_seed"], main["horizon_seconds"], main["video_id"])
    current = next(
        row
        for row in metrics
        if (row["panel_seed"], row["horizon_seconds"], row["video_id"]) == identity
        and row["arm_id"] == method.COMPARATOR_ARM.arm_id
    )
    pointwise = next(
        row
        for row in metrics
        if (row["panel_seed"], row["horizon_seconds"], row["video_id"]) == identity
        and row["arm_id"] == method.POINTWISE_ABLATION.arm_id
    )
    main["ordered_mse"] = 1.0
    main["persistence_mse"] = 10.0
    main["target_shuffle_mse"] = 2.0
    main["history_shuffle_mse"] = 2.0
    current["ordered_mse"] = 2.0
    pointwise["ordered_mse"] = 2.0
    summary = method._synthetic_summary(fits, metrics)
    assert summary["positive_passes"] is True
    assert summary["negative_passes"] is True

    current["ordered_mse"] = np.nextafter(2.0, 0.0)
    assert method._synthetic_summary(fits, metrics)["negative_passes"] is False
    current["ordered_mse"] = 2.0
    fits[0]["linear_system_residual"] = method.LINEAR_RESIDUAL_MAX
    assert method._synthetic_summary(fits, metrics)["positive_passes"] is True
    fits[0]["linear_system_residual"] = np.nextafter(method.LINEAR_RESIDUAL_MAX, math.inf)
    assert method._synthetic_summary(fits, metrics)["positive_passes"] is False
    fits[0]["linear_system_residual"] = 0.0
    main_fit = next(row for row in fits if row["arm_id"] == method.MAIN_ARM_ID and row["control_id"] == "ordered")
    main_fit["kernel_relative_error"] = method.KERNEL_ERROR_MAX
    assert method._synthetic_summary(fits, metrics)["positive_passes"] is True
    main_fit["kernel_relative_error"] = np.nextafter(method.KERNEL_ERROR_MAX, math.inf)
    assert method._synthetic_summary(fits, metrics)["positive_passes"] is False


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("section", "field", "replacement"),
    [
        ("real_fit_rows", "horizon_seconds", 1),
        ("real_fit_rows", "fold", False),
        ("real_fit_rows", "channels", 4.0),
        ("real_fit_rows", "weights_finite", 1),
        ("real_metric_rows", "test_rows", 60.0),
        ("real_metric_rows", "video_id", 10993),
        ("activity_rows", "rows", 60.0),
        ("activity_rows", "active", 1),
    ],
)
def test_validator_rejects_json_primitive_type_drift(
    complete_evidence: dict[str, object], section: str, field: str, replacement: object
) -> None:
    bad = deepcopy(complete_evidence)
    cast(list[dict[str, Any]], bad[section])[0][field] = replacement
    with pytest.raises(ValueError):
        method.validate_evidence(bad)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation", ["shared_normalizer", "target_input", "history_target", "paired_design"]
)
def test_validator_rejects_causal_binding_tamper(complete_evidence: dict[str, object], mutation: str) -> None:
    bad = deepcopy(complete_evidence)
    fits = cast(list[dict[str, Any]], bad["real_fit_rows"])
    if mutation == "shared_normalizer":
        fits[1]["normalizer_fingerprint"] = "f" * 64
    elif mutation == "target_input":
        row = next(item for item in fits if item["control_id"] == "target_shuffle")
        row["input_sha256"] = "f" * 64
    elif mutation == "history_target":
        row = next(item for item in fits if item["control_id"] == "history_shuffle")
        row["target_sha256"] = "f" * 64
    else:
        row = next(item for item in fits if item["horizon_seconds"] == 1.0)
        row["design_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        method.validate_evidence(bad)


def test_validator_rejects_derived_metric_and_kernel_tamper(
    complete_evidence: dict[str, object],
) -> None:
    ratio_bad = deepcopy(complete_evidence)
    metric = cast(list[dict[str, Any]], ratio_bad["real_metric_rows"])[0]
    metric["ordered_ratio"] = cast(float, metric["ordered_ratio"]) + 0.01
    with pytest.raises(ValueError, match="ordered_ratio"):
        method.validate_evidence(ratio_bad)

    kernel_bad = deepcopy(complete_evidence)
    fit = next(
        row
        for row in cast(list[dict[str, Any]], kernel_bad["synthetic_fit_rows"])
        if row["recovered_kernel"] is not None
    )
    cast(list[list[float]], fit["recovered_kernel"])[0][0] += 0.01
    with pytest.raises(ValueError, match="kernel"):
        method.validate_evidence(kernel_bad)


def test_real_support_thresholds_are_inclusive_and_nextafter_strict() -> None:
    exact: dict[str, Any] = {
        "arm_id": method.MAIN_ARM_ID,
        "ordered_mse": 1.0,
        "persistence_mse": 1.2,
        "target_shuffle_mse": 1.1,
        "history_shuffle_mse": 1.1,
    }
    assert method._supports(exact)
    for field, boundary in (
        ("persistence_mse", 1.2),
        ("target_shuffle_mse", 1.1),
        ("history_shuffle_mse", 1.1),
    ):
        below = dict(exact)
        below[field] = np.nextafter(boundary, 0.0)
        assert not method._supports(below)


def test_paired_advantage_boundary_and_zero_persistence_are_explicit() -> None:
    half = {"ordered_mse": 1.0, "persistence_mse": 2.0}
    one = {"ordered_mse": 1.1, "persistence_mse": 2.0}
    assert method._paired_advantage(half, one)
    one["ordered_mse"] = np.nextafter(1.1, 0.0)
    assert not method._paired_advantage(half, one)
    for half_persistence, one_persistence in ((0.0, 2.0), (2.0, 0.0), (0.0, 0.0)):
        half["persistence_mse"] = half_persistence
        one["persistence_mse"] = one_persistence
        assert not method._paired_advantage(half, one)


def test_activity_uses_floored_denominator_and_full_eight_by_eight_grid() -> None:
    ratio, active = method._activity_predicate(1e-4, 0.0)
    assert ratio == 1e11
    assert active is False
    assert method._activity_predicate(1e-4, 1e-3) == (0.1, True)
    assert method._activity_predicate(1e-4, 1.2e-4)[1] is True
    assert method._activity_predicate(1e-4, np.nextafter(1.2e-4, 0.0))[1] is False

    panel = method.matched_panel(_formal_raw_table(3, pixels=True))
    half = panel.current.copy()
    half[:, :, 0, :] += 0.1
    half[:, :, -1, :] += 0.1
    border_only = method.MatchedPanel(
        panel.video_ids,
        panel.timestamps,
        panel.previous,
        panel.current,
        half,
        panel.target_1p0,
    )
    rows = method._activity_rows(border_only)
    assert all(row["persistence_mse"] > 0.0 for row in rows if row["horizon_seconds"] == 0.5)


def _decision_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    controls = [
        {
            "domain": domain,
            "horizon_seconds": horizon,
            "arm_id": arm.arm_id,
            "target_shuffle_support": 0,
            "history_shuffle_support": 0 if arm.uses_history else None,
        }
        for domain in method.DOMAINS
        for horizon in method.HORIZONS
        for arm in method.ARMS
    ]
    arms = [
        {
            "domain": domain,
            "horizon_seconds": horizon,
            "arm_id": arm.arm_id,
            "supporting_videos": 0,
        }
        for domain in method.DOMAINS
        for horizon in method.HORIZONS
        for arm in method.ARMS
    ]
    paired = [
        {"domain": domain, "arm_id": arm.arm_id, "supporting_videos": 0}
        for domain in method.DOMAINS
        for arm in method.ARMS
    ]
    return controls, arms, paired


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("case", "expected"),
    [
        ("synthetic_positive", "invalid_MM005_synthetic_positive_control"),
        ("synthetic_negative", "invalid_MM005_synthetic_negative_control"),
        ("target_invalid", "invalid_MM005_real_negative_control"),
        ("target_inconclusive", "inconclusive_MM005_real_negative_control"),
        ("taesd_one", "matched_one_second_taesd_signal_supported"),
        ("taesd_one_border", "inconclusive_matched_one_second_taesd_signal"),
        ("taesd_mismatch", "half_second_horizon_mismatch_supported"),
        ("taesd_half", "inconclusive_half_second_taesd_signal"),
        ("taesd_half_border", "inconclusive_half_second_taesd_borderline"),
        ("pixel_one", "matched_one_second_pixel_signal_supported"),
        ("pixel_one_border", "inconclusive_matched_one_second_pixel_signal"),
        ("pixel_mismatch", "half_second_taesd_representation_failure_supported"),
        ("pixel_half", "inconclusive_half_second_pixel_signal"),
        ("pixel_half_border", "inconclusive_half_second_pixel_borderline"),
        ("objective", "half_second_tested_spatial_local_linear_objective_failure_supported"),
        ("inactive", "insufficient_half_second_source_change"),
        ("heterogeneous", "inconclusive_half_second_video_heterogeneity"),
        ("fallback", "MM005_diagnostic_inconclusive"),
    ],
)
def test_frozen_decision_order_covers_every_terminal_branch(case: str, expected: str) -> None:
    controls, arms, paired = _decision_inputs()
    synthetic_positive = case != "synthetic_positive"
    synthetic_negative = case != "synthetic_negative"
    activity: int | float = 0

    def arm(domain: str, horizon: float, arm_id: str = method.MAIN_ARM_ID) -> dict[str, Any]:
        return next(
            row
            for row in arms
            if row["domain"] == domain and row["horizon_seconds"] == horizon and row["arm_id"] == arm_id
        )

    if case == "target_invalid":
        controls[0]["target_shuffle_support"] = 6
    elif case == "target_inconclusive":
        controls[0]["target_shuffle_support"] = 3
    elif case == "taesd_one":
        arm("taesd", 1.0)["supporting_videos"] = 6
    elif case == "taesd_one_border":
        arm("taesd", 1.0)["supporting_videos"] = 3
    elif case == "taesd_mismatch":
        arm("taesd", 0.5)["supporting_videos"] = 6
        next(row for row in paired if row["domain"] == "taesd")["supporting_videos"] = 6
    elif case == "taesd_half":
        arm("taesd", 0.5)["supporting_videos"] = 6
        next(row for row in paired if row["domain"] == "taesd")["supporting_videos"] = 5
    elif case == "taesd_half_border":
        arm("taesd", 0.5)["supporting_videos"] = 3
    elif case == "pixel_one":
        arm("pixel", 1.0)["supporting_videos"] = 6
    elif case == "pixel_one_border":
        arm("pixel", 1.0)["supporting_videos"] = 3
    elif case == "pixel_mismatch":
        arm("pixel", 0.5)["supporting_videos"] = 6
        next(row for row in paired if row["domain"] == "pixel")["supporting_videos"] = 6
    elif case == "pixel_half":
        arm("pixel", 0.5)["supporting_videos"] = 6
        next(row for row in paired if row["domain"] == "pixel")["supporting_videos"] = 5
    elif case == "pixel_half_border":
        arm("pixel", 0.5)["supporting_videos"] = 3
    elif case == "objective":
        activity = 6
    elif case == "heterogeneous":
        activity = 3
    elif case == "fallback":
        activity = 2.5
    classification, labels = method._decision(
        synthetic_positive=synthetic_positive,
        synthetic_negative=synthetic_negative,
        controls=controls,
        arms=arms,
        paired=paired,
        half_activity_support=cast(int, activity),
    )
    assert classification == expected
    if case == "taesd_mismatch":
        assert labels == ["current_diff_3x3_horizon_mismatch"]


def test_primary_arm_wins_fixed_decision_order_when_both_qualify() -> None:
    controls, arms, paired = _decision_inputs()
    for arm in method.ARMS:
        next(
            row
            for row in arms
            if row["domain"] == "taesd" and row["horizon_seconds"] == 0.5 and row["arm_id"] == arm.arm_id
        )["supporting_videos"] = 6
        next(row for row in paired if row["domain"] == "taesd" and row["arm_id"] == arm.arm_id)["supporting_videos"] = 6
    classification, labels = method._decision(
        synthetic_positive=True,
        synthetic_negative=True,
        controls=controls,
        arms=arms,
        paired=paired,
        half_activity_support=0,
    )
    assert classification == "half_second_horizon_mismatch_supported"
    assert labels == ["current_diff_3x3_horizon_mismatch"]


def test_config_and_report_expose_frozen_scope_and_claim_boundary(
    complete_evidence: dict[str, object],
) -> None:
    config = method.config_record()
    assert config["matched_rows"] == 453
    assert config["real_fit_rows"] == 80
    assert config["synthetic_fit_rows"] == 120
    assert config["matched_identity_sha256"] == method.MATCHED_IDENTITY_SHA256
    assert cast(str, config["paired_rule"]).startswith("persistence_0p5 > 0 and persistence_1p0 > 0")
    assert [row["train_rows"] for row in cast(list[dict[str, Any]], config["folds"])] == [332, 335, 346, 346]

    summary = method.summarize(complete_evidence)
    report = method.report_text(summary)
    assert "MM-005 matched half-horizon replay" in report
    assert "teacher-free rollout capability" in cast(str, summary["claim_boundary"])
    with pytest.raises(ValueError):
        method.report_text({"schema_version": method.SCHEMA_VERSION, "experiment_id": "MM-004"})
