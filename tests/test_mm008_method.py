"""Pure-method, leakage, replay, and synthetic-calibration tests for MM-008."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import method
from bench.multimodal_resolution_diagnostics import method as mm007


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def synthetic_metrics() -> Iterator[dict[tuple[str, str], method.SyntheticMetrics]]:
    metrics: dict[tuple[str, str], method.SyntheticMetrics] = {}
    for scenario in method.SYNTHETIC_SCENARIOS:
        case = method.synthetic_case(scenario)
        for arm in method.ALL_ARMS:
            metrics[(scenario, arm)] = method.synthetic_metrics(case, arm)
    yield metrics


def test_geometry_candidates_and_backward_warp_are_frozen() -> None:
    assert method.GEOMETRY.coords.shape == (48 * 48, 2)
    assert np.array_equal(method.GEOMETRY.normalized_coords[[0, -1]], [[-1.0, -1.0], [1.0, 1.0]])
    assert set(method.GEOMETRY.macro_ids.tolist()) == set(range(36))
    assert np.bincount(method.GEOMETRY.macro_ids).tolist() == [64] * 36
    assert np.bincount(method.GEOMETRY.parities).tolist() == [1152, 1152]
    assert method.INITIAL_TRANSLATIONS == mm007.NATIVE_CANDIDATES
    assert method.COORDINATE_VALUES == (
        (0.0, -4.0, 4.0, -8.0, 8.0),
        (0.0, -4.0, 4.0, -8.0, 8.0),
        (0.0, -2.0, 2.0, -4.0, 4.0),
        (0.0, -2.0, 2.0, -4.0, 4.0),
        (0.0, -2.0, 2.0, -4.0, 4.0),
        (0.0, -2.0, 2.0, -4.0, 4.0),
    )
    y, x = np.indices((64, 64), dtype=np.float64)
    source = np.broadcast_to(10.0 * y + x, (1, 3, 64, 64)).copy()
    selected = np.zeros(len(method.GEOMETRY.coords), dtype=bool)
    selected[0] = True
    parameters = np.asarray([[0.5, 1.0, 0.0, 0.0, 0.0, 0.0]])
    sampled = method._sample_affine(source, parameters, selected)
    assert np.array_equal(sampled, np.full((1, 3, 1), 10.0 * 7.5 + 7.0))
    invalid = np.asarray([[8.0, 0.0, 2.0, 0.0, 0.0, 0.0]])
    assert method._admissible(invalid).tolist() == [False]


def test_exact_candidate_ties_and_probe_only_tolerance() -> None:
    losses = np.asarray([[1.0, 1.0], [1.0 + 5e-13, 1.0]])
    assert np.array_equal(method._first_minimum(losses), [0, 1])
    assert np.array_equal(method._optimizer_tolerance(np.asarray([0.0, 1e6])), [1e-12, 1e-4])


def test_appearance_solve_clips_raw_gain_and_raw_bias_independently() -> None:
    source = np.broadcast_to(np.asarray([1.0, 2.0]), (1, 3, 2)).copy()
    target = 10.0 * source + 1.0
    gains, biases = method._ols(source, target, np.ones((1, 2), dtype=bool))
    assert np.array_equal(gains, np.full((1, 3), 4.0))
    assert np.array_equal(biases, np.full((1, 3), 1.0))


def test_boundary_components_are_reported_separately() -> None:
    parameters = np.asarray([[8.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    gains = np.asarray([[4.0, 1.0, 1.0]])
    biases = np.asarray([[-4.0, 0.0, 0.0]])
    site, gradient, gain, bias, boundary = method._boundary_fractions(
        "combined", parameters, gains, biases
    )
    assert np.array_equal(site, [0.5])
    assert np.array_equal(gradient, [0.0])
    assert np.allclose(gain, [1.0 / 3.0], rtol=0.0, atol=1e-15)
    assert np.allclose(bias, [1.0 / 3.0], rtol=0.0, atol=1e-15)
    assert np.array_equal(boundary, [0.5])


def test_external_translation_sentinels_replay_mm007_exactly() -> None:
    rng = np.random.Generator(np.random.PCG64(8_008))
    source = rng.normal(size=(2, 3, 64, 64))
    target = rng.normal(size=source.shape)
    for arm, family in (
        ("global_translation", "global_translation"),
        ("quadrant_translation", "quadrant_flow"),
    ):
        full = method.estimate_sentinel_full(source, target, arm)  # type: ignore[arg-type]
        xfit = method.estimate_sentinel_xfit(source, target, arm)  # type: ignore[arg-type]
        expected_full = mm007._estimate_full(source, target, 64, family)
        expected_xfit = mm007._estimate_xfit(source, target, 64, family)
        assert np.array_equal(full.flow, expected_full.flow)
        assert np.array_equal(full.prediction, expected_full.prediction)
        assert np.array_equal(xfit.flow, expected_xfit.flow)
        assert np.array_equal(xfit.prediction, expected_xfit.prediction)


def test_translation_and_appearance_positive_controls(
    synthetic_metrics: dict[tuple[str, str], method.SyntheticMetrics],
) -> None:
    for arm in ("global_translation", "quadrant_translation", "affine", "combined"):
        assert synthetic_metrics[("translation", arm)].expectation_failures("translation") == ()
    # The appearance mechanism correctly does not create support, but the
    # prespecified >=0.90 isolation margin is too tight for two frozen rows.
    translation_appearance = synthetic_metrics[("translation", "appearance")]
    assert not np.any(translation_appearance.complete_support)
    assert translation_appearance.expectation_failures("translation") == ("negative_margin",)
    for arm in ("appearance", "combined"):
        assert synthetic_metrics[("appearance", arm)].expectation_failures("appearance") == ()
    for arm in ("global_translation", "quadrant_translation", "affine"):
        assert synthetic_metrics[("appearance", arm)].expectation_failures("appearance") == ()


def test_affine_control_exposes_coordinate_descent_endpoint_blocker(
    synthetic_metrics: dict[tuple[str, str], method.SyntheticMetrics],
) -> None:
    for arm in ("affine", "combined"):
        metrics = synthetic_metrics[("affine", arm)]
        assert np.all(metrics.full_mse <= 0.5 * metrics.persistence_mse)
        assert np.all(metrics.xfit_mse <= 0.5 * metrics.persistence_mse)
        assert metrics.full_parameter_error == 2.0
        assert metrics.xfit_parameter_error == 2.0
        failures = metrics.expectation_failures("affine")
        assert "full_parameter_endpoint" in failures
        assert "xfit_parameter_endpoint" in failures
    case = method.synthetic_case("affine")
    fitted = method.estimate_full(case.source, case.target, "affine")
    assert np.array_equal(fitted.parameters, np.tile([0.0, 0.0, 2.0, -2.0, 0.0, 0.0], (6, 1)))
    # The frozen single-coordinate probe cannot see the required joint
    # ayx->0, axx->-2 move, which is the identified optimizer failure mode.
    assert not np.any(fitted.probe_strict_improvement)


def test_combined_control_exposes_failed_factorial_isolation(
    synthetic_metrics: dict[tuple[str, str], method.SyntheticMetrics],
) -> None:
    combined = synthetic_metrics[("combined", "combined")]
    assert np.all(combined.xfit_mse <= 0.5 * combined.persistence_mse)
    assert np.all(combined.complete_support)
    assert "xfit_parameter_endpoint" in combined.expectation_failures("combined")
    for arm in ("global_translation", "quadrant_translation", "affine", "appearance"):
        excluded = synthetic_metrics[("combined", arm)]
        assert "negative_margin" in excluded.expectation_failures("combined")
        assert np.any(excluded.complete_support)


def test_stationary_and_independent_negative_controls_expose_only_real_failures(
    synthetic_metrics: dict[tuple[str, str], method.SyntheticMetrics],
) -> None:
    for arm in method.ALL_ARMS:
        stationary = synthetic_metrics[("stationary", arm)]
        assert np.array_equal(stationary.persistence_mse, np.zeros(6))
        assert not np.any(stationary.complete_support)
        assert stationary.expectation_failures("stationary") == ()
        independent = synthetic_metrics[("independent", arm)]
        assert "negative_margin" in independent.expectation_failures("independent")
        assert np.any(independent.complete_support)


@pytest.mark.parametrize("arm", method.FACTORIAL_ARMS)  # type: ignore[untyped-decorator]
def test_heldout_target_mutation_cannot_change_own_fit_or_prediction(arm: method.Arm) -> None:
    case = method.synthetic_case("combined")
    before = method.estimate_xfit(case.source[:2], case.target[:2], arm)
    held = method.GEOMETRY.parities == 0
    coords = method.GEOMETRY.coords[held].astype(int)
    mutated = case.target[:2].copy()
    mutated[:, :, coords[:, 0], coords[:, 1]] += 1_000.0
    after = method.estimate_xfit(case.source[:2], mutated, arm)
    assert np.array_equal(before.parameters[:, 0], after.parameters[:, 0])
    assert np.array_equal(before.gains[:, 0], after.gains[:, 0])
    assert np.array_equal(before.biases[:, 0], after.biases[:, 0])
    assert np.array_equal(before.prediction[:, :, held], after.prediction[:, :, held])


def test_deterministic_replay_is_bit_exact() -> None:
    case = method.synthetic_case("translation")
    first = method.estimate_xfit(case.source[:2], case.target[:2], "combined")
    second = method.estimate_xfit(case.source[:2], case.target[:2], "combined")
    for name in (
        "parameters",
        "gains",
        "biases",
        "prediction",
        "objective",
        "probe_strict_improvement",
        "probe_best_improvement",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
