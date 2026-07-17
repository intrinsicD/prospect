"""Focused source-only synthetic tests for the MM-009 predictor seam."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from bench.multimodal_causal_diagnostics import predictor
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

CONFIG_SHA256 = "9" * 64
EXPOSED_NONRESERVED_SEEDS = {
    "affine": 990_901,
    "appearance": 990_902,
    "combined": 990_903,
    "stationary": 990_904,
}


def _assert_hard_readonly(array: np.ndarray) -> None:
    assert array.flags.c_contiguous
    assert not array.flags.writeable
    assert array.data.readonly
    current = array
    while True:
        with pytest.raises(ValueError):
            current.setflags(write=True)
        if not isinstance(current.base, np.ndarray):
            assert isinstance(current.base, bytes)
            break
        current = current.base


@pytest.fixture(scope="module")
def positive_results() -> dict[str, tuple[synthetic.SyntheticCase, predictor.SourcePairResult]]:
    assert not set(EXPOSED_NONRESERVED_SEEDS.values()) & set(synthetic.FROZEN_SEED_MAP.values())
    output: dict[str, tuple[synthetic.SyntheticCase, predictor.SourcePairResult]] = {}
    for scenario in ("affine", "appearance", "combined"):
        case = synthetic.generate_case(
            cast(synthetic.Scenario, scenario),
            seed=EXPOSED_NONRESERVED_SEEDS[scenario],
        )
        output[scenario] = (
            case,
            predictor.fit_source_pair(
                case.source[0],
                case.target[0],
                config_sha256=CONFIG_SHA256,
            ),
        )
    return output


def test_public_fitting_api_has_no_future_or_target_argument() -> None:
    for pair_function in (
        predictor.fit_source_pair,
        predictor.fit_checkerboard_history,
    ):
        parameters = inspect.signature(pair_function).parameters
        assert tuple(parameters) == ("previous", "current", "config_sha256")
        assert not {"future", "target"} & set(parameters)
    for control_function in (
        predictor.fit_bias_only_control,
        predictor.fit_checkerboard_bias_only_history,
    ):
        parameters = inspect.signature(control_function).parameters
        assert tuple(parameters) == ("current", "config_sha256")
        assert not {"future", "target", "previous"} & set(parameters)
    assert predictor.V22_PROTOCOL_SHA256 == geometry.PROTOCOL_SHA256
    assert not hasattr(predictor, "PROTOCOL_SHA256")


def test_operator_application_is_spatial_first_once_and_unclipped() -> None:
    rng = np.random.Generator(np.random.PCG64(90_009))
    current = np.ascontiguousarray(rng.normal(size=(3, 64, 64)), dtype=np.float64)
    parameters = np.asarray((-4.0, 4.0, 0.0, 2.0, -2.0, 0.0), dtype=np.float64)
    gains = np.asarray((1.2, 0.8, 1.4), dtype=np.float64)
    biases = np.asarray((0.3, -0.2, 0.1), dtype=np.float64)
    sampled = geometry.sample_scalar(current, parameters, geometry.FULL_MASK)
    expected = gains[:, None] * sampled + biases[:, None]
    actual = predictor.apply_operator_once(current, parameters, gains, biases)
    assert np.array_equal(actual, expected)
    _assert_hard_readonly(actual)

    large = np.ascontiguousarray(current * 100.0, dtype=np.float64)
    unclipped = predictor.apply_operator_once(large, parameters, gains, biases)
    assert float(np.max(np.abs(unclipped))) > 4.0


@pytest.mark.parametrize("scenario", ("affine", "appearance", "combined"))
def test_constant_operator_recovery_and_one_step_forecast(
    positive_results: dict[str, tuple[synthetic.SyntheticCase, predictor.SourcePairResult]],
    scenario: str,
) -> None:
    case, result = positive_results[scenario]
    truth = case.truth
    assert truth is not None
    operator = result.operator(cast(predictor.Arm, scenario))
    expected_future = predictor.apply_operator_once(
        case.target[0],
        truth.theta_array(),
        truth.gain_array(),
        truth.bias_array(),
    )

    assert np.array_equal(operator.parameters, truth.theta_array())
    np.testing.assert_allclose(operator.gains, truth.gain_array(), rtol=0.0, atol=8e-15)
    np.testing.assert_allclose(operator.biases, truth.bias_array(), rtol=0.0, atol=8e-15)
    np.testing.assert_allclose(operator.forecast, expected_future, rtol=0.0, atol=2e-14)
    assert operator.forecast_sha256 == predictor.array_sha256(
        operator.forecast, role=f"{scenario}_forecast"
    )
    assert operator.forecast_sha256 != operator.history_reconstruction_sha256
    assert operator.history_reconstruction.shape == operator.forecast.shape == (3, 2304)
    _assert_hard_readonly(operator.history_reconstruction)
    _assert_hard_readonly(operator.forecast)

    if scenario in {"affine", "combined"}:
        history = cast(exact.GlobalResult, operator.history_fit)
        assert np.array_equal(operator.history_reconstruction, history.prediction)
        assert operator.history_reconstruction_sha256 == history.prediction_sha256


def test_fit_is_deterministic_and_independent_of_later_input_mutation(
    positive_results: dict[str, tuple[synthetic.SyntheticCase, predictor.SourcePairResult]],
) -> None:
    case, reference = positive_results["combined"]
    previous = case.source[0].copy()
    current = case.target[0].copy()
    replay = predictor.fit_source_pair(previous, current, config_sha256=CONFIG_SHA256)
    snapshots = {
        arm: replay.operator(cast(predictor.Arm, arm)).forecast.tobytes(order="C")
        for arm in ("affine", "appearance", "combined")
    }
    previous.fill(1_000.0)
    current.fill(-1_000.0)

    for arm in ("affine", "appearance", "combined"):
        replay_arm = replay.operator(cast(predictor.Arm, arm))
        reference_arm = reference.operator(cast(predictor.Arm, arm))
        assert replay_arm.forecast.tobytes(order="C") == snapshots[arm]
        assert replay_arm.forecast_sha256 == reference_arm.forecast_sha256
        assert replay_arm.history_reconstruction_sha256 == reference_arm.history_reconstruction_sha256
        assert np.array_equal(replay_arm.parameters, reference_arm.parameters)
        assert np.array_equal(replay_arm.gains, reference_arm.gains)
        assert np.array_equal(replay_arm.biases, reference_arm.biases)
        assert np.array_equal(replay_arm.forecast, reference_arm.forecast)
    assert replay.previous_sha256 == reference.previous_sha256
    assert replay.current_sha256 == reference.current_sha256


def test_source_baselines_are_exact_unclipped_and_role_hashed(
    positive_results: dict[str, tuple[synthetic.SyntheticCase, predictor.SourcePairResult]],
) -> None:
    case, result = positive_results["affine"]
    baselines = result.baselines
    previous = geometry.sample_scalar(case.source[0], 0, geometry.FULL_MASK)
    current = geometry.sample_scalar(case.target[0], 0, geometry.FULL_MASK)
    assert np.array_equal(baselines.persistence, current)
    assert np.array_equal(baselines.velocity, 2.0 * current - previous)
    assert baselines.persistence_sha256 != predictor.array_sha256(
        baselines.persistence, role="velocity_forecast"
    )
    predictor.validate_source_baselines(baselines, case.source[0], case.target[0])
    with pytest.raises(predictor.PredictorValidationError, match="hash differs"):
        replace(baselines, persistence_sha256="0" * 64)


def test_full_bias_only_control_is_current_only_constant_and_freshly_hashed(
    positive_results: dict[str, tuple[synthetic.SyntheticCase, predictor.SourcePairResult]],
) -> None:
    case, pair = positive_results["combined"]
    control = pair.bias_only
    standalone = predictor.fit_bias_only_control(
        case.target[0], config_sha256=CONFIG_SHA256
    )
    assert control.current_sha256 == standalone.current_sha256
    assert np.array_equal(control.history_reconstruction, standalone.history_reconstruction)
    assert np.array_equal(control.forecast, standalone.forecast)
    assert np.array_equal(
        control.forecast,
        np.broadcast_to(control.history_fit.biases[:, None], control.forecast.shape),
    )
    assert np.array_equal(control.history_reconstruction, control.forecast)
    assert control.history_reconstruction_sha256 != control.forecast_sha256
    assert control.forecast_sha256 == predictor.array_sha256(
        control.forecast, role="bias_only_forecast"
    )
    predictor.validate_bias_only_control(
        control, case.target[0], config_sha256=CONFIG_SHA256
    )
    _assert_hard_readonly(control.history_reconstruction)
    _assert_hard_readonly(control.forecast)


@pytest.fixture(scope="module")
def stationary_checkerboard() -> tuple[synthetic.SyntheticCase, predictor.CheckerboardHistoryResult]:
    case = synthetic.generate_case(
        "stationary", seed=EXPOSED_NONRESERVED_SEEDS["stationary"]
    )
    return (
        case,
        predictor.fit_checkerboard_history(
            case.source[0], case.target[0], config_sha256=CONFIG_SHA256
        ),
    )


def test_checkerboard_history_stitches_both_held_parities(
    stationary_checkerboard: tuple[
        synthetic.SyntheticCase, predictor.CheckerboardHistoryResult
    ],
) -> None:
    case, result = stationary_checkerboard
    expected = geometry.sample_scalar(case.target[0], 0, geometry.FULL_MASK)
    for arm in (result.affine, result.appearance, result.combined):
        assert len(arm.output_parity_fits) == 2
        for fit in arm.output_parity_fits:
            assert fit.prediction.shape == (3, 1152)
        np.testing.assert_allclose(
            arm.history_reconstruction, expected, rtol=0.0, atol=1.2e-14
        )
        assert arm.history_reconstruction_sha256 == predictor.array_sha256(
            arm.history_reconstruction,
            role=f"{arm.arm}_checkerboard_history_reconstruction",
        )
        _assert_hard_readonly(arm.history_reconstruction)

    bias = result.bias_only
    standalone = predictor.fit_checkerboard_bias_only_history(
        case.target[0], config_sha256=CONFIG_SHA256
    )
    assert np.array_equal(
        bias.history_reconstruction, standalone.history_reconstruction
    )
    for output_parity, history_fit in enumerate(bias.output_parity_fits):
        selected = geometry.PARITY_MASKS[output_parity]
        assert np.array_equal(
            bias.history_reconstruction[:, selected], history_fit.prediction
        )
        assert np.array_equal(
            history_fit.prediction,
            np.broadcast_to(history_fit.biases[:, None], history_fit.prediction.shape),
        )
    predictor.validate_checkerboard_bias_only_history(
        bias, case.target[0], config_sha256=CONFIG_SHA256
    )
    _assert_hard_readonly(bias.history_reconstruction)


def test_half_cycle_indices_are_stable_timestamp_ordered_and_within_video() -> None:
    video_ids = np.asarray(("b", "a", "b", "a", "b", "a", "b"))
    timestamps = np.asarray((3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0), dtype=np.float64)
    first = predictor.within_video_half_cycle_indices(video_ids, timestamps)
    second = predictor.within_video_half_cycle_indices(video_ids, timestamps)
    assert np.array_equal(first, second)
    assert np.array_equal(video_ids[first], video_ids)
    assert np.all(first != np.arange(len(first)))
    assert np.array_equal(np.sort(first), np.arange(len(first)))
    predictor.validate_half_cycle_indices(first, video_ids, timestamps)
    _assert_hard_readonly(first)

    with pytest.raises(predictor.PredictorValidationError, match="at least two"):
        predictor.within_video_half_cycle_indices(
            np.asarray(("a", "a", "b")), np.asarray((0.0, 1.0, 0.0))
        )


def test_strict_input_hash_and_deep_result_validators(
    positive_results: dict[str, tuple[synthetic.SyntheticCase, predictor.SourcePairResult]],
) -> None:
    case, result = positive_results["appearance"]
    predictor.validate_source_pair_result(
        result,
        case.source[0],
        case.target[0],
        config_sha256=CONFIG_SHA256,
    )
    with pytest.raises(predictor.PredictorValidationError, match="input hashes"):
        predictor.validate_source_pair_result(
            result,
            case.source[0] + 0.01,
            case.target[0],
            config_sha256=CONFIG_SHA256,
        )
    with pytest.raises(predictor.PredictorValidationError, match="float64"):
        predictor.validate_normalized_frame(case.source[0].astype(np.float32))
    with pytest.raises(predictor.PredictorValidationError, match="C-contiguous"):
        predictor.validate_normalized_frame(np.swapaxes(case.source[0], -2, -1))
    nonfinite = case.source[0].copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(predictor.PredictorValidationError, match="nonfinite"):
        predictor.validate_normalized_frame(nonfinite)
