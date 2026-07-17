"""Focused one-row integration tests for strict MM-008 v2.2 scoring."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from unittest.mock import patch

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

EXPOSED_SEED = 12_345
CONFIG_SHA256 = "3" * 64


@dataclass(frozen=True)
class _CompletedRun:
    case: synthetic.SyntheticCase
    result: scoring.RowScore
    grid_calls: tuple[tuple[np.ndarray, np.ndarray, tuple[exact.FitRequest, ...]], ...]


@pytest.fixture(scope="module")
def completed_run() -> _CompletedRun:
    case = synthetic.generate_case("translation", seed=EXPOSED_SEED)
    calls: list[tuple[np.ndarray, np.ndarray, tuple[exact.FitRequest, ...]]] = []
    original = exact.fit_global_contexts

    def recording_fit(
        source: np.ndarray,
        fit_mask: np.ndarray,
        output_mask: np.ndarray,
        requests: tuple[exact.FitRequest, ...],
        *,
        config_sha256: str,
    ) -> tuple[exact.GlobalResult, ...]:
        calls.append((fit_mask, output_mask, requests))
        return original(
            source,
            fit_mask,
            output_mask,
            requests,
            config_sha256=config_sha256,
        )

    with patch.object(scoring.exact, "fit_global_contexts", recording_fit):
        result = scoring.score_row(case, 0, config_sha256=CONFIG_SHA256)
    return _CompletedRun(case, result, tuple(calls))


@pytest.fixture(scope="module")
def transposed_run(completed_run: _CompletedRun) -> scoring.RowScore:
    return scoring.score_transposed_row(
        completed_run.case, 0, config_sha256=CONFIG_SHA256
    )


@pytest.fixture(scope="module")
def scoped_runs() -> tuple[scoring.RowScore, scoring.RowScore]:
    coupled = synthetic.generate_case("coupled_boundary", seed=EXPOSED_SEED)
    constant = synthetic.generate_case("constant_target", seed=EXPOSED_SEED)
    return (
        scoring.score_row(coupled, 0, config_sha256=CONFIG_SHA256),
        scoring.score_row(constant, 0, config_sha256=CONFIG_SHA256),
    )


def _context(arm: scoring.ArmScores, name: scoring.ContextName) -> scoring.ArmContextScore:
    return next(item for item in arm.contexts if item.plan.name == name)


def test_frozen_scenario_scope_has_the_exact_ordinary_coverage_census() -> None:
    expected_arms: dict[synthetic.Scenario, tuple[scoring.Arm, ...]] = {
        "translation": scoring.ARM_ORDER,
        "affine": scoring.ARM_ORDER,
        "appearance": scoring.ARM_ORDER,
        "combined": scoring.ARM_ORDER,
        "stationary": scoring.ARM_ORDER,
        "independent": scoring.ARM_ORDER,
        "coupled_boundary": ("affine", "combined"),
        "constant_target": ("appearance", "combined"),
    }
    assert dict(scoring.SCENARIO_ARMS) == expected_arms
    assert all(scoring.scenario_arms(scenario) == arms for scenario, arms in expected_arms.items())
    assert scoring.ORDINARY_COVERAGE == scoring.CoverageCounts(
        scenario_arm_banks=34,
        row_arms=204,
        fitted_contexts=1_428,
        grid_contexts=630,
        bias_contexts=336,
        persistence_records=48,
    )

    banks = sum(len(arms) for arms in expected_arms.values())
    grid_banks = sum(len(arms) for arms in scoring.SCENARIO_GRID_ARMS.values())
    assert (banks, grid_banks) == (34, 15)
    assert banks * 6 == 204
    assert banks * 6 * 7 == 1_428
    assert grid_banks * 6 * 7 == 630
    assert len(expected_arms) * 6 * 7 == 336
    assert len(expected_arms) * 6 == 48


def test_exact_context_order_three_source_streams_and_complete_counts(
    completed_run: _CompletedRun,
) -> None:
    result = completed_run.result
    assert result.input_orientation == "native"
    assert tuple(item.arm for item in result.arms) == scoring.ARM_ORDER
    assert tuple(item.plan.name for item in result.bias.contexts) == scoring.CONTEXT_ORDER
    assert all(
        tuple(item.plan.name for item in arm.contexts) == scoring.CONTEXT_ORDER
        for arm in result.arms
    )

    assert len(completed_run.grid_calls) == 3
    assert [len(requests) for _, _, requests in completed_run.grid_calls] == [2, 6, 6]
    full_fit, full_output, _ = completed_run.grid_calls[0]
    p0_fit, p0_output, _ = completed_run.grid_calls[1]
    p1_fit, p1_output, _ = completed_run.grid_calls[2]
    assert np.array_equal(full_fit, geometry.FULL_MASK)
    assert np.array_equal(full_output, geometry.FULL_MASK)
    assert np.array_equal(p0_fit, geometry.PARITY_MASKS[1])
    assert np.array_equal(p0_output, geometry.PARITY_MASKS[0])
    assert np.array_equal(p1_fit, geometry.PARITY_MASKS[0])
    assert np.array_equal(p1_output, geometry.PARITY_MASKS[1])

    assert tuple(item.stream for item in result.grid_streams) == ("full", "p0", "p1")
    assert all(item.grid_arms == scoring.GRID_ARMS for item in result.grid_streams)
    assert [len(item.consumer_keys) for item in result.grid_streams] == [2, 6, 6]
    assert len({item.source_grid.scope_sha256 for item in result.grid_streams}) == 3
    for stream in result.grid_streams:
        assert len(stream.source_grid.batch_records) == 22
        assert [len(batch.indices) for batch in stream.source_grid.batch_records] == [128] * 21 + [121]

    grid_contexts = tuple(
        context
        for arm in result.arms
        if arm.arm in {"affine", "combined"}
        for context in arm.contexts
    )
    assert len(grid_contexts) == 14
    for context in grid_contexts:
        estimate = context.estimate
        assert isinstance(estimate, exact.GlobalResult)
        stream_name = "full" if context.plan.name == "true_full" else context.plan.name[-2:]
        expected_grid = next(
            item.source_grid for item in result.grid_streams if item.stream == stream_name
        )
        assert estimate.source_grid is expected_grid

    assert result.persistence.count == 3 * 48 * 48
    assert all(record.count == 3 * 48 * 48 for record in (
        result.bias.true_full,
        result.bias.true_xfit,
        result.bias.near_xfit,
        result.bias.far_xfit,
    ))


def test_boundary_and_constant_scenarios_execute_only_their_scoped_arms(
    scoped_runs: tuple[scoring.RowScore, scoring.RowScore],
) -> None:
    coupled, constant = scoped_runs
    assert tuple(item.arm for item in coupled.arms) == ("affine", "combined")
    assert tuple(item.arm for item in constant.arms) == ("appearance", "combined")
    assert all(item.grid_arms == ("affine", "combined") for item in coupled.grid_streams)
    assert all(item.grid_arms == ("combined",) for item in constant.grid_streams)
    assert [len(item.consumer_keys) for item in coupled.grid_streams] == [2, 6, 6]
    assert [len(item.consumer_keys) for item in constant.grid_streams] == [1, 3, 3]
    assert len(coupled.dominance) == 2
    assert len(constant.dominance) == 2
    assert coupled.expectation_failures == ()
    assert constant.expectation_failures == ()


def test_every_prediction_is_scored_only_against_true_row_output_sites(
    completed_run: _CompletedRun,
) -> None:
    case = completed_run.case
    result = completed_run.result
    assert [(item.plan.target_kind, item.plan.target_row) for item in result.bias.contexts] == [
        ("true", 0),
        ("true", 0),
        ("true", 0),
        ("near", 1),
        ("near", 1),
        ("far", 3),
        ("far", 3),
    ]

    affine = result.arm("affine")
    near_p0 = _context(affine, "near_p0")
    true_output = fitting.target_values(case.target[0], geometry.PARITY_MASKS[0])
    wrong_output = fitting.target_values(case.target[1], geometry.PARITY_MASKS[0])
    prediction = near_p0.estimate.prediction
    assert isinstance(near_p0.estimate, exact.GlobalResult)
    assert near_p0.error == calibration.error_record(prediction, true_output)
    assert near_p0.error != calibration.error_record(prediction, wrong_output)

    assembled = np.empty((3, geometry.SITE_COUNT), dtype=np.float64)
    assembled[:, geometry.PARITY_MASKS[0]] = _context(affine, "true_p0").estimate.prediction
    assembled[:, geometry.PARITY_MASKS[1]] = _context(affine, "true_p1").estimate.prediction
    true_full = fitting.target_values(case.target[0], geometry.FULL_MASK)
    assert affine.true_xfit == calibration.error_record(assembled, true_full)
    assert _context(affine, "true_full").error == affine.true_full
    assert _context(affine, "true_full").error.count == 6_912
    assert all(context.error.count == 3_456 for context in affine.contexts[1:])


def test_endpoints_combined_carries_and_singleton_expectations_pass(
    completed_run: _CompletedRun,
) -> None:
    result = completed_run.result
    truth_index = geometry.state_index((4.0, -4.0, 0.0, 0.0, 0.0, 0.0))
    for arm_name in ("affine", "combined"):
        arm = result.arm(arm_name)
        assert arm.endpoints_pass
        assert arm.predicates.pair
        assert arm.predicates.performance
        assert arm.predicates.complete
        assert arm.predicates.strong
        for name in ("true_full", "true_p0", "true_p1"):
            estimate = _context(arm, name).estimate
            assert isinstance(estimate, exact.GlobalResult)
            assert estimate.selected.state_index == truth_index
            assert estimate.certificate.scalar_replay_bit_exact

    combined = result.arm("combined")
    appearance = result.arm("appearance")
    assert combined.carries is not None
    for combined_context, appearance_context in zip(
        combined.contexts, appearance.contexts, strict=True
    ):
        carry = combined_context.carries
        assert carry is not None
        estimate = combined_context.estimate
        assert isinstance(estimate, exact.GlobalResult)
        assert carry.owner_scope_sha256 == estimate.objective_cache.scope_sha256
        assert np.array_equal(carry.appearance_prediction, appearance_context.estimate.prediction)
        assert tuple(name for name, _ in carry.hashes) == tuple(
            sorted(name for name, _ in carry.hashes)
        )
        assert not carry.affine_prediction.flags.writeable
        with pytest.raises(ValueError):
            carry.affine_prediction.setflags(write=True)

    assert result.arm("global_translation").predicates.strong
    assert result.arm("quadrant_translation").predicates.strong
    assert result.dominates("global_translation", "appearance")
    assert result.expectations_pass
    assert result.expectation_failures == ()


def test_score_api_has_no_seed_subset_cache_or_scoring_target_seam(
    completed_run: _CompletedRun,
) -> None:
    signature = inspect.signature(scoring.score_row)
    assert tuple(signature.parameters) == ("case", "row", "config_sha256")
    assert signature.parameters["config_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    transpose_signature = inspect.signature(scoring.score_transposed_row)
    assert tuple(transpose_signature.parameters) == (
        "original_case",
        "row",
        "config_sha256",
    )
    assert (
        transpose_signature.parameters["config_sha256"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )

    with pytest.raises(scoring.ScoringV22Error, match="config SHA"):
        scoring.score_row(completed_run.case, 0, config_sha256="invalid")
    with pytest.raises(scoring.ScoringV22Error, match="row must be an integer"):
        scoring.score_row(completed_run.case, True, config_sha256=CONFIG_SHA256)

    forged_target = completed_run.case.target.copy()
    forged_target[0, 0, 0, 0] += 1.0
    forged = replace(completed_run.case, target=forged_target)
    with patch.object(
        scoring.exact,
        "fit_global_contexts",
        side_effect=AssertionError("fit must not run before case validation"),
    ) as grid_fit:
        with pytest.raises(scoring.ScoringV22Error, match="replay validation"):
            scoring.score_row(forged, 0, config_sha256=CONFIG_SHA256)
        grid_fit.assert_not_called()

    with patch.object(
        scoring.synthetic,
        "transpose_case",
        side_effect=AssertionError("transpose must not run before original validation"),
    ) as transpose:
        with pytest.raises(scoring.ScoringV22Error, match="replay validation"):
            scoring.score_transposed_row(forged, 0, config_sha256=CONFIG_SHA256)
        transpose.assert_not_called()


def test_trusted_transpose_path_has_metamorphic_truth_and_distinct_scopes(
    completed_run: _CompletedRun,
    transposed_run: scoring.RowScore,
) -> None:
    native = completed_run.result
    transformed = transposed_run
    assert transformed.input_orientation == "transposed"
    assert transformed.expectation_failures == ()
    assert tuple(item.arm for item in transformed.arms) == scoring.ARM_ORDER

    for arm_name in ("affine", "combined"):
        native_arm = native.arm(arm_name)
        transformed_arm = transformed.arm(arm_name)
        for context_name in ("true_full", "true_p0", "true_p1"):
            native_result = _context(native_arm, context_name).estimate
            transformed_result = _context(transformed_arm, context_name).estimate
            assert isinstance(native_result, exact.GlobalResult)
            assert isinstance(transformed_result, exact.GlobalResult)
            assert tuple(transformed_result.selected.parameters) == synthetic.transpose_theta(
                tuple(float(value) for value in native_result.selected.parameters)
            )
            assert native_result.context_key != transformed_result.context_key

    native_prediction = _context(native.arm("affine"), "true_full").estimate.prediction
    transformed_prediction = _context(
        transformed.arm("affine"), "true_full"
    ).estimate.prediction
    expected_prediction = np.swapaxes(
        native_prediction.reshape(3, 48, 48), -2, -1
    ).reshape(3, -1)
    np.testing.assert_allclose(transformed_prediction, expected_prediction, rtol=0.0, atol=1e-12)
    assert {
        item.source_grid.scope_sha256 for item in native.grid_streams
    }.isdisjoint(item.source_grid.scope_sha256 for item in transformed.grid_streams)

    with pytest.raises(scoring.ScoringV22Error, match="rejects native"):
        scoring.validate_transposed_row_score(native, completed_run.case)


def test_deep_row_score_replay_rejects_nested_evidence_forgery(
    completed_run: _CompletedRun,
) -> None:
    last = completed_run.result.expectations[-1]
    forged_expectations = (
        *completed_run.result.expectations[:-1],
        replace(last, passed=not last.passed),
    )
    forged = replace(completed_run.result, expectations=forged_expectations)
    with pytest.raises(scoring.ScoringV22Error, match="bit-exact replay"):
        scoring.validate_row_score(forged, completed_run.case)
