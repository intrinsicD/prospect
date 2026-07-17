"""Development-only tests for the pure MM-008 v2 optimizer and comparator core.

The only fresh entropy in this file is the declared development seed 12345.  The
older affine fixture is an already exposed v1 regression case; no v2 confirmation,
auditor-nonce, lifecycle, or real-data path is exercised here.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from typing import Literal, cast

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import method as v1
from bench.multimodal_mechanism_diagnostics import method_v2 as v2


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def wall_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(12_345))
    source = rng.normal(size=(3, 64, 64))
    source *= np.arange(64)[None, :, None] >= 32
    truth = np.asarray([4.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    target = source.copy()
    mask = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    coords = v1.GEOMETRY.coords.astype(int)
    target[:, coords[:, 0], coords[:, 1]] = v1._sample_affine(source[None], truth[None], mask)[0]
    return source, target, truth


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def wall_full(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> v2.Estimate:
    source, target, _ = wall_case
    return v2.estimate_full(source, target, "affine")


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def wrong_null_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(12_345))
    source = rng.normal(size=(3, 64, 64))
    wrong_source = rng.normal(size=(3, 64, 64))
    truth = np.asarray([-4.0, 4.0, 0.0, 2.0, -2.0, 0.0])
    mask = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    coords = v1.GEOMETRY.coords.astype(int)

    def transformed(base: np.ndarray) -> np.ndarray:
        target = base.copy()
        target[:, coords[:, 0], coords[:, 1]] = v1._sample_affine(base[None], truth[None], mask)[0]
        return target

    return source, transformed(wrong_source), transformed(source)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def wrong_null_estimate(
    wrong_null_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> v2.Estimate:
    source, wrong_target, _ = wrong_null_case
    return v2.estimate_null_xfit(source, wrong_target, "affine")


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def combined_dev_full(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> v2.Estimate:
    source, _, truth = wall_case
    gains = np.asarray([1.2, 0.8, 1.4])
    biases = np.asarray([0.1, -0.2, 0.3])
    mask = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    sampled = v1._sample_affine(source[None], truth[None], mask)[0]
    target = source.copy()
    coords = v1.GEOMETRY.coords.astype(int)
    target[:, coords[:, 0], coords[:, 1]] = gains[:, None] * sampled + biases[:, None]
    return v2.estimate_full(source, target, "combined")


def frozen_float64(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def test_protocol_grid_order_and_exact_uv_membership() -> None:
    assert v2.PROTOCOL_SHA256 == ("6bd9f35d13a36394ea2a17cdd951a0ea0adf0365909228e73671cc9484c19b5f")
    assert len(v2.CANONICAL_STATES) == v2.STATE_COUNT == 15_625
    assert v2.CANONICAL_GRID.dtype == np.dtype("<f8")
    assert v2.CANONICAL_GRID.flags.c_contiguous
    assert v2.CANDIDATE_ORDER_SHA256 == hashlib.sha256(v2.CANONICAL_GRID.tobytes(order="C")).hexdigest()
    keys = [state.canonical_key for state in v2.CANONICAL_STATES]
    assert keys == sorted(keys)

    current = (4.0, -4.0, 2.0, -2.0, 4.0, -4.0)
    indices = v2.neighborhood_indices(v2.state_index(np.asarray(current)), "UV")
    actual = {v2.CANONICAL_STATES[index].values for index in indices}
    expected = {(4.0, -4.0, u[0], w[0], u[1], w[1]) for u in v2.U_BLOCK for w in v2.V_BLOCK}
    assert len(indices) == len(set(indices)) == 625
    assert indices == tuple(sorted(indices))
    assert actual == expected


def test_order_agreement_anchors_tolerance_to_selected_large_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def tolerance(value: float) -> float:
        calls.append(value)
        return 2.0e12

    monkeypatch.setattr(v2, "optimizer_tolerance", tolerance)
    assert v2.objective_order_agrees(1.0e12, 1.0e6, 1.0e6)
    assert calls == [1.0e6]


def test_disagreeing_wrong_context_accepts_only_null_certification(
    wrong_null_estimate: v2.Estimate,
) -> None:
    assert wrong_null_estimate.certification_mode == "null"
    assert wrong_null_estimate.certified
    assert not any(direction.optimizer.certified for direction in wrong_null_estimate.directions)
    for direction in wrong_null_estimate.directions:
        optimizer = direction.optimizer
        assert optimizer.forward.certified
        assert optimizer.reverse.certified
        assert optimizer.null_certified
        assert not optimizer.objective_agreement
        assert not optimizer.prediction_agreement
        assert not optimizer.flow_agreement
        optimizer.validate(direction.context, certification_mode="null")
        with pytest.raises(v2.V2ValidationError, match="claim context is not certified"):
            optimizer.validate(direction.context, certification_mode="claim")


def test_optimizer_exposes_immutable_f_r_endpoints_and_fit_only_s_selection(
    wrong_null_estimate: v2.Estimate,
) -> None:
    for direction in wrong_null_estimate.directions:
        optimizer = direction.optimizer
        assert optimizer.forward_evaluation.state_index == optimizer.forward.endpoint_state_index
        assert optimizer.reverse_evaluation.state_index == optimizer.reverse.endpoint_state_index
        assert optimizer.forward_evaluation.objective == optimizer.forward.endpoint_objective
        assert optimizer.reverse_evaluation.objective == optimizer.reverse.endpoint_objective
        assert not optimizer.forward_prediction.flags.writeable
        assert not optimizer.reverse_prediction.flags.writeable
        candidates = {
            "F": optimizer.forward_evaluation,
            "R": optimizer.reverse_evaluation,
        }
        expected_label = min(
            ("F", "R"),
            key=lambda label: (
                candidates[label].selection_objective,
                candidates[label].state.canonical_key,
            ),
        )
        assert optimizer.selected_start == expected_label
        assert optimizer.selected_evaluation is candidates[expected_label]
        assert np.array_equal(
            optimizer.selected_prediction,
            optimizer.forward_prediction if expected_label == "F" else optimizer.reverse_prediction,
        )
        with pytest.raises(v2.V2ValidationError, match="unknown Q"):
            optimizer.evaluation_for(cast(v2.QLabel, "X"))
        with pytest.raises(v2.V2ValidationError, match="unknown Q"):
            optimizer.prediction_for(cast(v2.QLabel, "X"))


def test_public_q_panels_and_order_label_swap_preserve_minimum_decision(
    wrong_null_case: tuple[np.ndarray, np.ndarray, np.ndarray],
    wrong_null_estimate: v2.Estimate,
) -> None:
    _, _, scoring_target = wrong_null_case
    panels = {panel.label: panel.prediction for panel in wrong_null_estimate.q_panels()}
    assert tuple(panels) == v2.Q_LABEL_ORDER
    assert np.array_equal(panels["S"], wrong_null_estimate.prediction)
    for label in v2.Q_LABEL_ORDER:
        expected = np.full_like(panels[label], np.nan)
        for direction in wrong_null_estimate.directions:
            expected[:, direction.context.output_mask] = direction.prediction_for(label)
        assert np.array_equal(panels[label], expected)
        assert not panels[label].flags.writeable

    mask = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    scoring_values = v1._target_values(scoring_target[None], mask)[0]
    scores: dict[v2.QLabel, float] = {
        label: float(np.mean((panels[label] - scoring_values) ** 2, dtype=np.float64)) for label in v2.Q_LABEL_ORDER
    }
    selected = v2.select_q_minimum(scores)
    swapped: dict[v2.QLabel, float] = {
        "S": scores["S"],
        "F": scores["R"],
        "R": scores["F"],
    }
    swapped_selected = v2.select_q_minimum(swapped)
    assert selected.label == "F"
    assert scores["F"] < scores["S"]
    assert selected.value == swapped_selected.value
    assert (1.25 * selected.value <= 1.0) == (1.25 * swapped_selected.value <= 1.0)


def test_q_minimum_uses_exact_frozen_s_f_r_tie_order() -> None:
    assert v2.select_q_minimum({"S": 1.0, "F": 1.0, "R": 1.0}) == v2.QMinimum("S", 1.0)
    assert v2.select_q_minimum({"S": 2.0, "F": 1.0, "R": 1.0}) == v2.QMinimum("F", 1.0)
    assert v2.select_q_minimum({"S": 2.0, "F": 2.0, "R": 1.0}) == v2.QMinimum("R", 1.0)


def test_fit_context_owns_readonly_copies_and_excludes_held_target(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source, target, _ = wall_case
    caller_source = source.copy()
    caller_target = target.copy()
    context = v2.make_xfit_context(caller_source, caller_target, "affine", output_parity=0)
    original_scope = context.scope_sha256
    caller_source += 10_000.0
    output_coords = v1.GEOMETRY.coords[context.output_mask].astype(int)
    caller_target[:, output_coords[:, 0], output_coords[:, 1]] -= 10_000.0
    assert context.scope_sha256 == original_scope
    assert caller_source.flags.writeable and caller_target.flags.writeable
    assert not context.source.flags.writeable
    assert not context.fit_target.flags.writeable
    with pytest.raises(ValueError):
        context.source[0, 0, 0] = 0.0

    held_mutation = target.copy()
    held_mutation[:, output_coords[:, 0], output_coords[:, 1]] += 1_000.0
    mutated_context = v2.make_xfit_context(source, held_mutation, "affine", output_parity=0)
    assert context.scope_sha256 == mutated_context.scope_sha256
    assert np.array_equal(context.fit_target, mutated_context.fit_target)


def test_cache_scope_content_and_request_hashes_are_binary_exact(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source, target, _ = wall_case
    context = v2.make_full_context(source, target, "affine")
    manual_scope = hashlib.sha256()
    manual_scope.update(v2.CACHE_SCOPE_TAG)
    manual_scope.update(context.source.tobytes(order="C"))
    manual_scope.update(context.fit_target.tobytes(order="C"))
    manual_scope.update(np.asarray(context.fit_mask, dtype=np.uint8).tobytes(order="C"))
    manual_scope.update(np.asarray(context.output_mask, dtype=np.uint8).tobytes(order="C"))
    manual_scope.update(b"affine")
    manual_scope.update(bytes.fromhex(v2.CONFIG_SHA256))
    assert context.scope_sha256 == manual_scope.hexdigest()

    invalid_index = v2.state_index(np.asarray([8.0, 0.0, 4.0, 0.0, 0.0, 0.0]))
    cache = v2.ObjectiveCache(context)
    invalid = cache.evaluate(invalid_index, start="F")
    valid = cache.evaluate(v2.ZERO_STATE_INDEX, start="F")
    assert invalid.status == "inadmissible" and invalid.objective is None
    assert valid.status == "valid" and valid.objective is not None

    content = hashlib.sha256()
    content.update(v2.CACHE_CONTENT_TAG)
    content.update(bytes.fromhex(context.scope_sha256))
    for index, evaluation in sorted(cache.entries.items()):
        content.update(struct.pack("<H", index))
        if evaluation.status == "inadmissible":
            content.update(b"\x00")
        else:
            assert evaluation.objective is not None
            content.update(b"\x01")
            content.update(struct.pack("<d", evaluation.objective))
    assert cache.content_sha256() == content.hexdigest()

    requests = hashlib.sha256()
    requests.update(v2.CACHE_REQUEST_TAG)
    requests.update(bytes.fromhex(context.scope_sha256))
    requests.update(b"F")
    requests.update(struct.pack("<H", invalid_index))
    requests.update(struct.pack("<H", v2.ZERO_STATE_INDEX))
    assert cache.request_sha256("F") == requests.hexdigest()


def test_combined_cache_serializes_projection_fields_exactly() -> None:
    case = v1.synthetic_case("combined")
    context = v2.make_full_context(case.source[0], case.target[0], "combined")
    truth_index = v2.state_index(case.parameters[0])
    cache = v2.ObjectiveCache(context)
    evaluation = cache.evaluate(truth_index)
    assert evaluation.objective is not None
    assert evaluation.gains is not None and evaluation.biases is not None
    assert evaluation.retained_macros == tuple(sorted(evaluation.retained_macros))

    expected = hashlib.sha256()
    expected.update(v2.CACHE_CONTENT_TAG)
    expected.update(bytes.fromhex(context.scope_sha256))
    expected.update(struct.pack("<H", truth_index))
    expected.update(b"\x01")
    expected.update(struct.pack("<d", evaluation.objective))
    expected.update(np.asarray(evaluation.gains, dtype="<f8").tobytes(order="C"))
    expected.update(np.asarray(evaluation.biases, dtype="<f8").tobytes(order="C"))
    expected.update(struct.pack("<B", len(evaluation.retained_macros)))
    expected.update(bytes(evaluation.retained_macros))
    assert cache.content_sha256() == expected.hexdigest()


def test_all_pairs_search_repairs_the_exposed_v1_affine_trap() -> None:
    case = v1.synthetic_case("affine")
    old = v1.estimate_full(case.source[:1], case.target[:1], "affine")
    assert np.array_equal(old.parameters[0], [0.0, 0.0, 2.0, -2.0, 0.0, 0.0])
    assert not old.probe_strict_improvement[0]

    repaired = v2.estimate_full(case.source[0], case.target[0], "affine")
    truth = np.asarray([0.0, 0.0, 2.0, 0.0, 0.0, -2.0])
    assert repaired.certified
    assert np.array_equal(repaired.parameters[0], truth)
    assert repaired.objectives[0] <= 1e-12
    trace = repaired.directions[0].optimizer
    assert trace.forward.endpoint_state_index == v2.state_index(truth)
    assert trace.reverse.endpoint_state_index == v2.state_index(truth)
    assert tuple(record.block for record in trace.forward.terminal_neighborhoods) == (
        "T",
        "U",
        "V",
        "TU",
        "TV",
        "UV",
    )
    assert all(
        record.requested_indices == tuple(sorted(record.requested_indices))
        for record in trace.forward.terminal_neighborhoods
    )


def test_masked_admissibility_wall_recovers_full_and_xfit_exactly(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray], wall_full: v2.Estimate
) -> None:
    source, target, truth = wall_case
    xfit = v2.estimate_xfit(source, target, "affine")
    assert wall_full.certified and xfit.certified
    assert np.array_equal(wall_full.parameters, truth[None])
    assert np.array_equal(xfit.parameters, np.stack((truth, truth)))
    assert np.array_equal(wall_full.objectives, [0.0])
    assert np.array_equal(xfit.objectives, [0.0, 0.0])
    assert np.all(np.isfinite(xfit.prediction))
    assert bool(v1._admissible(truth[None])[0])
    beyond_positive = truth.copy()
    beyond_positive[0] = np.nextafter(8.0, np.inf)
    beyond_negative = np.zeros(6)
    beyond_negative[0] = np.nextafter(-8.0, -np.inf)
    exact_edges = np.asarray([[8.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-8.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert np.array_equal(v1._admissible(exact_edges), [True, True])
    assert not bool(v1._admissible(beyond_positive[None])[0])
    assert not bool(v1._admissible(beyond_negative[None])[0])


def test_held_target_mutation_preserves_fit_history_prediction_and_bias(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source, target, _ = wall_case
    before_context = v2.make_xfit_context(source, target, "affine", output_parity=0)
    output_coords = v1.GEOMETRY.coords[before_context.output_mask].astype(int)
    mutated = target.copy()
    mutated[:, output_coords[:, 0], output_coords[:, 1]] += 123.0
    after_context = v2.make_xfit_context(source, mutated, "affine", output_parity=0)
    before = v2.fit_direction(before_context)
    after = v2.fit_direction(after_context)
    assert np.array_equal(before.parameters, after.parameters)
    assert np.array_equal(before.prediction, after.prediction)
    assert before.retained_macros == after.retained_macros
    assert before.optimizer.forward.accepted_states == after.optimizer.forward.accepted_states
    assert before.optimizer.forward.request_sha256 == after.optimizer.forward.request_sha256
    before_bias = v2.fit_bias_only(before_context.fit_target, before_context.fit_mask, before_context.output_mask)
    after_bias = v2.fit_bias_only(after_context.fit_target, after_context.fit_mask, after_context.output_mask)
    assert before_bias.retained_macros == after_bias.retained_macros
    assert np.array_equal(before_bias.biases, after_bias.biases)
    assert np.array_equal(before_bias.prediction, after_bias.prediction)


def test_combined_estimate_materializes_both_correct_carries(
    combined_dev_full: v2.Estimate,
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, _, truth = wall_case
    estimate = combined_dev_full
    direction = estimate.directions[0]
    assert estimate.certified
    assert direction.affine_carry is not None
    assert direction.appearance_carry is not None
    assert np.array_equal(direction.parameters, truth)
    assert np.allclose(direction.gains, [1.2, 0.8, 1.4], rtol=0.0, atol=1e-12)
    assert np.allclose(direction.biases, [0.1, -0.2, 0.3], rtol=0.0, atol=1e-12)
    assert np.array_equal(direction.affine_carry.parameters, direction.parameters)
    assert np.array_equal(direction.affine_carry.gains, np.ones(3))
    assert np.array_equal(direction.affine_carry.biases, np.zeros(3))
    assert np.array_equal(direction.appearance_carry.parameters, np.zeros(6))
    for evaluation in (
        direction.optimizer.forward_evaluation,
        direction.optimizer.reverse_evaluation,
    ):
        assert evaluation.gains is not None and evaluation.biases is not None
        assert not evaluation.gains.flags.writeable
        assert not evaluation.biases.flags.writeable
        assert evaluation.retained_macros == tuple(sorted(evaluation.retained_macros))


def test_bias_only_uses_pass_one_retention_and_no_third_trim() -> None:
    channel_values = np.asarray([1.25, -0.75, 3.5])
    target = np.broadcast_to(channel_values[:, None, None], (3, 64, 64)).copy()
    full = v2.estimate_bias_full(target)
    xfit = v2.estimate_bias_xfit(target)
    assert np.array_equal(full.biases, channel_values[None])
    assert np.array_equal(xfit.biases, np.stack((channel_values, channel_values)))
    assert np.array_equal(full.objectives, [0.0])
    assert np.array_equal(xfit.objectives, [0.0, 0.0])
    assert full.directions[0].retained_macros == tuple(range(27))
    assert len(xfit.directions[0].retained_macros) == 14
    assert len(xfit.directions[1].retained_macros) == 14
    assert np.array_equal(full.prediction[:, 0], channel_values)


def test_exhaustive_oracle_confirms_the_masked_wall_global_endpoint(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray], wall_full: v2.Estimate
) -> None:
    source, target, truth = wall_case
    context = v2.make_full_context(source, target, "affine")
    oracle = v2.exhaustive_oracle(context, truth, optimizer=wall_full.directions[0].optimizer)
    assert oracle.candidate_count == 15_625
    assert 0 < oracle.admissible_count < oracle.candidate_count
    assert oracle.selected_state_index == v2.state_index(truth)
    assert oracle.minimum_objective == oracle.truth_objective == 0.0
    assert oracle.second_best_nonequivalent_gap > 1e-12
    assert len(oracle.cache_content_sha256) == 64


def test_validation_seams_fail_closed_on_context_and_result_drift(
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray], wall_full: v2.Estimate
) -> None:
    source, target, _ = wall_case
    good = v2.make_full_context(source, target, "affine")
    bypassed = v2.FitContext(
        source.copy(),
        good.fit_target.copy(),
        good.fit_mask.copy(),
        good.output_mask.copy(),
        "affine",
        v2.CONFIG_SHA256,
    )
    with pytest.raises(v2.V2ValidationError, match="read-only"):
        v2.fit_optimizer(bypassed)
    drifted = replace(
        wall_full.directions[0].optimizer,
        protocol_sha256="0" * 64,
    )
    with pytest.raises(v2.V2ValidationError, match="protocol hash"):
        drifted.validate(good)


def test_null_endpoint_prediction_drift_fails_closed(
    wrong_null_estimate: v2.Estimate,
) -> None:
    direction = wrong_null_estimate.directions[0]
    drifted = replace(
        direction.optimizer,
        forward_prediction=direction.optimizer.forward_prediction.copy(),
    )
    with pytest.raises(v2.V2ValidationError, match="immutable"):
        drifted.validate(direction.context, certification_mode="null")


def test_public_optimizer_replay_rejects_trace_request_cache_and_label_forges(
    wall_full: v2.Estimate,
) -> None:
    direction = wall_full.directions[0]
    optimizer = direction.optimizer
    terminal = optimizer.forward.terminal_neighborhoods[0]
    forged_terminal = replace(
        optimizer.forward,
        terminal_neighborhoods=(
            replace(terminal, best_improvement=terminal.best_improvement + 1.0),
            *optimizer.forward.terminal_neighborhoods[1:],
        ),
    )
    forged_history = replace(
        optimizer.forward,
        objective_history=(
            optimizer.forward.objective_history[0] + 1.0,
            *optimizer.forward.objective_history[1:],
        ),
    )
    forged_request = replace(optimizer.forward, request_sha256="0" * 64)
    forged_cache = replace(optimizer.cache, content_sha256="0" * 64)
    relabeled_forward = replace(optimizer.forward, start="R")
    relabeled_reverse = replace(optimizer.reverse, start="F")
    for forged in (
        replace(optimizer, forward=forged_terminal),
        replace(optimizer, forward=forged_history),
        replace(optimizer, forward=forged_request),
        replace(optimizer, cache=forged_cache),
        replace(
            optimizer,
            forward=relabeled_forward,
            reverse=relabeled_reverse,
        ),
    ):
        with pytest.raises(v2.V2ValidationError, match="deterministic replay"):
            forged.validate(direction.context, require_certified=False)


def test_public_optimizer_replay_rejects_combined_endpoint_field_forge(
    combined_dev_full: v2.Estimate,
) -> None:
    direction = combined_dev_full.directions[0]
    optimizer = direction.optimizer
    evaluation = optimizer.forward_evaluation
    assert evaluation.objective is not None
    assert evaluation.gains is not None
    forged_gains = frozen_float64(evaluation.gains + np.asarray([0.25, 0.0, 0.0]))
    forged_evaluation = replace(
        evaluation,
        objective=evaluation.objective + 1.0,
        gains=forged_gains,
        retained_macros=evaluation.retained_macros[1:],
    )
    forged = replace(optimizer, forward_evaluation=forged_evaluation)
    with pytest.raises(v2.V2ValidationError, match="deterministic replay"):
        forged.validate(direction.context, require_certified=False)


def test_f_r_predictions_reconstruct_directly_from_public_endpoint_fields(
    wrong_null_estimate: v2.Estimate,
    combined_dev_full: v2.Estimate,
) -> None:
    directions = (*wrong_null_estimate.directions, combined_dev_full.directions[0])
    for direction in directions:
        for label in cast(tuple[v2.QLabel, v2.QLabel], ("F", "R")):
            evaluation = direction.evaluation_for(label)
            sampled = v1._sample_affine(
                direction.context.source[None],
                evaluation.state.array()[None],
                direction.context.output_mask,
            )[0]
            if direction.context.arm == "combined":
                assert evaluation.gains is not None and evaluation.biases is not None
                expected = evaluation.gains[:, None] * sampled + evaluation.biases[:, None]
            else:
                expected = sampled
            assert np.array_equal(direction.prediction_for(label), expected)


def test_estimate_validate_checks_aggregates_masks_roles_and_deep_replay(
    wall_full: v2.Estimate,
    wrong_null_estimate: v2.Estimate,
) -> None:
    wall_full.validate()
    drifted_objectives = frozen_float64(wall_full.objectives + 1.0)
    with pytest.raises(v2.V2ValidationError, match="aggregates"):
        replace(wall_full, objectives=drifted_objectives).validate()
    with pytest.raises(v2.V2ValidationError, match="masks"):
        replace(
            wrong_null_estimate,
            directions=cast(
                tuple[v2.DirectionEstimate, ...],
                tuple(reversed(wrong_null_estimate.directions)),
            ),
        ).validate(require_certified=False)
    with pytest.raises(v2.V2ValidationError, match="certification role"):
        replace(wrong_null_estimate, certification_mode="claim").validate(require_certified=False)
    with pytest.raises(v2.V2ValidationError, match="invalid mode"):
        replace(wall_full, mode=cast(Literal["full", "xfit"], "broken")).validate()


def test_normal_fit_uses_one_private_build_and_never_public_deep_validation(
    monkeypatch: pytest.MonkeyPatch,
    wall_case: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source, target, _ = wall_case
    context = v2.make_full_context(source, target, "affine")
    original_build = v2._build_optimizer_result
    build_calls = 0

    def counted_build(
        current: v2.FitContext,
    ) -> tuple[v2.OptimizerResult, v2.ObjectiveCache]:
        nonlocal build_calls
        build_calls += 1
        return original_build(current)

    def forbidden_public_validation(
        self: v2.OptimizerResult,
        current: v2.FitContext,
        *,
        require_certified: bool = True,
        certification_mode: v2.CertificationMode = "claim",
    ) -> None:
        del self, current, require_certified, certification_mode
        raise AssertionError("ordinary fit called public deep replay")

    monkeypatch.setattr(v2, "_build_optimizer_result", counted_build)
    monkeypatch.setattr(v2.OptimizerResult, "validate", forbidden_public_validation)
    result = v2.fit_optimizer(context)
    assert result.certified
    assert build_calls == 1
