from __future__ import annotations

import math
from fractions import Fraction

import pytest

import bench.active_acquisition.problem as problem_module
from bench.active_acquisition import (
    ACQUISITION_ACTIONS,
    NONCANDIDATE_SIGN_INVARIANCE_ACTIONS,
    NUISANCE_SCAN,
    OVERPOWERED_NEGATIVE,
    OVERPOWERED_POSITIVE,
    SKIP,
    STRONG_NEGATIVE,
    STRONG_POSITIVE,
    WEAK_NEGATIVE,
    WEAK_POSITIVE,
    ActuatorRegime,
    FractionOracle,
    HiddenActuatorProblem,
    TerminalAction,
)


def test_signed_pulse_likelihoods_follow_the_hidden_actuator_regime() -> None:
    problem = HiddenActuatorProblem()

    assert problem.outcomes(STRONG_POSITIVE) == (-1, 1)
    for observed, expected in zip(
        problem.likelihoods(STRONG_POSITIVE),
        ((0.9, 0.1), (0.1, 0.9)),
        strict=True,
    ):
        assert observed == pytest.approx(expected)
    for observed, expected in zip(
        problem.likelihoods(STRONG_NEGATIVE),
        ((0.1, 0.9), (0.9, 0.1)),
        strict=True,
    ):
        assert observed == pytest.approx(expected)
    assert problem.posterior_direct(STRONG_POSITIVE, 1) == pytest.approx(0.9)
    assert problem.posterior_direct(STRONG_POSITIVE, -1) == pytest.approx(0.1)
    assert problem.posterior_direct(STRONG_NEGATIVE, 1) == pytest.approx(0.1)
    assert problem.posterior_direct(STRONG_NEGATIVE, -1) == pytest.approx(0.9)


def test_pulse_paths_fail_closed_when_declared_accuracy_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = HiddenActuatorProblem()
    monkeypatch.setattr(
        problem_module,
        "_PULSE_ACCURACY",
        {kind: value for kind, value in problem_module._PULSE_ACCURACY.items() if kind is not STRONG_POSITIVE.kind},
    )

    with pytest.raises(ValueError, match="likelihood requires a declared accuracy"):
        problem.likelihoods(STRONG_POSITIVE)
    with pytest.raises(ValueError, match="realization requires a declared accuracy"):
        problem.realize_acquisition(STRONG_POSITIVE, ActuatorRegime.DIRECT, unit_interval=0.5)


def test_policy_candidate_set_matches_the_five_action_protocol() -> None:
    assert ACQUISITION_ACTIONS == (
        SKIP,
        WEAK_POSITIVE,
        STRONG_POSITIVE,
        OVERPOWERED_POSITIVE,
        NUISANCE_SCAN,
    )
    assert tuple(action.action_id for action in ACQUISITION_ACTIONS) == (
        "skip",
        "weak",
        "strong",
        "overpowered",
        "nuisance",
    )
    assert NONCANDIDATE_SIGN_INVARIANCE_ACTIONS == (
        WEAK_NEGATIVE,
        STRONG_NEGATIVE,
        OVERPOWERED_NEGATIVE,
    )
    assert all(
        action not in ACQUISITION_ACTIONS and action.action_id.endswith(":sign_inverted_non_candidate")
        for action in NONCANDIDATE_SIGN_INVARIANCE_ACTIONS
    )


def test_hand_calculated_action_values_separate_information_from_cost() -> None:
    problem = HiddenActuatorProblem()
    expected = {
        SKIP: (0.00, 0.50),
        WEAK_POSITIVE: (0.13, 0.63),
        STRONG_POSITIVE: (0.24, 0.74),
        OVERPOWERED_POSITIVE: (-0.05, 0.45),
        NUISANCE_SCAN: (-0.01, 0.49),
    }

    for action, (net_increment, episode_value) in expected.items():
        diagnostics = problem.diagnose(action)
        assert diagnostics.net_incremental_value == pytest.approx(net_increment)
        assert diagnostics.expected_episode_value == pytest.approx(episode_value)

    weak = problem.diagnose(WEAK_POSITIVE)
    strong = problem.diagnose(STRONG_POSITIVE)
    overpowered = problem.diagnose(OVERPOWERED_POSITIVE)
    assert weak.expected_decision_value == pytest.approx(0.16)
    assert strong.expected_decision_value == pytest.approx(0.32)
    assert overpowered.expected_decision_value == pytest.approx(0.40)


def test_nuisance_scan_has_maximal_raw_entropy_but_no_regime_information() -> None:
    problem = HiddenActuatorProblem()
    nuisance = problem.diagnose(NUISANCE_SCAN)
    strong = problem.diagnose(STRONG_POSITIVE)

    assert nuisance.outcome_probabilities == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert nuisance.observation_entropy_nats == pytest.approx(math.log(4.0))
    assert nuisance.expected_information_gain_nats == pytest.approx(0.0)
    assert nuisance.expected_decision_value == pytest.approx(0.0)
    assert nuisance.observation_entropy_nats > strong.observation_entropy_nats


def test_fraction_oracle_independently_matches_all_float_value_terms() -> None:
    problem = HiddenActuatorProblem()
    exact = FractionOracle()
    expected_exact_totals = {
        SKIP: Fraction(1, 2),
        WEAK_POSITIVE: Fraction(63, 100),
        STRONG_POSITIVE: Fraction(37, 50),
        OVERPOWERED_POSITIVE: Fraction(9, 20),
        NUISANCE_SCAN: Fraction(49, 100),
    }

    for action in problem.acquisition_actions:
        float_row = problem.diagnose(action)
        exact_row = exact.evaluate(action)
        assert float_row.expected_immediate_payoff == pytest.approx(float(exact_row.expected_immediate_payoff))
        assert float_row.expected_decision_value == pytest.approx(float(exact_row.expected_decision_value))
        assert float_row.net_incremental_value == pytest.approx(float(exact_row.net_incremental_value))
        assert float_row.expected_episode_value == pytest.approx(float(exact_row.expected_episode_value))

    for action, expected in expected_exact_totals.items():
        assert exact.evaluate(action).expected_episode_value == expected


def test_realized_results_expose_outcomes_and_costs_but_not_the_regime() -> None:
    problem = HiddenActuatorProblem()

    correct_pulse = problem.realize_acquisition(
        STRONG_POSITIVE,
        ActuatorRegime.DIRECT,
        unit_interval=0.2,
    )
    flipped_pulse = problem.realize_acquisition(
        STRONG_POSITIVE,
        ActuatorRegime.DIRECT,
        unit_interval=0.95,
    )
    nuisance = problem.realize_acquisition(
        NUISANCE_SCAN,
        ActuatorRegime.REVERSED,
        unit_interval=0.80,
    )

    assert correct_pulse.outcome == 1
    assert correct_pulse.task_payoff == 1.0
    assert correct_pulse.net_reward == pytest.approx(0.42)
    assert flipped_pulse.outcome == -1
    assert flipped_pulse.task_payoff == 0.0
    assert flipped_pulse.net_reward == pytest.approx(-0.58)
    assert nuisance.outcome == 3
    assert nuisance.net_reward == pytest.approx(-0.01)
    assert not hasattr(correct_pulse, "regime")


def test_terminal_behavior_uses_the_posterior_and_paired_draw() -> None:
    problem = HiddenActuatorProblem()
    posterior = problem.posterior_direct(STRONG_POSITIVE, 1)

    assert problem.best_terminal_action(prior_direct=posterior) is TerminalAction.DIRECT
    assert problem.best_terminal_action(prior_direct=1.0 - posterior) is TerminalAction.REVERSED
    assert problem.terminal_value(prior_direct=posterior) == pytest.approx(0.82)
    assert problem.realize_terminal(
        TerminalAction.DIRECT,
        ActuatorRegime.DIRECT,
        unit_interval=0.85,
    ).success
    assert not problem.realize_terminal(
        TerminalAction.REVERSED,
        ActuatorRegime.DIRECT,
        unit_interval=0.85,
    ).success


def test_terminal_tie_uses_declared_positive_action_first() -> None:
    problem = HiddenActuatorProblem(prior_direct=0.5)

    assert problem.terminal_actions == (
        TerminalAction.DIRECT,
        TerminalAction.REVERSED,
    )
    assert problem.best_terminal_action() is TerminalAction.DIRECT


def test_invalid_actions_outcomes_and_random_draws_fail_closed() -> None:
    problem = HiddenActuatorProblem()

    with pytest.raises(ValueError, match="pulse command"):
        type(STRONG_POSITIVE)(STRONG_POSITIVE.kind, 0)
    with pytest.raises(ValueError, match="not an outcome"):
        problem.posterior_direct(STRONG_POSITIVE, 0)
    with pytest.raises(ValueError, match="unit_interval"):
        problem.realize_acquisition(
            STRONG_POSITIVE,
            ActuatorRegime.DIRECT,
            unit_interval=1.0,
        )
    with pytest.raises(ValueError, match="prior_direct"):
        HiddenActuatorProblem(prior_direct=1.1)
