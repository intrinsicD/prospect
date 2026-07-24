from __future__ import annotations

import pytest

from bench.active_acquisition import (
    NUISANCE_SCAN,
    OVERPOWERED_POSITIVE,
    SKIP,
    STRONG_POSITIVE,
    WEAK_POSITIVE,
    AcquisitionKind,
    HiddenActuatorProblem,
    eig_only,
    goal_only,
    oracle,
    prospect_voi,
    raw_entropy,
    shuffled_information,
    uniform_random,
)
from bench.active_acquisition.policies import ActionScore, _best


def test_predeclared_policy_controls_choose_distinct_expected_actions() -> None:
    problem = HiddenActuatorProblem()

    assert prospect_voi(problem).selected_action == STRONG_POSITIVE
    assert oracle(problem).selected_action == STRONG_POSITIVE
    assert goal_only(problem).selected_action == SKIP
    assert raw_entropy(problem).selected_action == NUISANCE_SCAN
    assert eig_only(problem).selected_action == OVERPOWERED_POSITIVE
    assert shuffled_information(problem).selected_action == WEAK_POSITIVE


def test_prospect_and_fraction_oracle_agree_on_every_action_value() -> None:
    problem = HiddenActuatorProblem()
    prospect = prospect_voi(problem)
    exact = oracle(problem)

    assert prospect.selected_action == exact.selected_action
    assert [row.action for row in prospect.scores] == [row.action for row in exact.scores]
    assert [row.value for row in prospect.scores] == pytest.approx([row.value for row in exact.scores])
    assert prospect.selected_score == pytest.approx(0.74)
    assert {row.score_kind for row in prospect.scores} == {"expected_episode_net_return"}
    assert {row.unit for row in prospect.scores} == {"return"}


def test_goal_only_excludes_information_but_still_pays_declared_costs() -> None:
    problem = HiddenActuatorProblem()
    decision = goal_only(problem)
    scores = {row.action: row.value for row in decision.scores}

    assert scores[SKIP] == pytest.approx(0.50)
    assert scores[STRONG_POSITIVE] == pytest.approx(0.42)
    assert scores[NUISANCE_SCAN] == pytest.approx(0.49)
    assert {row.score_kind for row in decision.scores} == {"goal_only_expected_return"}
    assert {row.unit for row in decision.scores} == {"return"}


def test_raw_entropy_and_eig_only_fail_for_different_reasons() -> None:
    problem = HiddenActuatorProblem()
    raw = raw_entropy(problem)
    eig = eig_only(problem)

    assert raw.selected_action.kind is AcquisitionKind.NUISANCE_SCAN
    assert eig.selected_action.kind is AcquisitionKind.OVERPOWERED_PULSE
    assert problem.diagnose(raw.selected_action).expected_decision_value == pytest.approx(0.0)
    assert problem.diagnose(eig.selected_action).net_incremental_value == pytest.approx(-0.05)


def test_shuffled_information_preserves_score_sources_but_breaks_action_linkage() -> None:
    problem = HiddenActuatorProblem()
    decision = shuffled_information(problem)
    scores = {row.action: row.value for row in decision.scores}

    # Weak receives the overpowered pulse's 0.40 EVSI while retaining its own
    # expected immediate payoff and 0.53 action cost.
    assert scores[WEAK_POSITIVE] == pytest.approx(0.87)
    # Strong retains its own direct utility/cost but receives weak's 0.16 EVSI.
    assert scores[STRONG_POSITIVE] == pytest.approx(0.58)
    # Skip and nuisance retain their own zero information values.
    assert scores[NUISANCE_SCAN] == pytest.approx(0.49)
    assert decision.selected_action == WEAK_POSITIVE


def test_uniform_random_is_seeded_and_selects_only_declared_actions() -> None:
    problem = HiddenActuatorProblem()

    first = uniform_random(problem, seed=1729)
    repeated = uniform_random(problem, seed=1729)
    alternatives = {uniform_random(problem, seed=seed).selected_action for seed in range(32)}

    assert first.selected_action == repeated.selected_action
    assert first.selected_action in problem.acquisition_actions
    assert len(alternatives) > 1


def test_uniform_random_uses_stable_sha256_modulo_vectors() -> None:
    problem = HiddenActuatorProblem()
    expected_ids = (
        "nuisance",
        "strong",
        "overpowered",
        "strong",
        "strong",
        "nuisance",
        "strong",
        "skip",
        "overpowered",
        "nuisance",
    )

    assert (
        tuple(uniform_random(problem, seed=seed).selected_action.action_id for seed in range(len(expected_ids)))
        == expected_ids
    )
    with pytest.raises(ValueError, match="integer"):
        uniform_random(problem, seed=True)
    with pytest.raises(ValueError, match="nonnegative"):
        uniform_random(problem, seed=-1)


def test_policy_ties_use_the_stable_action_identifier() -> None:
    problem = HiddenActuatorProblem()

    assert prospect_voi(problem).selected_action.command == 1
    assert oracle(problem).selected_action.command == 1
    assert eig_only(problem).selected_action.command == 1


def test_candidate_score_ties_use_declared_candidate_order() -> None:
    problem = HiddenActuatorProblem()
    tied_scores = tuple(ActionScore(action, 1.0, "test_tie", "return") for action in problem.acquisition_actions)

    assert _best("test_tie", tied_scores).selected_action == SKIP
