"""Semantic tests for the exact epistemic decision benchmark."""

from math import log2

import pytest

from bench.epistemic import (
    DiagnosticDecisionProblem,
    Evidence,
    ExploitAction,
    FutureEvidenceError,
    Probe,
    ProbeOutcome,
    binary_entropy,
)


def test_informative_probe_has_exact_bayes_posterior_information_and_value() -> None:
    problem = DiagnosticDecisionProblem()
    probe = Probe.informative(accuracy=0.9)

    assert problem.posterior_right(probe, ProbeOutcome.ONE) == pytest.approx(0.9)
    assert problem.posterior_right(probe, ProbeOutcome.ZERO) == pytest.approx(0.1)
    assert problem.expected_information_gain_bits(probe) == pytest.approx(1.0 - binary_entropy(0.9))
    assert problem.expected_value_of_sample_information(probe) == pytest.approx(0.25)


def test_future_evidence_cannot_leak_into_an_earlier_decision() -> None:
    problem = DiagnosticDecisionProblem()
    evidence = Evidence(Probe.informative(), ProbeOutcome.ONE, observed_at=4)

    with pytest.raises(FutureEvidenceError, match="unavailable"):
        problem.posterior_from_evidence(evidence, as_of=3)

    assert problem.posterior_from_evidence(evidence, as_of=4) == pytest.approx(0.9)


@pytest.mark.parametrize("probe", [Probe.irrelevant_bit(), Probe.noisy_signal()])
def test_high_entropy_nuisance_and_irreducible_noise_have_no_epistemic_or_decision_value(
    probe: Probe,
) -> None:
    problem = DiagnosticDecisionProblem()
    evaluation = problem.evaluate_probe(probe)

    assert evaluation.observation_entropy_bits == pytest.approx(1.0)
    assert evaluation.expected_information_gain_bits == pytest.approx(0.0)
    assert evaluation.expected_value_of_sample_information == pytest.approx(0.0)
    assert evaluation.net_value_of_information == pytest.approx(0.0)


def test_probe_selection_ignores_noisy_tv_and_irrelevant_bit_attraction() -> None:
    problem = DiagnosticDecisionProblem()
    informative = Probe.informative(cost=0.1)

    assert problem.select_probe([Probe.noisy_signal(), Probe.irrelevant_bit(), informative]) == informative
    assert problem.select_probe([Probe.noisy_signal(), Probe.irrelevant_bit()]) is None


def test_destructive_certainty_is_not_epistemic_evidence() -> None:
    problem = DiagnosticDecisionProblem()
    destructive = Probe.destructive_certainty()
    evaluation = problem.evaluate_probe(destructive)

    assert evaluation.physical_state_entropy_reduction_bits == pytest.approx(1.0)
    assert evaluation.observation_entropy_bits == pytest.approx(0.0)
    assert evaluation.expected_information_gain_bits == pytest.approx(0.0)
    assert evaluation.expected_value_of_sample_information == pytest.approx(0.0)
    assert not evaluation.is_admissible_epistemic_action
    assert problem.select_probe([destructive]) is None


def test_hypothesis_label_permutation_preserves_metrics_and_swaps_decisions() -> None:
    problem = DiagnosticDecisionProblem()
    probe = Probe.informative(accuracy=0.8)
    probability_right = 0.72
    permuted_probability_right = 1.0 - probability_right

    assert problem.expected_information_gain_bits(probe, prior_right=probability_right) == pytest.approx(
        problem.expected_information_gain_bits(probe, prior_right=permuted_probability_right)
    )
    assert problem.expected_value_of_sample_information(probe, prior_right=probability_right) == pytest.approx(
        problem.expected_value_of_sample_information(probe, prior_right=permuted_probability_right)
    )
    assert problem.posterior_right(probe, ProbeOutcome.ONE, prior_right=probability_right) == pytest.approx(
        1.0 - problem.posterior_right(probe, ProbeOutcome.ZERO, prior_right=permuted_probability_right)
    )
    assert problem.best_exploit(prior_right=probability_right).action is ExploitAction.GUESS_RIGHT
    assert problem.best_exploit(prior_right=permuted_probability_right).action is ExploitAction.GUESS_LEFT


def test_probe_cost_threshold_uses_decision_value_not_information_gain() -> None:
    problem = DiagnosticDecisionProblem()
    below_threshold = Probe.informative(cost=0.249)
    at_threshold = Probe.informative(cost=0.25)
    above_threshold = Probe.informative(cost=0.251)

    assert problem.best_exploit().action is ExploitAction.KNOWN_SAFE
    assert problem.expected_value_of_sample_information(below_threshold) == pytest.approx(0.25)
    assert problem.expected_information_gain_bits(below_threshold) == pytest.approx(
        1.0 + 0.9 * log2(0.9) + 0.1 * log2(0.1)
    )
    assert problem.select_probe([below_threshold]) == below_threshold
    assert problem.select_probe([at_threshold]) is None
    assert problem.select_probe([above_threshold]) is None
