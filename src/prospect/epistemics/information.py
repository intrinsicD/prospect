"""Exact finite epistemic calculations used by the reference semantics gate.

The functions here are intentionally small and auditable.  Neural or sampling
estimators may implement the same contracts later, but the exact finite case is the
oracle against which their meaning is checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, log
from typing import TypeAlias

ProbabilityVector: TypeAlias = tuple[float, ...]
LikelihoodMatrix: TypeAlias = tuple[ProbabilityVector, ...]
UtilityMatrix: TypeAlias = tuple[tuple[float, ...], ...]

_TOL = 1e-12


def _probabilities(values: tuple[float, ...], *, name: str) -> ProbabilityVector:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} contains a negative probability")
    total = sum(values)
    if total <= 0.0:
        raise ValueError(f"{name} has zero probability mass")
    normalized = tuple(value / total for value in values)
    if not isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=_TOL):
        raise ValueError(f"{name} could not be normalized")
    return normalized


def _likelihoods(likelihoods: LikelihoodMatrix, hypothesis_count: int) -> LikelihoodMatrix:
    if len(likelihoods) != hypothesis_count:
        raise ValueError("likelihood row count must equal the number of hypotheses")
    if not likelihoods:
        raise ValueError("likelihood matrix must not be empty")
    outcome_count = len(likelihoods[0])
    if outcome_count == 0 or any(len(row) != outcome_count for row in likelihoods):
        raise ValueError("likelihood rows must have one common non-zero length")
    return tuple(_probabilities(tuple(row), name=f"likelihood row {index}") for index, row in enumerate(likelihoods))


def entropy(probabilities: ProbabilityVector) -> float:
    """Shannon entropy in nats."""

    probs = _probabilities(tuple(probabilities), name="probabilities")
    return -sum(probability * log(probability) for probability in probs if probability)


def predictive_distribution(prior: ProbabilityVector, likelihoods: LikelihoodMatrix) -> ProbabilityVector:
    """Marginal distribution over evidence outcomes."""

    beliefs = _probabilities(tuple(prior), name="prior")
    rows = _likelihoods(likelihoods, len(beliefs))
    return tuple(
        sum(beliefs[hypothesis] * rows[hypothesis][outcome] for hypothesis in range(len(beliefs)))
        for outcome in range(len(rows[0]))
    )


def bayes_posterior(
    prior: ProbabilityVector,
    likelihoods: LikelihoodMatrix,
    observed_outcome: int,
) -> ProbabilityVector:
    """Posterior over hypotheses after one observed evidence outcome."""

    beliefs = _probabilities(tuple(prior), name="prior")
    rows = _likelihoods(likelihoods, len(beliefs))
    if not 0 <= observed_outcome < len(rows[0]):
        raise IndexError("observed outcome is outside the likelihood support")
    unnormalized = tuple(beliefs[hypothesis] * rows[hypothesis][observed_outcome] for hypothesis in range(len(beliefs)))
    if sum(unnormalized) <= 0.0:
        raise ValueError("observed outcome had zero prior-predictive probability")
    return _probabilities(unnormalized, name="posterior")


def expected_information_gain(prior: ProbabilityVector, likelihoods: LikelihoodMatrix) -> float:
    """Expected posterior entropy reduction, equal to mutual information.

    This measures information about the named hypothesis.  It is not automatically
    useful for a decision; use :func:`expected_value_of_sample_information` for that.
    """

    beliefs = _probabilities(tuple(prior), name="prior")
    rows = _likelihoods(likelihoods, len(beliefs))
    predictive = predictive_distribution(beliefs, rows)
    expected_posterior_entropy = 0.0
    for outcome, probability in enumerate(predictive):
        if probability <= 0.0:
            continue
        expected_posterior_entropy += probability * entropy(bayes_posterior(beliefs, rows, outcome))
    gain = entropy(beliefs) - expected_posterior_entropy
    return max(0.0, gain) if gain > -_TOL else gain


def _best_expected_utility(belief: ProbabilityVector, utilities: UtilityMatrix) -> tuple[int, float]:
    if not utilities:
        raise ValueError("at least one terminal decision is required")
    if any(len(row) != len(belief) for row in utilities):
        raise ValueError("each utility row must cover every hypothesis")
    values = tuple(
        sum(probability * utility for probability, utility in zip(belief, row, strict=True)) for row in utilities
    )
    index = max(range(len(values)), key=values.__getitem__)
    return index, values[index]


@dataclass(frozen=True, slots=True)
class InformationValueResult:
    """Expected information and utility effects of an evidence action."""

    expected_information_gain_nats: float
    expected_decision_value: float
    acquisition_cost: float
    net_value: float
    prior_best_decision: int


def expected_value_of_sample_information(
    prior: ProbabilityVector,
    likelihoods: LikelihoodMatrix,
    utilities: UtilityMatrix,
    *,
    acquisition_cost: float = 0.0,
) -> InformationValueResult:
    """Expected value of evidence for the specified downstream decision.

    The result is contextual: changing ``utilities`` can turn the same learnable
    evidence from useful to irrelevant without changing its mutual information.
    """

    if acquisition_cost < 0.0:
        raise ValueError("acquisition cost must be non-negative")
    beliefs = _probabilities(tuple(prior), name="prior")
    rows = _likelihoods(likelihoods, len(beliefs))
    prior_decision, prior_value = _best_expected_utility(beliefs, utilities)
    predictive = predictive_distribution(beliefs, rows)
    value_after_evidence = 0.0
    for outcome, probability in enumerate(predictive):
        if probability <= 0.0:
            continue
        posterior = bayes_posterior(beliefs, rows, outcome)
        _, posterior_value = _best_expected_utility(posterior, utilities)
        value_after_evidence += probability * posterior_value
    decision_value = value_after_evidence - prior_value
    if -_TOL < decision_value < 0.0:
        decision_value = 0.0
    return InformationValueResult(
        expected_information_gain_nats=expected_information_gain(beliefs, rows),
        expected_decision_value=decision_value,
        acquisition_cost=acquisition_cost,
        net_value=decision_value - acquisition_cost,
        prior_best_decision=prior_decision,
    )
