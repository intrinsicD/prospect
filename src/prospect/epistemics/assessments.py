"""Domain-linked proper scoring and posterior-change assessments.

These adapters deliberately implement only quantities that are justified by the
declared categorical distribution.  In particular, a posterior entropy change is
reported as an internal information effect; it is not relabelled as durable
knowledge or externally demonstrated improvement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from prospect.domain import (
    BeliefUpdate,
    Distribution,
    DomainInvariantError,
    EpistemicEffect,
    EpistemicEffectKind,
    ExperienceEvent,
    Prediction,
    ProperScore,
)

from .information import entropy
from .scoring import brier_score, categorical_log_score

OutcomeIndex = Callable[[ExperienceEvent], int]


def _real_parameter(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise DomainInvariantError("categorical parameters must contain only real numbers")
    return float(value)


def categorical_probabilities(distribution: Distribution) -> tuple[float, ...]:
    """Return normalized probabilities from a categorical domain distribution."""

    if distribution.family != "categorical":
        raise DomainInvariantError(f"categorical assessment cannot consume {distribution.family!r}")
    parameters = distribution.parameters
    if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes, bytearray)):
        raise DomainInvariantError("categorical parameters must be a numeric sequence")
    probabilities = tuple(_real_parameter(value) for value in cast(Sequence[object], parameters))
    if not probabilities:
        raise DomainInvariantError("categorical parameters must not be empty")
    if any(probability < 0.0 for probability in probabilities):
        raise DomainInvariantError("categorical probabilities must be non-negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise DomainInvariantError("categorical probabilities have zero mass")
    return tuple(probability / total for probability in probabilities)


@dataclass(frozen=True, slots=True)
class CategoricalScorer:
    """Score the immutable action-time forecast against a realized outcome."""

    outcome_index: OutcomeIndex
    rule: str = "log_score"
    scorer_version: str = "categorical-scorer-v1"

    def score(self, prediction: Prediction, experience: ExperienceEvent) -> ProperScore:
        """Return a lower-is-better proper score linked to real evidence."""

        probabilities = categorical_probabilities(prediction.distribution)
        observed_index = self.outcome_index(experience)
        if self.rule == "log_score":
            value = categorical_log_score(probabilities, observed_index)
            unit = "nats"
        elif self.rule == "brier":
            value = brier_score(probabilities, observed_index)
            unit = "squared_probability"
        else:
            raise ValueError(f"unsupported categorical scoring rule: {self.rule!r}")
        evidence = experience.outcome.evidence
        return ProperScore(
            score_id=(f"score:{self.scorer_version}:{prediction.prediction_id}:{evidence.evidence_id}"),
            prediction_id=prediction.prediction_id,
            realized_evidence_id=evidence.evidence_id,
            rule=self.rule,
            value=value,
            unit=unit,
            scorer_version=self.scorer_version,
            scored_at=experience.closed_at,
        )


@dataclass(frozen=True, slots=True)
class CategoricalEntropyEffect:
    """Measure posterior entropy change without making a knowledge claim."""

    evaluator_version: str = "categorical-entropy-v1"

    def effect(self, update: BeliefUpdate) -> EpistemicEffect:
        """Return the internal entropy reduction associated with one update."""

        before = entropy(categorical_probabilities(update.prior.distribution))
        after = entropy(categorical_probabilities(update.posterior.distribution))
        return EpistemicEffect(
            effect_id=f"effect:{self.evaluator_version}:{update.update_id}",
            belief_update_id=update.update_id,
            target_id=update.prior.target.target_id,
            kind=EpistemicEffectKind.INFORMATION_GAIN,
            measure="categorical_entropy",
            before=before,
            after=after,
            improvement=before - after,
            higher_is_better=False,
            evaluator_version=self.evaluator_version,
            evaluated_at=update.updated_at,
            externally_calibrated=False,
        )


__all__ = (
    "CategoricalEntropyEffect",
    "CategoricalScorer",
    "OutcomeIndex",
    "categorical_probabilities",
)
