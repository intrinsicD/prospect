"""Epistemic assessments derived from qualified beliefs and predictions.

This package deliberately separates ex-ante uncertainty, ex-post proper scores,
information gain, and decision value.  None of those quantities is a universal
substitute for the others.
"""

from .assessments import (
    CategoricalEntropyEffect,
    CategoricalScorer,
    OutcomeIndex,
    categorical_probabilities,
)
from .information import (
    InformationValueResult,
    bayes_posterior,
    entropy,
    expected_information_gain,
    expected_value_of_sample_information,
    predictive_distribution,
)
from .scoring import brier_score, categorical_log_score, diagonal_gaussian_nll

__all__ = [
    "CategoricalEntropyEffect",
    "CategoricalScorer",
    "InformationValueResult",
    "OutcomeIndex",
    "bayes_posterior",
    "brier_score",
    "categorical_log_score",
    "categorical_probabilities",
    "diagonal_gaussian_nll",
    "entropy",
    "expected_information_gain",
    "expected_value_of_sample_information",
    "predictive_distribution",
]
