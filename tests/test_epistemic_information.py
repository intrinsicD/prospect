from __future__ import annotations

import math

import pytest

from prospect.epistemics import (
    bayes_posterior,
    brier_score,
    categorical_log_score,
    diagonal_gaussian_nll,
    entropy,
    expected_information_gain,
    expected_value_of_sample_information,
    predictive_distribution,
)


def test_exact_binary_information_gain_matches_entropy_reduction() -> None:
    prior = (0.5, 0.5)
    perfect_probe = ((1.0, 0.0), (0.0, 1.0))
    assert expected_information_gain(prior, perfect_probe) == pytest.approx(math.log(2.0))
    assert predictive_distribution(prior, perfect_probe) == pytest.approx((0.5, 0.5))
    assert bayes_posterior(prior, perfect_probe, 0) == pytest.approx((1.0, 0.0))


def test_information_and_decision_value_are_not_interchangeable() -> None:
    prior = (0.5, 0.5)
    perfect_probe = ((1.0, 0.0), (0.0, 1.0))
    irrelevant_utility = ((1.0, 1.0),)
    result = expected_value_of_sample_information(prior, perfect_probe, irrelevant_utility)
    assert result.expected_information_gain_nats == pytest.approx(math.log(2.0))
    assert result.expected_decision_value == pytest.approx(0.0)
    assert result.net_value == pytest.approx(0.0)


def test_probe_is_selected_only_when_decision_value_exceeds_cost() -> None:
    prior = (0.5, 0.5)
    noisy_probe = ((0.8, 0.2), (0.2, 0.8))
    choose_hypothesis = ((1.0, 0.0), (0.0, 1.0))
    cheap = expected_value_of_sample_information(prior, noisy_probe, choose_hypothesis, acquisition_cost=0.1)
    expensive = expected_value_of_sample_information(prior, noisy_probe, choose_hypothesis, acquisition_cost=0.4)
    assert cheap.net_value > 0.0
    assert expensive.net_value < 0.0


def test_uninformative_noise_has_zero_information_and_decision_value() -> None:
    prior = (0.5, 0.5)
    noise = ((0.5, 0.5), (0.5, 0.5))
    utilities = ((1.0, 0.0), (0.0, 1.0))
    result = expected_value_of_sample_information(prior, noise, utilities)
    assert result.expected_information_gain_nats == pytest.approx(0.0)
    assert result.expected_decision_value == pytest.approx(0.0)


def test_scores_are_proper_and_finite_for_supported_outcomes() -> None:
    assert entropy((0.5, 0.5)) == pytest.approx(math.log(2.0))
    assert categorical_log_score((0.8, 0.2), 0) < categorical_log_score((0.2, 0.8), 0)
    assert brier_score((0.8, 0.2), 0) < brier_score((0.2, 0.8), 0)
    assert math.isfinite(diagonal_gaussian_nll((0.0,), (1.0,), (0.0,)))


def test_invalid_probability_shapes_fail_closed() -> None:
    with pytest.raises(ValueError):
        expected_information_gain((0.5, 0.5), ((1.0,),))
    with pytest.raises(IndexError):
        bayes_posterior((0.5, 0.5), ((1.0, 0.0), (0.0, 1.0)), 3)
