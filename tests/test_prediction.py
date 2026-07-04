"""Unit tests for the `Prediction` distribution contract (P0-001, ADR-0002).

`Prediction` must parameterize a real diagonal Gaussian: `log_prob` is concrete,
correct, finite under a variance floor, and the type is immutable.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import isfinite, log, tau

import pytest

from prospect.types import Prediction


def _pred(mean: list[float], var: list[float], **kwargs: float) -> Prediction:
    return Prediction(mean=mean, var=var, epistemic=0.1, aleatoric=0.2, **kwargs)


def test_log_prob_standard_normal_at_mean() -> None:
    # 2-dim standard normal evaluated at its mean: log N(0 | 0, 1) per dim = -0.5*log(2*pi)
    p = _pred([0.0, 0.0], [1.0, 1.0])
    assert p.log_prob([0.0, 0.0]) == pytest.approx(-log(tau))


def test_log_prob_matches_hand_computed_value() -> None:
    # N(mean=[1.0, -2.0], var=[0.5, 2.0]) at observed=[1.5, -1.0]; value computed by hand:
    # -0.5 * (0.5^2/0.5 + ln(2*pi*0.5) + 1.0^2/2.0 + ln(2*pi*2.0)) = -2.3378770664093453
    p = _pred([1.0, -2.0], [0.5, 2.0])
    assert p.log_prob([1.5, -1.0]) == pytest.approx(-2.3378770664093453, rel=1e-12)


def test_log_prob_higher_near_the_mean() -> None:
    p = _pred([0.0], [1.0])
    assert p.log_prob([0.1]) > p.log_prob([2.0])


def test_log_prob_finite_at_zero_variance() -> None:
    # The variance floor keeps surprise finite even for an overconfident prediction.
    p = _pred([0.0], [0.0])
    assert isfinite(p.log_prob([1.0]))


def test_log_prob_length_mismatch_raises() -> None:
    p = _pred([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="length mismatch"):
        p.log_prob([0.0, 1.0])


def test_prediction_is_frozen() -> None:
    p = _pred([0.0], [1.0])
    with pytest.raises(FrozenInstanceError):
        p.reward = 2.0  # type: ignore[misc]


def test_duration_defaults_to_one_step() -> None:
    assert _pred([0.0], [1.0]).duration == 1.0
    assert _pred([0.0], [1.0], duration=7.5).duration == 7.5
