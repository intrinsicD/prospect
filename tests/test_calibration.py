"""Harness calibration policy for U-003."""
from __future__ import annotations

import numpy as np
import pytest

from bench.calibration import audit_threshold, calibrate_threshold, exceedance_rate
from prospect.memory import UncertaintyMemoryRouter
from prospect.planning import HierarchicalManager


def test_nominal_calibration_publishes_the_target_quantile_as_a_float() -> None:
    scores = np.random.default_rng(11).normal(loc=4.0, scale=0.2, size=2_000)
    tracker = calibrate_threshold(scores, alpha=0.1)

    assert tracker.value == pytest.approx(float(np.quantile(scores, 0.9)), abs=0.03)
    assert tracker.trigger_rate == pytest.approx(0.1, abs=0.02)
    assert exceedance_rate(scores, tracker.value) == pytest.approx(0.1, abs=0.02)


def test_nominal_calibration_uses_a_disjoint_pilot_then_tracks_a_shift() -> None:
    rng = np.random.default_rng(22)
    pilot = rng.normal(size=1_000)
    shifted = rng.normal(loc=3.0, size=4_000)

    baseline = calibrate_threshold(pilot, alpha=0.1)
    tracker = calibrate_threshold(np.concatenate([pilot, shifted]), alpha=0.1)

    assert tracker.updates == len(shifted)
    assert tracker.value > baseline.value + 2.0
    assert tracker.value == pytest.approx(float(np.quantile(shifted, 0.9)), abs=0.15)
    assert exceedance_rate(shifted[-2_000:], tracker.value) == pytest.approx(0.1, abs=0.02)


def test_termination_and_retrieval_publish_separate_calibrated_thresholds() -> None:
    rng = np.random.default_rng(31)
    termination = calibrate_threshold(rng.normal(loc=8.0, size=2_000), alpha=0.01)
    retrieval = calibrate_threshold(rng.gamma(shape=2.0, scale=0.1, size=2_000), alpha=0.1)

    manager = HierarchicalManager(surprise_threshold=termination.value)
    router = UncertaintyMemoryRouter(threshold=retrieval.value)

    assert termination is not retrieval
    assert termination.alpha == 0.01 and retrieval.alpha == 0.1
    assert manager.surprise_threshold == termination.value
    assert router.threshold == retrieval.value


def test_independent_nominal_audit_has_finite_sample_tolerance() -> None:
    rng = np.random.default_rng(41)
    tracker = calibrate_threshold(rng.normal(size=5_000), alpha=0.01)
    rate, tolerance, accepted = audit_threshold(
        rng.normal(size=5_000), tracker, tolerance=0.006
    )

    assert rate == pytest.approx(0.01, abs=tolerance)
    assert tolerance < 0.01
    assert accepted is True


def test_rare_tail_calibration_is_measurable_and_target_sensitive() -> None:
    alpha, tolerance = 0.0001, 0.00015
    rng = np.random.default_rng(83)
    tracker = calibrate_threshold(rng.normal(size=100_000), alpha=alpha)
    rate, _, accepted = audit_threshold(
        rng.normal(size=100_000), tracker, tolerance=tolerance
    )

    assert 0.0 < tracker.trigger_rate == pytest.approx(alpha, abs=tolerance)
    assert 0.0 < rate == pytest.approx(alpha, abs=tolerance)
    assert accepted is True


@pytest.mark.parametrize("scores", [[], [0.0, float("nan")]])
def test_nominal_calibration_rejects_invalid_streams(scores: list[float]) -> None:
    with pytest.raises(ValueError, match="scores"):
        calibrate_threshold(scores, alpha=0.1)
    with pytest.raises(ValueError, match="scores"):
        exceedance_rate(scores, threshold=0.0)
