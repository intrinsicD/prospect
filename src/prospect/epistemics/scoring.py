"""Proper scoring rules for realized outcomes."""

from __future__ import annotations

from math import isfinite, log, tau

_VARIANCE_FLOOR = 1e-12


def _normalized(probabilities: tuple[float, ...]) -> tuple[float, ...]:
    if not probabilities or any(probability < 0.0 for probability in probabilities):
        raise ValueError("probabilities must be a non-empty non-negative vector")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("probabilities have zero mass")
    return tuple(probability / total for probability in probabilities)


def categorical_log_score(probabilities: tuple[float, ...], observed_index: int) -> float:
    """Negative log score in nats; lower is better."""

    probs = _normalized(probabilities)
    if not 0 <= observed_index < len(probs):
        raise IndexError("observed index is outside the categorical support")
    probability = probs[observed_index]
    return float("inf") if probability <= 0.0 else -log(probability)


def brier_score(probabilities: tuple[float, ...], observed_index: int) -> float:
    """Multiclass Brier score."""

    probs = _normalized(probabilities)
    if not 0 <= observed_index < len(probs):
        raise IndexError("observed index is outside the categorical support")
    return sum((probability - float(index == observed_index)) ** 2 for index, probability in enumerate(probs))


def diagonal_gaussian_nll(
    mean: tuple[float, ...],
    variance: tuple[float, ...],
    observed: tuple[float, ...],
) -> float:
    """Negative log-likelihood for a diagonal Gaussian."""

    if not (len(mean) == len(variance) == len(observed)) or not mean:
        raise ValueError("mean, variance, and observation must have equal non-zero length")
    if any(not isfinite(value) for value in (*mean, *variance, *observed)):
        raise ValueError("Gaussian parameters and observation must be finite")
    return 0.5 * sum(
        (value - center) ** 2 / max(spread, _VARIANCE_FLOOR) + log(tau * max(spread, _VARIANCE_FLOOR))
        for center, spread, value in zip(mean, variance, observed, strict=True)
    )
