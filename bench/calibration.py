"""Harness-owned adaptive calibration policy (U-003).

Core consumers still receive plain float thresholds.  The harness uses the first
fifth of a nominal stream as a disjoint scale/quantile pilot, then replays only the
remaining observations through the online ACI update in arrival order.  OOD/failure
evaluation scores are deliberately not fed back: adapting on anomalies would
eventually redefine persistent failures as normal and erase the false-alarm semantics.
"""
from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from prospect.voe import AdaptiveThreshold


def calibrate_threshold(scores: ArrayLike, alpha: float) -> AdaptiveThreshold:
    """Fit an online tracker on a non-empty nominal stream and return it live.

    ``eta`` is dimensionful, so the harness derives it from the nominal IQR.  The
    empirical quantile is only a pilot warm start; later scores causally exercise the
    exact decaying-step ACI update exposed by the core helper. The pilot is disjoint
    so no score is both used with look-ahead and counted as an online decision.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite one-dimensional stream")
    pilot_size = max(1, len(values) // 5)
    pilot, online = values[:pilot_size], values[pilot_size:]
    q25, initial, q75 = np.quantile(pilot, [0.25, 1.0 - alpha, 0.75])
    eta = max(0.1 * float(q75 - q25), 1e-12)
    tracker = AdaptiveThreshold(alpha=alpha, eta=eta, initial_value=float(initial))
    for score in online:
        tracker.update(float(score))
    return tracker


def exceedance_rate(scores: ArrayLike, threshold: float) -> float:
    """Retrospective exceedance rate under one fixed threshold."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite one-dimensional stream")
    return float(np.mean(values > threshold))


def audit_threshold(
    scores: ArrayLike, tracker: AdaptiveThreshold, *, tolerance: float
) -> tuple[float, float, bool]:
    """Audit a frozen threshold on an independent nominal stream.

    ``tolerance`` is a predeclared engineering bound, not a statistical confidence
    interval: nominal streams can be sequential or clustered, and ACI's online
    rate-control claim does not require exchangeability.
    """
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    values = np.asarray(scores, dtype=float)
    rate = exceedance_rate(values, tracker.value)
    return rate, tolerance, abs(rate - tracker.alpha) <= tolerance
