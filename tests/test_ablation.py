"""Unit tests for the P9-002 ablation harness: leave-one-out marginal value + classify."""
from __future__ import annotations

from bench.evals.p9_ablation import MARGIN, classify, marginals


def test_marginals_are_full_minus_ablated() -> None:
    marg = marginals(-19.0, {"planning": -73.0, "retrieval": -8.0, "exploit_penalty": -25.0})
    assert marg["planning"] == 54.0      # disabling planning hurts a lot -> load-bearing
    assert marg["retrieval"] == -11.0    # disabling retrieval HELPS -> a harmful component
    assert marg["exploit_penalty"] == 6.0


def test_classify_labels_by_sign_and_margin() -> None:
    assert classify(54.0) == "load-bearing"
    assert classify(-11.0) == "harmful"
    assert classify(2.0) == "negligible"   # within +/-MARGIN either way
    assert classify(-2.0) == "negligible"
    assert classify(MARGIN + 0.1) == "load-bearing"
