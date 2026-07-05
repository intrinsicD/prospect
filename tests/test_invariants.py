"""Unit tests for the P9-004 gate-overfit battery: every negative control + metamorphic
invariant holds, and the bootstrap CI distinguishes a real margin from noise."""
from __future__ import annotations

import numpy as np

from bench.evals.p9_invariants import bootstrap_ci, check_invariants


def test_all_negative_controls_and_invariants_hold() -> None:
    results = check_invariants()
    failed = [name for name, ok in results if not ok]
    assert not failed, f"gate-overfit checks failed: {failed}"
    assert len(results) >= 6  # a real battery, not a single check


def test_bootstrap_ci_straddles_zero_for_noise_and_excludes_for_a_real_margin() -> None:
    rng = np.random.default_rng(0)
    lo_noise, hi_noise = bootstrap_ci(rng.normal(0.0, 1.0, size=60), np.random.default_rng(1))
    lo_real, _ = bootstrap_ci(rng.normal(5.0, 1.0, size=60), np.random.default_rng(2))
    assert lo_noise < 0.0 < hi_noise  # a within-noise margin is not significant
    assert lo_real > 0.0              # a real margin excludes zero
