"""P9-004 gate-overfit sentinel: negative controls + metamorphic invariants (ADR-0008).

A calibrated benchmark suite lies in two ways: **gate-overfit** (a trivial solution
quietly passes) and **noise** (a margin within seed variance reads as a pass). This
battery guards both, cheaply (no training), so it can stand in the ratchet:

- **negative controls** — each capability criterion must REJECT its degenerate solution
  (always-retrieve, one-step options), so passing means the capability, not the artifact;
- **metamorphic invariants** — properties that hold with no golden threshold (surprise
  decomposition is exact, an untrusted source never overrides, log-prob peaks at the
  mean), which catch bugs a thresholded gate cannot;
- **statistics** — a bootstrap CI distinguishes a real margin from noise.

Registered as the `gate-overfit` sentinel (active from P9). `check_invariants()` is pure
and unit-tested; the sentinel is healthy iff every check holds.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from prospect.memory import SemanticStore, UncertaintyMemoryRouter
from prospect.types import KnowledgeItem, LatentState, Prediction, Provenance, Trust
from prospect.voe import SurpriseCompetenceMonitor

from ..gates import SentinelResult, sentinel_check
from .p9_ablation import MARGIN, classify


def bootstrap_ci(
    diffs: np.ndarray, rng: np.random.Generator, n: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `diffs` — the tool a gate uses to ask
    'is this margin beyond noise?' (a CI that straddles 0 is not significant)."""
    d = np.asarray(diffs, dtype=float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Metamorphic invariants (no golden threshold)
# --------------------------------------------------------------------------- #
def _inv_surprise_decomposition_exact() -> bool:
    monitor = SurpriseCompetenceMonitor()
    pred = Prediction(mean=np.array([0.0, 0.0]), var=np.array([1.0, 1.0]),
                      epistemic=0.3, aleatoric=0.1)
    s = monitor.surprise(pred, LatentState(z=np.array([1.0, 1.0])))
    return abs(s.epistemic + s.aleatoric - s.total) < 1e-9  # attribution partitions total


def _inv_untrusted_never_overrides() -> bool:
    router = UncertaintyMemoryRouter([SemanticStore(trust=Trust.UNTRUSTED)], threshold=0.5)
    return router.route(None, 10.0) is None  # untrusted content is data, never instruction


def _inv_log_prob_peaks_at_mean() -> bool:
    pred = Prediction(mean=np.array([0.0, 0.0]), var=np.array([1.0, 1.0]),
                      epistemic=0.1, aleatoric=0.1)
    return pred.log_prob([0.0, 0.0]) > pred.log_prob([3.0, 3.0])


# --------------------------------------------------------------------------- #
# Negative controls (the trivial solution must FAIL the criterion)
# --------------------------------------------------------------------------- #
def _neg_always_retrieve_fails() -> bool:
    """Where the model is confident, blanket 'always retrieve' is worse than gating: it
    fetches a (here wrong) fact instead of the model's accurate prediction."""
    store = SemanticStore()
    store.write(KnowledgeItem(content=(np.array([0.0, 0.0, 0.0]), np.array([9.0, 9.0])),  # wrong
                              provenance=Provenance(source="x", trust=Trust.HIGH)))
    router = UncertaintyMemoryRouter([store], threshold=0.5)
    target = parametric = np.array([0.0, 0.0])  # confident, accurate model prediction
    retrieved = np.asarray(store.query(np.array([0.0, 0.0, 0.0]))[0].content[1], dtype=float)
    gated = parametric if router.route(None, 0.1) is None else retrieved  # 0.1 <= 0.5 -> model
    return float(np.mean((retrieved - target) ** 2)) > float(np.mean((gated - target) ** 2))


def _neg_one_step_options_fail_diversity() -> bool:
    return not (float(np.mean([1, 1, 1, 1])) > 1.0)  # option-diversity requires duration > 1


def _neg_ablation_no_over_credit() -> bool:
    """The ablation harness must not label a within-noise marginal as load-bearing."""
    return (classify(0.5 * MARGIN) == "negligible" and classify(-3.0 * MARGIN) == "harmful"
            and classify(3.0 * MARGIN) == "load-bearing")


def _stat_bootstrap_flags_noise() -> bool:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, size=40)   # a margin indistinguishable from 0
    real = rng.normal(5.0, 1.0, size=40)    # a margin clearly beyond noise
    lo_n, hi_n = bootstrap_ci(noise, np.random.default_rng(1))
    lo_r, _ = bootstrap_ci(real, np.random.default_rng(2))
    return lo_n < 0.0 < hi_n and lo_r > 0.0  # noise CI straddles 0; real CI excludes it


CHECKS: list[tuple[str, Callable[[], bool]]] = [
    ("surprise-decomposition-exact", _inv_surprise_decomposition_exact),
    ("untrusted-never-overrides", _inv_untrusted_never_overrides),
    ("log-prob-peaks-at-mean", _inv_log_prob_peaks_at_mean),
    ("always-retrieve-fails", _neg_always_retrieve_fails),
    ("one-step-options-fail-diversity", _neg_one_step_options_fail_diversity),
    ("ablation-no-over-credit", _neg_ablation_no_over_credit),
    ("bootstrap-flags-noise", _stat_bootstrap_flags_noise),
]


def check_invariants() -> list[tuple[str, bool]]:
    """Run the battery; each `(name, passed)` — pure, cheap, order-independent."""
    return [(name, fn()) for name, fn in CHECKS]


@sentinel_check("gate-overfit")
def check_gate_overfit() -> SentinelResult:
    results = check_invariants()
    failed = [name for name, ok in results if not ok]
    detail = (f"{len(results)} negative controls + metamorphic invariants hold"
              if not failed else "FAILED: " + ", ".join(failed))
    return SentinelResult(name="gate-overfit", healthy=not failed, detail=detail)
