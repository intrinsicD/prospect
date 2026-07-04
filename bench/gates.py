"""Benchmark-gated delivery with collapse sentinels (ADR-0005, ADR-0006).

A phase ships only when (a) its **capability** gate passes AND (b) every applicable
integrity **sentinel** is healthy. Capability gates ask "did the model reach the
bar?"; sentinels ask "did it stay healthy while doing so?" — because collapse
(representation, uncertainty, generative-replay, option) is typically invisible in
the training loss: the trivial solution has low loss.

Each gate/sentinel has a *precise* criterion and a `check()` returning a result
object. Until the eval body exists, `check()` returns a PENDING (not-passed /
not-healthy) result and does not raise, so `make gate` prints cleanly.

Sentinel data contract (P0-005): training loops write per-step metrics to
`bench/runs/<run-id>/metrics.jsonl` via `bench.runlog.RunLog` — the keys come from
what `Learner.update()` returns plus held-out probes. A zero-argument sentinel
`check()` reads the run back (`bench.runlog.read_run`, default `latest_run()`) to
verify its criterion throughout training, not only at the capability checkpoint.
Each sentinel's criterion names the metric keys it requires.

Check registration (P0-006): the registries here hold criteria as *data*; eval
bodies live in `bench/evals/` and replace their PENDING check by decorating a
zero-arg callable with `@gate_check(phase)` / `@sentinel_check(name)` at import.

Seed policy (P0-006): every registered check owns an explicit seed list, evaluates
over those seeds, and records them in `GateResult.seeds`. `run_gate` persists a JSON
report (criterion, metrics, seeds, run-id) under `bench/results/`, so recorded
results are reproducible and docs-sync is mechanical.

Regression ratchet (P0-007): phases whose gate has passed are listed in
`bench/SHIPPED`. `make gate-all` re-runs every shipped gate and fails if any is
BLOCKED; CI runs it on every push, so a shipped capability cannot silently regress.
For gates whose evidence is an expensive training artifact, the re-run policy is:
re-run the *evaluation* against the persisted / regenerable artifact (e.g. the run
log), not full retraining — and say so in the report detail. Revisit via an
ADR-0005 amendment if a gate outgrows this.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PHASE_ORDER = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]


def _phase_at_least(phase: str, floor: str) -> bool:
    """True if `phase` is at or beyond `floor` in the roadmap order."""
    return PHASE_ORDER.index(phase) >= PHASE_ORDER.index(floor)


# --------------------------------------------------------------------------- #
# Capability gates: did the model reach the bar for this phase?
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    phase: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    detail: str = ""


@dataclass
class Gate:
    phase: str
    goal: str
    criterion: str  # the exact, human-readable pass condition
    check: Callable[[], GateResult]


def _pending_gate(phase: str, criterion: str) -> Callable[[], GateResult]:
    def _check() -> GateResult:
        return GateResult(phase=phase, passed=False, detail=f"PENDING — {criterion}")

    return _check


GATES: dict[str, Gate] = {}


def _register(phase: str, goal: str, criterion: str) -> None:
    GATES[phase] = Gate(phase, goal, criterion, _pending_gate(phase, criterion))


_register(
    "P0",
    "Scaffold",
    "Imports are clean and the smoke test suite passes (pytest exit code 0). "
    "Eval registered in bench/evals/p0_scaffold.py.",
)
_register(
    "P1",
    "Flat world model + calibrated uncertainty",
    "Latent 1-step prediction beats a persistence/linear baseline, AND on a "
    "stochastic variant epistemic uncertainty falls with more data while aleatoric "
    "persists (the two are separable).",
)
_register(
    "P2",
    "Planning beats reaction",
    "MPC/CEM in imagination beats a model-free baseline at EQUAL environment-step "
    "budget on the reference control task.",
)
_register(
    "P3",
    "VoE mastery test + curiosity curriculum",
    "Expected-vs-violated surprise differential is large and reliable (effect size "
    "over N seeds), AND curiosity-driven data collection improves sample efficiency "
    "vs random exploration.",
)
_register(
    "P4",
    "Skills with predictive preconditions",
    "simulate-to-select picks the correct skill above baseline, AND misapplication is "
    "flagged by a surprise spike.",
)
_register(
    "P5",
    "Hierarchical planning",
    "Two-level (jumpy option-model) planning beats flat planning on a long-horizon "
    "task at EQUAL compute.",
)
_register(
    "P6",
    "Any-to-any codec",
    "Swapping the single-modality codec for the universal codec preserves core-loop "
    "performance within tolerance.",
)
_register(
    "P7",
    "Continual improvement",
    "On a task sequence, retention of earlier skills stays above threshold (no "
    "catastrophic forgetting), AND plasticity is retained (late tasks learn as fast "
    "as early ones).",
)
_register(
    "P8",
    "Knowledge bases",
    "Uncertainty-gated retrieval improves accuracy vs no-retrieval on the use-case "
    "benchmark, AND performance is robust to a poisoned/low-trust source (provenance "
    "respected).",
)
_register(
    "P9",
    "Whole-system integration",
    "The fully-composed agent works end-to-end through the composition root: it "
    "controls better than a reactive baseline; within ONE run the single epistemic "
    "signal both sets the planner's explore/exploit coefficient AND gates retrieval "
    "(retrieval fires where the model is uncertain); retrieval-as-action does not "
    "degrade control; and all applicable collapse sentinels stay healthy.",
)


# --------------------------------------------------------------------------- #
# Collapse sentinels: did the model stay healthy? (ADR-0006)
# Integrity is enforced, not hoped for — collapse hides in a good loss curve.
# --------------------------------------------------------------------------- #
@dataclass
class SentinelResult:
    name: str
    healthy: bool
    metrics: dict[str, float] = field(default_factory=dict)
    detail: str = ""


@dataclass
class Sentinel:
    name: str
    detects: str  # the collapse mode this guards against
    applies_from: str  # phase from which this integrity check is active
    criterion: str  # the precise "healthy" condition
    check: Callable[[], SentinelResult]


def _pending_sentinel(name: str, criterion: str) -> Callable[[], SentinelResult]:
    def _check() -> SentinelResult:
        return SentinelResult(name=name, healthy=False, detail=f"PENDING — {criterion}")

    return _check


SENTINELS: dict[str, Sentinel] = {}


def _register_sentinel(name: str, detects: str, applies_from: str, criterion: str) -> None:
    SENTINELS[name] = Sentinel(name, detects, applies_from, criterion, _pending_sentinel(name, criterion))


_register_sentinel(
    "representation-integrity",
    "representation collapse (constant / low-rank latent)",
    "P1",
    "Latent per-dimension std stays above a floor AND effective rank (participation "
    "ratio of the latent covariance) stays above a threshold on held-out data "
    "throughout training — not only at the capability checkpoint.",
)
_register_sentinel(
    "uncertainty-reliability",
    "uncertainty / ensemble collapse (confident-but-wrong)",
    "P1",
    "Predicted epistemic uncertainty is rank-correlated with held-out prediction "
    "error above a threshold, and ensemble-member disagreement does not decay to ~0 "
    "in regions where error is high.",
)
_register_sentinel(
    "replay-fidelity",
    "generative-replay collapse (model autophagy / MAD)",
    "P3",
    "Rehearsal keeps a fixed real-data fraction; dreamed-sample diversity (feature "
    "variance / coverage vs real) stays above a floor and does not shrink across "
    "regenerations; rehearsal lineage depth is capped (no dream-of-dreams).",
)
_register_sentinel(
    "option-diversity",
    "option collapse (options become identical or one-step; abstraction lost)",
    "P5",
    "Option-usage entropy stays above a floor (no single option dominates), mean "
    "option duration stays above one step, and pairwise option outcome-distributions "
    "remain distinguishable.",
)


# --------------------------------------------------------------------------- #
# Check registration: eval modules in bench/evals/ replace PENDING checks.
# --------------------------------------------------------------------------- #
def gate_check(phase: str) -> Callable[[Callable[[], GateResult]], Callable[[], GateResult]]:
    """Decorator: register the real capability eval for `phase`, replacing PENDING.
    Criteria stay data in this module; eval bodies self-register on import."""
    if phase not in GATES:
        raise KeyError(f"unknown phase {phase!r}; known phases: {', '.join(GATES)}")

    def _register_check(fn: Callable[[], GateResult]) -> Callable[[], GateResult]:
        GATES[phase].check = fn
        return fn

    return _register_check


def sentinel_check(name: str) -> Callable[[Callable[[], SentinelResult]], Callable[[], SentinelResult]]:
    """Decorator: register the real integrity eval for sentinel `name`, replacing PENDING."""
    if name not in SENTINELS:
        raise KeyError(f"unknown sentinel {name!r}; known sentinels: {', '.join(SENTINELS)}")

    def _register_check(fn: Callable[[], SentinelResult]) -> Callable[[], SentinelResult]:
        SENTINELS[name].check = fn
        return fn

    return _register_check


def applicable_sentinels(phase: str) -> list[Sentinel]:
    """The integrity sentinels active by `phase` (those whose applies_from <= phase)."""
    return [s for s in SENTINELS.values() if _phase_at_least(phase, s.applies_from)]


def run_sentinels(phase: str) -> list[SentinelResult]:
    return [s.check() for s in applicable_sentinels(phase)]


# --------------------------------------------------------------------------- #
# Composite: a phase gate = capability AND all applicable sentinels healthy.
# --------------------------------------------------------------------------- #
@dataclass
class GateReport:
    phase: str
    capability: GateResult
    sentinels: list[SentinelResult] = field(default_factory=list)
    run_id: str | None = None  # the training run the eval refers to (report metadata)

    @property
    def passed(self) -> bool:
        return self.capability.passed and all(s.healthy for s in self.sentinels)

    def __str__(self) -> str:
        head = f"[{self.phase}] {'PASS' if self.passed else 'BLOCKED'}"
        cap = f"  capability: {'ok' if self.capability.passed else 'not met'} — {self.capability.detail}"
        lines = [head, cap]
        for s in self.sentinels:
            state = "healthy" if s.healthy else "NOT HEALTHY"
            lines.append(f"  sentinel[{s.name}]: {state} — {s.detail}")
        return "\n".join(lines)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _write_report(report: GateReport, results_dir: Path) -> Path:
    """Persist the report as JSON — the record the docs-sync step cites (committed)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path, n = results_dir / f"{report.phase}-{stamp}.json", 1
    while path.exists():
        path, n = results_dir / f"{report.phase}-{stamp}-{n}.json", n + 1
    payload = {
        "phase": report.phase,
        "passed": report.passed,
        "run_id": report.run_id,
        "written_at": stamp,
        "capability": {
            "criterion": GATES[report.phase].criterion,
            "passed": report.capability.passed,
            "metrics": report.capability.metrics,
            "seeds": report.capability.seeds,
            "detail": report.capability.detail,
        },
        "sentinels": [
            {
                "name": s.name,
                "criterion": SENTINELS[s.name].criterion,
                "healthy": s.healthy,
                "metrics": s.metrics,
                "detail": s.detail,
            }
            for s in report.sentinels
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


SHIPPED_FILE = Path(__file__).resolve().parent / "SHIPPED"


def shipped_phases(path: Path | None = None) -> list[str]:
    """Phases recorded as shipped in `bench/SHIPPED` (one per line; blank lines and
    `#` comments ignored). An unknown phase raises ValueError — the ratchet must
    fail loudly, never silently skip."""
    file = path or SHIPPED_FILE
    if not file.exists():
        return []
    phases: list[str] = []
    for lineno, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line not in GATES:
            raise ValueError(
                f"{file}:{lineno}: unknown phase {line!r} in SHIPPED; known phases: {', '.join(GATES)}"
            )
        phases.append(line)
    return phases


def run_shipped_gates(
    path: Path | None = None, results_dir: Path | None = None
) -> list[GateReport]:
    """Re-run every shipped phase's gate (the regression ratchet, P0-007)."""
    return [run_gate(phase, results_dir=results_dir) for phase in shipped_phases(path)]


def run_gate(phase: str, run_id: str | None = None, results_dir: Path | None = None) -> GateReport:
    """Run a phase's kill-gate: capability + all applicable integrity sentinels.

    The phase passes only if capability passes AND every applicable sentinel is
    healthy. Persists a JSON report under `bench/results/` (or `results_dir`).
    `run_id` names the training run the eval refers to — recorded in the report;
    zero-arg checks default to `runlog.latest_run()`. Raises KeyError (listing the
    known phases) for an unknown phase.
    """
    if phase not in GATES:
        raise KeyError(f"unknown phase {phase!r}; known phases: {', '.join(GATES)}")
    report = GateReport(
        phase=phase,
        capability=GATES[phase].check(),
        sentinels=run_sentinels(phase),
        run_id=run_id,
    )
    _write_report(report, results_dir or RESULTS_DIR)
    return report
