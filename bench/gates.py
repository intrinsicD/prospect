"""Benchmark-gated delivery with collapse sentinels and a quality floor
(ADR-0005, ADR-0006, ADR-0007).

A phase ships only when (a) its **capability** gate passes AND (b) every applicable
integrity **sentinel** is healthy AND (c) the **quality floor** is satisfied.
Capability gates ask "did the model reach the bar?"; sentinels ask "did it stay
healthy while doing so?" — because collapse (representation, uncertainty,
generative-replay, option) is typically invisible in the training loss: the trivial
solution has low loss. The floor asks a third, different question: "did capability
*regress* against a frozen probe — and if so, did we roll back?" A sentinel only
*detects*; the floor is the *actuator* the sentinels lack (ADR-0007).

Each gate/sentinel/floor has a *precise* criterion and a `check()` returning a result
object. Until the eval body exists, `check()` returns a PENDING (not-passed /
not-healthy / not-satisfied) result and does not raise, so `make gate` prints cleanly.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import nan

PHASE_ORDER = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]


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
    metric: float
    detail: str = ""


@dataclass
class Gate:
    phase: str
    goal: str
    criterion: str  # the exact, human-readable pass condition
    check: Callable[[], GateResult]


def _pending_gate(phase: str, criterion: str) -> Callable[[], GateResult]:
    def _check() -> GateResult:
        return GateResult(phase=phase, passed=False, metric=nan, detail=f"PENDING — {criterion}")

    return _check


GATES: dict[str, Gate] = {}


def _register(phase: str, goal: str, criterion: str) -> None:
    GATES[phase] = Gate(phase, goal, criterion, _pending_gate(phase, criterion))


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


# --------------------------------------------------------------------------- #
# Collapse sentinels: did the model stay healthy? (ADR-0006)
# Integrity is enforced, not hoped for — collapse hides in a good loss curve.
# --------------------------------------------------------------------------- #
@dataclass
class SentinelResult:
    name: str
    healthy: bool
    metric: float
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
        return SentinelResult(name=name, healthy=False, metric=nan, detail=f"PENDING — {criterion}")

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


def applicable_sentinels(phase: str) -> list[Sentinel]:
    """The integrity sentinels active by `phase` (those whose applies_from <= phase)."""
    return [s for s in SENTINELS.values() if _phase_at_least(phase, s.applies_from)]


def run_sentinels(phase: str) -> list[SentinelResult]:
    return [s.check() for s in applicable_sentinels(phase)]


# --------------------------------------------------------------------------- #
# Quality floor: did capability regress against a frozen probe? (ADR-0007)
# A sentinel *detects* collapse and reports health; a floor *measures regression
# against a frozen, held-out probe and acts* (restore the best-known checkpoint
# from the vault). It is the actuator the sentinels lack, and the enabling
# guarantee for continual improvement (R7). Active from P1 — put the floor down
# before the system starts changing itself.
# --------------------------------------------------------------------------- #
@dataclass
class FloorResult:
    name: str
    satisfied: bool
    metric: float
    detail: str = ""


@dataclass
class Floor:
    name: str
    guards: str  # the silent regression / failure this floor prevents
    applies_from: str  # phase from which the floor is enforced
    criterion: str  # the precise "satisfied" condition
    check: Callable[[], FloorResult]


def _pending_floor(name: str, criterion: str) -> Callable[[], FloorResult]:
    def _check() -> FloorResult:
        return FloorResult(name=name, satisfied=False, metric=nan, detail=f"PENDING — {criterion}")

    return _check


FLOORS: dict[str, Floor] = {}


def _register_floor(name: str, guards: str, applies_from: str, criterion: str) -> None:
    FLOORS[name] = Floor(name, guards, applies_from, criterion, _pending_floor(name, criterion))


_register_floor(
    "quality-floor",
    "silent regression on a frozen probe (a self-improving loop optimizing a failure)",
    "P1",
    "Every eval, the component is scored on a FROZEN, hashed, never-trained-on probe "
    "with >=2 uncorrelated metrics (at least one calibrated NLL — no single gameable "
    "number). Best-so-far weights + auxiliary state are stored content-addressed in a "
    "vault; a regression beyond a per-metric tolerance on ANY metric restores the best "
    "checkpoint and lowers the LR. A checkpoint is floor-eligible ONLY if its ADR-0006 "
    "sentinels are healthy, so a collapsed-but-low-loss model can never be promoted. A "
    "phase passes only if the current best has not regressed.",
)


def applicable_floors(phase: str) -> list[Floor]:
    """The quality floors enforced by `phase` (those whose applies_from <= phase)."""
    return [f for f in FLOORS.values() if _phase_at_least(phase, f.applies_from)]


def run_floors(phase: str) -> list[FloorResult]:
    return [f.check() for f in applicable_floors(phase)]


# --------------------------------------------------------------------------- #
# Composite: a phase gate = capability AND all applicable sentinels healthy AND
# the quality floor satisfied (ADR-0005, ADR-0006, ADR-0007).
# --------------------------------------------------------------------------- #
@dataclass
class GateReport:
    phase: str
    capability: GateResult
    sentinels: list[SentinelResult] = field(default_factory=list)
    floors: list[FloorResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.capability.passed
            and all(s.healthy for s in self.sentinels)
            and all(f.satisfied for f in self.floors)
        )

    def __str__(self) -> str:
        head = f"[{self.phase}] {'PASS' if self.passed else 'BLOCKED'}"
        cap = f"  capability: {'ok' if self.capability.passed else 'not met'} — {self.capability.detail}"
        lines = [head, cap]
        for s in self.sentinels:
            state = "healthy" if s.healthy else "NOT HEALTHY"
            lines.append(f"  sentinel[{s.name}]: {state} — {s.detail}")
        for fl in self.floors:
            state = "satisfied" if fl.satisfied else "NOT SATISFIED"
            lines.append(f"  floor[{fl.name}]: {state} — {fl.detail}")
        return "\n".join(lines)


def run_gate(phase: str) -> GateReport:
    """Run a phase's kill-gate: capability + applicable integrity sentinels + floor.

    The phase passes only if capability passes AND every applicable sentinel is
    healthy AND the quality floor is satisfied. Raises KeyError for an unknown phase.
    """
    return GateReport(
        phase=phase,
        capability=GATES[phase].check(),
        sentinels=run_sentinels(phase),
        floors=run_floors(phase),
    )
