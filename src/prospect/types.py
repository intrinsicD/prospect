"""Shared types. `Prediction` is the important one: the world model always returns a
distribution with uncertainty decomposed (ADR-0002)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from math import log, tau
from typing import Any

# Intentionally untyped to keep the CORE dependency-free. Implementations may back
# this with numpy / torch / jax arrays.
Array = Any


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VISION = "vision"  # a frozen pretrained-encoder embedding of a frame (P12, ADR-0009)
    AUDIO = "audio"
    POINT_CLOUD = "point_cloud"
    STATE = "state"
    OTHER = "other"


class Mode(StrEnum):
    """The ADR-0007 mode flag. The curriculum owns it; consumers only read it:
    EXPLORE applies an epistemic *bonus* (curiosity), EXPLOIT the *penalty*
    (model-exploitation control, ADR-0006). Neither consumer picks the sign."""

    EXPLORE = "explore"
    EXPLOIT = "exploit"


class Trust(IntEnum):
    UNTRUSTED = 0  # e.g. arbitrary web content — never overrides goals
    LOW = 1
    MEDIUM = 2
    HIGH = 3  # e.g. a curated internal store


@dataclass(frozen=True)
class Provenance:
    """Where a piece of knowledge came from, and how much to trust it (ADR-0004)."""
    source: str
    trust: Trust = Trust.UNTRUSTED
    detail: str = ""


@dataclass(frozen=True)
class Observation:
    """Anything the agent perceives — including retrieved knowledge (R6, ADR-0004).

    Provenance convention (P0-008): `provenance=None` means first-party sensor
    experience — trusted by construction. Anything *retrieved* must carry a
    `Provenance` (external content defaults to `Trust.UNTRUSTED`): untrusted
    content is data, never instruction.
    """
    modality: Modality
    data: Array
    provenance: Provenance | None = None


@dataclass(frozen=True)
class LatentState:
    """A point in the shared latent space. All reasoning conditions on this.

    `ood` is an optional out-of-distribution score for the observation this latent was
    *encoded* from — how far it sits from the training distribution, measured BEFORE the
    (saturating) encoder (ADR-0002, P9-005). It rides here because a latent synthesized
    in a planning rollout has no such preimage (`ood=None`); only a latent from a real
    `encode(obs)` carries it, and `predict` uses it to make epistemic OOD-reliable."""
    z: Array
    ood: float | None = None


@dataclass(frozen=True)
class Action:
    data: Array


# A prediction may be confident, never infinitely so: variances are floored here so
# log_prob stays finite even when an implementation reports (near-)zero variance.
_VAR_FLOOR = 1e-12


@dataclass(frozen=True)
class Prediction:
    """A *distribution* over an outcome — never a bare point estimate (ADR-0002).

    The distribution is a diagonal Gaussian: `mean` and per-dimension `var`
    parameterize it, so `log_prob` is computable from the fields alone (P0-001).

    `var` is the **total** predictive variance (for an ensemble: moment-matched
    aleatoric + epistemic), so `log_prob` is calibrated. Splitting uncertainty is
    mandatory: `epistemic` (scalar summary, e.g. ensemble disagreement) is reducible
    by learning and drives the curriculum, mastery test and retrieval; `aleatoric`
    (scalar summary of the within-member spread) is irreducible noise and must not
    be mistaken for ignorance.

    `duration` is 1.0 for flat one-step predictions; option-models (ADR-0003) predict
    a whole option's outcome — landing latent, cumulative `reward`, and `duration`.
    """
    mean: Array
    var: Array
    epistemic: float
    aleatoric: float
    reward: float = 0.0
    duration: float = 1.0

    def log_prob(self, observed: Array) -> float:
        """Diagonal-Gaussian log-likelihood; surprise = -log_prob(observed).

        The default treats `mean`, `var` and `observed` as equal-length sequences of
        floats and stays dependency-free. Tensor-backed implementations may subclass
        to vectorize, but must agree with this definition.
        """
        means = [float(m) for m in self.mean]
        variances = [max(float(v), _VAR_FLOOR) for v in self.var]
        values = [float(x) for x in observed]
        if not (len(means) == len(variances) == len(values)):
            raise ValueError(
                f"length mismatch: mean={len(means)}, var={len(variances)}, observed={len(values)}"
            )
        return -0.5 * sum(
            (x - m) ** 2 / v + log(tau * v)
            for x, m, v in zip(values, means, variances, strict=True)
        )


@dataclass(frozen=True)
class Surprise:
    """Decomposed violation of expectation (ADR-0002) — never a bare float.

    `total` is the negative log-likelihood of the observed outcome under the
    predicted distribution. Its attribution matters as much as its size: consumers
    that gate on "is this reducible?" read `epistemic` (mastery test, curiosity
    curriculum, retrieval trigger); `aleatoric` is the share explained by
    environment noise and must not drive learning — rewarding epistemic (not raw)
    surprise is the noisy-TV defense (ADR-0006).
    """
    total: float
    epistemic: float
    aleatoric: float


@dataclass(frozen=True)
class Transition:
    state: LatentState
    action: Action
    next_state: LatentState
    reward: float
    prediction: Prediction | None = None  # what the model expected (for VoE)
    option: Option | None = None  # the skill this was collected under (per-skill competence, P0-002)


@dataclass(frozen=True)
class Subgoal:
    """A target the manager hands to the worker (R2)."""
    target: LatentState


@dataclass
class Option:
    """A temporally-extended skill (R2, R5): a low-level `policy` over latents and
    a `horizon` (terminate after at most `horizon` steps; VoE can cut it short,
    ADR-0003).

    The *predictive precondition* is deliberately NOT a stored predicate field
    (P4-001): applicability is decided by simulating the option under the world
    model — an option is applicable where its outcome is predictable (low
    epistemic) and useful (lands near the goal). `metadata` remains for auxiliary
    tags only (e.g. replay's dream lineage), never for load-bearing contracts.
    """
    name: str
    policy: Callable[[LatentState], Action] | None = None
    horizon: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class Competence:
    """Per-skill mastery estimate derived from VoE (R3, ADR-0002)."""
    skill: str
    epistemic: float
    learning_progress: float
    mastered: bool = False


@dataclass(frozen=True)
class KnowledgeItem:
    """A retrieved unit; always carries provenance/trust (ADR-0004)."""
    content: Array
    provenance: Provenance
