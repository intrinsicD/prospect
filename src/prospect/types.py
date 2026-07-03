"""Shared types. `Prediction` is the important one: the world model always returns a
distribution with uncertainty decomposed (ADR-0002)."""
from __future__ import annotations

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
    AUDIO = "audio"
    POINT_CLOUD = "point_cloud"
    STATE = "state"
    OTHER = "other"


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
    """Anything the agent perceives — including retrieved knowledge (R6, ADR-0004)."""
    modality: Modality
    data: Array
    provenance: Provenance | None = None


@dataclass(frozen=True)
class LatentState:
    """A point in the shared latent space. All reasoning conditions on this."""
    z: Array


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

    Splitting uncertainty is mandatory: `epistemic` (scalar summary, e.g. ensemble
    disagreement) is reducible by learning and drives the curriculum, mastery test
    and retrieval; `aleatoric` (scalar summary of `var`) is irreducible noise and
    must not be mistaken for ignorance.

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
class Transition:
    state: LatentState
    action: Action
    next_state: LatentState
    reward: float
    prediction: Prediction | None = None  # what the model expected (for VoE)


@dataclass(frozen=True)
class Subgoal:
    """A target the manager hands to the worker (R2)."""
    target: LatentState


@dataclass
class Option:
    """A temporally-extended skill: initiation predicate, policy, termination (R2, R5).

    Concrete callables are supplied by the skill implementation; kept as metadata
    references here so the core stays behaviour-agnostic.
    """
    name: str
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
