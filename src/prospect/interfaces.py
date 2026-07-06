"""The contracts. Every component satisfies one of these `Protocol`s (structural
typing — implementations need not inherit). These are the seams an agent fills in."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import (
    Action,
    Array,
    Competence,
    KnowledgeItem,
    LatentState,
    Observation,
    Option,
    Prediction,
    Subgoal,
    Surprise,
    Transition,
    Trust,
)


@runtime_checkable
class Codec(Protocol):
    """Any input -> shared latent, latent -> any output (R6). Retrieved knowledge
    enters through the same encoder (knowledge-as-tokens, ADR-0004)."""

    def encode(self, obs: Observation) -> LatentState: ...
    def decode(self, state: LatentState, query: object) -> object: ...


@runtime_checkable
class WorldModel(Protocol):
    """Latent dynamics (R1, R4). Predicts a *distribution* over the next latent state
    and reward, with uncertainty decomposed (ADR-0002)."""

    def predict(self, state: LatentState, action: Action) -> Prediction: ...
    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]: ...


@runtime_checkable
class Learner(Protocol):
    """A component the harness can train (P0-003) — the uniform training seam.

    `update()` consumes a batch of transitions and returns a metrics dict (loss
    terms + integrity stats); the harness logs these to the run-metrics artifact
    (P0-005) that the ADR-0006 sentinels read. Kept separate from the inference
    contracts so consumers like the planner keep a narrow view. Expected to be
    satisfied, alongside their primary contract, by: the world model (P1), the
    option model (P5), the codec (P6)."""

    def update(self, batch: Sequence[Transition]) -> dict[str, float]: ...


@runtime_checkable
class ObservationLearner(Protocol):
    """Learn dynamics from ACTION-FREE observation (R7, R8, ADR-0010): infer a latent
    action between consecutive observations and predict forward from it — learning to
    predict by watching, and recovering the action structure without ever seeing an
    action. `observe()` is the action-free training verb (no `Transition`, no reward). The
    decorrelation-for-identifiability detail lives in the implementation (ADR-0010).
    Satisfied by `observation.LatentActionModel`."""

    def infer_action(self, obs: Array, next_obs: Array) -> Array: ...
    def predict(self, obs: Array, latent_action: Array) -> Array: ...
    def observe(self, obs: Array, next_obs: Array) -> dict[str, float]: ...


@runtime_checkable
class OptionModel(Protocol):
    """Temporally-abstract 'jumpy' model (R2, ADR-0003): the outcome of committing to
    an option — landing latent, cumulative discounted reward, duration, uncertainty.
    This is what makes high-level *planning* (not just reaction) possible."""

    def predict_option(self, state: LatentState, option: Option) -> Prediction: ...


@runtime_checkable
class Planner(Protocol):
    """Flat planning in imagination (R1): MPC/CEM/search over the world model."""

    def plan(self, state: LatentState, goal: Subgoal | None = None) -> Action: ...


@runtime_checkable
class HierarchicalPlanner(Protocol):
    """Two-level planning (R2, ADR-0003): a manager searches over the OptionModel and
    emits an option; the worker executes it; VoE terminates it early on surprise."""

    def plan_option(self, state: LatentState) -> Option: ...
    def should_terminate(self, transition: Transition) -> bool: ...


@runtime_checkable
class CompetenceMonitor(Protocol):
    """Violation of expectation as the unifying signal (R3, R7, ADR-0002).
    One object, many jobs: surprise, mastery, forgetting.

    surprise() returns a decomposed `Surprise` — never a bare float (P0-002):
    consumers gate on `.epistemic`, not the undecomposed total. Transitions
    collected under a skill carry `Transition.option` for per-skill attribution."""

    def surprise(self, prediction: Prediction, observed: LatentState) -> Surprise: ...
    def update(self, transition: Transition) -> None: ...
    def competence(self, skill: str) -> Competence: ...
    def is_mastered(self, skill: str) -> bool: ...
    def is_forgetting(self, skill: str) -> bool: ...


@runtime_checkable
class SkillLibrary(Protocol):
    """Options with predictive preconditions; select by simulate-to-match (R5).
    Only competence-gated (mastered) skills are offered to the high level."""

    def propose(self, state: LatentState, subgoal: Subgoal) -> list[Option]: ...
    def add(self, option: Option) -> None: ...


@runtime_checkable
class EpisodicMemory(Protocol):
    """Experience buffer + *generative* replay for rehearsal (R7, ADR-0004)."""

    def add(self, transition: Transition) -> None: ...
    def sample(self, n: int) -> list[Transition]: ...
    def generative_replay(self, n: int) -> list[Transition]: ...


@runtime_checkable
class KnowledgeSource(Protocol):
    """Internal or external knowledge / tools (R8, ADR-0004). Querying is an *action*
    the planner selects, gated by uncertainty; every item carries provenance/trust.

    `trust` is the source's provenance *floor* — the trust level the router uses for
    trust-ordered selection (P8-002). It is the source-level counterpart of each
    item's `Provenance.trust`; an `UNTRUSTED` source is data the router must never let
    override the agent's own prediction (ADR-0004: untrusted content is never
    instruction)."""

    name: str
    trust: Trust

    def query(self, query: object) -> list[KnowledgeItem]: ...


@runtime_checkable
class SemanticMemory(KnowledgeSource, Protocol):
    """Distilled facts consolidated from experience (R7, R8). The read side *is* a
    `KnowledgeSource` (P0-008): one query verb into every knowledge tier, so the
    router selects the semantic store like any other source — no parallel query
    path. `write` is its separate consolidation surface."""

    def write(self, item: KnowledgeItem) -> None: ...


@runtime_checkable
class ModeArbiter(Protocol):
    """The explore/exploit arbitration seam the composition root reads (ADR-0007,
    P9-001). `uncertainty_coefficient()` is the signed coefficient applied to
    per-step epistemic uncertainty — positive penalises it (exploit-mode planning),
    negative rewards it (explore-mode collection). The arbiter (the curriculum) owns
    the sign; the agent and planner never pick it. Satisfied by
    `voe.LearningProgressCurriculum`."""

    def uncertainty_coefficient(self) -> float: ...


@runtime_checkable
class UncertaintyTunable(Protocol):
    """A planner whose per-step epistemic coefficient the composition root can set
    from the `ModeArbiter` (P9-001). Kept separate from `Planner` because not every
    planner exposes the knob; the agent narrows to this before writing it. Satisfied
    by `planning.FlatPlanner`."""

    uncertainty_penalty: float


@runtime_checkable
class MemoryRouter(Protocol):
    """Chooses which tier to consult (parametric / internal / external) given the
    query and current epistemic uncertainty (R8, ADR-0004).

    Returning `None` means: confident — answer parametrically (from the model's
    weights), do not retrieve (P0-008). At P8 the router's decision surfaces to the
    planner as retrieval *options* (retrieval-as-action, ADR-0004 rule 2).

    Selection is provenance-respecting (P8-002): a source is eligible only if its
    `trust` clears the router's floor, and among eligible sources the highest-trust
    one wins — so an untrusted source can never override the agent's own prediction
    (returns `None` instead)."""

    def route(self, query: object, epistemic: float) -> KnowledgeSource | None: ...
