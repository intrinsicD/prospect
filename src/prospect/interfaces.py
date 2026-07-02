"""The contracts. Every component satisfies one of these `Protocol`s (structural
typing — implementations need not inherit). These are the seams an agent fills in."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from collections.abc import Sequence

from .types import (
    Action,
    Competence,
    KnowledgeItem,
    LatentState,
    Observation,
    Option,
    Prediction,
    Subgoal,
    Transition,
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
    One object, many jobs: surprise, mastery, forgetting."""

    def surprise(self, prediction: Prediction, observed: LatentState) -> float: ...
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
class SemanticMemory(Protocol):
    """Distilled, queryable facts consolidated from experience (R7, R8)."""

    def write(self, item: KnowledgeItem) -> None: ...
    def read(self, query: object) -> list[KnowledgeItem]: ...


@runtime_checkable
class KnowledgeSource(Protocol):
    """Internal or external knowledge / tools (R8, ADR-0004). Querying is an *action*
    the planner selects, gated by uncertainty; every item carries provenance/trust."""

    name: str

    def query(self, query: object) -> list[KnowledgeItem]: ...


@runtime_checkable
class MemoryRouter(Protocol):
    """Chooses which tier to consult (parametric / internal / external) given the
    query and current epistemic uncertainty (R8, ADR-0004)."""

    def route(self, query: object, epistemic: float) -> KnowledgeSource: ...
