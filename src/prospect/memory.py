"""Memory tiers (R7, R8). Episodic replay + *generative* replay (rehearsal), a
semantic store, and an uncertainty-gated router over the tiers. See ADR-0004.
Tasks: P3-003 (replay), P8-001 (router).
"""
from __future__ import annotations

from .interfaces import KnowledgeSource
from .types import KnowledgeItem, Transition


class ReplayBuffer:
    """Real experience for learning; generative_replay() dreams old experience from
    the world model to rehearse skills without hoarding raw data (anti-forgetting).

    Contract: interfaces.EpisodicMemory.
    """

    def add(self, transition: Transition) -> None:
        raise NotImplementedError("P3-003")

    def sample(self, n: int) -> list[Transition]:
        raise NotImplementedError("P3-003")

    def generative_replay(self, n: int) -> list[Transition]:
        raise NotImplementedError("P3-003")


class SemanticStore:
    """Distilled facts consolidated from experience. Its read side is a
    `KnowledgeSource` (one query verb into every tier, P0-008); `write` is the
    consolidation surface. Contract: interfaces.SemanticMemory."""

    name = "semantic"

    def write(self, item: KnowledgeItem) -> None:
        raise NotImplementedError("P8-001")

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("P8-001")


class UncertaintyMemoryRouter:
    """Route a query to a tier by current epistemic uncertainty: answer from the model
    when confident — `route()` returns `None`, the parametric tier (P0-008) — and
    retrieve when uncertain (retrieval-as-action).

    Contract: interfaces.MemoryRouter.
    """

    def route(self, query: object, epistemic: float) -> KnowledgeSource | None:
        raise NotImplementedError("P8-001")
