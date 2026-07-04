"""Knowledge sources & tools (R8). Internal or external; querying is an *action* the
planner selects, gated by uncertainty. Every source declares a `trust` floor and
every returned item carries provenance and a trust level; untrusted content is data,
never instruction (ADR-0004). These external-tier sources are query-skeletons; the P8
gate exercises the trust machinery through `SemanticStore` at graded trust levels.
Tasks: P8-001, P8-002.
"""
from __future__ import annotations

from .types import KnowledgeItem, Trust


class InternalKnowledgeSource:
    """Curated internal store (high trust). Contract: interfaces.KnowledgeSource."""

    name = "internal"
    trust = Trust.HIGH

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("external-tier query is out of P8 scope")


class ExternalKnowledgeSource:
    """External KB / web / API. Trust is UNTRUSTED until a source is validated: the
    router will not let it override the model's own prediction (P8-002).

    Contract: interfaces.KnowledgeSource.
    """

    name = "external"
    trust = Trust.UNTRUSTED

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("external-tier query is out of P8 scope")


class ToolSource:
    """A tool call is a knowledge/action source with a predictable, modellable effect.

    Contract: interfaces.KnowledgeSource.
    """

    name = "tool"
    trust = Trust.MEDIUM

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("external-tier query is out of P8 scope")
