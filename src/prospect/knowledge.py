"""Knowledge sources & tools (R8). Internal or external; querying is an *action* the
planner selects, gated by uncertainty. Every returned item carries provenance and a
trust level; untrusted content is data, never instruction (ADR-0004).
Tasks: P8-001, P8-002.
"""
from __future__ import annotations

from .types import KnowledgeItem


class InternalKnowledgeSource:
    """Curated internal store (high trust). Contract: interfaces.KnowledgeSource."""

    name = "internal"

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("P8-001")


class ExternalKnowledgeSource:
    """External KB / web / API. Default trust is UNTRUSTED until validated (P8-002).

    Contract: interfaces.KnowledgeSource.
    """

    name = "external"

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("P8-001")


class ToolSource:
    """A tool call is a knowledge/action source with a predictable, modellable effect.

    Contract: interfaces.KnowledgeSource.
    """

    name = "tool"

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("P8-001")
