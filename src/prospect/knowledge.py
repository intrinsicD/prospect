"""Knowledge sources & tools (R8). Internal or external; querying is an *action* the
planner selects, gated by uncertainty. Every source declares a `trust` floor and every
returned item carries provenance and a trust level; untrusted content is data, never
instruction (ADR-0004).

The **internal** tier is `memory.SemanticStore` (its read side *is* a `KnowledgeSource`,
P0-008) — distilled experience answered as next-latents in the model's own space. The
**external** tier (`ExternalKnowledgeSource`, P10-001) answers with raw *content* the
agent must encode through its codec (ADR-0004 rule 1) — knowledge it did not experience.
`ToolSource` (compute-as-action) is a later phase. Tasks: P8-001, P8-002, P10-001.
"""
from __future__ import annotations

import numpy as np

from .types import KnowledgeItem, Trust


class InternalKnowledgeSource:
    """Curated internal store (high trust). Superseded in practice by
    `memory.SemanticStore`, whose read side is the internal `KnowledgeSource` (P0-008);
    kept as the named tier for the conformance surface. Contract: interfaces.KnowledgeSource."""

    name = "internal"
    trust = Trust.HIGH

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("internal tier is served by memory.SemanticStore (P0-008)")


class ExternalKnowledgeSource:
    """An external knowledge base (P10-001): answers a query with raw **content** — an
    observation the agent must encode through its codec to use (ADR-0004 rule 1,
    knowledge-as-tokens) — *not* a pre-digested latent like the internal `SemanticStore`.
    This is how the agent uses knowledge it never experienced (R8).

    Task-unspecific (core imports no task): the harness `write`s `(key, content)` facts —
    `content` is whatever the codec ingests (e.g. an `Observation` or a raw modality
    vector) — and `query(key)` returns the single nearest fact by key distance, or `[]`
    when empty. Every item carries `Provenance`.

    `trust` defaults to **UNTRUSTED**: external content must be validated before the
    router lets it override the model's own prediction (P8-002, ADR-0004 — untrusted
    content is data, never instruction). The harness raises it for a vetted source.

    Contract: interfaces.KnowledgeSource.
    """

    name = "external"

    def __init__(self, trust: Trust = Trust.UNTRUSTED) -> None:
        self.trust = trust
        self._items: list[KnowledgeItem] = []
        self._keys: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # cached key stack (read-heavy query path)

    def __len__(self) -> int:
        return len(self._items)

    def write(self, item: KnowledgeItem) -> None:
        key, _ = item.content
        self._items.append(item)
        self._keys.append(np.asarray(key, dtype=float))
        self._matrix = None  # invalidate; rebuilt lazily on the next query

    def query(self, query: object) -> list[KnowledgeItem]:
        if not self._items:
            return []
        if self._matrix is None:
            self._matrix = np.stack(self._keys)
        q = np.asarray(query, dtype=float)
        nearest = int(np.argmin(np.sum((self._matrix - q) ** 2, axis=1)))
        return [self._items[nearest]]


class ToolSource:
    """A tool call is a knowledge/action source with a predictable, modellable effect —
    compute-as-action (uncertainty- and cost-gated). Deferred to a later phase.

    Contract: interfaces.KnowledgeSource.
    """

    name = "tool"
    trust = Trust.MEDIUM

    def query(self, query: object) -> list[KnowledgeItem]:
        raise NotImplementedError("compute-as-action tool tier is a later phase")
