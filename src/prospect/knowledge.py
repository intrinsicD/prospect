"""Knowledge sources & tools (R8). Internal or external; querying is an *action* the
planner selects, gated by uncertainty. Every source declares a `trust` floor and every
returned item carries provenance and a trust level; untrusted content is data, never
instruction (ADR-0004).

The **internal** tier is `memory.SemanticStore` (its read side *is* a `KnowledgeSource`,
P0-008) — distilled experience answered as next-latents in the model's own space. The
**external** tier (`ExternalKnowledgeSource`, P10-001) answers with raw *content* the
agent must encode through its codec (ADR-0004 rule 1) — knowledge it did not experience.
`ToolSource` (P11-001) *computes* its answer on demand (ADR-0004 rule 2) — exact for any
query, but each call has a cost, so it is gated by uncertainty AND cost. Tasks: P8-001,
P8-002, P10-001, P11-001.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .types import KnowledgeItem, Provenance, Trust


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
    """A tool call is a knowledge/action source that **computes** its answer on demand
    (compute-as-action, ADR-0004 rule 2) rather than looking it up — exact for any query,
    with no store or coverage limit, but each call has a COST. So it is worth invoking
    only where the cheap parametric model is unreliable: tool-use is an action gated by
    uncertainty AND cost (P11-001), not a blind pipeline.

    Task-unspecific (core imports no task): the harness supplies `compute`, a
    `query -> content` function (the actual tool — a simulator, solver, API). `query`
    wraps the result with provenance and **counts the call** — `calls` is the cost signal
    the harness reads. `content` is `(query, result)`, uniform with the other tiers, so a
    computed result ingests through the codec exactly like a retrieved observation. `trust`
    defaults to MEDIUM (a vetted tool). An unconfigured `ToolSource()` (no `compute`) still
    satisfies the protocol shape but raises if queried.

    Contract: interfaces.KnowledgeSource.
    """

    name = "tool"

    def __init__(
        self, compute: Callable[[object], object] | None = None,
        trust: Trust = Trust.MEDIUM, source: str = "tool",
    ) -> None:
        self._compute = compute
        self.trust = trust
        self._source = source
        self.calls = 0

    def query(self, query: object) -> list[KnowledgeItem]:
        if self._compute is None:
            raise NotImplementedError("ToolSource needs a harness-supplied compute function")
        self.calls += 1  # the cost signal: one invocation
        return [KnowledgeItem(content=(query, self._compute(query)),
                              provenance=Provenance(source=self._source, trust=self.trust))]
