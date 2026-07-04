# P0-008 — One query path into knowledge; the router may decline; provenance defaults

- **Status:** done
- **Phase:** P0
- **Requirements:** R8
- **ADRs:** ADR-0004 (amend)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P0"]` (registered by P0-006)

## Goal
A single query verb for all knowledge, a router that can answer "stay parametric",
and an explicit trust default for first-party observations. Today
`SemanticMemory.read(query)` and `KnowledgeSource.query(query)` are the same
operation with different names — so the semantic store cannot be routed to without a
wrapper; `MemoryRouter.route()` *must* return a source, so ADR-0004's tier one
(answer from weights when confident) is inexpressible; and
`Observation.provenance = None` has no defined meaning under the trust rule.

## Non-goals
- No retrieval, storage, or routing implementations (P8-001).
- No poisoning/robustness handling (P8-002).
- No retrieval-as-`Option` mechanism yet — record the intent in ADR-0004 (the
  router's decision surfaces to the planner as retrieval options), implement at P8.

## Interface to satisfy
`interfaces.SemanticMemory`, `interfaces.MemoryRouter`, `types.Observation` docs;
skeletons in `memory.py` / `knowledge.py` updated to match.

## Approach (brief)
- Make the read side of semantic memory *be* a knowledge source:
  `class SemanticMemory(KnowledgeSource, Protocol)` — it gains `name` and `query()`;
  `read()` is removed; `write()` stays as its consolidation surface. One query verb
  everywhere.
- `MemoryRouter.route(query, epistemic) -> KnowledgeSource | None` — `None` means
  "confident: answer parametrically, do not retrieve". Docstring spells this out.
- Document the provenance convention on `Observation.provenance`:
  `None` ⇒ first-party sensor experience, trusted by construction; anything retrieved
  **must** carry `Provenance` (and external content defaults to `UNTRUSTED`, as now).
- Amend ADR-0004: (a) the semantic store is an internal `KnowledgeSource` — no
  parallel query path; (b) `route() -> None` is the parametric tier; (c) the intended
  P8 mechanism is retrieval surfaced as planner-selectable options.

## Acceptance criteria
- [x] `SemanticMemory` extends `KnowledgeSource`; `SemanticStore` skeleton matches
      (`name = "semantic"`, `query()` replaces `read()`); smoke tests updated
      (conformance check for both protocols).
- [x] `MemoryRouter.route` returns `KnowledgeSource | None`, documented (interface,
      skeleton, and architecture.md).
- [x] `Observation.provenance` docstring defines the `None` convention.
- [x] ADR-0004 amended (three points above, as two consequence bullets).
- [x] `make test` green, `make lint` clean.

## Test plan
- Smoke: `isinstance(SemanticStore(), interfaces.KnowledgeSource)`; skeleton
  instantiation unchanged; protocol-presence checks for the new signatures.

## Docs-sync checklist
- [x] Task Status updated; gate result recorded below.
- [x] ADR-0004 amended.
- [x] `docs/architecture.md` memory component note reflects the unified query path
      and the `None` (parametric) routing contract.
- [x] Backlog P8-001 row references the `None` (parametric) contract (was already
      annotated at planning time; verified).

## Gate result
The P0 gate is registered and shipped; the ratchet re-runs it:

```
make gate-all : [P0] PASS — ratchet ok, 1 shipped gate(s) still green
make test     : 36 passed
make lint     : All checks passed!
```

Result: **PASS** (P0 criterion met). Coverage for this task: smoke conformance
asserts `SemanticStore` satisfies **both** `SemanticMemory` and `KnowledgeSource`,
`InternalKnowledgeSource` satisfies `KnowledgeSource`, and `UncertaintyMemoryRouter`
satisfies `MemoryRouter` with the new `KnowledgeSource | None` signature.
