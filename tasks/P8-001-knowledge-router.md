# P8-001 — Three-tier memory router + uncertainty-gated retrieval-as-action

- **Status:** done
- **Phase:** P8
- **Requirements:** R8
- **ADRs:** ADR-0004 (three tiers; retrieval is an action gated by epistemic
  uncertainty; knowledge as tokens), ADR-0002 (the gating signal is the VoE
  epistemic), P0-008 (`route() -> None` is the parametric tier; one query verb)
- **Depends on:** P3-003 (memory substrate), P1-001 (the epistemic signal)
- **Phase gate:** `bench/gates.py::GATES["P8"]` — this task delivers the
  uncertainty-gated-retrieval-beats-no-retrieval half; the poisoned/low-trust
  robustness half is P8-002, so the capability records progress and the gate
  stays honestly BLOCKED until P8-002.

## Goal
Retrieval becomes an uncertainty-gated action. `SemanticStore` is a queryable
`KnowledgeSource` of facts; `UncertaintyMemoryRouter` answers from the parametric
tier when the model is confident (`route() -> None`) and retrieves when it is
epistemically uncertain (`route() -> source`). The measurable claim (ADR-0004):
gated retrieval beats model-alone by fixing exactly the queries the model is
uncertain about — while retrieving only a fraction of the time (the gating is the
point).

## Non-goals
- No poisoned/low-trust robustness, no external/tool sources (P8-002 — this task's
  source is the high-trust internal store).
- No full planner-option surfacing of retrieval (ADR-0004's end state); the router
  IS the gating a planner would consult. Minimal: demonstrate the gating improves
  prediction, the measurable core.
- No new environment: the Pendulum, with the model trained on a LIMITED region so
  it is confident there and uncertain elsewhere.

## Interface to satisfy
`interfaces.SemanticMemory` (= `KnowledgeSource` + `write`) and
`interfaces.MemoryRouter` in `prospect/memory.py` (replace the skeletons).
`SemanticStore.query` returns the nearest stored fact; `UncertaintyMemoryRouter.
route(query, epistemic)` returns `None` (parametric) below a threshold, else a
source. Retrieved items carry `Provenance` (P0-008).

## Approach (brief)
- `SemanticStore`: fact items with `content = (key, answer)` (a query key and the
  answer it holds) + provenance; `write` appends; `query(key)` returns the nearest
  fact by key distance. Knowledge-as-tokens (ADR-0004): the answer is a next-latent
  in the model's own space, a drop-in for the model's prediction.
- `UncertaintyMemoryRouter(sources, threshold)`: `route(query, epistemic)` = `None`
  when `epistemic <= threshold` (confident: parametric tier), else the source.
- Gate eval (`bench/evals/p8_knowledge.py`, `@gate_check("P8")`, run `p8` carrying
  all four sentinels): train the model on `|omega| <= REGION` only; a store holds
  correct (state,action)->next-latent facts across the FULL range. Per test query:
  the model predicts; the router gates on `pred.epistemic` (threshold = a held-out
  epistemic quantile); confident -> parametric mean, uncertain -> retrieved answer.
  Compare 1-step MSE: gated vs no-retrieval (model alone) vs always-retrieve;
  report the retrieval rate. Pass-progress: gated << no-retrieval on every seed.

## Acceptance criteria
- [ ] Implements `SemanticMemory` + `MemoryRouter`; conformance holds; empty-store
      query returns `[]`; `route` returns `None` when confident (unit-tested).
- [ ] Gated retrieval beats no-retrieval (much lower held-out MSE) on every seed,
      retrieving only a fraction of queries (the uncertain region); recorded in the
      P8 gate metrics (composite BLOCKED pending P8-002).
- [ ] All four sentinels healthy on run `p8`.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_memory.py): store write/nearest-query/empty; router gating both
  ways; retrieved items carry provenance.
- Eval: `make gate PHASE=P8` — gated-vs-no-retrieval + retrieval rate per seed.

## Docs-sync checklist
- [x] Status → `done`; gate report (BLOCKED composite, gated>no-retrieval met) below.
- [x] architecture.md memory.py/knowledge notes still accurate.
- [x] Backlog: P8-001 done; P8-002 unblocked (start here).

## Gate result
`make gate PHASE=P8` (3 seeds; ~2m):

```
[P8] BLOCKED
  capability: not met — 1-step MSE: gated 0.0079 vs no-retrieval 0.0255
    (>= x1.5 better on every seed: MET) vs always-retrieve 0.0137; retrieval
    rate 55% (gated: model where confident, retrieve where uncertain);
    robustness half pending (P8-002)
  sentinel[representation-integrity]: healthy — min per-dim std 0.852 (floor
    0.3), min effective rank 2.16 (floor 2.0)
  sentinel[uncertainty-reliability]: healthy — disagreement-vs-error rank corr
    0.63 (min 0.3), high-error-decile disagreement 18.93x median (min 1.0) on a
    mixed in-dist + OOD probe set
  sentinel[replay-fidelity]: healthy — real fraction 0.50, dream diversity 0.93,
    diversity shrink 0.93, lineage depth max 3 (cap 3), dreams stored: 0
  sentinel[option-diversity]: healthy — usage entropy 0.77, mean executed
    duration 2.94 steps, min pairwise landing d' 0.74
```

**Accuracy half MET** — uncertainty-gated retrieval cuts 1-step MSE ~3.2x vs
model-alone on every seed, and beats always-retrieve too (the gating keeps the
model's accurate prediction where it is confident and fetches a fact only in the
uncertain region, retrieving 55% of the time). All four sentinels healthy: the
limited-region model stays representationally intact, and its natural OOD
(`|omega| > REGION`) exercises the uncertainty-reliability probe. The P8 composite
is honestly **BLOCKED** — the poisoned/low-trust robustness half is P8-002 — so P8
is **not** appended to `bench/SHIPPED` yet.
