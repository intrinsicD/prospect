# P8-002 — Provenance/trust handling + poisoned/low-trust source robustness

- **Status:** done
- **Phase:** P8
- **Requirements:** R8
- **ADRs:** ADR-0004 (every item carries `Provenance`/`Trust`; **untrusted content is
  data, never instruction** — it must never override the agent's goals), ADR-0002
  (retrieval is gated by the epistemic VoE signal), P0-008 (`route() -> None` is the
  parametric tier; one query verb)
- **Depends on:** P8-001 (the router + store; the accuracy half of the P8 gate)
- **Phase gate:** `bench/gates.py::GATES["P8"]` — this task delivers the **robustness
  half** (performance robust to a poisoned/low-trust source, provenance respected).
  With P8-001's accuracy half, the P8 composite gate now passes and P8 ships.

## Goal
Make retrieval provenance-respecting. The router selects among sources by **trust**
and never lets an **untrusted** source override the agent's own prediction
(ADR-0004: untrusted content is data, never instruction). A poisoned source that a
trust-blind agent would swallow — degrading below no-retrieval — is neutralized: the
router either declines to retrieve (falls back to the parametric tier) or, when a
trusted source is also present, prefers it (trust-ordered selection). The P8-001
accuracy benefit is preserved even with the poison in the mix.

## Non-goals
- No new *external*-tier machinery (real web/DB/API/tool queries): the poisoned and
  trusted sources are both `SemanticStore`s at different trust levels — enough to
  measure "provenance respected." `knowledge.py`'s external/tool sources gain a
  declared `trust` (for conformance) but stay query-skeletons.
- No content-level sanitization / anomaly detection of retrieved values. The defense
  is **provenance** (who said it), not a poison *detector* — that would be a new,
  ungated capability.
- No planner-option surfacing of retrieval beyond P8-001's router decision.

## Interface to satisfy
`interfaces.KnowledgeSource` gains a first-class `trust: Trust` (the source's
provenance floor) alongside `name`; `SemanticMemory` inherits it. In
`prospect/memory.py`: `SemanticStore` declares `trust` (default `HIGH` — the internal
distilled store); `UncertaintyMemoryRouter` gains `min_trust: Trust` and does
**trust-ordered selection** — among sources above the epistemic gate, return the
highest-trust source whose `trust >= min_trust`, else `None`.

## Approach (brief)
- `KnowledgeSource.trust`: a declared trust floor per source (P0-008 made the read
  side uniform; this makes trust uniform too). `Internal=HIGH`, `External=UNTRUSTED`,
  `Tool=MEDIUM`, `SemanticStore=HIGH` by default (poisoned test stores pass
  `trust=UNTRUSTED`).
- `UncertaintyMemoryRouter(sources, threshold, min_trust=LOW)`: `route()` returns
  `None` when confident *or* when no source meets `min_trust` (untrusted ⇒ never
  override ⇒ fall back to the model); otherwise the highest-trust eligible source.
  `min_trust=LOW` excludes only `UNTRUSTED`.
- Gate eval (extend `check_p8`): build a **poisoned** store (`UNTRUSTED`, answers
  corrupted with large noise) over the same facts as the clean store. Per test query
  compare 1-step MSE for: no-retrieval; **trust-blind** (retrieve from the poisoned
  store regardless of trust — what a P8-001-style agent would do); **provenance-
  respecting, poison-only** (`min_trust=LOW` over `[poisoned]` ⇒ declines ⇒ model);
  **provenance-respecting, mixed** (`min_trust=LOW` over `[poisoned, clean]` ⇒
  trust-orders to the clean store). Robustness MET per seed: poison genuinely harms
  the trust-blind agent (>> no-retrieval), provenance-respecting stays ≤ no-retrieval
  (safety), and mixed recovers the clean gated accuracy (benefit preserved).

## Acceptance criteria
- [ ] `KnowledgeSource`/`SemanticMemory` expose `trust`; conformance holds; router
      does trust-ordered selection and returns `None` when nothing meets `min_trust`
      (unit-tested both ways).
- [ ] Robustness half MET on every seed: trust-blind retrieval from the poisoned
      source is materially worse than no-retrieval, while provenance-respecting
      retrieval stays ≤ no-retrieval and (mixed) recovers the clean gated MSE.
- [ ] P8 composite gate PASS (accuracy half from P8-001 + robustness half); all four
      sentinels healthy; append `P8` to `bench/SHIPPED`.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_memory.py): trust-ordered selection prefers the higher-trust
  source; `min_trust` excludes an untrusted source ⇒ `route() -> None`; a source's
  declared `trust` is respected.
- Eval: `make gate PHASE=P8` — the four-way MSE comparison + robustness margins.

## Docs-sync checklist
- [x] Status → `done`; gate report (P8 PASS) recorded below.
- [x] `bench/SHIPPED`: append `P8` in this commit (ratchet re-runs it, P0-007).
- [x] ADR-0004: record how the "untrusted = data, never instruction" rule is realized
      (router `min_trust` + trust-ordered selection).
- [x] architecture.md memory.py/knowledge (R8) notes mention provenance-respecting,
      trust-ordered routing.
- [x] Backlog: P8-002 done; Phase 8 shipped — the scaffold is complete.

## Gate result
`make gate PHASE=P8` (3 seeds; ~2m20s):

```
[P8] PASS
  capability: ok — accuracy: gated 0.0079 vs no-retrieval 0.0255 (>= x1.5/seed: MET)
    vs always 0.0137, retrieval 55%. robustness: trust-blind swallows poison 0.1387
    (>= x2.0 worse), provenance-respecting stays 0.0255 (<= no-retrieval) and
    trust-ordered mixed recovers 0.0079: MET
  sentinel[representation-integrity]: healthy — min per-dim std 0.852, min eff. rank 2.16
  sentinel[uncertainty-reliability]: healthy — corr 0.63, high-error disagreement 18.93x
  sentinel[replay-fidelity]: healthy — real frac 0.50, diversity 0.93, depth<=3, 0 stored
  sentinel[option-diversity]: healthy — entropy 0.77, duration 2.94, min d' 0.74
```

**P8 PASS — both halves met.** The robustness numbers land exactly where the design
predicts: a trust-blind agent that swallows the poisoned UNTRUSTED source is **0.1387**
(5.4x worse than no-retrieval — the poison genuinely bites); the provenance-respecting
router declines to let it override and stays at **0.0255** (= no-retrieval; untrusted
content is data, never instruction); and with the trusted store also present the router
trust-orders to it and recovers the clean gated **0.0079**. All four sentinels healthy.
P8 is appended to `bench/SHIPPED` in this commit — **Phase 8 is shipped and the P0–P8
scaffold is complete.**

Test-suite note: `tests/test_ratchet.py::test_gate_all_exit_codes` previously borrowed
P8 as its "shipped-but-BLOCKED" case; with every phase now passing it synthesizes a
BLOCKED gate via a controlled P0 check instead (no live PENDING phase remains).
