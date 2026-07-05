# P10-002 — External-source trust robustness (poison never overrides)

- **Status:** done
- **Phase:** P10
- **Requirements:** R8 (external knowledge, safely)
- **ADRs:** ADR-0004 (untrusted content is data, never instruction; trust-ordered routing)
- **Depends on:** P10-001 (external knowledge through the codec), P8-002 (the trust
  machinery this reuses: `min_trust` floor + trust-ordered selection)
- **Phase gate:** completes `bench/gates.py::GATES["P10"]` — the robustness half; with it
  the composite P10 gate goes PASS and P10 ships (added to `bench/SHIPPED`).

## Goal
Make the external tier safe: a poisoned / `UNTRUSTED` external source — one that answers
with corrupted *content* over the same keys — must never override the agent's own
prediction. This is P8-002's guarantee carried to content-through-the-codec: the defense
is provenance (who said it), not inspecting the content.

## Non-goals
- No new trust mechanism — reuse `UncertaintyMemoryRouter`'s `min_trust` floor +
  trust-ordered selection (P8-002). No poison *detector* (that would be a separate,
  separately-gated capability).
- No change to the P10-001 capability path.

## Interface to satisfy
No new `Protocol`. The eval builds a poisoned `ExternalKnowledgeSource(trust=UNTRUSTED)`
(corrupted observations over the same keys) and exercises three router configurations
(trust-blind, provenance-respecting, provenance-respecting + a trusted source). Sets
`robustness_met` in `check_p10`, flipping the composite gate to PASS.

## Approach (brief)
Mirror P8-002 over the content-through-codec path:
- **trust-blind** (`min_trust=UNTRUSTED`): ingests the poisoned observation → codec
  encodes garbage → markedly WORSE than no-retrieval (the poison bites).
- **provenance-respecting** (default `min_trust=LOW`): the untrusted source never clears
  the floor → `route()` returns `None` → the agent falls back to the model (no override,
  no worse than no-retrieval).
- **respecting + trusted**: trust-orders to the trusted external source → recovers the
  clean P10-001 gated accuracy.

## Acceptance criteria
- [x] Trust-blind ingestion of the poison is ≥ POISON_HARM_FACTOR × worse than
      no-retrieval (measured **1.029 vs 0.0255 — ~40×**, the codec-encoded poison bites).
- [x] Provenance-respecting stays ≤ no-retrieval (0.0255 vs 0.0255 — untrusted never
      overrides), and with a trusted source present, trust-orders and recovers the clean
      gated accuracy (**0.0076**).
- [x] `robustness_met` set; **composite P10 gate PASS**; P10 appended to `bench/SHIPPED`.
- [x] `make gate-all` (P0–P10) green; `make test` / `lint` / `typecheck` clean.

## Test plan
- Eval: `make gate PHASE=P10` — capability + robustness → PASS. `make gate-all` regression.
- (Trust-ordering + the untrusted-never-overrides invariant are already unit/sentinel
  covered by P8-002 and the gate-overfit invariants.)

## Docs-sync checklist
- [x] Status → `done`; gate result recorded below; P10-001 status note updated (composite
      now PASS).
- [x] ADR-0004: the trust guarantee extends to external content-through-codec.
- [x] BACKLOG P10-002 done; Phase 10 shipped note.

## Gate result
`make gate PHASE=P10` → **[P10] PASS** (capability + robustness), all five collapse
sentinels healthy (~2m). Robustness measured (median over 3 seeds):

| path | 1-step MSE | verdict |
|---|---|---|
| no-retrieval (model alone) | 0.0255 | baseline |
| trust-blind (ingests poison) | **1.029** | ~40× worse — the poison bites |
| provenance-respecting (untrusted excluded) | 0.0255 | never overrides (= no-retrieval) |
| respecting + trusted (trust-ordered) | **0.0076** | recovers the clean gated accuracy |

The defense is *provenance* (who said it), not inspecting the content — a poisoned
observation encoded through the codec is arbitrary garbage, so the only safe rule is to
never let an UNTRUSTED source override the model (ADR-0004). **P10 ships** (`bench/SHIPPED`
now ratchets P0–P10).
