# P11-001 — Compute-as-action tools (uncertainty- and cost-gated)

- **Status:** done
- **Phase:** P11
- **Requirements:** R8 (tools as a knowledge source), R1/R4 (the epistemic signal that
  gates the call)
- **ADRs:** ADR-0004 (**rule 2**: retrieval and tool-use are actions the planner selects,
  gated by uncertainty), ADR-0001 (a tool result enters through the codec, like an obs)
- **Depends on:** P10 (external knowledge through the codec — the ingestion path), P9-005
  (distance-aware epistemic — what fires the gate OOD)
- **Phase gate:** new `bench/gates.py::GATES["P11"]` — a single-task phase (like P6/P7);
  PASS ships it.

## Goal
Add the third knowledge tier: a **tool** the agent *calls* as an action. Unlike a lookup
KB (P10), a tool **computes** its answer on demand — exact for any query, with no store or
coverage limit — but each call has a **cost**. So calling it is a decision gated by
uncertainty AND cost: invoke the expensive exact tool only where the cheap parametric
model is unreliable. This is what "retrieval/tool-use are uncertainty-gated actions"
(ADR-0004 rule 2) means when the source *computes* rather than looks up.

## Non-goals
- No real external process/API — the tool is an in-process exact oracle (a simulator)
  supplied by the harness (deterministic gate). A real API adapter is later.
- No planner-integration of the call (retrieval-into-planning is P9-007's concern) — the
  gate measures the *call decision* on 1-step prediction, not multi-step rollouts.
- No faulty-tool robustness here (a possible follow-up); tools default to MEDIUM trust.

## Interface to satisfy
- Core: implement `knowledge.ToolSource` (satisfies `interfaces.KnowledgeSource`) — a
  **task-unspecific** compute-as-action adapter: it holds a harness-supplied `compute`
  (`query -> content`), `query` wraps the result with provenance and **counts the call**
  (`calls` is the cost signal). No new `Protocol`.
- Ingestion reuses P10: the tool returns an observation, ingested via `codec.encode`.

## Approach (brief)
- **Tool:** an exact next-state oracle — `compute((obs, action))` runs the true env one
  step and returns the next observation. Exact for any query (no store, no distance gate).
- **Gate:** call the tool when the model's epistemic clears the seen-region threshold.
- **Measure** (1-step latent MSE + call counts): the tool vs the model on seen/OOD, and
  four policies — never-call, always-call, uncertainty-gated, random-gated (equal budget).

## Acceptance criteria (single-task phase — PASS ships)
- [x] **Tool helps where the model is uncertain:** on OOD, tool-call MSE ≪ model-alone
      (**0.0002 vs 0.0527 — 263×**; the exact oracle computes what the model can't).
- [x] **Uncertainty spends the budget well** (load-bearing): at an equal call budget,
      uncertainty-gated beats random-gated (**0.0008 vs 0.0128 — 16×**); call rate
      concentrated on OOD (seen **5%** / OOD **97%**).
- [x] **Cost-gating is the sweet spot:** uncertainty-gated is strictly better than
      never-call (**0.0008 vs 0.0255 — 32×**) at strictly fewer calls (**53%** vs
      always-call's 100%; always MSE 0.0002).
- [x] `make gate PHASE=P11` PASS with all sentinels healthy; P11 appended to
      `bench/SHIPPED`; `make gate-all` green; `make test`/`lint`/`typecheck` clean.

## Test plan
- Unit (tests/test_knowledge.py): `ToolSource` computes via the supplied function, wraps
  provenance, counts calls; conformance to `KnowledgeSource`.
- Eval: `bench/evals/p11_tools.py::check_p11` — the three criteria; registered as P11.

## Docs-sync checklist
- [x] Status → `done`; gate result recorded below.
- [x] `knowledge.py` docstring (tool tier implemented); ADR-0004 rule 2 exercised.
- [x] R8 traceability row (+P11); roadmap P11 row; BACKLOG P11 + shipped note.

## Gate result
`make gate PHASE=P11` → **[P11] PASS**, all five collapse sentinels healthy (~2m). Median
over 3 seeds:

| criterion | measured | bar |
|---|---|---|
| tool helps — OOD tool vs model MSE | **0.0002 vs 0.0527 (263×)** | ≥ 2× |
| uncertainty spends well — gated vs random @ equal budget | **0.0008 vs 0.0128 (16×)** | ≥ 1.2× |
| — call rate seen / OOD | 5% / 97% | seen ≤ 15% |
| cost sweet spot — gated vs never-call | **0.0008 vs 0.0255 (32×)** | ≥ 2× |
| — call rate vs always-call | 53% vs 100% | < 70% |

The tool is exact everywhere (always-call MSE 0.0002 ≈ 0), so correctness is never the
question — the *whole* value is the call decision. Uncertainty-gating targets the calls
where the model error (hence the benefit) is largest, so it beats a random policy at equal
budget and is the cost sweet spot between never- and always-calling. **P11 ships**
(`bench/SHIPPED` now ratchets P0–P11).
