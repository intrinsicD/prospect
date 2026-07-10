# U-108 — Episodic → semantic consolidation pathway

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R7, R8
- **ADRs:** ADR-0004 (three-tier knowledge)
- **Depends on:** none
- **Phase gate:** the triggering gate that needs experience-distilled facts; `["P8"]`
- **Source:** `docs/sota-review-2026-07.md` U-108 · [CoALA](https://arxiv.org/abs/2309.02427)

## Trigger (promote to `ready` when…)
A **gate needs facts distilled from the agent's own experience** rather than
harness-written — e.g. a use-case phase where the semantic store must be *populated by the
agent* (consolidate recurring episodic experience into semantic facts, prune stale ones),
not pre-seeded. The **upgrade-triggers** workflow step checks: if a new phase's task
requires the agent to write its own semantic tier, promote. Today facts are written
directly by the harness (memory.py `SemanticStore.write`), which is correct for the
current gates — a consolidation process is a real 2025-practice gap but building it before
a gate demands it violates the minimal-implementation rule (review RQ4).

## Goal
When triggered: add a consolidation process that distills recurring/high-value episodic
experience into semantic facts (key = latent+action, answer = next-latent) and
prunes/decays stale facts — moving the episodic→semantic handoff into the agent, as the
LLM-agent-memory literature (CoALA) does with a background consolidation step.

## Non-goals
- No LLM memory-graph machinery (PageRank, entity extraction) — not transferable
  (review SKIP); the transferable principle is *consolidation*, not the graph.
- Not before the trigger — direct harness writes are the correct minimal choice now.

## Interface to satisfy (when promoted)
A consolidation routine (harness-side or a `memory` method) that reads the episodic buffer
and writes distilled `KnowledgeItem`s into `SemanticStore` with provenance; a decay/prune
rule for stale facts. `SemanticMemory`/`KnowledgeSource` protocols unchanged.

## Approach (brief, when promoted)
- Batched/background: every N episodes, cluster recurring transitions, write cluster
  summaries as facts, decay unused facts — the standard episodic→semantic consolidation.

## Acceptance criteria (when promoted)
- [ ] Consolidation populates the semantic store from experience; retrieval over the
      self-consolidated store meets the triggering gate.
- [ ] Poisoning/trust guarantees (P8-002) preserved; `make gate-all` green; clean checks.

## Test plan (when promoted)
- Unit: consolidation writes facts that answer seen queries; stale facts decay out.
- Eval: the triggering use-case gate + `make gate PHASE=P8` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0004: record the consolidation pathway and its trigger; cite CoALA.
- [ ] `docs/sota-review-2026-07.md`: note U-108 outcome.

## Gate result
<deferred — no gate until promoted>
