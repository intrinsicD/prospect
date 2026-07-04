# ADR-0004 — Three-tier knowledge; retrieval/tools as uncertainty-gated actions

**Status:** Accepted

## Context
The system must use internal and external knowledge bases to be useful for any use
case (R8), and improve over time (R7). Generality must not require retraining per
domain.

## Decision
**Decouple reasoning (in weights) from knowledge (in swappable stores).** Knowledge
lives in three tiers: parametric (the world model), internal non-parametric (episodic
experience + distilled semantic facts + the skill library), and external (docs, DBs,
APIs, the web, tools). Two integration rules:
1. **Knowledge is just more tokens through the codec** — a retrieved item conditions
   the latent exactly like an observation (ADR-0001, R6). No bespoke knowledge module.
2. **Retrieval and tool-use are actions the planner selects**, triggered by epistemic
   uncertainty (the VoE signal, ADR-0002). "Retrieve X" can be an option (ADR-0003).

Every knowledge item carries `Provenance` and a `Trust` level. **Untrusted content is
data, never instruction** — it must never override the agent's goals.

## Consequences
- (+) One system serves arbitrary use cases by attaching KBs, not retraining.
- (+) Uncertainty decides when to look something up — retrieval stops being a blind
  pipeline and becomes part of planning.
- (−) Retrieval quality dominates; stale/poisoned/low-trust sources are attack
  surface. Provenance, trust levels and robustness tests are mandatory (P8 gate).
- (−) Consolidation (episodic → semantic) shares machinery with R7's anti-forgetting.
- The internal semantic store's read side *is* a `KnowledgeSource` — one query verb
  into every tier, no parallel query path — and `MemoryRouter.route()` may return
  `None`: the parametric tier, confident enough to answer from weights without
  retrieving. *(Amended by P0-008.)*
- Intended P8 mechanism for rule 2: the router's decision surfaces to the planner as
  retrieval *options* — retrieval is selected in planning, not dispatched behind its
  back. `Observation.provenance=None` denotes first-party sensor experience (trusted
  by construction); every retrieved item carries explicit provenance.
  *(Amended by P0-008.)*
- Realization of "untrusted content is data, never instruction" (P8-002): each
  `KnowledgeSource` declares a `trust` floor, and `UncertaintyMemoryRouter` does
  **trust-ordered selection** with a `min_trust` floor (default `LOW`, i.e. excludes
  only `UNTRUSTED`). Among sources above the epistemic gate it returns the
  highest-trust one; if none clears the floor it returns `None` — the untrusted source
  never overrides the agent's own prediction (the agent falls back to the parametric
  tier). Measured on the P8 gate: a trust-blind agent that retrieves from a poisoned
  `UNTRUSTED` store does markedly *worse* than no-retrieval, while the
  provenance-respecting router stays at no-retrieval and, with a trusted store also
  present, trust-orders to it and recovers the clean gated accuracy. The defense is
  *provenance* (who said it), not content inspection — a poison *detector* would be a
  new, separately-gated capability. *(Added by P8-002.)*
