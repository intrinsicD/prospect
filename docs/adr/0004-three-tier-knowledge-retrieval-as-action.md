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
