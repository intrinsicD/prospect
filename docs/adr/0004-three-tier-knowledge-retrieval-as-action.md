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
- **Retrieval-as-lookup obeys the curse of dimensionality** (P9-006). Rule 2 retrieves
  the nearest fact in a continuous key space (here `concat(latent, action)`), so the
  store's *density* must scale with the key-space *dimension* or the nearest fact is too
  far to be a right answer. Measured on env #2 (PointMass, a 6-D key): a sparse store
  (1500 facts) made retrieval fail to generalize (nearest-fact error 0.021 > the model's
  own 0.017), while a dimension-adequate store (40000) made it generalize (0.0135, ~22%
  better than no-retrieval) — with the *same* latent key. This corrects P9-005's
  hypothesis that a saturating encoder corrupted the key space: the latent key is fine
  (it even beats a raw standardized-input key); the shortfall was density, not the key.
  Consequence for external tiers: a store must be provisioned dense enough for its key
  dimension, and retrieval benefit degrades gracefully (not catastrophically) as density
  falls. *(Added by P9-006.)*
- **Rule 1 exercised — external knowledge enters through the codec** (P10-001). P8's
  internal store answers with next-latents in the model's own space (digested
  experience). The external tier (`knowledge.ExternalKnowledgeSource`) answers with raw
  *content* — an observation the agent never sensed — which it must `codec.encode`
  exactly like a first-party observation (rule 1, previously stated but untested). This
  lets the agent use knowledge it cannot derive from experience: measured on a pendulum
  OOD band the model can't extrapolate, codec-ingested external content cut 1-step MSE
  3.4× vs the model alone, while corrupting the retrieved observation worsened it 50×
  (the answer demonstrably flows through the codec). Two lessons composed: the external
  KB is *complementary* (OOD-only), so misapplying it to a seen query returns an
  irrelevant fact — which makes retrieval genuinely gated (not a trivial always-query
  oracle), and requires the P9-007 **distance gate** as well as the uncertainty gate:
  **consult** external knowledge when uncertain, **trust** a retrieved fact only when it
  is close. *(Added by P10-001.)*
- **The provenance defense holds for external content too** (P10-002). A poisoned
  `UNTRUSTED` external source answering with corrupted *observations* over the same keys
  is arbitrary garbage once encoded through the codec — there is nothing to inspect. The
  P8-002 guarantee carries over unchanged: a trust-blind agent ingests it and does ~40×
  worse than no-retrieval, while the provenance-respecting router (`min_trust` floor +
  trust-ordered selection) never lets it override the model and trust-orders to a trusted
  source when one is present. The defense is *who said it*, not *what it says*.
  *(Added by P10-002.)*
- **Rule 2 exercised — tool-use as a compute-as-action** (P11-001). The third tier is a
  tool that *computes* its answer on demand (`knowledge.ToolSource`) — exact for any
  query, no store or coverage limit, but each call carries a COST (`calls` is the signal).
  So invoking it is an action gated by epistemic uncertainty AND cost: call the expensive
  exact tool only where the cheap parametric model is unreliable. Measured with an exact
  next-state oracle on the OOD band: the tool result (ingested through the codec, reusing
  rule 1) cut 1-step MSE ~200× vs the model alone; uncertainty-gating spent an equal call
  budget far better than random (it calls where the model error — hence the benefit — is
  largest, not uniformly), and is the cost sweet spot — strictly better than never-calling
  on error at strictly fewer calls than always-calling. Unlike the retrieval tiers,
  correctness is never in question (the tool is exact); the *whole* decision is *when it
  is worth calling*, which is exactly what "uncertainty-gated action" buys. *(Added by
  P11-001.)*
- **Retrieval into *planning* is distance-gated, not certainty-asserting** (P9-007).
  Rule 2 makes retrieval an action the planner selects, so a retrieved fact substitutes
  for the model's prediction *inside CEM rollouts*. Two failure modes were measured and
  fixed. (1) At rollout depth the query is an *imagined* latent that wanders far from any
  real transition (median key-distance ~7× that of a real in-coverage query), so its
  nearest fact is fiction; substituting it corrupts multi-step optimisation — the P9-002
  harmful marginal. (2) Marking a retrieved row `epistemic = 0` (certain) removed the
  ADR-0006 exploit penalty exactly in the least-reliable region, *luring* CEM into the
  retrieval seam. Rule: substitute a fact **only when it is within a reliability radius**
  (calibrated to the store's coverage, as the epistemic gate is calibrated to the model's
  scale), and carry **honest epistemic scaled by distance** (`epi × min(1, dist/radius)`),
  never 0 — reliability *is* closeness (the P9-006 insight, now gating *whether* to
  retrieve). Measured: retrieval-into-planning went from a −3.1 (and, at a stronger
  exploit penalty, −15) marginal to −0.3 (negligible, safe); composed control improved
  (−23.6 → −9.7) and the entangled exploit-penalty marginal recovered (−6.0 → −1.6). The
  1-step P8/P9-006 retrieval role (queries are *real* states, in coverage) is unchanged:
  `reliability_radius=None` keeps substitute-and-trust there. *(Added by P9-007.)*
