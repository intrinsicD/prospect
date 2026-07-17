# ADR-0001 — Latent predictive world model as the spine

**Status:** Accepted — amended by ADR-0014

## Context
The system must predict consequences and plan (R1), identify the right patterns
(R4), and eventually accept any modality (R6). A single mechanism can serve all
three if chosen well.

## Decision
Make a **latent predictive world model** the core: encode observations into a shared
latent, and learn dynamics that predict the next latent state and reward. Predict in
**latent space, not observation/pixel space** (JEPA/Dreamer lineage). Planning,
uncertainty, skills and knowledge all attach to this latent.

## ADR-0014 amendment — belief and experience boundary
For the E-series, the reasoning state is a versioned `BeliefState`, not an
unversioned latent point and not the replay schema. Canonical replay stores raw
`Experience`; a belief updater derives the posterior state under a named
representation/model version. A `Prediction`, `Decision`, and
`EpistemicTransition` link that belief to the action-time expectation and later
outcome.

This preserves the latent predictive spine while allowing deterministic latent
models, recurrent stochastic models, and future substrates behind one belief
contract. The legacy-v1 `LatentState` contract and its P0–P14 evidence remain
historical. The amended contract is not implemented or evidenced until `E0-001`.

## Consequences
- (+) Predicting latents forces the representation to keep controllable,
  outcome-relevant structure and drop distractors — this *is* R4, for free.
- (+) One latent hub makes any-to-any I/O (R6) an encoder/decoder swap, not a rewrite.
- The shared latent is a *contract*: downstream components couple to its
  **distribution**, not just its shape. Replacing the encoder (P6) therefore
  requires distribution-matching — distill the new codec into the incumbent latent
  space — or a budgeted retrain of everything downstream. The swap typechecks
  either way; only distillation makes it cheap. *(Amended by P0-011.)*
- (−) Long imagined rollouts compound error (see ADR-0003 for the mitigation).
- (−) Latent-space prediction is collapse-prone (a constant / low-rank encoder makes
  prediction error — and therefore the whole VoE signal — meaningless); representation
  integrity is enforced per ADR-0006 and monitored by a standing sentinel.
- Prediction targets are distributions, not point estimates (see ADR-0002).
- For new E-series work, ADR-0014 supersedes ADR-0002 and governs prediction,
  uncertainty, and lifecycle linkage.
