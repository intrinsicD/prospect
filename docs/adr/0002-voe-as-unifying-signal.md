# ADR-0002 — Prediction error (VoE) as the single unifying signal

**Status:** Accepted

## Context
We need to test whether an action has been *learned* (R3). Raw error cannot: a large
error may mean the dynamics are unlearned (epistemic, reducible) or the environment
is stochastic (aleatoric, irreducible). Several other needs (curriculum, skill trust,
re-planning, forgetting, retrieval) also reduce to "how surprised am I?".

## Decision
Compute surprise as **negative log-likelihood of the observation under a predicted
distribution**, and always **decompose uncertainty into epistemic and aleatoric**
(e.g. ensemble disagreement vs. within-member spread). Define "learned" as *low
epistemic uncertainty plus flattened learning progress*. Reuse this one signal for
all six jobs listed in `docs/architecture.md`.

## Consequences
- (+) One backbone; new requirements plug in rather than spawning bespoke modules.
- (+) A mastery **test** and a curiosity **curriculum** come from the same quantity.
- (−) Everything depends on calibrated uncertainty; calibration degrades
  off-distribution and must be monitored (a P7 concern).
- **Contract:** the world model returns `types.Prediction` — a diagonal Gaussian
  (`mean` + per-dimension `var`) with the epistemic/aleatoric split and a **concrete**
  `log_prob` — never a bare float or an unimplemented distribution. Downstream code
  must not bypass this. *(Amended by P0-001: `var` and a working `log_prob` are part
  of the contract, so surprise is computable from any `Prediction` without
  subclassing.)*
- **Contract:** the surprise signal itself is `types.Surprise` — total NLL plus its
  epistemic/aleatoric attribution — never a bare float; consumers gate on
  `.epistemic`, not the undecomposed total (the same rule as `Prediction`, one level
  up). Transitions collected while executing a skill set `Transition.option`, so
  competence is attributable per skill. *(Amended by P0-002.)*
