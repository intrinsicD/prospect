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
- Consumers can conflict over the signal's *sign* — planning avoids epistemic
  uncertainty while curiosity seeks it; that arbitration is decided in ADR-0007
  (mode-dependent, curriculum-owned). And under distribution shift the signal is
  ambiguous across consumers: the forgetting detector (job 5) and the retrieval
  trigger (job 6) fire together, yet "I forgot", "the world changed", and "I'm
  off-distribution, so the uncertainty estimate itself is unreliable" are three
  different correct responses to one scalar. Disambiguation is expected to need
  context beyond the scalar (which skill, which regime) — a named P7 concern, not
  assumed away. *(Amended by P0-010.)*
- **Forgetting detection keys on prediction ERROR, not epistemic** (P7-001). The
  natural reading — "forgetting = epistemic rising on a mastered skill" — fails
  under distribution shift because the ensemble is often *confidently wrong*: its
  members agree on a wrong answer, so epistemic (disagreement) stays low even as
  the skill decays. Measured: an epistemic-keyed detector never fired on real
  continual-learning forgetting. `CompetenceMonitor.is_forgetting` therefore latches
  a skill's *prediction error* at mastery and fires when it rises; mastery still
  keys on epistemic (learned = low uncertainty). This resolves the P0-010-flagged
  forgetting-under-shift concern for the "I forgot" case. *(Amended by P7-001.)*
- **Epistemic is distance-aware, not ensemble-disagreement alone** (P9-005). The
  *confidently-wrong* failure has a second consequence beyond forgetting: ensemble
  disagreement under-detects out-of-distribution inputs, because the tanh encoder
  saturates and squashes far-away inputs into the seen latent region, so the members
  (sharing that encoder) agree even where the model is wrong. Measured: on a second
  environment (PointMass) epistemic rose only 1.75x out-of-region while error rose 10x,
  and the uncertainty-reliability signal did not generalize. Fix: `encode` computes a
  pre-encoder OOD score — the standardized input's excess energy over the training
  distribution's unit variance, measured *before* the saturating encoder — carried on
  the `LatentState`, and `predict` scales the epistemic scalar by `1 + w·ood`. In-
  distribution the score is ~0 (epistemic unchanged, self-calibrated gates preserved);
  out-of-distribution it rises by construction. Measured after the fix: OOD epistemic
  rise 1.75x→7.85x and epistemic-vs-error rank correlation 0.52→0.80 on PointMass; the
  uncertainty signal now generalizes (P9 gate). Only latents from a real `encode(obs)`
  carry the score; synthesized planning-rollout latents (`ood=None`) use ensemble
  disagreement alone, and `var`/`log_prob` stay the ensemble's calibrated total —
  distance-awareness is added to the *reducible-uncertainty scalar*, not the likelihood.
  *(Amended by P9-005.)*
