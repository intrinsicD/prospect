# ADR-0007 — Arbitration of the epistemic signal (seek vs avoid)

**Status:** Accepted

## Context
Two consumers pull the one epistemic signal (ADR-0002) in opposite directions.
Planning *avoids* high epistemic uncertainty: ADR-0006's model-exploitation control
penalizes uncertain rollouts (MBPO/MOPO) so the planner is repelled from regions the
model is wrong about. The curiosity curriculum (P3, R3/R7) *seeks* epistemic
uncertainty: it is the learning-progress signal that decides what to practice next.
Both are individually correct; unarbitrated they fight over the same quantity — a
planner that both seeks and avoids uncertainty does neither.

## Decision
The **sign applied to epistemic uncertainty is mode-dependent, and the mode is
chosen by the curriculum — never by the consumer.**

- **Exploit mode** (acting for external reward): planning applies the uncertainty
  *penalty* (ADR-0006's uncertainty-penalized rollouts).
- **Explore mode** (collecting data in order to learn): the objective applies an
  epistemic *bonus* (curiosity) — epistemic only; aleatoric noise is never rewarded
  (the noisy-TV defense, ADR-0002/0006).

One arbiter — the curriculum / learning-progress logic (P3-002) — sets the mode per
episode or rollout. The planner and the explorer read a mode flag and apply the
corresponding sign; neither owns the decision.

## Consequences
- (+) The conflict is resolved by construction: one signal, one arbiter, two modes —
  no bespoke second signal (the design-health test of ADR-0002 holds).
- (+) Planner and curriculum stay simple consumers of the mode flag.
- (−) The explore/exploit split becomes a tunable (mode schedule / budget). The P3
  gate disciplines it: curiosity-driven collection must measurably beat random
  exploration at equal budget, or the knob is not earning its keep.
- Under distribution shift the signal is ambiguous in a *second* way (forgetting vs
  world-changed vs off-distribution); that is the ADR-0002 amendment's territory and
  a named P7 concern — this ADR arbitrates only the seek-vs-avoid sign.
