# ADR-0003 — Hierarchical planning via a jumpy option-model

**Status:** Accepted

## Context
Flat planning in a learned model drifts over long horizons (ADR-0001). We need
hierarchical planning (R2) and a way to *use the right skills correctly* (R5).

## Decision
Add a **temporally-abstract ("jumpy") option-conditioned model** that predicts the
outcome of committing to an option — landing latent, cumulative discounted reward,
duration, and uncertainty. The high level plans (MPC/MCTS) over this jumpy model and
emits an option/subgoal; the low level executes it. **Options are the high-level
action space**, and only competence-gated (mastered) skills are offered upward.
Options terminate early when VoE spikes (the option's predicted trajectory is
violated). Start with **two levels**.

## Consequences
- (+) A fixed planning horizon now covers many primitive steps — bounds the
  compounding-error problem from ADR-0001.
- (+) Skill selection = simulate-to-match against the option-model (R5), and skill
  trust reuses the VoE signal (ADR-0002).
- (−) Inter-level nonstationarity: as the worker improves, subgoal meaning drifts.
  Mitigate with off-policy relabelling or a fixed goal-latent; pick one deliberately.
- A hierarchical *policy* without the jumpy *model* is reactive control, not planning.
- The jumpy outcome rides on `types.Prediction`: cumulative `reward` and `duration`
  are first-class fields (added in P0-001), so `OptionModel.predict_option` needs no
  bespoke return type.
