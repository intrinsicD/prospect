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

## Note (2026-07-02) — external evidence for simulate-to-select over a learned router
A sibling project (OmniLatent) built the obvious alternative to simulate-to-select — a
**learned key-matching router** over an expert registry with sparse top-k gating — and
measured it honestly (`docs/routing_ablation.md` there). Result: **routing did not beat
firing all experts** ("always-on") at their scale; the win was at best efficiency, not
quality, because attention-injected experts are already input-conditioned, making an
explicit router redundant *for quality*. We take this as evidence to keep skill
selection **grounded in the world model** (simulate the option-model, pick by predicted
outcome under uncertainty, gate by VoE) rather than adopting a separately-learned
router. We keep only the router's *abstention* idea — low confidence ⇒ gather evidence
(retrieve / read memory) instead of forcing a skill — which Prospect already expresses
as uncertainty-gated retrieval (ADR-0004). This note records a mechanism we chose
**not** to import and why.
