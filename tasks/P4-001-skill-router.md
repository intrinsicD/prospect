# P4-001 — Skill library with predictive preconditions + simulate-to-select router

- **Status:** done
- **Phase:** P4
- **Requirements:** R5
- **ADRs:** ADR-0002 (skill trust = VoE), ADR-0003 (options are the high-level
  action space; only mastered skills offered upward)
- **Depends on:** P3-001 (competence gating), P1-001 (the model to simulate on)
- **Phase gate:** `bench/gates.py::GATES["P4"]`

## Goal
`SkillRouter` satisfying `interfaces.SkillLibrary`: a registry of options and a
simulate-to-select router. Selection *is* the predictive precondition: each
candidate is rolled forward under the world model for its horizon; score =
imagined-landing distance to the subgoal + accumulated epistemic. Only
competence-gated (mastered) skills are offered upward (cold start: all, so
competence can accrue). Misapplication — executing a skill other than the one
whose outcome was predicted — is flagged by a surprise spike (the VoE signal's
job #3).

## Non-goals
- No jumpy option-model or hierarchical manager (P5 — the router's flat rollout
  `simulate()` is the primitive P5's learned option-model replaces).
- No skill *learning* (policies are supplied); no new environment.

## Interface to satisfy
`prospect.interfaces.SkillLibrary` — implement in `prospect/skills.py`. Typed
`Option` fields (the P0-review promotion): `policy: Callable[[LatentState],
Action] | None` and `horizon: int` — what simulation and execution need. The
*precondition* is deliberately NOT a stored predicate: it is predictive
(computed by simulation), which is the architecture's point; `metadata` stays
for auxiliary tags only (e.g. replay's dream lineage).

## Approach (brief)
- `SkillRouter(world_model, monitor, uncertainty_weight)`: `add()` rejects
  policy-less options; `propose(state, subgoal)` filters to mastered skills
  (when a monitor is attached and anything is mastered), ranks by
  `_score = ||landing − target||² + w·Σ epistemic`; `simulate(state, option)`
  exposes the landing `Prediction` for VoE against real execution.
- Gate eval (`bench/evals/p4_skills.py`, run `p4` with P1-probes + replay
  fidelity so all active sentinels judge this phase's model): three
  constant-torque skills (left/coast/right, horizon 6) on the Pendulum;
  per test case the target is a randomly chosen skill's REAL landing; ground
  truth = the skill whose real execution lands closest (scaled obs space);
  router accuracy vs the uniform-random baseline (1/3). Misapplication: surprise
  of the chosen skill's landing prediction vs the real landing of a *different*
  skill, compared to the correctly-executed case — AUC ≥ threshold per seed.
  Executors set `Transition.option` when feeding the monitor (P0-002).

## Acceptance criteria
- [x] Implements `interfaces.SkillLibrary`; conformance holds; `add()` rejects
      policy-less options.
- [x] Ranking is correct on a transparent stub model; the epistemic term breaks
      ties toward the predictable skill (the predictive precondition,
      unit-tested).
- [x] Competence gating: only mastered skills offered; cold start offers all;
      no monitor = no gating (unit-tested).
- [x] **Gate P4 PASS:** routing accuracy 0.92/0.83/0.83 per seed (uniform
      baseline 0.33, behavioral near-ties counted) AND paired misapplication
      win rate 0.98/0.95/0.97 (min 0.9); all sentinels healthy on run `p4`;
      `P4` appended to `bench/SHIPPED` in this commit.
- [x] `make test` green (74), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_skills.py): ranking, uncertainty tie-break, gating states,
  policy-less rejection, typed Option fields.
- Eval: `make gate PHASE=P4`; then `make gate-all` (P0–P4).

## Docs-sync checklist
- [x] Status → `done`; gate report below; `bench/SHIPPED` += P4.
- [x] architecture.md skills.py note still accurate (predictive preconditions,
      simulate-to-select, competence gating — all now literal).
- [x] types.py `Option` docstring explains why the precondition is computed,
      not stored (backlog note resolved in spirit).
- [x] Backlog: P4-001 done; **Phase 4 shipped**; P5-001 unblocked (start here).

## Gate result
`make gate PHASE=P4` — PASS record `bench/results/P4-20260704T062233Z.json`:

```
[P4] PASS
  capability: ok — routing accuracy per seed [0.92, 0.83, 0.83] (uniform
    baseline 0.33, near-ties count); paired misapplication win rate per seed
    [0.98, 0.95, 0.97] (min required 0.9)
  sentinels: representation-integrity healthy (min std 0.867, rank 2.02);
    uncertainty-reliability healthy (corr 0.70); replay-fidelity healthy
```

`gate-all`: 5 shipped gates green (~2m20s).

**What it took — three diagnosed root causes, each measured before fixed:**
(1) *Compounding fog*: on the default pendulum the model's open-loop landing
error was ~10x the inter-skill separation (ADR-0001's named limiter, P5's jumpy
model is the real answer) — the P4 reference task must keep flat rollouts
informative. (2) *Action blindness*: at dt 0.15 the full torque swing moved the
one-step prediction by only ~0.31 predictive std — the NLL loss absorbed the
action into noise, leaving routing AND misapplication VoE structurally at
chance; the reference config (gravity 1, damping 0.4, dt 0.4) makes the action
resolvable (ratio 1.2). Misapplication is judged CLOSED-LOOP and PAIRED (same
start, same intended skill), the way options actually terminate per ADR-0003 —
open-loop landing comparison measured AUC ~0.56, pooled-AUC ~0.7. (3)
*Uncalibrated gating*: with the default absolute mastery threshold, some seeds
gated the truly-best skill out of `propose()` (accuracy 0.33–0.55 vs 0.77–0.85
ungated, metric ceiling 1.00); the eval calibrates `mastery_epistemic` to 4x
the model's measured held-out epistemic floor. Also: attractor-dominated envs
concentrate long-episode data — frequent wide resets keep the manifold (and the
latent rank) spread.
