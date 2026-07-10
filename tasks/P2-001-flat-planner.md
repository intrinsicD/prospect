# P2-001 — Flat planner: MPC/CEM in imagination

- **Status:** done
- **Phase:** P2
- **Requirements:** R1
- **ADRs:** ADR-0001, ADR-0002, ADR-0006 (uncertainty-penalized rollouts), ADR-0007
  (exploit-mode sign)
- **Depends on:** P1-001
- **Phase gate:** `bench/gates.py::GATES["P2"]`

## Goal
A `Planner` that acts by imagining: cross-entropy-method (CEM) optimization over
action sequences rolled out in the world model's latent space, executed receding-
horizon (plan, act on the first action, re-plan). Planning must demonstrably beat
reaction: better return than a model-free baseline given the SAME number of
environment steps for learning.

## Non-goals
- No hierarchy, options, or goal-conditioned planning (P5 — `plan()`'s `goal`
  parameter is accepted and ignored until then, documented).
- No exploration logic — the planner is the ADR-0007 *exploit-mode* consumer:
  epistemic uncertainty is a penalty here; the bonus sign belongs to the
  curriculum (P3-002).
- No new environment: the Pendulum reference task from P1 is the testbed.

## Interface to satisfy
`prospect.interfaces.Planner` — implement `FlatPlanner` in `prospect/planning.py`
(replace the skeleton). Constructor takes the `WorldModel` to plan over. Uses the
model's `TrajectoryWorldModel.predict_member_batch` specialization when offered
(U-001), then vectorized `predict_batch`, falling back to the protocol's per-sample
`predict()` so any narrow `WorldModel` still works.

## Approach (brief)
- CEM: sample K candidate action sequences from a per-timestep Gaussian, clip to
  action bounds, roll each out in imagination (TS∞ member trajectories since
  U-001), score by discounted imagined reward **minus λ·propagated epistemic per
  step** (ADR-0006 model-exploitation control, exploit-mode per ADR-0007), refit
  the Gaussian to the elites, iterate.
- Receding horizon with warm start: keep the elite mean, shift by one step for the
  next `plan()` call; `reset()` clears it between episodes.
- Gate eval (`bench/evals/p2_planner.py`): equal-budget comparison on Pendulum.
  * MBRL agent: BUDGET random env steps → train `FlatWorldModel` (P1 recipe,
    probes logged to run `p2` so the P1-era sentinels judge THIS run's model) →
    CEM-plan on fresh evaluation episodes.
  * Model-free baseline: CEM-ES direct policy search (small tanh-MLP policy),
    fitness = episode return on the real env, every rollout counted against the
    SAME budget.
  * Random policy reported as the floor. Identical evaluation episodes (shared
    seeds) for all three.

## Acceptance criteria
- [x] Implements `interfaces.Planner`; typed conformance assertion updated
      (`FlatPlanner(FlatWorldModel())`).
- [x] The epistemic penalty is real: with λ=1 the planner avoids the
      high-reward-but-high-uncertainty arm a λ=0 planner chases (unit-tested).
- [x] CEM optimizes: finds the concave-reward optimum (0.7 ± 0.15) through the
      per-sample protocol fallback (unit-tested).
- [x] **Gate P2:** planner beats the model-free baseline AND the random floor on
      **every seed**, not just the median — returns (planner/baseline/random):
      s0 −49.99/−63.41/−73.46, s1 −58.26/−64.99/−67.57, s2 −61.05/−61.88/−63.91.
- [x] Sentinels healthy on the world model P2 actually trained (run log `p2`;
      values identical to P1's because the model-training computation is the same
      deterministic recipe on the same budget).
- [x] `make test` green (47), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_planner.py): protocol conformance; CEM finds the concave-reward
  optimum via the per-sample fallback; uncertainty penalty flips the chosen arm;
  warm start persists across calls and clears on `reset()`.
- Eval: `make gate PHASE=P2` — equal-budget comparison + sentinel health; seeds
  recorded in the report.

## Docs-sync checklist
- [x] Status → `done`; P2 `GateReport` pasted below.
- [x] `P2` appended to `bench/SHIPPED` in this commit; `make gate-all` re-runs
      P0+P1+P2 (all green, ~100s — noted for CI cost).
- [x] Requirement row R1 still accurate (world_model.py + planning.py).
- [x] architecture.md planning.py note still accurate (flat MPC/CEM in
      imagination; hierarchy arrives at P5).
- [x] Backlog updated (P2-001 done; P2-002 unblocked and next).

## Gate result
`make gate PHASE=P2` — report persisted as `bench/results/P2-20260703T175413Z.json`:

```
[P2] PASS
  capability: ok — median eval return over 5 shared episodes: planner -58.26 vs
    model-free ES baseline -63.41 vs random -67.57 (learning budget 4096 env
    steps each; ES used 4000)
  sentinel[representation-integrity]: healthy — min per-dim std 0.868 (floor
    0.3), min effective rank 2.18 (floor 2.0)
  sentinel[uncertainty-reliability]: healthy — worst-seed disagreement-vs-error
    rank corr 0.79 (min 0.3), high-error-decile disagreement 21x median
```

Seeds [0, 1, 2]; the planner wins on every individual seed (margins 13.4 / 6.7 /
0.8 return). Deterministic end to end — the ratchet re-run reproduces identical
numbers. First gate run passed without iteration: every seam it consumed
(`Prediction`, `predict_batch`, run log, sentinel bodies, `Environment`) was
already exercised by P1.
