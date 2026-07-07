# P14-001 — Observe → repeat (imitation from observation)

- **Status:** demonstrated (non-gated, on the hard-benchmark tier); numpy gate = follow-up
- **Phase:** P14 (roadmap). Capability shown here on a real task; the numpy kill-gate that
  formally ships P14 in the ratchet is the named follow-up (mirrors P12: gated on stand-ins,
  real vision demonstrated off-gate).
- **Requirements:** R5 (use the right learned patterns — reproduce a skill), R7 (improve
  from watching)
- **ADRs:** ADR-0012 (imitation-from-observation: action recovery to reproduce a demo),
  ADR-0010 (latent-action inference — the arc-faithful recovery route), ADR-0007 (the
  observe→repeat→explore curriculum), ADR-0011 (the non-gated tier this runs in)
- **Depends on:** P13 (LatentActionModel), P2 (planner/MBRL baseline), BH-001 + the A study
  (which established that exploration alone can't crack swingup — the motivation)
- **Phase gate:** none yet (non-gated demonstration). A numpy-gated toy imitation task is the
  follow-up to ship P14 in `bench/SHIPPED`.

## Goal
Reproduce a **demonstrated behaviour the agent never performed itself**, from the demo's
**observations only** (imitation from observation) — on cartpole **swingup**, the task the A
study proved exploration (even curiosity) cannot crack at feasible budgets. Then *explore*
(P3-002) fills what watching can't teach.

## Non-goals
- Not a numpy kill-gate (runs on dm_control, non-gated — ADR-0011). Shipping P14 in the
  ratchet is a separate follow-up.
- No expert *actions* used by the imitation routes — observations only (actions are oracle-only).
- No claim of SOTA control; the ceiling here is a cloned reactive policy, not optimal control.

## Interface / where it lives
- Harness only (task-specific → `bench/`): `bench/hard/imitation.py::run_imitation`. Reuses the
  **unchanged** core: P13 `LatentActionModel`, `FlatWorldModel`/`FlatPlanner` (from-scratch
  baseline), the `bench.Environment` seam. No `src/prospect/` change.

## Approach (brief)
Watch an expert swingup (observations only) → **recover its actions from observation** at the
same interaction budget a from-scratch agent gets (grounding = the agent's own labelled steps)
→ **clone** a closed-loop reactive policy → run it. Two recovery routes, both reported
(ADR-0012): direct **inverse-dynamics** (robust primary) and the P13 **latent-action** +
calibration (arc-faithful, for the fully action-free limit). Guards: oracle clone (ceiling),
shuffled-demo (negative control), from-scratch MBRL (same budget).

## Acceptance criteria (demonstration — "done" = reproduces and is reported honestly)
- [x] Reproduces swingup from observation, **beating from-scratch MBRL** and a **shuffled-demo**
      negative control, approaching the oracle-clone ceiling — measured over ≥3 seeds.
- [x] Both recovery routes reported; the P13 latent route's real-task variance stated honestly.
- [x] Runs via `make bench-hard`; isolation intact (numpy CI untouched). Report in `bench/hard/results/`.

## Test plan
- `bench/hard/imitation.py::run_imitation` (3 seeds) → the reproduction table + recovery R².
- Consolidated report `bench/hard/results/BH-001-report.md` §B; `tests/test_bench_hard.py`
  keeps the adapter covered (skips without the extra).

## Result
`make bench-hard` → report §B (dm_control 1.0.43, mujoco 3.10.0, numpy 2.4.6). Median over 3
seeds, expert demo return 115.9, grounding budget 4096 (== from-scratch):

| agent | swingup return | ×from-scratch |
|---|---|---|
| imitation — inverse-dynamics | **45.3** | 7.1× |
| imitation — latent-action (P13) | 14.0 (67 on seed 0; high-variance) | 2.2× |
| oracle clone (true actions, ceiling) | 76.4 | 11.9× |
| from-scratch MBRL (same budget) | 6.4 | 1× |
| shuffled demo (neg control) | 0.1 | 0.0× |

**Imitation from observation reproduces a swingup the agent never performed** — the inverse-
dynamics route gets a stable 45.3 (7× from-scratch, ~59% of the oracle ceiling); the shuffled
control collapses to 0.1 (it imitates the *specific* behaviour). This lands exactly where the A
study said exploration fails: watching converts the same budget from failure (6.4) into swingup.
**Honest weak spot:** the P13 latent-action route is high-variance on this real task (67 on one
seed, ~10–14 on others) — grounding a 1-D latent to executable actions across the grounding→demo
distribution shift is not yet reliable; the direct route is preferred when grounding labels exist
(ADR-0012). Watching is a prior; **explore (P3-002) still closes what watching can't teach.**
