# P14-001 — Observe → repeat (imitation from observation)

- **Status:** done — **numpy-gated (P14 shipped)** AND demonstrated on the real hard-benchmark tier
- **Phase:** P14. `GATES["P14"]` PASS on a numpy swing-up task (ships in `bench/SHIPPED`,
  ratchet green at 15 gates); the DMC swingup study is the non-gated real-task demonstration on
  top (mirrors P12: gated on stand-ins, real vision demonstrated off-gate).
- **Requirements:** R5 (use the right learned patterns — reproduce a skill), R7 (improve
  from watching)
- **ADRs:** ADR-0012 (imitation-from-observation: action recovery to reproduce a demo),
  ADR-0010 (latent-action inference + the `ground` reliability fix), ADR-0007 (the
  observe→repeat→explore curriculum), ADR-0011 (the non-gated tier the DMC demo runs in)
- **Depends on:** P13 (LatentActionModel), P2 (planner/MBRL baseline), BH-001 + the A study
  (which established that exploration alone can't crack swingup — the motivation)
- **Phase gate:** `bench/gates.py::GATES["P14"]` — a single-task phase; PASS ships it.

## Goal
Reproduce a **demonstrated behaviour the agent never performed itself**, from the demo's
**observations only** (imitation from observation) — on cartpole **swingup**, the task the A
study proved exploration (even curiosity) cannot crack at feasible budgets. Then *explore*
(P3-002) fills what watching can't teach.

## Non-goals
- No expert *actions* used by the imitation routes — observations only (actions are oracle-only).
- No claim of SOTA control; the ceiling here is a cloned reactive policy, not optimal control.
- No world-model *planner* in the reproduction path (the roadmap's "planner + latent actions"
  synthesis is a later refinement); reproduction is a cloned reactive policy for now.

## Interface to satisfy
- Core: `imitation.ObservationImitator` (satisfies `interfaces.ImitationLearner`): `ground()`
  learns action recovery from the agent's own labelled transitions (inverse dynamics),
  `clone()` watches the demo's observations and fits a reactive policy, `act()` reproduces.
  Also `LatentActionModel.ground()` (ADR-0010 amendment) — the P13 route's supervised grounding.
- Numpy gate: `bench/evals/p14_imitation.py::check_p14` on `bench.envs.PendulumSwingup`.
- Non-gated demo: `bench/hard/imitation.py::run_imitation` on DMC swingup (real task).

## Approach (brief)
Watch an expert swingup (observations only) → **recover its actions from observation** at the
same interaction budget a from-scratch agent gets (grounding = the agent's own labelled steps)
→ **clone** a closed-loop reactive policy → run it. Two recovery routes, both reported
(ADR-0012): direct **inverse-dynamics** (robust primary) and the P13 **latent-action** +
calibration (arc-faithful, for the fully action-free limit). Guards: oracle clone (ceiling),
shuffled-demo (negative control), from-scratch MBRL (same budget).

## Acceptance criteria (single-task phase — PASS ships)
- [x] **Reproduces the demo** it only watched: imitation score high (swings up).
- [x] **Recovers actions from observation**: recovered demo actions match the true hidden
      actions (R² ≥ floor) — recovery is real, not given.
- [x] **Specific behaviour**: a shuffled-demo control collapses toward the floor (negative control).
- [x] **Watching is what does it**: imitation beats cloning the agent's OWN random data by a margin.
- [x] `make gate PHASE=P14` PASS, all sentinels healthy; P14 appended to `bench/SHIPPED`;
      `make gate-all` green (15 gates); `make test`/`lint`/`typecheck` clean.

## Test plan
- Numpy gate: `bench/evals/p14_imitation.py::check_p14` (3 seeds) — the four criteria + sentinels.
- Unit: `tests/test_imitation.py` (ObservationImitator recovery + shapes); `tests/test_observation.py`
  (`LatentActionModel.ground`). Conformance to `ImitationLearner` (`tests/test_conformance.py`).
- Non-gated real-task demo: `bench/hard/imitation.py::run_imitation` (DMC swingup), report §B.

## Gate result
`make gate PHASE=P14` → **[P14] PASS**, all five sentinels healthy. Median over 3 seeds:

| criterion | measured | bar |
|---|---|---|
| reproduces the demo — imitation score | **0.99** | ≥ 0.6 |
| recovers actions from observation — R² | **1.000** | ≥ 0.5 |
| specific (shuffled-demo control) — score | **0.12** | ≤ 0.35 |
| watching matters — imitation − clone-own-random | **1.00** | ≥ 0.4 |

**P14 ships** (`bench/SHIPPED` ratchets P0–P14). The agent reproduces a swing-up it never
performed, from an expert's observations + a little grounding; the shuffled control collapses
and cloning its own random data fails — watching the *specific* expert is what does it.

## Real-task demonstration (DMC swingup, non-gated) + the Part-2 reliability fix
`make bench-hard` → report §B. Reproduces the DMC swingup (inverse-dynamics **45.3** vs
from-scratch **6.4**, shuffled 0.1, oracle 76.4). The P13-based route was initially high-variance
(the separate latent→action calibration extrapolated with a systematic bias — and recovery R²
didn't even predict reproduction). **Fixed** by **watch-then-ground** (`LatentActionModel.ground`,
ADR-0010 amendment): action-free pretraining then a supervised grounding step. Measured: in the
**512-label regime it beats from-scratch inverse dynamics (46 vs 33)** — watching is a low-data
prior for control; at full budget direct inverse dynamics is still best (the honest boundary).
Watching is a prior; **explore (P3-002) still closes what watching can't teach.**
