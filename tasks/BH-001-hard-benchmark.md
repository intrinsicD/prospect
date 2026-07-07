# BH-001 — Optional harder-benchmark probe (real MuJoCo control)

- **Status:** done
- **Phase:** none — this is a **NON-gated** tier (ADR-0011). It ships nothing and
  advances no phase; it does not gate P14.
- **Requirements:** R1 (predict/plan, stress-tested on foreign dynamics) — as evidence,
  not as a new gated capability.
- **ADRs:** ADR-0011 (the optional non-gated tier), ADR-0005 (the numpy-only gated core
  it is fenced away from), ADR-0007 (the exploit-mode epistemic penalty whose effect the
  probe surfaces), ADR-0002 (prediction error / epistemic — the calibration spot-check)
- **Depends on:** P2 (the claim being re-run), P1 (`FlatWorldModel`), P0-004 (the
  `bench.Environment` seam the adapter satisfies)
- **Phase gate:** none (by design). The deliverable is a committed **report artifact**,
  not a pass/fail gate.

## Goal
Answer one question the authored toys can't: does the P2 result — **MPC/CEM over a
learned `FlatWorldModel` beats a model-free baseline at equal env-step budget** — survive
on a standard MuJoCo control task the repo did not design? Produce an honest, auditable
report, not a number tuned to look good.

## Non-goals
- **Not a gate.** No pass bar ships anything; nothing here enters `bench/SHIPPED` or
  `make gate-all`. It must not slow or bind the numpy-only core CI (ADR-0011).
- **Not SOTA.** Small budget, small numpy world model, short horizons. Losing to
  published SAC would be unremarkable — irrelevant to the matched-budget question asked.
- **No pixels.** State observations only (no GL/render dep); a real visual encoder is the
  vision arc's concern (ADR-0009), orthogonal to this.
- **No core change.** If the core needs editing to run a real task, that is a finding, not
  a licence to edit — the whole point is that the `Environment` seam suffices.

## Interface to satisfy
- Harness only (task-specific ⇒ `bench/`, golden rule 3): `bench/hard/dmc_env.py::
  DMCEnvironment` satisfies the existing `bench.Environment` Protocol (`reset(seed)` /
  `step(action) -> (obs, reward, done)`), wrapping a DeepMind Control Suite task. It
  exposes `obs_dim`/`action_dim`/`action_low`/`action_high` so `FlatWorldModel`,
  `FlatPlanner` and the model-free baseline size themselves to the task with no per-task
  code. **The core (`src/prospect/`) is untouched.**

## Approach (brief)
- Re-run the P2 comparison with the **same machine** (BUDGET, EP_LEN, `FlatPlanner`
  defaults, a budget-matched CEM-ES model-free baseline all equal P2's) on DMC tasks.
  Deviating from P2's settings to flatter MBRL is forbidden — it would defeat the probe.
- Informative rungs (5 seeds): `cartpole-balance` (both learners should saturate — a
  sanity floor) and `cartpole-swingup` (the harder test).
- Reachability rungs (1 seed): `reacher-easy`, `point_mass-easy`, `finger-spin` — 2-D
  actions / different domains, to show the adapter generalizes; reported honestly even
  when they fall below the probe's resolution at this budget.
- A light **P1-calibration spot-check** rides along (seed 0): median ensemble epistemic
  should fall as the training set grows (reported, not gated).
- **Isolation:** `[bench-hard]` optional extra (dm_control + mujoco), `bench/hard/` never
  imported by the gate registry, skips cleanly when the extra is absent, a manual
  `workflow_dispatch` CI job — the numpy `ci.yml` and the ratchet never touch it.

## Acceptance criteria (non-gated — "done" = the probe runs and is reported honestly)
- [x] `DMCEnvironment` satisfies `bench.Environment` (typed test, `tests/test_bench_hard.py`,
      `importorskip` so the numpy CI skips it) and drives the **unchanged** core on real MuJoCo.
- [x] The P2 claim is re-run at matched budget on ≥2 informative tasks over ≥5 seeds; raw
      per-seed returns + spread + matched-budget deltas written to `bench/hard/results/`.
- [x] Isolation holds: `make test` / `make gate-all` / `make typecheck` stay green **without**
      the extra; the tier runs only via `make bench-hard` (needs `[bench-hard]`).
- [x] The report states the honest reading — including where MBRL does **not** decisively
      win and why — rather than a tuned headline.

## Test plan
- Unit (`tests/test_bench_hard.py`, skipped without the extra): protocol conformance,
  obs/action dims, reset/step contract, seed reproducibility, out-of-box action clipped.
- Probe (`python -m bench.hard`): writes `bench/hard/results/BH-001-report.{md,json}`.
- Isolation: the numpy-only suite (`make test`, `make gate-all`) never imports `bench.hard`.

## Docs-sync checklist
- [x] ADR-0011 added (Accepted); ADR index updated.
- [x] Roadmap: harder-benchmark tier noted (non-gated, ADR-0011). BACKLOG: BH-001 row.
- [x] README status: point the "next credibility jump needs harder environments" line at
      the new tier.
- [x] `pyproject.toml` `[bench-hard]` extra + mypy override; `Makefile` `bench-hard`
      target; `.github/workflows/bench-hard.yml` (manual only).

## Result
`make bench-hard` → report at `bench/hard/results/BH-001-report.{md,json}` (dm_control
1.0.43, mujoco 3.10.0, numpy 2.4.6). Median [min, max] over 5 seeds, 4096-step budget,
matched CEM-ES model-free baseline (4000 ≤ 4096 steps), P2 planner defaults:

| task | MBRL | model-free (matched) | random | MBRL ≥ both? |
|---|---|---|---|---|
| `cartpole-balance` | 92.4 [52.6, 98.9] | 99.5 [98.4, 99.7] | 78.5 [63.4, 80.7] | no |
| `cartpole-swingup` | 6.4 [1.3, 7.6] | 8.8 [7.5, 8.8] | 0.2 [0.1, 0.5] | no |

P1-calibration spot-check (seed 0): epistemic(full)/epistemic(256) = **0.116** — the
ensemble's uncertainty falls with data on a real task, as it should. Reachability (1 seed):
the adapter also drives `reacher-easy` (6/2), `point_mass-easy` (4/2), `finger-spin` (9/2)
unchanged, but all agents score ~0 there — below the probe's resolution at this budget.

**Honest reading (the point of the probe).** The seam works — the **unchanged** core acts
in real MuJoCo across 5 tasks / 4 domains / 2-D actions via one `Environment` adapter, and
both learners crush random (so nothing is broken on foreign dynamics). But the clean
*MBRL-beats-model-free* win from the authored Pendulum (P2) does **not** reproduce as a
decisive win at equal budget: on balance both saturate, on swingup the matched-budget
model-free baseline stays ahead, and MBRL is higher-variance. A reported contributing
factor: the exploit-mode epistemic penalty (ADR-0007) steers the planner away from the
high-uncertainty upright/target region a 4096-random-step model never visited — the penalty
working as designed, not a control failure. **Conclusion:** toy-benchmark wins are not
evidence of general control; the next credibility jump needs more budget, better model-
training exploration, and stronger baselines — not more phases. Recorded as a finding, not
tuned away. **This tier is non-gated: nothing ships, `bench/SHIPPED` is unchanged.**

## Follow-up studies (same report, chasing the swingup failure)
The probe surfaced that `cartpole-swingup` fails because random data never reaches the
upright goal. Two studies chase that, in the same consolidated report (§A, §B):

- **§A — curiosity (`bench/hard/curiosity.py`).** Swaps random collection for the P3-002
  curiosity curriculum. **Finding:** curiosity *reaches* the goal region random data can't
  (max reward 0.24 → 0.70) but does **not** convert it to control (MBRL 6.4 → 2.1); 3×
  budget doesn't fix it. Exploration is necessary but not sufficient here.
- **§B — imitation from observation (`bench/hard/imitation.py`, P14-001, ADR-0012).** Watch
  an expert swingup's *observations only* → recover its actions (inverse-dynamics primary;
  P13 latent-action route arc-faithful but high-variance) → clone a closed-loop policy.
  **Finding:** reproduces a swingup the agent never performed — inverse-dyn **45.3 vs
  from-scratch 6.4** (7×), shuffled control 0.1. Watching does what exploration can't at the
  same budget. The A→B arc: exploration reaches the region; a demonstration hands over the
  behaviour.
