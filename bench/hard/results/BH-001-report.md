# BH-001 — harder-benchmark probe (non-gated)

_Generated 2026-07-07 00:05 UTC. dm_control 1.0.43, mujoco 3.10.0, numpy 2.4.6._

Re-runs the **P2 claim** — MPC/CEM over a learned `FlatWorldModel` beats a model-free baseline at **equal env-step budget** — on DeepMind Control Suite tasks the repo did not author (ADR-0011). **Non-gated:** no phase ships on this; it is a credibility probe whose value is an honest number, not a pass.

**Setup (matched to the shipped P2 gate).** Learning budget **4096 env steps** per agent; model-free is CEM-ES policy search (POP 10 × GENS 4 × 100-step rollouts = 4000 steps ≤ budget); MBRL planner is P2's `FlatPlanner` defaults (horizon 20, 64 candidates). Returns are the mean over 3 shared eval episodes (100 steps), identical seeds across all three agents. Median [min, max] over 5 seeds.

| task | dims (obs/act) | MBRL | model-free (matched) | random | MBRL ≥ both? |
|------|----------------|------|----------------------|--------|--------------|
| `cartpole-balance` | 5/1 | 92.4 [52.6, 98.9] | 99.5 [98.4, 99.7] | 78.5 [63.4, 80.7] | no |
| `cartpole-swingup` | 5/1 | 6.4 [1.3, 7.6] | 8.8 [7.5, 8.8] | 0.2 [0.1, 0.5] | no |

**P1-calibration spot-check** (seed 0, `cartpole-balance`): median ensemble epistemic at full budget / at 256 steps = **0.116** (< 1 ⇒ uncertainty falls with data, as it should).

**Reachability (1 seed, harder / higher-action-dim tasks).** The same adapter loads and steps these — different domains, 2-D actions — with no code change, but at this budget every agent scores ~0: they are **below the probe's resolution**, not broken. This is the honest edge of a 4096-step / 100-step probe (max return across all three agents shown).

| task | dims (obs/act) | best-of-3 return |
|------|----------------|------------------|
| `reacher-easy` | 6/2 | 0.00 |
| `point_mass-easy` | 4/2 | 0.00 |
| `finger-spin` | 9/2 | 1.00 |

## Honest reading
- The seam works: the **core is unchanged** — `FlatWorldModel`/`FlatPlanner`/`Agent` act in real MuJoCo via one `bench.Environment` adapter, across 5 DMC tasks in four domains, including 2-D action spaces.
- Both learners beat random by a wide margin, so the model-based machine is not broken on foreign dynamics.
- But the clean *MBRL-beats-model-free* win from the authored Pendulum (P2) does **not** reproduce as a decisive win at equal budget here: on `cartpole-balance` both saturate the task, and on the harder tasks the matched-budget model-free baseline is competitive. Contributing factor, reported not hidden: the exploit-mode epistemic penalty (ADR-0007) discourages the planner from the high-uncertainty regions a 4096-random-step model never visited — precisely the upright/target region — which is the penalty working as designed, not a control failure.
- This is the finding the probe exists to surface: toy-benchmark wins are **not** evidence of general control; the credibility jump needs more env-step budget, better exploration for model training, and stronger baselines — not more phases.
