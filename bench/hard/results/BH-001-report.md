# BH-001 — harder-benchmark probe (non-gated)

_Generated 2026-07-07 07:42 UTC. dm_control 1.0.43, mujoco 3.10.0, numpy 2.4.6._

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
- This is the finding the probe exists to surface: toy-benchmark wins are **not** evidence of general control. `cartpole-swingup` is the sharp case — random data never reaches the upright goal, so the world model is ignorant exactly where the reward lives. Two follow-up studies below chase that: **A** asks whether *directed* exploration (curiosity) fixes it; **B** asks whether *watching a demonstration* does.

## A — curiosity-driven collection (does directed exploration fix swingup?)

Swaps the P2 probe's *random* data collection for the shipped **curiosity curriculum** (P3-002 / ADR-0007): an explore-mode planner whose epistemic coefficient is a *bonus*, steering collection toward high-uncertainty regions. Same downstream (exploit-mode MBRL), same 4096-step budget — only the collection policy differs. Median over 3 seeds.

| metric | random collection | curiosity collection |
|--------|-------------------|----------------------|
| goal coverage — max reward reached | 0.24 | **0.70** |
| goal coverage — frac steps near goal (r>0.5) | 0.0% | **0.3%** |
| **MBRL control return** (swingup) | **6.4** | 2.1 |

**Reading.** Curiosity's exploration *works* — it **reaches** the upright region random data never touches (max reward ~0.7 vs ~0.24), though it still rarely *dwells* there (fraction near the goal stays ≈0). And that partial coverage does **not** convert to better exploit control: curiosity data *hurts* the downstream MBRL. Curiosity chases novelty, so its data concentrates on wild high-energy states and under-covers the near-bottom region the exploit planner traverses, while the goal coverage it does gain is too sparse to learn the upright dynamics. More budget doesn't close it (measured separately: 3× budget still worse). **Exploration is necessary but not sufficient here** — which motivates B.

## B — imitation from observation (does watching a demo reproduce swingup?)

The agent **watches** an expert swingup — its *observations only*, actions hidden — then **recovers the actions from observation** at the same interaction budget a from-scratch agent gets, and **clones** a closed-loop policy. Two recovery routes: a direct inverse-dynamics model, and the P13 `LatentActionModel` (ADR-0010) + a tiny calibration (the arc-faithful, action-free route). Oracle = clone on the true (hidden) actions — the ceiling. Median over 3 seeds; expert demo return 115.9.

| agent | swingup return | vs from-scratch |
|-------|----------------|-----------------|
| **imitation — inverse-dynamics** | 45.3 | 7.1× |
| imitation — latent-action (P13) | 14.0 | 2.2× |
| oracle clone (true actions, ceiling) | 76.4 | 12.0× |
| from-scratch MBRL (same budget) | 6.4 | 1.0× |
| shuffled demo (neg control) | 0.1 | 0.0× |

Action-recovery R² vs the true demo actions: inverse-dynamics 0.62, latent-action (P13) 0.65.

**Reading.** Watching **works where exploration could not**: at the *same* budget the from-scratch agent fails swingup on (A), imitation-from-observation reproduces it — the inverse-dynamics route reproduces a swingup the agent never performed, well above from-scratch and approaching the oracle-clone ceiling, while the shuffled-demo control collapses (it is imitating the *specific* behaviour, not just moving). The P13 latent-action route is the honest weak spot: it can match the direct route on a good seed but is high-variance on this real task — recovering executable actions from a 1-D latent across a distribution shift (grounding states → the demo's upright states) is not yet reliable. **The A→B arc:** exploration reaches the goal region but can't convert it to control at feasible budgets; a demonstration hands over the goal-reaching behaviour directly — the sample-efficient route, and the substrate for learning from video (ADR-0009/0010).
