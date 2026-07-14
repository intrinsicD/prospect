# BC-001 protocol: BridgeControl causal coverage fixture

**Status:** frozen control criteria with post-stop integrity clarification; non-gated
research evidence

**Parent artifacts:**

- `docs/research/2026-07-13-transformational-research-prompt.md`
- `docs/research/2026-07-13-predictive-reliability-portfolio.md`, Evidence Program P1

**Scope:** `bench/bridge_control/` only. No production source, gate, task status, ADR,
backlog status, or shipped result may change from this experiment.

## Timing and anti-HARKing record

The parent portfolio predeclared the central null, the 2×2×2 factors, exact and
balanced-data positive controls, one permitted fixture redesign, and the rule that a
failed positive control invalidates the fixture before factorial interpretation.

Concrete implementation used excluded learner/planner seed `97` to validate and then
redesign the fixture. An independent adversarial design review identified that the
first implementation colocated bridge and rank interventions and used a goal-side
action compensator. Before the primary factorial, the implementation was changed once:

- bridge support is manipulated only at the directed door;
- action rank and unique controllable-support density are manipulated only in the
  disjoint post-bridge stabilization strip;
- the rank designs use identical coordinate-wise action marginals, removing the
  compensator;
- nuisance dynamics are the identity and are reward-irrelevant;
- coverage is measured over x-region × door/off-lane, not x alone.

The exact-model and balanced-data control criteria below were stated before the eight
fresh learner/planner blocks were inspected. This file is an execution ledger rather
than a timestamped third-party preregistration; it must not be represented as the
latter. Formal result artifacts hash this protocol and rerun the frozen controls.

After the first stopped execution, a read-only audit required stronger source/artifact
hashing and corrected an over-attribution in the localization prose. Those integrity
changes did not alter the environment, datasets, seeds, learner, planner, thresholds,
or stop rule. The diagnostic hybrid is explicitly treated as a joint
transition/representation/uncertainty intervention with no pass threshold; that
clarification is post-outcome and is not part of the preregistered causal test.

## Research question and null

Does a dataset's observed directed bridge add reproducible downstream-control
information beyond point coverage, coordinate-wise action marginals, unique
controllable-support density, and ordinary one-step outcome metrics?

The primary null is that, conditional on the matched construction, bridge support and
post-bridge action rank add no reproducible information about learned-model MPC return.

## Fixed environment

State is `(x, y, n)` and action is `(a0, a1)`, with all values clipped to fixed bounds.
The nuisance obeys `n_next = n` and never enters reward.

- Outside the post-bridge strip, `dx = 0.22 a0` and `dy = 0.12 a1`.
- In `0.30 <= x <= 0.75`, door-lane dynamics use sum/difference coordinates:
  `dx = 0.16(a0+a1)`, `dy = 0.16(a0-a1)`.
- Crossing `x=0` is blocked, leaving x unchanged, unless the proposed next state has
  `abs(y) <= 0.15`.
- Off-lane decoy states have `abs(y) >= 0.55` and are laterally absorbing.
- Reward is `1.25*x_next - 2*abs(y_next-0.25) - 0.01*(a0²+a1²)`.

Evaluation uses four fixed starts at `x=-0.90`, 14 steps, and success means final
`x>=0.75` and `abs(y-0.25)<=0.20`.

## Fixed evidence interventions

Every primary dataset has the same row count, x-region × lane macro-counts, nuisance
multiset, and action-coordinate histograms.

### B — observed directed bridge

At the bridge source, the same four door-lane states and same four actions are paired
differently. `B=0` contains zero observed forward bridge transitions; `B=1` contains
exactly 16. These rows are invariant to rank and density.

### R — post-bridge local action rank

At every stabilization-strip microstate:

- deficient: `(a0,a1)=(v,v)` for `v=(-.9,-.3,.3,.9)`;
- full: `a0=(-.9,-.3,.3,.9)`, `a1=(-.3,.9,-.9,.3)`.

Both action coordinates have identical marginal values, counts, means, variances, and
entropy. The centered local action design has minimum singular value `0` versus at
least `0.5`.

### D — unique controllable-support density

At each crossed stabilization state/action cell, `D=1` repeats one exact controllable
microstate over the same eight nuisance values; `D=8` uses eight guarded x/y
micropositions with the identical nuisance multiset. This is deterministic unique
support, not eight statistically independent samples.

## Manipulation checks before returns

- all eight primary cells exist and have equal row count;
- x-region × lane counts match exactly;
- nuisance levels and coordinate-wise action histograms match exactly;
- node coverage is 16 in every cell;
- bridge edge count is exactly `0` versus `16`, invariant to R and D;
- local minimum action singular value is at most `0.05` versus at least `0.5`,
  invariant to B and D;
- unique controllable microstates per stabilization cell are exactly `1` versus `8`,
  invariant to B and R;
- regeneration reproduces semantic dataset hashes.

Failure of any check permits no learned-return interpretation.

## Learner and planner

The production classes are unchanged:

- `FlatWorldModel(obs_dim=3, action_dim=2, latent_dim=6, hidden=48, ensemble=5)`;
- 1,800 update steps, batch size 64, identical per-seed minibatch index schedules;
- learner/planner blocks `0..7`; development seed `97` is excluded;
- `FlatPlanner(horizon=12, candidates=64, elites=8, iterations=3,
  uncertainty_penalty=0.03, action bounds [-1,1])`;
- identical starts, episode length, planner compute, and evaluation order in all arms.

## Sequential controls and stop rule

1. The exact dynamics/reward model with unchanged FlatPlanner must succeed on at least
   95% of fixed starts.
2. The fully balanced learned cell `B=1,R=full,D=8` must succeed on at least 80% of
   fixed starts across the eight fresh blocks and materially beat the random baseline.
3. Only if both pass may the other seven factorial cells and the action-permutation and
   nuisance-only controls be used for causal contrasts.

If step 2 fails after the one permitted redesign, label BC-001
`aborted_invalid_fixture`, do not inspect or estimate factorial effects, and run only a
localization ladder. Do not rescue the hypothesis with further bins, hyperparameters,
model changes, or coefficient tuning.

## Localization ladder after a failed balanced control

The diagnostic package may report:

- exact dynamics + exact reward planning;
- exact raw-state dynamics + the learned reward head + zero epistemic, reported only
  as a joint transition/representation/uncertainty diagnostic;
- learned one-step reward error;
- nearest-prototype next-region and door/off-lane classification;
- normalized target-latent one-step error as a secondary within-model metric;
- fixed-bank multi-step candidate-ranking correlation and action regret;
- final-state failure geography and epistemic exposure.

Cross-arm raw latent MSE is not a primary baseline because each model learns a
different online/EMA target encoder and the core has no raw-state decoder.

## Primary analysis only if controls pass

Use ±1 factor coding and compute seed-blocked B, R, D, and B×R effects. Require the B
contrast at full rank and B×R interaction to be positive in at least seven of eight
blocks before using a one-sided exact sign-randomization result. Compare return against
bridge reachability/cut support, action rank, unique support, reward error,
next-region classification, nearest support distance, and candidate-ranking regret.

A positive first topology would still require a second corridor length. Without that
replication, it establishes at most a fixture-specific bridge effect—not a percolation
law, general coverage theory, or production mechanism.
