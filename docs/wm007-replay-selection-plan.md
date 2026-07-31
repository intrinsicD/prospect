# WM-007 replay-selection efficiency plan

Status: proposed; exploratory planning only, not a sealed protocol. Every
outcome under this plan is claim-ineligible until a separate formal protocol
is prospectively frozen after qualification.

## Dependency

Reuses only accepted WM-001 machinery — the fixture family, the replay storage
seam under `src/prospect/storage/`, and the evaluation contract — without
touching any sealed WM-001 artifact. `WorldModelConfig.require_formal()` and
`OptimizerConfig.require_formal()` deliberately pin exact sealed
configurations; this task never modifies, subclasses around, or relabels them.
It declares its own exploratory configuration family inside its own benchmark
package. It is independent of the WM-002 → WM-006 chain: planning may proceed
in parallel, and execution starts only when it does not contend with the
single active scientific experiment.

## Falsifiable objective

Under matched consumed-transition, gradient-step, model, optimizer, and
evaluation budgets on a fixed regime set, at least one non-uniform
replay-selection policy reaches the frozen held-out target with fewer consumed
transitions and fewer optimizer steps than both uniform replay and
WM-001-style balanced replay, without a worst-regime or retention regression.

## The single causal delta

Only the replay-selection distribution varies. Candidate priority:

```text
p_i ∝ (eps + s_i)^alpha * (eps + d_i)^beta * (eps + l_i)^gamma * b_i
```

- `s_i`: immutable action-time surprise, recorded at collection and never
  recomputed after outcomes are known;
- `d_i`: ensemble disagreement at scoring time;
- `l_i`: observed loss reduction when the transition was previously sampled
  (learning progress);
- `b_i`: an explicit context/regime balancing factor.

Arms: uniform; balanced (WM-001-style); surprise-only; disagreement-only;
learning-progress-only; one prospectively frozen hybrid. Each arm declares its
importance-weighting/bias-correction rule before execution.

Naming rule, per the semantic contract: the sampling score is a selection
heuristic. It is never labeled "information gain" unless computed as an actual
prior/posterior change, and surprise, uncertainty, disagreement, and
information value remain separate named quantities in records and reports.

## Required controls

- uniform replay at identical budgets (floor);
- WM-001-style balanced replay (the shipped posture);
- a permuted-priority control: the winning policy's scores permuted across
  transitions while preserving the marginal selection distribution, which
  kills "any non-uniformity helps" explanations;
- seed-matched paired runs across all arms; and
- no policy may read held-out evaluation identities, future outcomes, or
  another arm's state.

## Gates

1. Freeze the regime set, seeds, budgets (consumed transitions and gradient
   steps), target thresholds, policy hyperparameters (`alpha`, `beta`,
   `gamma`, `eps`, the balance definition), importance weighting, and the
   evaluation schedule before any outcome is observed.
2. Verify that every arm consumed identical transition counts and optimizer
   steps; a policy that silently buys extra work invalidates the comparison.
3. Measure separately, never collapsed into one scalar: held-out negative log
   likelihood, calibration, transitions-to-target, optimizer-steps-to-target,
   retention after interference, regime coverage, and worst-regime score.
4. Bind every replayed sample to canonical experience ancestry; imagined
   evidence never enters the real-experience store.
5. Fresh-process persistence parity: the frozen updated snapshot reproduces
   bounded evaluation behavior after restore.
6. Promotion requires the winning policy to beat uniform, balanced, and its
   own permuted-priority control on the frozen efficiency endpoint with no
   worst-regime or retention regression.

## Exclusions

No change to model capacity, ensemble size, optimizer family, acquisition
policy, environment family, observation modality, or planner inside a
comparison. No sealed WM-001 file is edited and no WM-001 evidence is
relabeled. Success establishes matched-budget selection efficiency on this
fixture only; continual-scale retention under growing task counts and bounded
memory remains WM-004's question, and this task's terminal outcome only
informs WM-004's adaptive-replay candidate set.

## Deferred sibling questions

Each is explicitly outside this plan and requires its own plan document
before any work:

- member-wise ensemble update assignment (bootstrap masks or error-routed
  member updates) with mandatory ensemble-collapse guards: pairwise prediction
  correlation, disagreement calibration, mixture NLL, and per-member data
  coverage;
- planner-only distillation of the ensemble into a compact screening student
  with teacher reranking of top candidates, keeping canonical belief,
  evidence, and formal uncertainty on the teacher ensemble;
- progressive model-width growth — deprioritized: the accepted world model is
  already small, and environment interaction, repeated ensemble evaluation,
  protocol serialization, and held-out experiments dominate current budgets.

## Intended repository surface

`bench/world_model_efficiency/` (problem, replay policies, protocol, runner,
analysis, README), `tests/test_wm007_*.py`, and untracked generated results
below the benchmark package. `src/prospect/storage/` is consumed read-only; a
task-neutral selection contract moves into `src/prospect/` only if the
experiment proves it is required beyond this fixture.

## Next action

Specify the cheapest killing fixture: the smallest regime set and budget at
which uniform and balanced replay measurably differ from at least one
prioritized policy on transitions-to-target. If no separation exists at small
scale, record the negative result and stop before building the full harness.
