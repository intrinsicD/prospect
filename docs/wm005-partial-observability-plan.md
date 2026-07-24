# WM-005 long-horizon partial-observability plan

Status: queued; planning only, not a sealed protocol.

## Dependency

Begin after WM-003B reaches a terminal disposition; WM-004 is a sibling branch,
not a prerequisite. Freeze acquisition, model capacity, task distribution, and
one named simple replay policy (initially static balanced replay at a fixed
capacity). Introduce observation aliasing and a checkpointable
belief/latent-state update as the primary causal delta.

## Falsifiable objective

On a prospectively selected vector-observation POMDP with controllable horizon
and known simulator state, a learned recurrent belief state improves held-out
prediction and executed return as horizon and aliasing increase relative to
reactive and fixed-window controls.

## Required controls

- reactive current-observation model;
- fixed short observation window with matched capacity;
- temporal-order-shuffled recurrent model;
- recurrent-state reset ablation; and
- privileged full-state model as a ceiling.

## Gates

1. Bind the observation function, hidden state, horizons, aliasing levels,
   episode boundary, budgets, and held-out resets before training.
2. Prove a manipulation check: identical observations must correspond to
   action-relevant different hidden states, and the full-state ceiling must be
   able to exploit the distinction.
3. Link each recurrent belief and prediction to exactly the causally available
   prefix; reject future observations and hidden simulator state from the agent
   path.
4. Require improved held-out proper score or hidden-state diagnostic accuracy
   over reactive and shuffled controls.
5. Require a paired executed-return advantage that grows or remains robust over
   prospectively frozen longer horizons; predictive improvement alone is
   insufficient.
6. Check calibration by horizon and aliasing stratum rather than only in one
   pooled aggregate.
7. Restore at an episode boundary and reproduce latent-state initialization,
   model identity, and evaluation behavior.

## Exclusions

No pixels, language, audio, active acquisition, new replay mechanism,
mid-episode crash recovery, or reconciliation of real external side effects.
The hidden simulator state is audit-only and never an agent input.

## Intended repository surface

`bench/world_model_partial_observability/`, `tests/test_wm005_*.py`, plus a
task-neutral checkpointable latent-state protocol in `src/prospect/` only if
the benchmark demonstrates a stable interface.

## Next action

Compare maintained vector POMDP benchmarks for controllable horizon, exact
hidden-state access, deterministic reset custody, license, baseline strength,
and runtime cost. Select the cheapest environment whose aliasing manipulation
check is exact.
