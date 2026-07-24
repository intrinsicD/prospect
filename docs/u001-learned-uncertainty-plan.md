# U-001 learned uncertainty and calibration plan

Status: queued; planning only, not a sealed protocol.

## Dependency

Begin after WM-002 reaches a terminal disposition. Reuse its finite
active-acquisition fixture, truthful return accounting, action set, and Q0
oracle as audit scaffolding, but remove the agent's privileged known pulse
likelihoods. Learning uncertainty from real experience is the only primary
causal delta.

## Falsifiable objective

From a bounded stream of canonical acquisition transitions, one shared
Prospect model learns predictive distributions whose uncertainty is calibrated,
separates reducible model uncertainty from irreducible outcome noise, ranks
out-of-distribution error risk, changes acquisition choices when information is
decision-useful, and retains those improvements after fresh-process restore.

## Required controls

- fixed known-likelihood oracle as a ceiling, never an input to the learner;
- fixed prior/no-update learner;
- point-estimate learner with matched capacity and updates;
- outcome-label or likelihood-link shuffled learner;
- matched random/uniform acquisition; and
- capacity-matched ensemble or conjugate estimator as a reference where exact.

## Gates

1. Freeze a train/calibration/held-out split over likelihood regimes, real
   transition budgets, update work, seeds, proper scores, calibration bins,
   OOD strata, and acquisition opportunities before opening held-out outcomes.
2. Prove an uncertainty-learning manipulation: training data must contain
   repeated conditions with both epistemically reducible ignorance and
   irreducible stochastic outcome noise.
3. Require held-out log-score or Brier improvement over fixed-prior and
   point-estimate controls, with calibration error and coverage reported by
   regime rather than only pooled.
4. Require epistemic uncertainty to shrink with informative data while
   irreducible predictive entropy remains bounded near its generating level;
   shuffled evidence must break the former effect.
5. Require uncertainty or disagreement to rank held-out/OOD prediction error
   better than entropy-only and constant-confidence controls.
6. On a prospectively frozen acquisition-choice sweep, require learned
   uncertainty and learned VOI to select different actions from no-update and
   shuffled controls where the exact ceiling proves a positive opportunity,
   and require a paired executed-return benefit under matched interactions.
7. Restore in a fresh process and reproduce model identity, calibrated
   predictions, acquisition assessments, and selected actions without replaying
   training data.
8. Independently recompute proper scores, calibration, OOD ranking, action
   values, budgets, and ordered gates from raw transitions and predictions.

## Exclusions

No transfer to new task families, adaptive replay, long-horizon POMDP,
multimodal encoder, neural representation claim, or external-arena claim.
Success is bounded evidence that uncertainty was learned and used on this
fixture, not general calibrated intelligence.

## Intended repository surface

`bench/learned_uncertainty/`, `tests/test_u001_*.py`, and a task-neutral
predictive-uncertainty contract in `src/prospect/` only if the benchmark
demonstrates that existing domain records cannot express the required
decomposition without fixture leakage.

## Next action

Write the smallest conjugate or tabular likelihood-learning killing test that
contains separately controllable epistemic and aleatoric components. Derive
exact expected proper scores and acquisition choices before selecting any
learned function approximator.
