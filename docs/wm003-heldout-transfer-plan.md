# WM-003A zero-shot contextual-transfer plan

Status: queued; planning only, not a sealed protocol.

## Dependency

Begin after U-001 has a terminal accepted or rejected disposition. Freeze one
named collection policy, model family, optimizer, update budget, and planner
from the last scientifically supported snapshot. Do not perform any online
update on held-out contexts: zero-shot transfer is the only primary causal
delta.

## Falsifiable objective

One frozen shared Prospect world model, with no task-specific model copies,
heads, checkpoints, or held-out updates, provides better zero-shot prediction
and executed control on prospectively held-out observed dynamics contexts than
context-agnostic and context-shuffled controls.

## Required controls

- untrained/fixed-prior model;
- pooled model with context withheld;
- context-shuffled training;
- independent per-context models as a capacity control; and
- known-dynamics planning as a ceiling where the benchmark permits it.

## Gates

1. Bind one versioned, externally recognizable contextual-control benchmark,
   dependency closure, task set, budgets, seeds, and immutable
   train/interpolation/extrapolation context split.
2. Prove that no held-out context, reset, trajectory, normalization statistic,
   or task identity leaks into training or selection.
3. Require the shared model to learn the training contexts before any transfer
   claim is eligible.
4. Require paired positive zero-shot held-out predictive-score and
   executed-return effects over the non-oracle controls.
5. Include a prospectively frozen transfer-opportunity sweep with at least one
   near/interpolation and one far/extrapolation stratum; require the known
   dynamics or per-context ceiling to show usable signal in each scored
   stratum so a null result is interpretable.
6. Re-evaluate training contexts without updating and require fresh-process
   checkpoint parity.
7. Independently recompute every split, budget, metric, interval, and gate from
   raw rows.

Online adaptation is excluded and belongs to WM-003B. A later adaptation win
cannot rescue a failed WM-003A zero-shot claim.

## Exclusions

No hidden-context inference, uncertainty-directed collection, adaptive replay,
new observation encoder, partial observability, or broad state-of-the-art
claim. A single contextual benchmark does not establish general transfer.

## Intended repository surface

`bench/world_model_context_transfer/`, `tests/test_wm003_*.py`, and, only after
qualification, a versioned confirmation plan and operator runbook.

## Next action

Produce a claim-ineligible feasibility matrix for CARL-compatible
vector-control tasks: maintained version, context variables, observed-context
interface, immutable train/interpolation/extrapolation split support, baseline
availability, runtime cost, and compatibility with Prospect's existing model
and planner. Select one task family and one fixed collection policy before
writing protocol code.
