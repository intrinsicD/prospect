# WM-003B held-out online-adaptation plan

Status: queued; planning only, not a sealed protocol.

## Dependency

Begin after WM-003A reaches a terminal disposition. Reuse its frozen benchmark,
context split, model, collection policy, planner, budgets, and zero-shot
checkpoint. Enable matched online updates on held-out contexts as the only
primary causal delta.

## Falsifiable objective

Starting from the same frozen zero-shot checkpoint, one shared Prospect model
reaches a prospectively frozen held-out performance threshold using fewer real
transitions than scratch and retains bounded training-context behavior after
the adaptation budget.

## Required controls

- scratch learner with identical architecture and update budget;
- frozen zero-shot checkpoint with no adaptation;
- context-agnostic pooled checkpoint;
- context-shuffled adaptation transitions; and
- independent per-context learner as a capacity ceiling.

## Gates

1. Freeze held-out adaptation trajectories, transition and optimizer budgets,
   evaluation cadence, threshold, area-under-learning-curve metric, and
   checkpoint schedule before opening adaptation outcomes.
2. Bind every update to a canonical real held-out transition and forbid test
   evaluation outcomes, other contexts, or oracle dynamics from the update.
3. Require either fewer real transitions to the frozen threshold or a better
   paired learning curve than scratch, with zero-shot starting performance
   reported separately.
4. Demonstrate a positive adaptation opportunity: the frozen control must leave
   measurable headroom below the known-dynamics or per-context ceiling.
5. Re-evaluate training contexts after adaptation and enforce a bounded
   short-term retention guardrail; do not interpret this as continual-scale
   retention.
6. Restore the adapted model in a fresh process and reproduce model identity,
   predictions, and executed evaluation actions.
7. Independently recompute ancestry, budgets, learning curves, threshold
   crossing, retention guardrail, and ordered gates from raw rows.

## Exclusions

No zero-shot claim, new acquisition mechanism, adaptive replay, growing task
sequence, partial observability, observation encoder, or general transfer
claim. Retention across many tasks belongs to WM-004.

## Intended repository surface

`bench/world_model_context_adaptation/`, `tests/test_wm003b_*.py`, and only
task-neutral runtime changes proven necessary by the isolated adaptation path.

## Next action

After WM-003A selects a benchmark, derive the cheapest paired learning-curve
matrix that distinguishes warm-start transfer from extra update work and
capacity, then freeze one threshold and one retention guardrail.
