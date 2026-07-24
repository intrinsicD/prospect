# WM-004 bounded continual-scale plan

Status: queued; planning only, not a sealed protocol.

## Dependency

Begin after WM-003B reaches a terminal disposition. Freeze its task/context
representation, collection policy, model family, optimizer, planner, and
evaluation procedure so replay and bounded memory are the primary causal delta.

## Falsifiable objective

Under matched memory, real-transition, update, planning, and compute budgets,
an adaptive replay policy retains more prior-context behavior while preserving
new-context plasticity across an increasing task sequence than static balanced
or random replay.

## Required controls

- naive sequential learning with no replay;
- WM-001-style static balanced replay;
- random/reservoir replay at the same capacity;
- full replay as an unbounded-memory ceiling; and
- independent per-task models as a capacity ceiling, not an admissible agent.

## Gates

1. Freeze prospective task orders, task counts, capacity sweep, memory bytes,
   transition counts, optimizer work, and evaluation schedule.
2. Demonstrate that the naive learner both learns each new task and measurably
   interferes with at least one earlier task; without interference, retention
   is not tested.
3. Measure forward transfer, immediate plasticity, backward transfer/forgetting,
   worst-task retention, and average return separately.
4. Require the adaptive method to improve a prospectively frozen retention
   endpoint over both bounded static controls without an unacceptable
   plasticity loss.
5. Report performance, storage, update work, and wall-clock scaling at every
   task-count/capacity point; a larger hidden budget invalidates comparison.
6. Bind every replay sample to canonical experience ancestry and forbid
   imagined transitions from the real replay namespace.
7. Restore the complete long-sequence state in a fresh process and reproduce
   replay identities and bounded evaluation behavior.

## Exclusions

No change to acquisition, benchmark family, model capacity within a comparison,
observation modality, partial-observation horizon, or external evaluation.
Success does not establish unbounded continual learning.

## Intended repository surface

`bench/world_model_continual_scale/`, `tests/test_wm004_*.py`, and a reusable
task-neutral replay contract in `src/prospect/` only if the experiment proves
that the contract is required beyond this fixture.

## Next action

Specify the smallest three-point scale test—task counts, memory capacities, and
orders—that can expose both interference and a difference between adaptive and
static replay before committing to a full benchmark sequence.
