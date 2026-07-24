# A-002 lifecycle-concurrency plan

Status: queued; planning only.

## Dependency

Begin after A-001 defines the durable transaction state machine. Reuse its
operation identities and recovery dispositions rather than introducing a
second commit model.

## Falsifiable objective

Concurrent decide, observe, learn, checkpoint, and restore operations are
linearizable to one valid sequential lifecycle history, or are rejected before
producing externally visible partial state.

## Gates

1. Declare the allowed concurrent operations, lock order, snapshot cutoffs,
   version preconditions, cancellation behavior, and observable commit points.
2. Generate adversarial schedules covering every operation pair and critical
   boundary, including exceptions and cancellation while locks are held.
3. Require every completed history to admit a valid sequential ordering with
   unique experience, decision, receipt, and checkpoint identities.
4. Require stale snapshots, double observations, cross-agent records, and
   predecessor-version races to fail without mutation.
5. Prove or test freedom from deadlock, lock inversion, leaked locks, and
   starvation within a prospectively bounded schedule.
6. Combine representative concurrent failures with A-001 process recovery and
   require the same terminal state.

## Exclusions

No distributed consensus, multi-host shared storage, external tool
linearizability, performance-scaling claim, or hostile process outside the
declared cooperative runtime.

## Intended repository surface

Focused runtime synchronization changes, `tests/test_a002_concurrency.py`, and
deterministic schedule fixtures under `bench/assurance/`.

## Next action

Draw the current lifecycle operation/lock graph and enumerate conflicting
operation pairs. Select the smallest schedule set that covers every shared
mutable owner before changing locks.
