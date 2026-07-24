# A-001 durable exactly-once recovery plan

Status: queued; may proceed in parallel with WM-002.

## Dependency

Build on the current transactional learner, append-only experience and update
ledgers, component checkpoint manifests, and declared episode-boundary resume
contract. This is an engineering-assurance task, not an agent-learning claim.
The initial failure model is an abrupt single-process death on a local durable
filesystem with documented atomic-rename and fsync semantics. Power loss,
filesystem corruption, kernel faults, distributed writers, and storage-device
lying are explicitly outside A-001.

## Falsifiable objective

For an abrupt process death at every durable mutation boundary of a learning
transaction, fresh-process recovery produces exactly either the complete
pre-transaction state or the complete committed state, never a split state or
duplicate update.

## Gates

1. Specify one versioned transaction state machine, write ordering, fsync
   boundary, ownership rule, idempotency key, and recovery decision for every
   persistent component.
2. Enumerate all model, ledger, runtime, checkpoint, and journal mutation points
   and demonstrate fault injection immediately before and after each one.
3. After every injected death, require agreement among model bytes/version,
   update receipt, consumed experience ancestry, runtime snapshot, and
   transaction disposition.
4. Run recovery repeatedly and require byte-identical state after the first
   successful recovery; no duplicate receipt, replay insertion, or model
   update is permitted.
5. Reject truncated, reordered, duplicated, checksum-invalid, unknown-version,
   and stale-generation recovery records without silently guessing.
6. Reopen the recovered state in a fresh process and complete one subsequent
   valid learning transaction.

## Exclusions

No mid-episode environment recovery, non-idempotent external side effects,
concurrent lifecycle operations, cross-machine parity, or hostile-writer
resistance. Those belong to A-002 through A-005b.

## Intended repository surface

`src/prospect/runtime/recovery.py`, a narrowly scoped durable journal under
`src/prospect/storage/`, `tests/test_a001_crash_recovery.py`, and an adversarial
crash-matrix runner under `bench/assurance/`.

## Next action

Inventory every durable write in the current learn/checkpoint path and write
the transaction-state and crash-point table before implementing a journal.
