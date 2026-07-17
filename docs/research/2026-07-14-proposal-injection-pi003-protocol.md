# PI-003: administrative full rerun after PI-002 semantic-verifier defect

**Status:** frozen before PI-003 formal model seeds `0..7`  
**Date:** 2026-07-14  
**Scope:** non-gated BridgeControl research evidence; no production, task, ADR, or
benchmark-gate change

## Purpose

PI-003 is a fresh administrative rerun of the unchanged PI-001 proposal-injection
science. PI-001 remains the terminal report-order failure. PI-002 fixed report order,
reran every formal outcome, and passed the complete persisted-package verifier; its
separate semantic regeneration then compared JSON-decoded list fields with regenerated
tuple fields as raw Python objects. A second regeneration established equality after
canonical JSON normalization. PI-002 and its hashes are preserved in
`docs/research/2026-07-14-pi002-semantic-verifier-failure.md`.

PI-003 was frozen after both prior outcomes existed. None is an independent
replication, none may determine a new scientific choice, and none may be copied.
PI-003 retrains/evaluates every seed and applies the original frozen branch rule.

## Frozen delta from PI-002

Exactly three administrative/verifier values change:

1. experiment id `PI-002` becomes `PI-003`;
2. schema `proposal-injection-v2` becomes `proposal-injection-v3`; and
3. full semantic verification converts the regenerated result fields through the same
   finite, sorted-key JSON representation used by the artifact before comparing them
   with JSON-decoded saved fields.

PI-003 retains PI-002's lexicographically canonical rescue-report ordering. There is
no scientific change to data, learner, seeds, training, evaluation, planner, scoring,
injection, controls, metrics, thresholds, conditional branch, or interpretation.

The canonicalization is limited to representation equivalence: it may convert tuples
to JSON arrays/lists and mapping order to sorted-key order. It does not round floats,
drop fields, reorder arrays, tolerate non-finite values, change numeric types by
coercion, or use approximate equality. Any numeric, string, boolean, null, array-order,
or mapping-content difference still fails semantic verification.

## Inherited scientific protocol

PI-003 inherits every scientific field of PI-002 and PI-001:

- formal seeds `0..7`, frozen learner/data/schedule, four starts, and 14-step episodes;
- native zero-penalty TS-infinity, exact-reference injection, action-permuted control,
  and exact raw ceiling;
- unchanged 64×3×12 primary learned budget and eight first-round replacements;
- strict disabled-injection and OL-002 replay parity;
- rescue thresholds of 7/8 positive seeds, 50% exact-gap closure, and 80% success;
- enlarged search only after specific privileged rescue, time-permuted control only
  after non-specific rescue, and the action-commitment audit after no rescue; and
- all scope, abandonment, and no-post-hoc-arm rules.

PI-003 binds all PI-001 and PI-002 failed artifacts, source snapshots, protocols,
failure records, the canonical semantic-diagnostic finding, the new namespace shim,
and the unchanged parents. It is one-shot. Any further failure preserves PI-003 and
requires a new identifier.
