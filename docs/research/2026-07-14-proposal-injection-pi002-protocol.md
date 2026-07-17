# PI-002: administrative full rerun after PI-001 verifier defect

**Status:** frozen before PI-002 formal model seeds `0..7`  
**Date:** 2026-07-14  
**Scope:** non-gated BridgeControl research evidence; no production, task, ADR, or
benchmark-gate change

## Purpose

PI-002 is an exact administrative full rerun of the scientific experiment frozen in
`docs/research/2026-07-14-proposal-injection-pi001-protocol.md`. PI-001 is a preserved
terminal failure: after writing a complete package, its final report verifier rendered
two rescue bullets in JSON-sorted order while the saved report retained their in-memory
insertion order. The failure and all artifact hashes are recorded in
`docs/research/2026-07-14-pi001-verifier-failure.md`.

PI-002 was frozen after PI-001's numeric outcomes existed. It is therefore not an
independent replication and cannot increase scientific confidence by repetition.
PI-001 outcomes are neither accepted nor copied. PI-002 copies the same frozen BC-001
dataset, retrains formal seeds `0..7`, repeats every replay/parity/ceiling check, runs
all primary arms, applies the same conditional branch rule, creates fresh outcomes,
and undergoes a full deterministic semantic regeneration.

## Frozen delta from PI-001

Exactly three administrative/rendering values change:

1. experiment id `PI-001` becomes `PI-002`;
2. schema `proposal-injection-v1` becomes `proposal-injection-v2`; and
3. rescue-report bullets are rendered in lexicographic key order before both writing
   and verification, independent of mapping insertion or JSON serialization order.

There is no scientific change to data, learner, seeds, training, starts, success
predicate, planner, TS-infinity scorer, uncertainty coefficient, candidate budget,
injection count/position, exact-reference generator, negative transformation, exact
ceiling, recorded diagnostics, rescue thresholds, conditional branch, or
interpretation boundary.

The PI-002 protocol binds:

- all seven failed PI-001 artifacts and their manifest;
- the exact PI-001 source snapshot and runtime;
- the PI-001 protocol, loop prompt, trigger analysis, and failure record;
- the PI-002 namespace/rendering shim and its tests;
- the same sealed BC-001 and OL-002 parents.

## Inherited scientific protocol

PI-002 inherits the complete PI-001 protocol, including:

- eight independently trained model seeds and four repeated starts;
- native zero-penalty TS-infinity, eight exact-reference injections, action-coordinate
  permuted injections, and exact raw ceiling;
- exactly 64 learned candidates × three iterations × horizon 12 per primary MPC call;
- disabled-injection parity and byte-bound OL-002 native/exact replay;
- specific rescue requiring at least 7/8 positive seed-return differences, at least
  50% paired oracle-gap closure, at least 80% success, and no matching permuted rescue;
- enlarged 512×5 learned search only after specific privileged rescue;
- time-permuted control only after non-specific rescue;
- action-commitment audit without enlarged search after no privileged rescue; and
- the 50% statewise-transfer/final-retention audit thresholds.

PI-002 is one-shot. Any additional defect preserves PI-002 and requires a new
identifier. No PI-001 result may determine an arm, threshold, seed, or interpretation
outside the already frozen PI-001 branch rule.
