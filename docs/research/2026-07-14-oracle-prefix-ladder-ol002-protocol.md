# OL-002: administrative full rerun after OL-001 verifier defect

**Status:** frozen before OL-002 formal seeds 0--7  
**Date:** 2026-07-14  
**Scope:** non-gated research evidence; no production, task, ADR, or gate change

## Purpose

OL-002 is an exact administrative full rerun of the component-localization experiment
defined in `docs/research/2026-07-14-oracle-prefix-ladder-protocol.md`.  OL-001 is a
preserved terminal failure: its deterministic semantic rerun reached the final CSV
rendering check, where universal-newline translation made a correct CRLF byte file
compare unequal as text.  The failure and artifact hashes are recorded in
`docs/research/2026-07-14-ol001-verifier-failure.md`.

OL-002 was frozen after OL-001's numeric outcomes were available.  It is therefore
not an independent replication and cannot increase confidence through repetition;
matching OL-001/OL-002 numbers must be treated as one experiment and never
double-counted.  No OL-001 outcome is accepted or copied into OL-002.  OL-002 prepares a fresh
byte-bound copy of BC-001 `b1_r1_d8.npz`, trains formal seeds `0..7` again, repeats
the all-seed replay/parity/exact-ceiling gate, executes the same endpoint ladder and
conditional prefix rule, writes new raw outcomes, and performs the same full
deterministic semantic verification rerun.

## Frozen delta from OL-001

Exactly three administrative/rendering values change:

1. experiment id `OL-001` becomes `OL-002`;
2. schema `oracle-ladder-v1` becomes `oracle-ladder-v2`; and
3. canonical CSV text uses LF row terminators before both writing and verification.

The LF conversion is applied to the complete canonical CSV string after the same
field ordering and row serialization.  It does not alter models, seeds, datasets,
rungs, candidate banks, evaluations, contrasts, thresholds, conditional execution,
or reports derived from numeric results.

The OL-002 protocol record binds:

- the full OL-001 implementation sources reused for models, audits, execution, and
  semantic verification;
- the OL-002 namespace shim and its tests;
- the original OL-001 protocol and this delta protocol;
- the OL-001 failure record and all seven failed-artifact hashes; and
- the same BC-001 parent sources, prompt, portfolio, and frozen dataset evidence.

It also requires the exact OL-001 NumPy runtime (`2.4.6`) and machine-checks equality
of all inherited scientific fields before preparation, execution, and verification.

## Inherited scientific protocol

All scientific definitions and stopping rules are inherited unchanged from the
OL-001 protocol, including:

- eight independent model-seed blocks and four repeated starts per seed;
- A/B penalty, B/C TS-infinity-versus-mean, C/D transition stack, D/E refresh
  interface, and E/F online-encoding/reward-stack contrasts;
- fixed 120+8 common-candidate audits;
- materiality requiring at least 7/8 positive seed directions, at least 20% oracle
  gap closure, and median normalized-regret improvement;
- recovery requiring at least 80% success and at least 50% oracle-gap closure;
- conditional `k={1,2,4,8}` execution only if C-to-D is material; and
- no depth knee unless the full curve executes, every later depth remains recovered,
  and no material seed-level reversal occurs.

OL-002 is one-shot.  Any further defect preserves OL-002 and requires a new
identifier.  Search injection, enlarged search, MuJoCo replication, task activation,
and production edits remain outside this experiment.
