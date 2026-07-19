# WM-001 protocol 1.4.0 confirmation plan

Status: historical prospective plan, sealed before any v1.4.0 development or
formal outcome. The completed execution and adjudication failure are recorded
in the [v1.4 formal results review](wm001-v140-formal-results.md).

## Purpose

WM-001 v1.4.0 performs one fresh-seed confirmation of Prospect's existing
collect → learn → improve → retain → restart experiment. It repairs evidence
arithmetic and audit custody exposed by the rejected v1.3.0 attempt. It does
not change the formal agent, world model, optimizer, controller, data budgets,
controls, thresholds, exclusions, killing gates, or scientific claim.

The rejected v1.3.0 result remains immutable and rejected. Its outcomes were
used only to identify an auditor schedule defect and an underspecified
coverage endpoint. They were not used to select a performance threshold or
tune the system.

## Exact coverage contract

For every predictive row, the authoritative inputs are the exact persisted
little-endian IEEE-754 float32 target, member-mean, and member-log-variance
tensors. Scalars are promoted exactly to CPython binary64. In member order
0 through 4:

```text
z = (target - mean) * math.exp(-0.5 * log_variance)
cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
pit = math.fsum(member_cdfs) / 5
```

The finite binary64 PIT is converted with `as_integer_ratio()`. It is covered
iff `20*numerator >= denominator` and
`20*numerator <= 19*denominator`. Each raw row stores:

- `coverage_semantics = "wm001-mixture-pit-binary64-count-v1"`;
- `interval_90_covered_target_count`;
- `coverage_target_count = 4 * transition_count`; and
- `interval_90_coverage = covered / total`.

The independent auditor separately decodes the sidecar, duplicates the
operation sequence, and requires exact semantics, count, total, and fraction
agreement. For formal evidence it selects the arithmetic device only from the
pre-outcome binding, requires the result and live auditor runtime to match that
binding, and rehashes the bound installed dependency bytes. There is no
one-target, floating, self-declared-device, or cross-device tolerance.

Across the eight formal task-A/after-A rows, let `C` be the sum of covered
counts and `T` the sum of target counts (`8 * 6,400`). The unchanged inclusive
K3 bounds pass exactly when:

```text
10*C >= 7*T
100*C <= 99*T
```

Floating row fractions and their descriptive mean do not decide the gate.

## Fresh seeds

Master seed `i` is the first four bytes, interpreted as an unsigned big-endian
integer, of:

```text
SHA256("WM-001|1.4.0|<lane>-master|<i>")
```

Development uses indices 0–1:

```text
2439054559, 3246851043
```

Formal uses indices 0–7:

```text
339970590, 474769515, 550273937, 438984650,
2732731971, 2253809848, 2206960337, 3506881479
```

The verifier and auditor regenerate these masters and all 1,360 namespace
streams. They require 1,360 unique current streams, no internal collisions,
and no master or stream overlap with v1.3.0.

## Pre-binding rehearsal

Exactly one complete, two-seed, full-budget development rehearsal is run
before binding. It is permanently claim-ineligible. Its launch criteria are
engineering and evidence only:

- raw schema and matrix completeness;
- deterministic execution and exact seed parity;
- exact producer/auditor coverage count agreement;
- zero independent-audit failures or evidence gaps;
- component-complete restart parity; and
- complete immutable custody.

Development K3–K6 performance values are descriptive. Passing or failing them
cannot permit or prevent formal launch. No scientific tuning is allowed.

## Binding

After the rehearsal, the exact candidate is committed and the worktree must be
clean. A fresh formal binding records:

- protocol and schema digests;
- clean Git commit, tree, and complete implementation manifest;
- dependency closure and exact runtime;
- the full formal test report;
- Pendulum and independent-oscillator conformance;
- coverage endpoint-neighbor cases and the disclosed v1.3 boundary coordinate;
- producer coverage-source and independent-auditor source digests; and
- the fresh formal master schedule.

The binding and every supporting report are copied into the producer root
before the first reset. Immediately before any outcome-producing operation,
the launcher atomically creates the sole protocol-wide
`results/formal/formal-launch.json`, binding the formal-binding digest, attempt
directory, Git commit, and Git tree, and copies those exact bytes into the
producer root. An existing marker blocks every same-version binding.

## Sole formal attempt and adjudication

The first v1.4.0 formal environment reset begins the only permitted formal
producer attempt. It cannot be resumed or retried in place or in another
directory. A crash, incomplete evidence, producer-gate failure, audit failure,
or semantic-review failure retires v1.4.0. A same-version rerun or corrected
post-outcome auditor cannot upgrade it.

Acceptance requires, in order:

1. immutable producer custody and schema verification;
2. independently reopened raw evidence and exact K0–K7 recomputation;
3. exact coverage counts and integer K3 decision;
4. an audit report bound to the pre-outcome auditor, conformance, and test
   digests, then reproduced byte-for-byte by a fresh adjudication-time run of
   an exclusive descriptor-bound copy of the already-verified auditor bytes;
5. a separate adversarial semantic review; and
6. an external accepted adjudication package binding all evidence digests.

Only an accepted package may support the statement that Prospect collected
experience, learned from it, improved executed behavior, retained that
improvement through conflicting learning, and reproduced it after restart
within the bounded WM-001 environment.
