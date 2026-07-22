# Prospect

Prospect is an adaptive-agent runtime for linking collected experience to
persistent state changes, held-out behavior, and retained improvement.

The canonical contract is:

```text
decide -> execute -> observe -> store -> assimilate -> learn -> evaluate
```

Prediction, uncertainty, realized proper score, belief revision, information
gain, decision value, learning, and retention remain distinct. Stable identities
connect every step; there is no universal epistemic scalar or hidden “last
prediction.” See [the architecture](docs/architecture.md).

## Current boundary

The repository contains:

- backend-neutral domain records and protocols;
- exact Bayes, information-value, and proper-score reference semantics;
- transparent action assessment;
- one linked runtime with canonical experience and epistemic stores;
- failure-atomic in-process learning across owned model bytes, runtime state, and
  the update ledger;
- canonical replay custody plus optional TorchRL/TensorDict sampling;
- integrity-checked component checkpoints; and
- an executable probabilistic world-model, CEM control, retention, restart, and
  independent-evidence program in
  [WM-001](bench/world_model_lifecycle/README.md).

WM-001 protocols 1.3.0 and 1.4.0 have each completed one eight-seed formal
attempt. Both producer results passed K0–K7 with strong fixture-specific effects
and exact fresh-process parity. Version 1.4 also passed a direct run of its
corrected pre-bound auditor: 6,393,031 checks, zero failures, and zero coverage
gaps.

Version 1.4 still did not receive formal acceptance. Mandatory adjudication runs
the captured auditor under `python -I`; on the bound machine that hid the
user-site locations from which two bound distributions had been resolved, so
the adjudication-time audit could not reproduce the passing report. The
accepted package was refused, and the harness also refused to package the
failing audit as rejected. The repository therefore still has no accepted
demonstration of the complete claim. The
[v1.4 formal results review](docs/wm001-v140-formal-results.md) preserves the
strong bounded evidence, the exact failure, and the requirements for a new
protocol version. The earlier
[v1.3 review](docs/wm001-v130-formal-results.md) remains immutable history.

## Layout

```text
src/prospect/             current agent implementation
bench/epistemic/          exact semantic and lifecycle references
bench/world_model_lifecycle/
                          WM-001 protocol, implementation, evidence, and runbook
tests/                    active unit, adversarial, and integration tests
docs/architecture.md      canonical system definition
datasets/                 preserved reusable inputs and checksums
.agents/skills/           project research and results-audit skills
```

Generated experiment outputs belong under `bench/**/results/` and remain
untracked. Curated reusable inputs belong under `datasets/` with provenance and
checksums.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
make install
make install-runtime
make check
make check-runtime
make epistemic-diagnostics
make epistemic-gate
python -m bench.world_model_lifecycle.verify protocol
```

`make check` covers the backend-neutral core. `make check-runtime` adds the
world-model implementation and adversarial tests. Direct WM-001 execution
through `make wm001-development` is deliberately disabled so it cannot bypass
the versioned evidence lifecycle. Protocol 1.5 is retired after an unfinalized
development qualification exposed a lazy Gymnasium environment-custody gap;
no result, closure, binding, or formal marker was created. Protocol 1.6 fixed
that boundary, but its sole independent development audit exposed one missing
captured bootstrap support file and retired the version before closure.
Protocol 1.7 made that dependency explicit and its audit passed, but its sole
closure hit a 64 MiB whole-file limit while rechecking a 320 MB authenticated
result. Protocol 1.8 repaired that boundary and completed development, audit,
and closure, but a fresh-process closure recheck exposed nondeterministic
serialization of unordered matrix-contract sets. It retired before preformal
or formal authorization. Protocol 1.9 repaired that defect and completed its
producer, independent audit, closure, and sealed-runtime reopen, but its fixed
preformal suite then exposed a runner test whose closure path was not isolated
from live lifecycle state. The failed report retired v1.9 before binding or
formal launch; no K3–K6 value was opened or used.

Protocol 1.10 completed development, audit, closure, and sealed reopen, but its
preformal QA process re-entered a live runtime-only inventory verifier after
all ten commands had run. The expected QA-only packages caused an ordinary
exception before report publication, consuming the hidden claim without
authorizing a binding or formal launch. No v1.10 performance value was opened
or used.

Protocol 1.11 repaired that composition class and passed its sealed static
gates. Its result-free command-10 rehearsal completed semantically, but
PyTorch 2.9 emitted a benign deprecation warning on stderr when the harness
accessed legacy TF32 precision APIs. Command 10 requires exactly zero stderr
bytes, so the sealed version was retired before any development producer,
experience collection, training, or metric existed. Its terminal disposition
is preserved in the
[v1.11 result-free rehearsal failure](docs/wm001-v1110-result-free-rehearsal-failure.md).

Protocol 1.12 repaired the precision/stderr boundary, passed its result-free
rehearsal, completed development, accepted audit, closure, and all ten
preformal commands, then failed its sole binding transaction. The
formal-binding schema required `bytes >= 1` for every test-log row even though
the ten successful stderr logs were correctly empty. The failed binding was
outer-finalized and no formal launch occurred. Its terminal disposition is
preserved in the
[v1.12 binding-schema failure](docs/wm001-v1120-binding-schema-failure.md).

Protocol 1.13 repaired the stream/schema contradiction, passed command 10,
completed development, accepted audit, closure, and all ten preformal
commands, and assembled a root-schema-valid binding package. Its strict
consumer then applied the canonical live-bundle report verifier to the
preserved report inside the mixed binding directory. That role correctly
rejected the copied path and additional binding sidecars. The failed binding
was outer-finalized and no formal launch occurred. Its terminal disposition is
preserved in the
[v1.13 binding-verifier failure](docs/wm001-v1130-binding-verifier-failure.md).

Protocol 1.14 repaired that report-role boundary and passed its prospective
gates, result-free rehearsal, development, independent audit, closure, fresh
runtime reopen, and all ten preformal commands. Its sole binding transaction
then reached the independent development-archive auditor, which falsely
reported an extra member after verifying all 86 declared members. The archive
had no membership delta: the final check created a second `TarFile` iterator,
which replayed a cached member. The failed binding was outer-finalized and no
formal launch occurred. Its terminal disposition is preserved in the
[v1.14 independent archive-verifier failure](docs/wm001-v1140-development-archive-membership-failure.md).

Protocol 1.15 repaired the archive-reader boundary and passed its prospective
gates, result-free rehearsal, development, accepted audit, closure, fresh
runtime reopen, all ten preformal commands, accepted binding, and the
operator-recorded final stop/go sequence. Its sole formal invocation then
returned `1` before producer
custody. The operator-diagnostic traceback identifies the standard-library
launcher applying its 64 MiB control-file reader to the 320,977,868-byte
development result. The fsynced binding-keyed root is consumed and remains
empty; no formal marker, producer, outcome, audit, review, or adjudication
exists. Its terminal disposition and evidence caveat are preserved in the
[v1.15 formal-invocation failure](docs/wm001-v1150-formal-invocation-failure.md).
Protocol 1.16 repaired the streamed bulk-producer and result-qualification
boundaries and passed its prospective gates, result-free rehearsal,
development, accepted audit, closure, fresh-runtime reopen, all ten preformal
commands, accepted binding, and operator-recorded final stop/go checks. Its
operator-observed exact pre-root accepted-binding rehearsal then exposed a
separate consumer-contract mismatch:
the real audit terminal correctly names a content-addressed reproduction-runtime
sidecar, while the standard-library launcher required the stale fixed execution
filename. The rehearsal returned `1` before child dispatch or formal-root
creation. No formal marker, producer, outcome, audit, review, or adjudication
exists. Its terminal disposition is preserved in the
[v1.16 accepted-binding rehearsal failure](docs/wm001-v1160-accepted-binding-rehearsal-failure.md).
Protocol 1.17 is the active prospective candidate. It follows the authenticated
audit-reproduction receipt instead of duplicating sidecar filenames, exercises
the real 15-member producer-to-consumer audit package, and adds a
binding-keyed single-use rehearsal claim with an outer-finalized terminal.
Formal launch v3 binds that accepted rehearsal under held custody. Its
[confirmation plan](docs/wm001-v1170-confirmation-plan.md) and
[operator runbook](docs/wm001-v1170-operator-runbook.md) are pre-outcome; no
v1.17 development or formal outcome exists yet.
