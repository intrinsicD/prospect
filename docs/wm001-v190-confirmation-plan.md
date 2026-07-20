# WM-001 protocol 1.9.0 confirmation plan

Status: prospective. This document must be sealed with the v1.9 protocol,
schemas, implementation, dependency lock, tests, and independent harness review
before any v1.9 development or formal outcome is produced.

Protocol 1.8.0 is immutable and retired. Its sole development producer,
independent audit, canonical archive, closure marker, and closure attempt were
outer-finalized. A fresh interpreter then rejected the closure because its
matrix-contract digest serialized two frozensets without sorting; process hash
randomization produced a different digest. No v1.8 preformal report, binding,
formal launch, formal audit, semantic review, or adjudication exists. No v1.8
K3–K6 value was opened and none may be inspected, summarized, compared, or
used to select a v1.9 setting.

## Purpose and scientific freeze

WM-001 v1.9.0 permits one fresh-seed confirmation of the unchanged
collect → learn → improve → retain → restart experiment. It constructs every
matrix-contract array in sorted order and binds the canonical digest
`09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84`
in the protocol, closure writer, verifier, and independent auditor. Before the
canonical closure marker is published, a newly exec'd interpreter must reopen
the prospective marker through inherited bootstrap and runtime-seal
descriptors. Before preformal authorization, another sealed runtime must
reopen the accepted, outer-finalized closure attempt and its retained fresh
reopen receipt. Before the development producer exists, the result-free sealed
rehearsal must already execute the same nested descriptor/bootstrap route with
a fresh challenge and reproduce the matrix-contract golden without recursively
acquiring the outer-launch lock.

The repairs may change deterministic matrix serialization, cross-process
verification, closure/preformal authorization, tests, versioned paths,
schemas, seeds, and evidence plumbing. They may not change the world model,
learning algorithm, optimizer, planner, controller, task definitions,
experience or update budgets, controls, metrics, thresholds, exclusions,
killing order, or scientific claim.

The following continuity requirements are fatal gates:

- the canonical SHA-256 of the 17 scientific protocol blocks remains
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- `model.py`, `learning.py`, `planning.py`, and `runtime_lane.py` retain their
  sealed v1.4 source SHA-256 values; and
- every v1.9 protocol, verifier, auditor, prospective review, and result schema
  agrees on the new version, paths, seeds, and exact support-file set.

The v1.8 closure failure is engineering lineage only. It supplies no evidence
for or against the scientific claim.

## Assurance boundary

Every v1.9 seal, binding, operator record, audit receipt, and adjudication
record binds `prospect.wm001.trust-model.v1` with
`tamper_resistant: false`. The kernel, filesystem implementation, base
interpreter and standard library, invoking account, and every process able to
write the repository, environments, or results roots remain trusted. One
protocol-wide nonblocking advisory lock provides cooperative exclusive use for
each complete outer-launcher child lifetime.

Hashes, descriptor custody, stable-file checks, before/after inventories, and
no-replace publication detect accidental or persistent drift within this
boundary. They do not provide hostile-writer resistance, fs-verity, external
attestation, or protection from a privileged or same-principal writer.

## Fresh seed universe

Master seed `i` is the first four bytes, interpreted as an unsigned big-endian
integer, of:

```text
SHA256("WM-001|1.9.0|<lane>-master|<i>")
```

Development indices 0–1 are:

```text
86535224, 2906056242
```

Formal indices 0–7 are:

```text
1369779618, 2721934008, 2798280967, 926105433,
4118470289, 919763803, 2112633694, 2832104894
```

The verifier and independent auditor regenerate all 136 namespace streams for
each current master. They require:

- 10 unique current masters and 1,360 unique current derived streams;
- no current master/master, stream/stream, or master/stream collision;
- 80 unique exposed prior masters and 10,880 unique exposed prior streams from
  v1.0.0, v1.2.0, v1.3.0, v1.4.0, v1.5.0, v1.6.0, v1.7.0, and v1.8.0; and
- zero overlap in all current/prior master/stream cross classes.

No seed is replaced after any v1.9 outcome-producing path is created.

## Exact runtime and environment

The QA and producer environments are fresh, separate, non-editable virtual
environments. The producer runs the one reviewed Prospect wheel under CPython
`-I -S -B`, without `site.py`, user/system site inheritance, `.pth` execution,
or customization modules. Before any package-root import, the launcher and
producer bootstrap drop nonexistent absolute startup search entries, reject
every extant entry outside the inventoried standard-library tree, reject
site-package entries, and rewrite `sys.path` deterministically. This closes
CPython's normally retained but nonexistent `python312.zip` search slot rather
than allowing it to become an unbound import source later. The fixed process
environment is:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
LAZY_LEGACY_OP=False
LC_ALL=C.UTF-8
PATH=/usr/bin:/bin
PYGAME_HIDE_SUPPORT_PROMPT=hide
SDL_AUDIODRIVER=dsp
TZ=UTC
```

Only the protocol-declared accelerator and thread visibility variables may be
added, and every present key/value is bound exactly. The binding and launcher
recompute the interpreter bytes, standard library, complete package-root
inventory, distribution ownership, lock rows, installed Prospect source,
hardware, deterministic settings, and process environment. Missing,
additional, aliased, editable, symlinked, special, unowned, or byte-different
content is fatal.

The prospective development runtime seal and its deterministic completion
marker are the same inode with two links. Formal execution uses only the
singly linked `formal-binding.json` inside the one accepted, outer-finalized
binding attempt.

## Captured bootstrap support repair

`producer_bootstrap.py` is a required captured support for every full outcome
audit. The canonical sorted outcome-audit support set is:

```text
producer_bootstrap.py
protocol.json
schemas/raw-result.schema.json
```

The captured runner writes those exact bytes below its private capture root,
records their sizes and SHA-256 digests in the audit runtime manifest, and
reopens their bytes and descriptor identities after auditor execution.
`artifact_audit.py` may never reach through its temporary `HERE` to an
undeclared sibling.

The two restart-runtime validation branches have distinct provenance:

1. Development has no formal `source` block. Its captured
   `producer_bootstrap.py` must equal the installed, runtime-sealed bootstrap
   execution source and the producer execution identity.
2. Formal has a bound `source` block. Its captured
   `producer_bootstrap.py` must equal the exact
   `source/bench/world_model_lifecycle/producer_bootstrap.py` snapshot and its
   `source.implementation_files` row, as well as the formal producer execution
   identity.

There is no fallback to a repository sibling, a caller working directory, an
uncaptured package-root file, or whichever Prospect installation happens to be
importable. A missing support, an extra support, a digest mismatch, a source
branch mismatch, or a changed captured file is fatal.

The prebinding request uses its own smaller, purpose-specific support set.
Adding `producer_bootstrap.py` to full outcome audits must not silently broaden
unrelated conformance roles.

## Branch-exact result-free qualification

Before the canonical development root exists, the sealed rehearsal must:

1. import Gymnasium, create and close `Pendulum-v1` without reset or step, and
   recheck the exact environment and live bootstrap custody;
2. run the captured runner in private-path and inherited-descriptor source
   modes, with at least three byte-identical repetitions per mode;
3. execute a synthetic development restart-runtime validation with
   `source=None` and the exact three-file outcome support manifest;
4. execute a synthetic formal restart-runtime validation with a bound source
   manifest and snapshot and the same captured support bytes;
5. prove both branches still pass when no repository or installed sibling can
   satisfy an undeclared `HERE/producer_bootstrap.py` lookup;
6. mutate or omit each bootstrap support identity in turn and require
   deterministic refusal; and
7. recheck captured source/support bytes, root inventories, environment, and
   canonical reports after every execution.

The conformance invocation independently carries the sealed
`producer_bootstrap.py` SHA-256; the captured file may not supply its own
expected identity. Its canonical report and complete six-execution receipt
(three path, then three descriptor) are content-addressed and retained. The
preformal command log binds the complete audit-execution block, and formal
binding plus the independent verifier must reconstruct and match that same
block, report, receipt, runtime manifests, and invocation manifests.

These fixtures are result-free: they may not reset or step a task, collect
experience, train a model, read a prior result, inspect K3–K6, or create any
development/formal producer path. Unit tests must separately cover development
and formal branches; one branch may not be mocked away while the other is
claimed covered.

## One audit runner everywhere

Prebinding conformance, launch-time conformance, development audit, formal
audit, and adjudication replay use the same descriptor-capable captured runner.
The runtime manifest contains only pre-bindable interpreter, source, support,
root, standard-library, environment, and limit identities. The invocation
manifest separately binds arguments and working directory. Neither canonical
manifest contains a temporary path, descriptor number, process identifier,
timestamp, future result path, or mutable source path.

Before sealing, adversarial tests must reject user-site and `PYTHONPATH`
injection, extant ambient startup search roots, `.pth` and customization
imports, altered descriptors, support mutation, missing/extra bootstrap
support, noncanonical reports, changed root inventories, JSON type aliases in
the retained report or receipt, and branch provenance substitution. The real
captured child must prove that absent zip entries are dropped and that every
retained standard-library search directory is covered by the bound inventory.
Immediately before formal marker publication, a descriptor-mode replay must
equal the bound manifest and prebinding report byte-for-byte.

## One-shot development pipeline

There is exactly one v1.9 development producer:

```text
bench/world_model_lifecycle/results/development/qualification-v1.9.0
```

Exclusive creation of that root consumes the v1.9 development qualification.
It may never be resumed, overwritten, renamed into place, or replaced by a
numbered sibling. The sole canonical development audit and closure paths are:

```text
bench/world_model_lifecycle/results/operator-v1.9/audits/development-audit-v1.9.0
bench/world_model_lifecycle/results/operator-v1.9/closures/development-closure-v1.9.0
bench/world_model_lifecycle/results/development/development-closure-v1.9.0.json
```

The audit must execute and reproduce with the exact captured bootstrap support
manifest. Qualification depends only on engineering, custody, complete
evidence, exact K7 restart parity, and independent reconstruction. K3–K6
development values are descriptive and excluded from binding and selection.
The closure writer sorts both set-derived matrix arrays, recomputes the golden
digest, and requires a fresh `-I -S -B` child entered directly through the
captured bootstrap and runtime-seal descriptors to reopen the prospective
closure before publishing its canonical marker. The accepted attempt retains
a challenge-bound fresh-reopen report. Recursive launcher entry is forbidden
because the outer launcher already holds the protocol-wide lock.

Any failure after development-root creation—including producer, audit,
reproduction, closure, preformal, binding, or final stop/go failure—retires
v1.9. A terminal failure does not restore authorization merely because a later
artifact is absent. Repair requires a new protocol version and fresh seeds.

## Preformal, binding, and formal pipeline

After an accepted, outer-finalized development audit and closure, the preformal
report runs exactly ten command roles: eight isolated QA roles and two
sealed-runtime roles, with separate stdout/stderr evidence. Its first runtime
role reopens both the canonical marker and accepted outer-finalized closure
attempt, including the retained fresh-reopen report; its stdout is parsed
semantically rather than trusted as a generic zero exit code. It
binds the prospective review, Git commit/tree, protocol and schemas, dependency
lock, complete implementation manifest, development evidence, runtime
inventories, exact outcome support manifest, and branch-exact conformance
reports.
The branch-exact report and full repeated-execution receipt are preserved as
formal binding sidecars, not summarized into an uncheckable pass bit.

The only binding attempt is:

```text
bench/world_model_lifecycle/results/operator-v1.9/bindings/formal-binding-v1.9.0
```

It binds the eight fresh formal masters and the exact captured outcome-audit
runtime. All source, package, support, and launch-time conformance checks finish
before formal marker publication.

The repository-wide formal marker is:

```text
bench/world_model_lifecycle/results/formal/formal-launch-v1.9.0.json
```

The first canonical formal producer root or formal marker consumes the sole
formal attempt according to the runbook. A completed, failed, or interrupted
post-marker formal run cannot be retried. The canonical formal-audit claim and
adjudication claim are also single-use. A report-generation failure consumes
the formal-audit claim and may only lead to rejected evidence. Adjudication
replays an ordinary report exactly once; authenticated no-report failure
evidence receives zero replay and cannot be accepted.

Acceptance requires an outer-finalized formal producer, independently
recomputed K0–K7, a clean byte-reproduced audit, an independent semantic review
accepting every gate, an atomically published accepted adjudication package,
and successful strict post-publication verification. Only that package may
support the bounded WM-001 learning, improvement, retention, and restart
statement.
