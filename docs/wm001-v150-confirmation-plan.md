# WM-001 protocol 1.5.0 confirmation plan

Status: prospective plan. This document is sealed before any v1.5.0
development or formal outcome. Protocol 1.4.0 remains immutable and retired;
its completed execution and adjudication failure are recorded in the
[v1.4 formal results review](wm001-v140-formal-results.md).

## Purpose and scientific freeze

WM-001 v1.5.0 permits one fresh-seed confirmation of Prospect's existing
collect → learn → improve → retain → restart experiment. It repairs the
execution and evidence boundary exposed by v1.4:

- the direct auditor and adjudicator saw different dependency closures;
- prebinding did not execute the exact adjudication-time isolation path;
- launch custody still contained fallible work after the sole marker;
- a conformance-failing audit could not enter an official rejected package;
  and
- package publication was not directory-atomic or independently reopened.

The v1.5 repair may change only dependency isolation, source and runtime
custody, prebinding conformance, development qualification, launch ordering,
audit execution, rejection preservation, and package publication. It may not
change the world model, learning algorithm, optimizer, planner, controller,
experience budgets, controls, metrics, thresholds, exclusions, killing order,
or scientific claim.

This freeze is machine-checked in two ways:

1. the canonical SHA-256 of the 17 scientific protocol blocks must remain
   `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
2. `model.py`, `learning.py`, `planning.py`, and `runtime_lane.py` must retain
   their sealed v1.4 source SHA-256 values.

No v1.3 or v1.4 outcome is used to tune a scientific value. Those outcomes
only motivate the named evidence-harness repairs.

## Assurance and trust boundary

Every runtime, binding, operator, and adjudication record binds
`prospect.wm001.trust-model.v1` with `tamper_resistant: false`. The kernel,
filesystem implementation, base interpreter and standard library, invoking
account, and every process able to write the repository, isolated environment,
or results roots are trusted. Each WM-001 invocation has exclusive use of
those paths, enforced between cooperating harness processes by one
protocol-wide nonblocking advisory lock held by the standard-library outer
launcher for the complete child lifetime of each sealed-runtime invocation.
QA-only preformal commands remain covered by the exclusive-use assumption.
The lock is rooted at the canonical repository derived from the captured
bootstrap, so changing the caller's working directory cannot create a second
runtime-lock namespace.

Hashes, descriptor checks, pre/post inventories, and no-replace publication
detect accidental or persistent drift and application-level overwrite. They
do not resist the environment owner, a cooperating or malicious
same-principal writer, transient mutate-and-restore attacks, privileged
actors, or a compromised kernel. Here, “immutable” means protocol-level
append-only/no-replace evidence under that boundary. No fs-verity, read-only
root, TPM, external signer, or external attestation is claimed.

## Fresh seed universe

Master seed `i` is the first four bytes, interpreted as an unsigned big-endian
integer, of:

```text
SHA256("WM-001|1.5.0|<lane>-master|<i>")
```

Development indices 0–1 are:

```text
4085517670, 2227535912
```

Formal indices 0–7 are:

```text
1800791691, 1963228177, 2416009491, 3925214220,
1508934628, 2118526007, 4212585034, 530094003
```

The verifier and independent auditor regenerate all 136 namespace streams for
each of the ten current masters. They require:

- 1,360 unique current derived streams;
- no current master/master, stream/stream, or master/stream collision;
- 40 unique exposed prior masters and 5,440 unique exposed prior streams from
  v1.0.0, v1.2.0, v1.3.0, and v1.4.0; and
- zero overlap in all four current/prior master/stream cross classes.

## Isolated runtime contract

The producer uses a dedicated virtual environment with no editable
installation and no system-site inheritance. A standard-library-only bootstrap
adds the single verified package root without running `site.py`, `.pth` files,
or customization modules, and the formal process runs the non-editable Prospect
wheel under CPython `-I -S -B`. Its environment begins from
`env -i` and contains only the fixed locale, timezone, executable search path,
cuBLAS determinism setting, the fixed TorchRL `LAZY_LEGACY_OP=False` mode, and
explicitly bound accelerator/thread visibility variables.

The binding records and launch recomputes:

- the absolute CPython executable and its bytes;
- a stable inventory of the standard library;
- every explicitly authorized package root and every regular file below it;
- every installed distribution, with PEP 503 canonical name, version,
  non-editable status, declared-file count, and a digest over every stable
  `RECORD` declaration, including declarations without embedded hashes;
- the marker-resolved closure of the scientific roots; and
- exact producer flags, hardware, Torch/CUDA/driver/thread settings, and safe
  process environment.

Duplicate distributions, missing declared files, editable installs, symlinks,
special files, unowned additions, or any byte difference are fatal. The wheel
copy of every executed Prospect and WM-001 source must equal the committed
source manifest.

There are two typed runtime-seal cases. The prospective development seal is
itself an outer-finalized file: the seal and its deterministic completion
marker are the same inode with two links. Formal execution instead uses
`formal-binding.json`, which remains singly linked inside the one canonical
accepted binding attempt; its sibling terminal `operator-attempt.json` and
deterministic outer-completion marker establish the attempt's publication.
Copied, direct, failed, or unfinalized binding files are not runtime seals.

## Outcome-free preflight repair

The first development invocation stopped at the initial live-bootstrap custody
check, before any task reset, experience collection, training, evaluation, or
result write. The failed producer directory
`results/development/qualification-v1.5.0` and its original outer-finalized
`runtime-seal-v1.5.0.json` remain preserved as rejected evidence. No formal
marker was created.

The failure exposed one omitted environment input: importing the frozen
TorchRL dependency materialized its default `LAZY_LEGACY_OP=False` setting
after the pre-import seal, so the exact pre/post environment comparison
correctly rejected the process. The repair explicitly fixes and binds that
same default value, adds a fresh-process no-environment-mutation regression,
and moves the only outcome-producing development qualification and its runtime
seal to the non-overwriting `*-attempt-2` paths. No scientific component,
seed, budget, controller, metric, or threshold changed.

## One audit runner everywhere

Prebinding conformance, the official independent audit, launch-time
conformance, and adjudication replay use one captured runner. It launches the
auditor under `-I -S -B`, imports only the bound roots, and receives:

- the runtime manifest through an inherited regular-file descriptor;
- the invocation envelope through a separate inherited descriptor;
- the bootstrap through an inherited descriptor;
- the auditor through an exclusive private path or inherited descriptor; and
- support inputs through private captured-file tokens.

The runtime manifest contains only pre-bindable source, bootstrap, interpreter,
root, and safe-environment identities. Future artifact paths and arguments are
kept in the per-run invocation envelope. The canonical conformance report
contains no temporary path, file-descriptor number, process identifier,
timestamp, future outcome path, or mutable source path.

Auditor stdout is captured through an exclusive private regular-file
descriptor with a fixed OS-enforced byte ceiling; callers cannot select an
output path. This is the same capture mode used in prebinding, official audit,
and adjudication replay.

Before binding, private-path and inherited-descriptor source modes each run at
least three times. Stdout, stderr, return code, report bytes, source identities,
support identities, and normalized manifests must agree exactly. Adversarial
tests inject user-site packages, `PYTHONPATH`, `PYTHONUSERBASE`, `.pth`,
`sitecustomize`, mutated support files, changed descriptors, and noncanonical
reports. Every leak or mismatch must be rejected.

Immediately before formal marker publication, one descriptor-mode rerun must
match the bound runtime-manifest and canonical prebinding-report bytes exactly.

## Development qualification and closure

Exactly one fresh, complete, two-seed, full-budget development rehearsal is
used to qualify the candidate. It remains permanently claim-ineligible.
Qualification depends only on engineering and custody:

- exact v1.5 schema, seed, budget, and evidence-matrix completeness;
- deterministic producer execution;
- complete producer-manifest custody;
- exact predictive target/count reconstruction;
- zero independent-audit integrity failure or coverage gap;
- exact component-complete fresh-process restart parity;
- isolated runner execution with the bound dependency closure;
- byte-identical supplied and adjudication-mode audit reports; and
- passing adversarial package and runner tests.

K3–K6 development performance values are descriptive and must not be copied
into, compared by, summarized in, or selected by the binding.

Qualification ends by exclusively creating
`results/development/development-closure-v1.5.0.json`. The marker binds the
producer manifest, result, independent audit, audit-reproduction receipt,
runner manifest, auditor source, Git commit, and Git tree. Once it exists, the
v1.5 development CLI refuses every further rehearsal. A source, dependency,
protocol, schema, runner, or qualification change requires a new protocol
version.

Development publication is a three-step chain. First, the producer's terminal
manifest is committed by the outer launcher. Second, the canonical development
audit operator attempt is accepted and outer-finalized. Third, the closure
operator consumes that exact audit attempt, creates the exclusive closure
marker, and outer-finalizes the canonical closure attempt. Binding accepts no
other closure path or copy.

## Preformal stop/go audit

Before binding, the complete candidate must pass exactly ten command rows:

1. QA protocol/seal/scientific-continuity verification;
2. QA Ruff;
3. QA core mypy;
4. QA WM-001 mypy;
5. QA epistemic tests;
6. QA world-model tests;
7. QA real-subprocess audit-runner adversarial tests;
8. QA independent prospective-harness review verification;
9. sealed-runtime development producer/audit/closure evidence verification; and
10. sealed-runtime bootstrap, dependency, standard-library, package-root,
    installed-source, Pendulum, oscillator, coverage, and deterministic-tensor
    conformance.

The first eight rows run in a non-editable QA environment; the final two run
through the descriptor bootstrap in the exact prospective runtime. Each row
has separate stdout and stderr evidence, for exactly 20 ordered logs. Before
and after the QA rows, the report binds the exact QA distribution closure with
no editable installs. Before and after the runtime rows, it binds the runtime
executable, prospective seal, development closure, review, and complete
inventory. The machine-readable report records roles, command lines, exit
codes, output digests, Git commit, Git tree, both closure snapshots, and
overall status. A missing, mocked-only, noncanonical, or failing check is a
no-go.

## Formal binding and launch

The formal binding copies and binds:

- protocol, raw schema, binding schema, and dependency lock;
- clean Git commit/tree and the complete implementation manifest;
- the machine-readable test report and development closure;
- every execution-source digest, including runner, bootstrap, auditor,
  verifier, experiment launcher, binder, and adjudicator;
- the complete runtime, standard-library, package-root, and distribution
  inventories;
- full Pendulum, oscillator, and coverage reports;
- the isolated prebinding request, runtime manifest, bootstrap bytes, and
  canonical repeated-conformance report; and
- the eight fresh formal masters.

The binding is created only by the canonical binding operator. It consumes the
accepted outer-finalized closure attempt, publishes
`results/operator-v1.5/bindings/formal-binding-v1.5.0`, and registers that
attempt's terminal manifest with the outer launcher. Formal execution requires
the exact `formal-binding.json` inside that attempt.

All input preservation, live-binding verification, binding-attempt
verification, source/package inventory recomputation, and isolated launch-time
conformance finish before the marker. The producer copies both the binding
attempt terminal and its completion bytes. The launcher then:

1. writes and fsyncs canonical `formal-launch.json`, binding the canonical
   binding-attempt path plus terminal/completion identities, in the producer
   root;
2. atomically hard-links that same inode as the protocol-wide
   `results/formal/formal-launch-v1.5.0.json`; and
3. fsyncs both parent directories.

The retired unversioned v1.4 marker remains immutable. It neither blocks nor
authorizes v1.5. The first v1.5 formal environment reset after marker
publication begins the sole permitted formal attempt. It can never be resumed
or rerun under v1.5. On success or failure, the child writes
`producer-manifest.json`. Only after the child returns a coherent logical
status and the outer launcher rechecks the captured bootstrap, seal, receipt,
terminal bytes, and path does it hard-link that exact manifest to the
deterministic outer-completion marker. That link is the logical commit; an
unfinalized producer root is not evidence.

## Audit, semantic review, and atomic adjudication

The official audit remains outside the outer-finalized producer root. Exactly
one version-scoped formal-audit claim exists. Its canonical operator attempt
must terminate as accepted, rejected, or failure evidence and then receive its
outer-completion marker; even report-generation failure consumes the claim.

Audit identity fields say what result, source, runtime, and binding were
examined. Status fields separately say whether conformance, integrity,
engineering completeness, and claim completeness passed. A coherent failed
audit—or a canonical audit-execution-failure record if no ordinary report can
be emitted—must therefore remain publishable only as rejected evidence.
Identity mismatch is never converted into a clean status failure.

Adjudication consumes only that canonical audit attempt. An ordinary audit
report receives exactly one bound replay and must reproduce byte-for-byte. An
audit-execution failure, an authenticated attempt lacking outer completion, or
another no-report terminal receives zero replays and can only produce a
rejected package. After an independent version-2 adversarial semantic review,
adjudication consumes its own version-2 one-shot claim, durably records replay
start before the sole runner entry, builds a hidden staging directory, fsyncs
every member and directory, verifies the staged version-7 package, and
publishes it with one atomic no-replace directory rename. An ordinary
post-claim fault is automatically preserved as a strictly rejected
`adjudication_recovery_failure` without replay. Explicit sealed recovery of an
abrupt marker-only/hidden-staging interruption also performs no replay; an
already renamed exact package is only reverified and outer-finalized unchanged.
Completed recovery is single-use and refused. The strict verifier reopens the
final directory and rejects a missing, extra, symlinked, noncanonical,
digest-mismatched, nonterminal, replay-count-inconsistent,
disposition-inconsistent, recovery-inconsistent, or upstream-mutated member.
The package manifest is the adjudication terminal registered with the outer
launcher.

Acceptance requires all of the following:

1. one immutable completed formal producer root;
2. exact producer verification and independently recomputed K0–K7;
3. a clean audit generated and reproduced by the bound runner;
4. a separate semantic review accepting every gate without fatal finding;
5. an atomic accepted adjudication package; and
6. successful post-publication package verification.

A crash, producer failure, audit-execution failure, audit failure, semantic
rejection, or package-verification failure retires v1.5.0. No corrected audit,
code repair, dependency repair, or same-version rerun may upgrade it.

Only a verified accepted package may support the bounded statement that
Prospect collected experience, learned from it, improved executed behavior,
retained that improvement through conflicting learning, and reproduced it
after a fresh-process restart in WM-001.
