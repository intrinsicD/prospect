# WM-001 protocol 1.12.0 confirmation plan

Status: prospective. This document must be sealed with the v1.12 protocol,
schemas, implementation, dependency lock, tests, and independent harness review
before any v1.12 development or formal outcome is produced.

Protocols 1.10.0 and 1.11.0 are immutable and terminally retired. Version
1.10 reached its preformal gate before a cross-environment QA/runtime
composition error consumed that version's hidden claim. Version 1.11 repaired
that class, passed its sealed static gates, created and outer-finalized only
its prospective runtime seal, and completed the result-free command-10
bootstrap/inventory rehearsal semantically. The successful PyTorch 2.9 child
nevertheless emitted a benign TF32 deprecation `UserWarning` on stderr while
the harness accessed legacy `allow_tf32` and float32-matmul-precision APIs.
Command 10's prospective contract requires exactly zero stderr bytes, so the
sealed v1.11 bytes could never authorize preformal evidence and may not be
changed in place. No v1.11 development producer was created, no experience was
collected, no model was trained, and no development or formal metric exists.
No retained K or performance value was opened or used to select a v1.12
setting. The v1.11 runtime seal, same-inode outer completion, runtime lock,
wheel, environments, and console record remain terminal result-free evidence
and cannot authorize, block, resume, or be reused by v1.12. Their disposition
is preserved in the
[v1.11 result-free rehearsal failure](wm001-v1110-result-free-rehearsal-failure.md).

## Purpose and scientific freeze

WM-001 v1.12.0 permits one fresh-seed confirmation of the unchanged
collect → learn → improve → retain → restart experiment. It retains the
v1.10 canonical matrix contract and its digest
`09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84`.
It changes only the evidence machinery that failed or was found unsound:

1. every legacy TF32 getter/setter is replaced with PyTorch 2.9's explicit
   `fp32_precision` hierarchy; the runtime request and report bind the global,
   CUDA-matmul, cuDNN-backend, convolution, and RNN precision strings under
   `prospect.wm001.prebinding-conformance-request.v2` and
   `prospect.wm001.prebinding-conformance.v2`;
2. command 10, its repeated prebinding executions, and restart-runtime
   receipts retain the exact zero-stderr rule, while warnings are made fatal
   in static and fresh-subprocess regression gates;
3. deep package, source, standard-library, ownership, and executable checks
   remain in sealed-runtime closure creation/reopen, binding creation, and
   formal launch, while downstream QA consumers use explicit recorded-evidence
   validation with no ambient-inventory replay;
4. command 9's QA consumer independently parses canonical closure bytes,
   archive roles, member ordering and digests, accepted-attempt custody, and
   the sealed runtime receipt;
5. any ordinary semantic-validation exception becomes an exact failed check
   and truthful nonzero generation envelope, while `BaseException` subclasses
   still propagate;
6. command 10 emits the complete canonical package/root/standard-library/
   ownership inventory and the complete fresh-child identity receipt, allowing
   both advertised hashes to be recomputed;
7. the producer, verifier, independent auditor, and standard-library launcher
   require command 10's zero exit, passing status, empty stderr, typed objects,
   recomputed digests, and exact equality with binding dependencies;
8. the runtime-side binding consumer reopens the report's explicitly recorded
   QA executable and closure instead of substituting its intentionally smaller
   caller environment, and it is exercised read-only before the binding
   one-shot claim;
9. the standard-library launcher requires command 9's complete receipt and
   independently cross-links all five closure, producer, result, attempt, and
   completion digests; it also binds command 10's full conformance object to
   the binding's audit-execution record;
10. QA binding verification cross-links coverage arithmetic, Python, platform,
   closure, and outcome-audit identities to recorded runtime objects instead
   of the QA interpreter;
11. accepted binding and closure-attempt verification uses the recorded role,
   while `verify_live_binding` retains and tests the strict ambient runtime
   guard; and
12. adversarial tests cover every command-9 digest, command-10 object and
   stderr mutations, false closure status, ordinary exceptions, interrupt
   propagation, and genuinely unequal QA/runtime inventories.

The result-free sealed rehearsal still executes the same nested
descriptor/bootstrap route with a fresh challenge and reproduces the
matrix-contract golden without recursively acquiring the outer-launch lock.
Before the canonical closure marker is published, a newly exec'd interpreter
must reopen the prospective marker through inherited bootstrap and
runtime-seal descriptors. Before preformal authorization, another sealed
runtime must reopen the accepted, outer-finalized closure attempt and its
retained fresh-reopen receipt.

The v1.12 repair may change only TF32 runtime configuration and identity,
zero-stderr enforcement, warnings/source regression gates, versioned paths,
schemas, fresh seeds, and the evidence plumbing needed to bind those fields.
It may not change the world model, learning algorithm, optimizer, planner,
controller, task definitions, experience or update budgets, controls, metrics,
thresholds, exclusions, killing order, or scientific claim.

The following continuity requirements are fatal gates:

- the canonical SHA-256 of the 17 scientific protocol blocks remains
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- `model.py`, `learning.py`, `planning.py`, and `runtime_lane.py` retain their
  sealed v1.4 source SHA-256 values; and
- every v1.12 protocol, verifier, auditor, prospective review, and result schema
  agrees on the new version, paths, seeds, and exact support-file set.

The v1.10 and v1.11 failures are engineering lineage only. They supply no
evidence for or against the scientific claim.

## Assurance boundary

Every v1.12 seal, binding, operator record, audit receipt, and adjudication
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
SHA256("WM-001|1.12.0|<lane>-master|<i>")
```

Development indices 0–1 are:

```text
2530568307, 3822916726
```

Formal indices 0–7 are:

```text
402304386, 1582362517, 3717100311, 3870324956,
2551652339, 986753049, 4074588580, 1996653376
```

The verifier and independent auditor regenerate all 136 namespace streams for
each current master. They require:

- 10 unique current masters and 1,360 unique current derived streams;
- no current master/master, stream/stream, or master/stream collision;
- 110 unique exposed prior masters and 14,960 unique exposed prior streams from
  v1.0.0, v1.2.0, v1.3.0, v1.4.0, v1.5.0, v1.6.0, v1.7.0, v1.8.0,
  v1.9.0, v1.10.0, and v1.11.0; and
- zero overlap in all current/prior master/stream cross classes.

No seed is replaced after any v1.12 outcome-producing path is created.

## Exact runtime and environment

The QA and producer environments are fresh, separate, non-editable virtual
environments at
`/home/alex/.venvs/prospect-wm001-v112-reviewed` and
`/home/alex/.venvs/prospect-wm001-v112-reviewed-runtime`. Both install the one
reviewed wheel built in a fresh `prospect-wm001-v112-wheelhouse.*` directory.
The producer runs that wheel under CPython
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

PyTorch 2.9 precision custody uses only its explicit string-valued hierarchy.
The producer and independent CUDA replay select IEEE float32 through
`torch.backends.cuda.matmul.fp32_precision = "ieee"` and
`torch.backends.cudnn.conv.fp32_precision = "ieee"` and
`torch.backends.cudnn.rnn.fp32_precision = "ieee"`. Runtime identity records
the global, CUDA-matmul, cuDNN-backend, cuDNN-convolution, and cuDNN-RNN
`fp32_precision` strings without calling `allow_tf32`,
`get_float32_matmul_precision`, or another deprecated compatibility API. The
auditor restores the exact prior string values after replay. A source guard,
warnings-as-errors unit gate, and fresh pinned-runtime subprocess must reject
any legacy access or warning before a runtime seal can authorize development.

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
   canonical reports after every execution; and
8. capture stdout and stderr separately, require a zero process status and
   canonical passing command-10 stdout, and prove the captured stderr has
   exactly zero bytes before the development producer path may exist.

The conformance invocation independently carries the sealed
`producer_bootstrap.py` SHA-256; the captured file may not supply its own
expected identity. Its canonical report and complete six-execution receipt
(three path, then three descriptor) are content-addressed and retained. The
preformal command log binds the complete audit-execution block, and formal
binding plus the independent verifier must reconstruct and match that same
block, report, receipt, runtime manifests, and invocation manifests.

These fixtures are result-free: the initial Gymnasium smoke subcheck may not
reset or step a task. The captured-auditor conformance subchecks intentionally
reset and step isolated QA-only Pendulum and oscillator fixtures, but they may
not collect experiment experience, train a model, read a prior result, inspect
K3–K6, or create any development/formal producer path. Unit tests must
separately cover development and formal branches; one branch may not be mocked
away while the other is claimed covered.

The exact real command-10 subprocess is a prospective gate, not an informal
smoke test. It must use the reviewed runtime executable, `-I -S -B`, sanitized
environment, captured launch and producer bootstraps, canonical runtime seal,
and CUDA device. Its stderr is never merged into stdout or filtered. A
successful semantic JSON object accompanied by even one stderr byte retires
the version; warning suppression or allowlisting is forbidden.

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

There is exactly one v1.12 development producer:

```text
bench/world_model_lifecycle/results/development/qualification-v1.12.0
```

Exclusive creation of that root consumes the v1.12 development qualification.
It may never be resumed, overwritten, renamed into place, or replaced by a
numbered sibling. Seed-override diagnostics cannot accept a caller-selected
output and are confined to generated children of
`results/development/diagnostics-v1.12.0`; they cannot occupy any lifecycle
namespace. The sole canonical development audit and closure paths are:

```text
bench/world_model_lifecycle/results/operator-v1.12/audits/development-audit-v1.12.0
bench/world_model_lifecycle/results/operator-v1.12/closures/development-closure-v1.12.0
bench/world_model_lifecycle/results/development/development-closure-v1.12.0.json
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

Each operator attempt is claimed by exclusively creating its deterministic
hidden `.staging` sibling. A stranded hidden claim, a canonical attempt, or a
canonical closure marker consumes that stage. The closure marker is never a
resume token: interruption after marker publication forbids a second closure
invocation. Any failure after development-root creation—including producer,
audit, reproduction, closure, preformal, binding, or final stop/go
failure—retires v1.12. A terminal failure does not restore authorization merely
because a later artifact is absent. Repair requires a new protocol version and
fresh seeds.

## Preformal, binding, and formal pipeline

After an accepted, outer-finalized development audit and closure, the preformal
bundle must still be lexically absent. It is created exactly once at:

```text
bench/world_model_lifecycle/results/development/v1.12.0/preformal
```

The generator exclusively creates the deterministic hidden sibling
`.preformal.staging` and fsyncs its parent before the first command; either that
hidden claim or the final bundle consumes the one-shot attempt. Operator
attempt roots are created one component at a time, fsyncing each parent before
the corresponding deterministic hidden claim may be used. The report runs
exactly ten command roles: eight isolated QA roles and two sealed-runtime roles,
with separate stdout/stderr evidence. All ten subprocesses finish and their
outcomes are held in memory while the canonical bundle remains absent. The
generator writes and fsyncs the completed report and 20 logs only under the
hidden claim, then atomically publishes the whole directory with a no-replace
rename, so no command or reader can observe a partial canonical bundle. It
returns nonzero with `passed: false` and the exact
failed-command, identity-check, or semantic runtime-output diagnostic if the
completed report is not valid. An infrastructure interruption may leave only
the hidden claim; it still consumes v1.12 and cannot be deleted and retried.

Command 9 is exactly `runtime-accepted-closure-evidence`. It receives the
canonical closure and accepted closure attempt, and its seven captured input
identities are the closure, closure-attempt terminal, same-inode outer
completion, runtime seal, launch bootstrap, producer bootstrap, and prospective
review. Its canonical JSON stdout and empty stderr are parsed semantically by
both the producer-side verifier and independent auditor; a generic zero exit
code is insufficient. The public runtime CLI does not expose the obsolete
`development-evidence` mode. Command 10 is the sealed result-free
bootstrap/inventory conformance rehearsal. Its canonical stdout contains the
complete package rows, package-root summaries, standard-library summary,
package ownership, and fresh-child identity receipt alongside their recomputed
digests and the branch-exact audit-execution identities. Its command row must
have zero exit, `passed: true`, and an exact empty stderr log. The binding
producer, central verifier, independent auditor, and outer launcher each
require the inventory object to equal the recorded binding dependencies and
independently validate the fresh challenge, distinct process IDs, protocol,
matrix-contract digest, and pass status.

The report binds the prospective review, Git commit/tree, protocol and schemas,
dependency lock, complete implementation manifest, development evidence,
runtime inventories, exact outcome support manifest, and branch-exact
conformance reports.
The branch-exact report and full repeated-execution receipt are preserved as
formal binding sidecars, not summarized into an uncheckable pass bit.

The only binding attempt is:

```text
bench/world_model_lifecycle/results/operator-v1.12/bindings/formal-binding-v1.12.0
```

It binds the eight fresh formal masters and the exact captured outcome-audit
runtime. Before the binding attempt can be accepted, the exact independent
formal-input consumer must reconstruct and validate the report, command 9,
command 10, closure/archive, runtime, source, dependencies, and audit-execution
identity. The accepted attempt retains its canonical
`formal-input-preflight.json` receipt. Attempt verification independently reruns
the same consumer and requires byte-exact receipt equality. Before importing
the producer, the standard-library outer launcher also requires the
terminal-bound receipt and cross-checks its binding, report, closure, and bound
auditor identities. It independently reopens the command-9 and command-10
canonical outputs and requires their semantic-object SHA-256 values to equal
the corresponding receipt fields. The receipt is subsequently preserved in
the formal artifact. All source, package, support, and launch-time conformance
checks finish before formal marker publication.

The repository-wide formal marker is:

```text
bench/world_model_lifecycle/results/formal/formal-launch-v1.12.0.json
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
