# WM-001 protocol 1.20.0 confirmation plan

Status: prospective. This document must be sealed with the v1.20 protocol,
schemas, implementation, dependency lock, tests, and independent harness review
before any v1.20 development or formal outcome is produced.

Protocols 1.10.0 through 1.19.0 are immutable and terminally retired. Version
1.19 passed every prospective gate, completed its sole two-seed development
producer, and completed the full outcome audit twice with byte-identical
outputs and independently verified capacity evidence. Its sole development
closure then rejected the valid outer-finalized producer manifest because a
generic canonical-JSON reader silently required one link while the terminal
and its outer completion are deliberately one two-link inode.

The authenticated closure attempt is a terminal RuntimeError failure, and no
canonical closure, preformal bundle, binding, formal authority, or scientific
outcome exists. Every v1.19 producer, audit, failed-closure, seal, lock, and
outer-completion byte remains immutable and cannot be removed, resumed,
upgraded, repaired, or reused. No v1.19 K3–K6 value or other scientific
performance value was manually opened or used to select the repair. The exact
disposition is preserved in the
[v1.19 development-closure custody failure](wm001-v1190-development-closure-custody-failure.md).

The v1.20 protocol directly supersedes the exact sealed v1.19 protocol bytes
with SHA-256
`07c6fe364aeddbd5689fa4f638a6f9a38506b16e8845a947fffa87e01eb3854a`.
Skipping that immediate lineage or pointing at an earlier protocol is fatal.

## Purpose and scientific freeze

WM-001 v1.20.0 permits one fresh-seed confirmation of the unchanged
collect → learn → improve → retain → restart experiment. It retains the
v1.10 canonical matrix contract and its digest
`09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84`.
It changes only the finalized producer-manifest custody seam and the test
composition needed to repair that defect:

1. `create_development_closure()` captures the live finalized
   `producer-manifest.json` under an explicit two-link contract;
2. the closure canonical-parses and hashes those same captured bytes, requires
   byte equality with the public producer verifier's object, and performs no
   loose second digest read;
3. the binding layer's stable-file readers require every caller to state its
   one-link or two-link custody contract instead of inheriting a silent
   default;
4. every other live terminal-manifest consumer is audited for the same
   lifecycle transition from one-link precommit to two-link outer-finalized
   custody;
5. a real positive composition regression crosses the outer-finalized producer
   into the actual closure creator, while one-link, extra-link, alias, mutation,
   and verifier-divergence cases fail at that exact seam;
6. formal-binding v10, raw-result v9, protocol v9, prebinding v2, preformal v2,
   closure v2, runtime-seal v1, audit-runtime-manifest v2,
   captured-audit-execution v2, audit-reproduction v3,
   adjudication-audit-execution v3, formal-input-preflight v1, rehearsal v1,
   and formal-launch v3 remain unchanged;
7. the v1.19 role-specific timeout, measured elapsed receipts, capacity
   arithmetic, 8 GiB aggregate ceiling, and 2 GiB result ceiling are inherited
   exactly and remain independently recomputed at every consumer;
8. the central verifier and independent auditor retain exact historical tuple
   equality and add v1.19 to the prior master/stream universe, producing 190
   prior masters and 25,840 unique prior streams with zero collision in every
   declared class; and
9. every active namespace, master seed, environment, wheel, lock, seal,
   prospective review, development path, binding path, and formal authority is
   fresh for v1.20.

The result-free sealed rehearsal still executes the same nested
descriptor/bootstrap route with a fresh challenge and reproduces the
matrix-contract golden without recursively acquiring the outer-launch lock.
Before the canonical closure marker is published, a newly exec'd interpreter
must reopen the prospective marker through inherited bootstrap and
runtime-seal descriptors. Before preformal authorization, another sealed
runtime must reopen the accepted, outer-finalized closure attempt and its
retained fresh-reopen receipt.

The v1.20 repair may change only explicit finalized-manifest custody,
explicit binding-reader link contracts, single-capture object/digest
derivation, exact-seam regressions, versioned paths, and fresh seeds. It does
not change formal-binding v10, raw-result v9, or the v1.19
audit-runtime/capacity contract. It may not change the world model, learning algorithm,
optimizer, planner, controller, task definitions, experience or update
budgets, controls, metrics, thresholds, exclusions, killing order, or claim.

The following continuity requirements are fatal gates:

- `experiment.revision.supersedes` is exactly `1.19.0` and its pinned
  superseded-protocol SHA-256 is exactly
  `07c6fe364aeddbd5689fa4f638a6f9a38506b16e8845a947fffa87e01eb3854a`;
- the canonical SHA-256 of the 17 scientific protocol blocks remains
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- `model.py`, `learning.py`, `planning.py`, and `runtime_lane.py` retain their
  sealed v1.4 source SHA-256 values; and
- every v1.20 protocol, verifier, auditor, prospective review, and result schema
  agrees on the new version, paths, seeds, and exact support-file set.

The v1.10 through v1.19 failures are engineering lineage only. They supply no
evidence for or against the scientific claim.

## Assurance boundary

Every v1.20 seal, binding, operator record, audit receipt, and adjudication
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
SHA256("WM-001|1.20.0|<lane>-master|<i>")
```

Development indices 0–1 are:

```text
3626676950, 2572962267
```

Formal indices 0–7 are:

```text
3772418031, 1586188972, 155797552, 2704051827,
818738828, 4077496645, 1566512625, 2151461680
```

The verifier and independent auditor regenerate all 136 namespace streams for
each current master. They require:

- 10 unique current masters and 1,360 unique current derived streams;
- no current master/master, stream/stream, or master/stream collision;
- 190 unique exposed prior masters and 25,840 unique exposed prior streams from
  v1.0.0, v1.2.0, v1.3.0, v1.4.0, v1.5.0, v1.6.0, v1.7.0, v1.8.0,
  v1.9.0, v1.10.0, v1.11.0, v1.12.0, v1.13.0, v1.14.0, v1.15.0,
  v1.16.0, v1.17.0, v1.18.0, and v1.19.0; and
- zero overlap in all current/prior master/stream cross classes.

No seed is replaced after any v1.20 outcome-producing path is created.

## Exact runtime and environment

The QA and producer environments are fresh, separate, non-editable virtual
environments at
`/home/alex/.venvs/prospect-wm001-v120-reviewed` and
`/home/alex/.venvs/prospect-wm001-v120-reviewed-runtime`. Both install the one
reviewed wheel built in a fresh `prospect-wm001-v120-wheelhouse.*` directory.
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

## Captured outcome-audit runtime and bootstrap support

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

Every runtime manifest now has schema
`prospect.wm001.audit-runtime-manifest.v2` and exactly one closed execution
role. `conformance` binds `timeout_seconds: 600`; `outcome_audit` binds
`timeout_seconds: 10800`. The bootstrap derives the timeout solely from that
role and rejects an unknown role, a caller-selected timeout, or a mismatch.
Result-free prebinding, command-10, and launch-time rehearsals use
`conformance`. The two complete development audits, the sole formal audit, and
an ordinary-report adjudication replay use `outcome_audit`.

Every successful outcome audit is retained through
`prospect.wm001.captured-audit-execution.v2`. In addition to its exact stdout,
stderr, runtime manifest, invocation manifest, source, bootstrap, and support
identities, the receipt records the strict-positive monotonic subprocess
duration as `subprocess_elapsed_ns`. Development executes the complete
descriptor-mode outcome audit twice with byte-identical report, stderr,
runtime, and invocation bytes. `prospect.wm001.audit-reproduction.v3` binds both execution
receipts and embeds one `prospect.wm001.audit-capacity.v1` record.

Let `t1` and `t2` be those two elapsed durations, `D_total` the sum of every
byte row in the exact development producer manifest, and `D_result` its unique
`result.json` byte count. With integer arithmetic only, the capacity gate is:

```text
t = max(t1, t2)
aggregate_required_ns = ceil(2 * t * 8589934592 / D_total)
result_required_ns    = ceil(2 * t * 2147483648 / D_result)
combined_required_ns  = aggregate_required_ns + result_required_ns
combined_required_ns <= 10800 * 1000000000
```

The two ceilings are applied separately before addition. The aggregate term
projects the full 8 GiB producer ceiling; the result term independently
projects the 2 GiB result reread/materialization ceiling; the safety factor is
exactly `2/1`. The record also binds the producer-manifest SHA-256, calibration
sizes, both original elapsed values, selected maximum, each projected term,
their sum, and the available timeout. The central verifier, independent
auditor, formal launcher, and adjudicator recompute this record rather than
trusting its `passed` member. A failed capacity gate retires v1.20 before
binding. These elapsed values are engineering liveness calibration only and
cannot establish, interpret, or select a scientific endpoint.

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

## Canonical archive and authorization boundary

The development closure emits one byte-canonical USTAR archive whose physical
member sequence is exactly the closure's ordered member rows. The independent
reader opens it once, retains one stream iterator, matches every expected
member in order, and proves physical exhaustion with that same iterator. It
rejects an early terminator, any omitted, duplicated, reordered, or additional
member, a noncanonical header, unsafe path, link, special file, changed payload,
nonzero padding, truncation, trailing garbage, or stable-file custody change.
The 64 MiB per-retained-member limit remains, and total retained payload is
bounded at 256 MiB; large producer payloads are streamed rather than retained.

The archive is not authorized merely because its outer rows hash. The auditor
must parse `producer/producer-manifest.json`, require its complete canonical
schema, exact scalar types, ordered file rows, real UTC timestamps, and temporal
ordering, then require the physical `producer/*` member set to equal that
manifest exactly. The `evidence/*` namespace is an exact one-level set: result
qualification, independent audit, audit reproduction, producer runtime seal,
both `evidence/audit-execution-01.execution.json` and
`evidence/audit-execution-02.execution.json`, producer bootstrap, launch bootstrap,
and the closure-declared audit runtime, invocation, and stderr sidecars. The
writer validates these names before
combining namespaces, so an evidence key cannot overwrite or collide with a
producer member.

The retained result qualification must name the exact v1.20 development lane,
two fresh seeds, budgets, matrix contract, execution identity, and
claim-ineligible status. The retained audit, reproduction receipt, runtime
manifest, invocation manifest, stderr, runtime seal, and both bootstrap bytes
must form one digest-connected closure. The independent formal-input consumer
then reopens the live producer manifest and result plus every live audit,
reproduction, runtime, invocation, stderr, seal, and bootstrap input and joins
each byte string to its archived role. A self-consistent copied archive cannot
substitute for changed live authorization inputs.

All JSON comparisons at this boundary are scalar-type exact. Canonical integer
counts cannot be supplied as booleans or floats; booleans cannot be supplied as
integers; ordered rows cannot be supplied as sets; and an outer object cannot
vouch for itself by repeating its own filename or digest. The closure filename
is recomputed from its canonical content, its producer root must be the exact
v1.20 canonical root, and preflight inputs are reopened through stable,
single-link regular-file descriptors.

Before sealing, one real archive fixture must pass both the central verifier
and the independent formal-input archive reader without mocking either archive
seam, and the same physical payload mutation must fail at both readers. The
complete dedicated mutation set must additionally prove rejection of physical
extra, missing, reordered, duplicated, noncanonical, changed, truncated,
linked, or trailing members; incorrect producer membership or timestamps;
evidence namespace collisions; omitted or changed bootstraps; changed live
authorization roles; and JSON scalar aliases. A real producer, outer-finalized
audit, and closure fixture must traverse the production authorization
subhelpers and reject every archived/live role substitution. Separate complete
qualification and formal-input fixtures retain full semantic-consumer
coverage. A direct valid multi-member regression must fail if the old
second-iterator exhaustion pattern is restored.

## One-shot development pipeline

There is exactly one v1.20 development producer:

```text
bench/world_model_lifecycle/results/development/qualification-v1.20.0
```

Exclusive creation of that root consumes the v1.20 development qualification.
It may never be resumed, overwritten, renamed into place, or replaced by a
numbered sibling. Seed-override diagnostics cannot accept a caller-selected
output and are confined to generated children of
`results/development/diagnostics-v1.20.0`; they cannot occupy any lifecycle
namespace. The sole canonical development audit and closure paths are:

```text
bench/world_model_lifecycle/results/operator-v1.20/audits/development-audit-v1.20.0
bench/world_model_lifecycle/results/operator-v1.20/closures/development-closure-v1.20.0
bench/world_model_lifecycle/results/development/development-closure-v1.20.0.json
```

The audit must execute twice under the exact `outcome_audit` runtime and
reproduce byte-for-byte with the exact captured bootstrap support manifest.
Both captured-execution v2 receipts must retain strict-positive elapsed times;
the resulting reproduction v3 receipt must pass the independently recomputed
capacity formula above before closure. Qualification depends only on
engineering, custody, complete evidence, exact K7 restart parity, and
independent reconstruction. K3–K6
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
failure—retires v1.20. A terminal failure does not restore authorization merely
because a later artifact is absent. Repair requires a new protocol version and
fresh seeds.

## Preformal, binding, and formal pipeline

After an accepted, outer-finalized development audit and closure, the preformal
bundle must still be lexically absent. It is created exactly once at:

```text
bench/world_model_lifecycle/results/development/v1.20.0/preformal
```

The generator exclusively creates the deterministic hidden sibling
`.preformal.staging` and fsyncs its parent before the first command; either that
hidden claim or the final bundle consumes the one-shot attempt. Operator
attempt roots are created one component at a time, fsyncing each parent before
the corresponding deterministic hidden claim may be used. The report runs
exactly ten command roles: eight isolated QA roles and two sealed-runtime roles,
with separate stdout/stderr evidence. Every command must return zero and every
stderr stream must be the canonical empty stream; any stderr byte is a failed
gate. All ten subprocesses finish and their outcomes are held in memory while
the canonical bundle remains absent. The generator writes and fsyncs the
completed report and 20 logs only under the hidden claim, then atomically
publishes the whole directory with a no-replace rename, so no command or reader
can observe a partial canonical bundle. It returns nonzero with `passed: false`
and the exact
failed-command, identity-check, or semantic runtime-output diagnostic if the
completed report is not valid. An infrastructure interruption may leave only
the hidden claim; it still consumes v1.20 and cannot be deleted and retried.

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
bench/world_model_lifecycle/results/operator-v1.20/bindings/formal-binding-v1.20.0
```

It binds the eight fresh formal masters, the exact captured `outcome_audit`
runtime with its 10,800-second limit, both development elapsed receipts, and
the reproduction-v3 capacity record. Before the binding attempt can be
accepted, the exact independent
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
the formal artifact. The accepted attempt also retains the exact archived
`development-result-qualification.json`; the launcher joins its raw-result
digest to the streamed live result, and the formal auditor rejoins the live
attempt copy, formal-root copy, archived bytes, and binding digest.

After the final stop/go sequence and before creating the binding-keyed formal
root, the standard-library launcher consumes exactly one binding-keyed claim
and runs exactly once:

```text
preformal-runtime bootstrap-inventory-conformance --device cuda
```

against the accepted binding. Before full authorization or child dispatch, the
launcher hard-links `rehearsal-claim.json` into the deterministic
`results/rehearsals/v1.20/accepted-binding-<binding-sha256>.json` namespace.
It captures stdout, stderr, and the inherited outer receipt inside the fixed
five-file attempt directory, proves exit zero, canonical binding-equal stdout,
empty stderr and outer receipt, and the continued absence of every formal
root, marker, producer, audit, review, and adjudication path. It then publishes
an accepted or failed `rehearsal-terminal.json` as a same-inode deterministic
outer completion. An existing claim or terminal forbids another child;
recovery performs no dispatch. Before a canonical terminal exists it may
publish only failed evidence; if an intact immutable accepted or failed
terminal already exists, recovery may only hard-link those exact bytes as its
outer completion. It never rewrites or downgrades a terminal, and malformed or
contradictory states fail closed. Only the independently reopened accepted
package may authorize formal dispatch.

The producer-root `prospect.wm001.formal-launch.v3` record binds the accepted
package's exact claim, claim-marker, terminal, and outer-completion
path/byte/SHA-256 rows. The verifier, independent auditor, and adjudicator each
reopen the live package and require strict equality with those rows. All
source, package, support, rehearsal, and launch-time conformance checks finish
before formal marker publication. The producer keeps the accepted rehearsal
descriptors open across launch-record creation and rechecks them immediately
before and after the global formal-marker hard link; a post-link discrepancy
consumes and retires the formal attempt before any outcome reset.
Across the gap between rehearsal and formal invocation, the outer launcher
rechecks the complete current-version formal namespace immediately before the
formal subprocess. It allows only the canonical, non-aliased, empty
binding-keyed parent prepared by the operator; an alternate confirmation
directory, current marker or claim, audit/adjudication staging path, semantic
review, symlink, or nonempty parent blocks dispatch.

The repository-wide formal marker is:

```text
bench/world_model_lifecycle/results/formal/formal-launch-v1.20.0.json
```

The first canonical formal producer root or formal marker consumes the sole
formal attempt according to the runbook. A completed, failed, or interrupted
post-marker formal run cannot be retried. The canonical formal-audit claim and
adjudication claim are also single-use. A report-generation failure consumes
the formal-audit claim and may only lead to rejected evidence. Adjudication
replays an ordinary report exactly once; authenticated no-report failure
evidence receives zero replay and cannot be accepted.

The sole formal audit and any ordinary-report adjudication replay must use the
binding's exact audit-runtime-manifest v2 `outcome_audit` identity and its
10,800-second deadline. Their captured-execution v2 receipts must contain
strict-positive elapsed times. A conformance-role manifest, a 600-second
outcome manifest, a changed timeout, or a missing elapsed value is a terminal
contract failure; it cannot be repaired or replayed in the same version.

Acceptance requires an outer-finalized formal producer, independently
recomputed K0–K7, a clean byte-reproduced audit, an independent semantic review
accepting every gate, an atomically published accepted adjudication package,
and successful strict post-publication verification. Only that package may
support the bounded WM-001 learning, improvement, retention, and restart
statement.
