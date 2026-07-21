# WM-001 protocol 1.15.0 confirmation plan

Status: prospective. This document must be sealed with the v1.15 protocol,
schemas, implementation, dependency lock, tests, and independent harness review
before any v1.15 development or formal outcome is produced.

Protocols 1.10.0 through 1.14.0 are immutable and terminally retired. Versions
1.10 through 1.13 exposed and repaired, in order, a QA/runtime composition
error, deprecated precision access on command-10 stderr, a schema contradiction
for required empty stderr streams, and confusion between canonical and
preserved preformal-report verifier roles.

Version 1.14 passed its prospective gates, result-free command-10 rehearsal,
full-budget two-seed development producer, independently reproduced audit,
closure, fresh sealed-runtime closure reopen, and all ten preformal commands.
Its sole binding transaction then reached the independent formal-input archive
auditor. That auditor rejected the canonical development qualification archive
as containing an extra member even though it had already matched and verified
all 86 declared members and the archive had zero missing or extra physical
members.

The defect was the auditor's exhaustion check, not the archive. The member loop
created one iterator implicitly through `zip(members, stream)`, but the final
check called `next(iter(stream))`. A new `TarFile` iterator replayed the first
cached, already verified member and the auditor misclassified that replay as
new physical evidence. The failed binding attempt was outer-finalized. No
v1.14 formal-input preflight, accepted binding, formal launch, formal producer,
formal audit, semantic review, or adjudication exists or may be created.
Precommitted automation processed the claim-ineligible development artifact,
but no v1.14 K or performance value was manually opened, printed, summarized,
compared, thresholded, or used to choose a v1.15 setting. The exact disposition
is preserved in the
[v1.14 independent archive-verifier failure](wm001-v1140-development-archive-membership-failure.md).

All v1.10–v1.14 evidence remains claim-ineligible and cannot authorize, block,
resume, upgrade, repair, or be reused by v1.15. The v1.15 protocol must directly
supersede the exact sealed v1.14 protocol bytes with SHA-256
`39f5820a91c8a504355f971449726ae0a9067cc856111a575bb038455d1fd635`.
Skipping that immediate lineage or pointing at an earlier protocol is fatal.

## Purpose and scientific freeze

WM-001 v1.15.0 permits one fresh-seed confirmation of the unchanged
collect → learn → improve → retain → restart experiment. It retains the
v1.10 canonical matrix contract and its digest
`09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84`.
It changes only the archive, authorization, and prospective-coverage boundary
needed to repair that defect and the adjacent ambiguities found before another
one-shot attempt:

1. the independent USTAR pass retains one explicit iterator from the first
   expected member through the final exhaustion check, so cached-member replay
   cannot masquerade as an extra physical member;
2. the auditor parses the archived canonical producer manifest, requires its
   exact ordered rows and scalar types, and requires the physical `producer/*`
   namespace to equal that manifest rather than trusting only an outer closure
   projection;
3. the development qualification binds the archived launch and producer
   bootstrap bytes and joins every live producer, result, audit, reproduction,
   runtime, invocation, stderr, and bootstrap input to the corresponding
   archived role digest before formal authorization;
4. archive, closure, runtime, invocation, qualification, and authorization
   comparisons use strict JSON scalar identity, so `true`, `1`, and `1.0` are
   never aliases; producer timestamps are real canonical UTC instants in
   monotonic order rather than merely matching a string shape;
5. the archive writer validates evidence names before insertion, rejects any
   collision with the `producer/*` namespace, requires the exact one-level
   `evidence/*` set, preserves the per-member limit, and adds a bounded aggregate
   retained-payload limit;
6. stable descriptor reads, single-link custody, canonical content-addressed
   closure naming, exact producer roots, both captured bootstraps, and the full
   runtime/invocation semantics are checked on the independent boundary;
7. real unmocked multi-member USTAR acceptance, real physical-extra-member
   rejection, writer-collision, timestamp, strict-scalar, bootstrap, and
   live-input regressions exercise their production seams; a real
   producer/audit-to-closure authorization fixture rejects every archived/live
   role substitution, while separate complete qualification and formal-input
   tests exercise those semantic consumers; central and independent archive
   verifiers share the same generated valid archive and mutation expectations;
8. all ten preformal commands retain separate stdout/stderr custody and require
   exact empty stderr streams; formal-binding v10, raw-result v9, prebinding
   request/report v2, preformal report v2, precision custody, and the existing
   stream/source digest split remain unchanged;
9. the central verifier and independent auditor add v1.14 to the prior
   master/stream universe and reproduce the exact expanded collision counts;
   and
10. every active namespace, master seed, environment, wheel, lock, seal,
    prospective review, development path, binding path, and formal authority is
    fresh for v1.15.

The result-free sealed rehearsal still executes the same nested
descriptor/bootstrap route with a fresh challenge and reproduces the
matrix-contract golden without recursively acquiring the outer-launch lock.
Before the canonical closure marker is published, a newly exec'd interpreter
must reopen the prospective marker through inherited bootstrap and
runtime-seal descriptors. Before preformal authorization, another sealed
runtime must reopen the accepted, outer-finalized closure attempt and its
retained fresh-reopen receipt.

The v1.15 repair may change only independent archive iteration and semantics,
archive construction and limits, closure-to-archive-to-live-input linkage,
strict scalar and timestamp verification, the associated real regressions,
versioned paths, fresh seeds, and evidence plumbing needed to enforce those
roles. It does not change a serialized binding or result shape: formal-binding
stays v10 and raw-result stays v9, although their sealed schema bytes bind the
new protocol version and formal seeds. It may not change the world model,
learning algorithm, optimizer, planner, controller, task definitions,
experience or update budgets, controls, metrics, thresholds, exclusions,
killing order, or scientific claim.

The following continuity requirements are fatal gates:

- `experiment.revision.supersedes` is exactly `1.14.0` and its pinned
  superseded-protocol SHA-256 is exactly
  `39f5820a91c8a504355f971449726ae0a9067cc856111a575bb038455d1fd635`;
- the canonical SHA-256 of the 17 scientific protocol blocks remains
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- `model.py`, `learning.py`, `planning.py`, and `runtime_lane.py` retain their
  sealed v1.4 source SHA-256 values; and
- every v1.15 protocol, verifier, auditor, prospective review, and result schema
  agrees on the new version, paths, seeds, and exact support-file set.

The v1.10 through v1.14 failures are engineering lineage only. They supply no
evidence for or against the scientific claim.

## Assurance boundary

Every v1.15 seal, binding, operator record, audit receipt, and adjudication
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
SHA256("WM-001|1.15.0|<lane>-master|<i>")
```

Development indices 0–1 are:

```text
2388891654, 3201418215
```

Formal indices 0–7 are:

```text
2465968807, 3494485289, 1615601571, 2220840580,
280448223, 597199725, 712207456, 1727907751
```

The verifier and independent auditor regenerate all 136 namespace streams for
each current master. They require:

- 10 unique current masters and 1,360 unique current derived streams;
- no current master/master, stream/stream, or master/stream collision;
- 140 unique exposed prior masters and 19,040 unique exposed prior streams from
  v1.0.0, v1.2.0, v1.3.0, v1.4.0, v1.5.0, v1.6.0, v1.7.0, v1.8.0,
  v1.9.0, v1.10.0, v1.11.0, v1.12.0, v1.13.0, and v1.14.0; and
- zero overlap in all current/prior master/stream cross classes.

No seed is replaced after any v1.15 outcome-producing path is created.

## Exact runtime and environment

The QA and producer environments are fresh, separate, non-editable virtual
environments at
`/home/alex/.venvs/prospect-wm001-v115-reviewed` and
`/home/alex/.venvs/prospect-wm001-v115-reviewed-runtime`. Both install the one
reviewed wheel built in a fresh `prospect-wm001-v115-wheelhouse.*` directory.
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
producer bootstrap, launch bootstrap, and the closure-declared audit runtime,
invocation, and stderr sidecars. The writer validates these names before
combining namespaces, so an evidence key cannot overwrite or collide with a
producer member.

The retained result qualification must name the exact v1.15 development lane,
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
v1.15 canonical root, and preflight inputs are reopened through stable,
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

There is exactly one v1.15 development producer:

```text
bench/world_model_lifecycle/results/development/qualification-v1.15.0
```

Exclusive creation of that root consumes the v1.15 development qualification.
It may never be resumed, overwritten, renamed into place, or replaced by a
numbered sibling. Seed-override diagnostics cannot accept a caller-selected
output and are confined to generated children of
`results/development/diagnostics-v1.15.0`; they cannot occupy any lifecycle
namespace. The sole canonical development audit and closure paths are:

```text
bench/world_model_lifecycle/results/operator-v1.15/audits/development-audit-v1.15.0
bench/world_model_lifecycle/results/operator-v1.15/closures/development-closure-v1.15.0
bench/world_model_lifecycle/results/development/development-closure-v1.15.0.json
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
failure—retires v1.15. A terminal failure does not restore authorization merely
because a later artifact is absent. Repair requires a new protocol version and
fresh seeds.

## Preformal, binding, and formal pipeline

After an accepted, outer-finalized development audit and closure, the preformal
bundle must still be lexically absent. It is created exactly once at:

```text
bench/world_model_lifecycle/results/development/v1.15.0/preformal
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
the hidden claim; it still consumes v1.15 and cannot be deleted and retried.

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
bench/world_model_lifecycle/results/operator-v1.15/bindings/formal-binding-v1.15.0
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
bench/world_model_lifecycle/results/formal/formal-launch-v1.15.0.json
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
