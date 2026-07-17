# LCV-001 sealed-lineage/runtime-closure protocol

**Date:** 2026-07-16  
**Status:** implementation validated privately; canonical prepare/run not authorized  
**Role:** host-bound, non-scientific research-infrastructure gate

## Claim boundary

LCV-001 asks whether a successor assay can consume the exact sealed MM-007 frames,
causal identities, and source-current fold normalizers through a frozen consumer path.
It does not rerun or repair MM-001--MM-009, validate a predictor, improve a benchmark,
or change any prior classification. `PASS` only authorizes preparation of a newly
named assay after that assay revalidates this host receipt.

This is a trusted-bootstrap receipt, not a self-contained runtime. It pins the lexical
Python/stdlib/NumPy/`numpy.libs` aggregates and exact numerical canary on this host.
The live kernel, CPU/microcode, dynamic loader, glibc, libc, libm, libz, libgcc,
libstdc++, systemd user manager, and filesystem remain trusted host dependencies.

## Frozen inputs and independent release condition

The sole scientific parent is the exact 14-file MM-007 tree. All file SHA-256, byte,
and source-mode pins, the seven non-root directories, and their live `0775` modes are
enumerated in `2026-07-16-lcv001-config.json`. The primary anchors are:

- MM-007 artifact manifest: `db0b6654ab098dc9a3ec93e4a6de8820bbe5860d44974645e9a5ee7dad1537fb`
- MM-007 input manifest: `1f83c805e6c5d75f4f1d5a2102d471c15bbc6bb787960cb5ae630bd2260faa1f`
- MM-007 frame package: `fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f`

Canonical preparation additionally requires an external `GO` audit bound to the
config, this protocol, the frozen LCV source/tests, and the parent anchors. The audit
must independently affirm canonical-output absence, the claim boundary, parent pins,
runtime commitments, and source/tests. Development helpers may use separately named
temporary roots; public `prepare` and `run` accept canonical paths only.

## Descriptor custody and preparation transaction

Preparation parses no parent JSON or NPZ and imports no historical experiment. It
retains every ancestor and in-tree directory descriptor, uses `O_NOFOLLOW`, rejects
hard links/symlinks/special files, performs exact pre/post membership and identity
checks, and copies only authenticated in-memory snapshot bytes. The live parent and
copied package both have exact directory-mode contracts.

Preparation occurs in a private sibling staging directory. On any pre-publication
failure, only that agent-created tree is descriptor-removed and the destination
remains absent. Its 128-bit random name is selected absent, created with exclusive
`mkdir`, and recorded by the caller before any namespace mutation; allocation,
chmod, and parent `fsync` are inside the owned-cleanup boundary. Successful
publication uses Linux `renameat2(RENAME_NOREPLACE)`
through a retained parent descriptor. The namespace rename is the
commit boundary. The staged directory's device/inode identity is re-read if an
exception occurs at the syscall-return boundary, so an already published tree is not
reported absent or cleaned as pre-commit state. A successful parent-directory
`fsync` is reported explicitly; an
error after the rename returns
`prepared_namespace_committed_durability_unconfirmed`, never an ordinary failure
that implies the visible destination is absent. Formal execution retries and requires
that parent-directory `fsync` before it may create the marker. There is no
replace-on-race path.

The published package has 33 prepared files. Frozen inputs, config, audit, freeze
record, copied source/tests, and MM-007 bytes live below `prepared/`. A prepared-phase
anchor pre-creates `outcomes/`. Before publication:

- every file is `0444`;
- the output root and all `prepared/` directories are `0555`;
- only `outcomes/` remains `0755` for application-level `O_EXCL` creation of formal
  products.

The mode layout prevents accidental writes and blocks ordinary path replacement
without first changing permissions. It is not kernel immutability or an append-only
filesystem guarantee: the owning UID and live filesystem are trusted not to race the
gate by changing modes. Every custody pass still rejects observed replacement or
drift.

## Lexical host runtime and supervisor

Formal and semantic children use the exact prefix:

`/home/alex/Documents/prospect/.venv/bin/python -I -S -B`

The terminal interpreter is `/home/alex/miniconda3/bin/python3.12`, while lexical
`sys.executable` remains the venv path. The copied shadow source is the only project
source on `sys.path`; historical modules are forbidden. Python 3.12.9, NumPy 2.4.6,
the venv/terminal symlink chain, Python and NumPy object hashes, loaded OpenBLAS object,
all prefixes, exact environment, and the host platform receipt are checked.

The runtime receipt persists all 893 NumPy/`numpy.libs` and 916 stdlib per-file
`{bytes, sha256}` records and validates their count, byte total, and canonical-map
commitments. These are aggregate host commitments, not copies of the underlying
runtime libraries.

Formal and semantic work runs in a transient user-systemd cgroup-v2 service with
`KillMode=control-group`, `SIGKILL`, a 180-second ceiling, bounded 8 MiB aggregate
accepted output, and total-wall cleanup reserve. Output is sampled in temporary
files, so a fast writer may transiently overshoot before termination; no oversized
capture is accepted into a receipt. The service explicitly empties `LD_AUDIT`,
`LD_LIBRARY_PATH`, and `LD_PRELOAD`; the stdlib guard rechecks them before `execv`.
The user bus must be a current-UID-owned socket. Timeout, interruption, noisy-child,
and failure paths kill/reap the local process group, kill/stop/reset the unit, require
a non-active `failed`, `inactive`, or `unknown` state plus absent cgroup paths, and
fail closed on any cgroup-tree census error.

The exact 512x256 near-degenerate SVD canary pins input, U, S, Vh, reconstruction,
minimum singular gap, and combined bundle hashes with OpenBLAS thread count one.
Actual child controls also pin direct/cgroup parity plus venv-16-thread and base
NumPy thread-sensitivity endpoints.

## Consumer semantics

LCV-owned code checks only the successor-facing surface:

1. exact top-level schemas and every successor-consumed frame, classification, and
   alignment duplicate anchor in the MM-007 manifest, marker, evidence, result,
   summary, decision, frame package, and frame schema;
2. exact three NPZ members, order, ZIP metadata, NPY headers, dtypes, shapes, layouts,
   array hashes, 477 raw identities, and timestamps;
3. exactly 453 same-video half-second previous/current/future triples and per-video
   counts, with identity digest `d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b`;
4. full current R8 digest
   `587d28455a0bd0226f24c94a60ce6bd6ea9bee6bf05ec2a315089e6e10ffd787`;
5. per-fold selection of exact training-current native frame indices before pooling,
   frozen uint8-to-float32 and float64 block mean, float32 cast, contiguous NCHW
   strides `(768,256,32,4)`, and float64 mean/std;
6. four exact normalizer fingerprints, train counts `332/335/346/346`, strict
   `uses_target is False`, and identical rows shared by resolutions 8/16/32/64.

The concrete wrong implementation pools into non-contiguous NCHW strides
`(768,4,96,12)`. Its four preregistered fingerprints and differences of
`297/171/89/201` maximum ULP are recomputed, required to differ, injected into the
normalizer verifier, and required to classify `invalid_LCV001_artifact`.

No PCA, encoder/decoder, prediction model, deformation/flow fitter, scorer, media
decoder, scientific generator, or historical verifier is executed.

## Two-phase formal lifecycle

The formal child cannot write `PASS`. After the irreversible marker it writes only
runtime/closure/control receipts and `provisional-result.json` with classification
`PENDING_CGROUP_CLEANUP`. The supervisor then destroys and independently verifies
absence of that child cgroup and returns a structured cleanup receipt.

Only the parent may finalize. It descriptor-validates the exact 38-file canonical
provisional tree, requires the cleanup unit to equal the runtime-receipt unit, and
revalidates the live orchestrator source against the audit-frozen source. It then
copies only authenticated provisional bytes into a private same-filesystem sibling.
Cleanup receipt, result, report, and self-excluding artifact manifest are created
only in that sibling. The sibling is exact-sealed and must pass `_verify_in_process`
before promotion. Its 128-bit random absent name is created by exclusive `mkdir`
inside the same ownership/cleanup boundary; cleanup is restricted to that
invocation-owned path and never targets a pre-existing name.

Immediately before commit, the parent reacquires the unchanged canonical provisional
tree and live-source binding. Linux `renameat2(RENAME_EXCHANGE)` then atomically swaps
the already verified sibling into the canonical name through retained parent
descriptors. Pre-exchange device/inode identities are re-read if an exception occurs
at the syscall-return boundary, so an exchange that already committed cannot be
mistaken for a pre-commit failure. The completed package contains
exactly 42 files; every file is `0444` and all 17 directories including the root are
`0555`. The exchanged-out provisional tree is removed separately. Canonical `PASS`
acceptance requires the returned commit receipt to report atomic exchange, successful
parent-directory `fsync`, retired-tree removal, and no warnings. A post-commit fsync
or retired-tree-cleanup exception is reported as a commit warning and never rewrites
or terminalizes the already verified canonical package.
If an asynchronous failure lands after exchange but before the exact commit receipt
is returned, the parent first reconciles canonical membership. A verified completed
canonical or any indeterminate namespace is never terminalized; the call instead
fails as runtime receipt-loss/manual-inspection required and cannot authorize `PASS`.
Only a canonical whose membership is still provably a subset of the provisional
phase may receive a terminal-failure receipt.

Every catchable pre-commit failure that leaves a read-only-reconciled, structurally
exact provisional namespace removes the owned private candidate and terminal-seals
the original canonical tree. An unexpected or unstable namespace is preserved for
manual inspection and cannot authorize `PASS`. Terminal membership must be a subset of the 38 provisional
files plus `terminal-failure.json`; cleanup receipt, result, report, and artifact
manifest are forbidden there. Typed parent-byte/schema failures remain
`invalid_LCV001_artifact`; supervisor, timeout, interruption, host-I/O, and pre-marker
failures are `invalid_LCV001_runtime`; unexpected post-marker child or parent logic
failures are `invalid_LCV001_verifier`. A terminal canonical tree cannot be retried.
Any supervisor cleanup value is strictly validated before its bounded digest/summary
enters the terminal receipt; malformed, cyclic, or unserializable values become an
explicit invalid-cleanup summary and cannot escape terminal custody.
The child serializes and explicitly flushes its final JSON inside that classification
boundary. If structured diagnostics cannot survive an output-channel failure, exit
code `70` is the dedicated runtime-transport fallback; Python shutdown-flush code
`120` and interrupt endpoints `-2`/`130` are also runtime, while a contradictory
structured classification anywhere in stderr is a verifier failure. Ordinary typed
failure uses exit `2`, empty stdout, and exactly one canonical diagnostic object;
other nonzero codes cannot claim typed artifact/runtime ownership. Exit `0` requires
empty stderr and exactly one canonical, explicitly flushed JSON object on stdout.
The entire parent-side interpretation after the supervisor returns is inside terminal
custody. The four nested sensitivity-canary children use the same write/flush and
transport-code rules: spawn, timeout, interruption, and channel failures are runtime;
malformed output, endpoint mismatch, and unexpected child logic are verifier-owned.
Before interpreting any channel, the parent requires the exact supervised-result
type, launched argv, integer (not boolean/float) return code, string channels, and a
role-valid cleanup receipt. Exception text and cleanup attributes are reduced through
guarded bounded helpers, so malformed return objects or hostile exception formatting
cannot double-fault around reconciliation.

This transaction covers catchable process failures. Uncatchable `SIGKILL`, power
loss, arbitrary filesystem failure, or a hostile owning UID can leave an uncommitted
private candidate or an indeterminate durability boundary; those states require
manual custody inspection and do not authorize `PASS`.

The 25 mutation controls cover digest/schema/cross-link/duplicate-anchor mutations,
held-out/future/previous/reordered indices, frame bytes, normalizers/boolean type,
full-R8 endpoint, concrete wrong layout, runtime receipt/thread maps, and four actual
runtime child endpoints. Formal classifications are exactly `PASS`,
`invalid_LCV001_artifact`, `invalid_LCV001_runtime`, and
`invalid_LCV001_verifier`.

`verify` is a read-only exact-package/cross-link check. `verify-semantic` is a
read-only supervised lexical-runtime replay from copied inputs. The canonical output
is `bench/sealed_lineage_verifier/results/LCV-001`; it remains absent until a separate
independent audit authorizes the one-shot canonical prepare/run.
