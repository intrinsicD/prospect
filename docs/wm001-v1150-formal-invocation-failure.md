# WM-001 v1.15 formal-invocation preclaim failure

Date: 2026-07-21

Disposition: **retired — formal invocation failed before producer custody; no
formal authority**

## Terminal outcome

The exact WM-001 protocol 1.15.0 harness was sealed at Git commit
`30df10bb3ed14741ee01a06a7d47d3ba005d8db7`. Its protocol and schemas,
independent prospective review, 753-test suite, lint, two type-check scopes,
installed source snapshots, exact 45-row runtime dependency closure, separate
QA/runtime environments, warning-as-error precision probe, CUDA gate,
seed-domain audit, archive/auditor regressions, and formal-binding schema gate
passed.

The prospective runtime seal was created and outer-finalized. The sole
result-free command-10 rehearsal returned zero, emitted a canonical passing
receipt, and wrote exactly zero stderr bytes.

The sole development producer completed both fresh v1.15 masters and was
outer-finalized. Its independent development audit was accepted, reproduced
byte-for-byte, and outer-finalized. The sole closure transaction published the
canonical closure, retained its fresh sealed-runtime reopen, and outer-finalized
the accepted attempt. The required post-finalization sealed-runtime reopen
passed.

The one-shot preformal generator completed all ten fixed commands and published
the complete 20-log report. Both the QA-side report verifier and the
runtime-side canonical verifier accepted it. All ten command exit statuses were
zero and all ten stderr streams were the required zero-byte streams.

The sole binding transaction passed the independent formal-input consumer,
published a terminal-bound preflight receipt, and outer-finalized an accepted
binding attempt. The operator-recorded final stop/go sequence then exited zero
after protocol, development-result, preformal-report, binding, exact
preflight-replay, sealed-closure, source-custody, outer-completion,
clean-worktree, and formal-path-absence checks.

The operator exclusively created and fsynced the binding-keyed formal root:

```text
bench/world_model_lifecycle/results/formal/5282325024dd19a32607d9ad3b70067ccd06f52bbf44d1a7bd76f856e5df99de
```

The sole formal invocation then returned status `1` before it published the
formal marker or created `confirmation-v1.15.0`. Its separately captured stdout
was empty. Its diagnostic stderr contained 2,023 bytes.

The precommitted runbook makes creation of the binding-keyed root the final
operator preparation and declares any later invocation refusal or failure
terminal. The empty root must not be removed, reused, resumed, or replaced.
Protocol 1.15.0 therefore cannot be retried or repaired. Because no formal
producer exists, a formal audit, semantic review, or adjudication would have no
eligible input and must not be created.

## Evidence identity

- sealed Git commit:
  `30df10bb3ed14741ee01a06a7d47d3ba005d8db7`
- sealed Git tree:
  `bcb1149743d67250642793d1bcc7670f918dfa5d`
- protocol SHA-256:
  `8db5560044bbedfb491be12a26bd8b39c43fd6d6a314ce86d6afdc71f50486bb`
- prospective-review SHA-256:
  `1780298c53fe3d0aec1514f66c97596ae967fc873af6af42f9ce29ce472a673e`
- prospective implementation-manifest SHA-256:
  `00cd1b3edf220ec86745510d16cf7724af98e4d29c8e578ced5fab0e385f4bc9`
- runtime dependency-lock SHA-256:
  `b0e565e5769dd249bfef88442b881fe31f298eb073cd7fea72ae79cce312d314`
- runtime-seal SHA-256:
  `808b863e71be3becb9d0984983090b8881e7ca7a4aa58df9221aafb106af2a3d`
- initial temporary result-free command-10 rehearsal stdout: 10,096 bytes with
  SHA-256
  `f0d9f31aa556f106255694d25a4ddc4a6abb07b191a1a36b4863ed197e918061`
- initial temporary result-free command-10 rehearsal stderr: zero bytes with
  SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- terminal-bound preformal command-10 stdout: 10,096 bytes with SHA-256
  `d23ed41fa7c0241486589d8db0db4211e3cc2b0eae3772702e4b03bcf72f5779`
- terminal-bound preformal command-10 stderr: zero bytes with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- producer-manifest: 12,661 bytes with SHA-256
  `32610fc659218ee15bd39b0f6d2348debb4f66fedf6fc3744402bb7f3f2242db`
- raw development result: 320,977,868 bytes with SHA-256
  `e7d67758d770022516ab52692676e75941f5e71ac11918aec53bc5c1d3842dce`,
  authenticated and hashed as opaque bytes without decoding or semantic
  inspection
- accepted development-audit terminal: 3,271 bytes with SHA-256
  `c3cdf1fdd94484355e709114cd5b4783e03b68667030ce274b878fabfd2317cb`
- canonical development closure: 20,039 bytes with SHA-256
  `963741f25a857a626c9194848f34d5451418fe2094712347ee581c81cb61c674`
- accepted closure-attempt terminal: 5,368 bytes with SHA-256
  `8aba9a26dbd1abd8771552322a06ba098897d7368b030e91d342ff76eeae09fb`
- accepted preformal report: 59,457 bytes with SHA-256
  `6facbcf6d1a0e0551231c7f3dda6b731e88683ab72b501ca94ba8d3808b2e454`
- accepted formal binding: 36,015 bytes with SHA-256
  `5282325024dd19a32607d9ad3b70067ccd06f52bbf44d1a7bd76f856e5df99de`
- accepted formal-input preflight receipt: 705 bytes with SHA-256
  `8c6ac37aeffaa555c45c395f2221acc3aceba9bcc421b5eb231167713cbe3cf9`
- accepted, outer-finalized binding-attempt terminal: 14,890 bytes with
  SHA-256
  `a7b5249f5fb408113bd45fc2dbc2718ab9e6c7a5ec7886d48edc1c753eb80960`
  (the terminal and deterministic outer marker are inode `69234299` with
  link count two)
- operator-observed prepared binding-keyed formal root: inode `71602485`,
  ordinary directory link count two, zero children
- operator-observed formal invocation diagnostic stdout: zero bytes with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- operator-observed formal invocation diagnostic stderr: 2,023 bytes with SHA-256
  `87ff72e920546235122bd7311bff06f12d5a54b861c53cafd279c272bf58be06`
- canonical formal confirmation directory: absent
- repository-wide v1.15 formal launch marker: absent
- formal producer manifest and outer completion: absent
- formal audit attempt and marker: absent
- semantic review: absent
- adjudication marker and package: absent

The return code, empty-root metadata, and stdout/stderr captures are temporary
operator observations, not an outer-finalized formal terminal. Their identities
support diagnosis but do not substitute for producer custody. No result value,
performance summary, K value, prediction tensor, or trace was opened, printed,
compared, or used to choose a repair.

## Exact engineering cause

The standard-library outer launcher defines one generic limit:

```python
_MAX_CONTROL_BYTES = 64 << 20
```

`_verify_development_producer()` recursively visits every live producer file
except the manifest, reconstructs its row, and only then compares the exact
sorted live namespace with the producer-manifest rows. It passes each file
through `_regular_row()`, which passes it through `_open_regular()` and
`_read_descriptor()`.
`_read_descriptor()` rejects any regular file whose `st_size` exceeds
`_MAX_CONTROL_BYTES` before reading or hashing it.

The authenticated development `result.json` is 320,977,868 bytes, more than
four times the 67,108,864-byte limit. It is the first sorted producer data file
to reach the guard, but it is not the only incompatible file. Two per-seed JSON
files are 160,421,650 and 160,537,118 bytes, and two checkpoints are 78,204,724
and 78,205,824 bytes. A result-only exception would therefore leave the same
failure class in place. The formal diagnostic traceback ends at the first
guard with:

```text
canonical development producer file result.json exceeds its byte limit
```

This occurs inside formal binding custody verification, before the producer
bootstrap is dispatched, before `ProducerAttempt` can claim the confirmation
directory, and before formal-marker publication. The empty formal root and
absence of every producer/marker path agree with that ordering.

The diagnosis is independently supported by the sealed source branch, the
authenticated file size, and the diagnostic capture. It is not an
authenticated formal-producer failure: no producer terminal exists from which
a precommitted verifier could reconstruct a failure record.

## Why the prospective gates missed it

The closure and independent archive-custody paths deliberately stream large
qualification evidence. The standard-library launcher's separate live
development-producer reconstruction did not: it reused its bounded control-file
reader for bulk producer data. Other verification paths, including the explicit
result verifier, may still materialize the raw JSON and do not erase this
launcher-specific mismatch.

The launch-bootstrap unit fixture represents `result.json` with a tiny
four-field object. The real outer-launcher subprocess fixture proves bootstrap
dispatch and receipt custody against a minimal synthetic package, not the
accepted binding plus a production-sized development result. Command 10 uses
small synthetic development/formal restart artifacts. Consequently, the exact
formal-launch authorization branch was extensively mutation-tested but was
never passed a valid result larger than 64 MiB before the one-shot launch.

## Results-audit disposition

| Claim class | Evidence checked | Disposition |
| --- | --- | --- |
| Prospective harness, development, audit, closure, preformal, and accepted-binding custody | Sealed identities, precommitted verifiers, same-inode outer completions, exact preflight replay | **confirm, engineering scope only** |
| Development audit supports a scientific claim | Safe audit fields are `passed=true`, `integrity_passed=true`, and `engineering_complete=true`, but `complete_for_claim=false` and `audit_execution_conformance_verified=false` | **retire; development is qualification evidence only** |
| Formal invocation was consumed before producer custody | Empty fsynced binding root, return status, path absence, runbook one-shot rule | **narrow; confirmed operationally, but no authenticated producer terminal exists** |
| Generic 64 MiB reader rejected the 320,977,868-byte development result | Sealed source branch, authenticated size, diagnostic traceback | **confirm as engineering diagnosis** |
| The accepted binding was launchable against production-scale evidence | The separate standard-library outer consumer rejected the valid large producer namespace | **retire** |
| A v1.15 formal experiment completed | No marker, confirmation directory, manifest, result, or outer completion | **retire** |
| Prospect learned, improved, retained an improvement, or met a formal threshold in v1.15 | No authorized formal outcome; development values are permanently claim-ineligible | **retire** |
| Prospect has or lacks the general collect → learn → improve → retain capability | WM-001 v1.15 produced no authorized formal outcome | **unresolved** |

No scientific metric was recomputed because no formal result exists and
development outcomes cannot support the confirmatory claim.

## Required fresh-version repair

A successor protocol must:

1. preserve the strict 64 MiB bound for actual control files rather than
   raising one global limit to accommodate bulk evidence;
2. add a separate descriptor-based streaming row reader for producer data,
   with canonical-path and `O_NOFOLLOW` checks, regular-file and link-count
   checks, pre/post `fstat` identity, a post-read path-to-descriptor same-inode
   check, exact byte count, incremental SHA-256, and mutation/short-read
   rejection;
3. use that streaming path for every live producer file, including JSON and
   checkpoints; require exact live-to-manifest namespace equality and
   typed, sorted, unique rows; hash each file once and reuse that captured row;
   enforce exactly 4 GiB per live producer file and 16 GiB across the live
   producer namespace, matching the independent producer auditor, while
   retaining the 64 MiB bound for canonical control-object parsing;
4. copy the exact archived
   `prospect.wm001.development-result-qualification.v1` bytes into the accepted
   binding attempt as a terminal-bound sidecar; require its digest to equal
   `development_qualification.result_qualification_sha256`, then have the
   launcher and formal auditor rejoin its `raw_result_sha256` to the streamed
   raw-result digest; do not add a result-only bypass, a second unjoined
   metadata sidecar, or a full-result materialization path;
5. add valid 64 MiB boundary, 64 MiB-plus, production-sized sparse, short-read,
   growth, shrink, inode-swap, symlink, hard-link, digest-mismatch, and malformed
   metadata regressions through the real `_verify_development_producer()` path;
6. add a real accepted-binding-to-standard-library-custody integration test
   with all five large-file roles and, before creating the formal root, run the
   existing result-free outer path with the accepted binding as runtime seal:
   `preformal-runtime bootstrap-inventory-conformance --device cuda`;
7. require that exact preclaim rehearsal to return canonical success with
   empty stderr and prove it creates no formal marker, confirmation directory,
   outcome-producing experiment execution, experience, model update, semantic
   outcome decoding, or metric exposure;
8. keep the model, learning algorithm, optimizer, planner, controller, budgets,
   controls, metrics, thresholds, exclusions, scientific blocks, and raw-result
   semantics unchanged;
9. use a fresh version, paths, derivation domain, seeds, environments, wheel,
   dependency lock, review, seal, development evidence, binding, and formal
   authority; and
10. preserve every v1.15 producer, audit, closure, preformal, binding,
    preflight, outer-completion, and empty formal-root byte unchanged; record
    the temporary diagnostics by size and hash, preserve them unchanged if
    retained, and never treat them as authoritative formal evidence.

The next admissible move is a fresh-version engineering repair and prospective
large-result formal-authorization rehearsal. It is not interpretation of the
v1.15 development outcomes.
