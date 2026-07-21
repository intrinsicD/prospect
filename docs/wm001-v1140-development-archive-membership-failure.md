# WM-001 v1.14 independent archive-verifier failure

Date: 2026-07-21

Disposition: **retired — failed binding, no formal authority**

## Terminal outcome

The exact WM-001 protocol 1.14.0 harness was sealed at Git commit
`1f7a3acc299fed2dcb4c754167f3cedc2794058f`. Its protocol, independent
prospective review, 698-test suite, lint, two type-check scopes, installed
source snapshots, exact 45-row runtime lock, intentionally unequal QA/runtime
environments, warning-as-error PyTorch precision probe, CUDA gate, seed-domain
audit, formal-binding schema gate, canonical-versus-bound preformal verifier
tests, and real strict binding create-to-consume regression passed.

The prospective runtime seal was created and outer-finalized. The sole
result-free command-10 rehearsal returned zero, emitted canonical passing JSON,
and wrote exactly zero stderr bytes. The sole development producer then
completed both fresh v1.14 masters and was outer-finalized. Its independent
development audit was accepted, reproduced byte-for-byte, and outer-finalized.
The sole closure transaction published the canonical closure, retained its
fresh sealed-runtime reopen, and outer-finalized the accepted attempt. The
required post-finalization sealed-runtime reopen passed.

The one-shot preformal generator completed all ten fixed commands and published
the complete 20-log report. Both the separate QA-side report verifier and the
recorded-runtime canonical verifier accepted it. All ten command exit statuses
were zero and all ten stderr streams were the required zero-byte streams.

The sole binding transaction preserved and authenticated the accepted closure,
preformal report, 20 logs, conformance evidence, prospective review, runtime
seal, and source identities. It then invoked the independent formal-input
artifact auditor. That auditor rejected the development qualification archive
with the diagnostic:

```text
development archive contains an extra member
```

The transaction published and outer-finalized a terminal binding attempt with
status `failure`, error type `ArtifactAuditError`, and failure code
`artifact_audit_error`. Protocol 1.14.0 may not be repaired, retried, resumed,
rebound, or launched formally. No v1.14 formal-input preflight, formal launch
marker, formal producer, formal audit, semantic review, adjudication marker, or
adjudication package exists.

## Evidence identity

- sealed Git commit:
  `1f7a3acc299fed2dcb4c754167f3cedc2794058f`
- sealed Git tree:
  `37c9853d29f09cacfc76affbdeea1fff7b9e3ac1`
- protocol SHA-256:
  `39f5820a91c8a504355f971449726ae0a9067cc856111a575bb038455d1fd635`
- prospective-review SHA-256:
  `df7df1337318d492d8e3adfea83ff11045fabf77729217cea269ce9509d06b19`
- prospective implementation-manifest SHA-256:
  `3cc1823eb76edba366a30083a3981b7e5d3d98c68bd5c29512898d1080ceaf5c`
- runtime-seal SHA-256:
  `3ddc0cae0916685eba7b4e56dceebb45425499632f951f3e7b2db2975b24ef61`
- result-free command-10 stdout: 10,096 bytes with SHA-256
  `ca60dc4e155be0797b726c16b6d1986d3c68120a506dff040ba2241324ef0c8c`
- result-free command-10 stderr: zero bytes with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- canonical development-closure SHA-256:
  `0b3e8c822adf21bdbd05340e2594ebcefa6f51535369b1960853354a9cc1ecef`
- accepted preformal report: 59,384 bytes with SHA-256
  `119f6214d1b9d4aba05658977724b300e65c9ed580c2d11cd2bbd91c59c5601f`
- terminal formal-binding payload: 35,863 bytes with SHA-256
  `aef5ed40f9c328d97ff75a4cd0d477a3ae819dd207f68a0cad101eed4803cb52`
- binding execution-failure: 360 bytes with SHA-256
  `a11d0921879214289a3a81299453722fae12f9bef33ebef979db1d602ce133b4`
- binding error message: 44 bytes with SHA-256
  `4d57190a3992fb64674dc64b9972edfb4a92ec801a0d6f4b84c98b24f9e5445d`
- failed, outer-finalized binding-attempt terminal: 14,966 bytes with
  SHA-256
  `28242f2751210cda4ca411523986c5c8440894d410f88db715675fe4dad59249`
  (the terminal and deterministic outer marker are inode `69375155` with
  link count two)
- independently streamed qualification archive: 86 bound members, all
  metadata and payload digests verified, zero missing members, and zero extra
  members
- formal-input preflight: absent
- formal launch marker and formal confirmation root: absent
- formal audit marker: absent
- semantic review: absent
- adjudication marker and package: absent

No result content, performance value, or K value was opened, printed,
summarized, compared, or used to choose the repair.

## Exact cause

The development qualification archive is valid. Its central streaming verifier
consumed all 86 members, checked each member against the closure's ordered
identity rows, reached the archive terminator, and found no membership delta.
The independent auditor had already matched and hashed those same 86 members
before raising its error.

The false rejection came from the independent auditor's end-of-stream check.
It iterated with:

```python
for expected, member in zip(members, stream, strict=False):
    ...
```

and then tried to prove exhaustion with:

```python
next(iter(stream))
```

Here `stream` is a `tarfile.TarFile`, not the iterator object created for the
first loop. Python's `TarFile.__iter__()` replays members retained in the
object's internal `members` list when a new iterator is requested. Therefore,
after the valid final archive member had been consumed, `iter(stream)` created
a second iterator and yielded an already verified member. The auditor
misclassified that replay as a physical extra archive member.

A result-free two-member canonical USTAR reproduction exhibits the same
behavior: the first iterator consumes exactly `a` and `b`, while
`next(iter(stream))` returns `a`. Retaining the original iterator and checking
`next(original_iterator)` reaches `StopIteration` as intended.

No direct independent-auditor regression exercised a valid multi-member USTAR
through this exact helper. The central archive verifier used a single
`for member in archive` loop with an in-loop upper-bound check and therefore
did not share the defect. Higher-level binding tests either did not reach the
independent archive helper with a real archive or replaced that seam with a
fixture.

## Admissible claim

The evidence authenticates the sealed harness, completed development
execution, accepted development audit and closure, complete passing preformal
checks, a structurally assembled binding package, and a terminal independent
archive-verifier failure. It does not authorize a formal confirmation.

No claim that Prospect learned, improved, retained an improvement, or met any
formal performance threshold is supportable from protocol 1.14.0.

## Results-audit disposition

| Claim class | Evidence checked | Disposition |
| --- | --- | --- |
| Engineering custody and terminal failure | Authenticated attempt, same-inode outer completion, content identities, lifecycle-path absence | **confirm** |
| Archive membership mismatch | Independent 86-member stream verification plus result-free iterator reproduction | **refute** — the archive has no extra member |
| Independent-auditor implementation defect | Exact source branch plus synthetic canonical USTAR reproduction | **confirm** |
| Learning, improvement, retention, or performance | No authorized formal outcome; result-bearing payloads remained unopened | **retire** |

The audit recomputed no scientific metric because v1.14 never obtained formal
authority and opening development performance values could not support a
confirmatory claim. It inspected no raw result payload, prediction tensor,
trace, K value, performance summary, or captured command output. The decisive
follow-up is the fresh-version real-archive acceptance regression and full
one-shot confirmation specified below.

## Required fresh-version repair

A successor protocol must:

1. retain one iterator object for the independent USTAR member pass and use
   that same object for the final exhaustion check;
2. add a real canonical multi-member USTAR acceptance test that would fail if
   already consumed members are replayed;
3. retain negative tests for a physical extra member, omitted metadata row,
   reordered row, duplicate member, noncanonical header, changed payload,
   truncation, terminal garbage, symlink, hard link, and archive TOCTOU;
4. exercise the real development-closure-to-independent-formal-input audit
   seam without replacing archive verification;
5. compare the central and independent archive verifiers against the same
   generated valid archive and mutation corpus;
6. use fresh versioned paths, derivation domain, seeds, environments, wheel,
   lock, prospective review, seal, development evidence, binding, and formal
   authority;
7. directly supersede the exact v1.14 protocol SHA-256
   `39f5820a91c8a504355f971449726ae0a9067cc856111a575bb038455d1fd635`;
   and
8. preserve every v1.14 producer, audit, closure, preformal, failed-binding,
   execution-failure, and outer-completion byte unchanged.
