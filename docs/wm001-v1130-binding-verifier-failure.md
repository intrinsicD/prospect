# WM-001 v1.13 bound-report verifier failure

Date: 2026-07-21

Disposition: **retired — failed binding, no formal authority**

## Terminal outcome

The exact WM-001 protocol 1.13.0 harness was sealed at Git commit
`d5851e07ca10963241042bbe301ef69af35a0643`. Its protocol, independent
prospective review, 675-test suite, lint, two type-check scopes, installed
source snapshots, exact 45-row runtime lock, unequal QA/runtime environments,
warning-as-error PyTorch precision probe, CUDA gate, seed-domain audit, and
formal-binding root-schema gate passed.

The prospective runtime seal was created and outer-finalized. The exact
result-free command-10 rehearsal returned zero, emitted canonical passing JSON,
and wrote exactly zero stderr bytes. Its separately retained capture remained
result-free. The four real-subprocess audit modules then passed all 194 tests.

The sole development producer completed both fresh v1.13 masters and was
outer-finalized. Its independent development audit was accepted, reproduced
byte-for-byte, and outer-finalized. The sole closure transaction published the
canonical closure, retained its fresh sealed-runtime reopen, and outer-finalized
the accepted attempt. The required post-finalization sealed-runtime reopen
passed.

The one-shot preformal generator completed all ten fixed commands and published
the complete 20-log report. Both the separate QA-side report verifier and the
recorded-runtime verifier accepted it. All ten command exit statuses were zero,
all ten stdout streams were nonempty, and all ten stderr streams were the exact
zero-byte streams required by the protocol.

The sole binding transaction copied and authenticated the accepted evidence and
constructed a complete formal-binding v10 object. The repaired root schema
accepted that object, including all ten empty stderr rows. The transaction then
failed its immediate strict consumer check at:

```text
preformal report path is missing or aliased
```

The binding layer wrapped that diagnostic as:

```text
formal test report does not prove the complete fixed preformal check set
```

The transaction published and outer-finalized a terminal binding attempt with
status `failure`, error type `RuntimeError`, and failure code `runtime_error`.
Protocol 1.13.0 may not be repaired, retried, resumed, rebound, or launched
formally. No v1.13 formal-input preflight, formal launch marker, formal producer,
formal audit, semantic review, adjudication marker, or adjudication package
exists.

## Evidence identity

- sealed Git commit:
  `d5851e07ca10963241042bbe301ef69af35a0643`
- protocol SHA-256:
  `e7988e3605079b7b7830949d6fd107f26066059ac3cc3974c5bfe15af876dc0c`
- prospective-review SHA-256:
  `8c454efe3ce611767368a6e1f9089b9ea6692a44c34c1c28229d2c99ff2911e0`
- prospective implementation-manifest SHA-256:
  `ee7a6527dc04d1bc1550fa4255d6d2036f60fb5c9353d0e0ae6baca61b0928dd`
- runtime-seal SHA-256:
  `054c66d93ddf9fd0d85fc6ea7e883e88200ebbc38d45c4a1dc91cdcaa7db8e4f`
- result-free command-10 stdout: 10,096 bytes with SHA-256
  `e7efa4fd45ed260b0c3e1695547d1fd7957b2fc16716217e2c2ffdde5f96d8c7`
- result-free command-10 stderr: zero bytes with SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- producer-manifest SHA-256:
  `0e86e609cf8b34d147f150a69de4ebd2750f256b19c8e55efb8e439ffad32c10`
- raw-result SHA-256, authenticated without opening its content:
  `9bff723727390e4dd01d91ae1817841f91beb0c321aba86ee4a83e118a913784`
- accepted development-audit terminal SHA-256:
  `417d57d0235e63a733b36c258f7c8c11f19d52a3261717d9cd39b59acb03f1a8`
- canonical development-closure SHA-256:
  `41ce92c2137613c634173c9531f294cbdeaa5eb8d98db560aeb5c5e265062785`
- accepted closure-attempt terminal SHA-256:
  `ffa204f4fd20283fcc67ebe2dc44f230f854614b8e127cd318c4566e63a436e9`
- accepted preformal report: 58,690 bytes with SHA-256
  `4c8e108d369370b78fe0451b58220c7072c8302a72f7f596b200e4e686e70b8a`
- terminal formal-binding payload SHA-256:
  `fa09aadb339fce089f44e26bac9f95ce1d64c78050419f730332326ccf5f896a`
- binding execution-failure SHA-256:
  `0dfcb45a2a197e7b21aced201dc160a378b33dad5109c28002cd0df3cfe22df5`
- binding error message: 72 bytes with SHA-256
  `63e3a328dcdd890e0e26b75ef27f9caab535850cea5c974ff3ddb132d4573e08`
- failed, outer-finalized binding-attempt terminal SHA-256:
  `ee38047cc1258778bee28c3d0191165358fcbadf164f5029ab3f60899c5c566f`
  (the terminal and deterministic outer marker are inode `69234165` with
  link count two)
- independent strict-verifier diagnostic capture: zero-byte stdout; 2,121-byte
  stderr with SHA-256
  `20e457537a5ba4ccdd99c5cc075d25bfeebf4cd6dd633e7a2198258628c34447`
- recursive fixed-root evidence inventory
  (`qualification-v1.13.0`, runtime seal, development closure, preformal
  bundle, `operator-v1.13`, and `outer-completions/v1.13`): 173 sorted rows,
  comprising 10 directories and 163 regular files
  (1,118,107,403 regular-file bytes); compact key-sorted canonical row JSON is
  53,071 bytes with SHA-256
  `632a36c55354ca101da932c12ad296c4353fa627d8a30098c1529914f6d4ddc3`.
- formal-input preflight: absent
- formal launch marker and formal confirmation root: absent
- formal audit marker: absent
- semantic review: absent
- adjudication marker and package: absent

No result content, performance value, or K value was opened, printed,
summarized, compared, or used to choose the repair.

## Exact cause

Binding construction correctly preserved the canonical preformal report and its
20 referenced logs as root-level siblings in a self-contained binding package.
The source and preserved reports are byte-identical, but necessarily have
different inodes and paths.

`verify_binding()` passed the preserved sibling to
`verify_machine_test_report()`. That function always delegated to
`verify_recorded_preformal_report()`, whose role is to verify the live,
canonical development bundle. Its internal verifier unconditionally requires
the absolute report path to equal the canonical development
`PREFORMAL_REPORT_PATH`. Consequently, every legitimately copied report was
rejected.

Removing only that path equality would not be sufficient or secure. The same
live verifier also requires the report directory to contain exactly the report
and its 20 logs, whereas a valid binding package contains the binding, closure,
conformance, and audit sidecars too. It also reopens ambient live repository,
environment, seal, review, and closure paths rather than validating only the
preserved package and its binding cross-links.

Existing generated-binding tests validated the complete object only against
JSON Schema while mocking report verification. Operator tests mocked the
strict binding consumer. No test exercised the real producer-to-preserved-copy
to strict-consumer seam, so this verifier-role contradiction escaped the
prospective gate.

## Admissible claim

The evidence authenticates the sealed harness, completed development execution,
accepted development audit and closure, complete passing preformal checks, and
terminal binding-verifier failure. It does not authorize a formal confirmation.
No claim that Prospect learned, improved, retained an improvement, or met any
formal performance threshold is supportable from protocol 1.13.0.

## Required fresh-version repair

A successor protocol must:

1. retain the strict canonical verifier unchanged for the unique live
   development/preformal bundle;
2. add an explicitly separate bound-package verifier, not a permissive mode
   flag, that authenticates the preserved report, its exact 20 referenced
   sibling logs, and all report-to-binding cross-links without reopening
   recorded absolute paths;
3. permit unrelated legitimate binding sidecars while rejecting any missing,
   duplicate, reordered, traversing, symlinked, hard-linked, tampered, or extra
   reserved `preformal-*` evidence member;
4. preserve command, environment, source, Git, review, runtime-seal,
   command-9 closure, and command-10 runtime-conformance semantics using bound
   evidence and identities;
5. add an unmocked producer-to-consumer integration test that creates a
   complete binding package and immediately passes its real
   `formal-binding.json` to the real strict verifier;
6. prove the canonical verifier rejects the identical noncanonical copy and
   the bound verifier does not consult unavailable ambient original paths;
7. retain independent central and artifact-auditor implementations and run a
   shared mutation corpus against both;
8. use fresh versioned paths, derivation domain, seeds, environments, wheel,
   lock, prospective review, seal, development evidence, binding, and formal
   authority; and
9. preserve every v1.13 producer, audit, closure, preformal, failed-binding,
   execution-failure, and outer-completion byte unchanged.
