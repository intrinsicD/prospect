# WM-001 v1.9 preformal test failure

Date: 2026-07-20

Disposition: **retired — no binding or formal authority**

## Terminal outcome

The sole WM-001 protocol 1.9.0 development producer completed under commit
`adb586bdd5bc7cf1da3f462558b8f9c8b772226c` and was outer-finalized. Its
canonical independent development audit was accepted and outer-finalized. The
sole development-closure transaction published its canonical archive, retained
fresh-runtime receipt, accepted operator attempt, and same-inode outer
completion. The required post-finalization sealed-runtime reopen passed.

The prospective preformal generator then executed its fixed ten-command gate.
Nine commands passed, but command 6, `pytest-wm001`, returned 1 with one failed
test and 471 passed tests. The separately invoked public report verifier
therefore rejected the report:

```text
PreformalEvidenceError: preformal report runtime, source, or Git identity differs
```

The message was imprecise: every recorded identity was stable, while the
report's `all_pass` field was false. This is terminal failure evidence.
Protocol 1.9.0 may not be repaired, retried, resumed, upgraded, bound, or
completed. No v1.9 formal binding, formal producer, formal audit, semantic
review, or adjudication package may be created.

## Evidence identity

- protocol SHA-256:
  `3b97eaa1330066a7773345afd3445f086139d5e6090e8f86bfad87d14e93f090`
- prospective-review SHA-256:
  `a0b581995eea28d44dcb89008e1825edf342e1e67ed9a1fc9c9ad6af3e5ba1b6`
- implementation-manifest SHA-256:
  `eae945e19a04328465f687093b47c9eff3e8c1f145b6da7de7185e3b9c73b45c`
- runtime-seal SHA-256:
  `6a16ffcfb898d6dcf934b4f87d85a901087d3b8413af0b2777b26a4c9b0333f8`
- producer-manifest SHA-256:
  `fbdca9f7cea24f65a805ff6eeabcb9897b493a6b4931578648bc3bb1b07d1c0b`
- raw-result bytes:
  `319289566`
- raw-result SHA-256:
  `5b6fd1ed77c958005809c7a6acabd154a0fc51be3f791d5339eb0c0eab5b345a`
- accepted development-audit terminal SHA-256:
  `80404af0c8cdda67d831cf91f2e3b75ded5e7016ccf80bdd0319122afbed647b`
- qualification-archive bytes:
  `1116067840`
- qualification-archive SHA-256:
  `1c88a06944a071cc0d6b6ae8302e856a7ff41b931c2a9028e59ad47387ff998a`
- canonical development-closure SHA-256:
  `fdb5e1e7b014088be5dad24e84ef4647a6e6d5d522836d0e0ac9a5614d6b8fa1`
- accepted closure-attempt terminal SHA-256:
  `e59ba458b64d35e1a227ad5872fb9a59829acdb4fdd68332fd5a78551a4f4c79`
- preformal report SHA-256:
  `13ff706b7a436a91c60895bb490bd3a9d3c594de0d8ed809211522c1ab970ff7`

The development result, accepted audit, and accepted closure remain permanently
claim-ineligible. No K3–K6 value was opened, printed, summarized, compared, or
used to select a repair.

## Exact cause

`test_runtime_custody_refusal_precedes_producer_root_creation` redirected
`run.DEVELOPMENT_RESULTS_ROOT` and `run.DEVELOPMENT_QUALIFICATION_PATH` into a
temporary directory but did not redirect the development-closure path imported
inside `run.main()`. Before real evidence existed, the test reached its mocked
runtime-custody refusal and passed. After the canonical v1.9 closure existed,
the same test observed that live path and correctly returned from the earlier
same-version-closed guard:

```text
WM-001 protocol 1.9 development is closed; additional same-version
rehearsals are forbidden
```

The test was therefore dependent on lifecycle state outside its temporary
fixture. The pre-evidence suite could pass while the identical post-closure
suite failed.

Two additional reporting and independence defects were identified without
opening outcome values:

1. `generate-report` always emitted a `passed: true` generation envelope after
   writing a report, even when a command failed and the report contained
   `all_pass: false`.
2. The independent auditor's preformal command/input contract had drifted from
   the live generator contract, while circular fixtures generated from the
   auditor's own stale constants concealed the mismatch.

## Required fresh-version repair

A successor protocol must:

1. make every lifecycle path used by runner tests explicit and hermetically
   redirected by the fixture;
2. prove the complete WM-001 suite passes both before and after a synthetic
   development closure exists;
3. preserve an immutable failed report but make `generate-report` return a
   nonzero status and `passed: false` whenever any command fails;
4. report failed-command evidence separately from identity drift;
5. compare the independent auditor's preformal command and input contracts
   directly against a separately captured producer contract, with non-circular
   golden tests;
6. use fresh versioned paths, seeds, environments, schemas, seal, review, and
   binding; and
7. preserve all v1.9 producer, audit, closure, outer-completion, and preformal
   evidence unchanged.
