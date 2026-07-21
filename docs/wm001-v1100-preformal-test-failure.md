# WM-001 v1.10 preformal test failure

Date: 2026-07-21

Disposition: **retired — no binding or formal authority**

## Terminal outcome

The sole WM-001 protocol 1.10.0 development producer completed under commit
`b61f45cb1da0652d6c32efd0ef0b276b2d8ca9a9` and was outer-finalized. Its
canonical independent development audit was accepted, reproduced
byte-for-byte, and outer-finalized. The sole closure transaction published the
canonical archive, retained fresh-runtime receipt, accepted operator attempt,
and same-inode outer completion. The required post-finalization sealed-runtime
reopen passed.

The prospective preformal generator then executed all ten fixed commands and
wrote their 20 content-addressed logs below its deterministic hidden claim.
Before constructing or publishing the report, its producer-side semantic check
for command 9 raised:

```text
RuntimeError: isolated installed inventory must equal the exact runtime
dependency closure; extra=['ast-serialize', 'iniconfig', 'librt', 'mypy',
'mypy-extensions', 'pathspec', 'pip', 'pluggy', 'pygments', 'pytest', 'ruff'],
missing=[]
```

The canonical `results/development/v1.10.0/preformal/` bundle is absent.
The durable `results/development/v1.10.0/.preformal.staging/` claim contains
exactly 20 log files and no report. That hidden claim consumes the one-shot
preformal attempt. Protocol 1.10.0 may not be repaired, retried, resumed,
upgraded, bound, or completed. No v1.10 formal binding, formal launch, formal
producer, formal audit, semantic review, or adjudication package exists.

## Evidence identity

- protocol SHA-256:
  `fb2584cbbeab133692867e2396ee1ded5953ca7ceb7d68134febccd5aed3970b`
- prospective-review SHA-256:
  `f5f9bc08011fc0fcdeba8761d27adaff9838e02293ac57c4aa996726d21376c4`
- implementation-manifest SHA-256:
  `104436d3fbcc26814449d2c4470ebd1581c59485c18f61527d47f02f41ac02e8`
- runtime-seal SHA-256:
  `5a40cb72ae1bdfd31e64b2aef75e247b40e57217f9226fbd933866cfdbd6d12c`
- producer-manifest SHA-256:
  `a95f88d97a86503aa4f272220bc4c246a996fc41e7afee54e0a4abefd329464b`
- raw-result SHA-256, emitted by the accepted sealed closure reopen:
  `df396779f8745bae836bfd0055cb7fec461b13756ed5b37928c136ba7ac980bd`
- accepted development-audit terminal SHA-256:
  `7cb5d871f0ddad3387a4f8ce221a1b4b788c0f75e0755afea78150cd445ab92a`
- canonical development-closure SHA-256:
  `ce293d65e7c88aba9a93f3158a43a55c79041df911d6cc11771d15b526a7aba0`
- accepted closure-attempt terminal SHA-256:
  `15c9cb4faf9eb92606fd8c725b9a473580dea7226a18471a10948ad998a04c9d`
- newline-terminated sorted hidden-log-name manifest SHA-256:
  `6c68f9364b7ba642df86c8bfa3109d482e7da20f64dc0027d5196de25121ed14`
- canonical JSON array of the 20 sorted `{path,bytes,sha256}` hidden-log
  identities, serialized without a trailing LF: 5,462 bytes total and SHA-256
  `2b978fe6f253d2dc19449dfc5cf6de1b334b0d4e3ddbcfa441b3aa831262d99e`
- recursive retired-evidence inventory
  (`prospect.wm001.retired-evidence-inventory-summary.v1`): 132 sorted path
  rows comprising 8 directories and 124 regular files
  (2,238,884,602 regular-file bytes), with canonical rows SHA-256
  `e066db5d4955385a676faa0fc9e1377741ee6a958f5ff91cbd0908e49f5c6ece`;
  rows contain `path`, `type`, `mode`, `device`, `inode`, `links`, `bytes`, and
  `sha256`, serialized as compact, key-sorted UTF-8 JSON without a trailing
  newline; the content-addressed qualification archive is
  `bench/world_model_lifecycle/results/development/development-qualification-0dcb207529ac7094.tar`
  with SHA-256
  `0dcb207529ac7094934ac09f7da0a6cfb69a195c333a14d310d9a5442c0cfac1`.
  The eight inventoried roots are the archive,
  `results/.wm001-v1.10-runtime.lock`,
  `results/development/qualification-v1.10.0`,
  `results/development/runtime-seal-v1.10.0.json`,
  `results/development/development-closure-v1.10.0.json`,
  `results/development/v1.10.0/.preformal.staging`,
  `results/operator-v1.10`, and `results/outer-completions/v1.10`, all below
  `bench/world_model_lifecycle/` where no fuller prefix is shown.
  The empty `v1.10.0` parent directory itself is not a row; its only child and
  all regular evidence content are inventoried.
- preformal report: absent
- formal binding: absent
- formal launch marker: absent

The development result, accepted audit, and accepted closure remain permanently
claim-ineligible. No performance or K value was opened, printed, summarized,
compared, or used to select the repair.

## Exact cause

Command 9, `runtime-accepted-closure-evidence`, correctly ran inside the
minimal sealed runtime. It deeply verified the development closure, archive,
producer, audit, package inventory, retained fresh-runtime receipt, closure
attempt, and same-inode outer completion before emitting its canonical passing
receipt.

The generator then tried to validate that receipt semantically from the QA
process. `_accepted_closure_evidence_from_report()` called
`_verified_closure_member_digests()`, which called the full
`verify_development_closure()` routine a second time. That full verifier
correctly requires the current interpreter's installed distributions to equal
the minimal 45-row runtime closure. The QA environment intentionally also
contains pytest, Ruff, mypy, pip, and their support packages, so the
environment-sensitive verification correctly refused.

The boundary was therefore composed incorrectly: a result generated and fully
verified under sealed runtime custody was reverified through a routine whose
ambient-inventory precondition cannot hold in the QA process. Unit tests used
synthetic closure projections and did not execute this exact post-command
composition with real, unequal QA and runtime inventories. The semantic helper
also caught only `PreformalEvidenceError`; the ordinary `RuntimeError` escaped
before a truthful failed report or generation envelope could be constructed.

## Required fresh-version repair

A successor protocol must:

1. keep the deep, environment-sensitive closure verification in command 9's
   sealed runtime process;
2. make the QA-side semantic check environment-neutral and limited to
   canonical receipt parsing plus exact cross-links to the captured closure,
   closure-attempt terminal, same-inode completion, and closure-declared archive
   member identities;
3. never invoke minimal-runtime inventory validation from the QA process;
4. convert any ordinary semantic-check exception into a named failed check so
   a report and generation envelope cannot misleadingly disappear or pass;
5. include an integration test that uses unequal real QA/runtime package
   closures and exercises the post-command semantic composition end to end;
6. adversarially mutate every command-9 digest and prove rejection without
   depending on ambient package inventory;
7. re-audit command 10 and the independent formal-input consumer for the same
   cross-environment category;
8. use fresh versioned paths, seeds, environments, schemas, seal, prospective
   review, and binding; and
9. preserve all v1.10 producer, audit, closure, outer-completion, and hidden
   preformal evidence unchanged.
