# WM-001 v1.18 formal-audit timeout

Date: 2026-07-22

Disposition: **terminally rejected — the sole formal producer completed, but
the sole independent formal-audit execution reached its sealed 600-second
deadline before emitting a report; the outer-finalized adjudication performed
zero replay and forbids a same-version retry**

## Terminal outcome

The exact WM-001 protocol 1.18.0 harness was sealed at Git commit
`a4e126e99dcc234a3c6c8d17c0194d134befd998`, tree
`e3aa80dae0ae268d8885355520fc5e67cf5045af`. Its independent prospective
review, 908-test suite, focused archive and rehearsal regressions, lint, type
checks, schemas, installed-source equality, dependency closure, precision
probe, RTX 3050 CUDA gate, seed-collision audit, lifecycle-path absence checks,
runtime seal, and result-free command-10 rehearsal all passed.

The sole development producer completed. Its independent audit, closure,
fresh-runtime reopen, ten-command preformal report, formal binding, preflight,
final stop/go, and accepted-binding rehearsal all passed. The production
launcher stored exactly the child bytes, and the separate v1.18 verifier
accepted those same bytes using the sealed protocol matrix identity. This
confirms the v1.17 verifier defect was correctly repaired.

The sole formal producer then completed and was outer-finalized. The bound
preflight and development-result-qualification sidecars matched their binding
copies byte-for-byte. The producer stdout was retained unopened.

The sole formal audit did not produce an ordinary report. Its authenticated,
outer-finalized attempt records `status: failure`; the captured execution
records `phase: timeout`, no return code, zero stdout bytes, and zero stderr
bytes. The supported adjudication inspector classifies the evidence as
`execution_failure`, with `replay_performed: false`, no independent-audit
digest, and an empty reviewed-gate list.

An independent semantic review therefore requested `rejected`. The one-shot
adjudicator performed zero audit replay and published an outer-finalized,
independently verified rejection package with:

```text
outcome_kind: formal_audit_execution_failure
audit_execution_completed: false
audit_byte_identical: null
same_version_replay_permitted: false
disposition: rejected
```

Protocol 1.18 is permanently retired. Its audit must not be rerun, resumed, or
repaired in place.

## Evidence identity

- protocol SHA-256:
  `5def6aaa0fc474675483049dd0b8661abb8819bab459f8e42d4d33b919145cb1`
- prospective-review SHA-256:
  `8e5b5a66a7a17fdc6579f4841e459af962bfa1b756dbedfb6a02367e97a51f31`
- dependency-lock SHA-256:
  `30c776858aa2f38986f103a9e419063abcde99144cc205a4e8db97c58818e9b7`
- runtime seal: 1,944 bytes, SHA-256
  `d46f99fa916a18047a8d3bfd7f2935e0ffe0e3eac09e74b006c7c5b9fba8de80`,
  link count two
- accepted development-audit terminal: 3,271 bytes, SHA-256
  `f8df8c3fd6b5a6a500bbbd4ffbe207927145df73cbacd02c3f27c6d7415d4daf`,
  link count two
- development closure: 20,040 bytes, SHA-256
  `801d1addcfcbdb003a0f0635ef018add61fc3b89fb43e2c931b81f6200cbd46f`
- accepted closure terminal: 5,368 bytes, SHA-256
  `76ece88e830fa9c93ec07be25f19f336248114924768cc154fbf961141b39f02`,
  link count two
- preformal report: 57,958 bytes, SHA-256
  `f17b5b826e509953bbd7d64c9f0bd51428ab94684e47fa08d8282faa9e0fb151`
- formal binding: 36,298 bytes, SHA-256
  `e79d19f60d31e574dd9a692e33da44d903246a5ed9af693d0211f0172569546b`
- formal-input preflight: 705 bytes, SHA-256
  `a76430ffddc50acb8e5d5197a7149ae2f9d1ca4fe818e8adad003af70d5c6e97`
- accepted binding terminal: 15,028 bytes, SHA-256
  `dbaa4ce751ae1f5223564f1fc5ec0bb006229fa2b70c6bbc8bd0238dbf1a23f3`,
  link count two
- accepted rehearsal terminal: 2,018 bytes, SHA-256
  `4d62283b9edb38c6703ee8a9a0c342101064635ff1fbdafccb55c47159a9d83a`,
  link count two; launcher stdout and stderr were empty and the separate
  verifier accepted
- formal producer manifest: 68,188 bytes, SHA-256
  `1cdebf48944f59a79b08db8f5a5c9cb17135e7ab5597dc5640e7a11195750b0e`,
  link count two
- formal result SHA-256, bound by the supported inspector:
  `1c9416234d025fb1fd808cb9da3d02607b126987425f8f11c90b6133babed42f`
- formal-audit terminal: 2,661 bytes, SHA-256
  `a388df42b3785547e2f9551f43b44893309b15f22538fd5e6064d71af545dfcc`,
  link count two
- captured formal-audit execution failure: 1,495 bytes, SHA-256
  `10234f8b2adf83a59f1a60dd256684d4788b42b72edfa39d0007673c62be835f`
- semantic review: 1,740 bytes, SHA-256
  `684e9a053d7e7316a7bfaa821a68185fe713bccbd94db6c85d306c186fd0e72a`
- adjudication manifest: 9,477 bytes, SHA-256
  `74f4bb1a5c93b94bb388cdc3d390334240e550b20c77eeaaa0a22ae52530a529`,
  link count two

All development and formal performance values remained unopened by the
operator. Only precommitted producer, verifier, audit, inspector, semantic
review, and adjudication paths processed the raw evidence.

## Exact failure and deeper engineering cause

The proximate failure is conclusive: the captured auditor was killed by the
sealed 600-second child timeout. The authenticated failure receipt records
`phase: timeout`, `returncode: null`, empty streams, and the exact bound runtime
manifest whose limit is 600 seconds.

The deeper harness defect is a missing scale/liveness contract. Development and
formal outcome audits used the same timeout despite materially different
declared packages:

- development package: 1,119,343,819 bytes; result: 320,703,854 bytes;
- formal package: 4,404,192,892 bytes; result: 1,250,512,326 bytes; and
- formal/development scale ratios: approximately 3.93 and 3.90.

The independent auditor hashes every manifested file and then rereads,
materializes, parses, schema-validates, and semantically checks the result
inside the timed subprocess. Read-only filesystem timing around the accepted
development audit suggests roughly 210 seconds per replay; linear formal-scale
estimates are roughly 819–826 seconds, already beyond the sealed limit. That
timing is engineering calibration, not authenticated scientific evidence.

The timeout does **not** establish that a longer audit would pass. No ordinary
report exists, no gate was reviewed, and the failure has no phase telemetry
beyond `timeout`. A specific internal bottleneck and every scientific endpoint
remain unresolved.

## Why prospective verification missed it

The formal operator tests replace the real bound auditor, semantic-audit tests
use small fixtures, result-free conformance does no formal-scale semantic audit,
and the production-scale streaming regression verifies sparse-file custody
rather than audit completion under the timeout. No prospective gate related
the declared 4 GiB result and 16 GiB aggregate limits to a measured or
conservatively derived audit-time budget. Version 1.18 was the first retained
formal-audit attempt to reach this boundary.

## Claim disposition

| Claim | Disposition |
| --- | --- |
| The v1.18 schema-exact accepted-binding rehearsal repair works | Confirmed in engineering scope |
| One sealed formal producer transaction completed with authenticated custody | Confirmed in engineering scope |
| The independent formal audit completed or reproduced the result | Refuted; it timed out before a report |
| K0–K7 passed or failed | Not established; no gate was independently reviewed |
| Experience caused a persistent shared-model update | Not established |
| Held-out prediction or executed behavior improved | Not established |
| Plasticity, interference, replay retention, or fresh-process parity passed | Not established |
| Prospect demonstrated collect → learn → improve → retain | Not established |
| The scientific hypothesis failed | Also not established; this was an audit-execution failure |
| Prospect generally has or lacks the target capability | Unresolved and outside WM-001's fixture scope |

One-sentence verdict: **v1.18 completed a sealed formal producer, but its only
independent audit timed out before producing any report, so the experiment is
terminally rejected and establishes no scientific endpoint in either
direction.**

## Required fresh-version repair

A successor must:

1. preserve all v1.18 evidence unchanged and never replay its consumed audit;
2. preserve the scientific system, budgets, controls, metrics, thresholds,
   schemas, and serialized representations;
3. separate result-free conformance timeouts from full outcome-audit timeouts;
4. bind the selected role-specific timeout in the exact runtime manifest and
   use that value in the child process, rejecting arbitrary or mismatched
   values in launcher, bootstrap, verifier, auditor, binding, and adjudicator;
5. prospectively prove conservative audit-capacity arithmetic against the
   declared 4 GiB result and 16 GiB aggregate ceilings with at least 2×
   headroom, rather than tuning only to this observed formal artifact;
6. retain authenticated timeout-failure packaging and add regressions for role
   propagation, manifest mismatch, arbitrary-value rejection, and formal audit
   selection; and
7. use a fresh protocol digest, seed domain, environments, paths, prospective
   review, and one new one-shot confirmation. A larger timeout alone does not
   imply that the scientific audit will pass.
