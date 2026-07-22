# WM-001 v1.19 development-closure custody failure

Date: 2026-07-22

Disposition: **terminally retired — the sole development producer and its
twice-reproduced outcome audit completed, but the sole closure transaction
rejected the valid outer-finalized producer manifest under an inconsistent
one-link custody expectation; no closure, binding, formal authority, or
scientific outcome exists**

## Terminal outcome

The exact WM-001 protocol 1.19.0 harness was sealed at Git commit
`0c2bc26acb2781d42e15b05b9a0beb8e5261a543`, tree
`5013701aee6eb0023cd05b1a9fd8b8c680f41a16`. Its independent prospective
review, 925-test suite, focused preformal, archive, capacity, launcher, and
runtime-boundary regressions, lint, type checks, schemas, installed-source
equality, dependency closure, precision probe, RTX 3050 CUDA gate,
seed-collision audit, lifecycle-path absence checks, runtime seal, and
result-free command-10 rehearsal all passed.

The sole development producer completed for both fresh v1.19 development
masters and was outer-finalized. The sole development-audit operator then ran
the complete descriptor-mode `outcome_audit` twice under the sealed 10,800
second role. Its accepted attempt proves byte-identical reports, stderr,
runtime manifests, and invocation manifests, and its independently verified
reproduction-v3 receipt carries two strict-positive elapsed receipts plus a
passing capacity-v1 projection. This confirms the v1.18 audit-runtime and
capacity repair at development scale; it is not a formal scientific result.

The sole development-closure transaction then failed. Its authenticated,
outer-finalized operator attempt records `status: failure`, phase
`development_closure`, error type `RuntimeError`, and failure code
`runtime_error`. The canonical development-closure marker was never published.
The preformal bundle, binding attempt, formal marker, formal producer, formal
audit, semantic review, and adjudication package are all absent.

Protocol 1.19 is permanently retired. Its producer, audit, or closure must not
be rerun, resumed, removed, or repaired in place.

## Evidence identity

- protocol SHA-256:
  `07c6fe364aeddbd5689fa4f638a6f9a38506b16e8845a947fffa87e01eb3854a`
- prospective-review SHA-256:
  `a01841bc0f2d71d782f1fc3d560145cb00ec29031ac9afcf0e494acd378d4618`
- dependency-lock SHA-256:
  `80852b69d4f15a04a818e3c7d311b86db7e08d3301c68583f959ac1f1022ae93`
- runtime seal: 1,944 bytes, SHA-256
  `ddd2881ddb457a080cd5e88dd3f8f0c56ae6d72d30a4952261c7d5d160afd215`,
  link count two
- development producer manifest: 12,625 bytes, SHA-256
  `2c9339bc54c833609bdc7fd896fe3ecde2b3f07fcb06cbf2af08e617ce96a5f9`,
  link count two
- development result: 320,176,812 bytes, SHA-256
  `fbff2016a617cc9e6574afb9a468a8a74dd82a87e6b3f9e4ca3f8aa77ba74bcb`,
  link count one; its performance values remained unopened by the operator
- accepted development-audit terminal: 3,271 bytes, SHA-256
  `0c466ba451387fe2407a61af7a3dff14bcaf6d91d21498596297c28079e17b55`,
  link count two
- audit-reproduction v3 receipt: 2,618 bytes, SHA-256
  `66bf09789e3718ae61a532492b947d88705d14e8f23b2d03f849cfb6bc9b2dd2`
- failed development-closure terminal: 5,296 bytes, SHA-256
  `5d86300d39e108b4601845951f473a6acde565de0dc6d91208642f62056da124`,
  link count two
- canonical execution-failure record: 368 bytes, SHA-256
  `22d5337ef1fd8bd2f10961c40ee0e80563a5145f0f4b994e0fb64d12553f6a61`
- recorded exception-message identity: 66 bytes, SHA-256
  `405ba517d55241736d7206b2a7fd8a42e3f24339323e777980055f3fd1dbe35e`

The producer manifest, audit attempt, capacity evidence, and failed closure
attempt all pass their supported structural verifiers. No raw performance
value was opened or promoted while preparing this record.

## Exact failure and deeper engineering cause

The failure record intentionally stores only the exception type plus the byte
length and SHA-256 of its rendered message. The committed source renders:

```text
development producer manifest violates its 1-link custody contract
```

to exactly the recorded 66 bytes and SHA-256. The runtime path is unambiguous:

1. The outer launcher correctly hard-links `producer-manifest.json` to its
   deterministic completion marker. The valid terminal therefore has link
   count two.
2. Producer and audit custody verification correctly accept that two-link
   terminal.
3. `create_development_closure()` later rereads the same manifest through
   `_load_canonical_json()`.
4. That generic helper silently delegates to `_stable_regular_payload()` with
   its default `expected_nlink=1`.
5. The closure therefore rejects the valid two-link terminal before it can
   publish the canonical marker.

This is a sealed implementation bug, not corrupt operator evidence and not a
scientific failure. It is distinct from v1.18: the role-specific outcome audit
completed twice and its capacity evidence passed before this later custody
reader failed.

## Why prospective verification missed it

The archive and live-producer regression directly proves that an
outer-finalized manifest has link count two, and the operator tests exercise
closure sequencing. However, the real positive closure creator was not
composed with that real outer-finalized producer. The operator end-to-end test
substitutes the closure creator, while the archive test starts below the
canonical JSON reread. Each component passed separately; no prospective test
crossed their exact boundary.

The generic JSON reader's hidden one-link default made the mismatch easy to
miss in source review. A correct successor must make the terminal-manifest
custody expectation explicit at the call site and test the complete real
reader chain.

## Claim audit

| Claim | Disposition |
| --- | --- |
| The v1.19 role-specific audit timeout and measured capacity path works at development scale | Confirmed in engineering scope by one accepted twice-reproduced development audit |
| The v1.19 development closure completed | Refuted; its sole transaction is an authenticated failure |
| The failure indicates bad producer or audit evidence | Refuted; both upstream packages pass their strict verifiers and the rejected manifest has the protocol-required link count two |
| K0–K7 passed or failed formally | Not established; no formal producer or formal audit exists |
| Experience caused a persistent shared-model update | Not established as a formal claim |
| Held-out prediction or executed behavior improved | Not established as a formal claim |
| Plasticity, interference, replay retention, or fresh-process parity passed formally | Not established |
| Prospect demonstrated collect → learn → improve → retain | Not established |
| The scientific hypothesis failed | Not established; the terminal event is a closure-custody implementation failure |
| Prospect generally has or lacks the target capability | Unresolved and outside WM-001's fixture scope |

One-sentence verdict: **v1.19 repaired and exercised the development audit
capacity path, but an inconsistent one-link closure reader rejected the valid
two-link producer terminal, so the version is retired without formal authority
or a scientific conclusion.**

## Required fresh-version repair

A successor must:

1. preserve all v1.19 evidence unchanged and never replay its consumed
   producer, audit, or closure;
2. preserve the scientific system, budgets, controls, metrics, thresholds,
   schemas, capacity formula, and serialized representations;
3. read the live outer-finalized producer manifest once under an explicit
   two-link custody contract and derive both its object and digest from those
   same captured bytes;
4. audit every live terminal-manifest reader for the same link-count mismatch;
5. add a real positive composition test from outer-finalized producer through
   the actual closure creator, plus negative one-link, extra-link, symlink,
   mutation, and digest-substitution cases at that exact seam;
6. ensure restoring the hidden one-link default makes the new regression fail;
   and
7. use a fresh protocol digest, seed domain, environments, paths, prospective
   review, and one new one-shot confirmation.
