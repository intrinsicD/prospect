# WM-001 v1.17 independent rehearsal-verifier failure

Date: 2026-07-22

Disposition: **retired — the one-shot rehearsal was accepted and
outer-finalized, but the required independent verifier rejected it before any
formal authority existed**

## Terminal outcome

The exact WM-001 protocol 1.17.0 harness was sealed at Git commit
`d6d010b7a39cb4aac3b927f64dd8e8b98d52b9af`, tree
`5b12819f639644af16dbd5172b4582d09992ee73`. Its prospective review, complete
test suite, lint, type checks, schemas, source/install equality checks,
precision probe, CUDA gate, seed-collision audit, and lifecycle-path absence
checks passed before outcome-producing work.

The sole development producer completed and was outer-finalized. Its ordinary
independent audit was accepted and reproduced; the closure transaction,
sealed-runtime closure reopen, ten-command preformal report, formal binding,
preflight reconstruction, and final stop/go checks all passed. Development
evidence remains claim-ineligible and its scientific values were not used in
this diagnosis.

The binding-keyed accepted-rehearsal transaction then dispatched exactly one
result-free child. Its authenticated terminal records `status: accepted`,
`child_started: true`, return code zero, empty stderr, an empty outer receipt,
and formal-path absence both before and after dispatch. The claim and global
claim marker are one inode with link count two; the terminal and deterministic
outer-completion marker are also one inode with link count two.

The next mandatory gate, the separate `rehearsal.py` verifier, rejected this
accepted package with:

```text
RehearsalEvidenceError: accepted-binding rehearsal fresh runtime identity is malformed
```

The runbook forbids retrying a consumed rehearsal or repairing a sealed source
in place. Protocol 1.17 is therefore retired. No v1.17 binding-keyed formal
root, formal marker, formal producer, formal outcome, formal audit, semantic
review, or adjudication exists.

One operator invocation used the QA closure verifier before the prescribed
sealed-runtime closure verifier. The QA-only verifier correctly refused that
runtime package and performed no mutation. The exact prescribed runtime
verifier was then invoked once and passed. This procedural detour did not
produce or authorize evidence and is retained here for completeness.

## Evidence identity

- protocol SHA-256:
  `b915d70eef0b09c7562b04f7c9f2e416cd12249c1b512108e11759d008473905`
- prospective-review SHA-256:
  `bb810b989c815549ae4576dc00bbb180421ee8ae25952558cfb5bf48bf1546e1`
- dependency-lock SHA-256:
  `8458d99eb472375bf7f92d5453ecb6dd7e9e55564158bc97c643c087a80dba23`
- runtime seal: 1,953 bytes, SHA-256
  `fc58b6c80b609f569c94c4a6a102c7e8a7be6ca72ba0844b0121187b261f8697`,
  terminal and outer marker inode `69505635`, link count two
- accepted development-audit terminal: 3,271 bytes, SHA-256
  `5d833347d3746e09d3c7351fa1c219c3125a1f0db6cedc098f7a80dd93930a8c`,
  terminal and outer marker inode `69220132`, link count two
- development closure: 20,049 bytes, SHA-256
  `448f7d3f59a4174740ec3aad20823cb30d9561880972e2f5eb48a4ab35ef85f2`
- accepted closure terminal: 5,368 bytes, SHA-256
  `010cc3fbd7f7f39f8868ae4dc6feee17210a05248543654c18a57e9f906b3adf`,
  terminal and outer marker inode `69220137`, link count two
- preformal report: 61,659 bytes, SHA-256
  `3017cb2fd5926c915e2f230cd198222bd5d3470fb4c690c2971a72bfcd04dc84`
- formal binding: 36,309 bytes, SHA-256
  `c768d4e3319a7c56a6f0bcd7bc56690f67912e0d1d5a6791cc6b469de3f60b56`
- formal-input preflight: 705 bytes, SHA-256
  `05c38865b71426fa050de5315c206c2a5884fe61db03f8c8a6482bff1deea780`
- accepted binding terminal: 15,028 bytes, SHA-256
  `7caa2b0ab4572dbda9a1d01c77902ab62b3b90a6cc770c29d646acf6a56b588e`,
  terminal and outer marker inode `69220179`, link count two
- rehearsal claim: 1,424 bytes, SHA-256
  `b53e78404c4676f8ebcfd8dce43969e3d3221783c27e1390b099ba70b154719f`,
  claim and marker inode `69220184`, link count two
- rehearsal stdout: 10,102 bytes, SHA-256
  `211b05580302c091055d6259c131cf8d2bc601adb2933149cb6e64ff9f8018b6`
- rehearsal stderr and outer receipt: zero bytes, each SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- rehearsal terminal: 2,018 bytes, SHA-256
  `4f92b04665bb3bcf643a2c266e1e2357c281bf0629b08bd91647532c263e85ab`,
  terminal and outer marker inode `69220220`, link count two
- every v1.17 formal, formal-audit, review, and adjudication path: absent

## Exact engineering cause

The fresh-process child emitted matrix-contract SHA-256
`09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84`.
That is the exact reviewed identity bound by the sealed protocol. The
standard-library launcher validates this output against its independent golden
constant and correctly published the accepted terminal.

The separate verifier used a different contract. It read its expected value
from:

```python
binding["development_qualification"]["matrix_contract_sha256"]
```

Formal-binding representation v10 deliberately projects the development
closure into a performance-free identity and does not contain this field. Its
JSON Schema has `additionalProperties: false`, so adding the field would make a
production binding invalid. The lookup therefore returned `None`, and the
verifier rejected a valid output by comparing the reviewed digest against a
nonexistent binding member.

This diagnosis was checked three ways without opening performance values:

1. the actual binding's exact key set and v10 schema both prove that the field
   is absent by design;
2. the actual stdout digest equals both the sealed protocol identity and the
   launcher's reviewed constant; and
3. replacing only the nonexistent lookup in memory with the protocol-verified
   constant makes the complete independent package verifier and all four
   rehearsal identity projections pass. All formal paths remain absent.

The empty outer receipt is not a defect: the result-free producer entry is
specified to emit no outer payload, and both launcher and verifier require the
zero-byte receipt.

## Why the prospective gates missed it

The independent-verifier unit fixture invented
`development_qualification.matrix_contract_sha256`, then generated its stdout
from that invented field and monkeypatched strict binding verification. The
larger producer/launcher composition fixture made the same schema-forbidden
addition and its synthetic child also read it. These tests proved internal
agreement between two non-production fixtures, not agreement with the exact
formal-binding v10 projection.

The v1.17 work successfully repaired the prior real audit-package composition
gap, but its new rehearsal boundary introduced another fixture-to-production
shape gap. Test count and subprocess scale did not compensate for the missing
schema-exact composition assertion.

## Claim disposition

| Claim | Disposition |
| --- | --- |
| The v1.17 producer, audit, closure, preformal, and binding engineering chain completed | Confirmed in engineering scope |
| The accepted rehearsal child and launcher output were malformed | Refuted |
| The independent rehearsal verifier accepted the production v10 binding/output pair | Refuted; verifier contract is wrong |
| A v1.17 formal experiment completed | Refuted by authenticated path absence |
| Prospect learned, improved, and retained improvement in v1.17 | Not established; no formal outcome exists |
| Prospect generally has or lacks the target capability | Unresolved |

One-sentence verdict: **v1.17 reached an authentic accepted, outer-finalized
result-free rehearsal, but a schema-inconsistent independent verifier blocked
formal authority; it contributes engineering evidence only and no scientific
claim.**

## Required fresh-version repair

A successor must:

1. preserve v1.17 evidence unchanged and never retry its consumed rehearsal;
2. preserve the scientific protocol, budgets, controls, metrics, thresholds,
   representation v9, formal-binding v10, raw-result v9, and formal-launch v3;
3. make the independent verifier compare the emitted matrix digest with the
   reviewed identity already verified against the sealed protocol, not with a
   nonexistent binding field;
4. remove the forbidden field from every fixture and assert the exact v10
   development projection;
5. include a real producer-shaped binding/output-to-independent-verifier
   composition regression and a wrong-digest rejection regression; and
6. use a fresh protocol digest, fresh seed domain, fresh QA/runtime
   environments, fresh lifecycle paths, prospective review, and one new
   one-shot confirmation.
