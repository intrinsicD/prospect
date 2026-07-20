# WM-001 v1.8 development-closure verification failure

Date: 2026-07-20

Disposition: **retired — no preformal, binding, or formal authority**

## Terminal outcome

The sole WM-001 protocol 1.8.0 development producer completed under commit
`aa78239ab21a49eef8bec2f0d7bcf83a1d14be54` and was outer-finalized. Its
canonical independent development audit was accepted and outer-finalized. The
sole development-closure command also returned zero and published an
outer-finalized operator attempt with status `accepted`.

An independent fresh-process call to the public `verify_operator_attempt`
verifier then rejected that closure:

```text
RuntimeError: development result qualification does not prove exact
seeds/budgets/matrix
```

This is terminal failure evidence. Protocol 1.8.0 may not be repaired, retried,
resumed, upgraded, or completed. No v1.8 preformal report, formal binding,
formal producer, formal audit, semantic review, or adjudication package may be
created.

## Evidence identity

- protocol SHA-256:
  `3aa795e1a54b7cda04b94c77afc683f79639b8f9fffc3dae8be839d53b5d89bc`
- prospective-review SHA-256:
  `e79999a21312d0c227bd4fa5bc6c85bdb7a39fe7a376e63d7465f5d51b323c97`
- implementation-manifest SHA-256:
  `20c031bad88e3db1d006c2f1a1f4691ed45f007d785e204ae1fd3e2721f7f041`
- runtime-seal SHA-256:
  `ade182a27ae8e4c1eb656ba820fddd77ffe51545a236e7200589c1ff3bbe56b9`
- producer-manifest SHA-256:
  `1cfe4a00ebe36edd7856e86e62e32e510cde56775bae9fda0fbaa6fa30e79516`
- raw-result bytes:
  `320126662`
- raw-result SHA-256:
  `997f9122ac0210d75feddf414a9a9cbbb5451c08230ce31f3d5793c42f1070c2`
- accepted development-audit terminal SHA-256:
  `8ed8f69144cd12dfba078d9c66a061c0605f23f41fc808efd1c400ce826b65a7`
- qualification-archive bytes:
  `1118033920`
- qualification-archive SHA-256:
  `ee14079b9fcc52e2bafdfdd92dd5cdb4e0b77763c6a7c89fa9f4a9fbe7b16233`
- qualification-archive members:
  `86`
- closure-attempt terminal SHA-256:
  `dec96cb0f8ebfff8325826b9c895f519c5105b6b0a76c7aeb589d1373221e2f5`

The result and accepted development audit remain permanently claim-ineligible.
No K3–K6 value was opened, printed, summarized, compared, or used to select a
repair.

## Exact cause

`_development_matrix_contract_sha256()` built a canonical JSON object, but two
members of that object were populated by iterating `frozenset` values without
sorting:

```python
"predictive_contracts": [list(row) for row in PREDICTIVE_CONTRACTS],
"policy_contracts": [list(row) for row in EPISODE_CONTRACTS],
```

Python hash randomization changes those iteration orders across fresh
interpreters. The closure writer archived matrix-contract digest
`403f311db33d800abf2de8c8b78629e62897ea6e1e875674c3741210ea8b6052`.
Twelve independent interpreters subsequently produced twelve distinct digests
from the unchanged contract.

The writer failed to expose this prospectively because every closure
self-check ran in the same process as serialization. Set iteration remained
stable within that process, so the temporary marker, published marker, and
operator final checks all recomputed the same process-local digest. The first
fresh-process public verification correctly rejected the evidence.

All other result-qualification identity fields checked during diagnosis were
exact: schema, experiment, protocol and raw-result digests, lane,
claim-ineligibility, two master seeds, strict scalar types, and the declared
episode, transition, predictive, policy, update, and optimizer-manifest counts.

## Required fresh-version repair

A successor protocol must:

1. serialize every unordered contract collection in a declared total order;
2. bind one fixed expected matrix-contract digest;
3. test the digest across multiple fresh interpreters with distinct
   `PYTHONHASHSEED` values;
4. create a synthetic closure qualification in one process and verify it in
   another;
5. include that cross-process check in the result-free prospective rehearsal;
6. use fresh versioned paths, seeds, environments, schemas, seal, review, and
   binding; and
7. preserve the v1.8 producer, audit, archive, closure marker, and operator
   attempt unchanged as failure evidence.
