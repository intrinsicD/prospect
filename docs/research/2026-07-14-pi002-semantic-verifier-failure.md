# PI-002 terminal semantic-verifier failure

**Status:** preserved failed semantic-verification artifact  
**Date:** 2026-07-14

PI-002 completed its fresh formal run and passed protocol, input, parent, CSV, report,
decision-recomputation, and artifact-manifest verification. Its separately invoked
full semantic regeneration then retrained/evaluated all eight seeds and failed when
comparing the regenerated in-memory `rows` object with JSON-decoded saved rows.

The failure is a canonical container-type defect, not a numeric or decision mismatch:

- `ReferenceCall.as_dict()` preserves `reference_exact_scores` and
  `output_exact_scores` as Python tuples in regenerated provider diagnostics.
- JSON serializes those tuples as arrays and decodes them as Python lists.
- The semantic verifier compared raw Python objects rather than their canonical JSON
  representations, so `tuple != list` despite identical ordered numbers.
- A second complete regeneration compared both forms. Raw equality was false only for
  `rows`; canonical JSON-form equality was true for `rows`, `parity`, `decision`, and
  `executed_arms`. The first mismatch was the provider record for
  `privileged_injection`, seed 0; its provider record became equal after JSON
  canonicalization.

Under the PI-002 one-shot rule, the package remains unaccepted despite the confirmed
representation-only cause.

Frozen failed-artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `protocol.json` | `ed50d151a8bb6d7c485528bc53eaa64b8e26cb8a111003a4ef6019c1ca233da3` |
| `input-manifest.json` | `b692c700443958c5037711cf99601bdf92f491dcc5585ba151661117bba8a0e9` |
| `PI-002-results.json` | `6afcfe0da6d954b1cf1399bf66b99c5c084701a73188e1f25f705cfca4f0b4ad` |
| `PI-002-runs.csv` | `bb003b16ce91b918561381982ebd4e0c6eedae5c05ef3de59318a7bf8fd63c5e` |
| `PI-002-report.md` | `c462cc64e7bc02812be352a2f3d76bab237078ca54a88d7d39710eb2cdd0965f` |
| `inputs/BC-001-b1_r1_d8.npz` | `9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948` |
| `artifact-manifest.json` | `8a08f97da5c100191fcac1348d06fd1dbcbb5a615bb89672052583a71408fd3d` |

PI-002 remains untouched. A new administrative rerun must preserve the canonical
report-order fix and compare semantic regeneration only after normalizing both sides
through the artifact's canonical JSON representation. It must rerun all formal models
and outcomes; PI-001/PI-002 numbers cannot be copied or double-counted.
