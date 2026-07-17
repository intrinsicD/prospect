# PI-001 terminal verifier failure

**Status:** preserved failed formal artifact  
**Date:** 2026-07-14

PI-001 completed its formal model training, primary-arm evaluation, conditional branch
selection, JSON/CSV/Markdown rendering, and artifact-manifest write. It then failed the
final canonical Markdown comparison in `_verify_outcomes`.

The defect is deterministic mapping-order dependence in report rendering, not a
numeric mismatch:

- The in-memory decision inserted rescue records as `privileged_injection` followed by
  `action_permuted_injection`, and the saved report used that order.
- `_write_json(..., sort_keys=True)` serialized the same mapping alphabetically.
- The verifier read the JSON and regenerated the report as
  `action_permuted_injection` followed by `privileged_injection`.
- Saved and regenerated reports are both 1,049 decoded characters and differ first at
  character 522, where only those two bullet records exchange order.
- CSV, JSON, protocol, input, dataset, and artifact hashes were written, but the frozen
  PI-001 rule accepts no scientific conclusion after any final verifier failure.

Frozen failed-artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `protocol.json` | `f2a8281175c6b083fc5b685359967c7e8297d46fe1f9404de6f6b84cc498c006` |
| `input-manifest.json` | `e8fa658fc9a6cc0e05a0ece9059a5635ca2ba828177669014dc69d2275fb443f` |
| `PI-001-results.json` | `92f4035c9f0247e10288ad0f412aeb1eb59922f835f7eaea027eefca88cfc409` |
| `PI-001-runs.csv` | `bb003b16ce91b918561381982ebd4e0c6eedae5c05ef3de59318a7bf8fd63c5e` |
| `PI-001-report.md` | `f622c0faf8d29e2a81088a791a4349e2ae5e9f1f1f26f41aedac5a75bf900c54` |
| `inputs/BC-001-b1_r1_d8.npz` | `9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948` |
| `artifact-manifest.json` | `f6d95ad83aad7d94b9c71665808b41ccc38c75487a104fbf61f077fd7f15842c` |

PI-001 remains untouched. Any administrative rerun must use a new experiment/schema,
bind every failed artifact and the exact PI-001 source snapshot, train/evaluate all
formal seeds again, and make the report order canonical both before and after JSON
serialization. Its numbers may not be copied into or counted independently from the
rerun.
