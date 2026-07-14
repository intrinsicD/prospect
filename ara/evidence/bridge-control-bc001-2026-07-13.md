# BC-001 BridgeControl stopped control screen

**Date:** 2026-07-13  
**Scope:** non-gated research fixture; no production or shipped-claim change  
**Disposition:** `aborted_invalid_fixture`

## Bound artifacts

- Protocol: `docs/research/2026-07-13-bridge-control-protocol.md`
- Machine result: `bench/bridge_control/results/BC-001/BC-001-results.json`
- Human report: `bench/bridge_control/results/BC-001/BC-001-report.md`
- Artifact manifest: `bench/bridge_control/results/BC-001/artifact-manifest.json`
- Canonical protocol-record SHA-256:
  `c9f454674d5510552593594583554ca12ff3abac5ff7b2027102f80d8a824656`
- Dataset-manifest SHA-256:
  `30b01e4b58a202ed6e1b749636fc7a95c0295ecabfb8dd8b87d9476846590f3c`
- Result-file SHA-256:
  `3b1a6b092929bf27eb5fb9a46c32fdb358f02e526a23ddd25d55014c99f7e30f`
- Artifact-manifest SHA-256:
  `555a61a189a4e696ca68eb637aed2db2c79d8b4961fa17b36ea8e038b2e7faec`

The result records Git HEAD `2126fdf2529c475020082b9147e078f3730759ef`,
`dirty=true`, and explicit SHA-256 hashes for every BridgeControl source, the relevant
production model/planner interfaces, protocol, parent research artifacts, and tests.

## Manipulation screen

All eight primary cells contain 896 transitions and cover the same 16 x-region-by-lane
nodes. They have identical x-region-by-lane counts, nuisance multisets, and
coordinate-wise action histograms. The intended factors separate as follows:

| Factor diagnostic | Low | High |
|---|---:|---:|
| Directed door crossings | 0 | 16 |
| Local action minimum singular value | 0.00000 | 0.67082 |
| Unique controllable support per stabilization cell | 1 | 8 |

Ten deterministic datasets were prepared: eight primary cells, an action-permuted
control, and a constant-nuisance control. The latter two were hashed but not trained
because the learned positive control failed.

## Sequential control result

Each arm used eight formal learner/planner blocks and four fixed starts per block.
Development seed 97 was excluded.

| Arm | Mean return | Successes | Success rate | Role |
|---|---:|---:|---:|---|
| Exact dynamics + exact reward | 5.2730 | 32/32 | 1.0000 | positive control, pass |
| Balanced learned B1/Rfull/D8 | -2.7701 | 2/32 | 0.0625 | positive control, fail |
| Random policy | -20.9009 | 0/32 | 0.0000 | floor |
| Exact transition + learned reward + zero epistemic | 3.4387 | 27/32 | 0.84375 | joint diagnostic only |

The balanced learned arm missed the frozen 80% success floor after the one permitted
fixture redesign. The stop rule therefore prohibited training or interpreting the
seven other factorial cells, the two named negative controls, or a second topology.
`factorial_effects` is null. No bridge, rank, density, interaction, interval, or novelty
claim is supported.

The joint diagnostic changes transition, representation, and epistemic handling at
once. Its rescue establishes only that the learned reward head is not the sole blocker;
it does not identify which changed component caused the rescue.

## Verification

- `python -m bench.bridge_control verify` returned `verified_results`.
- Full unit suite: 155 passed, 1 skipped.
- Ruff: pass.
- mypy: pass over 77 source files.
- Isolated P0-P14 regression ratchet: all 15 reports passed; reports were written under
  `/tmp` so committed gate evidence was not changed.
- Portfolio structural validator: pass; this does not establish novelty.

The verifier binds current source and protocol hashes, regenerates all deterministic
datasets and diagnostics, recomputes success labels and aggregates, enforces the stop
branch, checks the canonical CSV, rerenders Markdown/SVG artifacts, and verifies the
artifact manifest.
