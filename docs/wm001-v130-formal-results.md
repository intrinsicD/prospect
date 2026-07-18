# WM-001 protocol 1.3.0 formal results

Disposition: **reject acceptance of this attempt; preserve strong,
fixture-specific mechanism evidence.**

WM-001 v1.3 ran end to end on eight prospectively fixed formal seeds. The
immutable producer result passed its K0–K7 numerical gates, including
held-out predictive learning, executed behavior, shared-model interference,
replay retention, and exact fresh-process persistence. The required pre-bound
independent audit nevertheless returned `passed: false` because of two defects
in the auditor that was sealed with the run. Under the protocol's zero-failure
rule, the complete lifecycle claim is therefore not established by this
attempt.

This is a rejection of the attempt's claim eligibility, not evidence that the
agent failed to learn and not a retroactive invalidation of the binding or
immutable producer evidence.

## Evidence identity

- Protocol: `1.3.0`, SHA-256
  `c15ec673be44caf2a2b30dc4b3ea9ee0e0e8108210cd6772567cd0cd1d9ff080`
- Implementation commit:
  `8cc5a8b734fb77afbd6ffc3e569906c37d4c17d9`
- Git tree: `e746cf43da1bb251836017f4a6c4555717987148`
- Formal binding:
  `fcb544699d62797999aa40aadcb94ff19af0acac39212f438302aad337a8709b`
- Raw result:
  `df10cdb74c3f9048070140a97aef3a9bbf404fa8cd30212ce8bc82cb72f6dc08`
- Producer manifest:
  `ada094e0f36b095be83fc13fb88669e49e8bec7bd2e253bb31ca59f48e9a495a`
- Independent audit:
  `9186f95e2c466aa2ba8af8ba8a7d767ec65f85a6393d2723c46a160000d03501`
- Semantic review:
  `9835b62b6c57b34ff9e18c7eb444961175dda5ea1c9041c20770ce6a00a50434`
- Rejected adjudication manifest:
  `25c454ddb1764825a932a70b91594ef606f65e6504c4db75f612be8e27a46c1a`
- Execution: CUDA, deterministic algorithms, Torch `2.9.0+cu128`,
  10,159.4 seconds
- Artifact: 4.1 GiB, 363 manifested files plus the manifest

Each replicate contains 496 episodes, 99,200 real transitions, 20 policy runs,
12 predictive evaluations, five committed optimizer phases, one rejected
zero-step probe, and all 15 required checkpoint components. The result verifier
accepted the result against the exact formal binding.

## Claim dispositions

| Claim | Disposition | Evidence |
|---|---|---|
| Binding, producer custody, budgets, and result envelope are internally consistent | **Confirm** | Clean bound commit; completed immutable manifest; binding-aware verifier passed |
| The declared real experience caused persistent shared-model byte changes used downstream | **Confirm** | K1–K2 ancestry, split isolation, update lineage, and rejected-probe atomicity reopened |
| The model learned held-out task-A dynamics from its own task-A experience | **Confirm, narrow** | Large NLL gains over frozen, corrupted-target, and learned-nuisance controls on the exact Pendulum fixture |
| That update improved executed task-A behavior | **Confirm, narrow** | Paired real-environment CEM return improved over cold/frozen and learned-nuisance controls |
| The same shared model learned task B and naive B learning interfered with task A | **Confirm, narrow** | Task-B predictive and return gains plus a large prespecified naive A-forgetting effect |
| Balanced replay retained task-A gain without unacceptable task-B loss | **Confirm, narrow** | Positive denominators for every seed; retained-fraction and replay-vs-naive gates passed |
| The retained state and behavior survived a fresh process | **Confirm** | All component, identity, prediction, action, and return differences were exactly zero |
| The auditor's seed-schedule finding is evidence against the run | **Retire** | The auditor alone contains `3280611227`; protocol, binding, verifier, analysis, and artifact contain sealed seed `3280610186` |
| The auditor's one-target coverage mismatch refutes K3 | **Retire** | It affects corrupted-control coverage, not the K3 after-A coverage row, and no gate decision changes |
| The complete collect → learn → improve → retain → persist claim is formally established | **Unresolved; reject for this attempt** | The mandatory, pre-bound independent audit did not pass |

“Narrow” means the evidence applies to this exact same-machine Pendulum and
independent-oscillator fixture, controller, budgets, and seed domain. It is not
a claim of broad agent maturity.

## Independently recomputed effects

All intervals are paired across the predeclared master-seed unit (`n = 8`) and
use a two-sided 95% Student-t interval with critical value
`2.3646242515927844`.

| Effect | Mean | 95% CI |
|---|---:|---:|
| Oscillator own-source NLL improvement, irrelevant vs cold | 3.3157 | [3.2501, 3.3813] |
| Task-A NLL improvement, after-A vs frozen | 3.5556 | [3.4764, 3.6349] |
| Task-A NLL improvement, after-A vs corrupted | 2.4416 | [2.3532, 2.5300] |
| Task-A NLL improvement, after-A vs learned nuisance | 3.8944 | [3.7105, 4.0782] |
| Task-A after-A 90% interval coverage | 0.9811 | [0.9733, 0.9888] |
| Task-A return improvement, after-A vs frozen | 863.12 | [643.14, 1083.11] |
| Task-A return improvement, after-A vs learned nuisance | 881.50 | [685.37, 1077.64] |
| Task-A oracle-normalized score | 0.7336 | [0.5051, 0.9620] |
| Task-B NLL improvement, replay vs before-B | 3.2165 | [2.8611, 3.5719] |
| Task-B return improvement, replay vs before-B | 1316.10 | [1215.88, 1416.32] |
| Naive task-A forgetting after B | 1182.20 | [954.25, 1410.15] |
| Retained task-A gain fraction | 1.3468 | [0.8979, 1.7958] |
| Replay task-A advantage over naive B learning | 1360.61 | [1269.99, 1451.22] |
| Replay-minus-naive task-B return | -9.94 | [-22.07, 2.20] |

Every seed-level retention denominator was positive (range `443.22` to
`1136.73`). The retained fractions ranged from `0.9127` to `2.2123`; values
above one mean the replay checkpoint exceeded that seed's original after-A gain.

The learned nuisance arm passed its own manipulation check, while its task-A NLL
change relative to frozen was negative: `-0.3387`
[`-0.5207`, `-0.1567`]. Its task-A return change relative to frozen was
`-18.38` [`-83.28`, `46.52`]. This supports specificity against this
prespecified learned nuisance source, not generic task relevance.

## Independent-audit disposition

The independent auditor completed 6,392,734 checks: 6,392,732 passed, two
failed, and no claim-coverage gaps.

### 1. Formal seed schedule

The bound auditor and a copied unit-test literal use `3_280_611_227` as the
second formal seed. The sealed protocol, formal binding, result verifier,
producer analysis, and immutable result all use `3_280_610_186`. This is an
auditor false negative. It reveals that duplicated constants were tested
against each other instead of against the sealed protocol.

### 2. Prediction coverage endpoint

Exactly one of 6,400 target coordinates differs:

- replicate `wm001-formal-3332986400`;
- task `pendulum_normal_torque`;
- checkpoint `corrupted`;
- sidecar row 207, target dimension 0;
- transition
  `wm001-formal-3332986400:validation-a:transition:4795`.

The bound CUDA/Torch float32 producer computes PIT
`0.05000000074505806` and includes the coordinate. NumPy/float64 computes
`0.049999998325471175` and excludes it; an 80-digit calculation from the
stored float32 tensors gives `0.049999998325471163…`. The stored coverage is
therefore `5838/6400 = 0.9121875`, while the independent arithmetic gives
`5837/6400 = 0.91203125`.

The protocol declares an inclusive `[0.05, 0.95]` interval but does not bind
the numerical semantics at its discontinuous endpoints. The auditor intended
to tolerate one target, but comparing the two fractions failed because their
binary-float difference exceeds `1/6400` by about `8e-17`. This is a
specification and comparator defect, not sidecar corruption. The affected row
is descriptive corrupted-control coverage; K3 uses the separately matched
after-A coverage aggregate. It cannot change any K0–K7 decision.

A post-hoc corrected audit is not independent confirmatory evidence. The
formal source snapshot binds the original auditor bytes, and the protocol
requires a clean pass from that pre-bound auditor. The immutable attempt must
not be repaired or relabeled.

## Commands and recomputation

The evidence was reopened with:

```bash
python -m bench.world_model_lifecycle.verify protocol
python -m bench.world_model_lifecycle.verify binding \
  artifacts/wm001-binding-20260718-v130-8cc5a8b/formal-binding.json
python -m bench.world_model_lifecycle.verify result \
  <producer-root>/result.json \
  --binding <producer-root>/formal-binding.json
python -m bench.world_model_lifecycle.artifact_audit \
  <producer-root> \
  --output artifacts/wm001-audits/formal-fcb54469-20260718-v130.json
sha256sum <producer-root>/result.json \
  <producer-root>/producer-manifest.json \
  artifacts/wm001-audits/formal-fcb54469-20260718-v130.json
```

A separate Python-standard-library recomputation opened the eight exact
per-replicate JSON files without importing producer `analysis.py`. For each
checkpoint it joined policy-run episode IDs to the 32 raw episode returns,
formed paired seed-level contrasts, and calculated
`mean ± t(0.975, 7) × sample_stdev / sqrt(8)`. It independently reproduced the
effects above, all positive retention denominators, and the per-seed retained
fractions. Separate NumPy, CUDA float32, and 80-digit replays isolated the sole
coverage-coordinate disagreement.

## Limits and unverified evidence

- The environment is a simple, low-dimensional, observed-context control
  fixture, not a broad, partially observed, multimodal, or real-world task.
- The uncertainty output is evaluated predictively; uncertainty reduction is
  not yet used as a calibrated intrinsic decision objective.
- The independent auditor reimplemented the algorithms, but some exact replay
  uses the same bound NumPy/Torch RNG and device primitives.
- The formal test report is content-addressed but has no machine-readable test
  result schema.
- The artifact is hash-bound but not externally signed or transparency-log
  anchored.
- Optimizer input IDs, target transformations, RNG manifests, budgets, and
  resulting bytes were reopened; every gradient operation was not independently
  retrained from scratch.
- No published baseline, external benchmark, architectural ablation, sample
  efficiency comparison, or broad generalization test is part of WM-001.

## Required next confirmation

Before another formal run:

1. source the formal seed schedule from the sealed protocol and test parity
   across protocol, binding, verifier, producer, and auditor;
2. compare coverage in integer target-count space and bind exact PIT endpoint
   arithmetic plus an explicit numerical guard;
3. adversarially test both fixes before any outcome is observed;
4. issue a new protocol version, fresh seed domain, clean implementation
   binding, and immutable attempt; and
5. treat any same-seed or corrected-auditor replay of this artifact as
   diagnostic only.

One-sentence verdict: **the immutable producer evidence is strongly consistent
with Prospect collecting experience, learning from it, improving executed
behavior, and retaining the gain after interference and restart, but WM-001
v1.3 did not formally establish that claim because its mandatory pre-bound
independent auditor did not pass.**
