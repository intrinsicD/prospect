# WM-001 protocol 1.4.0 formal results

Disposition: **reject formal acceptance; preserve strong fixture-specific
mechanism evidence and the adjudication-harness failure.**

WM-001 v1.4 ran end to end on its eight prospectively fixed fresh formal seeds.
The immutable producer result passed K0–K7. A direct run of the corrected,
pre-bound independent auditor reopened the raw package and passed 6,393,031
checks with zero failures and zero coverage gaps. One content-addressed
semantic review incorporating three adversarial referee passes then accepted
the protocol's narrow two-regime Pendulum claim.

The mandatory adjudication-time audit reproduction did not match that supplied
audit. Adjudication executes the captured auditor as `python -I -B` through an
inherited private file descriptor. On this machine, `-I` removes the user-site
locations from which two bound distributions had been resolved at binding and
direct-audit time. In particular, `farama-notifications` and `gymnasium` were
bound from `~/.local`, but they are not visible to isolated Python. The
reproduced audit therefore rejected the dependency closure, substituted an
invalid formal device, and returned 289 failures plus one CEM-replay coverage
gap.

The accepted package was refused before publication. A second attempt to
create a rejected package from the exactly preserved failing replay was also
refused because the adjudicator requires successful coverage conformance before
it will package any formal audit, including a rejected one. Neither an accepted
nor an official rejected adjudication package exists.

Under the sole-attempt rule, v1.4 is retired. The failure is in the
adjudication environment and packaging path, not evidence that the producer
failed to learn. It nevertheless prevents the complete claim from becoming
formally accepted.

## Evidence identity

- Protocol: `1.4.0`, SHA-256
  `17d67ea4a73f2c8d6e902cbaceadd315bf04b0addefa80e731a5eecfdd46c5e5`
- Implementation commit:
  `fb274c273857f7986668f86624374ebe28777549`
- Git tree: `b262388156991d63682fa3ddcd1e4ec68b328efc`
- Formal binding:
  `96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857`
- Formal launch:
  `e8cd636fa734c95183a2f6a6eea7cadc98886867f80ca38e1b65e4b096ec464a`
- Raw result:
  `bd759ac621494a732ead40b770c66c725cb84e1597085ec674cb747e2891bab0`
- Producer manifest:
  `af0a7702708b64fe95ecb2e888d39c9262971b921829b8a6f3a680b92008299a`
- Direct corrected audit:
  `e1b00f03afaab896db01da5de0991ff32ba46ecdfe909853d93b6a1b0bc9af28`
- Pre-adjudication accepted semantic review:
  `64427141d825615a183e311ca01dd19608bced63d70383381e02914923e20f87`
- Descriptor-bound failing audit replay:
  `11ac22d15db89e43b002ab05223ead05843a4abd55de45fe8777ab702f7dd226`
- Superseding rejected semantic review:
  `0ff74453796b2fe447ecf43554152c5ea026215690fa31d39f36da58643112c8`
- Corrected auditor source:
  `9cc6fe45e643981512e64f46ee5f398d637ba2fc22103f322c99498140f52e85`
- Adjudicator source:
  `a9de42a5922b8b94a3bc85b7bd757a877a25e1941d82f55d8ea9480a28fbc151`
- Execution: CUDA, deterministic algorithms, Torch `2.9.0+cu128`,
  14,786.1 seconds
- Producer artifact: 4,395,257,936 bytes (4.09 GiB), 365 manifested files
  plus the manifest

## Local evidence paths

- Formal binding:
  `artifacts/wm001-binding-20260718-v140-confirmation/formal-binding.json`
- Immutable formal producer root:
  `bench/world_model_lifecycle/results/formal/96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857/20260718-v140-confirmation`
- Direct corrected audit:
  `artifacts/wm001-audits/20260718-v140-formal-confirmation.json`
- Accepted pre-adjudication semantic review:
  `artifacts/wm001-audits/20260718-v140-formal-semantic-review.json`
- Descriptor-bound diagnostic replay:
  `artifacts/wm001-audits/20260719-v140-adjudication-replay-diagnostic-1.json`
- Superseding rejected semantic review:
  `artifacts/wm001-audits/20260719-v140-formal-semantic-review-rejected.json`
- Intended accepted package, confirmed absent:
  `artifacts/wm001-adjudications/formal-96628f7d-20260719-v140-accepted`
- Intended rejected package, confirmed absent:
  `artifacts/wm001-adjudications/formal-96628f7d-20260719-v140-rejected`

Each replicate contains 496 real 200-step episodes, 99,200 real transitions,
20 policy runs, 12 predictive evaluations, five committed 2,000-step optimizer
updates, one rejected zero-step probe, and all 15 required checkpoint
components. Producer custody and the binding-aware result verifier both pass.

The claim-ineligible full-budget development rehearsal also completed before
binding. Its qualifying second attempt passed 1,598,247 corrected-auditor
checks with zero failures and zero gaps. The preserved first attempt failed
producer finalization only because the raw-result schema omitted two sealed
exact-comparator labels; its independent audit found no second defect.

## Claim dispositions

| Claim | Disposition | Evidence |
|---|---|---|
| Binding, producer custody, exact fresh seed schedule, budgets, and result envelope are internally consistent | **Confirm** | Clean bound commit; sole launch; completed 365-file manifest; binding-aware verifier passed |
| Identified task-A experience caused persistent shared-model byte changes used downstream | **Confirm** | K1–K2 lineage, split isolation, transactional ancestry, and rejected-probe atomicity independently reopened |
| The model learned held-out task-A prediction | **Confirm, narrow** | Large NLL gains over frozen, joint-target-permuted, and verified learned-oscillator controls |
| Exact v1.4 coverage arithmetic is sealed and reproduced by the direct bound-runtime audit | **Confirm, narrow** | All 96 prediction sidecars matched; pooled `49,949/51,200` passes both exact integer predicates |
| The learned model improved executed task-A behavior | **Confirm, narrow** | Paired 200-step CEM return improved beyond cold/frozen and learned-oscillator controls |
| One shared model learned task B and naive B-only learning interfered with task A | **Confirm, narrow** | Task-B predictive/return gains, shared-parameter checks, and large prespecified naive forgetting |
| Balanced replay retained task-A gain without unacceptable task-B loss | **Confirm, narrow** | Positive denominators for all seeds; retained-fraction and replay-vs-naive gates passed |
| The retained state and continuation survived a fresh process | **Confirm, narrow** | Same-machine, same-dependency 15-component identity and behavior differences are exactly zero |
| v1.4 formally establishes the complete collect → learn → improve → retain → persist claim | **Retire for v1.4** | Mandatory adjudication-time audit reproduction failed; accepted package was refused |
| An official rejected v1.4 adjudication package was published | **Retire; false for v1.4** | Rejected packaging exited `2`; the intended output is absent |
| Prospect is a mature or general learning agent | **Unresolved** | WM-001 is one low-dimensional, observed-context fixture |
| WM-001 demonstrates precise 90% calibration | **Narrow** | It passes the predeclared broad pooled `[0.70, 0.99]` gate; some seed-level fractions exceed 0.99 |
| Uncertainty reduction drives action selection or reward | **Retire for WM-001** | Uncertainty is evaluated predictively; the planner uses deterministic ensemble-mean reward |
| Generalization, multimodal learning, novelty, sample efficiency, or published-benchmark superiority | **Unresolved** | Not tested |

“Narrow” means the evidence applies to this exact same-machine,
same-dependency Pendulum/oscillator fixture, model, controller, budgets,
controls, and seed domain. It is not a broad agent-capability claim.

## Independently recomputed effects

All intervals are paired across the predeclared master-seed unit (`n = 8`) and
use a two-sided 95% Student-t interval with critical value
`2.3646242515927844`.

| Effect | Mean | 95% CI |
|---|---:|---:|
| Oscillator own-source NLL improvement, learned vs cold | 3.3554 | [3.2798, 3.4309] |
| Task-A NLL improvement, after-A vs frozen | 3.5388 | [3.4511, 3.6266] |
| Task-A NLL improvement, after-A vs corrupted | 2.4672 | [2.3455, 2.5888] |
| Task-A NLL improvement, after-A vs learned oscillator | 3.8570 | [3.6462, 4.0678] |
| Task-A after-A 90% interval coverage | 0.9756 | [0.9595, 0.9917] |
| Task-A return improvement, after-A vs cold/frozen | 893.60 | [667.17, 1120.03] |
| Task-A return improvement, after-A vs learned oscillator | 961.66 | [780.23, 1143.10] |
| Task-A oracle-normalized score | 0.7568 | [0.5382, 0.9753] |
| Task-B NLL improvement, replay vs before-B | 3.6128 | [2.1105, 5.1150] |
| Task-B return improvement, replay vs before-B | 1350.84 | [1287.54, 1414.14] |
| Naive task-A forgetting after B | 1189.72 | [946.69, 1432.75] |
| Retained task-A gain fraction | 1.3501 | [0.8362, 1.8640] |
| Replay task-A advantage over naive B learning | 1368.68 | [1283.10, 1454.25] |
| Replay-minus-naive task-B return | -2.33 | [-9.33, 4.67] |

Every seed-level retention denominator is positive. Values of retained fraction
above one mean that the replay checkpoint exceeded that seed's original
after-A gain.

The learned oscillator arm first passes its own held-out manipulation check.
On task A, its NLL change relative to frozen is negative: `-0.3182`
[`-0.4898`, `-0.1466`]. Its task-A return change relative to frozen is
`-68.06` [`-152.45`, `16.32`]. This supports specificity against this one
prespecified learned nuisance source, not generic task relevance.

## Exact coverage result

The eight after-A covered-target counts are:

```text
6219, 6057, 6326, 6323, 6083, 6343, 6209, 6389
```

They sum to:

```text
C / T = 49,949 / 51,200 = 0.97556640625
10C = 499,490 >= 7T = 358,400
100C = 4,994,900 <= 99T = 5,068,800
```

The corrected auditor independently regenerated all 96 predictive sidecars
from their stored float32 tensors using the bound scalar-binary64 PIT rule.
Every stored covered count, target count, and derived fraction matched exactly.
Some individual seed fractions exceed 0.99, but the frozen K3 rule explicitly
uses pooled integer counts.

## Audit and adjudication disposition

The direct corrected audit reports:

```text
passed checks:        6,393,031
failed checks:                0
coverage gaps:                0
integrity_passed:          true
engineering_complete:     true
complete_for_claim:       true
passed:                   true
```

A separate raw-row recomputation ignored stored aggregates and gates and
reproduced both exactly. One content-addressed semantic review synthesized three
independent referee passes. Its quantitative pass reopened all eight raw
replicate records and all 96 prediction tensor sidecars and found no metric,
interval, count, threshold, control, or gate discrepancy. The other two passes
independently accepted the bounded causal and custody claims.

Acceptance still requires the adjudicator to reproduce the supplied audit
bytes. That step failed:

```text
adjudication package refused:
supplied independent audit report does not exactly match a fresh canonical
run of the pre-bound auditor
```

A preserved descriptor-bound diagnostic replay shows the cause:

```text
normal Python:   ENABLE_USER_SITE=true
isolated Python: ENABLE_USER_SITE=false

normal mode:
  farama-notifications 0.0.6  visible
  gymnasium 0.29.1            visible

python -I mode:
  farama-notifications        PackageNotFoundError
  gymnasium                   PackageNotFoundError
```

The binding correctly includes those distributions, but they are installed
under the user-site root. The isolated auditor therefore reports:

| Diagnostic finding | Count |
|---|---:|
| Bound input package invalid | 1 |
| Prediction evidence unavailable after invalid-device substitution | 96 |
| CEM action replay unavailable | 96 |
| Dependent CEM RNG replay mismatch | 96 |
| **Total failures** | **289** |

It also reports one blocker gap,
`cem_action_trace_replay_incomplete`, and returns:

```text
passed checks:        6,391,962
failed checks:              289
coverage gaps:                1
integrity_passed:         false
engineering_complete:    false
complete_for_claim:      false
passed:                  false
```

The auditor source still matches the pre-outcome binding exactly. The producer
manifest, result, binding, and sidecars remain unchanged and valid. The
difference comes from the adjudicator changing package visibility relative to
the runtime observed at binding and direct-audit time.

The rejected-package command then failed separately:

```text
adjudication package refused:
independent audit source, tests, or coverage conformance do not match the
pre-outcome binding
```

This occurs because the adjudicator requires
`coverage_conformance_verified=true` before accepting any formal audit identity,
even when the requested disposition is rejected. Both intended output
directories remain absent.

## Commands and outcomes

The decisive verification surfaces were:

```bash
make check-runtime
python -m bench.world_model_lifecycle.verify binding \
  artifacts/wm001-binding-20260718-v140-confirmation/formal-binding.json
python -m bench.world_model_lifecycle.verify result \
  bench/world_model_lifecycle/results/formal/96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857/20260718-v140-confirmation \
  --binding artifacts/wm001-binding-20260718-v140-confirmation/formal-binding.json
python -m bench.world_model_lifecycle.artifact_audit \
  bench/world_model_lifecycle/results/formal/96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857/20260718-v140-confirmation \
  --output artifacts/wm001-audits/20260718-v140-formal-confirmation.json
python -m bench.world_model_lifecycle.adjudication \
  --producer bench/world_model_lifecycle/results/formal/96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857/20260718-v140-confirmation \
  --audit artifacts/wm001-audits/20260718-v140-formal-confirmation.json \
  --semantic-review artifacts/wm001-audits/20260718-v140-formal-semantic-review.json \
  --output artifacts/wm001-adjudications/formal-96628f7d-20260719-v140-accepted \
  --disposition accepted
```

The fresh prebinding run passed lint, 33 normal mypy targets, two scoped
auditor/adjudicator mypy targets, 85 epistemic tests, and 198 world-model tests.
The binding and result verifiers passed. The direct audit passed. The accepted
adjudication command exited `2` with the byte-mismatch refusal above.

One diagnostic execution of the same private descriptor-bound audit was then
preserved and diffed against the direct audit. A rejected semantic review bound
that failing report. The rejected adjudication invocation was:

```bash
python -m bench.world_model_lifecycle.adjudication \
  --producer bench/world_model_lifecycle/results/formal/96628f7d551f14f50108a51ee454fe725e7e439501ca5ffe081cf891c0f17857/20260718-v140-confirmation \
  --audit artifacts/wm001-audits/20260719-v140-adjudication-replay-diagnostic-1.json \
  --semantic-review artifacts/wm001-audits/20260719-v140-formal-semantic-review-rejected.json \
  --output artifacts/wm001-adjudications/formal-96628f7d-20260719-v140-rejected \
  --disposition rejected
```

It also exited `2`. No accepted or rejected package was published.

## Corrections made

- No producer, binding, audit, threshold, seed, control, model, optimizer,
  controller, budget, or formal result byte was changed.
- The pre-adjudication accepted review remains preserved as evidence of what
  the referees concluded before the mandatory replay.
- A superseding rejected semantic review records the fatal reproduction
  mismatch.
- Repository status and result documentation are narrowed to the last valid
  evidence rung.

## Limits, unopened evidence, and follow-up

- Development K3–K6 outcomes were not used to decide binding or formal launch.
- The experiment is same-machine and same-dependency; it is not an
  independent-lab replication.
- The auditor regenerates model predictions and controller traces, but it does
  not independently retrain every AdamW gradient operation.
- Hash-bound local evidence is not externally signed or transparency-log
  anchored.
- The formal test report is content-addressed text rather than a
  machine-verifiable test-result schema.
- No published baseline, external benchmark, architectural ablation, sample
  efficiency comparison, or broad generalization test is part of WM-001.

A future protocol version must:

1. reproduce the exact adjudication-time auditor command before formal launch,
   using a no-outcome conformance artifact;
2. preserve the binding's dependency visibility under isolation, either by
   using a fully self-contained environment or by explicitly restoring only
   the bound distribution roots before descriptor execution;
3. test a transitive dependency installed exclusively in user site;
4. require the prebinding gate to compare direct and descriptor-bound canonical
   auditor outputs byte-for-byte;
5. allow a failing but identity-valid formal audit to enter a rejected package;
6. use a new protocol version and fresh seed domain; and
7. keep external benchmark and broader-task maturity work separate from this
   fixture confirmation.

One-sentence verdict: **WM-001 v1.4 produced strong, independently checked
fixture-specific evidence that Prospect collected identified experience,
updated one shared world model, improved held-out prediction and executed
behavior, retained that gain through demonstrated conflicting learning, and
reproduced it after fresh-process restart, but it did not formally establish
the complete claim because the mandatory adjudication-time audit could not
reproduce the passing audit in the sealed dependency environment.**
