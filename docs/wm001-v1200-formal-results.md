# WM-001 protocol 1.20.0 formal results

Disposition: **accepted for the exact bounded WM-001 claim.**

WM-001 v1.20 completed its prospectively sealed eight-seed formal lifecycle.
The producer passed K0–K7. The bound independent auditor reopened the raw
artifact and passed 6,393,061 checks with zero failures and zero coverage gaps.
Three adversarial semantic referee passes independently accepted the narrow
claim with no fatal finding. The single-use adjudicator then reproduced the
audit byte for byte and published a terminal accepted package.

This is Prospect's first formally accepted demonstration that, on the sealed
two-regime Pendulum/oscillator fixture, one shared agent can collect identified
experience, learn a persistent probabilistic world model because of that
experience, improve held-out prediction and executed control, learn a
conflicting contextual task, retain task-A behavior through replay under
demonstrated naive interference, and reproduce the retained state and behavior
after a same-machine, same-dependency fresh-process restore.

It is not evidence that Prospect is already a mature or general agent.

## Evidence identity

- Protocol: `1.20.0`, SHA-256
  `e3ca74157cde18ca9d1011d14a61e2bd756505e825f88965cd6ec4ea4749ffe2`
- Bound implementation commit:
  `98acfdcc51e76decab76835dc709f6e7400d06d4`
- Bound Git tree: `5c3721ed0a677762d8291b3879fbabe8f7f6aeb0`
- Formal binding:
  `4e4a88bc7ffa4245d1031263ef9d1bd9096ce87565c21ec15abb3516e4a743d6`
- Formal launch:
  `2f19e073ea9bd63aaf134e49234b1d057c9e9571dde0d498a60316c408ea70d1`
- Raw result:
  `27834c95f37a74474e458300f399930e600b885a79759a9ab5031bb4c81f40c3`
- Producer manifest:
  `25c3dfd6ddb2cd8163425cdfe6e461526edb9ae7000b6e95b13b486c0b468e79`
- Formal audit claim:
  `95260ff76471b4c849a5f0495cc462c68a7177893c3c98d94d24be28f1b5494e`
- Audit-attempt manifest:
  `e08c0402ad34583627b85300f99fa984dd14f6eff78a29209c077f3022712e3e`
- Independent and reproduced audit:
  `b55cb47eea9d230d8efd5e6b082a3d2acfdec8ab2c3ff7d91e4b2d4a65bae192`
- Semantic review:
  `feeeee756294e2ae726b653a3f04483abd225efed9741f100498bb1dd95a9f4c`
- Adjudication claim:
  `d2744e93f8c1f1b33a46ba50ddc997525764eacea39f2e3c1474696782be63a1`
- Adjudication manifest:
  `636804e7e74043386dd5ee8480a1bc48949ab38b434cc26a3ef769e1b3a9416c`
- Adjudication outer completion:
  `d8ac19e9d30a4f5fa88a951ee9d32d53766c2bc95e039aa4e539f5c78a243af7.json`,
  the same inode and bytes as the adjudication manifest
- Auditor source:
  `fd722bc63fd483df7875bbf6b7ec77acba7e17310c748fa6fbbadc76b6bbf84e`

The formal producer contains 429 manifested files plus its manifest: 430 files
and 4,401,639,070 bytes total. The declared payload is 4,401,570,918 bytes. It
ran from `2026-07-22T23:42:20.983445Z` through
`2026-07-23T03:46:28.711266Z`. The fresh formal seeds were:

```text
3772418031, 1586188972, 155797552, 2704051827,
818738828, 4077496645, 1566512625, 2151461680
```

## Claim dispositions

| Claim | Disposition |
|---|---|
| Exact experience ancestry caused transactional shared-model changes used downstream | **Confirm, fixture-bound** |
| Held-out task-A prediction improved beyond frozen, target-corrupted, and separately learned nuisance controls | **Confirm, fixture-bound** |
| The learned model improved executed task-A control | **Confirm, fixture-bound** |
| One shared model learned task B and naive B-only learning interfered with task A | **Confirm, fixture-bound** |
| Balanced replay retained task-A behavior without a material task-B cost | **Confirm, fixture-bound** |
| Retained state and behavior survived a fresh process | **Confirm, same machine and dependencies** |
| Prospect is a mature or generally capable agent | **Unresolved; not tested** |
| WM-001 demonstrates precise 90% calibration | **Narrow; not established** |
| Uncertainty reduction drives planning or reward | **Retire for WM-001** |
| Generalization, multimodal learning, autonomous exploration, novelty, sample efficiency, state-of-the-art superiority, or cross-hardware reproducibility | **Unresolved; not tested** |

WM-001 evaluates predictive uncertainty, but its controller optimizes
deterministic ensemble-mean task reward. The experiment therefore does not
test the broader hypothesis that uncertainty reduction itself should determine
action value or reward.

## Independently recomputed effects

All intervals are paired across the predeclared master-seed unit (`n = 8`) and
use a two-sided 95% Student-t interval. An independent raw-row pass ignored the
stored aggregate metrics and gate booleans, reproduced all 25 aggregate means
and intervals exactly, and passed all 33 K3–K6 numeric predicates.

| Effect | Mean | 95% CI |
|---|---:|---:|
| Learned oscillator own-source NLL improvement vs cold | 3.3762 | [3.3172, 3.4351] |
| Task-A NLL improvement, after-A vs frozen | 3.6213 | [3.5577, 3.6850] |
| Task-A NLL improvement, after-A vs corrupted targets | 2.5023 | [2.3694, 2.6352] |
| Task-A NLL improvement, after-A vs learned oscillator | 3.8708 | [3.7740, 3.9676] |
| Task-A interval coverage | 0.98480 | [0.98031, 0.98930] |
| Task-A executed return improvement vs cold/frozen | 955.49 | [761.59, 1149.39] |
| Task-A executed return improvement vs learned oscillator | 939.03 | [716.85, 1161.21] |
| Task-A oracle-normalized score | 0.80336 | [0.59568, 1.01104] |
| Task-B replay NLL improvement vs before-B | 2.96679 | [2.15302, 3.78056] |
| Task-B replay return improvement vs before-B | 1321.98 | [1264.34, 1379.63] |
| Task-B naive return improvement vs before-B | 1321.99 | [1260.63, 1383.34] |
| Naive task-A forgetting after B | 1224.09 | [1045.60, 1402.58] |
| Retained task-A gain fraction | 1.27603 | [0.77413, 1.77793] |
| Replay task-A advantage over naive B learning | 1370.55 | [1296.45, 1444.65] |
| Replay-minus-naive task-B return | -0.0055 | [-9.0167, 9.0057] |

Every seed-level retention denominator is positive and every seed retained at
least the predeclared fraction. One seed has a retained fraction of about
`2.745` because its original after-A denominator is comparatively small; the
large absolute replay-versus-naive task-A effect independently confirms that
the retention pass is not an artifact of that ratio.

The learned oscillator control first learned its own source. On task A, its NLL
effect relative to frozen was `-0.24948` with CI
`[-0.31496, -0.18400]`, while its executed-return effect was only `+16.46`
with CI `[-79.42, 112.34]`. Correct task-A experience therefore beat both an
untrained model and a model that learned the wrong process.

## Exact coverage and restore

The pooled after-A coverage is:

```text
C / T = 50,422 / 51,200 = 0.9848046875
10C = 504,220 >= 7T = 358,400
100C = 5,042,200 <= 99T = 5,068,800
```

This passes the prospectively fixed `[0.70, 0.99]` integer gate. Coverage is
conservative relative to a nominal 90% interval, so WM-001 supports a broad
coverage check, not precise calibration.

All K7 component, lineage, prediction, action, and episode-return mismatches or
differences are exactly zero across all eight seeds after a same-machine,
same-dependency fresh-process restore.

## Audit and adjudication

The sole formal audit reports:

```text
passed checks:        6,393,061
failed checks:                0
coverage gaps:                0
integrity_passed:          true
engineering_complete:     true
complete_for_claim:       true
passed:                   true
```

Its descriptor-bound execution returned `0`, wrote empty stderr, and took
1,081,023,805,064 ns. The adjudicator's mandatory replay used the same bound
auditor source and runtime, returned `0`, wrote empty stderr, and took
1,153,999,216,409 ns. Supplied and reproduced reports are byte-identical at
SHA-256 `b55cb47e...e192`.

The terminal package has 31 manifest-listed files plus its manifest, 32 files
and 1,926,157 bytes total. It has disposition `accepted`, outcome kind
`audit_report`, `audit_byte_identical: true`,
`supplied_audit_clean_for_claim: true`, and `terminal: true`. Same-version
replay is forbidden.

## What was checked

The results audit inspected the sealed protocol, formal binding, launch,
producer manifest, all raw replicate records, predictive sidecars, checkpoint
and lineage evidence, audit attempt, runtime and invocation receipts,
independent audit, semantic review, adjudication replay, and terminal package.
It also:

- recomputed all 25 aggregate means and Student-t intervals from raw replicate
  rows and obtained exact equality;
- independently applied all 33 K3–K6 numeric predicates;
- checked exact pooled coverage as integer arithmetic;
- verified all K0–K7 gate booleans from recomputed values;
- checked source, result, manifest, claim, audit, review, and adjudication
  hashes;
- checked the required hard-link custody relationships; and
- verified the outer-finalized accepted package with the supported package
  verifier.

One first post-outcome package-verifier invocation overlapped the documentation
edits in this results record. Its live preformal identity check correctly
rejected the dirty Git tree. No experiment, audit, adjudication, or evidence
write was repeated. After the documentation was committed, the same read-only
supported verifier was rerun from the exact clean bound commit
`98acfdcc51e76decab76835dc709f6e7400d06d4`; it exited `0` and printed
`accepted`.

The final documentation correction changes the repository's current status
from “no v1.20 result exists” and “the complete bounded lifecycle remains
unestablished” to the accepted, fixture-bounded result recorded here. Frozen
plans, runbooks, protocols, prospective reviews, prior failure records, and
prior result records remain unchanged.

## Limitations and unopened questions

- `n = 8` measures seed variation on one fixed task family, not task-family
  generalization.
- Task B is an observed-context torque-sign reversal of Pendulum, not an
  unannounced new domain.
- Retention uses stored task-A experience and explicit balanced replay.
- Data collection is seeded uniform exploration, not uncertainty-directed
  active learning.
- The representation, ensemble, optimizer, replay policy, and CEM controller
  are fixed rather than autonomously selected.
- No pixel, language, audio, multimodal, long-horizon open-world, or real-world
  environment was opened.
- No external arena or published benchmark baseline was run.
- Exact persistence is limited to the same machine and dependency closure.
- The independent implementation still uses the same bound NumPy/Torch RNG
  primitives and device kernels for the exact replay portions.
- Evidence is not externally signed or transparency-log anchored. The trusted
  account, kernel, filesystem, and repository writers remain inside the threat
  model.

These unopened dimensions cannot be inferred from K0–K7 and remain unresolved.

## Recommended next evidence

Freeze WM-001 as the causal-mechanism foundation. The next experiment should
test a broader task distribution with fresh held-out task families and compare
the same agent against strong published baselines under matched environment,
data, compute, and planning budgets. It should add:

1. autonomous or uncertainty-directed experience selection against uniform
   collection;
2. calibrated uncertainty diagnostics with proper scores and reliability
   curves;
3. more than two overlapping tasks, adaptive memory/replay selection, and
   explicit capacity scaling;
4. transfer to unseen dynamics or observation regimes without task-specific
   model copies;
5. longer-horizon restart and recovery tests; and
6. at least one externally recognizable benchmark or hosted evaluation.

The next project question is therefore no longer whether Prospect can exhibit
one real collect → learn → improve → retain → persist chain. It is how broadly,
efficiently, and autonomously that mechanism scales.

## Local evidence paths

- Formal producer:
  `bench/world_model_lifecycle/results/formal/4e4a88bc7ffa4245d1031263ef9d1bd9096ce87565c21ec15abb3516e4a743d6/confirmation-v1.20.0`
- Formal audit:
  `bench/world_model_lifecycle/results/operator-v1.20/audits/formal-audit-v1.20.0`
- Semantic review:
  `artifacts/wm001-reviews/formal-v1.20.0.json`
- Accepted adjudication:
  `bench/world_model_lifecycle/results/adjudication-v1.20/formal-adjudication-v1.20.0`
