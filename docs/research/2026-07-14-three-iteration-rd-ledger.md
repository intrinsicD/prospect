# Three-iteration evidence-driven R&D ledger

**Date:** 2026-07-14  
**Reusable prompt:** `docs/research/2026-07-14-evidence-driven-rd-loop-prompt.md`  
**Scope:** non-gated BridgeControl research; no production, task, ADR, gate, or shipped
result change

## Outcome first

The three-iteration loop falsified **simple proposal scarcity** as the primary cause of
the frozen learned controller's BridgeControl failure. Injecting simulator-optimized
sequences into the unchanged learned iCEM budget raised mean return from `-2.660520` to
`-1.368870`, but recovered only `16.28%` of the exact-controller gap, improved only
`5/8` model seeds, and reached `9.38%` success rather than the preregistered `80%`.
The action-permuted control also reached `9.38%` success. The frozen branch rule
therefore prohibited enlarged search.

The preregistered aggregate audit classified the failure as
`no_privileged_rescue:trigger_not_statewise`: only `9.60%` of 448 visited MPC calls
placed any injected reference in the first-round learned top elite, and none remained
the final best sequence. A labeled post-hoc stratification sharpened, but did not
change, that decision: transfer was `100%` at step 0, `34.375%` at step 1, and `0%`
from step 2 onward; even at step 0, learned-score refinement replaced the injected
first-round best in every call.

## Iteration 1 — reproduce before extending

**Question.** Does the sealed OL-002 evidence really satisfy its predeclared trigger
for a privileged-candidate injection test, and do fixed-bank metrics agree with
closed-loop control?

**Actions.**

- Ran the full OL-002 semantic verifier under the pinned `.venv` NumPy `2.4.6`
  environment: `verified_results`.
- Added deterministic trigger analysis in
  `bench/proposal_injection/trigger.py` and saved
  `bench/proposal_injection/results/PI-001-trigger.json`.
- Recomputed all quantities from the sealed JSON, treating OL-001/OL-002 as one
  scientific experiment.

**Evidence.** All eight exact-reference sequences ranked in the learned top eight on
all `32/32` seed/start fixed-bank blocks, an exact reference was selected on all
`32/32`, and native zero-penalty learned control succeeded on `6.25%`. The trigger
reproduced. However, `k=12` exact-transition scoring had better Pearson, Spearman, and
selected regret than `k=8`, while `k=8` had higher closed-loop success (`96.875%`
versus `84.375%`). Common-bank fidelity and closed-loop control pointed in opposite
directions.

**Decision/prompt refinement.** Freeze the smallest causal search test: replace eight
first-round native proposals with current-state exact references while preserving all
learned candidate evaluations and RNG draws. Include an action-coordinate-permuted
negative control. Do not enlarge search unless injection specifically rescues.

## Iteration 2 — causal proposal-injection test

**Question.** If high-exact-return candidates are physically present in the unchanged
learned scorer's first iCEM round, does closed-loop control recover?

**Actions.**

- Added a bench-only `ProposalInjectionPlanner` with exact zero-injection parity,
  fixed learned-budget accounting, provenance diagnostics, and exact-reference
  providers. `src/prospect/` remained unchanged.
- Froze PI-001 before formal outcomes with seeds `0..7`, four starts, 14 steps,
  native `64×3×12` learned scoring, eight first-round replacements, exact ceiling,
  action-permuted control, and rescue thresholds of `7/8`, `50%`, and `80%`.
- Preserved two administrative verifier failures without accepting their numbers:
  PI-001 report mapping order and PI-002 tuple/list semantic comparison. PI-003
  inherited identical scientific fields, reran all training/evaluation, and passed
  canonical persisted-package plus full semantic regeneration.

**Accepted PI-003 evidence.**

| Arm | Mean return | Success | Positive seeds vs native | Exact-gap closure |
|---|---:|---:|---:|---:|
| native zero-penalty | -2.660520 | 6.25% | — | — |
| privileged injection | -1.368870 | 9.38% | 5/8 | 16.28% |
| action-permuted injection | -1.991250 | 9.38% | 5/8 | 8.44% |
| exact raw | 5.273024 | 100.00% | — | 100.00% |

**Decision/prompt refinement.** Privileged injection did not rescue and the negative
control was not cleanly separated in success. Abandon simple proposal scarcity as the
primary explanation on this fixture. Per the frozen rule, do not run enlarged search;
route iteration 3 to the action-commitment audit.

## Iteration 3 — localize why injection failed

**Question.** Did the fixed-bank trigger fail to transfer to visited MPC states, did
iCEM refinement discard injected references, or did selected references still fail in
closed loop?

**Preregistered evidence.** Across 448 privileged-injection calls:

- at least one injected reference entered the first-round top eight on `9.598%`;
- an injected reference was first-round best on `7.366%`;
- an injected reference was final best on `0%`;
- success conditional on any top-elite hit was `9.302%`.

This met the frozen `trigger_not_statewise` classification rather than the 50%
statewise-transfer threshold.

**Exploratory post-hoc evidence.** The deterministic step audit in
`bench/proposal_injection_v3/results/PI-003-posthoc-step-audit.json` found:

| Real episode step | Calls | Any injected top-elite | Injected first-round best | Injected final best |
|---:|---:|---:|---:|---:|
| 0 | 32 | 100% | 100% | 0% |
| 1 | 32 | 34.375% | 3.125% | 0% |
| 2–13 (each) | 32 | 0% | 0% | 0% |

This does not amend the preregistered result. It suggests two separable mechanisms:
learned-score refinement displaces strong exact candidates before the first real
action, and the reference ranking ceases to transfer after the controller changes
state.

**Next prompt refinement.** The cheapest next experiment is an iteration-wise
learned-versus-exact candidate-landscape audit, not more search compute. At frozen
initial, step-1, and step-2 visited states, retain every candidate from each iCEM round
and score the identical sequences with the learned model and exact simulator. Kill the
model-exploitation hypothesis if learned-score ascent does not systematically reduce
exact score/rank relative to the injected first-round reference; otherwise require the
direction in at least `7/8` model seeds before considering a planner-side mitigation.

## Verification and provenance

- Accepted scientific package:
  `bench/proposal_injection_v3/results/PI-003/PI-003-results.json`
  (`bea3a1ad850099b97628e313d1b1a2a889d54912aee7ada0bfab82343f743251`).
- PI-003 report:
  `c5a3b8aaa6e8fe4a02ca0cef79075fa8c5b3358571e8bb30273b733d393af08b`.
- Frozen dataset:
  `9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948`.
- Final semantic command:
  `.venv/bin/python -m bench.proposal_injection_v3 verify-semantic` →
  `verified_semantic_results`.
- PI-001 and PI-002 are administrative predecessors of PI-003, not independent
  evidence. Their terminal failure records and hashes remain preserved.

## Limitations

- The experiment is an authored BridgeControl diagnostic, not DMC evidence or a
  production capability claim.
- Simulator references are privileged diagnostic inputs whose generation cost is
  excluded from the learned planner budget and reported separately.
- Four starts are repeated measures inside eight model seeds.
- The step-stratified localization was chosen after the aggregate PI-003 decision and
  remains exploratory until preregistered in a new experiment.
