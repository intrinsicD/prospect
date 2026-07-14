# OL-001: Simulator-oracle localization on BridgeControl

**Status:** frozen before formal seeds 0--7  
**Date:** 2026-07-14  
**Scope:** non-gated research evidence; no production, task, ADR, or benchmark-gate change

## Question

BC-001 falsified the assumption that deliberately balanced evidence was enough to
make the shipped learned controller reliable.  On its frozen `b1_r1_d8` dataset,
the learned controller succeeded in 2/32 seed-by-start episodes (6.25%), while the
exact simulator controller succeeded in 32/32.  A coarse exact-transition,
learned-reward, zero-epistemic hybrid succeeded in 27/32, but changed transition,
representation, and uncertainty together.

OL-001 is the smallest component-isolating continuation of portfolio direction P3.
It asks which stage prevents covered experience from becoming control: the planner's
epistemic coefficient, TS-infinity versus recursive-mean propagation, learned
transition recursion, the online/EMA refresh interface consumed by reward, or the
learned reward stack.  It is a BridgeControl precursor, not the full MuJoCo P3
program.

Parents:

- `docs/research/2026-07-13-transformational-research-prompt.md`
- `docs/research/2026-07-13-predictive-reliability-portfolio.md`
- sealed result `bench/bridge_control/results/BC-001`

## Frozen inputs and replay gate

- Load only a byte-for-byte copy of BC-001 `b1_r1_d8.npz`; never regenerate it.
- Formal execution trains only the eight balanced models, once per seed, because
  BC-001 did not serialize weights.  Strict result verification separately
  regenerates those deterministic models from the frozen schedule; those verifier
  regenerations are not reused as or counted as formal outcome fits.
- Reuse the exact BC-001 learner, 1,800 updates, minibatch schedule, evaluation
  starts, 14-step episodes, success predicate, and shipped planner budget.
- Formal model seeds are `0..7`.  Seed `97` is development-only and excluded from
  every formal result and decision.
- Before inference, every regenerated native baseline must replay its saved BC-001
  per-start return and final state within `1e-10`, with exact success vectors.
- BC-001 must verify as `verified_results` before OL-001 preparation, execution,
  analysis, and verification.

Preparation copies the selected NPZ into OL-001 and binds parent results, parent
protocol and manifests, selected source files, dataset bytes and semantics, and this
document.  Formal execution refuses any drift.

The semantic verifier is intentionally stronger than a file-hash check: for a
completed package it retrains seeds `0..7`, checks model fingerprints, reruns every
executed control evaluation, recomputes every fixed-bank score, and rechecks the
wrapper parity gates.  Thus later `verify` and `analyze` calls perform deterministic
verification regenerations; they never overwrite the one-shot formal package.

## Harness semantics

The recursive-mean prefix model carries planner state

`[active learned latent (6), exact raw sidecar (3), exact steps remaining (1)]`.

At the start of each MPC call, both the active latent and raw sidecar come from the
same real observation.  For the first `k` **imagined** transitions of that MPC call,
the wrapper advances the sidecar with exact BridgeControl dynamics and refreshes the
active latent from the exact next state.  It then resumes the unchanged learned
ensemble-mean transition recursion after step `k`.  Prefix depth is candidate-carried
state, never mutable wrapper state, and resets at every real MPC replan rather than
at episode boundaries.

The target-refresh form uses the EMA target encoding after exact transitions because
that is the learned dynamics target space.  A separate full-horizon online-refresh
arm uses the online encoding because that is the reward head's training input.  The
learned reward is always evaluated at the active current latent.  Oracle reward is
defined only for a full exact horizon, so no decoder or prototype inversion is used.

The prefix model exposes only the mean-batch planner path.  Its `k=0` form must match
a direct recursive-ensemble-mean adapter in scores, actions, returns, and final
states; a unit gate separately checks internal warm-start arrays.  Its full
exact-transition/exact-reward form must match the raw
`ExactBridgeModel` under identical planner seeds.  Any parity failure invalidates
the harness.

## Frozen endpoint ladder

The endpoint ladder runs in order on every replayed model:

| Rung | Propagation | Reward input/function | Epistemic coefficient |
|---|---|---|---:|
| `learned_tsinf_penalty` (A) | shipped TS-infinity | learned | 0.03 |
| `learned_tsinf_no_penalty` (B) | shipped TS-infinity | learned | 0.00 |
| `learned_mean_no_penalty` (C) | recursive ensemble mean | learned | 0.00 |
| `exact_target_learned_reward` (D) | exact, target refresh | learned | 0.00 |
| `exact_online_learned_reward` (E) | exact, online refresh | learned | 0.00 |
| `exact_online_oracle_reward` (F) | exact, online refresh | exact | 0.00 |
| `exact_raw` | exact raw-state control | exact | 0.00 |

Interpretations are deliberately narrow:

- A to B changes only the planner coefficient.  It does not test calibration.
- B to C changes member propagation and nonlinear reward averaging together.
- C to D tests the learned transition-mean/recursive-refresh stack conditional on
  the target interface; it is not a representation-capacity test.
- D to E tests the online-versus-EMA interface as consumed by learned reward.
- E to F holds online refresh fixed and tests the learned `reward composed with online encoding` stack, not reward
  head weights alone.
- Learned dynamics plus oracle reward is undefined here because recursive latents
  have no raw preimage.  OL-001 will not fake it with a decoder.

## Conditional prefix curve

The intermediate target-refresh rungs `k={1,2,4,8}` run only if C to D materially
closes the oracle gap under the decision rule below.  C is the `k=0` endpoint and D
is `k=12`.  Every prefix uses recursive mean, learned reward, and coefficient zero.

If D fails but E rescues, the result is classified as an interface failure and the
prefix curve is not run.  If the executed curve is non-monotone beyond paired seed
variation, no minimum-depth knee is reported.

## Outcomes and fixed-bank mechanism audit

Model seed is the independent block (`n=8`); the four starts are repeated measures
inside each seed.  The primary seed-level outcomes are mean episode return and mean
success.  Raw per-start returns, success labels, final states, and action traces are
retained.

A pre-hashed fixed horizon-12 bank is also scored from every start.  It contains
120 one-round colored proposals plus eight exact-model reference elites generated
with a separately labeled 512-candidate, five-iteration oracle search.  It is not a
replay of the shipped 64-candidate-by-three-iteration planner budget.  The bank is
diagnostic and is never injected into the formal planner.  For each rung it records correlation and
rank correlation with exact return, normalized exact regret of the selected
sequence, and the rank of the best reference sequence.

## Decisions and stopping rules

Apply rules in this order:

1. **Replay and parity.** In a first pass over all eight seeds, replay A, check mean
   `k=0` parity, check exact-wrapper/raw parity, and require aggregate exact success
   of at least 95%.  Stop `invalid_harness` on any failure.  Do not execute B, D, E,
   endpoint audits, or conditional prefixes until this all-seed gate has passed.
2. **Endpoint contrasts.** For a directed contrast to be material, seed-level mean
   return must improve in at least 7/8 seeds, close at least 20% of the paired oracle
   return gap, and improve median fixed-bank normalized regret.  A dominant effect
   closes at least 50% of the gap.  Report every raw contrast regardless.
3. **Recovery.** A rung recovers control only if aggregate success is at least 80%
   and it closes at least 50% of the paired oracle return gap.
4. **Prefix curve.** Run only under the C-to-D materiality condition above.  The
   smallest `k` satisfying recovery is a depth knee only if later rungs do not show
   a seed-level reversal.
5. **Search.** The exact ceiling rules out inadequate shipped iCEM on the true
   BridgeControl landscape, not proposal failure on a learned landscape.  OL-001
   reports the fixed-bank evidence but does not inject privileged sequences.  A
   compute-matched injection experiment is a separate follow-up only if a
   high-exact-return reference ranks in the learned scorer's top elite while native
   planning still fails.  Enlarged search follows only if injection rescues.
6. **Residual.** If no endpoint is material, do not force attribution; assign the
   residue to unresolved interactions and design a higher-resolution experiment.

The sign condition is descriptive but stringent: under a fair independent sign
null, at least 7/8 positive directions has one-sided probability `9/256`.  No
p-value is used as a publication claim.  No doubled-compute repeat, new rung,
threshold, or seed may be introduced after formal execution begins; a defect
requires a new experiment identifier.

## Interpretation boundary

OL-001 may localize a causal bottleneck inside this frozen authored fixture.  It
cannot establish benchmark gains, activate U-006/U-007/U-008, change an ADR or gate,
or justify production edits.  Any promoted mechanism requires a separate
falsification experiment and then a clean benchmark evaluation.
