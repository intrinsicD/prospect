# CL-001 protocol: learned-versus-exact candidate landscapes

**Status:** frozen before formal execution  
**Date frozen:** 2026-07-15  
**Scope:** non-gated BridgeControl mechanism audit  
**Parent:** accepted PI-003 proposal-injection package

## Question

PI-003 showed that exact-reference sequences were learned-best in every initial
first-round pool but never survived as the final best sequence. Reference transfer
then weakened at real step 1 and vanished from step 2 onward. CL-001 asks two distinct
questions:

1. Before any real action, does learned iCEM refinement increase its own score while
   moving to sequences with materially worse exact value?
2. At visited step-2 states, do exact-useful references lose learned rank even in a
   cold common bank, rather than only because warm-started candidate populations
   changed?

The experiment diagnoses a frozen authored fixture. It does not establish a general
planner, DMC, or production capability claim.

## Independence and units

- PI-003 model seeds `0..7` generated the post-hoc hypothesis. They are replayed only
  to validate instrumentation and recover the previously unrecorded candidate pools;
  they are excluded from confirmatory decisions.
- Untouched model seeds `8..19` are the 12 confirmatory units.
- Four fixed evaluation starts are repeated measures inside each model seed. Candidate
  rows and iCEM rounds are not independent replicates.
- Seed `97` is development-only and excluded from all formal artifacts and decisions.
- Both replay and confirmatory models use the frozen `b1_r1_d8` dataset, 1,800 updates,
  batch size 64, architecture, evaluation starts, and 14-step episodes.

The frozen `10/12` requirement is a descriptive model-seed robustness rule, not a
binomial p-value: restarts share one dataset, and each endpoint is a composite
mechanism signature rather than an exchangeable fair-sign statistic. It does not
support environment-level generality.

## Unchanged planner and replay

The audited arm is PI-003 `privileged_injection`:

- horizon 12, 64 candidates, eight elites, three iCEM rounds;
- colored beta 2, elite keep fraction 0.3, temperature 0.5;
- zero uncertainty penalty;
- eight current-state exact-reference sequences replace the last eight fresh rows of
  round 0 after the full native RNG draw;
- exact reference generation remains separate oracle diagnostic compute.

The sealed PI-003 planner and sources remain untouched. A new bench-only subclass
observes each live `_imagined_returns` call, returns the learned scores without change,
and scores a copy of the same pool with frozen exact BridgeControl dynamics. Every
14-step call is executed because planner RNG and provider call indices persist across
the four episodes; only steps `0`, `1`, and `2` are retained.

For replay seeds, require exact parity with PI-003 model fingerprints, actions,
returns, final states, success labels, plan diagnostics, provider records, and raw-
state hashes. For confirmatory seeds, require exact parity between an ordinary
privileged planner and the instrumented planner under independent identical providers.

## Retained evidence

For every retained call and each of its three iCEM rounds, save:

- all 64 action sequences with shape `(64, 12, 2)`;
- the unchanged learned score and exact simulator score of each identical sequence;
- current-call injected provenance, including carried injected elites;
- raw state, learned latent, real step, iCEM round, seed, and start;
- learned elite membership, selected identity, candidate hashes, and score/rank
  summaries.

The raw tensor package uses float64 arrays and a canonical semantic digest over each
array name, dtype, shape, and little-endian C-order bytes. JSON contains derived rows
and the semantic digest, not duplicate sequence tensors.

## Co-primary 1: within-call exploitation at step 0

Deduplicate the union of the three 64-row pools by canonical float64 sequence bytes,
retaining the first occurrence to match the planner's strict first-maximum rule.

For each seed/start call:

- `R` is the exact-best of the eight injected round-0 references;
- `B0` is the learned-best candidate in round 0;
- `C` is the planner's learned-best sequence over all three rounds;
- `refinement_learned_gain = learned(C) - learned(B0)`;
- `refinement_exact_delta = exact(C) - exact(B0)`;
- `refinement_exact_rank_damage = exact_rank(C) - exact_rank(B0)` within the
  deduplicated union, with descending average-tie ranks.

A start jointly supports within-call exploitation only if:

- `B0` is one of the eight injected references;
- `C` first appears after round 0;
- `refinement_learned_gain > 1e-12`;
- `refinement_exact_delta < -1e-12`; and
- `refinement_exact_rank_damage >= 8`, one frozen elite width.

A seed supports the mechanism only if at least three of its four starts jointly
support it. This prevents separate starts from satisfying different pieces of the
composite signature. The `R` comparison remains descriptive and does not determine
the primary endpoint.

The co-primary passes if at least `10/12` confirmatory seeds support it. Fewer than ten
is `not_supported_by_frozen_rule`; thresholds are not tuned after outcomes.

## Co-primary 2: visited-state scorer shift

For each seed/start, construct one identical cold common bank for all retained states:

- 40 deterministic colored native sequences generated once for that seed/start;
- the eight exact references injected at step 0;
- the eight exact references injected at step 1; and
- the eight exact references injected at step 2.

Reuse all 64 sequences byte-for-byte at steps 0, 1, and 2. This holds the companion
population fixed while retaining the state-specific exact-reference set for every
retained state.

Score each 64-sequence bank learned and exact. Let `R_t` be the exact-best reference at
step `t`, and define its rank residual as
`learned_rank(R_t) - exact_rank(R_t)`. A paired start supports visited-state scorer
shift only if:

- `R_0` and `R_2` both have exact rank at most 8 in their cold banks; and
- `rank_residual(R_2) - rank_residual(R_0) >= 8`.

A seed supports the mechanism only if at least three of its four paired starts
jointly support it. Using residual change prevents a deterioration in the reference's
own exact rank from being attributed directly to learned-score shift. The eight
state-specific target references still differ across steps, so the result remains a
statewise scorer/reference-alignment diagnosis rather than a same-sequence causal
effect. Unlike a bank with only the current state's references, however, no companion
row changes between the paired rank calculations.

The co-primary passes if at least `10/12` confirmatory seeds support it. Step 1 is a
frozen secondary onset diagnostic and cannot change the primary decision.

## Descriptive secondary metrics

Report without additional pass/fail claims:

- per-round Pearson and Spearman learned/exact correlation;
- learned/exact top-eight overlap;
- exact regret and exact rank of each round's learned argmax;
- per-round presence, learned/exact score, and learned/exact rank trajectories for
  `R`, `B0`, and `C`;
- cold-bank learned and exact reference ranks plus their residuals at steps 0, 1, and
  2;
- replay-seed versions of both primary metrics, labeled hypothesis-generating;
- learned and oracle sequence/transition evaluation counts separately.

Score magnitudes are never compared across learned and exact scorers; only within-
scorer differences and ranks are interpreted.

## Frozen decision

- Both co-primaries pass: `within_call_exploitation_and_statewise_scorer_shift`.
- Only co-primary 1 passes: `within_call_exploitation_only`.
- Only co-primary 2 passes: `statewise_scorer_shift_only`.
- Neither passes: `neither_mechanism_supported`.

No production mitigation is implemented from CL-001. If statewise shift passes, a
future state-source experiment may compare privileged-, native-, and exact-controller
states. If exploitation passes, a future intervention must target learned-score
reliability rather than proposal count. Intermediate patterns remain unresolved.

## Invalidity and stop rules

Invalidate the package and draw no mechanism conclusion on any of:

- PI-003 artifact, source, protocol, or copied-dataset drift;
- replay model fingerprint or full privileged-trajectory mismatch;
- confirmatory ordinary-versus-instrumented trajectory/provider mismatch;
- wrong pool count or shape, non-finite value, raw/latent state mismatch across rounds,
  or altered candidate/action/RNG schedule;
- stored learned scores differing from live planner scores;
- exact scores failing deterministic recomputation;
- first-round injection positions or later carried provenance violating PI-003 rules;
- selected action not reconstructible from captured scores;
- tensor, JSON, CSV, report, or manifest failing canonical verification;
- any seed, endpoint, threshold, branch, or retry changed after formal execution starts.

An atomic formal-start record is written before any formal model is trained. Its
presence without a complete verified package is terminal and forbids retrying the
identifier. Run every formal replay and confirmatory seed without outcome-dependent
stopping. A formal packaging, interruption, or verifier defect requires a fresh
experiment identifier; CL-001 is never repaired in place.

## Reproduction package

The package must contain the frozen protocol and input/source hashes, copied dataset,
raw candidate tensors, machine-readable result JSON, per-call CSV, concise report, and
artifact manifest. Verification must recompute all derived metrics from the tensors;
full semantic verification must retrain and regenerate every replay and confirmatory
seed. Run focused tests, the full default test suite, Ruff, strict mypy, and whitespace
validation. Production files, tasks, ADRs, gates, and sealed parent packages remain
unchanged.
