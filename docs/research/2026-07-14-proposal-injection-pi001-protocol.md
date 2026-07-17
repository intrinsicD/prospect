# PI-001: compute-matched proposal injection and conditional search test

**Status:** frozen before formal model seeds `0..7`  
**Date:** 2026-07-14  
**Scope:** non-gated BridgeControl research evidence; no production, task, ADR, or
benchmark-gate change

## Question and trigger

OL-002 localized BridgeControl failure to a mixed transition/reward stack, but its
fixed-bank audit exposed a narrower discrepancy. On every one of the 32
model-seed/start blocks, the eight privileged exact-reference sequences were in the
learned zero-penalty TS-infinity scorer's top eight and an exact reference was selected,
while native closed-loop learned control succeeded on only 2/32 episodes (6.25%). The
OL protocol predeclared this as the condition for a separate compute-matched candidate-
injection experiment and prohibited enlarged search unless injection first rescues.

PI-001 asks whether the native online failure is primarily proposal scarcity: does
placing high-exact-return sequences inside the unchanged learned scorer's first iCEM
round recover closed-loop control?

Parents:

- `docs/research/2026-07-14-evidence-driven-rd-loop-prompt.md`;
- `docs/research/2026-07-13-predictive-reliability-portfolio.md`;
- sealed BC-001 `b1_r1_d8` data and results;
- preserved failed OL-001 package and failure record;
- sealed OL-002 result package;
- deterministic `bench/proposal_injection/results/PI-001-trigger.json`.

OL-001 and OL-002 are one scientific experiment. PI-001 does not count their matching
numbers as replication.

## Frozen inputs, learner, and evaluation

- Copy only the byte-identical BC-001 `b1_r1_d8.npz` selected by OL-002.
- Train `FlatWorldModel(obs_dim=3, action_dim=2, latent_dim=6, hidden=48,
  ensemble=5)` for 1,800 updates with batch size 64 and the frozen per-seed minibatch
  schedule.
- Formal model seeds are `0..7`; seed `97` is development-only and excluded.
- Use the four frozen BridgeControl starts, 14 real steps, the unchanged success
  predicate, horizon 12, discount 0.99, action range `[-1,1]`, colored beta 2,
  temperature 0.5, and elite keep fraction 0.3.
- The primary learned scorer is shipped TS-infinity with uncertainty penalty zero.
  Zero was selected before PI-001 because OL-002 showed penalty removal was not
  material and it removes a separate coefficient from this search test.
- Model seed is the independent block. Starts and MPC calls are repeated measures.
- The exact simulator is used only to create diagnostic candidates and the ceiling;
  it is never represented as a deployable input.

PI-001 binds the current source, protocol, prompt, trigger, parent artifacts, copied
dataset, Python/NumPy runtime, and repository revision by hash. The formal package is
one-shot. A defect preserves PI-001 and requires a new identifier.

## Harness parity and candidate accounting

The bench-only injection planner mirrors `FlatPlanner.plan`. With injection disabled,
its actions, warm starts, elite state, returns, final states, and RNG state must match
the production planner at strict tolerance under identical seeds. Any mismatch makes
PI-001 `invalid_harness` before causal arms run.

At each real MPC call, the planner first samples the complete native first-round noise
tensor. Injection then replaces the final eight fresh rows; it does not skip RNG draws.
Rounds two and three use the unchanged learned iCEM update and carry logic. Thus every
primary arm scores exactly:

`64 candidates × 3 iterations × 12 imagined steps`

per MPC call. The eight injected candidates are not added to that budget. The separate
oracle search that generates them evaluates 512 exact candidates for five iterations,
then scores the final 128-sequence audit bank. Oracle transitions are counted and
reported separately; they cannot be described as compute-matched deployment cost.

Each plan call records:

- learned sequence and imagined-transition evaluation counts;
- injected sequence count;
- injected sequences in the first-round learned top eight;
- whether the best first-round sequence was injected;
- whether the final best sequence was an unchanged injected sequence;
- selected first action and its source;
- privileged and transformed-reference exact scores;
- RNG/provider seed and current raw-state hash.

## All-seed validity gate

Before inspecting a causal contrast, require across all eight model seeds:

1. OL-002 verifies under its pinned runtime and source snapshot.
2. The PI-001 trigger recomputes byte-for-byte from sealed OL-002 JSON.
3. Every regenerated native zero-penalty model replays its OL-002 per-start return,
   final state, success vector, and actions within `1e-10`.
4. The disabled-injection planner parity-matches `FlatPlanner` within `1e-10`.
5. The exact raw controller succeeds on at least 95% of frozen episodes and replays
   OL-002.
6. Candidate counts and replacement positions equal the protocol.
7. Exact-reference generation is deterministic from model seed, episode order, MPC
   step, and current raw state; the negative transformation is applied after the same
   reference generation.

Failure stops scientific execution. PI-001 may report the invalidating defect but no
proposal, search, or control conclusion.

## Iteration-2 primary arms

Run in a fixed order with fresh planners and identical model weights:

1. **`native_no_penalty`** — production `FlatPlanner`, 64 candidates, eight elites,
   three iterations, no injected candidates.
2. **`privileged_injection`** — replace eight final first-round proposals with the
   eight unique elites produced by the frozen exact 512×5 reference search from the
   current real raw state.
3. **`action_permuted_injection`** — generate the same kind of current-state exact
   references, then swap action coordinates `a0 ↔ a1` in every sequence before the
   identical replacement. This preserves boundedness and temporal shape while
   destroying the BridgeControl action semantics.
4. **`exact_raw`** — unchanged exact-model planning ceiling; this is a validity and gap
   reference, not a learned arm.

Each arm gets its own planner RNG initialized with the same model seed. The privileged
and permuted arms use the same deterministic oracle-search seed schedule, although
their visited states may diverge after their actions diverge.

## Primary outcomes and rescue rule

Retain all per-start returns, successes, final states, actions, plan-call diagnostics,
candidate counts, and provider manipulation checks. Aggregate first within model seed,
then across eight seeds.

For treatment `T` relative to native `N` and exact `E`, define paired gap closure as:

`mean(T - N) / mean(E - N)`.

A **specific privileged rescue** requires all of:

- privileged mean return exceeds native in at least 7/8 model seeds;
- privileged gap closure is at least 50%;
- privileged aggregate success is at least 80%;
- action-permuted injection does not itself satisfy all three preceding conditions.

The sign condition has one-sided fair-sign probability `9/256`; it is a descriptive
threshold, not a publication p-value. Report raw contrasts regardless of decision.

If both injected arms meet the numerical rescue thresholds, classify the result as
`non_specific_injection_rescue`. If privileged injection misses any threshold,
classify it as `no_privileged_rescue`. Do not tune injection count, candidate budget,
seeds, penalty, or threshold after execution.

## Iteration-3 conditional branch

Apply exactly one branch after the primary decision.

### Branch A — specific privileged rescue

Run `enlarged_native_search` with no oracle candidates: 512 learned candidates,
32 elites, five iterations, and otherwise identical TS-infinity zero-penalty planning.
This matches the reference search's proposal count/round count but uses only the
learned model. It is deliberately not compute-matched to native and must report its
13.333× learned sequence-evaluation multiplier.

An enlarged-search rescue uses the same 7/8, 50% gap-closure, and 80% success
thresholds. Success supports proposal scarcity conditional on this fixture; failure
shows that privileged proposal quality, not search scale alone, caused the injection
rescue.

### Branch B — non-specific injection rescue

Run `time_permuted_injection`: preserve each privileged reference's action-coordinate
multiset but apply a deterministic non-identity cyclic time shift before injection.
This tests whether generic structured/saturated proposal shape, rather than simulator-
optimized ordering, caused the common rescue. Use the same primary thresholds but make
no proposal-quality claim if this control also rescues.

### Branch C — no privileged rescue

Do not enlarge search. Run the predeclared action-commitment audit over the already
recorded plan calls:

- fraction with at least one injected reference in the first-round top eight;
- fraction whose first-round best candidate was injected;
- fraction whose final best candidate remained an injected reference;
- success conditional on each event;
- episode position where the fixed-bank-style ranking stops transferring.

Classify the failure narrowly:

- `trigger_not_statewise` if fewer than 50% of MPC calls place any injected reference
  in the learned top elite;
- `refinement_or_warm_start_loss` if at least 50% enter the first top elite but fewer
  than 50% of calls retain an injected reference as the final best sequence;
- `open_loop_closed_loop_mismatch` if at least 50% remain final best yet control still
  fails;
- `mixed_or_unresolved` otherwise.

This audit cannot establish causality by itself; it selects a later experiment.

## Interpretation boundary and abandonment

PI-001 can localize a search/proposal failure inside the frozen authored fixture. It
cannot validate a new exploration metric, activate U-006/U-007/U-008, change planner
production code, amend an ADR, or establish DMC improvement.

Abandon proposal scarcity as the primary explanation if privileged injection fails
its predeclared rescue. Abandon search scale as sufficient if injection rescues but
the conditional enlarged learned search does not. If the action-permuted or
time-permuted control rescues comparably, retain only the weaker hypothesis that
proposal geometry or optimizer regularization matters and design a new matched
mechanism experiment.

No further arm, doubled-compute repeat, threshold, seed, or post-hoc fixture change may
be added to PI-001 after formal execution begins.
