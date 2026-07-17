# SS-001 frozen cross-seed scorer-swap protocol

Status: frozen before formal execution. This document defines the one-shot SS-001
experiment. Any post-start defect terminates this identifier; a repair requires a
new experiment ID.

## Question and scope

CL-001 found exact-damaging within-call refinement, but not at its preregistered
10/12-seed gate. SS-001 asks whether that inherited signature, when it recurs in
untouched generator models, transfers to independently initialized scorers trained
on the same frozen data. It also asks whether a rank-only panel selector recovers
exactly better candidates offline.

This is a **cross-seed scorer swap**, not classical data cross-fitting. All models
share the frozen `BC-001/b1_r1_d8` dataset and architecture. Agreement therefore
supports robustness across training restarts only; it cannot establish new-data,
new-architecture, environment, or production-policy generalization.

## Inputs, roles, and units

- Sealed parent: the complete byte-pinned CL-001 package. Parent seeds 8..19 are
  outcome-visible and enter calibration/transport rows only.
- Auditor seeds: 20..31. Each auditor scores every pool and never changes candidate
  generation or a trajectory.
- Fresh generator seeds: 32..43. These are the primary experimental units.
- Development seed: 97 only, excluded from every formal result.
- Four BridgeControl starts are repeated measures. Auditors are one correlated
  measurement panel. Starts, rounds, candidates, and auditor-generator pairs are
  not independent replicates.

Every model is trained for 1,800 balanced-data updates with batch size 64. Fresh
generators run the unchanged privileged-injection planner for every step of all
four 14-step episodes. Only the three step-0 candidate pools are retained. Ordinary
and instrumented executions must have exact trajectory parity.

## Candidate identities and scorer swap

For each generator/start, flatten the three 64-candidate pools in round-major,
candidate-minor order and deduplicate exact float64 sequence bytes by first
occurrence. The expected union has 186 sequences.

- `B0`: the generator scorer's round-0 argmax.
- `C`: the generator scorer's global argmax over all three rounds.
- Exact and learned ranks are descending, one-based average-tie ranks on the union.

Each auditor encodes the raw state with its own encoder and scores the identical
union with TS-infinity model rollouts, horizon 12, discount 0.99, uncertainty
penalty 0, and no epistemic horizon bound. Raw scores are never compared or
averaged across models. For candidate `u` and auditor `a`:

```text
q_a(u) = (rank_a(u) - 1) / (|union| - 1)
Q(u)   = mean_a q_a(u)
X      = first canonical candidate minimizing Q(u)
```

An auditor rejects `C` when `rank_a(B0) < rank_a(C)`, transfers `C` when the reverse
holds, and otherwise ties. A call is `restart_rejected` at 9/12 rejecting auditors
and `shared_transfer` at 9/12 transferring auditors.

## Frozen primary signature and gates

A fresh call has the inherited harmful signature `H` only when all conditions hold:

```text
B0 is an injected exact-reference candidate
C first occurs after round 0
generator_score(C) - generator_score(B0) > 1e-12
exact(C) - exact(B0) < -1e-12
exact_rank(C) - exact_rank(B0) >= 8
```

The recurrence gate requires at least 10/12 fresh generators to have `H` on at
least 2/4 starts. Failure is `parent_signature_not_reproduced`.

For a generator with `H_g` harmful starts, a direction is supported when at least
`ceil(0.75 * H_g)` calls have that direction. Experiment-level support requires
10/12 fresh generators:

- rejection support: `restart_specific_exploitation`;
- transfer support: `same_data_shared_bias`;
- neither: `heterogeneous_cross_model_transfer`.

The directional branches are mutually exclusive.

On an `H` call, `X` is an exact rescue only when both hold:

```text
exact(X) - exact(C) > 1e-12
exact_rank(C) - exact_rank(X) >= 8
```

The same `ceil(0.75 * H_g)` and 10/12 generator aggregation yields either
`cross_seed_rank_rescue` or `no_robust_cross_seed_rank_rescue`.

## Frozen calibration control

CL-001's 11 exact-improving `C` calls, across visible parent generators
8, 9, 10, 11, 12, and 18, are a negative-direction calibration control only. A
calibration call passes when at least 9/12 auditors transfer `C` and `X` does not
materially degrade it. Material degradation requires both:

```text
exact(X) - exact(C) < -1e-12
exact_rank(X) - exact_rank(C) >= 8
```

A calibration generator passes on `ceil(0.75 * P_g)` of its improving calls; 5/6
generators must pass. Otherwise the result is `auditor_direction_control_failed`
and mechanism/rescue outputs are not interpretable.

## Classification order

1. Any input, source, parity, tensor, lineage, score, or artifact failure invalidates
   the package.
2. Recurrence failure emits `parent_signature_not_reproduced`.
3. Calibration failure emits `auditor_direction_control_failed`.
4. Otherwise emit one mechanism branch plus one rescue branch.

The 9/12, 10/12, 5/6, 2/4, and 75% thresholds are descriptive robustness rules,
not binomial tests or inferential p-values. Exact rescue is offline candidate
selection evidence, not an episode-return or control-policy result.

## Integrity and stopping rules

The complete CL-001 package is copied byte-for-byte at preparation, and original
and copied files remain pinned. The protocol, source hashes, parent identity,
copied dataset, seed assignments, tensor schemas, metrics, thresholds, tie rules,
and branch order freeze before any formal seed 20..43 is trained. A durable atomic
formal-start marker is written before formal computation. All seeds run without
outcome-dependent stopping, retries, auditor selection, or threshold changes.

The fast verifier checks frozen inputs/sources, exact recomputation, candidate
lineage, duplicate-score consistency, deterministic derivations, decisions, and
rendered artifacts. The semantic verifier retrains every generator and auditor,
regenerates every raw tensor and execution record in memory, and requires exact
equality. Neither verifier changes production code, tasks, ADRs, gates, or any
sealed parent artifact.
