# VP-001 frozen held-out validator-panel protocol

Status: frozen before formal execution. VP-001 is one-shot. Any defect after the
atomic formal-start marker terminates this identifier and requires a new namespace.

## Question and epistemic scope

SS-001's same-data auditor panel selected a materially exact-worse global mean-rank
candidate `X` on three exact-improving calibration calls. VP-001 asks whether those
three already-fixed winners are rejected by a disjoint same-data validator panel,
consistent with finite-panel winner's curse, or remain preferred across independent
training restarts, consistent with a restart-stable model-family blind spot.

This is a conditional fixed-case diagnostic, not a prevalence estimate, SS-001
replication, data holdout, or independent-architecture validation. Every SS-001/CL-001
outcome, candidate pool, and old auditor score is visible. Only validator outputs at
seeds 44..55 are prospective evidence. The three fixed target calls are the primary
case units, measured by one correlated 12-validator panel. Agreement means same-data,
same-architecture restart stability only. VP-001 cannot change or reclassify SS-001.

## Frozen inputs and model roles

- Parent: the complete byte-pinned 17-file SS-001 package, including its nested
  CL-001 package and dataset.
- Validators: untouched seeds 44..55.
- Development: seed 97 only, excluded from every formal array and decision.
- Frozen source generators: visible parent seeds 8..19 and fresh SS seeds 32..43.
- Training: unchanged `BC-001/b1_r1_d8`, 1,800 updates, batch size 64, identical
  model architecture.
- Scoring: each validator's own encoder, TS-infinity rollout, horizon 12, discount
  0.99, uncertainty penalty 0, no epistemic horizon bound.

Validators score every sequence in every existing 186-sequence first-occurrence
union. This supplies within-validator ranks for the already-fixed identities `B0`,
`C`, and `X`. Raw score magnitudes are never compared or averaged across validators.
VP-001 generates no candidate, changes no trajectory, computes no new panel winner,
and never replaces or reselects `X`.

## Fixed cases and controls

Primary targets, one per visible parent generator:

```text
T = {(8,0), (10,0), (11,2)}
```

All have exact `C > X` with material rank loss under SS-001. Exact-improving
direction controls are the 11 parent calls:

```text
D = {(8,0), (8,1), (9,0), (10,0), (10,3), (11,0), (11,2),
     (12,2), (12,3), (18,0), (18,3)}
```

Fixed-X sensitivity controls are SS-001's 25 fresh exact-rescue calls:

```text
G = {(32,1), (32,2), (32,3), (33,2), (33,3),
     (35,0), (35,2), (35,3), (36,2), (36,3), (37,0),
     (38,1), (38,3), (39,1), (39,3),
     (40,0), (40,1), (40,2), (40,3),
     (41,0), (41,2), (42,1), (42,2), (42,3), (43,2)}
```

Exact-identity sentinels are parent calls `(9,0)`, `(12,2)`, `(12,3)`, `(18,0)`,
and `(18,3)`, where `X` and `C` are byte-identical. Any validator score/rank non-tie
is an integrity failure. Parent controls `(8,1)` and `(10,3)` have submaterial exact
rank losses and remain descriptive. Parent `(11,0)` is a same-generator high-value
positive-X diagnostic with exact `X-C = +29.4601` and a 131-rank gain. Its `X` versus
`C` contrast is descriptive and non-gating; the call remains a direction-control unit
only for its preregistered `C` versus `B0` comparison.

## Ranking and votes

For each validator, descending average-tie ranks are computed over the frozen union.
For a fixed pair `A,B`, the validator prefers `A` iff `rank(A) < rank(B)`; equality is
a tie. No raw cross-model score comparison is allowed.

On each target:

```text
R = number of validators with rank(C) < rank(X)
S = number of validators with rank(X) < rank(C)
heldout_reject   iff R >= 9/12
heldout_transfer iff S >= 9/12
```

The thresholds are descriptive robustness rules, never binomial tests. The normalized
within-validator rank gap is `(rank(X) - rank(C)) / (186 - 1)`, so positive values
favor `C`. Report its median and interquartile range using NumPy's linear quantile
method. Do not pool 36 target votes or treat candidate rows as replicates.

## Frozen gate order and branches

0. Any input, source, hash, tensor, identity, lineage, finiteness, exact-score,
   duplicate-score, or identity-sentinel failure invalidates the package.
1. Direction calibration: on every `D` call compare `C` with `B0`; a call passes at
   >=9/12 `C` votes. Aggregate within each of the six parent seeds using
   `ceil(0.75 * D_g)` and require >=5/6 passing seeds. Additionally, every target must
   pass its local `C > B0` direction check. Aggregate failure yields
   `validator_direction_control_failed`; target-local failure yields
   `target_direction_confounded`.
2. Fixed-X sensitivity: on every `G` call compare exact-better fixed `X` with `C`; a
   call passes at >=9/12 `X` votes. Aggregate within the 11 eligible fresh generators
   using `ceil(0.75 * G_g)` and require >=9/11 passing seeds. Failure yields
   `validator_fixed_X_sensitivity_control_failed`.
3. After all gates:
   - all three targets `heldout_reject` -> `finite_panel_winners_curse_supported`;
   - all three `heldout_transfer` -> `same_data_shared_blind_spot_supported`;
   - at least one decisive reject and at least one decisive transfer ->
     `heterogeneous_target_failure`;
   - every other pattern -> `target_panel_inconclusive`.

Do not majority-label two of three targets. The primary cases are fixed repeated
stimuli measured by one correlated validator panel; no p-values or general prevalence
claim are permitted.

The other 60 visible calls are retained only for provenance and exploratory inspection.
They are ineligible for every confirmatory gate or claim. Any report-wide 11/25
call-pass totals are descriptive integrity summaries; classification aggregates within
source seed first and never treats starts as independent replicates.

## Integrity and stopping rules

Preparation copies the full SS-001 tree byte-for-byte and binds its exact file set,
manifest, canonical protocol, result, tensor, terminal classification, and nested
CL-001 bytes. Child protocol/source/input hashes and all identity indices and sequence
hashes freeze before seed 44 is trained. A durable `O_CREAT|O_EXCL` marker is written
and fsynced before any formal validator model is trained.

All 12 validators run once, with no retry, seed/model selection, early stop, outcome
inspection between models, threshold change, or candidate selection. A valid control
failure/null/ambiguous branch is a scientific result; any post-marker execution,
verifier, provenance, rendering, or semantic mismatch is a terminal integrity failure.

The fast verifier checks original/copied SS bytes, child sources and marker, raw tensor
schema/digest, frozen identity provenance, exact recomputation, duplicate consistency,
deterministic derivations, report/CSV bytes, artifact manifest, and exact closed child
file/result schemas; symlinks and unmanifested claims are forbidden. The semantic
verifier retrains validators 44..55 and requires byte-identical scores, fingerprints,
and derived records. It does not semantically rerun SS-001. No production source,
task, ADR, gate, SS-001 artifact, or CL-001 artifact is modified.
