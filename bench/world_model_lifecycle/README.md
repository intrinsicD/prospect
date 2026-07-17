# WM-001: World-model lifecycle

WM-001 is Prospect's first end-to-end causal learning experiment. It asks one
deliberately narrow question:

> Can one agent collect identified experience, update one persistent world model
> because of that experience, improve held-out prediction and executed behavior,
> learn a conflicting contextual task, retain the original gain, and reproduce
> that retained state after a fresh-process restart?

The scientific contract is frozen in [`protocol.json`](protocol.json). Its
raw-byte digest is recorded in [`SEALED_PROTOCOL.sha256`](SEALED_PROTOCOL.sha256).
The numerical thresholds are hypotheses fixed before outcomes, not statements
that Prospect already passes.

## Why this experiment comes first

The current runtime already defines auditable observations, decisions,
ExperienceEvents, epistemic transitions, update receipts, evaluation records, and
component checkpoints. It does not yet establish that learned predictive
parameters caused later behavior to improve and survived interference and
restart. WM-001 supplies that missing causal chain before adding learned
encoders, pixels, multimodality, actor learning, or large external benchmarks.

Pendulum-v1 keeps representation learning out of the first test. Task A uses
normal torque. Task B reverses torque and exposes a context scalar. Both tasks
must pass through the same five-member probabilistic MLP ensemble; task-specific
models, adapters, heads, and checkpoints are forbidden. The ensemble's
deterministic mean prediction feeds TorchRL 0.13.3's existing `CEMPlanner` through
a `ModelBasedEnvBase` adapter. Probabilistic uncertainty is evaluated but is not
an extra planning reward or a custom trajectory-sampling mechanism in WM-001.
Prospect remains responsible for lifecycle custody, transactions, identities,
evidence, and persistence.

## Frozen evidence program

Each of eight formal replicate seeds performs the same sequence:

1. initialize and hash a cold shared model;
2. collect eight complete task-A episodes (1,600 transitions);
3. evaluate cold prediction and executed control on disjoint complete episodes;
4. prepare, validate, and atomically commit a 2,000-step task-A update;
5. evaluate prediction and behavior with all learning and replay writes disabled;
6. run matched frozen and joint-target-permutation controls;
7. collect eight complete task-B episodes through the same agent;
8. fork the post-A state into balanced A/B replay and naive B-only updates;
9. require the naive condition to learn B and measurably forget A;
10. require replay to learn B while retaining the prespecified fraction of A;
11. checkpoint every declared stateful component at an episode boundary;
12. restore in a different process and require exact identities, predictions,
    actions, and paired episode returns.

Prediction validation uses eight held-out episodes per task and replicate.
Executed behavior uses 32 paired reset seeds per task, condition, and replicate.
The inferential unit is the replicate seed, not an individual transition or
episode. The protocol declares exact seed derivation, budgets, metrics,
Student-t intervals, thresholds, controls, and the K0–K7 killing order.

The required controls are:

- a frozen cold model;
- a learner with jointly permuted next-state/reward targets;
- a naive sequential B-only learner;
- an executed random-policy lower bound;
- a separately namespaced true-dynamics MPC ceiling.

If any killing gate fails, later numbers are descriptive only. In particular,
retention is not tested unless B improves and the naive learner reliably forgets
A.

## Two-stage sealing

There are two seals with different purposes:

1. The scientific protocol is sealed now. Outcomes cannot change its null,
   semantics, seeds, budgets, controls, metrics, or thresholds.
2. Once implementation is complete, a formal binding conforming to
   [`schemas/formal-binding.schema.json`](schemas/formal-binding.schema.json)
   records the clean Git commit and tree, every executed source-file digest,
   exact dependency lock and distributions, environment conformance report,
   deterministic runtime, and checkpoint implementation. It must be created
   before the first formal environment reset.

This avoids inventing a future source commit today while preventing code or
dependency changes after formal outcomes begin. Changing the scientific contract
requires a new protocol version. Changing a bound implementation requires a new
binding and a completely new formal run.

## Development and formal lanes

Development runs use only master seeds `101` and `211`. They may use reduced
budgets and exist to debug correctness, feasibility, and failure modes. They are
never claim-eligible and cannot be relabeled formal.

The formal lane uses the eight master seeds listed in the protocol and exactly
the declared budgets. No tuning, exclusions, retries, early stopping, extra
training, or analysis changes are permitted after launch. A crash or missing
replicate fails the active gate.

Raw evidence must conform to
[`schemas/raw-result.schema.json`](schemas/raw-result.schema.json). It retains
episode rows, transition lineage, update ancestry, optimizer batch manifests,
checkpoint component hashes, replicate-level values, and every threshold check.
Aggregate tables without their raw members are invalid.

## Killing gates

| Gate | What must be true |
|---|---|
| K0 | Protocol, implementation binding, seeds, completeness, and matched budgets are valid. |
| K1 | Real-transition custody and whole-episode held-out isolation are exact. |
| K2 | Learning is transactional, changes bytes, binds ancestry, and the committed bytes produce later predictions. |
| K3 | Correct task-A experience beats frozen and corrupted controls on held-out predictive NLL. |
| K4 | The learned model improves executed held-out task-A return under fixed MPC. |
| K5 | The same weights learn B, while the naive B-only update demonstrably forgets A. |
| K6 | Balanced replay retains A without unacceptable loss of B plasticity. |
| K7 | A component-complete fresh-process restore has exact state and behavior parity. |

The exact thresholds are in `protocol.json`; this table is only a reading aid.

## Implementation order

Implementation should preserve the gate order:

1. environment/context wrapper and semantic conformance tests;
2. canonical transition tensor codec and whole-episode split registry;
3. owned model/optimizer/version state with stable parameter digests;
4. transactional `prepare → validate → commit` learner;
5. held-out prediction evaluator;
6. CEM learned-model, random, and true-dynamics planners;
7. B replay and naive control;
8. component-complete checkpoint and fresh-process driver;
9. development runs on the two development seeds;
10. immutable implementation binding, then one formal run;
11. independent recomputation and semantic audit.

Do not start the formal lane merely because the code runs. First pass all
deterministic conformance, ancestry, leakage, rollback, digest-use, and restore
tests, then complete and verify the implementation binding.

## Verification

The verifier uses only the Python standard library:

```bash
python bench/world_model_lifecycle/verify.py protocol
python bench/world_model_lifecycle/verify.py seed collect_a_episode 104729 0
python bench/world_model_lifecycle/verify.py binding path/to/formal-binding.json
python bench/world_model_lifecycle/verify.py result path/to/result.json \
  --binding path/to/formal-binding.json
```

`protocol` is usable immediately. Binding and result verification intentionally
fail until those future artifacts are complete. JSON Schema files provide the
full interchange contract; the verifier additionally enforces cross-file hashes,
seed derivation, causal split custody, gate ordering, and binding consistency
that JSON Schema alone cannot express.
