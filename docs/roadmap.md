# Roadmap

## Objective

Prospect must demonstrate one identity-linked causal chain:

> one agent collects experience, learns from those exact records, improves
> executed held-out behavior because of the update, and retains the same gain
> through interference and restart.

The active architecture is ADR-0014. The superseded P-series code, tests, and
benchmark ratchet were removed during the cutover; Git history preserves them as
historical research, not active evidence.

## Evidence ladder

| Row | Claim | Minimum admissible evidence | Current disposition |
|---|---|---|---|
| E0 | Trace integrity | one canonical real experience per step; exact decision/execution/prediction/evidence links; visible partial-failure status | structural tests pass; durable recovery remains open |
| E1 | Epistemic semantics | exact Bayes/EIG/EVSI/proper-score oracle plus future-leakage, irrelevant-noise, destructive-certainty, cost, and relabeling controls | exact reference tests pass |
| E2 | Collect | one target-aware agent acquires useful evidence above matched random/raw-entropy controls | reference-only; current collector is disconnected from E3 |
| E3 | Learn | those exact E2 transition IDs change versioned model state and improve disjoint held-out proper score/calibration over frozen and corrupted-link controls | blocked; current fixture only assimilates task-local beliefs |
| E4 | Improve | the frozen post-update snapshot executes better held-out actions/outcomes at equal budget | blocked; current fixture computes analytic expected utility |
| E5 | Retain | that same gain remains above its pre-learning baseline after shared-state interference and production checkpoint/process restart | blocked; current task slots prevent interference and checkpoint is incomplete |

`make epistemic-diagnostics` checks the exact arithmetic and plumbing.
`make epistemic-gate` remains nonzero until every required row supports its
capability claim. Passing diagnostics may never be substituted for that gate.

## Next implementation sequence

### E0-R — durable lifecycle recovery

- Make the lifecycle journal durable and checkpointed.
- Define idempotent continuation for every partial boundary.
- Serialize concurrent interactions or provide an equivalent transaction protocol.
- Reconcile environment-side effects and pending intentions at restart.

### E3-L — real learner adapter

- Select an existing mature learning backend rather than another custom agent.
- Introduce a transactional prepare/validate/commit learner boundary.
- Prove that exact E2 experience IDs are the samples consumed by the update.
- Change model or representation versions and rebuild the resulting belief.
- Use a disjoint held-out split, frozen no-update control, marginal-preserving
  linkage permutation, irrelevant evidence, and calibration diagnostics.

### E4-B — executed behavioral evaluation

- Freeze the pre-update snapshot and evaluation stream.
- Execute actions and record real outcomes before and after learning.
- Match environment steps, model/planner calls, and other resources.
- Use paired seeds and a preregistered interval/effect rule.
- Keep evaluation updates disabled.

### E5-P — persistence, interference, and plasticity

- Checkpoint model/target/optimizer, replay/sampler/RNG, belief/filter,
  calibration, policy/regulator, knowledge, counters, identities, journals, and
  canonical ledgers.
- Compare uninterrupted and restored continuation.
- Apply a shared-parameter interference stream rather than independent task slots.
- Re-evaluate the original task and also measure plasticity on the new task.
- Ablate persisted state categories so a false checkpoint cannot pass.

### E6-X — external credibility

- Run the unchanged lifecycle on a foreign environment or agent benchmark.
- Compare with strong published baselines at matched budgets.
- Preserve raw traces, protocol, source/config/dependency identities, and
  checkpoint hashes in a sealed result package.

## Research boundary

The architecture is currently a disciplined composition of known ideas:
Bayesian decision value, proper scoring, append-only custody, replay,
checkpointing, and independent lifecycle evaluation. Novelty is not assumed.
Research candidates must survive the prior-art and transformation tests in
`.agents/skills/prospect-research-ideation/SKILL.md` before entering this roadmap.
The current audited candidates and killing experiments are in
[the linked-experience research portfolio](research/2026-07-17-linked-experience-research-portfolio.md);
the numeric claim boundary is in
[the lifecycle results audit](research/2026-07-17-epistemic-lifecycle-results-audit.md).
