# Repository Context: Prospect

This is a navigation aid, not authority. Verify it against
`docs/architecture.md`, `src/prospect/`, `bench/epistemic/`, `tests/`, and
`datasets/` before relying on it.

## Mission

Prospect is an adaptive-agent runtime intended to make one causal statement
testable:

> exact collected experience changed persistent agent state, caused better
> executed behavior on disjoint cases, and the same gain survived interference
> and fresh-process restoration.

Prediction, uncertainty, surprise, information gain, decision value, learning,
knowledge, and retention are separate quantities connected by explicit
identities and versions.

## Current substrate

- **Toolchain:** Python 3.11+, Hatchling, pytest, Ruff, and strict mypy. Core and
  exact references are standard-library-only.
- **Optional runtime:** pinned PyTorch, TorchRL, and TensorDict support an
  experience-replay index. Imports are lazy.
- **Core boundary:** `src/prospect/` is task-neutral. It contains immutable domain
  records, decision decomposition, exact epistemic calculations, the linked
  runtime, canonical stores, and checkpoint coordination.
- **Experiment boundary:** `bench/` owns environments, reference problems,
  baselines, scorers, and experiments. `bench/epistemic/` is the current exact
  semantic reference and active test dependency.
- **Data boundary:** curated reusable inputs live under `datasets/` with
  provenance and checksums. Generated outputs live under
  `bench/**/results/` and are not tracked.
- **Evidence boundary:** an exact finite fixture may validate a contract but does
  not establish mature learning unless one agent, learner, experience ancestry,
  held-out evaluation, interference path, and restored state form the same
  causal chain.

## Highest-value research surface

The nearest missing result is a small real shared-parameter learner that:

1. collects canonical experience through the authoritative runtime;
2. consumes those exact transition identities in a persistent update;
3. improves predictive score and executed utility on disjoint cases;
4. beats frozen, irrelevant-evidence, and marginal-preserving corruption
   controls;
5. experiences genuine cross-task interference through shared state; and
6. retains measurable improvement after complete fresh-process restoration.

Other open mechanisms include:

- transactional prepare/validate/commit semantics for learning;
- durable idempotent recovery across partial lifecycle steps;
- calibrated value-of-information forecasts under model and sensor
  misspecification;
- continual-learning response surfaces for plasticity, interference, and
  retention;
- causal attribution and revocation of individual experience groups; and
- external benchmark adapters with strong published baselines.

## Constraints

- Keep real observations separate from imagined model outcomes.
- Preserve immutable action-time predictions and evaluate them with proper
  scores after outcomes arrive.
- Keep task utility, information value, risk, and cost as distinct decision
  terms.
- Require a version-changing `UpdateReceipt` for learning; belief assimilation
  alone is not model learning.
- Use executed disjoint evaluation, not analytic expected utility or training
  telemetry, as behavior evidence.
- Require shared persistent state for an interference or retention claim.
- Never infer missing provenance, versions, or calibration metadata.
- Never claim novelty or capability from a passing unit test or exact fixture.

## Selected-candidate workflow

For a selected direction:

1. Create one self-contained experiment under `bench/<name>/`.
2. Predeclare the falsifiable claim, null, controls, data split, seeds, budgets,
   metrics, and abandonment rule before formal outcomes.
3. Put reusable inputs in `datasets/`; put generated outputs in
   `bench/<name>/results/`.
4. Keep reusable task-neutral behavior behind typed protocols in
   `src/prospect/`.
5. Add the narrowest adversarial tests needed to protect the new contract.
6. Run `make check`, then audit quantitative evidence with
   `prospect-results-audit`.
7. Update `docs/architecture.md` only if the accepted result changes stable
   system semantics.
