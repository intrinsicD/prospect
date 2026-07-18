# Repository Context: Prospect

This is a navigation aid, not authority. Verify it against `README.md`,
`docs/architecture.md`, `src/prospect/`, `bench/epistemic/`,
`bench/world_model_lifecycle/`, `tests/`, and `datasets/` before relying on it.

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
- **Optional runtime:** pinned PyTorch, TorchRL, TensorDict, Gymnasium, and
  JSON Schema support replay and the WM-001 world-model experiment. The
  backend-neutral core still declares no required runtime dependency.
- **Core boundary:** `src/prospect/` is task-neutral. It contains immutable domain
  records, decision decomposition, exact epistemic calculations, the linked
  runtime, canonical stores, and checkpoint coordination.
- **Experiment boundary:** `bench/` owns environments, reference problems,
  baselines, scorers, and experiments. `bench/epistemic/` is the exact semantic
  reference; `bench/world_model_lifecycle/` contains the current WM-001
  protocol, implementation, verification, audit, and runbook.
- **Data boundary:** curated reusable inputs live under `datasets/` with
  provenance and checksums. Generated outputs live under
  `bench/**/results/` and are not tracked.
- **Evidence boundary:** an exact finite fixture may validate a contract but does
  not establish mature learning unless one agent, learner, experience ancestry,
  held-out evaluation, interference path, and restored state form the same
  causal chain.

## Highest-value research surface

WM-001 protocol 1.3.0 now supplies the first bounded end-to-end implementation:
a probabilistic ensemble learns two observed-context Pendulum regimes, drives a
fixed-budget CEM controller, undergoes shared-state interference, uses balanced
replay for retention, and restores its retained state in a fresh process. Its
eight-seed immutable producer passed K0-K7, but the mandatory pre-bound
independent auditor returned two reproduced false negatives. The attempt is
formally rejected and the complete lifecycle claim remains unestablished; this
is not evidence that the agent failed to learn.

The next confirmatory step is to bind the formal seed schedule to the sealed
protocol, define coverage in integer target-count space with exact endpoint
semantics, adversarially test both repairs before outcomes, and then issue a new
protocol version, fresh seed domain, clean implementation binding, and immutable
attempt. Same-seed or corrected-auditor replay of v1.3 remains diagnostic only.

Other high-value open boundaries include:

- calibrated value-of-information estimates with adversarial controls;
- durable idempotent recovery across partial lifecycle steps and abrupt process
  death;
- serialization beyond the learning commit so concurrent callers cannot race
  across interaction-stage boundaries;
- exact mid-episode resume across environment, recurrent belief, pending action,
  external side effects, and RNG state;
- broader continual-learning plasticity and retention beyond two
  observed-context actuator regimes; and
- external benchmarks and strong published baselines before capability or
  novelty claims.

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
6. Run the scope-appropriate gate (`make check` for the backend-neutral core,
   `make check-runtime` for world-model/runtime changes), then audit quantitative
   evidence with `prospect-results-audit`.
7. Update `docs/architecture.md` only if the accepted result changes stable
   system semantics.
