# E0-001 — Epistemic lifecycle rewrite

- **Status:** in-progress
- **Phase:** E0 (E-series, Epistemic lifecycle v2)
- **Requirements:** R1, R3, R7; R8 at the typed knowledge/acquisition boundary
- **ADRs:** ADR-0014; amended ADR-0001, ADR-0004, ADR-0006, ADR-0008;
  ADR-0005 (preregistered, benchmark-gated evidence)
- **Depends on:** none; historical P-series results are not prerequisites
- **Phase gate:** exact reference diagnostics implemented; full capability gate
  currently fails with E2/E3 `reference_only` and E4/E5 `blocked`

## Goal
Produce one runnable flat agent lifecycle in which Prospect can trace the exact
experience it collects, update a versioned predictive model from that experience,
show a matched improvement in behavior, restore the learned state, and show that
the improvement remains after a prespecified interference block.

The result is four independently reported claims — **collect**, **learn**,
**improve**, and **retain** — backed by linked artifacts. The task is complete only
when all four meet preregistered criteria. At task creation, none is implemented or
evidenced.

## Evidence boundary
P0–P14 remain historical **legacy-v1 evidence** in Git history and authored
research narratives. Their active implementation, tests, evaluator registry, and
ratchet were removed at cutover. They do not pre-pass an E-series criterion.
Every new result requires new run, model, calibration, decision, transition, and
checkpoint identities.

## Non-goals
- No new hierarchy, option discovery, generative replay, multimodal codec,
  imitation, passive-observation learner, semantic consolidation, or external tool
  integration.
- No claim of a universal agent or state-of-the-art control.
- No custom replacement for a mature collector, tensor batch, replay, optimizer,
  model/planner, or checkpoint mechanism when an existing substrate satisfies the
  contract.
- No learned meta-regulator until the explicit value-of-information baseline and
  its ablations work.
- No exact mid-episode resume in E0; the first persistence contract is explicitly
  episode-boundary.

## Interfaces and records to satisfy
The target contracts are implementation-neutral at the core boundary and batched
in the backend. Exact fields may be refined before code lands, but semantic
collapsing requires an ADR amendment.

### Domain records
- `BeliefState`: posterior/control features, optional recurrent filter state,
  optional distribution parameters, and representation version.
- `Prediction`: named outcome/reward/continuation distributions, uncertainty with
  measure and units, horizon, action-time prediction identity, model version, and
  calibration identity.
- `Experience`: immutable raw observation/action/outcome fact with
  terminated/truncated, discount, episode/step/task/goal/decision identities, and
  behavior-policy version.
- `EpistemicTransition`: links one `Experience` to before/after beliefs, the exact
  action-time `Prediction`, a named proper score, and realized information gain.
- `Goal`, `InformationValue`, `UtilityEstimate`, and `Decision`: make task value,
  expected information gain, goal-conditioned information value, risk, resource
  cost, and selected action explicit.
- `UpdateContext` and `UpdateResult`: record consumed samples, before/after model
  versions, update identity, priorities, and metrics. Training metrics are
  telemetry, not lifecycle evidence.
- `KnowledgeClaim`, `CalibrationState`, and `CalibrationReport`: link scope,
  evidence, provenance, data split, model/representation version, method, and
  sample count.
- `CheckpointManifest`: declares schema/source/config identities, component
  versions, environment/update counters, replay snapshot, random states, and
  episode-boundary resume semantics.

### Protocols
- `BeliefUpdater.initial/assimilate`
- `PredictiveModel.predict/rollout`
- `SurpriseScorer.score`
- `InformationEvaluator.evaluate`
- `DecisionPolicy.decide`
- `Learner.update(batch, context) -> UpdateResult`
- `Calibrator.fit/apply/audit`
- central `CheckpointStore.save/load`

The runtime contract is `decide -> environment step -> observe`. The returned
`Decision` is passed explicitly into observation handling; no hidden pending
prediction or mutable planner sign is allowed.

## Existing substrate policy
Use existing solutions as the first reference implementation:

- TorchRL/TensorDict-style collector, typed batch, replay, sampler, and
  environment-transform mechanisms;
- an adapted TD-MPC2-class latent model/planner as the first reference controller;
- substrate-native/PyTorch component state dictionaries and replay serialization;
  and
- an independent DreamerV3-class result as an external algorithmic baseline when
  the lifecycle harness is stable.

Before adding a dependency, pin and record its version, license, source revision,
checkpoint semantics, and adapter surface. If an upstream API or license makes the
selected substrate unsuitable, update this task and the dependency decision before
substituting another implementation. Do not silently reimplement the algorithm.

Prospect-owned code is limited initially to domain contracts, adapters, explicit
epistemic/information-value regulation, calibration/evidence linkage, lifecycle
orchestration, and the independent evaluator.

## Approach and migration slices

The original side-by-side migration was shortened after audit: the P-series
interfaces could not represent the required identities and keeping compatibility
would have made the old semantics normative. The active cutover therefore kept
historical receipts in Git and removed the compatibility layer.

### A. Add contracts
1. Preserve legacy-v1 authored receipts and source history without treating them
   as characterization tests for v2.
2. Add ADR-0014 records/protocols as the sole active domain contract.
3. Require any future legacy or third-party adapter to fail on information it
   cannot represent rather than invent provenance or termination semantics.

### B. Cut over the runtime and storage
1. Introduce terminated/truncated environment results.
2. Make one runtime own `decide -> step -> observe`, storing exactly one raw
   `Experience`.
3. Move replay to the selected tensor substrate. Store raw experience only;
   imagined transitions use a separate schema and lineage.
4. Remove the legacy policy adapter and historical gates from the active runtime.

### C. Cut over model, planner, and learning
1. Wrap the selected reference model with `BeliefUpdater`,
   `PredictiveModel`, and `Learner`.
2. Wrap the planner as `DecisionPolicy`; every chosen action returns its action-time
   prediction and utility/information breakdown.
3. Make the update scheduler, not the learner, own sampling and update cadence.
4. Increment model/policy versions and invalidate or refresh calibration
   explicitly.

### D. Add persistence and epistemic regulation
1. Persist model/target/optimizer, replay/sampler, normalization,
   calibration/regulator/knowledge state, counters, configuration/source
   identities, and RNG state through one manifest.
2. Prove episode-boundary restore equivalence before using restart in retention
   evidence.
3. Start with a transparent estimator of expected information gain and
   goal-conditioned value of information. Record its utility contribution; do not
   use a hidden signed uncertainty coefficient.

### E. Run the independent lifecycle evaluator
1. Freeze formal splits, seeds, budgets, controls, metrics, thresholds, and
   interference stream before the first formal result.
2. Run and report each lifecycle claim separately.
3. Run component and mechanism ablations, including no update, frozen pre-update
   policy, zero information value, shuffled experience linkage, and checkpoint
   state omissions.
4. Preserve all formal manifests and receipts; a failed row remains a failed row.

### F. Remove superseded paths
Completed at the active-tree cutover: the old flat source modules, P-series tests,
benchmark registry/ratchet, and obsolete workflow were removed. Git history is the
archive; no compatibility package remains in the production import graph.

## 2026-07-17 implementation and audit result

The new domain, decision, runtime, storage, exact epistemic semantics, and
reference diagnostics are implemented. An adversarial scientist pass reproduced
the numeric results but rejected the full lifecycle interpretation:

| Row | Numeric diagnostic | Audited disposition |
|---|---|---|
| E2 | exact VOI selects the diagnostic probe at matched synthetic cost | reference-only; these experiences are not consumed by E3 |
| E3 | task-local exact posterior log score improves over controls | reference-only; no model/representation/policy update and no disjoint task split |
| E4 | analytic expected utility rises | blocked; no held-out action/outcome execution |
| E5 | task-keyed posterior values survive JSON round-trip | blocked; interference is excluded and canonical custody is omitted |

The full report deliberately emits `passed: false`. The exact fixture remains an
E0/E1 semantic/plumbing oracle and cannot promote E2–E5.

Artifacts:

- `docs/research/2026-07-17-epistemic-lifecycle-results-audit.md`
- `docs/research/2026-07-17-linked-experience-research-portfolio.md`
- `docs/runtime-substrate.md`

## Acceptance criteria

### Contract and trace integrity
- [ ] All target records and protocols exist with static conformance tests.
- [ ] Every environment step produces exactly one raw `Experience`; no missing,
      duplicate, or cross-episode decision links occur in the formal runs.
- [ ] The observed outcome is scored against the immutable action-time prediction,
      with compatible model/representation/calibration versions.
- [x] Raw real experience and imagined lineage cannot share a schema or storage
      namespace.
- [x] Utility and information contributions are visible in `Decision`; no runtime
      consumer mutates an uncertainty sign behind the decision interface.

### Existing-substrate adoption
- [x] Dependency/license/revision audit is recorded for the selected tensor/replay
      and checkpoint substrates; model/planner selection remains open.
- [ ] Tensor batches remain batched through collection, replay, and learning; the
      backend does not convert whole training batches into Python transition lists.
- [ ] Legacy reference behavior has parity adapters during migration, followed by
      removal or explicit legacy-baseline isolation.

### Independent lifecycle claims
- [ ] **Collect:** formal trace-integrity checks pass and the raw replay snapshot
      hash/count agrees with the environment-step ledger.
- [ ] **Learn:** on a disjoint held-out split, the updated model improves the
      preregistered proper score and calibration criterion over a matched no-update
      control. Training loss is reported but is not the pass criterion.
- [ ] **Improve:** at equal held-out evaluation budget, the post-update decision
      policy improves preregistered behavioral utility over the frozen pre-update
      policy, with the required paired uncertainty interval/effect criterion.
- [ ] **Persistence prerequisite:** save/reload is behaviorally equivalent to the
      in-memory post-update agent within a preregistered tolerance.
- [ ] **Retain:** after reload and the frozen interference/delay protocol, behavior
      remains above the pre-learning baseline under the preregistered paired
      criterion. Reload parity alone is not called retention.
- [ ] A four-row claim report preserves independent PASS/FAIL outcomes; the full
      lifecycle statement is emitted only if every row passes.
- [ ] Calibration fitting, learning evaluation, improvement evaluation, and
      retention evaluation use declared disjoint evidence or a preregistered
      leakage-safe design.

### Checkpoint and reproducibility
- [ ] The checkpoint manifest covers every stateful component and declares
      episode-boundary semantics.
- [ ] An interrupted-at-boundary run restored from checkpoint matches the declared
      uninterrupted reference for counters, replay/sample order, subsequent update
      metrics, and chosen actions within the frozen determinism tolerance.
- [ ] Formal result artifacts bind source/config/dependency identities, splits,
      seeds, budgets, model/policy/calibration versions, and checkpoint hashes.

### Quality and governance
- [x] New focused unit/contract/integration tests pass.
- [x] `make test`, `make lint`, and `make typecheck` are green.
- [ ] The new E0 evaluator passes its preregistered criteria; P0–P14 results are not
      counted.
- [x] Superseded production paths are deleted or isolated, and no new production
      import references them.
- [x] Confirmed last-run validation receipt is recorded below.

## Test and evidence plan

### Unit and contract
- Distribution family/scorer agreement; score is best at the forecast mode where
  applicable.
- Version mismatch and stale-calibration rejection.
- Decision-to-experience identity round trip.
- Terminated versus truncated bootstrapping semantics.
- Real/imagined lineage separation.
- Utility decomposition arithmetic and information-value units.
- Strict checkpoint-manifest missing/extra component checks.

### Runtime and integration
- One `decide -> step -> observe` call sequence per step, including reset and
  truncation.
- Replay sequence sampling never crosses episode boundaries.
- Learner consumes the declared replay snapshot and returns a new model version.
- Save/reload continuation compared with uninterrupted episode-boundary execution.
- Legacy adapters are exercised only by legacy-baseline tests.

### Formal lifecycle experiment
- At least a no-update control and frozen pre-update policy at matched environment
  and evaluation budgets.
- Held-out prediction/calibration data separate from behavior evaluation.
- Paired seeds and confidence/effect criteria frozen before the formal run.
- A prespecified task delay or interference stream followed by retention
  evaluation on the original task.
- Zero-information-value and shuffled-linkage negative controls.
- Ablation of each persisted state category to confirm the manifest catches
  retention-breaking omissions.

## Confirmed last-run validation
- **Completed:** 2026-07-17T16:00:20+02:00
- **Source state:** uncommitted E-series cutover based on
  `f126703304ec5807856e1280d610ba6df0029056`; active source/test/tooling manifest
  SHA-256 `a0fd78938bb45bce1a7a84b56145c935f0ab124fedcd6ef847783d70d69ea3b5`
- **Environment:** Python 3.12.9; PyTorch 2.9.0+cu128; TorchRL 0.13.3;
  TensorDict 0.13.0
- **Commands and outcomes:**
  - `make check` — PASS: Ruff; mypy over 32 source files; 85 tests; exact
    diagnostics. The diagnostic JSON correctly remains `passed: false`.
  - `make test-runtime` — PASS: 13 optional storage/TorchRL tests.
  - `ruff format --check src/prospect bench/epistemic tests/test_epistemic_*.py`
    — PASS: 32 files already formatted.
  - `python .agents/skills/prospect-research-ideation/scripts/validate_portfolio.py
    docs/research/2026-07-17-linked-experience-research-portfolio.md` — PASS;
    structural validation is not a novelty result.
  - `python -m pip wheel --no-deps --wheel-dir /tmp/prospect-wheel-check .` —
    PASS: `prospect-0.0.1-py3-none-any.whl`, SHA-256
    `92b20326618749e40f99e05c8c7f81f70fe3aadee5c22a47e87faabdd048a6b2`.
  - stale legacy-import scan and `git diff --check` — PASS.
  - `make epistemic-gate` — expected FAIL (exit 1): E2/E3
    `reference_only`, E4/E5 `blocked`, full report `passed: false`.

## Docs-sync checklist
- [x] Task Status updated; four claim results recorded independently.
- [x] Confirmed last-run validation receipt completed.
- [ ] E0 gate definition, formal protocol, and result artifacts linked.
- [x] ADR-0014 consequences updated for the implemented cutover.
- [x] Requirements and architecture documents reflect the current contracts.
- [x] P0–P14 remain marked legacy-v1 and their historical receipts remain
      unchanged.
- [x] Superseded production-path deletion/isolation is documented.

## Gate result
**BLOCKED.** Exact reference predicates run successfully, but the audited
capability report is intentionally false: E2/E3 are reference-only and E4/E5 are
blocked. No model-learning, behavioral-improvement, retention, or full-lifecycle
capability claim is made.
