# Prompt: three-iteration evidence-driven R&D loop

Use `$prospect-research-ideation` to audit candidate research directions and use
Prospect's normal benchmark discipline for every selected experiment. This is an
evidence-producing research/development task, not feature brainstorming. Work for
exactly three iterations, where an iteration must end in a durable artifact, a
reproducible computation or experiment, and a decision about the next branch.

Ask the researcher for direction whenever a missing choice would materially change
the scientific question, architectural contract, irreversible repository state, or
interpretation. Otherwise continue with the narrowest evidence-producing action and
state the assumption made.

## Objective

Continue Prospect's predictive-reliability program from the sealed BC-001 and OL-002
evidence. Determine whether the remaining BridgeControl failure is caused by the
learned scorer rejecting useful action sequences, the native iCEM proposal/search
process failing to present those sequences, or a receding-horizon effect not captured
by the fixed-bank audit.

Do not generalize an authored-fixture result to DMC, activate an upgrade task, or
modify production behavior. The intended output is a falsified explanation or a
sharper next experiment, not a benchmark claim.

## Live grounding required before action

Read and reconcile at least:

- `CLAUDE.md`, `docs/architecture.md`, `docs/requirements.md`, and
  `docs/roadmap.md`;
- `docs/sota-review-2026-07.md`, `tasks/BACKLOG.md`, relevant `U-*.md` tasks,
  and affected ADRs;
- `docs/research/2026-07-13-transformational-research-prompt.md` and
  `docs/research/2026-07-13-predictive-reliability-portfolio.md`;
- the BC-001, OL-001, and OL-002 protocols, failure record, source, tests, reports,
  JSON results, and artifact manifests;
- `src/prospect/planning.py`, `world_model.py`, `types.py`, and `agent.py`.

Verify repository state and artifact integrity before trusting summarized results.
Distinguish committed observations, fresh computations, hypotheses, and conclusions.
Never double-count OL-001 and OL-002: OL-002 is an administrative rerun of the same
scientific experiment after OL-001's verifier defect.

## Non-negotiable contracts

- Preserve the predictive-world-model spine, distributional `Prediction`, and the
  epistemic/aleatoric split.
- Keep `src/prospect/` task-agnostic and unchanged in this loop. Experimental
  planners, simulator access, datasets, and scorers belong under `bench/`.
- Do not mutate BC-001, OL-001, or OL-002 source or artifacts. Bind parent files by
  hash and preserve failed artifacts.
- Use the same frozen BC-001 balanced dataset, model seeds `0..7`, learner schedule,
  evaluation starts, episode length, and native planner budget unless a new protocol
  explicitly names a single changed factor.
- Development runs may use seed `97`; formal conclusions may not.
- Predeclare the hypothesis, arms, controls, metrics, thresholds, branch rule, and
  abandonment condition before formal outcomes are inspected.
- A formal defect gets a new experiment identifier. Never repair a sealed formal
  package in place.
- Capability interpretations require paired seed results, raw rows, exact controls,
  negative controls, and machine-verifiable artifacts. Report small samples and
  fixture limitations directly.

## Iteration protocol

### Iteration 1 — reproduce the decision trigger

1. Verify the OL-002 package and recompute, from its JSON rather than prose, the
   paired endpoint/prefix results and fixed-bank diagnostics.
2. Test the predeclared search-injection trigger from the OL protocol: a
   high-exact-return reference must rank in the learned scorer's top elite while
   native learned planning still fails.
3. Compare `prefix_8_target_no_penalty` with
   `exact_target_learned_reward`. Determine whether fixed-bank correlation/regret
   moves in the same direction as closed-loop success.
4. Save the computation and an iteration ledger with raw inputs, hashes, formulas,
   result, limitations, and the selected next branch.

If the trigger does not reproduce, stop the injection branch and explain the
discrepancy. Do not tune thresholds to recover it.

### Iteration 2 — cheapest causal search test

If iteration 1 reproduces the trigger, freeze and run a new non-gated candidate-
injection experiment:

- Native arm: the OL-002 learned TS-infinity scorer with zero uncertainty penalty
  and the unchanged `64 candidates × 3 iterations` iCEM budget.
- Privileged-injection arm: at every real MPC call, replace a predeclared number of
  first-round native proposals with high-exact-return sequences generated from the
  current exact simulator state. Do not add learned-model candidate evaluations.
- Negative control: inject action-coordinate-permuted versions of the same reference
  sequences under the identical replacement and RNG schedule.
- Exact ceiling: bind or replay the OL-002 exact result.

The experimental planner must parity-match `FlatPlanner` when injection is disabled.
Record whether injected sequences enter the learned top elite, whether one supplies
the chosen first action, paired returns/success, exact score of proposed and selected
sequences, candidate counts, and oracle-only diagnostic compute separately from the
learned planner budget.

Predeclare rescue as all of: at least 7/8 seed-level return improvements, at least
50% closure of the paired native-to-exact return gap, at least 80% aggregate success,
and no comparable rescue from the permuted control. A failure remains useful evidence
and must not be rescued by changing candidate count, thresholds, seeds, or injection
fraction after the run.

### Iteration 3 — branch on iteration 2, do not merely narrate it

- **Specific privileged rescue:** run one frozen enlarged-search arm with no oracle
  candidates. Change only native iCEM search compute according to the preregistered
  budget. Test whether search alone reproduces the injection rescue.
- **Privileged and permuted rescue:** run a matched non-oracle structured-proposal
  control to distinguish useful reference quality from generic proposal regularity.
- **No privileged rescue:** do not enlarge search. Run the smallest logged
  action-commitment audit that determines whether injected references were scored as
  elites but lost through iCEM refinement/warm-start or whether the fixed-bank trigger
  failed to transfer to visited MPC states.
- **Invalid harness or parent drift:** preserve the failed package and use iteration 3
  to localize the defect only; draw no scientific conclusion.

Freeze the iteration-3 branch before inspecting its outcomes. Apply the same paired
seed, control, artifact, and one-shot rules as iteration 2.

## Evaluation and adversarial audit

At each iteration, attempt to destroy the favored explanation. Separate:

- score fidelity on a common candidate set;
- proposal recall of high-exact-return sequences;
- optimizer selection/refinement behavior;
- first-action quality under receding-horizon replanning;
- closed-loop return and success.

Do not call a result novel merely because it is new to Prospect. Search prior work
only if making a research-novelty claim, and then record primary sources, queries,
cutoff date, nearest threat, and uncertainty. The candidate-injection sequence is
primarily a causal systems diagnostic; engineering value does not require novelty.

## Durable output contract

Produce:

1. this reusable prompt;
2. a three-entry iteration ledger under `docs/research/`;
3. one frozen protocol per new formal experiment;
4. task-specific implementation and tests under `bench/` / `tests/`;
5. immutable machine-readable raw results, a concise Markdown report, input/source
   hashes, and an artifact manifest;
6. a final synthesis stating what was falsified, what remains live, and the cheapest
   next experiment.

Validate targeted tests, Ruff, strict mypy, and the full default test suite. If a
shipped phase or production file is touched unexpectedly, stop and request direction
before continuing.
