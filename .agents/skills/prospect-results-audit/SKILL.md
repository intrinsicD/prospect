---
name: prospect-results-audit
description: Performs an adversarial referee pass over Prospect experiment, capability, and causal-mechanism claims. Use after a quantitative experiment or evidence session, before promoting any learning/improvement/retention claim, while reviewing a results-bearing change, or whenever reported numbers lack independent semantic verification. Do not use for generating new research directions.
---

# Results Audit

Act as an independent referee. Assume the producing session made a mistake in
protocol, data custody, control construction, accounting, statistics, or claim
wording and try to find it. The output is corrected evidence and bounded claims,
not reassurance.

## 1. Inventory claims

List every quantitative, causal, generality, capability, and retention statement
in the changed code, tests, architecture, README, and experiment report.

For each claim record:

- exact wording and scope;
- development versus disjoint/held-out status;
- source, dataset, configuration, model, and evaluator identities;
- raw evidence package;
- verification command; and
- current disposition: asserted, supported, weakened, refuted, or unresolved.

A claim with no explicit scope or raw evidence is already a finding.

## 2. Recompute from raw outcomes

Reconstruct metrics and decisions from raw rows or tensors rather than report
prose. Verify checksums, data membership, source/config versions, split
identities, checkpoint components, and causal ancestry. Run both a fast
structural verifier and an independent semantic recomputation when available.

Report rendering, schema validity, and green unit tests do not replace semantic
recomputation.

## 3. Bind the causal chain

For a learning claim, verify that the same continuous chain contains:

1. canonical collected experience IDs;
2. transition IDs derived from those experiences;
3. an `UpdateReceipt` that consumed those transitions and changed declared
   persistent versions;
4. a frozen post-update snapshot;
5. executed evaluation on disjoint cases at matched budget;
6. frozen, irrelevant-evidence, and corruption controls;
7. genuine shared-state interference; and
8. a complete checkpoint restored in a fresh process.

Reject a chain assembled from different agents, learners, task-local independent
slots, analytic utility, or an in-process round trip.

## 4. Audit protocol parity

Confirm that the executed claim, environment, data bytes, split, model and
evaluator seeds, budgets, thresholds, branch rule, dependency versions, and
source state match the frozen protocol. Development or replay data may not
silently become confirmatory evidence.

Give a defective formal run a new run identifier. Do not repair a result package
in place during the same study.

## 5. Recompute accounting and statistics

Check equal environment steps, model evaluations, tool calls, horizons,
training updates, and wall-clock scope. Recompute paired deltas, intervals,
success counts, calibration, proper scores, utility/regret, plasticity,
forgetting, and retention from the raw data. Do not treat correlated rows,
timestamps, or repeated starts as independent samples.

## 6. Audit controls

Require every predeclared positive/negative control and collapse sentinel to
behave as intended before interpreting the primary endpoint. Verify that
shuffled, poisoned, irrelevant, or ablated controls change only the intended
factor and preserve marginals, budgets, and denominators where required.

A secondary metric cannot rescue a failed primary rule or control.

## 7. State limits

List exactly what ran and what did not: unit tests, structural checks, semantic
regeneration, training, disjoint evaluation, interference, checkpoint restore,
external benchmark, or artifact inspection only. Keep optional or unavailable
work explicitly pending.

Use precise language:

- “failed to establish” rather than “proved absent”;
- “fixture-specific mechanism evidence” rather than “agent capability”;
- “reload parity” rather than “retention” without interference and behavior;
- “apparently novel under the searched scope” rather than absolute novelty.

## 8. Dispose of each claim

Assign one disposition:

- **confirm** — raw evidence, semantics, and scope all survive;
- **narrow** — rewrite to the supported endpoint, fixture, or maturity;
- **retire** — remove the unsupported statement; or
- **unresolved** — name the missing evidence and next decisive check.

Apply matching corrections to the result report, README, architecture, tests, or
code in the same change. Do not leave a stronger claim elsewhere.

## 9. Report

End with:

- the claim table and dispositions;
- commands run and outcomes;
- independent recomputations;
- corrections made;
- unverified evidence and why;
- unused or unopened data;
- explicit follow-up tests; and
- a one-sentence verdict on what the experiment actually establishes.

## Hard prohibitions

- Tune thresholds, controls, seeds, or branches after formal outcomes.
- Treat training loss or internal uncertainty reduction as external improvement.
- Count administrative reruns as independent evidence.
- Promote an exact fixture, development split, or analytic expectation into a
  general capability.
- Let a secondary endpoint rescue a failed primary endpoint.
- Report remembered numbers without reopening their raw source.

## Repository anchors

- `docs/architecture.md` — canonical semantics and evidence ladder.
- `src/prospect/` — linked runtime and persistent-state implementation.
- `bench/epistemic/` — exact reference semantics, not a mature capability proof.
- `bench/<experiment>/results/` — generated raw evidence for a new experiment.
- `datasets/` — curated inputs and checksums.
- `tests/` and `Makefile` — executable verification surfaces.
