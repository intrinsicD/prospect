# Prompt: Prospect transformational research run

Use `$prospect-research-ideation` for this request. Follow that skill's complete
workflow and required templates. This is a research and falsification task, not a
feature-brainstorming or production-implementation task.

## Research question

What falsifiable research programs could explain and improve **when
prediction-error-driven exploration or observation coverage becomes reliable
downstream control** in Prospect, while preserving the predictive-world-model
spine, the distributional `Prediction` contract, the epistemic/aleatoric split,
and the one-signal-many-jobs design?

The desired result is a diverse portfolio and a recommended smallest decisive
experiment. Do not assume that the answer is another model component. Search
especially for changes to the problem formulation, primitive objects,
measurement grammar, experimental protocol, or evidence base.

## Empirical starting point

Treat these as observations to explain, not conclusions to defend:

- The non-gated DMC probe reports that curiosity reaches the cartpole-swingup
  goal region more often than random collection but produces worse downstream
  MBRL control at the tested budget.
- Imitation from observation succeeds at that budget, while action-recovery
  `R²` was not by itself a reliable predictor of reproduced behavior.
- In the composed toy system, retrieval inside planning is now safe but earns
  little, and the exploit-mode epistemic penalty is near-negligible on the
  current task.
- The current planner aggregates per-step epistemic uncertainty as a scalar
  penalty and optional horizon bound. The repository has not established that
  pointwise uncertainty, maximum reached reward, or occupancy alone measures a
  *connected, repeatable, action-conditioned region of reliable prediction*.

Verify every item against the live repository before using it.

## Mandatory repository grounding

Read the live files, not only the skill's repository profile:

- `CLAUDE.md`, `README.md`, `docs/architecture.md`,
  `docs/requirements.md`, `docs/roadmap.md`;
- `docs/sota-review-2026-07.md`, `tasks/BACKLOG.md`, all relevant ready and
  deferred `U-*.md` tasks, and affected ADRs;
- `src/prospect/types.py`, `world_model.py`, `planning.py`, `voe.py`,
  `memory.py`, and `agent.py` as needed;
- `bench/gates.py`, relevant `bench/evals/`, `bench/SHIPPED`, and the committed
  BH-001 Markdown/JSON report.

Build a frontier map before generating final candidates. Explicitly separate:

- implemented mechanisms and measured results;
- ready maintenance/upgrades U-006--U-012;
- trigger-gated U-101--U-112 ideas that must not be relabeled as new;
- unresolved anomalies and missing measurements;
- architectural invariants that a candidate may not silently break.

## Search scope and evidence discipline

- Literature cutoff: **2026-07-13**.
- Search current primary sources: papers/preprints, official proceedings,
  theses, patents when relevant, and original project/code pages. Use reviews
  only to expand queries, not as sole support for a novelty judgment.
- Search exact terms, synonyms, older terminology, mathematical/functional
  descriptions, donor fields, recipient fields, and bridge fields.
- Record queries, source classes, inaccessible sources, and the strongest
  prior-art threat. Separate donor-method, correspondence, adaptation, and
  prediction novelty.
- Never claim absolute novelty. Use only scoped language such as "apparently
  unexplored under the stated search procedure and cutoff."
- Treat arXiv/OpenReview submissions and workshops as evidence of prior art,
  not as equivalent indicators of peer-reviewed validity.

## Independent generation requirements

Keep generation lanes independent until their first-pass lists exist:

1. At least 4 productive N1/N2 recombinations.
2. At least 4 assumption-surgery candidates.
3. At least 4 primitive/grammar candidates, including 2 new formal objects,
   observables, operators, representations, or problem definitions.
4. At least 3 new-evidence programs.
5. At least 6 mechanism transfers from at least 4 donor fields, with at least
   3 transfers from fields not commonly paired with model-based RL.

Include an explicit fixation anti-library. It must reject duplicates of the
repository's shipped/ready/deferred upgrade track and generic suggestions such
as larger models, component swaps, attention, another learned gate, raw
curiosity, plain multi-step loss, or another uncertainty threshold unless a
distinct falsifiable claim survives subtraction.

For transfers, provide donor-recipient maps and apply terminology-removal,
structural, causal-preservation, counter-analogy, native-baseline, and
historical-obviousness tests. Include at least one transfer of a measurement or
diagnostic practice rather than an algorithm.

## Required empirical work in this run

Perform at least one reproducible computation against committed repository
evidence before selecting the first experiment. At minimum, re-analyze the six
arm-seed observations in `bench/hard/results/BH-001-report.json` to test whether
the current point-coverage proxies (`cov_*`, `goalfrac_*`) are associated with
downstream `mbrl_*` return. Report raw rows, Pearson and Spearman associations,
an exact permutation result where feasible, sample size, and the collection-arm
confounder. Do not turn a small or non-significant result into a capability
claim.

Use that computation only to generate or refine a hypothesis. N4/N4-T requires
an actual new observation and a hypothesis that depends on it; otherwise keep
the candidate at N3/N3-T or below.

## Adversarial evaluation

After generation, switch roles and try to destroy every serious novelty claim.
Apply the novelty and transfer tests from the skill. Downgrade or reject ideas
that collapse into known exploration, reachability, model preconditions,
adaptive rollout horizons, temporal consistency, causal diagnostics, forecast
reconciliation, or ordinary coverage objectives.

Score survivors independently from 0--5 on apparent novelty, falsifiability,
explanatory value, importance, feasibility, first-test cost, interpretability of
results, baseline strength, robustness to negative results, and publication
potential. Preserve a Pareto set rather than forcing one scalar ranking.

For each N3/N3-T/N4/N4-T survivor, specify the cheapest killing experiment with:

- null hypothesis;
- signature under the proposed mechanism;
- signature under the strongest conventional explanation;
- named native and literature baselines;
- minimal data/compute, seeds, metrics, plots, and negative controls;
- leakage/bug/noise checks;
- predeclared abandonment criterion;
- useful artifact produced by an informative failure.

Prefer a two-day harness experiment over a new production subsystem.

## Output contract

Write the completed research artifact to:

`docs/research/2026-07-13-predictive-reliability-portfolio.md`

It must contain:

- repository/frontier map and functional problem signature;
- fixation anti-library;
- 3--5 productive recombinations;
- 3--5 exploratory candidates;
- at least 3 transformational candidates;
- at least 4 audited cross-domain transfers, including 2 rarely connected
  donor fields and 1 measurement transfer;
- at least 2 new-evidence discovery programs;
- the empirical re-analysis and its limitations;
- detailed nearest-work matrices for the strongest candidates;
- a scored Pareto frontier;
- one recommended first experiment, not a demand to implement the largest idea;
- source/query log and audit limitations.

Use the skill's idea-card and transfer-map fields. State unresolved assumptions
instead of inventing them. Cite repository paths/lines and link primary web
sources directly.

Then run:

```bash
python .agents/skills/prospect-research-ideation/scripts/validate_portfolio.py \
  docs/research/2026-07-13-predictive-reliability-portfolio.md
```

Structural validation is necessary but does not establish scientific novelty.

## Scope boundary

Do not modify `src/prospect/`, gates, tasks, backlog status, ADR status, or
shipped results in this run. A selected candidate can enter implementation only
later through a bounded task, any required ADR, a named baseline, a benchmark
gate, collapse sentinels, and the regression ratchet.
