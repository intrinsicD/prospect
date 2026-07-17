---
name: prospect-research-ideation
description: Generates and rigorously audits diverse, falsifiable research directions for Prospect's adaptive agent, including assumption surgery, new primitives and evidence programs, cross-domain mechanism transfer, and adversarial prior-art review. Use for novel research ideas, unexplored agent or world-model algorithms, high-risk/high-reward experiments, or publishable extensions that may enter Prospect's experiment and benchmark workflow. Do not use for ordinary feature brainstorming or implementation of an already selected method.
---

# Research Ideation (Transformational)

> **Provenance.** First-party skill by the repository owner (Alexander
> Dieckmann), MIT-licensed, adapted into Prospect from the
> `transformational-research-skill-kit` v1.0.0. It is a SKILL.md-with-companion
> discipline skill: `references/`, `assets/`, `evals/`, and `scripts/` are
> hand-authored, and `references/repository-context.md` is the Prospect-specific
> rewrite verified against the repository's live architecture and workflow.

> **Repository fit.** This skill is the ideation *front end* for the research
> mission. It proposes and audits; it does not modify `src/prospect/`, the benchmark
> harness, or repository state. A selected candidate enters Prospect as one
> bounded experiment under `bench/`, with a predeclared claim, named baselines,
> controls, budgets, killing criterion, and generated outputs under `results/`.
> Put reusable inputs in `datasets/` with provenance and checksums; keep
> `src/prospect/` task-agnostic and protocol-driven. Update
> `docs/architecture.md` only when stable system semantics change. Never fabricate
> prior art, results, or novelty claims.

## Mission

Produce a **diverse, falsifiable research portfolio**, not merely surprising prose. Always include useful recombinations, but deliberately search for ideas that alter the problem's primitives, assumptions, objectives, admissible operations, or evidence base.

Never claim global or absolute novelty. Use language such as:

- "apparently novel under the stated search procedure and cutoff date";
- "I did not find this under the searched formulations";
- "the apparently novel component is...";
- "the strongest prior-art threat is...".

Never say:

- "definitely novel";
- "no one has ever done this";
- "guaranteed publication".

## Activation boundary

Use this skill when the request concerns research ideation, scientific hypotheses, unconventional algorithm design, cross-field transfer, new experiments, or publishable extensions.

Do not use it for ordinary product features, small optimizations, routine refactoring, or direct implementation of a method the user has already chosen.

This skill proposes and audits research. It does not automatically modify
production code. A selected candidate must first become a bounded experiment
with a protocol, controls, and abandonment rule.

## Required resources

Read these files as needed, using paths relative to this skill root:

- `references/novelty-taxonomy.md` for novelty classes and transformation tests;
- `references/cross-domain-transfer.md` for mandatory donor-field search and transfer validation;
- `references/evidence-generation.md` for experiments that can create evidence absent from the initial corpus;
- `references/prior-art-audit.md` before making novelty claims;
- `references/repository-context.md` when present, but verify it against the actual repository;
- `assets/idea-card.md`, `assets/transfer-map.md`, and `assets/research-portfolio.md` for output structure.

## Inputs to establish

Infer from the repository and request where possible:

- target research question or desired capability;
- recipient field and adjacent fields;
- literature cutoff date and available sources;
- existing methods, assumptions, benchmarks, code, data, hardware, simulators, and proof tools;
- project duration and acceptable risk;
- intended venue or expected contribution level;
- known ideas that must not be repeated;
- repository constraints and validation workflow.

State unresolved assumptions instead of silently inventing them.

## Operating principles

1. Separate generation from evaluation.
2. Use independent idea lanes before allowing cross-contamination.
3. Maintain incompatible directions; do not converge early on one fashionable answer.
4. Treat novelty, correctness, usefulness, feasibility, and publishability as separate quantities.
5. Prefer causal mechanisms and falsifiable predictions over analogies and buzzwords.
6. Search for counterexamples and nearest prior art before elaborating a candidate.
7. Preserve strange-but-valid candidates in the portfolio.
8. Penalize generic templates such as "add attention", "make it adaptive", "use a neural network", or "make it multiscale" unless a specific new claim survives.
9. Let experiments, proofs, simulations, and repository evidence overrule rhetorical attractiveness.
10. Optimize a portfolio and Pareto frontier rather than a single scalar score.

# Workflow

## 1. Repository and frontier map

Inspect the actual repository before ideating. Read its architecture, current
methods, tests, reference benchmarks, datasets, and existing experiment code.
Do not treat the optional repository profile as authoritative when it conflicts
with files in the repo.

Construct a compact frontier map. For each important prior method or repository component, capture:

- problem;
- primitive objects and representation;
- assumptions;
- objective;
- core operator or causal mechanism;
- information available;
- optimization/inference procedure;
- evidence and benchmarks;
- failure modes and unresolved anomalies.

Also list:

- dominant paradigms;
- assumptions shared by most approaches;
- rarely questioned definitions;
- densely explored regions;
- sparse or poorly measurable regions;
- unexplained residuals and negative results.

Do not generate final ideas until this map exists.

## 2. Functional problem signature

Rewrite the target problem without domain-specific vocabulary. State:

- what enters the system;
- what hidden state must be inferred or controlled;
- what is transported, conserved, allocated, compressed, predicted, or optimized;
- what is local versus global;
- what is continuous versus discrete;
- what symmetries and boundary conditions exist;
- what uncertainty and partial observability exist;
- what limits identifiability, stability, quality, or compute.

Use this signature for cross-domain search.

## 3. Fixation and anti-library

List the default suggestions a competent LLM or conventional researcher is likely to produce. Include fashionable architectures, standard component swaps, obvious combinations, superficial uses of learning, routine loss changes, and saturated benchmark variants.

Candidates may use these components, but may not be labelled transformational unless a distinct scientific claim remains after subtracting them.

## 4. Parallel generation lanes

Generate the first pass in independent lanes. Do not let one lane collapse the others.

### Lane A — Productive recombination

Generate at least 4 useful N1/N2 candidates by combining complementary methods, transferring known techniques to unresolved limitations, or connecting theory with implementation.

### Lane B — Assumption surgery

Generate at least 4 candidates. For each major assumption, try to:

- remove or reverse it;
- make it local instead of global;
- make it dynamic instead of fixed;
- infer it as an output rather than provide it as an input;
- replace an object by a relation, operator, distribution, or equivalence class;
- exchange primal and dual viewpoints;
- inspect zero, singular, infinite, or asymptotic limits;
- model the failure of the assumption explicitly.

### Lane C — Primitive and grammar invention

Generate at least 4 candidates, including at least 2 that introduce a formal object, observable, operator, representation, or problem definition absent from the frontier map.

Ask:

- What is treated as fundamental but could be derived?
- What is derived but could instead be fundamental?
- Is the state space wrong?
- Can objects be replaced by transformations, measures, events, constraints, programs, flows, conserved quantities, or equivalence classes?
- Can a procedure be replaced by a characterization theorem or invariant?
- Can the output be specified implicitly instead of constructed?
- Is there a representation in which the apparent problem disappears?

### Lane D — New-evidence programs

Generate at least 3 investigations capable of producing evidence absent from the initial corpus. Read `references/evidence-generation.md`.

Each must state:

- what is varied;
- what is measured;
- what surprising outcome would matter;
- how the result could induce a new hypothesis;
- how bugs, leakage, or noise will be excluded.

### Lane E — Cross-domain mechanism transfer

This lane is mandatory. Read `references/cross-domain-transfer.md`.

Generate at least 6 candidates from at least 4 donor fields. At least 3 must come from fields not already commonly associated with the recipient field.

Transfer the underlying mechanism, measurement practice, invariant, decomposition, or failure theory—not just terminology or a black-box algorithm.

For every candidate provide an explicit donor-to-recipient map and identify:

- preserved causal mechanism;
- broken correspondence;
- required invention;
- adoption barrier;
- enabling technical or scientific change;
- recipient-specific prediction.

## 5. Transformation and transfer tests

Use `references/novelty-taxonomy.md` to apply:

- A-plus-B test;
- subtraction test;
- grammar test;
- prediction test;
- necessity test;
- compression test.

For cross-domain candidates also apply:

- terminology-removal test;
- structural/homomorphism test;
- causal-preservation test;
- counter-analogy test;
- native-baseline test;
- historical-obviousness test.

Downgrade or reject candidates that fail.

## 6. Independent adversarial prior-art audit

Switch roles. The evaluator must actively try to destroy each novelty claim and should not defend the generator's intent.

Read `references/prior-art-audit.md`. Search by exact terms, synonyms, mathematical form, functional description, older terminology, patents, theses, repositories, workshop work, and adjacent fields.

For transfers, search recipient, donor, and bridge fields.

Return nearest prior work and facet-level overlap. Classify each candidate as:

- likely known;
- known components, possibly new relationship;
- apparently unexplored;
- apparently transformational;
- insufficient evidence.

Include cutoff date, searched sources, and uncertainty.

## 7. Scientific-value audit

Score independently from 0–5:

- apparent novelty;
- falsifiability;
- explanatory value;
- expected scientific importance;
- feasibility;
- cost of first test;
- probability of interpretable results;
- availability of strong baselines;
- robustness to negative results;
- publication potential.

Do not collapse the scores too early. Keep a Pareto set containing, where possible:

- highest novelty;
- fastest validation;
- strongest theory direction;
- strongest systems direction;
- best high-risk/high-reward direction;
- best informative-negative-result direction.

## 8. Cheapest killing experiment

For every surviving N3, N3-T, N4, or N4-T candidate, design the smallest decisive test capable of showing the central claim is wrong.

Include:

- null hypothesis;
- signature if correct;
- signature under the strongest conventional explanation;
- minimal implementation/proof;
- required data and compute;
- decisive plots, metrics, or lemmas;
- confounders;
- abandonment criterion.

Prefer a two-day falsification test over a two-month implementation.

## 9. Final portfolio

Use `assets/research-portfolio.md` and `assets/idea-card.md`.

Minimum portfolio:

- 3–5 productive recombinations;
- 3–5 exploratory candidates;
- at least 3 transformational candidates;
- at least 2 new-evidence discovery programs;
- at least 4 cross-domain transfers;
- at least 2 transfers from rarely connected donor fields;
- at least 1 transfer focused on measurement, diagnostics, or experimental protocol rather than an algorithm;
- at least 1 N3-T candidate that changes the recipient formulation rather than merely its implementation.

For each idea include:

- title;
- central falsifiable claim;
- novelty class;
- known foundation;
- irreducible delta;
- why it is not merely A+B;
- changed grammar or preserved transfer mechanism;
- new prediction;
- cheapest killing test;
- prior-art threats;
- novelty confidence with cutoff and searched sources;
- publishable outcomes for success, partial success, and informative failure.

Conclude with a recommended **first experiment**, not a demand to implement the largest idea.

## 10. Optional deterministic check

When the output is saved as Markdown, run:

```bash
python scripts/validate_portfolio.py path/to/portfolio.md
```

The script checks structural completeness and prohibited certainty claims. It cannot verify scientific novelty.
