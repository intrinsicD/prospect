# Evidence-Generating Discovery Programs

## Purpose

The strongest route beyond corpus recombination is:

```text
existing knowledge -> intervention -> new observation -> new hypothesis
```

A proposed experiment is not itself N4. N4 requires an observation actually produced by the experiment and a hypothesis that depends on it.

## Candidate procedures

- exhaustive search on small instances;
- automated counterexample generation;
- symbolic regression over measured residuals;
- theorem, lemma, or invariant search;
- differentiable or high-fidelity simulation;
- random program or operator generation;
- quality-diversity search rather than single-objective optimization;
- adversarial data or geometry generation;
- extreme-regime and singular-limit tests;
- high-precision measurement of unexplained residuals;
- ablation grids designed to expose interactions;
- synthetic universes with altered assumptions;
- active experiment selection maximizing discrimination between theories;
- automated search for violations of assumed invariants.

## Required program card

For every program state:

### Search space

What varies? Include representations, operators, topologies, parameters, data-generating processes, or proof objects.

### Observable

What is measured that current methods do not usually measure?

### Conventional expectation

What should happen under the strongest existing explanation?

### Surprising signature

What reproducible pattern would require a new explanation?

### Promotion rule

What result is sufficient to promote the observation into a hypothesis or new research lineage?

### Falsification and controls

How will the procedure exclude:

- implementation bugs;
- numerical instability;
- data leakage;
- benchmark artifacts;
- random multiple-testing discoveries;
- uncalibrated uncertainty;
- simulator mismatch;
- invalid proof assumptions?

### Reproduction package

Specify seeds, versions, data provenance, test cases, tolerances, and negative controls.

## Productive failure

Design programs where a negative result still yields:

- a mapped impossibility region;
- a counterexample dataset;
- a diagnostic or benchmark;
- a bound;
- a taxonomy of failure modes;
- a refined assumption;
- a reusable experimental tool.

## Open-ended archive

When possible, preserve diverse discoveries by behavioral descriptors rather than only top score. Useful descriptors include:

- theory versus systems;
- local versus global;
- discrete versus continuous;
- deterministic versus probabilistic;
- representation-changing versus objective-changing;
- compute cost;
- failure type;
- evidence produced.

Generate mutations from underexplored archive regions. Do not repeatedly mutate only the current best candidate.
