# Adversarial Prior-Art and Novelty Audit

## Role separation

The auditor is an adversary, not an advocate. It should attempt to reconstruct or invalidate the candidate from prior work.

## Claim-facet decomposition

Audit separately:

- problem novelty;
- representation novelty;
- mechanism novelty;
- theoretical novelty;
- experimental novelty;
- combination novelty;
- cross-domain correspondence novelty;
- recipient-specific prediction novelty.

## Search layers

Search, when accessible:

- journal and conference papers;
- workshop and poster papers;
- preprints;
- theses and dissertations;
- patents;
- technical reports;
- books and older terminology;
- software repositories and documentation;
- issue discussions and experimental branches;
- adjacent fields and industrial practices.

For cross-domain candidates, search:

1. recipient field;
2. donor field;
3. bridge fields connecting them.

## Query expansion

Use:

- exact candidate terminology;
- synonyms and historical names;
- mathematical equations and objects;
- functional descriptions without field vocabulary;
- claimed mechanism;
- predicted outcome;
- failure mode;
- donor/recipient mapping;
- alternative motivations for the same mechanism.

## Nearest-work matrix

For each serious candidate, report a table such as:

| Prior work | Problem | Representation | Mechanism | Prediction | Evidence | Overlap threat |
|---|---:|---:|---:|---:|---:|---|

Attempt to reconstruct the candidate from one work, then from a combination of works. State the irreducible remainder.

## Audit labels

- **likely known** — central claim or mechanism already appears;
- **known components, possibly new relationship** — novelty may be in mapping or interaction;
- **apparently unexplored** — no close claim found under searched formulations;
- **apparently transformational** — changed grammar and distinct prediction survive the search;
- **insufficient evidence** — search coverage is too weak.

## Confidence report

Always include:

- cutoff date;
- databases and repositories searched;
- search terms or functional descriptions;
- sources not accessible;
- strongest prior-art threat;
- probability range, not false precision;
- reasons the range may be wrong.

## Novelty is not value

A candidate can be novel but trivial, infeasible, untestable, or unimportant. Keep novelty separate from scientific importance and feasibility.

## Historical backtest for skill evaluation

To evaluate this skill itself:

1. restrict sources to year `t`;
2. generate ideas;
3. compare with meaningful work from `t+1` to `t+3`;
4. measure anticipation of mechanisms and results, not keyword overlap;
5. assess feasibility, diversity, and whether the proposed killing tests would have been informative.
