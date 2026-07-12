# Cross-Domain Mechanism Transfer

## Goal

Search for mature mechanisms, representations, invariants, diagnostics, optimization principles, and experimental practices from distant fields that are absent or conceptually unrecognized in the recipient field.

A donor method being known does not make the transfer incremental. Novelty may lie in the structural correspondence, recipient-specific adaptation, changed formulation, or new prediction.

## 1. Build a domain-neutral functional signature

Describe the target without its native nouns. Include:

- state;
- observations;
- actions/operators;
- hidden variables;
- constraints and boundary conditions;
- objective or energy;
- invariants and symmetries;
- noise and uncertainty;
- local/global coupling;
- timescales;
- characteristic failure modes.

Example:

Instead of "fit Gaussian primitives to images", write "infer a structured latent state from incomplete noisy projections while preserving local consistency and enabling efficient forward synthesis".

## 2. Search donor families

Cover at least six plausible families and choose at least four for candidates. Do not select only fashionable areas.

Potential donor families:

- statistical physics and phase transitions;
- control theory and system identification;
- coding and information theory;
- operations research and scheduling;
- databases, query optimization, and streaming systems;
- compiler construction and program analysis;
- cryptography and error detection;
- causal inference and experimental design;
- ecology and evolutionary dynamics;
- economics, market design, and distributed pricing;
- neuroscience and predictive processing;
- materials science and defect theory;
- fluid dynamics and conservation laws;
- formal methods and abstract interpretation;
- topology and sheaf-like consistency;
- optimal transport;
- signal processing;
- robotics and active perception;
- numerical analysis and multigrid;
- computational biology.

Search by functional behavior, not only method name. Examples:

- systems that reconstruct hidden state from partial projections;
- systems that correct corruption using structured redundancy;
- systems that coordinate local decisions without central control;
- systems stable under delayed or uncertain feedback;
- systems that allocate finite precision adaptively;
- systems that derive global consistency from local observations;
- systems that discover conserved quantities from trajectories.

Inspect textbooks, older terminology, industrial practices, obsolete methods whose hardware limitations have changed, explanatory theories not operationalized as algorithms, and measurement protocols rather than only algorithms.

## 3. Remove donor terminology

Describe the donor mechanism in domain-neutral terms:

- state;
- transformation;
- invariant;
- feedback;
- update rule;
- convergence mechanism;
- robustness source;
- efficiency source.

Reject proposals that cannot remain precise without donor buzzwords.

## 4. Explicit transfer map

Use `assets/transfer-map.md`. Map at least four structurally meaningful roles:

| Structural role | Donor | Recipient |
|---|---|---|
| state | | |
| observation | | |
| action/operator | | |
| objective/energy | | |
| invariant | | |
| noise model | | |
| boundary condition | | |
| failure mode | | |

## 5. Identify the causal mechanism

State exactly why the donor method works, such as:

- negative feedback stabilizes a dynamic state;
- redundancy permits error correction;
- dual variables expose hidden constraints;
- local competition induces global organization;
- conserved flux constrains evolution;
- randomization avoids adversarial degeneracy;
- an information bottleneck removes nuisance variables;
- hierarchical decomposition separates timescales;
- pricing coordinates distributed resource allocation.

Then demonstrate that the same mechanism, or a valid substitute, exists in the recipient problem.

## 6. Adapt rather than copy

State:

- donor assumptions that fail;
- variables without direct counterparts;
- mathematical structure requiring generalization;
- components that must not be transferred;
- new recipient observables, states, losses, operators, or protocols required.

The mismatch is often the source of the actual invention.

## 7. Derive a recipient-specific prediction

The transfer must imply more than black-box performance. Seek:

- a scaling law;
- invariant or conservation relation;
- phase transition;
- failure boundary;
- equivalence between formulations;
- optimality condition;
- measurable diagnostic;
- algorithmic regime where it should dominate.

Without this, classify at most N2-T.

## Transfer tests

### Terminology-removal test

Remove donor vocabulary. Is the mechanism still precise?

### Structural/homomorphism test

Do the operations responsible for donor success correspond under the mapping? Conceptually ask whether `T(f_donor(x))` behaves like `f_recipient(T(x))` for the important operations.

### Causal-preservation test

Does the reason the donor method works survive the transfer? If not, identify a replacement mechanism or reject it.

### Counter-analogy test

List at least three important mismatches. Mark each harmless, correctable, scientifically productive, or fatal.

### Native-baseline test

Compare against the strongest native recipient method. Explain which repeated limitation the donor mechanism addresses.

### Historical-obviousness test

Could a competent researcher have proposed the transfer ten years ago? If so, why was it not adopted, and what has changed now?

## Mutation operators

- **Mechanism without implementation:** preserve the principle, invent a new implementation.
- **Implementation without interpretation:** transfer the procedure, derive a new recipient theory.
- **Dual transfer:** inspect both donor primal and dual formulations.
- **Failure-mode transfer:** import a mature failure theory or diagnostic, not the solution.
- **Measurement transfer:** import an observable, experimental design, or evaluation protocol.
- **Invariant transfer:** seek a recipient analogue of a conservation law, symmetry, potential, sufficient statistic, or order parameter.
- **Architecture transfer:** import a decomposition into hierarchy, modules, timescales, local/global, primal/dual, or fast/slow variables.
- **Reverse transfer:** ask whether the recipient field suggests a donor improvement.
- **Broken-analogy invention:** treat the key mismatch as the source of a new mechanism.

## Prior-art layers

Audit:

1. recipient-field use of the donor idea;
2. donor-field standard formulations;
3. bridge fields where the same transfer may already have occurred under different terminology.

Report separately:

- donor-method novelty;
- correspondence novelty;
- adaptation novelty;
- prediction novelty.
