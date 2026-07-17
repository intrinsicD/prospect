# CLAUDE.md — Prospect operating contract

Read `docs/architecture.md` before changing behavior.

## Semantic contract

- `Prediction` is a pre-outcome distribution with named target, horizon, and
  model/representation/calibration versions.
- `ProperScore` evaluates that immutable prediction after evidence arrives.
- `BeliefUpdate` links a prior, exact real experience, and posterior.
- `EpistemicEffect` is one named projection of that revision.
- `InformationValue` is prospective and goal-conditioned, not raw entropy.
- `UpdateReceipt` names the exact transitions consumed and persistent versions
  changed.
- `EvaluationRecord` is held-out evidence, not training telemetry.

The selected `DecisionRecord` must return through observation handling. Imagined
evidence has separate lineage and never enters the real-experience store. Do not
collapse uncertainty, surprise, information, reward, knowledge, and competence
into one scalar.

## Evidence contract

Collect, learn, improve, and retain are separate claims:

1. canonical experience is uniquely linked;
2. those exact identities drive a persistent version-changing update;
3. the frozen updated snapshot improves executed disjoint behavior against
   matched controls; and
4. the same gain survives genuine shared-state interference and fresh-process
   restoration.

Exact fixtures and in-process round trips may validate semantics, but not the
complete capability.

## Repository boundaries

- Put task-neutral contracts and runtime behavior in `src/prospect/`.
- Put reference problems and experiments in `bench/`.
- Put curated reusable inputs in `datasets/` with provenance and checksums.
- Put generated outputs in `bench/**/results/`; they are ignored by Git.
- Add typed protocols and adversarial tests for new public behavior.
- Keep `docs/architecture.md` aligned with stable system semantics.

Use `.agents/skills/prospect-research-ideation/SKILL.md` for unexplored research
directions and `.agents/skills/prospect-results-audit/SKILL.md` after quantitative
experiments or before capability claims.

## Experiment workflow

1. State one falsifiable claim and its cheapest killing criterion.
2. Freeze data, controls, budgets, seeds, metrics, and dependency versions.
3. Implement the smallest experiment under `bench/`.
4. Keep training and held-out evaluation identities disjoint.
5. Run the experiment, preserve raw outputs locally, and audit the result.
6. Update code, tests, and architecture only to the extent supported.

## Commands

```text
make install
make install-runtime
make test
make lint
make typecheck
make epistemic-diagnostics
make epistemic-gate
make check
```

## Definition of done

- [ ] `make check` passes.
- [ ] The experiment’s claim, controls, and abandonment rule were fixed before
      formal outcomes.
- [ ] Raw outcomes and source/data/config identities can be independently checked.
- [ ] Capability limits and blocked rows are reported explicitly.
- [ ] No unsupported generality, retention, or novelty claim remains.
