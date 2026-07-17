# CLAUDE.md — Prospect operating contract

Prospect is an adaptive-agent research runtime. Read this file,
`docs/architecture.md`, `docs/requirements.md`, `docs/roadmap.md`, ADR-0014,
and the active task before editing.

## Semantic contract

A predictive model may anticipate consequences, but one scalar may not stand for
uncertainty, surprise, information, reward, knowledge, and competence.

- A `Prediction` is a pre-outcome distribution with a named target, horizon,
  model, representation, and calibration version.
- A `ProperScore` evaluates that immutable prediction after evidence arrives.
- A `BeliefUpdate` identifies the prior, exact real experience, and posterior.
- An `EpistemicEffect` names one measured projection of that revision.
- `InformationValue` is prospective and goal-conditioned; it is not raw entropy.
- An `UpdateReceipt` identifies the exact transitions consumed and persistent
  versions changed.
- An `EvaluationRecord` is external held-out evidence, not training telemetry.

The exact `DecisionRecord` must return through observation handling. Imagined
evidence has separate lineage and may never enter the real-experience store.
There is no universal `.epistemic` float or hidden “last prediction.”

## Evidence contract

Collect, learn, improve, and retain are separate claims:

1. collected experience must be canonical and uniquely linked;
2. those exact experience/transition identities must appear in a version-changing
   learner receipt;
3. the resulting frozen snapshot must improve executed held-out behavior at a
   matched budget; and
4. that same gain must remain above its pre-learning baseline after shared-state
   interference and a production checkpoint/process restart.

A numeric diagnostic can pass while its capability claim remains unsupported.
Never promote an exact fixture, analytic expectation, in-process round-trip, or
same-task posterior update into a full lifecycle result.

## Workflow

- Work only the active E-series task and its acceptance criteria.
- Put task-neutral contracts/runtime in `src/prospect/`; environments, datasets,
  and reference problems stay in `bench/`.
- New public behavior implements a typed protocol and receives adversarial tests.
- Amend an ADR before contradicting a locked decision.
- Run `.agents/skills/prospect-results-audit/SKILL.md` after every quantitative
  evidence session and before promoting a claim.
- Use `.agents/skills/prospect-research-ideation/SKILL.md` for novel research
  directions; selected ideas still require a task, protocol, and killing test.
- Keep authored source, tests, protocols, audit narratives, and checksum pointers
  in Git. Generated datasets, runtimes, model blobs, and result packages remain
  outside Git under the repository ignore policy.
- Record exact validation commands, source state, dependency versions, outcomes,
  and unresolved limits in the active task.

## Commands

```text
make install                 editable package and development tools
make install-runtime         additionally install TorchRL/TensorDict
make test                    active tests
make lint                    active Ruff checks
make typecheck               active strict Mypy checks
make epistemic-diagnostics  run exact predicates; not a capability pass
make epistemic-gate         full claim status; expected to fail while blocked
make check                   lint + typecheck + tests + diagnostics
```

## Active tree

- `src/prospect/domain/` — immutable records and backend-neutral protocols.
- `src/prospect/decision/` — candidate assessment and deterministic selection.
- `src/prospect/epistemics/` — exact reference information/scoring semantics.
- `src/prospect/runtime/` — authoritative decide/step/observe/learn lifecycle.
- `src/prospect/storage/` — canonical custody, replay adapter, checkpoint bundle.
- `bench/epistemic/` — exact development diagnostics.
- `tests/test_epistemic_*.py` — the active test surface.

The old flat modules, P-series tests, benchmark registry, and ratchet are not in
the active tree. Git history is their archive. Do not restore compatibility
imports merely to make obsolete tests pass.

## Definition of done

- [ ] Active protocol and acceptance criteria are satisfied.
- [ ] `make check` passes.
- [ ] The capability gate is reported truthfully, including blocked rows.
- [ ] Results receive an independent semantic audit.
- [ ] Task receipt and affected docs match the implementation.
- [ ] No unsupported capability, generality, retention, or novelty claim remains.
