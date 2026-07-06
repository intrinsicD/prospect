# Prospect

A small, disciplined research scaffold for a **predictive-world-model agent**: an
agent that plans by simulating the consequences of its actions, tests what it has
learned by measuring its own surprise, and adapts across tasks by *attaching*
knowledge rather than retraining.

> Working name — rename freely (`Test Engine → IntrinsicEngine` energy).

## The one load-bearing idea
A predictive world model is the spine, and **prediction error (violation of
expectation) is the single signal** that threads through learning, mastery-testing,
skill selection, re-planning, forgetting-detection and retrieval. Most requirements
are *consumers* of that core, not separate systems. See `docs/architecture.md`.

## Status — P0–P13 shipped; validated on toy benchmarks
The design is realized, not skeletal. There are working numpy implementations of the
flat world model (ensemble Gaussian latent dynamics with an epistemic/aleatoric split,
EMA target encoder, anti-collapse regularization, inverse-dynamics + reward heads),
CEM/MPC planning, the jumpy option-model and hierarchical manager, skills, replay,
semantic memory + uncertainty-gated retrieval, the universal codec, external knowledge +
compute-as-action tools, a swappable vision seam, and latent-action learning from
action-free observation — plus five standing **collapse sentinels** (representation,
uncertainty, replay, option, gate-overfit). `bench/SHIPPED` ratchets **P0–P13**;
`make gate-all` re-runs every kill-gate in CI.

**Honest scope.** This is a disciplined *research scaffold*, validated on controlled toy
benchmarks — pendulum, a 2D point-mass, and synthetic visual blobs with deterministic
stand-in encoders. It is a machine for proving seams and surfacing failure modes, **not**
evidence of a capable general agent. The next credibility jump needs harder environments,
real embeddings, and stronger baselines — not more phases. Generality is *earned by a
gate*, never assumed; code grows one benchmark-gated phase at a time (`docs/roadmap.md`,
`tasks/BACKLOG.md`), and a phase ships only when its gate **and** its collapse sentinels pass.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # recommended: work in a venv
make install     # editable install + dev tools
make test        # unit + conformance tests (must stay green)
make lint        # ruff
make typecheck   # mypy — full type hints are enforced
make gate PHASE=P1   # inspect a phase kill-gate
make gate-all    # re-run every shipped gate (the regression ratchet)
make tree        # see the layout
```

## Where to start (as a human or an agent)
1. Read `CLAUDE.md` — how work is done here.
2. Read `docs/architecture.md`, `docs/requirements.md`, `docs/roadmap.md`.
3. Take the top unblocked task in `tasks/BACKLOG.md`. Phases P0–P13 are shipped
   (`bench/SHIPPED`); the current front is P14 (observe→repeat). For worked examples,
   `tasks/P1-001-flat-world-model.md` shows an implementation phase and
   `tasks/P13-001-learn-from-observation.md` a recent one.

## Layout
```
CLAUDE.md            agent operating manual (start here)
docs/                architecture, requirements, roadmap
docs/adr/            locked architecture decisions
tasks/               backlog, task template, specified tasks
src/prospect/        task-unspecific CORE (interfaces + one file per component)
bench/               task-specific HARNESS: kill-gates (the fitness function) + the Environment seam
tests/               unit + conformance tests
```
