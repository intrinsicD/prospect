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

## This repo is intentionally minimal
It ships the seams (typed `Protocol` contracts), the design (`docs/`, `docs/adr/`),
a benchmark-gated plan (`docs/roadmap.md`, `bench/gates.py`), and skeletons that
import cleanly and pass a smoke test — **not** a speculative implementation. Code
grows one benchmark-gated phase at a time.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # recommended: work in a venv
make install     # editable install + dev tools
make test        # smoke tests (green from commit one)
make lint        # ruff
make typecheck   # mypy — full type hints are enforced
make gate PHASE=P1   # inspect a phase kill-gate
make gate-all    # re-run every shipped gate (the regression ratchet)
make tree        # see the layout
```

## Where to start (as a human or an agent)
1. Read `CLAUDE.md` — how work is done here.
2. Read `docs/architecture.md`, `docs/requirements.md`, `docs/roadmap.md`.
3. Take the top unblocked task in `tasks/BACKLOG.md`. The Phase-0 contract-hardening
   tasks (`tasks/P0-001` … `P0-011`) come first and are fully specified;
   `tasks/P1-001-flat-world-model.md` is the worked example for implementation phases.

## Layout
```
CLAUDE.md            agent operating manual (start here)
docs/                architecture, requirements, roadmap
docs/adr/            locked architecture decisions
tasks/               backlog, task template, specified tasks
src/prospect/        task-unspecific CORE (interfaces + one file per component)
bench/               task-specific HARNESS: kill-gates (the fitness function) + the Environment seam
tests/               smoke tests
```
