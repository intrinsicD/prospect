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
make install     # editable install + dev tools
make test        # smoke tests (green from commit one)
make lint        # ruff
make gate PHASE=P1   # inspect a phase kill-gate
make tree        # see the layout
```

## Where to start (as a human or an agent)
1. Read `CLAUDE.md` — how work is done here.
2. Read `docs/architecture.md`, `docs/requirements.md`, `docs/roadmap.md`.
3. Take the top unblocked task in `tasks/BACKLOG.md`. The first one,
   `tasks/P1-001-flat-world-model.md`, is fully specified as the worked example.

## Layout
```
CLAUDE.md            agent operating manual (start here)
docs/                architecture, requirements, roadmap
docs/adr/            locked architecture decisions
tasks/               backlog, task template, specified tasks
src/prospect/        task-unspecific CORE (interfaces + one file per component)
bench/               task-specific HARNESS: the kill-gates (the fitness function)
tests/               smoke tests
```
