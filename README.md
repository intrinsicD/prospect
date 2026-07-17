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

## Status — P0–P14 shipped; validated on toy benchmarks
The design is realized, not skeletal. There are working numpy implementations of the
flat world model (ensemble Gaussian latent dynamics with an epistemic/aleatoric split,
EMA target encoder, anti-collapse regularization, inverse-dynamics + reward heads),
iCEM/MPC planning, the jumpy option-model and hierarchical manager, skills, replay,
semantic memory + uncertainty-gated retrieval, the universal codec, external knowledge +
compute-as-action tools, a swappable vision seam, latent-action learning from
action-free observation, and imitation from observation (observe→repeat) — plus five
standing **collapse sentinels** (representation, uncertainty, replay, option, gate-overfit).
`bench/SHIPPED` ratchets **P0–P14**; `make gate-all` re-runs every kill-gate in CI.

**Honest scope.** This is a disciplined *research scaffold*, validated on controlled toy
benchmarks — pendulum, a 2D point-mass, and synthetic visual blobs with deterministic
stand-in encoders. It is a machine for proving seams and surfacing failure modes, **not**
evidence of a capable general agent. The next credibility jump needs harder environments,
real embeddings, and stronger baselines — not more phases. A first step is in the repo: an
**optional, non-gated** harder-benchmark tier (`make bench-hard`, ADR-0011) that runs the
*unchanged* core on real DeepMind Control Suite (MuJoCo) tasks through the `Environment`
seam and reports honestly — including where the toy wins do *not* reproduce. It is fenced
off from the numpy-only gated CI (the `[bench-hard]` extra), so generality stays *earned by
a gate*, never assumed; code grows one benchmark-gated phase at a time (`docs/roadmap.md`,
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
3. Take the top unblocked task in `tasks/BACKLOG.md`. Phases P0–P14 are shipped
   (`bench/SHIPPED`). For worked examples, `tasks/P1-001-flat-world-model.md` shows an
   implementation phase and `tasks/P14-001-observe-repeat-imitation.md` a recent one.
4. For novel research directions, load
   `.agents/skills/prospect-research-ideation/SKILL.md`; route any selected idea
   through a task, ADR when needed, and a benchmark gate before implementation.
5. Before accepting results or promoting a claim, phase, or default, load
   `.agents/skills/prospect-results-audit/SKILL.md` for an independent scientist pass.

## Layout
```
CLAUDE.md            agent operating manual (start here)
.agents/skills/       project-scoped research-ideation and results-audit workflows
docs/                architecture, requirements, roadmap
docs/adr/            locked architecture decisions
tasks/               backlog, task template, specified tasks
src/prospect/        task-unspecific CORE (interfaces + one file per component)
bench/               task-specific HARNESS: kill-gates (the fitness function) + the Environment seam
artifact-pointers/   checksums and retrieval metadata for externally stored artifacts
tests/               unit + conformance tests
```
