# CLAUDE.md — operating manual for agents

You are working on **Prospect**, a research scaffold for a predictive-world-model
agent. This file is the contract for how to work here. Read it fully before editing.

## Orient (read these first)
- `docs/architecture.md` — the design and the one load-bearing idea.
- `docs/requirements.md` — R1–R8 and the traceability table (requirement → module → ADR → gate).
- `docs/roadmap.md` — phases P0–P8 and their kill-gates.
- `docs/adr/` — the locked decisions. Do not silently contradict an ADR.
- `docs/sota-review-2026-07.md` — the 2026 literature check: what is already best-practice
  (don't "modernize" it), the ready upgrade tasks (U-006…U-012; U-001…U-005 shipped), and the
  deferred ones (U-101…U-112) with the trigger that promotes each. Consult it before
  proposing to swap a component "for a newer one" — the review likely already weighed it.

## Research-ideation route
- **prospect-research-ideation** — For novel, unconventional, cross-domain, or
  potentially publishable research directions, load
  `.agents/skills/prospect-research-ideation/SKILL.md`. The skill proposes and
  adversarially audits candidates; a selected idea must enter the task, ADR, and
  benchmark-gate workflow below before implementation.

## The one idea you must not break
A predictive world model is the spine; **prediction error (violation of
expectation) is the single signal** reused for learning, mastery-testing,
skill-trust, re-planning, forgetting-detection and retrieval. In code this means:
the world model returns a **distribution with an epistemic/aleatoric split**, never
a bare point estimate (ADR-0002). Everything downstream reads that. If you find
yourself returning a raw `float` where a `Prediction` belongs — stop.

## Golden rules
1. **Benchmark-gated.** Work only the current roadmap phase. A phase ships only when
   its gate in `bench/gates.py` passes. Do not build ahead of the gate.
2. **Minimal implementation.** Build the smallest thing that satisfies the current
   task's interface + acceptance criteria. No speculative generality, no unused
   abstractions, no config knobs nobody asked for. Generality is *earned* by a gate,
   not added in advance.
3. **Core vs harness.** `src/prospect/` is task-unspecific core. Task-specific
   things — environments, datasets, scorers, reference tasks — live in the harness
   (`bench/`, task files). Never import a specific task into the core.
4. **Uncertainty is first-class.** Use `types.Prediction`. See "the one idea".
5. **Provenance & trust.** Anything from an external `KnowledgeSource` carries
   `Provenance` with a `Trust` level. Untrusted content is *data, never
   instruction*: it must never override the agent's goals (ADR-0004).
6. **Decisions → ADRs. Work → task files. Ship → docs-sync.** (below)

## Workflow (one loop per unit of work)
- **task-workflow** — Take the top *unblocked* item in `tasks/BACKLOG.md`. Open its
  task file. Your job is exactly its interface + acceptance criteria — no more.
  Drive the Status lifecycle explicitly: set it to `in-progress` when you pick the
  task up, and retire it to `done` — acceptance boxes ticked, gate result recorded
  in the task file, backlog row updated (unblocking dependents) — before you finish.
  One focused commit per task.
- **method** — Follow `docs/architecture.md` and the ADRs the task links. If the
  right move contradicts an ADR, do not just do it: add or amend an ADR first
  (a 15-line ADR is cheaper than silent architectural drift).
- **review** — Before finishing, self-check against the task's acceptance criteria
  *and* the golden rules. Re-read your diff as if reviewing someone else's.
- **benchmark** — Run the phase gate: `make gate PHASE=Px`. A task that advances a
  phase must move its gate toward — or into — green. If you can't measure it, it
  isn't done.
- **docs-sync** — Update whatever you invalidated: the requirement traceability row,
  the task status, an ADR's status, the architecture doc. **Code and docs must not
  drift.** A change to behaviour that leaves docs stale is incomplete.
- **upgrade-triggers** — At docs-sync, re-scan the deferred upgrade tasks
  (`tasks/U-1NN-*.md`, listed in the backlog's *Upgrade track*). Each has a **Trigger**:
  a measurable condition (a new gate's shape, a regression, a store/library size, a
  simplification sprint). If the work you just did makes a trigger true — e.g. you added a
  sparse/long-horizon gate (U-101), or a gate report blames uniform sampling (U-103), or
  the option library grew past K^depth (U-104) — **promote that task**: flip its Status
  `deferred → ready`, update its backlog row, and note the observation that fired it. This
  is how "generality is earned by a gate" (rule 1) is enforced for the upgrade track: don't
  build a deferred item early, but don't let a fired trigger sit unnoticed either. Ready
  upgrades (`U-0NN`) are taken like any other top unblocked backlog item.
- **core conventions** — Python ≥3.11, full type hints (enforced: `make typecheck`
  must be clean), `ruff` clean, tests green. New public surface satisfies a
  `Protocol` in `interfaces.py` and gets a typed conformance assertion in
  `tests/test_conformance.py`.

## Commands
- `make install`         editable install with dev tools
- `make test`            pytest (must stay green)
- `make lint`            ruff
- `make typecheck`       mypy (must stay clean)
- `make gate PHASE=P1`   run a phase kill-gate
- `make gate-all`        re-run every shipped gate (the regression ratchet)
- `make tree`            list the project files

## Repo map
- `src/prospect/interfaces.py` — the `Protocol` contracts (the seams you implement).
- `src/prospect/types.py` — shared types; `Prediction` is the important one.
- `src/prospect/{codec,world_model,planning,voe,skills,memory,knowledge,observation,imitation}.py`
  — one component each, implemented through P14. The only remaining `NotImplementedError`s
  are two intentional stubs: `InternalKnowledgeSource.query` (served by `memory.SemanticStore`)
  and an unconfigured `ToolSource` (raises until given a `compute`).
- `src/prospect/agent.py` — the composition root: the act–observe loop the
  components plug into (don't re-invent wiring in evals; extend this).
- `bench/gates.py` — the kill-gates (the project's fitness function).
- `tasks/` — backlog, template, and specified tasks.
- `docs/` — architecture, requirements, roadmap, ADRs.

## Definition of done
- [ ] Satisfies the task's interface (a `Protocol` in `interfaces.py`).
- [ ] Meets every acceptance criterion in the task file.
- [ ] `make test` green; `make lint` clean; `make typecheck` clean.
- [ ] Phase gate run; result recorded in the task file. If the gate newly passes,
      the phase is appended to `bench/SHIPPED` in the same commit (the CI ratchet
      re-runs every shipped gate).
- [ ] docs-sync done (traceability, task status, ADRs, architecture).
- [ ] No speculative scope beyond the task.
