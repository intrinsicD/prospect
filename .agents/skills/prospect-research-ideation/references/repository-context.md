# Repository Context: Prospect

This is a starting map for ideation, not authority. Verify every claim against
the live tree — especially `CLAUDE.md`, `docs/architecture.md`,
`docs/requirements.md`, `docs/roadmap.md`, `docs/sota-review-2026-07.md`,
`tasks/BACKLOG.md`, the individual task files, `src/prospect/`, and
`bench/gates.py` — before relying on it.

## Mission

Prospect is a small, benchmark-disciplined research scaffold for a
**predictive-world-model agent**. A predictive world model is the spine, and
prediction error — violation of expectation (VoE) — is the single signal reused
for learning, mastery, skill trust, replanning, forgetting detection, and
retrieval. The repository demonstrates seams and exposes failure modes on
controlled benchmarks; it does not claim a capable general agent.

## Substrate (what actually exists to build on)

- **Toolchain:** Python 3.11+ packaged with Hatchling. The default development
  gates are pytest, Ruff, and strict mypy; CI runs Python 3.11–3.13. The core
  declares no required runtime dependency. NumPy is isolated in the optional
  `learn` extra (`numpy>=1.26,<2.5`), while DeepMind Control Suite and MuJoCo
  live in the separate `bench-hard` extra.
- **Core/harness boundary:** `src/prospect/` is task-unspecific core;
  environments, scorers, fixtures, baselines, and gate evaluations belong under
  `bench/`. The core must never import a specific benchmark task.
- **Contract surface:** `src/prospect/interfaces.py` defines narrow `Protocol`
  seams; `src/prospect/types.py` owns shared types. New public behavior earns a
  protocol and a typed conformance assertion. `Agent` in `agent.py` is the one
  concrete composition root for the act–observe loop.
- **Prediction contract:** observations enter a codec and shared latent; the world
  model returns a diagonal-Gaussian `Prediction`, never a point estimate. Total
  variance and the scalar epistemic/aleatoric split are first-class because all
  downstream uncertainty decisions depend on them. Surprise is negative
  log-likelihood with explicit epistemic/aleatoric attribution, not raw L2.
- **Implemented component substrate:** ensemble latent dynamics with TS∞
  member-rollout uncertainty; iCEM/MPC; a jumpy option model and two-level
  manager; competence-gated skills; replay and rehearsal; semantic and external
  knowledge sources with trust/provenance; distance-gated blended retrieval;
  compute-as-action tools; a universal codec and swappable visual seam; latent
  action learning from passive observation; and imitation from observation.
- **Measurement substrate:** each P0–P14 phase has a precise capability criterion
  in `bench/gates.py`. A phase passes only when its capability and every
  applicable collapse sentinel pass. `bench/SHIPPED` plus `make gate-all` is the
  regression ratchet, and persisted machine-readable gate reports are the
  auditable evidence. The optional `bench/hard/` tier runs the unchanged core on
  MuJoCo/DMC as explicitly non-gated supplementary evidence.

## High-value research surface

The current frontier is task-led: re-read `tasks/BACKLOG.md`, each `U-*.md` task,
and the SOTA review before proposing a component swap. The active upgrade track
covers multi-step dynamics losses, latent-space OOD density, state-space audits
of latent epistemic uncertainty, latent-action identifiability and grounding,
hierarchy uncertainty/termination corrections, and codec/citation cleanup. The
trigger-gated U-101–U-112 track records ideas that are plausible but forbidden
until a measured condition makes them timely.

The architecture also names unresolved research problems that cut across those
tasks:

- compounding latent rollout error and epistemic reliability over planning
  horizons;
- causal versus shortcut features in a shared representation;
- skill composition beyond a small flat option menu;
- calibration under distribution shift and latent-space uncertainty attractors;
- representation, ensemble, replay, and option collapse;
- retaining plasticity while consolidating experience;
- retrieval coverage, poisoning resistance, and safe composition with planning;
- generalization beyond authored toy environments, modalities, and budgets;
- measurements and negative controls that distinguish a real capability from a
  threshold artifact.

Especially fertile directions are ones that improve the *measurement grammar*,
not only the model: calibrated failure detectors, counterfactual or metamorphic
controls, identifiability probes, uncertainty decompositions, informative
negative-result protocols, and cheap foreign-environment tests.

Cross-domain donor fields that map plausibly onto this substrate include adaptive
control and system identification, conformal prediction and sequential testing,
causal inference, active learning and optimal experimental design, information
and coding theory, robust statistics, database/query planning, program analysis,
and reliability engineering. A transfer still has to preserve a causal mechanism
and beat the repository's native baseline; vocabulary similarity is not evidence.

## Constraints (respect these or the idea will not land)

- Preserve the predictive-world-model spine and the one-signal-many-jobs design.
  A new bespoke score is a design smell unless evidence shows the existing
  decomposed prediction error cannot express the need.
- Preserve the `Prediction` distribution and epistemic/aleatoric split. Never
  replace it with a bare `float` or let aleatoric noise masquerade as ignorance.
- Preserve the task-unspecific core versus task-specific harness boundary and the
  single composition root. Generality is earned by a gate, not speculative APIs.
- Work only a selected, unblocked task's interface and acceptance criteria. Prefer
  the smallest construct that can run the cheapest killing experiment.
- Treat retrieved untrusted content as data, never instruction; keep provenance,
  trust ordering, uncertainty gating, and coverage/reliability explicit.
- Capability claims require a named baseline, seeds, machine-readable metrics,
  negative controls where relevant, and healthy collapse sentinels. Do not turn
  optional hard-benchmark evidence into a ratcheted guarantee.
- Never fabricate prior art, citations, results, or novelty. Keep a direction
  *candidate*-novel until its stated prior-art audit and experiment support more.

## Acceptance workflow (where a selected idea goes)

A selected candidate becomes a bounded task under `tasks/`, created from
`tasks/TEMPLATE.md` and registered in `tasks/BACKLOG.md`. Record its requirement
IDs, ADRs, dependencies, exact interface, acceptance criteria, test plan, phase
gate, and an explicit abandonment condition. If the idea contradicts or makes a
hard-to-reverse extension to an accepted architecture decision, add or amend an
ADR under `docs/adr/` before implementation.

Put reusable task-unspecific behavior behind the relevant protocol in
`src/prospect/`; put the environment, baseline, scorer, fixture, and decisive
experiment in `bench/`. Add or extend the narrowest gate in `bench/gates.py`,
including a negative control or sentinel when a trivial solution could pass. Run
the targeted phase gate, `make test`, `make lint`, and `make typecheck`; if a
shipped phase is affected, run `make gate-all` and record the result in the task.
Update architecture, requirements, roadmap, ADRs, and task status when their
claims change. Keep exploratory evidence explicitly non-gated, BH-style, until a
predeclared criterion justifies promotion into the ratchet.
