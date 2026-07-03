# P0-002 — Decomposed `Surprise` type + per-skill transition attribution

- **Status:** blocked (P0-001)
- **Phase:** P0
- **Requirements:** R3, R5, R7
- **ADRs:** ADR-0002 (amend contract consequence)
- **Depends on:** P0-001
- **Phase gate:** `bench/gates.py::GATES["P0"]` (registered by P0-006)

## Goal
The VoE signal itself carries the epistemic/aleatoric split, and a `Transition` can
name the option/skill it was collected under. Today `CompetenceMonitor.surprise()`
returns a bare `float` — the exact ambiguity ADR-0002 exists to remove (raw error
cannot distinguish "unlearned" from "noisy") — and `update(transition)` has no way to
attribute experience to a skill, though competence is tracked per skill.

## Non-goals
- No implementation of *how* the split is computed — that is P3-001's job. This task
  only fixes the seam so P3-001 has somewhere correct to put it.
- No curiosity/curriculum logic (P3-002).

## Interface to satisfy
`interfaces.CompetenceMonitor` (signature change) and `types.Transition` /
new `types.Surprise`. The `SurpriseCompetenceMonitor` skeleton in `voe.py` updates to
match (still raising `NotImplementedError("P3-001")`).

## Approach (brief)
- New frozen dataclass `types.Surprise`: `total: float` (the NLL),
  `epistemic: float`, `aleatoric: float` (attribution of the error). Docstring states
  the discipline: consumers that gate on "is this reducible?" read `.epistemic`
  (mastery, curiosity, retrieval); the noisy-TV defense in ADR-0006 depends on this.
- `CompetenceMonitor.surprise(prediction, observed) -> Surprise` (was `-> float`).
- `Transition` gains `option: Option | None = None` — set when the transition was
  collected while executing a skill, so `update()` can attribute it.
- Amend ADR-0002's contract consequence: "surprise is a `Surprise` (decomposed),
  never a bare float — the same rule as `Prediction`, one level up."

## Acceptance criteria
- [ ] `types.Surprise` exists with `total`, `epistemic`, `aleatoric`.
- [ ] `CompetenceMonitor.surprise` returns `Surprise`; skeleton and smoke tests match.
- [ ] `Transition.option` exists, defaults to `None`, and is documented.
- [ ] ADR-0002 consequence amended.
- [ ] `make test` green, `make lint` clean.

## Test plan
- Unit: `Surprise` instantiates and is frozen; `Transition(option=...)` round-trips;
  smoke protocol-conformance check still passes with the new signature.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] ADR-0002 contract consequence amended.
- [ ] `docs/architecture.md` "one signal, many jobs" section notes the decomposed
      return type.
- [ ] Backlog rows for P3-001 / P4-001 reference the new seam.

## Gate result
_not run yet_
