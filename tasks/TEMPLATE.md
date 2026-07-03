# P?-??? — <title>

- **Status:** backlog
- **Phase:** P?
- **Requirements:** R?
- **ADRs:** ADR-????
- **Depends on:** <task ids or "none">
- **Phase gate:** `bench/gates.py::GATES["P?"]`

## Goal
<one or two sentences: what capability exists after this task>

## Non-goals
<explicitly what this task does NOT do — protects against scope creep>

## Interface to satisfy
<the `Protocol` in `src/prospect/interfaces.py` this must implement, and where the
implementation lives>

## Approach (brief)
<the intended method, referencing architecture.md / ADRs; keep it minimal>

## Acceptance criteria
- [ ] Implements the named interface.
- [ ] <measurable criterion 1>
- [ ] Phase gate criterion met (or measurably advanced): <restate the gate>
- [ ] `make test` green, `make lint` clean.

## Test plan
<what tests/eval prove the criteria — unit + the gate>

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] If the phase gate newly passes: append the phase to `bench/SHIPPED` in the
      same commit (the ratchet re-runs it in CI, P0-007).
- [ ] Requirement traceability row still accurate.
- [ ] ADR status/consequences updated if anything changed.
- [ ] architecture.md updated if a contract or component changed.

## Gate result
<paste the GateResult once run>
