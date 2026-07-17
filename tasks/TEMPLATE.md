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
- [ ] Confirmed last-run validation receipt recorded below.

## Test plan
<what tests/eval prove the criteria — unit + the gate>

## Confirmed last-run validation
- **Completed:** <ISO-8601 timestamp with timezone>
- **Source state:** <commit/tree identity or exact tested change-set description>
- **Environment:** <result-relevant runtime and dependency versions, or "not bound">
- **Commands and outcomes:**
  - `<exact command>` — <PASS/FAIL/SKIP and counts or decisive result>

Publishing/handoff agents reuse this receipt when the relevant source and environment
inputs are unchanged. They rerun only missing, incomplete, failed, invalidated, or
explicitly requested checks.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] Confirmed last-run validation receipt updated.
- [ ] If the phase gate newly passes: append the phase to `bench/SHIPPED` in the
      same commit (the ratchet re-runs it in CI, P0-007).
- [ ] Requirement traceability row still accurate.
- [ ] ADR status/consequences updated if anything changed.
- [ ] architecture.md updated if a contract or component changed.

## Gate result
<paste the GateResult once run>
