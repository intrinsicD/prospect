# P0-007 — Regression ratchet: shipped gates stay green

- **Status:** blocked (P0-006)
- **Phase:** P0
- **Requirements:** — (delivery infrastructure; makes ADR-0005 self-enforcing)
- **ADRs:** ADR-0005 (amend consequence)
- **Depends on:** P0-006
- **Phase gate:** `bench/gates.py::GATES["P0"]`

## Goal
Once a phase ships, its gate is re-run automatically and a regression turns CI red.
Today gates gate *shipping* but nothing gates *staying shipped*: CI runs only ruff and
pytest, so a P3-era change can silently regress a passed P1 gate — which undercuts the
entire benchmark-gated premise.

## Non-goals
- No new gates or eval bodies.
- No scheduling/nightly infrastructure — a plain CI job is enough while gates are
  cheap; revisit (via an ADR-0005 amendment) when a gate becomes expensive to re-run.

## Interface to satisfy
Harness + CI surface: `bench/SHIPPED`, `make gate-all`, a CI step.

## Approach (brief)
- `bench/SHIPPED` — one phase ID per line; a phase is appended in the same commit
  that records its passing gate (part of the docs-sync checklist from now on).
  Starts containing `P0` once P0-006's gate passes.
- `make gate-all` — runs `run_gate` for every shipped phase; exits nonzero if any
  report is BLOCKED; prints the composite summary.
- CI: add a `gate-all` step after tests. Because sentinels read the run log (P0-005),
  document the policy for gates whose evidence is a training artifact: re-run the
  *evaluation* against the persisted/regenerable artifact, not full retraining — and
  note in the report which mode ran.
- Amend ADR-0005: "gates are re-run for every shipped phase in CI; shipping appends
  to `bench/SHIPPED`."

## Acceptance criteria
- [ ] `bench/SHIPPED` exists and lists `P0`.
- [ ] `make gate-all` green locally; deliberately breaking a smoke test makes it fail
      (verified once, then reverted).
- [ ] CI workflow runs `make gate-all` and would fail on a BLOCKED shipped gate.
- [ ] The expensive-gate re-run policy is documented (gates.py docstring or ADR-0005).
- [ ] `make test` green, `make lint` clean.

## Test plan
- Unit: `gate-all` aggregation logic (all pass ⇒ exit 0; one BLOCKED ⇒ nonzero;
  unknown phase in SHIPPED ⇒ clear error).
- Manual: the break-one-test experiment above; CI run on the branch.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] ADR-0005 consequence amended.
- [ ] `tasks/TEMPLATE.md` docs-sync checklist gains "append phase to `bench/SHIPPED`
      when the gate passes".
- [ ] CLAUDE.md "Definition of done" unchanged? — check; amend if the ratchet adds a
      step.

## Gate result
_not run yet_
