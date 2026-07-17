# Tasks

One task = one unit of agent work, small enough to finish and gate. Tasks are the
handoff between the roadmap and the code.

## Lifecycle (Status field)
`backlog` → `ready` (unblocked, specified) → `in-progress` → `review` → `done`.
Use `blocked` with a reason when a dependency isn't met.

## Conventions
- ID: `P<phase>-NNN` (e.g. `P1-001`). Filename `P1-001-short-slug.md`.
- Every task links its **Requirements (R#)**, **ADRs**, **Depends on**, and its
  **Phase gate**. If work reveals the ADR is wrong, amend the ADR first.
- A task's scope is *exactly* its interface + acceptance criteria. No scope creep.
- On completion, do the **docs-sync** checklist, record the gate result, and add the
  confirmed last-run validation receipt (timestamp, tested source state, relevant
  environment, exact commands, and outcomes).
- Handoff/publishing work must reuse a still-valid receipt instead of retesting.
  Rerun only if the receipt is absent/incomplete, relevant source or environment
  inputs changed, the recorded run failed, or the user explicitly requests it.

Copy `TEMPLATE.md` to start a new task. `P1-001` is the fully worked example.
