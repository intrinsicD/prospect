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
- On completion, do the **docs-sync** checklist and record the gate result here.

Copy `TEMPLATE.md` to start a new task. `P1-001` is the fully worked example.
