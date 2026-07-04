# P9-003 — Second environment + cross-environment generalization

- **Status:** blocked (P9-001)
- **Phase:** P9
- **Requirements:** R1, R4, R8 (do the capabilities generalize, or are they
  Pendulum-shaped?)
- **ADRs:** ADR-0005 (a capability is only real if it survives a second, structurally
  different task with the same code), ADR-0008
- **Depends on:** P0-004 (the `Environment` protocol), P1/P2/P8 (the load-bearing
  gates re-run), P9-001 (the E2E harness, re-run on env #2)
- **Phase gate:** contributes to `bench/gates.py::GATES["P9"]` — the
  "capabilities survive on a second environment" criterion.

## Goal
Add one structurally different environment to `bench/envs.py` and re-run the
load-bearing gates (P1 prediction, P2 planning, P8 retrieval) and the P9 E2E loop on
it with the **same core code** — only recalibrated thresholds. A capability that
survives is real; one that collapses was a Pendulum artifact. This is the direct
antidote to "everything is measured on one toy environment."

## Non-goals
- Not a realistic/hard benchmark — a *second, different* toy is enough to break
  single-environment overfit (e.g. a discrete gridworld or a different-dynamics
  continuous task). Minimal.
- No changes to `src/prospect/` core to make a gate pass on env #2; if a capability
  needs core changes to generalize, that is a **finding** → its own task.
- Not a full port of every gate — the load-bearing three (P1/P2/P8) + P9 E2E.

## Interface to satisfy
A new `Environment` (P0-004 protocol) in `bench/envs.py`, and env-parameterized
re-runs of the P1/P2/P8/P9 evals (harness-only; the core never learns which env it is
in). No new core `Protocol`.

## Approach (brief)
- Implement a second `Environment` with different state/dynamics (and, for P8, a
  region where the model is naturally uncertain).
- Parameterize the P1/P2/P8/P9 eval bodies by environment; run the same criteria with
  env-appropriate (recalibrated) thresholds.
- Assert each capability holds on env #2 on every seed. Where a threshold must move,
  record by how much (a large move is itself a fragility signal).

## Acceptance criteria
- [ ] A second `Environment` conforms to the protocol and is unit-tested.
- [ ] P1/P2/P8 capabilities + the P9 E2E loop hold on env #2 (recalibrated thresholds
      only, no core changes); results recorded.
- [ ] Any capability that fails to generalize is filed as a finding, not patched away.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (`tests/test_environment.py`): the second env's reset/step/(set_state) behave.
- Eval: `make gate PHASE=P9` includes the env-#2 re-runs.

## Docs-sync checklist
- [ ] Status → `done`; per-capability env-#2 results recorded below.
- [ ] architecture.md notes the two-environment validation; ADR-0008 updated.
- [ ] Backlog: P9-003 done.

## Gate result
_not run yet_
