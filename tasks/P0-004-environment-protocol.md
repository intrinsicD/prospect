# P0-004 — `Environment` protocol in the harness

- **Status:** ready
- **Phase:** P0
- **Requirements:** R1 (the P1/P2 gates need a reference task behind one seam)
- **ADRs:** ADR-0005; golden rule 3 (core vs harness)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P0"]` (registered by P0-006)

## Goal
A minimal `Environment` contract owned by the harness, so gate evals and training
loops target one seam. P1-001 already needs "a toy control task (harness)", but no
environment contract exists anywhere — without it, every phase invents its own
env-wiring, and wiring is where architecture erodes.

## Non-goals
- No real environment implementation — the toy control task ships *with* P1-001.
- No gym/gymnasium dependency; same shape, stdlib-only.
- No agent-loop composition root — that gets a home at P2 (backlog item P2-002).
- Nothing in `src/prospect/` — environments are task-specific (golden rule 3).

## Interface to satisfy
New `bench/envs.py` defining the protocol (the harness may import core types; the
core must never import the harness):

```python
@runtime_checkable
class Environment(Protocol):
    def reset(self, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool]: ...
        # -> (observation, reward, done)
```

## Approach (brief)
- Define the protocol in `bench/envs.py` with the docstring stating the import
  direction rule and that `seed` is mandatory plumbing (gate criteria quantify "over
  N seeds"; reproducibility starts at the env).
- A ~10-line dummy environment in the test file (not shipped) verifies the seam is
  usable end-to-end with core types.
- Add an import-direction guard test: no module under `src/prospect/` imports `bench`.

## Acceptance criteria
- [ ] `bench/envs.py` exists; `Environment` is `runtime_checkable`; exported from
      `bench/__init__.py`.
- [ ] A dummy env in the tests satisfies the protocol and steps with core types.
- [ ] Import-direction test: core never imports the harness.
- [ ] `tasks/P1-001` updated: the toy task implements `bench.Environment`.
- [ ] `make test` green, `make lint` clean.

## Test plan
- Smoke: dummy env `isinstance` check; one reset/step round-trip with `Observation` /
  `Action`; the import-direction scan over `src/prospect/`.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] `docs/architecture.md` (or README layout note) mentions the harness owns
      `Environment`.
- [ ] `tasks/P1-001` test-plan section references the seam.

## Gate result
_not run yet_
