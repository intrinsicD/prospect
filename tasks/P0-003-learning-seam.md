# P0-003 — A learning seam: `Learner` protocol

- **Status:** ready
- **Phase:** P0
- **Requirements:** R1, R7
- **ADRs:** ADR-0001, ADR-0005 (the harness must be able to drive training generically)
- **Depends on:** none (coordinate with P0-002, which also touches `Transition` docs)
- **Phase gate:** `bench/gates.py::GATES["P0"]` (registered by P0-006)

## Goal
A uniform seam through which the harness drives training. Every protocol in
`interfaces.py` is inference-only (`predict`, `plan`, `propose`, `query`), yet P1-001
says "train from a replay of transitions", and the P7 continual-learning gate must run
training *across a task sequence* without implementation-specific APIs. Without this
seam, each phase's gate eval entangles itself with one implementation's training loop.

## Non-goals
- No training loop, no optimizer, no model — this is the contract only.
- No trainer/scheduler abstraction, no callbacks (speculative generality).

## Interface to satisfy
New `interfaces.Learner` protocol; the `FlatWorldModel` skeleton grows a matching
`update` stub (raising `NotImplementedError("P1-001")`).

## Approach (brief)
- Add to `interfaces.py`:

  ```python
  @runtime_checkable
  class Learner(Protocol):
      """A component the harness can train. update() consumes a batch of transitions
      and returns a metrics dict (losses + integrity stats) — the harness logs these
      to the run-metrics artifact (P0-005) that the sentinels read."""
      def update(self, batch: Sequence[Transition]) -> dict[str, float]: ...
  ```

- Keep it a *separate* protocol rather than widening `WorldModel` — inference-only
  consumers (the planner) keep a narrow view; trainable components satisfy both
  structurally. Docstring names who is expected to satisfy it per phase: world model
  (P1), option model (P5), codec (P6).
- The `dict[str, float]` return is deliberate: it is the bridge to the sentinel
  run-log (P0-005) — training metrics flow out through the same call that trains.

## Acceptance criteria
- [ ] `interfaces.Learner` exists, `runtime_checkable`, with the docstring above.
- [ ] `FlatWorldModel` skeleton satisfies `Learner` structurally (smoke-tested).
- [ ] `tasks/P1-001` updated: `FlatWorldModel` must satisfy `WorldModel` **and**
      `Learner`.
- [ ] `make test` green, `make lint` clean.

## Test plan
- Smoke: `isinstance(FlatWorldModel(), interfaces.Learner)`; protocol registered in
  the conformance test alongside the others.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] `docs/architecture.md` components section notes the training seam.
- [ ] `tasks/P1-001` interface section updated.
- [ ] Requirement rows R1/R7 still accurate.

## Gate result
_not run yet_
