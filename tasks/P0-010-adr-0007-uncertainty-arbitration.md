# P0-010 — ADR-0007: arbitration of the epistemic signal (seek vs avoid)

- **Status:** ready
- **Phase:** P0
- **Requirements:** R1, R3, R7 (consumers of the one signal that currently conflict)
- **ADRs:** new ADR-0007; ADR-0002 (add consequence)
- **Depends on:** none (docs-only; can run in parallel with everything)
- **Phase gate:** `bench/gates.py::GATES["P0"]` (docs task — gate is `make test` green
  + docs consistent)

## Goal
Resolve, on paper and before P2/P3 build it, the tension between two consumers of the
same epistemic signal: ADR-0006 makes planning **avoid** high epistemic uncertainty
(MOPO-style uncertainty-penalized rollouts), while the P3 curiosity curriculum
**seeks** it. Both are individually correct; unarbitrated, they fight over the one
signal the architecture unifies — and this is a design decision, not an
implementation detail.

## Non-goals
- No code. No planner or curriculum implementation (P2-001 / P3-002).
- No new signal — the whole point is that both consumers read the same one.

## Interface to satisfy
None — documentation task. Deliverables: `docs/adr/0007-*.md`, ADR-0002 amendment,
architecture.md note.

## Approach (brief)
- **ADR-0007 decision sketch (to be finalized in the ADR):** the sign applied to
  epistemic uncertainty is **mode-dependent, chosen by the curriculum, never by the
  consumer**. Exploitation-mode planning (acting for external reward) applies the
  uncertainty *penalty* (ADR-0006's model-exploitation control). Exploration-mode
  data collection (curiosity, P3) applies the uncertainty *bonus*. One arbiter — the
  curriculum/learning-progress logic — decides the mode per episode/rollout; planner
  and explorer are consumers of a mode flag, not owners of the sign.
- **ADR-0002 amendment (shift disambiguation):** under distribution shift, the
  forgetting detector (job 5) and the retrieval trigger (job 6) fire together — "I
  forgot", "the world changed", and "I'm off-distribution so my uncertainty estimate
  is unreliable" are three different correct responses to one scalar. Note that
  disambiguation is expected to need context beyond the scalar (which skill, which
  regime) and is a named P7 concern — not silently assumed away.
- Add a short "arbitration" note under architecture.md's "one signal, many jobs"
  table, cross-referencing ADR-0007.

## Acceptance criteria
- [ ] ADR-0007 exists (Status/Context/Decision/Consequences), added to the ADR README
      index table.
- [ ] ADR-0002 gains the shift-disambiguation consequence.
- [ ] architecture.md "one signal, many jobs" section references the arbitration.
- [ ] No contradiction with ADR-0006 (its uncertainty-penalty paragraph now says
      "in exploitation mode" or equivalent cross-reference).
- [ ] `make test` green, `make lint` clean (unchanged code).

## Test plan
- Docs review only: re-read ADR-0002/0006/0007 together for consistency; check the
  backlog rows for P2-001/P3-002 reference ADR-0007.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] ADR README table updated.
- [ ] Backlog P2-001 and P3-002 rows link ADR-0007.

## Gate result
_not run yet_
