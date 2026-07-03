# P0-010 — ADR-0007: arbitration of the epistemic signal (seek vs avoid)

- **Status:** done
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
- [x] ADR-0007 exists (Status/Context/Decision/Consequences), added to the ADR README
      index table (`docs/adr/0007-epistemic-signal-arbitration.md`).
- [x] ADR-0002 gains the sign-arbitration + shift-disambiguation consequence.
- [x] architecture.md "one signal, many jobs" section references the arbitration
      (paragraph after the jobs table).
- [x] No contradiction with ADR-0006 — its model-exploitation paragraph now reads
      "in exploit mode … explore-mode data collection flips the sign under the
      curriculum's control (ADR-0007)".
- [x] `make test` green, `make lint` clean (unchanged code).

## Test plan
- Docs review only: re-read ADR-0002/0006/0007 together for consistency; check the
  backlog rows for P2-001/P3-002 reference ADR-0007.

## Docs-sync checklist
- [x] Task Status updated; gate result recorded below.
- [x] ADR README table updated (0007 · Accepted).
- [x] Backlog P2-001 and P3-002 rows link ADR-0007 (annotated at planning time;
      verified — P2-001 applies the penalty sign, P3-002 owns the mode flag).

## Gate result
Docs-only task; consistency review done (ADR-0002 ↔ 0006 ↔ 0007 cross-references
read together — the sign is owned once, by the curriculum; ADR-0006's penalty is
scoped to exploit mode; ADR-0002 points sign conflicts at ADR-0007). Code unchanged:

```
make test      : 37 passed
make lint      : All checks passed!
make typecheck : Success: no issues found in 25 source files
make gate-all  : [P0] PASS — ratchet ok
```

Result: **PASS** (P0 criterion met; docs consistent).
