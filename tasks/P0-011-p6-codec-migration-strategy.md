# P0-011 — Docs: the P6 codec swap is a representation change — name the migration strategy

- **Status:** ready
- **Phase:** P0
- **Requirements:** R6
- **ADRs:** ADR-0001 (add consequence); `docs/roadmap.md` P6 row
- **Depends on:** none (docs-only; can run in parallel with everything)
- **Phase gate:** `bench/gates.py::GATES["P0"]` (docs task — gate is `make test` green
  + docs consistent)

## Goal
Stop underplaying P6. The roadmap calls the universal codec "an interface change to a
working system" — structurally true (the `Codec` protocol makes it a drop-in), but
everything built in P1–P5 is *trained against* the old latent distribution: the
dynamics model, the option model, per-skill competence statistics, and every latent
stored in the replay buffer. Swapping encoders invalidates all of it unless the new
codec is matched to the old latent space. Protocols make the swap typecheck; they do
not make it cheap. The migration strategy must be named now, because it changes P6's
cost estimate by an order of magnitude and can influence earlier choices (e.g.
whether replay stores raw observations alongside latents).

## Non-goals
- No codec work, no distillation code (P6-001).
- No change to the P6 gate criterion ("swap preserves core-loop performance within
  tolerance") — this task explains what satisfying it will actually require.

## Interface to satisfy
None — documentation task. Deliverables: roadmap amendment, ADR-0001 consequence,
P6-001 backlog row update.

## Approach (brief)
- Amend the roadmap's P6 row + sequencing note: the swap is a **representation
  change**. Primary strategy: **distill the universal codec into the existing latent
  space** (train `UniversalCodec.encode` to match the P1 encoder's outputs on shared
  modalities before swapping), so downstream components remain valid within the gate's
  tolerance. Fallback: **budgeted full-stack retrain** behind the new codec if
  distillation cannot hit tolerance. State that P6's estimate includes this cost.
- Add an ADR-0001 consequence: "the shared latent is a *contract* — components
  trained on it couple to its distribution, not just its shape; replacing the encoder
  requires distribution-matching (distill) or downstream retraining."
- Concrete earlier-phase implication to record: the replay buffer (P3-003) should
  retain enough raw observation to re-encode experience under a future codec —
  note it on the P3-003 backlog row so the decision is made consciously there.

## Acceptance criteria
- [ ] Roadmap P6 row/sequencing note names the migration strategy (distill-first,
      retrain-fallback) and its cost.
- [ ] ADR-0001 consequence added.
- [ ] Backlog rows for P6-001 and P3-003 updated with the implications above.
- [ ] `make test` green, `make lint` clean (unchanged code).

## Test plan
- Docs review only: roadmap, ADR-0001, backlog rows consistent with each other and
  with the unchanged P6 gate criterion.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] architecture.md "what is deliberately hard" list gains the generality-tax /
      representation-migration point if not already covered.

## Gate result
_not run yet_
