# U-010 — Grounding labels *during* latent-action training (not only post-hoc)

- **Status:** ready
- **Phase:** U (upgrade track; re-gates P13/P14)
- **Requirements:** R7, R5
- **ADRs:** ADR-0010 (watch-then-ground), ADR-0012 (imitation)
- **Depends on:** none (extends P13-001 / P14-001)
- **Phase gate:** `bench/gates.py::GATES["P13"]` (transfer/low-label criterion) — must
  improve the low-label-regime margin
- **Source:** `docs/sota-review-2026-07.md` U-010 · [LAOM](https://arxiv.org/abs/2502.00379)
  · [CLAM](https://arxiv.org/abs/2505.04999)

## Goal
Post-hoc-only grounding is the weaker variant: the 2025 evidence (LAOM: 2.5% labels →
4.2× downstream; CLAM concurs) shows feeding a little action supervision *into* latent-action
training — so the latent space is shaped to be groundable — beats grounding purely
afterward. Interleave the few labelled transitions with the action-free `observe` loop
instead of only calling `ground` after pretraining.

## Non-goals
- Not abandoning the action-free objective — this *interleaves* a small supervised term
  with it, keeping the action-free-limit route (P13's arc-faithful path) intact.
- Not changing the imitation route (P14 `ObservationImitator`) — this improves the
  latent-action route (`observation.LatentActionModel`) the harness measures alongside it.

## Interface to satisfy
`observation.LatentActionModel` (observation.py): a combined step that runs `observe`
(action-free reconstruction + decorrelation, observation.py:84) and a weighted `ground`
term (supervised inverse toward the true action, observation.py:66) on the labelled
subset in the *same* update, rather than the harness looping `observe` then `ground`
separately. `ObservationLearner` protocol unchanged; add `observe_grounded(obs, next_obs,
labelled_batch, w_ground)` or a `ground_weight` on the existing loop.

## Approach (brief)
- Each step: reconstruction+decorrelation gradient (unlabelled stream) + `w_ground` ×
  supervised inverse gradient (labelled subset) into the same inverse model — LAOM/CLAM's
  "shape the latent to be groundable" rather than "ground a frozen latent".
- Expected effect: the low-label transfer margin the P13 gate measures (watch-first beats
  from-scratch) widens, and the P13→P14 latent route becomes more reliable (the review's
  strongest published upgrade to the observe→ground arc).

## Acceptance criteria
- [ ] Combined observe+ground step; unit test shows recovery R² at a small labelled
      budget ≥ the post-hoc-only baseline on the same data.
- [ ] **P13 low-label transfer criterion improves** (record the margin vs the shipped
      P13 report); `make gate-all` green.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_observation.py): interleaved grounding reaches target recovery R² with
  fewer labels than sequential observe-then-ground.
- Eval: `make gate PHASE=P13`, `make gate-all`; optionally re-run BH-001 §B latent route.

## Docs-sync checklist
- [ ] Status → done; low-label margin before/after recorded below.
- [ ] ADR-0010: note grounding-during-training strengthens watch-then-ground; cite
      LAOM/CLAM.
- [ ] `docs/sota-review-2026-07.md`: mark U-010 shipped.

## Gate result
<paste the GateResult once run>
