# U-111 — Jumpy-model cross-timescale consistency loss

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R2
- **ADRs:** ADR-0003 (jumpy option model bounds compounding error)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P5"]`
- **Source:** `docs/sota-review-2026-07.md` U-111 · [FAIR jumpy planning 2026](https://arxiv.org/abs/2602.19634)

## Trigger (promote to `ready` when…)
**Compounding jump error is measured at depth** — the jumpy option-model's rollout error
grows superlinearly in the number of composed jumps, visible as a P5 miss on deeper
option sequences or an option-model rollout-error probe rising with depth. The
**upgrade-triggers** workflow step checks: if a P5-class report attributes a miss to
multi-jump error, promote. Until then the single-jump option model is validated (P5-001
beats the flat rollout 4–6×; FAIR 2026 uses the same architecture) — the consistency loss
is a refinement, not a fix for a present problem (review RQ4).

## Goal
When triggered: add FAIR's cross-timescale consistency objective to `JumpyOptionModel.update`
— one long jump must agree with two composed short jumps — regularizing the option model
so composed multi-jump plans stay accurate.

## Non-goals
- Not before the trigger — no measured multi-jump error today.
- Not a new model class — a loss term on the existing ensemble.

## Interface to satisfy (when promoted)
`planning.JumpyOptionModel.update` (planning.py:189-235) gains a consistency term:
predict a 2-step-option landing directly and via two 1-step-option jumps, penalize their
disagreement. `OptionModel`/`Learner` protocols unchanged; metrics dict gains
`loss_consistency`.

## Approach (brief, when promoted)
- Where option durations compose, add `‖predict(2h) − predict(h)∘predict(h)‖`; cheap,
  reuses the members; directly targets the compounding-jump error ADR-0003 bounds.

## Acceptance criteria (when promoted)
- [ ] Consistency term added; multi-jump rollout error drops vs the shipped option model
      on a deep-sequence probe (the reason for promotion).
- [ ] **P5 gate PASS**; `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: composed 2×1-jump prediction converges to the direct 2-jump prediction after
  training with the term; `loss_consistency` present.
- Eval: `make gate PHASE=P5` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0003: record the consistency loss and the multi-jump-error measurement that
      triggered it.
- [ ] `docs/sota-review-2026-07.md`: note U-111 outcome.

## Gate result
<deferred — no gate until promoted>
