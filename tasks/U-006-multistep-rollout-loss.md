# U-006 — Multi-step (unrolled) dynamics loss term

- **Status:** ready
- **Phase:** U (upgrade track; re-gates against P1)
- **Requirements:** R1, R4
- **ADRs:** ADR-0001 (latent prediction; compounding rollout error is the named limiter)
- **Depends on:** none (composes with U-001)
- **Phase gate:** `bench/gates.py::GATES["P1"]` (1-step beats baseline) + a new
  multi-step-error probe; must not regress P2/P5
- **Source:** `docs/sota-review-2026-07.md` U-006 · [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985)

## Goal
The world model trains on one-step targets (world_model.py:249-336) but is consumed
autoregressively over 20 steps — the classic compounding-error mismatch, the repo's own
named "main limiter on R1". Add an unrolled k-step loss term (train the rollout the
planner actually uses), as V-JEPA 2-AC does with `L = L_teacher-forcing + L_rollout`.

## Non-goals
- Not replacing the one-step NLL — this *adds* a rollout term (weighted, default small),
  keeping the calibrated one-step likelihood intact.
- Not backprop-through-a-long-rollout instability territory: cap the unroll at a small k
  (2–3), matching the dream depth, so the through-time gradient stays short.
- No new heads; reuses the existing members.

## Interface to satisfy
`world_model.FlatWorldModel.update` (world_model.py:249) gains a rollout term: unroll the
ensemble-mean prediction k steps on the batch's action sequence (batches already carry
consecutive transitions via the buffer) and add `w_rollout · Σ ‖ẑ_{t+j} − target_{t+j}‖`
to the loss. Constructor: `w_rollout: float = 0.1`, `rollout_len: int = 2`. `Learner`
protocol unchanged; the returned metrics dict gains `loss_rollout`.

## Approach (brief)
- Reuse `_member_forward` to unroll k steps from each start; target is the EMA
  target-encoder latent at each future step (stop-grad, ADR-0006), consistent with the
  one-step target at world_model.py:261.
- Short unroll (k=2–3) keeps the hand-written backprop tractable and avoids the
  chaotic-long-rollout-gradient problem the review flagged.
- Directly attacks the named open problem; pairs with U-001 (honest propagated
  uncertainty) so the planner both predicts and *scores* multi-step more faithfully.

## Acceptance criteria
- [ ] `update` adds a k-step rollout loss; `loss_rollout` in the metrics dict.
- [ ] New probe: k-step rollout MSE drops vs the shipped one-step-only model on a held-out
      trajectory (measured; the point of the change).
- [ ] **P1 gate PASS** (1-step still beats baseline; collapse sentinels healthy); P2/P5
      not regressed; `make gate-all` green.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_world_model.py): `loss_rollout` present and finite; k-step held-out
  error lower than a `w_rollout=0` control after equal training.
- Eval: `make gate PHASE=P1`, `PHASE=P2`, `PHASE=P5`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; multi-step error before/after recorded below.
- [ ] architecture.md "Compounding rollout error" bullet: note the unrolled loss term as
      a mitigation (alongside hierarchy).
- [ ] `docs/sota-review-2026-07.md`: mark U-006 shipped.

## Gate result
<paste the GateResult once run>
