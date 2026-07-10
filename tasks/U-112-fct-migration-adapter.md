# U-112 — FCT-style migration adapter (old→new latent transformation)

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R6
- **ADRs:** ADR-0001 (the latent is a contract), P0-011 (distill-first, retrain-fallback)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P6"]` (codec swap preserves core-loop perf)
- **Source:** `docs/sota-review-2026-07.md` U-112 · [FCT](https://arxiv.org/abs/2112.02805)

## Trigger (promote to `ready` when…)
**Codec distillation misses a gate's tolerance** — the P0-011 distill-first path cannot
hit the P6 swap tolerance because a new encoder is much stronger than the incumbent 8-dim
latent can express (point-wise alignment plateaus — the documented BCT quality ceiling),
i.e. **the retrain-fallback is about to fire**. The **upgrade-triggers** workflow step
checks: if a P6-class codec-swap report shows distillation loss stuck above tolerance,
promote FCT as the cheaper middle path before a full retrain. Until then distillation into
the incumbent latent works (P6-001 shipped at ~1.0 swap ratio) — FCT is heavier and
unjustified (review RQ4 for the codec).

## Goal
When triggered: instead of constraining the new encoder to the old latent (BCT, which caps
its quality) or fully retraining downstream, learn an old→new latent transformation and
migrate downstream components behind it (FCT: store side-information + a learned mapping,
leaving the new encoder unconstrained — +18% retrieval over BCT in the source).

## Non-goals
- Not before the trigger — distillation is the correct cheap path now.
- Not the stationary/relative-representation redesign of the contract itself (heavier
  still; a separate future consideration).

## Interface to satisfy (when promoted)
A learned adapter mapping incumbent latents to the new encoder's space (or vice-versa);
the dynamics/option/competence components consume the mapped latent so they stay valid.
`Codec` protocol unchanged; the migration is harness-side.

## Approach (brief, when promoted)
- Train the new encoder unconstrained; learn old→new (or new→old) transformation on shared
  data; route downstream through it. The middle path between distill and full retrain.

## Acceptance criteria (when promoted)
- [ ] FCT adapter migrates downstream across the encoder swap; **P6 swap tolerance MET**
      where distillation missed it, at less cost than a full retrain.
- [ ] `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: mapped latents keep the dynamics model's 1-step error within tolerance across the
  swap.
- Eval: `make gate PHASE=P6` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0001 / P0-011: record FCT as the middle migration path and its trigger.
- [ ] `docs/sota-review-2026-07.md`: note U-112 outcome.

## Gate result
<deferred — no gate until promoted>
