# U-109 — Predict-then-invert imitation (PIDM)

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R5, R7
- **ADRs:** ADR-0012 (imitation from observation)
- **Depends on:** none (reuses the world model + IDM the repo already has)
- **Phase gate:** `bench/gates.py::GATES["P14"]`
- **Source:** `docs/sota-review-2026-07.md` U-109 · [Schäfer et al., ICML 2026](https://arxiv.org/abs/2601.21718)

## Trigger (promote to `ready` when…)
An **imitation gate is marginal at its demo budget** — P14 (or a new imitation phase)
passes only narrowly, or fails to reproduce a behaviour at a small number of demos where
predict-then-invert's variance reduction would help. The **upgrade-triggers** workflow
step checks: if a P14-class report shows a thin margin attributable to demo scarcity,
promote. Until then plain BC on recovered actions is best practice at one-demo/low-dim
scale (review C) — BC is horizon-optimal when implemented properly.

## Goal
When triggered: condition the cloned policy on a *predicted future state* (policy =
IDM(o_t, ô_{t+1}) with ô from the world model) instead of reactive BC — the review's most
promising published upgrade to imitation at this scale, needing zero new components (the
world model and IDM already exist). Reduces variance vs plain BC; BC needs ~3–5× more
demos to match.

## Non-goals
- No RL loop, no adversarial training, no video model (all review SKIPs at this scale).
- Not replacing the IDM action recovery — reusing it plus the world model's forward
  prediction.

## Interface to satisfy (when promoted)
`imitation.ObservationImitator` (imitation.py): the cloned policy becomes
`act(o_t) = inverse(o_t, world_model.predict(o_t, ·))` — predict the next latent, invert
to the action — instead of the reactive `_policy` (imitation.py:71-88). `ImitationLearner`
protocol may gain a world-model handle; keep the reactive clone as the fallback.

## Approach (brief, when promoted)
- At act time: predict ô_{t+1} with the world model, feed (o_t, ô_{t+1}) to the recovered
  inverse dynamics — predict-then-invert. Changes the policy interface, so gate it.

## Acceptance criteria (when promoted)
- [ ] Predict-then-invert policy; reproduces the demo at fewer labels than reactive BC on
      the same data (the variance-reduction claim, measured).
- [ ] **P14 gate PASS**; `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: predict-then-invert matches BC at large budget, beats it at small budget.
- Eval: `make gate PHASE=P14` (+ BH-001 §B optional) + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0012: record predict-then-invert as the marginal-budget upgrade and its trigger.
- [ ] `docs/sota-review-2026-07.md`: note U-109 outcome.

## Gate result
<deferred — no gate until promoted>
