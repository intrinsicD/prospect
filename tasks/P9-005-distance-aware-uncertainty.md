# P9-005 — Distance-aware epistemic uncertainty (make the signal OOD-reliable)

- **Status:** done
- **Phase:** P9
- **Requirements:** R1, R3, R4, R7, R8 (everything that reads the epistemic signal)
- **ADRs:** ADR-0002 (the one signal — amended here: epistemic is distance-aware, not
  ensemble-disagreement alone), ADR-0006 (uncertainty integrity), ADR-0008
- **Depends on:** P9-003 (surfaced the finding: uncertainty doesn't generalize)
- **Phase gate:** contributes to `bench/gates.py::GATES["P9"]` — the
  "uncertainty signal generalizes to a second environment" criterion.

## Goal
Repair the load-bearing crack P9-003 found: the epistemic uncertainty signal — the
spine every subsystem reads — is not OOD-reliable across environments. On PointMass the
ensemble is *confidently wrong* out-of-region (epistemic barely rises where the model is
wrong), so mastery, curiosity, retrieval-gating and re-planning all inherit an
unreliable signal. Make epistemic rise out-of-distribution with the same core, and gate
that it generalizes.

## Non-goals
- Not a new uncertainty *architecture* (no SNGP/GP/normalizing-flow) — the minimal
  distance-aware augmentation, earned by the measurement.
- Not fixing retrieval-generalization: the same encoder saturation also corrupts the
  retrieval *key* space (OOD queries match wrong facts), a distinct issue this task
  surfaces as a finding, not solves.
- No change to `var`/`log_prob` (likelihood calibration stays the ensemble's).

## Interface to satisfy
`types.LatentState` gains an optional `ood: float | None`; `world_model.FlatWorldModel.
encode` computes it (pre-encoder standardized-input excess energy) and `predict` scales
the epistemic scalar by `1 + w·ood`. Planning-rollout latents (`ood=None`) are
unchanged. No new `Protocol`.

## Approach (brief)
- **Diagnosis (measured first):** ensemble disagreement rose only 1.75x OOD on PointMass
  (error rose 10x); the encoder's tanh hidden layers saturate (~0.85 of latent dims),
  squashing OOD inputs into the seen latent region, so the ensemble can't detect OOD.
- **Fix:** a distance signal computed *before* the encoder — `mean(standardized_obs²) −
  1`, ~0 in-distribution and rising OOD — carried on the `LatentState` from a real
  `encode`, used by `predict` to scale epistemic (`w=1`). Rises OOD by construction,
  where saturated ensemble disagreement cannot.
- **Gate:** `p9_generalization` adds an uncertainty-reliability check on env #2 (the
  high-error-decile epistemic vs median ≥ floor), folded into the P9 gate.

## Acceptance criteria
- [ ] `encode` sets `ood` (0 in-distribution, rising OOD); `predict` scales epistemic by
      it; synthesized latents unaffected — unit-tested.
- [ ] The uncertainty signal generalizes to env #2 (high-error-decile ratio ≥ floor);
      folded into the P9 gate; **P9 still PASS**.
- [ ] No regression: `make gate-all` (P0–P9) green — the self-calibrated gates are
      preserved (in-distribution epistemic ≈ unchanged).
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_world_model.py): `ood` rises with input distance; `predict` boosts
  epistemic; an `ood=None` latent is unscaled.
- Eval: `make gate PHASE=P9` — the uncertainty-generalizes criterion; `make gate-all`
  for regression.

## Docs-sync checklist
- [x] Status → `done`; before/after numbers recorded below.
- [x] ADR-0002 amended (distance-aware epistemic).
- [x] Backlog: P9-005 done; the retrieval-key-saturation follow-up recorded.

## Gate result
`make gate PHASE=P9` → **PASS** with `uncertainty ✓ generalize` (high-error-decile
ratio 8.8 vs floor 3.0). `make gate-all` → **ratchet ok — 10 shipped gates still
green** (~11m): no regression from the global epistemic change.

**The fix, measured (diagnostic → after):**

| Signal (region-trained model, seen→OOD) | Pendulum | PointMass |
|---|---|---|
| ensemble-only epistemic rise | 5.6× → **16×** | **1.75× → 7.85×** |
| epistemic-vs-error rank corr | 0.57 → **0.60** | **0.52 → 0.80** |

The uncertainty signal now rises out-of-distribution on the environment where it was
broken, closely tracking the error rise (7.85× vs 9.93×). A **bonus**: the
uncertainty-reliability sentinel got much healthier across all phases (high-error-decile
disagreement 60–120× median, up from ~18×) — the OOD probes now separate cleanly.

**New findings surfaced by the fix (reported, not tuned away):**
1. **Retrieval still doesn't generalize** — the gate now fires (uncertainty fixed), but
   the same encoder saturation corrupts the retrieval *key* space, so OOD queries match
   wrong in-region facts. A distinct follow-up (a non-saturating key space, or keys in a
   pre-encoder feature).
2. **The ablation shifted:** with OOD-aware epistemic, the exploit-penalty flipped from
   +2.5 (negligible) to **−6.0 (harmful)** — penalising the now-larger OOD epistemic
   steers the planner away from OOD too aggressively; retrieval's harm shrank to −3.1
   (negligible). Consistent with the change; planning stays load-bearing (+49.5) and the
   gate PASSes.
