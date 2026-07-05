# ADR-0008 — Whole-system validation: the integration gate + ablation as the fitness

**Status:** Accepted

## Context
P0–P8 each prove one capability in its own harness. Passing them individually does
*not* prove that the composed agent works end-to-end, that every part is
load-bearing, that a capability generalizes beyond the single reference environment,
or that a gate measures the capability rather than a calibrated trivial solution. A
benchmark-gated project needs a fitness function for the **whole**, not only the
parts — otherwise "all gates green" can coexist with an agent that does not actually
work when assembled.

## Decision
Add **Phase 9 — whole-system validation** with four standing checks:
1. **Integration gate (P9-001):** run the fully-composed agent through the
   `agent.py` composition root and assert emergent, closed-loop properties — it
   controls end-to-end; within ONE run the single VoE (epistemic) signal both sets
   the planner's explore/exploit coefficient (via the curriculum, ADR-0007) *and*
   gates retrieval (ADR-0004); retrieval-as-action does not degrade control (a *gated*
   criterion since P9-007's distance-gating fix) — with all applicable collapse sentinels
   healthy.
2. **Ablation (P9-002):** a component earns its place only if disabling it measurably
   degrades the P9 metric. A no-op ablation is a *finding* (dead weight, or a test
   too blind to see the component's value) — never silently passed.
3. **Cross-environment (P9-003):** a capability is real only if it survives a second,
   structurally different environment with the **same core code** (recalibrated
   thresholds only). Collapse on env #2 means the capability was an artifact of the
   first environment.
4. **Invariants + negative controls (P9-004):** metamorphic invariants (no golden
   threshold) and per-gate negative controls that a trivial solution must *fail*
   guard against gate-overfit and against a margin that sits within seed noise.

The **integration gate + ablation are the standing whole-system fitness function**:
per-phase gates prove the parts; P9 proves the whole *and* that the parts matter. P9
joins the regression ratchet (ADR-0005, P0-007) like any other phase.

## Consequences
- (+) "Does it work together?" gets a measured, ratcheted answer — not an assumption.
- (+) Ablation turns "we assume each piece matters" into "we measured it"; no-op
  ablations expose dead weight or blind tests before they rot.
- (+) Cross-environment + negative controls directly de-risk the calibration/overfit
  inherent in hand-tuned single-environment gates (ADR-0006's spirit, applied to the
  gates themselves).
- (−) More CI compute: an end-to-end run plus ablations. Kept bounded — a short
  control budget, and ablations reuse the P9 harness rather than new machinery.
- (−) The integration gate composes many parts, so its numbers are noisier than a
  single-capability gate; its criteria are **relative / structural / paired**, not
  tight absolute thresholds, and it reports effect direction over precise magnitudes.

## Validation (P9-001)
The approach earned its keep on the first run: the integration gate surfaced a finding
no single-phase gate could see — **retrieval-into-planning degrades control** (the
composed agent returns −19.4 with retrieval vs −8.1 without). Retrieval improves 1-step
*prediction* (P8) but overriding the planner's multi-step rollout dynamics with
nearest-neighbour facts corrupts the optimisation. Per this ADR the result is *reported
as a finding* (measured, not gated) rather than tuned until it disappears, and it
becomes the first target of the P9-002 ablation. This is exactly the failure mode the
whole-system layer exists to catch: every part passed its own gate, yet composing them
naively hurt.

The P9-002 ablation (leave-one-out marginal control value `composed - ablated`)
then quantified it and found a second under-performer:

| Component | Marginal | Verdict |
|---|---|---|
| planning | +53.7 | load-bearing (the gated criterion) |
| retrieval | −9.5 | harmful — corrupts multi-step planning |
| exploit_penalty | +2.5 | negligible on this task |

Both under-performers were recorded as findings (harmful retrieval; a near-negligible
exploit penalty) rather than tuned away — the ablation is the standing tool that keeps
"every part earns its place" honest.

**Resolution (P9-007).** The harmful-retrieval finding was then fixed at its root, not
worked around. Diagnosis: inside CEM rollouts the query is an imagined latent that, at
depth, sits far from any real transition (median key-distance ~7× a real in-coverage
query), so its nearest fact is fiction; substituting it — and marking the row `epistemic
= 0`, which deleted the ADR-0006 exploit penalty exactly where the model was least
reliable — *lured* the planner into the retrieval seam. Fix: **distance-gated retrieval**
— substitute a fact only within a coverage-calibrated reliability radius, with honest
distance-scaled epistemic instead of 0 (reliability = closeness, the P9-006 insight).
Measured after the fix (median over seeds): retrieval marginal **−9.5 → −0.3**
(negligible, safe — now a *gated* criterion), composed control **−23.6 → −9.7**, and the
entangled exploit-penalty marginal recovered **−6.0 → −1.6**. The 1-step retrieval role
(real states, in coverage) is unchanged. The whole-system layer thus did its full job:
it *found* a naive-composition harm no single-phase gate could see, and the finding drove
a principled fix that a per-part view would never have motivated.

P9-003 then ran the capabilities on a **second, structurally different environment**
(`PointMass`, 2D nonlinear-drag; obs_dim 3→4, action_dim 1→2) with the same core:

| Capability | env #2 | Verdict |
|---|---|---|
| prediction (P1) | WM 0.0011 vs persistence 0.0295 | generalizes |
| planning (P2) | planner −15.5 vs random −52.8 | generalizes |
| uncertainty (P9-005) | high-error-decile epi 8.8× median | generalizes |
| retrieval (P8) | gated 0.0135 vs no-retrieval 0.0172 | generalizes (dense store) |

Prediction and planning generalize with only recalibrated eval params (no core change —
that no core change is required is the result). Retrieval's story took two corrections
before it generalized, both recorded honestly rather than patched silently:

- **P9-003** found retrieval did *not* generalize (gated 0.0183 vs no-retrieval 0.0172).
  Two causes were compounded. The first: the ensemble was *confidently wrong*
  out-of-region on PointMass, so the uncertainty gate barely fired (the ADR-0002
  limitation).
- **P9-005** fixed that first cause (distance-aware epistemic) — the gate now fires
  out-of-region — but retrieval *still* didn't help, and P9-005 **wrongly hypothesized**
  the residual cause was encoder saturation corrupting the retrieval *key* space.
- **P9-006** measured the residual cause and disproved the key-saturation hypothesis:
  the latent key is fine (it even beats a raw standardized-input key). The real cause is
  store **density vs key-space dimensionality** — the curse of dimensionality in the 6-D
  key. A sparse store (1500 facts) leaves the nearest fact too far to be right; a
  dimension-adequate store (40000) makes retrieval generalize (gated 0.0135, ~22% better
  than no-retrieval). Retrieval is now **gated** in the P9 generalization check.

Finally, P9-004 closed the phase by guarding the gates themselves: the standing
`gate-overfit` sentinel (7 cheap checks) asserts that trivial solutions FAIL their
criteria (negative controls), that metamorphic invariants hold with no golden
threshold, and that a bootstrap CI separates a real margin from noise. With it, "the
gates measure the capability, not the artifact" is enforced, not hoped — the same
discipline the ADR-0006 sentinels apply to the model, applied to the measurement.
The naive-composition harms this layer surfaced — retrieval degrading multi-step control,
and its entanglement with the exploit penalty — have since been driven to negligible by
the P9-007 distance-gating fix (both now gated/near-zero marginals). What remains honest
to say is narrower: retrieval into planning *earns little* on this task (a small negative
marginal, safe but not load-bearing), and the exploit penalty is near-negligible here —
the map of where the scaffold's real work remains.
