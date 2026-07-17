# ADR-0006 — Representation & uncertainty integrity (collapse prevention)

**Status:** Accepted — amended by ADR-0014

## Context
ADR-0001 chose to predict in *latent* space; ADR-0002 made a single calibrated
uncertainty signal the backbone for six jobs. Both choices are collapse-prone, and
because the whole system funnels through the shared latent and that one signal, a
collapse of *either* is a single point of failure that silently disables everything
downstream (learning, mastery test, skill-trust, re-planning, forgetting-detection,
retrieval). Collapse is usually invisible in the training loss, because the trivial
solution *has* low loss. Two further modes are self-inflicted: generative replay
(the anti-forgetting mechanism from ADR-0004/R7) is a recursive self-training loop
that can collapse (model autophagy / MAD), and the option layer (ADR-0003) can
collapse to identical or one-step options, losing temporal abstraction.

## Decision
**Representation anti-collapse.** Prevent the encoder's race-to-constant/low-rank
with: an EMA **target encoder with stop-gradient** on the target branch; **variance–
covariance regularization** (VICReg-style: a per-dimension std hinge plus off-diagonal
decorrelation); and **auxiliary heads that force controllable content** — inverse
dynamics (predict the action between consecutive latents) and reward prediction. A
reconstruction/decoder anchor is the robust fallback, but it re-introduces the
observation-detail pressure latent prediction was chosen to avoid, so it is **opt-in,
not default**.

**Uncertainty anti-collapse.** Maintain ensemble diversity — independent
initializations, decorrelated/bootstrapped data order, no over-shared trunk, periodic
member resets — and treat the uncertainty estimate as **unvalidated** until a
reliability check shows disagreement predicts held-out error.

**Generative-replay anti-collapse.** Never rehearse on dreams alone: a fixed fraction
of **real experience anchors every rehearsal batch**; cap **lineage depth** (regenerate
from a real-data-anchored checkpoint, never dream-of-dreams); and **quality-gate**
dreamed trajectories with the (validated) uncertainty estimate — rehearse only
in-distribution dreams. The fixed-budget real store combines a recent FIFO window
with an Algorithm-R reservoir over transitions that age out of it: freshness remains
available while older lifetime experience continues to accumulate statistically
instead of disappearing at a FIFO horizon. The two segments are disjoint and replay
sampling stays uniform over their union. *(Amended by U-004.)*

**Model-exploitation control.** Planning **in exploit mode** uses
**uncertainty-penalized rollouts** with short/branched horizons (MBPO/MOPO), so the
planner is repelled from regions the model is wrong about; explore-mode data
collection flips the sign under the curriculum's control (ADR-0007). This defense
*depends on* a healthy uncertainty estimate, which is why the reliability check is
upstream of it. Multi-step rollouts propagate one state per ensemble member (TS∞)
rather than repeatedly collapsing to the member mean: horizon epistemic is the
spread of those trajectories, while per-member aleatoric variance is accumulated
alongside them without sampling noise into the state. A planner may stop scoring
after accumulated epistemic crosses a calibrated bound. The uncertainty-reliability
sentinel rejects a return to mean-state rollouts by comparing both paths on an OOD
horizon probe. *(Amended by U-001.)*

**Integrity is gated, not hoped for.** Because collapse hides in a good loss, integrity
is enforced by standing **sentinels** in `bench/gates.py`. Every phase gate passes only
if its capability criterion passes **and** all applicable sentinels are healthy.

## ADR-0014 amendment — lifecycle and trace integrity
E-series integrity extends beyond representation rank and ensemble disagreement.
It also checks:

- one raw `Experience` per environment step, with complete episode, step, and
  decision links;
- an immutable action-time prediction rather than a forecast recomputed after an
  update;
- compatible model, representation, and calibration versions;
- separation of real and imagined transition lineage;
- checkpoint manifests that restore the declared learning state and random state;
  and
- disjoint training, calibration, improvement-evaluation, and retention evidence.

The existing sentinels remain valid legacy-v1 evidence for P0–P14. They do not
automatically satisfy these lifecycle checks. The E-series sentinel implementation
and results are pending in `E0-001`.

## Consequences
- (+) The two single-points-of-failure — the representation and the calibrated
  uncertainty — are actively protected and continuously monitored, not assumed.
- (+) The anti-forgetting mechanism cannot quietly become a collapse mechanism.
- (+) The auxiliary inverse-dynamics / reward heads double as the concrete mechanism
  for R4 (a controllable, outcome-relevant representation).
- (−) Extra losses/regularizers add tuning surface and compute; the std/rank floors and
  reliability thresholds are themselves hyperparameters that need calibrating per task.
- (−) Sentinels require held-out probes and add evaluation cost to every gate.
- Sentinels are fed by a run-metrics artifact (`bench/runs/<run-id>/metrics.jsonl`,
  P0-005): training loops log per-step integrity metrics — the dict
  `Learner.update()` returns plus held-out probes — and a zero-argument sentinel
  `check()` reads the run back, so "throughout training" is verifiable, not
  aspirational. *(Amended by P0-005.)*
- Out of scope here: **curiosity collapse** (noisy-TV) is already handled by rewarding
  *epistemic* (not raw) surprise in ADR-0002; **posterior collapse** applies only if a
  stochastic latent with a KL term is used, in which case add free-bits / KL-balancing
  at that time.
- **Gate-overfit is itself a collapse mode** (of the *measurement*, not the model): a
  calibrated gate can be satisfied by a trivial solution, or by a margin within seed
  noise. P9-004 adds the standing `gate-overfit` sentinel (active from P9): each
  capability criterion must REJECT its degenerate solution (always-retrieve, one-step
  options), the metamorphic invariants must hold with no golden threshold, and a
  bootstrap CI must separate a real margin from noise — the same "measured, not
  assumed" discipline this ADR applies to the model, now applied to the gates.
  *(Added by P9-004.)*
