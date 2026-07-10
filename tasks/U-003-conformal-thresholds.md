# U-003 — Adaptive conformal calibration of VoE thresholds (termination + retrieval gate)

- **Status:** done
- **Phase:** U (upgrade track; re-gates against P5/P8/P9)
- **Requirements:** R2, R3, R8
- **ADRs:** ADR-0002 (the one signal), ADR-0004 (retrieval gating)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P5"]`, `["P8"]`, `["P9"]` (must not regress)
- **Source:** `docs/sota-review-2026-07.md` U-003 · [ACI](https://arxiv.org/abs/2106.00170)
  · [decaying-step ACI](https://arxiv.org/pdf/2402.01139) · [conformal failure detection](https://arxiv.org/pdf/2503.08558)

## Goal
Replace one-shot termination-surprise and epistemic-retrieval cutoffs with
decaying-step adaptive conformal calibration: on each nominal calibration stream,
causally drive pre-update exceedances toward α, audit the frozen result on independent
nominal data, then publish the float threshold to the existing planning/memory
consumer. The update is
`θ_{t+1} = θ_t + η_t(1{s_t > θ_t} − α)`, with `η_t = η_0/√t`;
the online update is distribution-free and does not assume exchangeability. The
independent harness audit is an empirical engineering check, not that guarantee.

## Non-goals
- **Do NOT touch the forgetting detector's latched mastered-error floor**
  in `voe.py`: an adaptive threshold would adapt away exactly the sustained error
  rise it must catch. ACI is for the P5/P8/P9 *termination* and *retrieval* gates only.
- Not recalibrating the whole predictive distribution (would perturb the load-bearing
  epistemic/aleatoric split — cite Kuleshov-style recalibration as the rejected route).
- Not the PID conformal variant (unnecessary complexity at this scale).

## Interface to satisfy
A small `voe`-side helper `AdaptiveThreshold(alpha: float, eta: float)` with
`update(score) -> None` and `.value -> float` (plus an optional finite initial value).
`HierarchicalManager.should_terminate` reads a threshold tracked on nominal one-step
option-execution surprise; `memory.UncertaintyMemoryRouter` reads an epistemic
threshold tracked on a harness-defined nominal reference (one-step seen-region
epistemic for P8 and both P9 arms). **Separate trackers, separate α per consumer.**
No Protocol change: consumers still receive floats.

## Approach (brief)
- Two-line online quantile tracker on the running score stream; `η` decays
  (`η_t = η_0/√t`) so the threshold converges to the true quantile once the
  distribution settles ([ICML 2024](https://arxiv.org/pdf/2402.01139)).
- The harness (`bench/`) owns α and calibration policy. The first fifth of each
  nominal stream is a disjoint quantile/IQR pilot; only later scores count as causal
  online decisions. An independent nominal audit must fall inside a predeclared
  finite-sample band. OOD/failure evaluation scores never feed back.
- P9 retrieval inside planning uses a 100k-score nominal stream and α=.0001: the
  experiments showed that 1% one-step calibration is amplified into harmful CEM
  retrieval, while the much rarer target remains measurable rather than wrapping an
  inert fixed cutoff. Its separate PointMass one-step arm uses α=.01.
- **Pair with U-011:** feed the tracker the *epistemic-normalized* surprise (not raw
  total NLL) for termination, so the interrupt fires on "model wrong", not "world
  noisy". When U-011 lands, its normalized score can feed the same calibrator without
  changing the threshold API.

## Acceptance criteria
- [x] `AdaptiveThreshold` holds an empirical α-trigger-rate on a synthetic score stream
      (unit-tested: realized rate → α as the stream lengthens; converges under a shift).
- [x] Termination and retrieval gates read ACI-tracked thresholds; forgetting floor
      untouched (assert the mastered-error latch is still frozen).
- [x] **P5, P8, P9 gates PASS**; `make gate-all` green.
- [x] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (`tests/test_voe.py`, `tests/test_calibration.py`, `tests/test_memory.py`):
  realized 10%, 1%, and measurable 0.01% trigger rates; convergence after a
  distribution shift; disjoint pilot and target-sensitive independent audits;
  separate consumer trackers; gate-hit vs accepted-retrieval instrumentation;
  forgetting detector unchanged.
- Eval: `make gate PHASE=P5`, `PHASE=P8`, `PHASE=P9`, `make gate-all`.

## Docs-sync checklist
- [x] Status → done; gate results recorded below.
- [x] ADR-0002: add a consequence — thresholds have explicit nominal exceedance
      targets, forgetting floor deliberately excluded.
- [x] architecture.md: note the calibrated trigger-rate for termination/retrieval.
- [x] `docs/sota-review-2026-07.md`: mark U-003 shipped.
- [x] `tasks/BACKLOG.md` and `CLAUDE.md`: move U-003 from ready to shipped.

## Gate result

Final verification (2026-07-10): `make test` **144 passed, 1 skipped**;
`make lint` clean; `make typecheck` clean.

- **P5 PASS** — `bench/results/P5-20260710T162918Z.json`. The hierarchy still
  wins every seed (−9.8/−6.9/−4.7 vs compute-matched flat
  −50.2/−44.1/−35.2), all sentinels are healthy, and the independent termination
  audit is MET. For α=1%, causal online rates are 0.980–1.176% and independent
  audit rates are 0.745–1.118%.
- **P8 PASS** — `bench/results/P8-20260710T163008Z.json`. Gated MSE is 0.0076 vs
  0.0260 without retrieval, poison robustness remains MET, and the independent
  retrieval audit is MET. For α=10%, causal online rates are 9.063–10.375% and
  independent audit rates are 8.900–10.550%.
- **P9 PASS** — `bench/results/P9-20260710T163835Z.json`. The composed agent
  returns −17.4 vs −73.1 reactive; one-signal use and nominal calibration/audit
  are MET; retrieval remains safe (median marginal −3.3); and prediction,
  planning, uncertainty, and retrieval all generalize to PointMass. For the
  integrated α=0.01% policy, 80k causal updates produce nonzero online rates
  0.00375–0.01625%, while independent 100k-audit rates are 0.001–0.018%
  (predeclared tolerance ±0.015 percentage points). Runtime CEM gate-hit rates
  are 82.56–89.40%, but the separate distance gate limits accepted retrieval to
  1.76–6.28% of candidate rows (median 2.28%). The separate PointMass α=1%
  audit is 0.44–0.90% and passes.
- **Full ratchet PASS** — `make gate-all`: all 15 shipped gates P0–P14 green.

P9 exposed the important calibration edge case. α=1% on nominal one-step scores
was amplified inside CEM, producing 26.07% median accepted retrieval and a harmful
−19.4 retrieval marginal (`P9-20260710T151640Z.json`, FAIL). A planning-rollout
proxy then failed its independent audit (2.76–6.70% vs 1%) and eliminated accepted
retrieval, so one-signal use failed (`P9-20260710T152411Z.json`, FAIL). Retaining
the old `2× max` rule as an ACI pilot passed the capability gate but produced zero
nominal online/audit events—an inert tail, not target-sensitive calibration
(`P9-20260710T154410Z.json`, rejected despite PASS). The shipped policy restores
the ordinary pilot and makes the rare tail observable with α=0.01%, 100k
calibration + 100k independent audit scores, a target-sensitive audit band, and
required nonzero online and audit rates.
