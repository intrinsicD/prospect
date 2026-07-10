# U-106 — CUSUM / change-point option termination

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R2, R3
- **ADRs:** ADR-0002 (VoE signal), ADR-0003 (termination)
- **Depends on:** U-011 (epistemic-gated termination) + U-003 (conformal threshold) first
- **Phase gate:** `bench/gates.py::GATES["P5"]`
- **Source:** `docs/sota-review-2026-07.md` U-106 · [CPD-HRL](https://arxiv.org/abs/2510.24988)

## Trigger (promote to `ready` when…)
**Gate metrics show option termination chattering** — the interrupt fires on single-step
aleatoric spikes (false interrupts), visible as options terminating far below their
learned duration or as a P5 regression traced to over-termination. The **upgrade-triggers**
workflow step checks: if a P5-class report shows termination-rate anomalies after U-011
(epistemic-gated) and U-003 (conformal) are in, promote. Until then a well-calibrated
one-step epistemic trigger is fine (review RQ5) — CUSUM is only worth it against measured
chatter.

## Goal
When triggered: integrate surprise over steps (CUSUM: `S ← max(0, S + surprise − drift)`,
interrupt on crossing) instead of a single-step threshold, so a lone aleatoric spike no
longer false-terminates — robust to noise while staying the same VoE signal.

## Non-goals
- Not full Bayesian online change-point detection (BOCPD) — overkill at toy scale; CUSUM
  is the minimal version.
- Not replacing U-003's conformal calibration — CUSUM changes *what accumulates*, ACI
  still sets the crossing level.

## Interface to satisfy (when promoted)
`planning.HierarchicalManager.should_terminate` (planning.py:283-289): maintain a CUSUM
statistic over the running option's epistemic-attributed surprise; terminate when it
crosses the (conformal, U-003) level. `HierarchicalPlanner` protocol unchanged.

## Approach (brief, when promoted)
- Accumulate the epistemic surprise (U-011) with a drift term; the option interrupts on a
  *sustained* deviation, not one noisy step — CPD-guided termination outperforms
  single-step in the cited work.

## Acceptance criteria (when promoted)
- [ ] CUSUM termination; unit test: a single aleatoric spike does not interrupt, a
      sustained epistemic rise does; termination-rate anomaly from the trigger resolved.
- [ ] **P5 gate PASS**; `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: CUSUM ignores isolated spikes, fires on runs; parity with fixed threshold when
  drift→0 and horizon→1.
- Eval: `make gate PHASE=P5` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0003: record CUSUM termination and the chatter measurement that triggered it.
- [ ] `docs/sota-review-2026-07.md`: note U-106 outcome.

## Gate result
<deferred — no gate until promoted>
