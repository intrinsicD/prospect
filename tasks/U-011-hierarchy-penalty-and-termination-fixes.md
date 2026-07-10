# U-011 — Fix hierarchy penalty discounting + epistemic-gated option termination

- **Status:** ready
- **Phase:** U (upgrade track; re-gates P5)
- **Requirements:** R2, R3
- **ADRs:** ADR-0002 (gate on `.epistemic`, never undecomposed total), ADR-0003/0006/0007
- **Depends on:** none (pairs with U-003)
- **Phase gate:** `bench/gates.py::GATES["P5"]` — must hold or improve
- **Source:** `docs/sota-review-2026-07.md` U-011 · code review of planning.py

## Goal
Two internal inconsistencies found while reading, independent of the literature:
1. `HierarchicalManager.plan_option` (planning.py:276) subtracts the epistemic penalty
   **undiscounted**, while `FlatPlanner._imagined_returns` (planning.py:86) discounts it
   — later jumps are relatively over-penalized. Make the convention match (or document
   why it differs).
2. `should_terminate` (planning.py:283-289) fires on **undecomposed total NLL**, so it
   interrupts on aleatoric noise — contradicting architecture.md's own rule ("consumers
   gate on `.epistemic`, never the undecomposed total"). Gate the interrupt on the
   epistemic component (or NLL normalized by predicted aleatoric variance).

## Non-goals
- Not redesigning the manager search or the termination mechanism — two targeted
  corrections that align the code with its own ADRs.
- Not the conformal calibration of the threshold — that is U-003 (this makes the
  *signal* the threshold reads correct; U-003 makes the *threshold* adaptive; they compose).

## Interface to satisfy
`planning.HierarchicalManager` (planning.py:238-289): (1) multiply the epistemic penalty
by the running discount in `plan_option` to match `FlatPlanner`, or add a documented
constant-penalty rationale; (2) `should_terminate` computes surprise on the epistemic
share — reuse the `SurpriseCompetenceMonitor.surprise` decomposition (voe.py:104-109) or
normalize the NLL by `prediction.aleatoric` — and compares *that* to the threshold.
`HierarchicalPlanner` protocol unchanged.

## Approach (brief)
- Penalty: change planning.py:276 to `score -= discount * self.uncertainty_penalty *
  prediction.epistemic` (mirrors planning.py:86). Add a test asserting flat and
  hierarchical use the same discounting convention.
- Termination: the interrupt should mean "the model was *wrong*", not "the world was
  *noisy*". Fire on the epistemic-attributed surprise (or aleatoric-normalized NLL), so a
  high-aleatoric but well-predicted step does not chatter the option.

## Acceptance criteria
- [ ] Hierarchical epistemic penalty is discount-consistent with the flat planner (or the
      difference is documented in the docstring + ADR-0003).
- [ ] `should_terminate` gates on the epistemic component; unit test: a purely-aleatoric
      surprise spike does NOT terminate, an epistemic spike does.
- [ ] **P5 gate PASS** (two-level still beats compute-matched flat); `make gate-all` green.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_planner.py, tests/test_manager.py): discounting parity between
  planners; termination fires on epistemic not aleatoric surprise.
- Eval: `make gate PHASE=P5`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; gate result recorded below.
- [ ] ADR-0003: note the termination interrupt gates on epistemic (consistent with
      ADR-0002); penalty discounting convention documented.
- [ ] architecture.md: the re-planning interrupt (job #4) reads epistemic, matching the
      "gate on `.epistemic`" rule.
- [ ] `docs/sota-review-2026-07.md`: mark U-011 shipped.

## Gate result
<paste the GateResult once run>
