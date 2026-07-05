# P9-007 — Retrieval-into-planning: honest uncertainty instead of a free pass

- **Status:** in-progress
- **Phase:** P9
- **Requirements:** R1 (planning), R8 (retrieval), R4 (the epistemic signal both read)
- **ADRs:** ADR-0004 (retrieval-as-action — the composition rule), ADR-0006 (model-
  exploitation control), ADR-0007 (exploit-mode penalty), ADR-0008 (the finding)
- **Depends on:** P9-002 (quantified the harmful marginal), P9-006 (distance = retrieval
  reliability — the idea this fix rides on)
- **Phase gate:** `bench/gates.py::GATES["P9"]` — the ablation-marginal check (retrieval
  must not be *harmful* into planning) + no regression.

## Goal
Fix the standing P9 finding that uncertainty-gated retrieval **degrades multi-step
control** when composed into the planner's imagination rollouts. Make retrieval-into-
planning at worst neutral (marginal not harmful) by removing the model-exploitation seam
it currently opens.

## Non-goals
- Not changing the 1-step-prediction retrieval role (P8) or its generalization (P9-006).
- Not touching the exploit-penalty finding (a separate entangled marginal).
- No new retrieval index / ANN / learned key.

## The mechanism (diagnosed)
`RetrievalAugmentedWorldModel._rows` overrides the predicted mean with the nearest fact
AND sets `epistemic = 0.0` for that row. The planner scores `reward −
uncertainty_penalty · epistemic` per rollout step (ADR-0006/0007). Zeroing epistemic
exactly where the model is *least* reliable (retrieval only fires OOD) removes the
exploit penalty there, so CEM is **lured** into routing through the retrieval seam: the
penalty vanishes and a nearest-neighbour fact can look high-reward. That is model
exploitation (ADR-0006) through the retrieval mechanism — the harmful marginal.

## Interface to satisfy
No new `Protocol`. `RetrievalAugmentedWorldModel._rows` (src/prospect/memory.py) stops
asserting a retrieved fact is certain: it sets the row's epistemic from the **retrieval
distance** (nearest-neighbour key distance = reliability, P9-006), not to 0. A close
retrieval (store covers the query) → low epistemic (trusted); a far one (deep-rollout
fiction) → epistemic kept high (no free pass). `SemanticStore.query` returns the nearest
distance alongside the item so the wrapper can read it without recomputing.

## Approach (brief)
- Diagnosis measured first (a scratch experiment): reproduce the negative retrieval
  marginal, then test the honest-epistemic variants and keep the one that removes the
  harm with the least machinery.
- Fix: distance-as-epistemic on retrieved rows (details settled by the measurement).
- Gate: fold "retrieval is not *harmful* into planning" into the P9 ablation check.

## Acceptance criteria
- [ ] Retrieval-into-planning marginal is **not harmful** (≥ −MARGIN) on the P9 gate,
      with planning still load-bearing and the whole gate PASS.
- [ ] No regression: `make gate-all` (P0–P9) green.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.
- [ ] ADR-0004 amended with the composition rule; the finding text updated everywhere.

## Test plan
- Unit (tests/test_memory.py): a far retrieval keeps epistemic high; a close retrieval
  lowers it; `query` returns a sane distance.
- Eval: `make gate PHASE=P9` (the retrieval marginal); `make gate-all` for regression.

## Docs-sync checklist
- [ ] Status → `done`; gate result recorded below.
- [ ] ADR-0004 amended (retrieval-into-planning composition rule).
- [ ] ADR-0008 / architecture.md / p9_integration finding text updated.
- [ ] BACKLOG P9-007 row added.

## Gate result
<pending>
