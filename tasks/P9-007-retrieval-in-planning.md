# P9-007 — Retrieval-into-planning: honest uncertainty instead of a free pass

- **Status:** done
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

## Approach (brief, measured)
- **Diagnosis (scratch experiment).** Reproduced the harm at a fixed exploit penalty
  (marginal **−15**), then measured *where* retrieval fires in rollouts: median key-
  distance **0.667 vs 0.094** for a real in-coverage query (~7× farther) — rollout
  retrievals are mostly *fiction*. Two honest-epistemic variants (keep model-epi;
  distance-scale it) reduced the harm to −8..−11 but did **not** remove it: the mean was
  still being overridden with a far, wrong fact.
- **Fix (measured winner): distance-*gating*.** Substitute a fact only when its key-
  distance is within a coverage-calibrated `reliability_radius` (= 4× the median in-
  coverage distance); on a far query, keep the model. Carry honest distance-scaled
  epistemic (`epi × min(1, dist/radius)`), never 0. Sweep (C∈{1,2,4}) all landed the
  marginal in ±MARGIN; C=4 best. This suppresses ~99% of rollout retrievals (the far
  ones), leaving retrieval a rare, safe correction.
- **Gate:** folded "retrieval into planning is not *harmful*" (median marginal ≥ −MARGIN)
  into the P9 gate's PASS condition.

## Acceptance criteria
- [x] Retrieval-into-planning marginal is **not harmful** (≥ −MARGIN) on the P9 gate,
      with planning still load-bearing and the whole gate PASS.
- [x] No regression: `make gate-all` (P0–P9) green.
- [x] `make test` green, `make lint` clean, `make typecheck` clean.
- [x] ADR-0004 amended with the composition rule; the finding text updated everywhere.

## Test plan
- Unit (tests/test_memory.py): a far retrieval keeps epistemic high; a close retrieval
  lowers it; `query` returns a sane distance.
- Eval: `make gate PHASE=P9` (the retrieval marginal); `make gate-all` for regression.

## Docs-sync checklist
- [x] Status → `done`; gate result recorded below.
- [x] ADR-0004 amended (retrieval-into-planning composition rule).
- [x] ADR-0008 / architecture.md / p9_integration finding text updated.
- [x] BACKLOG P9-007 row added.

## Gate result
`make gate PHASE=P9` → **PASS** (all five sentinels healthy). `make gate-all` →
**ratchet ok** (no regression).

**The fix, measured (leave-one-out marginal control value; median over seeds):**

| Marginal | before (P9-002 era) | after P9-007 | verdict |
|---|---|---|---|
| retrieval into planning | −3.1 (−15 at a stronger penalty) | **−0.3** | negligible, **safe (gated)** |
| planning | +49.5 | **+63.4** | load-bearing |
| exploit_penalty | −6.0 (harmful) | **−1.6** | negligible (bonus: the entangled finding recovered) |
| composed control return | −23.6 | **−9.7** | improved |

Retrieval into planning is now a rare, safe correction (in-rollout retrieval rate ~0.4%,
down from ~25%): distance-gating skips the far, fictional facts that were overriding the
planner's rollout dynamics, and honest distance-scaled epistemic closes the `epi=0`
exploit seam. The 1-step P8/P9-006 retrieval role is untouched (`reliability_radius=None`).

**Reported, not tuned away:** retrieval into planning *earns little* here (safe but not
load-bearing, +/−0 marginal) — its value is 1-step prediction, not multi-step rollout
substitution; and the exploit-penalty is negligible on this task.
