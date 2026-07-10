# U-005 — k>1 distance-kernel-weighted retrieval blending (replace nearest-1 substitution)

- **Status:** ready
- **Phase:** U (upgrade track; re-gates against P8/P9/P10)
- **Requirements:** R8, R1
- **ADRs:** ADR-0004 (retrieval-as-action; distance-gated substitution, P9-007)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P8"]`, `["P9"]`, `["P10"]` — accuracy and the
  poisoned-source robustness check must hold or improve
- **Source:** `docs/sota-review-2026-07.md` U-005 · [kNN-LM gating](https://arxiv.org/abs/2210.15859)
  · [PoisonedRAG](https://arxiv.org/abs/2402.07867) · [RobustRAG](https://arxiv.org/abs/2405.15556)

## Goal
Replace nearest-1 hard substitution — the maximally noise- and poison-sensitive readout
— with k=2–3 distance-kernel-weighted blending of retrieved facts against the model's
*own* prediction. This is the converged answer of three literatures (kNN-LM adaptation,
episodic control, RAG poisoning defenses) and directly targets the observed P9-002
failure (far/noisy facts corrupting planning); the k>1 aggregation doubles as the
RobustRAG-style poisoning defense.

## Non-goals
- Keep the P9-007 reliability radius and the P8-002 trust floor as the outer gates —
  this changes the *readout*, not the gating (retrieve-when-uncertain AND trust-when-close
  are unchanged).
- No learned retriever / no learned gating network (review: non-learned similarity is
  sufficient at this scale — RA-DT finding; learned gating is not adopted).
- `SemanticStore.query` keeps returning a ranked list; the blend lives in the consumer.

## Interface to satisfy
`memory.SemanticStore.query` / `ExternalKnowledgeSource.query` return the k nearest
items (not 1) — memory.py:178-183, knowledge.py:70-77. `RetrievalAugmentedWorldModel._rows`
(memory.py:283-320) blends them: `pred ← (1-λ)·model_mean + λ·Σ softmax(-dist/τ)·answer`,
with λ from the distance-scaled reliability already computed at memory.py:315. `KnowledgeSource`
protocol: `query` already returns `list[KnowledgeItem]` — no signature change, just k>1.

## Approach (brief)
- Kernel weights `w_j = softmax(-dist_j / τ)` over the k nearest facts; blended answer
  `Σ w_j · answer_j`. τ calibrated to the store's key scale by the harness (as the
  radius is).
- Blend against the model rather than substitute: `mean_i ← (1-λ_i)·mean_i + λ_i·blend`,
  where `λ_i = min(1, ...)·(reliability)` reuses the P9-007 distance-scaling
  (memory.py:315) — an exact hit trusts the facts, a boundary hit keeps the model.
- Poisoning robustness: a single poisoned nearest neighbor no longer controls the output
  (RobustRAG aggregate-across-k); the P8-002 poisoned-source gate should hold at least as
  well, ideally better.

## Acceptance criteria
- [ ] Consumers blend k=2–3 distance-weighted facts with the model prediction; unit test
      shows a single far/poisoned neighbor moves the output strictly less than under
      nearest-1 substitution.
- [ ] **P8 accuracy PASS, P8-002 poisoned-source robustness PASS (≥ current), P9/P10
      PASS**; `make gate-all` green.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_memory.py, tests/test_knowledge.py): blend weights sum to 1; one
  poisoned neighbor among k=3 has bounded influence; k=1 + τ→0 reproduces the old
  substitution (backward-compat sanity).
- Eval: `make gate PHASE=P8`, `PHASE=P9`, `PHASE=P10`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; accuracy + poison robustness before/after recorded below.
- [ ] ADR-0004: amend — retrieval readout is k>1 distance-kernel blending against the
      model (poisoning robustness by aggregation), radius/trust unchanged as outer gates.
- [ ] architecture.md/memory docstring: nearest-1 → k>1 blended.
- [ ] `docs/sota-review-2026-07.md`: mark U-005 shipped.

## Gate result
<paste the GateResult once run>
