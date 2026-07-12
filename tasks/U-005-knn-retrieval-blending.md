# U-005 — k>1 distance-kernel-weighted retrieval blending (replace nearest-1 substitution)

- **Status:** done
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
items (not 1). `RetrievalAugmentedWorldModel` and the direct P8/P9/P10 consumers blend
them: `pred ← (1-λ)·model_mean + λ·Σ softmax(-dist/τ)·answer`, with λ from
distance-scaled reliability. `KnowledgeSource.query` already returns
`list[KnowledgeItem]` — no protocol change, just k>1.

## Approach (brief)
- Kernel weights `w_j = softmax(-dist_j / τ)` over the k nearest facts; blended answer
  `Σ w_j · answer_j`. τ calibrated to the store's key scale by the harness (as the
  radius is).
- Blend against the model rather than substitute: `mean_i ← (1-λ_i)·mean_i + λ_i·blend`,
  where a hard radius gives `λ_i = max(0, 1 - distance/radius)` — an exact hit
  trusts the facts, a boundary hit keeps the model. Without a hard radius, the
  store-calibrated kernel supplies soft reliability `exp(-distance/τ)`.
- Radius-covered consumers exclude any neighbor that does not cover every relevant
  ensemble particle; accepted blends retain honest epistemic `epi × (1-λ)`.
- Poisoning robustness: a single poisoned nearest neighbor no longer controls the output
  (RobustRAG aggregate-across-k); the P8-002 poisoned-source gate should hold at least as
  well, ideally better.

## Acceptance criteria
- [x] Consumers blend k=2–3 distance-weighted facts with the model prediction; unit test
      shows a single far/poisoned neighbor moves the output strictly less than under
      nearest-1 substitution.
- [x] **P8 accuracy PASS, P8-002 poisoned-source robustness PASS (≥ current), P9/P10
      PASS**; `make gate-all` green.
- [x] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_memory.py, tests/test_knowledge.py): blend weights sum to 1; one
  poisoned neighbor among k=3 has bounded influence; k=1 + τ→0 reproduces the old
  substitution (backward-compat sanity).
- Eval: `make gate PHASE=P8`, `PHASE=P9`, `PHASE=P10`, `make gate-all`.

## Docs-sync checklist
- [x] Status → done; accuracy + poison robustness before/after recorded below.
- [x] ADR-0004: amend — retrieval readout is k>1 distance-kernel blending against the
      model (poisoning robustness by aggregation), radius/trust unchanged as outer gates.
- [x] architecture.md/memory docstring: nearest-1 → k>1 blended.
- [x] `docs/sota-review-2026-07.md`: mark U-005 shipped.

## Gate result

Both stores now return up to three ranked neighbors using partial selection rather
than a full hot-path sort. One shared readout covers batched prediction, TS∞ member
rollouts, internal retrieval, PointMass generalization, and external observations
after each neighbor passes through the codec. Unit tests verify ranked top-3 results,
stable weights summing to one, bounded influence from one exact-key poisoned neighbor,
the exact k=1 compatibility case, boundary fallback, and deterministic outer gates.

**P8 PASS** — final ratchet report `bench/results/P8-20260710T174157Z.json`.
Against the U-004 baseline `P8-20260710T165149Z.json`, median gated MSE improves
`0.00761 → 0.00504` (34%) versus unchanged model-only `0.02599`; always-retrieve
improves `0.01381 → 0.00558`. The old independently random all-poison source was
averaged away by design, so the negative control was strengthened—not its threshold—
to a coherent source-level corruption. Trust-blind MSE is `0.07462` (>2× model),
provenance-respecting remains `0.02599`, and trust-ordering recovers `0.00504`.

**P9 PASS** — final ratchet report `bench/results/P9-20260710T175254Z.json`.
Composed return improves `-17.38 → -14.62`; retrieval's median marginal moves from
`-3.31` to `+1.42` (safe/neutral), and PointMass gated retrieval MSE improves
`0.01230 → 0.00687` against unchanged model-only `0.01532`. Prediction, planning,
uncertainty, retrieval, independent calibration audits, and every sentinel pass.

**P10 PASS** — final ratchet report `bench/results/P10-20260710T175345Z.json`.
External codec-ingested blending remains a >2× OOD gain (`0.02198` vs `0.05070`),
with seen no-harm (`0.00124` vs `0.00123`), the radius ablation still harmful
(`0.00575` seen MSE), and trust-blind poison `0.26343` while provenance-respecting
stays at `0.02599`. The narrower OOD gain versus nearest-1 (`0.01468`) is the cost of
keeping model weight; all predeclared P10 competence, codec, gating, and robustness
criteria remain met.

`make test`: 147 passed, 1 skipped; Ruff and mypy clean. Final `make gate-all`:
**P0–P14 PASS** (`ratchet ok — 15 shipped gate(s) still green`). No deferred
U-101–U-112 trigger fired.
