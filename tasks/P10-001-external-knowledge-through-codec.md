# P10-001 — External knowledge as content through the codec

- **Status:** done (capability half; composite P10 now PASS with P10-002)
- **Phase:** P10
- **Requirements:** R8 (external knowledge for any use case), R6 (codec ingestion), R4/R1
  (the epistemic signal that gates it)
- **ADRs:** ADR-0004 (retrieval-as-action; **rule 1** "knowledge is just more tokens
  through the codec" — exercised here for the first time), ADR-0001 (shared latent),
  ADR-0002 (uncertainty gates the query)
- **Depends on:** P8 (uncertainty-gated, trust-robust retrieval), P6 (the codec), P9-005
  (distance-aware epistemic — what makes OOD queries fire the gate)
- **Phase gate:** new `bench/gates.py::GATES["P10"]` — capability half here; composite
  PASS blocked pending P10-002 (trust robustness), mirroring P8-001/P8-002.

## Goal
Open the **external** knowledge tier for real. P8's `SemanticStore` answers with
next-latents *in the model's own space* (the agent's digested experience). This task
adds a source that answers with raw **content** the agent must **encode through the
codec** (ADR-0004 rule 1, so far unexercised) to recover a usable latent — extending
competence to a regime the parametric model **cannot derive from its own experience**.

## Non-goals
- No real network/web — "external" is an in-process oracle *treated* as external
  (deterministic gate). A real HTTP/DB source is a later, separately-gated adapter.
- No tool-use / compute-as-action (`ToolSource`) — that is Option B / P11.
- No new planner path — the gate measures **1-step prediction** (P8-001-style), not
  retrieval-into-planning (that composition is P9-007's concern, and is Option B's job).
- No trust/poison robustness here — that is P10-002 (reuses P8-002 machinery).

## Interface to satisfy
- Core: implement `knowledge.ExternalKnowledgeSource` (satisfies `interfaces.
  KnowledgeSource`) — a **task-unspecific** content source: nearest-key lookup over
  written `(key, content)` items, `trust` configurable. Answers carry raw content (an
  observation vector + modality), *not* a latent. The harness populates it (core imports
  no task).
- Ingestion is composed in the harness: `route → source.query → codec.encode(content) →
  latent`. The codec is `UniversalCodec` distilled to the model's `encode_target` (the
  prediction-target space), so an ingested fact is directly comparable to a prediction.

## Approach (brief, as built)
- **Env & split (reuse P8):** Pendulum, seen region (|omega| ≤ REGION) vs OOD band
  (|omega| in [REGION, FULL]). The model trains on the seen region → confident there and
  (P9-005 distance-aware) *uncertain* on OOD, so the gate fires on the external regime.
- **Codec (reuse P6):** distill a `UniversalCodec` to reproduce `encode_target` on the
  STATE modality over the full range, so `codec.encode(obs) ≈ encode_target(obs)`.
- **External KB:** `ExternalKnowledgeSource` holding **OOD-only** `(key=(latent,action),
  content=true-next-observation)` facts — deliberately *complementary* to the model, so
  misapplying it to a seen query returns an irrelevant fact (this is what makes the
  gating load-bearing rather than an always-query oracle).
- **Two-stage gate (the finding):** the uncertainty gate alone let false-consults on
  seen queries fetch irrelevant OOD facts and hurt. The P9-007 lesson applies to the
  external tier — retrieval must also be **distance-gated**: consult when uncertain
  (ADR-0004), ingest only when the fact is close/trustworthy (P9-007). On an accepted
  retrieval, `codec.encode(content)` stands in as the next-latent.

## Acceptance criteria (capability half; composite blocked pending P10-002)
- [x] **Competence extension:** on OOD queries, gated-external 1-step latent MSE beats
      model-alone by ≥ 2× (measured **3.4×**: 0.0153 vs 0.0527); seen no-harm (0.0014 vs
      0.0014).
- [x] **Codec-ingestion is load-bearing:** corrupting the retrieved content raises the
      MSE (metamorphic — the answer flows through the codec): **49.7×** (0.378 vs 0.0076);
      the ingested latent beats the model's OOD extrapolation (bypass control).
- [x] **Both gates load-bearing:** external ingested on OOD (83%) ~never on seen (3%); and
      removing the distance gate makes seen **4.4×** worse (0.0062 vs 0.0014).
- [x] `make test` green, `make lint` clean, `make typecheck` clean; P10 gate registered,
      capability recorded (composite BLOCKED pending P10-002 by design).

## Test plan
- Unit (tests/test_memory.py or test_knowledge.py): `ExternalKnowledgeSource` nearest-key
  lookup returns content; trust configurable; conformance to `KnowledgeSource`.
- Eval: `bench/evals/p10_external.py::check_p10` — the three capability criteria +
  metrics; registered as the P10 gate (reports capability; composite blocked).

## Docs-sync checklist
- [x] Status → capability `done`; gate result recorded below.
- [x] `knowledge.py` docstring updated (external tier implemented, not "out of scope").
- [x] ADR-0004: rule 1 now exercised (knowledge through the codec); the two-stage gate.
- [x] Requirement R8 traceability row: add P10 gate; roadmap P10 row.
- [x] BACKLOG P10-001 row.

## Gate result
`make gate PHASE=P10` → **[P10] BLOCKED** (composite blocked pending P10-002) with
**CAPABILITY MET** and all five collapse sentinels healthy (~2m). The capability half,
measured (median over 3 seeds):

| criterion | measured | bar |
|---|---|---|
| competence — OOD gated vs model MSE | **0.0153 vs 0.0527 (3.4×)** | ≥ 2× |
| competence — seen no-harm | 0.0014 vs 0.0014 | ≤ 1.3× |
| codec carries it — corrupted vs clean MSE | **0.378 vs 0.0076 (49.7×)** | ≥ 1.5× |
| gates — ingest rate seen / OOD | 3% / 83% | ≤ 5% / ≥ 50% |
| gates — remove distance-gate → seen MSE | 0.0062 vs 0.0014 (4.4×) | ≥ 1.5× |

**Finding (reported):** the uncertainty gate alone was not enough — false-consults on
seen queries fetched irrelevant OOD facts and hurt. The P9-007 distance-gating insight
generalizes to the external tier: reliability is closeness, so external retrieval needs
BOTH gates (consult-when-uncertain AND trust-when-close). The composite P10 gate ships
with P10-002 (external-source trust robustness).
