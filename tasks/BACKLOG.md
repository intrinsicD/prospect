# Backlog (ordered)

Take the top **unblocked** item. Each line: ID · status · requirement · blocked-by.
Expand a one-liner into a full task file (from `TEMPLATE.md`) when you pick it up.

> Every phase gate also enforces its applicable **collapse sentinels** (ADR-0006):
> `representation-integrity` & `uncertainty-reliability` (from P1), `replay-fidelity`
> (from P3), `option-diversity` (from P5). A task is not *done* if its phase's
> sentinels are unhealthy — integrity is part of the gate, not a nice-to-have.

## Phase 0 — contract & harness hardening (from the architecture review; pre-P1)
> These fix seams, not behaviour: cheap before P1, expensive after P3. All eleven
> have full task files. The two docs tasks (P0-010, P0-011) can run in parallel
> with anything.

- **P0-001** · `done` · R1,R3,R4 · — · `Prediction` parameterizes a real distribution: per-dim `var`, concrete `log_prob`, `duration` for option outcomes.
- **P0-002** · `done` · R3,R5,R7 · Decomposed `Surprise` type (no bare-float VoE) + `Transition.option` for per-skill attribution.
- **P0-003** · `done` · R1,R7 · `Learner` protocol — the uniform training seam the harness drives (P1 trains through it; P7's gate depends on it).
- **P0-004** · `done` · R1 · `Environment` protocol in `bench/` (harness-owned; core never imports the harness).
- **P0-005** · `done` · — · Run-metrics artifact (JSONL run log) — the data zero-arg sentinel `check()`s read to verify "throughout training" (ADR-0006).
- **P0-006** · `done` · — · Gate wiring: `@gate_check`/`@sentinel_check` registration, `metrics: dict`, persisted gate reports, register the P0 gate, friendly errors, explicit seed policy.
- **P0-007** · `done` · — · Regression ratchet: `bench/SHIPPED` + `make gate-all` + CI job — shipped gates stay green.
- **P0-008** · `done` · R8 · One query path into knowledge (`SemanticMemory` read-side *is* a `KnowledgeSource`), `route() -> KnowledgeSource | None` (None = parametric), provenance-`None` convention documented.
- **P0-009** · `done` · — · Enforce typing: mypy in CI, typed protocol-conformance assertions, ruff `I`, CI matrix 3.11–3.13.
- **P0-010** · `done` · R1,R3,R7 · ADR-0007: arbitration of the epistemic signal — curiosity seeks it, planning penalizes it; mode chosen by the curriculum. Plus shift-disambiguation note in ADR-0002. *(docs)*
- **P0-011** · `done` · R6 · Roadmap/ADR-0001 amendment: the P6 codec swap is a representation change — distill-first, retrain-fallback; replay keeps raw obs re-encodable. *(docs)*

> **Phase 0 complete** — all eleven contract & harness hardening tasks are `done`.

## Phase 1 — predictive core
- **P1-001** · `done` · R1,R4 · Flat latent world model + calibrated uncertainty. Gate **P1 PASS** (shipped; see task file for the report and the collapse-fighting lessons).

> **Phase 1 shipped** — `bench/SHIPPED` now ratchets P0 + P1.

## Phase 2 — planning
- **P2-001** · `done` · R1 · MPC/CEM planning in imagination; beat model-free at equal budget. Gate **P2 PASS** on every seed (shipped; see task file).
- **P2-002** · `done` · R1 · Composition root: `agent.py` act–observe loop + `bench.loop.run_episode` — one place the components meet; P2 gate reproduced byte-identically through it.

## Phase 3 — VoE, curriculum, replay
- **P3-001** · `done` · R3 · Calibrated surprise + decomposition + mastery test. Differential criterion **MET** (P(violated>expected) ≥ 0.93 every seed); P3 composite blocked pending P3-002 + P3-003 by design.
- **P3-002** · `done` · R3 · Curiosity curriculum. P3 **capability ok** (differential MET + curiosity MET: coverage ratio 0.26 vs 0.79 at equal budget); composite blocked only by the P3-003 sentinel.
- **P3-003** · `done` · R7 · Episodic replay + generative replay + `replay-fidelity` sentinel (real anchor 0.50, dream diversity 0.47, lineage ≤ 3, zero dreams stored). Gate **P3 PASS** (shipped).

> **Phase 3 shipped** — `bench/SHIPPED` now ratchets P0–P3. The one signal is
> live end-to-end: calibrated decomposed surprise, mastery, mode arbitration,
> curiosity, and collapse-guarded rehearsal.

## Phase 4 — skills
- **P4-001** · `done` · R5 · Skill router: simulate-to-select (accuracy 0.83–0.92 vs 0.33 baseline), paired closed-loop misapplication VoE (win rate ≥ 0.95), competence gating with calibrated mastery. `Option` gained typed `policy`/`horizon`; the precondition is *computed* (predictive), not stored. Gate **P4 PASS** (shipped).

> **Phase 4 shipped** — `bench/SHIPPED` now ratchets P0–P4.

## Phase 5 — hierarchy
- **P5-001** · `done` · R2 · Jumpy option-model (landing distribution, cumulative reward, duration): **beats the flat rollout 4–6x on every seed** — ADR-0003's compounding bound, measured. Composite blocked pending P5-002 by design.
- **P5-002** · `done` · R2 · Hierarchical manager (exhaustive search over the jumpy option-model) + VoE-triggered early termination + `option-diversity` sentinel. Two-level beats compute-matched flat on every seed (−9.1/−4.2/−4.7 vs −48.5/−34.9/−14.0), and beats full-compute flat too. Gate **P5 PASS** (shipped).

> **Phase 5 shipped** — `bench/SHIPPED` now ratchets P0–P5. Hierarchical
> *planning* (jumpy model + search + VoE termination) beats flat control at equal
> compute; all six phases green in ~3m30s.

## Phase 6 — any-to-any
- **P6-001** · `done` · R6 · Universal codec distilled into the incumbent latent (P0-011 migration, validated): swap-in 1-step MSE ratio ~1.0 for STATE **and** a rasterized IMAGE modality — the frozen core loop predicts from an image as from a state vector (any-to-any, measured). Gate **P6 PASS** (shipped).

> **Phase 6 shipped** — `bench/SHIPPED` now ratchets P0–P6. The distill-first
> migration (P0-011) is proven: the dynamics model is never retrained, yet its
> encoder swaps modality with <2% core-loop impact.

## Phase 7 — continual improvement
- **P7-001** · `done` · R7 · Continual improvement: `is_forgetting` (error-keyed, ADR-0002 amended) + rehearsal consolidation. Retention 3-5x better than naive; plasticity retained; naive loses both. Gate **P7 PASS** (shipped).

> **Phase 7 shipped** — `bench/SHIPPED` now ratchets P0–P7. The consolidation
> discipline preserves the memory AND plasticity that naive continual learning
> loses; forgetting detection keys on prediction error (the ensemble is
> confidently wrong under shift).

## Phase 8 — knowledge bases
- **P8-001** · `ready` · R8 · Three-tier memory router + retrieval-as-action (uncertainty-gated). `route()` may return `None` (answer parametrically); retrieval surfaces to the planner as options (ADR-0004, P0-008). **(start here — last capability phase)**
- **P8-002** · `blocked (P8-001)` · R8 · Provenance/trust handling + poisoned/low-trust source robustness.
