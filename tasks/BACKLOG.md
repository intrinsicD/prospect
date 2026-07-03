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
- **P1-001** · `ready` · R1,R4 · Flat latent world model + calibrated uncertainty. **(start here — fully specified; all P0 dependencies done)**

## Phase 2 — planning
- **P2-001** · `blocked (P1-001)` · R1 · MPC/CEM planning in imagination; beat model-free at equal budget. Uncertainty-penalty sign per ADR-0007 (P0-010).
- **P2-002** · `blocked (P2-001)` · R1 · Composition root: `agent.py` act–observe–learn loop (env + world model + planner + monitor) — one place the components meet, so gate evals stop re-inventing wiring.

## Phase 3 — VoE, curriculum, replay
- **P3-001** · `blocked (P1-001, P0-002)` · R3 · Calibrated surprise + epistemic/aleatoric decomposition + mastery test. Returns `types.Surprise` (P0-002), never a bare float.
- **P3-002** · `blocked (P3-001)` · R3 · Curiosity/intrinsic-motivation curriculum (learning-progress driven). Owns the explore/exploit mode flag per ADR-0007 (P0-010).
- **P3-003** · `blocked (P1-001)` · R7 · Episodic replay buffer + generative replay (rehearsal from the model). Enforce `replay-fidelity`: real-data anchor + lineage cap + uncertainty-gated dreams (ADR-0006). Retain raw observations so experience stays re-encodable under a future codec (P0-011).

## Phase 4 — skills
- **P4-001** · `blocked (P3-001)` · R5 · Skill library with predictive preconditions + simulate-to-select router (competence-gated). Promote the precondition to a **typed field** on `Option` when this lands — no metadata-dict convention. Executors set `Transition.option` so competence attribution works (P0-002).

## Phase 5 — hierarchy
- **P5-001** · `blocked (P4-001)` · R2 · Abstraction map φ + jumpy option-model (landing latent, cumulative reward, duration, uncertainty).
- **P5-002** · `blocked (P5-001)` · R2 · Hierarchical manager (search over option-model) + VoE-triggered early termination. Gate: 2-level > flat at equal compute. Enforce `option-diversity` (ADR-0006).

## Phase 6 — any-to-any
- **P6-001** · `blocked (P2-001)` · R6 · Universal codec (Perceiver-IO-style) wrapper; swap preserves core-loop performance. Migration per P0-011: **distill into the incumbent latent space first**; budgeted full-stack retrain only as fallback (ADR-0001).

## Phase 7 — continual improvement
- **P7-001** · `blocked (P3-003)` · R7 · Forgetting/plasticity metrics + consolidation policy; retention + plasticity gate.

## Phase 8 — knowledge bases
- **P8-001** · `blocked (P3-003, P0-008)` · R8 · Three-tier memory router + retrieval-as-action (uncertainty-gated). `route()` may return `None` (answer parametrically); retrieval surfaces to the planner as options (ADR-0004, P0-008).
- **P8-002** · `blocked (P8-001)` · R8 · Provenance/trust handling + poisoned/low-trust source robustness.
