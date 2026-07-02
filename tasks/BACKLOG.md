# Backlog (ordered)

Take the top **unblocked** item. Each line: ID · status · requirement · blocked-by.
Expand a one-liner into a full task file (from `TEMPLATE.md`) when you pick it up.

> Every phase gate also enforces its applicable **collapse sentinels** (ADR-0006):
> `representation-integrity` & `uncertainty-reliability` (from P1), `replay-fidelity`
> (from P3), `option-diversity` (from P5). A task is not *done* if its phase's
> sentinels are unhealthy — integrity is part of the gate, not a nice-to-have.

## Phase 1 — predictive core
- **P1-001** · `ready` · R1,R4 · — · Flat latent world model + calibrated uncertainty. **(fully specified — start here)**

## Phase 2 — planning
- **P2-001** · `blocked (P1-001)` · R1 · MPC/CEM planning in imagination; beat model-free at equal budget.

## Phase 3 — VoE, curriculum, replay
- **P3-001** · `blocked (P1-001)` · R3 · Calibrated surprise + epistemic/aleatoric decomposition + mastery test.
- **P3-002** · `blocked (P3-001)` · R3 · Curiosity/intrinsic-motivation curriculum (learning-progress driven).
- **P3-003** · `blocked (P1-001)` · R7 · Episodic replay buffer + generative replay (rehearsal from the model). Enforce `replay-fidelity`: real-data anchor + lineage cap + uncertainty-gated dreams (ADR-0006).

## Phase 4 — skills
- **P4-001** · `blocked (P3-001)` · R5 · Skill library with predictive preconditions + simulate-to-select router (competence-gated).

## Phase 5 — hierarchy
- **P5-001** · `blocked (P4-001)` · R2 · Abstraction map φ + jumpy option-model (landing latent, cumulative reward, duration, uncertainty).
- **P5-002** · `blocked (P5-001)` · R2 · Hierarchical manager (search over option-model) + VoE-triggered early termination. Gate: 2-level > flat at equal compute. Enforce `option-diversity` (ADR-0006).

## Phase 6 — any-to-any
- **P6-001** · `blocked (P2-001)` · R6 · Universal codec (Perceiver-IO-style) wrapper; swap preserves core-loop performance.

## Phase 7 — continual improvement
- **P7-001** · `blocked (P3-003)` · R7 · Forgetting/plasticity metrics + consolidation policy; retention + plasticity gate.

## Phase 8 — knowledge bases
- **P8-001** · `blocked (P3-003)` · R8 · Three-tier memory router + retrieval-as-action (uncertainty-gated).
- **P8-002** · `blocked (P8-001)` · R8 · Provenance/trust handling + poisoned/low-trust source robustness.
