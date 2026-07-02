# ADR-0008 — Continual-learning defense-in-depth (the R7 mechanism)

**Status:** Accepted

## Context
R7 (improve over time) and its P7 gate (retention above threshold **and** plasticity
retained) name *what* to achieve but no *mechanism*. `memory.py` promises episodic +
generative replay, nothing more. No single anti-forgetting technique is sufficient on
its own; the field's strongest results come from stacking several. External evidence:
the OmniLatent harness converged on a five-layer stack (replay + EMA + EWC/SI +
parameter-isolation + rollback) as the defense-in-depth for catastrophic forgetting.

## Decision
Implement continual learning as a **defense-in-depth stack**, introduced
**gate-by-gate** (minimal-implementation, ADR-0005) — not all at once:

1. **Experience replay** (substrate, from P3). Reservoir-sampled buffer; Dark
   Experience Replay (DER++): store the model's *outputs* at insertion and add a
   consistency term. Storage form is **compressed real data**, not dreams
   (ADR-0006 note).
2. **EMA teacher / self-distillation** (from P3). Reuse the *same* EMA target encoder
   ADR-0006 already mandates for representation anti-collapse — now also the LwF-style
   distillation teacher, the rollback target (ADR-0007), and the export weights. **One
   EMA, three jobs.**
3. **Online EWC + Synaptic Intelligence** (from P7, backbone). Anchor important
   parameters with a quadratic penalty; SI accumulates importance during the normal
   backward pass (cheap, no second pass); average the two importances.
4. **Parameter isolation** (from P4/P7). Additive, isolatable capacity units gated
   near zero so prior behaviour is recovered exactly — see ADR-0009.
5. **Gradient surgery (A-GEM, opt-in).** Project a batch gradient that would raise
   replay loss. Off by default; switch on per-component under persistent regression.
6. **Rollback (always).** The ADR-0007 quality floor is the last line of defense.

**Core vs harness.** The stack lives in the harness. The core exposes only the
VoE-derived signals the harness reads — `learning_progress`, a `Competence` estimate
— plus `state_dict()`/`load_state_dict()` on components. No task-specific memory
machinery enters `src/prospect/`.

**Metrics in Prospect's vocabulary.** Retention = held-out **calibrated surprise
(NLL)** on earlier tasks staying below threshold; plasticity = late tasks learning as
fast as early ones. Every step is guarded by the ADR-0006 `replay-fidelity` sentinel
and the ADR-0007 floor.

## Consequences
- (+) R7 gets a concrete, literature-grounded mechanism where each layer covers what
  the others miss.
- (+) Reuses machinery already in the design (the EMA target encoder, the VoE
  learning-progress signal, the floor) instead of spawning parallel systems.
- (−) Five composable methods are a real tuning surface; each is introduced only as
  the P3/P7 gate demands it, never ahead of it.
- Supersedes nothing; extends R7's implementation surface under ADR-0005's
  gate-ordered discipline. What we explicitly do **not** default to: generative replay
  (ADR-0006 note), a learned expert-router (ADR-0003 note).
