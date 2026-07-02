# ADR-0009 — Capacity growth by additive isolatable units, triggered by a VoE plateau

**Status:** Accepted

## Context
Generality across use cases (R5, R7) is supposed to come from *attaching* skills and
knowledge, not retraining a monolith (ADR-0004). A small model stays small by adding
**narrow** capacity where it is needed rather than scaling everything. That needs two
things a design usually leaves implicit: (a) a growth *unit* that cannot damage what
already works, and (b) a growth *trigger* that is principled rather than ad hoc.

## Decision
- **Growth unit — additive and isolatable.** New capacity is a fresh unit (a new
  option/skill, or an adapter-style module) registered with its influence **gated near
  zero**, so `gate ≈ 0` recovers prior behaviour **exactly** — improvements are
  additive, never catastrophic. Freeze the backbone, train only the new unit;
  optionally distill it back into the backbone later on a slow schedule, then drop it.
  This is the LoRA / Progressive-Networks family; OmniLatent's `LatentNeuralHook`
  (attention-injected latent tokens behind a near-zero sigmoid gate) is one concrete
  realization — Prospect's realization is a new option/skill, but the **additive-safety
  property is the invariant that matters**, not the mechanism.
- **Trigger — a VoE learning-progress plateau.** Grow when epistemic uncertainty stops
  falling and learning progress flattens despite priority. Prospect already defines
  exactly this quantity — "learned = low epistemic uncertainty + flattened learning
  progress" (ADR-0002) — so the trigger is the **native VoE signal**, not a bespoke
  plateau detector bolted on.
- **Safety — growth must pay its way.** A new unit is promoted only after it clears the
  ADR-0007 quality floor (no regression on the frozen probe); for options it must also
  keep the ADR-0006 `option-diversity` sentinel healthy (no collapse to identical /
  one-step options). New units become selectable by **simulate-to-select**
  (ADR-0003) — the world-model-grounded router — not a learned key-matching router
  (ADR-0003 note).

## Consequences
- (+) "Attach rather than retrain" gets a concrete, additive-safe growth mechanism with
  a principled trigger that reuses the one signal — no new bespoke detector.
- (+) Capacity expands on demand and is immediately usable by the planner, without
  touching frozen behaviour.
- (−) Unbounded growth is a real risk; it is bounded by the floor (a unit that does not
  earn probe improvement is not kept) and by gate-ordered introduction (options at P4,
  continual growth at P7).
- (−) Distill-back-and-drop is itself a small continual-learning problem — guarded by
  the ADR-0008 stack.
