# ADR-0009 — Omni-modal seams: any modality in/out, specialized per deployment

**Status:** Accepted

## Context
R6 is "process any kind of input, produce any kind of output." The codec (ADR-0001) is
already a Perceiver-style any-to-any seam: per-modality **input adapters** (data → token),
one **shared trunk** (token → the single latent), and per-modality **decoders** (latent →
data). Every modality distils into the ONE shared latent the world model reasons over,
modality-agnostically; P6 proved it at N=2 (a pendulum as STATE and as an IMAGE lands at
the *same* latent).

The goal now grows to a **universally useful** agent — usable across private-user,
industrial, scientific, and robotic deployments. But those differ in almost everything
*except* the architecture: their **modalities** (vision/audio/text vs. sensors/force/
thermal vs. measurements/text/images), their **action/output spaces** (motor commands vs.
text vs. control signals), their **data trust** (proprietary sensors vs. the open web),
and their **constraints** (real-time, safety-critical). "Universally useful" therefore
forks: it could mean *one omniscient checkpoint* (infeasible — data, compute, capacity,
cross-domain interference) or *one universally-adaptable architecture* (feasible). This ADR
commits to the second, precisely.

## Decision
- **Omni-modal seams.** The codec admits **any input and output modality** through its
  modality registry (adapter + shared trunk + decoder), all landing in the one shared
  latent (ADR-0001). The seam is universal; the world model never sees a modality, only
  the latent.
- **Universal seams, specialized weights.** The architecture and seams are universal; a
  **deployment instantiates and trains the subset of modalities its environment needs** —
  robotics (vision + proprioception + force + motor-out), science (time-series + text +
  images), a private user (vision + audio + text). Same architecture, specialized instance.
  "Universally useful" means universally **adaptable**, *not* one checkpoint that knows
  everything. (A shared *foundation* pretrain that seeds instances is a separate, much
  larger ambition — real backends and scale — not assumed here.)
- **One modality per gate.** The universal seam is cheap and stays general; **competence in
  each concrete modality is earned by its own kill-gate** — the golden rule that generality
  is *earned*, not added in advance. We do not instantiate all seams speculatively. Vision
  (P12) is the first; audio, proprioception, text, action-outputs, and true variable/
  missing-modality handling are named **future gates**, added as a deployment demands them.
- **Frozen, swappable modules; distill-first.** A seam that needs a big pretrained network
  (e.g., a vision encoder) keeps it **frozen and harness-side** (optional deps, ONNX/torch);
  the **core stays numpy over embeddings**. Upgrading a module is a distill-first migration
  (P0-011): re-distil its adapter to the incumbent latent, core untouched. Gates run over
  **committed fixtures** so CI stays numpy-only and deterministic; live sensors/cameras are
  non-gated runtime.
- **Provenance is per-seam.** Every modality/source carries trust (ADR-0004). This matters
  most here: an untrusted or compromised modality — a spoofed camera, poisoned web text, a
  faulty sensor — must **never override the agent's goals**. A seam *conditions prediction*;
  it does not *set objectives*. "Who trains it, and on what" is a first-class trust question.

## Consequences
- (+) One architecture serves every deployment class by *instantiating seams*, not
  rebuilding — the "universally useful" goal made concrete and buildable.
- (+) The hard parts stay contained: real modules are frozen/optional/harness-side; the core
  and all P0–P11 gates stay numpy-only; new modalities are additive via distill-first.
- (+) A precise safety story — per-seam trust means *learn from anything, be governed by
  nothing untrusted*.
- (−) True arbitrary/missing-modality input needs the multi-head **cross-attention** the
  codec already flags as earned-later — a named future gate, not free.
- (−) Cross-modal alignment at scale (same situation → same latent across *many* modalities)
  needs paired/aligned data and real training — beyond the toy numpy codec.
- (−) Output/action modalities (motor, text) make the decoder + action space
  deployment-specific — the planner acts in whatever action modality is wired.
- Supersedes the earlier vision-only draft of this ADR: P12 (vision) is now the **first
  instantiation** of omni-modal seams; ADR-0009 governs all seams.
