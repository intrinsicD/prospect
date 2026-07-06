# ADR-0010 — Learning from action-free observation via latent-action inference

**Status:** Accepted

## Context
The vision arc (ADR-0009) wants the agent to **learn from watching** — video / passive
observation, where there is a firehose of "predict what happens next" but **no actions and
no rewards**. This is the natural substrate for the predictive spine (ADR-0002): prediction
error over a stream is exactly the learning signal. But the world model, planner and VoE are
all *action-conditioned* (`predict(state, action)`), and a watched stream has no actions.

Two things can be learned from action-free observation: the **physics** (how the world
evolves) and the **behaviour** (what the actor did). Physics is learnable by predicting the
next observation; behaviour requires recovering the *action* structure without ever seeing
an action. Naively bottlenecking "the part of the next observation not explained by the
current one" does **not** recover the action — measured on Pendulum, a 1-D bottleneck
reconstructs the next observation ~1850× better than persistence yet its latent has ~0 (R²
0.02) recovery of the true action: it captures a *state-dependent* feature of the next state
(e.g. the next velocity), not the state-*independent* action. This is the known
identifiability problem in latent-action learning (LAPO / Genie / ILPO lineage).

## Decision
Add **latent-action inference** as the way to learn from action-free observation
(`observation.LatentActionModel`, P13):
- An **inverse model** infers a low-dimensional **latent action** between consecutive
  observations (`infer_action(o_t, o_{t+1})`), and a **forward model** predicts the next
  observation from it (`predict(o_t, latent_action)`). They train jointly to reconstruct
  `o_{t+1}` — learning the dynamics by watching, with the latent action as the bottleneck.
- **The load-bearing detail — a decorrelation penalty for identifiability.** The latent
  action is pushed to be **uncorrelated with the current observation**, so it captures the
  state-*independent* controllable factor — the action — rather than a state-feature of the
  next observation. Measured: this lifts action recovery from R² 0.02 to **~0.80** (linear
  corr ~0.9) while reconstruction still beats persistence >150×.
- **Reward-free and self-supervised** — no reward is learned here; the action-conditioned
  reward/control layers build on the learned representation. Watching provides a *prior*;
  acting (exploration, P3-002) closes what watching cannot teach — proprioception, contact,
  causal intervention. The value is the **low-data regime**: pretraining on action-free
  observation gives a head start when action-labelled interaction is scarce.
- **Provenance:** watched observation is external, UNTRUSTED (ADR-0004) — it conditions
  prediction, it never sets goals.

## Consequences
- (+) The predictive spine can now ingest a passive observation stream — the substrate for
  learning from video (ADR-0009): the same code runs on state vectors or visual embeddings.
- (+) A precise notion of "learn by watching": recover the action structure *and* transfer
  it, both measured (P13 gate), with the decorrelation penalty as the identifiability fix.
- (+) It composes with the existing loop: latent actions are candidate skills/behaviours to
  imitate (P14, observe→repeat), and the curriculum (ADR-0007) already arbitrates
  observe→exploit→explore.
- (−) The transfer benefit is a **low-data-regime** advantage — past a modest label budget,
  direct action-conditioned learning catches up (measured, reported not hidden). Watching is
  a prior, not a substitute for acting.
- (−) Identifiability is only *practically* resolved (decorrelation), not guaranteed; the
  latent action recovers the true action up to an invertible map, which is what transfer
  needs, not exact equality.
- (−) High-dimensional observation (real video) still needs the frozen perception encoder
  (ADR-0009); this ADR is about the action-free *learning*, orthogonal to perception.
