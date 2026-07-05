# Architecture

Prospect is a research scaffold for a **predictive-world-model agent**: it plans by
simulating the consequences of its actions, tests what it has learned by measuring
its own surprise, and adapts across use cases by attaching knowledge rather than
retraining.

## The one load-bearing idea
A predictive world model is the spine, and **prediction error — violation of
expectation (VoE) — is the single signal** reused throughout the system. Most
requirements are consumers of that core, not separate systems.

The model encodes an observation into a shared latent, and a dynamics function
predicts a *distribution* over the next latent state and reward, with uncertainty
split into **epistemic** (reducible by learning) and **aleatoric** (irreducible
noise). Prediction happens in *latent* space, not pixels — which forces the
representation to keep the controllable, outcome-relevant structure and drop
distractors (that is requirement R4).

## The one signal, many jobs
Surprise is computed as the negative log-likelihood of the observed outcome under
the predicted distribution (never raw L2), with epistemic isolated from aleatoric.
In code the signal is `types.Surprise` — the total NLL carrying its
epistemic/aleatoric attribution (P0-002); consumers gate on `.epistemic`, never on
the undecomposed total. The same quantity is reused as:

| # | Job | Requirement | Lives in |
|---|-----|-------------|----------|
| 1 | Learning signal — train where surprise is high | R1/R4 | world_model, voe |
| 2 | Mastery test — expected-vs-violated differential | R3 | voe |
| 3 | Skill-trust gate — select skill by predicted-outcome-under-uncertainty | R5 | skills |
| 4 | Re-planning interrupt — terminate an option on a surprise spike | R2 | planning |
| 5 | Forgetting detector — rising surprise on a mastered skill → rehearse | R7 | voe, memory |
| 6 | Retrieval trigger — high uncertainty → query a knowledge source | R8 | memory, knowledge |

Two of these jobs pull the signal in opposite directions: planning is *repelled*
from epistemic uncertainty (ADR-0006's exploitation control) while the curiosity
curriculum *seeks* it. The sign is mode-dependent — explore vs exploit — and the
curriculum owns the mode; neither consumer decides the sign itself (ADR-0007).

If a new requirement needs a brand-new bespoke signal, treat that as a smell: the
design's health is that additions plug into this backbone.

## Components (one file each, in `src/prospect/`)
- **agent.py** — the composition root (P2-002): the act–observe loop where the
  components meet — encode → plan → act; the monitor (P3), replay (P3) and
  retrieval-as-action (P8) plug into this one loop instead of re-inventing wiring.
- **codec.py** — universal encode/decode: any input → shared latent, latent → any
  output (R6). Retrieved knowledge enters through the *same* encoder (knowledge as
  tokens).
- **world_model.py** — latent dynamics; predicts a distribution over next latent +
  reward + uncertainty (R1, R4).
- **planning.py** — flat MPC/CEM in imagination (R1); the hierarchical manager over
  a *jumpy* option-model + VoE-triggered termination (R2).
- **voe.py** — calibrated surprise; epistemic/aleatoric split; competence/mastery
  (keyed on epistemic) and forgetting detection (keyed on rising prediction error —
  the ensemble is confidently wrong under shift, P7-001) (R3, R7); the
  learning-progress curriculum that owns the ADR-0007 explore/exploit mode flag
  (consumers read the sign, never pick it).
- **skills.py** — options with predictive preconditions; simulate-to-select router;
  only competence-gated (mastered) skills are offered upward (R5).
- **memory.py** — episodic replay + *generative* replay (rehearsal), a semantic
  store whose read side is a `KnowledgeSource` (one query verb, P0-008), and an
  uncertainty-gated, **provenance-respecting** router over the memory tiers: it may
  decline to retrieve (`None` = answer parametrically), and it selects among sources
  by `trust` — highest-trust above a `min_trust` floor wins, so an untrusted source
  never overrides the agent's own prediction (R7, R8; ADR-0004).
- **knowledge.py** — internal/external knowledge sources and tools as
  uncertainty-gated actions; each source declares a `trust` floor and each item
  carries provenance/trust — untrusted content is data, never instruction (R8).
- **types.py / interfaces.py** — shared types and the `Protocol` contracts every
  component satisfies. Components that learn additionally satisfy `Learner`
  (`update(batch) -> metrics dict`) — the uniform training seam the harness drives,
  and the channel through which sentinel metrics leave the training loop (P0-003).

## Hierarchy (R2), in one line
Hierarchical *planning* = a jumpy, option-conditioned world model + planning at each
level. A hierarchical *policy* alone gives reactive control; the abstract **model**
(predicting an option's landing state, cumulative reward, duration) is what lets the
manager plan. Two levels; generalise past two only if a gate demands it. See ADR-0003.

## Knowledge (R8), in one line
Generality across use cases comes from **decoupling reasoning (in weights) from
knowledge (in swappable stores)**. Three tiers — parametric (weights), internal
non-parametric (episodic + semantic + skills), external (docs/DBs/APIs/tools) — with
retrieval and tool-use as actions the planner selects, gated by uncertainty. See ADR-0004.

## What is deliberately hard (open problems we design *around*, not away)
- Compounding rollout error (bounded by hierarchy) — the main limiter on R1.
- Causal vs. spurious features (shortcut learning) — mitigated by acting = intervening.
- Skill composition beyond a flat menu.
- The generality tax of any-to-any I/O — including the P6 migration: a codec swap
  is a *representation* change, because everything downstream couples to the latent
  distribution (distill-first, retrain-fallback; ADR-0001, P0-011).
- Calibration under distribution shift — the whole VoE story rests on it.
- **Collapse of the shared latent** (constant / low-rank encoder) — because latent
  prediction is collapse-prone and everything reads this latent (ADR-0006).
- **Collapse of the uncertainty signal** (ensemble members agreeing where they are
  wrong) — removes the backbone all six VoE jobs depend on (ADR-0006).
- **Generative-replay collapse** (training on your own dreams → model autophagy) — the
  anti-forgetting mechanism turning into a forgetting one (ADR-0006).
- Catastrophic forgetting and loss of plasticity — the price of continual learning.
- Retrieval quality and trust of external sources — *and where retrieval is safe to
  apply*: the P9 integration gate found retrieval helps 1-step prediction (P8) yet
  **degrades multi-step planning** when it overrides the planner's rollout dynamics
  (ADR-0008; P9-002 dissects it).
- **Composing parts that each pass their own gate can still fail as a whole** — the
  reason Phase 9 validates the assembled agent, not just the components (ADR-0008).
- **Single-environment overfit** — capabilities are validated on a *second*,
  structurally different environment (`PointMass`, P9-003): prediction and planning
  generalize with the same core, but retrieval's benefit is env-dependent (the
  uncertainty signal must be OOD-sensitive, which it is not everywhere — ADR-0002/0008).

Each is named in the relevant ADR and gated by a benchmark; the collapse modes are
additionally guarded by standing integrity **sentinels** (ADR-0006), because they hide
in a good-looking loss. None is assumed solved.

## Glossary
- **VoE (violation of expectation)** — surprise; the calibrated prediction error.
- **Epistemic uncertainty** — reducible-by-learning; disagreement across an ensemble.
- **Aleatoric uncertainty** — irreducible environment noise; must not be mistaken for ignorance.
- **Latent world model** — dynamics learned in a compressed latent space (JEPA/Dreamer lineage).
- **Option / skill** — a temporally-extended behaviour: initiation predicate, policy, termination.
- **Jumpy / option-model** — a temporally-abstract model of an option's *whole* outcome.
- **Generative replay** — rehearsing old experience by dreaming it from the world model.
- **Competence gate** — the mastery test that decides which skills are trustworthy/selectable.
