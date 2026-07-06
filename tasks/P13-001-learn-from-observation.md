# P13-001 — Learn from passive observation (latent-action inference)

- **Status:** done
- **Phase:** P13
- **Requirements:** R7 (improve over time — from watching), R1 (predict), R8 (external
  observation as a knowledge source)
- **ADRs:** ADR-0010 (latent-action inference + the decorrelation identifiability fix),
  ADR-0002 (prediction error is the learning signal), ADR-0009 (the seam the observation
  stream enters through — state vectors now, visual embeddings later), ADR-0007 (the
  observe→exploit→explore curriculum this feeds)
- **Depends on:** P12 (perception seam), P1 (the predictive core / `_MLP`)
- **Phase gate:** new `bench/gates.py::GATES["P13"]` — a single-task phase; PASS ships it.

## Goal
Teach the agent to **learn from watching**: from a stream of observations with NO actions
and NO rewards, learn a predictive world model — and, crucially, recover the *action*
structure (behaviour), not just the physics — via latent-action inference. This is the
genuinely new component of the vision arc and the substrate for learning from video.

## Non-goals
- No imitation / reproducing behaviour — that is P14 (observe→repeat).
- No high-dim perception — P13 operates on observation *vectors* (state now; visual
  embeddings via P12 later); the perception encoder is ADR-0009's concern.
- No reward learning — action-free and self-supervised by construction.

## Interface to satisfy
- Core: `observation.LatentActionModel` (satisfies `interfaces.ObservationLearner`): an
  inverse model `infer_action(o_t, o_{t+1})` → latent action, a forward model
  `predict(o_t, latent_action)` → o_{t+1}, and `observe(o_t, o_{t+1})` — the action-free
  training verb (no `Transition`, no reward). Built on the world model's `_MLP`.

## Approach (brief, measured first)
- **Diagnosis (a prototype, before building):** a naive latent-action bottleneck
  reconstructs the next observation ~1850× better than persistence yet its latent recovers
  the true action at R² **0.02** — it captures a state-*dependent* feature (next velocity),
  not the action. This is the known identifiability problem.
- **Fix (ADR-0010):** a **decorrelation penalty** pushes the latent action uncorrelated
  with the current observation, so it captures the state-*independent* controllable factor
  — the action. Measured: recovery R² 0.02 → **~0.80** (linear corr ~0.9) with
  reconstruction still >150× better than persistence.
- **Transfer:** watching is a *low-data* prior — fit a small true-action → latent-action map
  on a few labels with the forward model frozen, and it beats from-scratch when labels are
  scarce (past a modest budget, direct learning catches up — the honest boundary).

## Acceptance criteria (single-task phase — PASS ships)
- [x] **Learns dynamics by watching:** latent-action reconstruction MSE beats a persistence
      baseline by ≥ RECON_MARGIN.
- [x] **Recovers hidden actions:** the latent action decodes to the true action with R² ≥
      0.5, and a shuffled-label control is ~0 (the negative control).
- [x] **Watching transfers:** at a small labelled budget, watch-first prediction MSE beats
      from-scratch by ≥ TRANSFER_MARGIN.
- [x] `make gate PHASE=P13` PASS, all sentinels healthy; P13 appended to `bench/SHIPPED`;
      `make gate-all` green; `make test`/`lint`/`typecheck` clean.

## Test plan
- Unit (tests/test_observation.py): action-free `observe` reduces reconstruction; infer/
  predict shapes (batch and single pair); decorrelation metric reported. Conformance to
  `ObservationLearner` (tests/test_conformance.py).
- Eval: `bench/evals/p13_observation.py::check_p13` — the three criteria + sentinels.

## Docs-sync checklist
- [x] Status → `done`; gate result recorded below.
- [x] ADR-0010 added (Accepted); ADR index updated.
- [x] R7 traceability row (+P13, ADR-0010); roadmap P13 row; BACKLOG P13 + shipped note.

## Gate result
`make gate PHASE=P13` → **[P13] PASS**, all five collapse sentinels healthy (~2m). Median
over 3 seeds:

| criterion | measured | bar |
|---|---|---|
| learns dynamics — recon vs persistence MSE | **0.0009 vs 0.3777 (420×)** | recon·5 ≤ persist |
| recovers hidden actions — decode R² | **0.811** (shuffled +0.010) | ≥ 0.5, shuffle ≤ 0.1 |
| transfers (N=40 labels) — watch-first vs from-scratch MSE | **0.0027 vs 0.0208 (7.7×)** | ≥ 1.5× |

The recovery R² 0.811 with a ~0 shuffled control is the headline: the decorrelation penalty
(ADR-0010) makes the latent action recover the *true* hidden action from an action-free
stream. **P13 ships** (`bench/SHIPPED` ratchets P0–P13). Honest boundary: the transfer win
is a low-data-regime advantage — past ~60 labels direct action-conditioned learning catches
up (measured in the prototype, reported not hidden). Watching is a prior, not a substitute
for acting — which is exactly why the loop ends in explore (P3-002).
