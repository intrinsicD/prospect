# P5-001 — The jumpy option-model (learned temporal abstraction)

- **Status:** done
- **Phase:** P5
- **Requirements:** R2
- **ADRs:** ADR-0003 (the jumpy model is what makes hierarchical *planning*
  possible; bounds ADR-0001's compounding error), ADR-0002 (distribution out,
  never a point)
- **Depends on:** P4-001 (options + the flat-rollout baseline it replaces)
- **Phase gate:** `bench/gates.py::GATES["P5"]` — the composite (two-level beats
  flat at equal compute) is P5-002's, and the `option-diversity` sentinel also
  arrives with P5-002; this task records the jumpy model's own measurable claim
  and the gate stays honestly BLOCKED.

## Goal
`JumpyOptionModel` satisfying `interfaces.OptionModel` + `interfaces.Learner`:
a *learned* ensemble model that predicts the outcome of committing to an option
in one jump — landing-latent distribution, cumulative discounted reward, and
duration (`Prediction` has carried `duration` since P0-001 for exactly this).
The measurable claim (ADR-0003's reason to exist): the jumpy landing prediction
beats the P4 router's flat one-step-composed rollout on held-out option
executions — temporal abstraction bounds compounding error.

## Non-goals
- No hierarchical manager, no VoE-triggered termination, no option-diversity
  sentinel (all P5-002).
- No new skills or environment: the P4 reference task and constant-torque
  options are reused (continuity with the router this model upgrades).
- No encoder of its own: the jumpy model lives in the shared latent (ADR-0001);
  anti-collapse belongs to the encoder's owner (the flat world model).

## Interface to satisfy
`interfaces.OptionModel` (`predict_option`) and `interfaces.Learner` (`update`)
— implement in `prospect/planning.py` (replace the skeleton). Training
convention (documented): option-transitions are latent-space — `state.z` /
`next_state.z` are latents (encode with E, target with Ē, mirroring the flat
model's objective), `option` set, `reward` = cumulative discounted; the duration
target is `option.metadata["duration"]` when present (varying durations arrive
with P5-002's early termination) else `option.horizon`.

## Approach (brief)
- Ensemble of MLPs over `[latent ⊕ one-hot(option)]` with residual Gaussian
  landing heads (disagreement = epistemic, predicted variance = aleatoric,
  moment-matched total — same discipline as P1); shared reward and duration
  heads; Adam; bootstrap resampling per member (decorrelation).
- Eval (`bench/evals/p5_options.py`, `@gate_check("P5")`, run `p5` with P1
  probes + replay records so active sentinels judge this phase's model): per
  seed, train the flat model (P4 recipe), collect real option executions,
  train the jumpy model, and compare held-out landing MSE (target-latent space)
  against the flat rollout — `jumpy < flat` required per seed for the partial
  criterion; reward/duration errors and calibration recorded. Capability stays
  `passed=False` with "manager pending (P5-002)" named.

## Acceptance criteria
- [x] Implements `OptionModel` + `Learner`; conformance assertions added.
- [x] Learns on synthetic option data: landing means, cumulative rewards, and
      durations recovered (incl. `metadata["duration"]` override); epistemic
      falls with training; unknown options (KeyError) and option-less
      transitions (ValueError) rejected loudly (unit-tested).
- [x] **Jumpy beats flat rollout on every seed** — landing MSE 0.55/0.65/0.57
      vs 2.44/2.05/3.32 (4–6x): one learned jump vs composing five one-step
      predictions. ADR-0003's claim, measured. Recorded in the P5 gate metrics.
- [x] `make test` green (79), `make lint` clean, `make typecheck` clean;
      `gate-all` (P0–P4) green.

## Test plan
- Unit (tests/test_options.py): synthetic two-option system — landing/reward/
  duration recovery, epistemic decline with training, unknown-option KeyError,
  option-less ValueError, protocol conformance.
- Eval: `make gate PHASE=P5` — jumpy-vs-flat metrics per seed; sentinel states
  as designed (P1-era + replay healthy on run `p5`; option-diversity PENDING).

## Docs-sync checklist
- [x] Status → `done`; gate report (BLOCKED composite, jumpy-vs-flat met) below.
- [x] architecture.md planning.py note still accurate (flat MPC + jumpy model +
      manager; the manager is what remains).
- [x] Backlog: P5-001 done; P5-002 unblocked (start here).

## Gate result
`make gate PHASE=P5` — record `bench/results/P5-20260704T064930Z.json`:

```
[P5] BLOCKED   (by design at this task — the manager is P5-002)
  capability: not met — jumpy landing MSE per seed [0.55, 0.65, 0.57] vs flat
    rollout [2.44, 2.05, 3.32] — jumpy beats flat on every seed: YES;
    hierarchical manager pending (P5-002)
  sentinels: representation-integrity / uncertainty-reliability /
    replay-fidelity healthy on run p5; option-diversity PENDING (P5-002)
```

The temporally-abstract model's reason to exist is now a measurement, not a
belief: one learned jump is 4–6x more accurate than composing one-step
predictions over the same options — the compounding-error bound ADR-0003
promised. A regression note for the record: registering the P5 check broke a
smoke test premised on P5 being PENDING (it silently ran the full eval inside
pytest); the ratchet caught the cascade within minutes and the test now uses
P6, the furthest still-pending phase. `gate-all`: 5 shipped gates green.
