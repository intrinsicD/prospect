# P3-002 — Curiosity curriculum: learning-progress-driven exploration

- **Status:** done
- **Phase:** P3
- **Requirements:** R3
- **ADRs:** ADR-0002 (epistemic, not raw, surprise — noisy-TV defense),
  ADR-0007 (the curriculum owns the explore/exploit mode flag)
- **Depends on:** P3-001, P2-001
- **Phase gate:** `bench/gates.py::GATES["P3"]` — this task completes the
  **capability** criterion (differential MET in P3-001 + curiosity here); the
  composite still waits on the `replay-fidelity` sentinel (P3-003).

## Goal
The ADR-0007 arbiter exists in code: `LearningProgressCurriculum` decides the
mode (EXPLORE until the skill is mastered, EXPLOIT after) and exposes the signed
epistemic coefficient consumers apply — the planner never chooses the sign. The
gate's measurable half: curiosity-driven data collection (explore-mode planning
toward epistemic uncertainty) trains a better world model than random collection
at the SAME env-step budget and training schedule.

## Non-goals
- No replay/rehearsal (P3-003). No skill library (P4).
- No new planner: explore mode = the P2 `FlatPlanner` with the curriculum's
  negative coefficient (epistemic bonus) — one consumer, one flag, per ADR-0007.
- The bonus is epistemic-only by construction (`Prediction.epistemic`): aleatoric
  noise is never rewarded (ADR-0002/0006 noisy-TV defense) — no extra mechanism
  needed.

## Interface to satisfy
No new `Protocol` (the curriculum is R3 logic in `voe.py`, reading the
`CompetenceMonitor`). New shared type: `types.Mode` (EXPLORE/EXPLOIT). Public
surface: `mode()`, `uncertainty_coefficient()`.

## Approach (brief)
- `types.Mode` StrEnum; `LearningProgressCurriculum(monitor, skill, explore_bonus,
  exploit_penalty)`: `mode()` = EXPLOIT iff `monitor.is_mastered(skill)`;
  `uncertainty_coefficient()` = `+exploit_penalty` in EXPLOIT, `-explore_bonus`
  in EXPLORE. Consumers assign it to `FlatPlanner.uncertainty_penalty`.
- Gate experiment (extends `bench/evals/p3_voe.py::check_p3`): per seed, two
  active-learning arms with identical budgets and training schedules —
  * curious arm: seed chunk random (untrained model), then chunks collected by
    an explore-mode planner (coefficient from the curriculum, monitor fed each
    round so the mode arbitration is the real mechanism, not a hardcoded sign);
  * random arm: all chunks random.
  Score both models on a uniform-coverage test set (states placed via
  `set_state`, so neither collector's visitation biases the eval) with the
  scale-free ratio `model MSE / persistence MSE` in each model's own target
  latent space (raw latent MSE is not comparable across different encoders).
  Curiosity criterion: curious ratio < random ratio (median over seeds).
- `check_p3.passed` becomes `differential_met AND curiosity_met` — the true P3
  capability; the composite stays BLOCKED on the pending replay sentinel.

## Acceptance criteria
- [x] Curriculum: unmastered ⇒ EXPLORE and a negative coefficient; mastered ⇒
      EXPLOIT and a positive one (unit-tested; the sign lives here only).
- [x] Gate experiment: curiosity beats random collection on **every seed**, not
      just the median — coverage ratios (curious/random): s0 0.26/1.33,
      s1 0.35/0.76, s2 0.17/0.79 (2–5x better model from the same budget).
- [x] `check_p3` capability **ok** (differential MET + curiosity MET); composite
      BLOCKED only by the pending `replay-fidelity` sentinel (P3-003).
- [x] `make test` green (62), `make lint` clean, `make typecheck` clean;
      ratchet (P0–P2) verified green below.

## Test plan
- Unit (tests/test_voe.py): mode arbitration + coefficient signs with a stub
  monitor in both states.
- Eval: `make gate PHASE=P3` — differential + curiosity metrics per seed.

## Docs-sync checklist
- [x] Status → `done`; gate report below.
- [x] ADR-0007 status note: implemented by P3-002 (decision unchanged).
- [x] architecture.md voe.py note mentions the curriculum owns the mode.
- [x] Backlog updated (P3-002 done; P3-003 next — last piece of P3).

## Gate result
`make gate PHASE=P3`:

```
[P3] BLOCKED   (capability OK; only the P3-003 sentinel remains)
  capability: ok — differential: P(violated surprise > expected) per seed
    [0.98, 0.93, 0.93], min 0.93 (>= 0.9) — MET; curiosity: coverage ratio
    (model/persistence MSE, lower is better) curious 0.26 vs random 0.79 at
    equal budget (1536 steps) — MET
  sentinel[representation-integrity]: healthy
  sentinel[uncertainty-reliability]: healthy
  sentinel[replay-fidelity]: NOT HEALTHY — PENDING (arrives with P3-003)
```

The P3 capability criterion is fully met; the phase ships when P3-003 implements
generative replay and its sentinel. Design note for the record: the curious arm's
mode flag is LIVE (monitor fed each round, curriculum consulted per chunk) — the
gate exercises the real ADR-0007 mechanism, not a hardcoded sign. Comparability
note: coverage is scored via the scale-free model/persistence MSE ratio because
raw latent MSE is incomparable across two independently-trained encoders.
