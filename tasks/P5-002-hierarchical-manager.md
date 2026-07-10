# P5-002 — Hierarchical manager + VoE termination + option-diversity sentinel

- **Status:** done
- **Phase:** P5
- **Requirements:** R2
- **ADRs:** ADR-0003 (manager plans over the jumpy model; options terminate on
  VoE), ADR-0006 (option-diversity sentinel; exploit-mode uncertainty penalty),
  ADR-0007 (the manager is an exploit-mode consumer)
- **Depends on:** P5-001
- **Phase gate:** `bench/gates.py::GATES["P5"]` — this task completes the
  composite: capability (two-level beats flat at equal compute) AND the
  `option-diversity` sentinel. If it passes, **P5 ships**.

## Goal
`HierarchicalManager` satisfying `interfaces.HierarchicalPlanner`: the high level
*plans* (not reacts) by searching option sequences over the learned
`JumpyOptionModel` — cumulative discounted reward with per-jump duration-aware
discounting, minus the exploit-mode epistemic penalty — and emits the first
option; the worker executes it; `should_terminate` cuts it early when one-step
VoE spikes (the re-planning interrupt, the one signal's job #4). Gate: at EQUAL
per-step planning compute, two-level planning beats flat CEM on the long-horizon
balance task.

## Non-goals
- No goal-conditioned subgoals (options are selected on reward; `Subgoal`
  emission can arrive when a consumer needs it).
- No more than two levels (ADR-0003: generalize past two only if a gate demands).
- No skill learning; the P4 constant-torque options are the option set.

## Interface to satisfy
`interfaces.HierarchicalPlanner` — implement in `prospect/planning.py` (replace
the skeleton). `plan_option(state)` and `should_terminate(transition)`
(transition in latent space with the act-time `prediction`, as the Agent's
monitor hook already produces).

## Approach (brief)
- Manager: exhaustive search over option sequences (K^depth is small for real
  skill libraries at shallow depth; 3^3=27 here), rolled through
  `predict_option`; score = Σ γ^(elapsed duration)·reward − λ·Σ epistemic;
  return the first option of the argmax sequence. `should_terminate`: one-step
  surprise `-prediction.log_prob(observed)` above a threshold the harness
  ACI-calibrates to α=.01 on nominal option-execution surprise (U-003).
- Compute accounting (the criterion's "equal compute"): planning cost counted in
  ensemble member-forward passes per environment step, VoE monitoring charged to
  the hierarchy. The flat arm's CEM parameters are derived at runtime to match
  the hierarchy's measured per-step budget; a full-compute flat reference (P2ish
  config) is reported for context, not gated. Data budgets are reported: the
  jumpy model's option executions are the hierarchy's one-time abstraction cost
  (ADR-0003 pays data to make per-decision planning cheap).
- Option-diversity sentinel (from the run log, per seed, over the hierarchy's
  eval episodes): normalized option-usage entropy ≥ floor, mean executed option
  duration > 1 primitive step, min pairwise landing d′ (mean separation over
  pooled std of REAL executed landings) ≥ floor.

## Acceptance criteria
- [x] Implements `HierarchicalPlanner`; conformance assertions added.
- [x] Deep lookahead is real: depth-1 is myopic, depth-2 picks the zero-reward
      "detour" that unlocks the high-reward room — planning, not reaction
      (unit-tested on a stub option-model).
- [x] Exploit-mode penalty avoids the lucrative-but-uncertain option; VoE
      termination fires on violation only (near-predicted: no; far: yes;
      no expectation: no) (unit-tested).
- [x] **Gate P5 PASS:** two-level return −9.1/−4.2/−4.7 vs compute-matched flat
      −48.5/−34.9/−14.0 per seed (~729 member-forwards/step both) — wins on
      every seed; **and beats the full-compute flat reference
      (−36.6/−47.3/−19.8 at ~5x the budget) on every seed too**.
      `option-diversity` healthy (entropy ≥ 0.80, executed duration 2.6–2.9 of
      horizon 3, min pairwise landing d′ ≥ 2.1); all other sentinels healthy on
      run `p5`. `P5` in `bench/SHIPPED`.
- [x] `make test` green (83), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_manager.py): greedy pick, detour-over-myopic at depth 2,
  epistemic avoidance, termination threshold both ways, protocol conformance.
- Eval: `make gate PHASE=P5`; then `make gate-all` (P0–P5).

## Docs-sync checklist
- [x] Status → `done`; the P5 PASS GateReport below; `bench/SHIPPED` += P5.
- [x] architecture.md planning.py + hierarchy notes still accurate (two levels,
      jumpy model, VoE termination — all literal now).
- [x] Backlog: P5-002 done; **Phase 5 shipped**; P6-001 unblocked.
- [x] Dropped the two stray duplicate P3 reports from the P5-001 commit.

## Gate result
`make gate PHASE=P5` — PASS record `bench/results/P5-20260704T073845Z.json`:

```
[P5] PASS
  capability: ok — two-level return per seed [-9.1, -4.2, -4.7] vs
    compute-matched flat [-48.5, -34.9, -14.0] (~729 member-forwards/step) —
    WINS on every seed; jumpy landing MSE beats composed flat rollout on every
    seed: YES
  sentinels: representation-integrity, uncertainty-reliability,
    replay-fidelity, option-diversity — ALL healthy
```

Bonus finding, stronger than the criterion: the hierarchy also beats the
FULL-compute flat reference (−36.6/−47.3/−19.8 at ~3840 member-forwards/step,
5x the budget) on every seed — temporal abstraction wins outright here, not
just at parity. `gate-all`: 6 shipped gates green (~3m30s).

**What it took (historical, before U-003; two diagnosed issues):** (1) the VoE termination threshold was
first calibrated on random-walk held-out surprise — the wrong reference
distribution for controlled trajectories — and cut healthy options at ~2 of 5
steps; recalibrating on option-execution stepwise surprise (then q99, now
superseded by U-003's α=.01 ACI policy) restored real
temporal abstraction (durations 2.6–2.9 of horizon 3, terminations present but
not dominant). (2) One seed's easy episodes were lost not on planning but on the
option set's control resolution — bang-bang ±2 cannot hold a target the way
continuous flat actions can; adding fine ±0.5 skills raised the hierarchy's
ceiling more than the recomputed budget helped the flat arm. The option set is
the hierarchy's action-space design surface, and it matters.
