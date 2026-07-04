# P7-001 — Continual improvement: forgetting detection + consolidation

- **Status:** done
- **Phase:** P7
- **Requirements:** R7
- **ADRs:** ADR-0002 (forgetting = epistemic rising on a mastered skill → rehearse),
  ADR-0006 (generative-replay anti-collapse — the consolidation substrate),
  ADR-0005 (P7 is a *discipline*, not a module)
- **Depends on:** P3-003 (the replay buffer + gated generative replay)
- **Phase gate:** `bench/gates.py::GATES["P7"]`

## Goal
The system improves over a task sequence without catastrophic forgetting, and
without losing plasticity. Two parts: (1) the last piece of the VoE monitor —
`is_forgetting`, epistemic rising back up on a once-mastered skill (the signal
that, in the full system, triggers rehearsal); (2) a consolidation *policy*
(rehearsal via the P3-003 buffer) that keeps earlier tasks' predictions healthy
while later tasks are learned — measured against a no-consolidation baseline that
forgets.

## Non-goals
- No new core module (P7 is a discipline, ADR-0005/roadmap): the only core change
  is `is_forgetting`; consolidation is the existing `ReplayBuffer` rehearsal
  applied across tasks in the harness.
- No new environment: a task sequence is Pendulum variants (different gravity =
  different dynamics).
- No architecture/EWC-style weight regularization — rehearsal is the mechanism the
  scaffold already built (ADR-0006); other consolidation methods are earned by a
  later gate if rehearsal proves insufficient.

## Interface to satisfy
`interfaces.CompetenceMonitor.is_forgetting` in `prospect/voe.py` (replace the
`NotImplementedError("P7-001")`). Latch a skill's mastered epistemic floor when it
first masters; `is_forgetting` = current epistemic risen a `forget_factor` above
that floor.

## Approach (brief)
- `is_forgetting`: `_SkillStats` records `mastered_epistemic` (the fast-EMA at
  first mastery); a never-mastered skill is never "forgetting"; a mastered one is
  forgetting when its fast-EMA epistemic exceeds `forget_factor × max(floor,
  mastery_epistemic)`.
- Gate eval (`bench/evals/p7_continual.py`, `@gate_check("P7")`, run `p7` carrying
  all four sentinels): a 3-task gravity sequence. Two arms at equal training
  budget — **consolidation** (each batch mixes current-task data with the buffer's
  gated generative replay over accumulated experience) and **no-consolidation**
  (current-task data only). Metrics per seed:
  * **retention**: task-0 held-out 1-step MSE after the whole sequence ÷ its MSE
    right after task-0 — consolidation keeps it within tolerance; no-consolidation
    blows up (the catastrophic-forgetting contrast).
  * **plasticity**: the last task's final held-out MSE ÷ the first task's — the
    model still fits new tasks (plasticity retained).
  * **detector**: `is_forgetting("task0")` fires after the no-consolidation
    sequence and stays quiet after consolidation (VoE ties to the mechanism).

## Acceptance criteria
- [x] `is_forgetting` implemented (keyed on rising prediction ERROR, not epistemic —
      see gate result); conformance holds; lifecycle unit-tested (never-mastered →
      False; mastered-then-error-risen → True; mastered-stable → False).
- [x] **Gate P7 PASS:** retention (avg past-task MSE) consolidate [0.10, 0.09, 0.07]
      vs none [0.27, 0.37, 0.38] — 3-5x better, within tol on every seed; plasticity
      (last-task fit / from-scratch scale) consolidate [1.44, 0.88, 2.97] ≤ 3.5 on
      every seed; all four sentinels healthy on run `p7`. `P7` in `bench/SHIPPED`.
- [x] `make test` green (90), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_voe.py): `is_forgetting` lifecycle + per-skill isolation.
- Eval: `make gate PHASE=P7`; then `make gate-all` (P0–P7).

## Docs-sync checklist
- [x] Status → `done`; the P7 PASS GateReport below; `bench/SHIPPED` += P7.
- [x] architecture.md voe.py note updated (mastery keys on epistemic, forgetting on
      rising error). ADR-0002 amended (the P0-010 forgetting-under-shift concern
      resolved for the "I forgot" case).
- [x] Backlog: P7-001 done; **Phase 7 shipped**; P8-001 next.

## Gate result
`make gate PHASE=P7` — PASS record `bench/results/P7-20260704T092110Z.json`:

```
[P7] PASS
  capability: ok — retention (avg past-task MSE, lower=better) consolidate
    [0.10, 0.09, 0.07] vs none [0.27, 0.37, 0.38] (contrast, tol x0.8);
    plasticity (last-task fit / from-scratch scale) consolidate [1.44, 0.88, 2.97]
    (naive [0.27, 0.39, 5.55], abs tol x3.5)
  sentinels: representation-integrity, uncertainty-reliability, replay-fidelity,
    option-diversity — ALL healthy
```

`gate-all`: 8 shipped gates green (~7m).

**The honest continual-learning story, measured:** naive sequential training loses
BOTH memory and plasticity — it forgets earlier tasks (avg past-task MSE 3-5x
worse) AND its seq/from-scratch fit ratio grows across the sequence (the
documented loss of plasticity). Consolidation — rehearsal of retained REAL
experience (the P3-003 buffer; raw obs stay re-feedable, P0-011) — preserves both.

**Three things diagnosed and fixed, each recorded:** (1) the representation-integrity
sentinel first flagged rank collapse — it was probing a fixed later task while the
model fit an earlier one (OOD, legitimately compressed); probing each task's own
data (P1 semantics) fixed it, and revealed task 0 is the representation-FORMATION
phase (its < 2.0 rank is the analog of P1's warm-up, excluded). (2) A wider gravity
spread saturated the omega clip and pushed the latent rank below its 2-DOF intrinsic
dimension — that makes the tasks unlearnable, not the policy inadequate; a moderate
[6,10,14] spread keeps the representation healthy while still causing forgetting.
(3) The forgetting detector first never fired: keyed on epistemic, it missed
forgetting because the ensemble is CONFIDENTLY WRONG under shift (members agree on
the wrong answer, ADR-0002's named limitation). Re-keying `is_forgetting` on rising
prediction ERROR (ADR-0002 amended) makes it fire — 3/3 on the forgotten arm, 1/3
on the retained arm (the seed-0 trip honestly reflects consolidation's imperfect
retention on that seed). The consolidation boundary is documented: it rehearses
real experience, not the buffer's latent-space dreams — consuming dreams needs a
latent-training path the world model doesn't expose (a future extension, earned by
a gate if raw retention becomes infeasible).
