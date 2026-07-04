# P9-003 — Second environment + cross-environment generalization

- **Status:** done
- **Phase:** P9
- **Requirements:** R1, R4, R8 (do the capabilities generalize, or are they
  Pendulum-shaped?)
- **ADRs:** ADR-0005 (a capability is only real if it survives a second, structurally
  different task with the same code), ADR-0008
- **Depends on:** P0-004 (the `Environment` protocol), P1/P2/P8 (the load-bearing
  gates re-run), P9-001 (the E2E harness, re-run on env #2)
- **Phase gate:** contributes to `bench/gates.py::GATES["P9"]` — the
  "capabilities survive on a second environment" criterion.

## Goal
Add one structurally different environment to `bench/envs.py` and re-run the
load-bearing gates (P1 prediction, P2 planning, P8 retrieval) and the P9 E2E loop on
it with the **same core code** — only recalibrated thresholds. A capability that
survives is real; one that collapses was a Pendulum artifact. This is the direct
antidote to "everything is measured on one toy environment."

## Non-goals
- Not a realistic/hard benchmark — a *second, different* toy is enough to break
  single-environment overfit (e.g. a discrete gridworld or a different-dynamics
  continuous task). Minimal.
- No changes to `src/prospect/` core to make a gate pass on env #2; if a capability
  needs core changes to generalize, that is a **finding** → its own task.
- Not a full port of every gate: prediction (P1) and planning (P2) are the gated
  generalization; retrieval (P8) is measured and reported. Re-running the *full
  composed* P9 loop on env #2 is out of scope — retrieval hurts control (P9-002), so
  the composed-with-retrieval agent is not the interesting object on a new env.

## Interface to satisfy
A new `Environment` (P0-004 protocol) in `bench/envs.py` — `PointMass`, a 2D nonlinear
(quadratic-drag) point mass: obs_dim=4, action_dim=2, no rotational/trig structure. And
`bench/evals/p9_generalization.py::generalizes()` re-running the capabilities on it with
the SAME core (constructed with the env's dimensions). No new core `Protocol`; the core
never learns which env it is in.

## Approach (brief)
- `PointMass`: quadratic drag (negligible at low speed, dominant at high speed) gives a
  seen/OOD velocity split; no spring, so the agent must act to reach the origin.
- `generalizes()`: on env #2, **prediction** (WM beats persistence, P1), **planning**
  (CEM beats random, P2), **retrieval** (gated beats no-retrieval, P8) — median over
  seeds (P2-style, robust to a lucky random start). Recalibrated eval params only
  (more training data + a shorter planning horizon for a 4-dim env — no core change).
- Fold into the P9 gate: gate on prediction + planning generalizing; record retrieval's
  result as a finding (its benefit is env-dependent).

## Acceptance criteria
- [ ] `PointMass` conforms to `Environment` + `set_state`; unit-tested (incl. the
      nonlinear drag).
- [ ] Prediction (P1) and planning (P2) generalize to env #2 with the same core
      (recalibrated eval params only); folded into the P9 gate; **P9 still PASS**.
- [ ] Retrieval's generalization is measured and, if it fails, filed as a finding —
      not patched away by a core change.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (`tests/test_environment.py`): the second env's reset/step/(set_state) behave.
- Eval: `make gate PHASE=P9` includes the env-#2 re-runs.

## Docs-sync checklist
- [x] Status → `done`; per-capability env-#2 results recorded below.
- [x] architecture.md notes the two-environment validation; ADR-0008 updated.
- [x] Backlog: P9-003 done.

## Gate result
`make gate PHASE=P9` (~6m50s; folds cross-env into the P9 gate):

```
[P9] PASS
  capability: ok — ... cross-env: prediction ✓ + planning ✓ generalize to a 2nd env
    (PointMass); retrieval ✗ (env-dependent). FINDINGS: retrieval hurts control
    (marginal -9.5) and does not generalize to PointMass; exploit-penalty negligible
  sentinel[*]: all four healthy
```

Per-capability on `PointMass` (medians over 2 seeds, same core, recalibrated eval
params only):

| Capability | env #2 result | Verdict |
|---|---|---|
| prediction (P1) | WM MSE **0.0011** vs persistence **0.0295** | generalizes (27x better) |
| planning (P2) | planner **−15.5** vs random **−52.8** | generalizes |
| retrieval (P8) | gated **0.0183** vs no-retrieval **0.0172** | **does NOT generalize** (finding) |

**P9-003 PASS — the core capabilities are real, not Pendulum artifacts.** Prediction
and planning both survive on a structurally different environment (2D nonlinear-drag
point mass; obs_dim 3→4, action_dim 1→2) with the **same core code** — only eval params
recalibrated (a WM built with the env's dimensions, `TRAIN_N`/`GEN_STEPS` raised for a
4-dim task, planning horizon 20→6). No `src/prospect/` change was needed — that no core
change is required IS the generalization result.

**Finding: retrieval does not generalize to PointMass.** Gated ≈ no-retrieval, because
the region-trained ensemble's epistemic barely rises out-of-region there (retrieval
fired ~0–9% vs 55% on Pendulum) — it is *confidently wrong* OOD (the ADR-0002
limitation), so the uncertainty gate never triggers. Retrieval's benefit is therefore
env-dependent: it rides on the uncertainty signal being OOD-sensitive, which is itself
env-dependent. Recorded, not patched — a real generalization limit for future work
(a better OOD-detection signal, or a per-env uncertainty calibration).

**Calibration note (honest):** planning failed at first (planner −455 vs random −88) —
model exploitation: an under-trained WM's long (horizon-20) rollouts diverge and CEM
optimises the fantasy. Raising training data/steps and shortening the horizon to 6
fixed it. The failure and its fix are the generalization work; median criteria (P2-style)
make the pass robust to a lucky random start.
