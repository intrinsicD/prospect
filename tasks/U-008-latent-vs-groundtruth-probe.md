# U-008 — Gate probe: latent-space epistemic vs ground-truth state-space error

- **Status:** ready
- **Phase:** U (upgrade track; new standing probe from P1)
- **Requirements:** R1, R3, R4
- **ADRs:** ADR-0006 (uncertainty integrity — measured, not assumed), ADR-0008
- **Depends on:** none
- **Phase gate:** folds into `bench/gates.py::GATES["P1"]`/`["P9"]` uncertainty-reliability
  sentinel as a new check
- **Source:** `docs/sota-review-2026-07.md` U-008 · [Biased Dreams](https://arxiv.org/abs/2604.25416)

## Goal
A 2026 result shows latent-space ensemble disagreement develops *attractors* toward
well-represented regions and can systematically overstate rollout quality — established
physical-space UQ results do not transfer to latent space automatically. Add a standing
gate probe that correlates latent-space epistemic against ground-truth *state-space*
prediction error on the toy envs (uniquely cheap here, where ground truth is available).

## Non-goals
- Not a new uncertainty mechanism — a *measurement* that keeps the existing signal
  honest (the review's explicit framing: nobody should trust latent disagreement without
  this probe).
- Not a GP calibration reference (that is the deferred bench-side option) — the direct
  ground-truth-error correlation only.

## Interface to satisfy
A new `@sentinel_check` (or an addition to the uncertainty-reliability sentinel) in
`bench/gates.py` + eval body in `bench/evals/`: on a held-out set, compute per-sample
latent-space epistemic and per-sample ground-truth state-space error (the toy env exposes
true next state), and require their rank correlation ≥ a floor. Reads the run-metrics
artifact (P0-005) like the other sentinels; zero-arg `check()`.

## Approach (brief)
- The toy envs expose the true next state, so state-space error is directly available —
  the cheap check the paper says latent models generally can't run.
- Correlate against latent epistemic (post-U-007 if landed); a low correlation is the
  attractor failure the paper warns of, surfaced as an unhealthy sentinel rather than a
  hidden bias.
- Standing check: active from P1 (where the latent + ensemble first exist), re-run by
  `gate-all`.

## Acceptance criteria
- [ ] New sentinel/check computes latent-epistemic vs ground-truth-error rank correlation
      on a held-out set; floor documented in `bench/gates.py` as data.
- [ ] Current shipped models PASS the floor (calibration recorded); a deliberately
      collapsed-epistemic control FAILS it (negative control, ADR-0008 discipline).
- [ ] `make gate-all` green; `make test`/`lint`/`typecheck` clean.

## Test plan
- Unit (tests/test_gates.py): the check passes on a healthy synthetic stream, fails on
  one where epistemic is decorrelated from error.
- Eval: `make gate PHASE=P1`, `PHASE=P9`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; measured correlation recorded below.
- [ ] ADR-0006: add the probe to the uncertainty-integrity sentinel list; note the
      latent-attractor risk it guards.
- [ ] roadmap.md sentinel note updated (new standing check).
- [ ] `docs/sota-review-2026-07.md`: mark U-008 shipped.

## Gate result
<paste the GateResult once run>
