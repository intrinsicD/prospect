# P9-004 — Metamorphic invariants, negative controls, statistics hardening

- **Status:** blocked (P9-001)
- **Phase:** P9
- **Requirements:** R1–R8 (guard the claims against gate-overfit and noise)
- **ADRs:** ADR-0006 (integrity is enforced, not hoped), ADR-0008
- **Depends on:** the existing per-phase gates; runs alongside P9-001
- **Phase gate:** hardens every gate; adds a `gate-overfit` negative-control check
  and metamorphic-invariant assertions.

## Goal
De-risk the two ways a calibrated benchmark suite lies: **gate-overfit** (a trivial
solution quietly passes) and **noise** (a margin within seed variance reads as a
pass). Add invariant checks that need no golden threshold, negative controls that a
degenerate solution must *fail*, and tighter statistics on the cheap gates.

## Non-goals
- Not new capabilities; this is measurement hardening.
- Not a rewrite of existing gates — additive assertions + a few more seeds.

## Interface to satisfy
`bench/evals/p9_invariants.py` (+ small additions to existing eval bodies): metamorphic
assertions, per-gate negative controls, and effect-size/CI reporting. Registered as a
`gate-overfit` sentinel and/or invariant checks. No new core `Protocol`.

## Approach (brief)
- **Metamorphic invariants** (no golden threshold): epistemic uncertainty is higher
  OOD than in-distribution; more data does not *raise* epistemic on seen regions; a
  mastered skill does not un-master without a distribution shift; a poisoned untrusted
  source never lowers accuracy below no-retrieval.
- **Negative controls / gate-overfit audit:** for each capability gate, name the
  trivial/degenerate solution (constant predictor, always-retrieve, one-step options,
  point-estimate) and assert it **fails** the gate — so passing means the capability,
  not the artifact.
- **Statistics hardening:** raise seed counts on the cheap gates; report effect size +
  a bootstrap CI; flag any gate whose margin sits within noise.

## Acceptance criteria
- [ ] The metamorphic invariants hold across the relevant gates (unit + eval).
- [ ] Each capability gate has a negative control its trivial solution fails.
- [ ] Cheap gates report effect size + CI; no shipped gate passes within noise.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit: each invariant + each negative control as a focused assertion.
- Eval: the `gate-overfit` sentinel runs with the phase gates.

## Docs-sync checklist
- [ ] Status → `done`; invariant/negative-control results recorded below.
- [ ] Register the `gate-overfit` sentinel in the sentinel table + smoke test.
- [ ] ADR-0006 consequences note the negative-control discipline; ADR-0008 updated.
- [ ] Backlog: P9-004 done.

## Gate result
_not run yet_
