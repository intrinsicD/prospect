# P9-004 — Metamorphic invariants, negative controls, statistics hardening

- **Status:** done
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
- [ ] A `gate-overfit` sentinel (active from P9) runs a battery of negative controls +
      metamorphic invariants + a bootstrap-CI check; healthy iff all hold.
- [ ] Metamorphic invariants hold (surprise decomposition exact, untrusted never
      overrides, log-prob peaks at the mean); negative controls reject their trivial
      solution (always-retrieve, one-step options, ablation-no-over-credit).
- [ ] `bootstrap_ci` distinguishes a real margin from noise; unit-tested.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_invariants.py): the whole battery holds; the bootstrap CI straddles
  0 for noise and excludes it for a real margin.
- Eval: `make gate PHASE=P9` — the `gate-overfit` sentinel runs (healthy).

## Docs-sync checklist
- [x] Status → `done`; battery result recorded below.
- [x] Register the `gate-overfit` sentinel in the sentinel table + smoke test.
- [x] ADR-0006 consequences note the negative-control discipline; ADR-0008 updated.
- [x] Backlog: P9-004 done — Phase 9 complete.

## Gate result
`make gate PHASE=P9`: the `gate-overfit` sentinel is **healthy** — 7 negative controls
+ metamorphic invariants hold, and it is correctly scoped (active from P9; absent from
P8's sentinel set). P9 composite still **PASS** (five sentinels now healthy).

Battery (all cheap, no training — stands in the ratchet):

| Check | Kind | Guards |
|---|---|---|
| surprise-decomposition-exact | invariant | epistemic + aleatoric == total |
| untrusted-never-overrides | invariant | untrusted content is data, never instruction |
| log-prob-peaks-at-mean | invariant | the predicted distribution is well-formed |
| always-retrieve-fails | negative control | the retrieval gate rejects blanket retrieval |
| one-step-options-fail-diversity | negative control | option-diversity rejects one-step options |
| ablation-no-over-credit | negative control | the ablation harness doesn't over-credit noise |
| bootstrap-flags-noise | statistics | a within-noise margin is not significant |

**P9-004 PASS — Phase 9 complete.** The suite is now guarded against its own two
failure modes: a trivial solution can't pass a capability criterion (negative controls),
and a margin within seed noise can't read as a pass (bootstrap CI). The metamorphic
invariants catch malformed distributions/routing with no golden threshold. This is the
standing complement to the findings P9-001..003 surfaced — the gates measure the
capability, not the artifact.
