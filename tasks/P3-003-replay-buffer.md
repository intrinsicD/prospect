# P3-003 — Episodic replay + generative replay + the replay-fidelity sentinel

- **Status:** done
- **Phase:** P3
- **Requirements:** R7 (groundwork; the retention gate itself is P7)
- **ADRs:** ADR-0006 (generative-replay anti-collapse), ADR-0004 (memory tiers),
  P0-011 (raw observations stay re-encodable)
- **Depends on:** P1-001
- **Phase gate:** `bench/gates.py::GATES["P3"]` — capability is already ok
  (P3-001/P3-002); this task implements the last unhealthy sentinel
  (`replay-fidelity`). If it goes healthy, **P3 ships**.

## Goal
`ReplayBuffer` satisfying `interfaces.EpisodicMemory`: real experience storage
(raw observations, P0-011) plus `generative_replay()` — rehearsal batches that are
anti-collapse **by construction** (ADR-0006): a fixed real-data fraction anchors
every batch; dreams always start from REAL stored states and roll the world model
forward at most `max_dream_depth` steps (dreamed states are never re-stored, so
dream-of-dreams is structurally impossible); dreamed steps are quality-gated by
epistemic uncertainty (rehearse only in-distribution dreams). The
`replay-fidelity` sentinel measures all of it from the run log.

## Non-goals
- No training ON dreams (P7-001 owns retention/consolidation; dreamed transitions
  live in latent space and are marked — the P7 trainer will consume them).
- No semantic store / router (P8-001).
- No new environment or model changes.

## Interface to satisfy
`prospect.interfaces.EpisodicMemory` — implement in `prospect/memory.py`
(replace the `ReplayBuffer` skeleton). Plus the composition-root hook promised in
P2-002: `Agent(memory=...)` buffers every `observe()`d transition.

## Approach (brief)
- Ring buffer (capacity, FIFO eviction). `sample(n)`: uniform with replacement;
  empty buffer raises.
- `generative_replay(n)`: `real_fraction` of the batch sampled real; the rest
  dreamed — start latents encoded from sampled REAL states (duck-typed
  `model.encode`, identity fallback), actions bootstrapped from stored real
  actions, next latents from `model.predict`, marked
  `Option("__dream__", metadata={"depth": k})`, depth ≤ `max_dream_depth`.
  Epistemic gate self-calibrates per call: threshold = multiplier × median
  depth-1 epistemic; gated-out dreams are replaced by extra real samples (the
  real fraction is a floor). Dreams carry `prediction` (the dreaming step).
- Sentinel eval (`bench/evals/p3_replay.py`): `check_p3` logs, per seed and per
  regeneration round, the measured real fraction, dream diversity (per-dim std of
  dreamed next-latents / real next-latents), max lineage depth, and
  buffer-growth-during-replay; `@sentinel_check("replay-fidelity")` reads the run
  back — healthy iff fraction ≥ floor, diversity ≥ floor and not shrinking across
  regenerations, depth ≤ cap, and zero dreams stored.

## Acceptance criteria
- [x] `ReplayBuffer` implements `EpisodicMemory`; conformance assertion holds.
- [x] Storage honest: FIFO eviction at capacity; dreams never stored (buffer
      length unchanged by `generative_replay` — unit-tested); raw observations
      in, raw out.
- [x] Rehearsal batch: real fraction is a floor (gated-out dreams topped up with
      real anchors); dreams marked `__dream__` with depth ≤ cap; latent-space
      dreams from the real model; the self-calibrating epistemic gate cuts
      off-distribution dreams (unit-tested with depth-one and bimodal stubs).
- [x] `Agent(memory=...)` buffers observed transitions (unit-tested).
- [x] **Sentinel `replay-fidelity` healthy**; **P3 composite PASSES**; `P3`
      appended to `bench/SHIPPED` in this commit.
- [x] `make test` green (69), `make lint` clean, `make typecheck` clean;
      `gate-all` green across all four shipped phases (~2m).

## Test plan
- Unit (tests/test_memory.py): add/sample/eviction; empty-sample error;
  real-fraction floor; dream marking + depth cap; no-storage-of-dreams;
  latent-space dream shapes with the real `FlatWorldModel`; gate behavior with a
  stub model whose epistemic explodes past depth 1; no-model error.
- Eval: `make gate PHASE=P3` — full composite; then `make gate-all` with P3.

## Docs-sync checklist
- [x] Status → `done`; the P3 PASS GateReport below; `bench/SHIPPED` += P3.
- [x] architecture.md memory.py note still accurate (episodic + generative
      replay + router — router remains P8-001, as the note's split implies).
- [x] Backlog: P3-003 done; **Phase 3 shipped**; P4-001 unblocked (start here);
      P7-001 and P8-001 unblocked.

## Gate result
`make gate PHASE=P3` — PASS record `bench/results/P3-20260704T055502Z.json`
(from the ratchet re-run; identical to the standalone run — deterministic):

```
[P3] PASS
  capability: ok — differential MET (P(violated>expected) min 0.93);
    curiosity MET (coverage ratio 0.26 vs 0.79 at equal budget)
  sentinel[representation-integrity]: healthy (min std 0.868, rank 2.18)
  sentinel[uncertainty-reliability]: healthy (worst-seed corr 0.79)
  sentinel[replay-fidelity]: healthy — across 5 regenerations x 3 seeds: real
    fraction min 0.50 (floor 0.3), dream diversity min 0.47 (floor 0.3),
    diversity shrink 0.86 (floor 0.5), lineage depth max 3 (cap 3),
    dreams stored: 0
```

Design notes for the record: dream-of-dreams is prevented *structurally*
(dreams are never stored; every dream roots at a real state), and the epistemic
gate self-calibrates per call (multiplier × median depth-1 epistemic) instead of
hardcoding a scale-dependent threshold. Dreamed transitions live in latent space
marked `Option("__dream__", {"depth": k})` and carry the dreaming `prediction`;
the P7 consolidation trainer is their consumer. `gate-all`: 4 shipped gates
green.
