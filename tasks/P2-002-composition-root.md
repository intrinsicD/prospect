# P2-002 — Composition root: the agent loop

- **Status:** done
- **Phase:** P2
- **Requirements:** R1 (the loop that makes the planner act in a world)
- **ADRs:** ADR-0001 (one latent hub), ADR-0005 (harness/core split)
- **Depends on:** P2-001
- **Phase gate:** `bench/gates.py::GATES["P2"]` (already shipped — this task must
  keep it green with identical metrics; the refactor is behavior-preserving)

## Goal
One place where the core components meet. Components are modular, but the *wiring*
(encode → plan → act → observe) had no home: each gate eval re-invented it, and
wiring is where architecture erodes. After this task: `prospect.agent.Agent` is the
act–observe loop, `bench.loop.run_episode` is the harness driver, and the P2 eval
consumes both instead of private wiring.

## Non-goals
- No new capability, no learning-loop changes — behavior-preserving refactor plus
  the new seam. The P2 gate must reproduce byte-identical metrics (deterministic).
- No Agent `Protocol` — it is a concrete composition root, not a seam to swap.
- Monitor/replay/retrieval integration — those phases plug in later; this task
  only documents where.

## Interface to satisfy
No new `Protocol`. `Agent` composes existing seams: an `encode` callable
(`Observation -> LatentState` — the Codec seam, so the P6 swap passes through one
place) and an `interfaces.Planner`. `bench.loop.run_episode` drives any object
with `act(Observation) -> Action` and `reset()`.

## Approach (brief)
- Core `src/prospect/agent.py`: `Agent(encode, planner)` with
  `act(obs)` (encode → plan), `observe(obs, action, next_obs, reward, ...)` →
  `Transition` carrying RAW modality data (P0-011: replay stays re-encodable),
  and `reset()` (forwards to the planner's warm-start reset when present).
  Docstring names the future plug-ins: monitor `update()` in `observe` (P3-001),
  replay `add()` (P3-003), retrieval-as-action (P8).
- Harness `bench/loop.py`: `run_episode(env, agent, steps, seed, collect=False)`
  → `(return, transitions)`; exactly the act/step/observe loop the P2 eval used,
  so behavior is preserved.
- Refactor `bench/evals/p2_planner.py` onto `Agent` + `run_episode` (policies
  wrapped in a tiny adapter); delete the private loop.

## Acceptance criteria
- [x] `Agent.act` = planner over encoded latent; `observe` builds a raw-modality
      `Transition` (option/prediction pass through); `reset` reaches the planner
      (unit-tested with stubs).
- [x] `run_episode` drives an `Agent` on a `bench.Environment`; `collect=True`
      returns raw-modality transitions; `Agent` satisfies the `Acting` seam
      (isinstance + typed assignment); seed-reproducible (unit-tested).
- [x] P2 eval uses `Agent` + `run_episode`; **`make gate PHASE=P2` reproduced the
      shipped metrics exactly** (planner −58.26 / baseline −63.41 / random −67.57
      medians — byte-identical, so the refactor is provably behavior-preserving).
- [x] `make test` green (52), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_agent.py): stub encode/planner — act routes through both;
  observe stores raw obs + passthrough fields; reset forwarded. `run_episode`
  on a countdown env: return summed, transitions collected, seeded reset.
- Regression: re-run gate P2; diff the capability metrics against the shipped
  report (must be identical — determinism makes this exact).

## Docs-sync checklist
- [x] Status → `done`; gate re-run result recorded below.
- [x] architecture.md components list gains the agent.py bullet.
- [x] CLAUDE.md repo map mentions agent.py ("don't re-invent wiring in evals").
- [x] Backlog updated (P2-002 done; Phase 3 next — P3-001 is the top unblocked item).

## Gate result
No new gate — P2 was already shipped; this task's obligation was to keep it green
through the refactor. `make gate PHASE=P2` through the new `Agent` + `run_episode`
wiring:

```
[P2] PASS
  capability: ok — median eval return over 5 shared episodes: planner -58.26 vs
    model-free ES baseline -63.41 vs random -67.57
  (sentinels healthy, unchanged)
```

Identical to the shipped report (`bench/results/P2-20260703T175413Z.json`) — the
determinism of the eval makes "behavior-preserving" checkable, not asserted.
No SHIPPED change (P2 already listed). Suite 52 passed; lint + typecheck clean.
