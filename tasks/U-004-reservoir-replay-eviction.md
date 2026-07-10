# U-004 — Hybrid FIFO + reservoir replay eviction

- **Status:** ready
- **Phase:** U (upgrade track; re-gates against P3/P7)
- **Requirements:** R7
- **ADRs:** ADR-0006 (generative-replay anti-collapse: real data must *accumulate*)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P7"]` (retention criterion) — must hold or
  improve; `["P3"]` replay-fidelity sentinel stays healthy
- **Source:** `docs/sota-review-2026-07.md` U-004 · [WMAR](https://arxiv.org/abs/2401.16650)
  · [accumulate-don't-replace](https://arxiv.org/abs/2404.01413)

## Goal
Fix the one place the memory design contradicts its own anti-forgetting purpose:
`ReplayBuffer` uses FIFO eviction (memory.py:66-71), so everything older than `capacity`
is gone exactly when rehearsal needs it. Add a reservoir half (uniform-over-lifetime
retention) alongside the FIFO (recency) half — the continual-learning standard,
validated inside a world-model agent (WMAR).

## Non-goals
- **Sampling stays uniform** — the review found uniform ≈ prioritized on stationary
  tasks even in the Curious Replay paper; this changes *eviction*, not sampling
  (prioritized sampling is the separate deferred U-103).
- No capacity increase for its own sake; the split is within the existing budget.
- No per-transition metadata beyond an insertion counter (reservoir needs only that).

## Interface to satisfy
`memory.ReplayBuffer` (memory.py:16-132): the ring buffer becomes two segments —
`fifo_capacity` (recent, FIFO as today) and `reservoir_capacity` (lifetime, reservoir
sampling: item k kept with probability `reservoir_capacity/k`). `add`, `sample`,
`__len__`, `generative_replay` unchanged in signature; `EpisodicMemory` protocol
unchanged. Constructor: `capacity` splits into `fifo_capacity` + `reservoir_capacity`
(default e.g. 60/40 of the old 50k).

## Approach (brief)
- Reservoir: standard Vitter algorithm-R (a few lines) — after the reservoir fills,
  the n-th arrival replaces a uniform-random slot with probability `reservoir_cap/n`.
- FIFO segment keeps recency (needed for on-policy freshness and the MAD "fresh-data
  loop" regime); reservoir segment keeps lifetime coverage (needed for rehearsal to
  reach old skills). `sample`/`generative_replay` draw from the union.
- This tightens the ADR-0006 "real data accumulates" prescription that FIFO undercut.

## Acceptance criteria
- [ ] Two-segment buffer; unit test shows lifetime coverage — after 10× capacity
      insertions, the reservoir still contains early-lifetime transitions (FIFO-only
      would have evicted them).
- [ ] **P7 retention criterion holds or improves**; P3 `replay-fidelity` sentinel
      healthy (real-anchor fraction, dream diversity, lineage cap unchanged).
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_memory.py): reservoir retains a marked early transition after heavy
  churn; segment sizes respected; `generative_replay` real-fraction floor still met.
- Eval: `make gate PHASE=P7`, `make gate PHASE=P3`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; retention before/after recorded below.
- [ ] ADR-0006: note reservoir eviction realizes the "real data accumulates" clause.
- [ ] architecture.md memory bullet: episodic buffer is FIFO+reservoir.
- [ ] `docs/sota-review-2026-07.md`: mark U-004 shipped.

## Gate result
<paste the GateResult once run>
