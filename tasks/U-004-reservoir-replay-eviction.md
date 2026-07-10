# U-004 — Hybrid FIFO + reservoir replay eviction

- **Status:** done
- **Phase:** U (upgrade track; re-gates against P3/P7)
- **Requirements:** R7
- **ADRs:** ADR-0006 (generative-replay anti-collapse: real data must *accumulate*)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P7"]` (retention criterion) — must hold or
  improve; `["P3"]` replay-fidelity sentinel stays healthy
- **Source:** `docs/sota-review-2026-07.md` U-004 · [WMAR](https://arxiv.org/abs/2401.16650)
  · [accumulate-don't-replace](https://arxiv.org/abs/2404.01413)

## Goal
Fix the one place the memory design contradicted its own anti-forgetting purpose:
at task start, `ReplayBuffer` used FIFO-only eviction, so everything older than
`capacity` was gone exactly when rehearsal needed it. Add a reservoir half
(uniform-over-lifetime retention) alongside the FIFO (recency) half — the
continual-learning standard, validated inside a world-model agent (WMAR).

## Non-goals
- **Sampling stays uniform** — the review found uniform ≈ prioritized on stationary
  tasks even in the Curious Replay paper; this changes *eviction*, not sampling
  (prioritized sampling is the separate deferred U-103).
- No capacity increase for its own sake; the split is within the existing budget.
- No per-transition metadata beyond an insertion counter (reservoir needs only that).

## Interface to satisfy
`memory.ReplayBuffer`: the former ring buffer becomes two segments —
`fifo_capacity` (recent FIFO) and `reservoir_capacity` (lifetime reservoir
sampling: item k kept with probability `reservoir_capacity/k`). `add`, `sample`,
`__len__`, `generative_replay` unchanged in signature; `EpisodicMemory` protocol
unchanged. Constructor: `capacity` splits into `fifo_capacity` + `reservoir_capacity`
(default e.g. 60/40 of the old 50k).

## Approach (brief)
- Reservoir: standard Vitter algorithm-R (a few lines) — after the reservoir fills,
  the n-th arrival replaces a uniform-random slot with probability `reservoir_cap/n`.
- Segments are disjoint: a transition stays in the recent FIFO until it ages out,
  then becomes an Algorithm-R candidate. This preserves `len()`/capacity semantics
  and avoids double-weighting recent entries while retaining a uniform sample of
  the older lifetime history.
- FIFO segment keeps recency (needed for on-policy freshness and the MAD "fresh-data
  loop" regime); reservoir segment keeps lifetime coverage (needed for rehearsal to
  reach old skills). `sample`/`generative_replay` draw from the union.
- This tightens the ADR-0006 "real data accumulates" prescription that FIFO undercut.

## Acceptance criteria
- [x] Two-segment buffer; unit test shows lifetime coverage — after 10× capacity
      insertions, the reservoir still contains early-lifetime transitions (FIFO-only
      would have evicted them).
- [x] **P7 retention criterion holds or improves**; P3 `replay-fidelity` sentinel
      healthy (real-anchor fraction, dream diversity, lineage cap unchanged).
- [x] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_memory.py): reservoir retains a marked early transition after heavy
  churn; segment sizes respected; `generative_replay` real-fraction floor still met.
- Eval: `make gate PHASE=P7`, `make gate PHASE=P3`, `make gate-all`.

## Docs-sync checklist
- [x] Status → done; retention before/after recorded below.
- [x] ADR-0006: note reservoir eviction realizes the "real data accumulates" clause.
- [x] architecture.md memory bullet: episodic buffer is FIFO+reservoir.
- [x] `docs/sota-review-2026-07.md`: mark U-004 shipped.

## Gate result

The implementation uses a 60/40 recent/history split within the existing capacity.
FIFO evictees are the reservoir stream, so entries are disjoint and Algorithm R is
uniform over the aged-out lifetime history. A fixed-seed unit test fills a capacity-10
buffer with 100 transitions: FIFO contains exactly `94..99`, while the four-slot
reservoir still contains marked transition `7`; uniform `sample()` draws exercise
both segments, and same-seed reservoir contents reproduce exactly.

**P7 PASS** — final ratchet report `bench/results/P7-20260710T165100Z.json`.
The before report is `P7-20260710T160637Z.json`; retention is unchanged on every
seed (`[0.09799, 0.09386, 0.07057]`, mean `0.08747`) and plasticity is unchanged
(`[1.4368, 0.8709, 2.9743]`, mean `1.7607`). This is the honest expected result:
P7 inserts 6,144 transitions, below the default 30,000-transition FIFO window, so
the gate is a regression check while the churn unit test is the eviction certificate.
Its replay sentinel remains healthy: real fraction `0.50`, minimum dream diversity
`0.6444`, shrink `0.9441`, lineage depth `3`, and zero dreams stored.

**P3 PASS** — final ratchet report `bench/results/P3-20260710T164743Z.json`.
Replay fidelity remains healthy: real fraction `0.50`, minimum dream diversity
`0.4745`, shrink `0.8542`, lineage depth `3`, and zero dreams stored. `make test`:
145 passed, 1 skipped; Ruff and mypy clean. Final `make gate-all`: **P0–P14 PASS**
(`ratchet ok — 15 shipped gate(s) still green`). No deferred U-101–U-112 trigger
fired; in particular, U-103 remains deferred because no gate identifies uniform
sampling as the limiter.
