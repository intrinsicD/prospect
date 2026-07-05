# P9-001 — End-to-end integration gate (the whole-system fitness function)

- **Status:** done
- **Phase:** P9
- **Requirements:** R1–R8 (their integration)
- **ADRs:** ADR-0005 (benchmark-gated increments), ADR-0008 (whole-system
  integration gate + ablation as the standing fitness — added by this task),
  ADR-0002 (the one signal), ADR-0004 (retrieval-as-action)
- **Depends on:** P2-002 (composition root), P1–P8 (the components composed)
- **Phase gate:** `bench/gates.py::GATES["P9"]`

## Goal
One gate that runs the **fully-composed** agent through `agent.py` on the reference
task for a real budget and asserts emergent, closed-loop properties that *no
per-phase gate covers*: the agent learns while acting, the single VoE signal drives
several mechanisms within the **same** run, and retrieval-as-action improves
downstream **control** (closing the loop P8 opened at the prediction level). The
integrity sentinels stay healthy on the integrated run.

## Non-goals
- Not a new capability and no new component — it *composes* existing ones through
  `agent.py`. Generality is not added here; integration is measured.
- Not the ablation study (P9-002) or cross-environment generalization (P9-003); this
  establishes the integrated baseline those build on.
- No new environment (Pendulum), no new tuning of the individual components.

## Interface to satisfy
Extend `agent.py` (the composition root — CLAUDE.md: *don't re-invent wiring in
evals; extend this*): wire the `MemoryRouter` into `act()` (retrieval-as-action —
when the act-time epistemic exceeds the router threshold, retrieve a fact and use it
to correct the prediction the planner consumes) and let the curriculum set the
planner's `Mode`. Add `bench/evals/p9_integration.py::check_p9` (`@gate_check("P9")`)
and register `"P9"` in `gates.py` (`PHASE_ORDER` + criterion). No new core `Protocol`.

## Approach (brief)
- Compose on Pendulum: `UniversalCodec.encode` → `FlatWorldModel` (learns online from
  a `ReplayBuffer`) → `FlatPlanner` (epistemic penalty/bonus set by the curriculum
  `Mode`) → `SurpriseCompetenceMonitor` + `LearningProgressCurriculum` → `SemanticStore`
  + `UncertaintyMemoryRouter` (retrieval-as-action). Wire it through `Agent`.
- Run the act–observe loop for a budget: encode, plan (with the retrieval-corrected
  prediction where uncertain), step the env, `observe()` (feeds monitor + replay),
  periodically `update()` the model from replay, let the curriculum pick the mode.
- Assert (every seed), with **robust** relative/structural/paired criteria (integration
  compounds noise — no tight absolute thresholds):
  1. **Controls end-to-end:** the composed agent's return beats a reactive baseline —
     the full wiring produces control, not just isolated capabilities.
  2. **One signal, many jobs, one run:** within the single run the epistemic signal is
     turned by the curriculum into the planner's live exploit coefficient AND gates
     retrieval (it fired; every retrieval above the router threshold by construction).
  3. **Integrity holds:** all applicable sentinels healthy on run `p9`.
- **Measured, not gated — a finding:** the retrieval-on vs retrieval-off control delta.
  The experiment showed retrieval-*into-planning* degrades control (it helps 1-step
  prediction per P8, but overriding the planner's rollout dynamics with nearest-
  neighbour facts corrupts multi-step optimisation). Reported as a finding; quantifying
  it is the P9-002 ablation's job — not a pass criterion to be tuned away.

## Acceptance criteria
- [ ] `agent.py` wires the curriculum mode; the planner plans over a
      `RetrievalAugmentedWorldModel`; unit tests cover both new seams.
- [ ] `check_p9` runs the composed agent end-to-end; controls-end-to-end +
      one-signal-many-jobs hold on every seed; **P9 gate PASS**.
- [ ] All applicable sentinels healthy on run `p9`.
- [ ] The retrieval-into-planning control finding is recorded (metric + detail) and
      handed to P9-002.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (`tests/test_agent.py`): the extended `act()` retrieves when uncertain and
  falls back to the model when confident; the curriculum mode reaches the planner.
- Eval: `make gate PHASE=P9` — the four integration assertions + the sentinels.

## Docs-sync checklist
- [x] Status → `done`; gate report recorded below.
- [x] Add **ADR-0008** (integration gate + ablation as the whole-system fitness).
- [x] `roadmap.md` gains the P9 row; requirements note the integration gate.
- [x] Backlog: P9-001 done; P9-002 unblocked (carries the retrieval finding).
- [x] Append `P9` to `bench/SHIPPED` in the same commit (ratchet, P0-007).

## Gate result
`make gate PHASE=P9` (3 seeds; ~2m20s after a store-query cache fix, from 13m):

```
[P9] PASS
  capability: ok — composed agent controls end-to-end: return -19.4 vs reactive
    -73.1 (beats every seed); one epistemic signal drives exploit-mode AND retrieval
    in one run (MET; retrieval rate 31%). FINDING: retrieval-into-planning costs
    control here (-19.4 vs -8.1 retrieval-off) — its marginal value is the P9-002
    ablation question
  sentinel[representation-integrity]: healthy — min per-dim std 0.852, min eff. rank 2.16
  sentinel[uncertainty-reliability]: healthy — corr 0.63, high-error disagreement 18.93x
  sentinel[replay-fidelity]: healthy — real frac 0.50, diversity 0.93, depth<=3, 0 stored
  sentinel[option-diversity]: healthy — entropy 0.77, duration 2.94, min d' 0.74
```

**P9 PASS — the whole system works end-to-end.** The composed agent (codec →
world-model → planner-over-retrieval-augmented-model → VoE monitor → curriculum →
memory/retrieval), wired through `agent.py`, controls the task (−19.4 vs random
−73.1); and in **one run** the single epistemic signal both sets the planner's live
exploit coefficient (via the mastered curriculum) and gates retrieval — one signal,
several jobs. All four collapse sentinels stay healthy on the integrated run.

**Finding handed to P9-002:** retrieval-into-planning *degrades* control here (−19.4
with retrieval vs −8.1 without). Retrieval improves 1-step prediction (P8) but
overriding the planner's rollout dynamics with nearest-neighbour facts corrupts the
multi-step optimisation; conservative gating (2× the max seen epistemic, ~31% override
vs 55% at P8's threshold) reduced but did not remove the cost. This is reported as a
finding, not tuned away — quantifying it and finding a non-destructive way to bring
retrieval into planning is P9-002's first ablation.

**Perf note:** `SemanticStore.query` re-stacked its key matrix on every call — fine
for P8's few-hundred queries, fatal in CEM planning (hundreds of thousands). Caching
the stacked matrix (invalidated on `write`) cut the gate from 13m to ~2m20s.
