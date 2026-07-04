# P9-001 — End-to-end integration gate (the whole-system fitness function)

- **Status:** ready
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
  1. **Learns while acting:** online 1-step prediction error at the end is materially
     below the start (it builds its world model from its own experience).
  2. **One signal, many jobs, one run:** within the single run, epistemic uncertainty
     simultaneously (a) drove explore→exploit (curriculum mode flips), (b) reached
     mastery (monitor), and (c) gated retrieval (invoked while uncertain → ~0 once
     learned).
  3. **Retrieval-as-action improves control:** the composed agent's task return is at
     least as good as the same agent with retrieval disabled, at equal budget — a
     control-level result beyond P8's 1-step MSE.
  4. **Integrity holds:** all applicable sentinels healthy on run `p9`.

## Acceptance criteria
- [ ] `agent.py` wires retrieval-as-action + curriculum mode; a unit test covers the
      new `act()` path (retrieve-when-uncertain, else the model's own prediction).
- [ ] `check_p9` runs the composed agent end-to-end and the four assertions hold on
      every seed; **P9 gate PASS**.
- [ ] All applicable sentinels healthy on run `p9`.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (`tests/test_agent.py`): the extended `act()` retrieves when uncertain and
  falls back to the model when confident; the curriculum mode reaches the planner.
- Eval: `make gate PHASE=P9` — the four integration assertions + the sentinels.

## Docs-sync checklist
- [ ] Status → `done`; gate report recorded below.
- [ ] Add **ADR-0008** (integration gate + ablation as the whole-system fitness).
- [ ] `roadmap.md` gains the P9 row; requirements note the integration gate.
- [ ] Backlog: P9-001 done; P9-002 unblocked.
- [ ] If PASS: append `P9` to `bench/SHIPPED` in the same commit (ratchet, P0-007).

## Gate result
_not run yet_
