# P9-002 — Ablation harness (prove every part is load-bearing)

- **Status:** done
- **Phase:** P9
- **Requirements:** R1–R8 (which component earns its place)
- **ADRs:** ADR-0008 (ablation is how a component *earns* its place — a part that
  can be removed with no measurable loss is either dead weight or untested)
- **Depends on:** P9-001 (the composed agent + the E2E metric ablations are measured
  against)
- **Phase gate:** contributes to `bench/gates.py::GATES["P9"]` — adds the
  "every ablation degrades the E2E metric" criterion / an `ablation` record.

## Goal
Systematically disable each component of the composed agent and measure the delta on
the P9 end-to-end metric. The claim: **removing any load-bearing part measurably
hurts.** A no-op ablation is a finding — either the component is not pulling its
weight (a design shortcoming) or the E2E gate can't see its value (a test
shortcoming). This turns "we assume each piece matters" into "we measured it."

**Start with the finding P9-001 already surfaced:** retrieval-into-planning has a
*negative* marginal value (disabling retrieval IMPROVED control, ~-19 → ~-8 return) —
retrieval helps 1-step prediction (P8) but overriding the planner's rollout dynamics
corrupts multi-step optimisation. Quantify the delta across gate settings, and
investigate whether retrieval can enter planning without the cost (e.g. depth-0 only,
or as a bounded residual correction rather than a blanket override). This is the first
ablation with a known, non-trivial answer — a good test that the harness measures what
it should.

## Non-goals
- Not a redesign of any component; ablations only *disable* (identity/point-estimate/
  no-op), they do not rewrite.
- Not cross-environment (P9-003) — ablations run on the P9 reference setup.
- No new capability.

## Interface to satisfy
`bench/evals/p9_ablation.py` — pure `marginals()` (leave-one-out `composed - ablated`)
and `classify()` (load-bearing / harmful / negligible). `check_p9` (the P9 gate) runs
the ablations over the P9 control loop's real components and records the table. The
ablatable components in that loop are exactly **planning** (reactive), **retrieval**
(bare model, no augmentation), and **exploit_penalty** (no curriculum, penalty pinned
to 0) — hierarchy/skills are not wired into this loop, so there is nothing to ablate
there. No new `Protocol` (ablations toggle existing seams).

## Approach (brief)
- Reuse the P9 harness: planning and retrieval marginals come from the reactive and
  bare conditions the gate already runs; add one `exploit_penalty` run. Marginal value =
  `composed - ablated` (positive => load-bearing; negative => harmful; ~0 => dead weight).
- Gate on the clearly load-bearing component (planning) being positive on every seed —
  a check that the harness measures what it should. Record every marginal; a harmful or
  negligible one is a **reported finding**, not tuned away (ADR-0008).

## Acceptance criteria
- [ ] `p9_ablation.py` computes leave-one-out marginal control value + classification;
      `check_p9` runs the three component ablations and records the table.
- [ ] Planning is confirmed load-bearing (positive marginal) on every seed — folded
      into the P9 gate; **P9 gate still PASS**.
- [ ] Retrieval's negative marginal (the P9-001 finding, now quantified) is recorded
      and classified `harmful`, not tuned away; the `exploit_penalty` marginal measured.
- [ ] Unit tests for the marginal / classify helpers.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit: each ablation constructor yields a runnable agent with exactly that seam off.
- Eval: `make gate PHASE=P9` reports the ablation deltas alongside the E2E result.

## Docs-sync checklist
- [x] Status → `done`; ablation table recorded below.
- [x] ADR-0008 consequences updated with the measured deltas.
- [x] Backlog: P9-002 done; the two findings (harmful retrieval, negligible
      exploit-penalty) recorded as follow-ups.

## Gate result
`make gate PHASE=P9` (3 seeds; ~3m50s):

```
[P9] PASS
  capability: ok — composed agent controls end-to-end: return -19.4 vs reactive
    -73.1 (beats every seed); one epistemic signal drives exploit-mode AND retrieval
    in one run (MET; retrieval rate 31%). ablation leave-one-out marginal control
    value: planning +53.7 (load-bearing), retrieval -9.5 (harmful),
    exploit_penalty +2.5 (negligible) [planning load-bearing every seed: MET].
    FINDING: retrieval hurts control here (marginal -9.5) ...
  sentinel[*]: all four healthy
```

**P9 PASS — the ablation harness works and immediately earned its keep.** Leave-one-out
marginal control value (`composed - ablated`) gave three distinct verdicts:

| Component | Marginal | Verdict |
|---|---|---|
| planning | **+53.7** | load-bearing (every seed — the gated criterion) |
| retrieval | **−9.5** | **harmful** (the P9-001 finding, now quantified) |
| exploit_penalty | **+2.5** | **negligible** (a second finding) |

Two findings surfaced, both reported rather than tuned away (ADR-0008):
1. **Retrieval-into-planning is harmful** (−9.5): it helps 1-step prediction (P8) but
   corrupts multi-step CEM optimisation. Open follow-up: can retrieval enter planning
   non-destructively (depth-0 only, or a bounded residual correction)?
2. **The exploit-mode penalty is ~negligible** (+2.5, within the ±5 margin) on this
   task: penalising epistemic uncertainty in planning barely helps here — either the
   control task isn't OOD-sensitive enough to reward it, or the penalty needs a
   different scale. Worth a dedicated ablation on a more OOD-forcing task (P9-003's
   second environment is the natural place).

The harness is exactly the tool ADR-0008 called for: it turns "we assume each piece
matters" into a measured table, and it caught two components not pulling their weight.
