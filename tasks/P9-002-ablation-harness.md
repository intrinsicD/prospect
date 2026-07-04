# P9-002 — Ablation harness (prove every part is load-bearing)

- **Status:** blocked (P9-001)
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
`bench/evals/p9_ablation.py` producing, for each ablation, the E2E metric delta vs.
the full agent. Ablations toggle existing seams (no new `Protocol`): point-estimate
world model (kill the epistemic/aleatoric split), no planning (reactive), no
hierarchy, no curiosity/curriculum, no retrieval, no anti-collapse regularization.
Feeds a criterion into the P9 gate and/or an `ablation-integrity` record.

## Approach (brief)
- For each ablation, construct the composed agent with that seam disabled and run the
  same P9 loop + evaluation. Record `full_metric − ablated_metric`.
- Assert each *designated load-bearing* ablation degrades the metric by at least a
  margin on every seed (paired). Document any ablation that does **not** hurt as an
  explicit finding (open a follow-up task; do not silently pass).
- Keep it cheap: reuse the P9 harness; ablations are configuration, not new code.

## Acceptance criteria
- [ ] Each load-bearing ablation (distribution, planning, retrieval, at minimum)
      degrades the P9 metric by a margin on every seed; the deltas are recorded.
- [ ] Any no-op ablation is surfaced as a finding with a follow-up task, not hidden.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit: each ablation constructor yields a runnable agent with exactly that seam off.
- Eval: `make gate PHASE=P9` reports the ablation deltas alongside the E2E result.

## Docs-sync checklist
- [ ] Status → `done`; ablation table recorded below.
- [ ] ADR-0008 consequences updated with the measured deltas.
- [ ] Backlog: P9-002 done; any no-op-ablation follow-ups filed.

## Gate result
_not run yet_
