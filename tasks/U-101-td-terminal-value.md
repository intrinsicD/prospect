# U-101 — TD-learned terminal value bootstrapping the CEM score

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R1, R2
- **ADRs:** ADR-0001, ADR-0006/0007 (penalty inside the rollout, value at the leaf)
- **Depends on:** a gated sparse/long-horizon task existing
- **Phase gate:** the future sparse/long-horizon phase gate that triggers this
- **Source:** `docs/sota-review-2026-07.md` U-101 · [TD-MPC2](https://arxiv.org/abs/2310.16828)
  · [TD-M(PC)2](https://arxiv.org/abs/2502.03550)

## Trigger (promote to `ready` when…)
A **gated sparse-reward or long-horizon task enters the roadmap** — e.g. cartpole
swingup promoted from the non-gated BH tier into a P-series gate, or any task whose
effective horizon exceeds the planner's finite horizon. The **upgrade-triggers** workflow
step (CLAUDE.md) checks this at docs-sync: if a new/edited gate's task exhibits sparse
reward or credit assignment beyond `FlatPlanner.horizon`, promote. Until then pure
finite-horizon imagined return is the defensible minimal choice (review RQ2).

## Goal
When the trigger fires: add a small TD-learned terminal value that bootstraps the CEM
score at the rollout leaf, so the planning horizon can shrink (which also curbs
compounding model error) without losing long-horizon credit assignment.

## Non-goals
- Do NOT build before the trigger — pure MPC is correct for the current dense-reward
  envs; a value head here would be speculative generality (ADR-0005).
- Not a full actor-critic / policy-prior triad (that is the further BMPC territory) —
  only the terminal value.

## Interface to satisfy (when promoted)
A value head (reusing `_MLP`) trained by TD on buffer transitions; `FlatPlanner._imagined_returns`
(planning.py:77-89) adds `discount^H · V(leaf_latent)` to each candidate's score. The
epistemic penalty stays inside the rollout; the value sits at the leaf.

## Approach (brief, when promoted)
- TD(0)/λ target on the replay buffer; **train the value on planner-visited actions, not
  policy samples** — TD-M(PC)2's diagnosis of value overestimation from planner/policy
  distribution mismatch.
- Compose cleanly with the epistemic penalty (penalty per step, value at the leaf) and
  with U-001's propagated uncertainty.

## Acceptance criteria (when promoted)
- [ ] Value head trained via `Learner`; planner score bootstraps it at the leaf.
- [ ] Beats pure finite-horizon MPC on the triggering sparse/long-horizon gate at equal
      compute; does not regress the dense-reward P2 gate.
- [ ] `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: value head TD update decreases target error; planner score includes the leaf term.
- Eval: the triggering gate + `make gate PHASE=P2` (no regression) + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready, then follow the standard lifecycle.
- [ ] ADR-0001/0003: record the value-bootstrap addition and its trigger.
- [ ] `docs/sota-review-2026-07.md`: note U-101 promoted/shipped.

## Gate result
<deferred — no gate until promoted>
