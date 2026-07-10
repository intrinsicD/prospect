# U-102 — LeJEPA/SIGReg replacing EMA+stop-grad+VICReg (SimNorm fallback)

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R1, R4
- **ADRs:** ADR-0006 (representation anti-collapse)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P1"]` (rank + capability no worse)
- **Source:** `docs/sota-review-2026-07.md` U-102 · [LeJEPA](https://arxiv.org/abs/2511.08544)
  · [TD-MPC2 SimNorm](https://arxiv.org/abs/2310.16828)

## Trigger (promote to `ready` when…)
The anti-collapse stack **blocks a gate or needs per-task retuning** (the ADR-0006
std/rank floors or VICReg weights require hand-tuning for a new env/modality), **or** a
deliberate simplification sprint is scheduled. The **upgrade-triggers** workflow step
checks: if a gate task reports EMA/VICReg tuning as a blocker, or the review's "simplify"
intent is picked up, promote. Until then the current EMA+VICReg+RankMe stack matches best
practice and passes every gate — do not swap a working mechanism.

## Goal
When triggered: replace EMA target + stop-gradient + VICReg with a single SIGReg term
(sketched isotropic-Gaussian regularization — random projections + a univariate
goodness-of-fit statistic), which LeJEPA proves optimal and which removes the teacher,
stop-grad, and schedulers. This is the review's one credible *simplification* (three
mechanisms → one, ~50 LOC, arguably simpler than the current pipeline).

## Non-goals
- Not adopting on faith — the evidence is large-scale vision SSL, not tiny RL latents, so
  this is a *gated* swap with a hard kill criterion, never a mid-review replacement.
- SimNorm (TD-MPC2 simplicial normalization) is the fallback if SIGReg underperforms.

## Interface to satisfy (when promoted)
`world_model.FlatWorldModel.update`: SIGReg term replaces the VICReg var/cov terms
(world_model.py:300-309) and the EMA target machinery (world_model.py:122-123, 166-178,
317-320); `encode_target` collapses into `encode`. `WorldModel`/`Learner` protocols
unchanged.

## Approach (brief, when promoted)
- SIGReg: project latents onto random 1-D directions, penalize deviation from a standard
  Gaussian via a closed-form statistic (Epps–Pulley) — linear time, one trade-off knob.
- Kill criterion: **effective rank and every gate metric no worse** with EMA/stop-grad/VICReg
  deleted. If SIGReg fails, try SimNorm (softmax-normalized latent groups, ~5 lines);
  if both fail, keep the current stack and mark the trigger closed with evidence.

## Acceptance criteria (when promoted)
- [ ] SIGReg (or SimNorm) replaces the stack; effective rank ≥ current; P1 capability ≥
      current; all collapse sentinels healthy.
- [ ] `make gate-all` green; net LOC reduced (the simplification is real).
- [ ] tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: SIGReg drives latents toward isotropic Gaussian on a synthetic batch; rank held.
- Eval: `make gate PHASE=P1` + `make gate-all` (regression is the kill criterion).

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0006: record the simplification and its measured kill/keep decision.
- [ ] `docs/sota-review-2026-07.md`: note U-102 outcome.

## Gate result
<deferred — no gate until promoted>
