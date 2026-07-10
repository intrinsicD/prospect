# U-107 — Continual backprop + plasticity diagnostics

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R7
- **ADRs:** ADR-0002 (forgetting detection), ADR-0006
- **Depends on:** none
- **Phase gate:** the triggering plasticity gate; `["P7"]`
- **Source:** `docs/sota-review-2026-07.md` U-107 · [Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7)
  · [ReDo](https://arxiv.org/abs/2302.12902)

## Trigger (promote to `ready` when…)
A **plasticity gate exists**, or P7-class results **show failure to re-learn after
detected forgetting** — error stays high and won't come back down (the error-rise monitor
detects the symptom but can't distinguish "world changed" from "network dying"). The
**upgrade-triggers** workflow step checks: if a continual-learning report shows sustained
non-recovery, or a plasticity phase is added, promote. The phenomenon is proven at
literally 5-unit-MLP scale (Nature 2024), so it *can* bite here — but only measure-then-fix.

## Goal
When triggered: (1) add plasticity *diagnostics* — dormant-neuron ratio (ReDo),
weight-norm, feature effective-rank — so the forgetting module distinguishes "world
changed" (error up, diagnostics stable) from "network dying" (error up, rank/dormancy
degrading); (2) add continual backprop — reinitialize a tiny fraction of low-utility
mature units — as the mitigation.

## Non-goals
- Do NOT build before the trigger — the current error-rise detector is not behind the
  literature (which proposes no online detector); this adds the *distinguishing diagnostic*
  and the *recovery mechanism* only when non-recovery is measured.
- Not a wholesale optimizer change.

## Interface to satisfy (when promoted)
`world_model._MLP`: per-unit contribution utility (|activation|·|outgoing weight|, EMA)
and a low-rate reinit of low-utility mature units (continual backprop, ~30 lines). A new
plasticity sentinel in `bench/gates.py` reads dormant ratio / effective rank from the run
metrics (P0-005).

## Approach (brief, when promoted)
- Diagnostics first (cheap, measure the failure mode); continual backprop only if the
  diagnostic confirms plasticity loss (generic L2-to-init / shrink-and-perturb is the
  simpler alternative the survey found often competitive — try it first).

## Acceptance criteria (when promoted)
- [ ] Dormant-ratio / effective-rank diagnostics in the run metrics + sentinel; continual
      backprop (or shrink-and-perturb) restores re-learning on the triggering task.
- [ ] P7 retention + the plasticity criterion PASS; `make gate-all` green; clean checks.

## Test plan (when promoted)
- Unit: dormant ratio rises on a frozen-into-saturation net; continual backprop lowers it
  and restores fit.
- Eval: the triggering plasticity gate + `make gate PHASE=P7` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0002/0006: record the plasticity diagnostic + mitigation and its trigger.
- [ ] `docs/sota-review-2026-07.md`: note U-107 outcome.

## Gate result
<deferred — no gate until promoted>
