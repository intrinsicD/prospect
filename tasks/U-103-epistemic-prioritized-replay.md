# U-103 — Epistemic-prioritized replay sampling

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R7
- **ADRs:** ADR-0002 (epistemic signal), ADR-0006
- **Depends on:** U-004 recommended first (fix eviction before prioritizing draws)
- **Phase gate:** the triggering nonstationarity/adaptation gate; `["P7"]`
- **Source:** `docs/sota-review-2026-07.md` U-103 · [UPER](https://arxiv.org/abs/2506.09270)
  · [Curious Replay](https://arxiv.org/abs/2306.15934)

## Trigger (promote to `ready` when…)
A **nonstationarity/adaptation gate exists**, or a continual/adaptation gate **shows
uniform sampling as the limiter** (the world model under-fits the changing region under
uniform draws). The **upgrade-triggers** workflow step checks: if a P7-class or new
adaptation gate's report attributes a miss to sampling, promote. Until then uniform
sampling is defensible — even the Curious Replay paper found uniform ≈ prioritized on
stationary tasks (review RQ1).

## Goal
When triggered: prioritize replay draws by **epistemic** uncertainty (information gain),
not raw TD/model loss — UPER's result is that raw-loss prioritization over-samples
aleatoric-noise transitions (a replay-side noisy-TV), and the epistemic/aleatoric split
this repo already carries is the correct priority. Because `Prediction` carries the
split, this is nearly free when needed.

## Non-goals
- Not vanilla TD-error PER (the review's explicit SKIP — over-samples noise).
- Not a learned prioritizer; the priority is the already-computed epistemic scalar.
- Sampling only — eviction is U-004.

## Interface to satisfy (when promoted)
`memory.ReplayBuffer.sample` (memory.py:73-77) draws proportional to a stored
per-transition epistemic priority (set from `Transition.prediction.epistemic` at `add`
time, or refreshed lazily). `EpisodicMemory` protocol unchanged.

## Approach (brief, when promoted)
- Store the act-time epistemic with each transition (already on `Transition.prediction`);
  sample with probability ∝ epistemic^ω (ω a small exponent), with an importance-sampling
  correction if the training loss requires unbiasedness.
- Prioritize by *epistemic*, never raw loss — the repo is unusually well positioned to do
  this right (UPER's whole point).

## Acceptance criteria (when promoted)
- [ ] Priority draws by epistemic; unit test: high-epistemic transitions over-sampled,
      high-*aleatoric* ones are NOT (the noisy-TV-of-replay defense).
- [ ] Beats uniform on the triggering adaptation gate; P7/P3 not regressed.
- [ ] `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: sampling distribution matches priorities; aleatoric-heavy transitions not favored.
- Eval: the triggering gate + `make gate PHASE=P7` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0002/0006: record epistemic-prioritized replay and its trigger.
- [ ] `docs/sota-review-2026-07.md`: note U-103 promoted/shipped.

## Gate result
<deferred — no gate until promoted>
