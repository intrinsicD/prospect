# P6-001 — Universal codec: any-to-any I/O via distillation into the incumbent latent

- **Status:** done
- **Phase:** P6
- **Requirements:** R6
- **ADRs:** ADR-0001 (one latent hub; the latent is a *contract*, amended by
  P0-011), ADR-0006 (reconstruction stays off the dynamics path)
- **Depends on:** P2-001 (a working core loop to preserve)
- **Phase gate:** `bench/gates.py::GATES["P6"]`

## Goal
`UniversalCodec` satisfying `interfaces.Codec`: any input modality → the shared
latent, latent → any output modality. The migration is the point (P0-011): the
codec is **distilled into the incumbent latent space** — its `encode` is trained
to match the frozen P1 encoder on the shared modality — so the dynamics model,
option model, competence stats and stored replay latents stay valid when it
replaces the single-modality encoder. Any-to-any is then real: a *second*
modality carrying the same situation distils to the SAME latent, so the frozen
core loop reasons over it identically.

## Non-goals
- No true multi-head cross-attention (the pooled trunk is the minimal Perceiver
  bottleneck: an arbitrary input → the fixed latent; cross-attention is earned by
  a later gate if a modality needs it).
- No retraining of the dynamics/option models — the whole point is they are NOT
  retrained (distill-first, ADR-0001/P0-011). A full-stack retrain is the
  documented fallback, not this task.
- No new environment; no pixels-into-dynamics (decode reconstruction is trained on
  the FROZEN latent, so it never pressures the dynamics latent — ADR-0006).

## Interface to satisfy
`prospect.interfaces.Codec` — implement `UniversalCodec` in `prospect/codec.py`
(replace the skeleton). `encode(Observation) -> LatentState` routes by modality;
`decode(LatentState, query) -> Observation` reads the latent out to the queried
modality. Distillation via dedicated methods (`distill_encode`, `fit_decode`) —
codec training is supervised latent-matching, not transition replay, so it does
NOT force-fit `Learner`.

## Approach (brief)
- Per-modality input adapter (`data -> token`), a shared trunk (`token ->
  latent`) — the bottleneck that lands ANY modality in the fixed latent — and
  per-modality decode heads (`latent -> data`), all small numpy MLPs; per-modality
  input standardization frozen from the first distill batch.
- Two modalities on the Pendulum: STATE `[cosθ, sinθ, ω]` (the incumbent
  modality) and IMAGE, a rasterized sensor view of the same situation (angular +
  velocity Gaussian bins). Distill both to the incumbent latent on paired data.
- Gate eval (`bench/evals/p6_codec.py`, `@gate_check("P6")`, run `p6` carrying all
  four applicable sentinels' records — one model per seed feeds probes, replay
  fidelity, a hierarchy rollout for option-diversity, and the codec swap): held-out
  1-step prediction MSE with the codec-swapped encoder (STATE and IMAGE) vs the
  incumbent encoder; pass iff both stay within a tolerance factor on every seed.
  Cross-modality latent agreement and decode reconstruction reported.

## Acceptance criteria
- [x] Implements `interfaces.Codec`; conformance holds; unknown modality (encode)
      and unknown query (decode) fail loudly (KeyError).
- [x] Distillation works: `encode(STATE)` reproduces the incumbent latent and
      `encode(IMAGE of the same state)` lands in ~the same latent (unit-tested);
      decode reconstructs the modality.
- [x] **Gate P6 PASS:** codec-swapped 1-step MSE ratios essentially 1.0 — STATE
      1.00/1.00/0.98, IMAGE 1.02/1.00/0.99 per seed (tolerance ×1.5): the swap is
      nearly transparent and the IMAGE modality drives the frozen core loop
      identically (any-to-any, measured). Cross-modality latent MSE ~0.001 (≈50x
      below the model's own prediction error); STATE reconstruction MSE ~0.005.
      All four applicable sentinels healthy on run `p6`. `P6` in `bench/SHIPPED`.
- [x] `make test` green (88), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_codec.py): protocol conformance; encode-distills-to-target;
  two modalities of the same state land in ~the same latent; decode reconstructs;
  unknown modality/query errors.
- Eval: `make gate PHASE=P6`; then `make gate-all` (P0–P6).

## Docs-sync checklist
- [x] Status → `done`; the P6 PASS GateReport below; `bench/SHIPPED` += P6.
- [x] architecture.md codec.py note still accurate (any input → shared latent,
      latent → any output — now literal); roadmap's distill-first migration note
      (P0-011) is now validated.
- [x] Backlog: P6-001 done; **Phase 6 shipped**; P7-001 / P8-001 next.

## Gate result
`make gate PHASE=P6` — PASS record `bench/results/P6-20260704T082650Z.json`:

```
[P6] PASS
  capability: ok — codec-swapped 1-step MSE / incumbent — STATE [1.0, 1.0, 0.98],
    IMAGE [1.02, 1.0, 0.99] (tolerance x1.5); the swap preserves the core loop
    and a second modality drives it identically on every seed: YES
  sentinels: representation-integrity, uncertainty-reliability, replay-fidelity,
    option-diversity — ALL healthy
```

The P0-011 thesis is now measured: the codec swap is a *representation* change,
and distilling the new codec into the incumbent latent is what makes it cheap —
the dynamics model is never retrained, yet swapping its encoder (even for an
entirely different modality, the rasterized IMAGE) moves 1-step prediction by
<2%. Cross-modality latent disagreement (~0.001) is ~50x smaller than the
model's own prediction error, so "same situation → same latent regardless of
modality" holds functionally, not just nominally.

One measurement-fidelity note: the option-diversity sentinel first read d'=0.38
(< 0.5 floor) because the codec eval undersampled the hierarchy with 6 episodes;
these are the *same* options that measured d'=2.12 in P5, so matching P5's
episode budget (EVAL_EPISODES) restored a faithful d'=0.66 — the sentinel was
right to flag an unreliable measurement, and the fix was to measure properly,
not to lower the floor. `gate-all`: 7 shipped gates green (~5m).
