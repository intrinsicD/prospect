# U-009 — Gate probe: endpoint-swap state-leakage test for latent actions

- **Status:** ready
- **Phase:** U (upgrade track; re-gates P13)
- **Requirements:** R7, R1
- **ADRs:** ADR-0010 (latent-action identifiability)
- **Depends on:** none
- **Phase gate:** folds into `bench/gates.py::GATES["P13"]` as an identifiability check
- **Source:** `docs/sota-review-2026-07.md` U-009 · [Garrido et al. 2026](https://arxiv.org/abs/2601.05230)

## Goal
The published identifiability test for latent-action models is the endpoint-swap probe:
swap the endpoints of transitions and check that forward-prediction error *spikes*; if it
doesn't, the "action" is smuggling state. Add it as a gate check for the decorrelation
penalty (`observation.LatentActionModel`, observation.py) — a concrete negative control
for the P13 identifiability claim the repo currently supports only via recovery R².

## Non-goals
- Not switching to VQ (the review validated continuous + decorrelation as *better*
  aligned with 2025–26 evidence than the VQ orthodoxy — this only *verifies* the choice).
- Not changing the decorrelation penalty itself — adding the probe that certifies it.

## Interface to satisfy
A new check in the P13 eval (`bench/evals/p13_observation.py`) + a criterion in
`bench/gates.py`: infer latent actions on a batch, then on the same batch with swapped
next-observations; require forward-prediction error under the swapped endpoints to exceed
the matched-endpoint error by a margin (no spike ⇒ state leakage ⇒ FAIL). Uses the
existing `LatentActionModel.infer_action`/`predict` (observation.py:51-64).

## Approach (brief)
- Matched: `predict(o_t, infer_action(o_t, o_{t+1}))` reconstructs `o_{t+1}` well.
- Swapped: feed a mismatched `o_{t+1}'`; a genuine action-only latent makes the forward
  model's reconstruction *worse* (it encodes the transition, not the endpoint); a
  state-leaking latent stays low-error because it copied the endpoint.
- Ratio (swapped error / matched error) ≥ floor is the identifiability certificate.

## Acceptance criteria
- [ ] Endpoint-swap check in the P13 eval with a documented floor; the shipped
      decorrelation-trained model PASSes (records the ratio).
- [ ] Negative control: a bottleneck trained *without* decorrelation FAILs the probe
      (state leakage detected) — matches the P13-001 recovery-R² story (0.02 vs 0.80).
- [ ] **P13 gate PASS**; `make gate-all` green; `make test`/`lint`/`typecheck` clean.

## Test plan
- Unit (tests/test_observation.py): swapped-error/matched-error ratio high for the
  decorrelated model, ~1 for a naive bottleneck.
- Eval: `make gate PHASE=P13`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; ratio recorded below.
- [ ] ADR-0010: add the endpoint-swap probe as the standing identifiability check;
      cite Garrido et al. 2026.
- [ ] `docs/sota-review-2026-07.md`: mark U-009 shipped.

## Gate result
<paste the GateResult once run>
