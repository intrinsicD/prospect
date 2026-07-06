# P12-001 — Swappable visual perception (real frozen eyes → predictive world model)

- **Status:** proposed (pending sign-off)
- **Phase:** P12
- **Requirements:** R6 (process any kind of input), R4 (right patterns), R1 (predict),
  R3 (VoE / surprise on what it sees)
- **ADRs:** ADR-0009 (omni-modal seams — **vision is the first instantiation**), ADR-0001
  (shared latent; the codec is the input seam), ADR-0002 (understanding = predictive, VoE),
  P0-011 (distill-first representation migration — the swap mechanism)
- **Depends on:** P6 (`UniversalCodec` + distillation), P0-011 (migration discipline)
- **Phase gate:** new `bench/gates.py::GATES["P12"]` — a single-task phase; PASS ships it.

## Goal
Give the agent **real eyes**: a frame passes through a *frozen pretrained* visual encoder
to an embedding, which the codec ingests into the shared latent, and the world model
**predicts over what it sees** and is **surprised** when wrong. And make vision a *swappable*
module — a better encoder drops in without retraining the core (the P0-011 guarantee), so
the project can upgrade its eyes as the field improves.

## Non-goals
- No **learning from video / latent actions** — that is P13 (this phase only proves the
  agent *sees and predicts*; it does not yet learn behavior from observation).
- No **live webcam in the gate** — a live camera is non-deterministic and needs hardware,
  so it is a runtime demo (a thin wrapper), not a CI gate (ADR-0009).
- No **training the vision encoder** — it is frozen; we only use it for inference.
- No **pixels in the core** — the core reasons over embeddings; the encoder is harness-side.

## Interface to satisfy
- Core: a `Modality.VISION` in `types`; the `UniversalCodec` gains a VISION adapter that
  distils embeddings into the incumbent latent (reuses `distill_encode`, P0-011). No new
  `Protocol` — an embedding-carrying `Observation` flows through the existing codec seam.
- Harness (`bench/`, optional `[vision]` extra, ONNX Runtime preferred): a
  `frame -> embedding` encoder used **once, offline**, to generate committed embedding
  **fixtures**. The P12 gate is numpy over those fixtures (deterministic, CI stays
  numpy-only).

## Approach (brief)
- **Fixtures:** render a few short *deterministic* clips (a moving/interacting shape — the
  content is controlled; the *seeing* is real), run each frame through TWO different frozen
  encoders offline, and commit the resulting per-frame embedding sequences as fixtures. (A
  small set of real image frames is an acceptable alternative; rendered clips avoid
  licensing/size and keep the gate reproducible.)
- **Ingest:** distil the codec's VISION adapter to land encoder-A embeddings in the
  incumbent latent (P6-style); train `FlatWorldModel` to predict next-visual-latent over
  the clips.
- **Swap:** distil a VISION adapter for encoder-B; show the frozen core loop is preserved
  within tolerance (the swap test, P6-style).

## Acceptance criteria (single-task phase — PASS ships)
- [ ] **Sees and predicts:** over held-out clip frames, the world model's 1-step
      next-visual-latent MSE beats a persistence baseline (it learned visual dynamics from
      *real* embeddings).
- [ ] **Vision is swappable (P0-011):** distilling a second, different frozen encoder into
      the incumbent latent preserves core-loop 1-step MSE within tolerance — a better
      vision module drops in without retraining the core.
- [ ] **Surprise is calibrated:** epistemic VoE is higher on genuinely novel / out-of-clip
      frames than on in-distribution ones (understanding = knowing what it did not expect).
- [ ] `make gate PHASE=P12` PASS with all sentinels healthy; P12 appended to `bench/SHIPPED`;
      `make gate-all` green; `make test`/`lint`/`typecheck` clean; the vision backend is an
      optional extra and CI stays numpy-only (gate runs over committed fixtures).

## Test plan
- Unit: the VISION modality round-trips through the codec; a fixture loader is deterministic.
- Eval: `bench/evals/p12_vision.py::check_p12` — the three criteria over committed fixtures.
- (Fixture generation is a separate offline dev script under the `[vision]` extra, not run
  in CI.)

## Docs-sync checklist
- [ ] Status → `done`; gate result recorded below.
- [ ] ADR-0009 → Accepted; codec docstring notes the VISION modality.
- [ ] R6 traceability row (+P12); roadmap P12 row; BACKLOG P12 + shipped note.

## Open decisions for sign-off
- **Encoder choice** for the fixtures (e.g., DINOv2-small vs. a CLIP/ViT vs. a tiny
  MobileNet) — smaller = faster fixtures, larger = more semantic. Two *different* ones are
  needed for the swap test.
- **Fixture content:** rendered deterministic clips (recommended) vs. a small committed set
  of real frames.
- **CI policy:** commit embedding fixtures so the gate needs no vision backend (recommended),
  vs. gate the vision extra directly in a separate CI job.

## Gate result
<pending>
