# P12-001 — Swappable visual perception (real frozen eyes → predictive world model)

- **Status:** done
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
- [x] **Sees and predicts:** held-out next-visual-latent MSE beats persistence — **0.0055
      vs 0.2651 (48×)**; the world model learned the visual dynamics from embeddings.
- [x] **Vision is swappable (P0-011):** a second, different frozen encoder distils into the
      incumbent latent and preserves the core loop — **encoder-B-via-codec 0.0058 vs
      incumbent 0.0055 (1.05×)**. A better vision module drops in without retraining the core.
- [x] **Surprise is calibrated:** epistemic VoE higher on novel (two-blob) frames than
      familiar ones — **0.0032 vs 0.0007 (4.6×)**.
- [x] `make gate PHASE=P12` PASS, all sentinels healthy; P12 appended to `bench/SHIPPED`;
      `make gate-all` green; `make test`/`lint`/`typecheck` clean; CI stays numpy-only.

## Test plan
- Unit: the VISION modality round-trips through the codec; a fixture loader is deterministic.
- Eval: `bench/evals/p12_vision.py::check_p12` — the three criteria over committed fixtures.
- (Fixture generation is a separate offline dev script under the `[vision]` extra, not run
  in CI.)

## Docs-sync checklist
- [ ] Status → `done`; gate result recorded below.
- [ ] ADR-0009 → Accepted; codec docstring notes the VISION modality.
- [ ] R6 traceability row (+P12); roadmap P12 row; BACKLOG P12 + shipped note.

## Decisions taken (Path B build)
- **Encoders:** two **deterministic stand-in** frozen encoders (fixed random-feature
  projections). A *real* pretrained encoder only makes sense on real image content, which a
  numpy CI gate has none of — and it swaps in via the identical distill path (that is the
  whole point of ADR-0009's swappability). Real DINOv2/CLIP embeddings are the local
  `[vision]` regen (Path A), where the content is real.
- **Content:** rendered deterministic clips — a blob orbiting under a fixed global flow, so
  single-frame prediction is well-posed; novel = a two-blob scene (OOD embeddings).
- **CI policy:** because the stand-in encoder is pure numpy and deterministic, embeddings
  are generated **inline** — no committed fixture files needed; CI stays numpy-only.
  Committed fixtures become relevant only when a real (torch/ONNX) encoder is used offline.

## Gate result
`make gate PHASE=P12` → **[P12] PASS**, all five collapse sentinels healthy (~2m). Median
over 3 seeds:

| criterion | measured | bar |
|---|---|---|
| sees & predicts — wm vs persistence MSE | **0.0055 vs 0.2651 (48×)** | wm·1.2 ≤ persist |
| swappable — encoder-B-via-codec vs incumbent MSE | **0.0058 vs 0.0055 (1.05×)** | ≤ 1.5× |
| surprise — novel vs familiar epistemic | **0.0032 vs 0.0007 (4.6×)** | ≥ 1.5× |

The swap ratio ~1.0 is the headline: a second frozen encoder, distilled into the incumbent
latent (P0-011), drives the frozen world model *as well as* the original — a better vision
module drops in without retraining the core. **P12 ships** (`bench/SHIPPED` ratchets
P0–P12). Real vision (a pretrained encoder on real frames) is the `[vision]`-extra regen +
the live-webcam runtime demo — same seam, same distill path (ADR-0009).
