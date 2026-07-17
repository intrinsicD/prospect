# Architecture

## A01: Harness-side frozen tiny multimodal preflight stack
- **Design**: Keep heavyweight modality backbones outside the task-unspecific Prospect
  core. MM-001 runs immutable TAESD vision and framewise video, SNAC-24k audio, and
  T5-Efficient-Tiny text sequentially in the benchmark harness, projects retained
  frontend representations to fixed 32-dimensional vectors, and trains only the
  existing small `FlatWorldModel` and `UniversalCodec` seams.
- **Selected path**: TAESD, SNAC, and T5 were selected for the smallest formal path;
  the earlier TAEHV, DAC, and FLAN-T5 alternatives were not executed or evaluated.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Evidence**: [`docs/research/2026-07-15-mm001-small-real-multimodal-preflight-protocol.md`,
  `bench/multimodal_preflight/`,
  `bench/multimodal_preflight/results/MM-001/input-manifest.json`]
- **From staging**: O35
