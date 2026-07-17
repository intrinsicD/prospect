# Datasets

This directory contains reusable inputs preserved independently of generated
experiment reports and result packages. Verify them from this directory with:

```bash
sha256sum -c SHA256SUMS
```

## BridgeControl

`bridge-control/` contains ten synthetic transition datasets with 896 rows each.
Every NPZ includes `states`, `actions`, `next_states`, `rewards`, region/lane/slot
identities, replicate identities, and the bridge/rank/density/control labels.
`dataset-manifest.json` records their original schema and hashes. Its dataset
paths are relative to the manifest.

These fixtures were generated within Prospect. Their original generator remains
available in Git history; the generated result reports and fitted outputs are not
part of the fresh tree.

## Perception Test

`perception-test/frames-64x64.npz` contains 477 RGB frames, timestamps, and video
identities derived from the official Google DeepMind Perception Test sample.
`perception-test/derived/` retains the associated frozen feature, pairing,
projection, transform, pixel-grid, and row datasets without the old experiment
reports.

Perception Test data is distributed under CC BY 4.0 and its repository software
under Apache 2.0:
<https://github.com/google-deepmind/perception_test>.

The upstream sample-video and annotation cache under
`~/.cache/prospect/perception_test_sample` is external to this repository and was
not modified by the cleanup.
