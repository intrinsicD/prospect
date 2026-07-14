# OL-001 terminal verifier failure

**Status:** preserved failed formal artifact  
**Date:** 2026-07-14

OL-001 completed its formal outcome pass and its deterministic semantic rerun.  It
then failed the final CSV rendering comparison in `_verify_outcomes`.

The defect is newline handling, not outcome drift:

- `OL-001-runs.csv` is 11,367 bytes and contains 89 CRLF row terminators.
- Its bytes exactly equal `_csv_text(results["rows"]).encode("utf-8")`.
- `Path.read_text()` applies universal-newline translation and returns LF text.
- The verifier compared that normalized text with `_csv_text`, which still contains
  CRLF, so text equality is false even though byte equality is true.

Frozen failed-artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `protocol.json` | `e5a01edaacf0150853db3db2349edd31e0c3b5386ac8740ef2826d03e43f768e` |
| `input-manifest.json` | `a6a241d1fce5ce1a71120c42a4779f2d5e4d13006fab9b5a87422f3af0113696` |
| `OL-001-results.json` | `0f0273f9974288ad66368d38494d3bd83f06bc42e2dca3a2a65075b881982faa` |
| `OL-001-runs.csv` | `a490af7967a63e8e5f05a22452f722a0ae0fdb5855d155babf86b70593cc0b50` |
| `OL-001-report.md` | `cfb50325d56b80d49350a1cfbb5032ec0f9ae89722fcdb8d3e54af2d9d6867d6` |
| `inputs/BC-001-b1_r1_d8.npz` | `9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948` |
| `artifact-manifest.json` | `b6956d01a87db378883d382f83ba98c4e135fa4793de5f8d37dad25514cca75d` |

The pinned artifact manifest is also parsed and every listed artifact, including the
copied NPZ, must reproduce these hashes before OL-002 can prepare.

OL-001 remains untouched.  Under its frozen defect rule, no conclusion is accepted
from it and no source or artifact is repaired in place.  OL-002 reruns the entire
formal experiment under a new identifier, with the only method change being
canonical LF CSV rendering so the saved bytes and verifier comparison share one
newline convention.
