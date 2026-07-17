# MM-009 terminal parent-verifier failure audit

## Verdict and claim boundary

MM-009 is terminally **`invalid_MM009`**. This is an infrastructure/package
verification failure before real row detachment, prediction, target isolation, or
scoring. It is not a causal-model result, a real-negative-control result, an
inconclusive family result, a clean NO-GO, or a GO.

The immutable canonical root remains
`bench/multimodal_causal_diagnostics/results/MM-009/`. It must not be repaired,
resumed, deleted, or reused.

### Artifact storage

The complete custody tree is preserved as the checksum-pinned release asset referenced
by
`artifact-pointers/MM-009.json`.
That archive, after both its archive SHA-256 and full-tree digest verify, is the
authoritative custody copy. A source-only Git checkout retains no generated MM-009
records or runtime; Git cannot represent the sealed read-only directory/file modes,
and repository policy excludes all generated evidence regardless of format or size.

From the repository root, reconstruct and verify the full canonical tree with:

```bash
python tools/materialize_artifact.py \
  artifact-pointers/MM-009.json
```

The materializer downloads to temporary storage and verifies before installation. If
the canonical destination already exists, it replaces it only after authenticating
the existing tree against the archive.

The same release includes
`prospect-MM-009-NumPy-2.4.6-NOTICES.txt`, an exact copy of NumPy 2.4.6's
consolidated notices for NumPy and its bundled OpenBLAS, LAPACK, and GCC runtime
libraries. Its checksum and repository copy under `THIRD_PARTY_NOTICES/` are bound by
the artifact pointer.

## Bound formal records

| Record | SHA-256 | Mode / time |
|---|---|---|
| protocol | `ca39f7cea6a2a5b041956b419bf3530dd54eb8403096963a044d7fcf1e2121cc` | copied before formal start |
| config file | `8f849f2a96f6cb1fce9dc18910dd83f35d2d6da00207c2ef6b984f4ec087d99d` | copied before formal start |
| input manifest | `33ff17dc8c50f590a8bdcf350057cf79a3a5a8cbcaf1dc487f649faa4b7133c9` | copied before formal start |
| freeze record | `1599d757cfa499b8656cab688b31d29b98fc1106bb856c61d6c3d49c540dff8a` | copied before formal start |
| formal marker | `430bc2e9bd7654059a336501ae779557995cf26e0e742390226518df094f2d56` | `0444`, 2026-07-16 21:48:51.020624787 +02:00 |
| formal controls | `ebe05b36394c94b202c96d7fb1c3d885f79a6be5f524e7402588e302960e46e7` | `0444`, 2026-07-16 21:58:09.705446740 +02:00 |

The durable formal controls pass the fitting-free semantic validator and reproduce
the preregistered evidence digest
`598c26a037453f199f07cc5864375689854c7377c376ef41cb85203a824f6476`.
This is synthetic-only evidence and licenses no real-mechanism claim.

## Exact failure location

After committing and validating the formal controls, the formal child entered
`_post_marker_inputs`. Its first operation called the live MM-007 verifier. The
recursive chain was MM-007 -> MM-006 -> MM-005 -> MM-004 -> MM-003, where MM-003
raised `ValueError("MM-003 transform records do not recompute")`. The wrapping chain
ended as `InvalidMM007ParentParity`, and the outer supervisor reported a failed
formal child. No copied MM-007 frame archive had yet been parsed by MM-009.

The formal service was
`mm009-custody-formal-3016538-3016538-175947298472501.service`; it exited with status
1 at 21:58:10 after 9 minutes 20.220 seconds. The preflight service was
`mm009-custody-preflight-3016538-3016538-175946719915848.service`. Both units are now
absent/inactive, and a post-audit process/cgroup census found no MM-009 residue.

## Root-cause experiment

MM-003 contains 24 sealed transform records. Refit comparisons found that all 20
non-PCA records are bit-exact across the tested formal environment. Only the four
`pca32_postz` records (one per fold) change:

| Quantity | Largest observed absolute difference |
|---|---:|
| PCA projection coefficient | `1.821e-14` |
| singular-value spectrum | `1.776e-14` |
| standardized output scale | `6.66e-16` |
| standardized output mean | `4.82e-17` |
| PCA subspace projector | `8.89e-16` |

With base Python 3.12.9, NumPy 2.1.3, and the original 16- or 32-thread OpenBLAS
regime, the complete sealed transform-record digest reproduces exactly as
`164f5563853f3fc25e9c4b649b116c94df47ef40b81d5d25feb91ec6e3e4a4c4`; the formal
one-thread digest is
`dfb9d29f542def4d079fa98c50d0290f0794777cb55ef329c5d8b53b59579959`.
Thread counts 1, 2, 4, and 8 each produce different exact digests while preserving
the PCA subspaces to numerical precision. MM-009 forced `OPENBLAS_NUM_THREADS=1`;
its inherited MM-003 verifier requires byte-exact refit hashes. This makes the
historical verifier thread-sensitive even though the scientific transform is
numerically equivalent.

A separate custody defect amplified the problem: the reviewed command began at the
lexical `.venv/bin/python` path with NumPy 2.4.6, but the cgroup helper called
`Path(command[0]).resolve()` and executed `/home/alex/miniconda3/bin/python3.12` with
NumPy 2.1.3. The formal controls happened to remain bit-identical, so they did not
detect this audited/formal runtime split.

The evidence supports a verifier/runtime-parity defect. It does not establish
corruption or scientific invalidity of MM-003 through MM-007.

## Absent outcomes

The canonical root contains no detachment index or detached row trees, no prediction
attempt, prediction commits, or prediction freeze, no pre-score budget, no
future-isolation record, no row scores, no aggregate evidence, no result, no report,
and no completed artifact manifest. Therefore no MM-009 future target was scored and
there is no real MM-009 outcome to interpret.

## Authorized continuation

The GO branch to a TAESD/MM-001 successor and the clean-NO-GO MM-010 analog/coverage
branch are both prohibited. MM-010 remains reserved. The only evidence-authorized
continuation is a new-identity, empty-root, independently audited lineage/runtime
conformance assay that:

1. authenticates the sealed MM-007 files and only the frame/normalizer semantics the
   causal assay actually consumes;
2. never recursively refits historical experiments;
3. binds both lexical and terminal interpreter identities plus NumPy/BLAS provenance;
4. exercises the exact eventual child environment before formal commitment; and
5. licenses a fresh causal assay identity only after passing its own mutation tests
   and independent audit.

An AIDE-style harness comparison remains a separate research direction and cannot
adapt on MM-009's withheld real targets.

## Independent audit

An independent referee rechecked marker/control ordering, hashes, the absent-output
census, journal cleanup, the exact formal exception chain, and the one-thread versus
default-thread PCA reproduction. Its verdict agrees with `invalid_MM009` and the
claim boundary above.
