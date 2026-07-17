# MM-011 off-grid sensitivity result audit

**Disposition:** `ACCEPT`, narrowly scoped  
**Decision:** `ABANDON_FINITE_GRID_BEFORE_REAL_DATA`  
**Scientific role:** generated, no-real-target pre-real diagnostic  
**Canonical MM-011 root:** absent

## Claim disposition

| Claim | Disposition | Supported boundary |
|---|---|---|
| The frozen six-cell audit completed without opening an LCV/MM-007 scientific frame. | Confirm | File-open tracing, an API with no frame path, and the immutable receipt support a synthetic-target-only run. |
| The exact MM-011/v2.2 coarse affine grid is safe to take to the real panel. | Refute | T1--T3 fail the preregistered directionality predicate; the protocol requires abandonment before canonical preparation. |
| All finite affine grids or all motion prediction fail. | Unsupported | No such general claim follows from six generated cells and one exact grid. |
| Continuous registration will solve the problem. | Unresolved | It is the selected successor hypothesis, not an observed result. |
| Prospect gained an end-to-end capability. | Unsupported | This was not a gate, learned model, real-media score, planner test, or capability evaluation. |

The receipt's phrase `target-free` is interpreted narrowly as **no real target**.
Generated futures were constructed and scored, so `synthetic-target-only` is the
precise description.

## Independent reconstruction

The frozen threshold is candidate-error/baseline-error `<= 0.8` for historical and
forward improvement. Directionality requires reversal error/persistence error `> 0.8`:
the same repeated-direction forecast must not improve by the formal `1.25` margin
when the previous frame is substituted for the future.

| Cell | Selected coarse theta | H/I | F/P | R/RP | Result |
|---|---|---:|---:|---:|---|
| T1 | `(0,0,-2,0,2,0)` | 0.57809 | 0.57414 | 0.59482 | fail directionality |
| T2 | `(0,-4,0,2,0,2)` | 0.61518 | 0.62426 | 0.69081 | fail directionality |
| T3 | `(4,-4,2,0,-2,0)` | 0.56842 | 0.56189 | 0.75176 | fail directionality |
| A1 | `(0,0,2,0,0,-2)` | 0.51816 | 0.63484 | 1.24329 | pass |
| A2 | `(0,0,0,2,-2,0)` | 0.52825 | 0.64981 | 1.23025 | pass |
| A3 | `(0,0,2,-2,0,2)` | 0.41841 | 0.51804 | 0.98085 | pass |

All six cells passed activity, historical identification, forward improvement, and
bit-exact one-step replay. Both broadband-identity and constant-low-texture controls
passed. The three translations look strong against future persistence, but the same
forecasts also approach the previous frame. The failure is therefore the exact
nondirectional interpolation signature the reversal control was designed to catch.
No aggregate or secondary metric may rescue those three declared coverage cells.

The independent reviewer additionally:

- regenerated all 18 previous/current/future input hashes using a separate point-loop
  bilinear sampler;
- recomputed every stored ratio, derivable predicate, failure code, decision, config
  digest, evidence digest, source hash, and protocol hash;
- traced file opens and found no LCV, MM-007, or other scientific dataset access; and
- ran a clean isolated full regeneration that was bit-exact to the immutable receipt.

## Artifact bindings

```text
protocol SHA-256                 07a29f4d4f22dc619ef3f28250c81dabbee8dab114a701326483bb4d47f144cd
audit file SHA-256               75d58254c2ef4add7e8c65d52b86df32631bc8775524381a78c2e598a72baca2
audit evidence SHA-256           adaf0c4221aa0f99073c337ee9376366ea64d0189a84e4eb4f6798fcbbc9a3c8
semantic receipt SHA-256         80e8a77499980971f5783742a2a46ed926956c808ba71ff8684b980d081f11b1
audit implementation SHA-256    b885828045303d37438242e1f2d44f66a03ce04557ba327cd1967ce119a9e9a7
```

Both receipts are canonical JSON at mode `0444`. The semantic receipt binds the audit
file record and records `bit_exact_replay=true`. The authoritative files are:

- `docs/research/2026-07-16-mm011-offgrid-sensitivity-audit.json`
- `docs/research/2026-07-16-mm011-offgrid-sensitivity-semantic-verification.json`

## Executed checks

- Frozen audit: six positive cells, two null controls, exact source-pair deep replay,
  one-thread NumPy/OpenBLAS. Result: terminal pre-real NO-GO.
- Clean-process semantic regeneration: bit-exact decision, records, failure codes,
  and evidence digest.
- Independent point-loop sampler, arithmetic reconstruction, source/protocol binding,
  and file-open trace: `ACCEPT` the bounded decision.
- Complete MM-011 focused regression suite: `140 passed in 158.26s`.
- Ruff over the MM-011 package/tests: pass.
- Repository-config mypy over all 21 MM-011 package modules: pass.

The first isolated launcher attempt lacked the repository `src/` path and failed
before importing the audit, drawing a seed, or creating an output. The frozen command
was then rerun with only that import path added; no scientific setting changed.

## Limitations and retained defects

These do not reverse the negative decision, but they constrain reuse:

1. The receipt does not contain enough primitives to reconstruct
   `forecast_replay_bit_exact` or the historical half of the identity control without
   semantic regeneration. The clean full replay supplies that check.
2. `validate_receipt` checks schema/config/digest integrity but does not itself replay
   predicate arithmetic or the decision. The independent audit did both.
3. The recorded source closure omits executed package initializers, including
   `bench/__init__.py`, and therefore is not a complete sealed execution closure.
4. The generic continuous sampler would index one past the array if a valid sample
   landed exactly on its last source index. None of the six declared fixtures comes
   close to that boundary; separate regeneration matched all hashes.

The broader unexecuted MM-011 harness is also `NO-GO` for a real run. Independent code
review found that formal and semantic cleanup were ephemeral return values rather than
canonical facts, cleanup units were not cross-linked to the actual-child runtime,
and the controller imported broad live repository/bytecode roots before source
validation. The copied-LCV live-root dependency was removed and short-timeout
supervision was corrected during construction, but the dead finite-grid candidate is
not being repaired into a real run.

## Next falsifiable experiment

Use a fresh identity for a generated **continuous-registration directionality** assay.
It should keep the same persistence, historical, forward, reversal, identity, and
low-texture predicates; add fresh development and untouched confirmation seeds; and
compare a continuous/subpixel translation estimate plus continuous affine refinement
against this exact coarse-grid candidate. Success requires every declared cell to
identify history, improve forward, and reject the previous-as-future reversal. Failure
must distinguish optimizer failure, non-identifiability, and nondirectional smoothing.

Only after that generated successor passes should a real LCV-backed assay be rebuilt.
That harness must execute a copied bytecode-free controller closure and write formal
and semantic cleanup into separate durable, unit-cross-linked canonical packages.
AIDE2 remains premature because the eight real futures are not an adaptive evaluator.
