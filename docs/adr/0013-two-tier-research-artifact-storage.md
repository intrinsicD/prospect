# ADR-0013 — Two-tier storage for research artifacts

**Status:** Accepted

## Context

Research runs can produce two very different classes of files. Authored source,
tests, tasks, ADRs, protocols, audit narratives, and small checksum pointers benefit
from ordinary Git review and history. Generated results, evidence records, fixtures,
copied runtimes, raw media, tensor bundles, and packaged data do not: they make clones
and pushes expensive, mix mutable outputs with source review, and can require custody
metadata that Git does not preserve.

MM-009 exposed the mismatch. Its complete terminal tree has 2,261 files and
106,800,773 logical file bytes, mostly a copied NumPy/BLAS and Python runtime. The Git
projection omitted 413 ignored files and normalized executable/read-only modes, so it
was both large and insufficient as the authoritative custody copy.

## Decision

Use two storage tiers:

- Git stores authored source, tests, tasks, ADRs, protocols, audit narratives, license
  notices, and external-artifact pointers only.
- Every generated `bench/**/results/` tree and the complete `ara/evidence/` layer stay
  outside Git regardless of size or text/binary format. Binary and packaged-data
  extensions are ignored repository-wide as a second guard.
- External artifacts may be kept in local or remote artifact storage. Every published
  archive is pinned by byte length and SHA-256; sealed trees additionally pin
  path/type/mode/content metadata with a deterministic full-tree digest and structural
  counts.
- A pointer records the release repository, tag, asset, canonical materialization
  path, integrity values, formal-record bindings, and the claim boundary. The URL
  alone is never authority.
- Pointers live under `artifact-pointers/`, never inside ignored result roots.
  Materializing an archive must verify it in a temporary directory, move aside and
  re-verify any existing destination, and restore that destination on a caught
  installation failure.
- Repository ignore rules cover all generated result/evidence roots and binary or
  packaged data. Authored benchmark implementation and protocol documentation remain
  reviewable source.
- External archives that redistribute third-party code or binaries ship the applicable
  notices as checksum-bound release material and reference them from the pointer.

## Consequences

- Git remains source-only, reviewable, and pushable without discarding locally or
  externally archived evidence needed for custody.
- A fresh checkout contains no generated evidence projection. The pinned archive
  reconstructs the full tree, including modes Git cannot represent.
- Tests that require archived evidence must materialize it explicitly and belong to an
  artifact-aware validation lane; a portable source checkout must not claim that
  omitted sealed replay has run.
- Release assets must remain available for convenient reconstruction. If an asset is
  moved or replaced, its pinned checksum detects the change, and a new pointer/version
  is required rather than silently changing the evidence.
- Publication now includes archive creation, independent extraction verification,
  release upload verification, and pointer validation.
