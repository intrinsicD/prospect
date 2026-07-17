# MM-009 — Causal deformation/appearance prediction

- **Status:** terminal-invalid
- **Phase:** non-gated real-multimodal diagnostic
- **Requirements:** R1, R4, R6
- **ADRs:** ADR-0001, ADR-0002, ADR-0009
- **Depends on:** sealed MM-007; MM-008 v2.2 development-validated scientific core
- **Phase gate:** experiment-owned frozen gate; no `bench/SHIPPED` change

## Goal

Determine whether smooth affine deformation, global channel appearance, or their
combination can make a genuinely causal half-second prediction on the existing eight
Perception Test videos. Fit only `previous -> current`, freeze the resulting prediction,
and expose `future` only to the scorer.

## Non-goals

- No target-aware parameter fitting, population claim, end-to-end Prospect capability
  claim, simultaneous multimodal fusion, action recovery, planning, or control.
- No edits to `src/prospect/`, MM-001 through MM-007, their sealed result packages, or
  the MM-008 v2.2 scientific core.
- No unchanged MM-001 retraining. A GO only selects a new MM-001/TAESD successor
  direction; that successor still requires its own task, protocol, implementation
  binding, pre-real audit, and empty output root.
- No full MM-008 sealed-runtime lifecycle. The user authorized superseding its untouched
  real-target path in favor of this causal experiment.

## Interface to satisfy

Experiment-owned typed functions under `bench/multimodal_causal_diagnostics/`:

- source-only row construction and fold normalizers;
- exact-global causal fit and application for `affine`, `appearance`, and `combined`;
- prediction-bundle freezing before target scoring;
- pure score aggregation, decision ladder, package verification, and semantic replay.

The benchmark harness remains task-specific and imports the pinned MM-008 v2.2 fit-only
core as a scientific dependency. No production protocol is added.

## Approach

Reuse MM-007's authenticated 477 R64 frames to reconstruct MM-005's exact 453
`previous,current,future` triplets and whole-video folds. Replay the current-only R8 fold
normalizer already frozen by MM-007. Before the formal marker, inspect/copy the pinned
MM-007 tree opaquely without parsing it. After the marker and formal synthetic gate,
run its existing fast verifier before any decode or row construction. Detach every row:
a fresh sandboxed predictor sees only that row's normalized `previous,current` and
preregistered source controls; its separate future row is not mounted. For each arm,
fit only `previous -> current`, then apply the selected transform once to `current`.
Compare the frozen prediction with future persistence only after all 453 supervisor
prediction commits are immutable. Retain checkerboard history reconstruction,
far-half-cycle derangement, temporal reversal, matched spatial-structure-erased
current-only bias, velocity, target-specificity, range, and synthetic controls.

## Acceptance criteria

- [ ] A protocol with exact parent/source/config hashes, causal transform semantics,
  controls, thresholds, decision branches, and abandonment criteria is frozen before
  any MM-009 real future is scored.
- [ ] Pre-marker preparation performs only exact opaque MM-007 census/hash/mode/copy.
  The census admits exactly the eight pinned files plus the required real `inputs/`
  lineage directory, which is non-authoritative, untraversed, unread, and uncopied;
  every other extra entry and every symlink fails closed. The existing MM-007 verifier
  runs after the formal marker and synthetic gate but before parent parsing or real
  row construction.
- [ ] Source-only prediction has no target argument or target import/path authority;
  mutating the scoring target leaves every parameter and prediction bit-exact.
- [ ] Translation, affine, appearance, and combined repeated-operator positives pass
  complete history/future/directional predicates at seeds `990900..990902`;
  stationary, affine/appearance reversal, paired independent-future, constant-target,
  coupled-boundary, channel-permutation, and forgery controls take their exact frozen
  branches at the registered seed map.
- [ ] Exact MM-007 parent verification and exact 453-row MM-005 identity/count parity pass.
- [ ] Prediction artifacts freeze before the scorer reads the future-target artifact.
- [ ] The fitting-free post-freeze gate validates, for every row, exact
  `PCG64(991000+ordinal)` random, two-axis reverse, preregistered deranged-future,
  `[0,8,8]` uint64-LSB, and rejected-NaN mutations while the 1,360 prediction-side
  files and all detached target files remain unchanged.
- [ ] An arm licenses the next end-to-end bridge only if all integrity controls pass,
  historical identifiability and future improvement hold jointly on at least 6/8
  videos, the same-video predicate `a<i and a<u and c<p and c<b` reaches at least 7/8,
  every fold is represented, activity holds on 8/8, and all
  shuffled/reversal/derangement null counts stay below the frozen preemption boundary.
- [ ] Any incomplete/invalid family, any null at 6/8, or any family-level
  inconclusive condition globally preempts GO even if a sibling family passes.
- [ ] Each child emits only temporary `predictions.npy` and `prediction.json`; the
  supervisor validates/copies them, writes the separate canonical predecessor-chained
  commit, and only then freezes the 453-row bundle.
- [ ] One fresh fitting-free `-I -S -B` score-only process produces each row's primitive
  score record after the future-isolation gate.
- [ ] Score and future-isolation custody launchers enter Landlock/seccomp; raw-open and
  manual-loader probes cannot reach the live repository, fitting sources, or sibling
  score targets, and the future gate commits through a fresh private output directory.
- [ ] The sandbox uses immutable pre-audit-bound NumPy and site-package-free stdlib
  copies plus a fitting-free custody runtime; after the recorded host bootstrap, no live
  Python package root is readable, and AF_INET/AF_INET6/AF_UNIX STREAM and DGRAM
  variants are denied.
- [ ] On first child failure, the shared supervisor terminates every registered process
  group. The complete formal and semantic lifecycles run in unique transient cgroup-v2
  user-systemd services with `RuntimeMaxSec=14400s` and `KillMode=control-group`; nested
  new sessions cannot survive normal exit, timeout, or client interruption, and every
  unit is confirmed inactive.
- [ ] Before future isolation or any scorer opens a target, immutable
  `pre-score-budget.json` authenticates the current package and proves that current
  bytes plus conservative per-file reserves remain within 2,000,000,000 bytes. Future
  and row-score children have hard 1 MB `RLIMIT_FSIZE` limits; fast verification
  replays the projection and all declared file bounds.
- [ ] Execution remains within 8 concurrent workers, 900 seconds per predictor/replay,
  14,400 seconds total, and 2,000,000,000 artifact bytes.
- [ ] The one-shot result is independently audited and semantically regenerated before
  any branch is acted upon.
- [ ] The durable formal synthetic record is semantically validated before parent
  access, and semantic verification reruns and bit-compares the complete panel.
- [ ] On GO, propose a new TAESD/MM-001 successor; on clean NO-GO, propose MM-010
  source-only coverage/analog diagnosis. Execute neither until its new task, protocol,
  implementation/config binding, synthetic gate, and independent pre-real audit GO
  exist. On invalid/inconclusive, do neither.
- [ ] Focused pytest, Ruff, and strict mypy pass; affected repository regressions remain
  green.

## Test plan

- Unit tests for row identities, normalizers, affine/appearance application, scoring
  arithmetic, branch precedence, canonical serialization, and fail-closed validation.
- Exact registered synthetic panel: translation/affine/appearance/combined positives;
  stationary, affine/appearance reversal, paired independent future, constant target,
  coupled boundary, and channel permutation. Acceleration and periodic ambiguity are
  not claimed as formal MM-009 controls; inherited v2.2 tests retain their narrower
  exact-grid/tie coverage.
- Mutation tests for future-target independence, detached-row authority, prediction
  bytes, parameters, hashes, row identity/order, source/target membership, and result
  decision fields.
- Clean-process prediction replay, fitting-free per-row scoring, and post-target
  semantic regeneration.
- Exact opaque MM-007 pin/copy validation during preparation; exact MM-007 fast
  verification only after the marker/synthetic gate, then again during package
  verification as permitted by the formal lifecycle.

## Docs-sync checklist

- [ ] Task status, result classification, and permitted successor direction recorded
  here and in `tasks/BACKLOG.md`.
- [ ] The immutable MM-009 supersession record and this protocol record that MM-008
  v2.2's untouched real-target route was retired before reserved/challenge/real use;
  do not edit the MM-008 protocol bytes whose SHA-256 is a pinned MM-009 dependency.
- [ ] Evidence and ARA records retain the development/formal/capability boundaries.
- [ ] No roadmap, requirement, ADR, README, or shipped-gate claim is promoted without a
  separate audited end-to-end result.

## Gate result

`invalid_MM009` — the immutable formal attempt stopped in the first
`_post_marker_inputs` operation, before parent parsing, row detachment, prediction,
future isolation, or scoring. Recursive MM-007 verification reached MM-003 and
re-fitted four PCA transforms under a different OpenBLAS thread/runtime regime; the
numerically equivalent PCA subspaces changed byte hashes and failed the historical
exact-equality verifier. The cgroup helper also resolved the reviewed venv interpreter
path to the base interpreter, splitting audited NumPy 2.4.6 from formal NumPy 2.1.3.

There is no real MM-009 result. The canonical output is terminal and cannot be retried
or repaired. The TAESD/MM-001 GO branch and MM-010 clean-NO-GO branch are prohibited.
The authorized continuation is LCV-001, a new-identity sealed-lineage/runtime closure
gate. See
`docs/research/2026-07-16-mm009-terminal-parent-verifier-failure.md`.
