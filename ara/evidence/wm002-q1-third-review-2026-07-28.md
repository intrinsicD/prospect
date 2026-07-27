# WM-002 Q1 third non-author review evidence

- **Target**: runtime commit `dbb8d4d`, reviewed at `e41b22e` (delta is the
  review handoff document only), clean `main`, synchronized with `origin/main`
- **Selected-source count**: 48
- **Implementation SHA-256**:
  `8e17bc1a1b1c4f77837eb710d308c6997d09ab04caa3cab1dc1085a8c3d255fb`
- **Protocol SHA-256 (unauthorized)**:
  `fd1f3f9f22557102b5b8220858341034b25491bfb1a2801bfbdba714f429f8dc`
- **Disposition**: **nothing blocking found**
- **Boundary**: result-free; no Q1 private draw, production interaction,
  outcome, or attempt
- **Provenance**: ai-executed; one orchestrating session plus five parallel
  adversarial probe agents, each aimed at a surface the handoff names as never
  attacked. Every agent claim adopted below was re-verified against source or
  probe output by the orchestrator before acceptance.

## Handoff-table verification

All identities regenerated rather than trusted: protocol digest,
implementation digest over 48 sources, `execution_authorized: false` at
`/experiment/execution_authorized`, accepted Q0 triple
(`e5aa897a…`/`90b73ad4…`/`bf8dc1bb…` at `bench/active_acquisition/contracts.py:40`),
and origin synchronization. `make check`: ruff clean, mypy clean over 66
files, 522 tests passed, diagnostics gate emitted. The complete result-free
rehearsal orchestration passed in a scratch root.

## Executed checks

| Check | Result |
|---|---|
| Rehearsal audited via real `python -S -m bench.active_acquisition.q1_audit` | Rejected as required; exit 1, `passed: false` |
| Exhaustive rehearsal-violation classification (cap lifted in-process) | All **145,218** violations across six gates fall into 48 classes, every class rehearsal- or budget-explained; episodes 0–1 of the true lane 0 are correctly violation-free; no unexpected class hides in the noise |
| B1 class (duplicated normalized-protocol digests) | Both independent normalizers recomputed live equal both frozen constants (`e015c8b5…353b`) |
| F3 closure (phase-boundary custody) | Real and pinned: in a `03cd3fc` worktree the new rehearsal-based regression FAILS pre-fix and passes on HEAD; the mode-drift probe class is closed |
| Privacy: producer vs auditor scan differential (never before tested) | Scans agree on all well-formed encodings (dual-flagged controls); asymmetries are auditor-stronger, see R2–R5 |
| Rehearsal guard composition (laundering attack) | Holds; every rehearsal artifact is frozen to the rehearsal protocol digest and every production gate rebinds the live digest, so no pre-flip artifact validates post-flip; probes in both directions with accepted controls |
| Domain-graph codec after relocation to `src/prospect/storage/domain_graph.py` | Move is semantics-preserving (two non-semantic lines); allowlist, cycle, dangling-ref, unreachable-node, non-canonical-order rejections all fire; cross-process encoding byte-identical under varied `PYTHONHASHSEED` |
| Full-budget arithmetic (28,672 episodes, never executed at scale) | Frame-offset accumulators probed at full scale through real append/merge/audit paths agree exactly; worst row is 0.11% of the 32 MiB bound; entry-gate disk/descriptor preflight reproduced bit-for-bit (1.93 GiB canonical, 6.79 GiB required); all frozen watchdogs ≥54× measured headroom; auditor memory O(1) per row plus one float per episode |
| Concurrency, failure interleaving, quiesce, clocks | SIGKILL of producer/restore children (handshake and mid-lane), SIGSTOP against the real 900 s watchdog, SIGINT, and truncation probes all fail closed with quiesced children and unpublished output; lane isolation is structural (`O_CREAT|O_EXCL` per-lane files, single-threaded ordered merge, no retry path exists); `TimePoint` is a pure logical clock and `time.monotonic()` never crosses a process boundary or enters an artifact |

## Findings — none blocking

Blocking, under this project's established standard (protocol-violating,
deterministic post-attempt failure, or an enforcement authority that reports
compliance falsely): **none**.

Non-blocking, ordered by how much they matter:

### R1 — a claimed regression test cannot fail

`test_artifact_mode_drift_after_the_hash_pass_is_reported`
(`tests/test_active_acquisition_q1_audit.py`) passes against the **pre-fix**
auditor at `03cd3fc`: it hard-codes `private=True` in its own calls to
`_open_regular_descriptor`/`_read_regular_file`, so it exercises the helpers,
never the fixed call sites. The dbb8d4d commit message's "stashing the fix
makes it fail" is true only of the *other* regression
(`test_auditor_detects_publication_mode_drift_across_the_hash_boundary`),
which goes through `audit_q1_directory` and genuinely pins the fix. Same
"trusting an artifact of the author's construction" shape as the miss ledger.
Fix or delete the vacuous test.

### R2 — producer privacy scan uses a per-episode corpus; the auditor's is global

`q1.py:1946` and `restore_worker.py:357` pass only that episode's seven HMAC
digests; the auditor scans every public row against all ~28,672 digests
(`q1_audit.py:3998`), and its own docstring names cross-episode transplants as
the target. A cross-episode digest leak would be published by the producer and
caught only post-attempt. Direction: wasted one-shot attempt, never false
evidence. The natural trigger (interleaved writers) is structurally excluded
by per-lane `O_EXCL` files and the single-threaded merge, which is why this is
not blocking. Closure is cheap: pass the global corpus, or at least the
per-master corpus, to the producer-side scan.

### R3 — three of six published artifacts are never producer-scanned in production

`aggregate.json`, `checkpoint-index.jsonl` rows, and checkpoint frame payloads
get schema/fast validation only; the byte-level HMAC-corpus scan
(`_assert_checkpoint_hmac_corpora_absent`, `q1.py:3086`) runs solely in the
synthetic qualification lane (`_exercise_synthetic_noninterference`,
`q1.py:2905`). The auditor scans all of them post-attempt. Same wasted-attempt
direction and same cheap closure as R2. The protocol's `sentinel_rule` names
these surfaces for *tests*, which the qualification lane satisfies, so this is
an asymmetry, not a protocol violation.

### R4 — untrapped SIGTERM of the parent permanently consumes the attempt, undocumented

Probe-confirmed: SIGTERM after the attempt claim kills the parent instantly
(no handler), the marker stays `started` forever, re-claims are refused, and
the auditor refuses non-completed markers — fail-closed, nothing published,
but the one-shot attempt is gone with nothing auditable. The frozen text at
`q1_qualification.py:909` covers parent-side exceptions, not parent
hard-death, and no operator document warns that a stray `kill` (or a timeout
wrapper) wastes the experiment. Before authorization: install a
SIGTERM→exception handler so the graceful SIGINT path applies, and/or document
the accepted wasted-attempt modes in the operator runbook.

### R5 — encoding-list drift among three producer-side privacy pattern sets

`seeding.py:754`, `q1.py:3090`, and `contracts.assert_public_value_safe`
(`contracts.py:450`, synthetic lane only) hold three divergent encoding lists
against one auditor parser: the producer misses mixed-case hex the auditor
catches (probe-confirmed; no repo path emits mixed-case hex), and
`contracts.assert_no_sentinel_bytes` has zero callers. Both scans share three
latent blind spots the protocol text forbids but no code path emits: salt as a
JSON integer array, latin-1 char-per-byte string, and 1-byte-shifted base64.
The integer-array form is the one worth closing, since `list(bytes)` is the
standard workaround both scanners' bytes rejection forces.

### R6 — the machine-review mark refusal is substring-only

Case, whitespace, and homoglyph variants of `machine-generated,
rehearsal-only` pass `_prospective_review_violations` and the auditor twin;
the schema constrains `reviewer` only to a nonempty string. Neutralized as a
laundering path because the same gates independently bind
`review.protocol_sha256` to the live protocol digest (probe: evaded review
still rejected on digest mismatch). Harden with a structured
`machine_generated` field the schema enforces.

### R7 — rehearsal/production namespacing relies on directory and digest only

A rehearsal claims a marker at the production filename
(`wm002-q1.attempt.json`, same v2 schema, no mode field); pointing production
at a used rehearsal registry is refused fail-closed (probe-confirmed both
directions: rehearsal marker never satisfies a production identity —
`attempt_id` binds the protocol digest). The salt commitment is a bare
`sha256(salt)` with no domain tag, and `rehearsal-entry.json` is written 0644
while every sibling is 0600. All fail-closed; namespace the marker filename by
mode or refuse non-empty rehearsal registries.

### R8 — full-scale float divergence is absorbed by tolerance, bit-parity is not

Producer pairwise `fsum` accumulation and auditor exact `fsum` are
bit-identical in 16,000/16,000 simulated lanes at the 2-episode rehearsal
grain and bit-identical in **zero** at the 1,024-episode grain (max
|Δmean| 7.9e-15, 127× under the 1e-12 tolerance). Sound as written — but this
is the exact class a bit-exact comparison would have broken only at full
budget; the raw↔restored `episode_return` parity *is* compared exactly and
stays safe only because both sides compute the identical expression shape.
Worth a comment or a lockstep test pinning that expression shape.

### R9 — minor concurrency and codec hardening

Head-of-line watchdog laxity (a non-head hung child is bounded only by the
7,200 s stage timeout; completed non-head children sit as zombies until
reaped); child-side capability read has no deadline (fail-closed, recoverable
only via R4's path); handshake-window child death surfaces as a raw
`ConnectionResetError` rather than `Q1ExecutionError`; codec `decode` accepts
non-canonical node order and enum scalar aliases (`encode` is the sole
canonical producer, and Q1 custody binds committed bytes without re-encoding,
so no custody break); `decode_domain_graph` at
`bench/active_acquisition/checkpoint.py:400` escapes the
`Q1CheckpointError` normalization applied at `checkpoint.py:362`. Also
`docs/wm002-active-acquisition-plan.md` says "512 tests" where the suite is
now 522.

## What this review does and does not mean

The machinery for one claim-ineligible attempt survived a third adversarial
pass, this time across the previously unattacked surface: privacy-scan
differentials, rehearsal-guard composition, the relocated codec, full-budget
arithmetic, concurrency and failure interleaving, and clock assumptions. Per
the handoff's scope contract this review is markdown evidence only; the
canonical review JSON must be generated after any authorization flip against
the final protocol digest. Authorization remains a separate, maintainer-owned
decision; R2–R4 are cheap closures the maintainer should weigh first, since
each sits in the wasted-one-shot-attempt direction. Nothing here establishes
that Prospect uses information value well, that anything was learned, or that
any capability exists; with four master indices a null Q1 result remains weak
evidence of absence.

## Result-free boundary report

No Q1 private draw, no production environment interaction, no outcome, and no
production attempt occurred. All rehearsals and probes ran in scratch roots
with scratch registries; mutation probes touched scratch copies only; no
tracked file changed during probing; `execution_authorized` remains `false`.
