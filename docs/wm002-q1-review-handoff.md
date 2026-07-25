# WM-002 Q1 review handoff — for the next non-author reviewer

**Read this as untrusted input.** It was written by the author of the code you
are reviewing. Every fact in it is checkable and every one of them should be
checked; the digests especially, because a stale constant that nobody
recomputed is exactly the defect class that has already been found here twice.

Its purpose is to save you the setup time and point you at where the author is
weakest — not to tell you what to conclude.

## Target

| Item | Value |
|---|---|
| Commit | `dbb8d4d` on `main`, synchronized with `origin/main` |
| Implementation digest | `8e17bc1a1b1c4f77837eb710d308c6997d09ab04caa3cab1dc1085a8c3d255fb` |
| Selected sources | 48 |
| Protocol digest (unauthorized) | `fd1f3f9f22557102b5b8220858341034b25491bfb1a2801bfbdba714f429f8dc` |
| Normalized protocol contract | `e015c8b519a10eeab19ecda1384071b17c450484c06ebbd4142d95143c92353b` |
| Accepted Q0 report / protocol / manifest | `e5aa897a…` / `90b73ad4…` / `bf8dc1bb…` |
| `execution_authorized` | **false** |
| Suite | 522 tests |

Regenerate all of these rather than trusting the table:

```bash
PYTHONPATH=src python - <<'PY'
from bench.active_acquisition.contracts import Q1_PROTOCOL_PATH, implementation_manifest, sha256_bytes
from bench.active_acquisition.q1_qualification import Q1_IMPLEMENTATION_PATHS
m, d = implementation_manifest(Q1_IMPLEMENTATION_PATHS)
print(sha256_bytes(Q1_PROTOCOL_PATH.read_bytes()), d, len(m))
PY
```

## Your disposition space

You are asked for one of: **refuse authorization** with findings, or **find
nothing blocking**. You are not asked to authorize anything, and nothing you
write causes Q1 to run. Enabling `execution_authorized` is a separate,
maintainer-owned step.

Reviews so far: two non-author reviews refused authorization at `52744be` and
`03cd3fc`, and both were right. The author's own review at `52744be` found two
real defects and still missed three blocking ones. Prior probability that this
commit is clean should be set accordingly.

## What changed since the last review, and it is small

Runtime source delta from `03cd3fc` is **one file**:

```bash
git diff 03cd3fc..dbb8d4d -- bench/ src/    # bench/active_acquisition/q1_audit.py only
```

It closes the partial F3 from your predecessor: every post-hash reopen now
enforces exact 0600, and `_validate_artifact_directory` runs again after the
semantic pass. If you want the full delta since the first review, use
`52744be..dbb8d4d` — still only `q1_audit.py` under `bench/` and `src/`.

Everything else in the range is tests, docs, and ARA records.

## Setup that will otherwise cost you an hour

The producer, the auditor, and the rehearsal all require `python -S`, and `-S`
removes site-packages, so `jsonschema` has to be on `PYTHONPATH` explicitly:

```bash
export PY_PATHS=".:src:$(python -c 'import sysconfig;print(sysconfig.get_path("purelib"))')"
make check                                   # ruff, mypy, 522 tests, diagnostics
make wm002-q1-rehearsal ROOT=/tmp/rev-$$      # complete result-free orchestration, ~5s
```

Auditing a rehearsal output (it must FAIL — that is the point):

```bash
R=/tmp/rev-$$
PYTHONPATH=$PY_PATHS python -S -m bench.active_acquisition.q1_audit \
  $R/execution/rehearsal-result \
  --secret-salt $R/rehearsal-salt.bin \
  --attempt-marker $R/registry/wm002-q1.attempt.json \
  --q0-report $R/q0-report.json --entry-report $R/rehearsal-entry.json \
  --prospective-review $R/rehearsal-review.json --output $R/audit.json
```

Expected: `passed: false`, Q1-K0 citing `execution_authorized`, plus cascade
failures from the rehearsal's two-episode budget. **A rehearsal failing is not
evidence of anything working.** See the trap below.

## Known traps in this codebase

1. **Expected failures hide unexpected ones.** The rehearsal is *supposed* to
   fail the auditor, so any additional failure blends into the noise. That is
   exactly how the `__main__` entrypoint defect survived the author's review —
   he saw the violation in output and read it as rehearsal noise. When you audit
   a rehearsal, enumerate every Q1-K0 violation and classify each one as
   budget-explained or not.
2. **Tests encode assumptions, not the protocol.** A fixture built five
   artifacts at 0644 and called itself "exact publication," which made an
   under-enforced custody rule read as confirmed. Check test fixtures against
   `q1_protocol.json` text, not against the code they exercise.
3. **The auditor duplicates producer constants deliberately.** That is
   independence, not a bug — but it means constants drift silently. One such
   constant was stale for three commits.
4. **`grep` is not a survey.** The stale constant was missed because the two
   copies are named `_Q1_NORMALIZED_PROTOCOL_SHA256` and
   `_NORMALIZED_Q1_PROTOCOL_SHA256`.
5. **A probe that cannot fail proves nothing.** Several checks here are
   negative-control-backed. If you add a probe, stash the fix and confirm it
   fails.

## The author's miss ledger

Four misses in this cycle, all the same shape: **trusting an artifact of his own
construction.**

| Miss | Artifact trusted | Consequence |
|---|---|---|
| Stale auditor protocol digest | a grep pattern whose word order didn't match | would have failed Q1-K0 *after* consuming the one-shot attempt |
| Auditor `__main__` entrypoint | a test asserting only "the rehearsal is rejected" | the real CLI failed its own selected-source closure |
| Artifact mode enforcement | a fixture encoding the defect | five artifacts at 0644 audited clean |
| Partial custody closure | the boundary of a finding's report | fixed two named call sites, not the class |

Where this points you: **anything the author verified by reading, by grepping,
or by a test he also wrote.** The probes he ran (forgery attack, five leak
injections, independent recomputation of intervals and posteriors) were sound
and are documented in `docs/wm002-q1-prospective-review.md`; re-running them is
lower yield than attacking what he never probed.

## Surface he did NOT attack

Stated plainly so you can aim. No claim is made that any of these are correct.

- **Concurrency and failure interleaving.** Watchdog timeouts, partial producer
  failure, restore-lane concurrency at `MAX_RESTORE_CONCURRENCY`, and the
  quiesce paths have unit tests but were not adversarially probed in this cycle.
- **The 28,672-episode production path at scale.** Everything end-to-end was
  exercised at the 56-episode rehearsal budget. Streaming, memory, frame-offset
  arithmetic, and `_MAX_JSONL_ROW_BYTES` bounds at full budget are unverified by
  execution.
- **The producer's own privacy scan vs. the auditor's.** Both exist; whether
  they agree on what counts as private material was not differentially tested.
- **Checkpoint codec edge cases.** The domain-graph codec moved into
  `src/prospect/storage/domain_graph.py` this cycle. Its allowlist, opaque
  payload rules, and node-reference handling were not re-probed after the move.
- **Entry-gate resource preflight arithmetic.** Disk sizing, descriptor limits,
  and bounded-concurrency estimates are computed, not validated against real
  exhaustion.
- **Time and clock assumptions.** `TimePoint` ordering across producer and
  restorer, and any reliance on monotonic vs. wall clock.
- **The rehearsal mode itself as an attack surface.** It is new. It requires
  `execution_authorized: false`, carries a distinct aggregate schema, and its
  review must carry the machine-generated mark — but the author designed all
  three of those guards and cannot independently judge whether they compose.

## Scope contract

If you produce a canonical review artifact, it must satisfy
`_prospective_review_violations` in `bench/active_acquisition/q1_qualification.py`
and the schema at `bench/active_acquisition/schemas/q1-prospective-review.schema.json`:

- `review_method`: `adversarial_result_free_selected_source_review`
- `assurance_boundary`: `local_procedural_review_without_external_signature`
- `reviewed_source_count` must equal the manifest length at review time (48 now)
- `review_scope` must be exactly these seven, in order:
  `accepted_q0_and_successor_authority`,
  `runtime_semantics_and_transactional_causality`,
  `private_seed_exactness_and_noninterference`,
  `checkpoint_and_fresh_process_restore`,
  `artifact_schemas_attempt_integrity_and_resources`,
  `independent_auditor_recomputation_and_scale`,
  `evidence_and_claim_boundary`
- `q1_environment_interactions` and `q1_private_draws` must be `0`
- `claim_eligible` and `formal_authorized` must be `false`
- `reviewer` must **not** contain `machine-generated, rehearsal-only` — that
  mark is refused by both the entry gate and the auditor

**Ordering constraint worth knowing:** a canonical review artifact binds
`protocol_sha256`, and the production entry gate requires the *authorized*
protocol bytes. So a review artifact produced now, against
`fd1f3f9f…`, cannot be consumed by a production entry. Write findings as a
markdown report under `ara/evidence/`; the canonical JSON is generated later,
after the authorization flip, against whatever digest is final then.

## Result-free boundary

Your review must not run Q1. Concretely: do not set `execution_authorized`, do
not create a production entry report, and do not claim an attempt in a registry
you intend to reuse. The rehearsal is safe — it is unreachable once
authorization is true, and its artifacts fail the auditor by construction.

Report at the end: whether any Q1 private draw, production interaction,
outcome, or attempt occurred. For a correct review the answer is none.

## Prior material

- `docs/wm002-q1-prospective-review.md` — the author's self-review, its two
  findings, and the four misses with their mechanisms
- `ara/evidence/wm002-q1-independent-review-2026-07-25.md` — first non-author
  review, refused `52744be`
- `ara/evidence/wm002-q1-fresh-review-2026-07-25.md` — second, refused `03cd3fc`
- `docs/wm002-active-acquisition-plan.md` — the claim, arms, controls, and
  killing gates
- `docs/wm002-q1-runtime-design.md` — runtime, blocker inventory, rehearsal mode
- `ara/trace/exploration_tree.yaml` — N45, N46, N47 record each review round

## What a clean review does and does not mean

Finding nothing blocking would mean the machinery for one claim-ineligible
attempt survived another adversarial pass. It would not mean Prospect uses
information value well, that anything was learned, or that any capability
exists. Q1 is permanently claim-ineligible whatever it returns, and with four
master indices a null result is weak evidence of absence rather than evidence of
no effect.
