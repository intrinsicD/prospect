# WM-002 Q1 prospective implementation review

Status: **Superseded in part. This self-review missed three blocking defects
that a non-author review then found at commit `52744be`; all three are fixed
below. Q1 authorization remains refused, and a fresh non-author review over the
changed implementation digest is required.**

The independent reviewer's evidence is
[`ara/evidence/wm002-q1-independent-review-2026-07-25.md`](../ara/evidence/wm002-q1-independent-review-2026-07-25.md).
Its findings, and what they say about the value of self-review, are recorded in
"What the independent review caught" below.

| Field | Value |
|---|---|
| Reviewer | Claude Opus 5, at the maintainer's request |
| Method | Adversarial result-free selected-source review with executable probes |
| Assurance boundary | Local procedural review without external signature, and without reviewer independence |
| Reviewed sources | 48 selected-source manifest members |
| Implementation digest at review | `5b03166f433b3ebd49eaf4f6feaf48d1a888db99a4ac72058ed63fe524041041` |
| Protocol digest at review | `fd1f3f9f22557102b5b8220858341034b25491bfb1a2801bfbdba714f429f8dc` (unauthorized) |
| Q1 environment interactions | 0 |
| Q1 private draws | 0 |

## Independence limitation

The entry gate checks a prospective review's schema, protocol and implementation
digests, method, assurance boundary, scope, source count, result-free counters,
and authorization flags. **It cannot check that the reviewer is independent of
the author.** For this review they are the same agent, so every finding below
carries the bias of self-review: the failure mode is not a wrong claim about
what the code does, it is a blind spot about what the code should have done.

Two mitigations exist and neither substitutes for a second reviewer. First,
every claim below was checked by running something — a forged artifact, a
mutated runtime, an independent recomputation — rather than by reading alone.
Second, the entry gate now refuses a review carrying the machine-generated
rehearsal mark (finding 2), so at least the harness's own review cannot be
mistaken for this one.

**Recommendation: obtain a second review from an agent or person who did not
write this code before enabling `execution_authorized`.**

## Findings

### 1. Fixed — the rehearsal helper chmod-ed a symlink target

`rehearsal._mkdir_exact` used `Path.mkdir(exist_ok=True)` followed by a
symlink-following `chmod` and `stat`. Pointing `root/execution` at another
directory caused that directory to be chmod-ed to `0700` before anything
refused it. The entry gate's resource preflight did refuse the run, so no
rehearsal proceeded, but the side effect landed first. The helper now rejects
non-directories and symlinks by `lstat`, requires current-user ownership, and
applies the mode through an `O_NOFOLLOW` directory descriptor whose identity it
reconfirms. Rehearsal-only; production always used the descriptor-checked
`_mkdir_private_exact`.

### 2. Fixed — the entry gate could not distinguish a machine-generated review

The rehearsal harness generates a schema-valid, passing prospective review. The
`reviewer` field is free text with `minLength: 1`, and every other check —
digests, scope, counters, flags — passes identically for a machine-generated
review and an independent one. Pointing the generator at an authorized protocol
digest would therefore have produced a review the production entry gate
accepted. The rule is now explicit in both directions: production refuses a
review carrying `machine-generated, rehearsal-only`, and a rehearsal refuses a
review without it, so a rehearsal can never consume the independent review
written for Q1. This stops accidental misuse. It does not stop a determined
operator, who is already inside the trust model.

### 3. Boundary — the killing test is underpowered by construction

The inferential unit is the master index and there are four of them, so every
paired interval has three degrees of freedom and `t = 3.1824`. The arithmetic
is correct: master-grain differences, sample standard deviation, and
`t·s/√4` reproduce hand computation exactly. The consequence is that a *passing*
K4 is meaningful — five conjunctive lower bounds strictly above zero is a strong
requirement, and the conjunction of level-α tests is itself level α — while a
*failing* K4 is weak evidence of absence rather than evidence of no effect. The
protocol's abandonment rule must be read accordingly: a null result kills this
formulation, but does not establish that active acquisition fails.

### 4. Boundary — source binding does not cover parent-process patching

`_validate_selected_module_origins` binds file bytes and module origins, not
runtime state: a parent process that monkeypatched a bound module before calling
`run_q1` would pass. The exposure is limited by architecture rather than by that
check. Producers and restorers are fresh `python -S` children spawned from bound
source; the parent only orchestrates, merges, and aggregates; the producer
aggregate is explicitly non-authoritative; and the independent auditor
recomputes every statistic from primitive rows in a separate process. A patched
parent can therefore corrupt artifacts but cannot fabricate a passing audit —
verified in the forgery probe below.

### 5. Boundary — public traces are correlated with the hidden sign by design

Privacy here means no private field, HMAC preimage, or schedule position in
public artifacts, all of which is scanned and enforced. It does not mean the
public rows are uninformative about `theta`: with `q = 0.9` the observed symbol
is strongly correlated with the hidden sign, because that correlation *is* the
experiment. Nothing is wrong here; the contract should just not be read as
statistical secrecy.

### 6. Boundary — reproducibility of accepted evidence needs a standing check

The accepted Q0 report silently stopped reproducing when two bound sources
changed, and surfaced only when an unrelated code path regenerated it. That is
now covered by a regeneration test for Q0. No equivalent standing check exists
for the *future* Q1 report, whose reproducibility will depend on preserved
artifacts rather than regeneration, since Q1 is a one-shot attempt.

## What was verified, and how

Each item was executed, not read.

| Scope | Check | Result |
|---|---|---|
| Q0 and successor authority | Regenerate Q0 and compare report, protocol, and manifest digests | Reproduces; drift now fails a test |
| Runtime semantics and transactional causality | Every core `CandidateAssessment` across all 280 rehearsal rows equals the exact Q0 return for its action | 0 untruthful rows |
| Runtime semantics and transactional causality | Arm objectives are unit-labelled and never relabelled as return: entropy and EIG arms score in `nats`, uniform random has no scalar score | Holds for all seven arms |
| Runtime semantics and transactional causality | The shuffled control executes `weak` but learns from `weak`'s true likelihood, not the `overpowered` likelihood it was shown | Posteriors are exactly `7/10` or `3/10` |
| Runtime semantics and transactional causality | Every posterior equals exact Bayes under the true executed likelihood | 56/56 rows |
| Private seed exactness and noninterference | Five injected leaks must be detected by the causal probe | All five detected |
| Private seed exactness and noninterference | Uniform-arm selection replays from public SHA-256 alone, without the salt | 0 mismatches |
| Checkpoint and fresh-process restore | Restore recomputes the terminal decision from restored posterior and the private schedule, then parity-checks 12 identity and outcome fields against the live episode | Recomputed, not copied |
| Artifact schemas, attempt integrity, resources | A second attempt claimed against a completed registry | Refused, experiment-global |
| Artifact schemas, attempt integrity, resources | Production aggregate still satisfies the frozen 28,672-episode schema after the execution-mode refactor, and a short lane is refused | Both hold |
| Independent auditor recomputation and scale | Forge a coherent producer lie: inflate one `episode_return` and re-sign the attempt marker | Caught twice — K1 primitive reconstruction and K5 restored parity |
| Independent auditor recomputation and scale | Paired interval arithmetic against hand computation | Exact match |
| Evidence and claim boundary | Rehearsal artifacts submitted to the independent auditor | Rejected at K0 on `execution_authorized`, plus K1, K3, K4, K5 |

## What the independent review caught

A non-author review of commit `52744be` returned three blocking findings this
self-review missed, plus one defense-in-depth gap. All four are now fixed.

### B1 — the auditor's frozen normalized-protocol digest was stale

`q1_audit.py` still held `14d22a05…` after the Q0 rebinding moved the value to
`e015c8b5…`. The auditor runs *after* the attempt, so an authorized run would
have passed the execution boundary, consumed the sole one-shot attempt, and
then deterministically failed Q1-K0 on a constant. This is the worst possible
failure shape: unrecoverable, and invisible until it is too late.

**How the self-review missed it.** When rebinding Q0 I grepped for
`NORMALIZED_PROTOCOL_SHA256`. The auditor's copy is named
`_NORMALIZED_Q1_PROTOCOL_SHA256` — the `Q1` sits in a different position, so the
pattern did not match, and I read the empty result as "only one copy exists."
The duplication is deliberate auditor independence, so the fix keeps both copies
and adds a test asserting they agree with each other *and* with a live
recomputation.

### B2 — the auditor's real entrypoint failed its own selected-source closure

`_validate_loaded_source_origins` demanded `bench.active_acquisition.q1_audit`
in `sys.modules`, but `python -S -m bench.active_acquisition.q1_audit` runs it
as `__main__`. The producer already handled this through `__main__.__spec__`;
the auditor did not. It now binds the running entrypoint by origin.

**How the self-review missed it.** I ran the auditor as a subprocess and *saw*
this violation in the output. I attributed it to the rehearsal context and moved
on, because my test asserted only that the rehearsal was rejected. An expected
failure had become a place for unrelated failures to hide. The test now asserts
that no entrypoint, source-origin, or interpreter violation appears among the
rejections.

### B3 — published-artifact custody was under-enforced

The protocol requires all six artifacts at exact mode 0600; the auditor enforced
it only for `private-audit.jsonl`, in both the directory check and the hashing
path. A disposable `aggregate.json` at 0644 was accepted. Both paths now enforce
0600 for all six.

**How the self-review missed it.** The unit-test helper `_make_exact_publication`
built five artifacts at 0644 and called the result an exact publication. I read
the test suite as a specification of intended behavior instead of checking it
against the protocol text, so a fixture encoding the defect read as
confirmation. The fixture is corrected.

### B4 — reviewer-mark parity

The entry gate refused the machine-generated rehearsal reviewer mark; the
auditor's review loader accepted it. I added the check in one place and did not
mirror it in the independent recomputation. Now both refuse it.

### What this says about self-review

The self-review below executed a forgery attack, five leak injections, and
several independent recomputations, and every one of those probes was sound.
It still missed three blocking defects, and the pattern is consistent: each
miss came from trusting an artifact of my own construction — my grep pattern, my
test's assertion, my test's fixture. Adversarial effort does not substitute for
a reviewer who does not share the author's mental model. **Do not treat a future
self-review as satisfying the independent-review requirement.**

## Disposition

**Q1 authorization refused at commit `52744be`.** Three blocking defects and one
gap from the independent review are fixed, along with the two defects found
here; six boundaries and misses are recorded. The implementation digest has
changed again, so the canonical review artifact, entry qualification, and run
identity must be regenerated, and a **fresh non-author review over the new
digest is required before authorization**.

This review does not authorize Q1. It does not establish that Prospect uses
information value well, that any agent learned anything, or that any capability
exists.
