# WM-002 Q1 prospective implementation review

Status: **Result-free review completed on 2026-07-25 with two findings fixed and
four boundaries recorded. It is NOT the independent review the entry gate
requires, because the reviewer wrote most of the reviewed code.**

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

## Disposition

No blocking finding. Two defects were found and fixed; four boundaries are
recorded above and belong in any reading of a future Q1 result. The
implementation digest changed with the fixes, so the canonical review artifact,
entry qualification, and run identity must all be regenerated against the final
bytes.

This review does not authorize Q1. It does not establish that Prospect uses
information value well, that any agent learned anything, or that any capability
exists. It states only that the machinery for a single claim-ineligible attempt
appears sound under adversarial probing by its own author.
