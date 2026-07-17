---
name: prospect-results-audit
description: Adversarial referee pass ("scientist pass") over Prospect's gate, experiment, capability, and causal-mechanism claims. Use before a quantitative or capability claim enters README.md, docs/roadmap.md, or ara/logic/claims.md; after any gate, formal experiment, or evidence session; before promoting a phase, task, default, or sealed result; when reviewing a results-bearing PR or commit; or whenever numbers lack an independent semantic verification. Do not use for generating new research directions (that is the prospect-research-ideation skill).
license: MIT
metadata:
  version: "1.0.0"
---

# Results Audit ("scientist pass")

> **Provenance.** Distilled 2026-07-15 from Prospect's own evidence history:
> the BC-001 read-only audit that strengthened artifact binding and narrowed an
> over-attributed localization claim; OL-001 and PI-001's terminal report-
> canonicalization failures; PI-002's tuple/list semantic-verifier failure; and
> the rule that an administrative rerun is the same scientific experiment, not
> an independent result. It also incorporates the sibling-repository referee
> pattern for correcting claims after apparently successful runs.

## Stance

Act as a referee, not the producing investigator. Assume the session made a
mistake in the protocol, artifact graph, arithmetic, control, or wording and
hunt for it. The deliverable is corrected evidence, narrower claims, and
truthful gate/task state, never reassurance. Report "confirmed" only after
filling the claim table and binding every row to independently checked evidence.

## When to run

- Before a quantitative, causal, generality, or capability claim enters
  `README.md`, `docs/roadmap.md`, or `ara/logic/claims.md`.
- After a phase gate, non-gated benchmark, formal research experiment, or ARA
  evidence session produces numbers.
- Before adding a phase to `bench/SHIPPED`, promoting a task/default, accepting
  a sealed package, or reviewing a results-bearing PR.
- On request: "audit", "referee pass", "scientist pass", "verify the claims".
- Periodically over old claims and parent packages, not only the newest result.

## Procedure

### 1. Inventory the claims

Sweep the changed `README.md`, `docs/roadmap.md`, `docs/requirements.md`, ADRs,
task files, `bench/results/`, experiment-local `bench/**/results/`,
`docs/research/`, `ara/logic/claims.md`, and `ara/evidence/`. Build:

| # | Claim (one sentence) | Scope/kind | Evidence package | Semantically verified now? |
|---|---|---|---|---|

Kind is one of: gated capability, non-gated mechanism, measured diagnostic, or
asserted. State scope explicitly: authored fixture versus foreign environment,
development versus held-out, replay versus fresh confirmation. An asserted row
or a claim whose scope is absent is already a finding.

### 2. Recompute from raw artifacts

Reconstruct decisions from result JSON/CSV/tensors, not from report prose. Run
the package's fast verifier and, where defined, its full semantic verifier that
re-trains or regenerates outcomes. Compare canonical serialized values rather
than raw Python container types. Verify `artifact-manifest.json`, input/source
hashes, nested parent receipts, model fingerprints, and copied protocols. A
final verifier failure invalidates the package even when its numbers look right.

### 3. Bind claims to executed checks

Gated claims bind to the exact `bench/gates.py` criterion, recorded seeds,
sentinels, and a fresh `make gate PHASE=Px`; shipped-capability claims also bind
to `make gate-all`. `make gate` may exit successfully while reporting BLOCKED,
so inspect the persisted `GateResult.passed` value and detail, not only shell
status. Formal non-gated claims bind to their package verifier,
targeted tests, and immutable raw artifacts. Record whether a gate evaluation
replayed a persisted artifact or actually retrained. Pending optional-dependency
or long-run checks remain pending; never restate them in the past tense.

### 4. Audit protocol and configuration parity

Confirm the executed phase/experiment ID, environment/task, dataset bytes,
model and evaluator seeds, train/development/confirmatory split, update and
planner budgets, thresholds, branch rule, parent identity, source snapshot, and
dependency versions match the claim. Replay/hypothesis-generating seeds must not
silently become confirmatory evidence. A formal defect receives a new identifier;
never repair or overwrite a sealed package in place.

### 5. Recompute accounting and statistics

Check equal environment steps, planner/model evaluations, tool or retrieval
calls, horizons, and wall-clock scope. Keep oracle-only diagnostic compute
separate from the learned planner budget. Recompute paired-seed deltas, gap
closure, success counts, ratios, intervals, and decision predicates from raw
rows. Do not double-count an administrative rerun (for example OL-001/OL-002)
as replication, or treat starts/calls from one model seed as independent seeds.

### 6. Audit controls and interpretation

Require the predeclared positive/negative controls and every applicable collapse
sentinel to pass before interpreting the primary endpoint. A diagnostic cannot
rescue a failed primary gate. Distinguish "failed to establish" from "proved
absent", mechanism evidence from production capability, a real-media preflight
from multimodal learning, and supplementary DMC evidence from the numpy gate
ratchet. Check that poisoned/shuffled/permuted controls change only the intended
factor and preserve budgets and denominators.

### 7. State environment and artifact limits

List exactly what ran in this session: tests, gate evaluation, fast verification,
semantic regeneration, optional DMC/multimodal workloads, or only artifact
inspection. State unavailable extras and unopened confirmatory data. A stored
report plus green unit tests is not semantic regeneration; a non-gated result is
not a shipped gate.

### 8. Dispose of every claim in the same change

For each row: **confirm** (evidence and scope remain valid), **narrow** (rewrite
to the supported fixture, endpoint, or maturity), or **retire** (remove the
claim but preserve its history). Update `ara/logic/claims.md`, matching
`ara/evidence/`, result report, roadmap/README/task/gate state as applicable in
the same commit. Failed sealed artifacts remain immutable and linked from their
replacement; negative results stay first-class.

### 9. Report

End with the claim table, verifier commands and outcomes, corrections made,
items still unverified and why, unopened/unused evidence, and explicit follow-up
experiments or task updates. Separate supported, refuted, and unresolved claims.

## Anti-patterns (hard nos)

- Tuning a threshold, seed set, branch, or control after seeing formal outcomes.
- Repairing, deleting, or overwriting a failed sealed package.
- Treating report rendering or manifest success as semantic verification.
- Counting administrative reruns as independent evidence.
- Promoting a non-gated, toy-fixture, development, or preflight result into a
  general capability claim.
- Letting a secondary metric rescue a failed primary rule or control precondition.
- Reporting remembered numbers without replaying their raw source and predicate.

## Repository anchors

`CLAUDE.md` — benchmark-gated workflow and definition of done ·
`bench/gates.py`, `bench/SHIPPED`, `bench/results/` — gate criteria, ratchet,
and reports · `bench/**/results/*/artifact-manifest.json` — sealed evidence
graphs · `docs/research/` — frozen protocols and failure records ·
`ara/logic/claims.md`, `ara/evidence/` — bounded claims and proof packages ·
`Makefile` — exact verification entry points.
