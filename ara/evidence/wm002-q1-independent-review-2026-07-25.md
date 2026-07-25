# WM-002 Q1 independent prospective review evidence

- **Target**: commit `52744be` on clean `main`, synchronized with `origin/main`
- **Reviewer role**: non-author adversarial reviewer
- **Disposition**: reject Q1 authorization
- **Boundary**: result-free; no Q1 private draw, production interaction, outcome, or attempt
- **Provenance**: ai-executed

## Executed checks

| Check | Result |
|---|---|
| `make check` | Ruff clean, Mypy clean over 66 source files, 516 tests passed |
| Complete Q1 rehearsal | Passed with real authenticated children, 56 episodes, and all six emitted artifacts initially at mode 0600 |
| Auditor via `python -S -m bench.active_acquisition.q1_audit` | Rejected rehearsal as required, but also emitted the unexpected Q1-K0 selected-source failure described below |
| Artifact mode mutation | Changing disposable `aggregate.json` from 0600 to 0644 was accepted by `_validate_artifact_directory` |
| Review-marker mutation | `_load_validated_prospective_review` accepted the rehearsal's machine-generated reviewer mark |

## Findings

### F1 — blocking: stale normalized protocol digest

`q1_audit.py` freezes
`14d22a0544377b4a1c754f109c25c1f23d67d05ad1f1aef978fcf04628a78fe5`.
Independent recomputation over the current protocol with only
`execution_authorized` normalized to false yields
`e015c8b519a10eeab19ecda1384071b17c450484c06ebbd4142d95143c92353b`,
which is also the value enforced by `q1_qualification.py`.

An authorized protocol gets past the execution-boundary check and then
deterministically fails Q1-K0 on this stale auditor constant. Because the audit
is post-run, this can strand the sole attempt after its result-bearing
execution.

Forensic bindings:

- `bench/active_acquisition/q1_audit.py:135`
- `bench/active_acquisition/q1_audit.py:1205`
- `bench/active_acquisition/q1_qualification.py:52`

### F2 — blocking: module entrypoint fails selected-source closure

The auditor requires `bench.active_acquisition.q1_audit` in `sys.modules` but
only looks it up by that name. Under its real `python -m` entrypoint, the
executed module is `__main__`, so the check reports:

`auditor essential selected modules were not loaded:['bench.active_acquisition.q1_audit']`

The producer already handles this case through `__main__.__spec__.name`; the
auditor does not. The rehearsal integration test invokes the correct module
entrypoint but asserts only that `execution_authorized` appears among Q1-K0
violations, so the extra failure is masked.

Forensic bindings:

- `bench/active_acquisition/q1_audit.py:787`
- `bench/active_acquisition/q1_audit.py:806`
- `bench/active_acquisition/q1.py:524`
- `tests/test_active_acquisition_rehearsal.py:181`

### F3 — blocking: published-artifact custody is under-enforced

The frozen protocol requires all six Q1 artifacts to be one-link regular files
with exact mode 0600. The auditor checks 0600 only for
`private-audit.jsonl`; its hashing path also requests private-mode enforcement
only for that one file. A disposable mutation of `aggregate.json` to mode 0644
was accepted.

The unit-test helper encodes the same mismatch by assigning 0644 to the other
five artifacts while naming the fixture an exact publication.

Forensic bindings:

- `bench/active_acquisition/q1_protocol.json:327`
- `bench/active_acquisition/q1_protocol.json:346`
- `bench/active_acquisition/q1_audit.py:2113`
- `bench/active_acquisition/q1_audit.py:2150`
- `tests/test_active_acquisition_q1_audit.py:756`

### F4 — nonblocking defense-in-depth gap: reviewer-mark parity

The production entry gate rejects the machine-generated rehearsal reviewer
mark, but the independent auditor's prospective-review loader requires only a
nonempty reviewer string. A legitimate production entry cannot normally carry
that review, so this is not independently sufficient to execute Q1, but the
auditor does not fully reproduce the entry gate's review semantics.

Forensic bindings:

- `bench/active_acquisition/q1_audit.py:613`
- `bench/active_acquisition/q1_audit.py:670`
- `bench/active_acquisition/q1_qualification.py:1525`

## Required closure before a new review

1. Repair F1–F3 and add regressions that exercise the auditor as a subprocess.
2. Mirror the machine-generated review-mark rejection in the auditor.
3. Re-run the full suite and complete result-free rehearsal.
4. Obtain a fresh non-author review over the new implementation digest.
5. Only then create the authorized protocol/review/entry bindings for the
   one-shot production attempt.
