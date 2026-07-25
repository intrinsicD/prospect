# WM-002 Q1 fresh non-author review evidence

- **Target commit**: `03cd3fc489a62eb687c6545b885c80a5c12602dc`
- **Selected-source count**: 48
- **Implementation SHA-256**:
  `cf7cdfe2ababbd361a10c0284544261a04cf5fad56e771326f7f2cafed5e5352`
- **Disposition**: reject Q1 authorization
- **Boundary**: result-free; no Q1 private draw, production interaction,
  outcome, or attempt
- **Provenance**: ai-executed

## Executed checks

| Check | Result |
|---|---|
| Repository state | Clean `main`, exactly synchronized with `origin/main` |
| Entry/auditor manifests | Both 48 files and exact digest `cf7cdfe2…5352` |
| `make check` | Ruff clean, Mypy clean over 66 source files, 520 tests passed |
| Complete result-free rehearsal | Passed with real authenticated children and all six artifacts initially at mode 0600 |
| Real auditor module entrypoint | Returned the expected rehearsal rejection, reported exact implementation digest `cf7cdfe2…5352`, and emitted no normalized-protocol or selected-source-origin failure |
| Protocol normalization | Entry implementation, auditor implementation, and both frozen constants all equal `e015c8b5…53b` |
| Rehearsal review mark | Independently rejected by the auditor |
| Post-hash mode-drift probe | Five artifacts remained at 0644 during the semantic pass; the auditor emitted no mode violation |

## Original-finding closure

- **F1, stale normalized protocol digest — closed.** Both independent
  normalizers and both frozen constants agree on
  `e015c8b519a10eeab19ecda1384071b17c450484c06ebbd4142d95143c92353b`.
- **F2, Python `-m` self-origin — closed.** The real site-disabled module
  entrypoint completed source closure and reported the exact new implementation
  digest.
- **F3, all-six exact-0600 custody — only partially closed.** The initial
  directory scan and committed hash pass now enforce 0600 on all six files, but
  the later semantic consumers do not.
- **F4, machine-generated review mark — closed.** The independent auditor now
  mirrors the production entry rejection while retaining the legitimate
  independent-review positive path.

## Blocking finding: semantic reopens drop exact-mode custody

The auditor performs two distinct artifact-consumption phases. The first hashes
all six files through `private=True`. Later:

- `_load_streaming_aggregate` reopens `aggregate.json` through
  `_read_regular_file` without `private=True`; and
- `_stream_audit_evidence` reopens the other five streams with
  `private=name == "private_audit"`.

A deterministic probe let the initial directory and committed-hash phases see
all six artifacts at 0600, then changed the five non-sidecar files to 0644
before the semantic phase. The semantic readers consumed those files, the
files remained 0644, and Q1-K0 contained no mode violation:

```text
post_hash_modes=['0o644', '0o644', '0o644', '0o600', '0o644', '0o644']
mode_violations=[]
```

The probe uses a hook only to place a persistent chmod exactly between the two
phases; it does not mutate-and-restore bytes, forge a digest, or require the
auditor to resist an undetectable same-account writer. At audit completion the
publication visibly violates the frozen all-six-0600 rule while the auditor is
silent. A production artifact with the same unchanged bytes can therefore pass
all content checks despite failed custody.

Forensic bindings:

- `bench/active_acquisition/q1_protocol.json:346`
- `bench/active_acquisition/q1_audit.py:2163`
- `bench/active_acquisition/q1_audit.py:2384`
- `bench/active_acquisition/q1_audit.py:2517`
- `bench/active_acquisition/q1_audit.py:2539`
- `tests/test_active_acquisition_q1_audit.py:1569`

## Required closure

1. Reopen the aggregate and every semantic stream with exact-0600 enforcement.
2. Revalidate the exact six-file publication after the semantic pass so the
   audit binds its terminal visible custody state.
3. Add a phase-boundary regression that mutates a non-sidecar mode after the
   committed hash and requires a Q1-K0 custody violation.
4. Re-run the full check and complete result-free rehearsal.
5. Obtain another fresh non-author review over the resulting implementation
   digest before changing authorization.
