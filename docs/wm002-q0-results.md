# WM-002 Q0 qualification results

Status: **independently accepted for Q0 protocol completeness only** on
2026-07-24. This result is permanently claim-ineligible, protocol 0.2.0-q is
unsealed, and Q1 and formal execution remain unauthorized.

## Verdict

WM-002 Q0 completely and reproducibly qualifies the exact known-model
arithmetic and normative Q0 protocol semantics of the finite hidden-actuator
fixture. It does not test or establish environment behavior, data collection,
training, model update, learned uncertainty, persistence, improvement,
retention, checkpoint restoration, external evaluation, or general agency.

The first produced report was rejected because correct marginalized arithmetic
did not prove the declared 88-cell coverage or full semantic parity. A repaired
report was rejected again when mutation probes found protocol fields protected
only by self-hashing. The final report was accepted only after exact path
enumeration, a sectioned normative protocol projection, targeted fail-closed
mutations, private-cell isolation, and a fresh independent recomputation.

## Exact result

The private analytical matrix contains 88 unique cells over both hidden signs,
all five acquisition actions, every supported observation, both terminal
decisions, and both terminal outcomes. Every action/terminal-decision
distribution normalizes to one.

| Acquisition action | Exact expected episode return |
|---|---:|
| `skip` | `1/2` |
| `weak` | `63/100` |
| `strong` | `37/50` |
| `overpowered` | `9/20` |
| `nuisance` | `49/100` |

The required deterministic selectors choose, in protocol order: Prospect
`strong`, independent Fraction oracle `strong`, goal-only `skip`, raw entropy
`nuisance`, EIG-only `overpowered`, and shuffled information `weak`. All five
public SHA-256 uniform-selector vectors reproduce their full digest, modulo
index, and selected action. The maximum Prospect-float versus exact-oracle
error is `1.1102230246251565e-16`, below the `1e-12` boundary.

## Evidence identities

| Evidence | SHA-256 |
|---|---|
| Canonical Q0 report | `779e8d8128312da2239107058137faac54751df620efb31291c0af98c2b8f243` |
| Protocol | `90b73ad4815380f113f91d0542bf7b91fd7e5196b5afd7f8c46b7fde9ec070cb` |
| Selected-source implementation manifest | `c9e6689a0ce66e5b79f733c057b839a155500908ba21a5adbf64637cb090c324` |
| Exact oracle | `e8987af521e00ace0ab15047847275cea1073d0bbe46fc77b3ec804c19bc8e55` |
| 88-cell matrix | `0a29e4a48ca9187e7825d2a8823f251699aa8df255fe0cf824ffeefcc5510e8e` |
| Uniform-vector rows | `a3f4be077e222a9c3e1aa674763cdc8f86b09dff7bac9528e77a475eb1d30879` |

The generated canonical report remains untracked at
`/tmp/wm002-q0-report.json`; it is reproduced with:

```bash
PYTHONPATH=src python -m bench.active_acquisition.run
```

## Verification and independent audit

- 82 focused active-acquisition tests passed.
- The repository-wide `make check` target passed Ruff, mypy over 42 source
  files, and 167 tests.
- The independent auditor regenerated the CLI report byte-for-byte,
  recomputed all 88 exact cells and five values without consuming producer
  aggregates, reproduced all selectors and uniform vectors, and matched all 11
  selected-source manifest members.
- A systematic 41-mutation sweep detected every one of the 37 normative Q0
  semantic or authority mutations. The four intentionally unbound mutations
  were prospective Q1-only mechanics.
- Q0 authority remained `claim_eligible=false`, `formal_authorized=false`, and
  zero-step; Q1 remained unimplemented, claim-ineligible, and unauthorized,
  with no formal version, seeds, thresholds, binding, or result schema.

The exact oracle is behaviorally and direct-call independent, not transitively
or process isolated. The implementation identity is deliberately a
selected-source manifest, not a complete dependency or environment closure.
Zero interaction is supported by direct-call rejection, fail-closed replacement
of both realization gateways, and fresh-run inspection; Q0 is analytical and
does not contain an instrumented general environment runtime.

## Next boundary

Q1 may be implemented against the
[authoritative-runtime design](wm002-q1-runtime-design.md), but it must not run
under protocol 0.2.0-q. Before the first environment interaction, a revised
prospective protocol must bind this audited report digest, freeze Q1
implementation and result/checkpoint schemas, pass recursive serialized-value
sentinel tests, freeze private-salt custody, and explicitly authorize Q1 while
keeping it permanently claim-ineligible. Only Q1 can test the linked
execute-to-update-to-act-to-restore path; neither Q0 nor Q1 can establish
learned uncertainty, which is isolated in [U-001](u001-learned-uncertainty-plan.md).
