# Prospect

Prospect is an adaptive-agent runtime for linking collected experience to
persistent state changes, held-out behavior, and retained improvement.

The canonical contract is:

```text
decide -> execute -> observe -> store -> assimilate -> learn -> evaluate
```

Prediction, uncertainty, realized proper score, belief revision, information
gain, decision value, learning, and retention remain distinct. Stable identities
connect every step; there is no universal epistemic scalar or hidden “last
prediction.” See [the architecture](docs/architecture.md).

## Current boundary

The repository contains:

- backend-neutral domain records and protocols;
- exact Bayes, information-value, and proper-score reference semantics;
- transparent action assessment;
- one linked runtime with canonical experience and epistemic stores;
- failure-atomic in-process learning across owned model bytes, runtime state, and
  the update ledger;
- canonical replay custody plus optional TorchRL/TensorDict sampling;
- integrity-checked component checkpoints; and
- an executable probabilistic world-model, CEM control, retention, restart, and
  independent-evidence program in
  [WM-001](bench/world_model_lifecycle/README.md).

WM-001 protocol 1.3.0 has completed one eight-seed formal attempt. Its immutable
producer evidence passed K0–K7 with strong fixture-specific effects and exact
fresh-process parity, but the attempt is formally rejected because its mandatory
pre-bound independent auditor contained two false-negative defects and returned
`passed: false`. The [formal results review](docs/wm001-v130-formal-results.md)
preserves both the mechanism evidence and the failed-acceptance boundary.
Therefore the repository still has no accepted demonstration of the complete
claim. Protocol 1.4.0 is the active fresh-seed confirmation: it keeps the
scientific system and thresholds unchanged, makes predictive coverage an exact
integer-count contract over persisted float32 evidence, binds endpoint
conformance and corrected auditor identity, and permits one formal attempt
after a claim-ineligible full-budget rehearsal.

## Layout

```text
src/prospect/             current agent implementation
bench/epistemic/          exact semantic and lifecycle references
bench/world_model_lifecycle/
                          WM-001 protocol, implementation, evidence, and runbook
tests/                    active unit, adversarial, and integration tests
docs/architecture.md      canonical system definition
datasets/                 preserved reusable inputs and checksums
.agents/skills/           project research and results-audit skills
```

Generated experiment outputs belong under `bench/**/results/` and remain
untracked. Curated reusable inputs belong under `datasets/` with provenance and
checksums.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
make install
make install-runtime
make check
make check-runtime
make epistemic-diagnostics
make epistemic-gate
python -m bench.world_model_lifecycle.verify protocol
make wm001-development
```

`make check` covers the backend-neutral core. `make check-runtime` adds the
world-model implementation and adversarial tests. `make wm001-development` runs
the two declared diagnostic seeds and is never claim-eligible. See the
[WM-001 executable runbook](bench/world_model_lifecycle/README.md) before
creating an implementation binding or formal attempt.
