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
- optional TorchRL/TensorDict replay; and
- integrity-checked component checkpoints.

It does not yet demonstrate one real agent learning a predictive model from its
own experience, improving executed held-out behavior because of that update, and
retaining the same gain through shared-state interference and process restart.
The next experiment must establish that chain before any maturity claim.

## Layout

```text
src/prospect/             current agent implementation
bench/epistemic/          exact semantic and lifecycle references
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
make epistemic-diagnostics
make epistemic-gate
```

`make epistemic-gate` remains nonzero until the complete learning and retention
chain is demonstrated.
