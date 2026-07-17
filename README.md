# Prospect

Prospect is a research runtime for making adaptive-agent claims auditable:
which experience was collected, what persistent state changed because of it,
whether behavior improved on held-out cases, and whether that gain survived
restart and interference.

## Core contract

The runtime keeps prediction, uncertainty, realized proper score, belief
revision, information gain, goal-conditioned information value, learning, and
retention distinct. Immutable identities link them through:

```text
decide -> environment step -> observe -> assimilate -> learn -> evaluate
```

There is no universal “epistemic” scalar and no hidden “last prediction.” See
[the architecture](docs/architecture.md) and
[ADR-0014](docs/adr/0014-linked-epistemic-lifecycle.md).

## Current state

Implemented:

- backend-neutral evidence, observation, belief, prediction, decision,
  experience, transition, update, snapshot, and evaluation records;
- exact finite Bayes, information-gain, EVSI, log-score, Brier, and Gaussian-NLL
  reference semantics;
- explicit utility/information/cost/risk decision decomposition;
- one authoritative runtime with canonical experience and epistemic ledgers;
- append-only partial-lifecycle evidence;
- an optional pinned TorchRL/TensorDict replay index; and
- atomic, integrity-checked component checkpoint bundles.

Not demonstrated:

- predictive-model learning from the agent’s own collected experience;
- executed held-out behavioral improvement caused by that update; or
- retention of the same gain through shared-state interference and a production
  process restart.

The exact binary reference diagnostics produce useful numbers, but an
independent audit found that their E2–E5 rows use different agents and do not
form one causal chain. The report therefore emits `passed: false`: E2/E3 are
`reference_only`, while E4/E5 are `blocked`. This is a semantic and plumbing
fixture, not a mature-agent result.

See the [independent results audit](docs/research/2026-07-17-epistemic-lifecycle-results-audit.md)
for the reproduced numbers and claim limits, and the
[research portfolio](docs/research/2026-07-17-linked-experience-research-portfolio.md)
for prior-art threats, candidate architectures, and the recommended first real
same-chain experiment.

The superseded P-series implementation, tests, and benchmark ratchet were
removed from the active tree during the E-series cutover. Their history remains
available in Git; they no longer constrain or masquerade as validation of the
new architecture.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
make install                 # editable package + development tools
make install-runtime         # additionally install pinned TorchRL/TensorDict
make check                   # lint + typecheck + 85 active tests + diagnostics
make epistemic-diagnostics  # exact predicates; report still says passed:false
make epistemic-gate         # capability status; intentionally nonzero today
```

## Layout

```text
src/prospect/domain/       immutable records and backend-neutral protocols
src/prospect/decision/     transparent candidate assessment and selection
src/prospect/epistemics/   exact information and proper-score semantics
src/prospect/runtime/      authoritative linked lifecycle and state custody
src/prospect/storage/      canonical stores, TorchRL replay, checkpoints
bench/epistemic/           exact reference diagnostics
tests/test_epistemic_*.py  active contract, adversarial, and integration tests
docs/                      architecture, ADRs, roadmap, research audits
tasks/                     active task and historical planning records
```

Start with `CLAUDE.md`, `docs/architecture.md`, and
`tasks/E0-001-epistemic-lifecycle-rewrite.md`.
