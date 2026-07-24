# WM-002: Exact active acquisition qualification

Status: **Q0 is independently accepted for protocol completeness only. The
claim-ineligible Q1 runtime, entry gate, and independent auditor are
implemented, but result-free qualification has been reopened and is not green
until every newly found blocker closes. Execution authorization remains false;
no Q1 outcome or capability evidence exists.**

Implemented safeguards now include a fixed experiment-global one-shot
`wm002-q1.attempt.json` tombstone, independent auditor reopening and validation
of the exact prospective review, and descriptor-stable selected-source reads.
The source binding covers the bytes observed during those reads, not
already-loaded bytecode, and is not protection from a malicious coordinated
same-account writer. The last completed combined result-free checkpoint before
subsequent integration changes reached 339 active-acquisition tests. That
checkpoint does not establish readiness.

This directory contains the Q0 exact-semantic runner and the implemented WM-002
Q1 qualification runtime. It contains no formal seal, formal seed set, or
accepted capability evidence. Q0 and Q1 remain permanently claim-ineligible;
formal evidence would require a separate prospectively reviewed protocol.

WM-002 asks one deliberately narrow question:

> On a finite hidden-actuator problem, can Prospect use full expected episode
> return, including decision-relevant value of information, to choose the
> acquisition action producing the best downstream executed net return against
> goal-only, raw-entropy, information-gain-only, shuffled-information, and
> uniform-random controls?

The first milestone is a cheap killing test, not a capability demonstration:

1. **Q0 — exact semantic matrix:** exhaustively verify the finite fixture,
   Bayes updates, EVSI decomposition, selectors, negative controls, and absence
   of hidden-state leakage.
2. **Q1 — claim-ineligible causal rehearsal (entry-gated):** run the
   complete decide → execute → observe → store → update → exploit → restore
   chain on four prospectively fixed master indices and 4,096 paired episodes
   per arm—28,672 episodes across seven arms—only after every protocol entry
   blocker closes.

Failure of either stage kills this formulation. It must not be rescued by
adding a neural model, tuning on the same outcomes, changing a threshold, or
opening a formal lane.

## Exact fixture

Each episode has a harness-private actuator sign
`theta ∈ {-1, +1}` with prior probability `1/2`. The agent gets one acquisition
action followed by one irrevocable terminal decision `d ∈ {-1, +1}`.

For a pulse with reliability `q`, the observed signal is `y ∈ {-1, +1}` and
`P(y = theta) = q`. A pulse also earns immediate task payoff
`1[y = +1]`. The terminal action succeeds with probability `0.9` when
`d = theta` and `0.1` otherwise. Net episode return is:

```text
immediate task payoff + terminal success - action_cost - acquisition_cost
```

At the symmetric prior, the exact expected net returns are:

| Acquisition action | Outcomes | Reliability | `action_cost` | `acquisition_cost` | Expected net return |
|---|---:|---:|---:|---:|---:|
| `skip` | one null outcome | — | 0.00 | 0.00 | 0.50 |
| `weak` | `y ∈ {-1,+1}` | 0.70 | 0.53 | 0.00 | 0.63 |
| `strong` | `y ∈ {-1,+1}` | 0.90 | 0.58 | 0.00 | 0.74 |
| `overpowered` | `y ∈ {-1,+1}` | 1.00 | 0.95 | 0.00 | 0.45 |
| `nuisance` | four equiprobable symbols independent of `theta` | — | 0.00 | 0.01 | 0.49 |

The required selector identities are:

| Arm | Required acquisition |
|---|---|
| Prospect expected episode return (with EVSI component) | `strong` |
| Independent exact oracle | `strong` |
| Goal-only | `skip` |
| Raw observation entropy | `nuisance` |
| EIG-only | `overpowered` |
| Shuffled information/action binding | `weak` |
| Uniform random | seeded uniform choice over all five actions |

The shuffled arm permutes selection diagnostics only (`weak <- overpowered`,
`strong <- weak`, `overpowered <- strong`). Core action assessments remain
truthful, and the environment and learner use the true likelihood of the action
actually executed. The control breaks only the selection-time action/value
binding.

## Interpretation boundary

In this exact fixture, Prospect and the independent oracle are supposed to
agree. The accepted Q0 validates only the 88-cell exact known-model arithmetic,
normative protocol parity, selector identities, deterministic tie/random rules,
selected-source binding, and field-level public-schema check. Q0 performs zero
environment interactions and establishes no learning,
persistence, executed improvement, or runtime-value isolation.

Only an authorized, passing Q1 could qualify the linked
execute → observe → update → act → restore chain. Its implementation, strict
schemas, actual serialized-value leakage tests, private-salt custody,
single-attempt marker, watchdogs, streaming publication, and independent
auditor now exist. The attempt marker is one fixed experiment-global tombstone,
not one marker per proposed run identity, and the auditor independently
revalidates and binds the exact result-free prospective review. These
implemented controls do not return the reopened qualification to green. The
final protocol/review/entry sequence remains mandatory after all blockers
close.
Neither stage shows learned or calibrated uncertainty, general active learning,
dual control, or superiority to a published method.

Run the claim-ineligible Q0 check with:

```bash
PYTHONPATH=src python -m bench.active_acquisition.run
```

The Q0 contract is [`protocol.json`](protocol.json); the successor Q1 contract
is [`q1_protocol.json`](q1_protocol.json). The exact audited Q0 disposition is
recorded in [`docs/wm002-q0-results.md`](../../docs/wm002-q0-results.md). The
human-readable implementation and evidence plan is
[`docs/wm002-active-acquisition-plan.md`](../../docs/wm002-active-acquisition-plan.md).
