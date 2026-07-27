# WM-002 active-acquisition qualification plan

Status: **Q0 is independently accepted for protocol completeness only and was
rebound on 2026-07-24 after two bound sources changed. Q1 protocol
`0.3.0-q1`, runtime, strict artifacts, entry gate, and independent auditor are
implemented, and both blockers that reopened result-free qualification are
closed. A third non-author review at implementation digest `8e17bc1a…` found
nothing blocking across the previously unprobed surface and recorded nine
non-blocking findings
([evidence](../ara/evidence/wm002-q1-third-review-2026-07-28.md)). Execution
authorization remains false, and no Q1 outcome or capability evidence
exists.**

The two closures are: causal private-input noninterference, now backed by five
mutation controls that make the probe fail when a leak is injected; and the
complete four-producer, 28-restore-lane, merge, validation, publication, and
completed-marker orchestration, now executed end to end by the result-free
rehearsal mode described in the
[runtime design](wm002-q1-runtime-design.md). A
[prospective review](wm002-q1-prospective-review.md) then fixed two further
defects, but it was self-review: successive non-author reviews then refused
authorization at commits `52744be` and `03cd3fc`, finding three blocking
defects the author had missed plus one incomplete closure. All are fixed, and a
fresh non-author review over the changed implementation digest is required
before authorization. The fixed experiment-global
one-shot tombstone, independent auditor validation of the exact prospective
review, and descriptor-stable selected-source reads are implemented. The source
binding covers bytes observed during the read, not already-loaded bytecode, and
does not defend against a malicious coordinated same-account writer. The
combined result-free suite is 522 tests; that is a test fact, not a readiness
claim, and the remaining authorization steps are unchanged.

WM-002 begins with one small question and one cheap killing fixture. It does
not yet test transfer, learned uncertainty, continual-learning scale,
multimodality, or an external benchmark.

The reproducible Q0 disposition and exact evidence identities are recorded in
the [Q0 results](wm002-q0-results.md).

## Decision to be tested

The architecture separates task utility, information value, cost, and risk.
WM-001 did not use information value to select actions. WM-002-Q asks whether
the exact decision decomposition can change a real acquisition action in the
right way and whether the acquired observation then changes downstream
executed behavior.

The prospective qualification statement is:

> With one acquisition opportunity and one terminal decision, Prospect's
> exact expected-episode-return selector, whose decomposition contains an EVSI
> term, chooses the strong diagnostic pulse, consumes that exact real
> transition in a persistent model update, and obtains greater paired executed
> net return than every declared non-oracle selector.

This is a killing statement, not a mature-agent claim. Prospect and the
independent oracle use the same known finite likelihood model and are expected
to agree. Agreement validates exact VOI use; it cannot establish that an
uncertainty estimator was learned or calibrated.

## Hidden-actuator fixture

Each episode samples a harness-private actuator sign
`theta ∈ {-1,+1}` from a symmetric prior. The agent may execute exactly one
acquisition action and must then make one irrevocable terminal decision
`d ∈ {-1,+1}`.

A diagnostic pulse produces `y ∈ {-1,+1}` with
`P(y = theta) = q`. It earns immediate task payoff `1[y = +1]`.
The terminal decision succeeds with probability `0.9` if `d = theta` and
`0.1` otherwise. Net episode return is:

```text
acquisition task payoff + terminal success - action_cost - acquisition_cost
```

The complete action matrix at the initial prior is fixed before implementation:

| Action | Observation model | `action_cost` | `acquisition_cost` | Exact expected net return |
|---|---|---:|---:|---:|
| `skip` | one null outcome | 0.00 | 0.00 | 0.50 |
| `weak` | `q = 0.70` | 0.53 | 0.00 | 0.63 |
| `strong` | `q = 0.90` | 0.58 | 0.00 | 0.74 |
| `overpowered` | `q = 1.00` | 0.95 | 0.00 | 0.45 |
| `nuisance` | four uniform symbols independent of `theta` | 0.00 | 0.01 | 0.49 |

The construction separates several quantities that would otherwise agree:

- `strong` maximizes expected episode net return;
- `overpowered` maximizes information gain but is too costly;
- `nuisance` maximizes raw observation entropy but says nothing about `theta`;
- `skip` is the goal-only choice when information value is removed; and
- `weak` is selected when the pulse information values are deliberately
  attached to the wrong actions.

The latent `theta` and its seed are private to the environment/auditor. They
must never appear as fields in agent-visible observations, outcomes,
experiences, transitions, receipts, checkpoints, or public result rows.

## Arms and causal controls

Every primary and non-oracle arm gets the same five candidates, one acquisition
step, one update invocation, one terminal step, and one pre-terminal
checkpoint/restore operation.

| Arm | Selection score and unit | Required selection |
|---|---|---|
| Prospect | Full expected episode net return (`return`) | `strong` |
| Independent oracle | Separate exact `Fraction` episode return (`return`) | `strong` |
| Goal-only | Prior terminal value + immediate payoff − both costs (`return`) | `skip` |
| Raw entropy | Observation entropy (`nats`) | `nuisance` |
| EIG-only | Entropy reduction about `theta`, ignoring cost (`nats`) | `overpowered` |
| Shuffled information | Full return with permuted EVSI diagnostics (`return`) | `weak` |
| Uniform random | Public SHA-256 index; no scalar score | seed-dependent |
Every core `CandidateAssessment` remains a truthful decomposition of the real
action under the true model, and its `total_value` remains an expected return in
return units for every arm. Arm-specific selection objectives are recorded in
bench-only diagnostic rows with an explicit score kind and unit; entropy, EIG,
or a random index is never relabeled as core utility.

The shuffled pulse mapping is fixed as:

```text
weak        <- overpowered information model
strong      <- weak information model
overpowered <- strong information model
```

The permutation changes selection diagnostics only. The environment and the
transactional learner retain the true likelihood of the action actually
executed. This preserves the information-value multiset while breaking only
the action/value binding used for selection.

## Continuous evidence chain

For each episode and arm:

```text
private theta and potential-outcome table
  -> immutable prior snapshot
  -> all five pre-action candidate assessments
  -> one selected and executed acquisition
  -> canonical observation, outcome, ExperienceEvent, and EpistemicTransition
  -> one persistent transactional update from that exact transition
  -> pre-terminal checkpoint
  -> frozen terminal assessment and executed success
  -> fresh-process restore of the same pre-terminal state
  -> identical restored assessment, action, success, and net return
```

The terminal observation cannot train the pre-terminal model. Another arm's
experience, private oracle values, latent sign, future outcome, or
counterfactual potential outcome is never update-eligible.

## Q0 — exact semantic qualification

Q0 performs no environment interaction and is permanently claim-ineligible.
It exhaustively enumerates both latent signs, all actions, all observations,
both terminal decisions, and both terminal outcomes.

Three paths must agree within their declared numeric boundary:

1. an independent `fractions.Fraction` oracle for likelihoods, posteriors,
   payoffs, terminal values, EVSI, costs, and episode returns;
2. independent explicit `math.log` formulas for entropy and EIG, which are not
   rational quantities; and
3. Prospect's generic floating Bayes/EIG/EVSI primitives.

Q0 passes only if:

- every reachable rational quantity is exact in the Fraction oracle and agrees
  with Prospect's float path within `1e-12`;
- every independently recomputed entropy and EIG agrees within `1e-12`;
- the independent rational values reproduce `0.50`, `0.63`, `0.74`, `0.45`,
  and `0.49` exactly;
- the six deterministic selectors choose their required actions;
- the nuisance scan has zero EIG and EVSI despite maximal raw entropy;
- sign-label permutation leaves value and selection semantics invariant;
- `action_cost` and `acquisition_cost` are each charged exactly once; and
- declared public record fields and emitted Q0 payload keys contain no private
  hidden-state field. Q1 must separately test actual serialized runtime values.

Any Q0 failure stops WM-002. A semantic failure is not repaired by adding a
learned model or changing the fixture after looking at Q1.

## Q1 — claim-ineligible causal killing test

Q1 is not yet authorized. Its implemented entry gate requires the bound,
independently audited passing Q0 report; exact selected-source and schema
manifests; recursive serialized-value privacy probes; exact-0700 execution and
registry directories; an exact-0600 private salt and commitment; an
actual-filesystem atomic no-replace publication probe; and a canonical
result-free prospective review.

After final authorization and entry, Q1 is a full-budget rehearsal but never
confirmation evidence:

- four prospectively fixed master indices under the `q1v3` private schedule;
- 1,024 paired episodes per master and arm;
- 4,096 episodes and 8,192 executed environment steps per arm, for 28,672
  episodes and 57,344 steps across seven arms;
- exactly 512 episodes of each hidden sign per master, privately reordered;
- one acquisition update and zero terminal updates per episode; and
- one live/restored terminal comparison per episode.

The master index is the inferential unit. Episode rows are averaged inside each
master before calculating paired differences and two-sided 95% Student-t
intervals.

Private schedules and potential outcomes use HMAC-SHA256 over semantic keys
`(master, episode, action, terminal decision, draw role)` with an
environment-only salt. Public artifacts bind the salt commitment and semantic
digests, not the salt, hidden sign, or private preimages; reconstruction
material is available only to the trusted private execution lane and the
independent auditor. Uniform-policy selection is a public SHA-256 rule and does
not use the private salt.

## Ordered killing gates

Q1 gates form a strict prefix:

1. **Q1-K0 — protocol and budgets:** accepted Q0 report digest, frozen Q1
   implementation/schemas, private-salt commitment, explicit Q1 authorization,
   exact matrix, four masters, all arms, episodes, steps, updates, and absent
   formal authority.
2. **Q1-K1 — trace and isolation:** unique linked records and recursive sentinel
   scans over actual serialized values; no hidden, future, terminal,
   oracle-private, other-arm, or counterfactual leakage.
3. **Q1-K2 — selector identity:** truthful return-unit core assessments and
   unit-labeled bench diagnostics are both recorded; each arm makes exactly its
   declared selection; shuffled updates use the true executed likelihood; and
   public SHA-256 random replay is exact.
4. **Q1-K3 — persistent change:** Prospect's strong acquisition transition is
   the sole cause of changed posterior-model bytes and version used by the
   terminal decision.
5. **Q1-K4 — executed behavior:** Prospect beats shuffled information—the
   analytically strongest non-oracle control—and every other non-oracle arm in
   paired master-level mean net return, with every 95% interval lower bound
   strictly above zero.
6. **Q1-K5 — restore:** verifier PIDs differ from producer PIDs, every checkpoint
   creates a fresh component graph, and a verifier may process a bounded batch;
   each restore exactly reproduces model, version, ancestry, assessment, action,
   paired terminal outcome, and net return.

The run stops at the first failed gate. Later values cannot rescue an earlier
failure.

## Independent audit

The producer's aggregates and gate booleans are non-authoritative. A separate
auditor must:

- descriptor-safely reopen the exact canonical prospective review, validate
  its schema, method, assurance boundary, scope, selected-source count, exact
  protocol and implementation digests, result-free counters, and authorization
  flags, then require the canonical entry report to bind its exact digest;
- reconstruct the private balanced sign schedule and keyed potential outcomes;
- independently recompute the Fraction oracle and every selector;
- resolve consumed transition identities against canonical real experience;
- reconstruct net returns from immediate payoff, cost, and terminal success;
- aggregate at master-seed grain and recompute every paired interval;
- reproduce the ordered killing prefix; and
- reopen both live and fresh-process terminal traces.

This qualification needs deterministic canonical JSON, bounded sidecars,
atomic no-replace publication, and a SHA-256 manifest. It does not need to copy
WM-001's Pendulum/CUDA-specific binding, multi-gigabyte artifact, or
adjudication machinery.

## Abandonment and promotion rules

Kill the formulation if:

- Prospect and the independent oracle select different actions;
- any deterministic control selects the wrong action;
- any hidden or future state reaches an agent-visible decision or update;
- the exact selected acquisition transition is not the sole update cause;
- Prospect's primary paired return interval includes zero; or
- fresh-process terminal parity fails.

Do not rescue a failure by tuning action costs, likelihoods, episode counts,
selectors, or thresholds on Q1 outcomes. A changed scientific fixture requires
a new qualification protocol and fresh seeds.

If Q0 and Q1 pass and an independent results audit finds no fatal defect, the
only permitted next step is to draft a **new** formal protocol. It must allocate
fresh seeds and prospectively bind its claim, thresholds, schemas,
implementation, and audit. Q0 and Q1 remain claim-ineligible forever and may
not be pooled with or relabeled as formal evidence.
