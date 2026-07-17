# Epistemic lifecycle results audit

**Audited:** 2026-07-17<br>
**Scope:** active E-series source tree and `bench/epistemic` diagnostics<br>
**Source state:** uncommitted E-series cutover based on `f126703`<br>
**Review standard:** independently reproduce the arithmetic, then test whether the
reported endpoint actually entails the named capability

## Verdict

The new runtime and exact epistemic semantics are useful, testable foundations.
They do **not** yet demonstrate that Prospect collects experience, learns from
those same records, improves its executed behavior, and retains that improvement.

All numeric predicates in the finite diagnostic fixture pass. The capability
report correctly emits `passed: false`:

| Row | Audited disposition | What the current result actually supports |
|---|---|---|
| E0 | supported, structurally | one authoritative step can preserve linked decision, execution, observation, experience, belief-update, score, transition, and learner-receipt custody |
| E1 | supported, as an exact oracle | finite Bayes, entropy, expected information gain, EVSI, proper scores, and typed uncertainty obey the frozen semantics |
| E2 | reference-only | a known finite channel model ranks the predefined diagnostic probe above the controls |
| E3 | reference-only | exact same-task Bayesian assimilation improves posterior log score |
| E4 | blocked | the fixture calculates expected utility; it does not execute held-out behavior |
| E5 | blocked | JSON state round-trip preserves independent task slots; shared-state retention is not exercised |
| Full lifecycle | blocked | E2–E5 do not describe one agent or one identity-linked causal chain |

## Reproduction

```bash
make test
make epistemic-diagnostics
make epistemic-gate
```

The first two commands are expected to succeed. The last command is expected to
exit nonzero until every capability row is supported. A successful
`--diagnostics` exit only means that the frozen arithmetic predicates reproduced;
the JSON still says `passed: false`.

## Independently reproduced numbers

| Diagnostic | Primary | Frozen/random control | Corrupted/irrelevant control |
|---|---:|---:|---:|
| E2 relevant-probe frequency | 1.0000 | random 0.3333 | raw observation entropy 0.0000 |
| E2 EIG per synthetic cost | 4.8186 | random 1.6062 | raw observation entropy 0.0000 |
| E3 mean log score, lower is better | 0.373935 | frozen 0.693147 | shuffled 2.037489; irrelevant 0.693147 |
| E4 analytic expected utility | 0.8832 | frozen 0.7600 | shuffled 0.2832; irrelevant 0.7600 |
| E5 task-A utility drift | 0.0000 | — | independent task slot prevents parameter interference |
| E5 restart drift | 0.0000 | — | serialization parity only |

These values are deterministic properties of the current finite fixture. They are
not estimates of general agent performance and have no sampling uncertainty.

## Claim-breaking findings

### 1. The collection and learning rows are causally disconnected

E2 exercises the authoritative runtime and creates canonical experiences and
epistemic transitions. E3 constructs another reference learner and never consumes
those transition identities. Consequently, the benchmark cannot answer the
central question: did the experience that Prospect chose and collected cause the
subsequent persistent change?

This is not repairable by relabeling a metric. One agent identity, one canonical
store, and an `UpdateReceipt` listing the exact E2 transition identities must span
the complete experiment.

### 2. E3 is belief assimilation, not model learning

The primary arm updates exact, task-local posterior probabilities under a known
channel model. Across its 100 update receipts, the model, representation, and
policy versions do not change. Only task posterior/configuration state changes.
Evaluation reuses the same task identities that were assimilated.

This remains a good Bayes oracle. It is not evidence that a predictive
model learned a rule that generalizes to disjoint held-out cases.

### 3. E4 does not execute behavior

The reported utility is computed analytically from the hidden binary-rule
distribution. The environment's label/action path is not used to produce held-out
actions and realized outcomes. Utility gain and regret reduction are therefore
the same algebraic quantity, not independent behavioral evidence.

A valid E4 freezes pre- and post-update snapshots, executes both on the same
held-out stream at equal resource budget, and scores externally observed outcomes
with updates disabled.

### 4. E5 excludes the failure mode it names

Task A and task B occupy independent posterior slots, so learning B cannot damage
A's parameters. The fixture also lacks a task-A pre-learning behavioral baseline.
Its checkpoint omits canonical experiences, epistemic transitions, and update
receipts: those stores contain 4/4/4 records before serialization and 0/0/0 after
restore, while the digest still passes.

The result proves a narrow JSON round-trip property. Retention requires one
shared-parameter learner, a pre-learning A baseline, learned A gain, interfering B
updates, post-B A evaluation, a production checkpoint, a fresh process, and a
post-restart A evaluation.

### 5. Several controls are too synthetic for a formal result

- Task identities encode the rule through an even/odd suffix convention.
- The shuffled arm deterministically inverts labels instead of using a
  marginal-preserving permutation with recorded parent lineage.
- The raw-entropy arm's zero is affected by deterministic tie ordering.
- The finite schedule exactly matches the channel proportions and has no
  sensitivity analysis over independent stochastic seeds.
- No sealed raw run package binds protocol, source, dependencies, configuration,
  splits, seeds, budgets, identities, and checkpoint receipts.

These do not invalidate the exact semantic oracle; they prevent promotion to a
capability claim.

## Claims permitted by this source state

It is accurate to say:

> Prospect now has typed, version-linked epistemic records; an authoritative
> one-step runtime; exact finite Bayesian and proper-score semantics; canonical
> in-memory custody; an optional TorchRL replay index; and integrity-checked
> episode-boundary checkpoint bundles.

It is not accurate to say:

> Prospect has demonstrated model learning, executed behavioral improvement,
> continual retention, a complete collect-to-retain causal chain, or a novel
> learning architecture.

## Minimum experiment that can change the verdict

Build one small but real shared-parameter learner behind the existing protocols
and run a single sealed lane:

1. freeze task generator, train/calibration/behavior/retention splits, seeds,
   budgets, and thresholds;
2. let the authoritative runtime select and collect E2 experiences;
3. train only from the exact recorded transition identities and issue a
   version-changing receipt;
4. compare frozen pre-update, no-update, true-link, and
   marginal-preserving-link-permutation arms on disjoint predictive data;
5. execute frozen pre/post policies on held-out outcomes at equal budget;
6. learn an overlapping interference task using the same parameters;
7. save every stateful category, restore in a fresh process, and re-evaluate the
   original gain and new-task plasticity; and
8. preserve the complete raw evidence bundle even when a row fails.

Until that lane exists, more elaborate models or external arenas would measure
something other than the lifecycle claim Prospect is trying to establish.
