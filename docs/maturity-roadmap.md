# Prospect maturity roadmap

Status: active research program.

This roadmap decomposes the maturity gaps left open by accepted WM-001 into
causally isolated scientific experiments and a parallel engineering-assurance
track. It is a planning index, not a sealed protocol or evidence authority.

## Operating rule

Each scientific experiment introduces one primary causal delta. Diagnostics
may be qualification gates, but acquisition, task distribution, replay,
representation, horizon, and evaluation regime must not all change in one
claim. A later task waits for the preceding task to reach a terminal accepted
or rejected disposition; it does not require the preceding hypothesis to win.

Every task progresses through:

```text
plan -> cheapest killing test -> claim-ineligible qualification
     -> prospectively frozen protocol -> formal run
     -> independent audit and bounded interpretation
```

Formal thresholds, seeds, budgets, controls, and exclusions are frozen only
after claim-ineligible qualification. Generated outputs stay untracked below
the relevant `bench/**/results/` directory.

## Scientific sequence

| Task | Status | Primary causal question | Depends on |
|---|---|---|---|
| WM-001 | Accepted and frozen | Can one linked agent collect, learn, improve, retain, and persist on the bounded two-regime fixture? | — |
| [WM-002](wm002-active-acquisition-plan.md) | Active | Does decision-relevant information value select more useful real experience under a matched budget? | WM-001 |
| [U-001](u001-learned-uncertainty-plan.md) | Queued | Can Prospect learn calibrated, decision-useful uncertainty from experience rather than receive a known likelihood? | terminal WM-002 |
| [WM-003A](wm003-heldout-transfer-plan.md) | Queued | Does one frozen shared model generalize zero-shot to prospectively held-out dynamics contexts? | terminal U-001 |
| [WM-003B](wm003b-online-adaptation-plan.md) | Queued | Does online updating adapt faster than scratch without unacceptable short-term forgetting? | terminal WM-003A |
| [WM-004](wm004-continual-scale-plan.md) | Queued | Does adaptive replay preserve plasticity and retention as task count grows under bounded memory? | terminal WM-003B |
| [WM-005](wm005-partial-observability-plan.md) | Queued | Does a learned belief state improve long-horizon prediction and control under observation aliasing? | terminal WM-003B |
| [WM-006](wm006-observation-regimes-plan.md) | Optional, queued | Does the lifecycle survive replacing privileged state vectors with learned observations? | terminal WM-005 |
| [EXT-001](ext001-external-evaluation-plan.md) | Queued | Does a frozen accepted snapshot reproduce a bounded result in an independently operated evaluation? | accepted WM-003B or later; A-004 |

## Parallel assurance track

These tasks establish engineering or evidence guarantees, not agent
intelligence:

| Task | Status | Assurance question | Depends on |
|---|---|---|---|
| [A-001](a001-durable-recovery-plan.md) | Queued, may start in parallel | Can interrupted durable learning recover exactly once? | current runtime |
| [A-002](a002-lifecycle-concurrency-plan.md) | Queued | Are concurrent lifecycle operations linearizable? | A-001 state machine |
| [A-003](a003-mid-episode-recovery-plan.md) | Queued | Can a declared simulator step resume without duplicated or omitted effects? | A-001, A-002 |
| [A-004](a004-cross-machine-restore-plan.md) | Queued | Can a checkpoint restore with prospectively bounded semantic parity on another machine? | A-001 |
| [A-005a](a005-external-attestation-plan.md) | Queued | Can an offline verifier validate a canonical signed evidence envelope and detect mutation or replay? | versioned evidence-envelope schema |
| [A-005b](a005b-transparency-lifecycle-plan.md) | Queued | Can an external trust service expose rollback, equivocation, revocation, and invalid key transitions? | A-005a; provider selected |

## Repository convention

Before implementation, each scientific task owns:

- one plan under `docs/`;
- one isolated `bench/<experiment>/` package containing its problem, protocol,
  runner, analysis, and README;
- focused `tests/test_<task>_*.py` tests; and
- untracked generated results below its benchmark package.

Fixture-specific mechanisms remain in `bench/`. Move a contract into
`src/prospect/` only when it is task-neutral, required by the runtime, and
covered by parity tests against the experiment that motivated it. Never modify
the sealed WM-001 protocol or relabel its evidence.

## Current action

WM-002 is the only active scientific experiment. Its claim-ineligible
[Q0 qualification](wm002-q0-results.md) is independently accepted for protocol
completeness: the 88-cell arithmetic, normative protocol projection, controls,
authority boundary, and reproducible report identities close. The
[authoritative Q1 runtime](wm002-q1-runtime-design.md), strict artifact
contracts, result-free entry gate, and independent auditor are implemented
but result-free qualification has been reopened and is not green until every
newly found blocker closes. The fixed experiment-global one-shot tombstone,
auditor-side validation of the exact prospective review, and descriptor-stable
selected-source reads are implemented; the source binding does not cover
already-loaded bytecode or a malicious coordinated same-account writer. The
last completed combined result-free checkpoint before subsequent integration
changes reached 339 tests, which is not readiness or evidence. Q1 execution
authorization remains false; its frozen budget is 4,096 episodes per arm and
28,672 total, and no Q1 outcome exists. Only after blocker closure and renewed
result-free review may final protocol authorization, canonical entry, the sole
claim-ineligible execution, and independent result audit proceed. Downstream
tasks remain planning-only.
