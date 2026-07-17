# Prospect Agent Skills

This directory contains two project-scoped [Agent Skills](https://agentskills.io).
They are checked in under `.agents/skills/` so agent harnesses can discover the
same repository-specific workflows without relying on ignored local `.claude/` state.

| Skill | Purpose | Triggers on |
| --- | --- | --- |
| `prospect-research-ideation` | Generate and adversarially audit diverse, falsifiable research portfolios, then hand selected ideas into Prospect's experiment workflow. First-party (MIT, Alexander Dieckmann), adapted from `transformational-research-skill-kit` v1.0.0. | Novel, unconventional, cross-domain, transformational, or publishable research directions; high-risk/high-reward experiments. Not ordinary feature brainstorming or implementation of an already-selected method. |
| `prospect-results-audit` | Run an adversarial scientist pass over gate, experiment, capability, and causal-mechanism claims; independently replay predicates and semantic verification; then confirm, narrow, or retire each claim with its evidence. | Before claims or phase/default promotion; after gates, formal experiments, or evidence sessions; while reviewing results-bearing changes; whenever numbers lack independent verification. |

The research-ideation skill carries hand-authored `references/`, `assets/`,
`evals/`, and `scripts/` companions. Repository context in either skill is a
navigation aid, not authority:
verify it against the live tree before using it.

The ideation skill proposes and audits candidates only. Route a selected
candidate into one bounded `bench/` experiment with a frozen protocol, controls,
budget, killing criterion, and generated `results/`. Keep reusable inputs in
`datasets/`. Use the results-audit skill after evidence exists and before its
claims or state transitions are accepted.
