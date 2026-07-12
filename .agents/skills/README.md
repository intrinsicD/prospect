# Prospect Agent Skills

This directory contains one project-scoped [Agent Skill](https://agentskills.io).
The skill is checked in under `.agents/skills/` so agent harnesses can discover the
same repository-specific workflow without relying on ignored local `.claude/` state.

| Skill | Purpose | Triggers on |
| --- | --- | --- |
| `prospect-research-ideation` | Generate and adversarially audit diverse, falsifiable research portfolios, then hand selected ideas into Prospect's task/ADR/benchmark workflow. First-party (MIT, Alexander Dieckmann), adapted from `transformational-research-skill-kit` v1.0.0. | Novel, unconventional, cross-domain, transformational, or publishable research directions; research roadmaps; high-risk/high-reward experiments. Not ordinary feature brainstorming or implementation of an already-selected method. |

The skill carries hand-authored `references/`, `assets/`, `evals/`, and
`scripts/` companions. Its repository context is a navigation aid, not authority:
verify it against the live tree before using it.

The skill proposes and audits only. Route a selected candidate through
`tasks/TEMPLATE.md`, an ADR when required, and the relevant benchmark gate before
implementation.
