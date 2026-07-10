# U-110 — Unsupervised skill discovery (METRA-class)

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R5, R2
- **ADRs:** ADR-0002 (competence/skill trust), ADR-0003 (options as action space)
- **Depends on:** none
- **Phase gate:** the triggering skill-discovery phase gate
- **Source:** `docs/sota-review-2026-07.md` U-110 · [METRA](https://arxiv.org/abs/2310.08887)

## Trigger (promote to `ready` when…)
The **roadmap adds a skill-discovery phase** — options are no longer harness-authored but
must be *discovered* by the agent (a phase whose gate rewards an agent that finds its own
temporally-extended behaviours). The **upgrade-triggers** workflow step checks: if a new
phase's task specifies option/skill discovery, promote. Today the option library is
harness-authored (`SkillRouter.add`), which is the correct minimal choice for the current
gates — discovery is a genuinely new capability, not a hygiene fix (review RQ4).

## Goal
When triggered: add unsupervised skill discovery — a temporal-distance-aware latent skill
space (METRA/LSD lineage) from which options are extracted — so the skill library and the
hierarchy's action space are learned rather than authored.

## Non-goals
- Not before the trigger — a numpy re-implementation is nontrivial and unjustified until a
  gate rewards discovery.
- Not replacing the competence gate (mastered-only-offered) — discovered skills still flow
  through it.

## Interface to satisfy (when promoted)
A discovery component producing `Option`s (with policies) that feed the existing
`SkillRouter` (skills.py) and `HierarchicalManager`; the competence monitor and jumpy
option-model consume discovered options unchanged.

## Approach (brief, when promoted)
- METRA: learn a skill latent maximizing temporal-distance coverage; decode skills to
  option policies; register mastered ones upward via the existing gate.

## Acceptance criteria (when promoted)
- [ ] Discovered options integrate with `SkillRouter`/`HierarchicalManager`; the
      discovery gate PASSes (discovered skills beat a no-discovery baseline).
- [ ] Existing P4/P5 gates not regressed; `make gate-all` green; clean checks.

## Test plan (when promoted)
- Unit: discovered options satisfy the `Option` interface and route/simulate correctly.
- Eval: the triggering discovery gate + `make gate PHASE=P4/P5` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0003: record skill discovery as a distinct phase and its trigger.
- [ ] `docs/sota-review-2026-07.md`: note U-110 outcome.

## Gate result
<deferred — no gate until promoted>
