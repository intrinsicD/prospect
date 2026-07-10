# U-104 — CEM / beam search over option sequences (replace exhaustive K^depth)

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R2
- **ADRs:** ADR-0003 (hierarchical planning)
- **Depends on:** none (reuses FlatPlanner's CEM loop)
- **Phase gate:** `bench/gates.py::GATES["P5"]` (equal-or-better at equal compute)
- **Source:** `docs/sota-review-2026-07.md` U-104 · [SkiMo](https://arxiv.org/abs/2207.07560)
  · [TAP](https://arxiv.org/abs/2208.10291)

## Trigger (promote to `ready` when…)
The **option library outgrows exhaustive K^depth** — concretely `K^depth` exceeds the
manager's compute budget (≈ K > 6 at depth 3, i.e. > 216 sequences), **or** manager
latency shows up as a P5-class gate cost. The **upgrade-triggers** workflow step checks:
if the registered option set size or planning depth pushes `product(options, repeat=depth)`
past a documented budget, promote. Until then exhaustive K^depth is *exact and cheaper*
than random shooting at small K (review RQ4: FAIR's 2026 system still uses random shooting;
exhaustive K³ beats it whenever K³ < 256).

## Goal
When triggered: replace `HierarchicalManager`'s `product(...)` enumeration
(planning.py:270) with sampling-based search over option sequences — reuse `FlatPlanner`'s
CEM loop over one-hot-relaxed / continuous option parameters (SkiMo), or beam search over
option codes (TAP), whichever the trigger's scale favors. Beam search is the natural
first fallback (prunes K^depth to beam_width·K per level, stays deterministic).

## Non-goals
- Do NOT build before the trigger — exhaustive search is exact and cheap at current K.
- Not a new planner class: reuse the existing CEM machinery over option parameters
  (the review's explicit guidance — don't write a second planner).
- Not skill *discovery* (that is U-110).

## Interface to satisfy (when promoted)
`HierarchicalManager.plan_option` (planning.py:266-281): swap the exhaustive loop for
CEM-over-options (candidates are option-sequence parameter vectors scored by the same
duration-aware discounted reward minus epistemic penalty) or beam search. `HierarchicalPlanner`
protocol unchanged.

## Approach (brief, when promoted)
- CEM-over-options: relax the one-hot option choice to a categorical/continuous parameter,
  run FlatPlanner's iterate-elite loop, decode to the best first option — the epistemic
  penalty (U-011-corrected) carries over unchanged.
- Beam search: keep the top-`beam_width` partial sequences per level; the P9-007-style
  distance/reliability term is the "stay near data" prior (TAP's OOD filter).

## Acceptance criteria (when promoted)
- [ ] Search-over-options replaces enumeration; matches exhaustive-search quality at
      small K (backward-compat) and stays within budget at the triggering large K.
- [ ] **P5 gate PASS** at equal compute; `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: on a small library, search recovers the exhaustive-search best first option;
  scales past K^depth budget without blowup.
- Eval: `make gate PHASE=P5` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0003: record the search replacement and its K-threshold trigger.
- [ ] `docs/sota-review-2026-07.md`: note U-104 promoted/shipped.

## Gate result
<deferred — no gate until promoted>
