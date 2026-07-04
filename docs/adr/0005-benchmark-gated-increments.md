# ADR-0005 — Benchmark-gated incremental delivery

**Status:** Accepted

## Context
The full system is close to the open problem of general intelligence; naive
all-at-once building produces sprawl and unfalsifiable progress. The team's working
style is benchmark-as-fitness with explicit kill-gates.

## Decision
Deliver in phases (`docs/roadmap.md`). Each phase has a **kill-gate** in
`bench/gates.py` with a *precise* pass criterion. A phase ships only when its gate
passes; agents work only the current phase. Pair this with a **minimal-implementation**
rule: build the smallest thing that satisfies the current task; generality is earned
by a gate, not added in advance.

## Consequences
- (+) Progress is measurable and falsifiable; dead ends die at their gate.
- (+) Guards against premature generality and over-engineering.
- (−) Requires discipline to keep gate criteria honest and to resist building ahead.
- Gate runs persist a JSON report under `bench/results/` (criterion, metrics, seeds,
  run-id) — the committed record the docs-sync step cites and CI can check. Eval
  bodies live in `bench/evals/` and self-register via `@gate_check` /
  `@sentinel_check`; criteria stay data in `bench/gates.py`. *(Amended by P0-006.)*
- Shipped phases are recorded in `bench/SHIPPED` (appended in the same commit that
  records the passing gate) and re-gated on every CI run via `make gate-all`: gates
  gate *staying* shipped, not only shipping. Expensive gates re-run their
  *evaluation* against persisted/regenerable artifacts, not full retraining (noted
  in the report detail). *(Amended by P0-007.)*
