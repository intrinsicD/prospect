# P9-006 — Retrieval generalization via a dimension-adequate store

- **Status:** done
- **Phase:** P9
- **Requirements:** R8 (knowledge retrieval), R1/R4 (the epistemic gate it rides on)
- **ADRs:** ADR-0004 (retrieval-as-action — amended here: retrieval-as-lookup obeys the
  curse of dimensionality), ADR-0008 (whole-system validation / generalization)
- **Depends on:** P9-005 (fixed the uncertainty gate so retrieval *fires* out-of-region;
  surfaced — and mis-attributed — the residual non-generalization)
- **Phase gate:** `bench/gates.py::GATES["P9"]` — the cross-env generalization criterion,
  now with retrieval folded in (all four capabilities must generalize).

## Goal
Make uncertainty-gated retrieval **generalize** to the second environment (PointMass),
and gate it — so the P9 generalization check covers prediction, planning, the epistemic
signal (P9-005) AND retrieval. Along the way, correct P9-005's wrong diagnosis of *why*
retrieval didn't generalize.

## Non-goals
- Not a new retrieval mechanism, index, or key transform. The fix is *store density*,
  not code — the latent `_key` is unchanged.
- Not fixing retrieval-*into-planning* (the P9-001/P9-002 finding that overriding the
  planner's rollout dynamics mid-CEM degrades control). That is a distinct composition
  limit, still reported not gated.
- Not an ANN/approximate index (the store's exact nearest-neighbour is fast enough at
  this scale — the cached key matrix keeps the query loop flat; see below).

## Interface to satisfy
No new `Protocol`. `SemanticStore` / `UncertaintyMemoryRouter` / the latent `_key` are
unchanged. The change is a single eval constant (`bench/evals/p9_generalization.py::
STORE_N`) plus folding `gen.retrieval_met` into the P9 gate's `generalizes_met`.

## Approach (brief)
- **Diagnosis (measured, overturning P9-005's hypothesis).** P9-005 guessed the encoder
  saturated and corrupted the retrieval *key* space. Two measurements disproved that:
  1. **Key comparison** — swapping the latent key for a raw standardized-*input* key (the
     pre-encoder feature P9-005 proposed) did **not** help; the latent key was *better*.
     So the key was never the problem.
  2. **Density sweep** — holding the latent key fixed and only growing the store:
     `STORE_N=1500` fails (gated 0.021 > model 0.017), `12000` improves but misses the
     margin, `25000`+ generalizes. The nearest fact gets *closer* (hence righter) purely
     as the store densifies.
- **Cause.** Retrieval returns the nearest fact in a continuous 6-D key space
  (`concat(4-D latent, 2-D action)`). Nearest-neighbour recall there obeys the **curse of
  dimensionality**: the store's density must scale with the key dimension or the nearest
  fact is too far to be a right answer. The first environment (Pendulum, 4-D key) happened
  to be dense enough at the old `STORE_N`; the second was not.
- **Fix.** Provision a **dimension-adequate** store: `STORE_N = 40000` (chosen for a
  comfortable margin — retrieval beats no-retrieval by ~22%, ~6% below the 1.2× gate
  threshold, robust across the CI matrix). Cost is ~3.4 s/seed of store-build; the
  cached-matrix query loop stays flat (~0.1 s). Then **gate** `retrieval_met`.

## Acceptance criteria
- [x] Uncertainty-gated retrieval generalizes to env #2: `gated * 1.2 <= none` on the
      median over seeds — with the unchanged latent key.
- [x] `retrieval_met` folded into the P9 gate's `generalizes_met`; **P9 still PASS**.
- [x] No regression: `make gate-all` (P0–P9) green.
- [x] `make test` green, `make lint` clean, `make typecheck` clean.
- [x] The wrong "key-space saturation" attribution corrected everywhere it appeared
      (P9-005 task, BACKLOG, ADR-0008, architecture.md, p9_integration docstring/notes).

## Test plan
- Diagnosis reproduced with two scratch experiments (key comparison + density sweep) —
  summarized above; the density sweep is the load-bearing evidence.
- Eval: `make gate PHASE=P9` — the generalization criterion now includes retrieval;
  `make gate-all` for regression.

## Docs-sync checklist
- [x] Task Status → `done`; gate result recorded below.
- [x] P9 already shipped (in `bench/SHIPPED`); the ratchet re-runs it — no append needed.
- [x] ADR-0004 amended (curse-of-dimensionality consequence); ADR-0008 generalization
      table + narrative corrected (P9-003 → P9-005 → P9-006 evolution).
- [x] architecture.md single-environment-overfit bullet corrected.
- [x] BACKLOG P9-005 finding corrected; P9-006 row added; Phase 9 summary updated.

## Gate result
`make gate PHASE=P9` → **PASS** with cross-env now `prediction ✓ + planning ✓ +
uncertainty ✓ + retrieval ✓`. `make gate-all` → **ratchet ok — all shipped gates green**.

**The diagnosis, measured (the key was never the problem; density was):**

| Store density (env #2, 6-D key) | none MSE | gated MSE | generalizes (gated·1.2 ≤ none) |
|---|---|---|---|
| STORE_N = 1500 (old) | 0.0172 | 0.0212 | ✗ |
| STORE_N = 12000 | 0.0172 | 0.0151 | ✗ (misses margin) |
| STORE_N = 25000 | 0.0172 | 0.0140 | ✓ (thin) |
| **STORE_N = 40000 (chosen)** | 0.0172 | **0.0135** | **✓ (~22% better)** |

Key comparison (STORE_N=4000, OOD band): latent-key retrieved-err **beat** the
standardized-input-key P9-005 proposed — confirming the residual shortfall was density,
not a saturating key.

**Finding (reported, not tuned away):** retrieval *into planning* still has a negative
ablation marginal (overriding rollout dynamics mid-CEM corrupts multi-step control). That
is a composition limit distinct from the 1-step-prediction generalization fixed here.
