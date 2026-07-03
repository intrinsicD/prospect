# P0-001 — `Prediction` parameterizes a real distribution

- **Status:** done
- **Phase:** P0
- **Requirements:** R1, R3, R4 (every consumer of the one signal reads this type)
- **ADRs:** ADR-0002 (amend contract consequence), ADR-0003 (option outcomes carry duration)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P0"]` (registered by P0-006; until then the P0
  criterion is: imports clean, `make test` green)

## Goal
`types.Prediction` — the one type everything reads — actually parameterizes a
distribution, so `log_prob` is computable from its fields, and it can express a jumpy
option outcome (duration). Today it is a point estimate plus two scalar summaries:
`log_prob` must raise `NotImplementedError`, forcing implementations to subclass a
frozen dataclass, and `OptionModel.predict_option` cannot express the duration
ADR-0003 requires.

## Non-goals
- No world-model implementation, no ensembles, no training (P1-001).
- No change to how epistemic is *estimated* — only to what the type can carry.
- Reward stays a point value; a reward distribution is deferred until a gate demands
  it (minimal rule).
- No `numpy`/`torch` in the core — it stays dependency-free.

## Interface to satisfy
`types.Prediction` itself is the contract. `interfaces.WorldModel` /
`interfaces.OptionModel` signatures are unchanged; skeleton docstrings updated.

## Approach (brief)
- Add `var: Array` — per-dimension variance of the predictive distribution over the
  next latent (the aleatoric spread). Keep scalar `epistemic` / `aleatoric` as summary
  fields (ensemble disagreement / mean predictive variance).
- Add `duration: float = 1.0` — one step for flat predictions; option-models predict
  multi-step outcomes (ADR-0003: landing latent, cumulative reward, **duration**).
- Make `log_prob` **concrete**: diagonal Gaussian NLL in pure Python over float
  sequences (`mean`, `var`, `observed` iterable). Clamp variance with a small floor so
  the result stays finite. Document that tensor-backed implementations may subclass to
  vectorize, but the default must be correct.
- Amend ADR-0002's contract consequence ("mean + var + epistemic/aleatoric split,
  `log_prob` concrete — never a bare float, never an unimplemented distribution") and
  add a one-line consequence to ADR-0003 (duration lives on `Prediction`).

## Acceptance criteria
- [x] `Prediction` has `mean`, `var`, `epistemic`, `aleatoric`, `reward`, `duration`.
- [x] `log_prob` is concrete and correct for a diagonal Gaussian: unit-tested against
      hand-computed values; higher near the mean; finite even at `var → 0` (floor).
- [x] `surprise = -log_prob(observed)` is directly usable — no subclass required.
- [x] ADR-0002 and ADR-0003 consequences amended; `tasks/P1-001` interface section
      updated to the new fields.
- [x] `make test` green, `make lint` clean.

## Test plan
- Unit: `log_prob` vs a hand-computed Gaussian NLL (two dims, known values);
  monotonicity (closer observation ⇒ higher log-prob); variance-floor finiteness;
  frozen immutability; `duration` defaults to 1.0.

## Docs-sync checklist
- [x] Task Status updated; gate result recorded below.
- [x] ADR-0002 contract consequence amended; ADR-0003 duration consequence added.
- [x] `docs/architecture.md` component note for types.py still accurate (verified —
      no change needed; it names `Prediction` as the important shared type, which holds).
- [x] `tasks/P1-001` "Interface to satisfy" updated (real `var`, concrete `log_prob`).

## Gate result
The P0 gate is not yet registered in `bench/gates.py` (that arrives with P0-006), so
the P0 criterion from the roadmap was applied directly:

```
imports clean, smoke tests green
make test : 15 passed (8 smoke + 7 new Prediction unit tests)
make lint : All checks passed!
```

Result: **PASS** (P0 criterion met). Tests covering this task:
`tests/test_prediction.py` — hand-computed diagonal-Gaussian NLL (2 cases),
monotonicity near the mean, variance-floor finiteness at `var=0`, length-mismatch
`ValueError`, frozen immutability, `duration` default.
