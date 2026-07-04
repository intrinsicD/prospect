# P0-009 — Enforce the typing rule: mypy in CI, conformance assertions, CI hygiene

- **Status:** done
- **Phase:** P0
- **Requirements:** — (process; protects every `Protocol` seam)
- **ADRs:** — (implements the CLAUDE.md "full type hints" rule, which currently has no enforcer)
- **Depends on:** none (best taken *after* the other code-touching P0 tasks so it
  typechecks the final contracts — but not blocked by them)
- **Phase gate:** `bench/gates.py::GATES["P0"]` (registered by P0-006)

## Goal
The typing mandate becomes machine-checked. `runtime_checkable` isinstance checks
verify method *presence*, not signatures — a `predict(self)` with the wrong arity
passes today's smoke test. A static checker plus explicitly-annotated conformance
assertions make the Protocol seams real.

## Non-goals
- No stub packages, no plugin config, no typing of future implementation internals.
- No ruff `ANN` rules — mypy's `disallow_untyped_defs` covers it without duplicating.

## Interface to satisfy
Tooling surface: `pyproject.toml`, `Makefile` (`make typecheck`),
`.github/workflows/ci.yml`, `tests/test_conformance.py`.

## Approach (brief)
- Add `mypy>=1.10` to the `dev` extra; configure in `pyproject.toml`:
  `strict = false` but `disallow_untyped_defs`, `check_untyped_defs`,
  `warn_return_any`, `no_implicit_optional` on for `src/`, `bench/`, `tests/`.
- `make typecheck` target; CI step after lint.
- New `tests/test_conformance.py` with typed assignments — one per skeleton/protocol
  pair — so mypy structurally verifies each implementation against its contract:

  ```python
  _wm: interfaces.WorldModel = FlatWorldModel()
  _pl: interfaces.Planner = FlatPlanner()
  # ... every skeleton
  ```

  (These also run as a test, subsuming the isinstance smoke checks.)
- Ruff: add `I` (import sorting) to the lint selection; autofix the tree.
- CI hygiene: build matrix Python 3.11 / 3.12 / 3.13 (requires-python is `>=3.11`);
  README quickstart gains a one-line venv note so the Makefile's
  `--break-system-packages` stops being the implied default workflow.

## Acceptance criteria
- [x] `make typecheck` exists and passes on the whole tree (25 files).
- [x] Conformance assertions cover **every** skeleton/protocol pair (15 typed
      assignments incl. `FlatWorldModel`-as-`Learner` and `SemanticStore`-as-
      `KnowledgeSource`); a deliberately wrong `predict` arity was caught by mypy
      at the conformance assertion with an expected-vs-got diff, then reverted.
- [x] CI runs lint + typecheck + tests + gate-all on 3.11/3.12/3.13.
- [x] Ruff `I` enabled; tree clean (import order autofixed).
- [x] `make test` green, `make lint` clean.

## Test plan
- The wrong-arity experiment above (mypy must fail).
- CI run on the branch across the matrix.

## Docs-sync checklist
- [x] Task Status updated; gate result recorded below.
- [x] CLAUDE.md updated: core conventions (typecheck enforced + conformance
      assertion for new surface), Commands list, Definition of done.
- [x] README quickstart updated (venv note, `make typecheck`, `make gate-all`).

## Gate result
The P0 gate is shipped; the ratchet re-runs it. Full verification:

```
make lint      : All checks passed!            (ruff E,F,B,UP,I)
make typecheck : Success: no issues found in 25 source files
make test      : 37 passed
make gate-all  : [P0] PASS — ratchet ok, 1 shipped gate(s) still green
```

Result: **PASS** (P0 criterion met). Wrong-signature experiment: dropping the
`action` parameter from `FlatWorldModel.predict` made mypy fail at the
conformance assertion (`tests/test_conformance.py`) with the expected-vs-got
signature diff — the Protocol seams are now machine-checked at signature level,
which `runtime_checkable` isinstance checks cannot do. Reverted; clean.
