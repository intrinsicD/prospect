# P0-009 — Enforce the typing rule: mypy in CI, conformance assertions, CI hygiene

- **Status:** ready
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
- [ ] `make typecheck` exists and passes on the whole tree.
- [ ] Conformance assertions cover **every** skeleton/protocol pair; a deliberately
      wrong signature is caught by mypy (verified once, then reverted).
- [ ] CI runs lint + typecheck + tests on 3.11/3.12/3.13.
- [ ] Ruff `I` enabled; tree clean.
- [ ] `make test` green, `make lint` clean.

## Test plan
- The wrong-arity experiment above (mypy must fail).
- CI run on the branch across the matrix.

## Docs-sync checklist
- [ ] Task Status updated; gate result recorded below.
- [ ] CLAUDE.md "core conventions" line mentions `make typecheck`.
- [ ] README quickstart updated (venv note, typecheck command).

## Gate result
_not run yet_
