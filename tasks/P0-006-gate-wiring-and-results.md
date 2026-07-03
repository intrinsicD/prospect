# P0-006 — Gate wiring, persisted results, the P0 gate, and a seed policy

- **Status:** ready
- **Phase:** P0
- **Requirements:** — (delivery infrastructure; enforces ADR-0005 for every requirement)
- **ADRs:** ADR-0005
- **Depends on:** none (P0-005 is complementary: sentinels read the run log)
- **Phase gate:** `bench/gates.py::GATES["P0"]` — registered by this task; until then:
  imports clean, `make test` green

## Goal
Real evals plug into the gate registry without editing registration internals; gate
runs persist a report artifact (so "paste the GateReport" in docs-sync is mechanical
and CI-checkable); every phase in `PHASE_ORDER` is runnable (`make gate PHASE=P0`
currently crashes with a `KeyError` traceback, as does a bare `make gate`); and
multi-criterion gates aren't forced through a single `metric: float`.

## Non-goals
- No eval bodies — all checks stay PENDING except P0's.
- No regression ratchet / CI gate job (that is P0-007, built on this).
- No changes to gate *criteria* — wording stays exactly as registered.

## Interface to satisfy
`bench/gates.py` (harness). Public surface after this task:
`@gate_check(phase)`, `@sentinel_check(name)`, `GateResult.metrics: dict[str, float]`,
`run_gate(phase, run_id=None) -> GateReport`, persisted report JSON.

## Approach (brief)
- **Registration:** `@gate_check("P1")` / `@sentinel_check("representation-integrity")`
  decorators replace the pending check on the already-registered entry (unknown
  phase/name → clear error). Eval modules live in `bench/evals/` and self-register on
  import; criteria remain data in `gates.py`.
- **Metrics:** replace `GateResult.metric: float` with `metrics: dict[str, float]`
  (P1's criterion is a conjunction of two measurements; one float cannot carry it).
  Same for `SentinelResult`.
- **Persistence:** `run_gate` writes `bench/results/<phase>-<timestamp>.json`
  (criterion, metrics, pass/health flags, seeds, run-id) and still returns/prints the
  report. Gitignore `bench/results/`? **No** — results are the record the docs-sync
  step cites; keep them committed (small JSON).
- **P0 gate:** register it — check = smoke suite green (run `pytest -q` via
  `subprocess`; passed ⇔ exit 0). Roadmap already names this criterion.
- **Friendly errors:** unknown phase → message listing valid phases, exit nonzero, no
  traceback; `make gate` without `PHASE` → usage line.
- **Seed policy:** each registered check owns an explicit seed list and records it in
  its `metrics`/report; the report JSON includes seeds so results are reproducible.
  Document this in the module docstring — gate criteria that say "over N seeds" now
  have an owner for N.

## Acceptance criteria
- [ ] `make gate PHASE=P0` runs and **passes** (smoke green).
- [ ] `make gate` (no PHASE) and `make gate PHASE=P9` print a friendly message and
      exit nonzero — no traceback.
- [ ] A dummy check registered via `@gate_check` in a test replaces PENDING and its
      metrics appear in the persisted JSON.
- [ ] `GateResult`/`SentinelResult` carry `metrics: dict[str, float]`; smoke tests
      updated.
- [ ] Report JSON written on every `run_gate` and includes criterion, metrics, seeds.
- [ ] `make test` green, `make lint` clean.

## Test plan
- Unit: decorator replaces pending check; unknown-phase error path; report JSON
  round-trip; P0 gate passes against the real suite.
- Manual: the three `make gate` invocations above.

## Docs-sync checklist
- [ ] Task Status updated; **paste the first passing P0 GateReport below**.
- [ ] ADR-0005 consequence notes results are persisted artifacts.
- [ ] `docs/roadmap.md` P0 row references the now-registered gate.
- [ ] `tasks/P1-001` eval section points at `bench/evals/` + `@gate_check`.

## Gate result
_not run yet_
