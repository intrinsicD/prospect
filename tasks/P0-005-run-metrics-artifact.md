# P0-005 — Run-metrics artifact: the data sentinels read

- **Status:** done
- **Phase:** P0
- **Requirements:** — (integrity infrastructure; serves ADR-0006, which protects R1/R3/R4/R7)
- **ADRs:** ADR-0006 (amend consequence: sentinels read the run log)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P0"]` (registered by P0-006)

## Goal
Define where training-time metrics live, so a zero-argument sentinel `check()` can
verify conditions like "latent std stays above the floor **throughout training** — not
only at the capability checkpoint". Today that criterion is unimplementable: sentinels
take no arguments and no artifact exists for them to read. Without this contract,
P1-001's implementer discovers the gap mid-task and invents the format under pressure.

## Non-goals
- No sentinel implementations (they arrive with P1-001 / P3-003 / P5-002).
- No experiment-tracking dependency (no wandb/tensorboard) — stdlib JSONL only.
- No plotting, no dashboards.

## Interface to satisfy
New `bench/runlog.py` (harness): a writer/reader pair over
`bench/runs/<run-id>/metrics.jsonl`.

## Approach (brief)
- `RunLog.log(step: int, metrics: dict[str, float])` appends one JSON record per call;
  `read_run(run_id)` / `latest_run()` return the records for sentinel consumption.
- Convention: each sentinel documents (in its `criterion` / eventual `check`) the
  metric keys it requires — e.g. `representation-integrity` reads
  `latent_std_min`, `latent_effective_rank`; `uncertainty-reliability` reads
  `disagreement_error_rank_corr`. Training loops emit these keys via the dict returned
  by `Learner.update()` (P0-003) plus held-out probes.
- `run-id` is supplied by the caller (harness), so gate runs can name the run they
  evaluate; `latest_run()` is the default for `make gate`.
- Add `bench/runs/` to `.gitignore`.

## Acceptance criteria
- [x] `bench/runlog.py` writer/reader round-trips records (step ordering preserved,
      floats intact).
- [x] `latest_run()` resolves correctly with multiple runs present.
- [x] `gates.py` module docstring documents the sentinel↔runlog contract (where a
      zero-arg `check()` gets its data).
- [x] `bench/runs/` gitignored.
- [x] `make test` green, `make lint` clean.

## Test plan
- Unit: write N records to a tmp run dir, read back, compare; two runs → `latest_run`
  picks the newer; malformed line raises a clear error.

## Docs-sync checklist
- [x] Task Status updated; gate result recorded below.
- [x] ADR-0006 consequence amended (sentinels are fed by the run log; evaluation cost
      note stands).
- [x] `tasks/P1-001` approach updated: training logs sentinel metrics via `runlog`.

## Gate result
The P0 gate is not yet registered in `bench/gates.py` (that arrives with P0-006), so
the P0 criterion from the roadmap was applied directly:

```
imports clean, smoke tests green
make test : 26 passed (22 prior + 4 new runlog tests)
make lint : All checks passed!
```

Result: **PASS** (P0 criterion met). Tests covering this task:
`tests/test_runlog.py` — write/read round-trip (order + values preserved),
`latest_run()` resolution across two runs (explicit mtimes, no flakiness),
no-runs `FileNotFoundError`, malformed-line `ValueError` with file:line location.
