# ADR-0011 — An optional, non-gated harder-benchmark tier (real MuJoCo control)

**Status:** Accepted

## Context
Every capability in the repo is validated on environments the project **authored**:
the Pendulum, the 2-D PointMass, synthetic visual blobs. That is deliberate — the
gates must run in the numpy-only CI, reproducibly, on any machine (ADR-0005) — but it
has a real limit an external review named directly: authored toys cannot distinguish
"the core works" from "the core was tuned to dynamics we designed." The P2 gate, for
instance, shows MPC-over-a-learned-model beating model-free control *on the Pendulum*.
Does that survive contact with dynamics the repo did not shape?

Answering that needs a **standard, externally-defined** benchmark — a MuJoCo control
task (DeepMind Control Suite). But its dependencies (`mujoco`, `dm_control`) are
heavyweight and violate two load-bearing invariants: the numpy-only CI and the
everything-is-gated rule. Pulling them into the gated core would slow every CI run and
make the ratchet depend on a native physics engine — a bad trade for the core's
reproducibility guarantee.

## Decision
Add a **fenced, optional, NON-gated benchmark tier** — not a phase, not a gate.
- **Isolation.** `dm_control`/`mujoco` live behind a `[bench-hard]` optional extra,
  never in `[dev,learn]`. The tier's code lives in `bench/hard/`, is **never imported
  by the gate registry** (`bench.evals`), and skips cleanly when the extra is absent.
  `make gate-all` / `python -m bench` and the numpy `ci.yml` never touch it; a separate
  **manual** `workflow_dispatch` job (`bench-hard.yml`) runs it on demand.
- **The seam does the work.** A `DMCEnvironment` adapter satisfies the existing
  `bench.Environment` Protocol (P0-004). The **core (`src/prospect/`) is not touched** —
  `FlatWorldModel`/`FlatPlanner`/`Agent` already take `obs_dim`/`action_dim`/action
  bounds, so a real MuJoCo task is just another environment. State observations only
  (no pixels ⇒ no GL/render dependency); a real visual encoder is the vision arc's
  concern (ADR-0009), orthogonal to this.
- **What it measures.** It re-runs the **P2 claim** on foreign dynamics with the *same
  machine* (P2's budget, planner defaults, and a budget-matched CEM-ES model-free
  baseline). Deviating from P2's settings to make MBRL look better is forbidden — that
  would defeat the probe.
- **The deliverable is an honest report, not a pass.** Output is an externally archived
  artifact (`bench/hard/results/`) with raw per-seed returns, seed spread, and
  matched-budget deltas. There is no threshold that ships anything. A small-budget planner losing to
  published SAC would be unremarkable; the only question asked is whether the
  model-based machine beats a model-free baseline **given the same env steps**.

## Consequences
- (+) A genuine credibility check against a standard benchmark, without weakening the
  numpy-only, fully-reproducible core CI or the gate ratchet.
- (+) Proves the `Environment` seam is real: the composed agent runs on MuJoCo with
  zero core edits, across multiple domains and action dimensionalities.
- (+) Surfaces honest boundaries (e.g. the exploit-mode epistemic penalty steering the
  planner away from regions a random-data model never saw) as *measured* findings
  rather than assertions — the report is auditable.
- (+) Becomes the home for follow-up *studies* on those boundaries, not just the one probe:
  the swingup failure it surfaced is chased by **A** (does curiosity-driven collection fix
  it? `bench/hard/curiosity.py`) and **B** (does imitation-from-observation reproduce it?
  `bench/hard/imitation.py`, ADR-0012) — both non-gated, both in the one consolidated report.
- (−) The probe is not reproducible in the default CI (needs the extra + a native
  physics engine); its numbers are evidence, not a ratcheted guarantee, and can drift
  with `dm_control`/`mujoco` versions (recorded in the report).
- (−) Being non-gated, nothing enforces that it stays green — by design. It is a
  telescope, not a tripwire. If a future decision wants a MuJoCo *gate*, that is a
  separate ADR that must also solve the CI-reproducibility cost.
- (−) It is not a SOTA effort: small budget, a small numpy world model, short horizons.
  It answers "does the core survive contact with real dynamics," not "is it good."
