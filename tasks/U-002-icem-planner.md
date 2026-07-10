# U-002 — iCEM planner: colored noise + keep/shift elites + execute-best + softmax weighting

- **Status:** done
- **Phase:** U (upgrade track; re-gates against P2/P5)
- **Requirements:** R1, R2
- **ADRs:** ADR-0001, ADR-0006/0007 (the epistemic penalty is untouched — proposal
  change only)
- **Depends on:** none (composes with U-001; land either order)
- **Phase gate:** `bench/gates.py::GATES["P2"]` (and `["P5"]` if the manager's inner
  scoring reuses the CEM loop) — must improve toward or hold the margin
- **Source:** `docs/sota-review-2026-07.md` U-002 · [iCEM](https://arxiv.org/abs/2008.06389)
  · [Pink Noise](https://openreview.net/forum?id=hQ9V5QN27eS) · [TD-MPC2](https://arxiv.org/abs/2310.16828)

## Goal
Bring `FlatPlanner` up to iCEM, the acknowledged baseline-beater at exactly this
low-dim/low-budget regime: sample temporally-correlated (colored) action sequences,
keep elites across CEM iterations and shift them across MPC steps, execute the
best-seen action rather than the elite mean, and update the elite mean/std with
softmax score weighting (the TD-MPC2/MPPI update).

## Non-goals
- No second planner class / no config-knob sprawl: these are edits to the existing
  loop, gated. Colored-noise `beta` gets one sensible default (~2.0), not a sweep knob.
- Not gradient-based or diffusion-annealed planning (SKIP per review — chaotic
  through-model gradients; annealing schedules are exactly the knob surface CLAUDE.md
  forbids without a gate).
- The per-step epistemic penalty (planning.py:86) is unchanged — this changes the
  *proposal distribution*, not the score.

## Interface to satisfy
`planning.FlatPlanner` (planning.py:17-111): `plan()` samples colored noise (replace
the white `self._rng.normal` at planning.py:68), retains an elite pool across the
`iterations` loop, returns the best-seen first action (replace `mean[0]` at
planning.py:75), and weights the elite update by `softmax(scores/temperature)`.
Constructor gains `colored_beta: float = 2.0`, `keep_elite_fraction: float = 0.3`,
`temperature: float = 0.5`. `Planner` protocol unchanged.

## Approach (brief)
- Colored noise: scale the white-noise FFT along the horizon axis by `f^(-beta/2)` and
  inverse-FFT (numpy.fft, ~10 lines) — temporally correlated candidates, the single
  largest contributor in iCEM's ablation. Use iCEM's fixed initial `σ=0.5` in
  normalized action coordinates so correlated trajectories do not saturate the
  physical action range.
- Keep-elites: carry a fraction of the previous iteration's elites into the next
  candidate pool; shift-elites: seed the next MPC step's pool with the shifted elite set
  (the warm start at planning.py:62 already shifts the mean — extend it to the pool).
- Execute-best: track the argmax-scoring sequence seen across iterations; return its
  first action.
- Softmax update: `mean = Σ w_i · seq_i`, `w = softmax(score/temperature)` over elites
  (uses score magnitude the current unweighted average throws away).

## Acceptance criteria
- [x] Colored-noise sampling, keep/shift elites, execute-best, softmax elite update all
      in the one loop; unit test confirms colored samples are temporally correlated
      (lag-1 autocorrelation > white-noise baseline).
- [x] **P2 gate PASS with margin ≥ current** on every seed (measured, not assumed);
      `make gate-all` green.
- [x] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_planner.py): autocorrelation of sampled sequences; execute-best
  returns the max-scoring first action on a rigged toy model; softmax weights sum to 1
  and recover the mean as temperature→∞.
- Eval: `make gate PHASE=P2` (report the margin delta vs the shipped P2 report),
  `make gate PHASE=P5`, `make gate-all`.

## Docs-sync checklist
- [x] Status → done; gate margin before/after recorded below.
- [x] architecture.md/planning docstring: "flat MPC/CEM" → iCEM note.
- [x] `docs/sota-review-2026-07.md`: mark U-002 shipped.

## Gate result

The first complete four-feature loop retained CEM's inherited initial standard
deviation (`σ=1.0` in normalized action coordinates). With beta-2 noise this made
whole trajectories saturate the action limits: P2 blocked on seed 1 (`-66.02` vs
the `-64.99` baseline). Restoring iCEM's fixed `σ=0.5` normalized proposal scale
removed that failure without adding a knob or changing the reward/epistemic score.

Final `make gate PHASE=P2`: **PASS**. The comparison floor is the stronger of the
model-free and random returns (model-free on all three seeds). “Before” is the
immediately preceding U-001 report, `P2-20260710T131536Z.json`.
The full-ratchet P2 evidence is `P2-20260710T140315Z.json`.

| seed | U-001 return | U-001 margin | U-002 return | U-002 margin | margin Δ |
|------|-------------:|-------------:|-------------:|-------------:|---------:|
| 0 | -46.15 | 17.26 | -45.23 | 18.19 | **+0.92** |
| 1 | -64.28 | 0.71 | -62.95 | 2.04 | **+1.33** |
| 2 | -59.58 | 2.30 | -56.73 | 5.15 | **+2.85** |

Median P2 return is `-56.73` vs model-free `-63.41` vs random `-67.57`; the
planner wins every seed. `make gate PHASE=P5`: **PASS**, hierarchy returns
`[-9.1, -4.2, -4.7]` vs compute-matched flat `[-49.2, -43.3, -35.2]` at about
729 member-forwards/step. `make test`: 131 passed, 1 skipped; lint and mypy clean.
Final `make gate-all` (P0–P14): **PASS**. Its P9 composition check improves to
`-16.3` vs reactive `-73.1`, with planning worth `+56.8` return and every sentinel
healthy.
