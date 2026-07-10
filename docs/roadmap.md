# Roadmap (benchmark-gated)

Each phase ships **only** when its kill-gate in `bench/gates.py` passes. Do not build
ahead of the current gate. Gates are the project's fitness function; keep their pass
criteria precise even while the eval body is a TODO.

| Phase | Goal | Deliverable | Kill-gate (summary) | Advances |
|-------|------|-------------|---------------------|----------|
| P0 | Scaffold | this repo | imports clean, smoke tests green (`GATES["P0"]`, registered) | — |
| P1 | Flat world model + calibrated uncertainty | `FlatWorldModel` | latent 1-step beats a persistence/linear baseline; epistemic falls with data while aleatoric persists on a stochastic variant | R1, R4 |
| P2 | Planning beats reaction | `FlatPlanner` (iCEM/MPC) | imagined MPC beats a model-free baseline at **equal env-step budget** | R1 |
| P3 | VoE + curriculum + replay | `SurpriseCompetenceMonitor`, `ReplayBuffer` | expected-vs-violated surprise differential reliable over seeds; curiosity beats random exploration | R3 (R7 groundwork) |
| P4 | Skills + router | `SkillRouter` | simulate-to-select picks the right skill above baseline; misapplication flagged by a surprise spike | R5 |
| P5 | Hierarchical planning | `JumpyOptionModel`, `HierarchicalManager` | two-level planning beats flat on a long-horizon task at **equal compute** | R2 |
| P6 | Any-to-any codec | `UniversalCodec` | swapping the single-modality codec for the universal one preserves core-loop performance within tolerance | R6 |
| P7 | Continual improvement | consolidation + metrics | on a task sequence: retention above threshold (no catastrophic forgetting) AND plasticity retained (late tasks learn as fast as early) | R7 |
| P8 | Knowledge bases | `UncertaintyMemoryRouter`, `KnowledgeSource`s | uncertainty-gated retrieval beats no-retrieval on the use-case benchmark AND stays robust to a poisoned/low-trust source | R8 |
| P9 | Whole-system validation | integration gate + ablation + 2nd env + invariants | the **composed** agent works end-to-end (learns while acting; one VoE signal drives explore/exploit, mastery and retrieval in one run; retrieval improves control); every part is load-bearing (ablation); capabilities survive a 2nd environment; no gate passes on a trivial solution or within noise | R1–R8 |
| P10 | External knowledge through the codec | `ExternalKnowledgeSource`, `UniversalCodec` | the agent answers OOD queries it can't derive from experience by retrieving external **content** and ingesting it through the codec (ADR-0004 rule 1), uncertainty- **and** distance-gated; beats the model alone, stays no-worse where it's already competent, and stays robust to a poisoned/low-trust source | R8 |
| P11 | Compute-as-action tools | `ToolSource` | the agent **calls** an exact compute tool (cost per call) as an action gated by uncertainty (ADR-0004 rule 2): the tool result (ingested through the codec) beats the model on OOD; the uncertainty signal spends the call budget on the right queries (beats random at equal budget); and gating is the cost sweet spot — better than never-calling at fewer calls than always-calling | R8 |

### Omni-modal seams & learning from observation (ADR-0009)
> **Universal adaptable seams, specialized per deployment, one modality per gate.** The
> codec admits any input/output modality into the shared latent; a deployment (private,
> industrial, science, robotics) instantiates and trains the subset it needs. Each concrete
> seam is gate-earned — vision first; the live-webcam / real-YouTube / live-sensor layer is
> a non-gated runtime demo on top. Then the agent learns from watching (observe→repeat→
> explore, which is ADR-0007's curriculum). Future seams (each its own gate, added on
> demand): **audio, proprioception, force, text, time-series; action-output modalities
> (motor, text); and true variable/missing-modality cross-attention** (the codec's
> earned-later item).

| Phase | Name | Key components | Kill-gate (one line) | Reqs |
|-------|------|----------------|----------------------|------|
| P12 | Swappable visual perception (first omni-modal seam) · **shipped** | frozen encoder → `UniversalCodec` VISION modality | the world model predicts over visual embeddings (48× better than persistence) and is surprised on novel frames (4.6×); a **better encoder swaps in** without retraining the core (1.05×, P0-011); built with deterministic stand-in encoders (CI numpy-only) — a real pretrained encoder swaps in via the same distill path | R6, R1, R3 |
| P13 | Learn from passive observation · **shipped** | action-free world model + **latent-action inference** (ADR-0010) | from an action-free stream: learns dynamics (beats persistence), recovers the hidden actions above chance (decorrelation identifiability fix), and **transfers** in the low-data regime (watch-first beats from-scratch at a small labelled budget) | R7, R1 |
| P14 | Observe → repeat (imitation) · **shipped** | `imitation.ObservationImitator` (recover actions from a demo's observations → clone a closed-loop policy) | from an expert's OBSERVATIONS ONLY (no actions), the agent reproduces a behaviour it never performed (recovery from observation, shuffled-demo control collapses, and it beats cloning its own random data — ADR-0012). Numpy-gated on `PendulumSwingup`; **demonstrated on real DMC swingup** where exploration can't reach (inverse-dyn 45.3 vs from-scratch 6.4). The P13 route is made reliable by **watch-then-ground** (ADR-0010): in the low-label regime it beats from-scratch inverse dynamics — watching as a low-data prior for control. Then **explore** (P3-002) fills what watching can't teach. | R5, R7 |

### Optional harder-benchmark tier (BH-001, ADR-0011) — non-gated
> A **fenced, non-gated** credibility probe that runs the *unchanged* core on real
> DeepMind Control Suite (MuJoCo) tasks through the `bench.Environment` seam, to test
> whether the authored-toy results survive foreign dynamics. It re-runs the **P2 claim**
> (MPC-over-a-learned-model vs a budget-matched model-free baseline) at P2's own settings.
> Deliberately outside the P-series: `dm_control` is the optional `[bench-hard]` extra,
> `bench/hard/` is never imported by the gate registry, and it runs only via
> `make bench-hard` (or a manual CI job) — the numpy-only `gate-all` ratchet is untouched.
> The deliverable is a committed **report** (`bench/hard/results/`), not a gate. See
> ADR-0011 for why the reproducible core CI stays numpy-only and this stays optional.
>
> The probe surfaced that **cartpole-swingup** fails (random data never reaches the upright
> goal), and the tier now chases that with two follow-up studies in the same report:
> **A** — does the curiosity curriculum (P3-002) fix it? *It reaches the goal region but
> can't convert sparse coverage to control — exploration necessary, not sufficient.*
> **B** — does imitation-from-observation reproduce it (P14, ADR-0012)? *Yes — watching an
> expert's observations + recovering its actions reproduces a swingup the agent never
> performed, at the budget where exploration fails.* The honest arc: explore reaches the
> region, a demonstration hands over the behaviour.

## Sequencing notes
- **Build the predictive core and its uncertainty estimate first** (P1). Every other
  requirement is a consumer of that core or a wrapper around it.
- **Introduce the episodic buffer + generative replay early** (P3): it is both the
  memory substrate and the anti-forgetting mechanism, so it earns its place twice.
- **The universal codec comes last** (P6) — but it is a **representation change**,
  not just an interface swap (P0-011): everything built in P1–P5 (the dynamics
  model, the option model, per-skill competence statistics, stored replay latents)
  is trained against the incumbent latent *distribution*, and a new encoder
  invalidates all of it unless matched. Migration strategy: **distill first** —
  train `UniversalCodec.encode` to match the incumbent encoder's outputs on shared
  modalities before swapping — with a **budgeted full-stack retrain as fallback**
  if distillation cannot hit the gate's tolerance. P6's cost estimate includes
  this; the Protocol makes the swap *typecheck*, distillation is what makes it
  *cheap*. (Prerequisite already in place: the replay buffer retains raw
  observations so experience stays re-encodable — see P3-003.)
- Treat **P7 as a discipline**, not a module: name the improvement metric, watch for
  forgetting via VoE, watch for plasticity loss.
- **Every phase gate also enforces its applicable collapse sentinels** (ADR-0006): a
  phase passes only if its capability criterion holds AND representation-integrity &
  uncertainty-reliability (active from P1), replay-fidelity (from P3) and
  option-diversity (from P5) are healthy. Collapse hides in a good loss, so integrity
  is measured, not assumed.
