# SOTA review — July 2026 (component-by-component literature check)

A full review of every architecture component against the 2023–2026 literature
(arXiv, GitHub, OpenReview), answering one question per component: *is this part
clearly outdated, and if so what replaces it — or is it already at (or ahead of)
current best practice?* Every finding became a task: **U-001…U-004 are shipped**, ready
upgrades are **U-005…U-012**, and deferred (trigger-gated) upgrades are
**U-101…U-112** — see the upgrade track in `tasks/BACKLOG.md` and the
**upgrade-triggers** workflow step in `CLAUDE.md`.

## Headline verdict

The architecture is *not* broadly outdated. No current SOTA world model
(DreamerV3/Dreamer 4, TD-MPC2, V-JEPA 2, IRIS/STORM, DIAMOND, Genie 3) natively
produces the epistemic/aleatoric split all six VoE jobs consume; where the field
needs an epistemic signal it bolts on exactly what this repo already has — a
probabilistic ensemble. There is no wholesale replacement to make. Twelve actionable
upgrades were identified (U-001…U-004 now shipped; eight remain ready), a separate dozen
are right-but-not-yet (deferred with explicit triggers), and three code/doc
inconsistencies surfaced during the read.

## Measurably behind → upgrade tasks (U-001…U-004 shipped)

| Task | Component | Finding (why replace) | Key sources |
|------|-----------|----------------------|-------------|
| U-001 · **shipped** | world_model/planning rollouts | Mean-latent imagination understated multi-step uncertainty; per-member TS∞ propagation + optional accumulated-epistemic truncation now ships, guarded by a strict OOD horizon-spread sentinel | [PETS](https://arxiv.org/abs/1805.12114) · [MACURA](https://arxiv.org/abs/2405.19014) · [Infoprop](https://arxiv.org/abs/2501.16918) |
| U-002 · **shipped** | FlatPlanner | Vanilla white-noise CEM was replaced by beta-2 colored proposals, keep/shift elites, execute-best, and softmax-weighted elite moments; a reference iCEM proposal scale prevents correlated trajectories saturating the action bounds | [iCEM](https://arxiv.org/abs/2008.06389) · [Pink Noise](https://openreview.net/forum?id=hQ9V5QN27eS) · [TD-MPC2](https://arxiv.org/abs/2310.16828) |
| U-003 · **shipped** | voe/planning/memory thresholds | One-shot termination-surprise and epistemic-retrieval cutoffs became separate decaying-step ACI policies with independent nominal audits; P9 planning uses a measurable 0.01% tail over 100k-score calibration/audit streams because CEM amplifies one-step crossings, while the forgetting latch stays fixed | [ACI](https://arxiv.org/abs/2106.00170) · [decaying-step ACI](https://arxiv.org/pdf/2402.01139) · [conformal failure detection](https://arxiv.org/pdf/2503.08558) |
| U-004 · **shipped** | ReplayBuffer eviction | FIFO-only eviction became a fixed-budget, disjoint 60/40 recent-FIFO + Algorithm-R lifetime reservoir; a 10×-capacity churn test retains early history while P3/P7 and the full ratchet stay green | [WMAR](https://arxiv.org/abs/2401.16650) · [accumulate-don't-replace](https://arxiv.org/abs/2404.01413) |
| U-005 | retrieval readout | Nearest-1 hard substitution is noise- and poison-sensitive; k=2–3 distance-kernel-weighted blending against the model's own prediction is the converged answer of the kNN-LM, episodic-control and RAG-security literatures | [kNN-LM gating](https://arxiv.org/abs/2210.15859) · [PoisonedRAG](https://arxiv.org/abs/2402.07867) · [RobustRAG](https://arxiv.org/abs/2405.15556) |
| U-006 | world_model training | One-step training + multi-step consumption is the classic compounding-error mismatch (the repo's own named "main limiter on R1"); current practice adds an unrolled multi-step loss term | [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) |
| U-007 | OOD/epistemic signal | The pre-encoder OOD score (P9-005) is a homebrew of the documented "feature collapse" fix but lives before the encoder; a latent-space Mahalanobis density is the literature's feature-space complement | [SNGP](https://arxiv.org/abs/2006.10108) · [DDU](https://arxiv.org/abs/2102.11582) |
| U-008 | gates | Latent-space ensemble disagreement develops attractors and must be validated against ground-truth state-space error — only a toy-env repo can gate this cheaply | [Biased Dreams](https://arxiv.org/abs/2604.25416) |
| U-009 | gates (latent actions) | The published identifiability test for latent-action models (endpoint-swap must spike forward error) is the right negative control for the decorrelation penalty | [Garrido et al. 2026](https://arxiv.org/abs/2601.05230) |
| U-010 | observation (LAM) | Post-hoc-only grounding is the weaker variant: tiny amounts of action supervision *during* latent-action training give multi-x downstream gains | [LAOM](https://arxiv.org/abs/2502.00379) · [CLAM](https://arxiv.org/abs/2505.04999) |
| U-011 | HierarchicalManager | Two internal inconsistencies: the epistemic penalty is undiscounted in the hierarchy (discounted in the flat planner), and option termination gates on undecomposed total NLL (fires on aleatoric noise) against ADR-0002's own rule | code review (planning.py) |
| U-012 | docs | "Perceiver-IO-style" mislabels the codec (it is adapter-alignment into a shared space + textbook backward-compatible training); ADRs lack the rejected-alternative and validation citations this review surfaced | [BCT](https://arxiv.org/abs/1912.03373) · [ImageBind](https://arxiv.org/abs/2305.05665) |

## Already at (or ahead of) best practice — keep, do not replace

- **Probabilistic ensemble for the epistemic/aleatoric split** — still the 2026
  standard; evidential deep learning is a documented trap (epistemic that doesn't
  shrink with data: [NeurIPS 2024 "Mirage"](https://arxiv.org/abs/2402.06160)).
- **EMA target + VICReg + effective-rank sentinel** — the mainstream anti-collapse
  recipe; the sentinel is [RankMe](https://arxiv.org/abs/2210.02885).
  ([LeJEPA/SIGReg](https://arxiv.org/abs/2511.08544) is the credible future
  *simplification* — deferred as U-102.)
- **Epistemic-only curiosity bonus (noisy-TV defense)** — validated by
  [AMA](https://arxiv.org/abs/2102.04399); disagreement exploration remains the
  substrate of 2025–26 work ([MaxInfoRL](https://arxiv.org/abs/2412.12098)).
- **LP-EMA mastery + error-keyed forgetting detection** — faithful minimal
  instance of the LP-curriculum lineage; "ensembles are confidently wrong under
  shift" is literature-supported; the Nature 2024 plasticity paper proposes no
  online detector, so error-rise detection is not behind.
- **Generative replay (real-anchored, never-store-dreams, epistemic-gated)** —
  precisely the one regime the model-autophagy literature certifies safe
  ([MAD, ICLR 2024](https://arxiv.org/abs/2307.01850)); the dream gate is
  MACURA-class rollout control avant la lettre.
- **Retrieval-as-action gated on epistemic; untrusted content is data, never
  instruction** — the LLM world independently converged on both
  ([FLARE](https://arxiv.org/abs/2305.06983), [CoALA](https://arxiv.org/abs/2309.02427),
  [CaMeL](https://arxiv.org/abs/2503.18813)); the separation here is *structural*
  (facts are numeric latents with no instruction channel), stronger than
  training-based LLM defenses. Exact NN is correct below ~10⁵ items.
- **Continuous latent actions + decorrelation (not VQ)** — the 2025–26 evidence
  favors this deviation from the LAPO/Genie VQ orthodoxy
  ([Garrido et al. 2026](https://arxiv.org/abs/2601.05230),
  [CLAM](https://arxiv.org/abs/2505.04999)).
- **Inverse-dynamics + BC imitation from observation** — still best practice at
  one-demo/low-dim/no-RL scale; foundation-scale systems (UniPi, LAPA) kept the IDM.
- **Jumpy option model + exhaustive K^depth search + VoE termination** — at or
  ahead: [FAIR's 2026 compositional jumpy planner](https://arxiv.org/abs/2602.19634)
  validates the architecture and still uses random shooting (exhaustive K³ is
  exact and cheaper at small K); VoE-triggered termination is more adaptive than
  Director's fixed-K switching.
- **Linear epistemic penalty (MOPO-form)** — still standard without a value
  function; penalty *form* matters less than the estimate
  ([ICLR 2022](https://arxiv.org/pdf/2110.04135)).
- **Distill-into-incumbent codec migration** — textbook
  [BCT](https://arxiv.org/abs/1912.03373); the documented BCT quality ceiling is
  exactly why the P0-011 retrain-fallback exists.

**Why not the big swaps:** DreamerV3/4, TD-MPC2-as-architecture, IRIS/STORM,
DIAMOND, Genie-class models were evaluated and rejected: (1) none produces an
epistemic/aleatoric split — adopting them deletes the one load-bearing idea;
(2) their machinery (reconstruction decoders, tokens, transformers, diffusion)
contradicts working ADR-0001/0006 choices; (3) they are 3–6 orders of magnitude
off in scale. Dreamer-style amortized actor-critic additionally removes
decision-time planning — the place the VoE signal plugs in. Deliberate forks,
not negligence.

## Right upgrade, wrong time → deferred tasks (trigger-gated)

Each deferred task file carries a **Trigger** — a measurable condition. The
**upgrade-triggers** workflow step (`CLAUDE.md`) re-checks these at every
docs-sync; a task is promoted to `ready` only when its trigger is observed, per
ADR-0005's "generality is earned by a gate".

| Task | Upgrade | Trigger (promote when…) | Key sources |
|------|---------|-------------------------|-------------|
| U-101 | TD-learned terminal value bootstrapping the CEM score | a *gated* sparse-reward / long-horizon task enters the roadmap (e.g. swingup promoted from the BH tier) | [TD-MPC2](https://arxiv.org/abs/2310.16828) · [TD-M(PC)2](https://arxiv.org/abs/2502.03550) |
| U-102 | LeJEPA/SIGReg replacing EMA+stop-grad+VICReg (SimNorm fallback) | the anti-collapse stack blocks a gate or needs per-task retuning; or a deliberate simplification sprint | [LeJEPA](https://arxiv.org/abs/2511.08544) · [TD-MPC2 SimNorm](https://arxiv.org/abs/2310.16828) |
| U-103 | Epistemic-prioritized replay sampling | a nonstationarity/adaptation gate exists, or a continual gate shows uniform sampling as the limiter | [UPER](https://arxiv.org/abs/2506.09270) · [Curious Replay](https://arxiv.org/abs/2306.15934) |
| U-104 | CEM/beam search over option sequences | the option library outgrows exhaustive K^depth (≈ K > 6 at depth 3, or manager latency hurts a gate) | [SkiMo](https://arxiv.org/abs/2207.07560) · [TAP](https://arxiv.org/abs/2208.10291) |
| U-105 | Last-layer Laplace epistemic | measured evidence 5-member disagreement is too coarse (epistemic-vs-error rank corr fails a floor despite U-007) | [laplax](https://arxiv.org/abs/2507.17013) · [CLAPS](https://arxiv.org/abs/2512.01384) |
| U-106 | CUSUM/change-point option termination | gate metrics show termination chattering (false interrupts on single-step aleatoric spikes) | [CPD-HRL](https://arxiv.org/abs/2510.24988) |
| U-107 | Continual backprop + plasticity diagnostics | a plasticity gate exists, or P7-class results show failure to re-learn after detected forgetting | [Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7) · [ReDo](https://arxiv.org/abs/2302.12902) |
| U-108 | Episodic→semantic consolidation pathway | a gate needs facts distilled from the agent's own experience (today the harness writes facts directly) | [CoALA](https://arxiv.org/abs/2309.02427) · agent-memory practice |
| U-109 | Predict-then-invert imitation (PIDM) | an imitation gate is marginal at its demo budget | [Schäfer et al., ICML 2026](https://arxiv.org/abs/2601.21718) |
| U-110 | Unsupervised skill discovery (METRA-class) | the roadmap adds a skill-discovery phase (options no longer harness-authored) | [METRA](https://arxiv.org/abs/2310.08887) |
| U-111 | Jumpy-model cross-timescale consistency loss | compounding jump error is measured at depth (option-model rollout error grows superlinearly in jumps) | [FAIR jumpy planning 2026](https://arxiv.org/abs/2602.19634) |
| U-112 | FCT-style migration adapter (old→new latent) | codec distillation misses a gate's tolerance (the P0-011 retrain-fallback is about to fire) | [FCT](https://arxiv.org/abs/2112.02805) |

## Review provenance

Conducted 2026-07-10 on the shipped P0–P14 state, via five parallel literature
sweeps (world models & uncertainty; planning & hierarchy; intrinsic
motivation/VoE/calibration; memory/replay/retrieval; codec/latent
actions/imitation). Load-bearing URLs were fetched and verified individually.
