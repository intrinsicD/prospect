# Backlog (ordered)

Take the top **unblocked** item. Each line: ID · status · requirement · blocked-by.
Expand a one-liner into a full task file (from `TEMPLATE.md`) when you pick it up.

> Every phase gate also enforces its applicable **collapse sentinels** (ADR-0006):
> `representation-integrity` & `uncertainty-reliability` (from P1), `replay-fidelity`
> (from P3), `option-diversity` (from P5). A task is not *done* if its phase's
> sentinels are unhealthy — integrity is part of the gate, not a nice-to-have.

## Phase 0 — contract & harness hardening (from the architecture review; pre-P1)
> These fix seams, not behaviour: cheap before P1, expensive after P3. All eleven
> have full task files. The two docs tasks (P0-010, P0-011) can run in parallel
> with anything.

- **P0-001** · `done` · R1,R3,R4 · — · `Prediction` parameterizes a real distribution: per-dim `var`, concrete `log_prob`, `duration` for option outcomes.
- **P0-002** · `done` · R3,R5,R7 · Decomposed `Surprise` type (no bare-float VoE) + `Transition.option` for per-skill attribution.
- **P0-003** · `done` · R1,R7 · `Learner` protocol — the uniform training seam the harness drives (P1 trains through it; P7's gate depends on it).
- **P0-004** · `done` · R1 · `Environment` protocol in `bench/` (harness-owned; core never imports the harness).
- **P0-005** · `done` · — · Run-metrics artifact (JSONL run log) — the data zero-arg sentinel `check()`s read to verify "throughout training" (ADR-0006).
- **P0-006** · `done` · — · Gate wiring: `@gate_check`/`@sentinel_check` registration, `metrics: dict`, persisted gate reports, register the P0 gate, friendly errors, explicit seed policy.
- **P0-007** · `done` · — · Regression ratchet: `bench/SHIPPED` + `make gate-all` + CI job — shipped gates stay green.
- **P0-008** · `done` · R8 · One query path into knowledge (`SemanticMemory` read-side *is* a `KnowledgeSource`), `route() -> KnowledgeSource | None` (None = parametric), provenance-`None` convention documented.
- **P0-009** · `done` · — · Enforce typing: mypy in CI, typed protocol-conformance assertions, ruff `I`, CI matrix 3.11–3.13.
- **P0-010** · `done` · R1,R3,R7 · ADR-0007: arbitration of the epistemic signal — curiosity seeks it, planning penalizes it; mode chosen by the curriculum. Plus shift-disambiguation note in ADR-0002. *(docs)*
- **P0-011** · `done` · R6 · Roadmap/ADR-0001 amendment: the P6 codec swap is a representation change — distill-first, retrain-fallback; replay keeps raw obs re-encodable. *(docs)*

> **Phase 0 complete** — all eleven contract & harness hardening tasks are `done`.

## Phase 1 — predictive core
- **P1-001** · `done` · R1,R4 · Flat latent world model + calibrated uncertainty. Gate **P1 PASS** (shipped; see task file for the report and the collapse-fighting lessons).

> **Phase 1 shipped** — `bench/SHIPPED` now ratchets P0 + P1.

## Phase 2 — planning
- **P2-001** · `done` · R1 · MPC/CEM planning in imagination; beat model-free at equal budget. Gate **P2 PASS** on every seed (shipped; see task file).
- **P2-002** · `done` · R1 · Composition root: `agent.py` act–observe loop + `bench.loop.run_episode` — one place the components meet; P2 gate reproduced byte-identically through it.

## Phase 3 — VoE, curriculum, replay
- **P3-001** · `done` · R3 · Calibrated surprise + decomposition + mastery test. Differential criterion **MET** (P(violated>expected) ≥ 0.93 every seed); P3 composite blocked pending P3-002 + P3-003 by design.
- **P3-002** · `done` · R3 · Curiosity curriculum. P3 **capability ok** (differential MET + curiosity MET: coverage ratio 0.26 vs 0.79 at equal budget); composite blocked only by the P3-003 sentinel.
- **P3-003** · `done` · R7 · Episodic replay + generative replay + `replay-fidelity` sentinel (real anchor 0.50, dream diversity 0.47, lineage ≤ 3, zero dreams stored). Gate **P3 PASS** (shipped).

> **Phase 3 shipped** — `bench/SHIPPED` now ratchets P0–P3. The one signal is
> live end-to-end: calibrated decomposed surprise, mastery, mode arbitration,
> curiosity, and collapse-guarded rehearsal.

## Phase 4 — skills
- **P4-001** · `done` · R5 · Skill router: simulate-to-select (accuracy 0.83–0.92 vs 0.33 baseline), paired closed-loop misapplication VoE (win rate ≥ 0.95), competence gating with calibrated mastery. `Option` gained typed `policy`/`horizon`; the precondition is *computed* (predictive), not stored. Gate **P4 PASS** (shipped).

> **Phase 4 shipped** — `bench/SHIPPED` now ratchets P0–P4.

## Phase 5 — hierarchy
- **P5-001** · `done` · R2 · Jumpy option-model (landing distribution, cumulative reward, duration): **beats the flat rollout 4–6x on every seed** — ADR-0003's compounding bound, measured. Composite blocked pending P5-002 by design.
- **P5-002** · `done` · R2 · Hierarchical manager (exhaustive search over the jumpy option-model) + VoE-triggered early termination + `option-diversity` sentinel. Two-level beats compute-matched flat on every seed (−9.1/−4.2/−4.7 vs −48.5/−34.9/−14.0), and beats full-compute flat too. Gate **P5 PASS** (shipped).

> **Phase 5 shipped** — `bench/SHIPPED` now ratchets P0–P5. Hierarchical
> *planning* (jumpy model + search + VoE termination) beats flat control at equal
> compute; all six phases green in ~3m30s.

## Phase 6 — any-to-any
- **P6-001** · `done` · R6 · Universal codec distilled into the incumbent latent (P0-011 migration, validated): swap-in 1-step MSE ratio ~1.0 for STATE **and** a rasterized IMAGE modality — the frozen core loop predicts from an image as from a state vector (any-to-any, measured). Gate **P6 PASS** (shipped).

> **Phase 6 shipped** — `bench/SHIPPED` now ratchets P0–P6. The distill-first
> migration (P0-011) is proven: the dynamics model is never retrained, yet its
> encoder swaps modality with <2% core-loop impact.

## Phase 7 — continual improvement
- **P7-001** · `done` · R7 · Continual improvement: `is_forgetting` (error-keyed, ADR-0002 amended) + rehearsal consolidation. Retention 3-5x better than naive; plasticity retained; naive loses both. Gate **P7 PASS** (shipped).

> **Phase 7 shipped** — `bench/SHIPPED` now ratchets P0–P7. The consolidation
> discipline preserves the memory AND plasticity that naive continual learning
> loses; forgetting detection keys on prediction error (the ensemble is
> confidently wrong under shift).

## Phase 8 — knowledge bases
- **P8-001** · `done` · R8 · Three-tier memory router + uncertainty-gated retrieval-as-action. Accuracy half **MET**: gated 1-step MSE ~3.2x lower than model-alone every seed (beats always-retrieve too), retrieving 55% of queries; all four sentinels healthy. P8 composite honestly BLOCKED pending P8-002.
- **P8-002** · `done` · R8 · Provenance/trust handling + poisoned/low-trust source robustness. `KnowledgeSource.trust` + trust-ordered routing with a `min_trust` floor: a trust-blind agent swallows the poison (5.4x worse than no-retrieval), the provenance-respecting router stays at no-retrieval (untrusted never overrides) and trust-orders to a trusted store to recover clean gated accuracy. Gate **P8 PASS** (shipped).

> **Phase 8 shipped** — `bench/SHIPPED` now ratchets P0–P8. Uncertainty-gated
> retrieval-as-action improves prediction where the model is uncertain, and provenance
> (trust-ordered selection + a `min_trust` floor) keeps a poisoned/low-trust source
> from ever overriding the agent — data, never instruction (ADR-0004).

> **P0–P8 complete.** Every roadmap phase has a passing kill-gate with its collapse
> sentinels healthy; the regression ratchet re-runs all nine in CI. The one signal —
> prediction error over a distribution with an epistemic/aleatoric split — is live
> end-to-end: prediction, planning, VoE/mastery/curiosity, skills, hierarchy,
> any-to-any I/O, continual improvement, and knowledge retrieval.

## Phase 9 — whole-system validation & hardening
> P0–P8 prove the *parts*. Phase 9 proves the *whole*, that each part is
> load-bearing, that the capabilities are not Pendulum artifacts, and that the gates
> measure capability rather than a calibrated trivial solution (ADR-0008).

- **P9-001** · `done` · R1–R8 · End-to-end integration gate. The composed agent controls the task (−19.4 vs random −73.1); in ONE run the single epistemic signal drives the planner's exploit coefficient AND retrieval; four sentinels healthy. Gate **P9 PASS** (shipped). **Finding:** retrieval-*into-planning* degrades control (−19.4 vs −8.1 off) — handed to P9-002.
- **P9-002** · `done` · R1–R8 · Ablation harness (leave-one-out marginal control value), folded into the P9 gate. Verdicts: planning **+53.7 load-bearing** (gated every seed), retrieval **−9.5 harmful** (the P9-001 finding, quantified), exploit_penalty **+2.5 negligible**. Two under-performers surfaced as findings, not tuned away. Gate **P9 PASS**.
- **P9-003** · `done` · R1,R4,R8 · Second environment (`PointMass`, 2D nonlinear-drag; obs 3→4, action 1→2) + cross-env generalization, folded into the P9 gate. **Prediction and planning generalize** with the same core (recalibrated eval params only). **Finding:** retrieval does NOT generalize (confidently-wrong OOD → gate never fires) — its benefit is env-dependent. Gate **P9 PASS**.
- **P9-004** · `done` · R1–R8 · Metamorphic invariants + per-gate negative controls + statistics hardening, as a standing `gate-overfit` sentinel (active from P9): 7 cheap checks — trivial solutions (always-retrieve, one-step options) FAIL their criteria; invariants hold (surprise decomposition exact, untrusted never overrides, log-prob peaks at mean); a bootstrap CI separates a real margin from noise. Sentinel **healthy**; P9 **PASS**.
- **P9-005** · `done` · R1,R3,R4,R7,R8 · Distance-aware epistemic uncertainty — the fix for the P9-003 finding. Diagnosed: the tanh encoder saturates, so ensemble disagreement can't detect OOD (epi rose 1.75x while error rose 10x on PointMass). Fix: `encode` computes a pre-encoder OOD score on the standardized input (`LatentState.ood`), `predict` scales epistemic by it. Result: OOD epi-rise 1.75x→7.85x, epi-vs-error rank corr 0.52→0.80; **the uncertainty signal now generalizes** (folded into the P9 gate, floor 3.0, measured 8.8). **New finding (hypothesis, later disproved):** retrieval still doesn't generalize — P9-005 guessed the same saturation corrupts the retrieval *key* space; **P9-006 measured it and found it was store density, not the key.** ADR-0002 amended. Gate **P9 PASS**.
- **P9-006** · `done` · R8,R1,R4 · Retrieval generalization via a dimension-adequate store — the fix for the P9-005 retrieval finding, and a correction of its cause. Measured: the latent key is fine (it even beats a raw standardized-input key); retrieval failed to generalize because the store was too *sparse* for its 6-D key space (the curse of dimensionality). Fix: `STORE_N` 1500→40000 (dimension-adequate); retrieval now generalizes (gated 0.0135 vs no-retrieval 0.0172, ~22% better) and is **gated** — the P9 cross-env check now covers prediction + planning + uncertainty + retrieval. ADR-0004 amended (curse-of-dimensionality consequence). Gate **P9 PASS**.
- **P9-007** · `done` · R1,R8,R4 · Retrieval-into-planning composition fix. Diagnosed the P9-002 harmful-retrieval marginal: inside CEM rollouts the query is an imagined latent far from any real fact (median key-dist ~7× a covered query), so the nearest fact is fiction, and marking the retrieved row `epi=0` deleted the exploit penalty exactly where the model was least reliable — luring CEM into the retrieval seam. Fix: **distance-gated retrieval** — substitute a fact only within a coverage-calibrated `reliability_radius`, with honest distance-scaled epistemic (never 0). Measured: retrieval marginal −3.1→**−0.3 (negligible, safe, now gated)**, composed control −23.6→−9.7, and the entangled exploit-penalty marginal −6.0→−1.6. ADR-0004 amended (composition rule). Gate **P9 PASS**.

> **Phase 9** — P9-001..007 shipped (all fold into the P9 gate; `bench/SHIPPED` ratchets
> P0–P9). The whole system is verified end-to-end (not just per part), a leave-one-out
> ablation quantifies every component's marginal value, the core capabilities — now
> *including the epistemic signal itself* (P9-005 distance-aware fix) *and retrieval*
> (P9-006 dimension-adequate store) — generalize to a second environment, and a standing
> `gate-overfit` sentinel keeps the gates from measuring artifacts or noise. The
> whole-system layer surfaced two naive-composition harms and then drove their fixes:
> retrieval degrading multi-step control, and its entanglement with the exploit penalty,
> both fixed by P9-007's distance-gating (retrieve into planning only trustworthy close
> facts, honest distance-scaled epistemic) and now gated at negligible marginals. Honest
> remainder: retrieval into planning earns little on this task (safe, not load-bearing);
> the exploit-penalty is negligible. P9-005's key-saturation guess for retrieval's
> non-generalization was measured and corrected to store density (curse of
> dimensionality) in P9-006.

> **P0–P9 complete.** Every roadmap phase ships with a passing kill-gate and healthy
> collapse sentinels; the ratchet re-runs all ten. Beyond the capabilities (P0–P8), the
> whole system is validated as an assembled agent, each part is measured for its keep,
> generalization is checked on a second environment — prediction, planning, the epistemic
> signal (P9-005) AND retrieval (P9-006) all survive it — and the gates are guarded
> against overfit and noise. The naive-composition harms the whole-system layer surfaced
> (retrieval degrading control; its exploit-penalty entanglement) were driven to
> negligible by P9-007's distance-gating. What remains honest to say is narrower:
> retrieval into planning earns little here, and the exploit-penalty is negligible — the
> map of where the scaffold's real work remains.

## Phase 10 — external knowledge & tools (beyond the toy loop)
> P0–P9 prove the predictive agent on toy control. Phase 10 opens the **external**
> knowledge tier: knowledge the agent never experienced, entering through the codec
> (ADR-0004 rule 1) — the step toward a real use case (R8). Option A (external knowledge
> base) first; Option B (compute-as-action tools) is a later phase.

- **P10-001** · `done` (capability; composite blocked pending P10-002) · R8,R6 · External knowledge through the codec. `ExternalKnowledgeSource` answers with raw content (an observation the agent never sensed) that it ingests via `codec.encode` (rule 1, first exercised), extending competence to an OOD band the model can't extrapolate: gated MSE 3.4× below model-alone, seen no-harm, corrupting the retrieved observation worsens it 50× (the answer flows through the codec). **Finding:** the uncertainty gate alone let seen false-consults fetch irrelevant OOD facts and hurt — the P9-007 distance gate is needed at the external tier too (consult-when-uncertain AND trust-when-close). Capability **MET**, all sentinels healthy; composite P10 BLOCKED pending P10-002.
- **P10-002** · `done` · R8 · External-source trust robustness. A poisoned UNTRUSTED external source (corrupted observations over the same keys) is the attack surface: a trust-blind agent ingests it through the codec and does ~40× worse than no-retrieval (1.029 vs 0.0255); the provenance-respecting router never lets it override (stays 0.0255) and trust-orders to a trusted source to recover clean accuracy (0.0076). The defense is provenance, not content inspection (reuses P8-002). Composite **P10 PASS**.

> **Phase 10 shipped** — `bench/SHIPPED` now ratchets P0–P10. The external knowledge tier
> is live: the agent answers OOD queries it can't derive from experience by retrieving
> external *content* and ingesting it through the codec (ADR-0004 rule 1, first
> exercised), gated by uncertainty (when to consult) AND distance (P9-007 — when to
> trust), and provenance keeps a poisoned source from ever overriding it. Next: Option B
> (compute-as-action tools, `ToolSource`) and/or a real (non-toy) environment.

## Phase 11 — compute-as-action tools
> The third knowledge tier (ADR-0004 rule 2): a tool the agent *calls*. Unlike a lookup
> KB, a tool **computes** its answer — exact for any query, but each call costs. So the
> decision is cleanly about uncertainty AND cost.

- **P11-001** · `done` · R8,R1 · Compute-as-action tools. `ToolSource` wraps a harness-supplied compute function (an exact next-state oracle) and counts calls (the cost signal); the tool result ingests through the codec (reusing P10). Uncertainty-gated tool-use: on OOD the tool beats the model ~200×; the uncertainty signal spends an equal call budget far better than random (it calls where model error — the benefit — is largest); and gating is the cost sweet spot (better than never-calling, fewer calls than always-calling). Gate **P11 PASS**; ships (`bench/SHIPPED` ratchets P0–P11).

## Phase 12+ — omni-modal seams & learning from observation · ADR-0009 accepted
> **Universal adaptable seams, specialized per deployment, one modality per gate**
> (ADR-0009). The codec admits any input/output modality into the shared latent; a
> deployment (private, industrial, science, robotics) instantiates and trains the subset it
> needs — same architecture, specialized weights. Then the agent learns from watching
> (observe→repeat→explore = ADR-0007's curriculum); the one genuinely new component is
> **latent-action inference** (learning *behavior*, not just physics, from action-free
> video). Honest walls: real modules need pretrained backends (harness-side, optional — the
> core stays numpy over embeddings), and real-video *at scale* is a runtime concern, so
> gates run on committed fixtures and scale is *demonstrated*, not gated. **Future seams,
> each its own gate, added on demand:** audio · proprioception · force · text · time-series
> · action-output modalities (motor, text) · true variable/missing-modality cross-attention.

- **P12-001** · `done` · R6,R1,R3 · Swappable visual perception — **the first omni-modal seam**. A frozen encoder turns a frame into an embedding; the codec's VISION modality distils it into the shared latent; the world model predicts over what it sees (48× better than persistence) and is **surprised** on novel frames (4.6×); and a **second, different encoder swaps in without retraining the core** (1.05× — P0-011). Built Path B: deterministic stand-in encoders on rendered rotating-blob clips (pure numpy, CI stays numpy-only, no fixture files); a real pretrained encoder on real frames swaps in via the same distill path (the `[vision]` regen + live-webcam demo). Gate **P12 PASS**, all sentinels healthy; ships (`bench/SHIPPED` ratchets P0–P12). (ADR-0009.)

> **Phase 12 shipped** — the first omni-modal seam is live: the agent sees through a
> (swappable) frozen encoder and predicts over what it sees, understanding it *predictively*
> (surprised when wrong). The mechanism — see → codec → shared latent → predict → swap →
> surprise — is proven in a numpy gate; real vision (a pretrained encoder on real content)
> and the live-webcam demo ride the same seam. Next in the arc: **P13 — learn from passive
> observation** (action-free world model + latent-action inference), then P14 (observe →
> repeat). Other seams (audio, proprioception, text, action-out, cross-attention) are
> named future gates, added on demand.
- **P13-001** · `backlog` · R7,R1 · Learn from passive observation: action-free world model + **latent-action inference** (infer the action between frames, Genie/LAPO-style) — learn dynamics AND behavior from a stream with no actions/rewards; gate transfer + latent-action recovery. Its own ADR when scoped.
- **P14-001** · `backlog` · R5,R7 · Observe → repeat: reproduce a demonstrated behavior the agent never performed itself (imitation-from-observation) via the planner + inferred latent actions; **explore** (P3-002) then fills what watching can't teach. Runtime layer on top: real-YouTube ingestion + live-webcam demo (non-gated).
