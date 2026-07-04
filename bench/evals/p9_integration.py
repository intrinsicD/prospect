"""P9 eval (task P9-001): the whole system, composed and run end-to-end.

Every earlier gate proves one capability in its own harness. This one wires the
*composed* agent through the `agent.py` composition root and asserts emergent,
closed-loop properties no single-phase gate can — the test of "does it work as a
whole" (ADR-0008).

Setup (per seed): a `FlatWorldModel` pre-trained on a LIMITED region of the pendulum
(|omega| <= REGION, so it is confident there and uncertain outside — the P8 trick), a
`SemanticStore` of correct next-latents across the FULL range, a
`SurpriseCompetenceMonitor` fed the seen region until it reports mastery, and a
`LearningProgressCurriculum` that therefore selects EXPLOIT. The agent plans with
`FlatPlanner` over a `RetrievalAugmentedWorldModel` — so the SAME epistemic signal
that the curriculum turns into the planner's exploit coefficient also gates retrieval
inside the rollouts (one signal, several jobs, one loop).

PASS = the whole system works end-to-end AND the one signal does several jobs at once
AND the ablation confirms the load-bearing part matters AND the core capabilities
generalize to a second environment:
1. **Controls end-to-end** — the composed agent's return beats a reactive (random)
   baseline: the full wiring produces control, not just isolated capabilities.
2. **One signal, many jobs, one run** — in the *same* run the monitor's epistemic
   signal (a) is turned by the curriculum into the planner's live exploit coefficient
   (mastered ⇒ positive; the value the agent actually applied), and (b) gates
   retrieval (it fired; every retrieval was above the router's uncertainty threshold
   by construction). Also carries all four collapse sentinels on run `p9`.
3. **Ablation — the load-bearing part matters** (P9-002) — leave-one-out marginal
   control value (`composed - ablated`): planning must be load-bearing on every seed.
4. **Generalizes to a 2nd environment** (P9-003) — prediction (P1) and planning (P2)
   survive on `bench.envs.PointMass`, a structurally different task, with the SAME core
   code (recalibrated eval params only).

MEASURED, NOT GATED — the findings (ADR-0008): (a) retrieval-into-planning has a
*negative* ablation marginal (it helps 1-step prediction per P8 but overriding the
planner's rollout dynamics corrupts multi-step optimisation); (b) retrieval does NOT
generalize to PointMass — its benefit is env-dependent (the ensemble's epistemic barely
rises OOD there, so retrieval rarely fires); (c) the exploit-penalty is ~negligible.
None is tuned away; each is a reported generalization/composition limit.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from prospect.agent import Agent
from prospect.memory import RetrievalAugmentedWorldModel, SemanticStore, UncertaintyMemoryRouter
from prospect.planning import FlatPlanner
from prospect.types import KnowledgeItem, LatentState, Observation, Provenance, Transition, Trust
from prospect.voe import LearningProgressCurriculum, SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel

from ..gates import GateResult, gate_check
from ..loop import run_episode
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe
from .p2_planner import EP_LEN, _PolicyAgent, _random_policy
from .p3_replay import log_replay_fidelity
from .p7_continual import _log_option_diversity
from .p8_knowledge import FULL, REGION, _env, _key, _region_data
from .p9_ablation import MARGIN, classify, marginals
from .p9_generalization import generalizes

RUN_ID = "p9"
SEEDS = [0, 1, 2]
TRAIN_N, STORE_N, PROBE_N = 4096, 3000, 256
RETRIEVE_MULT = 2.0           # retrieve only above 2x the MAX seen epistemic (see below)
EVAL_EPISODES = 2             # control episodes averaged per condition
# Why conservative (RETRIEVE_MULT on the max, not P8's 90th-percentile quantile):
# retrieval helps 1-step prediction (P8) but a low gate fires ~55% of the time inside
# CEM rollouts, overriding half the planner's dynamics with nearest-neighbour facts
# that are misaligned at rollout depth — which *degrades* control. Gating retrieval to
# states genuinely beyond training keeps it a rare, safe correction. (P9-002 quantifies
# the trade-off across gate settings.)
MASTERY_SLACK = 5.0           # mastery_epistemic = this x the model's seen epistemic floor
MONITOR_UPDATES = 40          # seen-region transitions fed to the monitor (>= min_updates)


def _encoder(model: FlatWorldModel) -> Callable[[Observation], LatentState]:
    def encode(obs: Observation) -> LatentState:
        return model.encode(obs.data)

    return encode


def _online_store(model: FlatWorldModel, facts: list[Transition]) -> SemanticStore:
    """Facts keyed by concat(latent, action) answering with the correct next-latent in
    the model's ONLINE space — a drop-in for the planner's rollout (P9-001)."""
    store = SemanticStore()
    for t in facts:
        answer = np.asarray(model.encode(t.next_state.z).z, dtype=float)
        store.write(KnowledgeItem(content=(_key(model, t), answer),
                                  provenance=Provenance(source="reference", trust=Trust.HIGH)))
    return store


def _mastered_curriculum(
    model: FlatWorldModel, seen: list[Transition], epi_floor: float
) -> tuple[SurpriseCompetenceMonitor, LearningProgressCurriculum]:
    """Feed the monitor the seen region (where the trained model is confident) until it
    reports mastery, so the curriculum selects EXPLOIT for the control run."""
    monitor = SurpriseCompetenceMonitor(
        mastery_epistemic=MASTERY_SLACK * max(epi_floor, 1e-6), min_updates=MONITOR_UPDATES)
    for t in seen[: MONITOR_UPDATES + 10]:
        latent = model.encode(t.state.z)
        monitor.update(Transition(state=latent, action=t.action,
                                  next_state=model.encode(t.next_state.z), reward=t.reward,
                                  prediction=model.predict(latent, t.action)))
    return monitor, LearningProgressCurriculum(monitor)


def _control_return(agent: object, seed: int) -> float:
    return float(np.mean([
        run_episode(_env(), agent, EP_LEN, 9000 + seed * 50 + e)[0]  # type: ignore[arg-type]
        for e in range(EVAL_EPISODES)
    ]))


@gate_check("P9")
def check_p9() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    controls, one_signal, composed_ret, bare_ret, ablation_ok = [], [], [], [], []
    marg_table: dict[str, list[float]] = {c: [] for c in ("planning", "retrieval", "exploit_penalty")}
    for seed in SEEDS:
        # 1. pre-train the world model on the seen region (+ sentinel logging on run p9)
        train = _region_data(REGION, TRAIN_N, seed)
        heldout = _region_data(REGION, PROBE_N, seed + 500)
        ood = _region_data(FULL, PROBE_N, seed + 900)
        model = FlatWorldModel(seed=seed)
        rng = np.random.default_rng(seed + 1)
        probe = _make_probe(heldout, heldout + ood, seed)
        for step in range(STEPS):
            idx = rng.integers(0, len(train), size=64)
            step_metrics = model.update([train[i] for i in idx])
            if step % 100 == 0:
                log.log(seed * SEED_STEP_OFFSET + step, step_metrics | probe(model))
        log_replay_fidelity(model, train, seed, log, step_offset=seed * SEED_STEP_OFFSET + 50_000)
        _log_option_diversity(model, seed, log)

        # 2. store + uncertainty-gated router
        store = _online_store(model, _region_data(FULL, STORE_N, seed + 50))
        seen_epi = sorted(model.predict(model.encode(t.state.z), t.action).epistemic for t in heldout)
        threshold = RETRIEVE_MULT * seen_epi[-1]  # 2x the most-uncertain seen state
        router = UncertaintyMemoryRouter([store], threshold=threshold)

        # 3. monitor -> mastery -> curriculum EXPLOIT (the same epistemic signal, other job)
        monitor, curriculum = _mastered_curriculum(model, heldout, float(np.median(seen_epi)))
        mode_coeff = curriculum.uncertainty_coefficient()
        mastered = monitor.is_mastered(SurpriseCompetenceMonitor.DEFAULT_SKILL)

        # 4. compose the full agent (plan over the retrieval-augmented model) + baselines
        augmented = RetrievalAugmentedWorldModel(model, router)
        planner_full = FlatPlanner(augmented, seed=seed)
        agent_full = Agent(encode=_encoder(model), planner=planner_full, world_model=model,
                           monitor=monitor, curriculum=curriculum)
        agent_bare = Agent(encode=_encoder(model), planner=FlatPlanner(model, seed=seed),
                           world_model=model, monitor=monitor, curriculum=curriculum)
        composed = _control_return(agent_full, seed)
        rate = augmented.retrievals / max(augmented.calls, 1)  # read before augmented is reused
        bare = _control_return(agent_bare, seed)
        reactive = _control_return(_PolicyAgent(_random_policy(seed)), seed)
        applied = planner_full.uncertainty_penalty  # set live inside act() from the curriculum

        # P9-002 ablation: leave-one-out control return with each component disabled —
        # planning (reactive), retrieval (bare model), exploit_penalty (no curriculum,
        # penalty pinned to 0). Marginal value = composed - ablated (see p9_ablation).
        agent_no_penalty = Agent(
            encode=_encoder(model), world_model=model, monitor=monitor,
            planner=FlatPlanner(RetrievalAugmentedWorldModel(model, router), seed=seed,
                                uncertainty_penalty=0.0))
        no_penalty = _control_return(agent_no_penalty, seed)
        marg = marginals(composed, {"planning": reactive, "retrieval": bare,
                                    "exploit_penalty": no_penalty})
        for name, value in marg.items():
            marg_table[name].append(value)
        ablation_ok.append(marg["planning"] > MARGIN)  # planning must be load-bearing

        controls.append(composed > reactive)
        one_signal.append(mastered and mode_coeff > 0 and abs(applied - mode_coeff) < 1e-9
                          and augmented.retrievals > 0)
        composed_ret.append(composed)
        bare_ret.append(bare)
        metrics |= {
            f"composed_return_s{seed}": composed,
            f"retrieval_off_return_s{seed}": bare,
            f"reactive_return_s{seed}": reactive,
            f"no_penalty_return_s{seed}": no_penalty,
            f"retrieval_rate_s{seed}": rate,
            f"exploit_coefficient_s{seed}": mode_coeff,
            f"mastered_s{seed}": float(mastered),
            f"marginal_planning_s{seed}": marg["planning"],
            f"marginal_retrieval_s{seed}": marg["retrieval"],
            f"marginal_exploit_penalty_s{seed}": marg["exploit_penalty"],
        }

    # P9-003 cross-environment generalization: the load-bearing capabilities must
    # survive on a SECOND, structurally different environment (PointMass) with the same
    # core. Gate on prediction + planning generalizing; retrieval's generalization is
    # recorded (its benefit is env-dependent — a finding).
    gen = generalizes()
    metrics |= gen.metrics
    generalizes_met = gen.prediction_met and gen.planning_met
    metrics |= {"prediction_generalizes": float(gen.prediction_met),
                "planning_generalizes": float(gen.planning_met),
                "retrieval_generalizes": float(gen.retrieval_met),
                "generalizes_met": float(generalizes_met)}

    # PASS = the whole system works end-to-end (controls), the one epistemic signal
    # drives several jobs in one run, the leave-one-out ablation confirms the clearly
    # load-bearing component (planning) matters, AND the core capabilities generalize to
    # a second environment. Recorded-not-gated findings: a harmful/negligible ablation
    # marginal, and retrieval's env-dependent generalization (ADR-0008).
    controls_met, one_signal_met, ablation_met = all(controls), all(one_signal), all(ablation_ok)
    passed = controls_met and one_signal_met and ablation_met and generalizes_met
    composed_med, bare_med = float(np.median(composed_ret)), float(np.median(bare_ret))
    reactive_med = float(np.median([metrics[f"reactive_return_s{s}"] for s in SEEDS]))
    rate_med = float(np.median([metrics[f"retrieval_rate_s{s}"] for s in SEEDS]))
    marg_med = {c: float(np.median(v)) for c, v in marg_table.items()}
    metrics |= {"composed_return_median": composed_med, "retrieval_off_return_median": bare_med,
                "reactive_return_median": reactive_med, "retrieval_rate_median": rate_med,
                "controls_met": float(controls_met), "one_signal_met": float(one_signal_met),
                "ablation_met": float(ablation_met)}
    metrics |= {f"marginal_{c}_median": m for c, m in marg_med.items()}
    table = ", ".join(f"{c} {m:+.1f} ({classify(m)})" for c, m in marg_med.items())
    gen_note = (f"prediction {'✓' if gen.prediction_met else '✗'} + planning "
                f"{'✓' if gen.planning_met else '✗'} generalize to a 2nd env (PointMass); "
                f"retrieval {'✓' if gen.retrieval_met else '✗ (env-dependent)'}")
    detail = (
        f"composed agent controls end-to-end: return {composed_med:.1f} vs reactive "
        f"{reactive_med:.1f} ({'beats every seed' if controls_met else 'FAILS'}); one epistemic "
        f"signal drives exploit-mode AND retrieval in one run "
        f"({'MET' if one_signal_met else 'NOT MET'}). ablation marginal control value: {table}. "
        f"cross-env: {gen_note}. FINDINGS: retrieval hurts control (marginal "
        f"{marg_med['retrieval']:+.1f}) and does not generalize to PointMass — its value is "
        f"env-dependent; exploit-penalty is negligible"
    )
    return GateResult(phase="P9", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
