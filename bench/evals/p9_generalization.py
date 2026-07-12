"""P9-003 cross-environment generalization: re-run the load-bearing capabilities on a
SECOND, structurally different environment (`bench.envs.PointMass`) with the SAME core
code — only recalibrated thresholds. A capability that survives is real; one that
collapses was a Pendulum artifact (ADR-0008).

On PointMass (2D nonlinear-drag point mass, obs_dim=4, action_dim=2):
- **prediction** — a world model beats a persistence baseline at 1-step latent MSE (P1);
- **planning** — CEM planning beats a random reactive baseline at control (P2);
- **uncertainty** — the epistemic signal is OOD-reliable: the highest-error-decile
  epistemic clears a floor over the median (P9-005 distance-aware fix);
- **retrieval** — uncertainty-gated retrieval beats no-retrieval at 1-step MSE (P8).

The core is constructed with the env's dimensions and is otherwise untouched — the
generalization test IS that no core change is needed. `generalizes()` returns per-
capability pass flags + metrics; `check_p9` folds them into the P9 gate.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from prospect.agent import Agent
from prospect.memory import SemanticStore, UncertaintyMemoryRouter, blend_retrieved_items
from prospect.planning import FlatPlanner
from prospect.types import Action, KnowledgeItem, LatentState, Observation, Provenance, Transition, Trust
from prospect.world_model import FlatWorldModel

from ..calibration import audit_threshold, calibrate_threshold, exceedance_rate
from ..envs import PointMass
from ..loop import run_episode
from .p2_planner import _PolicyAgent
from .p8_knowledge import _retrieval_temperature

GEN_SEEDS = [0, 1]
OBS_DIM, ACT_DIM = 4, 2
V_REGION, V_FULL, POS_RANGE = 2.0, 6.0, 2.0  # trained |v| <= V_REGION; store/test span V_FULL
TRAIN_N, TEST_N, PROBE_N = 4096, 300, 200
CALIBRATION_N, CALIBRATION_AUDIT_N = 5000, 5000
# STORE_N is dimension-adequate on purpose (P9-006). The retrieval key is 6-D
# (4 latent + 2 action); nearest-neighbour recall in a continuous key space suffers the
# curse of dimensionality — the store's density must scale with the key-space dimension
# or the nearest fact is too far to be right. A sparse store (1500) made retrieval FAIL
# to generalize here (nearest-fact error 0.021 vs the model's own 0.017); a
# dimension-adequate store (40000) makes it generalize with a comfortable margin
# (0.0135, ~22% better than no-retrieval). This is NOT the "key saturation" P9-005
# hypothesized — the latent key is fine (it even beats a raw-input key); it is density.
STORE_N = 40000
GEN_STEPS, BATCH = 2000, 64  # PointMass's 4-dim dynamics + reward head need more data
GEN_HORIZON = 6              # recalibrated planning horizon for this env (was 20 on Pendulum)
EP_LEN_GEN, EVAL_EP_GEN = 60, 2
RETRIEVAL_ALPHA_GEN = 0.01  # conservative nominal gate-hit rate on env #2
RETRIEVAL_AUDIT_TOLERANCE_GEN = 0.006
RETRIEVE_MARGIN = 1.2       # gated MSE must beat no-retrieval by at least this factor
UNCERTAINTY_FLOOR = 3.0     # high-error-decile epistemic must exceed this x median on env #2


def _encoder(model: FlatWorldModel) -> Callable[[Observation], LatentState]:
    def encode(obs: Observation) -> LatentState:
        return model.encode(obs.data)

    return encode


def _region_data(v_max: float, n: int, seed: int) -> list[Transition]:
    """Transitions with |vx|,|vy| <= v_max (positions in [-POS_RANGE, POS_RANGE]), placed
    via set_state so the seen/OOD velocity boundary is exact."""
    env = PointMass()
    rng = np.random.default_rng(seed)
    out: list[Transition] = []
    for _ in range(n):
        env.reset(seed=0)
        obs = env.set_state(rng.uniform(-POS_RANGE, POS_RANGE), rng.uniform(-POS_RANGE, POS_RANGE),
                            rng.uniform(-v_max, v_max), rng.uniform(-v_max, v_max))
        action = Action(data=rng.uniform(-1.0, 1.0, size=ACT_DIM))
        next_obs, reward, _ = env.step(action)
        out.append(Transition(state=LatentState(z=obs.data), action=action,
                              next_state=LatentState(z=next_obs.data), reward=reward))
    return out


def _key(model: FlatWorldModel, t: Transition) -> np.ndarray:
    return np.concatenate([np.asarray(model.encode(t.state.z).z, dtype=float),
                           np.asarray(t.action.data, dtype=float)])


def _store(model: FlatWorldModel, facts: list[Transition]) -> SemanticStore:
    """P8-style store: correct next-latent (target space) keyed by (latent, action)."""
    store = SemanticStore()
    for t in facts:
        answer = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        store.write(KnowledgeItem(content=(_key(model, t), answer),
                                  provenance=Provenance(source="reference", trust=Trust.HIGH)))
    return store


def _rollout(n: int, seed: int) -> list[Transition]:
    """Random-action rollouts (the visited distribution) — the P1/P2 training regime,
    resetting periodically for coverage. Distinct from `_region_data`'s exact-region
    sampling, which the retrieval capability (P8) needs for its seen/OOD split."""
    env = PointMass()
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    out: list[Transition] = []
    for i in range(n):
        if i % 50 == 0:
            obs = env.reset(seed=seed * 97 + i)
        action = Action(data=rng.uniform(-1.0, 1.0, size=ACT_DIM))
        next_obs, reward, _ = env.step(action)
        out.append(Transition(state=LatentState(z=obs.data), action=action,
                              next_state=LatentState(z=next_obs.data), reward=reward))
        obs = next_obs
    return out


def _train(model: FlatWorldModel, data: list[Transition], seed: int) -> FlatWorldModel:
    rng = np.random.default_rng(seed + 1)
    for _ in range(GEN_STEPS):
        idx = rng.integers(0, len(data), size=BATCH)
        model.update([data[i] for i in idx])
    return model


def _fresh_model(seed: int) -> FlatWorldModel:
    return FlatWorldModel(obs_dim=OBS_DIM, action_dim=ACT_DIM, seed=seed)


def _random_agent(seed: int) -> _PolicyAgent:
    rng = np.random.default_rng(seed + 700)

    def policy(obs: Observation) -> Action:
        return Action(data=rng.uniform(-1.0, 1.0, size=ACT_DIM))

    return _PolicyAgent(policy)


def _control_return(agent: object, seed: int) -> float:
    return float(np.mean([
        run_episode(PointMass(), agent, EP_LEN_GEN, 8000 + seed * 40 + e)[0]  # type: ignore[arg-type]
        for e in range(EVAL_EP_GEN)
    ]))


class Generalization(NamedTuple):
    prediction_met: bool     # world model beats persistence at 1-step latent MSE (P1)
    planning_met: bool       # CEM planning beats a random reactive baseline (P2)
    uncertainty_met: bool    # epistemic is OOD-reliable on env #2 (P9-005 distance-aware fix)
    retrieval_met: bool      # uncertainty-gated retrieval beats no-retrieval (P8) — gated (P9-006)
    metrics: dict[str, float]


def generalizes() -> Generalization:
    """Run the capabilities on the second environment; return per-capability pass flags
    (median over seeds, P2-style — robust to a lucky random start) + metrics. Prediction,
    planning, the uncertainty signal (P9-005) AND retrieval (P9-006) are all gated: they
    must generalize. Retrieval generalizes only given a dimension-adequate store — the
    curse of dimensionality in the 6-D key space, not the "key saturation" P9-005 guessed
    (see STORE_N above and the P9-006 task).
    """
    wm_mses, persist_mses, planner_rets, random_rets, gated_mses, none_mses = [], [], [], [], [], []
    unc_ratios: list[float] = []
    calibration_valid: list[bool] = []
    metrics: dict[str, float] = {}
    for seed in GEN_SEEDS:
        # P1 + P2: one model on the visited distribution (random rollouts).
        model = _train(_fresh_model(seed), _rollout(TRAIN_N, seed), seed)
        heldout = _rollout(PROBE_N, seed + 300)

        # 1. prediction beats a persistence baseline (P1)
        wm_err, persist_err = [], []
        for t in heldout:
            target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
            pred_mean = np.asarray(model.predict(model.encode(t.state.z), t.action).mean, dtype=float)
            persist = np.asarray(model.encode_target(t.state.z).z, dtype=float)  # predict "no change"
            wm_err.append(float(np.mean((pred_mean - target) ** 2)))
            persist_err.append(float(np.mean((persist - target) ** 2)))
        wm_mse, persist_mse = float(np.mean(wm_err)), float(np.mean(persist_err))
        wm_mses.append(wm_mse)
        persist_mses.append(persist_mse)

        # 2. planning beats a random reactive baseline (P2)
        planner = FlatPlanner(model, action_dim=ACT_DIM, action_low=-1.0, action_high=1.0,
                              horizon=GEN_HORIZON, seed=seed)
        planner_ret = _control_return(Agent(encode=_encoder(model), planner=planner), seed)
        random_ret = _control_return(_random_agent(seed), seed)
        planner_rets.append(planner_ret)
        random_rets.append(random_ret)

        # 3. uncertainty-gated retrieval beats no-retrieval (P8) — a SEPARATE model
        # trained on a limited region so it is confident there and uncertain outside.
        region_model = _train(_fresh_model(seed + 7), _region_data(V_REGION, TRAIN_N, seed), seed)
        region_held = _region_data(V_REGION, CALIBRATION_N, seed + 300)
        store = _store(region_model, _region_data(V_FULL, STORE_N, seed + 50))
        temperature = _retrieval_temperature(
            store,
            [
                _key(region_model, t)
                for t in _region_data(V_FULL, PROBE_N, seed + 450)
            ],
        )
        seen_epi = [
            region_model.predict(region_model.encode(t.state.z), t.action).epistemic
            for t in region_held
        ]
        retrieval = calibrate_threshold(seen_epi, alpha=RETRIEVAL_ALPHA_GEN)
        audit_epi = [
            region_model.predict(region_model.encode(t.state.z), t.action).epistemic
            for t in _region_data(V_REGION, CALIBRATION_AUDIT_N, seed + 400)
        ]
        audit_rate, audit_tolerance, audit_valid = audit_threshold(
            audit_epi, retrieval, tolerance=RETRIEVAL_AUDIT_TOLERANCE_GEN
        )
        calibration_valid.append(audit_valid)
        router = UncertaintyMemoryRouter([store], threshold=retrieval.value)
        none_err, gated_err, epi, retrieved = [], [], [], 0
        for t in _region_data(V_FULL, TEST_N, seed + 200):
            pred = region_model.predict(region_model.encode(t.state.z), t.action)
            target = np.asarray(region_model.encode_target(t.next_state.z).z, dtype=float)
            parametric = np.asarray(pred.mean, dtype=float)
            source = router.route(None, pred.epistemic)
            gated = parametric
            if source is not None:
                key = _key(region_model, t)
                gated, reliability, _ = blend_retrieved_items(
                    key, parametric, source.query(key), temperature
                )
                retrieved += int(reliability > 0.0)
            none_err.append(float(np.mean((parametric - target) ** 2)))
            gated_err.append(float(np.mean((gated - target) ** 2)))
            epi.append(pred.epistemic)
        none_mse, gated_mse = float(np.mean(none_err)), float(np.mean(gated_err))
        gated_mses.append(gated_mse)
        none_mses.append(none_mse)

        # Uncertainty-reliability on env #2 (P9-005): epistemic in the highest-error
        # decile vs the median — the same metric the uncertainty-reliability sentinel
        # uses, now required to GENERALIZE. Distance-aware epistemic is what lets it.
        epi_arr, err_arr = np.array(epi), np.array(none_err)
        hi = err_arr >= np.quantile(err_arr, 0.9)
        med = float(np.median(epi_arr))
        ratio = float(np.mean(epi_arr[hi]) / med) if med > 0 else 0.0
        unc_ratios.append(ratio)

        metrics |= {
            f"gen_wm_mse_s{seed}": wm_mse, f"gen_persist_mse_s{seed}": persist_mse,
            f"gen_planner_return_s{seed}": planner_ret, f"gen_random_return_s{seed}": random_ret,
            f"gen_retrieval_none_mse_s{seed}": none_mse, f"gen_retrieval_gated_mse_s{seed}": gated_mse,
            f"gen_retrieval_rate_s{seed}": retrieved / TEST_N, f"gen_uncertainty_ratio_s{seed}": ratio,
            f"gen_retrieval_calls_s{seed}": float(TEST_N),
            f"gen_retrieval_gate_hits_s{seed}": float(retrieved),
            f"gen_retrieval_alpha_s{seed}": retrieval.alpha,
            f"gen_retrieval_eta_s{seed}": retrieval.eta,
            f"gen_retrieval_threshold_s{seed}": retrieval.value,
            f"gen_retrieval_kernel_temperature_s{seed}": temperature,
            f"gen_retrieval_calibration_updates_s{seed}": float(retrieval.updates),
            f"gen_nominal_retrieval_online_rate_s{seed}": retrieval.trigger_rate,
            f"gen_nominal_retrieval_retrospective_rate_s{seed}": exceedance_rate(
                seen_epi, retrieval.value
            ),
            f"gen_nominal_retrieval_audit_rate_s{seed}": audit_rate,
            f"gen_nominal_retrieval_audit_tolerance_s{seed}": audit_tolerance,
            f"gen_nominal_retrieval_audit_valid_s{seed}": float(audit_valid),
        }
    # Median criteria (P2-style, robust to a lucky random start on one seed).
    wm_med, persist_med = float(np.median(wm_mses)), float(np.median(persist_mses))
    planner_med, random_med = float(np.median(planner_rets)), float(np.median(random_rets))
    gated_med, none_med = float(np.median(gated_mses)), float(np.median(none_mses))
    unc_med = float(np.median(unc_ratios))
    calibration_met = all(calibration_valid)
    metrics |= {"gen_wm_mse_median": wm_med, "gen_persist_mse_median": persist_med,
                "gen_planner_return_median": planner_med, "gen_random_return_median": random_med,
                "gen_retrieval_gated_mse_median": gated_med, "gen_retrieval_none_mse_median": none_med,
                "gen_uncertainty_ratio_median": unc_med,
                "gen_retrieval_calibration_met": float(calibration_met)}
    return Generalization(
        prediction_met=wm_med < persist_med,
        planning_met=planner_med > random_med,
        uncertainty_met=unc_med >= UNCERTAINTY_FLOOR,
        retrieval_met=gated_med * RETRIEVE_MARGIN <= none_med and calibration_met,
        metrics=metrics,
    )
