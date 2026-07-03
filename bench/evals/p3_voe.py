"""P3 eval — partial, from P3-001: the expected-vs-violated surprise differential.

Violation of expectation, constructed the way infant studies do it: the SAME
premises (held-out state-action pairs, in-distribution by construction) with
counterfactual outcomes — the next state computed under gravity-flipped physics
via `Pendulum.set_state`. A model trained on normal physics must be reliably more
surprised (`Surprise.total`, the calibrated NLL) by the counterfactual outcomes.
(Sampling violated *trajectories* instead is a broken design: a flipped pendulum
oscillates around θ≈0, exactly where the physics difference 2g·sinθ vanishes.)

Effect size: **probability of superiority (AUC)** — P(violated surprise >
expected surprise) >= AUC_MIN on every seed. NLL is heavy-tailed by construction
(quadratic in error), so pooled-std effect sizes understate near-total separation
(observed: AUC 0.93-0.98 while Cohen's d sits ~1.0 because the violated mean is
3x its median); the rank-based effect size is the honest measure here. Cohen's d
is still reported in the metrics for reference.

The full P3 capability is differential AND curiosity-beats-random (P3-002), and
the `replay-fidelity` sentinel (P3-003) is active-but-PENDING from P3 — so this
check reports `passed=False` with the pending half named until those land. The
differential result is recorded in the metrics either way: progress is measured,
not claimed (ADR-0005).
"""
from __future__ import annotations

import numpy as np

from prospect.agent import Agent
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState, Observation, Transition
from prospect.voe import LearningProgressCurriculum, SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, gate_check
from ..loop import run_episode
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe, _rollout, _train

RUN_ID = "p3"
SEEDS = [0, 1, 2]
TRAIN_N, PROBE_N = 4096, 256
AUC_MIN = 0.9  # probability of superiority, required on every seed
MASTERY_WARMUP = 200  # monitor updates fed from held-out data for the mastery demo
# Curiosity experiment (P3-002): identical budgets and training schedules; only the
# collection policy differs. Scored on a uniform-coverage set with the scale-free
# ratio model/persistence MSE (raw latent MSE is incomparable across encoders).
CURIOSITY_BUDGET, CHUNK, TRAIN_PER_ROUND, COVERAGE_N = 1536, 256, 250, 512
MONITOR_FEED = 20  # transitions per round that keep the curriculum's mode live


def _surprise_totals(
    model: FlatWorldModel, monitor: SurpriseCompetenceMonitor, transitions: list[Transition]
) -> np.ndarray:
    totals = []
    for t in transitions:
        prediction = model.predict(model.encode(t.state.z), t.action)
        totals.append(monitor.surprise(prediction, model.encode_target(t.next_state.z)).total)
    return np.array(totals)


def _cohens_d(violated: np.ndarray, expected: np.ndarray) -> float:
    pooled = np.sqrt((violated.var(ddof=1) + expected.var(ddof=1)) / 2.0)
    return float((violated.mean() - expected.mean()) / pooled)


def _auc(violated: np.ndarray, expected: np.ndarray) -> float:
    """Probability of superiority: P(a violated surprise exceeds an expected one)."""
    return float(np.mean(violated[:, None] > expected[None, :]))


def _counterfactual(expected: list[Transition], gravity: float) -> list[Transition]:
    """Same premises, different physics: re-step each (state, action) under the
    violated dynamics — a controlled violation with identical input marginals."""
    env = Pendulum(gravity=gravity)
    env.reset(seed=0)
    violated = []
    for t in expected:
        cos_theta, sin_theta, omega = np.asarray(t.state.z, dtype=float)
        env.set_state(float(np.arctan2(sin_theta, cos_theta)), omega)
        next_obs, reward, _ = env.step(t.action)
        violated.append(Transition(state=t.state, action=t.action,
                                   next_state=LatentState(z=next_obs.data), reward=reward))
    return violated


class _RandomAgent:
    """Acting-conforming random collector (the exploration baseline)."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> Action:
        return Action(data=self._rng.uniform(-2.0, 2.0, size=1))

    def reset(self) -> None:
        return None


def _coverage_set(seed: int) -> list[Transition]:
    """Uniform-coverage test set: states placed via `set_state`, so neither
    collector's visitation distribution biases the evaluation."""
    env = Pendulum()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(COVERAGE_N):
        obs = env.set_state(rng.uniform(-np.pi, np.pi), rng.uniform(-8.0, 8.0))
        action = Action(data=np.array([rng.uniform(-2.0, 2.0)]))
        next_obs, reward, _ = env.step(action)
        out.append(Transition(state=LatentState(z=obs.data), action=action,
                              next_state=LatentState(z=next_obs.data), reward=reward))
    return out


def _coverage_ratio(model: FlatWorldModel, coverage: list[Transition]) -> float:
    """Scale-free model skill: model MSE / persistence MSE, each in the model's
    own target-latent space."""
    current = np.stack([np.asarray(model.encode_target(t.state.z).z, dtype=float) for t in coverage])
    target = np.stack([np.asarray(model.encode_target(t.next_state.z).z, dtype=float) for t in coverage])
    predicted = np.stack(
        [np.asarray(model.predict(model.encode(t.state.z), t.action).mean, dtype=float) for t in coverage]
    )
    return float(np.mean((predicted - target) ** 2) / np.mean((current - target) ** 2))


def _active_learning(seed: int, curious: bool) -> float:
    """One collection arm: chunks of env steps + a fixed training schedule; the
    curious arm collects with an explore-mode planner whose epistemic coefficient
    comes from the LIVE curriculum (monitor fed each round, ADR-0007)."""
    model = FlatWorldModel(seed=seed)
    monitor = SurpriseCompetenceMonitor()
    curriculum = LearningProgressCurriculum(monitor)
    planner = FlatPlanner(model, horizon=8, candidates=32, elites=6, iterations=2, seed=seed)
    explorer = Agent(encode=lambda obs: model.encode(obs.data), planner=planner)
    rng = np.random.default_rng(seed + 17)
    data: list[Transition] = []
    for round_index in range(CURIOSITY_BUDGET // CHUNK):
        if curious and round_index > 0:  # the seed chunk is random for both arms
            planner.uncertainty_penalty = curriculum.uncertainty_coefficient()
            _, chunk = run_episode(Pendulum(), explorer, CHUNK, seed * 131 + round_index,
                                   collect=True)
        else:
            _, chunk = run_episode(Pendulum(), _RandomAgent(seed * 131 + round_index),
                                   CHUNK, seed * 131 + round_index, collect=True)
        data.extend(chunk)
        _train(model, data, TRAIN_PER_ROUND, rng)
        for t in chunk[:MONITOR_FEED]:  # keep the curriculum's mode decision live
            prediction = model.predict(model.encode(t.state.z), t.action)
            monitor.update(
                Transition(state=model.encode(t.state.z), action=t.action,
                           next_state=model.encode_target(t.next_state.z),
                           reward=t.reward, prediction=prediction)
            )
    return _coverage_ratio(model, _coverage_set(seed + 4400))


@gate_check("P3")
def check_p3() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    effect_sizes = []
    curious_ratios, random_ratios = [], []
    for seed in SEEDS:
        train = _rollout(Pendulum(), TRAIN_N, seed)
        heldout = _rollout(Pendulum(), PROBE_N, seed + 500)
        ood = _rollout(Pendulum(init_omega=14.0, omega_max=16.0), PROBE_N, seed + 900)
        model = FlatWorldModel(seed=seed)
        _train(model, train, STEPS, np.random.default_rng(seed + 1),
               probe=_make_probe(heldout, heldout + ood, seed), log=log,
               step_offset=seed * SEED_STEP_OFFSET)
        monitor = SurpriseCompetenceMonitor()
        expected = _rollout(Pendulum(), PROBE_N, seed + 1300)
        violated = _counterfactual(expected, gravity=-10.0)
        violated_s = _surprise_totals(model, monitor, violated)
        expected_s = _surprise_totals(model, monitor, expected)
        auc = _auc(violated_s, expected_s)
        effect_sizes.append(auc)
        # Mastery demo on the real signal: after training, in-distribution epistemic
        # is low and flat — the task-level "skill" should read as mastered.
        for t in heldout[:MASTERY_WARMUP]:
            prediction = model.predict(model.encode(t.state.z), t.action)
            monitor.update(
                Transition(state=model.encode(t.state.z), action=t.action,
                           next_state=model.encode_target(t.next_state.z),
                           reward=t.reward, prediction=prediction)
            )
        competence = monitor.competence(SurpriseCompetenceMonitor.DEFAULT_SKILL)
        curious_ratio = _active_learning(seed, curious=True)
        random_ratio = _active_learning(seed, curious=False)
        curious_ratios.append(curious_ratio)
        random_ratios.append(random_ratio)
        metrics |= {
            f"voe_auc_s{seed}": auc,
            f"cohens_d_s{seed}": _cohens_d(violated_s, expected_s),  # reference only
            f"violated_surprise_median_s{seed}": float(np.median(violated_s)),
            f"expected_surprise_median_s{seed}": float(np.median(expected_s)),
            f"mastered_s{seed}": float(competence.mastered),
            f"epistemic_ema_s{seed}": competence.epistemic,
            f"curious_coverage_ratio_s{seed}": curious_ratio,
            f"random_coverage_ratio_s{seed}": random_ratio,
        }
    auc_min = float(min(effect_sizes))
    differential_met = auc_min >= AUC_MIN
    curious_med = float(np.median(curious_ratios))
    random_med = float(np.median(random_ratios))
    curiosity_met = curious_med < random_med
    metrics |= {
        "voe_auc_min": auc_min,
        "differential_met": float(differential_met),
        "curious_coverage_ratio_median": curious_med,
        "random_coverage_ratio_median": random_med,
        "curiosity_met": float(curiosity_met),
    }
    detail = (
        f"differential: P(violated surprise > expected) per seed "
        f"{[round(a, 2) for a in effect_sizes]}, min {auc_min:.2f} (>= {AUC_MIN}) "
        f"— {'MET' if differential_met else 'NOT MET'}; "
        f"curiosity: coverage ratio (model/persistence MSE, lower is better) "
        f"curious {curious_med:.2f} vs random {random_med:.2f} at equal budget "
        f"({CURIOSITY_BUDGET} steps) — {'MET' if curiosity_met else 'NOT MET'}"
    )
    return GateResult(phase="P3", passed=differential_met and curiosity_met,
                      metrics=metrics, seeds=list(SEEDS), detail=detail)
