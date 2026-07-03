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

from prospect.types import LatentState, Transition
from prospect.voe import SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe, _rollout, _train

RUN_ID = "p3"
SEEDS = [0, 1, 2]
TRAIN_N, PROBE_N = 4096, 256
AUC_MIN = 0.9  # probability of superiority, required on every seed
MASTERY_WARMUP = 200  # monitor updates fed from held-out data for the mastery demo


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


@gate_check("P3")
def check_p3() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    effect_sizes = []
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
        metrics |= {
            f"voe_auc_s{seed}": auc,
            f"cohens_d_s{seed}": _cohens_d(violated_s, expected_s),  # reference only
            f"violated_surprise_median_s{seed}": float(np.median(violated_s)),
            f"expected_surprise_median_s{seed}": float(np.median(expected_s)),
            f"mastered_s{seed}": float(competence.mastered),
            f"epistemic_ema_s{seed}": competence.epistemic,
        }
    auc_min = float(min(effect_sizes))
    differential_met = auc_min >= AUC_MIN
    metrics |= {"voe_auc_min": auc_min, "differential_met": float(differential_met)}
    detail = (
        f"differential: P(violated surprise > expected) per seed "
        f"{[round(a, 2) for a in effect_sizes]}, min {auc_min:.2f} (criterion >= {AUC_MIN}) "
        f"— {'MET' if differential_met else 'NOT MET'}; curiosity criterion pending (P3-002)"
    )
    return GateResult(phase="P3", passed=False, metrics=metrics, seeds=list(SEEDS), detail=detail)
