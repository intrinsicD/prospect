"""P4 eval (task P4-001): simulate-to-select skill routing + misapplication VoE.

Three constant-torque skills (left / coast / right, horizon 5) on the P4
reference task: an action-dominant Pendulum (gravity 1, damping 0.4, dt 0.4).
Config rationale, measured, in two steps: (1) on the default pendulum the
model's open-loop landing error is ~10x the inter-skill separation — ADR-0001's
compounding-rollout-error limiter, whose real answer is P5's jumpy option-model;
(2) worse, at dt 0.15 the full action swing moves the one-step prediction by
only ~0.3 predictive std — the NLL loss absorbs the action effect into noise and
the model is action-blind (routing AND misapplication VoE structurally at
chance). The reference config makes torque the dominant force (action/std ratio
1.2), so the ROUTER mechanism is what is being tested, not those open problems.

Routing: per test case the target is the REAL landing of a randomly chosen
skill; ground truth = the skill whose real execution lands closest (scaled obs
space); router accuracy must beat the uniform baseline (1/3) on every seed.

Misapplication (the VoE signal's skill-trust job) is judged CLOSED-LOOP, the way
options actually terminate (ADR-0003): at each executed step, one-step surprise
against the prediction under the INTENDED skill's action. The detector metric is
PAIRED — for the same start and intended skill, the misapplied execution must
out-surprise the correct one (win rate >= WIN_RATE_MIN per seed). Pooled AUC is
the wrong statistic here: correct-case surprise varies across states by more
than the misapplication gap, fogging a per-event detector (measured ~0.6-0.78);
pairing cancels the state noise. (Open-loop landing comparison is worse still:
compounding rollout error fogs both outcomes — measured AUC ~0.56.)

Routing correctness counts behavioral near-ties: the chosen skill is correct if
its REAL landing distance is within 10% of the best skill's — when two skills
land equivalently for a target, either choice is right. Exact argmin match is
reported alongside.

Run `p4` carries the P1 probes AND the replay-fidelity records so every active
sentinel judges this phase's own model.
"""
from __future__ import annotations

import numpy as np

from prospect.skills import SkillRouter
from prospect.types import Action, LatentState, Option, Subgoal, Transition
from prospect.voe import SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, _make_probe, _rollout, _train
from .p3_replay import log_replay_fidelity

RUN_ID = "p4"
SEEDS = [0, 1, 2]
TRAIN_N, TESTS, SKILL_HORIZON = 4096, 60, 5
TRAIN_STEPS = 1500
RESET_EVERY = 40  # frequent wide resets: attractor-dominated envs concentrate data
GRAVITY, DAMPING, DT = 1.0, 0.4, 0.4  # P4 reference config (see module docstring)
COMPETENCE_FEED = 30  # per-skill monitor updates before routing (mastery accrual)
WIN_RATE_MIN = 0.9
TIE_TOLERANCE = 1.1  # chosen real distance within 10% of best counts as correct
OBS_SCALE = np.array([1.0, 1.0, 1.0 / 8.0])  # cos, sin, omega/omega_max


def _env(**kwargs: float) -> Pendulum:
    return Pendulum(gravity=GRAVITY, damping=DAMPING, dt=DT, **kwargs)


def _collect(n: int, seed: int) -> list[Transition]:
    """Training data with frequent, wide resets: the gentle env is
    attractor-dominated, so long episodes concentrate on the attractors and
    starve the encoder of manifold coverage (measured: effective rank falls).
    Short episodes from wide initial states keep the data — and therefore the
    latent — spread."""
    from prospect.types import LatentState as _L  # local alias for clarity

    env = _env(init_omega=8.0)
    rng = np.random.default_rng(seed)
    transitions: list[Transition] = []
    obs = env.reset(seed=seed * 7919 + 1)
    for i in range(n):
        if i % RESET_EVERY == 0 and i > 0:
            obs = env.reset(seed=seed * 7919 + i)
        action = Action(data=np.array([float(rng.uniform(-2.0, 2.0))]))
        next_obs, reward, _ = env.step(action)
        transitions.append(Transition(state=_L(z=obs.data), action=action,
                                      next_state=_L(z=next_obs.data), reward=reward))
        obs = next_obs
    return transitions


def _constant_policy(torque: float) -> Option:
    def policy(latent: LatentState) -> Action:
        return Action(data=np.array([torque]))

    return Option(name=f"torque{torque:+.0f}", policy=policy, horizon=SKILL_HORIZON)


def _skills() -> list[Option]:
    return [_constant_policy(-2.0), _constant_policy(0.0), _constant_policy(2.0)]


def _execute(theta: float, omega: float, option: Option) -> np.ndarray:
    """Really run the skill from an exact start state; returns the landing obs."""
    env = _env()
    env.reset(seed=0)
    obs = env.set_state(theta, omega)
    assert option.policy is not None
    for _ in range(option.horizon):
        obs, _, _ = env.step(option.policy(LatentState(z=obs.data)))
    return np.asarray(obs.data, dtype=float)


def _stepwise_surprise(
    model: FlatWorldModel,
    monitor: SurpriseCompetenceMonitor,
    theta: float,
    omega: float,
    intended: Option,
    executed: Option,
) -> float:
    """Closed-loop VoE during execution (how options terminate, ADR-0003): each
    real step is judged against the one-step prediction under the INTENDED
    skill's action, while the EXECUTED skill actually drives the env."""
    env = _env()
    env.reset(seed=0)
    obs = env.set_state(theta, omega)
    assert intended.policy is not None and executed.policy is not None
    surprises = []
    for _ in range(intended.horizon):
        latent = model.encode(obs.data)
        expected = model.predict(latent, intended.policy(latent))
        obs, _, _ = env.step(executed.policy(latent))
        surprises.append(monitor.surprise(expected, model.encode_target(obs.data)).total)
    return float(np.mean(surprises))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(((a - b) * OBS_SCALE) ** 2))


@gate_check("P4")
def check_p4() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    accuracies, aucs = [], []
    for seed in SEEDS:
        train = _collect(TRAIN_N, seed)
        heldout = _collect(256, seed + 500)
        ood = _rollout(_env(init_omega=14.0, omega_max=16.0), 256, seed + 900)
        model = FlatWorldModel(seed=seed)
        _train(model, train, TRAIN_STEPS, np.random.default_rng(seed + 1),
               probe=_make_probe(heldout, heldout + ood, seed), log=log,
               step_offset=seed * SEED_STEP_OFFSET)
        log_replay_fidelity(model, train, seed, log,
                            step_offset=seed * SEED_STEP_OFFSET + 50_000)

        # Mastery threshold calibrated to the model's achievable epistemic floor
        # (an absolute default would arbitrarily gate skills in or out per seed).
        heldout_epistemic = [
            model.predict(model.encode(t.state.z), t.action).epistemic for t in heldout[:128]
        ]
        monitor = SurpriseCompetenceMonitor(
            mastery_epistemic=4.0 * float(np.median(heldout_epistemic))
        )
        router = SkillRouter(model, monitor=monitor)
        skills = _skills()
        rng = np.random.default_rng(seed + 71)
        for skill in skills:
            router.add(skill)
            for _ in range(COMPETENCE_FEED):  # executors set Transition.option (P0-002)
                theta = float(rng.uniform(-np.pi, np.pi))
                omega = float(rng.uniform(-6.0, 6.0))
                start = _env().set_state(theta, omega)  # placed probe state
                assert skill.policy is not None
                latent = model.encode(start.data)
                prediction = model.predict(latent, skill.policy(latent))
                monitor.update(Transition(state=latent, action=skill.policy(latent),
                                          next_state=LatentState(z=prediction.mean),
                                          reward=0.0, prediction=prediction, option=skill))
        mastered = sum(monitor.is_mastered(s.name) for s in skills)

        correct = exact = 0
        wins = 0
        for _ in range(TESTS):
            theta = float(rng.uniform(-np.pi, np.pi))
            omega = float(rng.uniform(-6.0, 6.0))
            start_obs = np.asarray(_env().set_state(theta, omega).data, dtype=float)
            target_obs = _execute(theta, omega, skills[int(rng.integers(len(skills)))])
            proposal = router.propose(
                model.encode(start_obs),
                Subgoal(target=model.encode_target(target_obs)),
            )
            top = proposal[0]
            real_landings = {s.name: _execute(theta, omega, s) for s in skills}
            distances = {name: _distance(landing, target_obs)
                         for name, landing in real_landings.items()}
            best = min(distances.values())
            exact += distances[top.name] == best
            correct += distances[top.name] <= TIE_TOLERANCE * best
            # Misapplication VoE, closed-loop and PAIRED: same start, same intended
            # skill; the misapplied execution must out-surprise the correct one.
            others = [s for s in skills if s.name != top.name]
            wrong = others[int(rng.integers(len(others)))]
            ok = _stepwise_surprise(model, monitor, theta, omega, top, top)
            mis = _stepwise_surprise(model, monitor, theta, omega, top, wrong)
            wins += mis > ok

        accuracy = correct / TESTS
        win_rate = wins / TESTS
        accuracies.append(accuracy)
        aucs.append(win_rate)
        metrics |= {
            f"routing_accuracy_s{seed}": accuracy,
            f"routing_exact_match_s{seed}": exact / TESTS,
            f"misapplication_win_rate_s{seed}": win_rate,
            f"skills_mastered_s{seed}": float(mastered),
        }

    baseline = 1.0 / len(_skills())
    accuracy_min, win_rate_min = float(min(accuracies)), float(min(aucs))
    passed = accuracy_min > baseline and win_rate_min >= WIN_RATE_MIN
    metrics |= {"routing_accuracy_min": accuracy_min,
                "misapplication_win_rate_min": win_rate_min,
                "random_baseline": baseline}
    detail = (
        f"routing accuracy per seed {[round(a, 2) for a in accuracies]} "
        f"(uniform baseline {baseline:.2f}, near-ties count); paired misapplication "
        f"win rate per seed {[round(a, 2) for a in aucs]} (min required {WIN_RATE_MIN})"
    )
    return GateResult(phase="P4", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
