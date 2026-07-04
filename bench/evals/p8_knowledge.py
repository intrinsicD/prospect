"""P8 eval — partial, from P8-001: uncertainty-gated retrieval beats no-retrieval.

Retrieval is an action gated by epistemic uncertainty (ADR-0004): answer from the
parametric tier (the world model's weights) where the model is confident, retrieve
from a knowledge store where it is uncertain. The model is trained on a LIMITED
region of the pendulum's state space (|omega| <= REGION), so it is confident there
and uncertain outside it; a `SemanticStore` holds correct (state,action)->
next-latent facts across the FULL range (the knowledge base). Per test query the
`UncertaintyMemoryRouter` gates on the model's `prediction.epistemic` (threshold =
a held-out seen-region epistemic quantile).

Measured claim: gated 1-step MSE beats model-alone (no-retrieval) on every seed —
and typically beats ALWAYS-retrieve too, because gating keeps the model's accurate
prediction in the confident region and only fetches facts where the model is
wrong, retrieving a fraction of the time (the gating IS the point).

The full P8 capability is gated-beats-no-retrieval AND robustness to a
poisoned/low-trust source (P8-002), so this check reports `passed=False` with the
pending half named. Run `p8` carries all four sentinels: the limited-region model
must still be healthy, and its natural OOD (|omega| > REGION) feeds the
uncertainty-reliability probe.
"""
from __future__ import annotations

import numpy as np

from prospect.memory import SemanticStore, UncertaintyMemoryRouter
from prospect.types import Action, KnowledgeItem, LatentState, Provenance, Transition, Trust
from prospect.world_model import FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe
from .p3_replay import log_replay_fidelity
from .p7_continual import _log_option_diversity

RUN_ID = "p8"
SEEDS = [0, 1, 2]
GRAVITY, DAMPING, DT = 10.0, 0.2, 0.2
REGION, FULL = 4.0, 8.0  # trained |omega| <= REGION; store & test span |omega| <= FULL
TRAIN_N, STORE_N, TEST_N, PROBE_N = 4096, 3000, 400, 256
EPISTEMIC_QUANTILE = 0.9  # retrieve above the 90th-percentile seen-region epistemic
IMPROVEMENT_MARGIN = 1.5  # gated MSE must beat no-retrieval by at least this factor


def _env() -> Pendulum:
    return Pendulum(gravity=GRAVITY, damping=DAMPING, dt=DT)


def _region_data(omega_max: float, n: int, seed: int) -> list[Transition]:
    """Transitions with |omega| <= omega_max, states placed via set_state so the
    region boundary is exact (not visitation-dependent)."""
    env = _env()
    rng = np.random.default_rng(seed)
    out: list[Transition] = []
    for _ in range(n):
        env.reset(seed=0)
        obs = env.set_state(float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-omega_max, omega_max)))
        action = Action(data=np.array([float(rng.uniform(-2.0, 2.0))]))
        next_obs, reward, _ = env.step(action)
        out.append(Transition(state=LatentState(z=obs.data), action=action,
                              next_state=LatentState(z=next_obs.data), reward=reward))
    return out


def _key(model: FlatWorldModel, t: Transition) -> np.ndarray:
    return np.concatenate([np.asarray(model.encode(t.state.z).z, dtype=float),
                           np.asarray(t.action.data, dtype=float)])


def _build_store(model: FlatWorldModel, facts: list[Transition]) -> SemanticStore:
    """Knowledge-as-tokens: each fact answers (state,action) with the correct
    next-latent in the MODEL's own space (a drop-in for its prediction)."""
    store = SemanticStore()
    for t in facts:
        answer = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        store.write(KnowledgeItem(content=(_key(model, t), answer),
                                  provenance=Provenance(source="reference", trust=Trust.HIGH)))
    return store


@gate_check("P8")
def check_p8() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    beats_no_retrieval = []
    for seed in SEEDS:
        train = _region_data(REGION, TRAIN_N, seed)
        heldout = _region_data(REGION, PROBE_N, seed + 500)  # in-distribution (seen region)
        ood = _region_data(FULL, PROBE_N, seed + 900)  # spans the unseen region
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

        store = _build_store(model, _region_data(FULL, STORE_N, seed + 50))
        seen_epistemic = sorted(
            model.predict(model.encode(t.state.z), t.action).epistemic for t in heldout)
        threshold = seen_epistemic[int(EPISTEMIC_QUANTILE * len(seen_epistemic))]
        router = UncertaintyMemoryRouter([store], threshold=threshold)

        test = _region_data(FULL, TEST_N, seed + 200)
        none_err, gated_err, always_err, retrieved = [], [], [], 0
        for t in test:
            pred = model.predict(model.encode(t.state.z), t.action)
            target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
            parametric = np.asarray(pred.mean, dtype=float)
            retrieved_answer = np.asarray(store.query(_key(model, t))[0].content[1], dtype=float)
            source = router.route(None, pred.epistemic)
            gated = parametric if source is None else retrieved_answer
            if source is not None:
                retrieved += 1
            none_err.append(float(np.mean((parametric - target) ** 2)))
            gated_err.append(float(np.mean((gated - target) ** 2)))
            always_err.append(float(np.mean((retrieved_answer - target) ** 2)))
        none_mse, gated_mse = float(np.mean(none_err)), float(np.mean(gated_err))
        always_mse, rate = float(np.mean(always_err)), retrieved / len(test)
        beats_no_retrieval.append(gated_mse * IMPROVEMENT_MARGIN <= none_mse)
        metrics |= {
            f"no_retrieval_mse_s{seed}": none_mse,
            f"gated_mse_s{seed}": gated_mse,
            f"always_retrieve_mse_s{seed}": always_mse,
            f"retrieval_rate_s{seed}": rate,
        }

    accuracy_met = all(beats_no_retrieval)
    none_med = float(np.median([metrics[f"no_retrieval_mse_s{s}"] for s in SEEDS]))
    gated_med = float(np.median([metrics[f"gated_mse_s{s}"] for s in SEEDS]))
    always_med = float(np.median([metrics[f"always_retrieve_mse_s{s}"] for s in SEEDS]))
    rate_med = float(np.median([metrics[f"retrieval_rate_s{s}"] for s in SEEDS]))
    metrics |= {"no_retrieval_mse_median": none_med, "gated_mse_median": gated_med,
                "always_retrieve_mse_median": always_med, "retrieval_rate_median": rate_med,
                "accuracy_half_met": float(accuracy_met)}
    detail = (
        f"1-step MSE: gated {gated_med:.4f} vs no-retrieval {none_med:.4f} "
        f"(>= x{IMPROVEMENT_MARGIN} better on every seed: {'MET' if accuracy_met else 'NOT MET'}) "
        f"vs always-retrieve {always_med:.4f}; retrieval rate {rate_med:.0%} (gated: model where "
        f"confident, retrieve where uncertain); robustness half pending (P8-002)"
    )
    return GateResult(phase="P8", passed=False, metrics=metrics, seeds=list(SEEDS), detail=detail)
