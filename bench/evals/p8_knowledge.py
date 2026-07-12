"""P8 eval: uncertainty-gated retrieval beats no-retrieval (P8-001) AND is robust to
a poisoned/low-trust source (P8-002) — the two halves of the P8 capability.

Retrieval is an action gated by epistemic uncertainty (ADR-0004): answer from the
parametric tier (the world model's weights) where the model is confident, retrieve
from a knowledge store where it is uncertain. The model is trained on a LIMITED
region of the pendulum's state space (|omega| <= REGION), so it is confident there
and uncertain outside it; a `SemanticStore` holds correct (state,action)->
next-latent facts across the FULL range (the knowledge base). Per test query the
`UncertaintyMemoryRouter` gates on the model's `prediction.epistemic` (threshold
ACI-calibrated to a 10% exceedance rate on held-out seen-region scores); accepted
queries aggregate the ranked top three facts with a store-calibrated distance kernel
and blend that result with the model prediction (U-005).

Accuracy half (P8-001): gated 1-step MSE beats model-alone (no-retrieval) on every
seed — and typically beats ALWAYS-retrieve too, because gating keeps the model's
accurate prediction in the confident region and only fetches facts where the model
is wrong, retrieving a fraction of the time (the gating IS the point).

Robustness half (P8-002): a poisoned store (UNTRUSTED provenance, corrupted answers)
over the same keys is the attack surface. A trust-blind agent that retrieves from it
does WORSE than no-retrieval (the poison bites). A provenance-respecting router
(`min_trust` floor + trust-ordered selection) never lets the untrusted source
override the model — it declines to retrieve and stays at no-retrieval — and when the
trusted store is also present it trust-orders to it, recovering the clean gated
accuracy. Untrusted content is data, never instruction (ADR-0004).

Run `p8` carries all four sentinels: the limited-region model must still be healthy,
and its natural OOD (|omega| > REGION) feeds the uncertainty-reliability probe.
"""
from __future__ import annotations

import numpy as np

from prospect.interfaces import KnowledgeSource
from prospect.memory import SemanticStore, UncertaintyMemoryRouter, blend_retrieved_items
from prospect.types import Action, KnowledgeItem, LatentState, Provenance, Transition, Trust
from prospect.world_model import FlatWorldModel

from ..calibration import audit_threshold, calibrate_threshold, exceedance_rate
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
CALIBRATION_N, CALIBRATION_AUDIT_N = 2000, 2000
RETRIEVAL_ALPHA = 0.1  # nominal seen-region epistemic gate-hit rate
RETRIEVAL_AUDIT_TOLERANCE = 0.03
IMPROVEMENT_MARGIN = 1.5  # gated MSE must beat no-retrieval by at least this factor
POISON_SCALE = 0.5  # noise added to the poisoned store's answers (target-latent units)
POISON_HARM_FACTOR = 2.0  # trust-blind retrieval from the poison must be >= this x worse
ROBUST_TOL = 1.05  # provenance-respecting MSE must stay within this of its clean target


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


def _build_poisoned_store(
    model: FlatWorldModel, facts: list[Transition], seed: int
) -> SemanticStore:
    """An UNTRUSTED store over the same keys as the clean store, but whose answers are
    corrupted by one coherent source-level bias — a plausible poisoned/low-trust
    source (P8-002) that k-neighbor aggregation cannot average away. Its provenance
    is UNTRUSTED, so a provenance-respecting router must never let it override the
    model's own prediction (ADR-0004: untrusted content is data)."""
    rng = np.random.default_rng(seed)
    store = SemanticStore(trust=Trust.UNTRUSTED)
    offset: np.ndarray | None = None
    for t in facts:
        correct = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        if offset is None:
            direction = rng.normal(size=correct.shape)
            offset = POISON_SCALE * direction / float(np.sqrt(np.mean(direction**2)))
        poison = correct + offset
        store.write(KnowledgeItem(content=(_key(model, t), poison),
                                  provenance=Provenance(source="poisoned", trust=Trust.UNTRUSTED)))
    return store


def _routed_answer(
    router: UncertaintyMemoryRouter,
    key: np.ndarray,
    epistemic: float,
    parametric: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """The answer a provenance-respecting agent uses: the router's chosen source's
    distance-kernel blend when it retrieves, else its parametric prediction."""
    source = router.route(None, epistemic)
    if source is None:
        return parametric
    blended, _, _ = blend_retrieved_items(
        key, parametric, source.query(key), temperature
    )
    return blended


def _retrieval_temperature(source: KnowledgeSource, queries: list[np.ndarray]) -> float:
    """Median k-th-neighbor squared distance: a store-scale kernel calibration."""
    scales: list[float] = []
    for query in queries:
        items = source.query(query)
        if items:
            key = np.asarray(items[-1].content[0], dtype=float)
            scales.append(float(np.sum((query - key) ** 2)))
    return max(float(np.median(scales)), float(np.finfo(float).eps))


@gate_check("P8")
def check_p8() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    beats_no_retrieval, robust, calibration_valid = [], [], []
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

        facts = _region_data(FULL, STORE_N, seed + 50)
        store = _build_store(model, facts)
        poisoned = _build_poisoned_store(model, facts, seed + 70)
        temperature = _retrieval_temperature(store, [_key(model, t) for t in ood])
        calibration_epistemic = [
            model.predict(model.encode(t.state.z), t.action).epistemic
            for t in _region_data(REGION, CALIBRATION_N, seed + 600)
        ]
        retrieval = calibrate_threshold(calibration_epistemic, alpha=RETRIEVAL_ALPHA)
        audit_epistemic = [
            model.predict(model.encode(t.state.z), t.action).epistemic
            for t in _region_data(REGION, CALIBRATION_AUDIT_N, seed + 700)
        ]
        audit_rate, audit_tolerance, audit_valid = audit_threshold(
            audit_epistemic, retrieval, tolerance=RETRIEVAL_AUDIT_TOLERANCE
        )
        calibration_valid.append(audit_valid)
        threshold = retrieval.value
        # P8-001 accuracy path (a single trusted store); P8-002 robustness paths over a
        # poisoned UNTRUSTED source: trust-blind (ignores provenance — a P8-001-style
        # agent), provenance-respecting alone, and provenance-respecting alongside the
        # trusted store (trust-ordered selection). Default min_trust=LOW excludes UNTRUSTED.
        gated_router = UncertaintyMemoryRouter([store], threshold=threshold)
        blind_router = UncertaintyMemoryRouter([poisoned], threshold=threshold,
                                               min_trust=Trust.UNTRUSTED)
        respecting_router = UncertaintyMemoryRouter([poisoned], threshold=threshold)
        mixed_router = UncertaintyMemoryRouter([poisoned, store], threshold=threshold)

        test = _region_data(FULL, TEST_N, seed + 200)
        none_err, gated_err, always_err = [], [], []
        blind_err, resp_err, mixed_err, retrieved = [], [], [], 0
        for t in test:
            key = _key(model, t)
            pred = model.predict(model.encode(t.state.z), t.action)
            target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
            parametric = np.asarray(pred.mean, dtype=float)
            clean_answer, _, _ = blend_retrieved_items(
                key, parametric, store.query(key), temperature
            )
            retrieved += int(pred.epistemic > threshold)
            none_err.append(float(np.mean((parametric - target) ** 2)))
            gated_err.append(float(np.mean(
                (_routed_answer(
                    gated_router, key, pred.epistemic, parametric, temperature
                ) - target) ** 2)))
            always_err.append(float(np.mean((clean_answer - target) ** 2)))
            blind_err.append(float(np.mean(
                (_routed_answer(
                    blind_router, key, pred.epistemic, parametric, temperature
                ) - target) ** 2)))
            resp_err.append(float(np.mean(
                (_routed_answer(
                    respecting_router, key, pred.epistemic, parametric, temperature
                ) - target) ** 2)))
            mixed_err.append(float(np.mean(
                (_routed_answer(
                    mixed_router, key, pred.epistemic, parametric, temperature
                ) - target) ** 2)))
        none_mse, gated_mse = float(np.mean(none_err)), float(np.mean(gated_err))
        always_mse, rate = float(np.mean(always_err)), retrieved / len(test)
        blind_mse, resp_mse, mixed_mse = (
            float(np.mean(blind_err)), float(np.mean(resp_err)), float(np.mean(mixed_err)))
        beats_no_retrieval.append(gated_mse * IMPROVEMENT_MARGIN <= none_mse)
        # Robustness (every seed): the poison genuinely harms a trust-blind agent,
        # provenance-respecting retrieval stays no worse than no-retrieval (untrusted
        # never overrides), and trust-ordering recovers the clean gated accuracy.
        robust.append(blind_mse >= none_mse * POISON_HARM_FACTOR
                      and resp_mse <= none_mse * ROBUST_TOL
                      and mixed_mse <= gated_mse * ROBUST_TOL)
        metrics |= {
            f"no_retrieval_mse_s{seed}": none_mse,
            f"gated_mse_s{seed}": gated_mse,
            f"always_retrieve_mse_s{seed}": always_mse,
            f"retrieval_rate_s{seed}": rate,
            f"retrieval_alpha_s{seed}": retrieval.alpha,
            f"retrieval_eta_s{seed}": retrieval.eta,
            f"retrieval_threshold_s{seed}": retrieval.value,
            f"retrieval_kernel_temperature_s{seed}": temperature,
            f"retrieval_calibration_updates_s{seed}": float(retrieval.updates),
            f"nominal_retrieval_online_rate_s{seed}": retrieval.trigger_rate,
            f"nominal_retrieval_retrospective_rate_s{seed}": exceedance_rate(
                calibration_epistemic, retrieval.value
            ),
            f"nominal_retrieval_audit_rate_s{seed}": audit_rate,
            f"nominal_retrieval_audit_tolerance_s{seed}": audit_tolerance,
            f"nominal_retrieval_audit_valid_s{seed}": float(audit_valid),
            f"poison_blind_mse_s{seed}": blind_mse,
            f"poison_respecting_mse_s{seed}": resp_mse,
            f"poison_mixed_mse_s{seed}": mixed_mse,
        }

    accuracy_met, robustness_met = all(beats_no_retrieval), all(robust)
    calibration_met = all(calibration_valid)
    passed = accuracy_met and robustness_met and calibration_met

    def med(key: str) -> float:
        return float(np.median([metrics[f"{key}_s{s}"] for s in SEEDS]))

    none_med, gated_med, always_med, rate_med = (
        med("no_retrieval_mse"), med("gated_mse"), med("always_retrieve_mse"), med("retrieval_rate"))
    blind_med, resp_med, mixed_med = (
        med("poison_blind_mse"), med("poison_respecting_mse"), med("poison_mixed_mse"))
    metrics |= {"no_retrieval_mse_median": none_med, "gated_mse_median": gated_med,
                "always_retrieve_mse_median": always_med, "retrieval_rate_median": rate_med,
                "poison_blind_mse_median": blind_med, "poison_respecting_mse_median": resp_med,
                "poison_mixed_mse_median": mixed_med, "accuracy_half_met": float(accuracy_met),
                "robustness_half_met": float(robustness_met),
                "retrieval_calibration_met": float(calibration_met)}
    detail = (
        f"accuracy: gated {gated_med:.4f} vs no-retrieval {none_med:.4f} "
        f"(>= x{IMPROVEMENT_MARGIN}/seed: {'MET' if accuracy_met else 'NOT MET'}) vs always "
        f"{always_med:.4f}, retrieval {rate_med:.0%}. "
        f"robustness: trust-blind swallows poison {blind_med:.4f} (>= x{POISON_HARM_FACTOR} "
        f"worse), provenance-respecting stays {resp_med:.4f} (<= no-retrieval) and trust-ordered "
        f"mixed recovers {mixed_med:.4f}: {'MET' if robustness_met else 'NOT MET'}. "
        f"independent nominal retrieval-rate audit: {'MET' if calibration_met else 'NOT MET'}"
    )
    return GateResult(phase="P8", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
