"""P10 eval: EXTERNAL knowledge through the codec (task P10-001 capability half).

P8's `SemanticStore` answers with next-latents in the model's own space — the agent's
digested experience. This phase adds a genuinely *external* source: it answers with raw
**content** (an observation the agent never sensed) that the agent must **encode through
its codec** (ADR-0004 rule 1, knowledge-as-tokens) to recover a usable latent —
extending competence to a regime the parametric model cannot derive from experience (R8).

Setup (per seed): a `FlatWorldModel` trained on a LIMITED region of the pendulum
(|omega| <= REGION) — confident there, and (P9-005 distance-aware) uncertain outside it.
A `UniversalCodec` is distilled to reproduce the model's `encode_target` (the prediction-
target space) so an ingested observation is directly comparable to a prediction. An
`ExternalKnowledgeSource` holds OOD-ONLY facts `(key=(latent,action), content=true-next-
observation)` — knowledge the model can't extrapolate, and DELIBERATELY complementary:
querying it for a seen state (which the model already handles) returns an irrelevant OOD
fact. So retrieval rides the mature two-stage gate: the **uncertainty** gate decides when
to consult external knowledge (ADR-0004: an action taken only when the model is lost), and
the **distance** gate decides whether to trust the retrieved fact (P9-007: reliability is
closeness — a far fact at a seen query is skipped). On an accepted retrieval the
ranked top-three observations are individually `codec.encode`d, distance-kernel
aggregated, and blended with the model's next-latent (U-005).

Capability half (P10-001), three criteria — all in 1-step latent MSE:
1. **Competence extension** — on OOD queries the two-stage-gated external (codec-ingested)
   MSE is a large factor below model-alone, while on SEEN queries it is no worse (the
   distance gate skips the irrelevant OOD facts that a seen false-consult would fetch).
2. **The codec carries it** — corrupting the retrieved observation before ingestion raises
   the MSE (metamorphic: the answer flows through the codec from the content), and the
   codec-ingested latent beats the model's own OOD extrapolation (a bypass control).
3. **Both gates are load-bearing** — external knowledge is ingested on OOD (relevant) and
   ~never on seen; and REMOVING the distance gate (ingest whenever uncertainty fires)
   makes seen accuracy worse, because seen false-consults then fetch far, wrong facts.

Robustness half (P10-002): a poisoned `UNTRUSTED` external source (corrupted observations
over the same keys) is the attack surface. A trust-blind agent ingests it and does WORSE
than no-retrieval (the poison, encoded through the codec, bites); a provenance-respecting
router (`min_trust` floor + trust-ordered selection, P8-002) never lets the untrusted
source override the model, and with a trusted source also present trust-orders to it and
recovers the clean accuracy. Untrusted content is data, never instruction (ADR-0004).

Run `p10` carries all four data-sentinels (the limited-region model feeds the P1 probes,
replay fidelity and an option-diversity rollout), plus the P9 gate-overfit sentinel.
"""
from __future__ import annotations

import numpy as np

from prospect.codec import UniversalCodec
from prospect.knowledge import ExternalKnowledgeSource
from prospect.memory import UncertaintyMemoryRouter, blend_retrieved_items
from prospect.types import (
    Action,
    KnowledgeItem,
    LatentState,
    Modality,
    Observation,
    Provenance,
    Transition,
    Trust,
)
from prospect.world_model import FlatWorldModel

from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe
from .p3_replay import log_replay_fidelity
from .p7_continual import _log_option_diversity
from .p8_knowledge import (
    FULL,
    REGION,
    TRAIN_N,
    _env,
    _key,
    _region_data,
    _retrieval_temperature,
)

RUN_ID = "p10"
SEEDS = [0, 1, 2]
STATE_DIM = 3  # pendulum observation [cos theta, sin theta, omega]
STORE_N, TEST_N, PROBE_N = 4000, 400, 256  # external KB is OOD-only; test spans full range
DISTILL_N, DISTILL_STEPS, DISTILL_BATCH = 3072, 800, 128
EPISTEMIC_QUANTILE = 0.9    # consult external above the 90th-percentile seen-region epistemic
RELIABILITY_MULT = 2.0      # trust a retrieved fact only within 2x the KB's coverage distance (P9-007)
COMPETENCE_FACTOR = 2.0     # gated OOD MSE * this <= model-alone OOD MSE
SEEN_TOL = 1.3              # gated SEEN MSE <= model-alone SEEN MSE * this (no harm)
SEEN_INGEST_MAX = 0.05      # external ingested on <= this fraction of seen queries (concentrated on OOD)
OOD_INGEST_MIN = 0.5        # external ingested on >= this fraction of OOD queries (actually used)
CORRUPT_MARGIN = 1.5        # corrupted-content MSE >= clean gated MSE * this (codec carries it)
DISTANCE_MARGIN = 1.5       # no-distance-gate SEEN MSE >= gated SEEN MSE * this (distance gate load-bearing)
CORRUPT_SCALE = 1.0         # obs-space noise added to a retrieved observation (metamorphic)
POISON_SCALE = 2.0          # obs-space noise in the poisoned UNTRUSTED store's content (P10-002)
POISON_HARM_FACTOR = 2.0    # trust-blind ingestion of poison must be >= this x worse than no-retrieval
ROBUST_TOL = 1.1            # provenance-respecting MSE must stay within this of its clean target


def _ood_data(n: int, seed: int) -> list[Transition]:
    """Transitions in the OOD velocity band |omega| in [REGION, FULL] — the knowledge the
    model can't extrapolate. External facts and (part of) the test set live here."""
    env = _env()
    rng = np.random.default_rng(seed)
    out: list[Transition] = []
    for _ in range(n):
        env.reset(seed=0)
        omega = float(rng.uniform(REGION, FULL)) * (1.0 if rng.uniform() < 0.5 else -1.0)
        obs = env.set_state(float(rng.uniform(-np.pi, np.pi)), omega)
        action = Action(data=np.array([float(rng.uniform(-2.0, 2.0))]))
        next_obs, reward, _ = env.step(action)
        out.append(Transition(state=LatentState(z=obs.data), action=action,
                              next_state=LatentState(z=next_obs.data), reward=reward))
    return out


def _distill_codec(model: FlatWorldModel, seed: int) -> UniversalCodec:
    """Distil a UniversalCodec to reproduce `encode_target` (the prediction-target space)
    on the STATE modality over the FULL state range — so `codec.encode(obs)` lands where
    predictions are compared, for any obs including OOD (perception, not dynamics)."""
    codec = UniversalCodec({Modality.STATE: STATE_DIM}, latent_dim=model.latent_dim, seed=seed)
    rng = np.random.default_rng(seed + 202)
    states = np.stack([
        np.asarray(_env().set_state(float(rng.uniform(-np.pi, np.pi)),
                                    float(rng.uniform(-FULL, FULL))).data, dtype=float)
        for _ in range(DISTILL_N)])
    targets = np.stack([np.asarray(model.encode_target(s).z, dtype=float) for s in states])
    fit_rng = np.random.default_rng(seed + 303)
    for _ in range(DISTILL_STEPS):
        idx = fit_rng.integers(0, DISTILL_N, size=DISTILL_BATCH)
        codec.distill_encode(states[idx], Modality.STATE, targets[idx])
    return codec


def _external_store(model: FlatWorldModel, facts: list[Transition]) -> ExternalKnowledgeSource:
    """External KB: (state,action) -> the true next OBSERVATION (raw content, to be
    ingested through the codec) — not a pre-digested latent. Trust HIGH (a vetted source;
    the poisoned/UNTRUSTED case is P10-002)."""
    store = ExternalKnowledgeSource(trust=Trust.HIGH)
    for t in facts:
        content = Observation(modality=Modality.STATE, data=np.asarray(t.next_state.z, dtype=float))
        store.write(KnowledgeItem(content=(_key(model, t), content),
                                  provenance=Provenance(source="reference", trust=Trust.HIGH)))
    return store


def _poisoned_store(model: FlatWorldModel, facts: list[Transition], seed: int) -> ExternalKnowledgeSource:
    """An UNTRUSTED external source over the SAME keys as the clean KB, but whose content
    is a corrupted observation (P10-002). A provenance-respecting router must never let it
    override the model; a trust-blind one ingests garbage through the codec (ADR-0004)."""
    rng = np.random.default_rng(seed)
    store = ExternalKnowledgeSource(trust=Trust.UNTRUSTED)
    for t in facts:
        corrupt = np.asarray(t.next_state.z, dtype=float) + rng.normal(0.0, POISON_SCALE,
                                                                       size=np.asarray(t.next_state.z).shape)
        content = Observation(modality=Modality.STATE, data=corrupt)
        store.write(KnowledgeItem(content=(_key(model, t), content),
                                  provenance=Provenance(source="poisoned", trust=Trust.UNTRUSTED)))
    return store


def _ingest(codec: UniversalCodec, obs: Observation) -> np.ndarray:
    return np.asarray(codec.encode(obs).z, dtype=float)


def _ingest_content(codec: UniversalCodec, content: object) -> np.ndarray:
    if not isinstance(content, Observation):
        raise TypeError("external retrieval content must be an Observation")
    return _ingest(codec, content)


def _routed_latent(router: UncertaintyMemoryRouter, key: np.ndarray, epistemic: float,
                   radius: float, temperature: float, codec: UniversalCodec,
                   fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    """The two-stage gate for one router config: CONSULT external when uncertain (route
    respects trust), then TRUST the fact only when close (distance gate, P9-007). Returns
    (latent, ingested); on no-consult or a far fact it falls back to the model's own."""
    source = router.route(None, epistemic)
    if source is None:
        return fallback, False  # confident, or nothing clears the trust floor
    blended, reliability, _ = blend_retrieved_items(
        key,
        fallback,
        source.query(key),
        temperature,
        radius,
        answer_transform=lambda content: _ingest_content(codec, content),
    )
    return blended, reliability > 0.0


def _reliability_radius(model: FlatWorldModel, store: ExternalKnowledgeSource,
                        probes: list[Transition]) -> float:
    """Calibrate the distance gate to the KB's coverage (P9-007): the median key-distance
    from an in-coverage (OOD) query to its nearest fact, scaled up. A seen query sits far
    outside this — its nearest OOD fact is not to be trusted."""
    covered = [
        float(np.sum((_key(model, t) - np.asarray(store.query(_key(model, t))[0].content[0],
                                                   dtype=float)) ** 2))
        for t in probes
    ]
    return RELIABILITY_MULT * float(np.median(covered))


def _seed_metrics(seed: int, log: RunLog) -> dict[str, float]:
    """One seed: train the limited-region model (+ sentinel logging), distil the codec,
    build the OOD-only external KB, and measure the three capability criteria."""
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

    codec = _distill_codec(model, seed)
    facts = _ood_data(STORE_N, seed + 50)
    store = _external_store(model, facts)
    poisoned = _poisoned_store(model, facts, seed + 70)  # same keys, corrupted content (P10-002)
    retrieval_probes = _ood_data(PROBE_N, seed + 900)
    radius = _reliability_radius(model, store, retrieval_probes)
    temperature = _retrieval_temperature(
        store, [_key(model, t) for t in retrieval_probes]
    )
    seen_epi = sorted(model.predict(model.encode(t.state.z), t.action).epistemic for t in heldout)
    threshold = seen_epi[int(EPISTEMIC_QUANTILE * len(seen_epi))]
    # P10-001 capability path (a single trusted store); P10-002 robustness paths over a
    # poisoned UNTRUSTED source: trust-blind (ignores provenance), provenance-respecting
    # alone, and respecting alongside the trusted store (trust-ordered). Default
    # min_trust=LOW excludes UNTRUSTED.
    clean_router = UncertaintyMemoryRouter([store], threshold=threshold)
    blind_router = UncertaintyMemoryRouter([poisoned], threshold=threshold, min_trust=Trust.UNTRUSTED)
    resp_router = UncertaintyMemoryRouter([poisoned], threshold=threshold)
    mixed_router = UncertaintyMemoryRouter([poisoned, store], threshold=threshold)

    def err(latent: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((latent - target) ** 2))

    model_err, gated_err, corrupt_err, nodist_err = [], [], [], []
    blind_err, resp_err, mixed_err = [], [], []
    consulted, ingested, is_ood = [], [], []
    rng_c = np.random.default_rng(seed + 404)
    for t in _region_data(FULL, TEST_N, seed + 200):
        key = _key(model, t)
        pred = model.predict(model.encode(t.state.z), t.action)
        target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        model_mean = np.asarray(pred.mean, dtype=float)
        model_err.append(err(model_mean, target))

        clean_items = store.query(key)
        consult = clean_router.route(None, pred.epistemic) is not None  # uncertainty gate (ADR-0004)
        clean_latent, clean_reliability, _ = blend_retrieved_items(
            key,
            model_mean,
            clean_items,
            temperature,
            radius,
            answer_transform=lambda content: _ingest_content(codec, content),
        )
        ingest = consult and clean_reliability > 0.0  # + distance gate (P9-007)
        gated_err.append(err(clean_latent if consult else model_mean, target))
        # Infinite radius is the no-distance-gate ablation: every ranked neighbor is
        # eligible and the readout fully trusts their kernel aggregate.
        nodist_latent, _, _ = blend_retrieved_items(
            key,
            model_mean,
            clean_items,
            temperature,
            np.inf,
            answer_transform=lambda content: _ingest_content(codec, content),
        )
        nodist_err.append(err(nodist_latent if consult else model_mean, target))

        def corrupt_content(content: object) -> np.ndarray:
            if not isinstance(content, Observation):
                raise TypeError("external retrieval content must be an Observation")
            corrupt = Observation(
                modality=content.modality,
                data=content.data
                + rng_c.normal(0.0, CORRUPT_SCALE, size=content.data.shape),
            )
            return _ingest(codec, corrupt)

        corrupt_latent, _, _ = blend_retrieved_items(
            key,
            model_mean,
            clean_items,
            temperature,
            radius,
            answer_transform=corrupt_content,
        )
        corrupt_err.append(err(corrupt_latent if consult else model_mean, target))
        # P10-002: the same two-stage gate under each trust config.
        blind_err.append(err(_routed_latent(
            blind_router, key, pred.epistemic, radius, temperature, codec, model_mean
        )[0], target))
        resp_err.append(err(_routed_latent(
            resp_router, key, pred.epistemic, radius, temperature, codec, model_mean
        )[0], target))
        mixed_err.append(err(_routed_latent(
            mixed_router, key, pred.epistemic, radius, temperature, codec, model_mean
        )[0], target))
        consulted.append(consult)
        ingested.append(ingest)
        is_ood.append(abs(float(t.state.z[2])) > REGION)  # obs = [cos, sin, omega]

    model_e, gated_e, corr_e, nodist_e = map(np.array, (model_err, gated_err, corrupt_err, nodist_err))
    blind_e, resp_e, mixed_e = map(np.array, (blind_err, resp_err, mixed_err))
    ingest_a, ood_a = np.array(ingested), np.array(is_ood)

    def sub(arr: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(arr[mask])) if mask.any() else 0.0

    return {
        f"model_ood_mse_s{seed}": sub(model_e, ood_a),
        f"gated_ood_mse_s{seed}": sub(gated_e, ood_a),
        f"model_seen_mse_s{seed}": sub(model_e, ~ood_a),
        f"gated_seen_mse_s{seed}": sub(gated_e, ~ood_a),
        f"nodist_seen_mse_s{seed}": sub(nodist_e, ~ood_a),
        f"gated_mse_s{seed}": float(np.mean(gated_e)),
        f"corrupt_gated_mse_s{seed}": float(np.mean(corr_e)),
        f"none_mse_s{seed}": float(np.mean(model_e)),
        f"blind_mse_s{seed}": float(np.mean(blind_e)),
        f"respecting_mse_s{seed}": float(np.mean(resp_e)),
        f"mixed_mse_s{seed}": float(np.mean(mixed_e)),
        f"retrieval_kernel_temperature_s{seed}": temperature,
        f"seen_ingest_rate_s{seed}": sub(ingest_a.astype(float), ~ood_a),
        f"ood_ingest_rate_s{seed}": sub(ingest_a.astype(float), ood_a),
    }


@gate_check("P10")
def check_p10() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    for seed in SEEDS:
        metrics |= _seed_metrics(seed, log)

    def med(key: str) -> float:
        return float(np.median([metrics[f"{key}_s{s}"] for s in SEEDS]))

    model_ood, gated_ood = med("model_ood_mse"), med("gated_ood_mse")
    model_seen, gated_seen = med("model_seen_mse"), med("gated_seen_mse")
    nodist_seen = med("nodist_seen_mse")
    gated, corrupt = med("gated_mse"), med("corrupt_gated_mse")
    seen_ingest, ood_ingest = med("seen_ingest_rate"), med("ood_ingest_rate")
    none_mse, blind, resp, mixed = med("none_mse"), med("blind_mse"), med("respecting_mse"), med("mixed_mse")

    competence_met = gated_ood * COMPETENCE_FACTOR <= model_ood and gated_seen <= model_seen * SEEN_TOL
    codec_met = corrupt >= gated * CORRUPT_MARGIN and gated_ood < model_ood
    gates_met = (seen_ingest <= SEEN_INGEST_MAX and ood_ingest >= OOD_INGEST_MIN
                 and nodist_seen >= gated_seen * DISTANCE_MARGIN)
    capability_met = competence_met and codec_met and gates_met
    # P10-002: a trust-blind agent swallows the poison; a provenance-respecting router
    # never lets the UNTRUSTED source override the model; trust-ordering to a trusted
    # source recovers the clean accuracy (ADR-0004).
    robustness_met = (blind >= none_mse * POISON_HARM_FACTOR and resp <= none_mse * ROBUST_TOL
                      and mixed <= gated * ROBUST_TOL)
    passed = capability_met and robustness_met

    metrics |= {"model_ood_mse_median": model_ood, "gated_ood_mse_median": gated_ood,
                "model_seen_mse_median": model_seen, "gated_seen_mse_median": gated_seen,
                "nodist_seen_mse_median": nodist_seen, "gated_mse_median": gated,
                "corrupt_gated_mse_median": corrupt, "seen_ingest_rate_median": seen_ingest,
                "ood_ingest_rate_median": ood_ingest, "none_mse_median": none_mse,
                "blind_mse_median": blind, "respecting_mse_median": resp, "mixed_mse_median": mixed,
                "competence_met": float(competence_met), "codec_ingestion_met": float(codec_met),
                "gates_met": float(gates_met), "capability_met": float(capability_met),
                "robustness_met": float(robustness_met)}
    detail = (
        f"external knowledge through the codec — competence: OOD gated {gated_ood:.4f} vs model "
        f"{model_ood:.4f} (>= x{COMPETENCE_FACTOR}), seen no-harm {gated_seen:.4f} vs "
        f"{model_seen:.4f}: {'MET' if competence_met else 'NOT MET'}. codec carries it: corrupted "
        f"{corrupt:.4f} vs clean {gated:.4f} (>= x{CORRUPT_MARGIN}): {'MET' if codec_met else 'NOT MET'}. "
        f"gates load-bearing: ingest seen {seen_ingest:.0%}/OOD {ood_ingest:.0%}, no-distance-gate "
        f"seen {nodist_seen:.4f} vs {gated_seen:.4f} (>= x{DISTANCE_MARGIN}): "
        f"{'MET' if gates_met else 'NOT MET'}. robustness: trust-blind swallows poison {blind:.4f} "
        f"(>= x{POISON_HARM_FACTOR} vs no-retrieval {none_mse:.4f}), provenance-respecting stays "
        f"{resp:.4f} (<= no-retrieval), trust-ordered recovers {mixed:.4f}: "
        f"{'MET' if robustness_met else 'NOT MET'}. P10 {'PASS' if passed else 'BLOCKED'}"
    )
    return GateResult(phase="P10", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
