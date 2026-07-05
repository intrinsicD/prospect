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
observation is `codec.encode`d and stands in for the model's next-latent.

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

Robustness half (poisoned/UNTRUSTED external source never overrides) is P10-002; until it
lands the composite P10 gate is BLOCKED by design (mirrors P8-001/P8-002).

Run `p10` carries all four data-sentinels (the limited-region model feeds the P1 probes,
replay fidelity and an option-diversity rollout), plus the P9 gate-overfit sentinel.
"""
from __future__ import annotations

import numpy as np

from prospect.codec import UniversalCodec
from prospect.knowledge import ExternalKnowledgeSource
from prospect.memory import UncertaintyMemoryRouter
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
from .p8_knowledge import FULL, REGION, TRAIN_N, _env, _key, _region_data

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


def _ingest(codec: UniversalCodec, obs: Observation) -> np.ndarray:
    return np.asarray(codec.encode(obs).z, dtype=float)


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
    store = _external_store(model, _ood_data(STORE_N, seed + 50))
    radius = _reliability_radius(model, store, _ood_data(PROBE_N, seed + 900))
    seen_epi = sorted(model.predict(model.encode(t.state.z), t.action).epistemic for t in heldout)
    threshold = seen_epi[int(EPISTEMIC_QUANTILE * len(seen_epi))]
    router = UncertaintyMemoryRouter([store], threshold=threshold)

    # Per query: model-alone (bypass), external-ingested (clean / corrupted content), the
    # two-stage decision (uncertainty consult AND distance trust), and the no-distance-gate
    # control (uncertainty consult only). OOD = the un-learnable band.
    model_err, ext_err, corrupt_err = [], [], []
    consulted, ingested, is_ood = [], [], []
    rng_c = np.random.default_rng(seed + 404)
    for t in _region_data(FULL, TEST_N, seed + 200):
        key = _key(model, t)
        pred = model.predict(model.encode(t.state.z), t.action)
        target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        model_err.append(float(np.mean((np.asarray(pred.mean, dtype=float) - target) ** 2)))
        item = store.query(key)[0]
        content: Observation = item.content[1]
        ext_err.append(float(np.mean((_ingest(codec, content) - target) ** 2)))
        corrupt = Observation(modality=content.modality,
                              data=content.data + rng_c.normal(0.0, CORRUPT_SCALE, size=content.data.shape))
        corrupt_err.append(float(np.mean((_ingest(codec, corrupt) - target) ** 2)))
        consult = router.route(None, pred.epistemic) is not None  # uncertainty gate (ADR-0004)
        within = float(np.sum((key - np.asarray(item.content[0], dtype=float)) ** 2)) <= radius
        consulted.append(consult)
        ingested.append(consult and within)  # distance gate too (P9-007)
        is_ood.append(abs(float(t.state.z[2])) > REGION)  # obs = [cos, sin, omega]

    model_e, ext_e, corr_e = np.array(model_err), np.array(ext_err), np.array(corrupt_err)
    consult_a, ingest_a, ood_a = np.array(consulted), np.array(ingested), np.array(is_ood)
    gated = np.where(ingest_a, ext_e, model_e)          # two-stage gated
    corrupt_gated = np.where(ingest_a, corr_e, model_e)  # gated, but ingest corrupted content
    nodist = np.where(consult_a, ext_e, model_e)         # no-distance-gate control

    def sub(arr: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(arr[mask])) if mask.any() else 0.0

    return {
        f"model_ood_mse_s{seed}": sub(model_e, ood_a),
        f"gated_ood_mse_s{seed}": sub(gated, ood_a),
        f"model_seen_mse_s{seed}": sub(model_e, ~ood_a),
        f"gated_seen_mse_s{seed}": sub(gated, ~ood_a),
        f"nodist_seen_mse_s{seed}": sub(nodist, ~ood_a),
        f"gated_mse_s{seed}": float(np.mean(gated)),
        f"corrupt_gated_mse_s{seed}": float(np.mean(corrupt_gated)),
        f"seen_ingest_rate_s{seed}": sub(ingest_a.astype(float), ~ood_a),
        f"ood_ingest_rate_s{seed}": sub(ingest_a.astype(float), ood_a),
        f"seen_consult_rate_s{seed}": sub(consult_a.astype(float), ~ood_a),
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

    competence_met = gated_ood * COMPETENCE_FACTOR <= model_ood and gated_seen <= model_seen * SEEN_TOL
    codec_met = corrupt >= gated * CORRUPT_MARGIN and gated_ood < model_ood
    gates_met = (seen_ingest <= SEEN_INGEST_MAX and ood_ingest >= OOD_INGEST_MIN
                 and nodist_seen >= gated_seen * DISTANCE_MARGIN)
    capability_met = competence_met and codec_met and gates_met
    robustness_met = False  # P10-002 (poisoned/UNTRUSTED external source never overrides)
    passed = capability_met and robustness_met

    metrics |= {"model_ood_mse_median": model_ood, "gated_ood_mse_median": gated_ood,
                "model_seen_mse_median": model_seen, "gated_seen_mse_median": gated_seen,
                "nodist_seen_mse_median": nodist_seen, "gated_mse_median": gated,
                "corrupt_gated_mse_median": corrupt, "seen_ingest_rate_median": seen_ingest,
                "ood_ingest_rate_median": ood_ingest, "competence_met": float(competence_met),
                "codec_ingestion_met": float(codec_met), "gates_met": float(gates_met),
                "capability_met": float(capability_met), "robustness_met": float(robustness_met)}
    detail = (
        f"external knowledge through the codec — competence: OOD gated {gated_ood:.4f} vs model "
        f"{model_ood:.4f} (>= x{COMPETENCE_FACTOR}), seen no-harm {gated_seen:.4f} vs "
        f"{model_seen:.4f}: {'MET' if competence_met else 'NOT MET'}. codec carries it: corrupted "
        f"{corrupt:.4f} vs clean {gated:.4f} (>= x{CORRUPT_MARGIN}): {'MET' if codec_met else 'NOT MET'}. "
        f"gates load-bearing: ingest seen {seen_ingest:.0%}/OOD {ood_ingest:.0%}, remove distance-gate "
        f"-> seen {nodist_seen:.4f} vs {gated_seen:.4f} (>= x{DISTANCE_MARGIN}): "
        f"{'MET' if gates_met else 'NOT MET'}. CAPABILITY {'MET' if capability_met else 'NOT MET'}; "
        f"composite BLOCKED pending P10-002 (external-source trust robustness)"
    )
    return GateResult(phase="P10", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
