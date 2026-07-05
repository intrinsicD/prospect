"""P11 eval: compute-as-action tools (task P11-001).

A tool is a knowledge source that COMPUTES its answer on demand (ADR-0004 rule 2), exact
for any query with no store or coverage limit — but each call has a COST. So calling it is
an action gated by uncertainty AND cost: invoke the expensive exact tool only where the
cheap parametric model is unreliable. This is the third tier (parametric / retrieved /
computed) and the clean case for the cost gate — unlike a lookup KB (P10), correctness is
never in question, only *when it is worth calling*.

Setup (per seed): a limited-region pendulum `FlatWorldModel` (confident in-region,
P9-005-uncertain outside), a `UniversalCodec` distilled to `encode_target` (so a tool
result ingests into the prediction space, reusing P10), and a `ToolSource` whose compute
is an EXACT next-state oracle — it runs the true env one step. Per query the model's
epistemic gates the call.

Three criteria (1-step latent MSE + call counts):
1. **Tool helps where the model is uncertain** — on OOD the tool result (codec-ingested)
   MSE is far below the model alone (the tool computes what the model can't extrapolate).
2. **Uncertainty spends the budget well** — at an EQUAL call budget the uncertainty-gated
   policy beats a RANDOM-gated one: the VoE signal calls the tool on the OOD queries, not
   the seen ones (seen call-rate ~0). The load-bearing negative control.
3. **Cost-gating is the sweet spot** — uncertainty-gated is strictly better than
   never-calling (it catches the OOD errors) AND uses strictly fewer calls than
   always-calling (it skips the calls it doesn't need where the model is already right).

Run `p11` carries all four data-sentinels plus the P9 gate-overfit sentinel.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np

from prospect.knowledge import ToolSource
from prospect.memory import UncertaintyMemoryRouter
from prospect.types import Action, Modality, Observation, Trust
from prospect.world_model import FlatWorldModel

from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe
from .p3_replay import log_replay_fidelity
from .p7_continual import _log_option_diversity
from .p8_knowledge import FULL, REGION, TRAIN_N, _env, _region_data
from .p10_external import PROBE_N, TEST_N, _distill_codec, _ingest

RUN_ID = "p11"
SEEDS = [0, 1, 2]
EPISTEMIC_QUANTILE = 0.95   # call the tool above the 95th-percentile seen-region epistemic
TOOL_HELP_FACTOR = 2.0      # tool OOD MSE * this <= model-alone OOD MSE
RANDOM_MARGIN = 1.2         # random-gated MSE >= uncertainty-gated MSE * this
NEVER_FACTOR = 2.0          # gated MSE * this <= never-call MSE (gating strictly helps)
CALL_FRACTION = 0.7         # gated call-rate <= this (strictly fewer calls than always-call=1.0)
SEEN_RATE_MAX = 0.15        # tool called on <= this fraction of seen queries


def _oracle() -> ToolSource:
    """An exact next-state tool: compute((obs, action)) runs the true env one step. Exact
    for any query — the tool COMPUTES, it does not look up (contrast the P10 KB)."""
    def compute(query: object) -> Observation:
        obs_raw, action = cast("tuple[Any, Any]", query)
        obs = np.asarray(obs_raw, dtype=float)
        env = _env()
        env.reset(seed=0)
        env.set_state(float(np.arctan2(obs[1], obs[0])), float(obs[2]))
        next_obs, _, _ = env.step(Action(data=np.asarray(action, dtype=float)))
        return Observation(modality=Modality.STATE, data=np.asarray(next_obs.data, dtype=float))

    return ToolSource(compute=compute, trust=Trust.MEDIUM, source="oracle")


def _seed_metrics(seed: int, log: RunLog) -> dict[str, float]:
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
    tool = _oracle()
    router = UncertaintyMemoryRouter([tool], threshold=0.0)  # threshold set below
    seen_epi = sorted(model.predict(model.encode(t.state.z), t.action).epistemic for t in heldout)
    router.threshold = seen_epi[int(EPISTEMIC_QUANTILE * len(seen_epi))]

    model_err, tool_err, consult, is_ood = [], [], [], []
    for t in _region_data(FULL, TEST_N, seed + 200):
        pred = model.predict(model.encode(t.state.z), t.action)
        target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        model_err.append(float(np.mean((np.asarray(pred.mean, dtype=float) - target) ** 2)))
        source = router.route(None, pred.epistemic)  # uncertainty gate (ADR-0004 rule 2)
        item = tool.query((t.state.z, t.action.data))[0]  # the tool computes (and counts the call)
        tool_err.append(float(np.mean((_ingest(codec, item.content[1]) - target) ** 2)))
        consult.append(source is not None)
        is_ood.append(abs(float(t.state.z[2])) > REGION)  # obs = [cos, sin, omega]

    model_e, tool_e = np.array(model_err), np.array(tool_err)
    consult_a, ood_a = np.array(consult), np.array(is_ood)
    gated = np.where(consult_a, tool_e, model_e)
    budget = int(consult_a.sum())
    rand_mask = np.zeros(len(gated), dtype=bool)
    rand_mask[np.random.default_rng(seed + 606).permutation(len(gated))[:budget]] = True
    random_gated = np.where(rand_mask, tool_e, model_e)

    def sub(arr: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(arr[mask])) if mask.any() else 0.0

    return {
        f"model_ood_mse_s{seed}": sub(model_e, ood_a),
        f"tool_ood_mse_s{seed}": sub(tool_e, ood_a),
        f"never_mse_s{seed}": float(np.mean(model_e)),
        f"always_mse_s{seed}": float(np.mean(tool_e)),
        f"gated_mse_s{seed}": float(np.mean(gated)),
        f"random_gated_mse_s{seed}": float(np.mean(random_gated)),
        f"gated_call_rate_s{seed}": float(np.mean(consult_a)),
        f"seen_call_rate_s{seed}": sub(consult_a.astype(float), ~ood_a),
        f"ood_call_rate_s{seed}": sub(consult_a.astype(float), ood_a),
    }


@gate_check("P11")
def check_p11() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    for seed in SEEDS:
        metrics |= _seed_metrics(seed, log)

    def med(key: str) -> float:
        return float(np.median([metrics[f"{key}_s{s}"] for s in SEEDS]))

    model_ood, tool_ood = med("model_ood_mse"), med("tool_ood_mse")
    never, always, gated, random_gated = med("never_mse"), med("always_mse"), med("gated_mse"), med("random_gated_mse")
    gated_rate, seen_rate, ood_rate = med("gated_call_rate"), med("seen_call_rate"), med("ood_call_rate")

    tool_helps = tool_ood * TOOL_HELP_FACTOR <= model_ood
    spends_well = random_gated >= gated * RANDOM_MARGIN and seen_rate <= SEEN_RATE_MAX
    cost_sweet_spot = gated * NEVER_FACTOR <= never and gated_rate <= CALL_FRACTION
    passed = tool_helps and spends_well and cost_sweet_spot

    metrics |= {"model_ood_mse_median": model_ood, "tool_ood_mse_median": tool_ood,
                "never_mse_median": never, "always_mse_median": always, "gated_mse_median": gated,
                "random_gated_mse_median": random_gated, "gated_call_rate_median": gated_rate,
                "seen_call_rate_median": seen_rate, "ood_call_rate_median": ood_rate,
                "tool_helps": float(tool_helps), "spends_well": float(spends_well),
                "cost_sweet_spot": float(cost_sweet_spot)}
    detail = (
        f"compute-as-action tools — tool helps: OOD tool {tool_ood:.4f} vs model {model_ood:.4f} "
        f"(>= x{TOOL_HELP_FACTOR}): {'MET' if tool_helps else 'NOT MET'}. uncertainty spends well: "
        f"gated {gated:.4f} vs random {random_gated:.4f} (>= x{RANDOM_MARGIN}), call-rate seen "
        f"{seen_rate:.0%}/OOD {ood_rate:.0%}: {'MET' if spends_well else 'NOT MET'}. cost sweet spot: "
        f"gated {gated:.4f} vs never {never:.4f} (>= x{NEVER_FACTOR}) at {gated_rate:.0%} calls vs "
        f"always 100% (always MSE {always:.4f}): {'MET' if cost_sweet_spot else 'NOT MET'}. "
        f"P11 {'PASS' if passed else 'BLOCKED'}"
    )
    return GateResult(phase="P11", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
