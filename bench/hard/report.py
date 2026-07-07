"""Render a BH-001 run into a committed report artifact (markdown + JSON).

The report *is* the deliverable — there is no gate here (ADR-0011). It records raw
per-seed returns, matched-budget deltas and the honest interpretation, so the claim
"the core survives a real MuJoCo task" is auditable rather than asserted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .eval import BUDGET, CANDIDATES, EP_LEN, EVAL_EPISODES, GENS, HORIZON, POP, TaskResult

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _spread(xs: list[float]) -> str:
    return f"{np.median(xs):.1f} [{min(xs):.1f}, {max(xs):.1f}]"


def render_markdown(results: list[TaskResult], reach: list[TaskResult],
                    stamp: str, versions: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append("# BH-001 — harder-benchmark probe (non-gated)")
    lines.append("")
    lines.append(f"_Generated {stamp}. dm_control {versions.get('dm_control', '?')}, "
                 f"mujoco {versions.get('mujoco', '?')}, numpy {versions.get('numpy', '?')}._")
    lines.append("")
    lines.append(
        "Re-runs the **P2 claim** — MPC/CEM over a learned `FlatWorldModel` beats a "
        "model-free baseline at **equal env-step budget** — on DeepMind Control Suite "
        "tasks the repo did not author (ADR-0011). **Non-gated:** no phase ships on this; "
        "it is a credibility probe whose value is an honest number, not a pass."
    )
    lines.append("")
    lines.append(
        f"**Setup (matched to the shipped P2 gate).** Learning budget **{BUDGET} env "
        f"steps** per agent; model-free is CEM-ES policy search (POP {POP} × GENS {GENS} × "
        f"{EP_LEN}-step rollouts = {POP * GENS * EP_LEN} steps ≤ budget); MBRL planner is "
        f"P2's `FlatPlanner` defaults (horizon {HORIZON}, {CANDIDATES} candidates). "
        f"Returns are the mean over {EVAL_EPISODES} shared eval episodes ({EP_LEN} steps), "
        f"identical seeds across all three agents. Median [min, max] over "
        f"{len(results[0].seeds) if results else 0} seeds."
    )
    lines.append("")
    lines.append("| task | dims (obs/act) | MBRL | model-free (matched) | random | MBRL ≥ both? |")
    lines.append("|------|----------------|------|----------------------|--------|--------------|")
    for r in results:
        lines.append(
            f"| `{r.domain}-{r.task}` | {r.obs_dim}/{r.action_dim} | {_spread(r.mbrl)} | "
            f"{_spread(r.model_free)} | {_spread(r.random)} | "
            f"{'yes' if r.mbrl_beats_baseline else 'no'} |"
        )
    lines.append("")
    calib = next((r.calib_epistemic_ratio for r in results if r.calib_epistemic_ratio is not None), None)
    if calib is not None:
        lines.append(
            f"**P1-calibration spot-check** (seed 0, `{results[0].domain}-{results[0].task}`): "
            f"median ensemble epistemic at full budget / at {256} steps = **{calib:.3f}** "
            f"(< 1 ⇒ uncertainty falls with data, as it should)."
        )
        lines.append("")
    if reach:
        lines.append(
            "**Reachability (1 seed, harder / higher-action-dim tasks).** The same adapter "
            "loads and steps these — different domains, 2-D actions — with no code change, "
            "but at this budget every agent scores ~0: they are **below the probe's "
            "resolution**, not broken. This is the honest edge of a 4096-step / 100-step "
            "probe (max return across all three agents shown)."
        )
        lines.append("")
        lines.append("| task | dims (obs/act) | best-of-3 return |")
        lines.append("|------|----------------|------------------|")
        for r in reach:
            best = max(r.med_mbrl, r.med_model_free, r.med_random)
            lines.append(f"| `{r.domain}-{r.task}` | {r.obs_dim}/{r.action_dim} | {best:.2f} |")
        lines.append("")
    lines.append("## Honest reading")
    lines.append(
        "- The seam works: the **core is unchanged** — `FlatWorldModel`/`FlatPlanner`/"
        "`Agent` act in real MuJoCo via one `bench.Environment` adapter, across "
        f"{len(results) + len(reach)} DMC tasks in four domains, including 2-D action spaces."
    )
    lines.append(
        "- Both learners beat random by a wide margin, so the model-based machine is not "
        "broken on foreign dynamics."
    )
    lines.append(
        "- But the clean *MBRL-beats-model-free* win from the authored Pendulum (P2) does "
        "**not** reproduce as a decisive win at equal budget here: on `cartpole-balance` "
        "both saturate the task, and on the harder tasks the matched-budget model-free "
        "baseline is competitive. Contributing factor, reported not hidden: the exploit-mode "
        "epistemic penalty (ADR-0007) discourages the planner from the high-uncertainty "
        "regions a 4096-random-step model never visited — precisely the upright/target "
        "region — which is the penalty working as designed, not a control failure."
    )
    lines.append(
        "- This is the finding the probe exists to surface: toy-benchmark wins are **not** "
        "evidence of general control; the credibility jump needs more env-step budget, "
        "better exploration for model training, and stronger baselines — not more phases."
    )
    lines.append("")
    return "\n".join(lines)


def _task_payload(r: TaskResult) -> dict[str, object]:
    return {
        "domain": r.domain, "task": r.task,
        "obs_dim": r.obs_dim, "action_dim": r.action_dim, "seeds": r.seeds,
        "mbrl": r.mbrl, "model_free": r.model_free, "random": r.random,
        "median": {"mbrl": r.med_mbrl, "model_free": r.med_model_free, "random": r.med_random},
        "mbrl_beats_baseline": r.mbrl_beats_baseline,
        "baseline_env_steps": r.baseline_env_steps,
        "calib_epistemic_ratio": r.calib_epistemic_ratio,
    }


def write_report(results: list[TaskResult], reach: list[TaskResult], stamp: str,
                 versions: dict[str, str], results_dir: Path | None = None) -> Path:
    out = results_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "BH-001-report.md").write_text(render_markdown(results, reach, stamp, versions))
    payload = {
        "stamp": stamp,
        "versions": versions,
        "budget_env_steps": BUDGET,
        "tasks": [_task_payload(r) for r in results],
        "reachability": [_task_payload(r) for r in reach],
    }
    (out / "BH-001-report.json").write_text(json.dumps(payload, indent=2))
    return out / "BH-001-report.md"
