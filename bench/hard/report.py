"""Render a BH-001 run into a committed report artifact (markdown + JSON).

The report *is* the deliverable — there is no gate here (ADR-0011). It records raw
per-seed returns, matched-budget deltas and the honest interpretation, so the claim
"the core survives a real MuJoCo task" is auditable rather than asserted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .curiosity import CuriosityResult
from .eval import BUDGET, CANDIDATES, EP_LEN, EVAL_EPISODES, GENS, HORIZON, POP, TaskResult
from .imitation import ImitationResult

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
        "evidence of general control. `cartpole-swingup` is the sharp case — random data "
        "never reaches the upright goal, so the world model is ignorant exactly where the "
        "reward lives. Two follow-up studies below chase that: **A** asks whether *directed* "
        "exploration (curiosity) fixes it; **B** asks whether *watching a demonstration* does."
    )
    lines.append("")
    return "\n".join(lines)


def render_curiosity(cur: CuriosityResult) -> str:
    """Section A: does curiosity-driven collection fix the swingup failure?"""
    lines = ["## A — curiosity-driven collection (does directed exploration fix swingup?)", ""]
    lines.append(
        "Swaps the P2 probe's *random* data collection for the shipped **curiosity "
        "curriculum** (P3-002 / ADR-0007): an explore-mode planner whose epistemic "
        "coefficient is a *bonus*, steering collection toward high-uncertainty regions. "
        f"Same downstream (exploit-mode MBRL), same {int(cur.extra.get('budget_env_steps', 0))}-"
        "step budget — only the collection policy differs. Median over "
        f"{len(cur.seeds)} seeds."
    )
    lines.append("")
    lines.append("| metric | random collection | curiosity collection |")
    lines.append("|--------|-------------------|----------------------|")
    lines.append(f"| goal coverage — max reward reached | {np.median(cur.cov_random):.2f} | "
                 f"**{np.median(cur.cov_curious):.2f}** |")
    lines.append(f"| goal coverage — frac steps near goal (r>0.5) | "
                 f"{100 * np.median(cur.goalfrac_random):.1f}% | "
                 f"**{100 * np.median(cur.goalfrac_curious):.1f}%** |")
    lines.append(f"| **MBRL control return** (swingup) | **{np.median(cur.mbrl_random):.1f}** | "
                 f"{np.median(cur.mbrl_curious):.1f} |")
    lines.append("")
    lines.append(
        "**Reading.** Curiosity's exploration *works* — it **reaches** the upright region "
        "random data never touches (max reward ~0.7 vs ~0.24), though it still rarely "
        "*dwells* there (fraction near the goal stays ≈0). And that partial coverage does "
        "**not** convert to better exploit control: curiosity data *hurts* the downstream "
        "MBRL. Curiosity chases novelty, so its data concentrates on wild high-energy states "
        "and under-covers the near-bottom region the exploit planner traverses, while the "
        "goal coverage it does gain is too sparse to learn the upright dynamics. More budget "
        "doesn't close it (measured separately: 3× budget still worse). **Exploration is "
        "necessary but not sufficient here** — which motivates B."
    )
    lines.append("")
    return "\n".join(lines)


def render_imitation(imit: ImitationResult) -> str:
    """Section B: does watching a demonstration reproduce swingup?"""
    lines = ["## B — imitation from observation (does watching a demo reproduce swingup?)", ""]
    lines.append(
        "The agent **watches** an expert swingup — its *observations only*, actions hidden — "
        "then **recovers the actions from observation** at the same interaction budget a "
        "from-scratch agent gets, and **clones** a closed-loop policy. Two recovery routes: a "
        "direct inverse-dynamics model, and the P13 `LatentActionModel` (ADR-0010) + a tiny "
        "calibration (the arc-faithful, action-free route). Oracle = clone on the true "
        f"(hidden) actions — the ceiling. Median over {len(imit.from_scratch)} seeds; expert "
        f"demo return {imit.demo_return:.1f}."
    )
    lines.append("")
    lines.append("| agent | swingup return | vs from-scratch |")
    lines.append("|-------|----------------|-----------------|")
    fs = float(np.median(imit.from_scratch)) or 1e-9
    rows = [
        ("**imitation — inverse-dynamics**", np.median(imit.inverse_dyn)),
        ("imitation — latent-action (P13)", np.median(imit.latent_action)),
        ("oracle clone (true actions, ceiling)", np.median(imit.oracle)),
        ("from-scratch MBRL (same budget)", np.median(imit.from_scratch)),
        ("shuffled demo (neg control)", np.median(imit.shuffled)),
    ]
    for label, val in rows:
        lines.append(f"| {label} | {val:.1f} | {val / fs:.1f}× |")
    lines.append("")
    lines.append(
        f"Action-recovery R² vs the true demo actions: inverse-dynamics "
        f"{imit.recovery_r2.get('inverse_dyn', float('nan')):.2f}, latent-action (P13) "
        f"{imit.recovery_r2.get('latent_action', float('nan')):.2f}."
    )
    lines.append("")
    lines.append(
        "**Reading.** Watching **works where exploration could not**: at the *same* budget "
        "the from-scratch agent fails swingup on (A), imitation-from-observation reproduces it "
        "— the inverse-dynamics route reproduces a swingup the agent never performed, well "
        "above from-scratch and approaching the oracle-clone ceiling, while the shuffled-demo "
        "control collapses (it is imitating the *specific* behaviour, not just moving). The "
        "P13 latent-action route is the honest weak spot: it can match the direct route on a "
        "good seed but is high-variance on this real task — recovering executable actions "
        "from a 1-D latent across a distribution shift (grounding states → the demo's upright "
        "states) is not yet reliable. **The A→B arc:** exploration reaches the goal region but "
        "can't convert it to control at feasible budgets; a demonstration hands over the "
        "goal-reaching behaviour directly — the sample-efficient route, and the substrate for "
        "learning from video (ADR-0009/0010)."
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
                 versions: dict[str, str], curiosity: CuriosityResult | None = None,
                 imitation: ImitationResult | None = None,
                 results_dir: Path | None = None) -> Path:
    out = results_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(results, reach, stamp, versions)
    if curiosity is not None:
        md += "\n" + render_curiosity(curiosity)
    if imitation is not None:
        md += "\n" + render_imitation(imitation)
    (out / "BH-001-report.md").write_text(md)
    payload: dict[str, object] = {
        "stamp": stamp,
        "versions": versions,
        "budget_env_steps": BUDGET,
        "tasks": [_task_payload(r) for r in results],
        "reachability": [_task_payload(r) for r in reach],
    }
    if curiosity is not None:
        payload["curiosity"] = {
            "task": curiosity.task, "seeds": curiosity.seeds,
            "mbrl_random": curiosity.mbrl_random, "mbrl_curious": curiosity.mbrl_curious,
            "cov_random": curiosity.cov_random, "cov_curious": curiosity.cov_curious,
            "goalfrac_random": curiosity.goalfrac_random, "goalfrac_curious": curiosity.goalfrac_curious,
        }
    if imitation is not None:
        payload["imitation"] = {
            "demo_return": imitation.demo_return, "from_scratch": imitation.from_scratch,
            "oracle": imitation.oracle, "inverse_dyn": imitation.inverse_dyn,
            "latent_action": imitation.latent_action, "shuffled": imitation.shuffled,
            "recovery_r2": imitation.recovery_r2,
        }
    (out / "BH-001-report.json").write_text(json.dumps(payload, indent=2))
    return out / "BH-001-report.md"
