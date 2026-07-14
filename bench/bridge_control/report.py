"""Deterministic Markdown and SVG reporting for BC-001."""
from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, cast

import numpy as np


def _rows_by_arm(results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cast(list[dict[str, Any]], results["rows"]):
        grouped[str(row["arm"])].append(row)
    return grouped


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_diagnostic(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(cast(dict[str, float], row["diagnostics"])[key]) for row in rows]
    return float(np.mean(values))


def _control_return_svg(path: Path, grouped: dict[str, list[dict[str, Any]]]) -> None:
    arms = [
        ("exact_dynamics_exact_reward", "Exact", "#2f855a"),
        ("b1_r1_d8", "Balanced learned", "#2b6cb0"),
        (
            "exact_transition_learned_reward_zero_epistemic",
            "Exact transition + reward, epi=0",
            "#805ad5",
        ),
        ("random_policy", "Random", "#718096"),
    ]
    values = {
        name: [float(row["mean_eval_return"]) for row in grouped.get(name, [])]
        for name, _, _ in arms
    }
    all_values = [value for rows in values.values() for value in rows]
    low = min(all_values + [-1.0])
    high = max(all_values + [1.0])
    width, height = 920, 440
    left, top, plot_w, plot_h = 70, 35, 810, 320

    def y(value: float) -> float:
        return top + (high - value) / max(high - low, 1e-9) * plot_h

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="22" font-family="sans-serif" font-size="16">'
        "BC-001 control returns by learner/planner block</text>",
    ]
    for tick in np.linspace(low, high, 5):
        tick_y = y(float(tick))
        chunks.extend(
            [
                f'<line x1="{left}" y1="{tick_y:.1f}" x2="{left + plot_w}" '
                f'y2="{tick_y:.1f}" stroke="#edf2f7" stroke-width="1"/>',
                f'<text x="{left - 8}" y="{tick_y + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="10">{float(tick):.1f}</text>',
            ]
        )
    zero_y = y(0.0)
    chunks.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" '
        'stroke="#a0aec0" stroke-width="1"/>'
    )
    group_width = plot_w / len(arms)
    for arm_index, (name, label, color) in enumerate(arms):
        rows = values[name]
        center = left + group_width * (arm_index + 0.5)
        for seed, value in enumerate(rows):
            x = center + (seed - (len(rows) - 1) / 2) * 12
            chunks.append(
                f"<g><title>seed {seed}: {value:.3f}</title>"
                f'<circle cx="{x:.1f}" cy="{y(value):.1f}" r="4" fill="{color}"/></g>'
            )
        mean = float(np.mean(rows)) if rows else 0.0
        chunks.append(
            f'<line x1="{center - 52:.1f}" y1="{y(mean):.1f}" '
            f'x2="{center + 52:.1f}" y2="{y(mean):.1f}" stroke="{color}" stroke-width="4"/>'
        )
        chunks.append(
            f'<text x="{center:.1f}" y="390" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{escape(label)}</text>'
        )
    chunks.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _final_state_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 640, 480
    left, top, plot_w, plot_h = 70, 45, 520, 360

    def x_coord(x: float) -> float:
        return left + (x + 1.0) / 2.0 * plot_w

    def y_coord(y: float) -> float:
        return top + (0.8 - y) / 1.6 * plot_h

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="25" font-family="sans-serif" font-size="16">'
        "Balanced learned control: final-state geography</text>",
        f'<rect x="{x_coord(0.75):.1f}" y="{y_coord(0.45):.1f}" '
        f'width="{x_coord(1.0) - x_coord(0.75):.1f}" '
        f'height="{y_coord(0.05) - y_coord(0.45):.1f}" fill="#c6f6d5" opacity="0.7"/>',
        f'<line x1="{left}" y1="{y_coord(0):.1f}" x2="{left + plot_w}" '
        f'y2="{y_coord(0):.1f}" stroke="#cbd5e0"/>',
        f'<line x1="{x_coord(0):.1f}" y1="{top}" x2="{x_coord(0):.1f}" '
        f'y2="{top + plot_h}" stroke="#cbd5e0"/>',
    ]
    for row in rows:
        seed = int(row["seed"])
        for start_index, final in enumerate(cast(list[list[float]], row["final_states"])):
            chunks.append(
                f"<g><title>seed {seed}, start {start_index}: "
                f"x={float(final[0]):.3f}, y={float(final[1]):.3f}</title>"
                f'<circle cx="{x_coord(float(final[0])):.1f}" '
                f'cy="{y_coord(float(final[1])):.1f}" r="4" fill="#2b6cb0" '
                'opacity="0.75"/></g>'
            )
    chunks.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="455" text-anchor="middle" '
            'font-family="sans-serif" font-size="12">x</text>',
            '<text x="18" y="225" text-anchor="middle" font-family="sans-serif" '
            'font-size="12" transform="rotate(-90 18 225)">y</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _topology_svg(path: Path) -> None:
    width, height = 760, 260
    centers = [(70 + index * 88, 95) for index in range(8)]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" '
        'orient="auto"><path d="M0,0 L0,7 L7,3.5 z" fill="#c53030"/></marker></defs>',
        '<text x="30" y="25" font-family="sans-serif" font-size="16">'
        "BridgeControl causal layout</text>",
    ]
    for index in range(7):
        x1, y1 = centers[index]
        x2, y2 = centers[index + 1]
        color = "#c53030" if index == 3 else "#718096"
        width_px = 5 if index == 3 else 2
        marker = ' marker-end="url(#arrow)"' if index == 3 else ""
        chunks.append(
            f'<line x1="{x1 + 18}" y1="{y1}" x2="{x2 - 18}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width_px}"'
            f"{marker}/>"
        )
    for index, (x, y) in enumerate(centers):
        fill = "#bee3f8" if index in (5, 6) else "#edf2f7"
        chunks.append(f'<circle cx="{x}" cy="{y}" r="18" fill="{fill}" stroke="#4a5568"/>')
        chunks.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{index}</text>'
        )
    chunks.extend(
        [
            '<text x="340" y="58" text-anchor="middle" font-family="sans-serif" '
            'font-size="12" fill="#c53030">B: observed directed door</text>',
            '<text x="555" y="145" text-anchor="middle" font-family="sans-serif" '
            'font-size="12" fill="#2b6cb0">R,D: disjoint stabilization strip</text>',
            '<text x="380" y="215" text-anchor="middle" font-family="sans-serif" '
            'font-size="12">Off-lane decoy nodes exist at every x region</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _report(results: dict[str, Any], manifest: dict[str, Any]) -> str:
    grouped = _rows_by_arm(results)
    decision = cast(dict[str, Any], results["control_decision"])
    exact = cast(dict[str, float], decision["exact"])
    random = cast(dict[str, float], decision["random"])
    balanced = cast(dict[str, float], decision["balanced_learned"])
    hybrid_rows = grouped["exact_transition_learned_reward_zero_epistemic"]
    hybrid_return = _mean(hybrid_rows, "mean_eval_return")
    hybrid_success = _mean(hybrid_rows, "success_rate")
    balanced_rows = grouped["b1_r1_d8"]

    stopped = not bool(decision["passed"])
    balanced_pass = bool(
        balanced["mean_success_rate"] >= 0.80
        and balanced["mean_return"] > random["mean_return"]
    )
    if stopped:
        outcome = [
            "The factorial was stopped before causal contrasts. Exact dynamics with the unchanged",
            f"planner achieved {100 * exact['mean_success_rate']:.1f}% success, but the designated",
            f"balanced learned arm achieved only {100 * balanced['mean_success_rate']:.1f}% against",
            "the frozen 80% positive-control floor after the one permitted fixture redesign.",
            "Estimating bridge, rank, or density effects would therefore use a BC-001 configuration",
            "that fails its own learned positive control.",
            "",
            "This is an informative negative result about the assay, not evidence against every",
            "transition-support hypothesis. It rejects BC-001 as a valid causal test in its current",
            "form and triggers the simulator-oracle/model-error branch. Under the parent portfolio's",
            "predeclared abandonment rule, T1 is retired as the active next mechanism program rather",
            "than rescued with post-hoc changes.",
        ]
    else:
        outcome = [
            "Both frozen positive controls passed and all eight factorial cells were evaluated.",
            "The estimates below remain fixture-specific until the bridge contrast replicates on a",
            "second bottleneck length; completion alone does not establish a general coverage law.",
        ]
    localization = (
        "Replacing learned latent transition/uncertainty rollouts with exact raw-state transitions "
        f"and zero epistemic changed success from {100 * balanced['mean_success_rate']:.1f}% to "
        f"{100 * hybrid_success:.1f}%. This diagnostic rescue shows that the learned reward head is "
        "not the sole blocker, but it does not separate transition, representation, and epistemic "
        "effects."
    )

    lines = [
        "# BC-001 — BridgeControl causal coverage experiment",
        "",
        f"**Status:** `{results['status']}` (non-gated research evidence)",
        "",
        "## Outcome",
        "",
        *outcome,
        "",
        localization,
        "",
        "## Sequential control decision",
        "",
        "| Control | Mean return | Success rate | Criterion | Verdict |",
        "|---|---:|---:|---|---|",
        f"| Exact dynamics + exact reward | {exact['mean_return']:.3f} | "
        f"{exact['mean_success_rate']:.3f} | success ≥ 0.95 | "
        f"{'PASS' if exact['mean_success_rate'] >= 0.95 else 'FAIL'} |",
        f"| Fully balanced learned B1/Rfull/D8 | {balanced['mean_return']:.3f} | "
        f"{balanced['mean_success_rate']:.3f} | success ≥ 0.80 and beats random | "
        f"{'PASS' if balanced_pass else 'FAIL'} |",
        f"| Random policy | {random['mean_return']:.3f} | {random['mean_success_rate']:.3f} | "
        "named floor | reference |",
        f"| Exact transition + learned reward + zero epistemic | {hybrid_return:.3f} | "
        f"{hybrid_success:.3f} | localization-only; interpreted under stopped branch | diagnostic |",
        "",
        "### Fresh block rows",
        "",
        "| Seed | Exact return | Exact success | Balanced return | Balanced success | "
        "Exact-transition/learned-reward/zero-epistemic return | Hybrid success |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    exact_rows = {int(row["seed"]): row for row in grouped["exact_dynamics_exact_reward"]}
    hybrid_by_seed = {int(row["seed"]): row for row in hybrid_rows}
    for row in balanced_rows:
        seed = int(row["seed"])
        exact_row = exact_rows[seed]
        hybrid_row = hybrid_by_seed[seed]
        lines.append(
            f"| {seed} | {float(exact_row['mean_eval_return']):.3f} | "
            f"{float(exact_row['success_rate']):.3f} | {float(row['mean_eval_return']):.3f} | "
            f"{float(row['success_rate']):.3f} | {float(hybrid_row['mean_eval_return']):.3f} | "
            f"{float(hybrid_row['success_rate']):.3f} |"
        )

    lines.extend(
        [
            "",
            "![Control returns](plots/control-returns.svg)",
            "",
            "![Balanced final states](plots/balanced-final-states.svg)",
            "",
            "## Manipulation validity",
            "",
            "All manipulation checks passed before the control screen:",
            "",
            "| Cell | Rows | Nodes | Bridge edges | Local σmin | Unique support/cell |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    checks = cast(dict[str, dict[str, Any]], results["manipulation_checks"])
    for name in sorted(checks):
        check = checks[name]
        lines.append(
            f"| {name} | {int(check['rows'])} | {int(check['node_coverage'])} | "
            f"{int(check['bridge_edge_count'])} | "
            f"{float(check['local_action_min_singular']):.3f} | "
            f"{int(check['controllable_unique_per_cell_min'])} |"
        )
    lines.extend(
        [
            "",
            "Every primary cell has identical x-region × door/off-lane counts, nuisance multisets,",
            "and coordinate-wise action histograms. For a fixed bridge setting, bridge-source",
            "state/action/next-state/reward rows are byte-identical across rank and density arms.",
            "",
            "![Causal layout](plots/topology.svg)",
            "",
            "## Learned-model localization diagnostics",
            "",
            "These are averaged over the eight balanced learned models. Target-latent error is",
            "secondary because each model owns a different learned target encoder.",
            "",
            "| Diagnostic | Mean |",
            "|---|---:|",
            f"| Reward RMSE | {_mean_diagnostic(balanced_rows, 'reward_rmse'):.5f} |",
            f"| Target-latent MSE / persistence | "
            f"{_mean_diagnostic(balanced_rows, 'target_latent_mse_over_persistence'):.5f} |",
            f"| Prototype raw next-state MSE | "
            f"{_mean_diagnostic(balanced_rows, 'prototype_raw_next_mse'):.5f} |",
            f"| Next-region accuracy | {_mean_diagnostic(balanced_rows, 'next_region_accuracy'):.5f} |",
            f"| Door/off-lane accuracy | {_mean_diagnostic(balanced_rows, 'next_lane_accuracy'):.5f} |",
            f"| Candidate-rank Spearman | "
            f"{_mean_diagnostic(balanced_rows, 'candidate_rank_spearman'):.5f} |",
            f"| Candidate action regret | "
            f"{_mean_diagnostic(balanced_rows, 'candidate_action_regret'):.5f} |",
            f"| Mean epistemic | {_mean_diagnostic(balanced_rows, 'mean_epistemic'):.5f} |",
            "",
            "The candidate-ranking metrics use one fixed 60-sequence, horizon-five bank at two",
            "diagnostic starts; zero bank regret is not a global planning guarantee.",
            "",
            "Final-state geography shows that some controllers ended left of the door, while others",
            "ended at high x outside the y success band. Endpoints alone do not identify the cause.",
            "Because the designated balanced arm contains the bridge, full local action rank, and",
            "eight unique microstates per stabilization cell yet misses its control floor, BC-001",
            "cannot support its intended causal contrasts.",
        ]
    )
    if stopped:
        lines.extend(["", "## Not run by design", ""])
        for item in cast(list[str], results["not_run"]):
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "No bridge main effect, bridge×rank interaction, bootstrap interval, or novelty",
                "upgrade is reported. The action-permutation and nuisance-only datasets were",
                "prepared and hashed but not trained because the learned positive control failed.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Factorial effects",
                "",
                "| Effect | Mean return contrast | Per-seed contrasts |",
                "|---|---:|---|",
            ]
        )
        effects = cast(dict[str, dict[str, Any]], results["factorial_effects"])
        for name in ("bridge", "rank", "density", "bridge_x_rank"):
            effect = effects[name]
            values = ", ".join(f"{float(value):.3f}" for value in effect["per_seed"])
            lines.append(f"| {name} | {float(effect['mean']):.3f} | {values} |")
        lines.extend(
            [
                "",
                "These are first-topology estimates. They remain exploratory until a second",
                "corridor length reproduces the predeclared bridge signature.",
                "",
                "## Not run by design",
                "",
            ]
        )
        for item in cast(list[str], results["not_run"]):
            lines.append(f"- {item}")

    repository = cast(dict[str, Any], results["repository"])
    protocol = cast(dict[str, Any], results["protocol"])
    source_hashes = cast(dict[str, str], repository["source_sha256"])
    lines.extend(
        [
            "",
            "## Reproduction and provenance",
            "",
            f"- Canonical protocol-record SHA-256: `{results['protocol_sha256']}`",
            f"- Protocol-document SHA-256: `{protocol['protocol_document_sha256']}`",
            f"- Dataset-manifest file SHA-256: `{results['manifest_sha256']}`",
            f"- Dataset count: {len(cast(dict[str, Any], manifest['datasets']))}",
            f"- Git HEAD at execution: `{repository['head']}`; dirty worktree: "
            f"`{str(bool(repository['dirty'])).lower()}`",
            f"- Python: `{cast(dict[str, Any], results['versions'])['python']}`; "
            f"NumPy: `{cast(dict[str, Any], results['versions'])['numpy']}`",
            "- Development learner/planner seed 97 was excluded; formal blocks are 0–7.",
            "- Full machine-readable rows are in `BC-001-results.json` and `BC-001-runs.csv`.",
            "- `artifact-manifest.json` binds the protocol, manifest, result, CSV, report, and plots.",
            "",
            "### Frozen source snapshot",
            "",
        ]
    )
    for path, digest in sorted(source_hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.extend(
        [
            "",
            "Run `python -m bench.bridge_control verify` to recheck current source/protocol binding,",
            "deterministic datasets, raw aggregates, stop-rule semantics, and rendered artifacts.",
            "This fixture is non-gated and changes no shipped claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report_artifacts(
    output: Path,
    results: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    grouped = _rows_by_arm(results)
    plots = output / "plots"
    _control_return_svg(plots / "control-returns.svg", grouped)
    _final_state_svg(plots / "balanced-final-states.svg", grouped["b1_r1_d8"])
    _topology_svg(plots / "topology.svg")
    (output / "BC-001-report.md").write_text(_report(results, manifest), encoding="utf-8")
