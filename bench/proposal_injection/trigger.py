"""Recompute the predeclared proposal-injection trigger from sealed OL-002 data.

This module is deliberately read-only with respect to OL-002.  It turns the raw
machine result into a compact, deterministic iteration-1 record; it does not train a
model, rerun an outcome, or strengthen the scientific evidence by repetition.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("bench/oracle_ladder_v2/results/OL-002/OL-002-results.json")
DEFAULT_OUTPUT = Path("bench/proposal_injection/results/PI-001-trigger.json")
NATIVE_RUNG = "learned_tsinf_no_penalty"
PREFIX_RUNG = "prefix_8_target_no_penalty"
EXACT_PREFIX_RUNG = "exact_target_learned_reward"
EXPECTED_BLOCKS = 32


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return cast(
        dict[str, Any],
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant),
    )


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _audit_summary(rows: list[dict[str, Any]], rung: str) -> dict[str, float | int]:
    selected = [row for row in rows if row.get("rung") == rung]
    if len(selected) != EXPECTED_BLOCKS:
        raise ValueError(f"{rung} must contain {EXPECTED_BLOCKS} seed/start audit blocks")
    diagnostics = [cast(dict[str, Any], row["diagnostics"]) for row in selected]
    return {
        "blocks": len(selected),
        "all_reference_sequences_in_top_elite_blocks": sum(
            float(item["reference_top_k_fraction"]) == 1.0 for item in diagnostics
        ),
        "selected_exact_reference_blocks": sum(
            item["selected_source"] == "exact_reference" for item in diagnostics
        ),
        "mean_pearson": fmean(float(item["pearson"]) for item in diagnostics),
        "mean_spearman": fmean(float(item["spearman"]) for item in diagnostics),
        "mean_normalized_selected_regret": fmean(
            float(item["normalized_selected_regret"]) for item in diagnostics
        ),
        "median_normalized_selected_regret": median(
            float(item["normalized_selected_regret"]) for item in diagnostics
        ),
        "median_exact_selected_rank": median(
            float(item["exact_selected_rank"]) for item in diagnostics
        ),
    }


def analyze_trigger(results_path: Path = REPO_ROOT / DEFAULT_INPUT) -> dict[str, Any]:
    """Return the deterministic PI-001 decision trigger from an OL-002 result."""

    results = _read_json(results_path)
    if results.get("experiment_id") != "OL-002":
        raise ValueError("trigger input must be the sealed OL-002 result")
    if results.get("schema_version") != "oracle-ladder-v2":
        raise ValueError("trigger input has an unexpected OL-002 schema")
    if results.get("status") != "completed_localization":
        raise ValueError("OL-002 must have completed localization")

    decision = cast(dict[str, Any], results["decision"])
    aggregates = cast(dict[str, dict[str, float]], decision["rung_aggregates"])
    audits = cast(list[dict[str, Any]], results["audit_rows"])
    native = aggregates[NATIVE_RUNG]
    prefix = aggregates[PREFIX_RUNG]
    exact_prefix = aggregates[EXACT_PREFIX_RUNG]
    native_audit = _audit_summary(audits, NATIVE_RUNG)
    prefix_audit = _audit_summary(audits, PREFIX_RUNG)
    exact_prefix_audit = _audit_summary(audits, EXACT_PREFIX_RUNG)

    top_elite_everywhere = (
        native_audit["all_reference_sequences_in_top_elite_blocks"] == EXPECTED_BLOCKS
    )
    selected_reference_everywhere = (
        native_audit["selected_exact_reference_blocks"] == EXPECTED_BLOCKS
    )
    native_control_failed = float(native["success_rate"]) < 0.80
    trigger_reproduced = bool(
        top_elite_everywhere and selected_reference_everywhere and native_control_failed
    )

    closed_loop_prefers_prefix_8 = float(prefix["success_rate"]) > float(
        exact_prefix["success_rate"]
    )
    common_bank_prefers_exact_prefix = bool(
        float(exact_prefix_audit["mean_pearson"])
        > float(prefix_audit["mean_pearson"])
        and float(exact_prefix_audit["mean_spearman"])
        > float(prefix_audit["mean_spearman"])
        and float(exact_prefix_audit["mean_normalized_selected_regret"])
        < float(prefix_audit["mean_normalized_selected_regret"])
    )

    return {
        "schema_version": "proposal-injection-trigger-v1",
        "iteration": 1,
        "source": {
            "path": str(results_path.relative_to(REPO_ROOT)),
            "sha256": _file_hash(results_path),
            "experiment_id": results["experiment_id"],
            "scientific_independence": (
                "OL-002 is the administrative rerun of OL-001 and is treated as one experiment"
            ),
        },
        "native_rung": {
            "name": NATIVE_RUNG,
            "mean_return": float(native["mean_return"]),
            "success_rate": float(native["success_rate"]),
            "fixed_bank": native_audit,
        },
        "prefix_comparison": {
            "prefix_8": {
                "mean_return": float(prefix["mean_return"]),
                "success_rate": float(prefix["success_rate"]),
                "fixed_bank": prefix_audit,
            },
            "exact_prefix_12": {
                "mean_return": float(exact_prefix["mean_return"]),
                "success_rate": float(exact_prefix["success_rate"]),
                "fixed_bank": exact_prefix_audit,
            },
            "prefix_8_minus_exact_prefix_12": {
                "mean_return": float(prefix["mean_return"])
                - float(exact_prefix["mean_return"]),
                "success_rate": float(prefix["success_rate"])
                - float(exact_prefix["success_rate"]),
                "mean_pearson": float(prefix_audit["mean_pearson"])
                - float(exact_prefix_audit["mean_pearson"]),
                "mean_spearman": float(prefix_audit["mean_spearman"])
                - float(exact_prefix_audit["mean_spearman"]),
                "mean_normalized_selected_regret": float(
                    prefix_audit["mean_normalized_selected_regret"]
                )
                - float(exact_prefix_audit["mean_normalized_selected_regret"]),
            },
            "closed_loop_prefers_prefix_8": closed_loop_prefers_prefix_8,
            "common_bank_prefers_exact_prefix_12": common_bank_prefers_exact_prefix,
            "directions_disagree": bool(
                closed_loop_prefers_prefix_8 and common_bank_prefers_exact_prefix
            ),
        },
        "decision": {
            "top_elite_everywhere": top_elite_everywhere,
            "selected_reference_everywhere": selected_reference_everywhere,
            "native_control_failed": native_control_failed,
            "trigger_reproduced": trigger_reproduced,
            "next_branch": (
                "freeze_compute_matched_privileged_candidate_injection"
                if trigger_reproduced
                else "stop_and_explain_trigger_failure"
            ),
        },
        "limitations": [
            "This is a deterministic re-analysis of one sealed authored-fixture experiment, not a replication.",
            "The fixed bank contains privileged exact-reference sequences and is not the native online proposal set.",
            "Four starts are repeated measures inside each of eight independently trained model seeds.",
            "The result can trigger a causal search diagnostic but cannot establish a production mechanism.",
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the PI-001 trigger from OL-002")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze_trigger(args.input)
    _write_json(args.output, result)
    print(f"trigger_reproduced: {result['decision']['trigger_reproduced']}")


if __name__ == "__main__":
    main()
