"""Exploratory step-stratified audit of the verified PI-003 diagnostics.

This analysis is explicitly post-hoc.  It cannot change PI-003's preregistered
decision; it only resolves where the aggregate statewise-transfer failure appears and
selects a cheaper next experiment.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("bench/proposal_injection_v3/results/PI-003/PI-003-results.json")
DEFAULT_OUTPUT = Path("bench/proposal_injection_v3/results/PI-003-posthoc-step-audit.json")


def _read_json(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is forbidden: {value}")
            ),
        ),
    )


def _fraction(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def analyze_steps(path: Path = REPO_ROOT / DEFAULT_INPUT) -> dict[str, object]:
    results = _read_json(path)
    if results.get("experiment_id") != "PI-003":
        raise ValueError("post-hoc audit requires PI-003")
    if results.get("status") != "completed_proposal_injection":
        raise ValueError("post-hoc audit requires completed PI-003 results")
    decision = cast(dict[str, Any], results["decision"])
    if decision.get("primary_branch") != "no_privileged_rescue":
        raise ValueError("step audit is defined only for PI-003's no-rescue branch")

    rows = [
        row
        for row in cast(list[dict[str, Any]], results["rows"])
        if row["arm"] == "privileged_injection"
    ]
    if len(rows) != 8:
        raise ValueError("PI-003 must contain eight privileged-injection seed rows")
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for call in cast(list[dict[str, Any]], row["plan_diagnostics"]):
            by_step[int(call["step"])].append(call)
    if set(by_step) != set(range(14)) or any(len(calls) != 32 for calls in by_step.values()):
        raise ValueError("PI-003 diagnostics must contain 32 calls at each of 14 episode steps")

    step_rows: list[dict[str, object]] = []
    for step in sorted(by_step):
        calls = by_step[step]
        step_rows.append(
            {
                "step": step,
                "calls": len(calls),
                "any_injected_top_elite_fraction": _fraction(
                    [int(call["injected_top_elite_count"]) > 0 for call in calls]
                ),
                "first_round_best_injected_fraction": _fraction(
                    [bool(call["first_round_best_injected"]) for call in calls]
                ),
                "final_best_injected_fraction": _fraction(
                    [bool(call["best_sequence_injected"]) for call in calls]
                ),
            }
        )

    initial = step_rows[0]
    step_one = step_rows[1]
    later = step_rows[2:]
    return {
        "schema_version": "proposal-injection-posthoc-step-v1",
        "status": "exploratory_posthoc",
        "source": {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "experiment_id": "PI-003",
            "semantic_verification": "verified_semantic_results",
        },
        "predeclared_decision_unchanged": True,
        "step_rows": step_rows,
        "observations": {
            "initial_reference_transfer": (
                initial["any_injected_top_elite_fraction"] == 1.0
                and initial["first_round_best_injected_fraction"] == 1.0
            ),
            "initial_reference_survives_refinement": (
                float(cast(float, initial["final_best_injected_fraction"])) > 0.0
            ),
            "step_one_top_elite_fraction": step_one[
                "any_injected_top_elite_fraction"
            ],
            "step_two_onward_any_top_elite": any(
                float(cast(float, row["any_injected_top_elite_fraction"])) > 0.0
                for row in later
            ),
        },
        "interpretation": (
            "Exploratory: the fixed-start trigger transfers perfectly into PI-003's "
            "first iCEM round, but learned-score refinement replaces every injected "
            "reference; after one real action the ranking transfer weakens and from "
            "step 2 onward it disappears. This points to optimizer/model-exploitation "
            "and visited-state shift, not simple proposal scarcity."
        ),
        "recommended_next_experiment": {
            "name": "iteration-wise learned-versus-exact candidate-landscape audit",
            "minimal_test": (
                "At frozen initial, step-1, and step-2 visited states, retain every "
                "candidate pool from each iCEM iteration and score the same sequences "
                "with both the learned scorer and exact simulator."
            ),
            "null": (
                "Learned-score refinement does not systematically reduce exact score "
                "relative to the injected first-round reference."
            ),
            "killing_signature": (
                "Learned score rises while exact score or exact rank falls in at least "
                "7/8 model seeds, with the effect present at step 0 before state shift."
            ),
            "abandonment": (
                "If learned and exact refinement agree at step 0, abandon within-call "
                "model exploitation and isolate the step-1/step-2 state shift instead."
            ),
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-stratify verified PI-003 diagnostics")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze_steps(args.input)
    _write_json(args.output, result)
    print("posthoc: exploratory_step_audit_complete")


if __name__ == "__main__":
    main()
