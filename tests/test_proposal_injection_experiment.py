from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from bench.proposal_injection import experiment


def _row(arm: str, seed: int, value: float, success: float) -> dict[str, object]:
    diagnostics: list[dict[str, object]] = []
    if arm == "privileged_injection":
        diagnostics = [
            {
                "injected_top_elite_count": 8,
                "first_round_best_injected": True,
                "best_sequence_injected": True,
                "episode_success": bool(success),
            }
        ]
    return {
        "arm": arm,
        "seed": seed,
        "mean_eval_return": value,
        "success_rate": success,
        "plan_diagnostics": diagnostics,
    }


def _primary_rows(privileged: float, permuted: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in experiment.MODEL_SEEDS:
        rows.extend(
            [
                _row("native_no_penalty", seed, 0.0, 0.0),
                _row("privileged_injection", seed, privileged, 1.0 if privileged > 5.0 else 0.0),
                _row("action_permuted_injection", seed, permuted, 1.0 if permuted > 5.0 else 0.0),
                _row("exact_raw", seed, 10.0, 1.0),
            ]
        )
    return rows


def test_decision_selects_enlarged_search_only_for_specific_rescue() -> None:
    decision = experiment._decision(_primary_rows(privileged=8.0, permuted=1.0))
    assert decision["primary_branch"] == "specific_privileged_rescue"
    assert decision["expected_conditional_arm"] == "enlarged_native_search"
    assert str(decision["classification"]).startswith("pending_conditional")


def test_decision_selects_time_control_for_non_specific_rescue() -> None:
    decision = experiment._decision(_primary_rows(privileged=8.0, permuted=7.0))
    assert decision["primary_branch"] == "non_specific_injection_rescue"
    assert decision["expected_conditional_arm"] == "time_permuted_injection"


def test_no_rescue_runs_commitment_audit_instead_of_more_search() -> None:
    decision = experiment._decision(_primary_rows(privileged=1.0, permuted=0.5))
    assert decision["primary_branch"] == "no_privileged_rescue"
    assert decision["expected_conditional_arm"] is None
    audit = cast(dict[str, object], decision["commitment_audit"])
    assert audit["classification"] == "open_loop_closed_loop_mismatch"


def test_prepare_freezes_inputs_without_formal_outcomes(tmp_path: Path) -> None:
    output = tmp_path / "PI-001"
    prepared = experiment.prepare(output)
    assert prepared["status"] == "prepared_only"
    assert prepared["outcomes"] == "prepared_only"
    assert (output / "protocol.json").exists()
    assert (output / "input-manifest.json").exists()
    assert (output / experiment.INPUT_COPY).exists()
    with pytest.raises(ValueError, match="results are required"):
        experiment.verify(output, require_results=True)
