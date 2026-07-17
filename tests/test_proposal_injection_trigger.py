from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.proposal_injection.trigger import DEFAULT_INPUT, REPO_ROOT, analyze_trigger


def test_sealed_ol002_reproduces_candidate_injection_trigger() -> None:
    result = analyze_trigger(REPO_ROOT / DEFAULT_INPUT)
    decision = result["decision"]
    comparison = result["prefix_comparison"]
    native = result["native_rung"]

    assert decision == {
        "top_elite_everywhere": True,
        "selected_reference_everywhere": True,
        "native_control_failed": True,
        "trigger_reproduced": True,
        "next_branch": "freeze_compute_matched_privileged_candidate_injection",
    }
    assert native["success_rate"] == pytest.approx(0.0625)
    assert native["fixed_bank"]["blocks"] == 32
    assert comparison["closed_loop_prefers_prefix_8"] is True
    assert comparison["common_bank_prefers_exact_prefix_12"] is True
    assert comparison["directions_disagree"] is True


def test_trigger_rejects_wrong_parent_identity(tmp_path: Path) -> None:
    source = json.loads((REPO_ROOT / DEFAULT_INPUT).read_text(encoding="utf-8"))
    source["experiment_id"] = "not-OL-002"
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed OL-002"):
        analyze_trigger(path)
