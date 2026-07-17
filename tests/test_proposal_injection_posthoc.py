from __future__ import annotations

from typing import cast

from bench.proposal_injection.posthoc import DEFAULT_INPUT, REPO_ROOT, analyze_steps


def test_verified_pi003_step_audit_localizes_transfer_loss() -> None:
    result = analyze_steps(REPO_ROOT / DEFAULT_INPUT)
    observations = cast(dict[str, object], result["observations"])
    rows = cast(list[dict[str, object]], result["step_rows"])

    assert result["predeclared_decision_unchanged"] is True
    assert observations["initial_reference_transfer"] is True
    assert observations["initial_reference_survives_refinement"] is False
    assert observations["step_one_top_elite_fraction"] == 0.34375
    assert observations["step_two_onward_any_top_elite"] is False
    assert len(rows) == 14
    assert all(row["calls"] == 32 for row in rows)
