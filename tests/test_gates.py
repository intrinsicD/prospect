"""Tests for the gate wiring (P0-006): registration decorators, persisted report
JSON, friendly CLI errors, and the registered P0 gate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import bench
from bench.__main__ import main
from bench.gates import GateResult, gate_check, sentinel_check


def test_gate_check_replaces_pending_and_report_persists(tmp_path: Path) -> None:
    original = bench.GATES["P8"].check
    try:
        @gate_check("P8")
        def _fake_p8() -> GateResult:
            return GateResult(
                phase="P8", passed=True, metrics={"accuracy_gain": 0.12}, seeds=[0, 1, 2]
            )

        report = bench.run_gate("P8", run_id="run-x", results_dir=tmp_path)
        assert report.capability.passed
        assert "PENDING" not in report.capability.detail
        [path] = tmp_path.glob("P8-*.json")
        payload = json.loads(path.read_text())
        assert payload["phase"] == "P8"
        assert payload["run_id"] == "run-x"
        assert payload["capability"]["metrics"] == {"accuracy_gain": 0.12}
        assert payload["capability"]["seeds"] == [0, 1, 2]
        assert payload["capability"]["criterion"] == bench.GATES["P8"].criterion
        assert {s["name"] for s in payload["sentinels"]} == {s.name for s in report.sentinels}
    finally:
        bench.GATES["P8"].check = original


def test_registering_unknown_phase_or_sentinel_fails() -> None:
    with pytest.raises(KeyError, match="unknown phase"):
        gate_check("P99")  # a phase that does not exist (P9 is now real)
    with pytest.raises(KeyError, match="unknown sentinel"):
        sentinel_check("nonexistent-sentinel")


def test_run_gate_unknown_phase_lists_known(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="known phases"):
        bench.run_gate("P99", results_dir=tmp_path)


def test_cli_is_friendly_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()
    assert main(["P99"]) == 2
    out = capsys.readouterr().out
    assert "P99" in out and "P1" in out


@pytest.mark.skipif(
    os.environ.get("PROSPECT_IN_P0_GATE") == "1",
    reason="already inside the P0 gate's pytest run",
)
def test_p0_gate_passes_against_real_suite(tmp_path: Path) -> None:
    report = bench.run_gate("P0", results_dir=tmp_path)
    assert report.capability.passed, report.capability.detail
    assert report.sentinels == []  # no integrity sentinels apply before P1
    assert report.passed
