"""Tests for the regression ratchet (P0-007): SHIPPED parsing and the gate-all
aggregation exit codes (all pass -> 0, any BLOCKED -> 1, malformed SHIPPED -> 2).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import bench
from bench.__main__ import main
from bench.gates import GateResult


def test_shipped_records_p0() -> None:
    assert "P0" in bench.shipped_phases()


def test_shipped_parser_skips_comments_and_blanks(tmp_path: Path) -> None:
    shipped = tmp_path / "SHIPPED"
    shipped.write_text("# comment\n\nP0\nP1\n")
    assert bench.shipped_phases(shipped) == ["P0", "P1"]


def test_shipped_unknown_phase_fails_loudly(tmp_path: Path) -> None:
    shipped = tmp_path / "SHIPPED"
    shipped.write_text("P42\n")
    with pytest.raises(ValueError, match="unknown phase"):
        bench.shipped_phases(shipped)


def test_missing_shipped_file_means_nothing_shipped(tmp_path: Path) -> None:
    assert bench.shipped_phases(tmp_path / "SHIPPED") == []


def test_gate_all_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    shipped = tmp_path / "SHIPPED"

    # Empty SHIPPED -> nothing to ratchet -> 0.
    shipped.write_text("")
    assert main(["--all"], shipped_path=shipped, results_dir=tmp_path) == 0

    # A shipped phase whose gate is BLOCKED (P8 is still PENDING) -> 1.
    shipped.write_text("P8\n")
    assert main(["--all"], shipped_path=shipped, results_dir=tmp_path) == 1
    assert "RATCHET FAILED" in capsys.readouterr().out

    # Malformed SHIPPED -> 2.
    shipped.write_text("P42\n")
    assert main(["--all"], shipped_path=shipped, results_dir=tmp_path) == 2

    # All shipped gates pass -> 0 (fake a passing P0 check; P0 has no sentinels).
    original = bench.GATES["P0"].check
    try:
        bench.GATES["P0"].check = lambda: GateResult(phase="P0", passed=True, detail="fake pass")
        shipped.write_text("P0\n")
        assert main(["--all"], shipped_path=shipped, results_dir=tmp_path) == 0
        assert "ratchet ok" in capsys.readouterr().out
    finally:
        bench.GATES["P0"].check = original
