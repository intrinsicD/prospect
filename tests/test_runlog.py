"""Tests for the run-metrics artifact (P0-005) — the data sentinels read."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.runlog import RunLog, latest_run, read_run


def test_round_trip_preserves_order_and_values(tmp_path: Path) -> None:
    log = RunLog("run-a", root=tmp_path)
    log.log(0, {"loss": 1.5, "latent_std_min": 0.9})
    log.log(1, {"loss": 1.25})
    records = read_run("run-a", root=tmp_path)
    assert [r.step for r in records] == [0, 1]
    assert records[0].metrics == {"loss": 1.5, "latent_std_min": 0.9}
    assert records[1].metrics == {"loss": 1.25}


def test_latest_run_picks_the_newest(tmp_path: Path) -> None:
    RunLog("run-old", root=tmp_path).log(0, {"loss": 1.0})
    RunLog("run-new", root=tmp_path).log(0, {"loss": 2.0})
    os.utime(tmp_path / "run-old" / "metrics.jsonl", (1_000, 1_000))
    os.utime(tmp_path / "run-new" / "metrics.jsonl", (2_000, 2_000))
    assert latest_run(root=tmp_path) == "run-new"


def test_latest_run_with_no_runs_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        latest_run(root=tmp_path)


def test_malformed_line_raises_with_location(tmp_path: Path) -> None:
    log = RunLog("run-bad", root=tmp_path)
    log.log(0, {"loss": 1.0})
    with log.path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    with pytest.raises(ValueError, match="malformed run-log record"):
        read_run("run-bad", root=tmp_path)
