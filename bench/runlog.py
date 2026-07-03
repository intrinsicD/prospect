"""Run-metrics artifact (P0-005): the contract between training loops and the
ADR-0006 sentinels.

Training writes one JSONL record per logged step to
`bench/runs/<run-id>/metrics.jsonl`; a zero-argument sentinel `check()` reads the
run back to verify its integrity criterion *throughout training*, not only at the
capability checkpoint. Metric keys come from what `Learner.update()` returns plus
held-out probes; each sentinel's criterion names the keys it requires.

Stdlib-only. The run-id is supplied by the caller (the harness), so a gate run can
name the run it evaluates; `latest_run()` is the default for `make gate`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass(frozen=True)
class Record:
    """One logged training step."""
    step: int
    metrics: dict[str, float]


class RunLog:
    """Append-only JSONL writer for one training run."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.path = (root or RUNS_DIR) / run_id / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        record = {"step": int(step), "metrics": {k: float(v) for k, v in metrics.items()}}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_run(run_id: str, root: Path | None = None) -> list[Record]:
    """All records of a run, in the order they were logged."""
    path = (root or RUNS_DIR) / run_id / "metrics.jsonl"
    records: list[Record] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            metrics = {k: float(v) for k, v in payload["metrics"].items()}
            records.append(Record(step=int(payload["step"]), metrics=metrics))
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"{path}:{lineno}: malformed run-log record: {line!r}") from err
    return records


def latest_run(root: Path | None = None) -> str:
    """The id of the most recently written run — the default a gate run evaluates."""
    base = root or RUNS_DIR
    candidates = (
        [p for p in base.iterdir() if (p / "metrics.jsonl").exists()] if base.exists() else []
    )
    if not candidates:
        raise FileNotFoundError(f"no runs found under {base}")
    return max(candidates, key=lambda p: (p / "metrics.jsonl").stat().st_mtime).name
