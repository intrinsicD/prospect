"""Command-line entry point for result-free WM-002 Q0 qualification."""

from __future__ import annotations

from pathlib import Path

from bench.active_acquisition.qualification import PROTOCOL_PATH, run_qualification


def main(protocol_path: Path = PROTOCOL_PATH) -> int:
    """Print one canonical report and succeed exactly when Q0 passes."""

    report = run_qualification(protocol_path)
    print(report.to_json())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
