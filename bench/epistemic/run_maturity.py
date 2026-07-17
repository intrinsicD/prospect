"""Command-line entry point for the audited E2--E5 reference report."""

from __future__ import annotations

import argparse

from bench.epistemic.maturity import run_maturity_benchmark


def main(argv: list[str] | None = None) -> int:
    """Print canonical JSON.

    The default status is capability-gate compatible and therefore fails while
    any row is reference-only or blocked.  ``--diagnostics`` instead verifies
    only that the exact numeric predicates execute and pass; it never changes the
    report's ``passed: false`` capability disposition.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="exit on numeric reference-check status, not capability support",
    )
    arguments = parser.parse_args(argv)
    report = run_maturity_benchmark().report
    print(report.to_json())
    if arguments.diagnostics:
        return 0 if all(gate.diagnostic_checks_passed for gate in report.gates) else 1
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
