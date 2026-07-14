"""Command-line entry point for the one-shot OL-002 replication."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import DEFAULT_OUTPUT, analyze, prepare, run, verify


def main() -> None:
    parser = argparse.ArgumentParser(description="OL-002 oracle-ladder replication")
    parser.add_argument("command", choices=("prepare", "run", "verify", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare(args.output)
        print(f"prepare: {result['status']}")
    elif args.command == "run":
        result = run(args.output)
        print(f"run: {result['status']}")
    elif args.command == "analyze":
        result = analyze(args.output)
        print(f"analyze: {result['decision']['classification']}")
    else:
        result = verify(args.output, require_results=True)
        print(f"verify: {result['outcomes']}")


if __name__ == "__main__":
    main()
