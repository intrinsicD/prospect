"""CLI for the non-gated BridgeControl experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import DEFAULT_OUTPUT, prepare, run, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify", "all"), nargs="?", default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run":
        result = run(args.output)
    elif args.command == "verify":
        result = verify(args.output, require_results=True)
    else:
        prepare(args.output)
        result = run(args.output)
        verify(args.output, require_results=True)
    display = result.get(
        "outcomes",
        result.get("status", result.get("experiment_id", "complete")),
    )
    print(f"{args.command}: {display}")


if __name__ == "__main__":
    main()
