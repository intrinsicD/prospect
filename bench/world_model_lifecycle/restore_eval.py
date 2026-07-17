#!/usr/bin/env python3
"""Fresh-interpreter entry point for WM-001 restart parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .parity import evaluate_checkpoint, save_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    specification = json.loads(Path(arguments.spec).read_text(encoding="utf-8"))
    if (
        not isinstance(specification, dict)
        or set(specification) != {"schema", "task_reset_seeds", "device"}
        or specification.get("schema") != "prospect.wm001.restart-spec.v1"
    ):
        raise ValueError("unsupported restart specification")
    torch.use_deterministic_algorithms(True)
    result = evaluate_checkpoint(
        arguments.checkpoint,
        task_reset_seeds=specification["task_reset_seeds"],
        device=str(specification["device"]),
    )
    save_evaluation(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
