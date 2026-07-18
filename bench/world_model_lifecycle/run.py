#!/usr/bin/env python3
"""CLI for the sealed WM-001 development and formal lanes."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def _ensure_deterministic_cuda_environment() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8":
        return
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.execve(sys.executable, [sys.executable, "-m", "bench.world_model_lifecycle.run", *sys.argv[1:]], environment)


def main() -> int:
    _ensure_deterministic_cuda_environment()
    import torch

    from .artifact import (
        ProducerAttempt,
        claim_formal_launch,
        copy_file_exclusive,
        formal_launch_marker_path,
    )
    from .binding import verify_live_binding
    from .experiment import ExperimentConfig, run_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("development", "formal"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--master-seed",
        type=int,
        action="append",
        help="development-only subset of the two declared diagnostic seeds",
    )
    arguments = parser.parse_args()
    existing_launch_marker: Path | None = None
    if arguments.lane == "formal":
        if arguments.master_seed:
            parser.error("formal lane cannot override the eight sealed master seeds")
        if arguments.binding is None:
            parser.error("formal lane requires --binding")
        config = ExperimentConfig.formal(device=arguments.device)
        binding_digest = hashlib.sha256(arguments.binding.read_bytes()).hexdigest()
        output = arguments.output or (
            Path(__file__).resolve().parent
            / "results"
            / "formal"
            / binding_digest
            / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
        )
        existing_launch_marker, _ = formal_launch_marker_path(
            arguments.binding,
            output,
        )
        if existing_launch_marker.exists():
            print(
                "WM-001 protocol 1.4 formal launch already consumed; "
                "same-version retry is forbidden",
                file=sys.stderr,
            )
            return 1
    else:
        seeds = arguments.master_seed or None
        config = ExperimentConfig.development(
            master_seeds=(seeds if seeds is not None else (2439054559, 3246851043)),
            device=arguments.device,
        )
        stamp = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
        output = arguments.output or (Path(__file__).resolve().parent / "results" / "development" / stamp)
    attempt = ProducerAttempt(output, lane=arguments.lane)
    try:
        with attempt:
            binding_for_run: Path | None = None
            if arguments.lane == "formal":
                if arguments.binding is None:
                    raise RuntimeError("formal binding path disappeared after argument validation")
                binding_for_run = attempt.preserve_formal_inputs(arguments.binding)
                torch.use_deterministic_algorithms(True)
                verify_live_binding(binding_for_run, device=config.device)
                print("WM-001 live implementation binding verified", flush=True)
                launch_marker = claim_formal_launch(
                    arguments.binding,
                    attempt.output_directory,
                )
                copy_file_exclusive(
                    launch_marker,
                    attempt.output_directory / launch_marker.name,
                )
                print("WM-001 sole formal launch atomically claimed", flush=True)
            _, result_path = run_experiment(
                config,
                output_directory=attempt.output_directory,
                formal_binding_path=binding_for_run,
                output_prepared=True,
            )
            print(result_path, flush=True)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        if not (attempt.output_directory / "producer-manifest.json").is_file():
            print(f"WM-001 launch failed before attempt custody: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
