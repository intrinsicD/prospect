"""Result-free WM-002 Q1 orchestration rehearsal.

The Q1 runtime spawns four producer masters, twenty-eight fresh-process restore
lanes, merges and revalidates six artifacts, publishes them atomically, and
finalizes a durable attempt marker.  None of that shape was covered end to end,
because every entry path requires ``execution_authorized: true`` and that flag
must stay false until the authorized attempt.

This module closes that gap without touching the authorization boundary.  It
prepares genuine rehearsal inputs, runs the identical orchestration under the
miniature rehearsal budget, and refuses to run at all once the protocol
authorizes Q1.  A rehearsal is mechanics coverage only:

* it requires ``execution_authorized: false``, so it is unreachable exactly
  when the production attempt is reachable;
* its lanes hold two episodes instead of 1,024, so its hidden-regime schedule
  is unbalanced and its arm means are statistically meaningless;
* its aggregate carries the rehearsal schema, so the frozen Q1 aggregate
  contract and the independent auditor reject it; and
* it writes a machine-generated prospective review that names itself as
  rehearsal-only and can never bind the production protocol digest.

Nothing produced here is a Q1 draw, outcome, interval, or capability evidence.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bench.active_acquisition.contracts import (
    Q0_REPORT_SHA256,
    Q1_PROTOCOL_PATH,
    Q1_PROTOCOL_VERSION,
    canonical_json_bytes,
    implementation_manifest,
    sha256_bytes,
)
from bench.active_acquisition.q1 import (
    Q1ExecutionError,
    Q1RunOutputs,
    _protocol_execution_mode,
    run_q1_orchestration_rehearsal,
)
from bench.active_acquisition.q1_qualification import (
    PROSPECTIVE_REVIEW_SCHEMA,
    PROSPECTIVE_REVIEW_SCOPE,
    Q1_IMPLEMENTATION_PATHS,
    run_entry_qualification,
    write_report,
)
from bench.active_acquisition.qualification import PROTOCOL_PATH as Q0_PROTOCOL_PATH
from bench.active_acquisition.qualification import run_qualification
from bench.active_acquisition.seeding import Q1ExecutionMode

REHEARSAL_REVIEWER: Final = (
    "wm002-orchestration-rehearsal-harness (machine-generated, rehearsal-only; "
    "not the independent prospective review required before Q1 authorization)"
)
_SALT_BYTES: Final = 32


@dataclass(frozen=True, slots=True)
class RehearsalInputs:
    """Exact rehearsal-only paths accepted by the Q1 entry and runtime."""

    q0_report_path: Path
    secret_salt_path: Path
    prospective_review_path: Path
    entry_report_path: Path
    execution_root: Path
    attempt_registry_directory: Path
    output_directory: Path


def require_rehearsal_protocol(protocol_path: Path = Q1_PROTOCOL_PATH) -> str:
    """Return the protocol digest only while the rehearsal boundary holds."""

    protocol_sha256, mode = _protocol_execution_mode(protocol_path)
    if mode is not Q1ExecutionMode.REHEARSAL:
        raise Q1ExecutionError(
            "the orchestration rehearsal is disabled once the protocol authorizes Q1 execution"
        )
    return protocol_sha256


def prepare_rehearsal_inputs(root: Path) -> RehearsalInputs:
    """Write the accepted Q0 report, salt, review, and private roots under ``root``."""

    protocol_sha256 = require_rehearsal_protocol()
    _mkdir_exact(root, mode=0o700)
    execution_root = root / "execution"
    attempt_registry_directory = root / "registry"
    _mkdir_exact(execution_root, mode=0o700)
    _mkdir_exact(attempt_registry_directory, mode=0o700)

    q0_report_path = root / "q0-report.json"
    q0_payload = canonical_json_bytes(run_qualification(Q0_PROTOCOL_PATH).as_dict(), newline=True)
    if sha256_bytes(q0_payload) != Q0_REPORT_SHA256:
        raise Q1ExecutionError("regenerated Q0 report differs from the accepted Q0 digest")
    _write_exact(q0_report_path, q0_payload, mode=0o600)

    secret_salt_path = root / "rehearsal-salt.bin"
    _write_exact(secret_salt_path, os.urandom(_SALT_BYTES), mode=0o600)

    manifest, implementation_sha256 = implementation_manifest(Q1_IMPLEMENTATION_PATHS)
    prospective_review_path = root / "rehearsal-review.json"
    _write_exact(
        prospective_review_path,
        canonical_json_bytes(
            _rehearsal_review(
                protocol_sha256=protocol_sha256,
                implementation_sha256=implementation_sha256,
                reviewed_source_count=len(manifest),
            ),
            newline=True,
        ),
        mode=0o600,
    )

    entry_report_path = root / "rehearsal-entry.json"
    report = run_entry_qualification(
        q0_report_path=q0_report_path,
        secret_salt_path=secret_salt_path,
        prospective_review_path=prospective_review_path,
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
        execution_mode=Q1ExecutionMode.REHEARSAL,
    )
    if not report.passed:
        failed = [check.name for check in report.checks if not check.passed]
        raise Q1ExecutionError(f"rehearsal entry qualification failed: {', '.join(failed)}")
    write_report(report, entry_report_path)
    return RehearsalInputs(
        q0_report_path=q0_report_path,
        secret_salt_path=secret_salt_path,
        prospective_review_path=prospective_review_path,
        entry_report_path=entry_report_path,
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
        output_directory=execution_root / "rehearsal-result",
    )


def run_rehearsal(root: Path) -> Q1RunOutputs:
    """Prepare inputs and execute one complete result-free orchestration."""

    inputs = prepare_rehearsal_inputs(root)
    return run_q1_orchestration_rehearsal(
        entry_report_path=inputs.entry_report_path,
        q0_report_path=inputs.q0_report_path,
        secret_salt_path=inputs.secret_salt_path,
        prospective_review_path=inputs.prospective_review_path,
        execution_root=inputs.execution_root,
        attempt_registry_directory=inputs.attempt_registry_directory,
        output_directory=inputs.output_directory,
    )


def _rehearsal_review(
    *,
    protocol_sha256: str,
    implementation_sha256: str,
    reviewed_source_count: int,
) -> dict[str, object]:
    return {
        "schema": PROSPECTIVE_REVIEW_SCHEMA,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "reviewer": REHEARSAL_REVIEWER,
        "review_method": "adversarial_result_free_selected_source_review",
        "assurance_boundary": "local_procedural_review_without_external_signature",
        "reviewed_source_count": reviewed_source_count,
        "review_scope": list(PROSPECTIVE_REVIEW_SCOPE),
        "blocking_findings": [],
        "nonblocking_findings": [],
        "q1_environment_interactions": 0,
        "q1_private_draws": 0,
        "claim_eligible": False,
        "formal_authorized": False,
        "passed": True,
        "statement": (
            "Machine-generated rehearsal review. It binds only the unauthorized "
            "rehearsal protocol digest, asserts no independent judgement, and cannot "
            "satisfy the independent prospective review required before Q1."
        ),
    }


def _mkdir_exact(path: Path, *, mode: int) -> None:
    path.mkdir(mode=mode, parents=True, exist_ok=True)
    os.chmod(path, mode)
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != mode:
        raise Q1ExecutionError(f"rehearsal directory custody differs: {path}")


def _write_exact(path: Path, payload: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Q1ExecutionError(f"rehearsal input write made no progress: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    """Run one rehearsal and print its published artifact directory."""

    if sys.flags.no_site != 1:
        raise SystemExit("the Q1 orchestration rehearsal requires invocation with Python -S")
    parser = argparse.ArgumentParser(description="Run one result-free Q1 orchestration rehearsal.")
    parser.add_argument("--root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    outputs = run_rehearsal(arguments.root)
    print(str(outputs.output_directory))
    return 0


__all__ = (
    "REHEARSAL_REVIEWER",
    "RehearsalInputs",
    "main",
    "prepare_rehearsal_inputs",
    "require_rehearsal_protocol",
    "run_rehearsal",
)


if __name__ == "__main__":
    raise SystemExit(main())
