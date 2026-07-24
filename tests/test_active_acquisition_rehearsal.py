"""Result-free happy-path coverage for the complete Q1 orchestration.

The production entry path requires ``execution_authorized: true``, so until now
no test executed the four-producer, 28-restore-lane, merge, validation,
publication, and completed-marker sequence as one whole.  These tests run that
exact sequence under the rehearsal budget and pin the boundary that keeps a
rehearsal from ever being read as Q1 evidence.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from bench.active_acquisition import q1, rehearsal
from bench.active_acquisition.contracts import (
    ARM_ORDER,
    Q1_PROTOCOL_PATH,
    ContractError,
    canonical_json_bytes,
    validate_artifact,
)
from bench.active_acquisition.q1_qualification import (
    _prospective_review_violations,
    _protocol_boundary_violations,
    _protocol_snapshot,
)
from bench.active_acquisition.seeding import MASTER_COUNT, Q1ExecutionMode, episodes_per_master

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REHEARSAL_EPISODES = episodes_per_master(Q1ExecutionMode.REHEARSAL)
REHEARSAL_TOTAL = MASTER_COUNT * len(ARM_ORDER) * REHEARSAL_EPISODES
EXPECTED_SELECTION = {
    "prospect_expected_return": "strong",
    "independent_fraction_oracle": "strong",
    "goal_only": "skip",
    "raw_observation_entropy": "nuisance",
    "eig_only": "overpowered",
    "shuffled_information": "weak",
}
CANONICAL_ARTIFACTS = (
    "aggregate.json",
    "checkpoint-frames.bin",
    "checkpoint-index.jsonl",
    "private-audit.jsonl",
    "raw-trace.jsonl",
    "restored-trace.jsonl",
)


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run exactly one complete orchestration rehearsal in a fresh process."""

    root = tmp_path_factory.mktemp("q1-rehearsal")
    completed = subprocess.run(
        [sys.executable, "-S", "-m", "bench.active_acquisition.rehearsal", "--root", str(root)],
        cwd=REPOSITORY_ROOT,
        env=q1._q1_child_environment(),
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"rehearsal failed:\n{completed.stderr.decode('utf-8', 'replace')[-4000:]}")
    output_directory = Path(completed.stdout.decode().strip())
    return {
        "root": root,
        "output_directory": output_directory,
        "marker": json.loads((root / "registry" / "wm002-q1.attempt.json").read_bytes()),
        "aggregate": json.loads((output_directory / "aggregate.json").read_bytes()),
        "raw": _read_rows(output_directory / "raw-trace.jsonl"),
        "restored": _read_rows(output_directory / "restored-trace.jsonl"),
        "index": _read_rows(output_directory / "checkpoint-index.jsonl"),
        "private": _read_rows(output_directory / "private-audit.jsonl"),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").splitlines()]


def test_rehearsal_publishes_exactly_the_canonical_private_artifact_set(published: dict[str, Any]) -> None:
    output_directory = published["output_directory"]
    metadata = output_directory.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert sorted(path.name for path in output_directory.iterdir()) == sorted(CANONICAL_ARTIFACTS)
    for name in CANONICAL_ARTIFACTS:
        artifact = output_directory / name
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
        assert artifact.stat().st_size > 0
    assert not (output_directory.parent / f"{output_directory.name}.incomplete").exists()


def test_rehearsal_rows_cover_every_master_arm_lane_in_canonical_order(published: dict[str, Any]) -> None:
    for name in ("raw", "restored", "index", "private"):
        assert len(published[name]) == REHEARSAL_TOTAL, name
    expected = [
        (master, arm, episode)
        for master in range(MASTER_COUNT)
        for arm in ARM_ORDER
        for episode in range(REHEARSAL_EPISODES)
    ]
    assert [(row["master"], row["arm"], row["episode"]) for row in published["raw"]] == expected
    assert [(row["master"], row["arm"], row["episode"]) for row in published["restored"]] == expected
    assert [(row["master"], row["arm_id"], row["episode"]) for row in published["private"]] == expected


def test_rehearsal_selects_each_arm_declared_action(published: dict[str, Any]) -> None:
    selected: defaultdict[str, set[str]] = defaultdict(set)
    for row in published["raw"]:
        selected[row["arm"]].add(row["acquisition"]["selected_action"])
    for arm, action in EXPECTED_SELECTION.items():
        assert selected[arm] == {action}, arm
    assert selected["uniform_random"]


def test_rehearsal_restores_every_episode_in_a_distinct_fresh_process(published: dict[str, Any]) -> None:
    restorer_pids = {row["restorer_pid"] for row in published["restored"]}
    producer_pids = {row["producer_pid"] for row in published["restored"]}
    assert len(producer_pids) == MASTER_COUNT
    assert len(restorer_pids) == MASTER_COUNT * len(ARM_ORDER)
    assert not restorer_pids & producer_pids


def test_rehearsal_marker_completes_and_binds_every_published_artifact(published: dict[str, Any]) -> None:
    marker = published["marker"]
    assert marker["status"] == "completed"
    assert sorted(marker["artifact_sha256"]) == sorted(
        {
            "aggregate",
            "checkpoint_frames",
            "checkpoint_index",
            "private_audit",
            "raw_trace",
            "restored_trace",
        }
    )
    assert marker["q0_report_sha256"] == published["aggregate"]["q0_report_sha256"]
    assert marker["run_id"] == published["aggregate"]["run_id"]


def test_rehearsal_aggregate_carries_the_exact_rehearsal_budget(published: dict[str, Any]) -> None:
    aggregate = published["aggregate"]
    assert aggregate["schema"] == q1.REHEARSAL_AGGREGATE_SCHEMA
    assert aggregate["claim_eligible"] is False
    assert aggregate["formal_authorized"] is False
    assert aggregate["producer_analysis_authoritative"] is False
    assert aggregate["counts"] == {
        "masters": MASTER_COUNT,
        "arms": len(ARM_ORDER),
        "episodes": REHEARSAL_TOTAL,
        "environment_steps": 2 * REHEARSAL_TOTAL,
        "transitions": 2 * REHEARSAL_TOTAL,
        "acquisition_updates": REHEARSAL_TOTAL,
        "terminal_updates": 0,
        "checkpoints": REHEARSAL_TOTAL,
        "restores": REHEARSAL_TOTAL,
    }
    assert len(aggregate["arm_means"]) == MASTER_COUNT * len(ARM_ORDER)
    assert all(row["episode_count"] == REHEARSAL_EPISODES for row in aggregate["arm_means"])


def test_rehearsal_aggregate_cannot_satisfy_the_frozen_q1_aggregate_contract(published: dict[str, Any]) -> None:
    aggregate = published["aggregate"]
    with pytest.raises(ContractError):
        validate_artifact("aggregate", aggregate)
    with pytest.raises(q1.Q1ExecutionError):
        q1._validate_aggregate(aggregate, execution_mode=Q1ExecutionMode.PRODUCTION)
    q1._validate_aggregate(aggregate, execution_mode=Q1ExecutionMode.REHEARSAL)


def test_independent_auditor_rejects_the_rehearsal_as_q1_evidence(
    published: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The auditor is the last line: a rehearsal must fail it, not merely differ."""

    root = published["root"]
    audit_output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "bench.active_acquisition.q1_audit",
            str(published["output_directory"]),
            "--secret-salt",
            str(root / "rehearsal-salt.bin"),
            "--attempt-marker",
            str(root / "registry" / "wm002-q1.attempt.json"),
            "--q0-report",
            str(root / "q0-report.json"),
            "--entry-report",
            str(root / "rehearsal-entry.json"),
            "--prospective-review",
            str(root / "rehearsal-review.json"),
            "--output",
            str(audit_output),
        ],
        cwd=REPOSITORY_ROOT,
        env=q1._q1_child_environment(),
        capture_output=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode != 0 or audit_output.exists()
    audit = json.loads(audit_output.read_bytes())
    assert audit["passed"] is False
    gates = {row["gate"]: row for row in audit["gates"]}
    assert gates["Q1-K0"]["passed"] is False
    assert any("execution_authorized" in violation for violation in gates["Q1-K0"]["violations"])
    assert gates["Q1-K4"]["passed"] is False


def test_public_rehearsal_rows_carry_no_private_hidden_state(published: dict[str, Any]) -> None:
    public = json.dumps([published["raw"], published["index"], published["restored"], published["aggregate"]])
    for fragment in ("theta", "counterfactual", "hidden_regime", "hmac", "private_salt", "schedule_position"):
        assert fragment not in public


def test_the_two_execution_modes_are_mutually_exclusive_protocol_boundaries() -> None:
    _digest, protocol = _protocol_snapshot(Q1_PROTOCOL_PATH)
    assert _protocol_boundary_violations(protocol, execution_mode=Q1ExecutionMode.REHEARSAL) == []
    production = _protocol_boundary_violations(protocol, execution_mode=Q1ExecutionMode.PRODUCTION)
    assert "experiment boundary mismatch:execution_authorized" in production
    assert q1._protocol_execution_mode(Q1_PROTOCOL_PATH)[1] is Q1ExecutionMode.REHEARSAL


def test_rehearsal_is_disabled_once_the_protocol_authorizes_q1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rehearsal,
        "_protocol_execution_mode",
        lambda path: ("0" * 64, Q1ExecutionMode.PRODUCTION),
    )
    with pytest.raises(q1.Q1ExecutionError, match="disabled once the protocol authorizes"):
        rehearsal.require_rehearsal_protocol()


def test_production_entry_refuses_the_machine_generated_rehearsal_review(tmp_path: Path) -> None:
    """The gate cannot verify independence, but it can refuse its own harness's review.

    Nothing else in the entry chain distinguishes a machine-generated review
    from an independent one: the reviewer field is free text and every digest,
    scope, and counter check passes either way.
    """

    review = rehearsal._rehearsal_review(
        protocol_sha256="a" * 64,
        implementation_sha256="b" * 64,
        reviewed_source_count=7,
    )
    path = tmp_path / "review.json"
    path.write_bytes(canonical_json_bytes(review, newline=True))
    _digest, production = _prospective_review_violations(
        path,
        protocol_sha256="a" * 64,
        implementation_sha256="b" * 64,
        reviewed_source_count=7,
        execution_mode=Q1ExecutionMode.PRODUCTION,
    )
    assert "prospective review is the machine-generated rehearsal review" in production
    _digest, rehearsal_violations = _prospective_review_violations(
        path,
        protocol_sha256="a" * 64,
        implementation_sha256="b" * 64,
        reviewed_source_count=7,
        execution_mode=Q1ExecutionMode.REHEARSAL,
    )
    assert rehearsal_violations == []


def test_rehearsal_entry_refuses_to_consume_an_independent_review(tmp_path: Path) -> None:
    """A rehearsal must never burn the independent review written for Q1."""

    review = dict(
        rehearsal._rehearsal_review(
            protocol_sha256="a" * 64,
            implementation_sha256="b" * 64,
            reviewed_source_count=7,
        )
    )
    review["reviewer"] = "an independent human reviewer"
    path = tmp_path / "independent-review.json"
    path.write_bytes(canonical_json_bytes(review, newline=True))
    _digest, violations = _prospective_review_violations(
        path,
        protocol_sha256="a" * 64,
        implementation_sha256="b" * 64,
        reviewed_source_count=7,
        execution_mode=Q1ExecutionMode.REHEARSAL,
    )
    assert "rehearsal entry must not consume an independent prospective review" in violations


def test_rehearsal_directory_setup_refuses_a_symlinked_root(tmp_path: Path) -> None:
    """Refuse the link before chmod-ing whatever it points at."""

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o755)
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    (root / "execution").symlink_to(elsewhere)
    with pytest.raises(q1.Q1ExecutionError, match="not a regular directory"):
        rehearsal.prepare_rehearsal_inputs(root)
    assert stat.S_IMODE(elsewhere.stat().st_mode) == 0o755


def test_rehearsal_review_is_self_declared_rehearsal_only_and_result_free() -> None:
    review = rehearsal._rehearsal_review(
        protocol_sha256="a" * 64,
        implementation_sha256="b" * 64,
        reviewed_source_count=7,
    )
    assert review["reviewer"] == rehearsal.REHEARSAL_REVIEWER
    assert "rehearsal-only" in str(review["reviewer"])
    assert review["q1_environment_interactions"] == 0
    assert review["q1_private_draws"] == 0
    assert review["claim_eligible"] is False
    assert review["formal_authorized"] is False


def test_rehearsal_inputs_are_private_and_regenerate_the_accepted_q0_report(tmp_path: Path) -> None:
    inputs = rehearsal.prepare_rehearsal_inputs(tmp_path / "inputs")
    for path in (inputs.q0_report_path, inputs.secret_salt_path, inputs.prospective_review_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for directory in (inputs.execution_root, inputs.attempt_registry_directory):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert os.path.getsize(inputs.secret_salt_path) >= 32
    entry = json.loads(inputs.entry_report_path.read_bytes())
    assert entry["passed"] is True
    assert entry["claim_eligible"] is False
    assert entry["formal_authorized"] is False
    assert entry["q1_environment_interactions"] == 0
    assert entry["q1_private_draws"] == 0
