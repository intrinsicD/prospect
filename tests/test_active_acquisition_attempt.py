from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest

import bench.active_acquisition.attempt as attempt_module
from bench.active_acquisition.attempt import (
    ATTEMPT_MARKER_FILENAME,
    ATTEMPT_MARKER_SCHEMA,
    AttemptRegistryError,
    Q1AttemptMarker,
    Q1RunIdentity,
    attempt_marker_path,
    derive_run_identity,
)
from bench.active_acquisition.attempt import (
    claim_attempt as _claim_attempt_api,
)
from bench.active_acquisition.attempt import (
    finalize_attempt as _finalize_attempt_api,
)
from bench.active_acquisition.attempt import (
    load_attempt_marker as _load_attempt_marker_api,
)

_WORKER_CAPABILITY_SHA256 = hashlib.sha256(b"wm002-test-worker-capability").hexdigest()


def claim_attempt(registry_directory: Path, identity: Q1RunIdentity) -> Path:
    return _claim_attempt_api(
        registry_directory,
        identity,
        worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    )


def load_attempt_marker(
    path: Path,
    *,
    expected_identity: Q1RunIdentity,
) -> Q1AttemptMarker:
    return _load_attempt_marker_api(
        path,
        expected_identity=expected_identity,
        expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    )


def finalize_attempt(
    path: Path,
    identity: Q1RunIdentity,
    *,
    status: Literal["completed", "failed"],
    artifacts: Mapping[str, Path] | None = None,
) -> Q1AttemptMarker:
    return _finalize_attempt_api(
        path,
        identity,
        status=status,
        expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
        artifacts=artifacts,
    )


def _identity() -> Q1RunIdentity:
    return derive_run_identity(
        protocol_version="0.3.0-q1",
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        q0_report_sha256="3" * 64,
        entry_qualification_sha256="4" * 64,
        salt_commitment_sha256="5" * 64,
    )


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry"
    path.mkdir(mode=0o700)
    return path


def _other_identity() -> Q1RunIdentity:
    return derive_run_identity(
        protocol_version="0.3.0-q1",
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        q0_report_sha256="3" * 64,
        entry_qualification_sha256="a" * 64,
        salt_commitment_sha256="b" * 64,
    )


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for name in (
        "aggregate",
        "checkpoint_frames",
        "checkpoint_index",
        "private_audit",
        "raw_trace",
        "restored_trace",
    ):
        path = tmp_path / name
        path.write_bytes(f"{name}\n".encode())
        artifacts[name] = path
    return artifacts


def test_run_identity_binds_every_immutable_execution_input() -> None:
    identity = _identity()
    assert identity.run_id == f"wm002-q1-{identity.run_sha256}"
    assert identity.attempt_id == f"{identity.run_id}-attempt-0001"
    changed = derive_run_identity(
        protocol_version=identity.protocol_version,
        protocol_sha256=identity.protocol_sha256,
        implementation_sha256=identity.implementation_sha256,
        q0_report_sha256=identity.q0_report_sha256,
        entry_qualification_sha256="a" * 64,
        salt_commitment_sha256=identity.salt_commitment_sha256,
    )
    assert changed.run_id != identity.run_id


def test_run_identity_matches_the_frozen_canonical_golden_vector() -> None:
    identity = _identity()

    assert identity.run_sha256 == "ea6feeb09ddfb67aaeec7f26b5b36753fb2457a236554cdcb93b1ac22c3f7bc3"
    assert identity.run_id == ("wm002-q1-ea6feeb09ddfb67aaeec7f26b5b36753fb2457a236554cdcb93b1ac22c3f7bc3")


def test_attempt_claim_is_private_durable_and_exclusive(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    path = claim_attempt(registry, identity)

    marker = load_attempt_marker(path, expected_identity=identity)
    assert marker.status == "started"
    assert registry.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["schema"] == ATTEMPT_MARKER_SCHEMA
    with pytest.raises(AttemptRegistryError, match="already been claimed"):
        claim_attempt(registry, identity)


def test_marker_v2_commitment_is_required_and_preserved_exactly(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    with pytest.raises(AttemptRegistryError, match="status is invalid"):
        Q1AttemptMarker(identity, "invalid", _WORKER_CAPABILITY_SHA256)  # type: ignore[arg-type]

    with pytest.raises(AttemptRegistryError, match="worker_capability_sha256"):
        _claim_attempt_api(registry, identity, worker_capability_sha256="invalid")

    marker_path = claim_attempt(registry, identity)
    started = load_attempt_marker(marker_path, expected_identity=identity)
    assert started.worker_capability_sha256 == _WORKER_CAPABILITY_SHA256
    marker = finalize_attempt(marker_path, identity, status="failed")
    assert marker.worker_capability_sha256 == _WORKER_CAPABILITY_SHA256
    assert json.loads(marker_path.read_text())["worker_capability_sha256"] == _WORKER_CAPABILITY_SHA256

    with pytest.raises(AttemptRegistryError, match="capability commitment mismatch"):
        _load_attempt_marker_api(
            marker_path,
            expected_identity=identity,
            expected_worker_capability_sha256="f" * 64,
        )


def test_every_run_identity_claims_the_same_experiment_global_path(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = _identity()
    second = _other_identity()

    first_path = attempt_marker_path(registry, first)
    second_path = attempt_marker_path(registry, second)
    assert first.run_id != second.run_id
    assert first_path == second_path == registry / ATTEMPT_MARKER_FILENAME

    claim_attempt(registry, first)
    with pytest.raises(AttemptRegistryError, match="already been claimed"):
        claim_attempt(registry, second)
    with pytest.raises(AttemptRegistryError, match="identity mismatch"):
        load_attempt_marker(first_path, expected_identity=second)


def test_attempt_claim_fchmod_overrides_a_restrictive_umask(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    old_umask = os.umask(0o777)
    try:
        marker_path = claim_attempt(registry, _identity())
    finally:
        os.umask(old_umask)

    assert marker_path.stat().st_mode & 0o777 == 0o600


def test_partial_claim_still_consumes_the_global_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)

    def fail_after_prefix(descriptor: int, payload: bytes) -> None:
        os.write(descriptor, payload[:8])
        raise OSError("injected marker initialization failure")

    monkeypatch.setattr(attempt_module, "_write_all", fail_after_prefix)
    with pytest.raises(OSError, match="injected marker initialization failure"):
        claim_attempt(registry, _identity())

    marker_path = registry / ATTEMPT_MARKER_FILENAME
    assert marker_path.read_bytes()
    assert marker_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(AttemptRegistryError, match="already been claimed"):
        claim_attempt(registry, _other_identity())


def test_hard_linked_attempt_marker_is_rejected_at_point_of_use(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    os.link(marker_path, tmp_path / "marker-alias")

    with pytest.raises(AttemptRegistryError, match="exactly one hard link"):
        load_attempt_marker(marker_path, expected_identity=identity)


def test_attempt_marker_mode_0400_is_rejected_at_point_of_use(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    marker_path.chmod(0o400)

    with pytest.raises(AttemptRegistryError, match="exactly 0600"):
        load_attempt_marker(marker_path, expected_identity=identity)


def test_completed_attempt_binds_exact_six_artifacts(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    artifacts = _artifacts(tmp_path)

    marker = finalize_attempt(
        marker_path,
        identity,
        status="completed",
        artifacts=artifacts,
    )
    assert marker.status == "completed"
    assert marker.worker_capability_sha256 == _WORKER_CAPABILITY_SHA256
    assert tuple(name for name, _ in marker.artifact_sha256) == tuple(sorted(artifacts))
    assert load_attempt_marker(marker_path, expected_identity=identity) == marker
    with pytest.raises(AttemptRegistryError, match="started"):
        finalize_attempt(marker_path, identity, status="failed")


def test_completed_attempt_rejects_a_symlink_artifact(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    artifacts = _artifacts(tmp_path)
    link = tmp_path / "aggregate-link"
    link.symlink_to(artifacts["aggregate"])
    artifacts["aggregate"] = link

    with pytest.raises(AttemptRegistryError, match="open result artifact"):
        finalize_attempt(marker_path, identity, status="completed", artifacts=artifacts)
    assert load_attempt_marker(marker_path, expected_identity=identity).status == "started"


def test_completed_attempt_rejects_a_hard_linked_artifact(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    artifacts = _artifacts(tmp_path)
    os.link(artifacts["aggregate"], tmp_path / "aggregate-alias")

    with pytest.raises(AttemptRegistryError, match="exactly one hard link"):
        finalize_attempt(marker_path, identity, status="completed", artifacts=artifacts)
    assert load_attempt_marker(marker_path, expected_identity=identity).status == "started"


def test_conflicting_concurrent_finalizations_allow_exactly_one_winner(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    artifacts = _artifacts(tmp_path)
    barrier = threading.Barrier(2)

    def complete() -> Q1AttemptMarker:
        barrier.wait(timeout=5)
        return finalize_attempt(
            marker_path,
            identity,
            status="completed",
            artifacts=artifacts,
        )

    def fail() -> Q1AttemptMarker:
        barrier.wait(timeout=5)
        return finalize_attempt(marker_path, identity, status="failed")

    successes: list[Q1AttemptMarker] = []
    failures: list[AttemptRegistryError] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(complete), executor.submit(fail))
        for future in futures:
            try:
                successes.append(future.result(timeout=10))
            except AttemptRegistryError as error:
                failures.append(error)

    assert len(successes) == 1
    assert len(failures) == 1
    assert "only a started marker" in str(failures[0])
    assert load_attempt_marker(marker_path, expected_identity=identity) == successes[0]


def test_failed_attempt_remains_and_cannot_bind_artifacts(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    marker = finalize_attempt(marker_path, identity, status="failed")
    assert marker.status == "failed"
    assert marker_path.exists()
    with pytest.raises(AttemptRegistryError, match="already been claimed"):
        claim_attempt(registry, identity)


def test_marker_tamper_and_0755_registry_permissions_fail_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    identity = _identity()
    marker_path = claim_attempt(registry, identity)
    value = json.loads(marker_path.read_text())
    value["run_id"] = "wm002-q1-" + "f" * 64
    marker_path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.chmod(marker_path, 0o600)
    with pytest.raises(AttemptRegistryError, match="identity mismatch"):
        load_attempt_marker(marker_path, expected_identity=identity)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    os.chmod(unsafe, 0o755)
    with pytest.raises(AttemptRegistryError, match="exactly 0700"):
        claim_attempt(unsafe, identity)


def test_registry_symlink_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    link = tmp_path / "registry-link"
    link.symlink_to(registry, target_is_directory=True)
    with pytest.raises(AttemptRegistryError, match="non-symlink"):
        claim_attempt(link, _identity())
