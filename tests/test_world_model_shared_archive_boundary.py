from __future__ import annotations

import copy
import hashlib
import tarfile
from pathlib import Path
from typing import cast

import pytest

import bench.world_model_lifecycle.artifact_audit as artifact_audit
import bench.world_model_lifecycle.binding as binding


def test_one_writer_archive_crosses_both_real_readers_and_rejects_stale_member_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    development = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
    )
    producer = repository / "producer"
    development.mkdir(parents=True)
    producer.mkdir()
    producer_payload = b'{"lane":"development"}\n'
    evidence_payload = b'{"passed":true}\n'
    (producer / "result.json").write_bytes(producer_payload)

    monkeypatch.setattr(binding, "REPO", repository)
    monkeypatch.setattr(
        binding,
        "DEVELOPMENT_RESULTS_ROOT",
        development,
    )
    monkeypatch.chdir(repository)

    archive_path, archive_identity = binding._write_qualification_archive(
        destination_directory=development,
        producer_root=producer,
        evidence_payloads={
            "evidence/independent-audit.json": evidence_payload,
        },
    )
    retained_members = {
        "evidence/independent-audit.json",
        "producer/result.json",
    }
    expected_retained = {
        "evidence/independent-audit.json": evidence_payload,
        "producer/result.json": producer_payload,
    }

    central_retained = binding._stream_qualification_archive(
        archive_path,
        archive_identity,
        retained_members=retained_members,
    )
    independent_retained = (
        artifact_audit._verify_development_qualification_archive(
            archive_identity,
            members=cast(
                list[object],
                archive_identity["members"],
            ),
            retain_members=frozenset(retained_members),
        )
    )
    assert central_retained == independent_retained == expected_retained

    with tarfile.open(archive_path, mode="r:") as archive:
        member = archive.getmember("evidence/independent-audit.json")
        payload_offset = member.offset_data
    corrupted_payload = bytearray(archive_path.read_bytes())
    corrupted_payload[payload_offset] ^= 1
    corrupted_digest = hashlib.sha256(corrupted_payload).hexdigest()
    corrupted_path = development / (
        f"development-qualification-{corrupted_digest[:16]}.tar"
    )
    corrupted_path.write_bytes(corrupted_payload)
    corrupted_identity = copy.deepcopy(archive_identity)
    corrupted_identity.update(
        {
            "file": corrupted_path.name,
            "canonical_path": corrupted_path.relative_to(
                repository
            ).as_posix(),
            "sha256": corrupted_digest,
        }
    )

    with pytest.raises(RuntimeError, match="member digest changed"):
        binding._stream_qualification_archive(
            corrupted_path,
            corrupted_identity,
            retained_members=retained_members,
        )
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="member bytes changed",
    ):
        artifact_audit._verify_development_qualification_archive(
            corrupted_identity,
            members=cast(
                list[object],
                corrupted_identity["members"],
            ),
            retain_members=frozenset(retained_members),
        )
