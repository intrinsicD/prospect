from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bench.active_acquisition.q1_qualification as qualification
from bench.active_acquisition.contracts import (
    Q0_IMPLEMENTATION_SHA256,
    Q0_PROTOCOL_SHA256,
    Q0_REPORT_SHA256,
    Q1_PROTOCOL_PATH,
    Q1_SCHEMA_PATHS,
    canonical_json_bytes,
    implementation_manifest,
    protocol_document,
    sha256_bytes,
)
from bench.active_acquisition.q1_qualification import (
    _MAX_QUALIFICATION_INPUT_BYTES,
    _UNDECLARED_HIDDEN_STATE_SENTINEL_KEY,
    ENTRY_CHECK_ORDER,
    PROSPECTIVE_REVIEW_SCHEMA,
    PROSPECTIVE_REVIEW_SCOPE,
    Q1_IMPLEMENTATION_PATHS,
    _emitted_artifact_schema_violations,
    _non_strict_object_paths,
    _prospective_review_violations,
    _protocol_code_parity_violations,
    _q0_binding_violations,
    _read_private_salt,
    _read_stable_json_document,
    _resource_preflight,
    _write_exclusive_durable,
    validate_entry_report,
)


def test_qualification_cli_requires_python_no_site_mode() -> None:
    if qualification.sys.flags.no_site == 1:
        pytest.skip("test requires a site-enabled parent interpreter")
    with pytest.raises(SystemExit, match="Q1 qualification requires invocation with Python -S"):
        qualification._require_no_site_cli()


def test_private_salt_reader_checks_the_open_regular_descriptor(tmp_path: Path) -> None:
    salt_path = tmp_path / "salt.bin"
    salt = bytes(range(32))
    salt_path.write_bytes(salt)
    salt_path.chmod(0o600)
    assert _read_private_salt(salt_path) == salt

    salt_path.chmod(0o640)
    with pytest.raises(ValueError, match="permissions"):
        _read_private_salt(salt_path)

    salt_path.chmod(0o400)
    with pytest.raises(ValueError, match="permissions"):
        _read_private_salt(salt_path)

    salt_path.chmod(0o600)
    symlink_path = tmp_path / "salt-link.bin"
    symlink_path.symlink_to(salt_path)
    with pytest.raises(ValueError, match="opened safely"):
        _read_private_salt(symlink_path)

    with pytest.raises(ValueError, match="regular"):
        _read_private_salt(tmp_path)


def test_private_salt_reader_rejects_hard_links_and_enforces_exact_size_bound(tmp_path: Path) -> None:
    salt_path = tmp_path / "salt.bin"
    salt_path.write_bytes(b"s" * _MAX_QUALIFICATION_INPUT_BYTES)
    salt_path.chmod(0o600)

    assert len(_read_private_salt(salt_path)) == _MAX_QUALIFICATION_INPUT_BYTES

    hard_link = tmp_path / "salt-hard-link.bin"
    os.link(salt_path, hard_link)
    with pytest.raises(ValueError, match="exactly one hard link"):
        _read_private_salt(salt_path)
    hard_link.unlink()

    salt_path.write_bytes(b"s" * (_MAX_QUALIFICATION_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="bounded document limit"):
        _read_private_salt(salt_path)


def test_stable_json_reader_accepts_exact_bound_and_rejects_oversize(tmp_path: Path) -> None:
    path = tmp_path / "bounded.json"
    empty_payload = canonical_json_bytes({"value": ""}, newline=True)
    value = "v" * (_MAX_QUALIFICATION_INPUT_BYTES - len(empty_payload))
    payload = canonical_json_bytes({"value": value}, newline=True)
    assert len(payload) == _MAX_QUALIFICATION_INPUT_BYTES
    path.write_bytes(payload)

    digest, decoded = _read_stable_json_document(
        path,
        label="bounded test document",
        require_canonical=True,
    )
    assert digest == hashlib.sha256(payload).hexdigest()
    assert decoded == {"value": value}

    path.write_bytes(canonical_json_bytes({"value": value + "v"}, newline=True))
    with pytest.raises(ValueError, match="bounded document limit"):
        _read_stable_json_document(
            path,
            label="bounded test document",
            require_canonical=True,
        )


def test_stable_json_reader_rejects_nested_duplicate_object_keys_without_echo(
    tmp_path: Path,
) -> None:
    private_key = "recognizable-private-duplicate-key"
    path = tmp_path / "duplicate.json"
    path.write_bytes(('{"outer":{"' + private_key + '":1,"' + private_key + '":2}}\n').encode("ascii"))

    with pytest.raises(
        ValueError,
        match="duplicate-key test document is not valid finite UTF-8 JSON",
    ) as captured:
        _read_stable_json_document(
            path,
            label="duplicate-key test document",
            require_canonical=False,
        )

    assert private_key not in str(captured.value)
    assert captured.value.__cause__ is not None
    assert str(captured.value.__cause__) == "duplicate JSON object key is forbidden"


def test_stable_json_reader_rejects_in_place_mutation_without_echo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_text = "recognizable-private-mutated-value"
    path = tmp_path / "mutating.json"
    path.write_bytes(canonical_json_bytes({"safe": True}, newline=True))
    real_read = qualification.os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, length: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, length)
        if chunk and not mutated:
            mutated = True
            with path.open("ab") as stream:
                stream.write(private_text.encode("ascii"))
        return chunk

    monkeypatch.setattr(qualification.os, "read", mutate_after_first_read)
    with pytest.raises(ValueError, match="changed during its descriptor read") as captured:
        _read_stable_json_document(
            path,
            label="mutating test document",
            require_canonical=True,
        )
    assert mutated
    assert private_text not in str(captured.value)


def test_stable_json_reader_rejects_rename_away_and_replacement_with_old_link_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_text = "recognizable-private-renamed-value"
    path = tmp_path / "renamed.json"
    backup = tmp_path / "renamed-backup.json"
    replacement = tmp_path / "renamed-replacement.json"
    payload_a = canonical_json_bytes({"private": private_text}, newline=True)
    payload_b = canonical_json_bytes({"safe": True}, newline=True)
    path.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    real_fstat = qualification.os.fstat
    swapped = False
    fstat_calls = 0

    def rename_away_after_final_descriptor_stat(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls, swapped
        metadata = real_fstat(descriptor)
        fstat_calls += 1
        if fstat_calls == 2:
            swapped = True
            path.rename(backup)
            replacement.rename(path)
            assert real_fstat(descriptor).st_nlink == 1
        return metadata

    monkeypatch.setattr(qualification.os, "fstat", rename_away_after_final_descriptor_stat)
    with pytest.raises(ValueError, match="path changed during its descriptor read") as captured:
        _read_stable_json_document(
            path,
            label="rename-away test document",
            require_canonical=True,
        )

    assert swapped
    assert fstat_calls == 2
    assert backup.read_bytes() == payload_a
    assert backup.stat().st_nlink == 1
    assert path.read_bytes() == payload_b
    assert private_text not in str(captured.value)


def test_recursive_schema_strictness_detects_nested_open_objects() -> None:
    strict = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"nested": {"type": "object", "additionalProperties": False, "properties": {}}},
    }
    open_nested = copy.deepcopy(strict)
    properties = open_nested["properties"]
    assert isinstance(properties, dict)
    nested = properties["nested"]
    assert isinstance(nested, dict)
    nested["additionalProperties"] = {"type": "integer"}

    assert _non_strict_object_paths(strict) == ()
    assert _non_strict_object_paths(open_nested) == ("$.properties.nested",)


def test_hidden_state_sentinel_mutation_rejects_all_six_actual_development_samples() -> None:
    from bench.active_acquisition.q1 import run_development_qualification_probe

    probe = run_development_qualification_probe(
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        q0_report_sha256="3" * 64,
        salt_commitment_sha256="4" * 64,
    )
    assert probe.violations == ()
    assert set(probe.artifact_samples) == set(Q1_SCHEMA_PATHS)
    private_audit = probe.artifact_samples["private_audit"]
    assert isinstance(private_audit, dict)
    assert "theta" in private_audit
    for sample in probe.artifact_samples.values():
        assert isinstance(sample, dict)
        assert _UNDECLARED_HIDDEN_STATE_SENTINEL_KEY not in sample

    assert _emitted_artifact_schema_violations(probe.artifact_samples) == []


def test_entry_report_bytes_are_durable_and_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "entry.json"
    payload = b'{"fixture":true}\n'

    previous_umask = os.umask(0o777)
    try:
        _write_exclusive_durable(path, payload)
    finally:
        os.umask(previous_umask)
    assert path.read_bytes() == payload
    assert (path.stat().st_mode & 0o777) == 0o644
    assert path.stat().st_nlink == 1
    with pytest.raises(ValueError, match="already exists"):
        _write_exclusive_durable(path, b"replacement")
    assert path.read_bytes() == payload


def _review() -> dict[str, object]:
    return {
        "schema": PROSPECTIVE_REVIEW_SCHEMA,
        "protocol_version": "0.3.0-q1",
        "protocol_sha256": "1" * 64,
        "implementation_sha256": "2" * 64,
        "reviewer": "independent-test-reviewer",
        "review_method": "adversarial_result_free_selected_source_review",
        "assurance_boundary": "local_procedural_review_without_external_signature",
        "reviewed_source_count": 41,
        "review_scope": list(PROSPECTIVE_REVIEW_SCOPE),
        "blocking_findings": [],
        "nonblocking_findings": ["External immutable attestation remains future assurance."],
        "q1_environment_interactions": 0,
        "q1_private_draws": 0,
        "claim_eligible": False,
        "formal_authorized": False,
        "passed": True,
        "statement": "Result-free review of the exact selected-source closure.",
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value, newline=True))


def test_strict_prospective_review_accepts_exact_result_free_scope(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    _write(path, _review())

    digest, violations = _prospective_review_violations(
        path,
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        reviewed_source_count=41,
    )
    assert len(digest) == 64
    assert violations == []


def test_prospective_review_rejects_scope_or_source_count_drift(tmp_path: Path) -> None:
    value = _review()
    value["review_scope"] = list(reversed(PROSPECTIVE_REVIEW_SCOPE))
    value["reviewed_source_count"] = 40
    path = tmp_path / "review.json"
    _write(path, value)

    _, violations = _prospective_review_violations(
        path,
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        reviewed_source_count=41,
    )
    assert "review scope differs from the exact required scope and order" in violations
    assert "reviewed source count differs from the selected-source closure" in violations


def test_prospective_review_schema_rejects_extra_or_result_bearing_fields(tmp_path: Path) -> None:
    value = _review()
    value["theta"] = 1
    value["q1_environment_interactions"] = 1
    path = tmp_path / "review.json"
    _write(path, value)

    _, violations = _prospective_review_violations(
        path,
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        reviewed_source_count=41,
    )
    assert "review schema violation" in violations
    assert "prospective review is not result-free" in violations


def test_prospective_review_schema_diagnostic_never_echoes_private_value(tmp_path: Path) -> None:
    private_text = "recognizable-private-review-value"
    value = _review()
    value[private_text] = private_text
    path = tmp_path / "private-valued-review.json"
    _write(path, value)

    _digest_value, violations = _prospective_review_violations(
        path,
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        reviewed_source_count=41,
    )

    assert "review schema violation" in violations
    assert private_text not in repr(violations)


def test_prospective_review_path_replacement_cannot_pair_digest_a_with_validation_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_text = "recognizable-private-review-a"
    review_a = _review()
    review_a[private_text] = private_text
    payload_a = canonical_json_bytes(review_a, newline=True)
    payload_b = canonical_json_bytes(_review(), newline=True)
    path = tmp_path / "review.json"
    replacement = tmp_path / "replacement.json"
    path.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    real_read = qualification.os.read
    replaced = False

    def replace_after_first_read(descriptor: int, length: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, length)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(qualification.os, "read", replace_after_first_read)
    digest, violations = _prospective_review_violations(
        path,
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        reviewed_source_count=41,
    )

    assert replaced
    assert path.read_bytes() == payload_b
    assert digest == ""
    assert violations == ["prospective review validation failed:ValueError"]
    assert private_text not in repr(violations)


def test_q0_path_replacement_cannot_pair_digest_a_with_validation_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_a = {
        "schema": "prospect.wm002.active-acquisition.q0-qualification.v1",
        "protocol_sha256": Q0_PROTOCOL_SHA256,
        "implementation_sha256": Q0_IMPLEMENTATION_SHA256,
        "passed": False,
        "claim_eligible": False,
        "formal_authorized": False,
        "environment_interactions": 0,
    }
    report_b = {**report_a, "passed": True}
    payload_a = canonical_json_bytes(report_a, newline=True)
    payload_b = canonical_json_bytes(report_b, newline=True)
    digest_a = hashlib.sha256(payload_a).hexdigest()
    path = tmp_path / "q0.json"
    replacement = tmp_path / "q0-replacement.json"
    path.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    monkeypatch.setattr(qualification, "Q0_REPORT_SHA256", digest_a)
    protocol = {
        "q0_binding": {
            "report_sha256": digest_a,
            "protocol_sha256": Q0_PROTOCOL_SHA256,
            "implementation_sha256": Q0_IMPLEMENTATION_SHA256,
        }
    }
    real_read = qualification.os.read
    replaced = False

    def replace_after_first_read(descriptor: int, length: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, length)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(qualification.os, "read", replace_after_first_read)
    violations = _q0_binding_violations(path, protocol)

    assert replaced
    assert path.read_bytes() == payload_b
    assert violations == ["Q0 report validation failed:ValueError"]


def test_protocol_path_replacement_is_rejected_before_snapshot_can_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_a = {
        "schema": "prospect.wm002.active-acquisition.q1-protocol.v1",
        "experiment": {"protocol_version": "0.3.0-q1"},
        "snapshot": "a",
    }
    protocol_b = {**protocol_a, "snapshot": "b"}
    payload_a = canonical_json_bytes(protocol_a, newline=True)
    payload_b = canonical_json_bytes(protocol_b, newline=True)
    path = tmp_path / "protocol.json"
    replacement = tmp_path / "protocol-replacement.json"
    path.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    real_read = qualification.os.read
    replaced = False

    def replace_after_first_read(descriptor: int, length: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, length)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(qualification.os, "read", replace_after_first_read)
    with pytest.raises(ValueError, match="exactly one hard link"):
        qualification._protocol_snapshot(path)

    assert replaced
    assert path.read_bytes() == payload_b


def test_artifact_schema_digest_and_validation_share_snapshot_and_reject_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_a = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["safe"],
        "properties": {"safe": {"const": True}},
    }
    schema_b = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
    }
    payload_a = canonical_json_bytes(schema_a, newline=True)
    payload_b = canonical_json_bytes(schema_b, newline=True)
    path = tmp_path / "schema.json"
    replacement = tmp_path / "schema-replacement.json"
    path.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    monkeypatch.setattr(qualification, "Q1_SCHEMA_PATHS", {"fixture": path})
    schemas, digests = qualification._artifact_schema_snapshots()
    assert dict(digests) == {"fixture": hashlib.sha256(payload_a).hexdigest()}
    assert qualification._schema_accepts(schemas["fixture"], {"safe": True})
    assert not qualification._schema_accepts(schemas["fixture"], {"private": "recognizable"})

    real_read = qualification.os.read
    replaced = False

    def replace_after_first_read(descriptor: int, length: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, length)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, path)
        return chunk

    monkeypatch.setattr(qualification.os, "read", replace_after_first_read)
    with pytest.raises(ValueError, match="exactly one hard link"):
        qualification._artifact_schema_snapshots()

    assert replaced
    assert path.read_bytes() == payload_b


def _resource_samples() -> dict[str, object]:
    return {
        "raw_trace_max_bytes": 4096,
        "private_audit_max_bytes": 2048,
        "checkpoint_index_max_bytes": 1024,
        "checkpoint_frame_max_bytes": 16384,
        "restored_trace_max_bytes": 4096,
        "probe_duration_under_30_seconds": True,
    }


def test_resource_preflight_is_deterministic_and_requires_private_directories(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)

    first, first_violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )
    second, second_violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )
    assert first == second
    assert first_violations == second_violations == []
    assert first.passed
    assert first.execution_root is not None
    assert first.attempt_registry_directory is not None
    execution_metadata = execution_root.stat()
    registry_metadata = registry.stat()
    assert first.execution_root.as_dict() == {
        "canonical_path": str(execution_root.resolve()),
        "file_type": "directory",
        "mode": "0700",
        "st_dev": execution_metadata.st_dev,
        "st_gid": execution_metadata.st_gid,
        "st_ino": execution_metadata.st_ino,
        "st_uid": execution_metadata.st_uid,
    }
    assert first.attempt_registry_directory.as_dict() == {
        "canonical_path": str(registry.resolve()),
        "file_type": "directory",
        "mode": "0700",
        "st_dev": registry_metadata.st_dev,
        "st_gid": registry_metadata.st_gid,
        "st_ino": registry_metadata.st_ino,
        "st_uid": registry_metadata.st_uid,
    }
    assert first.max_restore_concurrency == 4
    assert first.required_free_bytes > first.estimated_canonical_bytes
    assert list(execution_root.iterdir()) == []


def test_resource_preflight_rejects_atomic_noreplace_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.active_acquisition import q1

    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)

    def reject_probe(_execution_root: Path) -> None:
        raise q1.Q1ExecutionError("synthetic unsupported filesystem")

    monkeypatch.setattr(q1, "probe_atomic_noreplace_publication", reject_probe)
    report, violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )

    assert not report.passed
    assert (
        "atomic no-replace publication probe failed:Q1ExecutionError:synthetic unsupported filesystem"
    ) in violations
    assert list(execution_root.iterdir()) == []


def test_resource_preflight_rejects_missing_samples_and_unsafe_registry(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o777)
    registry.chmod(0o777)

    report, violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples={},
    )
    assert not report.passed
    assert any(item.startswith("invalid or missing resource sample:") for item in violations)
    assert "attempt registry permissions must be exactly 0700" in violations


@pytest.mark.parametrize(
    ("directory_name", "expected_violation"),
    [
        ("execution", "execution root permissions must be exactly 0700"),
        ("registry", "attempt registry permissions must be exactly 0700"),
    ],
)
def test_resource_preflight_rejects_0755_directories(
    tmp_path: Path,
    directory_name: str,
    expected_violation: str,
) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)
    (execution_root if directory_name == "execution" else registry).chmod(0o755)

    report, violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )

    assert not report.passed
    assert expected_violation in violations


@pytest.mark.parametrize("target", ("execution", "registry"))
def test_resource_preflight_rejects_same_path_rename_recreate_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    from bench.active_acquisition import q1

    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)
    target_path = execution_root if target == "execution" else registry
    displaced = tmp_path / f"{target}-displaced"

    def replace_directory_at_same_path(_execution_root: Path) -> None:
        target_path.rename(displaced)
        target_path.mkdir(mode=0o700)

    monkeypatch.setattr(
        q1,
        "probe_atomic_noreplace_publication",
        replace_directory_at_same_path,
    )
    report, violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )

    label = "execution root" if target == "execution" else "attempt registry"
    assert not report.passed
    assert f"{label} identity changed during atomic publication probe" in violations
    bound_identity = report.execution_root if target == "execution" else report.attempt_registry_directory
    assert bound_identity is not None
    assert bound_identity.st_ino == displaced.stat().st_ino
    assert bound_identity.st_ino != target_path.stat().st_ino


def test_directory_identity_capture_rejects_path_replacement_during_descriptor_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "private"
    path.mkdir(mode=0o700)
    displaced = tmp_path / "displaced"
    original_inode = path.stat().st_ino
    real_fstat = os.fstat
    replaced = False

    def replace_after_descriptor_snapshot(descriptor: int) -> os.stat_result:
        nonlocal replaced
        metadata = real_fstat(descriptor)
        if metadata.st_ino == original_inode and not replaced:
            replaced = True
            path.rename(displaced)
            path.mkdir(mode=0o700)
        return metadata

    monkeypatch.setattr(qualification.os, "fstat", replace_after_descriptor_snapshot)
    with pytest.raises(ValueError, match="identity changed during its descriptor verification"):
        qualification._capture_private_directory_identity(path, label="private directory")
    assert replaced


@pytest.mark.parametrize(
    ("relationship", "expected_violation"),
    [
        ("same", "execution root and attempt registry must be distinct directories"),
        ("registry_inside_execution", "execution root and attempt registry must not be nested"),
        ("execution_inside_registry", "execution root and attempt registry must not be nested"),
    ],
)
def test_resource_preflight_rejects_shared_or_nested_authority_roots(
    tmp_path: Path,
    relationship: str,
    expected_violation: str,
) -> None:
    if relationship == "same":
        execution_root = tmp_path / "private"
        execution_root.mkdir(mode=0o700)
        registry = execution_root
    elif relationship == "registry_inside_execution":
        execution_root = tmp_path / "execution"
        execution_root.mkdir(mode=0o700)
        registry = execution_root / "registry"
        registry.mkdir(mode=0o700)
    else:
        registry = tmp_path / "registry"
        registry.mkdir(mode=0o700)
        execution_root = registry / "execution"
        execution_root.mkdir(mode=0o700)

    report, violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )

    assert not report.passed
    assert expected_violation in violations


def test_resource_preflight_rejects_non_owner_and_symlinked_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)
    effective_user = os.geteuid()
    monkeypatch.setattr(qualification.os, "geteuid", lambda: effective_user + 1)

    report, violations = _resource_preflight(
        execution_root=execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )
    assert not report.passed
    assert report.execution_root is None
    assert report.attempt_registry_directory is None
    assert "execution root must be owned by the effective user" in violations
    assert "attempt registry must be owned by the effective user" in violations

    monkeypatch.undo()
    linked_execution_root = tmp_path / "execution-link"
    linked_execution_root.symlink_to(execution_root, target_is_directory=True)
    linked_report, linked_violations = _resource_preflight(
        execution_root=linked_execution_root,
        attempt_registry_directory=registry,
        resource_samples=_resource_samples(),
    )
    assert not linked_report.passed
    assert linked_report.execution_root is None
    assert "execution root cannot be opened safely as a directory" in linked_violations


def test_protocol_code_parity_detects_arm_prior_budget_and_runtime_drift() -> None:
    protocol = copy.deepcopy(protocol_document())
    assert _protocol_code_parity_violations(protocol) == []

    arms = protocol["arms"]
    runtime = protocol["runtime"]
    budget = protocol["budget"]
    assert isinstance(arms, list)
    assert isinstance(runtime, dict)
    assert isinstance(budget, dict)
    assert isinstance(arms[0], dict)
    arms[0]["selection_unit"] = "nats"
    runtime["initial_model_sha256"] = "f" * 64
    runtime["max_restore_concurrency"] = 28
    process_watchdogs = runtime["process_watchdogs"]
    assert isinstance(process_watchdogs, dict)
    process_watchdogs["restore_child_timeout_seconds"] = 0.0
    budget["terminal_updates_total"] = 1
    violations = _protocol_code_parity_violations(protocol)
    assert "arm kind, unit, prior action, or shuffled-likelihood contract mismatch" in violations
    assert "canonical zero-evidence prior model mismatch" in violations
    assert "runtime contract mismatch:max_restore_concurrency" in violations
    assert "runtime contract mismatch:process_watchdogs" in violations
    assert "terminal update budget mismatch" in violations


def test_protocol_code_parity_rejects_entry_check_order_mutations() -> None:
    mutations = (
        list(reversed(ENTRY_CHECK_ORDER)),
        ["renamed_check", *ENTRY_CHECK_ORDER[1:]],
        list(ENTRY_CHECK_ORDER[:-1]),
        [*ENTRY_CHECK_ORDER, "extra_check"],
    )
    for mutated_order in mutations:
        protocol = copy.deepcopy(protocol_document())
        entry = protocol["entry_qualification"]
        assert isinstance(entry, dict)
        entry["check_order"] = mutated_order
        assert "entry qualification check order mismatch" in _protocol_code_parity_violations(protocol)


def _synthetic_directory_identity(canonical_path: str, *, st_ino: int) -> dict[str, object]:
    return {
        "canonical_path": canonical_path,
        "file_type": "directory",
        "mode": "0700",
        "st_dev": 1,
        "st_gid": os.getegid(),
        "st_ino": st_ino,
        "st_uid": os.geteuid(),
    }


def _entry_report(*, passed: bool) -> dict[str, object]:
    manifest, implementation_sha256 = implementation_manifest(Q1_IMPLEMENTATION_PATHS)
    checks = [
        {
            "name": name,
            "passed": passed,
            "summary": "Synthetic test check.",
            "violations": [] if passed else ["intentionally blocked"],
        }
        for name in ENTRY_CHECK_ORDER
    ]
    return {
        "schema": "prospect.wm002.active-acquisition.q1-entry-qualification.v1",
        "protocol_version": "0.3.0-q1",
        "protocol_sha256": sha256_bytes(Q1_PROTOCOL_PATH.read_bytes()),
        "q0_report_sha256": Q0_REPORT_SHA256,
        "q0_protocol_sha256": Q0_PROTOCOL_SHA256,
        "q0_implementation_sha256": Q0_IMPLEMENTATION_SHA256,
        "implementation_sha256": implementation_sha256 if passed else "",
        "implementation_manifest": [row.as_dict() for row in manifest] if passed else [],
        "dependency_versions": {"jsonschema": "4.25.1"},
        "schema_sha256": {name: sha256_bytes(path.read_bytes()) for name, path in sorted(Q1_SCHEMA_PATHS.items())},
        "salt_commitment_sha256": "a" * 64 if passed else "",
        "prospective_review_sha256": "b" * 64 if passed else "",
        "interpreter_identity": "test interpreter",
        "q1_environment_interactions": 0,
        "q1_private_draws": 0,
        "synthetic_development_interactions": 19 if passed else 0,
        "resource_preflight": {
            "execution_root": _synthetic_directory_identity("/private/execution", st_ino=2),
            "attempt_registry_directory": _synthetic_directory_identity(
                "/private/registry",
                st_ino=3,
            ),
            "raw_trace_max_bytes": 1 if passed else 0,
            "private_audit_max_bytes": 1 if passed else 0,
            "checkpoint_index_max_bytes": 1 if passed else 0,
            "checkpoint_frame_max_bytes": 1 if passed else 0,
            "restored_trace_max_bytes": 1 if passed else 0,
            "estimated_canonical_bytes": 1,
            "required_free_bytes": 1,
            "max_restore_concurrency": 4 if passed else 0,
            "sampled_arms": 7,
            "probe_duration_under_30_seconds": passed,
            "passed": passed,
        },
        "claim_eligible": False,
        "formal_authorized": False,
        "checks": checks,
        "passed": passed,
    }


@pytest.mark.parametrize("mutation", ("reordered", "renamed", "missing", "extra"))
def test_entry_report_rejects_check_name_and_order_mutations(mutation: str) -> None:
    value = _entry_report(passed=False)
    checks = value["checks"]
    assert isinstance(checks, list)
    if mutation == "reordered":
        checks[0], checks[1] = checks[1], checks[0]
    elif mutation == "renamed":
        first = checks[0]
        assert isinstance(first, dict)
        first["name"] = "renamed_check"
    elif mutation == "missing":
        checks.pop()
    else:
        checks.append(copy.deepcopy(checks[-1]))

    with pytest.raises(ValueError, match="canonical contract"):
        validate_entry_report(value)


def test_entry_report_resource_coherence_uses_named_canonical_check() -> None:
    value = _entry_report(passed=False)
    resources = value["resource_preflight"]
    assert isinstance(resources, dict)
    resources["passed"] = True

    with pytest.raises(ValueError, match="resource preflight pass value"):
        validate_entry_report(value)


def test_entry_report_directory_identity_schema_is_closed_and_pass_requires_bindings() -> None:
    failed_value = _entry_report(passed=False)
    failed_resources = failed_value["resource_preflight"]
    assert isinstance(failed_resources, dict)
    failed_resources["execution_root"] = None
    validate_entry_report(failed_value)

    passed_value = _entry_report(passed=True)
    passed_resources = passed_value["resource_preflight"]
    assert isinstance(passed_resources, dict)
    passed_resources["execution_root"] = None
    with pytest.raises(ValueError, match="lacks bound directory identities"):
        validate_entry_report(passed_value)

    extra_value = _entry_report(passed=False)
    extra_resources = extra_value["resource_preflight"]
    assert isinstance(extra_resources, dict)
    extra_identity = extra_resources["execution_root"]
    assert isinstance(extra_identity, dict)
    extra_identity["undeclared"] = True
    with pytest.raises(ValueError, match="schema violation"):
        validate_entry_report(extra_value)

    wrong_mode_value = _entry_report(passed=False)
    wrong_mode_resources = wrong_mode_value["resource_preflight"]
    assert isinstance(wrong_mode_resources, dict)
    wrong_mode_identity = wrong_mode_resources["execution_root"]
    assert isinstance(wrong_mode_identity, dict)
    wrong_mode_identity["mode"] = "0755"
    with pytest.raises(ValueError, match="schema violation"):
        validate_entry_report(wrong_mode_value)


def test_entry_report_rejects_shared_and_nested_directory_identity_bindings() -> None:
    shared_value = _entry_report(passed=True)
    shared_resources = shared_value["resource_preflight"]
    assert isinstance(shared_resources, dict)
    shared_resources["attempt_registry_directory"] = copy.deepcopy(shared_resources["execution_root"])
    with pytest.raises(ValueError, match="not distinct"):
        validate_entry_report(shared_value)

    nested_value = _entry_report(passed=True)
    nested_resources = nested_value["resource_preflight"]
    assert isinstance(nested_resources, dict)
    nested_registry = nested_resources["attempt_registry_directory"]
    assert isinstance(nested_registry, dict)
    nested_registry["canonical_path"] = "/private/execution/registry"
    with pytest.raises(ValueError, match="nested"):
        validate_entry_report(nested_value)


def test_entry_report_accepts_exact_bound_pass_and_rejects_manifest_drift() -> None:
    value = _entry_report(passed=True)
    validate_entry_report(value)

    manifest = value["implementation_manifest"]
    assert isinstance(manifest, list)
    assert isinstance(manifest[0], dict)
    manifest[0]["size_bytes"] += 1
    with pytest.raises(ValueError, match="manifest differs from selected source"):
        validate_entry_report(value)


def test_entry_report_accepts_failed_diagnostic_and_rejects_flag_incoherence() -> None:
    value = _entry_report(passed=False)
    validate_entry_report(value)

    checks = value["checks"]
    assert isinstance(checks, list)
    assert isinstance(checks[0], dict)
    checks[0]["passed"] = True
    with pytest.raises(ValueError, match="pass/violation mismatch"):
        validate_entry_report(value)


def test_entry_report_rejects_undeclared_private_field() -> None:
    value = _entry_report(passed=True)
    value["secret_salt"] = "forbidden"
    with pytest.raises(ValueError, match="entry report schema violation"):
        validate_entry_report(value)


def test_selected_source_closure_covers_fresh_process_repo_import_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    script = r"""
import json
import sys
from pathlib import Path

import bench.active_acquisition.q1
import bench.active_acquisition.q1_audit
import bench.active_acquisition.q1_qualification
import bench.active_acquisition.restore_worker

root = Path.cwd().resolve()
loaded = set()
for module in tuple(sys.modules.values()):
    filename = getattr(module, "__file__", None)
    if not filename:
        continue
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        continue
    if relative.suffix != ".py":
        continue
    if relative.parts[0] == "bench" or relative.parts[:2] == ("src", "prospect"):
        loaded.add(relative.as_posix())
print(json.dumps(sorted(loaded)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))
    selected = set(Q1_IMPLEMENTATION_PATHS)
    assert loaded <= selected, f"unbound imported sources: {sorted(loaded - selected)}"
