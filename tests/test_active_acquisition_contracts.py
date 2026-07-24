from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

import bench.active_acquisition.contracts as contracts
from bench.active_acquisition.contracts import (
    ACTION_ORDER,
    ARM_ORDER,
    CHECKPOINT_COMPONENTS,
    Q0_IMPLEMENTATION_SHA256,
    Q0_PROTOCOL_SHA256,
    Q0_REPORT_SHA256,
    Q1_PROTOCOL_PATH,
    Q1_PROTOCOL_VERSION,
    Q1_SCHEMA_PATHS,
    TERMINAL_ORDER,
    ContractError,
    assert_no_sentinel_bytes,
    assert_public_value_safe,
    canonical_json_bytes,
    implementation_manifest,
    protocol_document,
    schema_documents,
)


def test_q1_protocol_freezes_runtime_identity_and_q0_binding() -> None:
    protocol = protocol_document()
    experiment = protocol["experiment"]
    q0 = protocol["q0_binding"]
    assert isinstance(experiment, dict)
    assert isinstance(q0, dict)
    assert experiment["protocol_version"] == Q1_PROTOCOL_VERSION
    assert experiment["claim_eligible"] is False
    assert experiment["formal_authorized"] is False
    assert experiment["execution_authorized"] is False
    assert q0["report_sha256"] == Q0_REPORT_SHA256
    assert q0["protocol_sha256"] == Q0_PROTOCOL_SHA256
    assert q0["implementation_sha256"] == Q0_IMPLEMENTATION_SHA256


def test_bound_q0_digests_still_regenerate_from_the_current_sources() -> None:
    """A bound Q0 report that no longer reproduces is stale evidence, not evidence.

    The originally accepted digests silently stopped reproducing when two files
    inside the Q0 selected-source manifest changed. Regenerating Q0 here fails
    the moment that happens again.
    """

    from bench.active_acquisition.qualification import PROTOCOL_PATH, run_qualification

    report = run_qualification(PROTOCOL_PATH)
    payload = canonical_json_bytes(report.as_dict(), newline=True)
    assert report.passed is True
    assert report.protocol_sha256 == Q0_PROTOCOL_SHA256
    assert report.implementation_sha256 == Q0_IMPLEMENTATION_SHA256
    assert contracts.sha256_bytes(payload) == Q0_REPORT_SHA256


def test_protocol_document_rejects_nested_duplicate_keys_without_echo(tmp_path: Path) -> None:
    private_key = "recognizable-private-duplicate-key"
    path = tmp_path / "duplicate-protocol.json"
    path.write_text(
        (
            "{\"schema\":\"prospect.wm002.active-acquisition.q1-protocol.v1\","
            "\"experiment\":{\"protocol_version\":\"0.3.0-q1\","
            f"\"{private_key}\":1,\"{private_key}\":2}}}}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="cannot decode JSON document") as captured:
        protocol_document(path)
    assert private_key not in str(captured.value)
    assert str(captured.value.__cause__) == "duplicate JSON object key is forbidden"


def test_q1_protocol_freezes_orders_arms_components_and_schemas() -> None:
    protocol = protocol_document()
    candidates = protocol["candidates"]
    runtime = protocol["runtime"]
    schemas = protocol["schemas"]
    assert isinstance(candidates, dict)
    assert isinstance(runtime, dict)
    assert isinstance(schemas, dict)
    arms = protocol["arms"]
    candidate_rows = candidates["acquisition_order"]
    assert isinstance(arms, list)
    assert isinstance(candidate_rows, list)
    assert tuple(row["semantic_id"] for row in candidate_rows if isinstance(row, dict)) == ACTION_ORDER
    assert tuple(candidates["terminal_order"]) == TERMINAL_ORDER
    assert tuple(row["id"] for row in arms if isinstance(row, dict)) == ARM_ORDER
    assert tuple(runtime["checkpoint_components"]) == CHECKPOINT_COMPONENTS
    assert set(schemas.values()) == {f"schemas/{path.name}" for path in Q1_SCHEMA_PATHS.values()}


def test_all_q1_artifact_schemas_are_strict_and_metaschema_valid() -> None:
    schemas = schema_documents()
    assert set(schemas) == set(Q1_SCHEMA_PATHS)
    for schema in schemas.values():
        assert schema["additionalProperties"] is False


def test_schema_documents_and_compiled_validators_are_cached_per_process() -> None:
    assert contracts._schema_documents_cached() is contracts._schema_documents_cached()
    assert contracts._compiled_validator("raw_trace") is contracts._compiled_validator("raw_trace")


def test_q1_schema_mutation_cannot_silently_add_a_public_field() -> None:
    from jsonschema import Draft202012Validator

    schema = schema_documents()["checkpoint_frame"]
    valid = {
        "schema": "prospect.wm002.active-acquisition.q1-checkpoint-frame.v1",
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": "wm002-q1-" + "0" * 64,
        "attempt_id": "wm002-q1-" + "0" * 64 + "-attempt-0001",
        "entry_qualification_sha256": "2" * 64,
        "master": 0,
        "arm": ARM_ORDER[0],
        "episode": 0,
        "frame_offset": 0,
        "frame_header_bytes": 8,
        "frame_length": 1,
        "checkpoint_sha256": "0" * 64,
        "component_sha256": {name: "1" * 64 for name in CHECKPOINT_COMPONENTS},
    }
    Draft202012Validator(schema).validate(valid)
    mutated = copy.deepcopy(valid)
    mutated["theta"] = 1
    errors = tuple(Draft202012Validator(schema).iter_errors(mutated))
    assert errors


def test_permissioned_private_sidecar_matches_frozen_schema() -> None:
    from jsonschema import Draft202012Validator

    from bench.active_acquisition.seeding import PrivateQ1SeedSchedule

    sidecar = PrivateQ1SeedSchedule(bytes(range(32))).private_audit_material(
        0,
        0,
        ARM_ORDER[0],
    )
    private_row = {
        "run_id": "wm002-q1-" + "0" * 64,
        "attempt_id": "wm002-q1-" + "0" * 64 + "-attempt-0001",
        **sidecar.as_private_dict(),
    }
    Draft202012Validator(schema_documents()["private_audit"]).validate(private_row)


def test_public_recursive_scanner_rejects_private_keys_and_values() -> None:
    assert_public_value_safe(
        {
            "salt_commitment": "a" * 64,
            "semantic_key_sha256": "b" * 64,
            "observation": {"observed_symbol": 1},
        },
        private_sentinels=("PRIVATE-SENTINEL",),
    )
    with pytest.raises(ContractError, match="private field name"):
        assert_public_value_safe({"theta": 1})
    with pytest.raises(ContractError, match="private sentinel"):
        assert_public_value_safe(
            {"observed_symbol": "PRIVATE-SENTINEL"},
            private_sentinels=("PRIVATE-SENTINEL",),
        )


def test_opaque_serialization_scanner_checks_raw_and_hex_private_bytes() -> None:
    secret = bytes.fromhex("cafe" * 8)
    assert_no_sentinel_bytes((b"public",), private_sentinels=(secret,))
    with pytest.raises(ContractError, match="private sentinel"):
        assert_no_sentinel_bytes((secret.hex().encode(),), private_sentinels=(secret,))


def test_selected_source_manifest_is_sorted_and_content_bound(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    rows, digest = implementation_manifest(("b.py", "a.py"), repository_root=tmp_path)
    assert tuple(row.relative_path for row in rows) == ("a.py", "b.py")
    assert tuple(row.sha256 for row in rows) == (
        "87428fc522803d31065e7bce3cf03fe475096631e5e07bbd7a0fde60c4cf25c7",
        "0263829989b6fd954f72baaf2fc64bc2e2f01d692d4de72986ea808f6e99813f",
    )
    assert digest == "1f94727401d294edeab0124bf4ca436082438780df4c21c38c2c8b6d34bb1b3e"
    second.write_text("changed\n", encoding="utf-8")
    _, changed = implementation_manifest(("b.py", "a.py"), repository_root=tmp_path)
    assert changed != digest


def test_selected_source_manifest_rejects_a_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("safe\n", encoding="utf-8")
    (tmp_path / "selected.py").symlink_to(target)

    with pytest.raises(ContractError, match="cannot safely read implementation member"):
        implementation_manifest(("selected.py",), repository_root=tmp_path)


def test_selected_source_manifest_rejects_a_symlinked_intermediate_directory(tmp_path: Path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    (real_directory / "selected.py").write_text("safe\n", encoding="utf-8")
    (tmp_path / "linked").symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ContractError, match="cannot safely read implementation member"):
        implementation_manifest(("linked/selected.py",), repository_root=tmp_path)


def test_selected_source_manifest_rejects_signature_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected.py"
    selected.write_text("before\n", encoding="utf-8")
    real_read = contracts._read_bounded_descriptor

    def mutate_after_read(descriptor: int, *, limit: int, label: str) -> bytes:
        payload = real_read(descriptor, limit=limit, label=label)
        selected.write_text("after-mutation\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(contracts, "_read_bounded_descriptor", mutate_after_read)
    with pytest.raises(ContractError, match="changed during its descriptor read"):
        implementation_manifest(("selected.py",), repository_root=tmp_path)


def test_selected_source_manifest_rejects_path_to_descriptor_signature_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "selected.py").write_text("selected\n", encoding="utf-8")
    (tmp_path / "replacement.py").write_text("replacement\n", encoding="utf-8")
    real_stat = os.stat
    selected_stat_calls = 0

    def substitute_second_path_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes] | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal selected_stat_calls
        if path == "selected.py" and dir_fd is not None:
            selected_stat_calls += 1
            if selected_stat_calls == 2:
                return real_stat(
                    "replacement.py",
                    dir_fd=dir_fd,
                    follow_symlinks=False,
                )
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(contracts.os, "stat", substitute_second_path_stat)
    with pytest.raises(ContractError, match="path-to-descriptor signature changed"):
        implementation_manifest(("selected.py",), repository_root=tmp_path)
    assert selected_stat_calls == 2


def test_selected_source_manifest_rejects_nonregular_member(tmp_path: Path) -> None:
    (tmp_path / "selected.py").mkdir()

    with pytest.raises(ContractError, match="regular non-symlink file"):
        implementation_manifest(("selected.py",), repository_root=tmp_path)


def test_selected_source_manifest_rejects_hard_linked_member(tmp_path: Path) -> None:
    selected = tmp_path / "selected.py"
    selected.write_text("safe\n", encoding="utf-8")
    os.link(selected, tmp_path / "alias.py")

    with pytest.raises(ContractError, match="exactly one hard link"):
        implementation_manifest(("selected.py",), repository_root=tmp_path)


def test_selected_source_manifest_enforces_a_bounded_full_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "selected.py").write_bytes(b"12345")
    monkeypatch.setattr(contracts, "_MAX_IMPLEMENTATION_MEMBER_BYTES", 4)

    with pytest.raises(ContractError, match="4-byte read limit"):
        implementation_manifest(("selected.py",), repository_root=tmp_path)


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_protocol_is_valid_finite_json() -> None:
    decoded = json.loads(Q1_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert decoded["schema"] == "prospect.wm002.active-acquisition.q1-protocol.v1"
