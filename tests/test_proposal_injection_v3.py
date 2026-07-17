from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest

from bench.proposal_injection import experiment as base
from bench.proposal_injection_v3 import experiment as v3


def test_failed_predecessor_is_hash_bound_and_json_equivalent() -> None:
    record = v3._failed_predecessor()
    assert record["artifact_sha256"] == v3.FAILED_HASHES
    assert record["artifact_manifest_entries_verified"] is True
    assert record["source_snapshot_matches_protocol"] is True
    assert record["raw_container_equality"] is False
    assert record["canonical_json_equality"] is True


def test_canonical_json_value_normalizes_containers_but_not_content() -> None:
    left = {"scores": (1.0, 2.0), "nested": {"b": True, "a": None}}
    right = {"nested": {"a": None, "b": True}, "scores": [1.0, 2.0]}
    assert v3._canonical_json_value(left) == right
    assert v3._canonical_json_value({"scores": (1.0, 3.0)}) != right
    with pytest.raises(ValueError):
        v3._canonical_json_value({"bad": math.nan})


def test_protocol_inherits_all_scientific_fields() -> None:
    with v3._configured():
        protocol = v3._protocol_record_v3()
    failed = base._read_json(base.REPO_ROOT / v3.FAILED_OUTPUT / "protocol.json")
    assert {field: protocol[field] for field in v3.SCIENTIFIC_FIELDS} == {
        field: failed[field] for field in v3.SCIENTIFIC_FIELDS
    }
    delta = cast(dict[str, object], protocol["method_delta"])
    assert delta["scientific_changes"] == []


def test_prepare_freezes_pi003_and_protects_predecessors(tmp_path: Path) -> None:
    output = tmp_path / "PI-003"
    prepared = v3.prepare(output)
    assert prepared["status"] == "prepared_only"
    assert prepared["outcomes"] == "prepared_only"
    with pytest.raises(ValueError, match="preserved predecessor"):
        v3.prepare(base.REPO_ROOT / v3.FAILED_OUTPUT)
