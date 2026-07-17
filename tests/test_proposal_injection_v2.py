from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from bench.proposal_injection import experiment as base
from bench.proposal_injection_v2 import experiment as v2


def test_failed_predecessor_is_hash_bound_and_order_only() -> None:
    record = v2._failed_predecessor()
    assert record["artifact_sha256"] == v2.FAILED_HASHES
    assert record["artifact_manifest_entries_verified"] is True
    assert record["source_snapshot_matches_protocol"] is True
    assert record["saved_report_matches_pre_serialization_order"] is True
    assert record["saved_report_matches_post_serialization_order"] is False
    assert record["first_difference_character"] == 522


def test_report_delta_is_order_canonical_and_identifier_only() -> None:
    failed = base._read_json(base.REPO_ROOT / v2.FAILED_OUTPUT / "PI-001-results.json")
    decision = dict(cast(dict[str, Any], failed["decision"]))
    rescues = cast(dict[str, Any], decision["rescues"])
    decision["rescues"] = {
        "privileged_injection": rescues["privileged_injection"],
        "action_permuted_injection": rescues["action_permuted_injection"],
    }
    insertion_order = dict(failed)
    insertion_order["decision"] = decision
    serialized = cast(
        dict[str, Any],
        json.loads(json.dumps(insertion_order, sort_keys=True)),
    )
    assert v2._report_text_v2(insertion_order) == v2._report_text_v2(serialized)
    assert v2._report_text_v2(serialized).startswith("# PI-002 proposal-injection result")


def test_protocol_inherits_all_scientific_fields() -> None:
    with v2._configured():
        protocol = v2._protocol_record_v2()
    failed = base._read_json(base.REPO_ROOT / v2.FAILED_OUTPUT / "protocol.json")
    assert {field: protocol[field] for field in v2.SCIENTIFIC_FIELDS} == {
        field: failed[field] for field in v2.SCIENTIFIC_FIELDS
    }
    delta = cast(dict[str, object], protocol["method_delta"])
    assert delta["scientific_changes"] == []


def test_prepare_freezes_pi002_and_protects_pi001(tmp_path: Path) -> None:
    output = tmp_path / "PI-002"
    prepared = v2.prepare(output)
    assert prepared["status"] == "prepared_only"
    assert prepared["outcomes"] == "prepared_only"
    with pytest.raises(ValueError, match="preserved PI-001"):
        v2.prepare(base.REPO_ROOT / v2.FAILED_OUTPUT)
