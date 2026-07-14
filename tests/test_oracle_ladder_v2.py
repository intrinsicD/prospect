from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import pytest

from bench.oracle_ladder import experiment as base
from bench.oracle_ladder_v2 import experiment as v2
from bench.oracle_ladder_v2.experiment import (
    EXPERIMENT_ID,
    FAILED_HASHES,
    SCHEMA_VERSION,
    _configured,
    _csv_text_v2,
    _failed_predecessor,
    _protocol_record_v2,
    prepare,
    verify,
)


def test_failed_predecessor_is_hash_bound_and_newline_only() -> None:
    record = _failed_predecessor()
    assert record["artifact_sha256"] == FAILED_HASHES
    assert record["artifact_manifest_entries_verified"] is True
    assert record["source_snapshot_matches_protocol"] is True
    assert record["canonical_csv_bytes_match"] is True
    assert record["csv_crlf_count"] == 89
    assert record["normalized_text_matches_crlf_canonical"] is False


def test_namespace_context_is_scoped_and_restored() -> None:
    original_id = base.EXPERIMENT_ID
    original_schema = base.SCHEMA_VERSION
    with _configured():
        assert base.EXPERIMENT_ID == EXPERIMENT_ID
        assert base.SCHEMA_VERSION == SCHEMA_VERSION
    assert base.EXPERIMENT_ID == original_id
    assert base.SCHEMA_VERSION == original_schema


def test_protocol_enforces_runtime_and_ol002_stop_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    with _configured():
        protocol = _protocol_record_v2()
    runtime = cast(dict[str, str], protocol["runtime_constraints"])
    stop_rules = cast(list[str], protocol["stop_rules"])
    assert runtime["numpy_version"] == "2.4.6"
    assert all("OL-001" not in rule for rule in stop_rules)
    monkeypatch.setattr(base.np, "__version__", "0.0.0")
    with pytest.raises(ValueError, match="requires the OL-001 NumPy runtime 2.4.6"):
        _failed_predecessor()


def test_failed_input_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_failure = tmp_path / "failed-OL-001"
    shutil.copytree(base.REPO_ROOT / v2.FAILED_OUTPUT, copied_failure)
    monkeypatch.setattr(v2, "FAILED_OUTPUT", copied_failure)
    copied_input = copied_failure / "inputs/BC-001-b1_r1_d8.npz"
    payload = bytearray(copied_input.read_bytes())
    payload[-1] ^= 1
    copied_input.write_bytes(payload)
    with pytest.raises(ValueError, match="failed artifacts have drifted"):
        _failed_predecessor()


def test_csv_delta_is_only_crlf_to_lf() -> None:
    rows = [
        {
            "rung": "example",
            "seed": 0,
            "model_sha256": "0" * 64,
            "mean_eval_return": 1.0,
            "success_rate": 0.5,
            "planner_candidate_transition_count": 1,
            "oracle_candidate_transition_count": 0,
        }
    ]
    original = base._csv_text(rows)
    revised = _csv_text_v2(rows)
    assert revised == original.replace("\r\n", "\n")
    assert "\r" not in revised


def test_prepare_freezes_ol002_and_strict_verify_rejects_prepared_only(tmp_path: Path) -> None:
    output = tmp_path / "OL-002"
    prepared = prepare(output)
    assert prepared["status"] == "prepared_only"
    assert prepared["outcomes"] == "prepared_only"
    assert verify(output)["outcomes"] == "prepared_only"
    with pytest.raises(ValueError, match="complete verified"):
        verify(output, require_results=True)
    protocol = base._read_json(output / "protocol.json")
    assert protocol["experiment_id"] == EXPERIMENT_ID
    assert protocol["schema_version"] == SCHEMA_VERSION
    assert protocol["failed_predecessor"]["artifact_sha256"] == FAILED_HASHES
