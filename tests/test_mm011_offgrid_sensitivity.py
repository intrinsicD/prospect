from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from bench.multimodal_causal_assay import offgrid_sensitivity as audit
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry

PROTOCOL_SHA256 = "07a29f4d4f22dc619ef3f28250c81dabbee8dab114a701326483bb4d47f144cd"
RECEIPT = Path("docs/research/2026-07-16-mm011-offgrid-sensitivity-audit.json")
SEMANTIC = Path(
    "docs/research/2026-07-16-mm011-offgrid-sensitivity-semantic-verification.json"
)


def test_frozen_offgrid_config_is_exact_and_source_bound() -> None:
    config = audit.config_record()

    assert config["activity_sse_min"] == 0.6912
    assert config["primary_factor"] == 1.25
    assert config["null_sse_max"] == 1e-24
    assert config["required_positive_passes"] == 6
    assert [item[1] for item in audit.CASE_SPECS] == ["T1", "T2", "T3", "A1", "A2", "A3"]
    source_hashes = config["source_sha256"]
    assert isinstance(source_hashes, dict)
    assert set(source_hashes) == {str(path) for path in audit.SOURCE_PATHS}
    assert all(isinstance(value, str) and len(value) == 64 for value in source_hashes.values())
    for _seed, _name, theta in audit.CASE_SPECS:
        with pytest.raises(geometry.GeometryValidationError, match="not an exact member"):
            geometry.state_index(theta)


def test_halo_fixture_is_deterministic_broadband_and_does_not_read_real_frames() -> None:
    first_frame, first_metrics, first_halo = audit._fixture(991_100)  # noqa: SLF001
    replay_frame, replay_metrics, replay_halo = audit._fixture(991_100)  # noqa: SLF001

    assert first_frame.shape == (3, 64, 64)
    assert first_halo.shape == (3, 96, 96)
    assert first_metrics.failure_reasons() == ()
    assert replay_metrics == first_metrics
    np.testing.assert_array_equal(replay_frame, first_frame)
    np.testing.assert_array_equal(replay_halo, first_halo)
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "lcv_parent" not in source
    assert "MM-007-frames-64x64.npz" not in source


def test_every_declared_continuous_fixture_is_boundary_safe() -> None:
    full_y, full_x = audit._full_coordinates()  # noqa: SLF001
    central_y, central_x = audit._central_coordinates()  # noqa: SLF001
    fixtures = {seed: audit._fixture(seed) for seed in (991_100, 991_101, 991_102)}  # noqa: SLF001

    for seed, _name, theta in audit.CASE_SPECS:
        _previous, _metrics, halo = fixtures[seed]
        current = audit._sample_continuous(  # noqa: SLF001
            halo,
            theta,
            full_y,
            full_x,
            source_offset=audit.HALO_OFFSET,
        ).reshape(3, 64, 64)
        future = audit._sample_continuous(  # noqa: SLF001
            current,
            theta,
            central_y,
            central_x,
            source_offset=0,
        )
        assert current.shape == (3, 64, 64)
        assert future.shape == (3, 2304)
        assert np.all(np.isfinite(current))
        assert np.all(np.isfinite(future))


def test_durable_offgrid_result_is_bound_and_rejects_mutation() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    validated = audit.validate_receipt(value, protocol_sha256=PROTOCOL_SHA256)

    assert validated["decision"] == "ABANDON_FINITE_GRID_BEFORE_REAL_DATA"
    assert validated["failure_codes"] == ["positive:T1", "positive:T2", "positive:T3"]
    positive_cases = cast(list[dict[str, audit.JsonValue]], validated["positive_cases"])
    controls = cast(list[dict[str, audit.JsonValue]], validated["controls"])
    assert sum(item["passed"] is True for item in positive_cases) == 3
    assert all(item["passed"] is True for item in controls)
    mutated = deepcopy(value)
    mutated["positive_cases"][0]["passed"] = True
    with pytest.raises(audit.OffgridSensitivityError, match="digest differs"):
        audit.validate_receipt(mutated, protocol_sha256=PROTOCOL_SHA256)


def test_semantic_receipt_binds_bit_exact_replay() -> None:
    result = json.loads(RECEIPT.read_text(encoding="utf-8"))
    semantic = json.loads(SEMANTIC.read_text(encoding="utf-8"))

    assert semantic["audit_file"]["sha256"] == hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    assert semantic["bit_exact_replay"] is True
    assert semantic["decision"] == result["decision"]
    assert semantic["evidence_sha256"] == result["evidence_sha256"]
