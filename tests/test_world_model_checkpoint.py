from __future__ import annotations

import hashlib
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bench.world_model_lifecycle.checkpoint import (
    CANONICAL_COMPONENT_IDS,
    OPAQUE_COMPONENT_IDS,
    ComponentPayload,
    RNGStateError,
    WMCheckpointError,
    canonical_json_bytes,
    load_checkpoint,
    manifest_schema_sha256,
    restore_numpy_generator,
    restore_numpy_rng,
    restore_python_rng,
    save_checkpoint,
    snapshot_numpy_generator,
    snapshot_numpy_rng,
    snapshot_python_rng,
)
from prospect.domain import TimePoint
from prospect.storage import CheckpointComponent, CheckpointCoordinator


def _components() -> dict[str, ComponentPayload]:
    return {
        component_id: ComponentPayload(
            component_id=component_id,
            logical_version=f"{component_id}-v1",
            payload=f"payload:{component_id}".encode(),
            media_type="application/x-wm001-fixture",
            predecessor_sha256=(
                None if component_id == "world_model" else hashlib.sha256(f"prior:{component_id}".encode()).hexdigest()
            ),
        )
        for component_id in CANONICAL_COMPONENT_IDS
    }


def test_component_complete_checkpoint_round_trips_and_reports_digests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wm001.prospect-checkpoint"
    components = _components()
    report = save_checkpoint(
        path,
        checkpoint_id="checkpoint-after-b",
        agent_id="wm001-agent",
        created_at=TimePoint(tick=1600, clock_id="interaction"),
        components=components,
        versions={"model": "post-b-v1"},
        metadata={"replicate_id": "formal-000"},
    )
    first_bytes = path.read_bytes()
    loaded = load_checkpoint(path, expected_agent_id="wm001-agent")

    assert loaded.report.manifest_sha256 == report.manifest_sha256
    assert tuple(row["component_id"] for row in report.component_rows()) == (CANONICAL_COMPONENT_IDS)
    assert [row["sha256"] for row in report.component_rows()] == [
        hashlib.sha256(components[component_id].payload).hexdigest() for component_id in CANONICAL_COMPONENT_IDS
    ]
    assert loaded.payload("world_model") == b"payload:world_model"
    assert len(manifest_schema_sha256()) == 64

    save_checkpoint(
        path,
        checkpoint_id="checkpoint-after-b",
        agent_id="wm001-agent",
        created_at=TimePoint(tick=1600, clock_id="interaction"),
        components=components,
        versions={"model": "post-b-v1"},
        metadata={"replicate_id": "formal-000"},
    )
    assert path.read_bytes() == first_bytes


def test_save_and_restore_reject_missing_or_extra_components_before_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wm001.prospect-checkpoint"
    components = _components()
    del components["optimizer"]
    with pytest.raises(WMCheckpointError, match="missing=.*optimizer"):
        save_checkpoint(
            path,
            checkpoint_id="incomplete",
            agent_id="wm001-agent",
            created_at=TimePoint(tick=1),
            components=components,
        )
    assert not path.exists()

    complete = _components()
    save_checkpoint(
        path,
        checkpoint_id="complete",
        agent_id="wm001-agent",
        created_at=TimePoint(tick=2),
        components=complete,
    )
    loaded = load_checkpoint(path)
    called: list[str] = []
    restorers = {
        component_id: (lambda _payload, record: called.append(record.component_id))
        for component_id in CANONICAL_COMPONENT_IDS
        if component_id != "planner_rng"
    }
    with pytest.raises(WMCheckpointError, match="missing=.*planner_rng"):
        loaded.restore(restorers)
    assert called == []

    extra_restorers = {component_id: (lambda _payload, _record: None) for component_id in CANONICAL_COMPONENT_IDS}
    extra_restorers["unknown_state"] = lambda _payload, _record: None
    with pytest.raises(WMCheckpointError, match="extra=.*unknown_state"):
        loaded.restore(extra_restorers)


def test_load_rejects_noncanonical_archive_component_set(tmp_path: Path) -> None:
    path = tmp_path / "generic.prospect-checkpoint"
    CheckpointCoordinator().save(
        path,
        checkpoint_id="wrong-shape",
        agent_id="wm001-agent",
        created_at=TimePoint(tick=1),
        components={
            "world_model": CheckpointComponent(
                name="world_model",
                version="v1",
                payload=b"model",
            )
        },
        versions={"model": "v1"},
    )
    called: list[str] = []
    with pytest.raises(WMCheckpointError, match="archive components.*missing"):
        loaded = load_checkpoint(path)
        loaded.restore({"world_model": lambda _payload, _record: called.append("called")})
    assert called == []


def test_python_and_numpy_rng_codecs_restore_exact_streams() -> None:
    python_rng = random.Random(101)
    python_payload = snapshot_python_rng(python_rng)
    expected_python = [python_rng.random() for _ in range(8)]
    restore_python_rng(python_payload, python_rng)
    assert [python_rng.random() for _ in range(8)] == expected_python

    np.random.seed(202)
    numpy_payload = snapshot_numpy_rng()
    expected_numpy = np.random.standard_normal(8)
    restore_numpy_rng(numpy_payload)
    np.testing.assert_array_equal(np.random.standard_normal(8), expected_numpy)

    generator = np.random.default_rng(303)
    generator_payload = snapshot_numpy_generator(generator)
    expected_generator = generator.integers(0, 2**31, size=8)
    restore_numpy_generator(generator_payload, generator)
    np.testing.assert_array_equal(
        generator.integers(0, 2**31, size=8),
        expected_generator,
    )


def test_rng_codecs_reject_noncanonical_or_incompatible_payloads() -> None:
    payload = snapshot_python_rng(random.Random(1))
    noncanonical = b'{ "schema": "not-canonical" }'
    with pytest.raises(RNGStateError, match="not canonical JSON"):
        restore_python_rng(noncanonical, random.Random())

    value = {
        "gaussian_cache": None,
        "internal_state": [1],
        "python_version": "0.0.0",
        "random_state_version": 3,
        "schema": "prospect.rng.python",
        "schema_version": 1,
    }
    with pytest.raises(RNGStateError, match="different Python version"):
        restore_python_rng(canonical_json_bytes(value), random.Random())

    assert payload == snapshot_python_rng(random.Random(1))


def test_bundle_can_be_loaded_by_a_fresh_interpreter(tmp_path: Path) -> None:
    path = tmp_path / "fresh-process.prospect-checkpoint"
    report = save_checkpoint(
        path,
        checkpoint_id="fresh-process",
        agent_id="wm001-agent",
        created_at=TimePoint(tick=99),
        components=_components(),
    )
    script = (
        "from bench.world_model_lifecycle.checkpoint import load_checkpoint;"
        f"c=load_checkpoint({str(path)!r},expected_agent_id='wm001-agent');"
        "print(c.report.manifest_sha256);"
        "print(','.join(k for k,_ in c.payloads))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    assert lines == [
        report.manifest_sha256,
        ",".join(CANONICAL_COMPONENT_IDS),
    ]


def test_opaque_component_constant_is_exact_complement() -> None:
    assert set(OPAQUE_COMPONENT_IDS) == set(CANONICAL_COMPONENT_IDS) - {
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "torch_accelerator_rng",
        "collection_rng",
        "planner_rng",
    }
