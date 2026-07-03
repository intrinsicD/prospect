"""Smoke tests: the scaffold imports and the contracts are wired. These stay green
from commit one; real behaviour tests arrive with each task."""
from __future__ import annotations

import bench
import prospect
from prospect import interfaces, types
from prospect.codec import UniversalCodec
from prospect.knowledge import ExternalKnowledgeSource, InternalKnowledgeSource, ToolSource
from prospect.memory import ReplayBuffer, SemanticStore, UncertaintyMemoryRouter
from prospect.planning import FlatPlanner, HierarchicalManager, JumpyOptionModel
from prospect.skills import SkillRouter
from prospect.voe import SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel


def test_package_version() -> None:
    assert prospect.__version__


def test_types_instantiate() -> None:
    s = types.LatentState(z=[0.0])
    a = types.Action(data=[0.0])
    p = types.Provenance(source="unit-test", trust=types.Trust.LOW)
    assert s.z == [0.0]
    assert a.data == [0.0]
    assert p.trust is types.Trust.LOW


def test_skeletons_satisfy_protocols() -> None:
    # runtime_checkable protocols verify method presence (structural typing).
    assert isinstance(FlatWorldModel(), interfaces.WorldModel)
    assert isinstance(FlatWorldModel(), interfaces.Learner)  # the training seam (P0-003)
    assert isinstance(FlatPlanner(), interfaces.Planner)
    assert isinstance(SurpriseCompetenceMonitor(), interfaces.CompetenceMonitor)
    assert isinstance(SkillRouter(), interfaces.SkillLibrary)
    assert isinstance(ReplayBuffer(), interfaces.EpisodicMemory)


def test_all_skeletons_instantiate() -> None:
    for cls in (
        UniversalCodec,
        FlatWorldModel,
        FlatPlanner,
        JumpyOptionModel,
        HierarchicalManager,
        SurpriseCompetenceMonitor,
        SkillRouter,
        ReplayBuffer,
        SemanticStore,
        UncertaintyMemoryRouter,
        InternalKnowledgeSource,
        ExternalKnowledgeSource,
        ToolSource,
    ):
        assert cls() is not None


def test_all_gates_registered() -> None:
    assert set(bench.GATES) == {"P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}
    for gate in bench.GATES.values():
        assert gate.criterion  # every gate has a precise criterion


def test_all_sentinels_registered() -> None:
    expected = {
        "representation-integrity",
        "uncertainty-reliability",
        "replay-fidelity",
        "option-diversity",
    }
    assert set(bench.SENTINELS) == expected
    for sentinel in bench.SENTINELS.values():
        assert sentinel.criterion
        assert sentinel.detects
        assert sentinel.applies_from in bench.gates.PHASE_ORDER


def test_phase_gate_is_composite_and_pending(tmp_path) -> None:
    report = bench.run_gate("P5", results_dir=tmp_path)
    assert isinstance(report, bench.GateReport)
    # pending capability + pending sentinels => phase is not passable yet
    assert report.passed is False
    assert "PENDING" in report.capability.detail
    names = {s.name for s in report.sentinels}
    # by P5, the P1 and P3 sentinels are still active, plus the P5 option sentinel
    assert {"representation-integrity", "uncertainty-reliability", "replay-fidelity",
            "option-diversity"} <= names


def test_sentinels_activate_by_phase(tmp_path) -> None:
    p1 = {s.name for s in bench.run_gate("P1", results_dir=tmp_path).sentinels}
    assert "representation-integrity" in p1
    assert "uncertainty-reliability" in p1
    assert "replay-fidelity" not in p1  # generative replay arrives at P3
    assert "option-diversity" not in p1  # options arrive at P5
