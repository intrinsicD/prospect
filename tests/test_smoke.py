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
    assert set(bench.GATES) == {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}
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


def test_all_floors_registered() -> None:
    expected = {"quality-floor"}
    assert set(bench.FLOORS) == expected
    for floor in bench.FLOORS.values():
        assert floor.criterion
        assert floor.guards
        assert floor.applies_from in bench.gates.PHASE_ORDER


def test_phase_gate_is_composite_and_pending() -> None:
    report = bench.run_gate("P5")
    assert isinstance(report, bench.GateReport)
    # pending capability + pending sentinels + pending floor => not passable yet
    assert report.passed is False
    assert "PENDING" in report.capability.detail
    names = {s.name for s in report.sentinels}
    # by P5, the P1 and P3 sentinels are still active, plus the P5 option sentinel
    assert {"representation-integrity", "uncertainty-reliability", "replay-fidelity",
            "option-diversity"} <= names
    # the quality floor (ADR-0007) is the third leg of the composite, active from P1
    floor_names = {f.name for f in report.floors}
    assert "quality-floor" in floor_names
    assert all(not f.satisfied for f in report.floors)


def test_sentinels_activate_by_phase() -> None:
    p1 = {s.name for s in bench.run_gate("P1").sentinels}
    assert "representation-integrity" in p1
    assert "uncertainty-reliability" in p1
    assert "replay-fidelity" not in p1  # generative replay arrives at P3
    assert "option-diversity" not in p1  # options arrive at P5


def test_quality_floor_active_from_p1() -> None:
    # the floor goes down before the system starts changing itself (ADR-0007)
    p1_floors = {f.name for f in bench.run_gate("P1").floors}
    assert "quality-floor" in p1_floors
