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
    assert isinstance(FlatPlanner(FlatWorldModel()), interfaces.Planner)
    assert isinstance(SurpriseCompetenceMonitor(), interfaces.CompetenceMonitor)
    assert isinstance(SkillRouter(), interfaces.SkillLibrary)
    assert isinstance(ReplayBuffer(), interfaces.EpisodicMemory)
    # one query verb into every knowledge tier (P0-008)
    assert isinstance(SemanticStore(), interfaces.SemanticMemory)
    assert isinstance(SemanticStore(), interfaces.KnowledgeSource)
    assert isinstance(InternalKnowledgeSource(), interfaces.KnowledgeSource)
    assert isinstance(UncertaintyMemoryRouter(), interfaces.MemoryRouter)


def test_all_skeletons_instantiate() -> None:
    for factory in (
        UniversalCodec,
        FlatWorldModel,
        lambda: FlatPlanner(FlatWorldModel()),  # planners plan over a world model
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
        assert factory() is not None


def test_all_gates_registered() -> None:
    assert set(bench.GATES) == {"P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"}
    for gate in bench.GATES.values():
        assert gate.criterion  # every gate has a precise criterion


def test_all_sentinels_registered() -> None:
    expected = {
        "representation-integrity",
        "uncertainty-reliability",
        "replay-fidelity",
        "option-diversity",
        "gate-overfit",
    }
    assert set(bench.SENTINELS) == expected
    for sentinel in bench.SENTINELS.values():
        assert sentinel.criterion
        assert sentinel.detects
        assert sentinel.applies_from in bench.gates.PHASE_ORDER


def test_phase_gate_is_composite_and_capability_gated() -> None:
    # The composite invariant: a phase passes only if its capability passes AND
    # every applicable sentinel is healthy — a not-met capability BLOCKS the phase
    # no matter how healthy the sentinels are. Every phase now ships a real,
    # model-training capability eval, so this is built synthetically to stay a cheap
    # structural check (running P8's live eval trains three models).
    not_met = bench.GateResult(phase="P8", passed=False, detail="capability not met")
    names = {s.name for s in bench.applicable_sentinels("P8")}
    # by P8, all four integrity sentinels are active
    assert {"representation-integrity", "uncertainty-reliability", "replay-fidelity",
            "option-diversity"} <= names
    all_healthy = [bench.SentinelResult(name=n, healthy=True) for n in names]
    report = bench.GateReport(phase="P8", capability=not_met, sentinels=all_healthy)
    assert isinstance(report, bench.GateReport)
    assert report.passed is False  # capability gates the composite, even all-healthy


def test_sentinels_activate_by_phase() -> None:
    # applicable_sentinels, not run_gate: P1's capability check trains real models.
    p1 = {s.name for s in bench.applicable_sentinels("P1")}
    assert "representation-integrity" in p1
    assert "uncertainty-reliability" in p1
    assert "replay-fidelity" not in p1  # generative replay arrives at P3
    assert "option-diversity" not in p1  # options arrive at P5
