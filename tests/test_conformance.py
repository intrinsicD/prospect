"""Static protocol-conformance assertions (P0-009).

Each typed assignment below makes mypy verify the implementation *structurally*
against its `Protocol` — signatures included, which the runtime_checkable
isinstance smoke checks cannot do (they verify method presence only). The function
also runs as a plain test, so instantiation stays covered at runtime.
"""
from __future__ import annotations

from prospect import interfaces
from prospect.codec import UniversalCodec
from prospect.knowledge import ExternalKnowledgeSource, InternalKnowledgeSource, ToolSource
from prospect.memory import ReplayBuffer, SemanticStore, UncertaintyMemoryRouter
from prospect.planning import FlatPlanner, HierarchicalManager, JumpyOptionModel
from prospect.skills import SkillRouter
from prospect.voe import SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel


def test_every_skeleton_conforms_to_its_protocol() -> None:
    codec: interfaces.Codec = UniversalCodec()
    world_model: interfaces.WorldModel = FlatWorldModel()
    learner: interfaces.Learner = FlatWorldModel()  # the training seam (P0-003)
    planner: interfaces.Planner = FlatPlanner(FlatWorldModel())
    option_model: interfaces.OptionModel = JumpyOptionModel()
    manager: interfaces.HierarchicalPlanner = HierarchicalManager()
    monitor: interfaces.CompetenceMonitor = SurpriseCompetenceMonitor()
    skills: interfaces.SkillLibrary = SkillRouter()
    episodic: interfaces.EpisodicMemory = ReplayBuffer()
    semantic: interfaces.SemanticMemory = SemanticStore()
    semantic_as_source: interfaces.KnowledgeSource = SemanticStore()  # P0-008
    router: interfaces.MemoryRouter = UncertaintyMemoryRouter()
    internal: interfaces.KnowledgeSource = InternalKnowledgeSource()
    external: interfaces.KnowledgeSource = ExternalKnowledgeSource()
    tool: interfaces.KnowledgeSource = ToolSource()

    for impl in (
        codec,
        world_model,
        learner,
        planner,
        option_model,
        manager,
        monitor,
        skills,
        episodic,
        semantic,
        semantic_as_source,
        router,
        internal,
        external,
        tool,
    ):
        assert impl is not None
