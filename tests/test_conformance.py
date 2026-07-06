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
from prospect.memory import (
    ReplayBuffer,
    RetrievalAugmentedWorldModel,
    SemanticStore,
    UncertaintyMemoryRouter,
)
from prospect.observation import LatentActionModel
from prospect.planning import FlatPlanner, HierarchicalManager, JumpyOptionModel
from prospect.skills import SkillRouter
from prospect.voe import LearningProgressCurriculum, SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel


def test_every_skeleton_conforms_to_its_protocol() -> None:
    codec: interfaces.Codec = UniversalCodec()
    world_model: interfaces.WorldModel = FlatWorldModel()
    learner: interfaces.Learner = FlatWorldModel()  # the training seam (P0-003)
    planner: interfaces.Planner = FlatPlanner(FlatWorldModel())
    option_model: interfaces.OptionModel = JumpyOptionModel()
    option_learner: interfaces.Learner = JumpyOptionModel()  # trains on option jumps (P5-001)
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
    # P9-001 seams: the arbiter the composition root reads, the tunable planner it
    # writes, and the retrieval-augmented model the planner plans over.
    arbiter: interfaces.ModeArbiter = LearningProgressCurriculum(SurpriseCompetenceMonitor())
    tunable: interfaces.UncertaintyTunable = FlatPlanner(FlatWorldModel())
    augmented: interfaces.WorldModel = RetrievalAugmentedWorldModel(
        FlatWorldModel(), UncertaintyMemoryRouter())
    observer: interfaces.ObservationLearner = LatentActionModel(obs_dim=3)  # P13-001, ADR-0010

    for impl in (
        codec,
        world_model,
        learner,
        planner,
        option_model,
        option_learner,
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
        arbiter,
        tunable,
        augmented,
        observer,
    ):
        assert impl is not None
