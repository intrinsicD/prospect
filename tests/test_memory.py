"""Unit tests for ReplayBuffer (P3-003, U-004): hybrid real storage,
real-anchored rehearsal, lineage-capped latent-space dreams, and the epistemic gate."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from prospect import interfaces
from prospect.memory import (
    ReplayBuffer,
    RetrievalAugmentedWorldModel,
    SemanticStore,
    UncertaintyMemoryRouter,
)
from prospect.types import (
    Action,
    KnowledgeItem,
    LatentState,
    MemberRollout,
    Prediction,
    Provenance,
    Transition,
    Trust,
)
from prospect.world_model import FlatWorldModel


def _transition(value: float) -> Transition:
    return Transition(
        state=LatentState(z=np.array([value, 0.0, 0.0])),
        action=Action(data=np.array([value / 10.0])),
        next_state=LatentState(z=np.array([value + 1.0, 0.0, 0.0])),
        reward=-value,
    )


def _filled(buffer: ReplayBuffer, n: int) -> ReplayBuffer:
    for i in range(n):
        buffer.add(_transition(float(i)))
    return buffer


class _DepthOneModel:
    """Stub: confident at depth 1 (epistemic 0.01), exploding beyond (1.0) —
    exercises the self-calibrating gate. Tracks state via a step marker dim."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        z = np.asarray(state.z, dtype=float)
        depth_marker = z[-1]
        epistemic = 0.01 if depth_marker == 0.0 else 1.0
        mean = z + np.array([0.0] * (len(z) - 1) + [1.0])  # bump the marker
        return Prediction(mean=mean, var=np.ones_like(z), epistemic=epistemic,
                          aleatoric=0.1, reward=0.0)

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        return [self.predict(state, a) for a in actions]


def test_buffer_keeps_recent_fifo_and_early_lifetime_reservoir() -> None:
    capacity = 10
    buffer = ReplayBuffer(capacity=capacity, seed=7)
    for i in range(10 * capacity):
        buffer.add(_transition(float(i)))
        assert len(buffer) == min(i + 1, capacity)

    assert isinstance(buffer, interfaces.EpisodicMemory)
    assert buffer.fifo_capacity == 6
    assert buffer.reservoir_capacity == 4
    assert len(buffer._fifo) == buffer.fifo_capacity
    assert len(buffer._reservoir) == buffer.reservoir_capacity

    fifo_values = {float(np.asarray(t.state.z)[0]) for t in buffer._fifo}
    reservoir_values = {float(np.asarray(t.state.z)[0]) for t in buffer._reservoir}
    assert fifo_values == set(range(94, 100))
    assert reservoir_values <= set(range(94))
    assert 7.0 in reservoir_values  # marked early item survives 10x-capacity churn

    sampled_values = {float(np.asarray(t.state.z)[0]) for t in buffer.sample(2_000)}
    assert {7.0, 99.0} <= sampled_values  # uniform draws cover reservoir and FIFO

    twin = _filled(ReplayBuffer(capacity=capacity, seed=7), 10 * capacity)
    twin_reservoir = [float(np.asarray(t.state.z)[0]) for t in twin._reservoir]
    assert twin_reservoir == [float(np.asarray(t.state.z)[0]) for t in buffer._reservoir]


def test_single_slot_buffer_degrades_to_fifo() -> None:
    buffer = _filled(ReplayBuffer(capacity=1), 3)
    assert buffer.fifo_capacity == 1
    assert buffer.reservoir_capacity == 0
    assert len(buffer) == 1
    assert float(np.asarray(buffer.sample(1)[0].state.z)[0]) == 2.0


def test_sampling_empty_buffer_raises() -> None:
    with pytest.raises(ValueError, match="empty replay buffer"):
        ReplayBuffer().sample(1)


def test_generative_replay_requires_a_model() -> None:
    with pytest.raises(ValueError, match="needs a world model"):
        _filled(ReplayBuffer(), 8).generative_replay(8)


def _split(batch: list[Transition]) -> tuple[list[Transition], list[Transition]]:
    reals: list[Transition] = []
    dreams: list[Transition] = []
    for t in batch:  # split by marker: dataclass == on array fields is ill-defined
        if t.option is not None and t.option.name == ReplayBuffer.DREAM_SKILL:
            dreams.append(t)
        else:
            reals.append(t)
    return reals, dreams


def test_rehearsal_batch_is_real_anchored_and_lineage_capped() -> None:
    model = FlatWorldModel(seed=0)
    buffer = _filled(ReplayBuffer(model, real_fraction=0.5, max_dream_depth=3, seed=1), 64)
    before = len(buffer)
    batch = buffer.generative_replay(32)
    reals, dreams = _split(batch)
    assert len(batch) == 32
    assert len(reals) >= 16  # the real fraction is a floor
    assert dreams, "expected some dreams from an untrained-but-usable model"
    assert len(buffer) == before  # dreams are NEVER stored (no dream-of-dreams)
    for dream in dreams:
        assert dream.option is not None
        assert 1 <= dream.option.metadata["depth"] <= 3
        assert len(np.asarray(dream.next_state.z)) == model.latent_dim  # latent space
        assert dream.prediction is not None  # carries the dreaming step


def test_epistemic_gate_cuts_deep_dreams() -> None:
    buffer = _filled(
        ReplayBuffer(_DepthOneModel(), real_fraction=0.25, max_dream_depth=5,
                     epistemic_multiplier=4.0, seed=2),
        32,
    )
    batch = buffer.generative_replay(32)
    reals, dreams = _split(batch)
    assert len(reals) == 8 and len(dreams) == 24  # every start yields its depth-1 dream
    assert max(d.option.metadata["depth"] for d in dreams if d.option) == 1  # depth 2+ gated


class _BimodalModel(_DepthOneModel):
    """Some start states are wildly off-distribution: gated at depth 1."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        prediction = super().predict(state, action)
        z = np.asarray(state.z, dtype=float)
        if z[0] >= 24.0:  # a minority of starts (buffer holds values 0..31)
            return Prediction(mean=prediction.mean, var=prediction.var,
                              epistemic=1000.0, aleatoric=0.1)
        return prediction


def test_gated_out_dreams_are_topped_up_with_real_anchors() -> None:
    buffer = _filled(
        ReplayBuffer(_BimodalModel(), real_fraction=0.25, max_dream_depth=5,
                     epistemic_multiplier=4.0, seed=3),
        32,
    )
    batch = buffer.generative_replay(32)
    reals, dreams = _split(batch)
    assert dreams, "in-distribution starts still dream"
    assert len(batch) == 32  # shortage refilled
    assert len(reals) > 8  # the real fraction is a floor, not a ceiling

def _fact(key: list[float], answer: list[float], trust: Trust = Trust.HIGH) -> KnowledgeItem:
    prov = Provenance(source="unit", trust=trust)
    return KnowledgeItem(content=(np.array(key), np.array(answer)), provenance=prov)


def test_semantic_store_returns_nearest_fact_with_provenance() -> None:
    store = SemanticStore()
    assert store.query(np.array([0.0, 0.0])) == []  # empty store
    store.write(_fact([0.0, 0.0], [1.0]))
    store.write(_fact([5.0, 5.0], [2.0]))
    [near] = store.query(np.array([0.2, -0.1]))
    assert near.content[1][0] == 1.0  # nearest to [0,0]
    assert near.provenance is not None
    [far] = store.query(np.array([4.8, 5.1]))
    assert far.content[1][0] == 2.0


def test_router_gates_on_epistemic() -> None:
    store = SemanticStore()
    router = UncertaintyMemoryRouter([store], threshold=0.5)
    assert router.route(np.zeros(2), epistemic=0.2) is None  # confident -> parametric
    assert router.route(np.zeros(2), epistemic=0.9) is store  # uncertain -> retrieve
    empty = UncertaintyMemoryRouter([], threshold=0.5)
    assert empty.route(np.zeros(2), epistemic=0.9) is None  # nothing to retrieve from


def test_semantic_store_declares_trust() -> None:
    assert SemanticStore().trust is Trust.HIGH  # internal distilled store, trusted
    assert SemanticStore(trust=Trust.UNTRUSTED).trust is Trust.UNTRUSTED


def test_router_is_trust_ordered() -> None:  # P8-002: highest-trust eligible wins
    low, high = SemanticStore(trust=Trust.LOW), SemanticStore(trust=Trust.HIGH)
    assert UncertaintyMemoryRouter([low, high], threshold=0.5).route(None, 0.9) is high
    assert UncertaintyMemoryRouter([high, low], threshold=0.5).route(None, 0.9) is high


def test_router_never_lets_untrusted_override() -> None:  # P8-002: data, not instruction
    poisoned = SemanticStore(trust=Trust.UNTRUSTED)
    trusted = SemanticStore(trust=Trust.HIGH)
    # default floor (LOW) excludes an untrusted-only set -> fall back to the model
    assert UncertaintyMemoryRouter([poisoned], threshold=0.5).route(None, 0.9) is None
    # a trust-blind caller (floor UNTRUSTED) would consult the poison...
    blind = UncertaintyMemoryRouter([poisoned], threshold=0.5, min_trust=Trust.UNTRUSTED)
    assert blind.route(None, 0.9) is poisoned
    # ...but trust-ordering still prefers a trusted source when one is present
    mixed = UncertaintyMemoryRouter([poisoned, trusted], threshold=0.5, min_trust=Trust.UNTRUSTED)
    assert mixed.route(None, 0.9) is trusted


def test_store_and_router_satisfy_protocols() -> None:
    assert isinstance(SemanticStore(), interfaces.SemanticMemory)
    assert isinstance(SemanticStore(), interfaces.KnowledgeSource)
    assert isinstance(UncertaintyMemoryRouter(), interfaces.MemoryRouter)


class _EpiModel:
    """Protocol-only base world model (P9-001 test): fixed epistemic; predict bumps
    dim-0 by 1, so the base prediction is distinguishable from a retrieved fact."""

    def __init__(self, epistemic: float) -> None:
        self._epi = epistemic

    def predict(self, state: LatentState, action: Action) -> Prediction:
        return Prediction(mean=np.asarray(state.z, dtype=float) + np.array([1.0, 0.0]),
                          var=np.ones(2), epistemic=self._epi, aleatoric=0.1)

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        return [self.predict(state, a) for a in actions]


class _MemberEpiModel(_EpiModel):
    def predict_member_batch(
        self,
        member_latents: np.ndarray,
        actions: np.ndarray,
        initial_ood: float | None = None,
    ) -> MemberRollout:
        states = np.asarray(member_latents, dtype=float)
        if states.ndim == 2:
            states = np.repeat(states[None, :, :], 2, axis=0)
        next_states = states.copy()
        next_states[0, :, 0] += 1.0
        next_states[1, :, 0] -= 1.0
        epistemic = next_states.var(axis=0).mean(axis=1)
        if initial_ood is not None:
            epistemic *= 1.0 + initial_ood
        return MemberRollout(
            states=next_states,
            variances=np.ones_like(next_states),
            rewards=np.zeros((2, len(actions))),
            epistemic=epistemic,
        )


def test_retrieval_augments_only_when_uncertain() -> None:  # P9-001
    store = SemanticStore()
    store.write(_fact([0.0, 0.0, 0.5], [9.0, 9.0]))  # key = latent(2) + action(1)
    router = UncertaintyMemoryRouter([store], threshold=0.5)
    state, action = LatentState(z=np.array([0.0, 0.0])), Action(data=np.array([0.5]))

    confident = RetrievalAugmentedWorldModel(_EpiModel(0.2), router)  # 0.2 <= 0.5
    passthrough = confident.predict(state, action)
    assert list(passthrough.mean) == [1.0, 0.0] and confident.retrievals == 0  # base, no retrieval
    assert confident.gate_hits == 0

    uncertain = RetrievalAugmentedWorldModel(_EpiModel(0.9), router)  # 0.9 > 0.5 -> retrieve
    corrected = uncertain.predict(state, action)
    assert list(corrected.mean) == [9.0, 9.0]  # the retrieved fact stands in for the guess
    assert corrected.epistemic == 0.0 and uncertain.retrievals == 1 and uncertain.calls == 1
    assert uncertain.gate_hits == 1
    assert isinstance(uncertain, interfaces.WorldModel)


def test_retrieval_is_applied_inside_member_trajectory_rollouts() -> None:  # U-001
    store = SemanticStore()
    store.write(_fact([0.0, 0.0, 0.5], [9.0, 9.0]))
    router = UncertaintyMemoryRouter([store], threshold=0.25)
    augmented = RetrievalAugmentedWorldModel(_MemberEpiModel(0.9), router)
    rollout = augmented.predict_member_batch(
        np.zeros((1, 2)), np.array([[0.5]]), initial_ood=None
    )
    assert np.allclose(rollout.states, 9.0)  # the fact covers and corrects both members
    assert float(np.asarray(rollout.epistemic)[0]) == 0.0
    assert augmented.retrievals == 1 and augmented.calls == 1


def test_member_retrieval_requires_distance_coverage_for_every_particle() -> None:
    store = SemanticStore()
    store.write(_fact([2.0, 0.0, 0.5], [9.0, 9.0]))  # exact at the member-mean query
    router = UncertaintyMemoryRouter([store], threshold=0.25)
    augmented = RetrievalAugmentedWorldModel(
        _MemberEpiModel(0.9), router, reliability_radius=2.0
    )
    member_states = np.array([[[0.0, 0.0]], [[4.0, 0.0]]])
    rollout = augmented.predict_member_batch(member_states, np.array([[0.5]]))
    # Each particle is squared-distance 4 from the fact, outside radius 2: the
    # mean query alone is insufficient evidence, so the base particles stand.
    assert not np.allclose(rollout.states, 9.0)
    assert augmented.gate_hits == 1 and augmented.retrievals == 0 and augmented.calls == 1


def test_augmented_imagine_propagates_the_base_member_trajectories() -> None:
    augmented = RetrievalAugmentedWorldModel(
        _MemberEpiModel(0.9), UncertaintyMemoryRouter(threshold=10.0)
    )
    predictions = augmented.imagine(
        LatentState(z=np.zeros(2)), [Action(data=np.zeros(1)) for _ in range(3)]
    )
    assert predictions[-1].epistemic > predictions[0].epistemic


def test_retrieval_distance_gated_in_planning() -> None:  # P9-007
    store = SemanticStore()
    store.write(_fact([0.0, 0.0, 0.5], [9.0, 9.0]))  # fact key = latent(2) + action(1)
    router = UncertaintyMemoryRouter([store], threshold=0.5)
    action = Action(data=np.array([0.5]))

    # A FAR query (key-distance 50 > radius): the nearest fact is fiction at this point,
    # so it is NOT substituted — the model's own prediction stands, uncertainty intact.
    far = RetrievalAugmentedWorldModel(_EpiModel(0.9), router, reliability_radius=2.0)
    far_pred = far.predict(LatentState(z=np.array([5.0, 5.0])), action)
    assert list(far_pred.mean) == [6.0, 5.0]  # base (state + [1,0]), not the fact
    assert far.gate_hits == 1 and far.retrievals == 0
    assert far_pred.epistemic == 0.9  # untouched, no free pass

    # A CLOSE query (key-distance 1.0 <= radius 2.0): substitute, but carry HONEST
    # distance-scaled epistemic (0.9 * dist/radius) instead of the certain epi=0.
    near = RetrievalAugmentedWorldModel(_EpiModel(0.9), router, reliability_radius=2.0)
    near_pred = near.predict(LatentState(z=np.array([1.0, 0.0])), action)
    assert list(near_pred.mean) == [9.0, 9.0] and near.gate_hits == near.retrievals == 1
    assert 0.0 < near_pred.epistemic < 0.9  # not zeroed: reliability = closeness
    assert abs(near_pred.epistemic - 0.9 * (1.0 / 2.0)) < 1e-12
