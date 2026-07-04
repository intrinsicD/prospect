"""Unit tests for the ReplayBuffer (P3-003): honest storage, real-anchored
rehearsal batches, lineage-capped latent-space dreams, and the epistemic gate."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from prospect import interfaces
from prospect.memory import ReplayBuffer, SemanticStore, UncertaintyMemoryRouter
from prospect.types import (
    Action,
    KnowledgeItem,
    LatentState,
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


def test_buffer_satisfies_protocol_and_evicts_fifo() -> None:
    buffer = _filled(ReplayBuffer(capacity=4), 6)
    assert isinstance(buffer, interfaces.EpisodicMemory)
    assert len(buffer) == 4
    stored_first_dims = {float(np.asarray(t.state.z)[0]) for t in buffer.sample(200)}
    assert stored_first_dims <= {2.0, 3.0, 4.0, 5.0}  # 0 and 1 evicted first


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
