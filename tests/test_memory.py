"""Unit tests for the ReplayBuffer (P3-003): honest storage, real-anchored
rehearsal batches, lineage-capped latent-space dreams, and the epistemic gate."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from prospect import interfaces
from prospect.memory import ReplayBuffer
from prospect.types import Action, LatentState, Prediction, Transition
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