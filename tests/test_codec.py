"""Unit tests for the UniversalCodec (P6-001): distillation into a target latent,
any-to-any (two modalities of the same situation → the same latent), decode
reconstruction, and loud errors on unknown modalities."""
from __future__ import annotations

import numpy as np
import pytest

from prospect import interfaces
from prospect.codec import UniversalCodec
from prospect.types import LatentState, Modality, Observation


def _target_latent(x: np.ndarray) -> np.ndarray:
    """A fixed nonlinear map standing in for the incumbent encoder. Depends on all
    of the state (incl. omega) so the latent is invertible — as the real incumbent
    latent is, since its inverse-dynamics/reward heads force controllable content."""
    w = np.array([[0.5, -0.3, 0.2, 0.1, 0.4, -0.2, 0.3, -0.1],
                  [-0.2, 0.4, 0.1, -0.3, 0.2, 0.5, -0.1, 0.3],
                  [0.3, 0.1, -0.4, 0.25, -0.15, 0.35, 0.2, -0.3]])
    return np.asarray(np.tanh(x[:3] @ w), dtype=float)


def _paired(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """State (3-dim), a 6-dim 'image' of the same underlying value, and the shared
    target latent — both modalities carry the same two-dof situation."""
    theta = rng.uniform(-np.pi, np.pi, n)
    omega = rng.uniform(-4.0, 4.0, n)
    state = np.stack([np.cos(theta), np.sin(theta), omega], axis=1)
    image = np.stack([np.cos(theta), np.sin(theta), np.sin(2 * theta),
                      np.cos(2 * theta), omega / 4.0, np.tanh(omega)], axis=1)
    target = np.stack([_target_latent(s) for s in state])
    return state, image, target


def _distilled(steps: int = 800) -> UniversalCodec:
    codec = UniversalCodec({Modality.STATE: 3, Modality.IMAGE: 6}, latent_dim=8, seed=0)
    rng = np.random.default_rng(1)
    state, image, target = _paired(rng, 2048)
    fit = np.random.default_rng(2)
    for _ in range(steps):
        idx = fit.integers(0, len(state), 64)
        codec.distill_encode(state[idx], Modality.STATE, target[idx])
        codec.distill_encode(image[idx], Modality.IMAGE, target[idx])
        codec.fit_decode(target[idx], state[idx], Modality.STATE)
    return codec


def test_codec_conforms_to_protocol() -> None:
    assert isinstance(UniversalCodec(), interfaces.Codec)


def test_encode_distills_to_the_target_latent() -> None:
    codec = _distilled()
    rng = np.random.default_rng(9)
    state, _, target = _paired(rng, 64)
    errors = [float(np.mean((np.asarray(codec.encode(Observation(Modality.STATE, s)).z) - t) ** 2))
              for s, t in zip(state, target, strict=True)]
    assert np.mean(errors) < 0.01  # reproduces the incumbent latent (the migration)


def test_two_modalities_of_the_same_state_share_a_latent() -> None:
    codec = _distilled()
    rng = np.random.default_rng(10)
    state, image, _ = _paired(rng, 64)
    gaps = [float(np.mean((np.asarray(codec.encode(Observation(Modality.STATE, s)).z)
                           - np.asarray(codec.encode(Observation(Modality.IMAGE, im)).z)) ** 2))
            for s, im in zip(state, image, strict=True)]
    assert np.mean(gaps) < 0.02  # any-to-any: same situation, same latent


def test_decode_reconstructs_the_modality() -> None:
    codec = _distilled()
    rng = np.random.default_rng(11)
    state, _, target = _paired(rng, 64)
    mean, std = codec._stats[Modality.STATE]
    errors = []
    for s, t in zip(state, target, strict=True):
        out = np.asarray(codec.decode(LatentState(z=t), Modality.STATE).data)
        errors.append(float(np.mean((out * std + mean - s) ** 2)))
    assert np.mean(errors) < 0.1  # latent -> the queried modality (R6, produce any output)


def test_unknown_modality_and_query_fail_loudly() -> None:
    codec = UniversalCodec({Modality.STATE: 3})
    with pytest.raises(KeyError, match="no adapter"):
        codec.encode(Observation(Modality.IMAGE, np.zeros(3)))
    with pytest.raises(KeyError, match="no decoder"):
        codec.decode(LatentState(z=np.zeros(8)), Modality.AUDIO)


def test_vision_embedding_ingests_and_distils_into_the_latent() -> None:  # P12-001
    """A VISION seam (a frozen-encoder embedding) is just another modality: it lands in
    the shared latent and distils toward an incumbent latent (P0-011, ADR-0009)."""
    codec = UniversalCodec({Modality.VISION: 16}, latent_dim=8, seed=0)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(96, 16))
    target = np.tanh(emb[:, :8] * 0.7)  # a fixed incumbent latent to reproduce
    assert len(codec.encode(Observation(Modality.VISION, emb[0])).z) == 8  # lands in the latent

    def err() -> float:
        return float(np.mean([np.mean((np.asarray(codec.encode(Observation(Modality.VISION, e)).z) - t) ** 2)
                              for e, t in zip(emb, target, strict=True)]))

    before = err()
    fit = np.random.default_rng(1)
    for _ in range(400):
        idx = fit.integers(0, len(emb), 32)
        codec.distill_encode(emb[idx], Modality.VISION, target[idx])
    assert err() < before * 0.5  # the VISION adapter distils toward the incumbent latent
