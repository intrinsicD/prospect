"""Parity and provenance tests for candidate-landscape capture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from bench.candidate_landscape import CandidateLandscapePlanner, CandidatePoolAudit
from bench.candidate_landscape import experiment as landscape_experiment
from bench.proposal_injection import PlannerState, ProposalInjectionPlanner
from prospect.types import Action, LatentState, Prediction


class _FirstActionModel:
    def predict(self, state: LatentState, action: Action) -> Prediction:
        reward = float(np.asarray(action.data, dtype=float)[0])
        return Prediction(
            mean=np.asarray(state.z, dtype=float),
            var=np.ones_like(np.asarray(state.z, dtype=float)),
            epistemic=0.0,
            aleatoric=1.0,
            reward=reward,
        )

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        return [self.predict(state, action) for action in actions]


class _Provider:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, raw: np.ndarray, count: int, horizon: int, action_dim: int) -> np.ndarray:
        self.calls.append(raw.copy())
        values = np.linspace(0.7, 0.9, count)[:, None, None]
        return np.broadcast_to(values, (count, horizon, action_dim)).copy()


def _state(value: float) -> PlannerState:
    return PlannerState(
        latent=LatentState(z=np.array([0.0, 0.0])),
        raw_sidecar=np.array([value, 0.1, -0.2]),
    )


def _exact_score(raw: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    return np.asarray(np.sum(sequences, axis=(1, 2)) + raw[0], dtype=float)


def test_capture_is_action_warm_state_and_rng_parity() -> None:
    common: dict[str, Any] = {
        "action_dim": 2,
        "action_low": -1.0,
        "action_high": 1.0,
        "horizon": 12,
        "candidates": 8,
        "elites": 2,
        "iterations": 3,
        "uncertainty_penalty": 0.0,
        "keep_elite_fraction": 0.5,
        "seed": 812,
        "injection_count": 2,
    }
    base_provider = _Provider()
    audit_provider = _Provider()
    base = ProposalInjectionPlanner(
        _FirstActionModel(),
        sequence_provider=base_provider,
        **common,  # type: ignore[arg-type]
    )
    audited = CandidateLandscapePlanner(
        _FirstActionModel(),
        sequence_provider=audit_provider,
        episode_steps=3,
        audit_steps=(0, 1),
        exact_scorer=_exact_score,
        **common,
    )

    for value in (0.0, 0.1, 0.2):
        state = _state(value)
        base_action = base.plan(state)
        audited_action = audited.plan(state)
        assert np.array_equal(audited_action.data, base_action.data)
        assert base._warm_mean is not None and audited._warm_mean is not None
        assert base._warm_elites is not None and audited._warm_elites is not None
        assert np.array_equal(audited._warm_mean, base._warm_mean)
        assert np.array_equal(audited._warm_elites, base._warm_elites)

    assert all(
        np.array_equal(left, right) for left, right in zip(base_provider.calls, audit_provider.calls, strict=True)
    )
    assert np.array_equal(base._sample_colored_noise(4), audited._sample_colored_noise(4))

    pools = audited.pool_audits
    assert len(pools) == 2 * 3
    assert [(pool.step, pool.iteration) for pool in pools] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert np.array_equal(pools[0].injected, [False] * 6 + [True, True])
    assert np.count_nonzero(pools[1].injected) == 1
    # On the warm second call, one carried native/injected elite occupies the last
    # row, so replacements occupy the two rows immediately before it.
    assert np.array_equal(
        pools[3].injected,
        [False] * 5 + [True, True, False],
    )
    for pool in pools:
        assert np.array_equal(pool.exact_scores, _exact_score(pool.raw_state, pool.sequences))
        assert pool.sequences.flags.writeable is False
        assert pool.latent_state.flags.writeable is False
        assert pool.learned_scores.flags.writeable is False
        assert pool.exact_scores.flags.writeable is False
        assert pool.injected.flags.writeable is False


def test_reset_preserves_provider_aligned_call_index() -> None:
    provider = _Provider()
    planner = CandidateLandscapePlanner(
        _FirstActionModel(),
        action_dim=2,
        horizon=12,
        candidates=4,
        elites=1,
        iterations=1,
        injection_count=1,
        sequence_provider=provider,
        episode_steps=2,
        audit_steps=(0,),
        exact_scorer=_exact_score,
    )
    planner.plan(_state(0.0))
    planner.plan(_state(0.1))
    planner.reset()
    planner.plan(_state(0.2))

    assert [(pool.call_index, pool.episode_index, pool.step) for pool in planner.pool_audits] == [
        (0, 0, 0),
        (2, 1, 0),
    ]


def test_formal_pool_lineage_matches_top_three_carry_rule() -> None:
    planner = CandidateLandscapePlanner(
        _FirstActionModel(),
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=12,
        candidates=64,
        elites=8,
        iterations=3,
        keep_elite_fraction=0.3,
        injection_count=8,
        sequence_provider=_Provider(),
        exact_scorer=_exact_score,
    )
    planner.plan(_state(0.0))
    landscape_experiment._validate_pool_lineage(list(planner.pool_audits))


def test_capture_configuration_and_record_validation() -> None:
    common: dict[str, Any] = {
        "action_dim": 2,
        "horizon": 12,
        "candidates": 4,
        "elites": 1,
        "iterations": 1,
        "sequence_provider": _Provider(),
    }
    with pytest.raises(ValueError, match="positive injection"):
        CandidateLandscapePlanner(_FirstActionModel(), injection_count=0, **common)
    with pytest.raises(ValueError, match="inside the episode"):
        CandidateLandscapePlanner(_FirstActionModel(), injection_count=1, episode_steps=2, audit_steps=(2,), **common)

    with pytest.raises(ValueError, match="shape"):
        CandidatePoolAudit(
            call_index=0,
            episode_index=0,
            step=0,
            iteration=0,
            raw_state=np.zeros(3),
            latent_state=np.zeros(2),
            latent_ood=None,
            sequences=np.zeros((4, 11, 2)),
            learned_scores=np.zeros(4),
            exact_scores=np.zeros(4),
            injected=np.zeros(4, dtype=bool),
        )
