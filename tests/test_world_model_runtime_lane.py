from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytest

from bench.world_model_lifecycle.runtime_lane import (
    INDEPENDENT_OSCILLATOR_SOURCE,
    IndependentPhaseOscillatorEnvironment,
    PendulumEpisodeSession,
    PresetActionController,
    RuntimeCustody,
    run_episode,
    transition_lineage_row,
)


class _StaticBackend:
    version = "wm001-test-static-backend-v1"
    digest = hashlib.sha256(version.encode("ascii")).hexdigest()

    def predict_ensemble(
        self,
        observation: np.ndarray,
        context: float,
        action: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del context, action
        means = np.repeat(np.asarray(observation, dtype=np.float64)[None, :], 2, axis=0)
        variances = np.ones_like(means)
        return means, variances


class _SequenceController:
    version = "wm001-test-action-sequence-v1"

    def __init__(self, actions: Sequence[float]) -> None:
        self._actions = tuple(float(action) for action in actions)
        self._index = 0

    def select(self, observation: np.ndarray, context: float) -> tuple[float, float]:
        del observation, context
        action = self._actions[self._index]
        self._index += 1
        return action, 0.0


def test_independent_phase_oscillator_has_exact_seeded_reset_and_dynamics() -> None:
    environment = IndependentPhaseOscillatorEnvironment()
    initial, reset_info = environment.reset(seed=710_003)

    np.testing.assert_allclose(
        initial,
        np.asarray([0.9428705678878576, -0.3331592595303768, 1.2768786504270468]),
        rtol=0.0,
        atol=1e-15,
    )
    assert reset_info == {"reset_seed": 710_003}
    phase = math.atan2(float(initial[1]), float(initial[0]))
    expected_phase = phase + environment.STEP_SECONDS * float(initial[2])
    next_observation, reward, terminated, truncated, step_info = environment.step(np.asarray([1.75], dtype=np.float32))
    np.testing.assert_allclose(
        next_observation,
        np.asarray([math.cos(expected_phase), math.sin(expected_phase), initial[2]]),
        rtol=0.0,
        atol=1e-15,
    )
    assert reward == pytest.approx(math.cos(expected_phase), abs=1e-15)
    assert (terminated, truncated, step_info) == (False, False, {"step_index": 1})
    assert np.isfinite(next_observation).all()
    assert -1.0 <= reward <= 1.0

    repeated, _ = environment.reset(seed=710_003)
    np.testing.assert_array_equal(repeated, initial)


def test_independent_phase_oscillator_is_action_independent_for_full_horizon() -> None:
    left = IndependentPhaseOscillatorEnvironment()
    right = IndependentPhaseOscillatorEnvironment()
    left_initial, _ = left.reset(seed=810_007)
    right_initial, _ = right.reset(seed=810_007)
    np.testing.assert_array_equal(left_initial, right_initial)

    for step_index in range(200):
        left_step = left.step(np.asarray([-2.0], dtype=np.float32))
        right_step = right.step(np.asarray([2.0], dtype=np.float32))
        np.testing.assert_array_equal(left_step[0], right_step[0])
        assert left_step[1:] == right_step[1:]
        assert left_step[2] is False
        assert left_step[3] is (step_index == 199)
    with pytest.raises(RuntimeError, match="200-step horizon"):
        left.step(np.asarray([0.0], dtype=np.float32))


def test_independent_phase_oscillator_uses_canonical_runtime_evidence() -> None:
    actions = np.linspace(-2.0, 2.0, 200)
    custody = RuntimeCustody.create("wm001-independent-oscillator")
    episode, _ = run_episode(
        run_id="independent-control",
        task_id="independent_phase_oscillator",
        episode_id="independent-control:episode:0",
        reset_seed=910_009,
        controller=_SequenceController(actions),
        backend=_StaticBackend(),
        custody=custody,
    )

    assert len(episode.transitions) == 200
    assert episode.intended_actions == tuple(float(action) for action in actions)
    assert episode.applied_actions == (0.0,) * 200
    assert all(not transition.experience.terminated for transition in episode.transitions)
    assert all(not transition.experience.truncated for transition in episode.transitions[:-1])
    assert episode.transitions[-1].experience.truncated
    assert custody.ledger.transition_count == len(custody.store) == 200

    first = episode.transitions[0]
    reset = first.experience.decision.belief.information_set.observations[0]
    assert reset.modality == "independent_phase_oscillator_reset_state"
    assert reset.evidence.lineage.provenance.source_id == f"{INDEPENDENT_OSCILLATOR_SOURCE}:reset"
    for transition in episode.transitions:
        event = transition.experience
        observation = event.observation
        outcome = event.outcome
        physical = np.asarray(observation.evidence.payload["physical_observation"], dtype=np.float64)
        assert event.task_id == "independent_phase_oscillator"
        assert observation.modality == "independent_phase_oscillator_state"
        assert observation.evidence.payload["task_context"] == 2.0
        assert observation.evidence.lineage.provenance.source_id == INDEPENDENT_OSCILLATOR_SOURCE
        assert outcome.evidence.lineage.provenance.source_id == INDEPENDENT_OSCILLATOR_SOURCE
        assert outcome.evidence.payload["applied_torque"] == 0.0
        assert np.isfinite(physical).all()
        assert -1.0 <= float(outcome.evidence.payload["reward"]) <= 1.0
        lineage = transition_lineage_row(transition, split="irrelevant_evidence")
        assert lineage["task_context"] == 2.0
        assert lineage["applied_action"] == 0.0


def test_two_interleaved_episode_sessions_match_sequential_authoritative_runs() -> None:
    backend = _StaticBackend()
    actions_a = 1.75 * np.sin(np.linspace(0.0, 5.0 * np.pi, 200))
    actions_b = 1.5 * np.cos(np.linspace(0.0, 7.0 * np.pi, 200))
    reset_seeds = (710_003, 710_009)

    interleaved_custody = RuntimeCustody.create("wm001-session-interleaved")
    sessions = (
        PendulumEpisodeSession(
            run_id="interleaved",
            task_id="pendulum_normal_torque",
            episode_id="interleaved:episode:0",
            reset_seed=reset_seeds[0],
            backend=backend,
            custody=interleaved_custody,
        ),
        PendulumEpisodeSession(
            run_id="interleaved",
            task_id="pendulum_normal_torque",
            episode_id="interleaved:episode:1",
            reset_seed=reset_seeds[1],
            backend=backend,
            custody=interleaved_custody,
        ),
    )
    assert sessions[0].context == sessions[1].context == 0.0
    protected_initial = sessions[0].current_observation
    protected_initial[:] = 99.0
    assert not np.array_equal(sessions[0].current_observation, protected_initial)

    insertion_order = []
    global_tick = 0
    for index in range(200):
        first = sessions[0].step(float(actions_a[index]), global_tick)
        insertion_order.append(first.transition)
        global_tick = first.transition.created_at.tick + 1
        second = sessions[1].step(float(actions_b[index]), global_tick)
        insertion_order.append(second.transition)
        global_tick = second.transition.created_at.tick + 1

    interleaved = (sessions[0].finish(), sessions[1].close())
    assert sessions[0].finish() is interleaved[0]
    assert all(session.done and session.step_index == 200 for session in sessions)
    assert len(interleaved_custody.store) == 400
    assert interleaved_custody.ledger.transition_count == 400
    assert all(left.created_at.tick <= right.created_at.tick for left, right in pairwise(insertion_order))

    transition_ids = [transition.transition_id for episode in interleaved for transition in episode.transitions]
    prediction_ids = [prediction_id for episode in interleaved for prediction_id in episode.prediction_ids]
    assert len(set(transition_ids)) == len(transition_ids) == 400
    assert len(set(prediction_ids)) == len(prediction_ids) == 400
    for transition in insertion_order:
        assert interleaved_custody.ledger.get_transition(transition.transition_id) is transition
        assert interleaved_custody.store.get(transition.experience.experience_id) is transition.experience

    sequential_custody = RuntimeCustody.create("wm001-session-sequential")
    sequential_first, _ = run_episode(
        run_id="sequential",
        task_id="pendulum_normal_torque",
        episode_id="sequential:episode:0",
        reset_seed=reset_seeds[0],
        controller=_SequenceController(actions_a),
        backend=backend,
        custody=sequential_custody,
    )
    sequential_second, _ = run_episode(
        run_id="sequential",
        task_id="pendulum_normal_torque",
        episode_id="sequential:episode:1",
        reset_seed=reset_seeds[1],
        controller=_SequenceController(actions_b),
        backend=backend,
        custody=sequential_custody,
        start_tick=sequential_first.final_tick + 1,
    )
    sequential = (sequential_first, sequential_second)

    for batched_episode, sequential_episode in zip(interleaved, sequential, strict=True):
        assert batched_episode.undiscounted_return == sequential_episode.undiscounted_return
        assert batched_episode.intended_actions == sequential_episode.intended_actions
        assert batched_episode.applied_actions == sequential_episode.applied_actions
        np.testing.assert_array_equal(
            batched_episode.transitions[-1].experience.observation.evidence.payload["physical_observation"],
            sequential_episode.transitions[-1].experience.observation.evidence.payload["physical_observation"],
        )


def test_episode_session_rejects_missing_or_invalid_external_actions() -> None:
    controller = PresetActionController()
    with pytest.raises(ValueError, match=r"within \[-2, 2\]"):
        controller.preset(3.0)
    with pytest.raises(ValueError, match="predicted value"):
        controller.preset(0.0, float("nan"))
    with pytest.raises(RuntimeError, match="no preset action"):
        controller.select(np.zeros(3), 0.0)
    controller.preset(0.25, 7.5)
    assert controller.select(np.zeros(3), 0.0) == (0.25, 7.5)
    with pytest.raises(RuntimeError, match="no preset action"):
        controller.select(np.zeros(3), 0.0)
