from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("torchrl")
planning = pytest.importorskip("bench.world_model_lifecycle.planning")
torch = pytest.importorskip("torch")
model = pytest.importorskip("bench.world_model_lifecycle.model")


def test_analytic_oracle_conforms_to_gymnasium_for_both_tasks() -> None:
    report = planning.run_pendulum_conformance(samples_per_task=16, seed=73)

    assert report["passed"] is True
    assert report["cases"] == 32
    assert report["terminated_or_truncated_cases"] == 0
    assert len(report["report_sha256"]) == 64


def test_task_b_reverses_intended_torque_and_preserves_reward_semantics() -> None:
    observation = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    action = torch.tensor([1.0], dtype=torch.float64)

    next_a, reward_a, applied_a = planning.analytic_pendulum_step(
        observation,
        planning.TASK_A_CONTEXT,
        action,
    )
    next_b, reward_b, applied_b = planning.analytic_pendulum_step(
        observation,
        planning.TASK_B_CONTEXT,
        action,
    )

    assert applied_a.item() == pytest.approx(1.0)
    assert applied_b.item() == pytest.approx(-1.0)
    assert next_a[2].item() == pytest.approx(0.15)
    assert next_b[2].item() == pytest.approx(-0.15)
    assert reward_a.item() == pytest.approx(-0.001)
    assert reward_b.item() == pytest.approx(-0.001)


class _FakeEnsemble:
    version = "fake-model-v3"

    def predict_ensemble(
        self,
        observation: Any,
        context: Any,
        action: Any,
    ) -> tuple[Any, Any]:
        del context, action
        member_one = torch.zeros((*observation.shape[:-1], 4), dtype=observation.dtype)
        member_two = torch.zeros_like(member_one)
        member_one[..., 2] = 1.0
        member_two[..., 2] = 3.0
        member_one[..., 3] = -2.0
        member_two[..., 3] = -4.0
        means = torch.stack((member_one, member_two))
        return means, torch.ones_like(means)


def test_learned_adapter_uses_member_mean_and_carries_context() -> None:
    step = planning.EnsembleMeanTensorDictStep(_FakeEnsemble())
    tensordict = planning.TensorDict(
        {
            planning.STATE_KEY: torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
            planning.ACTION_KEY: torch.tensor([[0.5]]),
        },
        batch_size=(1,),
    )

    result = step(tensordict)

    assert torch.equal(result[planning.STATE_KEY][..., 3:], torch.ones(1, 1))
    assert torch.equal(result[planning.STATE_KEY][..., 2], torch.tensor([2.0]))
    assert torch.equal(result[planning.REWARD_KEY], torch.tensor([[-3.0]]))


def test_cem_controller_rng_checkpoint_reproduces_next_action_exactly() -> None:
    state = torch.tensor([0.0, 1.0, 0.0, 0.0])
    controller = planning.CEMController(planning.make_true_dynamics_env(), seed=101)
    controller.act(state)
    checkpoint = controller.state_dict()
    expected = controller.act(state)

    restored = planning.CEMController(planning.make_true_dynamics_env(), seed=999)
    restored.load_state_dict(checkpoint)
    actual = restored.act(state)

    assert torch.equal(actual, expected)
    assert restored.rng_digest == controller.rng_digest
    assert torch.all(actual >= -2.0)
    assert torch.all(actual <= 2.0)
    assert controller.version == "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1"
    selected, predicted_return = controller.select(np.asarray([0.0, 1.0, 0.0]), 0.0)
    assert -2.0 <= selected <= 2.0
    assert predicted_return == 0.0


def test_random_controller_rng_checkpoint_reproduces_next_action_exactly() -> None:
    state = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    controller = planning.UniformRandomController(seed=211)
    controller.act(state)
    checkpoint = controller.state_dict()
    expected = controller.act(state)

    restored = planning.UniformRandomController(seed=999)
    restored.load_state_dict(checkpoint)
    actual = restored.act(state)

    assert torch.equal(actual, expected)
    assert restored.rng_digest == controller.rng_digest
    assert controller.version == "wm001-uniform-random-controller-v1"
    selected, predicted_return = controller.select(np.asarray([1.0, 0.0, 0.0]), 1.0)
    assert -2.0 <= selected <= 2.0
    assert predicted_return == 0.0


def test_learned_controller_exposes_the_live_model_version() -> None:
    controller = planning.CEMController(
        planning.make_learned_model_env(_FakeEnsemble()),
        seed=17,
    )

    assert controller.version == "fake-model-v3"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_stacked_predictor_matches_source_for_single_and_batched_inputs(device: str) -> None:
    source = model.ProbabilisticEnsemble(
        model.WorldModelConfig(ensemble_members=3, hidden_dimensions=(19, 13)),
        initialization_seed=1301,
    ).to(device)
    stacked = planning.StackedEnsemblePredictor.try_from(source)
    assert stacked is not None
    assert stacked.version == source.version

    generator = torch.Generator(device=device).manual_seed(1709)
    for leading_shape in ((), (11,), (2, 7)):
        observation = torch.randn((*leading_shape, 3), generator=generator, device=device)
        direction = observation[..., :2]
        observation[..., :2] = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.randint(0, 2, (*leading_shape, 1), generator=generator, device=device).float()
        action = torch.empty((*leading_shape, 1), device=device).uniform_(
            -2.0,
            2.0,
            generator=generator,
        )

        expected_means, expected_variances = source.predict_ensemble(observation, context, action)
        actual_means, actual_variances = stacked.predict_ensemble(observation, context, action)
        repeated_means, repeated_variances = stacked.predict_ensemble(observation, context, action)

        assert torch.allclose(actual_means, expected_means, rtol=2e-6, atol=2e-6)
        assert torch.allclose(actual_variances, expected_variances, rtol=2e-6, atol=2e-6)
        assert torch.equal(repeated_means, actual_means)
        assert torch.equal(repeated_variances, actual_variances)
        expected_target = expected_means.mean(dim=0)
        actual_target = stacked.predict_mean_target(observation, context, action)
        assert torch.allclose(actual_target, expected_target, rtol=2e-6, atol=2e-6)
        expected_next = source.project_next(observation, expected_target[..., :3])
        actual_next = stacked.project_next(observation, actual_target[..., :3])
        assert torch.allclose(actual_next, expected_next, rtol=2e-6, atol=2e-6)


def test_learned_env_vectorizes_compatible_model_and_safely_falls_back() -> None:
    source = model.ProbabilisticEnsemble(
        model.WorldModelConfig(ensemble_members=2, hidden_dimensions=(9, 7)),
        initialization_seed=1901,
    )

    vectorized_env = planning.make_learned_model_env(source)
    fallback_env = planning.make_learned_model_env(_FakeEnsemble())

    assert isinstance(vectorized_env.world_model.predictor, planning.StackedEnsemblePredictor)
    assert vectorized_env.world_model.predictor.source is source
    assert vectorized_env.world_model.predictor.version == source.version
    assert fallback_env.world_model.predictor.__class__ is _FakeEnsemble


def test_stacked_predictor_refreshes_after_source_parameter_change() -> None:
    source = model.ProbabilisticEnsemble(
        model.WorldModelConfig(ensemble_members=2, hidden_dimensions=(11, 7)),
        initialization_seed=2003,
    )
    stacked = planning.StackedEnsemblePredictor(source)
    observation = torch.tensor([[0.8, 0.6, -0.25], [-0.6, 0.8, 0.75]])
    context = torch.tensor([[0.0], [1.0]])
    action = torch.tensor([[-1.0], [1.5]])
    version_before = stacked.version

    with torch.no_grad():
        source.members[0].network[0].bias.add_(0.125)

    expected = source.predict_ensemble(observation, context, action)
    actual = stacked.predict_ensemble(observation, context, action)

    assert stacked.version == source.version
    assert stacked.version != version_before
    assert torch.allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
    assert torch.allclose(actual[1], expected[1], rtol=2e-6, atol=2e-6)
