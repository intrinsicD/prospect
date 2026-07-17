"""WM-001 planning substrates with an exact, separately namespaced oracle.

This module contains no imports from Prospect's record or replay namespaces.
Every transition produced here is imagined inside a TorchRL
``ModelBasedEnvBase`` rollout. Real environment interaction and evidence custody
remain the responsibility of the experiment harness.

The formal controller deliberately delegates CEM itself to
``torchrl.modules.CEMPlanner`` 0.13.3. The only additions are:

* a stateless TensorDict adapter for deterministic ensemble-mean predictions;
* an exact analytic Pendulum-v1 adapter used only by the oracle control; and
* explicit custody of the Torch RNG state that CEMPlanner consumes internally.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor, nn
from torch.nn import functional as F
from torchrl import __version__ as torchrl_version
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs.model_based import ModelBasedEnvBase
from torchrl.modules import CEMPlanner

STATE_KEY = "state"
ACTION_KEY = "action"
REWARD_KEY = "reward"
TASK_A_CONTEXT = 0.0
TASK_B_CONTEXT = 1.0
EXPECTED_TORCHRL_VERSION = "0.13.3"


@dataclass(frozen=True, slots=True)
class PendulumSemantics:
    """The Pendulum-v1 constants sealed by WM-001."""

    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
    time_step: float = 0.05
    max_speed: float = 8.0
    max_torque: float = 2.0
    horizon: int = 200


PENDULUM_SEMANTICS = PendulumSemantics()


@dataclass(frozen=True, slots=True)
class CEMBudget:
    """Arguments passed unchanged to TorchRL's CEMPlanner."""

    planning_horizon: int = 10
    optim_steps: int = 3
    num_candidates: int = 64
    top_k: int = 8

    def __post_init__(self) -> None:
        if self.planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive")
        if self.optim_steps <= 0:
            raise ValueError("optim_steps must be positive")
        if self.num_candidates <= 1:
            raise ValueError("num_candidates must exceed one")
        if not 1 < self.top_k <= self.num_candidates:
            raise ValueError("top_k must be in [2, num_candidates]")


WM001_CEM_BUDGET = CEMBudget()


@runtime_checkable
class EnsemblePredictionAPI(Protocol):
    """Physical-unit prediction interface consumed by the learned adapter.

    Implementations return member-first means and variances with shape
    ``[members, *batch, 4]``. The four targets are three physical observation
    deltas followed by reward.
    """

    def predict_ensemble(
        self,
        observation: Tensor,
        context: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


class VectorizedEnsembleCompatibilityError(TypeError):
    """Raised when a predictor is not the sealed three-linear-layer ensemble."""


def _broadcast_planning_feature(
    value: Tensor | float | list[float] | tuple[float, ...],
    leading_shape: torch.Size,
    *,
    device: torch.device,
) -> Tensor:
    """Broadcast one scalar feature with the same contract as the backend."""

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.shape == leading_shape:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim == 0:
        tensor = tensor.reshape((1,) * len(leading_shape) + (1,))
    if tensor.shape[-1:] != (1,):
        raise ValueError("context and action must be scalar features")
    try:
        return torch.broadcast_to(tensor, (*leading_shape, 1))
    except RuntimeError as error:
        raise ValueError(
            f"feature shape {tuple(tensor.shape)} cannot broadcast to {tuple(leading_shape) + (1,)}"
        ) from error


class StackedEnsemblePredictor(nn.Module):
    """Planning-only batched implementation of the sealed ensemble.

    ``ProbabilisticEnsemble.forward`` intentionally uses a transparent Python
    loop because that is convenient for training and per-member bootstrap
    losses.  CEM inference calls the same five networks many times on medium
    batches; launching each of their three linear layers separately is
    needlessly expensive, especially on CUDA.  This immutable planning view
    stacks corresponding weights and evaluates each layer with one batched
    matrix multiplication.

    Construction is deliberately structural rather than coupled to a concrete
    backend class.  Only the exact ``Linear, SiLU, Linear, SiLU, Linear`` shape
    is accepted.  Callers use :meth:`try_from`; incompatible predictors retain
    the original inference path without any semantic adaptation.
    """

    _LINEAR_POSITIONS = (0, 2, 4)
    _source: Any
    _source_parameters: tuple[nn.Parameter, ...]
    _source_parameter_versions: tuple[int, ...]
    _weight_0: Tensor
    _weight_1: Tensor
    _weight_2: Tensor
    _bias_0: Tensor
    _bias_1: Tensor
    _bias_2: Tensor

    def __init__(self, source: EnsemblePredictionAPI) -> None:
        super().__init__()
        layers, config = self._compatible_layers(source)
        object.__setattr__(self, "_source", source)
        object.__setattr__(
            self,
            "_source_parameters",
            tuple(parameter for member_layers in layers for layer in member_layers for parameter in layer.parameters()),
        )
        object.__setattr__(
            self,
            "_source_parameter_versions",
            tuple(parameter._version for parameter in self._source_parameters),
        )
        self._member_count = len(layers)
        self._input_dimension = int(config.input_dimension)
        self._output_dimension = int(config.output_dimension)
        self._log_variance_min = float(config.log_variance_min)
        self._log_variance_max = float(config.log_variance_max)
        self._observation_scale = tuple(float(value) for value in config.scaling.observation)
        self._context_scale = float(config.scaling.context)
        self._action_scale = float(config.scaling.action)
        self._target_scale = tuple(float(value) for value in config.scaling.target)
        try:
            source_version = cast(Any, source).version
        except AttributeError as error:
            raise VectorizedEnsembleCompatibilityError("predictor has no source version") from error
        if source_version is None:
            raise VectorizedEnsembleCompatibilityError("predictor has no source version")
        self._source_version = str(source_version)
        for layer_index in range(3):
            self.register_buffer(
                f"_weight_{layer_index}",
                torch.stack(
                    [member_layers[layer_index].weight.detach() for member_layers in layers],
                    dim=0,
                ).clone(),
                persistent=False,
            )
            self.register_buffer(
                f"_bias_{layer_index}",
                torch.stack(
                    [member_layers[layer_index].bias.detach() for member_layers in layers],
                    dim=0,
                ).clone(),
                persistent=False,
            )

    @classmethod
    def try_from(cls, source: EnsemblePredictionAPI) -> StackedEnsemblePredictor | None:
        """Return a vectorized view when structurally safe, otherwise ``None``."""

        try:
            return cls(source)
        except VectorizedEnsembleCompatibilityError:
            return None

    @staticmethod
    def _compatible_layers(
        source: EnsemblePredictionAPI,
    ) -> tuple[tuple[tuple[nn.Linear, nn.Linear, nn.Linear], ...], Any]:
        if not isinstance(source, nn.Module):
            raise VectorizedEnsembleCompatibilityError("predictor is not a torch module")
        config = getattr(source, "config", None)
        required_config = (
            "input_dimension",
            "output_dimension",
            "log_variance_min",
            "log_variance_max",
            "scaling",
        )
        if config is None or any(not hasattr(config, name) for name in required_config):
            raise VectorizedEnsembleCompatibilityError("predictor has no compatible configuration")
        scaling = config.scaling
        if any(not hasattr(scaling, name) for name in ("observation", "context", "action", "target")):
            raise VectorizedEnsembleCompatibilityError("predictor has no compatible fixed scaling")
        if int(config.input_dimension) != 5 or int(config.output_dimension) != 4:
            raise VectorizedEnsembleCompatibilityError("predictor dimensions differ from WM-001")
        untyped_members = getattr(source, "members", None)
        if not isinstance(untyped_members, (nn.ModuleList, list, tuple)) or len(untyped_members) < 1:
            raise VectorizedEnsembleCompatibilityError("predictor has no ensemble members")
        members = cast(Sequence[nn.Module], untyped_members)

        extracted: list[tuple[nn.Linear, nn.Linear, nn.Linear]] = []
        expected_shapes: tuple[tuple[int, int], ...] | None = None
        expected_device: torch.device | None = None
        expected_dtype: torch.dtype | None = None
        for member in members:
            network = getattr(member, "network", None)
            if not isinstance(network, nn.Sequential) or len(network) != 5:
                raise VectorizedEnsembleCompatibilityError("member network is not the sealed sequence")
            if not isinstance(network[1], nn.SiLU) or not isinstance(network[3], nn.SiLU):
                raise VectorizedEnsembleCompatibilityError("member activations are not SiLU")
            selected = tuple(network[position] for position in StackedEnsemblePredictor._LINEAR_POSITIONS)
            if not all(isinstance(layer, nn.Linear) for layer in selected):
                raise VectorizedEnsembleCompatibilityError("member layers are not linear")
            typed = cast(tuple[nn.Linear, nn.Linear, nn.Linear], selected)
            if any(layer.bias is None for layer in typed):
                raise VectorizedEnsembleCompatibilityError("member linear layers require bias")
            shapes = tuple((layer.out_features, layer.in_features) for layer in typed)
            if expected_shapes is None:
                expected_shapes = shapes
                expected_device = typed[0].weight.device
                expected_dtype = typed[0].weight.dtype
            if shapes != expected_shapes:
                raise VectorizedEnsembleCompatibilityError("ensemble member shapes differ")
            if any(
                layer.weight.device != expected_device
                or layer.weight.dtype != expected_dtype
                or layer.bias.device != expected_device
                or layer.bias.dtype != expected_dtype
                for layer in typed
            ):
                raise VectorizedEnsembleCompatibilityError("ensemble member devices or dtypes differ")
            extracted.append(typed)

        assert expected_shapes is not None
        if expected_shapes[0][1] != int(config.input_dimension):
            raise VectorizedEnsembleCompatibilityError("first layer input dimension differs from config")
        if expected_shapes[-1][0] != int(config.output_dimension) * 2:
            raise VectorizedEnsembleCompatibilityError("last layer output dimension differs from config")
        if not callable(getattr(source, "project_next", None)):
            raise VectorizedEnsembleCompatibilityError("predictor has no projection implementation")
        return tuple(extracted), config

    @property
    def device(self) -> torch.device:
        return self._weight_0.device

    @property
    def version(self) -> str:
        """Expose exactly the digest/version of the snapshotted source state."""

        self._refresh_if_source_changed()
        return self._source_version

    @property
    def source(self) -> EnsemblePredictionAPI:
        """The original predictor, retained for identity and mutation checks."""

        return cast(EnsemblePredictionAPI, self._source)

    def _refresh_if_source_changed(self) -> None:
        versions = tuple(parameter._version for parameter in self._source_parameters)
        if versions == self._source_parameter_versions:
            return
        layers, _ = self._compatible_layers(self._source)
        destination = self.device
        for layer_index in range(3):
            weight = torch.stack(
                [member_layers[layer_index].weight.detach() for member_layers in layers],
                dim=0,
            ).to(destination)
            bias = torch.stack(
                [member_layers[layer_index].bias.detach() for member_layers in layers],
                dim=0,
            ).to(destination)
            setattr(self, f"_weight_{layer_index}", weight)
            setattr(self, f"_bias_{layer_index}", bias)
        object.__setattr__(self, "_source_parameter_versions", versions)
        self._source_version = str(self._source.version)

    def _normalized_outputs(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        self._refresh_if_source_changed()
        inputs = inputs.to(device=self.device, dtype=torch.float32)
        if inputs.shape[-1:] != (self._input_dimension,):
            raise ValueError(f"inputs must have trailing dimension {self._input_dimension}")
        leading_shape = inputs.shape[:-1]
        flattened = inputs.reshape(-1, self._input_dimension)
        hidden = torch.matmul(flattened.unsqueeze(0), self._weight_0.transpose(1, 2))
        hidden = F.silu(hidden + self._bias_0.unsqueeze(1))
        hidden = torch.bmm(hidden, self._weight_1.transpose(1, 2))
        hidden = F.silu(hidden + self._bias_1.unsqueeze(1))
        outputs = torch.bmm(hidden, self._weight_2.transpose(1, 2))
        outputs = outputs + self._bias_2.unsqueeze(1)
        outputs = outputs.reshape(self._member_count, *leading_shape, self._output_dimension * 2)
        means = outputs[..., : self._output_dimension]
        log_variances = outputs[..., self._output_dimension :].clamp(
            self._log_variance_min,
            self._log_variance_max,
        )
        return means, log_variances

    def predict_normalized(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized means/log-variances with member-first shape."""

        with torch.no_grad():
            return self._normalized_outputs(inputs)

    def _encoded_inputs(
        self,
        observation: Tensor | list[float] | tuple[float, ...],
        context: Tensor | float | list[float] | tuple[float, ...],
        action: Tensor | float | list[float] | tuple[float, ...],
    ) -> tuple[Tensor, Tensor]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if observation_tensor.shape[-1:] != (3,):
            raise ValueError("observation must end in dimension 3")
        leading_shape = observation_tensor.shape[:-1]
        context_tensor = _broadcast_planning_feature(context, leading_shape, device=self.device)
        action_tensor = _broadcast_planning_feature(action, leading_shape, device=self.device)
        observation_scale = observation_tensor.new_tensor(self._observation_scale)
        inputs = torch.cat(
            (
                observation_tensor / observation_scale,
                context_tensor / self._context_scale,
                action_tensor / self._action_scale,
            ),
            dim=-1,
        )
        return observation_tensor, inputs

    def predict_ensemble(
        self,
        observation: Tensor | list[float] | tuple[float, ...],
        context: Tensor | float | list[float] | tuple[float, ...],
        action: Tensor | float | list[float] | tuple[float, ...],
    ) -> tuple[Tensor, Tensor]:
        """Return physical-unit means/variances exactly like the source."""

        _, inputs = self._encoded_inputs(observation, context, action)
        normalized_means, normalized_log_variances = self.predict_normalized(inputs)
        target_scale = normalized_means.new_tensor(self._target_scale)
        means = normalized_means * target_scale
        variances = normalized_log_variances.exp() * target_scale.square()
        return means, variances

    def predict_mean_target(
        self,
        observation: Tensor,
        context: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Return the physical ensemble-mean target used by deterministic MPC."""

        _, inputs = self._encoded_inputs(observation, context, action)
        normalized_means, _ = self.predict_normalized(inputs)
        target_scale = normalized_means.new_tensor(self._target_scale)
        return normalized_means.mean(dim=0) * target_scale

    def project_next(self, observation: Tensor, predicted_delta: Tensor) -> Tensor:
        """Delegate projection to the source model's sealed implementation."""

        return cast(Tensor, self._source.project_next(observation, predicted_delta))


def _floating_tensor(value: Tensor | np.ndarray | list[float] | tuple[float, ...]) -> Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def _split_state(state: Tensor) -> tuple[Tensor, Tensor]:
    if state.ndim < 1 or state.shape[-1] != 4:
        raise ValueError("state must have trailing shape [cos(theta), sin(theta), angular_velocity, context]")
    return state[..., :3], state[..., 3:4]


def project_physical_observation(
    observation: Tensor,
    delta: Tensor,
    *,
    max_speed: float = PENDULUM_SEMANTICS.max_speed,
) -> Tensor:
    """Apply the protocol's unit-circle and angular-velocity projection."""

    if observation.shape[-1:] != (3,) or delta.shape[-1:] != (3,):
        raise ValueError("observation and delta must both have trailing dimension 3")
    proposed = observation + delta
    unit_pair = proposed[..., :2]
    norm = torch.linalg.vector_norm(unit_pair, dim=-1, keepdim=True).clamp_min(1e-8)
    projected_pair = unit_pair / norm
    angular_velocity = proposed[..., 2:3].clamp(-max_speed, max_speed)
    return torch.cat((projected_pair, angular_velocity), dim=-1)


def analytic_pendulum_step(
    observation: Tensor | np.ndarray | list[float] | tuple[float, ...],
    context: Tensor | float,
    intended_action: Tensor | np.ndarray | list[float] | tuple[float, ...],
    *,
    semantics: PendulumSemantics = PENDULUM_SEMANTICS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact Pendulum-v1 next observation, reward, and applied torque.

    Task A (context 0) applies the clipped intended torque. Task B (context 1)
    reverses it. Reward is computed from the pre-action physical state and the
    applied torque, exactly as Gymnasium Pendulum-v1 does.
    """

    physical = _floating_tensor(observation)
    action = torch.as_tensor(intended_action, dtype=physical.dtype, device=physical.device)
    task_context = torch.as_tensor(context, dtype=physical.dtype, device=physical.device)
    if physical.ndim < 1 or physical.shape[-1] != 3:
        raise ValueError("observation must have trailing dimension 3")
    if action.ndim == 0:
        action = action.unsqueeze(-1)
    if action.shape[-1] != 1:
        raise ValueError("intended_action must have trailing dimension 1")
    if task_context.ndim == 0:
        task_context = task_context.unsqueeze(-1)
    if task_context.shape[-1] != 1:
        raise ValueError("context must be scalar or have trailing dimension 1")

    cosine, sine, angular_velocity, intended, encoded_context = torch.broadcast_tensors(
        physical[..., 0],
        physical[..., 1],
        physical[..., 2],
        action[..., 0],
        task_context[..., 0],
    )
    theta = torch.atan2(sine, cosine)
    normalized_theta = torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi
    clipped_intended = intended.clamp(-semantics.max_torque, semantics.max_torque)
    direction = torch.where(
        encoded_context >= 0.5,
        clipped_intended.new_tensor(-1.0),
        clipped_intended.new_tensor(1.0),
    )
    applied = clipped_intended * direction

    cost = normalized_theta.square() + 0.1 * angular_velocity.square() + 0.001 * applied.square()
    acceleration = (
        3.0 * semantics.gravity / (2.0 * semantics.length) * torch.sin(theta)
        + 3.0 / (semantics.mass * semantics.length**2) * applied
    )
    next_angular_velocity = (angular_velocity + acceleration * semantics.time_step).clamp(
        -semantics.max_speed,
        semantics.max_speed,
    )
    next_theta = theta + next_angular_velocity * semantics.time_step
    next_observation = torch.stack(
        (torch.cos(next_theta), torch.sin(next_theta), next_angular_velocity),
        dim=-1,
    )
    return next_observation, -cost.unsqueeze(-1), applied.unsqueeze(-1)


def analytic_pendulum_state_step(
    state: Tensor | np.ndarray | list[float] | tuple[float, ...],
    intended_action: Tensor | np.ndarray | list[float] | tuple[float, ...],
    *,
    semantics: PendulumSemantics = PENDULUM_SEMANTICS,
) -> tuple[Tensor, Tensor]:
    """Step a combined physical-observation/context state."""

    state_tensor = _floating_tensor(state)
    observation, context = _split_state(state_tensor)
    next_observation, reward, _ = analytic_pendulum_step(
        observation,
        context,
        intended_action,
        semantics=semantics,
    )
    next_context = torch.broadcast_to(context, (*next_observation.shape[:-1], 1))
    return torch.cat((next_observation, next_context), dim=-1), reward


class AnalyticPendulumTensorDictStep(nn.Module):
    """Stateless TensorDict world model for the separately namespaced oracle."""

    def __init__(
        self,
        *,
        state_key: str = STATE_KEY,
        action_key: str = ACTION_KEY,
        reward_key: str = REWARD_KEY,
        semantics: PendulumSemantics = PENDULUM_SEMANTICS,
    ) -> None:
        super().__init__()
        self.state_key = state_key
        self.action_key = action_key
        self.reward_key = reward_key
        self.semantics = semantics

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_state, reward = analytic_pendulum_state_step(
            tensordict.get(self.state_key),
            tensordict.get(self.action_key),
            semantics=self.semantics,
        )
        return tensordict.set(self.state_key, next_state).set(self.reward_key, reward)


def _prediction_means(prediction: Any) -> Tensor:
    if isinstance(prediction, tuple):
        if not prediction:
            raise ValueError("predict_ensemble returned an empty tuple")
        means = prediction[0]
    elif isinstance(prediction, Mapping):
        for key in ("means", "mean", "member_means"):
            if key in prediction:
                means = prediction[key]
                break
        else:
            raise ValueError("prediction mapping has no means field")
    elif hasattr(prediction, "means"):
        means = prediction.means
    elif hasattr(prediction, "mean") and isinstance(prediction.mean, Tensor):
        means = prediction.mean
    else:
        means = prediction
    if not isinstance(means, Tensor):
        raise TypeError("prediction means must be a torch.Tensor")
    return means


class EnsembleMeanTensorDictStep(nn.Module):
    """Stateless deterministic rollout adapter around a probabilistic ensemble."""

    def __init__(
        self,
        predictor: EnsemblePredictionAPI,
        *,
        state_key: str = STATE_KEY,
        action_key: str = ACTION_KEY,
        reward_key: str = REWARD_KEY,
    ) -> None:
        super().__init__()
        if isinstance(predictor, nn.Module):
            self.predictor: EnsemblePredictionAPI = predictor
        else:
            object.__setattr__(self, "predictor", predictor)
        self.state_key = state_key
        self.action_key = action_key
        self.reward_key = reward_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        state = tensordict.get(self.state_key)
        action = tensordict.get(self.action_key)
        observation, context = _split_state(state)
        with torch.no_grad():
            predict_mean_target = getattr(self.predictor, "predict_mean_target", None)
            if callable(predict_mean_target):
                predicted_target = predict_mean_target(observation, context, action)
                if predicted_target.shape != (*state.shape[:-1], 4):
                    raise ValueError(
                        "predict_mean_target must have shape "
                        f"[{', '.join(str(value) for value in state.shape[:-1])}, 4]; "
                        f"received {tuple(predicted_target.shape)}"
                    )
            else:
                prediction = self.predictor.predict_ensemble(observation, context, action)
                member_means = _prediction_means(prediction)
                expected_shape = (state.shape[:-1], 4)
                if member_means.ndim != state.ndim + 1 or member_means.shape[1:] != (*expected_shape[0], 4):
                    raise ValueError(
                        "predict_ensemble means must be member-first with shape "
                        f"[members, {', '.join(str(value) for value in state.shape[:-1])}, 4]; "
                        f"received {tuple(member_means.shape)}"
                    )
                if member_means.shape[0] < 1:
                    raise ValueError("predict_ensemble must return at least one member")
                predicted_target = member_means.mean(dim=0)
            delta = predicted_target[..., :3]
            reward = predicted_target[..., 3:4]
            project_next = getattr(self.predictor, "project_next", None)
            if callable(project_next):
                next_observation = project_next(observation, delta)
            else:
                next_observation = project_physical_observation(observation, delta)
            if next_observation.shape != observation.shape:
                raise ValueError("projected next observation shape does not match the input observation")
            next_state = torch.cat((next_observation, context), dim=-1)
        return tensordict.set(self.state_key, next_state).set(self.reward_key, reward)


class PendulumModelBasedEnv(ModelBasedEnvBase):
    """Batch-unlocked model environment accepted by TorchRL CEMPlanner."""

    def __init__(
        self,
        world_model: nn.Module,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        state_key: str = STATE_KEY,
    ) -> None:
        self.state_key = state_key
        super().__init__(world_model, device=device)
        state_low = torch.tensor([-1.0, -1.0, -8.0, 0.0], device=self.device, dtype=dtype)
        state_high = torch.tensor([1.0, 1.0, 8.0, 1.0], device=self.device, dtype=dtype)
        state_spec = Bounded(low=state_low, high=state_high, shape=(4,), dtype=dtype, device=self.device)
        self.observation_spec = Composite({state_key: state_spec.clone()}, device=self.device)
        self.state_spec = Composite({state_key: state_spec.clone()}, device=self.device)
        self.action_spec = Bounded(
            low=-PENDULUM_SEMANTICS.max_torque,
            high=PENDULUM_SEMANTICS.max_torque,
            shape=(1,),
            dtype=dtype,
            device=self.device,
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=dtype, device=self.device)

    def _reset(self, tensordict: TensorDictBase | None = None, **_: Any) -> TensorDictBase:
        if tensordict is not None and self.state_key in tensordict.keys():
            state = tensordict.get(self.state_key).to(device=self.device, dtype=self.action_spec.dtype)
            return TensorDict(
                {self.state_key: state},
                batch_size=state.shape[:-1],
                device=self.device,
            )
        state = torch.tensor(
            [1.0, 0.0, 0.0, TASK_A_CONTEXT],
            device=self.device,
            dtype=self.action_spec.dtype,
        )
        return TensorDict({self.state_key: state}, batch_size=(), device=self.device)

    def _set_seed(self, seed: int | None) -> int | None:
        return seed


def make_learned_model_env(
    predictor: EnsemblePredictionAPI,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PendulumModelBasedEnv:
    """Construct the deterministic ensemble-mean learned rollout environment."""

    planning_predictor = StackedEnsemblePredictor.try_from(predictor) or predictor
    return PendulumModelBasedEnv(
        EnsembleMeanTensorDictStep(planning_predictor),
        device=device,
        dtype=dtype,
    )


def make_true_dynamics_env(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PendulumModelBasedEnv:
    """Construct the exact oracle rollout environment."""

    return PendulumModelBasedEnv(AnalyticPendulumTensorDictStep(), device=device, dtype=dtype)


class CEMController:
    """Receding-horizon controller with checkpointable TorchRL planner RNG.

    ``select`` supplies the scalar bridge used by WM-001's Prospect runtime
    adapter. TorchRL's public CEMPlanner result contains the selected action but
    not the final elite return, so that bridge reports ``0.0`` as non-evidentiary
    predicted-return metadata. Executed and held-out return metrics are computed
    by the experiment harness from real outcomes.
    """

    def __init__(
        self,
        env: PendulumModelBasedEnv,
        *,
        seed: int,
        budget: CEMBudget = WM001_CEM_BUDGET,
        allow_nonprotocol_budget: bool = False,
        require_exact_torchrl: bool = True,
    ) -> None:
        if not allow_nonprotocol_budget and budget != WM001_CEM_BUDGET:
            raise ValueError("non-protocol CEM budget requires allow_nonprotocol_budget=True")
        if require_exact_torchrl and torchrl_version != EXPECTED_TORCHRL_VERSION:
            raise RuntimeError(f"WM-001 requires torchrl {EXPECTED_TORCHRL_VERSION}; found {torchrl_version}")
        self.env = env
        self.budget = budget
        self.seed = int(seed)
        self.device = torch.device(env.device)
        self.dtype = env.action_spec.dtype
        self._planner = CEMPlanner(
            env=env,
            planning_horizon=budget.planning_horizon,
            optim_steps=budget.optim_steps,
            num_candidates=budget.num_candidates,
            top_k=budget.top_k,
            reward_key=("next", REWARD_KEY),
            action_key=ACTION_KEY,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        self._rng_state = generator.get_state().clone()
        self._actions_emitted = 0

    @property
    def version(self) -> str:
        predictor = getattr(self.env.world_model, "predictor", None)
        learned_version = getattr(predictor, "version", None)
        if learned_version is not None:
            return str(learned_version)
        return "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1"

    @contextmanager
    def _owned_global_rng(self) -> Iterator[None]:
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self._rng_state, device=self.device)
            else:
                torch.random.set_rng_state(self._rng_state)
            try:
                yield
            finally:
                if self.device.type == "cuda":
                    self._rng_state = torch.cuda.get_rng_state(device=self.device).clone()
                else:
                    self._rng_state = torch.random.get_rng_state().clone()

    def act(self, state: Tensor | np.ndarray | list[float] | tuple[float, ...]) -> Tensor:
        """Plan from one state or a batch and return bounded intended torque."""

        state_tensor = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        _split_state(state_tensor)
        initial = TensorDict(
            {STATE_KEY: state_tensor},
            batch_size=state_tensor.shape[:-1],
            device=self.device,
        )
        with self._owned_global_rng(), torch.inference_mode():
            planned = cast(Tensor, self._planner(initial).get(ACTION_KEY)).detach().clone()
        self._actions_emitted += int(state_tensor.numel() // state_tensor.shape[-1])
        return planned

    def act_numpy(self, state: Tensor | np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return self.act(state).cpu().numpy()

    def select(self, observation: np.ndarray, context: float) -> tuple[float, float]:
        physical = np.asarray(observation, dtype=np.float32)
        if physical.shape != (3,):
            raise ValueError("select observation must have shape (3,)")
        state = np.concatenate((physical, np.asarray([context], dtype=np.float32)))
        torque = float(self.act(state).item())
        return torque, 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "prospect.wm001.cem-controller-state.v1",
            "seed": self.seed,
            "actions_emitted": self._actions_emitted,
            "rng_state": self._rng_state.clone(),
            "budget": asdict(self.budget),
            "device_type": self.device.type,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "prospect.wm001.cem-controller-state.v1":
            raise ValueError("unsupported CEM controller state schema")
        if CEMBudget(**state["budget"]) != self.budget:
            raise ValueError("CEM controller budget differs from checkpoint")
        if state.get("device_type") != self.device.type:
            raise ValueError("CEM controller device type differs from checkpoint")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("invalid CEM controller RNG state")
        self.seed = int(state["seed"])
        self._actions_emitted = int(state["actions_emitted"])
        self._rng_state = rng_state.detach().cpu().clone()

    @property
    def rng_digest(self) -> str:
        return hashlib.sha256(self._rng_state.detach().cpu().numpy().tobytes()).hexdigest()


class UniformRandomController:
    """Checkpointable uniform intended-torque controller."""

    def __init__(
        self,
        *,
        seed: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.seed)
        self._actions_emitted = 0

    @property
    def version(self) -> str:
        return "wm001-uniform-random-controller-v1"

    def act(
        self,
        state: Tensor | np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> Tensor:
        batch_shape: tuple[int, ...] = ()
        if state is not None:
            state_tensor = torch.as_tensor(state)
            _split_state(state_tensor)
            batch_shape = tuple(state_tensor.shape[:-1])
        action = torch.empty((*batch_shape, 1), device=self.device, dtype=self.dtype)
        action.uniform_(
            -PENDULUM_SEMANTICS.max_torque,
            PENDULUM_SEMANTICS.max_torque,
            generator=self._generator,
        )
        self._actions_emitted += max(1, int(np.prod(batch_shape, dtype=np.int64)))
        return action

    def act_numpy(
        self,
        state: Tensor | np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> np.ndarray:
        return self.act(state).cpu().numpy()

    def select(self, observation: np.ndarray, context: float) -> tuple[float, float]:
        physical = np.asarray(observation, dtype=np.float32)
        if physical.shape != (3,):
            raise ValueError("select observation must have shape (3,)")
        state = np.concatenate((physical, np.asarray([context], dtype=np.float32)))
        torque = float(self.act(state).item())
        return torque, 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "prospect.wm001.random-controller-state.v1",
            "seed": self.seed,
            "actions_emitted": self._actions_emitted,
            "rng_state": self._generator.get_state().clone(),
            "device_type": self.device.type,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "prospect.wm001.random-controller-state.v1":
            raise ValueError("unsupported random controller state schema")
        if state.get("device_type") != self.device.type:
            raise ValueError("random controller device type differs from checkpoint")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("invalid random controller RNG state")
        self.seed = int(state["seed"])
        self._actions_emitted = int(state["actions_emitted"])
        self._generator.set_state(rng_state.detach().cpu())

    @property
    def rng_digest(self) -> str:
        state = self._generator.get_state().detach().cpu().numpy().tobytes()
        return hashlib.sha256(state).hexdigest()


def run_pendulum_conformance(
    *,
    samples_per_task: int = 128,
    seed: int = 99173,
    observation_atol: float = 2e-6,
    reward_atol: float = 1e-9,
    planner_observation_atol: float = 2e-6,
    planner_reward_atol: float = 2e-5,
) -> dict[str, Any]:
    """Compare semantic and actual float32 planner paths against Gymnasium."""

    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    unwrapped: Any = env.unwrapped
    expected_parameters = {
        "g": PENDULUM_SEMANTICS.gravity,
        "m": PENDULUM_SEMANTICS.mass,
        "l": PENDULUM_SEMANTICS.length,
        "dt": PENDULUM_SEMANTICS.time_step,
        "max_speed": PENDULUM_SEMANTICS.max_speed,
        "max_torque": PENDULUM_SEMANTICS.max_torque,
    }
    parameter_errors = {
        name: abs(float(getattr(unwrapped, name)) - expected) for name, expected in expected_parameters.items()
    }
    spec_horizon = getattr(env.spec, "max_episode_steps", None)
    rng = np.random.default_rng(seed)
    max_observation_error = 0.0
    max_reward_error = 0.0
    max_planner_observation_error = 0.0
    max_planner_reward_error = 0.0
    terminated_or_truncated = 0
    try:
        env.reset(seed=seed)
        for context in (TASK_A_CONTEXT, TASK_B_CONTEXT):
            for _ in range(samples_per_task):
                theta = float(rng.uniform(-np.pi, np.pi))
                angular_velocity = float(
                    rng.uniform(
                        -1.25 * PENDULUM_SEMANTICS.max_speed,
                        1.25 * PENDULUM_SEMANTICS.max_speed,
                    )
                )
                intended = float(
                    np.float32(
                        rng.uniform(
                            -1.5 * PENDULUM_SEMANTICS.max_torque,
                            1.5 * PENDULUM_SEMANTICS.max_torque,
                        )
                    )
                )
                unwrapped.state = np.asarray([theta, angular_velocity], dtype=np.float64)
                applied_to_gym = intended if context == TASK_A_CONTEXT else -intended
                actual_observation, actual_reward, terminated, truncated, _ = unwrapped.step(
                    np.asarray([applied_to_gym], dtype=np.float32)
                )
                source_observation = torch.tensor(
                    [np.cos(theta), np.sin(theta), angular_velocity],
                    dtype=torch.float64,
                )
                expected_observation, expected_reward, _ = analytic_pendulum_step(
                    source_observation,
                    context,
                    torch.tensor([intended], dtype=torch.float64),
                )
                planner_observation, planner_reward, _ = analytic_pendulum_step(
                    source_observation.to(torch.float32),
                    np.float32(context),
                    torch.tensor([intended], dtype=torch.float32),
                )
                observation_error = float(
                    np.max(np.abs(actual_observation.astype(np.float64) - expected_observation.detach().cpu().numpy()))
                )
                reward_error = abs(float(actual_reward) - float(expected_reward.item()))
                planner_observation_error = float(
                    np.max(
                        np.abs(
                            actual_observation.astype(np.float64)
                            - planner_observation.detach().cpu().numpy().astype(np.float64)
                        )
                    )
                )
                planner_reward_error = abs(float(actual_reward) - float(planner_reward.item()))
                max_observation_error = max(max_observation_error, observation_error)
                max_reward_error = max(max_reward_error, reward_error)
                max_planner_observation_error = max(
                    max_planner_observation_error,
                    planner_observation_error,
                )
                max_planner_reward_error = max(
                    max_planner_reward_error,
                    planner_reward_error,
                )
                terminated_or_truncated += int(bool(terminated) or bool(truncated))
    finally:
        env.close()

    parameters_match = all(error == 0.0 for error in parameter_errors.values())
    passed = (
        parameters_match
        and spec_horizon == PENDULUM_SEMANTICS.horizon
        and terminated_or_truncated == 0
        and max_observation_error <= observation_atol
        and max_reward_error <= reward_atol
        and max_planner_observation_error <= planner_observation_atol
        and max_planner_reward_error <= planner_reward_atol
    )
    report: dict[str, Any] = {
        "schema": "prospect.wm001.pendulum-conformance.v1",
        "environment_id": "Pendulum-v1",
        "gymnasium_version": gym.__version__,
        "seed": seed,
        "samples_per_task": samples_per_task,
        "cases": 2 * samples_per_task,
        "semantic_parameters": expected_parameters,
        "semantic_parameter_absolute_errors": parameter_errors,
        "spec_horizon": spec_horizon,
        "max_observation_absolute_error": max_observation_error,
        "max_reward_absolute_error": max_reward_error,
        "planner_dtype": "float32",
        "max_planner_observation_absolute_error": max_planner_observation_error,
        "max_planner_reward_absolute_error": max_planner_reward_error,
        "terminated_or_truncated_cases": terminated_or_truncated,
        "observation_atol": observation_atol,
        "reward_atol": reward_atol,
        "planner_observation_atol": planner_observation_atol,
        "planner_reward_atol": planner_reward_atol,
        "passed": passed,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


__all__ = [
    "ACTION_KEY",
    "CEMBudget",
    "CEMController",
    "EnsembleMeanTensorDictStep",
    "EnsemblePredictionAPI",
    "PENDULUM_SEMANTICS",
    "PendulumModelBasedEnv",
    "PendulumSemantics",
    "REWARD_KEY",
    "STATE_KEY",
    "StackedEnsemblePredictor",
    "TASK_A_CONTEXT",
    "TASK_B_CONTEXT",
    "UniformRandomController",
    "WM001_CEM_BUDGET",
    "VectorizedEnsembleCompatibilityError",
    "analytic_pendulum_state_step",
    "analytic_pendulum_step",
    "make_learned_model_env",
    "make_true_dynamics_env",
    "project_physical_observation",
    "run_pendulum_conformance",
]
