"""Probabilistic world-model backend for the sealed WM-001 experiment.

The module deliberately owns only tensor semantics and learned state.  Runtime
custody, transaction commit, environment interaction, and planning live in
their respective layers.

The public surface is:

* :class:`TransitionBatch` for identified real transitions;
* :class:`ProbabilisticEnsemble` for the shared five-member model;
* :func:`prepare_candidate` for non-mutating, fixed-budget learning;
* :func:`evaluate_mixture` for the sealed predictive metrics; and
* :func:`optimizer_to_bytes` / :func:`optimizer_from_bytes` for safe restart.

No ``torch.save`` or pickle payload is used.  Model and optimizer snapshots are
canonical JSON headers followed by checksummed, fixed-endian tensor bytes.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

MODEL_FORMAT = "prospect.wm001.probabilistic-ensemble.v1"
OPTIMIZER_FORMAT = "prospect.wm001.adamw.v1"
SAMPLING_FORMAT = "prospect.wm001.bootstrap-manifest.v1"
PREDICTION_FORMAT = "prospect.wm001.predictive-evidence.v1"
COVERAGE_SEMANTICS = "wm001-mixture-pit-binary64-count-v1"
_CONTAINER_MAGIC = b"PROSPECT-WM001\0"
_LOG_TWO_PI = math.log(2.0 * math.pi)


class ModelValidationError(ValueError):
    """Raised when learned state or transition tensors violate WM-001."""


@dataclass(frozen=True)
class FixedScaling:
    """Protocol-fixed, non-fitted divisors used by WM-001."""

    observation: tuple[float, float, float] = (1.0, 1.0, 8.0)
    context: float = 1.0
    action: float = 2.0
    delta: tuple[float, float, float] = (2.0, 2.0, 16.0)
    reward: float = 16.2736044

    @property
    def target(self) -> tuple[float, float, float, float]:
        return (*self.delta, self.reward)


@dataclass(frozen=True)
class WorldModelConfig:
    """Architecture and distribution contract.

    Defaults exactly match the sealed protocol.  Smaller configurations are
    accepted only so deterministic unit tests can exercise the same code path.
    A formal binding must call :meth:`require_formal`.
    """

    ensemble_members: int = 5
    input_dimension: int = 5
    hidden_dimensions: tuple[int, int] = (256, 256)
    output_dimension: int = 4
    log_variance_min: float = -10.0
    log_variance_max: float = 0.5
    scaling: FixedScaling = FixedScaling()

    def validate(self) -> None:
        if self.ensemble_members < 1:
            raise ModelValidationError("ensemble_members must be positive")
        if self.input_dimension != 5:
            raise ModelValidationError("WM-001 input dimension must be 5")
        if len(self.hidden_dimensions) != 2 or any(width < 1 for width in self.hidden_dimensions):
            raise ModelValidationError("WM-001 requires two positive hidden dimensions")
        if self.output_dimension != 4:
            raise ModelValidationError("WM-001 output dimension must be 4")
        if not self.log_variance_min < self.log_variance_max:
            raise ModelValidationError("invalid log-variance bounds")
        divisors = (
            *self.scaling.observation,
            self.scaling.context,
            self.scaling.action,
            *self.scaling.delta,
            self.scaling.reward,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in divisors):
            raise ModelValidationError("all fixed scaling divisors must be finite and positive")

    def require_formal(self) -> None:
        expected = WorldModelConfig()
        if self != expected:
            raise ModelValidationError(f"formal WM-001 requires the sealed model configuration: {expected!r}")


@dataclass(frozen=True)
class OptimizerConfig:
    """Sealed AdamW training configuration."""

    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 1e-5
    batch_size: int = 256
    gradient_clip_l2: float = 10.0

    def validate(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ModelValidationError("learning rate must be finite and positive")
        if len(self.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ModelValidationError("AdamW betas must be in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ModelValidationError("AdamW epsilon must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ModelValidationError("weight decay must be finite and non-negative")
        if self.batch_size < 1:
            raise ModelValidationError("batch size must be positive")
        if not math.isfinite(self.gradient_clip_l2) or self.gradient_clip_l2 <= 0.0:
            raise ModelValidationError("gradient clip must be finite and positive")

    def require_formal(self) -> None:
        expected = OptimizerConfig()
        if self != expected:
            raise ModelValidationError(f"formal WM-001 requires the sealed optimizer configuration: {expected!r}")


@dataclass(frozen=True)
class TransitionBatch:
    """Identified real transition tensors at one consistent row grain."""

    transition_ids: tuple[str, ...]
    observations: Tensor
    contexts: Tensor
    actions: Tensor
    next_observations: Tensor
    rewards: Tensor

    @classmethod
    def from_arrays(
        cls,
        *,
        transition_ids: Sequence[str],
        observations: object,
        contexts: object,
        actions: object,
        next_observations: object,
        rewards: object,
    ) -> TransitionBatch:
        batch = cls(
            transition_ids=tuple(transition_ids),
            observations=torch.as_tensor(observations, dtype=torch.float32).detach().cpu().contiguous(),
            contexts=_column_tensor(contexts),
            actions=_column_tensor(actions),
            next_observations=torch.as_tensor(next_observations, dtype=torch.float32).detach().cpu().contiguous(),
            rewards=_column_tensor(rewards),
        )
        batch.validate()
        return batch

    @classmethod
    def concatenate(cls, *batches: TransitionBatch) -> TransitionBatch:
        if not batches:
            raise ModelValidationError("at least one transition batch is required")
        combined = cls(
            transition_ids=tuple(identity for batch in batches for identity in batch.transition_ids),
            observations=torch.cat([batch.observations for batch in batches], dim=0),
            contexts=torch.cat([batch.contexts for batch in batches], dim=0),
            actions=torch.cat([batch.actions for batch in batches], dim=0),
            next_observations=torch.cat([batch.next_observations for batch in batches], dim=0),
            rewards=torch.cat([batch.rewards for batch in batches], dim=0),
        )
        combined.validate()
        return combined

    def validate(self) -> None:
        rows = len(self.transition_ids)
        if rows < 1:
            raise ModelValidationError("transition batch cannot be empty")
        if len(set(self.transition_ids)) != rows or any(not identity for identity in self.transition_ids):
            raise ModelValidationError("transition IDs must be unique non-empty strings")
        expected_shapes = {
            "observations": (rows, 3),
            "contexts": (rows, 1),
            "actions": (rows, 1),
            "next_observations": (rows, 3),
            "rewards": (rows, 1),
        }
        for name, expected in expected_shapes.items():
            tensor = getattr(self, name)
            if tensor.shape != expected:
                raise ModelValidationError(f"{name} shape {tuple(tensor.shape)} does not match {expected}")
            if tensor.dtype != torch.float32:
                raise ModelValidationError(f"{name} must use float32")
            if not bool(torch.isfinite(tensor).all()):
                raise ModelValidationError(f"{name} contains non-finite values")
        valid_context = (self.contexts == 0.0).logical_or(self.contexts == 1.0).logical_or(self.contexts == 2.0)
        if not bool(valid_context.all()):
            raise ModelValidationError("WM-001 contexts must be exactly 0, 1, or 2")
        if not bool((self.actions.abs() <= 2.0 + 1e-6).all()):
            raise ModelValidationError("intended actions exceed Pendulum bounds")

    def encoded(self, scaling: FixedScaling, *, device: torch.device | str | None = None) -> tuple[Tensor, Tensor]:
        self.validate()
        destination = torch.device(device) if device is not None else torch.device("cpu")
        observations = self.observations.to(destination)
        contexts = self.contexts.to(destination)
        actions = self.actions.to(destination)
        next_observations = self.next_observations.to(destination)
        rewards = self.rewards.to(destination)
        observation_scale = observations.new_tensor(scaling.observation)
        delta_scale = observations.new_tensor(scaling.delta)
        inputs = torch.cat(
            (
                observations / observation_scale,
                contexts / scaling.context,
                actions / scaling.action,
            ),
            dim=-1,
        )
        targets = torch.cat(
            (
                (next_observations - observations) / delta_scale,
                rewards / scaling.reward,
            ),
            dim=-1,
        )
        return inputs, targets

    def select(self, indices: Tensor | Sequence[int]) -> TransitionBatch:
        index = torch.as_tensor(indices, dtype=torch.long)
        identities = tuple(self.transition_ids[int(row)] for row in index.tolist())
        return TransitionBatch(
            transition_ids=identities,
            observations=self.observations[index],
            contexts=self.contexts[index],
            actions=self.actions[index],
            next_observations=self.next_observations[index],
            rewards=self.rewards[index],
        )

    def __len__(self) -> int:
        return len(self.transition_ids)


class _GaussianMember(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        hidden_a, hidden_b = config.hidden_dimensions
        self.network = nn.Sequential(
            nn.Linear(config.input_dimension, hidden_a),
            nn.SiLU(),
            nn.Linear(hidden_a, hidden_b),
            nn.SiLU(),
            nn.Linear(hidden_b, config.output_dimension * 2),
        )
        self._output_dimension = config.output_dimension
        self._log_variance_min = config.log_variance_min
        self._log_variance_max = config.log_variance_max
        with torch.no_grad():
            output = self.network[-1]
            assert isinstance(output, nn.Linear)
            output.bias[self._output_dimension :].fill_(-3.0)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.network(inputs)
        mean = outputs[..., : self._output_dimension]
        raw_log_variance = outputs[..., self._output_dimension :]
        log_variance = raw_log_variance.clamp(self._log_variance_min, self._log_variance_max)
        return mean, log_variance


class ProbabilisticEnsemble(nn.Module):
    """One shared diagonal-Gaussian ensemble for both contextual tasks."""

    def __init__(self, config: WorldModelConfig | None = None, *, initialization_seed: int = 0) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        self.config.validate()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(initialization_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(initialization_seed)
            self.members = nn.ModuleList(_GaussianMember(self.config) for _ in range(self.config.ensemble_members))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_sha256(self) -> str:
        return sha256(self.to_bytes()).hexdigest()

    @property
    def version(self) -> str:
        return f"wm001-sha256:{self.parameter_sha256}"

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized means and log variances, member-first."""

        member_outputs = [member(inputs) for member in self.members]
        means = torch.stack([output[0] for output in member_outputs], dim=0)
        log_variances = torch.stack([output[1] for output in member_outputs], dim=0)
        return means, log_variances

    def predict_normalized(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            return self(inputs.to(device=self.device, dtype=torch.float32))

    def predict_ensemble(
        self,
        observation: Tensor | Sequence[float],
        context: Tensor | float,
        action: Tensor | Sequence[float] | float,
    ) -> tuple[Tensor, Tensor]:
        """Return member-first ``(delta observation, reward)`` means/variances.

        Values are in physical, unscaled units.  For a batched observation
        ``[B, 3]``, each result has shape ``[members, B, 4]``.  For a single
        observation ``[3]``, each result has shape ``[members, 4]``.
        """

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if observation_tensor.shape[-1:] != (3,):
            raise ModelValidationError("observation must end in dimension 3")
        leading_shape = observation_tensor.shape[:-1]
        context_tensor = _broadcast_feature(context, leading_shape, self.device)
        action_tensor = _broadcast_feature(action, leading_shape, self.device)
        observation_scale = observation_tensor.new_tensor(self.config.scaling.observation)
        inputs = torch.cat(
            (
                observation_tensor / observation_scale,
                context_tensor / self.config.scaling.context,
                action_tensor / self.config.scaling.action,
            ),
            dim=-1,
        )
        normalized_means, normalized_log_variances = self.predict_normalized(inputs)
        target_scale = normalized_means.new_tensor(self.config.scaling.target)
        means = normalized_means * target_scale
        variances = normalized_log_variances.exp() * target_scale.square()
        return means, variances

    @staticmethod
    def project_next(observation: Tensor, predicted_delta: Tensor) -> Tensor:
        """Apply the sealed Pendulum rollout projection."""

        next_observation = observation + predicted_delta
        direction = next_observation[..., :2]
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
        projected_direction = direction / norm
        velocity = next_observation[..., 2:3].clamp(-8.0, 8.0)
        return torch.cat((projected_direction, velocity), dim=-1)

    def validate_finite(self) -> None:
        expected = {
            f"members.{member}.{suffix}" for member in range(self.config.ensemble_members) for suffix in _MEMBER_KEYS
        }
        actual = set(self.state_dict())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ModelValidationError(f"model state keys differ; missing={missing}, extra={extra}")
        for name, tensor in self.state_dict().items():
            if tensor.dtype != torch.float32:
                raise ModelValidationError(f"{name} must use float32")
            if not bool(torch.isfinite(tensor).all()):
                raise ModelValidationError(f"{name} contains non-finite values")

    def to_bytes(self) -> bytes:
        self.validate_finite()
        metadata = {"format": MODEL_FORMAT, "config": _config_dict(self.config)}
        return _encode_container(metadata, self.state_dict())

    @classmethod
    def from_bytes(cls, payload: bytes, *, device: torch.device | str | None = None) -> ProbabilisticEnsemble:
        metadata, tensors = _decode_container(payload)
        if metadata.get("format") != MODEL_FORMAT:
            raise ModelValidationError("snapshot has the wrong model format")
        config = _config_from_dict(metadata.get("config"))
        model = cls(config, initialization_seed=0)
        expected = model.state_dict()
        if set(tensors) != set(expected):
            raise ModelValidationError("snapshot tensor names do not match the declared architecture")
        for name, expected_tensor in expected.items():
            tensor = tensors[name]
            if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
                raise ModelValidationError(f"snapshot tensor {name} shape or dtype mismatch")
        model.load_state_dict(tensors, strict=True)
        if device is not None:
            model.to(device)
        model.validate_finite()
        if model.to_bytes() != payload:
            raise ModelValidationError("model snapshot is not canonical")
        return model

    def clone(self, *, device: torch.device | str | None = None) -> ProbabilisticEnsemble:
        return self.from_bytes(self.to_bytes(), device=device if device is not None else self.device)


@dataclass(frozen=True)
class PredictiveMetrics:
    mixture_nll_nats_per_target_dimension: float
    normalized_rmse: float
    interval_90_coverage: float
    interval_90_covered_target_count: int
    coverage_target_count: int
    coverage_semantics: str
    transition_count: int
    prediction_rows_sha256: str
    prediction_payload: bytes
    per_target_nll: tuple[float, float, float, float]
    per_target_rmse: tuple[float, float, float, float]
    per_target_coverage: tuple[float, float, float, float]


@dataclass(frozen=True)
class PreparedCandidate:
    """A trained candidate that has not been installed in any live runtime."""

    model: ProbabilisticEnsemble
    optimizer_bytes: bytes
    predecessor_parameter_sha256: str
    candidate_parameter_sha256: str
    predecessor_model_version: str
    candidate_model_version: str
    optimizer_steps: int
    training_mode: Literal["normal", "joint_target_permuted"]
    sampling_manifest: bytes
    sampling_manifest_sha256: str
    sampled_id_counts: tuple[tuple[str, int], ...]
    consumed_multiset_sha256: str
    target_permutation_sha256: str | None
    target_permutation_payload: bytes | None
    loss_history: tuple[float, ...]

    def validate(self) -> None:
        self.model.validate_finite()
        if self.model.parameter_sha256 != self.candidate_parameter_sha256:
            raise ModelValidationError("candidate parameter digest does not match candidate bytes")
        if self.model.version != self.candidate_model_version:
            raise ModelValidationError("candidate version does not match candidate bytes")
        if sha256(self.sampling_manifest).hexdigest() != self.sampling_manifest_sha256:
            raise ModelValidationError("sampling manifest digest mismatch")
        if len(self.consumed_multiset_sha256) != 64:
            raise ModelValidationError("consumed multiset digest is not SHA-256")
        if (self.target_permutation_sha256 is None) != (self.target_permutation_payload is None):
            raise ModelValidationError("target permutation digest and payload must be present together")
        if (
            self.target_permutation_payload is not None
            and sha256(self.target_permutation_payload).hexdigest() != self.target_permutation_sha256
        ):
            raise ModelValidationError("target permutation digest does not match its payload")
        if self.optimizer_steps != len(self.loss_history):
            raise ModelValidationError("loss history does not match optimizer-step count")
        if not all(math.isfinite(loss) for loss in self.loss_history):
            raise ModelValidationError("loss history contains non-finite values")


def make_optimizer(model: ProbabilisticEnsemble, config: OptimizerConfig | None = None) -> torch.optim.AdamW:
    config = config or OptimizerConfig()
    config.validate()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )
    optimizer._prospect_wm001_config = config  # type: ignore[attr-defined]
    return optimizer


def optimizer_to_bytes(
    model: ProbabilisticEnsemble,
    optimizer: torch.optim.AdamW,
    *,
    config: OptimizerConfig | None = None,
) -> bytes:
    """Serialize AdamW slots without pickle or executable object graphs."""

    config = config or _optimizer_config_from_instance(optimizer)
    config.validate()
    named_parameters = list(model.named_parameters())
    parameter_names = [name for name, _ in named_parameters]
    state = optimizer.state_dict()
    if len(state["param_groups"]) != 1:
        raise ModelValidationError("WM-001 optimizer must have exactly one parameter group")
    parameter_ids = state["param_groups"][0]["params"]
    if len(parameter_ids) != len(parameter_names):
        raise ModelValidationError("optimizer parameter count does not match model")
    tensors: dict[str, Tensor] = {}
    initialized: list[str] = []
    for parameter_id, (name, parameter) in zip(parameter_ids, named_parameters, strict=True):
        slot = state["state"].get(parameter_id)
        if slot is None:
            continue
        if set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ModelValidationError(f"unexpected AdamW state fields for {name}: {sorted(slot)}")
        step = torch.as_tensor(slot["step"]).detach().cpu()
        if step.numel() != 1 or not bool(torch.isfinite(step).all()):
            raise ModelValidationError(f"invalid AdamW step for {name}")
        exp_avg = torch.as_tensor(slot["exp_avg"]).detach().cpu()
        exp_avg_sq = torch.as_tensor(slot["exp_avg_sq"]).detach().cpu()
        if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
            raise ModelValidationError(f"AdamW slot shape mismatch for {name}")
        if not bool(torch.isfinite(exp_avg).all()) or not bool(torch.isfinite(exp_avg_sq).all()):
            raise ModelValidationError(f"non-finite AdamW slot for {name}")
        tensors[f"{name}.step"] = step
        tensors[f"{name}.exp_avg"] = exp_avg
        tensors[f"{name}.exp_avg_sq"] = exp_avg_sq
        initialized.append(name)
    metadata = {
        "format": OPTIMIZER_FORMAT,
        "config": asdict(config),
        "model_parameter_sha256": model.parameter_sha256,
        "parameter_names": parameter_names,
        "initialized_parameters": initialized,
    }
    return _encode_container(metadata, tensors)


def optimizer_from_bytes(
    model: ProbabilisticEnsemble,
    payload: bytes,
    *,
    expected_config: OptimizerConfig | None = None,
) -> torch.optim.AdamW:
    metadata, tensors = _decode_container(payload)
    if metadata.get("format") != OPTIMIZER_FORMAT:
        raise ModelValidationError("snapshot has the wrong optimizer format")
    config = _optimizer_config_from_dict(metadata.get("config"))
    if expected_config is not None and config != expected_config:
        raise ModelValidationError("optimizer snapshot configuration differs from expected configuration")
    if metadata.get("model_parameter_sha256") != model.parameter_sha256:
        raise ModelValidationError("optimizer snapshot is bound to different model parameters")
    optimizer = make_optimizer(model, config)
    named_parameters = list(model.named_parameters())
    parameter_names = [name for name, _ in named_parameters]
    if metadata.get("parameter_names") != parameter_names:
        raise ModelValidationError("optimizer parameter ordering differs from model")
    initialized = metadata.get("initialized_parameters")
    if not isinstance(initialized, list) or any(not isinstance(name, str) for name in initialized):
        raise ModelValidationError("invalid initialized optimizer parameter list")
    if len(initialized) != len(set(initialized)) or not set(initialized).issubset(parameter_names):
        raise ModelValidationError("invalid initialized optimizer parameter identities")
    expected_tensor_names = {f"{name}.{field}" for name in initialized for field in ("step", "exp_avg", "exp_avg_sq")}
    if set(tensors) != expected_tensor_names:
        raise ModelValidationError("optimizer slot tensor set differs from metadata")
    raw_state = optimizer.state_dict()
    parameter_ids = raw_state["param_groups"][0]["params"]
    rebuilt_state: dict[int, dict[str, Tensor]] = {}
    model_device = model.device
    parameter_by_name = dict(named_parameters)
    for parameter_id, name in zip(parameter_ids, parameter_names, strict=True):
        if name not in initialized:
            continue
        parameter = parameter_by_name[name]
        step = tensors[f"{name}.step"]
        exp_avg = tensors[f"{name}.exp_avg"]
        exp_avg_sq = tensors[f"{name}.exp_avg_sq"]
        if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
            raise ModelValidationError(f"optimizer slot shape mismatch for {name}")
        rebuilt_state[parameter_id] = {
            "step": step.cpu(),
            "exp_avg": exp_avg.to(model_device),
            "exp_avg_sq": exp_avg_sq.to(model_device),
        }
    raw_state["state"] = rebuilt_state
    optimizer.load_state_dict(raw_state)
    if optimizer_to_bytes(model, optimizer, config=config) != payload:
        raise ModelValidationError("optimizer snapshot is not canonical")
    return optimizer


def prepare_candidate(
    source: ProbabilisticEnsemble,
    transitions: TransitionBatch,
    *,
    optimizer_steps: int,
    bootstrap_seeds: Sequence[int],
    minibatch_order_seed: int | None = None,
    optimizer_config: OptimizerConfig | None = None,
    predecessor_optimizer_bytes: bytes | None = None,
    training_mode: Literal["normal", "joint_target_permuted"] = "normal",
    target_permutation_seed: int | None = None,
    balanced_tasks: bool = False,
    device: torch.device | str | None = None,
) -> PreparedCandidate:
    """Train a deep candidate while proving the source model stayed unchanged."""

    transitions.validate()
    source.validate_finite()
    source_bytes_before = source.to_bytes()
    predecessor_sha256 = sha256(source_bytes_before).hexdigest()
    predecessor_version = source.version
    if optimizer_steps < 1:
        raise ModelValidationError("optimizer_steps must be positive")
    optimizer_config = optimizer_config or OptimizerConfig()
    optimizer_config.validate()
    if len(bootstrap_seeds) != source.config.ensemble_members:
        raise ModelValidationError("one bootstrap seed is required per ensemble member")
    if len(set(int(seed) for seed in bootstrap_seeds)) != len(bootstrap_seeds):
        raise ModelValidationError("ensemble bootstrap seeds must be distinct")
    if training_mode not in {"normal", "joint_target_permuted"}:
        raise ModelValidationError(f"unsupported training mode: {training_mode}")
    if training_mode == "joint_target_permuted" and target_permutation_seed is None:
        raise ModelValidationError("corrupted-target training requires a target permutation seed")
    if training_mode == "normal" and target_permutation_seed is not None:
        raise ModelValidationError("normal training cannot declare a target permutation seed")
    if balanced_tasks and optimizer_config.batch_size % 2 != 0:
        raise ModelValidationError("balanced task batches require an even batch size")

    training_device = torch.device(device) if device is not None else source.device
    candidate = ProbabilisticEnsemble.from_bytes(source_bytes_before, device=training_device)
    if predecessor_optimizer_bytes is None:
        optimizer = make_optimizer(candidate, optimizer_config)
    else:
        optimizer = optimizer_from_bytes(candidate, predecessor_optimizer_bytes, expected_config=optimizer_config)
    inputs, targets = transitions.encoded(candidate.config.scaling, device=training_device)
    permutation_digest: str | None = None
    permutation_payload: bytes | None = None
    if training_mode == "joint_target_permuted":
        permutation_generator = torch.Generator(device="cpu")
        permutation_generator.manual_seed(int(target_permutation_seed))
        permutation = torch.randperm(len(transitions), generator=permutation_generator)
        targets = targets[permutation.to(training_device)]
        permutation_payload = permutation.numpy().astype("<u4", copy=False).tobytes(order="C")
        permutation_digest = sha256(permutation_payload).hexdigest()

    task_indices: tuple[Tensor, Tensor] | None = None
    if balanced_tasks:
        context_cpu = transitions.contexts[:, 0]
        task_a = torch.nonzero(context_cpu == 0.0, as_tuple=False)[:, 0]
        task_b = torch.nonzero(context_cpu == 1.0, as_tuple=False)[:, 0]
        if task_a.numel() == 0 or task_b.numel() == 0:
            raise ModelValidationError("balanced training requires at least one transition from each task")
        task_indices = (task_a, task_b)

    generators: list[torch.Generator] = []
    for seed in bootstrap_seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        generators.append(generator)
    manifest_indices = np.empty(
        (optimizer_steps, candidate.config.ensemble_members, optimizer_config.batch_size),
        dtype="<u4",
    )
    for step in range(optimizer_steps):
        for member_index, generator in enumerate(generators):
            indices = _sample_indices(
                len(transitions),
                optimizer_config.batch_size,
                generator,
                task_indices=task_indices,
            )
            manifest_indices[step, member_index] = indices.numpy().astype("<u4", copy=False)
    if minibatch_order_seed is not None:
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(int(minibatch_order_seed))
        step_order = torch.randperm(optimizer_steps, generator=order_generator).numpy()
        manifest_indices = manifest_indices[step_order].copy()

    losses: list[float] = []
    candidate.train()
    for step in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        member_losses: list[Tensor] = []
        for member_index, member in enumerate(candidate.members):
            indices = torch.from_numpy(manifest_indices[step, member_index].astype(np.int64))
            member_inputs = inputs[indices.to(training_device)]
            member_targets = targets[indices.to(training_device)]
            means, log_variances = member(member_inputs)
            member_nll = 0.5 * (
                _LOG_TWO_PI + log_variances + (member_targets - means).square() * (-log_variances).exp()
            )
            member_losses.append(member_nll.mean())
        loss = torch.stack(member_losses).mean()
        if not bool(torch.isfinite(loss)):
            raise ModelValidationError(f"non-finite candidate loss at optimizer step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), optimizer_config.gradient_clip_l2)
        if not bool(torch.isfinite(gradient_norm)):
            raise ModelValidationError(f"non-finite candidate gradient at optimizer step {step}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    candidate.eval()
    candidate.validate_finite()
    optimizer_payload = optimizer_to_bytes(candidate, optimizer, config=optimizer_config)
    sampling_manifest = _encode_sampling_manifest(manifest_indices, transitions.transition_ids)
    sampled_counts_array = np.bincount(manifest_indices.reshape(-1), minlength=len(transitions))
    sampled_id_counts = tuple(
        (transition_id, int(count))
        for transition_id, count in zip(transitions.transition_ids, sampled_counts_array, strict=True)
        if count
    )
    consumed_multiset_sha256 = _consumed_multiset_sha256(
        manifest_indices,
        transitions.transition_ids,
    )
    if source.to_bytes() != source_bytes_before or source.version != predecessor_version:
        raise ModelValidationError("candidate preparation mutated the source model")
    result = PreparedCandidate(
        model=candidate,
        optimizer_bytes=optimizer_payload,
        predecessor_parameter_sha256=predecessor_sha256,
        candidate_parameter_sha256=candidate.parameter_sha256,
        predecessor_model_version=predecessor_version,
        candidate_model_version=candidate.version,
        optimizer_steps=optimizer_steps,
        training_mode=training_mode,
        sampling_manifest=sampling_manifest,
        sampling_manifest_sha256=sha256(sampling_manifest).hexdigest(),
        sampled_id_counts=sampled_id_counts,
        consumed_multiset_sha256=consumed_multiset_sha256,
        target_permutation_sha256=permutation_digest,
        target_permutation_payload=permutation_payload,
        loss_history=tuple(losses),
    )
    result.validate()
    return result


def evaluate_mixture(
    model: ProbabilisticEnsemble,
    transitions: TransitionBatch,
    *,
    batch_size: int = 4096,
    device: torch.device | str | None = None,
) -> PredictiveMetrics:
    """Compute exact sealed mixture NLL, RMSE, and mixture-PIT coverage."""

    transitions.validate()
    model.validate_finite()
    if batch_size < 1:
        raise ModelValidationError("evaluation batch size must be positive")
    evaluation_device = torch.device(device) if device is not None else model.device
    original_device = model.device
    if evaluation_device != original_device:
        model = model.clone(device=evaluation_device)
    inputs, targets = transitions.encoded(model.config.scaling, device=evaluation_device)
    log_prob_rows: list[Tensor] = []
    squared_error_rows: list[Tensor] = []
    mean_rows: list[Tensor] = []
    log_variance_rows: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(transitions), batch_size):
            stop = min(start + batch_size, len(transitions))
            batch_targets = targets[start:stop]
            means, log_variances = model(inputs[start:stop])
            target_by_member = batch_targets.unsqueeze(0)
            member_log_prob = -0.5 * (
                _LOG_TWO_PI + log_variances + (target_by_member - means).square() * (-log_variances).exp()
            )
            mixture_log_prob = torch.logsumexp(member_log_prob, dim=0) - math.log(model.config.ensemble_members)
            ensemble_mean = means.mean(dim=0)
            squared_error = (batch_targets - ensemble_mean).square()
            log_prob_rows.append(mixture_log_prob.cpu())
            squared_error_rows.append(squared_error.cpu())
            mean_rows.append(means.cpu())
            log_variance_rows.append(log_variances.cpu())

    log_prob = torch.cat(log_prob_rows, dim=0).to(torch.float64)
    squared_error = torch.cat(squared_error_rows, dim=0).to(torch.float64)
    per_target_nll_tensor = -log_prob.mean(dim=0)
    per_target_rmse_tensor = squared_error.mean(dim=0).sqrt()
    values = (per_target_nll_tensor, per_target_rmse_tensor)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ModelValidationError("predictive metrics contain non-finite values")
    member_means = torch.cat(mean_rows, dim=1)
    member_log_variances = torch.cat(log_variance_rows, dim=1)
    prediction_payload = encode_prediction_evidence(
        transitions.transition_ids,
        targets.cpu(),
        member_means,
        member_log_variances,
    )
    _, stored_targets, stored_means, stored_log_variances = decode_prediction_evidence(prediction_payload)
    covered_target_count, coverage_target_count, covered_by_target = (
        canonical_interval_90_coverage_counts(
            stored_targets,
            stored_means,
            stored_log_variances,
        )
    )
    per_target_coverage = tuple(
        covered / len(transitions)
        for covered in covered_by_target
    )
    return PredictiveMetrics(
        mixture_nll_nats_per_target_dimension=float(per_target_nll_tensor.mean()),
        normalized_rmse=float(squared_error.mean().sqrt()),
        interval_90_coverage=covered_target_count / coverage_target_count,
        interval_90_covered_target_count=covered_target_count,
        coverage_target_count=coverage_target_count,
        coverage_semantics=COVERAGE_SEMANTICS,
        transition_count=len(transitions),
        prediction_rows_sha256=sha256(prediction_payload).hexdigest(),
        prediction_payload=prediction_payload,
        per_target_nll=tuple(float(value) for value in per_target_nll_tensor),
        per_target_rmse=tuple(float(value) for value in per_target_rmse_tensor),
        per_target_coverage=per_target_coverage,
    )


def binary64_pit_is_in_inclusive_90_percent_interval(pit: float) -> bool:
    """Classify one finite binary64 PIT value using exact rational endpoints."""

    if not math.isfinite(pit):
        raise ModelValidationError("mixture PIT must be finite")
    numerator, denominator = pit.as_integer_ratio()
    return 20 * numerator >= denominator and 20 * numerator <= 19 * denominator


def canonical_binary64_mixture_pit(
    target: float,
    member_means: Sequence[float],
    member_log_variances: Sequence[float],
) -> float:
    """Compute one mixture PIT in the sealed scalar and member order."""

    if (
        not math.isfinite(target)
        or len(member_means) == 0
        or len(member_means) != len(member_log_variances)
        or not all(math.isfinite(value) for value in (*member_means, *member_log_variances))
    ):
        raise ModelValidationError("mixture PIT inputs must be finite matched member values")
    sqrt_two = math.sqrt(2.0)
    member_cdfs = []
    for mean, log_variance in zip(member_means, member_log_variances, strict=True):
        z_score = (target - mean) * math.exp(-0.5 * log_variance)
        member_cdfs.append(0.5 * (1.0 + math.erf(z_score / sqrt_two)))
    mixture_pit = math.fsum(member_cdfs) / len(member_cdfs)
    if not math.isfinite(mixture_pit):
        raise ModelValidationError("mixture PIT is non-finite")
    return mixture_pit


def canonical_interval_90_coverage_counts(
    normalized_targets: Tensor,
    member_means: Tensor,
    member_log_variances: Tensor,
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Apply the sealed scalar-binary64 coverage contract to float32 evidence.

    Tensor traversal and member reduction order are deliberately explicit.  The
    formal producer calls this only after a prediction payload has been encoded
    and decoded, making these values a function of the exact persisted bytes.
    """

    targets = normalized_targets.detach().cpu().contiguous()
    means = member_means.detach().cpu().contiguous()
    log_variances = member_log_variances.detach().cpu().contiguous()
    if targets.ndim != 2 or targets.shape[1] != 4:
        raise ModelValidationError("coverage targets must have shape [rows, 4]")
    if means.ndim != 3 or means.shape[1:] != targets.shape or means.shape[0] < 1:
        raise ModelValidationError("coverage means must have shape [members, rows, 4]")
    if log_variances.shape != means.shape:
        raise ModelValidationError("coverage log variances must match means")
    if not all(
        tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
        for tensor in (targets, means, log_variances)
    ):
        raise ModelValidationError("coverage evidence tensors must be finite float32")

    member_count = int(means.shape[0])
    row_count = int(targets.shape[0])
    covered_by_target = [0, 0, 0, 0]
    for row_index in range(row_count):
        for target_index in range(4):
            target = float(targets[row_index, target_index])
            mixture_pit = canonical_binary64_mixture_pit(
                target,
                tuple(
                    float(means[member_index, row_index, target_index])
                    for member_index in range(member_count)
                ),
                tuple(
                    float(log_variances[member_index, row_index, target_index])
                    for member_index in range(member_count)
                ),
            )
            if binary64_pit_is_in_inclusive_90_percent_interval(mixture_pit):
                covered_by_target[target_index] += 1

    covered_target_count = sum(covered_by_target)
    coverage_target_count = 4 * row_count
    return (
        covered_target_count,
        coverage_target_count,
        (
            covered_by_target[0],
            covered_by_target[1],
            covered_by_target[2],
            covered_by_target[3],
        ),
    )


def save_bytes(path: str | Path, payload: bytes) -> None:
    """Persist already-canonical bytes; callers own atomic file replacement."""

    Path(path).write_bytes(payload)


def _column_tensor(values: object) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor.contiguous()


def _broadcast_feature(value: object, leading_shape: torch.Size, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    target_shape = (*leading_shape, 1)
    if tensor.ndim == 0:
        tensor = tensor.reshape((1,) * len(target_shape)).expand(target_shape)
    elif tensor.shape == leading_shape:
        tensor = tensor.unsqueeze(-1)
    elif tensor.shape != target_shape:
        try:
            tensor = torch.broadcast_to(tensor, target_shape)
        except RuntimeError as error:
            message = f"feature shape {tuple(tensor.shape)} cannot broadcast to {target_shape}"
            raise ModelValidationError(message) from error
    return tensor


def _sample_indices(
    row_count: int,
    batch_size: int,
    generator: torch.Generator,
    *,
    task_indices: tuple[Tensor, Tensor] | None,
) -> Tensor:
    if task_indices is None:
        return torch.randint(row_count, (batch_size,), generator=generator, dtype=torch.long)
    half = batch_size // 2
    task_a, task_b = task_indices
    selected_a = task_a[torch.randint(task_a.numel(), (half,), generator=generator)]
    selected_b = task_b[torch.randint(task_b.numel(), (half,), generator=generator)]
    # Shuffle within the batch using the same member-specific generator.
    combined = torch.cat((selected_a, selected_b))
    return combined[torch.randperm(batch_size, generator=generator)]


def _encode_sampling_manifest(indices: np.ndarray, transition_ids: tuple[str, ...]) -> bytes:
    identity_payload = json.dumps(
        list(transition_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata = {
        "format": SAMPLING_FORMAT,
        "shape": list(indices.shape),
        "dtype": "uint32-le",
        "transition_ids_sha256": sha256(identity_payload).hexdigest(),
        "payload_sha256": sha256(indices.tobytes(order="C")).hexdigest(),
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _CONTAINER_MAGIC + struct.pack(">Q", len(header)) + header + indices.tobytes(order="C")


def _consumed_multiset_sha256(
    indices: np.ndarray,
    transition_ids: tuple[str, ...],
) -> str:
    """Hash exact optimizer consumption order, retaining every repetition."""

    encoded = tuple(identity.encode("utf-8") + b"\n" for identity in transition_ids)
    digest = sha256()
    for index in indices.reshape(-1):
        digest.update(encoded[int(index)])
    return digest.hexdigest()


def encode_prediction_evidence(
    transition_ids: Sequence[str],
    normalized_targets: Tensor,
    member_means: Tensor,
    member_log_variances: Tensor,
) -> bytes:
    """Encode the raw tensors required to recompute every predictive metric."""

    identities = tuple(transition_ids)
    row_count = len(identities)
    if len(set(identities)) != row_count or any(not identity for identity in identities):
        raise ModelValidationError("prediction evidence transition IDs must be unique and nonempty")
    targets = normalized_targets.detach().cpu().contiguous()
    means = member_means.detach().cpu().contiguous()
    log_variances = member_log_variances.detach().cpu().contiguous()
    if targets.shape != (row_count, 4):
        raise ModelValidationError("prediction evidence targets must have shape [rows, 4]")
    if means.ndim != 3 or means.shape[1:] != targets.shape:
        raise ModelValidationError("prediction evidence means must have shape [members, rows, 4]")
    if log_variances.shape != means.shape:
        raise ModelValidationError("prediction evidence log variances must match means")
    if not all(
        tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
        for tensor in (targets, means, log_variances)
    ):
        raise ModelValidationError("prediction evidence tensors must be finite float32")
    return _encode_container(
        {
            "format": PREDICTION_FORMAT,
            "transition_ids": list(identities),
        },
        {
            "member_log_variances": log_variances,
            "member_means": means,
            "normalized_targets": targets,
        },
    )


def decode_prediction_evidence(
    payload: bytes,
) -> tuple[tuple[str, ...], Tensor, Tensor, Tensor]:
    """Safely decode and canonicality-check predictive evidence."""

    metadata, tensors = _decode_container(payload)
    if set(metadata) != {"format", "transition_ids"} or metadata.get("format") != PREDICTION_FORMAT:
        raise ModelValidationError("prediction evidence metadata has the wrong schema")
    raw_ids = metadata.get("transition_ids")
    if not isinstance(raw_ids, list) or any(not isinstance(identity, str) for identity in raw_ids):
        raise ModelValidationError("prediction evidence transition IDs are malformed")
    if set(tensors) != {"normalized_targets", "member_means", "member_log_variances"}:
        raise ModelValidationError("prediction evidence tensor set differs from the contract")
    identities = tuple(raw_ids)
    targets = tensors["normalized_targets"]
    means = tensors["member_means"]
    log_variances = tensors["member_log_variances"]
    canonical = encode_prediction_evidence(identities, targets, means, log_variances)
    if canonical != payload:
        raise ModelValidationError("prediction evidence is not canonical")
    return identities, targets, means, log_variances


def _config_dict(config: WorldModelConfig) -> dict[str, object]:
    return {
        "ensemble_members": config.ensemble_members,
        "input_dimension": config.input_dimension,
        "hidden_dimensions": list(config.hidden_dimensions),
        "output_dimension": config.output_dimension,
        "log_variance_min": config.log_variance_min,
        "log_variance_max": config.log_variance_max,
        "scaling": {
            "observation": list(config.scaling.observation),
            "context": config.scaling.context,
            "action": config.scaling.action,
            "delta": list(config.scaling.delta),
            "reward": config.scaling.reward,
        },
    }


def _config_from_dict(raw: object) -> WorldModelConfig:
    if not isinstance(raw, dict):
        raise ModelValidationError("model configuration must be an object")
    expected_keys = {
        "ensemble_members",
        "input_dimension",
        "hidden_dimensions",
        "output_dimension",
        "log_variance_min",
        "log_variance_max",
        "scaling",
    }
    if set(raw) != expected_keys:
        raise ModelValidationError("model configuration fields differ from the safe schema")
    scaling_raw = raw["scaling"]
    if not isinstance(scaling_raw, dict) or set(scaling_raw) != {"observation", "context", "action", "delta", "reward"}:
        raise ModelValidationError("fixed scaling fields differ from the safe schema")
    try:
        config = WorldModelConfig(
            ensemble_members=int(raw["ensemble_members"]),
            input_dimension=int(raw["input_dimension"]),
            hidden_dimensions=tuple(int(value) for value in raw["hidden_dimensions"]),  # type: ignore[arg-type]
            output_dimension=int(raw["output_dimension"]),
            log_variance_min=float(raw["log_variance_min"]),
            log_variance_max=float(raw["log_variance_max"]),
            scaling=FixedScaling(
                observation=tuple(float(value) for value in scaling_raw["observation"]),  # type: ignore[arg-type]
                context=float(scaling_raw["context"]),
                action=float(scaling_raw["action"]),
                delta=tuple(float(value) for value in scaling_raw["delta"]),  # type: ignore[arg-type]
                reward=float(scaling_raw["reward"]),
            ),
        )
    except (TypeError, ValueError) as error:
        raise ModelValidationError("invalid model configuration value") from error
    config.validate()
    return config


def _optimizer_config_from_instance(optimizer: torch.optim.AdamW) -> OptimizerConfig:
    attached = getattr(optimizer, "_prospect_wm001_config", None)
    if isinstance(attached, OptimizerConfig):
        return attached
    if len(optimizer.param_groups) != 1:
        raise ModelValidationError("WM-001 optimizer must have one parameter group")
    group = optimizer.param_groups[0]
    return OptimizerConfig(
        learning_rate=float(group["lr"]),
        betas=tuple(float(value) for value in group["betas"]),
        epsilon=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
    )


def _optimizer_config_from_dict(raw: object) -> OptimizerConfig:
    if not isinstance(raw, dict):
        raise ModelValidationError("optimizer configuration must be an object")
    expected = {"learning_rate", "betas", "epsilon", "weight_decay", "batch_size", "gradient_clip_l2"}
    if set(raw) != expected:
        raise ModelValidationError("optimizer configuration fields differ from the safe schema")
    try:
        config = OptimizerConfig(
            learning_rate=float(raw["learning_rate"]),
            betas=tuple(float(value) for value in raw["betas"]),  # type: ignore[arg-type]
            epsilon=float(raw["epsilon"]),
            weight_decay=float(raw["weight_decay"]),
            batch_size=int(raw["batch_size"]),
            gradient_clip_l2=float(raw["gradient_clip_l2"]),
        )
    except (TypeError, ValueError) as error:
        raise ModelValidationError("invalid optimizer configuration value") from error
    config.validate()
    return config


_DTYPE_TO_NUMPY: Mapping[torch.dtype, np.dtype[object]] = {
    torch.float32: np.dtype("<f4"),
    torch.float64: np.dtype("<f8"),
    torch.int64: np.dtype("<i8"),
    torch.int32: np.dtype("<i4"),
    torch.uint8: np.dtype("u1"),
    torch.bool: np.dtype("?"),
}
_NUMPY_NAME_TO_TORCH: Mapping[str, torch.dtype] = {
    dtype.str: torch_dtype for torch_dtype, dtype in _DTYPE_TO_NUMPY.items()
}


def _encode_container(metadata: Mapping[str, object], tensors: Mapping[str, Tensor]) -> bytes:
    entries: list[dict[str, object]] = []
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        numpy_dtype = _DTYPE_TO_NUMPY.get(tensor.dtype)
        if numpy_dtype is None:
            raise ModelValidationError(f"unsupported canonical tensor dtype: {tensor.dtype}")
        array = tensor.numpy().astype(numpy_dtype, copy=False)
        payload = array.tobytes(order="C")
        entries.append(
            {
                "name": name,
                "dtype": numpy_dtype.str,
                "shape": list(tensor.shape),
                "offset": offset,
                "bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        )
        payloads.append(payload)
        offset += len(payload)
    header_object = {"metadata": dict(metadata), "tensors": entries, "payload_bytes": offset}
    header = json.dumps(header_object, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _CONTAINER_MAGIC + struct.pack(">Q", len(header)) + header + b"".join(payloads)


def _decode_container(payload: bytes) -> tuple[dict[str, object], dict[str, Tensor]]:
    prefix_length = len(_CONTAINER_MAGIC) + 8
    if len(payload) < prefix_length or not payload.startswith(_CONTAINER_MAGIC):
        raise ModelValidationError("snapshot has invalid canonical-container magic")
    header_length = struct.unpack(">Q", payload[len(_CONTAINER_MAGIC) : prefix_length])[0]
    if header_length > 16 * 1024 * 1024 or prefix_length + header_length > len(payload):
        raise ModelValidationError("snapshot header length is invalid")
    try:
        header = json.loads(payload[prefix_length : prefix_length + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelValidationError("snapshot header is not valid canonical JSON") from error
    if not isinstance(header, dict) or set(header) != {"metadata", "tensors", "payload_bytes"}:
        raise ModelValidationError("snapshot header fields differ from safe schema")
    metadata = header["metadata"]
    entries = header["tensors"]
    payload_bytes = header["payload_bytes"]
    if not isinstance(metadata, dict) or not isinstance(entries, list) or not isinstance(payload_bytes, int):
        raise ModelValidationError("snapshot header types are invalid")
    data = payload[prefix_length + header_length :]
    if payload_bytes != len(data):
        raise ModelValidationError("snapshot tensor payload length mismatch")
    tensors: dict[str, Tensor] = {}
    expected_offset = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "dtype", "shape", "offset", "bytes", "sha256"}:
            raise ModelValidationError("snapshot tensor entry differs from safe schema")
        name = entry["name"]
        dtype_name = entry["dtype"]
        shape = entry["shape"]
        offset = entry["offset"]
        byte_count = entry["bytes"]
        expected_sha256 = entry["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or name in tensors
            or not isinstance(dtype_name, str)
            or dtype_name not in _NUMPY_NAME_TO_TORCH
            or not isinstance(shape, list)
            or any(not isinstance(size, int) or size < 0 for size in shape)
            or not isinstance(offset, int)
            or not isinstance(byte_count, int)
            or not isinstance(expected_sha256, str)
        ):
            raise ModelValidationError("snapshot tensor metadata is invalid")
        if offset != expected_offset or byte_count < 0 or offset + byte_count > len(data):
            raise ModelValidationError("snapshot tensor offsets are invalid")
        tensor_payload = data[offset : offset + byte_count]
        if sha256(tensor_payload).hexdigest() != expected_sha256:
            raise ModelValidationError(f"snapshot tensor checksum mismatch: {name}")
        numpy_dtype = np.dtype(dtype_name)
        element_count = math.prod(shape)
        if element_count * numpy_dtype.itemsize != byte_count:
            raise ModelValidationError(f"snapshot tensor byte size does not match shape: {name}")
        array = np.frombuffer(tensor_payload, dtype=numpy_dtype).reshape(shape).copy()
        tensor = torch.from_numpy(array)
        expected_torch_dtype = _NUMPY_NAME_TO_TORCH[dtype_name]
        if tensor.dtype != expected_torch_dtype:
            tensor = tensor.to(expected_torch_dtype)
        tensors[name] = tensor
        expected_offset += byte_count
    if expected_offset != len(data):
        raise ModelValidationError("snapshot contains unclaimed tensor bytes")
    return metadata, tensors


_MEMBER_KEYS = (
    "network.0.weight",
    "network.0.bias",
    "network.2.weight",
    "network.2.bias",
    "network.4.weight",
    "network.4.bias",
)


__all__ = (
    "COVERAGE_SEMANTICS",
    "FixedScaling",
    "ModelValidationError",
    "OptimizerConfig",
    "PredictiveMetrics",
    "PreparedCandidate",
    "ProbabilisticEnsemble",
    "TransitionBatch",
    "WorldModelConfig",
    "binary64_pit_is_in_inclusive_90_percent_interval",
    "canonical_binary64_mixture_pit",
    "canonical_interval_90_coverage_counts",
    "decode_prediction_evidence",
    "encode_prediction_evidence",
    "evaluate_mixture",
    "make_optimizer",
    "optimizer_from_bytes",
    "optimizer_to_bytes",
    "prepare_candidate",
    "save_bytes",
)
