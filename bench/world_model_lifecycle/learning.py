"""Concrete transactional learning adapter for the WM-001 world model."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch

from prospect.domain import (
    AgentSnapshot,
    Belief,
    EpistemicTransition,
    TimePoint,
    UpdateReceipt,
    UpdateStatus,
)
from prospect.runtime import (
    ModelState,
    PreparedLearningUpdate,
    TransactionalLearner,
    VersionedModelOwner,
)

from .model import (
    OptimizerConfig,
    PreparedCandidate,
    ProbabilisticEnsemble,
    TransitionBatch,
    WorldModelConfig,
    make_optimizer,
    optimizer_from_bytes,
    optimizer_to_bytes,
    prepare_candidate,
)
from .runtime_lane import PredictiveBackend, transition_arrays

_FORMAT = "prospect.wm001.owned-model-state.v1"
_MAGIC = b"PROSPECT-WM001-STATE\0"


class WorldModelStateError(ValueError):
    """The compound model/optimizer state is malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class CompoundModelState:
    """Canonical bytes for the model parameters and their AdamW slots."""

    model_bytes: bytes
    optimizer_bytes: bytes

    def to_bytes(self) -> bytes:
        metadata = {
            "format": _FORMAT,
            "model_bytes": len(self.model_bytes),
            "model_sha256": hashlib.sha256(self.model_bytes).hexdigest(),
            "optimizer_bytes": len(self.optimizer_bytes),
            "optimizer_sha256": hashlib.sha256(self.optimizer_bytes).hexdigest(),
        }
        header = json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return _MAGIC + struct.pack(">Q", len(header)) + header + self.model_bytes + self.optimizer_bytes

    @classmethod
    def from_bytes(cls, payload: bytes) -> CompoundModelState:
        if not isinstance(payload, bytes) or not payload.startswith(_MAGIC):
            raise WorldModelStateError("owned model state has the wrong magic")
        offset = len(_MAGIC)
        if len(payload) < offset + 8:
            raise WorldModelStateError("owned model state is truncated")
        (header_size,) = struct.unpack(">Q", payload[offset : offset + 8])
        offset += 8
        if header_size > 1 << 20 or len(payload) < offset + header_size:
            raise WorldModelStateError("owned model state header is truncated or unreasonably large")
        raw_header = payload[offset : offset + header_size]
        offset += header_size
        try:
            metadata = json.loads(raw_header)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WorldModelStateError("owned model state header is not valid JSON") from error
        expected_keys = {
            "format",
            "model_bytes",
            "model_sha256",
            "optimizer_bytes",
            "optimizer_sha256",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_keys or metadata["format"] != _FORMAT:
            raise WorldModelStateError("owned model state header has an unsupported schema")
        canonical_header = json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if canonical_header != raw_header:
            raise WorldModelStateError("owned model state header is not canonical JSON")
        model_size = metadata["model_bytes"]
        optimizer_size = metadata["optimizer_bytes"]
        if (
            not isinstance(model_size, int)
            or isinstance(model_size, bool)
            or model_size < 1
            or not isinstance(optimizer_size, int)
            or isinstance(optimizer_size, bool)
            or optimizer_size < 1
            or len(payload) != offset + model_size + optimizer_size
        ):
            raise WorldModelStateError("owned model state component sizes are invalid")
        model_bytes = payload[offset : offset + model_size]
        optimizer_bytes = payload[offset + model_size :]
        if hashlib.sha256(model_bytes).hexdigest() != metadata["model_sha256"]:
            raise WorldModelStateError("owned model state parameter digest mismatch")
        if hashlib.sha256(optimizer_bytes).hexdigest() != metadata["optimizer_sha256"]:
            raise WorldModelStateError("owned model state optimizer digest mismatch")
        state = cls(model_bytes=model_bytes, optimizer_bytes=optimizer_bytes)
        if state.to_bytes() != payload:
            raise WorldModelStateError("owned model state bytes are not canonical")
        return state

    @property
    def payload(self) -> bytes:
        return self.to_bytes()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def version(self) -> str:
        return f"wm001-state-sha256:{self.digest}"


class _ValidatedStateCache:
    """Deserialize candidates before reservation and retain the validated object."""

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self._device = torch.device(device)
        self._models: dict[str, ProbabilisticEnsemble] = {}
        self._states: dict[str, CompoundModelState] = {}

    def validate(self, state: ModelState) -> None:
        expected = f"wm001-state-sha256:{state.digest}"
        if state.version != expected:
            raise WorldModelStateError("owned model version does not bind its exact compound bytes")
        compound = CompoundModelState.from_bytes(state.payload)
        model = ProbabilisticEnsemble.from_bytes(compound.model_bytes, device=self._device)
        optimizer_from_bytes(model, compound.optimizer_bytes)
        self._models[state.digest] = model
        self._states[state.digest] = compound

    def model(self, state: ModelState) -> ProbabilisticEnsemble:
        try:
            return self._models[state.digest]
        except KeyError:
            self.validate(state)
            return self._models[state.digest]

    def compound(self, state: ModelState) -> CompoundModelState:
        try:
            return self._states[state.digest]
        except KeyError:
            self.validate(state)
            return self._states[state.digest]


class WorldModelRuntime(PredictiveBackend):
    """Live numeric view whose bytes are exclusively owned by Prospect."""

    def __init__(
        self,
        owner: VersionedModelOwner,
        cache: _ValidatedStateCache,
    ) -> None:
        self.owner = owner
        self._cache = cache

    @classmethod
    def initialize(
        cls,
        *,
        initialization_seed: int,
        model_config: WorldModelConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> WorldModelRuntime:
        model = ProbabilisticEnsemble(model_config, initialization_seed=initialization_seed)
        optimizer = make_optimizer(model, optimizer_config)
        compound = CompoundModelState(
            model_bytes=model.to_bytes(),
            optimizer_bytes=optimizer_to_bytes(model, optimizer, config=optimizer_config),
        )
        payload = compound.payload
        state = ModelState(
            version=f"wm001-state-sha256:{hashlib.sha256(payload).hexdigest()}",
            payload=payload,
        )
        cache = _ValidatedStateCache(device=device)
        owner = VersionedModelOwner(state, validator=cache.validate)
        return cls(owner, cache)

    @classmethod
    def from_payload(
        cls,
        payload: bytes,
        *,
        device: torch.device | str = "cpu",
    ) -> WorldModelRuntime:
        digest = hashlib.sha256(payload).hexdigest()
        state = ModelState(version=f"wm001-state-sha256:{digest}", payload=payload)
        cache = _ValidatedStateCache(device=device)
        owner = VersionedModelOwner(state, validator=cache.validate)
        return cls(owner, cache)

    def fork(self, *, device: torch.device | str | None = None) -> WorldModelRuntime:
        state = self.owner.snapshot_state()
        target_device = self.model.device if device is None else device
        return self.from_payload(state.payload, device=target_device)

    @property
    def version(self) -> str:
        return self.owner.version

    @property
    def digest(self) -> str:
        """Digest of model parameters only, excluding optimizer slots."""

        return self.model.parameter_sha256

    @property
    def live_state_digest(self) -> str:
        return self.owner.digest

    @property
    def model(self) -> ProbabilisticEnsemble:
        return self._cache.model(self.owner.snapshot_state())

    @property
    def optimizer_bytes(self) -> bytes:
        return self._cache.compound(self.owner.snapshot_state()).optimizer_bytes

    @property
    def model_bytes(self) -> bytes:
        return self._cache.compound(self.owner.snapshot_state()).model_bytes

    def predict_ensemble(
        self,
        observation: np.ndarray,
        context: float,
        action: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        means, variances = model.predict_ensemble(observation, context, action)
        physical_delta = means[..., :3]
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=means.device)
        expanded_observation = observation_tensor.expand_as(physical_delta)
        next_means = model.project_next(expanded_observation, physical_delta)
        physical_variances = variances[..., :3].clamp_min(1e-12)
        return (
            next_means.detach().cpu().numpy().astype(np.float64, copy=False),
            physical_variances.detach().cpu().numpy().astype(np.float64, copy=False),
        )


@dataclass(frozen=True, slots=True)
class LearningEvidence:
    """Non-domain evidence retained for the result and independent audit."""

    phase: str
    consumed_transition_ids: tuple[str, ...]
    consumed_multiset_sha256: str
    predecessor_parameter_sha256: str
    candidate_parameter_sha256: str
    predecessor_live_state_sha256: str
    candidate_live_state_sha256: str
    optimizer_steps: int
    sampling_manifest: bytes
    sampling_manifest_sha256: str
    sampled_id_counts: tuple[tuple[str, int], ...]
    target_permutation_sha256: str | None
    target_permutation_payload: bytes | None
    loss_history: tuple[float, ...]


class TransactionalWorldModelLearner(TransactionalLearner):
    """Prepare a complete candidate from immutable model/optimizer bytes."""

    def __init__(
        self,
        *,
        phase: str,
        bootstrap_seeds: Sequence[int],
        minibatch_order_seed: int,
        optimizer_steps: int,
        optimizer_config: OptimizerConfig | None = None,
        training_mode: Literal["normal", "joint_target_permuted"] = "normal",
        target_permutation_seed: int | None = None,
        balanced_tasks: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.phase = phase
        self.bootstrap_seeds = tuple(int(seed) for seed in bootstrap_seeds)
        self.minibatch_order_seed = int(minibatch_order_seed)
        self.optimizer_steps = optimizer_steps
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.training_mode = training_mode
        self.target_permutation_seed = target_permutation_seed
        self.balanced_tasks = balanced_tasks
        self.device = torch.device(device)
        self.last_prepared: PreparedCandidate | None = None
        self.last_evidence: LearningEvidence | None = None

    def prepare(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
        current_model: ModelState,
    ) -> PreparedLearningUpdate:
        if snapshot.model_version != current_model.version:
            raise WorldModelStateError("learner snapshot does not identify the supplied immutable model")
        compound = CompoundModelState.from_bytes(current_model.payload)
        source = ProbabilisticEnsemble.from_bytes(compound.model_bytes, device=self.device)
        optimizer_from_bytes(
            source,
            compound.optimizer_bytes,
            expected_config=self.optimizer_config,
        )
        batch = _transition_batch(transitions)
        candidate = prepare_candidate(
            source,
            batch,
            optimizer_steps=self.optimizer_steps,
            bootstrap_seeds=self.bootstrap_seeds,
            minibatch_order_seed=self.minibatch_order_seed,
            optimizer_config=self.optimizer_config,
            predecessor_optimizer_bytes=compound.optimizer_bytes,
            training_mode=self.training_mode,
            target_permutation_seed=self.target_permutation_seed,
            balanced_tasks=self.balanced_tasks,
            device=self.device,
        )
        candidate.model.to("cpu")
        candidate.model.eval()
        candidate_compound = CompoundModelState(
            model_bytes=candidate.model.to_bytes(),
            optimizer_bytes=candidate.optimizer_bytes,
        )
        candidate_payload = candidate_compound.payload
        candidate_state = ModelState(
            version=f"wm001-state-sha256:{hashlib.sha256(candidate_payload).hexdigest()}",
            payload=candidate_payload,
        )
        completed_at = TimePoint(snapshot.captured_at.tick + 1, snapshot.captured_at.clock_id)
        resulting_belief = _resulting_belief(
            transitions[-1].belief_update.posterior,
            model_version=candidate_state.version,
            phase=self.phase,
            digest=candidate.candidate_parameter_sha256,
            completed_at=completed_at,
        )
        transition_ids = tuple(transition.transition_id for transition in transitions)
        multiset_digest = candidate.consumed_multiset_sha256
        receipt = UpdateReceipt(
            receipt_id=(
                f"wm001:{self.phase}:receipt:{candidate.candidate_parameter_sha256[:16]}:{multiset_digest[:16]}"
            ),
            agent_id=snapshot.agent_id,
            transitions=tuple(transitions),
            learner_version="wm001-transactional-ensemble-learner-v1",
            status=UpdateStatus.APPLIED,
            previous_configuration_version=snapshot.configuration_version,
            new_configuration_version=f"wm001-config:{candidate_state.version}",
            previous_model_version=current_model.version,
            new_model_version=candidate_state.version,
            previous_representation_version=snapshot.representation_version,
            new_representation_version=snapshot.representation_version,
            previous_policy_version=snapshot.policy_version,
            new_policy_version=snapshot.policy_version,
            started_at=snapshot.captured_at,
            completed_at=completed_at,
            resulting_belief=resulting_belief,
            metrics=(
                ("optimizer_steps", float(candidate.optimizer_steps)),
                ("training_loss_first", float(candidate.loss_history[0])),
                ("training_loss_last", float(candidate.loss_history[-1])),
                ("consumed_unique_transitions", float(len(transition_ids))),
            ),
        )
        self.last_prepared = candidate
        self.last_evidence = LearningEvidence(
            phase=self.phase,
            consumed_transition_ids=transition_ids,
            consumed_multiset_sha256=multiset_digest,
            predecessor_parameter_sha256=candidate.predecessor_parameter_sha256,
            candidate_parameter_sha256=candidate.candidate_parameter_sha256,
            predecessor_live_state_sha256=current_model.digest,
            candidate_live_state_sha256=candidate_state.digest,
            optimizer_steps=candidate.optimizer_steps,
            sampling_manifest=candidate.sampling_manifest,
            sampling_manifest_sha256=candidate.sampling_manifest_sha256,
            sampled_id_counts=candidate.sampled_id_counts,
            target_permutation_sha256=candidate.target_permutation_sha256,
            target_permutation_payload=candidate.target_permutation_payload,
            loss_history=candidate.loss_history,
        )
        return PreparedLearningUpdate(
            receipt=receipt,
            source_model_digest=current_model.digest,
            candidate_model=candidate_state,
        )


class RejectedUpdateProbeLearner(TransactionalLearner):
    """Predeclared forged-source probe for the transaction failure invariant."""

    def __init__(self, *, phase: str = "rejected_update_probe") -> None:
        self.phase = phase

    def prepare(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
        current_model: ModelState,
    ) -> PreparedLearningUpdate:
        completed_at = TimePoint(snapshot.captured_at.tick + 1, snapshot.captured_at.clock_id)
        forged_version = f"{current_model.version}:rejected-probe"
        resulting_belief = _resulting_belief(
            transitions[-1].belief_update.posterior,
            model_version=forged_version,
            phase=self.phase,
            digest=current_model.digest,
            completed_at=completed_at,
        )
        receipt = UpdateReceipt(
            receipt_id=f"wm001:{self.phase}:receipt:{current_model.digest[:24]}",
            agent_id=snapshot.agent_id,
            transitions=tuple(transitions),
            learner_version="wm001-rejected-source-digest-probe-v1",
            status=UpdateStatus.APPLIED,
            previous_configuration_version=snapshot.configuration_version,
            new_configuration_version=f"wm001-config:{forged_version}",
            previous_model_version=current_model.version,
            new_model_version=forged_version,
            previous_representation_version=snapshot.representation_version,
            new_representation_version=snapshot.representation_version,
            previous_policy_version=snapshot.policy_version,
            new_policy_version=snapshot.policy_version,
            started_at=snapshot.captured_at,
            completed_at=completed_at,
            resulting_belief=resulting_belief,
            metrics=(("probe_expected_rejection", 1.0),),
        )
        return PreparedLearningUpdate(
            receipt=receipt,
            source_model_digest="0" * 64,
            candidate_model=ModelState(
                version=forged_version,
                payload=current_model.payload,
            ),
        )


def _transition_batch(transitions: Sequence[EpistemicTransition]) -> TransitionBatch:
    observations, contexts, actions, targets, transition_ids = transition_arrays(transitions)
    return TransitionBatch.from_arrays(
        transition_ids=transition_ids,
        observations=observations,
        contexts=contexts,
        actions=actions,
        next_observations=observations + targets[:, :3],
        rewards=targets[:, 3],
    )


def _resulting_belief(
    source: Belief,
    *,
    model_version: str,
    phase: str,
    digest: str,
    completed_at: TimePoint,
) -> Belief:
    return replace(
        source,
        belief_id=f"wm001:{phase}:resulting-belief:{digest[:24]}",
        formed_at=completed_at,
        model_version=model_version,
    )


__all__ = (
    "CompoundModelState",
    "LearningEvidence",
    "RejectedUpdateProbeLearner",
    "TransactionalWorldModelLearner",
    "WorldModelRuntime",
    "WorldModelStateError",
)
