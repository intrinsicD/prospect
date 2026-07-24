"""Safe, exact pre-terminal checkpoint contract for WM-002 Q1.

This module owns no environment state.  It serializes all and only the five
public runtime components declared by the Q1 design and reconstructs a fresh
canonical storage/state/model graph after every byte and semantic check passes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

from prospect.decision import CounterIdentitySource
from prospect.domain import (
    AgentSnapshot,
    EpistemicTransition,
    ExperienceEvent,
    TimePoint,
    UpdateReceipt,
    UpdateStatus,
)
from prospect.runtime import AgentState, ModelStateValidator, VersionedModelOwner
from prospect.storage import (
    CheckpointComponent,
    CheckpointCoordinator,
    CheckpointFormatError,
    CheckpointIntegrityError,
    CheckpointManifest,
    EpistemicLedger,
    InMemoryExperienceStore,
    checkpoint_manifest_sha256,
    decode_domain_graph,
    encode_domain_graph,
)

Q1_CHECKPOINT_SCHEMA = "prospect.wm002.q1.checkpoint.v1"
Q1_COMPONENT_IDS = (
    "domain_custody",
    "episode_accumulator",
    "identity_counter",
    "posterior_model",
    "qualification_binding",
)
# The real Q1 preterminal path reaches 43; the development rollback probe can
# consume three additional abandoned learner IDs before succeeding at 46.
Q1_MAX_IDENTITY_NEXT_COUNTER = 64
FORBIDDEN_PRIVATE_COMPONENT_IDS = frozenset(
    {
        "derivation_key",
        "environment_salt",
        "future_outcome",
        "hidden_regime",
        "master_seed",
        "potential_outcomes",
        "private_schedule",
        "secret_salt",
        "seed",
        "terminal_draw",
        "unchosen_potential_outcomes",
    }
)

_COMPONENT_VERSIONS = MappingProxyType(
    {
        "domain_custody": "prospect.wm002.q1.domain-custody.v1",
        "episode_accumulator": "prospect.wm002.q1.episode-accumulator.v1",
        "identity_counter": "prospect.counter-identity.v1",
        "posterior_model": "prospect.wm002.q1.posterior-model-component.v1",
        "qualification_binding": "prospect.wm002.q1.qualification-binding.v1",
    }
)
_COMPONENT_MEDIA_TYPES = MappingProxyType(
    {component_id: f"application/vnd.{version}+json" for component_id, version in _COMPONENT_VERSIONS.items()}
)
_MANIFEST_VERSIONS = MappingProxyType({"wm002_q1_checkpoint": Q1_CHECKPOINT_SCHEMA})
_ACTION_ORDINALS = MappingProxyType(
    {
        "skip": 0,
        "weak": 1,
        "strong": 2,
        "overpowered": 3,
        "nuisance": 4,
    }
)
_ACTION_OUTCOME_SUPPORT = MappingProxyType(
    {
        "skip": frozenset({0}),
        "weak": frozenset({-1, 1}),
        "strong": frozenset({-1, 1}),
        "overpowered": frozenset({-1, 1}),
        "nuisance": frozenset({0, 1, 2, 3}),
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"wm002-q1-[0-9a-f]{64}\Z")
_DEFAULT_RUN_ID = "wm002-q1-" + "0" * 64
_MAX_COMPONENT_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 32 * 1024 * 1024
_FORBIDDEN_VALUE_KEYS = frozenset(
    {
        "derivation_key",
        "environment_salt",
        "future_outcome",
        "hidden_regime",
        "master_seed",
        "potential_outcomes",
        "private_seed",
        "private_schedule_position",
        "realized_theta",
        "regime_value",
        "salt",
        "seed",
        "secret_salt",
        "terminal_draw",
        "theta_value",
        "unchosen_potential_outcome",
        "unchosen_potential_outcomes",
    }
)
_PRIVATE_VALUE_KEY_FRAGMENTS = (
    "counterfactual",
    "derivation",
    "future_outcome",
    "hmac",
    "potential_outcome",
    "regime",
    "salt",
    "schedule_position",
    "seed",
    "terminal_draw",
    "theta",
)


class Q1CheckpointError(ValueError):
    """A Q1 logical bundle violates its exact public checkpoint contract."""


@dataclass(frozen=True, slots=True)
class QualificationBinding:
    """External qualification identities that a checkpoint cannot self-attest."""

    protocol_version: str
    protocol_sha256: str
    implementation_sha256: str
    q0_report_sha256: str
    entry_qualification_sha256: str
    salt_commitment_sha256: str
    run_id: str = _DEFAULT_RUN_ID
    attempt_id: str = _DEFAULT_RUN_ID + "-attempt-0001"

    def __post_init__(self) -> None:
        if not self.protocol_version or not self.protocol_version.strip():
            raise Q1CheckpointError("protocol_version must be nonempty")
        for label, digest in (
            ("protocol_sha256", self.protocol_sha256),
            ("implementation_sha256", self.implementation_sha256),
            ("q0_report_sha256", self.q0_report_sha256),
            ("entry_qualification_sha256", self.entry_qualification_sha256),
            ("salt_commitment_sha256", self.salt_commitment_sha256),
        ):
            _require_sha256(digest, label=label)
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise Q1CheckpointError("run_id must bind a lowercase SHA-256")
        if self.attempt_id != f"{self.run_id}-attempt-0001":
            raise Q1CheckpointError("attempt_id must name the sole frozen run attempt")

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "implementation_sha256": self.implementation_sha256,
            "entry_qualification_sha256": self.entry_qualification_sha256,
            "protocol_sha256": self.protocol_sha256,
            "protocol_version": self.protocol_version,
            "q0_report_sha256": self.q0_report_sha256,
            "run_id": self.run_id,
            "schema": _COMPONENT_VERSIONS["qualification_binding"],
            "salt_commitment_sha256": self.salt_commitment_sha256,
        }


@dataclass(frozen=True, slots=True)
class EpisodeAccumulator:
    """Primitive acquisition return fields charged before the terminal action."""

    task_payoff: float
    physical_action_cost: float
    information_acquisition_cost: float

    def __post_init__(self) -> None:
        for label, value in (
            ("task_payoff", self.task_payoff),
            ("physical_action_cost", self.physical_action_cost),
            ("information_acquisition_cost", self.information_acquisition_cost),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise Q1CheckpointError(f"{label} must be a finite number")
        if float(self.task_payoff) not in {0.0, 1.0}:
            raise Q1CheckpointError("task_payoff must be a binary fixture payoff")
        if self.physical_action_cost < 0.0 or self.information_acquisition_cost < 0.0:
            raise Q1CheckpointError("acquisition costs must be nonnegative")

    @property
    def net_acquisition_return(self) -> float:
        return float(self.task_payoff - self.physical_action_cost - self.information_acquisition_cost)

    def as_dict(self) -> dict[str, object]:
        return {
            "information_acquisition_cost": float(self.information_acquisition_cost),
            "physical_action_cost": float(self.physical_action_cost),
            "schema": _COMPONENT_VERSIONS["episode_accumulator"],
            "task_payoff": float(self.task_payoff),
        }


@dataclass(frozen=True, slots=True)
class CheckpointComponentDigest:
    component_id: str
    sha256: str
    size_bytes: int
    version: str


@dataclass(frozen=True, slots=True)
class Q1CheckpointReport:
    checkpoint_id: str
    agent_id: str
    component_digests: tuple[CheckpointComponentDigest, ...]
    manifest_sha256: str
    aggregate_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Q1CheckpointArtifact:
    """One complete deterministic frame and its independently usable digests."""

    payload: bytes
    report: Q1CheckpointReport


@dataclass(frozen=True, slots=True)
class RestoredQ1Checkpoint:
    """Fresh authoritative objects reconstructed from one verified Q1 frame."""

    report: Q1CheckpointReport
    manifest: CheckpointManifest
    model_owner: VersionedModelOwner
    model_predecessor_sha256: str
    experience_store: InMemoryExperienceStore
    ledger: EpistemicLedger
    agent_state: AgentState
    identity_source: CounterIdentitySource
    accumulator: EpisodeAccumulator
    binding: QualificationBinding
    snapshot: AgentSnapshot
    experience: ExperienceEvent
    transition: EpistemicTransition
    receipt: UpdateReceipt


def dump_q1_checkpoint(
    *,
    checkpoint_id: str,
    agent_id: str,
    created_at: TimePoint,
    model_owner: VersionedModelOwner,
    predecessor_model_sha256: str,
    snapshot: AgentSnapshot,
    experience: ExperienceEvent,
    transition: EpistemicTransition,
    receipt: UpdateReceipt,
    identity_source: CounterIdentitySource,
    accumulator: EpisodeAccumulator,
    binding: QualificationBinding,
) -> Q1CheckpointArtifact:
    """Encode the exact five-component pre-terminal Q1 state in memory."""

    _require_sha256(predecessor_model_sha256, label="predecessor_model_sha256")
    model_state = model_owner.snapshot_state()
    if model_state.version != f"wm002-model-sha256:{model_state.digest}":
        raise Q1CheckpointError("posterior model version does not bind its exact bytes")
    if model_state.digest == predecessor_model_sha256:
        raise Q1CheckpointError("posterior model digest must differ from its predecessor")
    _validate_domain_roots(
        agent_id=agent_id,
        snapshot=snapshot,
        experience=experience,
        transition=transition,
        receipt=receipt,
        model_version=model_state.version,
        predecessor_sha256=predecessor_model_sha256,
    )
    _validate_accumulator_against_experience(accumulator, experience)
    model_component = {
        "model_payload_base64": base64.b64encode(model_state.payload).decode("ascii"),
        "model_sha256": model_state.digest,
        "model_version": model_state.version,
        "predecessor_sha256": predecessor_model_sha256,
        "schema": _COMPONENT_VERSIONS["posterior_model"],
    }
    domain_component = {
        "domain_graph": encode_domain_graph(
            {
                "acquisition_experience": experience,
                "acquisition_receipt": receipt,
                "acquisition_transition": transition,
                "preterminal_snapshot": snapshot,
            }
        ),
        "schema": _COMPONENT_VERSIONS["domain_custody"],
    }
    _reject_private_value_keys(domain_component)
    components = {
        "domain_custody": _json_component("domain_custody", domain_component),
        "episode_accumulator": _json_component("episode_accumulator", accumulator.as_dict()),
        "identity_counter": CheckpointComponent(
            name="identity_counter",
            version=_COMPONENT_VERSIONS["identity_counter"],
            payload=identity_source.checkpoint_bytes(),
            media_type=_COMPONENT_MEDIA_TYPES["identity_counter"],
        ),
        "posterior_model": _json_component("posterior_model", model_component),
        "qualification_binding": _json_component("qualification_binding", binding.as_dict()),
    }
    coordinator = _coordinator()
    payload = coordinator.dump_bytes(
        checkpoint_id=checkpoint_id,
        agent_id=agent_id,
        created_at=created_at,
        components=components,
        versions=_MANIFEST_VERSIONS,
        metadata=_binding_metadata(binding),
    )
    loaded = coordinator.load_bytes(payload, expected_agent_id=agent_id)
    return Q1CheckpointArtifact(payload=payload, report=_report(payload, loaded.manifest))


def load_q1_checkpoint(
    payload: bytes,
    *,
    expected_agent_id: str,
    expected_aggregate_sha256: str,
    expected_binding: QualificationBinding,
    model_validator: ModelStateValidator,
) -> RestoredQ1Checkpoint:
    """Verify a Q1 frame completely, then construct an entirely fresh graph."""

    _require_sha256(expected_aggregate_sha256, label="expected_aggregate_sha256")
    actual_aggregate = hashlib.sha256(payload).hexdigest()
    if actual_aggregate != expected_aggregate_sha256:
        raise Q1CheckpointError("checkpoint aggregate SHA-256 differs from the external frame binding")
    coordinator = _coordinator()
    try:
        loaded = coordinator.load_bytes(payload, expected_agent_id=expected_agent_id)
    except (CheckpointFormatError, CheckpointIntegrityError) as error:
        raise Q1CheckpointError(str(error)) from error
    manifest = loaded.manifest
    _validate_manifest_contract(manifest, expected_binding)

    raw_binding = _decode_json_component(
        loaded.payload("qualification_binding"),
        component_id="qualification_binding",
    )
    binding = _decode_binding(raw_binding)
    if binding != expected_binding:
        raise Q1CheckpointError("qualification binding differs from the external accepted binding")
    if _binding_metadata(binding) != dict(manifest.metadata):
        raise Q1CheckpointError("checkpoint metadata differs from its qualification component")

    model_version, model_payload, predecessor_sha256 = _decode_model_component(
        _decode_json_component(loaded.payload("posterior_model"), component_id="posterior_model")
    )
    accumulator = _decode_accumulator(
        _decode_json_component(
            loaded.payload("episode_accumulator"),
            component_id="episode_accumulator",
        )
    )
    raw_domain = _decode_json_component(
        loaded.payload("domain_custody"),
        component_id="domain_custody",
    )
    _reject_private_value_keys(raw_domain)
    domain = _exact_mapping(
        raw_domain,
        {"domain_graph", "schema"},
        label="domain custody",
    )
    if domain["schema"] != _COMPONENT_VERSIONS["domain_custody"]:
        raise Q1CheckpointError("domain custody has an unsupported schema")
    roots = decode_domain_graph(domain["domain_graph"])
    snapshot, experience, transition, receipt = _decode_domain_roots(roots)
    _validate_domain_roots(
        agent_id=expected_agent_id,
        snapshot=snapshot,
        experience=experience,
        transition=transition,
        receipt=receipt,
        model_version=model_version,
        predecessor_sha256=predecessor_sha256,
    )
    _validate_accumulator_against_experience(accumulator, experience)

    identity_source = _decode_identity_source(loaded.payload("identity_counter"))
    model_owner = VersionedModelOwner.from_checkpoint(
        version=model_version,
        payload=model_payload,
        validator=model_validator,
    )
    experience_store = InMemoryExperienceStore()
    experience_store.append(experience)
    ledger = EpistemicLedger(experience_store)
    canonical_transition = ledger.append_rehydrated_transition(transition)
    canonical_receipt = ledger.append_rehydrated_update(receipt)
    canonical_belief = canonical_receipt.resulting_belief
    if canonical_belief is None:
        raise Q1CheckpointError("applied acquisition receipt has no resulting belief")
    canonical_snapshot = replace(
        snapshot,
        belief=canonical_belief,
        latest_update=canonical_receipt,
    )
    agent_state = AgentState(canonical_snapshot)
    report = _report(payload, manifest)
    return RestoredQ1Checkpoint(
        report=report,
        manifest=manifest,
        model_owner=model_owner,
        model_predecessor_sha256=predecessor_sha256,
        experience_store=experience_store,
        ledger=ledger,
        agent_state=agent_state,
        identity_source=identity_source,
        accumulator=accumulator,
        binding=binding,
        snapshot=canonical_snapshot,
        experience=experience,
        transition=canonical_transition,
        receipt=canonical_receipt,
    )


def _coordinator() -> CheckpointCoordinator:
    return CheckpointCoordinator(
        max_component_bytes=_MAX_COMPONENT_BYTES,
        max_total_bytes=_MAX_TOTAL_BYTES,
    )


def _json_component(component_id: str, value: object) -> CheckpointComponent:
    return CheckpointComponent(
        name=component_id,
        version=_COMPONENT_VERSIONS[component_id],
        payload=_canonical_json_bytes(value),
        media_type=_COMPONENT_MEDIA_TYPES[component_id],
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise Q1CheckpointError("checkpoint component is not canonical-JSON encodable") from error


def _decode_json_component(payload: bytes, *, component_id: str) -> object:
    if len(payload) > _MAX_COMPONENT_BYTES:
        raise Q1CheckpointError(f"{component_id} exceeds its decoded byte limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q1CheckpointError(f"{component_id} is not valid UTF-8 JSON") from error
    if _canonical_json_bytes(value) != payload:
        raise Q1CheckpointError(f"{component_id} is not canonical JSON")
    return value


def _decode_binding(value: object) -> QualificationBinding:
    raw = _exact_mapping(
        value,
        {
            "attempt_id",
            "entry_qualification_sha256",
            "implementation_sha256",
            "protocol_sha256",
            "protocol_version",
            "q0_report_sha256",
            "run_id",
            "salt_commitment_sha256",
            "schema",
        },
        label="qualification binding",
    )
    if raw["schema"] != _COMPONENT_VERSIONS["qualification_binding"]:
        raise Q1CheckpointError("qualification binding has an unsupported schema")
    return QualificationBinding(
        protocol_version=_require_string(raw["protocol_version"], label="protocol_version"),
        protocol_sha256=_require_string(raw["protocol_sha256"], label="protocol_sha256"),
        implementation_sha256=_require_string(raw["implementation_sha256"], label="implementation_sha256"),
        q0_report_sha256=_require_string(raw["q0_report_sha256"], label="q0_report_sha256"),
        entry_qualification_sha256=_require_string(
            raw["entry_qualification_sha256"], label="entry_qualification_sha256"
        ),
        salt_commitment_sha256=_require_string(raw["salt_commitment_sha256"], label="salt_commitment_sha256"),
        run_id=_require_string(raw["run_id"], label="run_id"),
        attempt_id=_require_string(raw["attempt_id"], label="attempt_id"),
    )


def _decode_model_component(value: object) -> tuple[str, bytes, str]:
    raw = _exact_mapping(
        value,
        {
            "model_payload_base64",
            "model_sha256",
            "model_version",
            "predecessor_sha256",
            "schema",
        },
        label="posterior model component",
    )
    if raw["schema"] != _COMPONENT_VERSIONS["posterior_model"]:
        raise Q1CheckpointError("posterior model component has an unsupported schema")
    encoded = _require_string(raw["model_payload_base64"], label="model_payload_base64")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise Q1CheckpointError("posterior model payload is not strict base64") from error
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise Q1CheckpointError("posterior model payload is not canonical base64")
    digest = hashlib.sha256(payload).hexdigest()
    declared_digest = _require_string(raw["model_sha256"], label="model_sha256")
    _require_sha256(declared_digest, label="model_sha256")
    if digest != declared_digest:
        raise Q1CheckpointError("posterior model bytes disagree with their component digest")
    version = _require_string(raw["model_version"], label="model_version")
    if version != f"wm002-model-sha256:{digest}":
        raise Q1CheckpointError("posterior model version does not bind its exact bytes")
    predecessor = _require_string(raw["predecessor_sha256"], label="predecessor_sha256")
    _require_sha256(predecessor, label="predecessor_sha256")
    if predecessor == digest:
        raise Q1CheckpointError("posterior model digest must differ from its predecessor")
    return version, payload, predecessor


def _decode_accumulator(value: object) -> EpisodeAccumulator:
    raw = _exact_mapping(
        value,
        {
            "information_acquisition_cost",
            "physical_action_cost",
            "schema",
            "task_payoff",
        },
        label="episode accumulator",
    )
    if raw["schema"] != _COMPONENT_VERSIONS["episode_accumulator"]:
        raise Q1CheckpointError("episode accumulator has an unsupported schema")
    return EpisodeAccumulator(
        task_payoff=_require_number(raw["task_payoff"], label="task_payoff"),
        physical_action_cost=_require_number(raw["physical_action_cost"], label="physical_action_cost"),
        information_acquisition_cost=_require_number(
            raw["information_acquisition_cost"],
            label="information_acquisition_cost",
        ),
    )


def _decode_identity_source(payload: bytes) -> CounterIdentitySource:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q1CheckpointError("identity counter is not valid JSON") from error
    raw = _exact_mapping(
        value,
        {"namespace", "next_counter", "schema_version"},
        label="identity counter",
    )
    namespace = _require_string(raw["namespace"], label="identity namespace")
    next_counter = raw["next_counter"]
    if (
        isinstance(next_counter, int)
        and not isinstance(next_counter, bool)
        and next_counter > Q1_MAX_IDENTITY_NEXT_COUNTER
    ):
        raise Q1CheckpointError(f"identity checkpoint counter must not exceed {Q1_MAX_IDENTITY_NEXT_COUNTER}")
    source = CounterIdentitySource(namespace)
    try:
        source.restore_bytes(payload)
    except ValueError as error:
        raise Q1CheckpointError(str(error)) from error
    return source


def _decode_domain_roots(
    roots: Mapping[str, object],
) -> tuple[AgentSnapshot, ExperienceEvent, EpistemicTransition, UpdateReceipt]:
    expected = {
        "acquisition_experience",
        "acquisition_receipt",
        "acquisition_transition",
        "preterminal_snapshot",
    }
    if set(roots) != expected:
        raise Q1CheckpointError("domain custody roots differ from the exact Q1 set")
    snapshot = roots["preterminal_snapshot"]
    experience = roots["acquisition_experience"]
    transition = roots["acquisition_transition"]
    receipt = roots["acquisition_receipt"]
    for value, expected_type, label in (
        (snapshot, AgentSnapshot, "preterminal snapshot"),
        (experience, ExperienceEvent, "acquisition experience"),
        (transition, EpistemicTransition, "acquisition transition"),
        (receipt, UpdateReceipt, "acquisition receipt"),
    ):
        if type(value) is not expected_type:
            raise Q1CheckpointError(f"{label} has an unexpected record type")
    return (
        cast(AgentSnapshot, snapshot),
        cast(ExperienceEvent, experience),
        cast(EpistemicTransition, transition),
        cast(UpdateReceipt, receipt),
    )


def _validate_domain_roots(
    *,
    agent_id: str,
    snapshot: AgentSnapshot,
    experience: ExperienceEvent,
    transition: EpistemicTransition,
    receipt: UpdateReceipt,
    model_version: str,
    predecessor_sha256: str,
) -> None:
    if snapshot.agent_id != agent_id or experience.agent_id != agent_id or receipt.agent_id != agent_id:
        raise Q1CheckpointError("domain custody contains another agent")
    if experience.step_index != 0 or experience.terminated or experience.truncated:
        raise Q1CheckpointError("pre-terminal custody requires one open acquisition step at index zero")
    if experience.decision is None or experience.execution is None:
        raise Q1CheckpointError("acquisition experience lacks its decision or execution")
    action = experience.decision.selected_assessment.action
    parameters = _exact_mapping(
        action.parameters,
        {"ordinal", "phase", "semantic_action"},
        label="acquisition action",
    )
    semantic_action = _require_string(parameters["semantic_action"], label="semantic acquisition action")
    ordinal = parameters["ordinal"]
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or parameters["phase"] != "acquisition"
        or action.action_kind != "wm002_acquisition"
        or _ACTION_ORDINALS.get(semantic_action) != ordinal
        or action.action_id != f"acquisition:{ordinal:02d}:{semantic_action}"
    ):
        raise Q1CheckpointError("acquisition action identity, ordinal, and semantic ID differ")
    observation = _exact_mapping(
        experience.observation.evidence.payload,
        {"observed_symbol", "phase", "semantic_action"},
        label="acquisition observation payload",
    )
    if observation["phase"] != "acquisition" or observation["semantic_action"] != semantic_action:
        raise Q1CheckpointError("acquisition observation does not name the executed semantic action")
    observed_symbol = observation["observed_symbol"]
    if (
        isinstance(observed_symbol, bool)
        or not isinstance(observed_symbol, int)
        or observed_symbol not in _ACTION_OUTCOME_SUPPORT[semantic_action]
    ):
        raise Q1CheckpointError("acquisition observed_symbol is outside the executed action support")
    if transition.experience is not experience:
        raise Q1CheckpointError("acquisition transition does not share the canonical experience")
    if len(receipt.transitions) != 1 or receipt.transitions[0] is not transition:
        raise Q1CheckpointError("acquisition receipt does not name exactly the canonical transition")
    if snapshot.latest_update is not receipt:
        raise Q1CheckpointError("pre-terminal snapshot does not share the acquisition receipt")
    if snapshot.belief is not receipt.resulting_belief:
        raise Q1CheckpointError("pre-terminal snapshot does not share the receipt's resulting belief")
    if snapshot.pending_intentions:
        raise Q1CheckpointError("pre-terminal snapshot must have no pending intention")
    if snapshot.model_version != model_version or receipt.new_model_version != model_version:
        raise Q1CheckpointError("snapshot, receipt, and model component versions differ")
    if receipt.status is not UpdateStatus.APPLIED or receipt.resulting_belief is None:
        raise Q1CheckpointError("pre-terminal checkpoint requires one applied acquisition receipt")
    if receipt.previous_model_version != f"wm002-model-sha256:{predecessor_sha256}":
        raise Q1CheckpointError("acquisition receipt does not bind the predecessor model digest")
    if not receipt.previous_configuration_version.endswith(predecessor_sha256):
        raise Q1CheckpointError("acquisition receipt configuration does not bind the predecessor digest")
    model_digest = model_version.removeprefix("wm002-model-sha256:")
    if _SHA256.fullmatch(model_digest) is None or not snapshot.configuration_version.endswith(model_digest):
        raise Q1CheckpointError("configuration version does not bind the model digest")


def _validate_accumulator_against_experience(
    accumulator: EpisodeAccumulator,
    experience: ExperienceEvent,
) -> None:
    payload = experience.outcome.evidence.payload
    raw = _exact_mapping(
        payload,
        {
            "information_acquisition_cost",
            "net_reward",
            "physical_action_cost",
            "task_payoff",
        },
        label="acquisition outcome payload",
    )
    expected = (
        accumulator.task_payoff,
        accumulator.physical_action_cost,
        accumulator.information_acquisition_cost,
        accumulator.net_acquisition_return,
    )
    actual = (
        _require_number(raw["task_payoff"], label="outcome task_payoff"),
        _require_number(raw["physical_action_cost"], label="outcome physical_action_cost"),
        _require_number(raw["information_acquisition_cost"], label="outcome information_acquisition_cost"),
        _require_number(raw["net_reward"], label="outcome net_reward"),
    )
    if any(
        not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(expected, actual, strict=True)
    ):
        raise Q1CheckpointError("episode accumulator differs from primitive executed acquisition fields")


def _validate_manifest_contract(manifest: CheckpointManifest, binding: QualificationBinding) -> None:
    names = tuple(component.name for component in manifest.components)
    if names != Q1_COMPONENT_IDS:
        forbidden = sorted(set(names) & FORBIDDEN_PRIVATE_COMPONENT_IDS)
        suffix = f"; forbidden private components={forbidden}" if forbidden else ""
        raise Q1CheckpointError(f"Q1 checkpoint must contain all and only five canonical components{suffix}")
    if dict(manifest.versions) != dict(_MANIFEST_VERSIONS):
        raise Q1CheckpointError("Q1 checkpoint manifest version map is not exact")
    for component in manifest.components:
        if component.version != _COMPONENT_VERSIONS[component.name]:
            raise Q1CheckpointError(f"{component.name} has an unsupported component version")
        if component.media_type != _COMPONENT_MEDIA_TYPES[component.name]:
            raise Q1CheckpointError(f"{component.name} has an unsupported media type")
    if dict(manifest.metadata) != _binding_metadata(binding):
        raise Q1CheckpointError("Q1 checkpoint metadata differs from the external binding")


def _binding_metadata(binding: QualificationBinding) -> dict[str, str]:
    return {
        "entry_qualification_sha256": binding.entry_qualification_sha256,
        "implementation_sha256": binding.implementation_sha256,
        "protocol_sha256": binding.protocol_sha256,
        "protocol_version": binding.protocol_version,
        "q0_report_sha256": binding.q0_report_sha256,
        "salt_commitment_sha256": binding.salt_commitment_sha256,
    }


def _report(payload: bytes, manifest: CheckpointManifest) -> Q1CheckpointReport:
    return Q1CheckpointReport(
        checkpoint_id=manifest.checkpoint_id,
        agent_id=manifest.agent_id,
        component_digests=tuple(
            CheckpointComponentDigest(
                component_id=entry.name,
                sha256=entry.sha256,
                size_bytes=entry.size_bytes,
                version=entry.version,
            )
            for entry in manifest.components
        ),
        manifest_sha256=checkpoint_manifest_sha256(manifest),
        aggregate_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _reject_private_value_keys(value: object) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            forbidden = {key for key in current if isinstance(key, str) and _is_private_value_key(key)}
            encoded_mapping = current.get("$mapping")
            if "$mapping" in current:
                if not isinstance(encoded_mapping, list):
                    raise Q1CheckpointError("checkpoint domain graph has a malformed encoded mapping")
                for entry in encoded_mapping:
                    if not isinstance(entry, list) or len(entry) != 2:
                        raise Q1CheckpointError("checkpoint domain graph has a malformed encoded mapping entry")
                    forbidden.update(key for key in _encoded_key_strings(entry[0]) if _is_private_value_key(key))
            if forbidden:
                raise Q1CheckpointError(f"checkpoint value contains private keys: {sorted(forbidden)}")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _is_private_value_key(value: str) -> bool:
    normalized = value.lower()
    return value in _FORBIDDEN_VALUE_KEYS or any(fragment in normalized for fragment in _PRIVATE_VALUE_KEY_FRAGMENTS)


def _encoded_key_strings(value: object) -> set[str]:
    """Return every string atom used to encode one domain-graph mapping key."""

    strings: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            strings.add(current)
        elif isinstance(current, Mapping):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return strings


def _exact_mapping(value: object, expected: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise Q1CheckpointError(f"{label} must contain exactly {sorted(expected)}")
    return cast(dict[str, object], value)


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise Q1CheckpointError(f"{label} must be a nonempty string")
    return value


def _require_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Q1CheckpointError(f"{label} must be a finite number")
    return float(value)


def _require_sha256(value: str, *, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise Q1CheckpointError(f"{label} must be lowercase SHA-256 hexadecimal")


__all__ = (
    "FORBIDDEN_PRIVATE_COMPONENT_IDS",
    "Q1_CHECKPOINT_SCHEMA",
    "Q1_COMPONENT_IDS",
    "CheckpointComponentDigest",
    "EpisodeAccumulator",
    "Q1CheckpointArtifact",
    "Q1CheckpointError",
    "Q1CheckpointReport",
    "QualificationBinding",
    "RestoredQ1Checkpoint",
    "dump_q1_checkpoint",
    "load_q1_checkpoint",
)
