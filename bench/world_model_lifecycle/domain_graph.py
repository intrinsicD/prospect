"""Explicit canonical JSON graph codec for WM-001 domain custody.

The checkpoint must retain immutable domain records as a graph, not merely as
summary rows.  This codec is deliberately narrow:

* only explicitly allowlisted Prospect domain records and enums are accepted;
* opaque payloads may contain only finite JSON scalars, string-keyed mappings,
  lists, and tuples;
* object sharing is represented by deterministic node references; and
* cross-component links are explicit stable external references supplied by
  the caller.

There is no pickle support, import-by-name, arbitrary class lookup, or generic
dataclass reflection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from prospect.domain import (
    Action,
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    CandidateAssessment,
    DecisionRecord,
    Distribution,
    EpistemicEffect,
    EpistemicEffectKind,
    EpistemicTarget,
    EpistemicTransition,
    Evidence,
    EvidenceLineage,
    EvidenceOrigin,
    ExecutedAction,
    ExecutionStatus,
    ExperienceEvent,
    ExperienceKind,
    Goal,
    InformationSet,
    InformationValue,
    IntendedAction,
    Observation,
    Outcome,
    Prediction,
    ProperScore,
    Provenance,
    ResourceLedger,
    ResourceUse,
    TimePoint,
    TrustLevel,
    UncertaintyEstimate,
    UncertaintyKind,
    UpdateReceipt,
    UpdateStatus,
    Utility,
)

GRAPH_SCHEMA = "prospect.wm001.domain-graph.v1"
MAX_GRAPH_JSON_BYTES = 512 * 1024 * 1024
MAX_GRAPH_NODES = 250_000
MAX_OBSERVATION_SEQUENCES = 20_000
MAX_OBSERVATION_SEQUENCE_LENGTH = 512
MAX_GRAPH_DEPTH = 64
MAX_GRAPH_VALUES = 5_000_000

# The field lists are part of the checkpoint format.  Keeping them explicit
# makes additions fail closed until the codec and its tests are reviewed.
_RECORD_SPECS: dict[str, tuple[type[object], tuple[str, ...]]] = {
    "TimePoint": (TimePoint, ("tick", "clock_id")),
    "Provenance": (Provenance, ("source_id", "trust", "source_kind", "detail")),
    "EvidenceLineage": (
        EvidenceLineage,
        (
            "evidence_id",
            "origin",
            "provenance",
            "parent_evidence_ids",
            "producer_version",
        ),
    ),
    "Evidence": (
        Evidence,
        ("evidence_id", "payload", "occurred_at", "available_at", "lineage"),
    ),
    "Observation": (
        Observation,
        ("observation_id", "agent_id", "modality", "evidence"),
    ),
    "InformationSet": (
        InformationSet,
        (
            "information_set_id",
            "agent_id",
            "as_of",
            "observations",
            "memory_version",
        ),
    ),
    "EpistemicTarget": (
        EpistemicTarget,
        ("target_id", "description", "target_kind"),
    ),
    "Distribution": (
        Distribution,
        (
            "distribution_id",
            "family",
            "support",
            "parameters",
            "representation_version",
            "event_shape",
        ),
    ),
    "Belief": (
        Belief,
        (
            "belief_id",
            "agent_id",
            "target",
            "information_set",
            "distribution",
            "formed_at",
            "model_version",
            "representation_version",
        ),
    ),
    "Action": (Action, ("action_id", "action_kind", "parameters")),
    "UncertaintyEstimate": (
        UncertaintyEstimate,
        (
            "estimate_id",
            "kind",
            "measure",
            "value",
            "unit",
            "target_id",
            "estimator_version",
            "assessed_at",
            "calibration_version",
        ),
    ),
    "IntendedAction": (
        IntendedAction,
        ("intention_id", "agent_id", "action", "intended_at"),
    ),
    "ExecutedAction": (
        ExecutedAction,
        (
            "execution_id",
            "intention",
            "status",
            "started_at",
            "ended_at",
            "realized_action",
            "deviation_reason",
        ),
    ),
    "Prediction": (
        Prediction,
        (
            "prediction_id",
            "prior_belief",
            "action",
            "target",
            "distribution",
            "issued_at",
            "horizon_end",
            "model_version",
            "representation_version",
            "calibration_version",
            "uncertainties",
        ),
    ),
    "Goal": (
        Goal,
        (
            "goal_id",
            "task_id",
            "target",
            "description",
            "issued_at",
            "preference_version",
            "deadline",
        ),
    ),
    "Utility": (
        Utility,
        (
            "utility_id",
            "goal_id",
            "prediction_id",
            "expected_value",
            "unit",
            "evaluator_version",
            "assessed_at",
        ),
    ),
    "InformationValue": (
        InformationValue,
        (
            "information_value_id",
            "prior_belief_id",
            "action_id",
            "target_id",
            "expected_reduction",
            "expected_cost",
            "unit",
            "evaluator_version",
            "assessed_at",
        ),
    ),
    "CandidateAssessment": (
        CandidateAssessment,
        (
            "assessment_id",
            "action",
            "prediction",
            "utility",
            "information_value",
            "expected_action_cost",
            "expected_risk",
            "admissible",
            "constraint_reasons",
            "constraint_penalty",
            "total_value",
            "unit",
            "evaluator_version",
            "assessed_at",
        ),
    ),
    "DecisionRecord": (
        DecisionRecord,
        (
            "decision_id",
            "agent_id",
            "belief",
            "goal",
            "intended_action",
            "alternatives",
            "selected_assessment",
            "policy_version",
            "decided_at",
        ),
    ),
    "Outcome": (Outcome, ("outcome_id", "evidence", "execution_id")),
    "ExperienceEvent": (
        ExperienceEvent,
        (
            "experience_id",
            "agent_id",
            "run_id",
            "task_id",
            "episode_id",
            "step_index",
            "kind",
            "observation",
            "outcome",
            "terminated",
            "truncated",
            "discount",
            "behavior_policy_version",
            "closed_at",
            "decision",
            "execution",
        ),
    ),
    "BeliefUpdate": (
        BeliefUpdate,
        ("update_id", "prior", "experience", "posterior", "updater_version", "updated_at"),
    ),
    "ProperScore": (
        ProperScore,
        (
            "score_id",
            "prediction_id",
            "realized_evidence_id",
            "rule",
            "value",
            "unit",
            "scorer_version",
            "scored_at",
        ),
    ),
    "EpistemicEffect": (
        EpistemicEffect,
        (
            "effect_id",
            "belief_update_id",
            "target_id",
            "kind",
            "measure",
            "before",
            "after",
            "improvement",
            "higher_is_better",
            "evaluator_version",
            "evaluated_at",
            "externally_calibrated",
        ),
    ),
    "EpistemicTransition": (
        EpistemicTransition,
        (
            "transition_id",
            "experience",
            "belief_update",
            "proper_scores",
            "effects",
            "created_at",
        ),
    ),
    "UpdateReceipt": (
        UpdateReceipt,
        (
            "receipt_id",
            "agent_id",
            "transitions",
            "learner_version",
            "status",
            "previous_configuration_version",
            "new_configuration_version",
            "previous_model_version",
            "new_model_version",
            "previous_representation_version",
            "new_representation_version",
            "previous_policy_version",
            "new_policy_version",
            "started_at",
            "completed_at",
            "resulting_belief",
            "rollback_of",
            "metrics",
        ),
    ),
    "ResourceUse": (ResourceUse, ("resource", "amount", "unit")),
    "ResourceLedger": (
        ResourceLedger,
        ("ledger_id", "started_at", "completed_at", "uses"),
    ),
    "AgentSnapshot": (
        AgentSnapshot,
        (
            "snapshot_id",
            "agent_id",
            "captured_at",
            "belief",
            "configuration_version",
            "memory_version",
            "knowledge_version",
            "model_version",
            "representation_version",
            "policy_version",
            "resources",
            "pending_intentions",
            "latest_update",
        ),
    ),
}
_TYPE_TO_SPEC = {record_type: (name, fields) for name, (record_type, fields) in _RECORD_SPECS.items()}

_ENUM_SPECS: dict[str, type[object]] = {
    "TrustLevel": TrustLevel,
    "EvidenceOrigin": EvidenceOrigin,
    "UncertaintyKind": UncertaintyKind,
    "ExecutionStatus": ExecutionStatus,
    "ExperienceKind": ExperienceKind,
    "EpistemicEffectKind": EpistemicEffectKind,
    "UpdateStatus": UpdateStatus,
}
_ENUM_TYPE_TO_NAME = {enum_type: name for name, enum_type in _ENUM_SPECS.items()}


def encode_domain_graph(
    roots: Mapping[str, object],
    *,
    external_references: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Encode named roots and shared records using only the WM-001 allowlist.

    ``external_references`` maps ``id(record)`` to a stable cross-component
    reference.  The caller remains responsible for ensuring those references
    identify canonical objects in the component decoded earlier.
    """

    encoder = _GraphEncoder(external_references or {})
    encoded_roots: dict[str, object] = {}
    for name, value in roots.items():
        _require_name(name, label="graph root")
        encoded_roots[name] = encoder.encode(value)
    graph = {
        "schema": GRAPH_SCHEMA,
        "roots": encoded_roots,
        "nodes": encoder.nodes,
        "observation_sequences": encoder.observation_sequences,
    }
    if len(encoded_roots) > 16 or len(encoder.nodes) > MAX_GRAPH_NODES:
        raise ValueError("domain graph exceeds its root/node bound")
    if len(encoder.observation_sequences) > MAX_OBSERVATION_SEQUENCES:
        raise ValueError("domain graph exceeds its observation-sequence bound")
    _validate_encoded_bounds(graph)
    return graph


def decode_domain_graph(
    value: object,
    *,
    external_objects: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Decode a canonical graph with a caller-owned external-reference table."""

    graph = _exact_mapping(
        value,
        {"schema", "roots", "nodes", "observation_sequences"},
        label="domain graph",
    )
    if graph["schema"] != GRAPH_SCHEMA:
        raise ValueError("domain graph has an unsupported schema")
    roots = graph["roots"]
    nodes = graph["nodes"]
    observation_sequences = graph["observation_sequences"]
    if not isinstance(roots, dict) or not isinstance(nodes, list) or not isinstance(observation_sequences, list):
        raise ValueError("domain graph roots/nodes are malformed")
    if len(roots) > 16 or len(nodes) > MAX_GRAPH_NODES:
        raise ValueError("domain graph exceeds its root/node bound")
    if len(observation_sequences) > MAX_OBSERVATION_SEQUENCES:
        raise ValueError("domain graph exceeds its observation-sequence bound")
    for name in roots:
        _require_name(name, label="graph root")
    _validate_encoded_bounds(value)
    decoder = _GraphDecoder(nodes, observation_sequences, external_objects or {})
    result = {name: decoder.decode(encoded) for name, encoded in roots.items()}
    decoder.require_all_nodes_consumed()
    return result


def transition_external_reference(transition_id: str) -> str:
    """Return the stable cross-component reference for one transition."""

    return f"transition:{_require_name(transition_id, label='transition ID')}"


def update_external_reference(receipt_id: str) -> str:
    """Return the stable cross-component reference for one update receipt."""

    return f"update:{_require_name(receipt_id, label='update receipt ID')}"


def belief_external_reference(belief_id: str) -> str:
    """Return the stable cross-component reference for one canonical belief."""

    return f"belief:{_require_name(belief_id, label='belief ID')}"


class _GraphEncoder:
    def __init__(self, external_references: Mapping[int, str]) -> None:
        self._external = dict(external_references)
        for object_identity, reference in self._external.items():
            if isinstance(object_identity, bool) or not isinstance(object_identity, int):
                raise TypeError("external graph object identities must be integers")
            _require_name(reference, label="external graph reference")
        self._record_refs: dict[int, str] = {}
        self.nodes: list[dict[str, object]] = []
        self._observation_sequence_refs: dict[tuple[int, ...], str] = {}
        self.observation_sequences: list[dict[str, object]] = []

    def encode(self, value: object) -> object:
        external = self._external.get(id(value))
        if external is not None:
            return {"$external": external}
        enum_name = _ENUM_TYPE_TO_NAME.get(type(value))
        if enum_name is not None:
            return {
                "$enum": enum_name,
                "value": cast(Any, value).value,
            }
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("domain graph cannot encode a non-finite float")
            return value
        record_spec = _TYPE_TO_SPEC.get(type(value))
        if record_spec is not None:
            return self._encode_record(value, *record_spec)
        if isinstance(value, tuple):
            if value and all(type(item) is Observation for item in value):
                return {
                    "$observation_sequence": self._encode_observation_sequence(cast(tuple[Observation, ...], value))
                }
            return {"$tuple": [self.encode(item) for item in value]}
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, dict):
            items: list[list[object]] = []
            if any(not isinstance(key, str) for key in value):
                raise TypeError("domain graph mappings require string keys")
            for key in sorted(cast(dict[str, object], value)):
                items.append([key, self.encode(value[key])])
            return {"$mapping": items}
        raise TypeError(f"unsupported WM-001 graph value: {type(value).__name__}")

    def _encode_record(
        self,
        value: object,
        type_name: str,
        field_names: tuple[str, ...],
    ) -> dict[str, str]:
        object_identity = id(value)
        existing = self._record_refs.get(object_identity)
        if existing is not None:
            return {"$ref": existing}
        reference = f"n{len(self.nodes):08d}"
        self._record_refs[object_identity] = reference
        node: dict[str, object] = {
            "ref": reference,
            "type": type_name,
            "fields": {},
        }
        self.nodes.append(node)
        encoded_fields = {field_name: self.encode(getattr(value, field_name)) for field_name in field_names}
        node["fields"] = encoded_fields
        return {"$ref": reference}

    def _encode_observation_sequence(
        self,
        observations: tuple[Observation, ...],
    ) -> str:
        if len(observations) > MAX_OBSERVATION_SEQUENCE_LENGTH:
            raise ValueError("domain graph observation sequence exceeds its length bound")
        identity_key = tuple(id(observation) for observation in observations)
        existing = self._observation_sequence_refs.get(identity_key)
        if existing is not None:
            return existing
        prefix = None if len(observations) == 1 else self._encode_observation_sequence(observations[:-1])
        reference = f"s{len(self.observation_sequences):08d}"
        self._observation_sequence_refs[identity_key] = reference
        self.observation_sequences.append(
            {
                "ref": reference,
                "prefix": prefix,
                "item": self.encode(observations[-1]),
            }
        )
        return reference


class _GraphDecoder:
    def __init__(
        self,
        raw_nodes: Sequence[object],
        raw_observation_sequences: Sequence[object],
        external_objects: Mapping[str, object],
    ) -> None:
        self._nodes: dict[str, dict[str, object]] = {}
        self._node_order: list[str] = []
        for index, raw in enumerate(raw_nodes):
            node = _exact_mapping(raw, {"ref", "type", "fields"}, label=f"domain graph node {index}")
            expected_ref = f"n{index:08d}"
            if node["ref"] != expected_ref:
                raise ValueError("domain graph node references are not canonical")
            type_name = node["type"]
            if not isinstance(type_name, str) or type_name not in _RECORD_SPECS:
                raise ValueError("domain graph node uses a non-allowlisted record type")
            fields = node["fields"]
            expected_fields = set(_RECORD_SPECS[type_name][1])
            if not isinstance(fields, dict) or set(fields) != expected_fields:
                raise ValueError(f"domain graph {type_name} fields differ from the allowlist")
            self._nodes[expected_ref] = node
            self._node_order.append(expected_ref)
        self._observation_sequences: dict[str, dict[str, object]] = {}
        self._observation_sequence_order: list[str] = []
        observation_sequence_lengths: dict[str, int] = {}
        for index, raw in enumerate(raw_observation_sequences):
            sequence = _exact_mapping(
                raw,
                {"ref", "prefix", "item"},
                label=f"domain graph observation sequence {index}",
            )
            expected_ref = f"s{index:08d}"
            if sequence["ref"] != expected_ref:
                raise ValueError("domain graph observation-sequence references are not canonical")
            prefix = sequence["prefix"]
            if prefix is not None:
                if not isinstance(prefix, str) or prefix not in self._observation_sequences:
                    raise ValueError("domain graph observation-sequence prefix is not prior")
            length = 1 if prefix is None else observation_sequence_lengths[cast(str, prefix)] + 1
            if length > MAX_OBSERVATION_SEQUENCE_LENGTH:
                raise ValueError("domain graph observation sequence exceeds its length bound")
            self._observation_sequences[expected_ref] = sequence
            self._observation_sequence_order.append(expected_ref)
            observation_sequence_lengths[expected_ref] = length
        self._external = dict(external_objects)
        for reference in self._external:
            _require_name(reference, label="external graph reference")
        self._decoded: dict[str, object] = {}
        self._decoding: set[str] = set()
        self._decoded_observation_sequences: dict[str, tuple[Observation, ...]] = {}

    def decode(self, value: object) -> object:
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("domain graph contains a non-finite float")
            return value
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if not isinstance(value, dict):
            raise ValueError("domain graph contains a non-JSON value")
        if set(value) == {"$ref"}:
            reference = value["$ref"]
            if not isinstance(reference, str):
                raise ValueError("domain graph node reference is malformed")
            return self._decode_record(reference)
        if set(value) == {"$external"}:
            reference = value["$external"]
            if not isinstance(reference, str) or reference not in self._external:
                raise ValueError("domain graph names an unknown external reference")
            return self._external[reference]
        if set(value) == {"$tuple"}:
            items = value["$tuple"]
            if not isinstance(items, list):
                raise ValueError("domain graph tuple payload is malformed")
            return tuple(self.decode(item) for item in items)
        if set(value) == {"$observation_sequence"}:
            reference = value["$observation_sequence"]
            if not isinstance(reference, str):
                raise ValueError("domain graph observation-sequence reference is malformed")
            return self._decode_observation_sequence(reference)
        if set(value) == {"$enum", "value"}:
            enum_name = value["$enum"]
            if not isinstance(enum_name, str) or enum_name not in _ENUM_SPECS:
                raise ValueError("domain graph uses a non-allowlisted enum")
            try:
                return cast(Any, _ENUM_SPECS[enum_name])(value["value"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"domain graph contains an invalid {enum_name}") from error
        if set(value) == {"$mapping"}:
            items = value["$mapping"]
            if not isinstance(items, list):
                raise ValueError("domain graph mapping payload is malformed")
            result: dict[str, object] = {}
            prior_key: str | None = None
            for item in items:
                if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                    raise ValueError("domain graph mapping entry is malformed")
                key = item[0]
                if prior_key is not None and key <= prior_key:
                    raise ValueError("domain graph mapping keys are not canonical")
                prior_key = key
                result[key] = self.decode(item[1])
            return result
        raise ValueError("domain graph contains an unsupported tagged mapping")

    def _decode_observation_sequence(
        self,
        reference: str,
    ) -> tuple[Observation, ...]:
        existing = self._decoded_observation_sequences.get(reference)
        if existing is not None:
            return existing
        sequence = self._observation_sequences.get(reference)
        if sequence is None:
            raise ValueError("domain graph names an unknown observation sequence")
        prefix_ref = sequence["prefix"]
        prefix = () if prefix_ref is None else self._decode_observation_sequence(cast(str, prefix_ref))
        item = self.decode(sequence["item"])
        if type(item) is not Observation:
            raise ValueError("domain graph observation sequence contains another record type")
        decoded = (*prefix, item)
        self._decoded_observation_sequences[reference] = decoded
        return decoded

    def _decode_record(self, reference: str) -> object:
        existing = self._decoded.get(reference)
        if existing is not None:
            return existing
        node = self._nodes.get(reference)
        if node is None:
            raise ValueError("domain graph names an unknown node reference")
        if reference in self._decoding:
            raise ValueError("domain graph contains a record cycle")
        self._decoding.add(reference)
        try:
            type_name = cast(str, node["type"])
            record_type, field_names = _RECORD_SPECS[type_name]
            raw_fields = cast(dict[str, object], node["fields"])
            decoded_fields = {field_name: self.decode(raw_fields[field_name]) for field_name in field_names}
            record = cast(Any, record_type)(**decoded_fields)
        except (TypeError, ValueError) as error:
            raise ValueError(f"domain graph could not construct {node['type']}") from error
        finally:
            self._decoding.remove(reference)
        self._decoded[reference] = record
        return record

    def require_all_nodes_consumed(self) -> None:
        unreferenced = set(self._node_order) - set(self._decoded)
        if unreferenced:
            raise ValueError("domain graph contains unreachable nodes")
        unused_sequences = set(self._observation_sequence_order) - set(self._decoded_observation_sequences)
        if unused_sequences:
            raise ValueError("domain graph contains unreachable observation sequences")


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return cast(dict[str, object], value)


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _validate_encoded_bounds(value: object) -> None:
    """Bound nested JSON work before constructing any domain record."""

    stack: list[tuple[object, int]] = [(value, 0)]
    values_seen = 0
    while stack:
        current, depth = stack.pop()
        values_seen += 1
        if values_seen > MAX_GRAPH_VALUES:
            raise ValueError("domain graph exceeds its encoded-value bound")
        if depth > MAX_GRAPH_DEPTH:
            raise ValueError("domain graph exceeds its nesting-depth bound")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


__all__ = (
    "GRAPH_SCHEMA",
    "MAX_GRAPH_DEPTH",
    "MAX_GRAPH_JSON_BYTES",
    "MAX_GRAPH_NODES",
    "MAX_GRAPH_VALUES",
    "MAX_OBSERVATION_SEQUENCES",
    "MAX_OBSERVATION_SEQUENCE_LENGTH",
    "belief_external_reference",
    "decode_domain_graph",
    "encode_domain_graph",
    "transition_external_reference",
    "update_external_reference",
)
