from __future__ import annotations

import copy
from collections.abc import Callable

import pytest

from bench.world_model_lifecycle import domain_graph
from bench.world_model_lifecycle.checkpoint import canonical_json_bytes
from bench.world_model_lifecycle.domain_graph import (
    decode_domain_graph,
    encode_domain_graph,
)
from prospect.domain import (
    Evidence,
    EvidenceLineage,
    EvidenceOrigin,
    InformationSet,
    Observation,
    Provenance,
    TimePoint,
    TrustLevel,
)


def _observation(index: int) -> Observation:
    identity = f"observation-{index}"
    point = TimePoint(index)
    return Observation(
        observation_id=identity,
        agent_id="agent",
        modality="fixture",
        evidence=Evidence(
            evidence_id=identity,
            payload={"index": index, "$ref": "opaque-payload-key"},
            occurred_at=point,
            available_at=point,
            lineage=EvidenceLineage(
                evidence_id=identity,
                origin=EvidenceOrigin.OBSERVED,
                provenance=Provenance(
                    source_id="fixture",
                    trust=TrustLevel.VERIFIED,
                    source_kind="unit_test",
                ),
            ),
        ),
    )


def test_domain_graph_round_trip_preserves_shared_prefixes_in_linear_space() -> None:
    observations: tuple[Observation, ...] = ()
    information_sets: list[InformationSet] = []
    for index in range(200):
        observations = (*observations, _observation(index))
        information_sets.append(
            InformationSet(
                information_set_id=f"information-{index}",
                agent_id="agent",
                as_of=TimePoint(index),
                observations=observations,
                memory_version=f"memory-{index}",
            )
        )

    encoded = encode_domain_graph({"sets": tuple(information_sets)})
    payload = canonical_json_bytes(encoded)
    decoded = decode_domain_graph(encoded)
    reencoded = encode_domain_graph(decoded)
    decoded_sets = decoded["sets"]

    assert isinstance(decoded_sets, tuple)
    assert len(encoded["observation_sequences"]) == 200
    assert len(encoded["nodes"]) < 1_500
    assert len(payload) < 1_000_000
    assert canonical_json_bytes(reencoded) == payload
    assert decoded_sets[-1].observations[0] is decoded_sets[0].observations[-1]
    assert decoded_sets[-1].observations[99] is decoded_sets[99].observations[-1]
    assert decoded_sets[-1].observations[-1].evidence.payload["$ref"] == ("opaque-payload-key")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda graph: graph["roots"].__setitem__("root", {"$unknown": "x"}),
            "unsupported tagged mapping",
        ),
        (
            lambda graph: graph["roots"].__setitem__("root", {"$ref": "n99999999"}),
            "unknown node reference",
        ),
        (
            lambda graph: graph["roots"].__setitem__("root", {"$external": "transition:missing"}),
            "unknown external reference",
        ),
        (
            lambda graph: graph["nodes"][0].__setitem__("type", "ArbitraryClass"),
            "non-allowlisted record type",
        ),
    ],
)
def test_domain_graph_rejects_unknown_tags_types_and_references(
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    graph = encode_domain_graph({"root": TimePoint(1)})
    mutation(graph)

    with pytest.raises(ValueError, match=match):
        decode_domain_graph(graph)


def test_domain_graph_rejects_unreachable_nodes_and_explicit_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = encode_domain_graph({"root": TimePoint(1)})
    unreachable = copy.deepcopy(graph)
    unreachable["roots"]["root"] = 1
    with pytest.raises(ValueError, match="unreachable nodes"):
        decode_domain_graph(unreachable)

    monkeypatch.setattr(domain_graph, "MAX_GRAPH_NODES", 0)
    with pytest.raises(ValueError, match="root/node bound"):
        decode_domain_graph(graph)

    monkeypatch.setattr(domain_graph, "MAX_GRAPH_NODES", 250_000)
    monkeypatch.setattr(domain_graph, "MAX_GRAPH_DEPTH", 1)
    with pytest.raises(ValueError, match="nesting-depth bound"):
        decode_domain_graph(graph)

    monkeypatch.setattr(domain_graph, "MAX_GRAPH_DEPTH", 64)
    monkeypatch.setattr(domain_graph, "MAX_OBSERVATION_SEQUENCE_LENGTH", 1)
    with pytest.raises(ValueError, match="observation sequence exceeds"):
        encode_domain_graph({"observations": (_observation(1), _observation(2))})
