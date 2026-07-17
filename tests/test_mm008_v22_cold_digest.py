from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import cold_digest_v22 as cold
from bench.multimodal_mechanism_diagnostics import evidence_v22 as evidence


@dataclass(frozen=True, slots=True)
class _Probe:
    value: object


@dataclass(frozen=True, slots=True)
class _Pair:
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class _Derived:
    source: int
    derived: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "derived", self.source + 1)


@dataclass(slots=True)
class _Cycle:
    value: object = None


@dataclass(frozen=True, slots=True)
class _UnicodeField:
    café: int


class _ArraySubclass(np.ndarray):
    pass


def _digest(value: object, cls: type[object]) -> str:
    return cold.DevelopmentEvidenceDigest(frozenset({cls})).root(
        value,
        root_type=cls,
    )


def _immutable_array(values: list[float]) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return cast(np.ndarray, array)


def test_annotation_closed_evidence_type_universe_is_frozen() -> None:
    assert len(cold.EVIDENCE_DATACLASS_TYPES) == cold.EXPECTED_TYPE_IDENTITY_COUNT == 61
    assert (
        cold.EVIDENCE_TYPE_IDENTITY_FINGERPRINT
        == cold.EXPECTED_TYPE_IDENTITY_FINGERPRINT
        == "bfca8e7d93c4249611fa92655b6db7019dcb08cfd0f71b3fa7993b137d81f1df"
    )
    assert evidence.ExposedSeedScienceEvidence in cold.EVIDENCE_DATACLASS_TYPES
    assert cold.FORMAL_AUTHORITY is False


def test_primitive_types_signed_zero_and_tuple_order_are_distinct() -> None:
    assert _digest(_Probe(True), _Probe) != _digest(_Probe(1), _Probe)
    assert _digest(_Probe(0.0), _Probe) != _digest(_Probe(-0.0), _Probe)
    assert _digest(_Probe((1, 2)), _Probe) != _digest(_Probe((2, 1)), _Probe)
    assert _digest(_Pair(1, 2), _Pair) != _digest(_Pair(2, 1), _Pair)


def test_array_bytes_and_init_false_fields_are_covered() -> None:
    first = _immutable_array([0.0, 1.0])
    changed = _immutable_array([0.0, np.nextafter(1.0, np.inf)])
    assert _digest(_Probe(first), _Probe) != _digest(_Probe(changed), _Probe)
    derived = _Derived(1)
    changed_derived = _Derived(1)
    object.__setattr__(changed_derived, "derived", 3)
    assert _digest(derived, _Derived) != _digest(changed_derived, _Derived)


def test_array_shape_dtype_and_signed_zero_are_covered() -> None:
    flat = _immutable_array([0.0, 1.0])
    shaped = flat.reshape(1, 2)
    shaped.setflags(write=False)
    positive_zero = _immutable_array([0.0])
    negative_zero = _immutable_array([-0.0])
    empty_float = np.array([], dtype=np.float64)
    empty_bool = np.array([], dtype=np.bool_)
    empty_float.setflags(write=False)
    empty_bool.setflags(write=False)
    assert _digest(_Probe(flat), _Probe) != _digest(_Probe(shaped), _Probe)
    assert _digest(_Probe(positive_zero), _Probe) != _digest(_Probe(negative_zero), _Probe)
    assert _digest(_Probe(empty_float), _Probe) != _digest(_Probe(empty_bool), _Probe)


def test_alias_and_value_identical_copy_have_same_value_digest() -> None:
    shared = _immutable_array([1.0, 2.0])
    copied = shared.copy()
    copied.setflags(write=False)
    assert _digest(_Pair(shared, shared), _Pair) == _digest(_Pair(shared, copied), _Pair)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "value",
    [
        {"a": 1},
        [1, 2],
        {1, 2},
        np.float64(1.0),
        float("inf"),
        float("nan"),
        "not-ascii-\N{SNOWMAN}",
    ],
)
def test_unsupported_or_noncanonical_leaves_fail_closed(value: object) -> None:
    with pytest.raises(cold.ColdDigestV22Error):
        _digest(_Probe(value), _Probe)


def test_noncanonical_arrays_fail_closed() -> None:
    writeable = np.array([1.0], dtype=np.float64)
    wrong_dtype = np.array([1], dtype=np.int64)
    nonfinite = _immutable_array([float("inf")])
    non_c: np.ndarray = np.arange(4, dtype=np.float64).reshape(2, 2).T
    non_c.setflags(write=False)
    subclass = cast(_ArraySubclass, _immutable_array([1.0]).view(_ArraySubclass))
    subclass.setflags(write=False)
    for value in (writeable, wrong_dtype, nonfinite, non_c, subclass):
        with pytest.raises(cold.ColdDigestV22Error):
            _digest(_Probe(value), _Probe)


def test_wrong_root_and_unexpected_nested_dataclass_fail_closed() -> None:
    with pytest.raises(cold.ColdDigestV22Error):
        cold.DevelopmentEvidenceDigest(frozenset({_Probe})).root(
            _Pair(1, 2),
            root_type=_Pair,
        )
    with pytest.raises(cold.ColdDigestV22Error):
        _digest(_Probe(_Pair(1, 2)), _Probe)


def test_cycles_and_non_ascii_field_names_fail_with_digest_error() -> None:
    cycle = _Cycle()
    cycle.value = cycle
    with pytest.raises(cold.ColdDigestV22Error, match="dataclass cycle"):
        _digest(cycle, _Cycle)
    with pytest.raises(cold.ColdDigestV22Error, match="field name"):
        _digest(_UnicodeField(1), _UnicodeField)


def test_expected_census_root_is_a_stable_algorithm_kat() -> None:
    assert (
        cold.DevelopmentEvidenceDigest(frozenset({evidence.EvidenceCensus})).root(
            evidence.EXPECTED_CENSUS,
            root_type=evidence.EvidenceCensus,
        )
        == "bcb4eafec39c65df4047ca68e17e2a888b5112cd590682f85f2902ffef436e82"
    )
