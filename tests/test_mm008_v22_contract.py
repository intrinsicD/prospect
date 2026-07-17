"""Independent byte KATs and fail-closed tests for MM-008 v2.2 contracts."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import contract_v22 as contract

_RAW_CONFIG_HASH = bytes(range(32))
_CONTEXT_KEY = "scenario/seed/row"
_EXPECTED_BASE_SCOPE = "ed7ad7a61816f6a7dc4e97b60bccb536bf589d2a206af6dc442b3053b7a36579"


def _stdlib_canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _stdlib_array_hash(
    scope: bytes,
    role: str,
    dtype_code: int,
    shape: tuple[int, ...],
    payload: bytes,
) -> str:
    role_bytes = role.encode("ascii")
    framing = b"".join(
        (
            b"MM008-v2.2-array\0",
            scope,
            struct.pack("<H", len(role_bytes)),
            role_bytes,
            struct.pack("<B", dtype_code),
            struct.pack("<B", len(shape)),
            *(struct.pack("<I", dimension) for dimension in shape),
            payload,
        )
    )
    return hashlib.sha256(framing).hexdigest()


def test_protocol_binding_and_formal_emission_are_explicitly_incomplete() -> None:
    assert contract.PROTOCOL_SHA256 == (
        "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
    )
    status = contract.FORMAL_AUTHORITY_STATUS
    assert not status.complete
    assert not status.formal_config_emission_available
    assert not status.exhaustive_output_role_bindings_available
    assert not status.native_runtime_policy_bindings_available
    assert status.missing_bindings == (
        "exhaustive output-schema array-role/scope/shape/dtype bindings",
        "native runtime policy and role bindings",
    )
    with pytest.raises(
        contract.FormalContractIncompleteError,
        match="formal config emission is unavailable",
    ):
        contract.require_formal_config_emission_authority()
    with pytest.raises(FrozenInstanceError):
        status.formal_config_emission_available = True  # type: ignore[misc]


def test_authority_status_rejects_plausible_inconsistent_or_malformed_states() -> None:
    with pytest.raises(contract.ContractV22Error, match="exact booleans"):
        contract.FormalAuthorityStatus(
            formal_config_emission_available=np.bool_(False),  # type: ignore[arg-type]
            exhaustive_output_role_bindings_available=False,
            native_runtime_policy_bindings_available=False,
            missing_bindings=("missing",),
        )
    with pytest.raises(contract.ContractV22Error, match="tuple of nonempty"):
        contract.FormalAuthorityStatus(False, False, False, ("",))
    with pytest.raises(contract.ContractV22Error, match="unique"):
        contract.FormalAuthorityStatus(False, False, False, ("missing", "missing"))
    with pytest.raises(contract.ContractV22Error, match="exactly match"):
        contract.FormalAuthorityStatus(True, True, True, ("still missing",))
    with pytest.raises(contract.ContractV22Error, match="exactly match"):
        contract.FormalAuthorityStatus(False, True, True, ())

    complete = contract.FormalAuthorityStatus(True, True, True, ())
    assert complete.complete


def test_canonical_json_plus_lf_has_one_unique_ascii_encoding() -> None:
    value: contract.JsonValue = {
        "z": [1, 2.5, True, None, "é"],
        "a": {"b": "x"},
    }
    expected = b'{"a":{"b":"x"},"z":[1,2.5,true,null,"\\u00e9"]}\n'
    assert contract.canonical_json_bytes(value) == expected
    assert contract.parse_canonical_json(expected) == value
    assert all(byte < 128 for byte in expected)
    assert expected.count(b"\n") == 1


@pytest.mark.parametrize(
    "record",
    (
        b'{"a":1, "b":2}\n',
        b'{"b":2,"a":1}\n',
        b' {"a":1,"b":2}\n',
        b'{"a":1,"b":2} \n',
        b'{"a":1,"b":2}\r\n',
        b'{"a":1,"b":2}',
        b'{"a":1,"b":2}\n\n',
        b'{"a":1.0e0,"b":2}\n',
        b'{"a":-0,"b":2}\n',
        b'{"a":"\\u0061","b":2}\n',
    ),
)
def test_parser_rejects_alternate_whitespace_and_nonroundtripping_bytes(
    record: bytes,
) -> None:
    with pytest.raises(contract.CanonicalJsonError):
        contract.parse_canonical_json(record)


@pytest.mark.parametrize(
    "record",
    (
        b'{"a":1,"a":2}\n',
        b'{"a":{"b":1,"b":2}}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":-Infinity}\n',
        b'{"x":1e400}\n',
        '{"x":"é"}\n'.encode(),
    ),
)
def test_parser_rejects_duplicate_nonfinite_and_nonascii_inputs(record: bytes) -> None:
    with pytest.raises(contract.CanonicalJsonError):
        contract.parse_canonical_json(record)


def test_serializer_rejects_non_json_types_nonfinite_values_and_cycles() -> None:
    invalid: tuple[object, ...] = (
        (1, 2),
        {1: "value"},
        np.int64(3),
        {"x": float("nan")},
        {"x": float("inf")},
    )
    for value in invalid:
        with pytest.raises(contract.CanonicalJsonError):
            contract.canonical_json_bytes(value)  # type: ignore[arg-type]

    cyclic: list[contract.JsonValue] = []
    cyclic.append(cyclic)
    with pytest.raises(contract.CanonicalJsonError, match="reference cycle"):
        contract.canonical_json_bytes(cyclic)
    with pytest.raises(contract.CanonicalJsonError, match="immutable bytes"):
        contract.parse_canonical_json(bytearray(b"null\n"))  # type: ignore[arg-type]


def test_excessive_json_nesting_fails_with_the_contract_exception() -> None:
    nested: object = 0
    for _ in range(2_000):
        nested = [nested]
    with pytest.raises(contract.CanonicalJsonError, match="nesting depth"):
        contract.canonical_json_bytes(nested)  # type: ignore[arg-type]

    nested_bytes = b"[" * 2_000 + b"0" + b"]" * 2_000 + b"\n"
    with pytest.raises(contract.CanonicalJsonError):
        contract.parse_canonical_json(nested_bytes)


def test_output_schema_and_config_hashes_match_independent_stdlib_kats() -> None:
    schema_value = {
        "array_roles": [],
        "schema_version": "kat-1",
        "unicode": "λ",
    }
    config_value = {
        "batch_size": 128,
        "enabled": True,
        "limits": [-4.0, 4.0],
        "output_schema_sha256": "ab" * 32,
    }
    schema_record = _stdlib_canonical(schema_value)
    config_record = _stdlib_canonical(config_value)
    expected_schema = hashlib.sha256(
        b"MM008-v2.2-output-schema\0" + schema_record[:-1]
    ).hexdigest()
    expected_config = hashlib.sha256(
        b"MM008-v2.2-config\0" + config_record[:-1]
    ).hexdigest()

    assert expected_schema == "f6537b4089db2f9b61ba15ecd2fa71881bc37710443836c83e0af47adc061bf5"
    assert expected_config == "01891fe4166f9e8cdf6e3591b588ba9f677e8534b9178c1ca061524063d53638"
    assert contract.output_schema_sha256(schema_record) == expected_schema
    assert contract.output_schema_digest(schema_record) == bytes.fromhex(expected_schema)
    assert contract.config_sha256(config_record) == expected_config
    assert contract.config_digest(config_record) == bytes.fromhex(expected_config)

    assert hashlib.sha256(b"MM008-v2.2-output-schema\0" + schema_record).hexdigest() != expected_schema
    assert hashlib.sha256(b"MM008-v2.2-config\0" + config_record).hexdigest() != expected_config
    for mutation in (schema_record[:-1], b" " + schema_record, schema_record + b"\n"):
        with pytest.raises(contract.CanonicalJsonError):
            contract.output_schema_digest(mutation)
    for non_object in (b"null\n", b"[]\n", b'"config"\n'):
        with pytest.raises(contract.CanonicalJsonError, match="JSON object"):
            contract.output_schema_digest(non_object)
        with pytest.raises(contract.CanonicalJsonError, match="JSON object"):
            contract.config_digest(non_object)


def test_base_record_scope_matches_independent_uint16_framing_and_is_strict() -> None:
    key = _CONTEXT_KEY.encode("ascii")
    expected = hashlib.sha256(
        b"MM008-v2.2-record-scope\0"
        + _RAW_CONFIG_HASH
        + struct.pack("<H", len(key))
        + key
    ).digest()
    assert expected.hex() == _EXPECTED_BASE_SCOPE
    assert contract.base_record_scope_digest(_RAW_CONFIG_HASH, _CONTEXT_KEY) == expected
    assert contract.base_record_scope_sha256(_RAW_CONFIG_HASH, _CONTEXT_KEY) == expected.hex()
    typed_scope = contract.base_record_scope(_RAW_CONFIG_HASH, _CONTEXT_KEY)
    assert typed_scope == contract.ScientificScope(
        contract.ScientificScopeKind.BASE_RECORD,
        expected,
    )

    with pytest.raises(contract.ContractV22Error, match="32 immutable raw bytes"):
        contract.base_record_scope_digest(b"x" * 31, _CONTEXT_KEY)
    with pytest.raises(contract.ContractV22Error, match="32 immutable raw bytes"):
        contract.base_record_scope_digest(bytearray(32), _CONTEXT_KEY)  # type: ignore[arg-type]
    with pytest.raises(contract.ContractV22Error, match="nonempty"):
        contract.base_record_scope_digest(_RAW_CONFIG_HASH, "")
    with pytest.raises(contract.ContractV22Error, match="ASCII"):
        contract.base_record_scope_digest(_RAW_CONFIG_HASH, "λ")
    with pytest.raises(contract.ContractV22Error, match="uint16"):
        contract.base_record_scope_digest(_RAW_CONFIG_HASH, "x" * 65_536)


def test_all_generic_array_dtype_codes_match_independent_stdlib_kats() -> None:
    scope = contract.base_record_scope(_RAW_CONFIG_HASH, _CONTEXT_KEY)
    assert scope.digest.hex() == _EXPECTED_BASE_SCOPE
    cases = (
        (
            contract.DevelopmentKatRole.UINT8,
            0,
            (3,),
            np.asarray((0, 1, 255), dtype=np.uint8),
            bytes((0, 1, 255)),
            "988eca5c1e5d6806741355e63fc5d965b6de701ed469327c3cff406f95a6e6a0",
        ),
        (
            contract.DevelopmentKatRole.UINT16,
            1,
            (2,),
            np.asarray((1, 513), dtype="<u2"),
            struct.pack("<HH", 1, 513),
            "2319ddfd35e1202e710a9746f456d154234e32f807c6102b2e99047a36222e9e",
        ),
        (
            contract.DevelopmentKatRole.UINT32,
            2,
            (2, 2),
            np.asarray(((0, 1), (65_537, 2**32 - 1)), dtype="<u4"),
            struct.pack("<IIII", 0, 1, 65_537, 2**32 - 1),
            "584e22ee875cede8a7d11fa5de2bd74fba2c43bc768345e41ac33f98b11d15e1",
        ),
        (
            contract.DevelopmentKatRole.INT64,
            3,
            (2,),
            np.asarray((-1, 2**40), dtype="<i8"),
            struct.pack("<qq", -1, 2**40),
            "43d3a028588468ab7c265676f909b3e84b54139b456a5653ed25c5c60b5ecb05",
        ),
        (
            contract.DevelopmentKatRole.FLOAT64,
            4,
            (2, 3),
            np.asarray(((0.0, -0.0, 1.5), (-2.25, 1e100, -1e-100)), dtype="<f8"),
            struct.pack("<6d", 0.0, -0.0, 1.5, -2.25, 1e100, -1e-100),
            "f436134041d65363a83c9c22ae429cca488f558800ddd953c2eb49f6af500fae",
        ),
        (
            contract.DevelopmentKatRole.BOOLEAN_MASK,
            0,
            (2, 3),
            np.asarray(((True, False, True), (False, False, True)), dtype=np.bool_),
            bytes((1, 0, 1, 0, 0, 1)),
            "14c3bd1c1ea653ecfc2e6bdc8da2ad02d991388b6580a69ab988fb2fc365213c",
        ),
    )
    for role, dtype_code, shape, array, payload, frozen_hash in cases:
        spec = contract.development_kat_role_spec(role)
        assert not spec.formal
        assert int(spec.dtype) == dtype_code
        assert spec.shape == shape
        expected = _stdlib_array_hash(scope.digest, role.value, dtype_code, shape, payload)
        assert expected == frozen_hash
        assert contract.scientific_array_sha256(scope, spec, array) == expected
        assert contract.scientific_array_digest(scope, spec, array) == bytes.fromhex(expected)


def test_array_roles_are_immutable_closed_and_cannot_self_authorize() -> None:
    scope = contract.base_record_scope(_RAW_CONFIG_HASH, _CONTEXT_KEY)
    spec = contract.development_kat_role_spec(contract.DevelopmentKatRole.FLOAT64)
    array = np.zeros((2, 3), dtype="<f8")
    with pytest.raises(FrozenInstanceError):
        spec.shape = (6,)  # type: ignore[misc]
    with pytest.raises(contract.ScientificArrayContractError, match="module-issued"):
        contract.ScientificArrayRoleSpec(object())
    with pytest.raises(contract.ScientificArrayContractError, match="KAT role"):
        contract.development_kat_role_spec("contract-kat/float64")  # type: ignore[arg-type]

    forged = object.__new__(contract.ScientificArrayRoleSpec)
    object.__setattr__(forged, "role", spec.role)
    object.__setattr__(forged, "scope_kind", spec.scope_kind)
    object.__setattr__(forged, "shape", spec.shape)
    object.__setattr__(forged, "dtype", spec.dtype)
    object.__setattr__(forged, "boolean_mask", spec.boolean_mask)
    object.__setattr__(forged, "formal", spec.formal)
    with pytest.raises(contract.ScientificArrayContractError, match="closed issued set"):
        contract.scientific_array_digest(scope, forged, array)

    wrong_scope = contract.ScientificScope(
        contract.ScientificScopeKind.OBJECTIVE,
        scope.digest,
    )
    with pytest.raises(contract.ScientificArrayContractError, match="different scope"):
        contract.scientific_array_digest(wrong_scope, spec, array)


def test_array_hash_rejects_shape_dtype_layout_endian_and_float_mutations() -> None:
    scope = contract.base_record_scope(_RAW_CONFIG_HASH, _CONTEXT_KEY)
    float_spec = contract.development_kat_role_spec(contract.DevelopmentKatRole.FLOAT64)
    uint16_spec = contract.development_kat_role_spec(contract.DevelopmentKatRole.UINT16)
    mask_spec = contract.development_kat_role_spec(contract.DevelopmentKatRole.BOOLEAN_MASK)

    with pytest.raises(contract.ScientificArrayContractError, match="shape"):
        contract.scientific_array_digest(scope, float_spec, np.zeros((6,), dtype="<f8"))
    with pytest.raises(contract.ScientificArrayContractError, match="dtype"):
        contract.scientific_array_digest(scope, float_spec, np.zeros((2, 3), dtype="<f4"))
    with pytest.raises(contract.ScientificArrayContractError, match="C-contiguous"):
        contract.scientific_array_digest(
            scope,
            float_spec,
            np.asfortranarray(np.zeros((2, 3), dtype="<f8")),
        )
    with pytest.raises(contract.ScientificArrayContractError, match="finite"):
        contract.scientific_array_digest(
            scope,
            float_spec,
            np.asarray(((0.0, 1.0, np.nan), (2.0, 3.0, 4.0)), dtype="<f8"),
        )
    with pytest.raises(contract.ScientificArrayContractError, match="finite"):
        contract.scientific_array_digest(
            scope,
            float_spec,
            np.asarray(((0.0, 1.0, np.inf), (2.0, 3.0, 4.0)), dtype="<f8"),
        )
    with pytest.raises(contract.ScientificArrayContractError, match="little-endian"):
        contract.scientific_array_digest(
            scope,
            uint16_spec,
            np.asarray((1, 513), dtype=">u2"),
        )
    with pytest.raises(contract.ScientificArrayContractError, match="bool input"):
        contract.scientific_array_digest(
            scope,
            mask_spec,
            np.asarray(((1, 0, 1), (0, 0, 1)), dtype=np.uint8),
        )
    with pytest.raises(contract.ScientificArrayContractError, match="shape"):
        contract.scientific_array_digest(
            scope,
            mask_spec,
            np.asarray((True, False, True, False, False, True), dtype=np.bool_),
        )


def test_typed_scope_rejects_mutable_or_wrong_length_digests() -> None:
    with pytest.raises(contract.ScientificArrayContractError, match="32 immutable raw bytes"):
        contract.ScientificScope(contract.ScientificScopeKind.BASE_RECORD, b"x" * 31)
    with pytest.raises(contract.ScientificArrayContractError, match="32 immutable raw bytes"):
        contract.ScientificScope(
            contract.ScientificScopeKind.BASE_RECORD,
            bytearray(32),  # type: ignore[arg-type]
        )
