"""Fail-closed byte-contract primitives for the MM-008 v2.2 lifecycle.

This module implements only the small, already-frozen framings in protocol
sections 3.1 and 6.3.  It deliberately does not define an output schema, emit a
formal config, create an artifact, or grant formal runtime authority.  The
development KAT role specs below exercise the generic ndarray framing without
claiming membership in the future formal output schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias, cast

import numpy as np

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-contract-primitives-v1"

_OUTPUT_SCHEMA_TAG: Final = b"MM008-v2.2-output-schema\0"
_CONFIG_TAG: Final = b"MM008-v2.2-config\0"
_RECORD_SCOPE_TAG: Final = b"MM008-v2.2-record-scope\0"
_ARRAY_TAG: Final = b"MM008-v2.2-array\0"
_SHA256_BYTES: Final = 32
_UINT16_MAX: Final = (1 << 16) - 1
_UINT32_MAX: Final = (1 << 32) - 1
_UINT8_MAX: Final = (1 << 8) - 1

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ContractV22Error(ValueError):
    """Raised when bytes or values violate a frozen v2.2 primitive grammar."""


class CanonicalJsonError(ContractV22Error):
    """Raised when JSON is not the unique canonical JSON-plus-LF encoding."""


class ScientificArrayContractError(ContractV22Error):
    """Raised when a scientific array is outside its issued typed role."""


class FormalContractIncompleteError(RuntimeError):
    """Raised whenever code attempts formal config emission before authority exists."""


def _validate_json_value(value: object, path: str, ancestors: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return
    if value_type is float:
        if not math.isfinite(cast(float, value)):
            raise CanonicalJsonError(f"{path} contains a nonfinite number")
        return
    if value_type is list:
        identity = id(value)
        if identity in ancestors:
            raise CanonicalJsonError(f"{path} contains a reference cycle")
        ancestors.add(identity)
        try:
            for index, item in enumerate(cast(list[object], value)):
                _validate_json_value(item, f"{path}[{index}]", ancestors)
        finally:
            ancestors.remove(identity)
        return
    if value_type is dict:
        identity = id(value)
        if identity in ancestors:
            raise CanonicalJsonError(f"{path} contains a reference cycle")
        ancestors.add(identity)
        try:
            for key, item in cast(dict[object, object], value).items():
                if type(key) is not str:
                    raise CanonicalJsonError(f"{path} has a non-string object key")
                _validate_json_value(item, f"{path}.{key}", ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise CanonicalJsonError(
        f"{path} has unsupported JSON value type {value_type.__name__}"
    )


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Return the unique protocol JSON encoding followed by exactly one LF."""

    try:
        _validate_json_value(value, "$", set())
    except RecursionError as error:
        raise CanonicalJsonError("JSON value exceeds the supported nesting depth") from error
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise CanonicalJsonError("value cannot be encoded as canonical ASCII JSON") from error
    return encoded + b"\n"


def _reject_nonfinite_constant(token: str) -> NoReturn:
    raise CanonicalJsonError(f"nonfinite JSON constant {token!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJsonError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def parse_canonical_json(data: bytes) -> JsonValue:
    """Parse canonical JSON-plus-LF, rejecting every alternate byte spelling."""

    if type(data) is not bytes:
        raise CanonicalJsonError("canonical JSON input must be immutable bytes")
    if not data.endswith(b"\n"):
        raise CanonicalJsonError("canonical JSON must end in exactly one LF")
    try:
        text = data[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise CanonicalJsonError("canonical JSON bytes must be ASCII") from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except CanonicalJsonError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise CanonicalJsonError("invalid canonical JSON syntax") from error
    try:
        _validate_json_value(parsed, "$", set())
    except RecursionError as error:
        raise CanonicalJsonError("JSON value exceeds the supported nesting depth") from error
    value = cast(JsonValue, parsed)
    if canonical_json_bytes(value) != data:
        raise CanonicalJsonError("JSON bytes are not the unique canonical round trip")
    return value


def _canonical_object_payload(record: bytes, label: str) -> bytes:
    value = parse_canonical_json(record)
    if type(value) is not dict:
        raise CanonicalJsonError(f"{label} must be a canonical JSON object")
    return record[:-1]


def output_schema_digest(record: bytes) -> bytes:
    """Hash one canonical output-schema record, excluding its trailing LF."""

    return hashlib.sha256(
        _OUTPUT_SCHEMA_TAG + _canonical_object_payload(record, "output schema")
    ).digest()


def output_schema_sha256(record: bytes) -> str:
    """Return the lowercase hexadecimal output-schema SHA-256."""

    return output_schema_digest(record).hex()


def config_digest(record: bytes) -> bytes:
    """Hash canonical config bytes without granting formal-config authority."""

    return hashlib.sha256(
        _CONFIG_TAG + _canonical_object_payload(record, "config")
    ).digest()


def config_sha256(record: bytes) -> str:
    """Return the lowercase hexadecimal config SHA-256."""

    return config_digest(record).hex()


def _require_digest(value: bytes, label: str) -> bytes:
    if type(value) is not bytes or len(value) != _SHA256_BYTES:
        raise ContractV22Error(f"{label} must be exactly 32 immutable raw bytes")
    return value


def _ascii_nonempty(value: str, label: str, maximum: int) -> bytes:
    if type(value) is not str or not value:
        raise ContractV22Error(f"{label} must be a nonempty string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ContractV22Error(f"{label} must contain only ASCII characters") from error
    if len(encoded) > maximum:
        raise ContractV22Error(f"{label} exceeds its uint16 byte-length field")
    return encoded


def base_record_scope_digest(raw_config_hash: bytes, context_key: str) -> bytes:
    """Hash one base record scope using the protocol's exact uint16 framing."""

    config_hash = _require_digest(raw_config_hash, "raw config hash")
    key = _ascii_nonempty(context_key, "context key", _UINT16_MAX)
    framed = _RECORD_SCOPE_TAG + config_hash + struct.pack("<H", len(key)) + key
    return hashlib.sha256(framed).digest()


def base_record_scope_sha256(raw_config_hash: bytes, context_key: str) -> str:
    """Return the lowercase hexadecimal base-record-scope SHA-256."""

    return base_record_scope_digest(raw_config_hash, context_key).hex()


class ScientificScopeKind(Enum):
    """Owner categories already named by the frozen generic-array grammar."""

    BASE_RECORD = "base-record"
    NONGRID_FIT = "nongrid-fit"
    PERSISTENCE = "persistence"
    OBJECTIVE = "objective"


@dataclass(frozen=True, slots=True)
class ScientificScope:
    """A raw scientific scope digest paired with its semantic owner category."""

    kind: ScientificScopeKind
    digest: bytes

    def __post_init__(self) -> None:
        if type(self.kind) is not ScientificScopeKind:
            raise ScientificArrayContractError("scientific scope kind is not issued")
        try:
            _require_digest(self.digest, "scientific scope hash")
        except ContractV22Error as error:
            raise ScientificArrayContractError(str(error)) from error


def base_record_scope(raw_config_hash: bytes, context_key: str) -> ScientificScope:
    """Construct the typed base scope accepted by matching array-role specs."""

    return ScientificScope(
        ScientificScopeKind.BASE_RECORD,
        base_record_scope_digest(raw_config_hash, context_key),
    )


class ScientificDType(IntEnum):
    """The five one-byte dtype codes frozen by protocol section 3.1."""

    UINT8 = 0
    UINT16 = 1
    UINT32 = 2
    INT64 = 3
    FLOAT64 = 4


_DTYPES: Final[dict[ScientificDType, np.dtype[np.generic]]] = {
    ScientificDType.UINT8: np.dtype("u1"),
    ScientificDType.UINT16: np.dtype("<u2"),
    ScientificDType.UINT32: np.dtype("<u4"),
    ScientificDType.INT64: np.dtype("<i8"),
    ScientificDType.FLOAT64: np.dtype("<f8"),
}

_ROLE_ISSUER: Final = object()


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ScientificArrayRoleSpec:
    """An immutable, module-issued role/scope/shape/dtype binding.

    There is intentionally no public constructor or role-registration API.  A
    future exhaustive output-schema implementation must issue the formal closed
    set.  Identity validation prevents a caller-created lookalike from granting
    itself authority.
    """

    role: str
    scope_kind: ScientificScopeKind
    shape: tuple[int, ...]
    dtype: ScientificDType
    boolean_mask: bool
    formal: bool

    def __new__(cls, issuer: object) -> ScientificArrayRoleSpec:
        if issuer is not _ROLE_ISSUER:
            raise ScientificArrayContractError("array role specs are module-issued only")
        return object.__new__(cls)

    def __init__(self, issuer: object) -> None:
        del issuer


def _issue_role_spec(
    role: str,
    scope_kind: ScientificScopeKind,
    shape: tuple[int, ...],
    dtype: ScientificDType,
    *,
    boolean_mask: bool = False,
    formal: bool = False,
) -> ScientificArrayRoleSpec:
    role_bytes = _ascii_nonempty(role, "array role", _UINT16_MAX)
    if len(role_bytes) != len(role):
        raise RuntimeError("ASCII array role length changed unexpectedly")
    if type(scope_kind) is not ScientificScopeKind or type(dtype) is not ScientificDType:
        raise RuntimeError("array role spec uses an unissued enum value")
    if type(shape) is not tuple or len(shape) > _UINT8_MAX:
        raise RuntimeError("array role spec rank is outside uint8")
    if any(type(dimension) is not int or not 0 <= dimension <= _UINT32_MAX for dimension in shape):
        raise RuntimeError("array role spec dimension is outside uint32")
    if type(boolean_mask) is not bool or type(formal) is not bool:
        raise RuntimeError("array role spec flags must be exact booleans")
    if boolean_mask and dtype is not ScientificDType.UINT8:
        raise RuntimeError("boolean masks must serialize under the uint8 dtype code")
    spec = ScientificArrayRoleSpec(_ROLE_ISSUER)
    object.__setattr__(spec, "role", role)
    object.__setattr__(spec, "scope_kind", scope_kind)
    object.__setattr__(spec, "shape", shape)
    object.__setattr__(spec, "dtype", dtype)
    object.__setattr__(spec, "boolean_mask", boolean_mask)
    object.__setattr__(spec, "formal", formal)
    return spec


class DevelopmentKatRole(Enum):
    """Closed non-formal roles used solely to test every primitive dtype branch."""

    UINT8 = "contract-kat/uint8"
    UINT16 = "contract-kat/uint16"
    UINT32 = "contract-kat/uint32"
    INT64 = "contract-kat/int64"
    FLOAT64 = "contract-kat/float64"
    BOOLEAN_MASK = "contract-kat/boolean-mask"


_KAT_ROLE_SPECS: Final = MappingProxyType(
    {
        DevelopmentKatRole.UINT8: _issue_role_spec(
            DevelopmentKatRole.UINT8.value,
            ScientificScopeKind.BASE_RECORD,
            (3,),
            ScientificDType.UINT8,
        ),
        DevelopmentKatRole.UINT16: _issue_role_spec(
            DevelopmentKatRole.UINT16.value,
            ScientificScopeKind.BASE_RECORD,
            (2,),
            ScientificDType.UINT16,
        ),
        DevelopmentKatRole.UINT32: _issue_role_spec(
            DevelopmentKatRole.UINT32.value,
            ScientificScopeKind.BASE_RECORD,
            (2, 2),
            ScientificDType.UINT32,
        ),
        DevelopmentKatRole.INT64: _issue_role_spec(
            DevelopmentKatRole.INT64.value,
            ScientificScopeKind.BASE_RECORD,
            (2,),
            ScientificDType.INT64,
        ),
        DevelopmentKatRole.FLOAT64: _issue_role_spec(
            DevelopmentKatRole.FLOAT64.value,
            ScientificScopeKind.BASE_RECORD,
            (2, 3),
            ScientificDType.FLOAT64,
        ),
        DevelopmentKatRole.BOOLEAN_MASK: _issue_role_spec(
            DevelopmentKatRole.BOOLEAN_MASK.value,
            ScientificScopeKind.BASE_RECORD,
            (2, 3),
            ScientificDType.UINT8,
            boolean_mask=True,
        ),
    }
)
_AUTHORIZED_ROLE_SPECS: Final = frozenset(id(spec) for spec in _KAT_ROLE_SPECS.values())


def development_kat_role_spec(role: DevelopmentKatRole) -> ScientificArrayRoleSpec:
    """Return one closed, explicitly non-formal KAT role spec."""

    if type(role) is not DevelopmentKatRole:
        raise ScientificArrayContractError("development KAT role is not issued")
    return _KAT_ROLE_SPECS[role]


def _authorized_spec(spec: ScientificArrayRoleSpec) -> ScientificArrayRoleSpec:
    if type(spec) is not ScientificArrayRoleSpec or id(spec) not in _AUTHORIZED_ROLE_SPECS:
        raise ScientificArrayContractError("array role spec is not in the closed issued set")
    if spec.formal:
        raise ScientificArrayContractError("formal array-role authority is not yet available")
    return spec


def _array_payload(spec: ScientificArrayRoleSpec, array: np.ndarray) -> bytes:
    if type(array) is not np.ndarray:
        raise ScientificArrayContractError("scientific array must be an exact numpy ndarray")
    if array.shape != spec.shape:
        raise ScientificArrayContractError(
            f"array shape {array.shape!r} does not match issued shape {spec.shape!r}"
        )
    if not array.flags.c_contiguous:
        raise ScientificArrayContractError("scientific array must already be C-contiguous")
    if spec.boolean_mask:
        if array.dtype != np.dtype(np.bool_):
            raise ScientificArrayContractError("boolean-mask role requires bool input")
        normalized = np.asarray(array, dtype=np.uint8, order="C")
        if np.any((normalized != 0) & (normalized != 1)):
            raise ScientificArrayContractError("boolean normalization produced a non-bit value")
        return normalized.tobytes(order="C")
    expected_dtype = _DTYPES[spec.dtype]
    if array.dtype != expected_dtype or array.dtype.str != expected_dtype.str:
        raise ScientificArrayContractError(
            f"array dtype {array.dtype.str!r} does not match issued little-endian dtype "
            f"{expected_dtype.str!r}"
        )
    if spec.dtype is ScientificDType.FLOAT64 and not bool(np.all(np.isfinite(array))):
        raise ScientificArrayContractError("floating scientific array must be finite")
    little_endian = array.astype(expected_dtype.newbyteorder("<"), order="C", casting="no", copy=False)
    return little_endian.tobytes(order="C")


def scientific_array_digest(
    scope: ScientificScope,
    spec: ScientificArrayRoleSpec,
    array: np.ndarray,
) -> bytes:
    """Hash an array under one closed typed role and matching owner scope."""

    issued = _authorized_spec(spec)
    if type(scope) is not ScientificScope or scope.kind is not issued.scope_kind:
        raise ScientificArrayContractError("array role is bound to a different scope kind")
    _require_digest(scope.digest, "scientific scope hash")
    role = issued.role.encode("ascii")
    dimensions = b"".join(struct.pack("<I", dimension) for dimension in issued.shape)
    framed = b"".join(
        (
            _ARRAY_TAG,
            scope.digest,
            struct.pack("<H", len(role)),
            role,
            struct.pack("<B", int(issued.dtype)),
            struct.pack("<B", len(issued.shape)),
            dimensions,
            _array_payload(issued, array),
        )
    )
    return hashlib.sha256(framed).digest()


def scientific_array_sha256(
    scope: ScientificScope,
    spec: ScientificArrayRoleSpec,
    array: np.ndarray,
) -> str:
    """Return the lowercase hexadecimal typed scientific-array SHA-256."""

    return scientific_array_digest(scope, spec, array).hex()


@dataclass(frozen=True, slots=True)
class FormalAuthorityStatus:
    """Immutable statement of the deliberately incomplete formal contract."""

    formal_config_emission_available: bool
    exhaustive_output_role_bindings_available: bool
    native_runtime_policy_bindings_available: bool
    missing_bindings: tuple[str, ...]

    def __post_init__(self) -> None:
        flags = (
            self.formal_config_emission_available,
            self.exhaustive_output_role_bindings_available,
            self.native_runtime_policy_bindings_available,
        )
        if any(type(flag) is not bool for flag in flags):
            raise ContractV22Error("formal authority flags must be exact booleans")
        if type(self.missing_bindings) is not tuple or any(
            type(binding) is not str or not binding for binding in self.missing_bindings
        ):
            raise ContractV22Error(
                "formal authority missing bindings must be a tuple of nonempty strings"
            )
        if len(set(self.missing_bindings)) != len(self.missing_bindings):
            raise ContractV22Error("formal authority missing bindings must be unique")
        prerequisites_complete = (
            self.exhaustive_output_role_bindings_available
            and self.native_runtime_policy_bindings_available
            and not self.missing_bindings
        )
        if self.formal_config_emission_available is not prerequisites_complete:
            raise ContractV22Error(
                "formal config authority must exactly match complete role/runtime bindings"
            )

    @property
    def complete(self) -> bool:
        """Return whether all authority needed for formal config emission exists."""

        return (
            self.formal_config_emission_available
            and self.exhaustive_output_role_bindings_available
            and self.native_runtime_policy_bindings_available
            and not self.missing_bindings
        )


FORMAL_AUTHORITY_STATUS: Final = FormalAuthorityStatus(
    formal_config_emission_available=False,
    exhaustive_output_role_bindings_available=False,
    native_runtime_policy_bindings_available=False,
    missing_bindings=(
        "exhaustive output-schema array-role/scope/shape/dtype bindings",
        "native runtime policy and role bindings",
    ),
)


def require_formal_config_emission_authority() -> NoReturn:
    """Fail closed until the exhaustive schema and native-runtime bindings exist."""

    missing = "; ".join(FORMAL_AUTHORITY_STATUS.missing_bindings)
    raise FormalContractIncompleteError(
        f"MM-008 v2.2 formal config emission is unavailable: {missing}"
    )


__all__ = [
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "CanonicalJsonError",
    "ContractV22Error",
    "DevelopmentKatRole",
    "FORMAL_AUTHORITY_STATUS",
    "FormalAuthorityStatus",
    "FormalContractIncompleteError",
    "JsonScalar",
    "JsonValue",
    "ScientificArrayContractError",
    "ScientificArrayRoleSpec",
    "ScientificDType",
    "ScientificScope",
    "ScientificScopeKind",
    "base_record_scope",
    "base_record_scope_digest",
    "base_record_scope_sha256",
    "canonical_json_bytes",
    "config_digest",
    "config_sha256",
    "development_kat_role_spec",
    "output_schema_digest",
    "output_schema_sha256",
    "parse_canonical_json",
    "require_formal_config_emission_authority",
    "scientific_array_digest",
    "scientific_array_sha256",
]
