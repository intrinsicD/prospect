"""Fail-closed development digest for MM-008 v2.2 exposed-seed evidence.

The digest is a test instrument for comparing complete in-memory scientific
graphs across clean Python processes.  It is deliberately not a serializer,
manifest, receipt, config authority, or formal protocol artifact.  Only the
closed immutable runtime universe reachable from
``ExposedSeedScienceEvidence`` is accepted.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Final, cast, get_args, get_type_hints

import numpy as np

from bench.multimodal_mechanism_diagnostics import evidence_v22 as evidence

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-dev-cold-root-v1"
FORMAL_AUTHORITY: Final = False
EXPECTED_TYPE_IDENTITY_COUNT: Final = 61
EXPECTED_TYPE_IDENTITY_FINGERPRINT: Final = "bfca8e7d93c4249611fa92655b6db7019dcb08cfd0f71b3fa7993b137d81f1df"

if evidence.PROTOCOL_SHA256 != PROTOCOL_SHA256:
    raise RuntimeError("MM-008 v2.2 cold digest dependency binds a different protocol")

_NODE_DOMAIN: Final = SCHEMA_VERSION.encode("ascii") + b"\0"
_TYPE_IDENTITY_DOMAIN: Final = b"MM008-v2.2-dev-root-type-identities-v2\0"
_ARRAY_DTYPES: Final = frozenset({"|b1", "<f8"})


class ColdDigestV22Error(ValueError):
    """Raised when a graph is outside the closed development digest universe."""


def _node(tag: bytes, parts: tuple[bytes, ...]) -> bytes:
    digest = hashlib.sha256(_NODE_DOMAIN)
    digest.update(struct.pack("<Q", len(tag)))
    digest.update(tag)
    digest.update(struct.pack("<Q", len(parts)))
    for part in parts:
        digest.update(struct.pack("<Q", len(part)))
        digest.update(part)
    return digest.digest()


def _type_parts(cls: type[object]) -> tuple[bytes, bytes]:
    try:
        return cls.__module__.encode("ascii"), cls.__qualname__.encode("ascii")
    except UnicodeEncodeError as error:
        raise ColdDigestV22Error("non-ASCII runtime type identity") from error


def _int_payload(value: int) -> bytes:
    magnitude = abs(value)
    size = max(1, (magnitude.bit_length() + 7) // 8)
    sign = b"\x01" if value < 0 else b"\x00"
    return sign + magnitude.to_bytes(size, "big")


def _dataclass_type_closure(root: type[object]) -> frozenset[type[object]]:
    pending = [root]
    found: set[type[object]] = set()
    while pending:
        cls = pending.pop()
        if cls in found:
            continue
        if not is_dataclass(cls):
            raise ColdDigestV22Error("annotation closure contains a non-dataclass root")
        found.add(cls)
        hints = get_type_hints(cls, include_extras=True)
        for field in fields(cls):
            try:
                annotation = hints[field.name]
            except KeyError as error:
                raise ColdDigestV22Error("dataclass field lacks a resolved annotation") from error
            annotations: list[object] = [annotation]
            while annotations:
                candidate = annotations.pop()
                if isinstance(candidate, type) and is_dataclass(candidate):
                    pending.append(candidate)
                else:
                    annotations.extend(get_args(candidate))
    return frozenset(found)


def _type_identity_fingerprint(types: frozenset[type[object]]) -> str:
    """Bind the exact module/qualified-name set, not dataclass schemas."""

    identities = sorted(_type_parts(cls) for cls in types)
    digest = hashlib.sha256(_TYPE_IDENTITY_DOMAIN)
    digest.update(struct.pack("<Q", len(identities)))
    for module, qualname in identities:
        for part in (module, qualname):
            digest.update(struct.pack("<Q", len(part)))
            digest.update(part)
    return digest.hexdigest()


EVIDENCE_DATACLASS_TYPES: Final = _dataclass_type_closure(evidence.ExposedSeedScienceEvidence)
EVIDENCE_TYPE_IDENTITY_FINGERPRINT: Final = _type_identity_fingerprint(EVIDENCE_DATACLASS_TYPES)

if (
    len(EVIDENCE_DATACLASS_TYPES) != EXPECTED_TYPE_IDENTITY_COUNT
    or EVIDENCE_TYPE_IDENTITY_FINGERPRINT != EXPECTED_TYPE_IDENTITY_FINGERPRINT
):
    raise RuntimeError("MM-008 v2.2 evidence type-identity universe drifted")


class DevelopmentEvidenceDigest:
    """Hash a closed immutable value graph without encoding object aliases."""

    def __init__(self, allowed_dataclasses: frozenset[type[object]]) -> None:
        if not allowed_dataclasses or any(
            not isinstance(cls, type) or not is_dataclass(cls) for cls in allowed_dataclasses
        ):
            raise ColdDigestV22Error("digest allowlist must contain exact dataclass types")
        self._allowed = allowed_dataclasses
        self._memo: dict[int, bytes] = {}
        self._active: set[int] = set()

    def root(self, value: object, *, root_type: type[object]) -> str:
        """Return a hexadecimal root only when the exact root type is present."""

        if type(value) is not root_type or root_type not in self._allowed:
            raise ColdDigestV22Error("wrong or unallowlisted evidence root type")
        return self._digest(value).hex()

    def _digest(self, value: object) -> bytes:
        value_type = type(value)

        if value is None:
            return _node(b"none", ())
        if value_type is bool:  # bool must precede int.
            checked_bool = cast(bool, value)
            return _node(b"bool", (b"\x01" if checked_bool else b"\x00",))
        if value_type is int:
            return _node(b"int", (_int_payload(cast(int, value)),))
        if value_type is float:
            checked_float = cast(float, value)
            if not math.isfinite(checked_float):
                raise ColdDigestV22Error("nonfinite Python float")
            return _node(b"float64", (struct.pack("<d", checked_float),))
        if value_type is str:
            checked_str = cast(str, value)
            try:
                encoded = checked_str.encode("ascii")
            except UnicodeEncodeError as error:
                raise ColdDigestV22Error("non-ASCII evidence string") from error
            return _node(b"str", (encoded,))

        if value_type is tuple:
            checked_tuple = cast(tuple[object, ...], value)
            identity = id(value)
            cached = self._memo.get(identity)
            if cached is not None:
                return cached
            if identity in self._active:
                raise ColdDigestV22Error("tuple cycle")
            self._active.add(identity)
            try:
                if checked_tuple and all(type(item) is int for item in checked_tuple):
                    result = _node(
                        b"tuple/int",
                        tuple(_int_payload(cast(int, item)) for item in checked_tuple),
                    )
                else:
                    result = _node(
                        b"tuple",
                        tuple(self._digest(item) for item in checked_tuple),
                    )
            finally:
                self._active.remove(identity)
            self._memo[identity] = result
            return result

        if value_type is np.ndarray:
            checked_array = cast(np.ndarray, value)
            identity = id(value)
            cached = self._memo.get(identity)
            if cached is not None:
                return cached
            if (
                checked_array.dtype.str not in _ARRAY_DTYPES
                or not checked_array.flags.c_contiguous
                or checked_array.flags.writeable
            ):
                raise ColdDigestV22Error("unexpected ndarray dtype, layout, or mutability")
            if checked_array.dtype.str == "<f8" and not bool(np.all(np.isfinite(checked_array))):
                raise ColdDigestV22Error("nonfinite ndarray")
            shape = struct.pack("<I", checked_array.ndim) + b"".join(
                struct.pack("<Q", size) for size in checked_array.shape
            )
            result = _node(
                b"ndarray",
                (
                    *_type_parts(value_type),
                    checked_array.dtype.str.encode("ascii"),
                    shape,
                    checked_array.tobytes(order="C"),
                ),
            )
            self._memo[identity] = result
            return result

        if isinstance(value, np.ndarray):
            raise ColdDigestV22Error("ndarray subclass is outside the closed universe")
        if isinstance(value, np.generic):
            raise ColdDigestV22Error("NumPy scalar is outside the closed universe")

        if is_dataclass(value) and not isinstance(value, type):
            if value_type not in self._allowed:
                raise ColdDigestV22Error("unexpected dataclass type")
            identity = id(value)
            cached = self._memo.get(identity)
            if cached is not None:
                return cached
            if identity in self._active:
                raise ColdDigestV22Error("dataclass cycle")
            self._active.add(identity)
            try:
                parts = list(_type_parts(value_type))
                for field in fields(value):
                    try:
                        field_name = field.name.encode("ascii")
                    except UnicodeEncodeError as error:
                        raise ColdDigestV22Error("non-ASCII dataclass field name") from error
                    parts.append(
                        _node(
                            b"field",
                            (
                                field_name,
                                self._digest(getattr(value, field.name)),
                            ),
                        )
                    )
                result = _node(b"dataclass", tuple(parts))
            finally:
                self._active.remove(identity)
            self._memo[identity] = result
            return result

        if isinstance(value, Mapping):
            raise ColdDigestV22Error("mapping is outside the closed universe")
        raise ColdDigestV22Error(f"unsupported evidence type: {value_type!r}")


def digest_exposed_seed_science(
    value: evidence.ExposedSeedScienceEvidence,
) -> str:
    """Hash one complete development graph under the frozen type universe."""

    return DevelopmentEvidenceDigest(EVIDENCE_DATACLASS_TYPES).root(
        value,
        root_type=evidence.ExposedSeedScienceEvidence,
    )


__all__ = [
    "ColdDigestV22Error",
    "DevelopmentEvidenceDigest",
    "EVIDENCE_DATACLASS_TYPES",
    "EVIDENCE_TYPE_IDENTITY_FINGERPRINT",
    "EXPECTED_TYPE_IDENTITY_COUNT",
    "EXPECTED_TYPE_IDENTITY_FINGERPRINT",
    "FORMAL_AUTHORITY",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "digest_exposed_seed_science",
]
