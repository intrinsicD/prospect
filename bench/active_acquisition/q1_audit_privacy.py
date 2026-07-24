"""Low-level global private-prefix scanning for the WM-002 Q1 auditor.

The scanner intentionally knows nothing about the Q1 seed schedule.  Its caller
supplies every private byte string in one batch, allowing the scanner to catch
a private value transplanted into a different episode's public artifact.

Only the first 16 raw bytes of each private value are indexed.  Public strings
are searched for that raw prefix and for hexadecimal or base64 representations
that encode at least those 16 bytes.  Each public byte position performs a
constant amount of work and set membership checks, so scanning does not become
``private_value_count * public_text_length``.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

PRIVATE_PREFIX_BYTES: Final = 16
_HEX_CHARS: Final = frozenset(b"0123456789abcdefABCDEF")
_BASE64_CHARS: Final = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/-_="
)
_BASE64_MIN_CHARS: Final = 22
_BASE64_SCAN_CHARS: Final = 24


class Q1PrivacyScanError(ValueError):
    """The private corpus or public value cannot be scanned safely."""


@dataclass(frozen=True, slots=True)
class PrivatePrefixScanner:
    """An immutable, deduplicated index of 16-byte private prefixes."""

    _raw_prefixes: frozenset[bytes]
    _hex_prefixes: frozenset[bytes]

    @classmethod
    def from_private_values(cls, private_values: Iterable[bytes]) -> PrivatePrefixScanner:
        """Build one global index from private byte strings.

        Values shorter than :data:`PRIVATE_PREFIX_BYTES` and non-``bytes``
        inputs are rejected rather than silently weakening the scan.
        """

        prefixes: set[bytes] = set()
        for index, value in enumerate(private_values):
            if not isinstance(value, bytes):
                raise Q1PrivacyScanError(f"private value {index} is not immutable bytes")
            if len(value) < PRIVATE_PREFIX_BYTES:
                raise Q1PrivacyScanError(
                    f"private value {index} is shorter than {PRIVATE_PREFIX_BYTES} bytes"
                )
            prefixes.add(value[:PRIVATE_PREFIX_BYTES])
        raw_prefixes = frozenset(prefixes)
        return cls(
            _raw_prefixes=raw_prefixes,
            _hex_prefixes=frozenset(prefix.hex().encode("ascii") for prefix in raw_prefixes),
        )

    @property
    def prefix_count(self) -> int:
        """Return the number of distinct indexed private prefixes."""

        return len(self._raw_prefixes)

    def scan(self, value: object) -> tuple[str, ...]:
        """Return sorted, unique paths to strings containing a private prefix.

        Paths use ``$`` for the root, JSON-quoted brackets for mapping values,
        integer brackets for list elements, and ``<key:...>`` for mapping keys.
        Cycles and values outside the JSON data model fail closed.
        """

        matches: set[str] = set()
        active_containers: set[int] = set()
        try:
            self._visit(
                value,
                path="$",
                matches=matches,
                active_containers=active_containers,
            )
        except Q1PrivacyScanError:
            raise
        except Exception as error:
            raise Q1PrivacyScanError(f"public value traversal failed: {error}") from error
        return tuple(sorted(matches))

    def _visit(
        self,
        value: object,
        *,
        path: str,
        matches: set[str],
        active_containers: set[int],
    ) -> None:
        if value is None or isinstance(value, bool | int):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise Q1PrivacyScanError(f"{path} contains a nonfinite JSON number")
            return
        if isinstance(value, str):
            if self._contains_private_prefix(value, path=path):
                matches.add(path)
            return
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active_containers:
                raise Q1PrivacyScanError(f"{path} contains a mapping cycle")
            active_containers.add(identity)
            try:
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise Q1PrivacyScanError(f"{path} contains a non-string mapping key")
                    rendered_key = _render_key(key)
                    if self._contains_private_prefix(key, path=f"{path}<key:{rendered_key}>"):
                        matches.add(f"{path}<key:{rendered_key}>")
                    self._visit(
                        child,
                        path=f"{path}[{rendered_key}]",
                        matches=matches,
                        active_containers=active_containers,
                    )
            finally:
                active_containers.remove(identity)
            return
        if isinstance(value, list):
            identity = id(value)
            if identity in active_containers:
                raise Q1PrivacyScanError(f"{path} contains a list cycle")
            active_containers.add(identity)
            try:
                for index, child in enumerate(value):
                    self._visit(
                        child,
                        path=f"{path}[{index}]",
                        matches=matches,
                        active_containers=active_containers,
                    )
            finally:
                active_containers.remove(identity)
            return
        raise Q1PrivacyScanError(
            f"{path} contains non-JSON value of type {type(value).__name__}"
        )

    def _contains_private_prefix(self, text: str, *, path: str) -> bool:
        try:
            payload = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise Q1PrivacyScanError(f"{path} contains a string that is not valid UTF-8") from error
        if not self._raw_prefixes or len(payload) < PRIVATE_PREFIX_BYTES:
            return False

        final_offset = len(payload) - PRIVATE_PREFIX_BYTES
        for offset in range(final_offset + 1):
            if payload[offset : offset + PRIVATE_PREFIX_BYTES] in self._raw_prefixes:
                return True
            if self._hex_prefix_matches(payload, offset):
                return True
            if self._base64_prefix_matches(payload, offset):
                return True
        return False

    def _hex_prefix_matches(self, payload: bytes, offset: int) -> bool:
        end = offset + 2 * PRIVATE_PREFIX_BYTES
        if end > len(payload):
            return False
        candidate = payload[offset:end]
        if any(character not in _HEX_CHARS for character in candidate):
            return False
        return candidate.lower() in self._hex_prefixes

    def _base64_prefix_matches(self, payload: bytes, offset: int) -> bool:
        available = min(_BASE64_SCAN_CHARS, len(payload) - offset)
        length = 0
        while length < available and payload[offset + length] in _BASE64_CHARS:
            length += 1
        if length < _BASE64_MIN_CHARS:
            return False
        candidate = payload[offset : offset + length]
        padding = b"=" * (-len(candidate) % 4)
        for altchars in (None, b"-_"):
            try:
                decoded = base64.b64decode(
                    candidate + padding,
                    altchars=altchars,
                    validate=True,
                )
            except (binascii.Error, ValueError):
                continue
            if (
                len(decoded) >= PRIVATE_PREFIX_BYTES
                and decoded[:PRIVATE_PREFIX_BYTES] in self._raw_prefixes
            ):
                return True
        return False


def find_private_prefix_paths(
    value: object,
    *,
    private_values: Iterable[bytes],
) -> tuple[str, ...]:
    """Build a global prefix index and scan one JSON-safe public value."""

    return PrivatePrefixScanner.from_private_values(private_values).scan(value)


def _render_key(key: str) -> str:
    return json.dumps(
        key,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


__all__ = (
    "PRIVATE_PREFIX_BYTES",
    "PrivatePrefixScanner",
    "Q1PrivacyScanError",
    "find_private_prefix_paths",
)
