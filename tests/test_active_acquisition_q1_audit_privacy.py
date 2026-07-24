from __future__ import annotations

import base64
import hashlib
import math

import pytest

from bench.active_acquisition.q1_audit_privacy import (
    PrivatePrefixScanner,
    Q1PrivacyScanError,
    find_private_prefix_paths,
)

_PRIVATE_A = bytes.fromhex(
    "fbffef00112233445566778899aabbcc"
    "ddeeff102132435465768798a9bacbdc"
)
_PRIVATE_B = bytes.fromhex(
    "102030405060708090a0b0c0d0e0f000"
    "112233445566778899aabbccddeeff00"
)


def test_global_index_detects_cross_row_private_transplant() -> None:
    scanner = PrivatePrefixScanner.from_private_values((_PRIVATE_A, _PRIVATE_B))
    public_rows = [
        {"episode": 0, "semantic_reference": hashlib.sha256(b"public-a").hexdigest()},
        {"episode": 1, "semantic_reference": _PRIVATE_A.hex()},
    ]

    assert scanner.scan(public_rows) == ('$[1]["semantic_reference"]',)


@pytest.mark.parametrize("prefix_length", (16, 17, 32))
def test_hex_truncations_and_embedded_encodings_are_detected(prefix_length: int) -> None:
    scanner = PrivatePrefixScanner.from_private_values((_PRIVATE_A,))
    truncated = _PRIVATE_A[:prefix_length]
    value = {
        "lower": f"before:{truncated.hex()}:after",
        "upper": f"before:{truncated.hex().upper()}:after",
    }

    assert scanner.scan(value) == ('$["lower"]', '$["upper"]')


@pytest.mark.parametrize("prefix_length", (16, 17, 32))
@pytest.mark.parametrize("urlsafe", (False, True))
@pytest.mark.parametrize("padded", (False, True))
def test_base64_truncations_and_embedded_encodings_are_detected(
    prefix_length: int,
    urlsafe: bool,
    padded: bool,
) -> None:
    scanner = PrivatePrefixScanner.from_private_values((_PRIVATE_A,))
    truncated = _PRIVATE_A[:prefix_length]
    encoded = (
        base64.urlsafe_b64encode(truncated)
        if urlsafe
        else base64.b64encode(truncated)
    ).decode("ascii")
    if not padded:
        encoded = encoded.rstrip("=")
    value = f"embedded::{encoded}::inside"

    assert scanner.scan(value) == ("$",)


def test_base64_detection_survives_base64_alphabet_surrounding_text() -> None:
    scanner = PrivatePrefixScanner.from_private_values((_PRIVATE_A,))
    encoded = base64.urlsafe_b64encode(_PRIVATE_A[:16]).decode("ascii").rstrip("=")

    assert scanner.scan(f"prefixABC{encoded}suffixXYZ") == ("$",)


def test_raw_utf8_prefix_and_mapping_key_are_detected_once_per_path() -> None:
    private = b"0123456789abcdef-private-tail"
    scanner = PrivatePrefixScanner.from_private_values((private, private))
    key = f"public-{private[:16].decode('ascii')}-key"
    value = {key: f"{private[:16].decode('ascii')}:{private.hex()}"}

    assert scanner.prefix_count == 1
    assert scanner.scan(value) == (
        f'$<key:"{key}">',
        f'$["{key}"]',
    )


def test_salt_commitment_is_not_treated_as_private_salt() -> None:
    salt = b"0123456789abcdef0123456789abcdef"
    commitment = hashlib.sha256(salt).hexdigest()

    assert find_private_prefix_paths(
        {"salt_commitment_sha256": commitment},
        private_values=(salt,),
    ) == ()


@pytest.mark.parametrize(
    "private_values",
    (
        (b"",),
        (b"fifteen-bytes!!",),
        ("not-bytes",),
        (bytearray(b"0123456789abcdef"),),
    ),
)
def test_private_inputs_shorter_than_16_bytes_or_not_bytes_fail_closed(
    private_values: object,
) -> None:
    with pytest.raises(Q1PrivacyScanError):
        PrivatePrefixScanner.from_private_values(private_values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    (
        {1: "non-string key"},
        {"bytes": b"not JSON"},
        {"tuple": ("not", "JSON")},
        {"nan": math.nan},
        {"positive_infinity": math.inf},
        {"negative_infinity": -math.inf},
        {"set": {"not JSON"}},
        {"object": object()},
        {"invalid_utf8_scalar": "\ud800"},
    ),
)
def test_malformed_json_values_fail_closed(value: object) -> None:
    scanner = PrivatePrefixScanner.from_private_values((_PRIVATE_A,))

    with pytest.raises(Q1PrivacyScanError):
        scanner.scan(value)


def test_mapping_and_list_cycles_fail_closed() -> None:
    scanner = PrivatePrefixScanner.from_private_values((_PRIVATE_A,))
    mapping_cycle: dict[str, object] = {}
    mapping_cycle["self"] = mapping_cycle
    list_cycle: list[object] = []
    list_cycle.append(list_cycle)

    with pytest.raises(Q1PrivacyScanError, match="cycle"):
        scanner.scan(mapping_cycle)
    with pytest.raises(Q1PrivacyScanError, match="cycle"):
        scanner.scan(list_cycle)
