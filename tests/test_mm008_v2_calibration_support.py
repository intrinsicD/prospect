"""Pure support tests for MM-008 v2; no frozen nonce, seed, or data is used."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from hashlib import sha256
from typing import cast

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import calibration_v2 as calibration

FAKE_PROTOCOL_SHA256 = "01" * 32
FAKE_NONCE_HEX = "ab" * 32
DEV_SOURCE_SEED = 12_345


def _unit_error(mse: float, *, count: int = 10) -> calibration.ErrorRecord:
    return calibration.ErrorRecord(sse=mse * count, count=count, mse=mse)


def _receipt_dict() -> dict[str, object]:
    return {
        "created_at_utc": "2026-01-02T03:04:05Z",
        "nonce_hex": FAKE_NONCE_HEX,
        "protocol_sha256": FAKE_PROTOCOL_SHA256,
        "reviewer_id": calibration.NONCE_REVIEWER_ID,
        "schema_version": calibration.NONCE_SCHEMA_VERSION,
    }


def test_v21_protocol_binding_accepts_only_the_frozen_protocol_identity() -> None:
    assert calibration.PROTOCOL_SHA256 == (
        "6bd9f35d13a36394ea2a17cdd951a0ea0adf0365909228e73671cc9484c19b5f"
    )
    current = _receipt_dict()
    current["protocol_sha256"] = calibration.PROTOCOL_SHA256
    receipt = calibration.validate_nonce_receipt(
        current, expected_protocol_sha256=calibration.PROTOCOL_SHA256
    )
    assert receipt.protocol_sha256 == calibration.PROTOCOL_SHA256

    retired = dict(current)
    retired["protocol_sha256"] = (
        "14a9f6a0f72c118a2107a938dc162b806a53bfc583cb15d3ac308a71fb264b32"
    )
    with pytest.raises(ValueError, match="differs from the expected protocol"):
        calibration.validate_nonce_receipt(
            retired, expected_protocol_sha256=calibration.PROTOCOL_SHA256
        )


def test_nonce_receipt_canonical_bytes_hash_and_fake_seed_derivation() -> None:
    receipt = calibration.validate_nonce_receipt(
        _receipt_dict(), expected_protocol_sha256=FAKE_PROTOCOL_SHA256
    )
    expected = (
        '{"created_at_utc":"2026-01-02T03:04:05Z",'
        f'"nonce_hex":"{FAKE_NONCE_HEX}",'
        f'"protocol_sha256":"{FAKE_PROTOCOL_SHA256}",'
        f'"reviewer_id":"{calibration.NONCE_REVIEWER_ID}",'
        f'"schema_version":"{calibration.NONCE_SCHEMA_VERSION}"}}\n'
    ).encode("ascii")
    assert calibration.canonical_nonce_receipt_bytes(receipt) == expected
    assert calibration.nonce_receipt_sha256(receipt) == sha256(expected).hexdigest()
    assert (
        calibration.parse_canonical_nonce_receipt_bytes(
            expected, expected_protocol_sha256=FAKE_PROTOCOL_SHA256
        )
        == receipt
    )

    expected_seeds = tuple(
        int.from_bytes(
            sha256(
                bytes.fromhex(FAKE_PROTOCOL_SHA256)
                + bytes.fromhex(FAKE_NONCE_HEX)
                + f"{calibration.CHALLENGE_LABEL_PREFIX}{index}".encode("ascii")
            ).digest()[:8],
            "big",
        )
        for index in range(calibration.CHALLENGE_COUNT)
    )
    assert tuple(
        calibration.derive_challenge_seed(FAKE_PROTOCOL_SHA256, FAKE_NONCE_HEX, index)
        for index in range(calibration.CHALLENGE_COUNT)
    ) == expected_seeds
    assert len(set(expected_seeds)) == calibration.CHALLENGE_COUNT
    with pytest.raises(ValueError, match="differs from the expected protocol"):
        calibration.derive_challenge_seeds(receipt)


def test_nonce_receipt_rejects_membership_format_identity_and_binding_drift() -> None:
    invalid: list[dict[str, object]] = []

    missing = _receipt_dict()
    del missing["reviewer_id"]
    invalid.append(missing)

    extra = _receipt_dict()
    extra["extra"] = "forbidden"
    invalid.append(extra)

    non_string = _receipt_dict()
    non_string["nonce_hex"] = 7
    invalid.append(non_string)

    uppercase_nonce = _receipt_dict()
    uppercase_nonce["nonce_hex"] = FAKE_NONCE_HEX.upper()
    invalid.append(uppercase_nonce)

    wrong_protocol = _receipt_dict()
    wrong_protocol["protocol_sha256"] = "02" * 32
    invalid.append(wrong_protocol)

    wrong_reviewer = _receipt_dict()
    wrong_reviewer["reviewer_id"] = "someone-else"
    invalid.append(wrong_reviewer)

    wrong_schema = _receipt_dict()
    wrong_schema["schema_version"] = "mm008-v2-challenge-nonce-v2"
    invalid.append(wrong_schema)

    fractional_time = _receipt_dict()
    fractional_time["created_at_utc"] = "2026-01-02T03:04:05.000Z"
    invalid.append(fractional_time)

    impossible_time = _receipt_dict()
    impossible_time["created_at_utc"] = "2026-02-30T03:04:05Z"
    invalid.append(impossible_time)

    for value in invalid:
        with pytest.raises(ValueError):
            calibration.validate_nonce_receipt(
                value, expected_protocol_sha256=FAKE_PROTOCOL_SHA256
            )

    with pytest.raises(ValueError, match="expected_protocol_sha256"):
        calibration.validate_nonce_receipt(
            _receipt_dict(), expected_protocol_sha256="not-a-sha"
        )


def test_nonce_receipt_bytes_must_match_the_complete_canonical_file() -> None:
    receipt = calibration.validate_nonce_receipt(
        _receipt_dict(), expected_protocol_sha256=FAKE_PROTOCOL_SHA256
    )
    canonical = calibration.canonical_nonce_receipt_bytes(receipt)
    noncanonical = (
        json.dumps(receipt.as_dict(), sort_keys=False, indent=2).encode("ascii"),
        canonical[:-1],
        canonical + b"\n",
        canonical.replace(b'"nonce_hex"', b'"nonce_hex" '),
    )
    for payload in noncanonical:
        with pytest.raises(ValueError, match="canonical"):
            calibration.parse_canonical_nonce_receipt_bytes(
                payload, expected_protocol_sha256=FAKE_PROTOCOL_SHA256
            )
    with pytest.raises(ValueError, match="immutable bytes"):
        calibration.parse_canonical_nonce_receipt_bytes(
            cast(bytes, bytearray(canonical)), expected_protocol_sha256=FAKE_PROTOCOL_SHA256
        )


def test_challenge_seed_derivation_rejects_noncanonical_identity_or_index() -> None:
    for index in (-1, calibration.CHALLENGE_COUNT, True, 1.5):
        with pytest.raises(ValueError):
            calibration.derive_challenge_seed(
                FAKE_PROTOCOL_SHA256, FAKE_NONCE_HEX, cast(int, index)
            )
    with pytest.raises(ValueError, match="protocol_sha256"):
        calibration.derive_challenge_seed("A" * 64, FAKE_NONCE_HEX, 0)
    with pytest.raises(ValueError, match="nonce_hex"):
        calibration.derive_challenge_seed(FAKE_PROTOCOL_SHA256, "cd" * 31, 0)


def _development_raw_source() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(DEV_SOURCE_SEED))
    epsilon = rng.normal(
        size=(
            calibration.SYNTHETIC_ROWS,
            calibration.CHANNELS,
            calibration.NATIVE_SIZE,
            calibration.NATIVE_SIZE,
        )
    )
    offsets = rng.normal(
        size=(calibration.SYNTHETIC_ROWS, calibration.CHANNELS, 1, 1)
    )
    return np.asarray(epsilon + 0.5 * offsets, dtype=np.float64)


def test_source_only_normalizer_uses_exact_r8_pool_and_round_trips() -> None:
    raw_source = _development_raw_source()
    normalizer = calibration.fit_source_only_normalizer(raw_source)
    blocked = raw_source.reshape(6, 3, 8, 8, 8, 8)
    pooled = np.mean(blocked, axis=(3, 5), dtype=np.float64)
    expected_mean = np.mean(pooled, axis=(0, 2, 3), keepdims=True, dtype=np.float64)
    expected_scale = np.maximum(
        np.std(pooled, axis=(0, 2, 3), keepdims=True, dtype=np.float64),
        calibration.SCALE_FLOOR,
    )
    assert np.array_equal(calibration.area_pool_r8(raw_source), pooled)
    assert np.array_equal(normalizer.mean, expected_mean)
    assert np.array_equal(normalizer.scale, expected_scale)
    assert not normalizer.mean.flags.writeable
    assert not normalizer.scale.flags.writeable

    normalized = normalizer.apply(raw_source)
    reconstructed = normalizer.invert(normalized)
    assert np.allclose(reconstructed, raw_source, rtol=0.0, atol=2e-15)

    target = raw_source + 100.0
    _ = normalizer.apply(target)
    assert np.array_equal(normalizer.mean, expected_mean)
    assert np.array_equal(normalizer.scale, expected_scale)


def test_source_only_normalizer_floor_and_input_validation_fail_closed() -> None:
    zeros = np.zeros((6, 3, 64, 64), dtype=np.float64)
    normalizer = calibration.fit_source_only_normalizer(zeros)
    assert np.array_equal(normalizer.mean, np.zeros((1, 3, 1, 1)))
    assert np.array_equal(
        normalizer.scale, np.full((1, 3, 1, 1), calibration.SCALE_FLOOR)
    )

    with pytest.raises(ValueError, match="exactly 6 rows"):
        calibration.fit_source_only_normalizer(zeros[:5])
    with pytest.raises(ValueError, match="shape"):
        calibration.area_pool_r8(np.zeros((6, 3, 32, 32)))
    contaminated = zeros.copy()
    contaminated[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        calibration.fit_source_only_normalizer(contaminated)
    with pytest.raises(ValueError, match="scale"):
        calibration.SourceOnlyNormalizer(
            mean=np.zeros((1, 3, 1, 1)), scale=np.zeros((1, 3, 1, 1))
        )


def test_broadband_metrics_accept_development_white_source() -> None:
    raw_source = _development_raw_source()
    normalized = calibration.fit_source_only_normalizer(raw_source).apply(raw_source)
    metrics = calibration.broadband_validity_metrics(normalized)
    assert metrics.valid
    assert metrics.failure_reasons() == ()
    assert metrics.matrix_rank == 18
    assert metrics.singular_value_ratio > calibration.MIN_SINGULAR_RATIO
    assert min(metrics.central_variance) > calibration.MIN_CENTRAL_VARIANCE
    assert max(abs(value) for value in metrics.lag_correlation) < calibration.MAX_ABS_LAG_CORRELATION
    assert all(metrics.lag_denominator_positive)


def test_broadband_metrics_record_degenerate_failures_without_nonfinite_evidence() -> None:
    metrics = calibration.broadband_validity_metrics(
        np.zeros((6, 3, 64, 64), dtype=np.float64)
    )
    assert not metrics.valid
    assert metrics.failure_reasons() == (
        "nonpositive_row_rms",
        "matrix_rank",
        "singular_value_ratio",
        "central_variance",
        "lag_denominator",
    )
    assert all(math.isfinite(value) for value in metrics.row_rms)
    assert math.isfinite(metrics.singular_value_ratio)
    assert all(math.isfinite(value) for value in metrics.central_variance)
    assert all(math.isfinite(value) for value in metrics.lag_correlation)

    with pytest.raises(ValueError, match="exactly 6 rows"):
        calibration.broadband_validity_metrics(np.zeros((5, 3, 64, 64)))
    nonfinite = np.zeros((6, 3, 64, 64))
    nonfinite[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        calibration.broadband_validity_metrics(nonfinite)


def test_bias_residual_predicates_use_exact_boundaries_and_bias_gate() -> None:
    persistence = 1.0
    ordered = 0.8
    bias_true = 1.0
    bias_wrong = 1.0
    wrong = calibration.PAIRING_FACTOR * ordered

    assert calibration.performance_support(persistence, ordered)
    assert calibration.beats_bias(ordered, bias_true)
    assert calibration.residual_pairing(ordered, wrong, bias_true, bias_wrong)
    assert calibration.pair_support(
        ordered, wrong, wrong, bias_true, bias_wrong, bias_wrong
    )
    for arm in (
        "global_translation",
        "quadrant_translation",
        "affine",
        "appearance",
        "combined",
    ):
        assert calibration.complete_support(
            arm,
            persistence,
            ordered,
            wrong,
            wrong,
            bias_true,
            bias_wrong,
            bias_wrong,
        )

    weak_bias = 0.5
    large_wrong = 2.0
    assert not calibration.beats_bias(ordered, weak_bias)
    assert calibration.complete_support(
        "affine",
        persistence,
        ordered,
        large_wrong,
        large_wrong,
        weak_bias,
        bias_wrong,
        bias_wrong,
    )
    assert not calibration.complete_support(
        "appearance",
        persistence,
        ordered,
        large_wrong,
        large_wrong,
        weak_bias,
        bias_wrong,
        bias_wrong,
    )


def test_q_envelope_uses_exact_applicable_membership_and_frozen_ties() -> None:
    assert calibration.applicable_q_labels("global_translation") == ("S",)
    assert calibration.applicable_q_labels("quadrant_translation") == ("S",)
    assert calibration.applicable_q_labels("appearance") == ("S",)
    assert calibration.applicable_q_labels("affine") == ("S", "F", "R")
    assert calibration.applicable_q_labels("combined") == ("S", "F", "R")

    singleton = calibration.q_envelope("appearance", {"S": _unit_error(0.4)})
    assert singleton.labels == ("S",)
    assert singleton.minimum_label == "S"
    assert singleton.minimum_mse == 0.4

    all_tied = calibration.q_envelope(
        "affine",
        {
            "R": _unit_error(0.4),
            "F": _unit_error(0.4),
            "S": _unit_error(0.4),
        },
    )
    assert all_tied.labels == ("S", "F", "R")
    assert all_tied.minimum_label == "S"
    forward_tie = calibration.q_envelope(
        "combined",
        {
            "R": _unit_error(0.3),
            "S": _unit_error(0.5),
            "F": _unit_error(0.3),
        },
    )
    assert forward_tie.minimum_label == "F"

    with pytest.raises(ValueError, match="membership"):
        calibration.q_envelope(
            "affine", {"S": _unit_error(0.4), "F": _unit_error(0.4)}
        )
    with pytest.raises(ValueError, match="membership"):
        calibration.q_envelope(
            "appearance", {"S": _unit_error(0.4), "F": _unit_error(0.4)}
        )
    with pytest.raises(ValueError, match="unknown panel"):
        calibration.q_envelope(
            "affine",
            {
                "S": _unit_error(0.4),
                "F": _unit_error(0.4),
                "R": _unit_error(0.4),
                "X": _unit_error(0.4),
            },
        )
    with pytest.raises(ValueError, match="counts must agree"):
        calibration.q_envelope(
            "affine",
            {
                "S": _unit_error(0.4),
                "F": _unit_error(0.4, count=11),
                "R": _unit_error(0.4),
            },
        )


def test_q_residual_pairing_is_conjunction_over_every_applicable_panel() -> None:
    ordered = 0.8
    failing = calibration.q_envelope(
        "affine",
        {
            "S": _unit_error(1.0),
            "F": _unit_error(0.7),
            "R": _unit_error(1.2),
        },
    )
    record = calibration.q_residual_pairing(ordered, failing, 1.0, 1.0)
    assert record.aggregation == "all"
    assert record.by_label() == {"S": True, "F": False, "R": True}
    assert record.envelope.minimum_label == "F"
    assert record.envelope.minimum_mse == 0.7
    assert not record.passed
    assert not calibration.residual_pairing(
        ordered, failing.minimum_mse, 1.0, 1.0
    )

    passing = calibration.q_envelope(
        "affine",
        {
            "S": _unit_error(1.0),
            "F": _unit_error(calibration.PAIRING_FACTOR * ordered),
            "R": _unit_error(1.2),
        },
    )
    near_far = calibration.q_pair_support(
        ordered, passing, passing, 1.0, 1.0, 1.0
    )
    assert near_far.near.passed
    assert near_far.far.passed
    assert near_far.passed
    assert calibration.q_complete_support(
        "affine", 1.0, ordered, passing, passing, 1.0, 1.0, 1.0
    )
    assert calibration.q_c_support(
        "affine", 1.0, ordered, passing, passing, 1.0, 1.0, 1.0
    )


def test_q_hit_uses_or_but_counts_the_wrong_target_only_once() -> None:
    envelope = calibration.q_envelope(
        "combined",
        {
            "S": _unit_error(1.2),
            "F": _unit_error(0.7),
            "R": _unit_error(0.6),
        },
    )
    hit = calibration.q_wrong_target_hit(1.0, envelope, 1.0)
    assert hit.aggregation == "any"
    assert hit.by_label() == {"S": False, "F": True, "R": True}
    assert hit.passed
    assert int(hit.passed) == 1
    assert sum(hit.by_label().values()) == 2

    singleton = calibration.q_envelope("appearance", {"S": _unit_error(0.7)})
    singleton_hit = calibration.q_wrong_target_hit(1.0, singleton, 1.0)
    assert singleton_hit.by_label() == {"S": True}
    assert singleton_hit.passed


def test_q_envelope_decisions_are_invariant_to_forward_reverse_label_swap() -> None:
    left = calibration.q_envelope(
        "combined",
        {
            "S": _unit_error(0.95),
            "F": _unit_error(0.7),
            "R": _unit_error(1.1),
        },
    )
    right = calibration.q_envelope(
        "combined",
        {
            "S": _unit_error(0.95),
            "F": _unit_error(1.1),
            "R": _unit_error(0.7),
        },
    )
    assert left.minimum_mse == right.minimum_mse == 0.7
    assert left.minimum_label == "F"
    assert right.minimum_label == "R"
    assert (
        calibration.q_residual_pairing(0.8, left, 1.0, 1.0).passed
        == calibration.q_residual_pairing(0.8, right, 1.0, 1.0).passed
    )
    assert (
        calibration.q_wrong_target_hit(1.0, left, 1.0).passed
        == calibration.q_wrong_target_hit(1.0, right, 1.0).passed
    )


def test_iterative_start_certificate_preempts_if_either_start_fails() -> None:
    certified = calibration.IterativeStartCertificationRecord("affine", True, True)
    assert not certified.preempts
    assert certified.failure_codes == ()

    forward_failed = calibration.IterativeStartCertificationRecord(
        "combined", False, True
    )
    assert forward_failed.preempts
    assert forward_failed.failure_codes == ("forward_terminal_certificate",)
    both_failed = calibration.IterativeStartCertificationRecord(
        "affine", False, False
    )
    assert both_failed.preempts
    assert both_failed.failure_codes == (
        "forward_terminal_certificate",
        "reverse_terminal_certificate",
    )
    with pytest.raises(ValueError, match="iterative arm"):
        calibration.IterativeStartCertificationRecord(
            cast(calibration.IterativeArm, "appearance"), True, True
        )
    with pytest.raises(ValueError, match="Python booleans"):
        calibration.IterativeStartCertificationRecord(
            "affine", cast(bool, np.bool_(True)), True
        )


def test_real_x_f_c_hit_marginal_and_dominance_primitives() -> None:
    persistence = 1.0
    ordered = 0.8
    wrong = calibration.PAIRING_FACTOR * ordered
    assert calibration.x_support("affine", persistence, ordered, 0.5)
    assert not calibration.x_support("appearance", persistence, ordered, 0.5)
    assert calibration.x_support("combined", persistence, ordered, 1.0)
    assert calibration.f_support("appearance", persistence, ordered, 1.0)
    assert calibration.c_support(
        "combined", persistence, ordered, wrong, wrong, 1.0, 1.0, 1.0
    )
    assert calibration.wrong_target_hit(persistence, ordered, 1.0)
    assert calibration.marginal(persistence, ordered, ordered)
    assert calibration.dominates(0.0, 1.0, 0.0, 0.5)
    assert not calibration.dominates(0.0, 0.0, 0.0, 0.0)

    for invalid in (math.nan, math.inf, -1.0, cast(float, "0.1")):
        assert not calibration.performance_support(1.0, invalid)
        assert not calibration.beats_bias(0.1, invalid)
        assert not calibration.residual_pairing(0.1, 1.0, 1.0, invalid)
        assert not calibration.wrong_target_hit(1.0, 0.1, invalid)
        assert not calibration.marginal(1.0, 0.1, invalid)
    assert not calibration.residual_pairing(1e308, 1e308, 1e308, 1e308)


def test_error_and_endpoint_records_are_finite_recomputable_and_nonempty() -> None:
    actual = np.asarray([1.0, 2.0])
    expected = np.zeros(2)
    score = calibration.error_record(actual, expected)
    assert score == calibration.ErrorRecord(sse=5.0, count=2, mse=2.5)
    endpoint = calibration.endpoint_record(actual, expected)
    assert endpoint == calibration.EndpointRecord(
        sse=5.0, count=2, mse=2.5, max_abs_error=2.0
    )
    assert not endpoint.passes()

    with pytest.raises(ValueError, match="reproduce"):
        calibration.ErrorRecord(sse=5.0, count=2, mse=2.4)
    with pytest.raises(ValueError, match="positive Python integer"):
        calibration.ErrorRecord(sse=0.0, count=cast(int, True), mse=0.0)
    with pytest.raises(ValueError, match="positive Python integer"):
        calibration.ErrorRecord(sse=0.0, count=0, mse=0.0)
    with pytest.raises(ValueError, match="equal shape"):
        calibration.error_record(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="empty"):
        calibration.endpoint_record(np.asarray([]), np.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        calibration.endpoint_record(np.asarray([np.inf]), np.asarray([0.0]))
    with pytest.raises(ValueError, match="non-finite"):
        calibration.error_record(np.asarray([1e308]), np.asarray([-1e308]))


def test_endpoint_and_count_records_reject_missing_nonfinite_or_impossible_values() -> None:
    endpoint = calibration.EndpointErrorRecord(
        full=calibration.ENDPOINT_TOLERANCE,
        parity0=0.0,
        parity1=calibration.ENDPOINT_TOLERANCE,
    )
    assert endpoint.passes()
    assert not replace(endpoint, parity1=np.nextafter(calibration.ENDPOINT_TOLERANCE, np.inf)).passes()
    assert not endpoint.passes(math.nan)

    for invalid in (math.nan, math.inf, -1.0, None):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            calibration.EndpointErrorRecord(
                full=cast(float, invalid), parity0=0.0, parity1=0.0
            )

    valid = calibration.DecisionCountRecord(
        x_support=cast(int, np.int64(2)),
        complete_support=2,
        full_support=2,
        hit_near=2,
        hit_far=2,
        marginal=2,
    )
    assert valid.video_count == 8
    with pytest.raises(ValueError, match="outside"):
        replace(valid, x_support=9)
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(valid, x_support=1, complete_support=2)
    with pytest.raises(ValueError, match="integer"):
        replace(valid, hit_near=True)
    with pytest.raises(ValueError, match="exactly eight"):
        replace(valid, video_count=7)


def test_clean_fail_is_per_family_and_every_preemption_fails_closed() -> None:
    counts = calibration.DecisionCountRecord(
        x_support=2,
        complete_support=2,
        full_support=2,
        hit_near=2,
        hit_far=2,
        marginal=2,
    )
    for arm in ("affine", "appearance", "combined"):
        assert calibration.clean_fail(
            arm, counts, denominators_valid=True, controls_valid=True
        )

    for field in ("x_support", "complete_support", "full_support", "hit_near", "hit_far"):
        changes = {field: 3}
        if field == "complete_support":
            changes["x_support"] = 3
        blocked = replace(counts, **changes)
        assert not calibration.clean_fail(
            "affine", blocked, denominators_valid=True, controls_valid=True
        )

    appearance_marginal = replace(counts, marginal=3)
    assert not calibration.clean_fail(
        "appearance", appearance_marginal, denominators_valid=True, controls_valid=True
    )
    assert calibration.clean_fail(
        "combined", appearance_marginal, denominators_valid=True, controls_valid=True
    )

    assert not calibration.clean_fail(
        "affine", counts, denominators_valid=False, controls_valid=True
    )
    assert not calibration.clean_fail(
        "affine", counts, denominators_valid=True, controls_valid=False
    )
    for flag in (
        "optimizer_preemption",
        "order_preemption",
        "boundary_preemption",
        "range_preemption",
    ):
        assert not calibration.clean_fail(
            "affine",
            counts,
            denominators_valid=True,
            controls_valid=True,
            **{flag: True},
        )
