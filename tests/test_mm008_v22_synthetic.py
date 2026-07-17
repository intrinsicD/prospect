"""Focused exposed-seed tests for the pure MM-008 v2.2 synthetic generators."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

EXPOSED_SEED = 12_345


@pytest.fixture(scope="module")
def cases() -> dict[synthetic.Scenario, synthetic.SyntheticCase]:
    return {
        scenario: synthetic.generate_case(scenario, seed=EXPOSED_SEED)
        for scenario in synthetic.SCENARIOS
    }


def _draw_raw(rng: np.random.Generator) -> np.ndarray:
    epsilon = rng.standard_normal(size=(6, 3, 64, 64))
    offsets = rng.standard_normal(size=(6, 3, 1, 1))
    return np.asarray(epsilon + 0.50 * offsets, dtype="<f8")


def _same_bits(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def test_protocol_seed_declarations_and_truths_are_exact_and_immutable() -> None:
    assert synthetic.PROTOCOL_SHA256 == geometry.PROTOCOL_SHA256 == calibration.PROTOCOL_SHA256
    assert synthetic.SCENARIOS == (
        "translation",
        "affine",
        "appearance",
        "combined",
        "stationary",
        "independent",
        "coupled_boundary",
        "constant_target",
    )
    assert dict(synthetic.FROZEN_SEED_MAP) == {
        "translation": 820_800,
        "affine": 820_801,
        "appearance": 820_802,
        "combined": 820_803,
        "stationary": 820_804,
        "independent": 820_805,
        "coupled_boundary": 820_806,
        "constant_target": 820_807,
    }
    with pytest.raises(TypeError):
        synthetic.FROZEN_SEED_MAP["translation"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        synthetic.TRUTHS["translation"] = None  # type: ignore[index]

    expected = {
        "translation": ((4.0, -4.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        "affine": ((0.0, 0.0, 2.0, 0.0, 0.0, -2.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        "appearance": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.25, 0.75, 1.5), (0.35, -0.25, 0.15)),
        "combined": ((-4.0, 4.0, 0.0, 2.0, -2.0, 0.0), (1.2, 0.8, 1.4), (0.3, -0.2, 0.1)),
        "stationary": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        "coupled_boundary": ((4.0, 0.0, 4.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    }
    for scenario, values in expected.items():
        truth = synthetic.TRUTHS[scenario]  # type: ignore[index]
        assert truth is not None
        assert (truth.theta, truth.gains, truth.biases) == values
        for array in (truth.theta_array(), truth.gain_array(), truth.bias_array()):
            assert array.dtype.str == "<f8"
            assert not array.flags.writeable
    assert synthetic.TRUTHS["independent"] is None
    assert synthetic.TRUTHS["constant_target"] is None


def test_all_scenarios_use_exact_source_draw_and_normalized_base_evidence(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    rng = np.random.Generator(np.random.PCG64(EXPOSED_SEED))
    expected_raw_source = _draw_raw(rng)
    first = cases["translation"]
    for case in cases.values():
        assert _same_bits(case.raw_source, expected_raw_source)
        assert _same_bits(case.raw_source, first.raw_source)
        assert _same_bits(case.normalized_base, case.normalizer.apply(case.raw_source))
        assert case.source_broadband == calibration.broadband_validity_metrics(case.normalized_base)
        assert case.source_broadband.valid
        assert case.generator_failure_codes() == ()
        for array in (
            case.raw_source,
            case.raw_target,
            case.normalized_base,
            case.source,
            case.target,
        ):
            assert array.shape == (6, 3, 64, 64)
            assert array.dtype.str == "<f8"
            assert array.flags.c_contiguous
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)

    independent = cases["independent"]
    expected_raw_target = _draw_raw(rng)
    assert _same_bits(independent.raw_target, expected_raw_target)
    assert _same_bits(independent.target, independent.normalizer.apply(expected_raw_target))
    assert independent.independent_target_broadband is not None
    assert independent.independent_target_broadband.valid


def test_exposed_seed_target_byte_known_answers(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    expected = {
        "translation": (
            "52f8872e702823a35388a9e5920b1dd822f9e7704083103e64280657e3d0eeaa",
            "905d5487bfb596343eee0d610eb03553ce5806bae92d610f851b89aac68f92c5",
        ),
        "affine": (
            "db36181f13b3b42004b8fdc495340e8e17c483ee2e69ed8a80c0aa6546029905",
            "53ced9f02ca90768b9b0b7346480b811a7663d3782533e326f18158a4be78e7c",
        ),
        "appearance": (
            "45a53c0b18b93de72798861f6404a46f15b59c2e6276a23652bee05c61ccf17d",
            "960ab3140e53fea9b072792c2e155a765a40f75619581f5d49c260865fb4a1e1",
        ),
        "combined": (
            "8a0e210f80e6edee2de3e56d22eba5887f17963de1535aa684e4eb2302991734",
            "450d8cd9b8ed72ac217582cfc7f61ef9b0842d6763eefaa79ca8f813e8c3adc6",
        ),
        "stationary": (
            "5087a90e6f0d09b1c12974fd4b6ce7992fc4c6ca3a5a1a115e85b09351d1ed3c",
            "cba31ab4d892fdba80fb56b9a6e171a409a6c29ffe94d3ac75e026f92df4dddf",
        ),
        "independent": (
            "968ceeca3c6a0f1df385b4c04d21a83425b1447403593bd5532fe6737992813b",
            "9001abcd1c7834bde697736184a2fbce8fbe0898a9e421472416176101ae4003",
        ),
        "coupled_boundary": (
            "6b17194bbefd853712b303f3ffd9c65b9d1a2a68a9042e6f6bdea335b7eea3ef",
            "65026830177a167a8c43685abb1254fceb84c3f54470cdddae8a1416271d471f",
        ),
        "constant_target": (
            "3bd3d18b6f719a187ca316bc3fc8f514d39106ccd13a829a171e19a0fc549506",
            "9e526c202f1c16c4dc593d4ec132167a2fab758c94180980e274e2646d1de7c0",
        ),
    }
    for scenario, case in cases.items():
        actual = tuple(
            hashlib.sha256(array.tobytes(order="C")).hexdigest()
            for array in (case.raw_target, case.target)
        )
        assert actual == expected[scenario]


def test_constant_target_draw_order_and_stationary_bit_equality(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    rng = np.random.Generator(np.random.PCG64(EXPOSED_SEED))
    _draw_raw(rng)
    constants = rng.standard_normal(size=(6, 3))
    constant = cases["constant_target"]
    normalized = np.broadcast_to(constants[:, :, None, None], (6, 3, 64, 64)).copy()
    expected_raw = constant.normalizer.invert(normalized)
    expected_target = constant.normalizer.apply(expected_raw)
    assert _same_bits(constant.raw_target, expected_raw)
    assert _same_bits(constant.target, expected_target)
    assert np.array_equal(np.ptp(constant.target, axis=(-2, -1)), np.zeros((6, 3)))

    stationary = cases["stationary"]
    assert _same_bits(stationary.raw_source, stationary.raw_target)
    assert _same_bits(stationary.source, stationary.target)
    assert stationary.independent_target_broadband is None

    source = stationary.source.copy()
    target = stationary.target.copy()
    source[0, 0, 0, 0] = 0.0
    target[0, 0, 0, 0] = -0.0
    assert np.array_equal(source, target)
    signed_zero_case = replace(stationary, source=source, target=target)
    assert "stationary_normalized_copy" in signed_zero_case.generator_failure_codes()


def test_positive_targets_use_geometry_scalar_sampling_and_source_round_trip(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    for scenario in (
        "translation",
        "affine",
        "appearance",
        "combined",
        "coupled_boundary",
    ):
        case = cases[scenario]
        assert case.truth is not None
        target_normalized = synthetic.transform_central(case.source, case.truth)
        expected_raw = case.normalizer.invert(target_normalized)
        expected_target = case.normalizer.apply(expected_raw)
        assert _same_bits(case.raw_target, expected_raw)
        assert _same_bits(case.target, expected_target)
        assert not target_normalized.flags.writeable

        sampled = geometry.sample_scalar(case.source[0], case.truth.theta, geometry.FULL_MASK)
        transformed = case.truth.gain_array()[:, None] * sampled + case.truth.bias_array()[:, None]
        coords = geometry.GEOMETRY.coords.astype(np.intp)
        actual = target_normalized[0][:, coords[:, 0], coords[:, 1]]
        assert _same_bits(actual, transformed)


def test_coupled_boundary_masks_after_normalization_and_reaches_joint_wall(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    case = cases["coupled_boundary"]
    assert np.array_equal(case.source[:, :, :32], np.zeros_like(case.source[:, :, :32]))
    assert _same_bits(case.source[:, :, 32:], case.normalized_base[:, :, 32:])
    assert case.truth is not None
    theta = case.truth.theta
    u = geometry.GEOMETRY.normalized_coords[:, 0]
    v = geometry.GEOMETRY.normalized_coords[:, 1]
    dy = (theta[0] + theta[2] * u) + theta[3] * v
    dx = (theta[1] + theta[4] * u) + theta[5] * v
    assert float(np.min(dy)) == 0.0
    assert float(np.max(dy)) == 8.0
    assert np.array_equal(dx, np.zeros_like(dx))
    assert geometry.is_admissible(theta)


def test_near_far_derangements_and_row_bundle_are_exact_and_immutable(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    near, far = synthetic.derangements()
    assert np.array_equal(near, [1, 0, 3, 2, 5, 4])
    assert np.array_equal(far, [3, 4, 5, 0, 1, 2])
    assert near.dtype.str == far.dtype.str == "<i8"
    assert not near.flags.writeable and not far.flags.writeable
    with pytest.raises(ValueError):
        near.setflags(write=True)

    bundle = synthetic.row_targets(cases["combined"], 2)
    assert (bundle.row, bundle.near_row, bundle.far_row) == (2, 3, 5)
    assert _same_bits(bundle.source, cases["combined"].source[2])
    assert _same_bits(bundle.true_target, cases["combined"].target[2])
    assert _same_bits(bundle.near_target, cases["combined"].target[3])
    assert _same_bits(bundle.far_target, cases["combined"].target[5])
    assert all(
        not value.flags.writeable
        for value in (bundle.source, bundle.true_target, bundle.near_target, bundle.far_target)
    )


def test_deterministic_replay_and_transpose_helpers(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    for scenario, case in cases.items():
        replay = synthetic.generate_case(scenario, seed=EXPOSED_SEED)
        assert synthetic.case_replays_exactly(case, replay)
        synthetic.validate_case(case)
    assert not synthetic.case_replays_exactly(cases["combined"], cases["affine"])

    theta = (-4.0, 4.0, 0.0, 2.0, -2.0, 0.0)
    assert synthetic.transpose_theta(theta) == (4.0, -4.0, 0.0, -2.0, 2.0, 0.0)
    truth = cases["combined"].truth
    assert truth is not None
    transposed_truth = synthetic.transpose_truth(truth)
    assert transposed_truth.theta == synthetic.transpose_theta(truth.theta)
    assert transposed_truth.gains == truth.gains
    assert transposed_truth.biases == truth.biases

    transposed = synthetic.transpose_case(cases["combined"])
    assert _same_bits(transposed.source, np.swapaxes(cases["combined"].source, -2, -1))
    assert _same_bits(transposed.target, np.swapaxes(cases["combined"].target, -2, -1))
    assert transposed.generator_failure_codes() == ()
    assert synthetic.case_replays_exactly(cases["combined"], synthetic.transpose_case(transposed))

    forged = replace(
        cases["translation"],
        raw_target=cases["translation"].raw_source,
        target=cases["translation"].source,
    )
    assert forged.generator_failure_codes() == ()
    with pytest.raises(synthetic.SyntheticV22Error, match="generator replay"):
        synthetic.validate_case(forged)


def test_generator_and_helper_inputs_fail_closed() -> None:
    for bad_seed in (True, -1, 2**64, 1.5):
        with pytest.raises(synthetic.SyntheticV22Error):
            synthetic.generate_case("translation", seed=bad_seed)  # type: ignore[arg-type]
    with pytest.raises(synthetic.SyntheticV22Error):
        synthetic.generate_case("unknown", seed=EXPOSED_SEED)  # type: ignore[arg-type]
    with pytest.raises(synthetic.SyntheticV22Error):
        synthetic.derangements(5)
    with pytest.raises(synthetic.SyntheticV22Error):
        synthetic.row_targets(synthetic.generate_case("stationary", seed=EXPOSED_SEED), True)
    with pytest.raises(synthetic.SyntheticV22Error):
        synthetic.transpose_theta((0.0,) * 5)
    with pytest.raises(synthetic.SyntheticV22Error):
        synthetic.transpose_theta((0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    with pytest.raises(synthetic.SyntheticV22Error):
        synthetic.transpose_r64(np.zeros((3, 64, 64), dtype=np.float32))
