"""Development-seed checks for the pure MM-008 v2 synthetic generators.

This file injects only seed 12345.  It does not read a nonce, derive a
challenge seed, execute an ordinary frozen seed, fit an arm, or run an oracle.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from typing import cast

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import calibration_v2 as calibration
from bench.multimodal_mechanism_diagnostics import method as v1
from bench.multimodal_mechanism_diagnostics import method_v2 as v2
from bench.multimodal_mechanism_diagnostics import synthetic_v2 as synthetic

DEV_SEED = 12_345


def _development_raw_draw(rng: np.random.Generator) -> np.ndarray:
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
    return np.asarray(epsilon + 0.50 * offsets, dtype=np.float64)


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def dev_cases() -> dict[synthetic.Scenario, synthetic.SyntheticCase]:
    return {
        scenario: synthetic.generate_case(scenario, seed=DEV_SEED)
        for scenario in synthetic.SCENARIOS
    }


def test_exact_scenario_catalog_and_transform_truths() -> None:
    assert synthetic.PROTOCOL_SHA256 == (
        "6bd9f35d13a36394ea2a17cdd951a0ea0adf0365909228e73671cc9484c19b5f"
    )
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
    expected = {
        "translation": (
            (4.0, -4.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ),
        "affine": (
            (0.0, 0.0, 2.0, 0.0, 0.0, -2.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ),
        "appearance": (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.25, 0.75, 1.5),
            (0.35, -0.25, 0.15),
        ),
        "combined": (
            (-4.0, 4.0, 0.0, 2.0, -2.0, 0.0),
            (1.2, 0.8, 1.4),
            (0.3, -0.2, 0.1),
        ),
        "stationary": (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ),
        "coupled_boundary": (
            (4.0, 0.0, 4.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ),
    }
    for scenario, values in expected.items():
        truth = synthetic.TRUTHS[cast(synthetic.Scenario, scenario)]
        assert truth is not None
        assert (truth.theta, truth.gains, truth.biases) == values
    assert synthetic.TRUTHS["independent"] is None
    assert synthetic.TRUTHS["constant_target"] is None


def test_synthetic_config_is_complete_canonical_and_independently_hashed() -> None:
    payload = synthetic.SYNTHETIC_CONFIG.as_dict()
    assert set(payload) == {
        "arm_order",
        "boundary_expectations",
        "broadband",
        "candidate_order_sha256",
        "challenge",
        "combined_carry_dominance",
        "derangements",
        "dominance",
        "exhaustive_modes",
        "exhaustive_pairs",
        "exhaustive_rules",
        "metamorphic_controls",
        "method_config_sha256",
        "normalizer",
        "ordinary_seed_map",
        "positive_arms",
        "predicates",
        "protocol_sha256",
        "q_envelope",
        "required_arms",
        "scenario_construction",
        "scenario_order",
        "schema_version",
        "scoring",
        "source_generator",
        "truths",
    }
    assert payload["protocol_sha256"] == (
        "6bd9f35d13a36394ea2a17cdd951a0ea0adf0365909228e73671cc9484c19b5f"
    )
    assert payload["scenario_order"] == list(synthetic.SCENARIOS)
    assert payload["ordinary_seed_map"] == {
        "translation": 820_800,
        "affine": 820_801,
        "appearance": 820_802,
        "combined": 820_803,
        "stationary": 820_804,
        "independent": 820_805,
        "coupled_boundary": 820_806,
        "constant_target": 820_807,
    }
    assert payload["source_generator"] == {
        "bit_generator": "PCG64",
        "dtype": "float64",
        "draw_order": ["epsilon", "c"],
        "epsilon_distribution": "standard_normal",
        "epsilon_shape": [6, 3, 64, 64],
        "c_distribution": "standard_normal",
        "c_shape": [6, 3, 1, 1],
        "c_factor": 0.5,
        "raw_source_expression": "epsilon+0.50*c",
    }
    assert payload["normalizer"] == {
        "fit_array": "raw_source_only",
        "pool": "area_mean_R64_to_R8",
        "pool_block": 8,
        "pooled_size": 8,
        "channelwise": True,
        "statistic_axes": ["row", "r8_y", "r8_x"],
        "mean": "float64_mean",
        "scale": "float64_population_std",
        "scale_floor": 1e-6,
        "apply_resolution": 64,
    }
    assert payload["broadband"] == {
        "evaluated_array": "normalized_base_before_optional_mask",
        "matrix_shape": [18, 4096],
        "required_rank": 18,
        "require_positive_row_rms": True,
        "min_singular_ratio_strict": 0.5,
        "central_slice": [8, 56],
        "min_central_variance_strict": 0.5,
        "lag_axes": ["x", "y"],
        "require_positive_lag_denominators": True,
        "max_abs_lag_correlation_strict": 0.1,
    }
    assert payload["derangements"] == {
        "near": [1, 0, 3, 2, 5, 4],
        "far": [3, 4, 5, 0, 1, 2],
    }
    assert payload["required_arms"] == {
        "translation": list(synthetic.ARMS),
        "affine": list(synthetic.ARMS),
        "appearance": list(synthetic.ARMS),
        "combined": list(synthetic.ARMS),
        "stationary": list(synthetic.ARMS),
        "independent": list(synthetic.ARMS),
        "coupled_boundary": ["affine", "combined"],
        "constant_target": ["appearance", "combined"],
    }
    assert payload["dominance"] == {
        "translation": [["global_translation", "appearance"]],
        "affine": [
            ["affine", "global_translation"],
            ["affine", "quadrant_translation"],
            ["affine", "appearance"],
        ],
        "appearance": [
            ["appearance", "global_translation"],
            ["appearance", "quadrant_translation"],
            ["appearance", "affine"],
        ],
        "combined": [
            ["combined", "global_translation"],
            ["combined", "quadrant_translation"],
            ["combined", "affine"],
            ["combined", "appearance"],
        ],
        "stationary": [],
        "independent": [],
        "coupled_boundary": [],
        "constant_target": [],
    }
    assert payload["q_envelope"] == {
        "tie_order": ["S", "F", "R"],
        "applicability": {
            "global_translation": ["S"],
            "quadrant_translation": ["S"],
            "affine": ["S", "F", "R"],
            "appearance": ["S"],
            "combined": ["S", "F", "R"],
        },
        "target_independent_scenarios": ["independent", "constant_target"],
        "claim_true_scenarios": [
            "translation",
            "affine",
            "appearance",
            "combined",
            "stationary",
            "coupled_boundary",
        ],
        "wrong_mode": "null",
        "selection": "complete_panel_minimum_after_target_blind_assembly",
        "pairing_reduction": "ALL",
        "hit_reduction": "ANY_once_per_wrong_target",
    }
    assert payload["predicates"] == {
        "persistence_factor": 1.25,
        "pairing_factor": 1.10,
        "strong_factor": 2.0,
        "endpoint_tolerance": 1e-10,
        "constant_bias_tolerance": 1e-12,
        "coupled_objective_tolerance": 1e-12,
        "flow_equivalence_tolerance": 1e-12,
        "prediction_agreement_tolerance": 1e-12,
        "formulas": {
            "Pair": (
                "b_T>0 and b_N>0 and b_F>0 and "
                "1.10*o_m*b_N<=min_q(n_m^q)*b_T and "
                "1.10*o_m*b_F<=min_q(s_m^q)*b_T"
            ),
            "Perf": "p>0 and 1.25*o_m<=p",
            "BeatsBias": "b_T>0 and 1.25*o_m<=b_T",
            "Complete": (
                "Perf(m) and Pair(m) and "
                "(m not in {appearance,combined} or BeatsBias(m))"
            ),
            "Strong": (
                "p>0 and 2.0*f_m<=p and 2.0*o_m<=p and Pair(m) and "
                "Complete(m) and all_applicable_endpoint_error<=1e-10"
            ),
            "Dom": (
                "f_b>f_a and 1.25*f_a<=f_b and "
                "o_b>o_a and 1.25*o_a<=o_b"
            ),
            "NoBiasGain": "b_T>0 and 1.25*o_m>b_T",
            "Hit": (
                "p>0 and b_wrong>0 and 1.25*min_q(wrong_m^q)<=p and "
                "1.25*min_q(wrong_m^q)<=b_wrong"
            ),
        },
    }
    assert payload["exhaustive_pairs"] == [
        ["translation", "affine"],
        ["affine", "affine"],
        ["combined", "combined"],
        ["coupled_boundary", "affine"],
    ]
    assert payload["exhaustive_modes"] == ["full", "p0", "p1"]
    assert payload["exhaustive_rules"] == {
        "state_count": 15_625,
        "enumeration": "full_canonical_order",
        "selection_key": ["objective", "canonical_state_key"],
        "selected_equals_first_global_minimum": True,
        "selected_equals_injected_truth": True,
        "objective_tolerance": "max(1e-12,1e-10*abs(final_objective))",
        "selected_vs_truth": "selected<=direct_truth+tolerance",
        "flow_equivalence_atol": 1e-12,
        "second_best_non_equivalent_flow_gap_strict": 1e-12,
        "ranks": ["total_zero_based", "admissible_zero_based"],
        "saved_counts": ["candidate", "admissible"],
    }
    assert payload["challenge"] == {
        "scenario": "independent",
        "count": 8,
        "label_prefix": "MM-008-v2-independent-challenge:",
        "nonce_schema_version": "mm008-v2-challenge-nonce-v1",
        "digest": "sha256",
        "digest_input_order": [
            "bytes.fromhex(protocol_sha256)",
            "bytes.fromhex(auditor_nonce_hex)",
            "ascii(label_prefix+decimal_index)",
        ],
        "digest_slice": [0, 8],
        "integer_endian": "big",
        "integer_signed": False,
    }
    metamorphic = cast(dict[str, object], payload["metamorphic_controls"])
    mutation = cast(dict[str, object], metamorphic["held_target_mutation"])
    assert mutation == {
        "delta": 123.0,
        "operation": "add_to_all_channels_at_held_output_parity_sites",
        "row": 0,
        "scenario_arm_pairs": [
            ["affine", "affine"],
            ["combined", "combined"],
        ],
        "output_parities": [0, 1],
        "preserve": [
            "parameters",
            "gains",
            "biases",
            "retained_macrocell_ids",
            "optimizer_histories_and_probes",
            "bias_only_fit",
            "held_parity_prediction",
        ],
        "comparison": "bit_exact",
    }
    assert metamorphic["transpose_scenario_arm_pairs"] == [
        ["translation", "affine"],
        ["translation", "combined"],
        ["affine", "affine"],
        ["affine", "combined"],
        ["appearance", "combined"],
        ["combined", "combined"],
    ]
    assert payload["boundary_expectations"] == {
        "sampler_accepts": [-8.0, 8.0],
        "sampler_rejects": ["nextafter(-8,-inf)", "nextafter(8,+inf)"],
        "interior_zero_clip_scenarios": [
            "translation",
            "affine",
            "appearance",
            "combined",
        ],
        "interior_zero_site_flow_boundary_scenarios": [
            "translation",
            "affine",
            "appearance",
            "combined",
        ],
        "interior_zero_gradient_boundary_scenarios": [
            "translation",
            "affine",
            "appearance",
            "combined",
        ],
        "coupled_boundary": {
            "clip_occupancy": 0.0,
            "site_flow_boundary": True,
            "gradient_boundary": True,
            "truth_flow_reaches": 8.0,
            "truth_ayy_reaches": 4.0,
        },
    }

    manual_digest = sha256(
        b"MM008-v2.1-synthetic-config\0"
        + synthetic.SYNTHETIC_CONFIG.canonical_json.encode("ascii")
    ).hexdigest()
    assert manual_digest == synthetic.SYNTHETIC_CONFIG_SHA256
    assert manual_digest == "974eb8f44de119ddce27d7663db185ec7bbce797829ea59a14d72748c85b2c52"

    omitted = dict(payload)
    del omitted["truths"]
    with pytest.raises(synthetic.SyntheticV2Error, match="differs"):
        synthetic.validate_synthetic_config(
            synthetic.SyntheticConfig.from_payload(omitted)
        )
    drifted = cast(dict[str, object], json.loads(json.dumps(payload)))
    predicates = cast(dict[str, object], drifted["predicates"])
    predicates["pairing_factor"] = 1.11
    with pytest.raises(synthetic.SyntheticV2Error, match="differs"):
        synthetic.validate_synthetic_config(
            synthetic.SyntheticConfig.from_payload(drifted)
        )


def test_all_development_cases_are_valid_finite_readonly_r64_banks(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    shape = (
        calibration.SYNTHETIC_ROWS,
        calibration.CHANNELS,
        calibration.NATIVE_SIZE,
        calibration.NATIVE_SIZE,
    )
    for scenario, case in dev_cases.items():
        assert case.scenario == scenario
        assert case.seed == DEV_SEED
        assert case.generator_failure_codes() == ()
        assert case.source_broadband.valid
        for array in (
            case.raw_source,
            case.raw_target,
            case.normalized_base,
            case.source,
            case.target,
        ):
            assert array.shape == shape
            assert array.dtype == np.float64
            assert array.flags.c_contiguous
            assert not array.flags.writeable
            assert np.all(np.isfinite(array))
        assert np.array_equal(
            case.normalizer.apply(case.raw_source), case.normalized_base
        )
        assert np.array_equal(case.normalizer.apply(case.raw_target), case.target)

        if scenario == "independent":
            assert case.independent_target_broadband is not None
            assert case.independent_target_broadband.valid
        else:
            assert case.independent_target_broadband is None


def test_positive_targets_use_the_declared_central_transform_then_round_trip(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    positive: tuple[synthetic.Scenario, ...] = (
        "translation",
        "affine",
        "appearance",
        "combined",
        "coupled_boundary",
    )
    central = np.zeros(
        (calibration.NATIVE_SIZE, calibration.NATIVE_SIZE), dtype=bool
    )
    coords = v1.GEOMETRY.coords.astype(int)
    central[coords[:, 0], coords[:, 1]] = True
    for scenario in positive:
        case = dev_cases[scenario]
        assert case.truth is not None
        expected = synthetic._central_target(case.source, case.truth)
        assert np.allclose(case.target, expected, rtol=0.0, atol=2e-15)
        assert np.allclose(
            case.target[:, :, ~central],
            case.source[:, :, ~central],
            rtol=0.0,
            atol=2e-15,
        )


def test_stationary_and_constant_target_controls_are_constructed_exactly(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    stationary = dev_cases["stationary"]
    assert np.array_equal(stationary.raw_target, stationary.raw_source)
    assert np.array_equal(stationary.target, stationary.source)

    constant = dev_cases["constant_target"]
    spatial_ranges = np.ptp(constant.target, axis=(-2, -1))
    assert np.array_equal(spatial_ranges, np.zeros_like(spatial_ranges))
    assert not np.array_equal(constant.target, constant.source)


def test_coupled_boundary_uses_valid_premask_base_and_exact_half_plane(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    case = dev_cases["coupled_boundary"]
    assert np.array_equal(case.source[:, :, :32], np.zeros_like(case.source[:, :, :32]))
    assert np.array_equal(case.source[:, :, 32:], case.normalized_base[:, :, 32:])
    assert case.source_broadband == calibration.broadband_validity_metrics(
        case.normalized_base
    )
    assert case.truth is not None
    flow = v1._affine_flow(case.truth.theta_array()[None])
    assert np.min(flow[:, :, 0]) == 0.0
    assert np.max(flow[:, :, 0]) == 8.0
    assert np.array_equal(flow[:, :, 1], np.zeros_like(flow[:, :, 1]))


def test_independent_generator_replays_and_draws_a_second_complete_bank(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    direct = dev_cases["independent"]
    replay = synthetic.generate_independent_case(seed=DEV_SEED)
    assert synthetic.case_replays_exactly(direct, replay)
    assert not np.array_equal(direct.raw_source, direct.raw_target)
    assert not np.array_equal(direct.source, direct.target)


def test_source_independent_and_constant_draw_order_is_exact(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    independent_rng = np.random.Generator(np.random.PCG64(DEV_SEED))
    expected_source = _development_raw_draw(independent_rng)
    expected_independent_target = _development_raw_draw(independent_rng)
    independent = dev_cases["independent"]
    assert np.array_equal(independent.raw_source, expected_source)
    assert np.array_equal(independent.raw_target, expected_independent_target)

    constant_rng = np.random.Generator(np.random.PCG64(DEV_SEED))
    assert np.array_equal(_development_raw_draw(constant_rng), expected_source)
    expected_constants = constant_rng.normal(
        size=(calibration.SYNTHETIC_ROWS, calibration.CHANNELS)
    )
    constant = dev_cases["constant_target"]
    assert np.allclose(
        constant.target[:, :, 0, 0],
        expected_constants,
        rtol=0.0,
        atol=2e-15,
    )


def test_all_scenarios_replay_exactly_and_share_the_seeded_source_draw(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    first = dev_cases[synthetic.SCENARIOS[0]]
    for scenario, case in dev_cases.items():
        replay = synthetic.generate_case(scenario, seed=DEV_SEED)
        assert synthetic.case_replays_exactly(case, replay)
        assert np.array_equal(case.raw_source, first.raw_source)
        assert np.array_equal(case.normalized_base, first.normalized_base)


def test_frozen_derangements_are_exact_and_have_no_fixed_points() -> None:
    near, far = synthetic.derangements()
    rows = np.arange(calibration.SYNTHETIC_ROWS)
    assert np.array_equal(near, [1, 0, 3, 2, 5, 4])
    assert np.array_equal(far, [3, 4, 5, 0, 1, 2])
    assert np.all(near != rows)
    assert np.all(far != rows)
    assert np.array_equal(np.sort(near), rows)
    assert np.array_equal(np.sort(far), rows)


def test_declared_mutation_dispatch_is_exact_and_does_not_fit(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    for scenario, arm in synthetic.DECLARED_MUTATION_PAIRS:
        dispatch = synthetic.declared_mutation_dispatch(dev_cases[scenario], arm)
        assert tuple(item.output_parity for item in dispatch) == (0, 1)
        for item in dispatch:
            assert item.scenario == scenario
            assert item.arm == arm
            assert item.row == 0
            assert item.delta == 123.0
            assert item.original.scope_sha256 == item.mutated.scope_sha256
            assert np.array_equal(item.original.fit_target, item.mutated.fit_target)
    with pytest.raises(synthetic.SyntheticV2Error, match="declared mutation"):
        synthetic.declared_mutation_dispatch(dev_cases["translation"], "affine")


def test_q_panel_scoring_is_canonical_target_blind_and_fail_closed() -> None:
    expected = np.zeros((3, 4), dtype=np.float64)
    panels = (
        v2.QPanel("R", np.ones_like(expected)),
        v2.QPanel("S", np.full_like(expected, 2.0)),
        v2.QPanel("F", np.ones_like(expected)),
    )
    envelope = synthetic.score_q_panels("affine", panels, expected)
    assert envelope.labels == ("S", "F", "R")
    assert envelope.by_label()["S"].mse == 4.0
    assert envelope.by_label()["F"].mse == 1.0
    assert envelope.by_label()["R"].mse == 1.0
    assert envelope.minimum_label == "F"

    singleton = synthetic.score_q_panels(
        "appearance", (v2.QPanel("S", np.ones_like(expected)),), expected
    )
    assert singleton.labels == ("S",)
    with pytest.raises(ValueError, match="membership"):
        synthetic.score_q_panels("affine", panels[:2], expected)
    with pytest.raises(ValueError, match="membership"):
        synthetic.score_q_panels("appearance", panels, expected)
    with pytest.raises(synthetic.SyntheticV2Error, match="duplicated"):
        synthetic.score_q_panels(
            "affine",
            (*panels, v2.QPanel("F", np.zeros_like(expected))),
            expected,
        )


def test_scorer_routes_true_and_wrong_iterative_contexts_without_fitting(
    dev_cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios: tuple[tuple[synthetic.Scenario, v2.CertificationMode], ...] = (
        ("independent", "null"),
        ("constant_target", "null"),
        ("translation", "claim"),
        ("stationary", "claim"),
        ("coupled_boundary", "claim"),
    )
    start_record = calibration.IterativeStartCertificationRecord(
        "affine", True, True
    )
    direction_record = synthetic.DirectionCertificationRecord(
        start_record, True, True, True
    )

    def make_fitters(
        current_panels: tuple[v2.QPanel, ...],
        current_calls: list[tuple[str, v2.CertificationMode]],
    ) -> tuple[object, object]:
        def fake_full(
            source: np.ndarray,
            target: np.ndarray,
            arm: synthetic.Arm,
            *,
            require_certified: bool,
            certification_mode: v2.CertificationMode,
        ) -> synthetic._FitOutput:
            del source, target, arm, require_certified
            current_calls.append(("full", certification_mode))
            return synthetic._FitOutput(
                prediction=current_panels[0].prediction,
                parameters=np.zeros(6),
                gains=np.ones(3),
                biases=np.zeros(3),
                objectives=(0.0,),
                certified=True,
                q_panels=current_panels,
                start_certifications=(start_record,),
                direction_certifications=(direction_record,),
            )

        def fake_xfit(
            source: np.ndarray,
            target: np.ndarray,
            arm: synthetic.Arm,
            *,
            require_certified: bool,
            certification_mode: v2.CertificationMode,
        ) -> synthetic._FitOutput:
            del source, target, arm, require_certified
            current_calls.append(("xfit", certification_mode))
            return synthetic._FitOutput(
                prediction=current_panels[0].prediction,
                parameters=np.zeros((2, 6)),
                gains=np.ones((2, 3)),
                biases=np.zeros((2, 3)),
                objectives=(0.0, 0.0),
                certified=True,
                q_panels=current_panels,
                start_certifications=(start_record, start_record),
                direction_certifications=(direction_record, direction_record),
            )

        return fake_full, fake_xfit

    for scenario, expected_true_mode in scenarios:
        case = dev_cases[scenario]
        target_values = synthetic._target_values(case, 0)
        panels = (
            v2.QPanel("S", target_values + 2.0),
            v2.QPanel("F", target_values + 1.0),
            v2.QPanel("R", target_values + 1.0),
        )
        calls: list[tuple[str, v2.CertificationMode]] = []
        fake_full, fake_xfit = make_fitters(panels, calls)
        monkeypatch.setattr(synthetic, "_fit_full", fake_full)
        monkeypatch.setattr(synthetic, "_fit_xfit", fake_xfit)
        bias_error = calibration.error_record(target_values + 1.0, target_values)
        bias = synthetic.BiasRowScores(
            true_full=bias_error,
            true_xfit=bias_error,
            near_xfit=bias_error,
            far_xfit=bias_error,
        )
        score = synthetic.score_arm_row(case, 0, "affine", bias=bias)

        assert calls == [
            ("full", expected_true_mode),
            ("xfit", expected_true_mode),
            ("xfit", "null"),
            ("xfit", "null"),
        ]
        assert score.true_full_q.labels == ("S", "F", "R")
        assert score.true_xfit_q.labels == ("S", "F", "R")
        assert score.near_xfit_q.labels == ("S", "F", "R")
        assert score.far_xfit_q.labels == ("S", "F", "R")
        assert score.true_full.mse == 4.0
        assert score.true_xfit.mse == (
            1.0 if expected_true_mode == "null" else 4.0
        )
        assert len(score.certifications.true_full_starts) == 1
        assert len(score.certifications.true_xfit_starts) == 2
        assert len(score.certifications.near_xfit_starts) == 2
        assert len(score.certifications.far_xfit_starts) == 2
        assert not score.certifications.preempts
        with pytest.raises(synthetic.SyntheticV2Error, match="predicate summary"):
            replace(score, complete=not score.complete)
        forged_true_xfit = (
            score.true_xfit_q.by_label()["S"]
            if expected_true_mode == "null"
            else score.true_xfit_q.minimum.error
        )
        with pytest.raises(synthetic.SyntheticV2Error, match="scalar score"):
            replace(score, true_xfit=forged_true_xfit)
        with pytest.raises(synthetic.SyntheticV2Error, match="aggregate fit"):
            replace(score.certifications, near_xfit=False)
        combined_q = calibration.QEnvelope("combined", score.true_full_q.panels)
        with pytest.raises(synthetic.SyntheticV2Error, match="Q-envelope arm"):
            replace(score, true_full_q=combined_q)

    noniterative = synthetic.FitCertificationScores(
        true_mode="claim",
        true_full=True,
        true_xfit=True,
        near_xfit=True,
        far_xfit=True,
        true_full_starts=(),
        true_xfit_starts=(),
        near_xfit_starts=(),
        far_xfit_starts=(),
        true_full_directions=(),
        true_xfit_directions=(),
        near_xfit_directions=(),
        far_xfit_directions=(),
    )
    with pytest.raises(synthetic.SyntheticV2Error, match="noniterative"):
        replace(noniterative, true_full=False)
