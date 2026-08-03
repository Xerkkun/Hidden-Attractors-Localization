"""Directed tests for the integer two-phase lead-lag PLL reference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from hidden_attractors.seed_generation import find_integer_lure_omega_gain_candidates_direct
from hidden_attractors.systems import get_system
from hidden_attractors.systems.pll_lead_lag import (
    pll_lead_lag_parameters,
    pll_lure_transfer,
    pll_original_rhs,
    pll_original_to_shifted,
    pll_shifted_sine_harmonics,
)
from hidden_attractors.workflows.pll_lead_lag import (
    continue_pll_running_cycle,
    find_pll_unstable_separator,
    pll_cylinder_distance,
    pll_direct_route_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pll_lead_lag_integer_lure_reference"
EVIDENCE = ROOT / "validation" / "reference_cases" / "pll_lead_lag_integer_q1"


@pytest.mark.unit
def test_pll_is_registered_with_exact_shifted_scalar_lure() -> None:
    system = get_system("pll-lead-lag-2015")
    assert system.dimension == 2
    assert system.lure is not None
    assert system.metadata["lure_form"] == "T1_exact_scalar_after_locked_equilibrium_shift"
    assert system.metadata["state_space"] == "R_times_S1"
    assert list(system.equilibrium_points()) == ["E_focus", "E_saddle"]

    for state in (
        np.zeros(2),
        np.array([0.003, -0.4]),
        np.array([-0.012, 1.3]),
    ):
        assert np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state)) < 1.0e-12

    point = np.array([0.004, -0.31])
    delta = 1.0e-7
    numerical = np.column_stack(
        [
            (system.evaluate(point + delta * direction) - system.evaluate(point - delta * direction))
            / (2.0 * delta)
            for direction in np.eye(2)
        ]
    )
    assert np.linalg.norm(system.jacobian_matrix(point) - numerical) < 3.0e-7

    original = np.array([0.007, 0.43])
    shifted = pll_original_to_shifted(original, system.parameters)
    assert np.linalg.norm(system.evaluate(shifted) - pll_original_rhs(original, system.parameters)) < 1.0e-12


@pytest.mark.unit
def test_pll_transfer_df_and_cylinder_contract_are_explicit() -> None:
    system = get_system("pll-lead-lag-2015")
    p = pll_lead_lag_parameters(system.parameters)
    spectral_point = complex(-0.2, 13.0)
    matrix_value = complex(
        system.lure.output_vector
        @ np.linalg.solve(
            system.lure.matrix - spectral_point * np.eye(2),
            system.lure.input_vector,
        )
    )
    assert pll_lure_transfer(spectral_point, system.parameters, convention="code") == pytest.approx(
        matrix_value, abs=1.0e-12
    )
    assert pll_lure_transfer(
        spectral_point, system.parameters, convention="standard"
    ) == pytest.approx(-matrix_value, abs=1.0e-12)

    harmonics = pll_shifted_sine_harmonics(1.0, system.parameters)
    assert harmonics["quadrature"] == 0.0
    assert harmonics["dc"] != 0.0
    assert harmonics["gain"] == pytest.approx(system.lure.describing_function(1.0))
    assert pll_cylinder_distance([0.0, 0.0], [0.0, 2.0 * np.pi], system.parameters) < 1.0e-14
    assert system.equilibrium_points()["E_saddle"][1] == pytest.approx(
        np.pi - 2.0 * p["theta_focus"]
    )


@pytest.mark.unit
def test_pll_direct_route_is_analytically_rejected_without_frequency_scan() -> None:
    system = get_system("pll-lead-lag-2015")
    assert find_integer_lure_omega_gain_candidates_direct(
        system.lure, wmax=float("inf"), compatible_only=False
    ) == []
    diagnostic = pll_direct_route_diagnostic(system)
    assert diagnostic["decision"] == "rejected_no_positive_imaginary_part_root"
    assert diagnostic["frequency_scan_used"] is False
    assert diagnostic["published_initial_conditions_used"] is False
    assert diagnostic["positive_imaginary_numerator"]["omega_coefficient"] > 0.0
    assert diagnostic["positive_imaginary_numerator"]["omega_cubed_coefficient"] > 0.0


@pytest.mark.integration
def test_andronov_continuation_generates_stable_cycle_and_separator() -> None:
    system = get_system("pll-lead-lag-2015")
    cycles = continue_pll_running_cycle([0.0, 250.0, 500.0], system.parameters)
    stable = cycles[-1]
    separator = find_pll_unstable_separator(stable, system.parameters, bracket_samples=120)

    assert stable.section_velocity == pytest.approx(140.125084912788, abs=2.0e-9)
    assert stable.section_x == pytest.approx(0.004908904250041, abs=2.0e-12)
    assert stable.period == pytest.approx(0.0826766624974, abs=2.0e-11)
    assert 0.0 < stable.multiplier < 1.0
    assert separator.section_velocity == pytest.approx(135.172047419637, abs=2.0e-9)
    assert separator.section_x == pytest.approx(0.005535958796674, abs=2.0e-12)
    assert separator.multiplier > 1.0
    assert separator.section_x > stable.section_x


@pytest.mark.unit
def test_pll_example_keeps_published_initial_conditions_posthoc() -> None:
    cfg = yaml.safe_load((EXAMPLE / "reproducibility.yaml").read_text(encoding="utf-8"))
    hiddenness = cfg["hiddenness"]
    expected_count = (
        len(hiddenness["required_equilibrium_classes"])
        * len(hiddenness["radii"])
        * int(hiddenness["samples_per_radius"])
    )
    assert expected_count == hiddenness["expected_probe_count"] == 96
    assert cfg["direct_route"]["fallback_route"] is None
    assert cfg["direct_route"]["frequency_scan_used"] is False
    assert cfg["continuation"]["start_loop_gain"] == 0.0
    assert cfg["published_regression"]["role"] == "post_derivation_regression_only_not_seed_input"
    serialized_continuation = json.dumps(cfg["continuation"], sort_keys=True)
    assert "x0" not in serialized_continuation
    assert "theta0" not in serialized_continuation


@pytest.mark.regression
def test_promoted_pll_quick_evidence_contains_exactly_96_probes() -> None:
    summary_path = EVIDENCE / "06_verification_summary.json"
    probes_path = EVIDENCE / "05_hiddenness_probes.csv"
    if not summary_path.exists() or not probes_path.exists():
        pytest.skip("promoted PLL quick evidence has not been generated in this checkout")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with probes_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["final_status"] == "ok"
    assert summary["hiddenness"]["summary"]["n_probes"] == 96
    assert summary["hiddenness"]["summary"]["target_hits"] == 0
    assert len(rows) == 96
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["target_hit"] == "False" for row in rows)
