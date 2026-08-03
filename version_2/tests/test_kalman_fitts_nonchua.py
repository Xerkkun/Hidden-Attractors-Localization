"""Regression tests for the non-Chua Kalman--Fitts integer example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from hidden_attractors.seed_generation import find_integer_lure_omega_gain_candidates_direct
from hidden_attractors.systems import get_system
from hidden_attractors.workflows import ContinuationPlan
from hidden_attractors.workflows.switching_lure import (
    continue_integer_lure_nonlinearity,
    find_sign_switching_cycle_seed,
    sign_nonlinearity,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_kalman_fitts_is_registered_as_exact_scalar_lure() -> None:
    system = get_system("kalman-fitts-2019")
    assert system.dimension == 4
    assert system.lure is not None
    assert system.metadata["lure_form"] == "exact_scalar"
    assert list(system.equilibrium_points()) == ["E0"]

    for state in (
        np.zeros(4),
        np.array([0.2, -0.4, 0.7, -0.1]),
        np.array([-2.0, 0.5, -0.8, 0.3]),
    ):
        assert np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state)) < 1.0e-12

    point = np.array([0.3, -0.2, 0.015, 0.4])
    delta = 1.0e-7
    numerical = np.column_stack(
        [
            (
                system.evaluate(point + delta * direction)
                - system.evaluate(point - delta * direction)
            )
            / (2.0 * delta)
            for direction in np.eye(4)
        ]
    )
    assert np.linalg.norm(system.jacobian_matrix(point) - numerical) < 2.0e-7


@pytest.mark.unit
def test_direct_route_recomputes_but_rejects_incompatible_gain() -> None:
    system = get_system("kalman-fitts-2019")
    unrestricted = find_integer_lure_omega_gain_candidates_direct(
        system.lure, wmax=20.0, compatible_only=False
    )
    compatible = find_integer_lure_omega_gain_candidates_direct(
        system.lure, wmax=20.0, compatible_only=True
    )

    assert len(unrestricted) == 1
    assert unrestricted[0][0] == pytest.approx(1.0054352291420865, abs=2.0e-12)
    assert unrestricted[0][1] == pytest.approx(-0.04316870115738453, abs=2.0e-12)
    assert compatible == []
    assert system.lure.describing_function(1.0e-5) > 0.0


@pytest.mark.integration
def test_switching_seed_is_generated_from_generic_section_point() -> None:
    system = get_system("kalman-fitts-2019")
    seed = find_sign_switching_cycle_seed(system, [-4.0, -4.0, 0.0, -4.0])

    assert seed.return_period == 2
    assert seed.iterations < 200
    assert seed.convergence_error <= 1.0e-8
    assert abs(float(system.lure.output_vector @ seed.seed)) <= 1.0e-12
    assert np.linalg.norm(seed.seed - np.asarray(seed.initial_section_state)) > 1.0


@pytest.mark.integration
def test_sign_to_tanh_continuation_is_a_separate_reusable_route() -> None:
    system = get_system("kalman-fitts-2019")
    seed = find_sign_switching_cycle_seed(
        system,
        [-4.0, -4.0, 0.0, -4.0],
        convergence_tolerance=1.0e-6,
    )
    steps = continue_integer_lure_nonlinearity(
        system,
        seed,
        sign_nonlinearity,
        plan=ContinuationPlan(
            (0.0, 0.5, 1.0),
            {"internal_parameter": "mu", "route": "sign_to_tanh_test"},
        ),
        t_transient=0.05,
        t_keep=0.05,
        h=0.01,
        div_threshold=100.0,
    )

    assert [step.lambda_value for step in steps] == [0.0, 0.5, 1.0]
    assert all(step.status == "ok" for step in steps)
    assert steps[-1].provenance["source_nonlinearity"] == "sign_nonlinearity"


@pytest.mark.unit
def test_reproducible_example_keeps_reference_point_out_of_seed_inputs() -> None:
    config_path = (
        ROOT
        / "examples"
        / "kalman_fitts_integer_lure_reference"
        / "reproducibility.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["direct_route"]["route"] == "direct_integer_transfer"
    assert config["direct_route"]["fallback_route"] is None
    assert "nscan" not in config["direct_route"]
    assert config["switching_seed"]["initial_section_state"] == [-4.0, -4.0, 0.0, -4.0]
    assert config["published_regression"]["role"] == "post_derivation_regression_only_not_seed_input"
    assert config["published_regression"]["third_cycle_point"] != config["switching_seed"]["initial_section_state"]
