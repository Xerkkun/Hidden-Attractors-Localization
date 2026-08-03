"""Unit tests for the integer modified Van der Pol--Duffing declaration."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.systems import get_system, list_systems
from hidden_attractors.systems.modified_van_der_pol_duffing import (
    MAVPD_2023_PARAMETERS,
    mavpd_2023_system,
    mavpd_lure_system,
)


@pytest.mark.unit
def test_mavpd_is_registered_with_published_parameters() -> None:
    system = get_system("modified-van-der-pol-duffing")

    assert "modified-van-der-pol-duffing" in list_systems()
    assert system.dimension == 3
    assert system.parameters == MAVPD_2023_PARAMETERS
    assert system.state_names == ("y1", "y2", "y3")
    assert system.lure is not None
    assert system.metadata["lure_form"] == "exact_scalar"
    assert system.reference["doi"] == "10.3390/math11030591"


@pytest.mark.unit
def test_mavpd_rhs_matches_exact_scalar_lure_split() -> None:
    for xi in (3.5, 3.1, 4.2):
        system = mavpd_2023_system({"xi": xi})
        assert system.lure is not None
        assert system.parameters["xi"] == xi
        assert system.lure.matrix[1, 1] == -xi

        for state in (
            np.zeros(3),
            np.array([0.2, -0.4, 0.7]),
            np.array([-1.2, 0.5, -0.8]),
        ):
            assert np.allclose(
                system.evaluate(state),
                system.lure.evaluate(state),
                rtol=0.0,
                atol=1.0e-12,
            )


@pytest.mark.unit
def test_mavpd_analytic_jacobian_matches_centered_difference() -> None:
    system = mavpd_2023_system({"xi": 3.1})
    point = np.array([0.3, -0.2, 0.015])
    step = 1.0e-7
    numerical = np.column_stack(
        [
            (
                system.evaluate(point + step * direction)
                - system.evaluate(point - step * direction)
            )
            / (2.0 * step)
            for direction in np.eye(3)
        ]
    )

    assert np.allclose(system.jacobian_matrix(point), numerical, rtol=1.0e-8, atol=2.0e-7)


@pytest.mark.unit
def test_mavpd_equilibria_are_complete_and_stationary() -> None:
    system = get_system("modified-van-der-pol-duffing")
    equilibria = system.equilibrium_points()
    root = np.sqrt(0.1)

    assert list(equilibria) == ["E0", "E+", "E-"]
    assert np.allclose(equilibria["E+"], [root, 0.0, root])
    assert np.allclose(equilibria["E-"], [-root, 0.0, -root])
    for equilibrium in equilibria.values():
        assert np.allclose(system.evaluate(equilibrium), 0.0, atol=1.0e-13)

    assert list(mavpd_2023_system({"gamma": -0.1}).equilibrium_points()) == ["E0"]


@pytest.mark.unit
def test_cubic_describing_function_and_inverse_are_consistent() -> None:
    lure = mavpd_lure_system()

    assert np.array_equal(lure.input_vector, [-100.0, 0.0, 0.0])
    assert np.array_equal(lure.output_vector, [1.0, 0.0, 0.0])
    assert lure.nonlinearity(-2.0) == -8.0
    for amplitude in (0.1, 0.75, 2.0):
        gain = 0.75 * amplitude**2
        assert lure.describing_function(amplitude) == pytest.approx(gain)
        assert lure.is_gain_compatible(gain)
        assert lure.solve_amplitude(gain) == pytest.approx(amplitude)

    assert not lure.is_gain_compatible(0.0)
    assert not lure.is_gain_compatible(-1.0)
    with pytest.raises(ValueError, match="amplitude"):
        lure.describing_function(0.0)
    with pytest.raises(RuntimeError, match="gain"):
        lure.solve_amplitude(-1.0)
