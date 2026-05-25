"""Algebra checks for the official Wu2023 arctan Chua parameter set."""

from __future__ import annotations

import numpy as np

from hidden_attractors.models import (
    chua_arctan_wu2023_parameters,
    chua_parameters,
    equilibria_arctan,
    jacobian_arctan,
    rhs_arctan,
)
from hidden_attractors.systems import get_system


Q_WU2023 = 0.99
EXPECTED = {
    "E0": np.array([0.0, 0.0, 0.0]),
    "E+": np.array([0.60967911698, 2.6247941849e-4, -0.60941663756]),
    "E-": np.array([-0.60967911698, -2.6247941849e-4, 0.60941663756]),
}
EXPECTED_EIGENVALUES = {
    "E0": np.array([2.335992046121269 + 0j, -1.0004421730606339 + 2.438870245845308j, -1.0004421730606339 - 2.438870245845308j]),
    "E+": np.array([-3.649223882578855 + 0j, 0.20653022914273128 + 2.7072975063472904j, 0.20653022914273128 - 2.7072975063472904j]),
}


def _sorted(values: np.ndarray) -> np.ndarray:
    return np.array(sorted(values, key=lambda value: (round(value.real, 12), round(value.imag, 12))))


def _finite_difference(state: np.ndarray, step: float = 1.0e-7) -> np.ndarray:
    p = chua_arctan_wu2023_parameters()
    result = np.zeros((3, 3))
    for column in range(3):
        shift = np.zeros(3)
        shift[column] = step
        result[:, column] = (rhs_arctan(state + shift, p) - rhs_arctan(state - shift, p)) / (2.0 * step)
    return result


def test_three_wu2023_equilibria_have_negligible_residual() -> None:
    params = chua_arctan_wu2023_parameters()
    equilibria = equilibria_arctan(params)

    assert set(equilibria) == {"E0", "E+", "E-"}
    for name, expected in EXPECTED.items():
        assert np.allclose(equilibria[name], expected, atol=1.0e-10)
        assert np.linalg.norm(rhs_arctan(equilibria[name], params)) < 1.0e-9


def test_registered_arctan_system_returns_all_equilibria() -> None:
    system = get_system("fractional_chua_arctan_wu2023")

    assert set(system.equilibrium_points()) == {"E0", "E+", "E-"}


def test_smooth_jacobian_matches_finite_difference_and_expected_eigenvalues() -> None:
    params = chua_arctan_wu2023_parameters()
    equilibria = equilibria_arctan(params)
    for name, state in equilibria.items():
        analytic = jacobian_arctan(state, params)
        assert np.linalg.norm(analytic - _finite_difference(state)) < 1.0e-7
        expected = EXPECTED_EIGENVALUES["E0" if name == "E0" else "E+"]
        assert np.allclose(_sorted(np.linalg.eigvals(analytic)), _sorted(expected), atol=1.0e-10)


def test_matignon_classification_at_q099_is_unstable_for_all_equilibria() -> None:
    threshold = Q_WU2023 * np.pi / 2.0
    params = chua_arctan_wu2023_parameters()
    for state in equilibria_arctan(params).values():
        margins = np.abs(np.angle(np.linalg.eigvals(jacobian_arctan(state, params)))) - threshold
        assert np.any(margins <= 0.0)


def test_arctan_nonlinearity_ignores_piecewise_slopes() -> None:
    state = np.array([0.4, -0.1, 0.2])
    left = chua_parameters(model="arctan", m0=-99.0, m1=77.0, a1=0.4, a2=-1.5585, rho=1.0)
    right = chua_parameters(model="arctan", m0=5.0, m1=-8.0, a1=0.4, a2=-1.5585, rho=1.0)

    assert np.array_equal(rhs_arctan(state, left), rhs_arctan(state, right))
