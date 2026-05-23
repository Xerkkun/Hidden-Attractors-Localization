"""Algebra regression checks for Danca's fractional non-smooth Chua case."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.models import (
    chua_nonsmooth_parameters,
    equilibria_nonsmooth,
    jacobian_nonsmooth,
    rhs_nonsmooth,
)
from hidden_attractors.seed_generation.chua import (
    find_harmonic_seed,
    find_omega_gain_candidates,
    transfer_function,
)


Q_DANCA = 0.9998
MATIGNON_THRESHOLD = Q_DANCA * np.pi / 2.0


def test_equilibria_and_regional_jacobians_match_matlab_validation() -> None:
    params = chua_nonsmooth_parameters()
    equilibria = equilibria_nonsmooth(params)
    expected_outer = np.array([6.5883078865388685, 0.0028364022560936064, -6.585471484282775])

    assert np.allclose(equilibria["E+"], expected_outer, atol=1.0e-12)
    assert np.allclose(equilibria["E-"], -expected_outer, atol=1.0e-12)
    for state in equilibria.values():
        assert np.linalg.norm(rhs_nonsmooth(state, params)) < 2.0e-14

    expected_inner = np.array(
        [[-6.96114384, 8.4562, 0.0], [1.0, -1.0, 1.0], [0.0, -12.0732, -0.0052]]
    )
    expected_outer_j = np.array(
        [[1.24137016, 8.4562, 0.0], [1.0, -1.0, 1.0], [0.0, -12.0732, -0.0052]]
    )
    assert np.allclose(jacobian_nonsmooth(equilibria["E0"], params), expected_inner, atol=1.0e-12)
    assert np.allclose(jacobian_nonsmooth(equilibria["E+"], params), expected_outer_j, atol=1.0e-12)
    with pytest.raises(ValueError, match="undefined"):
        jacobian_nonsmooth(np.array([1.0, 0.0, 0.0]), params)


def test_matignon_classification_matches_danca_and_matlab() -> None:
    params = chua_nonsmooth_parameters()
    equilibria = equilibria_nonsmooth(params)
    eig_inner = np.linalg.eigvals(jacobian_nonsmooth(equilibria["E0"], params))
    eig_outer = np.linalg.eigvals(jacobian_nonsmooth(equilibria["E+"], params))

    inner_margins = np.abs(np.angle(eig_inner)) - MATIGNON_THRESHOLD
    outer_margins = np.abs(np.angle(eig_outer)) - MATIGNON_THRESHOLD
    assert np.min(inner_margins) > 0.0
    assert np.min(outer_margins) < 0.0


def test_lure_branches_match_matlab_after_transfer_sign_normalization() -> None:
    params = chua_nonsmooth_parameters()
    expected = (
        (2.0402860510794905, 0.2100227929621122, 5.8517677854863281),
        (3.2449267309745160, 0.9569454049276507, 1.0530166102567644),
    )
    pairs = find_omega_gain_candidates(Q_DANCA, params, nscan=20_000)

    assert len(pairs) == 2
    for index, (expected_omega, expected_gain, expected_amplitude) in enumerate(expected):
        omega, gain = pairs[index]
        seed = find_harmonic_seed(q=Q_DANCA, params=params, branch_index=index, nscan=20_000)
        w_code = transfer_function(omega, Q_DANCA, params)
        w_report = -w_code
        assert omega == pytest.approx(expected_omega, abs=1.0e-11)
        assert gain == pytest.approx(expected_gain, abs=1.0e-11)
        assert seed.amplitude == pytest.approx(expected_amplitude, abs=2.0e-11)
        assert abs(1.0 + gain * w_code) < 1.0e-10
        assert abs(1.0 - gain * w_report) < 1.0e-10
