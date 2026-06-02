"""Fast algebraic and harmonic-balance validation for scientific software."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hidden_attractors.models import (
    chua_arctan_wu2023_parameters,
    chua_nonsmooth_parameters,
    rhs_arctan,
    rhs_nonsmooth,
)
from hidden_attractors.seed_generation.core import fractional_iomega_power
from hidden_attractors.seed_generation.lure import (
    LURE_TRANSFER_CONVENTION,
    LURE_TRANSFER_NORMALIZED_CONVENTION,
    find_lure_harmonic_seed,
    find_lure_omega_gain_candidates,
    lure_describing_function,
    lure_transfer_function,
)
from hidden_attractors.systems import get_system
from hidden_attractors.systems.lure import LureSystem
from hidden_attractors.verification import (
    classify_equilibrium_stability,
    compute_jacobian,
    solve_equilibria,
)


ROOT = Path(__file__).resolve().parents[1]


def _finite_difference(system, state: np.ndarray, step: float = 1.0e-7) -> np.ndarray:
    result = np.zeros((system.dimension, system.dimension))
    for column in range(system.dimension):
        shift = np.zeros(system.dimension)
        shift[column] = step
        result[:, column] = (
            system.evaluate(state + shift) - system.evaluate(state - shift)
        ) / (2.0 * step)
    return result


def _controlled_system(matrix: np.ndarray, q: float):
    return SimpleNamespace(
        parameters={"model": "nonsmooth", "alpha": 1.0, "m0": 0.0, "m1": 0.0, "q": q},
        lure=SimpleNamespace(matrix=np.asarray(matrix, dtype=float)),
    )


def _manual_2x2_p_minus_s_transfer(
    matrix: np.ndarray, bvec: np.ndarray, cvec: np.ndarray, s: complex
) -> complex:
    a = complex(matrix[0, 0]) - s
    b = complex(matrix[0, 1])
    c = complex(matrix[1, 0])
    d = complex(matrix[1, 1]) - s
    determinant = a * d - b * c
    adjugate_times_b = np.array([d * bvec[0] - b * bvec[1], -c * bvec[0] + a * bvec[1]])
    return complex(cvec @ adjugate_times_b / determinant)


def _quadrature_df(system: LureSystem, amplitude: float) -> float:
    theta = np.linspace(0.0, np.pi, 100_001)
    values = np.array(
        [system.nonlinearity(amplitude * np.cos(value)) * np.cos(value) for value in theta]
    )
    return float(2.0 * np.trapezoid(values, theta) / (np.pi * amplitude))


def test_solve_equilibria_substitution_and_symmetry_for_both_chua_models() -> None:
    cases = [
        ("chua-nonsmooth", rhs_nonsmooth, chua_nonsmooth_parameters()),
        ("fractional-chua-arctan-wu2023", rhs_arctan, chua_arctan_wu2023_parameters()),
    ]
    for system_id, rhs, parameters in cases:
        equilibria = solve_equilibria(get_system(system_id))
        assert set(equilibria) == {"E0", "E+", "E-"}
        assert np.allclose(equilibria["E-"], -equilibria["E+"], atol=1.0e-12, rtol=0.0)
        for state in equilibria.values():
            assert np.linalg.norm(rhs(state, parameters)) < 1.0e-8


def test_compute_jacobian_matches_regional_and_smooth_reference_formulas() -> None:
    nonsmooth = get_system("chua-nonsmooth")
    params = chua_nonsmooth_parameters()
    for x_value, slope in [(0.2, params.m0), (1.2, params.m1)]:
        state = np.array([x_value, -0.1, 0.3])
        expected = np.array(
            [
                [-params.alpha * (1.0 + slope), params.alpha, 0.0],
                [1.0, -1.0, 1.0],
                [0.0, -params.beta, -params.gamma],
            ]
        )
        assert np.allclose(compute_jacobian(nonsmooth, state), expected, atol=1.0e-12, rtol=0.0)
        assert np.allclose(compute_jacobian(nonsmooth, state), _finite_difference(nonsmooth, state), atol=1.0e-5)

    arctan = get_system("fractional-chua-arctan-wu2023")
    params = chua_arctan_wu2023_parameters()
    state = np.array([0.4, -0.1, 0.3])
    residual_slope = params.a2 * params.rho / (1.0 + (params.rho * state[0]) ** 2)
    expected = arctan.lure.matrix.copy()
    expected[0, 0] -= params.alpha * residual_slope
    assert np.allclose(compute_jacobian(arctan, state), expected, atol=1.0e-12, rtol=0.0)
    assert np.allclose(compute_jacobian(arctan, state), _finite_difference(arctan, state), atol=1.0e-5)


@pytest.mark.parametrize(
    ("matrix", "q", "expected"),
    [
        (np.diag([-1.0, -2.0, -3.0]), 0.8, "stable"),
        (np.diag([1.0, -2.0, -3.0]), 0.8, "unstable"),
        (np.array([[-0.1, -2.0], [2.0, -0.1]]), 0.8, "stable"),
        (np.array([[0.1, -2.0], [2.0, 0.1]]), 0.99, "unstable"),
        (np.diag([-1.0, -2.0, -3.0]), 1.0, "stable"),
        (np.diag([1.0, -2.0, -3.0]), 1.0, "unstable"),
    ],
)
def test_matignon_classification_on_controlled_spectra(
    matrix: np.ndarray, q: float, expected: str
) -> None:
    result = classify_equilibrium_stability(_controlled_system(matrix, q), np.zeros(matrix.shape[0]))
    assert result["stability_class"] == expected
    if q < 1.0:
        alpha_min = float(np.min(np.abs(np.angle(np.linalg.eigvals(matrix)))))
        assert result["instability_measure"] == pytest.approx(q - 2.0 * alpha_min / np.pi, abs=1.0e-12)


@pytest.mark.parametrize("q", [1.0, 0.6])
def test_lure_transfer_function_preserves_p_minus_s_i_convention(q: float) -> None:
    matrix = np.array([[-1.0, 2.0], [-3.0, -4.0]])
    bvec = np.array([1.5, -0.5])
    cvec = np.array([0.25, 2.0])
    system = LureSystem("synthetic-lure", matrix, bvec, cvec, lambda _s: 0.0, lambda _a: 0.0, lambda _a, _m: 0.0)
    omega = 1.7
    spectral_value = 1j * omega if q == 1.0 else fractional_iomega_power(omega, q)
    expected = _manual_2x2_p_minus_s_transfer(matrix, bvec, cvec, spectral_value)

    assert LURE_TRANSFER_CONVENTION == "c^T (P - s I)^(-1) b"
    assert LURE_TRANSFER_NORMALIZED_CONVENTION.endswith("= -W_code(s)")
    assert lure_transfer_function(omega, q, system) == pytest.approx(expected, abs=1.0e-10)
    assert -lure_transfer_function(omega, q, system) == pytest.approx(-expected, abs=1.0e-10)
    if q < 1.0:
        assert abs(spectral_value - 1j * omega) > 1.0e-3


def test_cross_tool_fractional_transfer_artifact_uses_the_same_sign_convention() -> None:
    artifact = ROOT / "validation" / "02_algebraic_validation" / "transfer_function_check.csv"
    if not artifact.exists():
        pytest.skip("Cross-tool transfer artifact is not available.")
    row = next(csv.DictReader(artifact.open(newline="", encoding="utf-8")))
    system = get_system("chua-nonsmooth").lure
    value = lure_transfer_function(float(row["omega"]), 0.9998, system)

    assert value.real == pytest.approx(float(row["w_code_real"]), abs=1.0e-8)
    assert value.imag == pytest.approx(float(row["w_code_imag"]), abs=1.0e-8)
    assert -value.real == pytest.approx(float(row["w_report_real"]), abs=1.0e-8)


@pytest.mark.parametrize(
    ("omega", "q", "expected"),
    [
        (2.0, 1.0, 2.0j),
        (4.0, 0.5, np.sqrt(4.0) * (np.cos(np.pi / 4.0) + 1j * np.sin(np.pi / 4.0))),
        (2.0, 0.9998, (2.0**0.9998) * np.exp(1j * 0.9998 * np.pi / 2.0)),
    ],
)
def test_fractional_iomega_power_principal_branch(omega: float, q: float, expected: complex) -> None:
    assert fractional_iomega_power(omega, q) == pytest.approx(expected, abs=1.0e-12)


@pytest.mark.parametrize(("omega", "q"), [(0.0, 0.8), (-1.0, 0.8), (1.0, 0.0), (1.0, 1.1)])
def test_fractional_iomega_power_rejects_invalid_inputs(omega: float, q: float) -> None:
    with pytest.raises(ValueError):
        fractional_iomega_power(omega, q)


@pytest.mark.parametrize(
    ("system_id", "amplitudes"),
    [("chua-nonsmooth", [0.5, 1.0, 2.0]), ("fractional-chua-arctan-wu2023", [0.5, 1.0, 2.0])],
)
def test_describing_function_matches_first_harmonic_quadrature(
    system_id: str, amplitudes: list[float]
) -> None:
    system = get_system(system_id).lure
    for amplitude in amplitudes:
        assert lure_describing_function(amplitude, system).real == pytest.approx(
            _quadrature_df(system, amplitude), abs=1.0e-6
        )


@pytest.mark.published
def test_kuznetsov2017_integer_nyquist_seed_reproduction() -> None:
    expected = json.loads((ROOT / "validation" / "references" / "kuznetsov2017_expected.json").read_text(encoding="utf-8"))
    system = get_system("chua-nonsmooth").lure
    pairs = find_lure_omega_gain_candidates(1.0, system, nscan=10_000)
    seed = find_lure_harmonic_seed(q=1.0, system=system, nscan=10_000)
    response = lure_transfer_function(seed.omega, 1.0, system)

    assert pairs[0][0] == pytest.approx(expected["omega0"], abs=1.0e-8)
    assert seed.gain == pytest.approx(expected["k"], abs=1.0e-8)
    assert seed.amplitude == pytest.approx(expected["a0"], abs=1.0e-8)
    assert np.allclose(seed.seed, expected["seed_plus"], atol=1.0e-7, rtol=0.0)
    assert abs(response.imag) < 1.0e-8
    assert abs(1.0 + seed.gain * response) < 1.0e-8

