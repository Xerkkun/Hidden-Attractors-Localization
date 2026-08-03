from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.fractional.caputo_fabrizio import (
    CAPUTO_FABRIZIO_REFERENCES,
    caputo_fabrizio_derivative,
    caputo_fabrizio_derivative_reference,
)
from hidden_attractors.fractional.contracts import get_fractional_method


def test_constant_has_zero_cf_derivative_and_research_evidence() -> None:
    samples = np.full(201, 3.25)
    result = caputo_fabrizio_derivative(samples, 0.01, 0.63)

    assert np.array_equal(result.values, np.zeros_like(samples))
    assert result.implementation_status == "research_required"
    assert result.stability == "experimental"
    assert result.definition == "caputo_fabrizio_exponential_kernel"
    assert result.normalization_description == "M(alpha)=1 (documented library default)"
    assert result.semantics["caputo_equivalence_claimed"] is False
    assert CAPUTO_FABRIZIO_REFERENCES == (
        "https://doi.org/10.12785/pfda/010201",
        "https://doi.org/10.1515/fca-2020-0032",
    )


def test_ramp_matches_exact_cf_formula_with_configurable_normalization() -> None:
    alpha = 0.37
    step = 0.005
    lower_terminal = -1.25
    elapsed = np.arange(401, dtype=float) * step
    samples = 4.0 + elapsed
    normalization = lambda order: 2.0 / (2.0 - order)
    normalization_value = normalization(alpha)

    result = caputo_fabrizio_derivative(
        samples,
        step,
        alpha,
        lower_terminal=lower_terminal,
        normalization=normalization,
        normalization_name="M(alpha)=2/(2-alpha)",
    )
    expected = normalization_value / alpha * (
        1.0 - np.exp(-alpha * elapsed / (1.0 - alpha))
    )

    assert result.values == pytest.approx(expected, rel=3e-14, abs=3e-14)
    assert result.times[0] == lower_terminal
    assert result.times[-1] == pytest.approx(lower_terminal + elapsed[-1])
    assert result.normalization_value == pytest.approx(normalization_value)
    assert result.normalization_description == "M(alpha)=2/(2-alpha)"


def test_recurrence_matches_direct_quadratic_reference_and_python_fallback() -> None:
    rng = np.random.default_rng(20260802)
    samples = np.cumsum(rng.normal(size=(96, 3)), axis=0)

    numba_result = caputo_fabrizio_derivative(samples, 0.0125, 0.81)
    python_result = caputo_fabrizio_derivative(
        samples,
        0.0125,
        0.81,
        backend="python",
    )
    reference = caputo_fabrizio_derivative_reference(samples, 0.0125, 0.81)

    assert numba_result.values == pytest.approx(reference.values, rel=2e-14, abs=2e-14)
    assert python_result.values == pytest.approx(reference.values, rel=2e-14, abs=2e-14)
    assert python_result.backend == "python"
    assert reference.backend == "python_reference"
    assert "validation only" in reference.semantics["complexity"]


def test_multicomponent_linearity_and_shape_are_preserved() -> None:
    step = 0.01
    elapsed = np.arange(121, dtype=float) * step
    samples = np.column_stack((elapsed, 2.0 * elapsed + 7.0, -3.0 * elapsed))
    result = caputo_fabrizio_derivative(samples, step, 0.42)

    assert result.values.shape == samples.shape
    assert result.values[:, 1] == pytest.approx(2.0 * result.values[:, 0])
    assert result.values[:, 2] == pytest.approx(-3.0 * result.values[:, 0])


def test_alpha_one_is_explicit_backward_difference() -> None:
    samples = np.array([2.0, 2.1, 2.4, 3.0])
    result = caputo_fabrizio_derivative(samples, 0.1, 1.0)

    assert result.values == pytest.approx([0.0, 1.0, 3.0, 6.0])
    assert result.method == "backward_difference"
    assert result.kernel_rate is None
    assert result.semantics["alpha_one_extension"].startswith("backward difference")


def test_alpha_zero_extension_and_small_alpha_limit_are_documented() -> None:
    step = 0.02
    elapsed = np.arange(101, dtype=float) * step
    samples = 5.0 + np.sin(elapsed)
    endpoint = caputo_fabrizio_derivative(samples, step, 0.0)
    near_endpoint = caputo_fabrizio_derivative(samples, step, 1.0e-9)

    assert endpoint.values == pytest.approx(samples - samples[0])
    assert endpoint.method == "alpha_zero_exact_extension"
    assert near_endpoint.values == pytest.approx(endpoint.values, rel=3e-9, abs=3e-9)
    assert endpoint.semantics["alpha_zero_extension"] == "M(0)*(x(t)-x(lower_terminal))"


@pytest.mark.parametrize(
    ("samples", "step", "alpha", "kwargs", "message"),
    [
        ([], 0.1, 0.5, {}, "samples"),
        ([1.0, np.nan], 0.1, 0.5, {}, "finite"),
        ([1.0, 2.0], 0.0, 0.5, {}, "step"),
        ([1.0, 2.0], 0.1, -0.1, {}, "alpha"),
        ([1.0, 2.0], 0.1, 1.01, {}, "alpha"),
        ([1.0, 2.0], 0.1, 0.5, {"normalization": 0.0}, "M\\(alpha\\)"),
        ([1.0, 2.0], 0.1, 0.5, {"lower_terminal": np.inf}, "lower_terminal"),
        ([1.0, 2.0], 0.1, 0.5, {"backend": "gpu"}, "backend"),
        ([1.0, 2.0], 0.1, 1.0, {"normalization": 2.0}, "M\\(1\\)=1"),
    ],
)
def test_invalid_inputs_are_rejected(samples, step, alpha, kwargs, message) -> None:
    with pytest.raises((ValueError, RuntimeError), match=message):
        caputo_fabrizio_derivative(samples, step, alpha, **kwargs)


def test_cf_operator_is_public_and_registered_without_enabling_an_fde_solver() -> None:
    from hidden_attractors import fractional

    method = get_fractional_method("cf_direct_recursive")
    assert method.implementation_status == "experimental"
    assert method.derivative_families == ("caputo_fabrizio",)
    assert fractional.caputo_fabrizio_derivative is caputo_fabrizio_derivative
