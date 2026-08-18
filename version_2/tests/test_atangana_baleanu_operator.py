from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfcx

from hidden_attractors.fractional.atangana_baleanu import (
    ABC_REFERENCES,
    abc_piecewise_linear_weights,
    atangana_baleanu_caputo_derivative,
    atangana_baleanu_caputo_derivative_reference,
    atangana_baleanu_normalization,
)
from hidden_attractors.fractional.contracts import get_fractional_method
from hidden_attractors.fractional.references import get_fractional_reference


def test_constant_has_zero_abc_derivative_and_research_contract() -> None:
    samples = np.full(129, 4.25)
    result = atangana_baleanu_caputo_derivative(samples, 1.0 / 128.0, 0.42)

    assert np.array_equal(result.values, np.zeros_like(samples))
    assert result.definition == "atangana_baleanu_caputo_mittag_leffler_kernel"
    assert result.implementation_status == "research_required"
    assert result.stability == "experimental"
    assert result.semantics["fde_solver"] is False
    assert result.semantics["caputo_equivalence_claimed"] is False
    assert result.semantics["analysed_order_interval"] == "0 < alpha <= 0.5"
    assert ABC_REFERENCES == (
        "https://doi.org/10.2298/TSCI160111018A",
        "https://doi.org/10.1016/j.chaos.2018.11.009",
        "https://doi.org/10.1515/fca-2020-0032",
    )


def test_half_order_ramp_matches_independent_erfcx_identity() -> None:
    alpha = 0.5
    step = 1.0 / 256.0
    elapsed = step * np.arange(257, dtype=np.float64)
    slope = 2.75
    samples = -3.0 + slope * elapsed
    normalization = atangana_baleanu_normalization(alpha)

    result = atangana_baleanu_caputo_derivative(
        samples,
        step,
        alpha,
        lower_terminal=-2.0,
        normalization=atangana_baleanu_normalization,
        normalization_name="B(alpha)=1-alpha+alpha/Gamma(alpha)",
    )
    root_time = np.sqrt(elapsed)
    kernel_integral = erfcx(root_time) - 1.0 + 2.0 * root_time / np.sqrt(np.pi)
    expected = normalization / (1.0 - alpha) * slope * kernel_integral

    assert result.values == pytest.approx(expected, rel=4e-13, abs=4e-13)
    assert result.normalization_value == pytest.approx(normalization)
    assert result.normalization_description == (
        "B(alpha)=1-alpha+alpha/Gamma(alpha)"
    )
    assert result.times[0] == -2.0
    assert result.times[-1] == pytest.approx(-1.0)


def test_numba_python_and_fft_convolutions_agree() -> None:
    rng = np.random.default_rng(20260803)
    samples = np.cumsum(rng.normal(size=(193, 3)), axis=0)
    kwargs = {"step": 1.0 / 192.0, "alpha": 0.37}

    numba_result = atangana_baleanu_caputo_derivative(
        samples, backend="numba", **kwargs
    )
    python_result = atangana_baleanu_caputo_derivative_reference(samples, **kwargs)
    fft_result = atangana_baleanu_caputo_derivative(
        samples, backend="fft", **kwargs
    )

    assert numba_result.values == pytest.approx(
        python_result.values, rel=4e-13, abs=4e-13
    )
    assert fft_result.values == pytest.approx(
        python_result.values, rel=5e-13, abs=5e-13
    )
    assert numba_result.backend == "numba_direct"
    assert python_result.backend == "python_direct"
    assert fft_result.backend == "numpy_fft_offline"
    assert fft_result.fft_length is not None


def test_weights_are_positive_monotone_and_backends_match() -> None:
    numba_weights = abc_piecewise_linear_weights(0.01, 0.48, 101, backend="numba")
    python_weights = abc_piecewise_linear_weights(0.01, 0.48, 101, backend="python")

    assert numba_weights.values == pytest.approx(
        python_weights.values, rel=5e-14, abs=5e-14
    )
    assert np.all(numba_weights.values > 0.0)
    assert np.all(np.diff(numba_weights.values) <= 0.0)
    assert numba_weights.values[0] <= 1.0
    assert numba_weights.max_terms_used < numba_weights.max_series_terms
    assert numba_weights.max_cancellation_condition < 1.0e10


def test_multicomponent_linearity_and_single_sample_contract() -> None:
    times = np.linspace(0.0, 1.0, 97)
    samples = np.column_stack((times**2, 3.0 * times**2 + 7.0, -times**2))
    result = atangana_baleanu_caputo_derivative(
        samples, times[1] - times[0], 0.31
    )
    singleton = atangana_baleanu_caputo_derivative([8.0], 0.1, 0.31)

    assert result.values.shape == samples.shape
    assert result.values[:, 1] == pytest.approx(3.0 * result.values[:, 0])
    assert result.values[:, 2] == pytest.approx(-result.values[:, 0])
    assert singleton.values == pytest.approx([0.0])
    assert singleton.max_kernel_argument == 0.0


@pytest.mark.parametrize(
    ("samples", "step", "alpha", "kwargs", "error", "message"),
    [
        ([], 0.1, 0.4, {}, ValueError, "samples"),
        ([1.0, np.nan], 0.1, 0.4, {}, ValueError, "finite"),
        ([1.0, 2.0], 0.0, 0.4, {}, ValueError, "step"),
        ([1.0, 2.0], 0.1, 0.0, {}, ValueError, "0 < alpha"),
        ([1.0, 2.0], 0.1, 0.51, {}, ValueError, "0 < alpha"),
        (
            [1.0, 2.0],
            0.1,
            0.4,
            {"normalization": 0.0},
            ValueError,
            "B\\(alpha\\)",
        ),
        (
            [1.0, 2.0],
            0.1,
            0.4,
            {"backend": "gpu"},
            ValueError,
            "backend",
        ),
        (
            np.linspace(0.0, 1.0, 33),
            1.0 / 32.0,
            0.5,
            {"max_series_terms": 8},
            ArithmeticError,
            "Mittag-Leffler",
        ),
    ],
)
def test_invalid_or_unverified_requests_are_rejected(
    samples, step, alpha, kwargs, error, message
) -> None:
    with pytest.raises(error, match=message):
        atangana_baleanu_caputo_derivative(samples, step, alpha, **kwargs)


def test_common_normalization_has_documented_endpoints() -> None:
    assert atangana_baleanu_normalization(0.0) == 1.0
    assert atangana_baleanu_normalization(1.0) == 1.0
    assert 0.0 < atangana_baleanu_normalization(0.4) < 1.0
    with pytest.raises(ValueError, match="alpha"):
        atangana_baleanu_normalization(1.01)


def test_abc_operator_is_public_registered_and_not_an_fde_solver() -> None:
    from hidden_attractors import fractional

    method = get_fractional_method("abc_sampled_convolution")
    reference = get_fractional_reference("yadav_pandey_shukla2019")

    assert method.implementation_status == "implemented"
    assert method.execution_kind == "sampled_operator"
    assert method.derivative_families == ("atangana_baleanu_caputo",)
    assert reference.doi == "10.1016/j.chaos.2018.11.009"
    assert (
        fractional.atangana_baleanu_caputo_derivative
        is atangana_baleanu_caputo_derivative
    )
