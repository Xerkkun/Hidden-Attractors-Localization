from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.fractional.fast_grunwald_letnikov import (
    FAST_GL_REFERENCES,
    FFT_AUTO_THRESHOLD,
    fast_grunwald_letnikov_derivative,
    gl_linear_convolution_fft_length,
)
from hidden_attractors.fractional.grunwald_letnikov import (
    grunwald_letnikov_derivative,
    grunwald_letnikov_weights,
)
from hidden_attractors.fractional.contracts import get_fractional_method


@pytest.mark.parametrize("sample_count", [64, 65])
@pytest.mark.parametrize(
    "definition",
    ["grunwald_letnikov", "riemann_liouville_gl", "caputo_shifted"],
)
def test_fft_matches_public_direct_operator_for_even_and_odd_histories(
    sample_count: int,
    definition: str,
) -> None:
    step = 0.007
    times = np.arange(sample_count, dtype=float) * step
    samples = 0.4 + np.sin(2.3 * times) + 0.2 * times**2

    expected = grunwald_letnikov_derivative(
        samples,
        step,
        0.63,
        definition=definition,
    )
    actual = fast_grunwald_letnikov_derivative(
        samples,
        step,
        0.63,
        definition=definition,
        backend="fft",
    )

    assert np.allclose(actual.values, expected.values, rtol=2e-13, atol=2e-12)
    assert actual.fft_length is not None
    assert actual.fft_length >= 2 * sample_count - 1
    assert actual.execution_mode == "batch_offline"
    assert actual.memory_policy == "full_history"
    assert actual.backend == "scipy.fft.pocketfft"
    assert actual.backend_version
    assert actual.references == FAST_GL_REFERENCES


def test_fft_q_one_reduces_to_first_backward_difference() -> None:
    step = 0.125
    samples = np.array([1.5, 2.0, -1.0, 3.25, 3.0])

    result = fast_grunwald_letnikov_derivative(
        samples,
        step,
        1.0,
        backend="fft",
    )
    expected = np.empty_like(samples)
    expected[0] = samples[0] / step
    expected[1:] = np.diff(samples) / step

    assert np.allclose(result.values, expected, rtol=2e-14, atol=2e-14)


def test_raw_and_caputo_shifted_constants_keep_distinct_semantics() -> None:
    samples = np.full(73, 2.75)
    raw = fast_grunwald_letnikov_derivative(
        samples,
        0.01,
        0.5,
        definition="riemann_liouville_gl",
        backend="fft",
    )
    shifted = fast_grunwald_letnikov_derivative(
        samples,
        0.01,
        0.5,
        definition="caputo_shifted",
        backend="fft",
    )

    direct_raw = grunwald_letnikov_derivative(
        samples,
        0.01,
        0.5,
        definition="riemann_liouville_gl",
    )
    assert np.allclose(raw.values, direct_raw.values, rtol=2e-13, atol=2e-12)
    assert np.allclose(shifted.values, 0.0, rtol=0.0, atol=1e-15)
    assert abs(raw.values[-1]) > 0.0


@pytest.mark.parametrize("orders", [0.42, [0.42, 0.71, 1.0]])
def test_fft_supports_scalar_and_componentwise_multicomponent_orders(orders) -> None:
    rng = np.random.default_rng(20260802)
    samples = rng.normal(size=(67, 3)).cumsum(axis=0)

    expected = grunwald_letnikov_derivative(
        samples,
        0.025,
        orders,
        definition="caputo_shifted",
    )
    actual = fast_grunwald_letnikov_derivative(
        samples,
        0.025,
        orders,
        definition="caputo_shifted",
        backend="fft",
    )

    assert actual.values.shape == samples.shape
    assert np.allclose(actual.values, expected.values, rtol=3e-13, atol=3e-12)
    assert actual.dimension == 3
    assert actual.estimated_workspace_bytes > actual.values.nbytes


def test_auto_selector_has_explicit_reproducible_threshold() -> None:
    direct = fast_grunwald_letnikov_derivative(
        np.arange(15.0),
        0.1,
        0.7,
        backend="auto",
        auto_threshold=16,
    )
    fft = fast_grunwald_letnikov_derivative(
        np.arange(16.0),
        0.1,
        0.7,
        backend="auto",
        auto_threshold=16,
    )

    assert direct.backend == "numba"
    assert direct.fft_length is None
    assert direct.requested_backend == "auto"
    assert fft.backend == "scipy.fft.pocketfft"
    assert fft.fft_length is not None
    assert fft.auto_threshold == 16
    assert FFT_AUTO_THRESHOLD > 1


def test_zero_padding_prevents_circular_wraparound_aliasing() -> None:
    rng = np.random.default_rng(8102)
    sample_count = 35
    samples = rng.normal(size=sample_count)
    order = 0.58
    weights = grunwald_letnikov_weights(order, sample_count)

    # Deliberately wrong: length-N FFT computes circular convolution, folding
    # the tail of the length-(2N-1) linear convolution into early indices.
    circular = np.fft.irfft(
        np.fft.rfft(samples, n=sample_count) * np.fft.rfft(weights, n=sample_count),
        n=sample_count,
    )
    direct = grunwald_letnikov_derivative(samples, 1.0, order).values
    padded = fast_grunwald_letnikov_derivative(
        samples,
        1.0,
        order,
        backend="fft",
    )

    assert np.max(np.abs(circular - direct)) > 1e-3
    assert np.allclose(padded.values, direct, rtol=2e-13, atol=2e-13)
    assert padded.fft_length == gl_linear_convolution_fft_length(sample_count)
    assert padded.fft_length >= 2 * sample_count - 1


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"samples": np.empty((0,)), "step": 0.1, "orders": 0.5}, ValueError, "samples"),
        (
            {"samples": [1.0, np.nan], "step": 0.1, "orders": 0.5},
            ValueError,
            "finite",
        ),
        ({"samples": [1.0], "step": 0.0, "orders": 0.5}, ValueError, "step"),
        (
            {"samples": np.ones((3, 2)), "step": 0.1, "orders": [0.5, 0.6, 0.7]},
            ValueError,
            "received 3",
        ),
        (
            {"samples": [1.0], "step": 0.1, "orders": 0.0},
            ValueError,
            "lie in",
        ),
        (
            {
                "samples": [1.0],
                "step": 0.1,
                "orders": 0.5,
                "definition": "caputo",
            },
            ValueError,
            "definition",
        ),
        (
            {"samples": [1.0], "step": 0.1, "orders": 0.5, "backend": "gpu"},
            ValueError,
            "backend",
        ),
        (
            {
                "samples": [1.0],
                "step": 0.1,
                "orders": 0.5,
                "auto_threshold": 0,
            },
            ValueError,
            "auto_threshold",
        ),
    ],
)
def test_invalid_inputs_are_rejected(kwargs, error, message: str) -> None:
    with pytest.raises(error, match=message):
        fast_grunwald_letnikov_derivative(**kwargs)


@pytest.mark.parametrize("invalid", [0, -1, 2.5, True])
def test_fft_length_rejects_invalid_sample_counts(invalid) -> None:
    with pytest.raises((TypeError, ValueError), match="sample_count"):
        gl_linear_convolution_fft_length(invalid)


def test_fast_gl_is_public_and_registered_as_an_offline_operator() -> None:
    from hidden_attractors import fractional

    method = get_fractional_method("gl_fft_offline")
    assert method.execution_kind == "sampled_operator"
    assert method.memory_policies == ("full_history",)
    assert (
        fractional.fast_grunwald_letnikov_derivative
        is fast_grunwald_letnikov_derivative
    )
