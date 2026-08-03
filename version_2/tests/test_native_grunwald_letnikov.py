from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.fractional.grunwald_letnikov import (
    grunwald_letnikov_derivative,
    grunwald_letnikov_weights,
)
from hidden_attractors.fractional.native_grunwald_letnikov import (
    NativeGLBackendUnavailable,
    NativeGrunwaldLetnikovBackend,
    native_grunwald_letnikov_convolution,
    native_grunwald_letnikov_derivative,
    native_grunwald_letnikov_weights,
)


@pytest.fixture(scope="module")
def native_backend() -> NativeGrunwaldLetnikovBackend:
    try:
        return NativeGrunwaldLetnikovBackend.build()
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler or shared-library loader unavailable: {exc}")


def test_native_weights_match_numba_recurrence(
    native_backend: NativeGrunwaldLetnikovBackend,
) -> None:
    expected = grunwald_letnikov_weights(0.731, 257)
    result = native_grunwald_letnikov_weights(0.731, 257, fallback=False)

    assert result.backend == "native_c"
    assert result.build.abi_version == 1
    assert result.build.kernel_id == "hafo_gl_direct_v1"
    assert result.build.openmp_requested is True
    assert isinstance(result.build.openmp_active, bool)
    assert len(result.build.source_sha256) == 64
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("definition", ["grunwald_letnikov", "caputo_shifted"])
@pytest.mark.parametrize("history_window", [None, 19])
def test_native_multicomponent_derivative_matches_numba(
    native_backend: NativeGrunwaldLetnikovBackend,
    definition: str,
    history_window: int | None,
) -> None:
    rng = np.random.default_rng(20260802)
    samples = rng.normal(size=(173, 3)).cumsum(axis=0)
    orders = np.asarray([0.41, 0.73, 1.0])
    step = 0.0025
    expected = grunwald_letnikov_derivative(
        samples,
        step,
        orders,
        definition=definition,
        history_window=history_window,
    )
    result = native_grunwald_letnikov_derivative(
        samples,
        step,
        orders,
        definition=definition,
        history_window=history_window,
        fallback=False,
    )

    assert result.backend == "native_c"
    assert result.operation == "derivative"
    assert result.memory_policy == (
        "full_history" if history_window is None else "finite_window"
    )
    np.testing.assert_allclose(result.values, expected.values, rtol=2e-14, atol=2e-12)


def test_native_convolution_is_unscaled_numba_history(
    native_backend: NativeGrunwaldLetnikovBackend,
) -> None:
    times = np.linspace(0.0, 2.0, 101)
    samples = np.column_stack((1.0 + times**2, np.sin(times)))
    orders = [0.5, 0.82]
    expected = grunwald_letnikov_derivative(
        samples,
        1.0,
        orders,
        definition="caputo_shifted",
        history_window=23,
    )
    result = native_grunwald_letnikov_convolution(
        samples,
        orders,
        definition="caputo_shifted",
        history_window=23,
        fallback=False,
    )

    assert result.operation == "convolution"
    assert result.step is None
    assert result.metadata["finite_window_changes_operator"] is True
    np.testing.assert_allclose(result.values, expected.values, rtol=2e-14, atol=2e-14)


def test_native_wrapper_preserves_vector_shape(
    native_backend: NativeGrunwaldLetnikovBackend,
) -> None:
    samples = np.linspace(0.0, 1.0, 51) ** 2
    result = native_grunwald_letnikov_derivative(
        samples,
        0.02,
        0.6,
        definition="caputo_shifted",
        fallback=False,
    )
    assert result.values.shape == samples.shape


def test_missing_compiler_or_loader_falls_back_to_numba(monkeypatch) -> None:
    def unavailable(cls, output_name=None):
        raise OSError("synthetic compiler absence")

    monkeypatch.setattr(
        NativeGrunwaldLetnikovBackend,
        "build",
        classmethod(unavailable),
    )
    samples = np.arange(12, dtype=float).reshape(6, 2)
    expected = grunwald_letnikov_derivative(
        samples,
        0.1,
        [0.5, 0.8],
        definition="caputo_shifted",
    )
    result = native_grunwald_letnikov_derivative(
        samples,
        0.1,
        [0.5, 0.8],
        definition="caputo_shifted",
    )

    assert result.backend == "numba_fallback"
    assert result.build.available is False
    assert "synthetic compiler absence" in (result.build.fallback_reason or "")
    np.testing.assert_array_equal(result.values, expected.values)
    with pytest.raises(NativeGLBackendUnavailable, match="unavailable"):
        native_grunwald_letnikov_derivative(
            samples,
            0.1,
            [0.5, 0.8],
            fallback=False,
        )


def test_invalid_requests_fail_before_entering_native_code() -> None:
    with pytest.raises(ValueError, match="positive"):
        native_grunwald_letnikov_derivative([1.0, 2.0], 0.0, 0.5)
    with pytest.raises(ValueError, match="one of"):
        native_grunwald_letnikov_convolution(
            [1.0, 2.0],
            0.5,
            definition="not_a_fractional_definition",
        )


def test_native_gl_functions_are_exported_from_fractional_namespace() -> None:
    from hidden_attractors import fractional

    assert (
        fractional.native_grunwald_letnikov_derivative
        is native_grunwald_letnikov_derivative
    )
    assert (
        fractional.native_grunwald_letnikov_convolution
        is native_grunwald_letnikov_convolution
    )
