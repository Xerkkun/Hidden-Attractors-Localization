"""Short native checks for the extensive fractional variational backend."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis.lyapunov_fractional import fractional_variational_abm_qr
from hidden_attractors.native import FractionalLyapunovRequest, NativeFractionalVariationalBackend


@pytest.fixture(scope="module")
def backend() -> NativeFractionalVariationalBackend:
    try:
        return NativeFractionalVariationalBackend.build("fractional_variational_lyapunov_test")
    except Exception as exc:
        pytest.skip(f"Native C toolchain unavailable: {exc}")


def _rf_request(**overrides) -> FractionalLyapunovRequest:
    values = {
        "system_id": "rabinovich_fabrikant",
        "x0": np.asarray([0.1, 0.1, 0.1]),
        "parameters": {"a": 0.1, "b": 0.98},
        "q": 0.9,
        "h": 0.01,
        "t_final": 0.2,
        "reorthonormalization_time": 0.05,
    }
    values.update(overrides)
    return FractionalLyapunovRequest(**values)


@pytest.mark.native
def test_native_rf_rhs_and_jacobian(backend: NativeFractionalVariationalBackend) -> None:
    rhs, jacobian = backend.rhs_jacobian("rabinovich_fabrikant", {"a": 0.1, "b": 0.98}, [0.1, 0.1, 0.1])
    np.testing.assert_allclose(rhs, [-0.079, 0.139, -0.198], atol=1e-14)
    np.testing.assert_allclose(
        jacobian,
        [[0.12, -0.89, 0.1], [1.27, 0.1, 0.3], [-0.02, -0.02, -1.98]],
        atol=1e-14,
    )


@pytest.mark.native
def test_native_lorenz_rhs_and_jacobian(backend: NativeFractionalVariationalBackend) -> None:
    rhs, jacobian = backend.rhs_jacobian("lorenz", {"sigma": 10.0, "beta": 8.0 / 3.0, "rho": 200.0}, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rhs, [10.0, 195.0, -6.0], atol=1e-14)
    np.testing.assert_allclose(
        jacobian,
        [[-10.0, 10.0, 0.0], [197.0, -1.0, -1.0], [2.0, 1.0, -8.0 / 3.0]],
        atol=1e-14,
    )


@pytest.mark.native
@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 20, 64])
def test_native_fft_block_matches_direct_full_history_qr(
    backend: NativeFractionalVariationalBackend,
    block_size: int,
) -> None:
    direct = backend.run(_rf_request(convolution_mode="direct"))
    accelerated = backend.run(_rf_request(convolution_mode="fft_block", fft_block_size=block_size))
    assert direct.status == accelerated.status == "ok"
    np.testing.assert_allclose(accelerated.final_state, direct.final_state, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(accelerated.exponents, direct.exponents, atol=1e-12, rtol=1e-12)


@pytest.mark.native
def test_native_dk2018_contract_is_explicit(backend: NativeFractionalVariationalBackend) -> None:
    result = backend.run(
        _rf_request(
            execution_contract="dk2018_block_restart_abm_gs",
            convolution_mode="direct",
        )
    )
    assert result.status == "ok"
    assert result.execution_contract == "dk2018_block_restart_abm_gs"
    assert np.all(np.isfinite(result.exponents))


@pytest.mark.native
def test_native_direct_full_history_qr_matches_python_reference(
    backend: NativeFractionalVariationalBackend,
) -> None:
    a, b = 0.1, 0.98
    rhs = lambda x: np.asarray([
        x[1] * (x[2] - 1.0 + x[0] * x[0]) + a * x[0],
        x[0] * (3.0 * x[2] + 1.0 - x[0] * x[0]) + a * x[1],
        -2.0 * x[2] * (b + x[0] * x[1]),
    ])
    jacobian = lambda x: np.asarray([
        [2.0 * x[0] * x[1] + a, x[0] * x[0] + x[2] - 1.0, x[1]],
        [-3.0 * x[0] * x[0] + 3.0 * x[2] + 1.0, a, 3.0 * x[0]],
        [-2.0 * x[1] * x[2], -2.0 * x[0] * x[2], -2.0 * (x[0] * x[1] + b)],
    ])
    reference = fractional_variational_abm_qr(
        rhs,
        jacobian,
        np.asarray([0.1, 0.1, 0.1]),
        q=0.9,
        h=0.01,
        t_final=0.2,
        reorthonormalization_time=0.05,
    )
    native = backend.run(_rf_request(convolution_mode="direct"))
    assert reference.status == native.status == "ok"
    np.testing.assert_allclose(native.exponents, reference.exponents, atol=1e-12, rtol=1e-12)
