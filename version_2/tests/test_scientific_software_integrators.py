"""Fast direct validation of the public ABM and EFORK integration APIs."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from hidden_attractors.integrations.abm import caputo_abm_integrate
from hidden_attractors.integrations.efork import _python_efork3_integrate, efork3_coefficients, efork_integrate
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems.lure import LureSystem


NO_EARLY_STOP = {"enabled": False}


def _mittag_leffler_q(z: float, q: float) -> float:
    """Independent reference series for E_q(z)."""

    result = 0.0
    for index in range(500):
        term = z**index / math.gamma(q * index + 1.0)
        result += term
        if abs(term) < 1.0e-16:
            break
    return result


def _linear_lure_system(decay: float = -1.0):
    lure = LureSystem(
        "scalar-linear-reference",
        np.array([[decay]]),
        np.array([0.0]),
        np.array([1.0]),
        lambda _sigma: 0.0,
        lambda _amplitude: 0.0,
        lambda _amplitude, _mu: 0.0,
    )
    return SimpleNamespace(lure=lure)


@pytest.mark.parametrize("q", [0.5, 0.8, 0.9998])
def test_caputo_abm_directly_converges_to_mittag_leffler(q: float) -> None:
    errors = []
    for h in [1.0 / 40.0, 1.0 / 80.0, 1.0 / 160.0]:
        times, states, status = caputo_abm_integrate(
            lambda _time, state: -state,
            np.array([1.0]),
            q,
            h,
            1.0,
            memory_mode="full",
            use_c_backend=False,
            early_stop_config=NO_EARLY_STOP,
        )
        exact = np.array([_mittag_leffler_q(-(time**q), q) for time in times])
        errors.append(float(np.max(np.abs(states[:, 0] - exact))))
        assert status == "ok"
        assert np.all(np.isfinite(states))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_abm_window_memory_is_a_declared_approximation_not_full_caputo() -> None:
    kwargs = {
        "rhs": lambda _time, state: -state,
        "x0": np.array([1.0]),
        "q": 0.8,
        "h": 0.025,
        "t_final": 1.0,
        "use_c_backend": False,
        "early_stop_config": NO_EARLY_STOP,
    }
    _, full_states, full_status = caputo_abm_integrate(memory_mode="full", **kwargs)
    _, window_states, window_status = caputo_abm_integrate(
        memory_mode="window", memory_window_length=4, **kwargs
    )

    assert full_status == window_status == "ok"
    module_doc = __import__("hidden_attractors.integrations.abm", fromlist=["__doc__"]).__doc__ or ""
    assert "finite-memory" in module_doc
    assert "equivalent to full Caputo" in module_doc
    assert not np.allclose(window_states, full_states, atol=1.0e-10, rtol=0.0)


def test_efork_full_memory_converges_toward_full_memory_abm_reference() -> None:
    system = _linear_lure_system()
    differences = []
    for h in [1.0 / 20.0, 1.0 / 40.0, 1.0 / 80.0]:
        _, abm_states, abm_status = caputo_abm_integrate(
            lambda _time, state: -state,
            np.array([1.0]),
            0.5,
            h,
            1.0,
            memory_mode="full",
            use_c_backend=False,
            early_stop_config=NO_EARLY_STOP,
        )
        _, efork_states, efork_status = efork_integrate(
            system,
            np.array([1.0]),
            0.5,
            h,
            1.0,
            memory_mode="full",
            use_c_backend=False,
            early_stop_config=NO_EARLY_STOP,
        )
        differences.append(float(abs(efork_states[-1, 0] - abm_states[-1, 0])))
        assert abm_status == efork_status == "ok"
    assert differences[1] < differences[0]
    assert differences[2] < differences[1]


def test_q1_uses_documented_integer_route_not_fractional_efork3() -> None:
    with pytest.raises(ValueError, match="standard integrator"):
        efork3_coefficients(1.0)

    times, states, status = efork_integrate(
        _linear_lure_system(),
        np.array([1.0]),
        1.0,
        0.025,
        0.1,
        use_c_backend=False,
        early_stop_config=NO_EARLY_STOP,
    )
    assert status == "ok"
    assert len(times) == len(states) == 5
    assert np.all(np.isfinite(states))


@pytest.mark.native
def test_native_efork_backend_matches_python_fallback_when_available() -> None:
    rhs = lambda _time, state: -state
    kwargs = {
        "rhs": rhs,
        "x0": np.array([1.0]),
        "q": 0.5,
        "h": 0.025,
        "t_final": 0.25,
        "method": "efork",
        "memory_mode": "full",
        "early_stop_config": NO_EARLY_STOP,
    }
    try:
        native_times, native_states, native_status, _ = fractional_integrate(
            use_c_backend=True, allow_python_fallback=False, return_history=True, **kwargs
        )
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler unavailable: {exc}")

    python_times, python_states, python_status = _python_efork3_integrate(
        rhs=rhs,
        x0=np.array([1.0]),
        q=0.5,
        h=0.025,
        t_final=0.25,
        memory_mode="full",
        early_stop_config=NO_EARLY_STOP,
    )
    assert native_status == python_status == "ok"
    assert np.allclose(native_times, python_times, atol=0.0, rtol=0.0)
    assert np.allclose(native_states, python_states, atol=1.0e-12, rtol=0.0)
