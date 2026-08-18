from __future__ import annotations

import sys
import pytest
import numpy as np
from pathlib import Path
from types import SimpleNamespace

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.integrations.selector import validate_integrator_compatibility, integrate


def test_validate_integrator_compatibility():
    # Valid combos
    assert validate_integrator_compatibility("abm", 0.95) == "abm"
    assert validate_integrator_compatibility("efork3", 0.98) == "efork3"
    assert validate_integrator_compatibility("rk4", 1.0) == "rk4"
    assert validate_integrator_compatibility("efork_q1", 1.0) == "efork_q1"

    # Redirection warning
    with pytest.warns(UserWarning, match="redirects to the integer-order"):
        assert validate_integrator_compatibility("efork3", 1.0) == "efork3"

    # Invalid q ranges
    with pytest.raises(ValueError, match="must be in"):
        validate_integrator_compatibility("efork3", 0.0)
    with pytest.raises(ValueError, match="must be in"):
        validate_integrator_compatibility("efork3", 1.2)
    with pytest.raises(ValueError, match="finite"):
        validate_integrator_compatibility("efork3", np.nan)

    # Incompatible combos
    with pytest.raises(ValueError, match="requires q < 1"):
        validate_integrator_compatibility("abm", 1.0)
    with pytest.raises(ValueError, match="only supports integer-order"):
        validate_integrator_compatibility("rk4", 0.95)
    with pytest.raises(ValueError, match="Unknown integrator"):
        validate_integrator_compatibility("unregistered_integer_scheme", 1.0)


def test_integrate_dispatch():
    # Simple linear RHS: dx/dt = -x
    def rhs(t, x):
        return -x

    # Integer integration
    t, x, status = integrate(rhs, np.array([1.0]), q=1.0, h=0.01, t_final=1.0, integrator="rk4")
    assert status == "ok"
    assert len(t) == 101
    assert np.allclose(x[-1], np.exp(-1.0), rtol=1e-2)

    # Fractional integration (using Python fallback or general solver)
    t_frac, x_frac, status_frac = integrate(
        rhs, np.array([1.0]), q=0.98, h=0.01, t_final=0.5, integrator="efork3", use_c_backend=False
    )
    assert status_frac == "ok"
    assert len(t_frac) == 51


@pytest.mark.parametrize(
    ("q", "integrator"),
    ((1.0, "rk4"), (0.9, "abm"), (0.9, "efork3")),
)
def test_fixed_step_integrators_reject_horizon_overshoot(q, integrator) -> None:
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        integrate(
            lambda _t, state: np.zeros_like(state),
            np.array([1.0]),
            q=q,
            h=0.3,
            t_final=1.0,
            integrator=integrator,
            use_c_backend=False,
        )


@pytest.mark.parametrize(
    ("q", "integrator"),
    ((1.0, "rk4"), (0.9, "abm"), (0.9, "efork3")),
)
def test_fixed_step_integrators_accept_aligned_horizon(q, integrator) -> None:
    times, _, status = integrate(
        lambda _t, state: np.zeros_like(state),
        np.array([1.0]),
        q=q,
        h=0.3,
        t_final=0.9,
        integrator=integrator,
        use_c_backend=False,
    )

    assert status == "ok"
    assert len(times) == 4
    assert times[-1] == pytest.approx(0.9)


def test_selector_does_not_mask_rhs_body_typeerror() -> None:
    calls = 0

    def rhs(t, x):
        nonlocal calls
        calls += 1
        del t, x
        raise TypeError("selector-rhs-body-typeerror")

    _, _, status = integrate(
        rhs,
        np.array([1.0]),
        q=1.0,
        h=0.01,
        t_final=0.1,
        integrator="rk4",
        use_c_backend=False,
    )

    assert status == "solver_exception:selector-rhs-body-typeerror"
    assert calls == 1


def test_generic_selector_directs_adm_to_specialized_api() -> None:
    with pytest.raises(
        ValueError,
        match=r"adm_wu2023_integrate\(params, x0, q, h, N",
    ):
        integrate(
            lambda state: -state,
            np.array([1.0]),
            q=0.98,
            h=0.01,
            t_final=0.1,
            integrator="adm_wu2023",
            use_c_backend=False,
        )


def test_near_integer_order_remains_fractional() -> None:
    """Only exact q=1 may change the model class to an ODE."""

    for q_near_one in (1.0 - 5.0e-11, np.nextafter(1.0, 0.0)):
        with pytest.raises(ValueError, match="only supports integer-order"):
            validate_integrator_compatibility("rk4", q_near_one)
        assert validate_integrator_compatibility("efork3", q_near_one) == "efork3"


def test_selector_uses_explicit_rhs_when_system_is_only_an_acceleration_hint() -> None:
    system = SimpleNamespace(evaluate=lambda state: -100.0 * state)

    t, x, status = integrate(
        lambda state: -2.0 * state,
        np.array([1.0]),
        q=1.0,
        h=0.001,
        t_final=0.1,
        integrator="rk4",
        system=system,
        use_c_backend=False,
        early_stop_config={"enabled": False},
    )

    assert status == "ok"
    assert len(t) == 101
    assert x[-1, 0] == pytest.approx(np.exp(-0.2), rel=2.0e-6)


def test_selector_propagates_python_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_integrate_general(**kwargs):
        observed.update(kwargs)
        return np.array([0.0]), np.array([[1.0]]), "ok"

    monkeypatch.setattr(
        "hidden_attractors.integrations.selector.get_integrator_fn",
        lambda: fake_integrate_general,
    )

    integrate(
        lambda state: -state,
        np.array([1.0]),
        q=0.9,
        h=0.01,
        t_final=0.1,
        integrator="efork3",
        allow_python_fallback=False,
    )

    assert observed["allow_python_fallback"] is False


def test_use_c_backend_false_disables_integer_numba_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hidden_attractors.integrations import general

    calls = {"numba": 0}

    def fake_numba(**kwargs):
        calls["numba"] += 1
        raise AssertionError("Numba backend must not run when use_c_backend=False")

    monkeypatch.setattr(general, "_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(general, "integrate_efork3_q1_numba", fake_numba)

    t, x, status = general.integrate_general(
        lambda state: -state,
        np.array([1.0]),
        q=1.0,
        h=0.01,
        t_final=0.1,
        integrator="efork3",
        system=SimpleNamespace(),
        use_c_backend=False,
    )

    assert calls["numba"] == 0
    assert status == "ok"
    assert len(t) == len(x) == 11


def test_general_dispatcher_forwards_python_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hidden_attractors.integrations import general

    observed: dict[str, object] = {}

    def fake_fractional_integrate(**kwargs):
        observed.update(kwargs)
        return (
            np.array([0.0, 0.1]),
            np.array([[1.0], [0.9]]),
            "ok",
            {},
        )

    monkeypatch.setattr(general, "fractional_integrate", fake_fractional_integrate)

    general.integrate_general(
        lambda state: -state,
        np.array([1.0]),
        q=0.9,
        h=0.1,
        t_final=0.1,
        integrator="efork3",
        use_c_backend=True,
        allow_python_fallback=False,
    )

    assert observed["allow_python_fallback"] is False


def test_integer_backend_failure_respects_disabled_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hidden_attractors.integrations import general

    def fail_numba(**kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(general, "_NUMBA_AVAILABLE", True)
    monkeypatch.setattr(
        general,
        "integrate_efork3_q1_numba",
        fail_numba,
    )

    with pytest.raises(
        RuntimeError,
        match="allow_python_fallback=False",
    ):
        general.integrate_general(
            lambda state: -state,
            np.array([1.0]),
            q=1.0,
            h=0.01,
            t_final=0.1,
            integrator="efork3",
            system=SimpleNamespace(),
            use_c_backend=True,
            allow_python_fallback=False,
        )
