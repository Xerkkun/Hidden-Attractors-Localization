from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from hidden_attractors.integrations.abm import _python_abm_integrate, caputo_abm_integrate
from hidden_attractors.integrations.efork import _python_efork3_integrate
from hidden_attractors.integrations._history import validate_prehistory
from hidden_attractors.continuation.continuation_fractional import (
    run_fractional_continuation_abm_monolithic,
)


Integrator = Callable[..., tuple[np.ndarray, np.ndarray, str]]


@pytest.fixture(params=[_python_abm_integrate, _python_efork3_integrate])
def direct_integrator(request: pytest.FixtureRequest) -> Integrator:
    return request.param


def _base_kwargs() -> dict[str, object]:
    return {
        "rhs": lambda _time, state: -np.asarray(state),
        "x0": np.array([1.0]),
        "q": 0.8,
        "h": 0.1,
        "t_final": 0.2,
        "divergence_norm": None,
        "early_stop_config": {"enabled": False},
    }


def test_direct_integrators_require_paired_history(direct_integrator: Integrator) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        direct_integrator(
            **_base_kwargs(),
            history_times=np.array([-0.1, 0.0]),
            history_states=None,
        )


@pytest.mark.parametrize(
    ("times", "states", "message"),
    [
        (np.array([-0.2, -0.05, 0.0]), np.ones((3, 1)), "same h grid"),
        (np.array([-0.2, -0.1, 0.01]), np.ones((3, 1)), "end at t=0"),
        (np.array([-0.2, -0.1, 0.0]), np.ones((2, 1)), "must have shape"),
        (np.array([-0.2, -0.1, 0.0]), np.array([[1.0], [np.nan], [1.0]]), "finite"),
        (np.array([-0.2, -0.1, 0.0]), np.array([[1.0], [1.0], [2.0]]), "must equal x0"),
    ],
)
def test_direct_integrators_reject_malformed_history(
    direct_integrator: Integrator,
    times: np.ndarray,
    states: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        direct_integrator(
            **_base_kwargs(),
            history_times=times,
            history_states=states,
        )


def test_direct_integrators_reject_unknown_memory_mode(
    direct_integrator: Integrator,
) -> None:
    with pytest.raises(ValueError, match="exactly 'full' or 'window'"):
        direct_integrator(**_base_kwargs(), memory_mode="widnow")


def test_direct_integrators_require_positive_integer_window(
    direct_integrator: Integrator,
) -> None:
    with pytest.raises(ValueError, match="required"):
        direct_integrator(**_base_kwargs(), memory_mode="window")
    with pytest.raises(TypeError, match="positive integer"):
        direct_integrator(
            **_base_kwargs(),
            memory_mode="window",
            memory_window_length=2.0,
        )


def test_direct_integrators_accept_anchored_uniform_history(
    direct_integrator: Integrator,
) -> None:
    times, states, status = direct_integrator(
        **_base_kwargs(),
        history_times=np.array([-0.2, -0.1, 0.0]),
        history_states=np.ones((3, 1)),
    )
    assert status == "ok"
    assert times[-1] == pytest.approx(0.2)
    assert np.allclose(np.diff(times), 0.1)
    assert states.shape[1] == 1


@pytest.mark.parametrize(
    "x0",
    [np.empty(0), np.array([[1.0]]), np.array([np.nan])],
)
def test_direct_integrators_reject_invalid_initial_state(
    direct_integrator: Integrator,
    x0: np.ndarray,
) -> None:
    kwargs = _base_kwargs()
    kwargs["x0"] = x0
    with pytest.raises(ValueError, match="non-empty finite one-dimensional"):
        direct_integrator(**kwargs)


@pytest.mark.parametrize("q", [float("nan"), 0.0, 1.1])
def test_direct_integrators_reject_invalid_order(
    direct_integrator: Integrator,
    q: float,
) -> None:
    kwargs = _base_kwargs()
    kwargs["q"] = q
    with pytest.raises(ValueError, match="q must satisfy"):
        direct_integrator(**kwargs)


@pytest.mark.parametrize(
    ("rhs", "message"),
    [
        (lambda _time, _state: 0.0, "rhs must return shape \\(2,\\)"),
        (lambda _time, _state: np.array([0.0, np.nan]), "finite derivatives"),
    ],
)
def test_direct_integrators_reject_invalid_rhs_result(
    direct_integrator: Integrator,
    rhs: Callable[..., object],
    message: str,
) -> None:
    kwargs = _base_kwargs()
    kwargs.update(rhs=rhs, x0=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match=message):
        direct_integrator(**kwargs)


def test_direct_integrators_preserve_rhs_exception(
    direct_integrator: Integrator,
) -> None:
    def broken_rhs(_time: float, _state: np.ndarray) -> np.ndarray:
        raise RuntimeError("rhs callback failed")

    kwargs = _base_kwargs()
    kwargs["rhs"] = broken_rhs
    with pytest.raises(RuntimeError, match="rhs callback failed"):
        direct_integrator(**kwargs)


@pytest.mark.parametrize(
    ("equilibrium", "message"),
    [
        (np.array([0.0]), "must have shape \\(2,\\)"),
        (np.array([0.0, np.nan]), "finite values"),
    ],
)
def test_direct_integrators_reject_invalid_equilibria(
    direct_integrator: Integrator,
    equilibrium: np.ndarray,
    message: str,
) -> None:
    kwargs = _base_kwargs()
    kwargs["x0"] = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match=message):
        direct_integrator(**kwargs, equilibria=[equilibrium])


def test_prehistory_validation_does_not_mutate_caller_states() -> None:
    x0 = np.array([1.0])
    states = np.array([[0.5], [1.0 + 8.0e-15]])
    original = states.copy()
    _, canonical = validate_prehistory(
        np.array([-0.1, 0.0]),
        states,
        x0=x0,
        h=0.1,
        caller="test",
    )
    np.testing.assert_array_equal(states, original)
    assert canonical is not None
    np.testing.assert_array_equal(canonical[-1], x0)


def test_abm_memory_window_time_must_align_with_h() -> None:
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        _python_abm_integrate(
            **_base_kwargs(),
            memory_mode="window",
            memory_window_time=0.15,
        )


def test_public_abm_memory_window_time_must_align_with_h() -> None:
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        caputo_abm_integrate(
            **_base_kwargs(),
            memory_mode="window",
            memory_window_time=0.15,
            use_c_backend=False,
        )


@pytest.mark.parametrize("divergence_norm", [float("nan"), float("-inf"), -1.0])
def test_direct_integrators_reject_invalid_divergence_norm(
    direct_integrator: Integrator,
    divergence_norm: float,
) -> None:
    kwargs = _base_kwargs()
    kwargs["divergence_norm"] = divergence_norm
    with pytest.raises(ValueError, match="divergence_norm must be positive"):
        direct_integrator(**kwargs)


@pytest.mark.parametrize("divergence_norm", [None, float("inf")])
def test_direct_integrators_accept_disabled_divergence_cutoff(
    direct_integrator: Integrator,
    divergence_norm: float | None,
) -> None:
    kwargs = _base_kwargs()
    kwargs["divergence_norm"] = divergence_norm
    times, _, status = direct_integrator(**kwargs)
    assert status == "ok"
    assert times[-1] == pytest.approx(0.2)


def test_monolithic_continuation_rejects_unilateral_history() -> None:
    system = SimpleNamespace(parameters={"q": 0.8})
    with pytest.raises(ValueError, match="must be provided together"):
        run_fractional_continuation_abm_monolithic(
            system=system,
            seed_x0=np.array([1.0]),
            k_gain=1.0,
            lambda_values=[0.0],
            h=0.1,
            t_transient=0.1,
            t_keep=0.1,
            history_times=np.array([-0.1, 0.0]),
            history_states=None,
        )
