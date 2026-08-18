from __future__ import annotations

import ctypes
from math import gamma
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import hidden_attractors.fractional.tempered_caputo_solver as tempered_solver
from hidden_attractors.fractional import (
    FractionalProblem,
    integrate_tempered_caputo_abm,
    solve_fractional_problem,
)
from hidden_attractors.integrations.fractional_c import (
    GeneralFractionalCBackend,
    fractional_integrate,
)


def _zero_rhs(time: float, state: np.ndarray) -> np.ndarray:
    del time
    return np.zeros_like(state)


def _constant_rhs(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    del time
    return np.ones_like(state) * parameters[0]


def _tempered_problem(**updates: object) -> FractionalProblem:
    arguments: dict[str, object] = {
        "derivative": "tempered_caputo",
        "method": "tempered_caputo_abm_pece_transform",
        "orders": 0.63,
        "initial_state": [0.2],
        "step": 0.01,
        "t_span": (1.25, 1.45),
        "kernel_parameters": {"tempering": 0.4},
        "allow_experimental": True,
    }
    arguments.update(updates)
    return FractionalProblem(**arguments)  # type: ignore[arg-type]


def test_manufactured_tempered_power_solution_uses_shifted_physical_time() -> None:
    order = 0.62
    tempering = 0.7
    lower = 1.3
    duration = 0.4
    power = 2.0
    initial = np.array([0.75, -0.2])
    amplitudes = np.array([1.2, -0.35])
    coefficient = gamma(power + 1.0) / gamma(power + 1.0 - order)
    seen_times: list[float] = []

    def manufactured_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del state
        seen_times.append(float(time))
        tau = time - lower
        return (
            np.exp(-tempering * tau)
            * coefficient
            * tau ** (power - order)
            * amplitudes
        )

    result = integrate_tempered_caputo_abm(
        manufactured_rhs,
        initial,
        order,
        tempering=tempering,
        lower_terminal=lower,
        upper_terminal=lower + duration,
        step=0.002,
        use_acceleration=False,
        divergence_norm=None,
    )
    tau = result.times - lower
    expected = np.exp(-tempering * tau)[:, None] * (
        initial[None, :] + tau[:, None] ** power * amplitudes[None, :]
    )

    assert result.status == "ok"
    np.testing.assert_allclose(result.states, expected, rtol=1.2e-5, atol=2.0e-7)
    assert min(seen_times) == pytest.approx(lower)
    assert max(seen_times) == pytest.approx(lower + duration)


@pytest.mark.parametrize(
    ("memory_policy", "history_window"),
    [("full_history", None), ("finite_window", 7)],
)
def test_zero_tempering_is_exact_caputo_abm_software_reduction(
    memory_policy: str,
    history_window: int | None,
) -> None:
    lower = 1.2
    duration = 0.2
    step = 0.01
    order = 0.67
    initial = np.array([0.2, -0.3])

    def physical_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.array(
            [0.2 * time - 0.1 * state[0], -0.15 * state[1] + 0.03 * time]
        )

    tempered = integrate_tempered_caputo_abm(
        physical_rhs,
        initial,
        order,
        tempering=0.0,
        lower_terminal=lower,
        upper_terminal=lower + duration,
        step=step,
        memory_policy=memory_policy,
        history_window=history_window,
        use_acceleration=False,
        divergence_norm=None,
    )

    def shifted_rhs(local_time: float, state: np.ndarray) -> np.ndarray:
        return physical_rhs(lower + local_time, state)

    direct_times, direct_states, direct_status, direct_info = fractional_integrate(
        rhs=shifted_rhs,
        x0=initial,
        q=order,
        h=step,
        t_final=float(np.nextafter(duration, -np.inf)),
        method="abm",
        memory_mode="full" if memory_policy == "full_history" else "window",
        memory_window_length=history_window,
        use_c_backend=False,
        divergence_norm=float("inf"),
        return_history=True,
        allow_python_fallback=True,
        early_stop_config={"enabled": False},
    )

    assert tempered.status == direct_status == "ok"
    np.testing.assert_array_equal(tempered.times, lower + direct_times)
    np.testing.assert_array_equal(tempered.states, direct_states)
    np.testing.assert_array_equal(tempered.transformed_states, direct_states)
    assert tempered.solver_info["lambda_zero_reduction"] is True
    assert tempered.solver_info["n_steps_completed"] == direct_info["n_steps_completed"]


def test_finite_window_is_explicitly_recorded_and_changes_long_history() -> None:
    arguments = dict(
        rhs=_constant_rhs,
        initial_state=[0.0],
        order=0.58,
        parameters=np.array([1.0]),
        tempering=0.25,
        lower_terminal=0.5,
        upper_terminal=1.0,
        step=0.01,
        use_acceleration=False,
        divergence_norm=None,
    )
    full = integrate_tempered_caputo_abm(**arguments)
    window = integrate_tempered_caputo_abm(
        **arguments,
        memory_policy="finite_window",
        history_window=5,
    )

    assert full.memory_policy == "full_history"
    assert full.history_window is None
    assert full.solver_info["truncated_memory"] is False
    assert window.memory_policy == "finite_window"
    assert window.history_window == 5
    assert window.solver_info["truncated_memory"] is True
    assert not np.allclose(full.states[-1], window.states[-1])


def test_accelerated_backend_agrees_with_python_when_native_is_available() -> None:
    arguments = dict(
        rhs=_constant_rhs,
        initial_state=[0.3, -0.2],
        order=0.67,
        parameters=np.array([0.15]),
        tempering=0.35,
        lower_terminal=1.0,
        upper_terminal=1.2,
        step=0.01,
        divergence_norm=None,
    )
    reference = integrate_tempered_caputo_abm(
        **arguments,
        use_acceleration=False,
    )
    accelerated = integrate_tempered_caputo_abm(
        **arguments,
        use_acceleration=True,
        allow_python_fallback=True,
    )

    assert reference.backend == "python_numpy_tempered_abm_physical"
    assert reference.solver_info["used_c_backend"] is False
    np.testing.assert_array_equal(accelerated.times, reference.times)
    np.testing.assert_allclose(
        accelerated.states,
        reference.states,
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    if accelerated.solver_info["used_c_backend"]:
        assert accelerated.backend == "native_c_tempered_abm_physical"
        assert "c_backend_error" not in accelerated.solver_info
    else:
        assert accelerated.backend == "python_numpy_tempered_abm_physical"
        assert "c_backend_error" in accelerated.solver_info


def test_native_failure_falls_back_only_when_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(cls) -> None:
        del cls
        raise OSError("synthetic-native-unavailable")

    monkeypatch.setattr(
        GeneralFractionalCBackend,
        "get_instance",
        classmethod(unavailable),
    )
    arguments = dict(
        rhs=_zero_rhs,
        initial_state=[0.4],
        order=0.6,
        tempering=0.3,
        lower_terminal=0.0,
        upper_terminal=0.1,
        step=0.01,
        use_acceleration=True,
        divergence_norm=None,
    )
    fallback = integrate_tempered_caputo_abm(
        **arguments,
        allow_python_fallback=True,
    )
    assert fallback.status == "ok"
    assert fallback.backend == "python_numpy_tempered_abm_physical"
    assert fallback.solver_info["used_c_backend"] is False
    assert "synthetic-native-unavailable" in fallback.solver_info["c_backend_error"]

    with pytest.raises(RuntimeError, match="allow_python_fallback=False"):
        integrate_tempered_caputo_abm(
            **arguments,
            allow_python_fallback=False,
        )


def test_divergence_is_truncated_at_first_crossing_in_physical_state() -> None:
    threshold = 0.2
    result = integrate_tempered_caputo_abm(
        _constant_rhs,
        [0.0],
        0.5,
        np.array([5.0]),
        tempering=2.0,
        lower_terminal=2.0,
        upper_terminal=2.2,
        step=0.01,
        use_acceleration=False,
        divergence_norm=threshold,
    )

    assert result.status == "diverged"
    assert len(result.times) == 2
    assert np.linalg.norm(result.states[-2]) <= threshold
    assert np.linalg.norm(result.states[-1]) > threshold
    assert result.actual_upper_terminal == pytest.approx(result.times[-1])
    assert result.actual_upper_terminal < result.requested_upper_terminal
    assert result.solver_info["underlying_status"] == "diverged"
    assert result.solver_info["divergence_coordinate"] == "physical_state"
    assert result.solver_info["physical_divergence_norm"] == threshold
    assert result.solver_info["transformed_divergence_norm"] is None
    assert result.solver_info["n_steps_completed"] == 1


def test_structured_result_preserves_transform_grid_backend_and_scope() -> None:
    lower = 1.1
    result = integrate_tempered_caputo_abm(
        _zero_rhs,
        [0.8, -0.4],
        0.61,
        tempering=0.5,
        lower_terminal=lower,
        upper_terminal=1.3,
        step=0.01,
        use_acceleration=False,
        divergence_norm=None,
    )
    factors = np.exp(0.5 * (result.times - lower))

    assert result.method == "tempered_caputo_abm_pece_transform"
    assert result.grid_coordinate == "physical_time"
    assert result.scope == "finite_numerical_trajectory_only"
    assert result.n_steps_requested == 20
    assert result.actual_upper_terminal == pytest.approx(1.3)
    assert result.solver_info["conjugation"] == "v=exp(tempering*(t-a))*x"
    assert result.solver_info["underlying_method"] == "caputo_abm_pece_damped_history"
    assert result.solver_info["maximum_exponent"] == pytest.approx(0.1)
    assert isinstance(result.solver_info, MappingProxyType)
    np.testing.assert_allclose(
        result.transformed_states,
        result.states * factors[:, None],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_array_equal(result.trajectory[:, 0], result.times)
    np.testing.assert_array_equal(result.trajectory[:, 1:], result.states)
    assert "https://doi.org/10.3934/dcdsb.2019026" in result.references
    assert "https://doi.org/10.1016/j.jcp.2014.04.024" in result.references


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"rhs": None}, TypeError, "callable"),
        ({"initial_state": []}, ValueError, "at least one finite"),
        ({"initial_state": [1.0 + 1.0j]}, TypeError, "real-valued"),
        ({"initial_state": [np.nan]}, ValueError, "finite"),
        ({"order": True}, TypeError, "order"),
        ({"order": 0.0}, ValueError, "strictly"),
        ({"order": 1.0}, ValueError, "strictly"),
        ({"tempering": True}, TypeError, "tempering"),
        ({"tempering": -0.1}, ValueError, "nonnegative"),
        ({"tempering": np.inf}, ValueError, "finite"),
        ({"lower_terminal": np.nan}, ValueError, "finite"),
        ({"upper_terminal": 0.0}, ValueError, "greater"),
        ({"step": True}, TypeError, "step"),
        ({"step": 0.0}, ValueError, "positive"),
        ({"use_acceleration": 1}, TypeError, "Boolean"),
        ({"allow_python_fallback": 1}, TypeError, "Boolean"),
        ({"divergence_norm": True}, TypeError, "divergence_norm"),
        ({"divergence_norm": 0.0}, ValueError, "positive or None"),
    ],
)
def test_invalid_direct_solver_contract_is_rejected(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "rhs": _zero_rhs,
        "initial_state": [0.0],
        "order": 0.5,
        "tempering": 0.1,
        "lower_terminal": 0.0,
        "upper_terminal": 0.2,
        "step": 0.01,
        "use_acceleration": False,
    }
    arguments.update(updates)
    with pytest.raises(error, match=match):
        integrate_tempered_caputo_abm(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"memory_policy": "fast_history"}, "memory_policy"),
        ({"memory_policy": "finite_window"}, "history_window >= 2"),
        (
            {"memory_policy": "finite_window", "history_window": True},
            "history_window >= 2",
        ),
        (
            {"memory_policy": "finite_window", "history_window": 1},
            "history_window >= 2",
        ),
        ({"history_window": 5}, "only valid"),
    ],
)
def test_invalid_memory_contract_is_rejected(
    updates: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "rhs": _zero_rhs,
        "initial_state": [0.0],
        "order": 0.5,
        "tempering": 0.1,
        "lower_terminal": 0.0,
        "upper_terminal": 0.2,
        "step": 0.01,
        "use_acceleration": False,
    }
    arguments.update(updates)
    with pytest.raises(ValueError, match=match):
        integrate_tempered_caputo_abm(**arguments)  # type: ignore[arg-type]


def test_nonintegral_grid_is_rejected_but_large_tempering_stays_physical() -> None:
    with pytest.raises(ValueError, match="integer number"):
        integrate_tempered_caputo_abm(
            _zero_rhs,
            [0.0],
            0.5,
            tempering=0.1,
            lower_terminal=0.0,
            upper_terminal=0.205,
            step=0.01,
            use_acceleration=False,
        )
    result = integrate_tempered_caputo_abm(
        _zero_rhs,
        [0.0],
        0.5,
        tempering=1000.0,
        lower_terminal=0.0,
        upper_terminal=1.0,
        step=0.1,
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.status == "ok"
    assert np.all(result.states == 0.0)
    assert result.transformed_states is None
    assert result.solver_info["transformed_states_stored"] is False


def test_native_step_limit_is_rejected_before_output_allocation() -> None:
    native_step_limit = int(np.iinfo(np.int32).max) - 3
    with pytest.raises(ValueError, match="supported limit"):
        integrate_tempered_caputo_abm(
            _zero_rhs,
            [0.0],
            0.5,
            tempering=0.1,
            lower_terminal=0.0,
            upper_terminal=float(native_step_limit + 1),
            step=1.0,
            use_acceleration=False,
        )


def test_output_capacities_are_checked_before_solver_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_shapes: list[tuple[int, ...]] = []
    original = tempered_solver.checked_array_capacity

    def recording_capacity(shape, dtype, *, caller, max_bytes=None):
        checked_shapes.append(tuple(shape))
        return original(shape, dtype, caller=caller, max_bytes=max_bytes)

    monkeypatch.setattr(
        tempered_solver,
        "checked_array_capacity",
        recording_capacity,
    )
    result = integrate_tempered_caputo_abm(
        _zero_rhs,
        [0.0, 1.0],
        0.5,
        tempering=0.1,
        lower_terminal=0.0,
        upper_terminal=0.2,
        step=0.1,
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.status == "ok"
    assert checked_shapes == [(3,), (3, 2)]


def test_native_tempered_abi_receives_element_capacities_and_finite_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeNativeFunction:
        argtypes = None
        restype = None

        def __call__(self, *arguments):
            captured["arguments"] = arguments
            arguments[11][:] = np.array([0.0, 0.1, 0.2])
            arguments[12][:] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            ctypes.cast(arguments[15], ctypes.POINTER(ctypes.c_int))[0] = 3
            ctypes.cast(arguments[16], ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0

    callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_void_p,
    )
    function = FakeNativeFunction()
    backend = SimpleNamespace(
        RHS_CALLBACK=callback_type,
        lib=SimpleNamespace(integrate_tempered_caputo_abm_c=function),
    )
    monkeypatch.setattr(
        GeneralFractionalCBackend,
        "get_instance",
        classmethod(lambda cls: backend),
    )

    result = integrate_tempered_caputo_abm(
        _zero_rhs,
        [0.0, 1.0],
        0.5,
        tempering=0.1,
        lower_terminal=0.0,
        upper_terminal=0.2,
        step=0.1,
        use_acceleration=True,
        allow_python_fallback=False,
        divergence_norm=None,
    )

    arguments = captured["arguments"]
    assert function.argtypes[13:15] == [ctypes.c_size_t, ctypes.c_size_t]
    assert arguments[10].value == sys.float_info.max
    assert arguments[13].value == 3
    assert arguments[14].value == 6
    assert result.solver_info["used_c_backend"] is True


@pytest.mark.parametrize(
    "bad_rhs",
    [
        lambda time, state: np.ones(state.size + 1),
        lambda time, state: np.full_like(state, np.nan),
        lambda time, state: np.full(state.shape, 1.0j),
    ],
)
def test_invalid_initial_rhs_is_rejected(bad_rhs) -> None:
    with pytest.raises((TypeError, ValueError), match="rhs"):
        integrate_tempered_caputo_abm(
            bad_rhs,
            [0.0],
            0.5,
            tempering=0.1,
            lower_terminal=0.0,
            upper_terminal=0.1,
            step=0.01,
            use_acceleration=False,
        )


def test_internal_rhs_typeerror_is_not_reinterpreted_as_a_signature() -> None:
    calls = 0

    def broken_rhs(time: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        del time, state
        calls += 1
        raise TypeError("tempered-internal-typeerror")

    with pytest.raises(TypeError, match="tempered-internal-typeerror"):
        integrate_tempered_caputo_abm(
            broken_rhs,
            [0.0],
            0.5,
            tempering=0.1,
            lower_terminal=0.0,
            upper_terminal=0.1,
            step=0.01,
            use_acceleration=False,
        )
    assert calls == 1


def test_python_backend_should_propagate_late_rhs_shape_failure() -> None:
    def changing_shape_rhs(time: float, state: np.ndarray) -> np.ndarray:
        if time > 0.025:
            return np.empty(state.size + 1)
        return np.ones_like(state)

    with pytest.raises(ValueError, match="shape"):
        integrate_tempered_caputo_abm(
            changing_shape_rhs,
            [0.0],
            0.5,
            tempering=0.1,
            lower_terminal=0.0,
            upper_terminal=0.1,
            step=0.01,
            use_acceleration=False,
            allow_python_fallback=True,
        )


def test_native_callback_propagates_late_rhs_shape_failure_when_available() -> None:
    probe = integrate_tempered_caputo_abm(
        _zero_rhs,
        [0.0],
        0.5,
        tempering=0.1,
        lower_terminal=0.0,
        upper_terminal=0.02,
        step=0.01,
        use_acceleration=True,
        allow_python_fallback=True,
    )
    if not probe.solver_info["used_c_backend"]:
        pytest.skip("The native C callback path is unavailable on this host.")

    def changing_shape_rhs(time: float, state: np.ndarray) -> np.ndarray:
        if time > 0.025:
            return np.empty(state.size + 1)
        return np.ones_like(state)

    with pytest.raises(ValueError, match="shape"):
        integrate_tempered_caputo_abm(
            changing_shape_rhs,
            [0.0],
            0.5,
            tempering=0.1,
            lower_terminal=0.0,
            upper_terminal=0.1,
            step=0.01,
            use_acceleration=True,
            allow_python_fallback=True,
        )


def test_fractional_problem_accepts_promoted_solver_without_opt_in() -> None:
    problem = _tempered_problem(allow_experimental=False)
    problem.validate_executable()
    result = solve_fractional_problem(
        problem,
        _zero_rhs,
        use_acceleration=False,
    )
    assert result.status == "ok"


def test_fractional_problem_dispatches_and_preserves_kernel_metadata() -> None:
    problem = _tempered_problem(problem_id="tempered-contract")
    result = solve_fractional_problem(
        problem,
        _constant_rhs,
        np.array([0.3]),
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.status == "ok"
    assert result.backend == "python_numpy_tempered_abm_physical"
    assert result.times[0] == pytest.approx(1.25)
    assert result.times[-1] == pytest.approx(1.45)
    assert result.metadata["problem_id"] == "tempered-contract"
    assert result.metadata["derivative"] == "tempered_caputo"
    assert result.metadata["method"] == "tempered_caputo_abm_pece_transform"
    assert result.metadata["kernel_parameters"] == {"tempering": 0.4}
    assert result.metadata["allow_experimental"] is True
    assert result.metadata["backend"] == result.backend
    assert result.metadata["backend_info"]["tempering"] == pytest.approx(0.4)
    assert result.metadata["backend_info"]["memory_policy"] == "full_history"
    assert result.metadata["backend_info"]["actual_upper_terminal"] == pytest.approx(
        1.45
    )
    assert result.metadata["claims"] == "finite_numerical_trajectory_only"


def test_fractional_problem_forwards_finite_window_without_hidden_defaults() -> None:
    problem = _tempered_problem(
        memory_policy="finite_window",
        history_window=6,
    )
    dispatched = solve_fractional_problem(
        problem,
        _constant_rhs,
        np.array([0.3]),
        use_acceleration=False,
        divergence_norm=None,
    )
    direct = integrate_tempered_caputo_abm(
        _constant_rhs,
        [0.2],
        0.63,
        np.array([0.3]),
        tempering=0.4,
        lower_terminal=1.25,
        upper_terminal=1.45,
        step=0.01,
        memory_policy="finite_window",
        history_window=6,
        use_acceleration=False,
        divergence_norm=None,
    )

    np.testing.assert_array_equal(dispatched.times, direct.times)
    np.testing.assert_array_equal(dispatched.states, direct.states)
    assert dispatched.status == direct.status == "ok"
    assert dispatched.metadata["memory_policy"] == "finite_window"
    assert dispatched.metadata["history_window"] == 6
    assert dispatched.metadata["backend_info"]["truncated_memory"] is True
    assert dispatched.metadata["backend_info"]["history_window"] == 6


def test_fractional_problem_rejects_unconsumed_options_before_execution() -> None:
    unused_method_option = _tempered_problem(method_options={"startup_tolerance": 1e-8})
    with pytest.raises(NotImplementedError, match="does not yet consume method_options"):
        solve_fractional_problem(
            unused_method_option,
            _zero_rhs,
            use_acceleration=False,
        )

    unused_kernel_option = _tempered_problem(
        kernel_parameters={"tempering": 0.4, "quadrature": "silent-no-more"}
    )
    with pytest.raises(ValueError, match="unsupported kernel_parameters"):
        solve_fractional_problem(
            unused_kernel_option,
            _zero_rhs,
            use_acceleration=False,
        )


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"kernel_parameters": {}}, ValueError, "tempering"),
        (
            {"kernel_parameters": {"tempering": True}},
            TypeError,
            "real kernel_parameters",
        ),
        (
            {"kernel_parameters": {"tempering": -0.1}},
            ValueError,
            "tempering.*>= 0",
        ),
        ({"orders": 1.0}, ValueError, "requires 0 < order < 1"),
        (
            {"orders": [0.5, np.nextafter(0.5, np.inf)], "initial_state": [0.0, 0.0]},
            ValueError,
            "componentwise",
        ),
        (
            {
                "grid_coordinate": "log_t_over_lower_terminal",
                "t_span": (1.0, float(np.exp(0.2))),
            },
            ValueError,
            "implemented only",
        ),
        (
            {"memory_policy": "finite_window", "history_window": 1},
            ValueError,
            "history_window >= 2",
        ),
        ({"history_window": 3}, ValueError, "only valid"),
    ],
)
def test_fractional_problem_validation_rejects_invalid_tempered_contracts(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _tempered_problem(**updates)


def test_fractional_problem_round_trip_keeps_tempering_and_experimental_flag() -> None:
    original = _tempered_problem(
        memory_policy="finite_window",
        history_window=8,
    )
    restored = FractionalProblem.from_mapping(original.to_mapping())

    assert restored == original
    assert restored.kernel_parameters["tempering"] == pytest.approx(0.4)
    assert restored.memory_policy == "finite_window"
    assert restored.history_window == 8
    assert restored.allow_experimental is True
