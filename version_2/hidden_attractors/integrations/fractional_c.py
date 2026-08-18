"""Unified C-backed fractional integrator interface.

Architecture
------------
* :class:`GeneralFractionalCBackend` is a **singleton** that lazily compiles
  the native C library (``fractional_integrators.c``) and exposes the
  ``integrate_fractional_c`` entry point via :mod:`ctypes`.
* :func:`fractional_integrate` is the public API.  It attempts the C backend
  first; if that fails *and* ``allow_python_fallback=True`` it emits a warning,
  records the native error in the returned provenance, and uses the pure-Python
  ABM or EFORK-3 solver.
"""

import ctypes
import math
import sys
import threading
import warnings
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .._rhs import bind_rhs
from .._time_grid import checked_array_capacity, exact_fixed_step_count
from ..native.rhs_registry import get_c_rhs_and_params
from ..parallel import load_ctypes_library
from ..paths import get_native_cache
from ._history import (
    validate_divergence_norm,
    validate_equilibria,
    validate_memory_policy,
    validate_prehistory,
    validate_rhs_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eval_rhs(rhs: Callable, t: float, x: np.ndarray) -> np.ndarray:
    """Evaluate a supported RHS signature without masking internal errors."""

    return np.asarray(bind_rhs(rhs)(t, x), dtype=float)


def _shared_suffix() -> str:
    """Return the platform-appropriate shared-library extension."""
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < 1 or normalized > np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be between 1 and INT32_MAX.")
    return normalized


def _binary_flag(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return int(value)
    raise ValueError(f"{name} must be boolean or 0/1.")


def _finite_float(value: Any, name: str, *, positive: bool = False,
                  non_negative: bool = False) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if non_negative and normalized < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized


def _enforce_effective_horizon(times: np.ndarray, t_final: float) -> np.ndarray:
    """Reject a real overshoot and canonicalize round-off at the horizon."""

    if len(times) == 0:
        raise RuntimeError("fractional integrator returned no time samples.")
    last = float(times[-1])
    scale = max(abs(last), abs(t_final), 1.0)
    tolerance = max(
        64.0 * float(np.finfo(np.float64).eps) * scale,
        8.0 * math.ulp(last),
        8.0 * math.ulp(t_final),
    )
    if last > t_final + tolerance:
        raise RuntimeError("fractional trajectory exceeded the validated t_final.")
    if abs(last - t_final) <= tolerance:
        if not times.flags.writeable:
            times = times.copy()
        times[-1] = t_final
    return times


def _physical_trajectory_start(times: np.ndarray) -> int:
    """Return the index of the shared ``t=0`` initial condition."""

    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise RuntimeError("fractional integrator returned an invalid time grid.")
    start = int(np.searchsorted(times, 0.0, side="left"))
    if start >= times.size:
        raise RuntimeError("fractional integrator did not return the t=0 state.")
    scale = max(1.0, float(np.max(np.abs(times))))
    tolerance = 64.0 * float(np.finfo(np.float64).eps) * scale
    if abs(float(times[start])) > tolerance:
        raise RuntimeError("fractional integrator did not return the t=0 state.")
    return start


# ---------------------------------------------------------------------------
# Singleton C backend
# ---------------------------------------------------------------------------

class GeneralFractionalCBackend:
    """Singleton wrapper for the compiled generic C fractional integrator.

    Call :meth:`get_instance` to obtain (and lazily initialise) the singleton.
    """

    # Class-level singleton cache — never store instance on the instance itself
    _instance: Optional["GeneralFractionalCBackend"] = None
    _instance_lock = threading.RLock()

    # Ctypes callback type — set during initialisation
    RHS_CALLBACK: Any = None

    def __init__(self) -> None:
        self.lib: Any = None

    @classmethod
    def get_instance(cls) -> "GeneralFractionalCBackend":
        """Return the singleton, compiling the C library on first call."""
        with cls._instance_lock:
            if cls._instance is not None:
                return cls._instance

            native_cache = get_native_cache()
            src_path = (
                Path(__file__).resolve().parent.parent
                / "native" / "csrc" / "fractional_integrators.c"
            )
            out_path = native_cache / f"fractional_integrators{_shared_suffix()}"
            result, lib = load_ctypes_library(
                src_path,
                out_path,
                openmp=False,
                expected_symbols=(
                    "fractional_integrators_abi_version",
                    "integrate_fractional_c",
                    "chua_saturation_rhs_c",
                    "chua_arctan_rhs_c",
                ),
                expected_abi_version=3,
                abi_version_symbol="fractional_integrators_abi_version",
            )

            # Callback type: void (*RhsCallback)(double t, const double *x,
            #                                    double *dx, int n, void *params)
            cls.RHS_CALLBACK = ctypes.CFUNCTYPE(
                None,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_void_p,
            )
            vector = np.ctypeslib.ndpointer(
                dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
            )
            lib.integrate_fractional_c.argtypes = [
                cls.RHS_CALLBACK, ctypes.c_void_p, ctypes.c_int, vector,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                vector, vector, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_double, vector, vector, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                ctypes.c_int, ctypes.c_double, vector, ctypes.c_int,
                ctypes.c_size_t,
            ]
            lib.integrate_fractional_c.restype = ctypes.c_int

            backend = cls()
            backend.lib = lib
            cls._instance = backend
            return backend


# ---------------------------------------------------------------------------
# Public integration API
# ---------------------------------------------------------------------------

def fractional_integrate(
    rhs: Any,
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    method: str,                          # "abm" or "efork" / "efork3"
    memory_mode: str,                     # "full" or "window"
    memory_window_length: Optional[int] = None,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    system: Optional[Any] = None,
    params: Optional[Any] = None,         # unused; kept for API compatibility
    use_c_backend: bool = True,
    divergence_norm: float | None = 120.0,
    return_history: bool = False,
    allow_python_fallback: bool = False,
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, str, dict]:
    """Integrate a fractional-order ODE system.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(t, x) -> ndarray``.  A legacy signature
        ``rhs(x)`` is also accepted.
    x0 : array_like, shape (n,)
        Initial state.
    q : float
        Caputo fractional order.  Must satisfy ``0 < q < 1`` (q=1 is
        rejected — use ``integrate_general`` for integer-order systems).
    h : float
        Time step.
    t_final : float
        Integration end time.
    method : {"abm", "efork", "efork3"}
        Integration method.  ``"efork_q1"`` is rejected.
    memory_mode : {"full", "window"}
        Memory truncation mode for the fractional memory sum.
    memory_window_length : int or None
        Number of steps to keep when ``memory_mode="window"``.
    history_times : ndarray or None
        Pre-history time stamps (negative times).
    history_states : ndarray or None
        Pre-history states, shape ``(history_len, dim)``.
    system : object or None
        Lur'e system instance.  If it has a registered C RHS, that is
        used directly without a Python callback.
    use_c_backend : bool
        Attempt the native C integrator.  Falls back to Python if False
        or if compilation/execution fails (requires
        ``allow_python_fallback=True``).
    divergence_norm : float or None
        Halt integration when ``‖x‖ > divergence_norm``. ``None`` and positive
        infinity disable this cutoff; the native ABI receives the largest
        finite double for that disabled state.
    return_history : bool
        Include the pre-history segment in the returned arrays.
    allow_python_fallback : bool
        If True and the C backend fails, warn, record ``c_backend_error`` in
        ``info``, and switch to the Python solver.
    early_stop_config : dict or None
        Early-stopping configuration dictionary.
    equilibria : list of ndarray or None
        Equilibrium points used for convergence early-stopping.

    Returns
    -------
    times : ndarray, shape (N,)
    states : ndarray, shape (N, dim)
    status : str  — "ok" | "diverged" | "diverged_early" | "converged_equilibrium_early" | …
    info : dict
        Backend metadata. ``n_steps``/``n_steps_completed`` count completed
        integration increments; ``n_samples`` counts all stored points and
        ``n_samples_returned`` counts points after the requested history slice.
    """
    q = _finite_float(q, "q")
    h = _finite_float(h, "h", positive=True)
    t_final = _finite_float(t_final, "t_final", non_negative=True)
    normalized_divergence_norm = validate_divergence_norm(
        divergence_norm,
        caller="fractional_integrate",
    )
    native_divergence_norm = (
        sys.float_info.max
        if normalized_divergence_norm is None
        else normalized_divergence_norm
    )
    if not isinstance(return_history, (bool, np.bool_)):
        raise TypeError("return_history must be boolean.")
    native_step_limit = int(np.iinfo(np.int32).max) - 2
    nsteps = exact_fixed_step_count(
        h,
        t_final,
        caller="fractional_integrate",
        max_steps=native_step_limit,
    )

    # Guard: fractional backend is only defined for 0 < q < 1
    if q >= 1.0:
        raise ValueError(
            f"fractional_integrate: requires 0 < q < 1, got q={q}. "
            "For integer order use integrate_general."
        )
    if q <= 0.0:
        raise ValueError(f"fractional_integrate: q must be positive, got q={q}.")

    # Validate method and memory selectors before lowering them to C enums.
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    method_l = method.lower()
    if method_l == "efork_q1":
        raise ValueError(
            f"Integrator '{method}' is not valid for fractional dynamics (q<1). "
            "Use 'abm' or 'efork3'."
        )
    if method_l == "abm":
        meth_val = 0
    elif method_l in {"efork", "efork3"}:
        meth_val = 1
    else:
        raise ValueError(
            f"fractional_integrate: unknown method '{method}'. "
            "Must be 'abm' or 'efork3'."
        )

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
    bound_rhs = bind_rhs(rhs)

    x0_raw = np.asarray(x0, dtype=np.float64)
    if x0_raw.ndim != 1 or x0_raw.size == 0:
        raise ValueError("x0 must have shape (dim,) with dim >= 1.")
    if x0_raw.size > np.iinfo(np.int32).max:
        raise ValueError("x0 dimension exceeds the native INT32 range.")
    if not np.all(np.isfinite(x0_raw)):
        raise ValueError("x0 must contain only finite values.")
    x0_arr = np.ascontiguousarray(x0_raw, dtype=np.float64)
    dim = x0_arr.size

    # Memory parameters
    if not isinstance(memory_mode, str):
        raise TypeError("memory_mode must be a string.")
    memory_mode_l = memory_mode.lower()
    memory_mode_l, validated_window = validate_memory_policy(
        memory_mode_l,
        memory_window_length,
        caller="fractional_integrate",
    )
    if validated_window is not None and validated_window > np.iinfo(np.int32).max:
        raise ValueError("memory_window_length must not exceed INT32_MAX.")
    mem_val = 0 if memory_mode_l == "full" else 1
    win_len = 0 if validated_window is None else validated_window
    memory_mode = memory_mode_l
    method = method_l

    # Early-stop config parsing (supports both flat and nested formats)
    esc = early_stop_config if early_stop_config is not None else {}
    if not isinstance(esc, Mapping):
        raise TypeError("early_stop_config must be a mapping or None.")
    div_section = esc.get("divergence", {})
    eq_section = esc.get("equilibrium", {})
    if not isinstance(div_section, Mapping) or not isinstance(eq_section, Mapping):
        raise TypeError("early-stop divergence/equilibrium sections must be mappings.")
    es_enabled = _binary_flag(esc.get("enabled", True), "early_stop.enabled")

    div_enabled = _binary_flag(esc.get(
        "divergence_enabled",
        div_section.get("enabled", True),
    ), "early_stop.divergence.enabled")
    div_norm_esc = _finite_float(esc.get(
        "divergence_norm",
        div_section.get("norm", 80.0),
    ), "early_stop.divergence.norm", positive=True)
    div_consec = _strict_positive_int(esc.get(
        "divergence_consecutive_steps",
        div_section.get("consecutive_steps", 5),
    ), "early_stop.divergence.consecutive_steps")
    div_growth = _finite_float(esc.get(
        "divergence_growth_factor",
        div_section.get("growth_factor", 1.25),
    ), "early_stop.divergence.growth_factor", positive=True)

    eq_enabled = _binary_flag(esc.get(
        "equilibrium_enabled",
        eq_section.get("enabled", True),
    ), "early_stop.equilibrium.enabled")
    eq_tol = _finite_float(esc.get(
        "equilibrium_tol",
        eq_section.get("tol", 1e-3),
    ), "early_stop.equilibrium.tol", positive=True)
    eq_deriv = _finite_float(esc.get(
        "equilibrium_derivative_tol",
        eq_section.get("derivative_tol", 1e-4),
    ), "early_stop.equilibrium.derivative_tol", positive=True)
    eq_consec = _strict_positive_int(esc.get(
        "equilibrium_consecutive_steps",
        eq_section.get("consecutive_steps", 200),
    ), "early_stop.equilibrium.consecutive_steps")
    eq_min_t = _finite_float(esc.get(
        "equilibrium_min_time",
        eq_section.get("min_time", 5.0),
    ), "early_stop.equilibrium.min_time", non_negative=True)

    # Equilibria flat buffer
    validated_equilibria = validate_equilibria(
        equilibria,
        dim=dim,
        caller="fractional_integrate",
    )
    if validated_equilibria:
        eq_pts = np.ascontiguousarray(np.concatenate(validated_equilibria))
        num_eq = len(validated_equilibria)
        if num_eq > np.iinfo(np.int32).max:
            raise ValueError("number of equilibria exceeds the native INT32 range.")
    else:
        eq_pts = np.zeros(1, dtype=np.float64)   # non-empty sentinel for ctypes
        num_eq = 0

    # Pre-history normalisation
    validated_history_times, validated_history_states = validate_prehistory(
        history_times,
        history_states,
        x0=x0_arr,
        h=h,
        caller="fractional_integrate",
    )
    if validated_history_times is not None:
        assert validated_history_states is not None
        history_len = int(validated_history_times.size)
        if history_len > np.iinfo(np.int32).max:
            raise ValueError("history length exceeds the native INT32 range.")
        history_times_arr = validated_history_times
        history_states_flat = np.ascontiguousarray(
            validated_history_states.reshape(-1)
        )
        history_times_for_solver: Optional[np.ndarray] = history_times_arr
        history_states_for_solver: Optional[np.ndarray] = validated_history_states
    else:
        history_times_arr = np.zeros(1, dtype=np.float64)   # sentinel
        history_states_flat = np.zeros(dim, dtype=np.float64)  # sentinel
        history_times_for_solver = None
        history_states_for_solver = None
        history_len = 0

    H_eff = max(history_len, 1)
    total_capacity = H_eff + nsteps
    checked_array_capacity(
        (total_capacity,),
        np.float64,
        caller="fractional_integrate output times",
    )
    checked_array_capacity(
        (total_capacity, dim),
        np.float64,
        caller="fractional_integrate output states",
    )

    info: dict = {
        "method": method,
        "memory_mode": memory_mode,
        "memory_window_length": memory_window_length,
        "n_dim": dim,
        "history_len_in": history_len,
        "divergence_norm": normalized_divergence_norm,
        "requested_t_final": t_final,
        "effective_t_final": t_final,
        "allow_python_fallback": allow_python_fallback,
        "used_c_backend": False,
        "rhs_source": "python_native",
    }
    callback_failures: list[BaseException] = []

    # -----------------------------------------------------------------------
    # Attempt C backend
    # -----------------------------------------------------------------------
    if use_c_backend:
        try:
            backend = GeneralFractionalCBackend.get_instance()

            out_times = np.zeros(total_capacity, dtype=np.float64)
            out_states = np.zeros((total_capacity, dim), dtype=np.float64).reshape(-1)

            out_steps_c = ctypes.c_int(0)
            status_code_c = ctypes.c_int(0)

            # Retrieve native C RHS pointer if available in registry
            rhs_ptr, params_struct = get_c_rhs_and_params(system, backend.lib)

            if rhs_ptr is not None:
                # --- Registered pre-compiled C RHS (fastest path) ---
                c_rhs = backend.RHS_CALLBACK(rhs_ptr)
                c_params = ctypes.cast(
                    ctypes.byref(params_struct), ctypes.c_void_p
                )
                info["rhs_source"] = "compiled_c_registry"
            else:
                # --- Generic Python callback ---

                def _py_rhs_wrapper(
                    t_val: float,
                    x_ptr: "ctypes.POINTER(ctypes.c_double)",
                    dx_ptr: "ctypes.POINTER(ctypes.c_double)",
                    n_val: int,
                    params_val: int,
                    _rhs=bound_rhs,
                    _dim=dim,
                ) -> None:
                    x_arr = np.ctypeslib.as_array(x_ptr, shape=(_dim,))
                    dx_arr = np.ctypeslib.as_array(dx_ptr, shape=(_dim,))
                    if callback_failures:
                        dx_arr.fill(np.nan)
                        return
                    try:
                        deriv = validate_rhs_result(
                            _rhs(t_val, x_arr),
                            dim=_dim,
                            caller="fractional_integrate",
                        )
                        dx_arr[:] = deriv
                    except BaseException as exc:
                        callback_failures.append(exc)
                        dx_arr.fill(np.nan)

                c_rhs = backend.RHS_CALLBACK(_py_rhs_wrapper)
                c_params = ctypes.c_void_p(None)
                info["rhs_source"] = "python_callback_wrapped"

            rc = backend.lib.integrate_fractional_c(
                c_rhs,
                c_params,
                ctypes.c_int(dim),
                x0_arr,
                ctypes.c_double(q),
                ctypes.c_double(h),
                ctypes.c_double(t_final),
                ctypes.c_int(meth_val),
                ctypes.c_int(mem_val),
                ctypes.c_int(win_len),
                history_times_arr,
                history_states_flat,
                ctypes.c_int(history_len),
                ctypes.c_size_t(history_times_arr.size),
                ctypes.c_size_t(history_states_flat.size),
                ctypes.c_double(native_divergence_norm),
                out_times,
                out_states,
                ctypes.c_size_t(out_times.size),
                ctypes.c_size_t(out_states.size),
                ctypes.byref(out_steps_c),
                ctypes.byref(status_code_c),
                # Early-stopping
                ctypes.c_int(es_enabled),
                ctypes.c_int(div_enabled),
                ctypes.c_double(div_norm_esc),
                ctypes.c_int(div_consec),
                ctypes.c_double(div_growth),
                ctypes.c_int(eq_enabled),
                ctypes.c_double(eq_tol),
                ctypes.c_double(eq_deriv),
                ctypes.c_int(eq_consec),
                ctypes.c_double(eq_min_t),
                eq_pts,
                ctypes.c_int(num_eq),
                ctypes.c_size_t(eq_pts.size if num_eq else 0),
            )

            if callback_failures:
                raise callback_failures[0]

            if rc < 0:
                raise RuntimeError(
                    f"integrate_fractional_c returned error code {rc}."
                )

            actual_samples = out_steps_c.value
            if actual_samples < 1 or actual_samples > total_capacity:
                raise RuntimeError(
                    "integrate_fractional_c returned an invalid sample count "
                    f"{actual_samples} for capacity {total_capacity}."
                )
            times = out_times[:actual_samples]
            states = out_states[: actual_samples * dim].reshape(actual_samples, dim)
            times = _enforce_effective_horizon(times, t_final)

            # Map C status codes to string labels
            _STATUS_MAP = {
                0: "ok",
                1: "diverged",
                2: "nonfinite_solution",
                3: "diverged_early",
                4: "converged_equilibrium_early",
            }
            status = _STATUS_MAP.get(status_code_c.value, f"unknown_{status_code_c.value}")

            physical_start = _physical_trajectory_start(times)
            start_slice = 0 if return_history else physical_start
            completed_steps = max(actual_samples - physical_start - 1, 0)
            info.update({
                "used_c_backend": True,
                "rhs_source": info["rhs_source"],
                "n_steps": completed_steps,
                "n_steps_completed": completed_steps,
                "n_samples": actual_samples,
                "n_samples_returned": actual_samples - start_slice,
                "effective_t_final": float(times[-1]),
                "status_code": status_code_c.value,
                "truncated_memory": (memory_mode == "window"),
            })

            return times[start_slice:], states[start_slice:], status, info

        except Exception as exc:
            if callback_failures:
                raise callback_failures[0]
            if not allow_python_fallback:
                raise RuntimeError(
                    f"C backend failed and allow_python_fallback=False. "
                    f"Original error: {exc}"
                ) from exc
            # Fall through to Python solvers
            info["c_backend_error"] = str(exc)
            warnings.warn(
                "The native fractional backend failed; using the requested "
                f"Python fallback. Native error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    # -----------------------------------------------------------------------
    # Python fallback solvers
    # -----------------------------------------------------------------------
    info["used_c_backend"] = False
    info["rhs_source"] = "python_native"
    info["truncated_memory"] = (memory_mode == "window")

    if method_l == "abm":
        from .abm import _python_abm_integrate

        t_arr, x_arr, status = _python_abm_integrate(
            bound_rhs, x0_arr, q=q, h=h, t_final=t_final,
            divergence_norm=normalized_divergence_norm,
            history_times=history_times_for_solver,
            history_states=history_states_for_solver,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            early_stop_config=early_stop_config,
            equilibria=validated_equilibria,
        )
    else:
        from .efork import _python_efork3_integrate

        def _rhs_t(t_val: float, x_val: np.ndarray) -> np.ndarray:
            return np.asarray(bound_rhs(t_val, x_val), dtype=float)

        t_arr, x_arr, status = _python_efork3_integrate(
            rhs=_rhs_t,
            x0=x0_arr,
            q=q,
            h=h,
            t_final=t_final,
            divergence_norm=normalized_divergence_norm,
            history_times=history_times_for_solver,
            history_states=history_states_for_solver,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            early_stop_config=early_stop_config,
            equilibria=validated_equilibria,
        )

    if len(t_arr) < 1:
        raise RuntimeError("fractional integrator returned no time samples.")
    # The Python EFORK reference returns only the physical 0..T segment,
    # whereas ABM and the C ABI retain prehistory internally.  Normalize both
    # representations here before applying the public ``return_history`` view.
    if history_len > 1:
        time_scale = max(1.0, abs(float(t_arr[0])))
        zero_tolerance = 64.0 * float(np.finfo(np.float64).eps) * time_scale
        if float(t_arr[0]) >= -zero_tolerance:
            assert validated_history_states is not None
            t_arr = np.concatenate((history_times_arr[:-1], t_arr))
            x_arr = np.vstack((validated_history_states[:-1], x_arr))
    actual_samples = len(t_arr)
    t_arr = _enforce_effective_horizon(t_arr, t_final)
    physical_start = _physical_trajectory_start(t_arr)
    start_slice = 0 if return_history else physical_start
    completed_steps = max(actual_samples - physical_start - 1, 0)
    info.update({
        "n_steps": completed_steps,
        "n_steps_completed": completed_steps,
        "n_samples": actual_samples,
        "n_samples_returned": actual_samples - start_slice,
        "effective_t_final": float(t_arr[-1]),
    })
    return t_arr[start_slice:], x_arr[start_slice:], status, info
