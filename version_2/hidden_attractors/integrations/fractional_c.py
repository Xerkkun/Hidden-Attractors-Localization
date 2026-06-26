"""Unified C-backed fractional integrator interface.

Architecture
------------
* :class:`GeneralFractionalCBackend` is a **singleton** that lazily compiles
  the native C library (``fractional_integrators.c``) and exposes the
  ``integrate_fractional_c`` entry point via :mod:`ctypes`.
* :func:`fractional_integrate` is the public API.  It attempts the C backend
  first; if that fails *and* ``allow_python_fallback=True`` it silently
  falls back to the pure-Python ABM or EFORK-3 solvers.

Path injection
--------------
The ``hidden_attractors`` package lives inside ``version_2/`` in the
workspace root.  The root-level ``hidden_attractors/`` folder contains only
the ``native/`` sub-directory; all Python modules (``parallel``, ``paths``,
etc.) reside in ``version_2/hidden_attractors/``.  We inject ``version_2``
at the front of ``sys.path`` so the correct package wins regardless of the
working directory.
"""

import ctypes
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# sys.path injection — must happen before any hidden_attractors import
# ---------------------------------------------------------------------------
_file_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.dirname(os.path.dirname(_file_dir))
_version_2_dir = os.path.join(_workspace_root, "version_2")
if _version_2_dir not in sys.path:
    sys.path.insert(0, _version_2_dir)

from hidden_attractors.parallel import compile_c_target  # noqa: E402
from hidden_attractors.paths import NATIVE_CACHE          # noqa: E402
from ..native.rhs_registry import get_c_rhs_and_params   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eval_rhs(rhs: Callable, t: float, x: np.ndarray) -> np.ndarray:
    """Evaluate ``rhs(t, x)`` or fall back to ``rhs(x)`` for legacy callables.

    Used throughout the Python fallback paths so that non-autonomous RHS
    functions receive the correct current time instead of a frozen ``t=0.0``.
    """
    try:
        return np.asarray(rhs(t, x), dtype=float)
    except TypeError:
        return np.asarray(rhs(x), dtype=float)


def _shared_suffix() -> str:
    """Return the platform-appropriate shared-library extension."""
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


# ---------------------------------------------------------------------------
# Singleton C backend
# ---------------------------------------------------------------------------

class GeneralFractionalCBackend:
    """Singleton wrapper for the compiled generic C fractional integrator.

    Call :meth:`get_instance` to obtain (and lazily initialise) the singleton.
    """

    # Class-level singleton cache — never store instance on the instance itself
    _instance: Optional["GeneralFractionalCBackend"] = None

    # Ctypes callback type — set during initialisation
    RHS_CALLBACK: Any = None

    def __init__(self) -> None:
        self.lib: Any = None

    @classmethod
    def get_instance(cls) -> "GeneralFractionalCBackend":
        """Return the singleton, compiling the C library on first call."""
        if cls._instance is not None:
            return cls._instance

        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)

        src_path = (
            Path(__file__).resolve().parent.parent
            / "native" / "csrc" / "fractional_integrators.c"
        )
        header_path = src_path.with_suffix(".h")
        source_fingerprint = hashlib.sha256(
            src_path.read_bytes() + header_path.read_bytes()
        ).hexdigest()[:12]
        out_path = NATIVE_CACHE / (
            f"fractional_integrators_{source_fingerprint}{_shared_suffix()}"
        )

        result = compile_c_target(
            src_path,
            out_path,
            target_kind="shared",
            openmp=False,
        )

        lib = ctypes.CDLL(str(result.path.resolve()))

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

        # ------------------------------------------------------------------
        # integrate_fractional_c argument types (must match the C header)
        # ------------------------------------------------------------------
        lib.integrate_fractional_c.argtypes = [
            cls.RHS_CALLBACK,                                             # rhs
            ctypes.c_void_p,                                              # params
            ctypes.c_int,                                                 # dim
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags="C_CONTIGUOUS"),                 # x0
            ctypes.c_double,                                              # q
            ctypes.c_double,                                              # h
            ctypes.c_double,                                              # t_final
            ctypes.c_int,                                                 # method
            ctypes.c_int,                                                 # memory_mode
            ctypes.c_int,                                                 # memory_window_length
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags="C_CONTIGUOUS"),                 # history_times
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags="C_CONTIGUOUS"),                 # history_states (flat)
            ctypes.c_int,                                                 # history_len
            ctypes.c_double,                                              # divergence_norm
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags="C_CONTIGUOUS"),                 # out_times
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags="C_CONTIGUOUS"),                 # out_states (flat)
            ctypes.POINTER(ctypes.c_int),                                 # out_steps
            ctypes.POINTER(ctypes.c_int),                                 # status_code
            # Early-stopping parameters
            ctypes.c_int,    # early_stop_enabled
            ctypes.c_int,    # div_early_enabled
            ctypes.c_double, # div_early_norm
            ctypes.c_int,    # div_consec_steps
            ctypes.c_double, # div_growth_factor
            ctypes.c_int,    # eq_early_enabled
            ctypes.c_double, # eq_tol
            ctypes.c_double, # eq_deriv_tol
            ctypes.c_int,    # eq_consec_steps
            ctypes.c_double, # eq_min_time
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags="C_CONTIGUOUS"),                 # equilibria_pts
            ctypes.c_int,    # num_equilibria
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
    divergence_norm: float = 120.0,
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
        Integration method.  ``"heun"`` and ``"efork_q1"`` are rejected.
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
    divergence_norm : float
        Halt integration when ``‖x‖ > divergence_norm``.
    return_history : bool
        Include the pre-history segment in the returned arrays.
    allow_python_fallback : bool
        If True and the C backend fails, silently switch to the Python solver.
    early_stop_config : dict or None
        Early-stopping configuration dictionary.
    equilibria : list of ndarray or None
        Equilibrium points used for convergence early-stopping.

    Returns
    -------
    times : ndarray, shape (N,)
    states : ndarray, shape (N, dim)
    status : str  — "ok" | "diverged" | "diverged_early" | "converged_equilibrium_early" | …
    info : dict   — backend metadata
    """
    q = float(q)

    # Guard: fractional backend is only defined for 0 < q < 1
    if q >= 1.0:
        raise ValueError(
            f"fractional_integrate: requires 0 < q < 1, got q={q}. "
            "For integer order use integrate_general."
        )
    if q <= 0.0:
        raise ValueError(f"fractional_integrate: q must be positive, got q={q}.")

    # Validate method
    method_l = method.lower()
    if method_l in {"heun", "efork_q1"}:
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

    x0_arr = np.ascontiguousarray(x0, dtype=np.float64)
    dim = x0_arr.size

    # Memory parameters
    mem_val = 0 if memory_mode.lower() == "full" else 1
    win_len = int(memory_window_length) if memory_window_length is not None else 0

    # Early-stop config parsing (supports both flat and nested formats)
    esc = early_stop_config if early_stop_config is not None else {}
    es_enabled = int(esc.get("enabled", True))

    div_enabled = int(esc.get(
        "divergence_enabled",
        esc.get("divergence", {}).get("enabled", True),
    ))
    div_norm_esc = float(esc.get(
        "divergence_norm",
        esc.get("divergence", {}).get("norm", 80.0),
    ))
    div_consec = int(esc.get(
        "divergence_consecutive_steps",
        esc.get("divergence", {}).get("consecutive_steps", 5),
    ))
    div_growth = float(esc.get(
        "divergence_growth_factor",
        esc.get("divergence", {}).get("growth_factor", 1.25),
    ))

    eq_enabled = int(esc.get(
        "equilibrium_enabled",
        esc.get("equilibrium", {}).get("enabled", True),
    ))
    eq_tol = float(esc.get(
        "equilibrium_tol",
        esc.get("equilibrium", {}).get("tol", 1e-3),
    ))
    eq_deriv = float(esc.get(
        "equilibrium_derivative_tol",
        esc.get("equilibrium", {}).get("derivative_tol", 1e-4),
    ))
    eq_consec = int(esc.get(
        "equilibrium_consecutive_steps",
        esc.get("equilibrium", {}).get("consecutive_steps", 200),
    ))
    eq_min_t = float(esc.get(
        "equilibrium_min_time",
        esc.get("equilibrium", {}).get("min_time", 5.0),
    ))

    # Equilibria flat buffer
    if equilibria is not None and len(equilibria) > 0:
        eq_pts = np.ascontiguousarray(
            np.concatenate([np.asarray(eq, dtype=np.float64).ravel()
                            for eq in equilibria]),
            dtype=np.float64,
        )
        num_eq = len(equilibria)
    else:
        eq_pts = np.empty(1, dtype=np.float64)   # non-empty sentinel for ctypes
        num_eq = 0

    # Pre-history normalisation
    if history_times is not None and history_states is not None and len(history_times) > 0:
        history_times_arr = np.ascontiguousarray(history_times, dtype=np.float64)
        # history_states may be 2-D (N, dim) or 1-D (N*dim,); normalise to 1-D for C
        hs = np.asarray(history_states, dtype=np.float64)
        history_states_flat = np.ascontiguousarray(hs.reshape(-1), dtype=np.float64)
        history_len = len(history_times_arr)
    else:
        history_times_arr = np.empty(1, dtype=np.float64)   # sentinel
        history_states_flat = np.empty(dim, dtype=np.float64)  # sentinel
        history_len = 0

    nsteps = int(np.ceil(t_final / h))
    H_eff = max(history_len, 1)
    total_capacity = H_eff + nsteps + 1

    info: dict = {
        "method": method,
        "memory_mode": memory_mode,
        "memory_window_length": memory_window_length,
        "n_dim": dim,
        "history_len_in": history_len,
        "divergence_norm": divergence_norm,
        "allow_python_fallback": allow_python_fallback,
        "used_c_backend": False,
        "rhs_source": "python_native",
    }

    # -----------------------------------------------------------------------
    # Attempt C backend
    # -----------------------------------------------------------------------
    if use_c_backend:
        try:
            backend = GeneralFractionalCBackend.get_instance()

            out_times = np.zeros(total_capacity, dtype=np.float64)
            out_states = np.zeros(total_capacity * dim, dtype=np.float64)

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
                    _rhs=rhs,
                    _dim=dim,
                ) -> None:
                    x_arr = np.ctypeslib.as_array(x_ptr, shape=(_dim,))
                    dx_arr = np.ctypeslib.as_array(dx_ptr, shape=(_dim,))
                    try:
                        deriv = np.asarray(_rhs(t_val, x_arr), dtype=np.float64)
                    except TypeError:
                        deriv = np.asarray(_rhs(x_arr), dtype=np.float64)
                    dx_arr[:] = deriv[:_dim]

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
                ctypes.c_double(divergence_norm),
                out_times,
                out_states,
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
            )

            if rc < 0:
                raise RuntimeError(
                    f"integrate_fractional_c returned error code {rc}."
                )

            actual_steps = out_steps_c.value
            times = out_times[:actual_steps]
            states = out_states[: actual_steps * dim].reshape(actual_steps, dim)

            # Map C status codes to string labels
            _STATUS_MAP = {
                0: "ok",
                1: "diverged",
                2: "nonfinite_solution",
                3: "diverged_early",
                4: "converged_equilibrium_early",
            }
            status = _STATUS_MAP.get(status_code_c.value, f"unknown_{status_code_c.value}")

            info.update({
                "used_c_backend": True,
                "rhs_source": info["rhs_source"],
                "n_steps": actual_steps,
                "status_code": status_code_c.value,
                "truncated_memory": (memory_mode == "window"),
            })

            start_slice = history_len if (return_history is False and history_len > 0) else 0
            return times[start_slice:], states[start_slice:], status, info

        except Exception as exc:
            if not allow_python_fallback:
                raise RuntimeError(
                    f"C backend failed and allow_python_fallback=False. "
                    f"Original error: {exc}"
                ) from exc
            # Fall through to Python solvers
            info["c_backend_error"] = str(exc)

    # -----------------------------------------------------------------------
    # Python fallback solvers
    # -----------------------------------------------------------------------
    info["used_c_backend"] = False
    info["rhs_source"] = "python_native"
    info["truncated_memory"] = (memory_mode == "window")

    if method_l == "abm":
        from .abm import _python_abm_integrate

        t_arr, x_arr, status = _python_abm_integrate(
            rhs, x0_arr, q=q, h=h, t_final=t_final,
            divergence_norm=divergence_norm,
            history_times=history_times,
            history_states=history_states,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            early_stop_config=early_stop_config,
            equilibria=equilibria,
        )
    else:
        from .efork import _python_efork3_integrate

        def _rhs_t(t_val: float, x_val: np.ndarray) -> np.ndarray:
            return eval_rhs(rhs, t_val, x_val)

        t_arr, x_arr, status = _python_efork3_integrate(
            rhs=_rhs_t,
            x0=x0_arr,
            q=q,
            h=h,
            t_final=t_final,
            divergence_norm=divergence_norm,
            history_times=history_times,
            history_states=history_states,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            early_stop_config=early_stop_config,
            equilibria=equilibria,
        )

    info["n_steps"] = len(t_arr)
    start_slice = history_len if (return_history is False and history_len > 0) else 0
    return t_arr[start_slice:], x_arr[start_slice:], status, info
