import ctypes
import os
import sys
from pathlib import Path
import numpy as np
from typing import Any, Tuple, Optional, List

from hidden_attractors.parallel import compile_c_target
from hidden_attractors.paths import NATIVE_CACHE
from ..native.rhs_registry import get_c_rhs_and_params

def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"

class GeneralFractionalCBackend:
    """Wrapper for compiled generic C fractional integrator."""
    
    lib: Any = None
    _cache = {}
    
    @classmethod
    def get_instance(cls) -> "GeneralFractionalCBackend":
        if "instance" in cls._cache:
            return cls._cache["instance"]
            
        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
        src_path = Path(__file__).resolve().parent.parent / "native" / "csrc" / "fractional_integrators.c"
        out_path = NATIVE_CACHE / f"fractional_integrators{_shared_suffix()}"
        
        # Compile on-the-fly using version_2 parallel policy
        result = compile_c_target(
            src_path,
            out_path,
            target_kind="shared",
            openmp=False
        )
        
        lib = ctypes.CDLL(str(result.path.resolve()))
        
        # Define callback signature: RhsCallback(double t, const double *x, double *dx, int n, void *params)
        cls.RHS_CALLBACK = ctypes.CFUNCTYPE(
            None,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_void_p
        )
        
        # Declare argument and return types for integrate_fractional_c
        lib.integrate_fractional_c.argtypes = [
            cls.RHS_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            
            # Early stopping arguments
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int
        ]
        lib.integrate_fractional_c.restype = ctypes.c_int
        
        backend = cls()
        backend.lib = lib
        cls._cache["instance"] = backend
        return backend

def fractional_integrate(
    rhs: Any,
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    method: str,                   # "abm" or "efork"
    memory_mode: str,              # "full" or "window"
    memory_window_length: Optional[int] = None,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    system: Optional[Any] = None,
    params: Optional[Any] = None,
    use_c_backend: bool = True,
    divergence_norm: float = 120.0,
    return_history: bool = False,
    allow_python_fallback: bool = False,
    
    # Early stopping config
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, str, dict]:
    """Unified generic integration interface for fractional differential equations with early stopping support."""
    x0_arr = np.asarray(x0, dtype=np.float64)
    dim = x0_arr.size
    
    # 1. Normalize parameters and method identifiers
    if not callable(rhs):
        raise TypeError("rhs must be a callable function")
        
    meth_val = 0 if method.lower() == "abm" else 1
    mem_val = 0 if memory_mode.lower() == "full" else 1
    win_len = int(memory_window_length) if memory_window_length is not None else 0
    
    # Parse early stopping config
    esc = early_stop_config if early_stop_config is not None else {}
    es_enabled = int(esc.get("enabled", True))
    
    # Support both flat and nested fields for early stop config
    div_enabled = int(esc.get("divergence_enabled", esc.get("divergence", {}).get("enabled", True)))
    div_norm = float(esc.get("divergence_norm", esc.get("divergence", {}).get("norm", 80.0)))
    div_consec = int(esc.get("divergence_consecutive_steps", esc.get("divergence", {}).get("consecutive_steps", 5)))
    div_growth = float(esc.get("divergence_growth_factor", esc.get("divergence", {}).get("growth_factor", 1.25)))
    
    eq_enabled = int(esc.get("equilibrium_enabled", esc.get("equilibrium", {}).get("enabled", True)))
    eq_t = float(esc.get("equilibrium_tol", esc.get("equilibrium", {}).get("tol", 1e-3)))
    eq_deriv = float(esc.get("equilibrium_derivative_tol", esc.get("equilibrium", {}).get("derivative_tol", 1e-4)))
    eq_consec = int(esc.get("equilibrium_consecutive_steps", esc.get("equilibrium", {}).get("consecutive_steps", 200)))
    eq_min_t = float(esc.get("equilibrium_min_time", esc.get("equilibrium", {}).get("min_time", 5.0)))
    
    if equilibria is not None and len(equilibria) > 0:
        eq_pts = np.ascontiguousarray(np.concatenate([np.asarray(eq, dtype=np.float64) for eq in equilibria]), dtype=np.float64)
        num_eq = len(equilibria)
    else:
        eq_pts = np.empty(0, dtype=np.float64)
        num_eq = 0
        
    # Handle prehistory normalization
    if history_times is not None and history_states is not None:
        history_times_arr = np.ascontiguousarray(history_times, dtype=np.float64)
        history_states_arr = np.ascontiguousarray(history_states, dtype=np.float64).reshape(-1, dim)
        history_len = len(history_times_arr)
    else:
        history_times_arr = np.empty(0, dtype=np.float64)
        history_states_arr = np.empty((0, dim), dtype=np.float64)
        history_len = 0
        
    H_eff = history_len if history_len > 0 else 1
    nsteps = int(np.ceil(t_final / h))
    total_capacity = H_eff + nsteps
    
    used_c_backend = False
    status = "ok"
    status_code_c = ctypes.c_int(0)
    out_steps_c = ctypes.c_int(0)
    
    info = {
        "method": method,
        "memory_mode": memory_mode,
        "memory_window_length": memory_window_length,
        "n_dim": dim,
        "history_len_in": history_len,
        "divergence_norm": divergence_norm,
        "allow_python_fallback": allow_python_fallback
    }

    # 2. Execute C Backend if requested
    if use_c_backend:
        try:
            backend = GeneralFractionalCBackend.get_instance()
            
            # Out buffers
            out_times = np.zeros(total_capacity, dtype=np.float64)
            out_states = np.zeros(total_capacity * dim, dtype=np.float64)
            
            # Fetch registered native RHS function address and params structure if available
            rhs_ptr, params_struct = get_c_rhs_and_params(system, backend.lib)
            
            if rhs_ptr is not None:
                # Registered pre-compiled C RHS
                c_rhs = backend.RHS_CALLBACK(rhs_ptr)
                c_params = ctypes.cast(ctypes.byref(params_struct), ctypes.c_void_p)
                info["rhs_source"] = "compiled_c_registry"
            else:
                # General Python callback RHS wrapped for C CDLL
                def py_rhs_wrapper(t_val, x_ptr, dx_ptr, n_val, params_val):
                    x_arr = np.ctypeslib.as_array(x_ptr, shape=(dim,))
                    dx_arr = np.ctypeslib.as_array(dx_ptr, shape=(dim,))
                    try:
                        deriv = np.asarray(rhs(t_val, x_arr), dtype=np.float64)
                    except TypeError:
                        deriv = np.asarray(rhs(x_arr), dtype=np.float64)
                    for d in range(dim):
                        dx_arr[d] = deriv[d]
                        
                c_rhs = backend.RHS_CALLBACK(py_rhs_wrapper)
                c_params = None
                info["rhs_source"] = "python_callback_wrapped"
                
            rc = backend.lib.integrate_fractional_c(
                c_rhs,
                c_params,
                dim,
                x0_arr,
                float(q),
                float(h),
                float(t_final),
                meth_val,
                mem_val,
                win_len,
                history_times_arr,
                history_states_arr,
                history_len,
                float(divergence_norm),
                out_times,
                out_states,
                ctypes.byref(out_steps_c),
                ctypes.byref(status_code_c),
                
                # New early stop parameters
                es_enabled,
                div_enabled,
                div_norm,
                div_consec,
                div_growth,
                eq_enabled,
                eq_t,
                eq_deriv,
                eq_consec,
                eq_min_t,
                eq_pts,
                num_eq
            )
            
            if rc < 0:
                raise RuntimeError(f"C integrate_fractional_c returned error code {rc}")
                
            actual_steps = out_steps_c.value
            times = out_times[:actual_steps]
            states = out_states[:actual_steps * dim].reshape(-1, dim)
            
            # Interpret status code
            if status_code_c.value == 1:
                status = "diverged"
            elif status_code_c.value == 2:
                status = "nonfinite_solution"
            elif status_code_c.value == 3:
                status = "diverged_early"
            elif status_code_c.value == 4:
                status = "converged_equilibrium_early"
            else:
                status = "ok"
                
            used_c_backend = True
            info["used_c_backend"] = True
            info["n_steps"] = actual_steps
            info["status_code"] = status_code_c.value
            info["truncated_memory"] = (memory_mode == "window")
            
            # Return either complete history + integrated run, or integrated run only
            if return_history:
                return times, states, status, info
            else:
                start_slice = history_len if history_len > 0 else 0
                return times[start_slice:], states[start_slice:], status, info
                
        except Exception as exc:
            if not allow_python_fallback:
                raise RuntimeError(f"Failed executing fractional C backend: {exc}")
                
    # 3. Fallback Python Solvers (if C failed and fallback is allowed, or C not requested)
    info["used_c_backend"] = False
    info["rhs_source"] = "python_native"
    info["truncated_memory"] = (memory_mode == "window")
    
    # Standard Python Fallbacks
    if method.lower() == "abm":
        from .abm import _python_abm_integrate
        # Wrapper to handle t-dependent rhs
        def rhs_dep(x):
            try:
                return rhs(0.0, x)
            except TypeError:
                return rhs(x)
                
        t_arr, x_arr, status = _python_abm_integrate(
            rhs_dep, x0_arr, q=q, h=h, t_final=t_final,
            divergence_norm=divergence_norm,
            history_times=history_times,
            history_states=history_states,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            early_stop_config=early_stop_config,
            equilibria=equilibria
        )
        info["n_steps"] = len(t_arr)
        if return_history:
            return t_arr, x_arr, status, info
        else:
            start_slice = history_len if history_len > 0 else 0
            return t_arr[start_slice:], x_arr[start_slice:], status, info
    else:
        # Python EFORK
        if memory_mode == "window":
            raise ValueError("Python fallback does not support EFORK + windowed memory. C backend is required.")
        else:
            from hidden_attractors.solvers.efork_published import efork3_caputo_integrate
            def rhs_t(t_val, x_val):
                try:
                    return rhs(t_val, x_val)
                except TypeError:
                    return rhs(x_val)
            try:
                t_arr, x_arr = efork3_caputo_integrate(rhs_t, x0_arr, alpha=q, h=h, t_final=t_final)
                
                # Perform post-integration analysis or early stopping emulation
                status = "ok"
                last_idx = len(t_arr)
                
                # Emulate step-by-step checks
                prev_norm = -1.0
                div_consec_count = 0
                growth_consec_count = 0
                
                eq_consec_counts = [0] * len(equilibria) if equilibria else []
                
                for idx in range(len(t_arr)):
                    state = x_arr[idx]
                    norm = np.linalg.norm(state)
                    
                    if not np.isfinite(norm):
                        status = "nonfinite_solution"
                        last_idx = idx + 1
                        break
                        
                    # Divergence Check
                    if divergence_norm is not None and norm > divergence_norm:
                        status = "diverged"
                        last_idx = idx + 1
                        break
                        
                    # Early stop checks
                    if es_enabled:
                        # 1. Divergence checks
                        if div_enabled:
                            if norm > div_norm:
                                div_consec_count += 1
                            else:
                                div_consec_count = 0
                                
                            if prev_norm >= 0.0:
                                if norm > div_growth * prev_norm:
                                    growth_consec_count += 1
                                else:
                                    growth_consec_count = 0
                            prev_norm = norm
                            
                            if div_consec_count >= div_consec or growth_consec_count >= div_consec:
                                status = "diverged_early"
                                last_idx = idx + 1
                                break
                        else:
                            prev_norm = norm
                            
                        # 2. Equilibrium checks
                        if eq_enabled and equilibria and t_arr[idx] >= eq_min_t:
                            converged_eq_idx = -1
                            for k, eq in enumerate(equilibria):
                                diff_norm = np.linalg.norm(state - eq)
                                deriv_norm = np.linalg.norm(rhs_t(t_arr[idx], state))
                                
                                if diff_norm < eq_t and deriv_norm < eq_deriv:
                                    eq_consec_counts[k] += 1
                                else:
                                    eq_consec_counts[k] = 0
                                    
                                if eq_consec_counts[k] >= eq_consec:
                                    converged_eq_idx = k
                                    break
                            if converged_eq_idx != -1:
                                status = "converged_equilibrium_early"
                                last_idx = idx + 1
                                break
                                
                info["n_steps"] = last_idx
                return t_arr[:last_idx], x_arr[:last_idx], status, info
            except Exception as exc:
                raise RuntimeError(f"Python EFORK fallback failed: {exc}")
