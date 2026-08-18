"""Native C backend wrappers.

These wrappers centralize ctypes signatures and compilation policy.  They do
not introduce a new numerical method; they expose the existing C EFORK and basin
classifiers behind a Python API suitable for experiments and examples.
"""

from __future__ import annotations

import ctypes
import csv
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .._rhs import bind_rhs
from ..models.chua import ChuaParameters, chua_nonsmooth_parameters
from ..parallel import compile_c_target, load_ctypes_library
from ..paths import PACKAGE_ROOT, get_native_cache
from .._time_grid import checked_array_capacity, exact_fixed_step_count
from .contracts import FractionalLyapunovRequest, FractionalLyapunovResult


C_SOURCE_ROOT = PACKAGE_ROOT / "native" / "csrc"
_C_INT_MAX = int(np.iinfo(np.int32).max)
_NATIVE_STEP_LIMIT = _C_INT_MAX - 2
_LIBRARY_LOCKS_GUARD = threading.Lock()
_LIBRARY_LOCKS: dict[Path, threading.RLock] = {}


def _library_transaction_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _LIBRARY_LOCKS_GUARD:
        return _LIBRARY_LOCKS.setdefault(resolved, threading.RLock())


def _finite_state(values: Sequence[float], dimension: int, name: str) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64)
    if state.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(state)


def _finite_scalar(value: float, name: str, *, positive: bool = False,
                   non_negative: bool = False) -> float:
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if non_negative and normalized < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized


def _bounded_positive_ceil_count(value: float, step: float, *, caller: str) -> int:
    ratio = value / step
    if not math.isfinite(ratio):
        raise ValueError(f"{caller}: value/h must be finite.")
    count = math.ceil(ratio)
    if count < 1 or count > _NATIVE_STEP_LIMIT:
        raise ValueError(
            f"{caller}: value/h exceeds the supported native C int range."
        )
    return count


def _checked_empty(
    shape: tuple[int, ...], dtype: np.dtype | type, *, caller: str
) -> np.ndarray:
    checked_array_capacity(shape, dtype, caller=caller)
    return np.empty(shape, dtype=dtype)

_FRACTIONAL_SYSTEM_IDS = {"rabinovich_fabrikant": 1, "lorenz": 2}
_FRACTIONAL_CONTRACT_IDS = {
    "dk2018_block_restart_abm_gs": 1,
    "fixed_lower_limit_full_history_qr": 2,
}
_FRACTIONAL_CONVOLUTION_IDS = {"direct": 1, "fft_block": 2}
_FRACTIONAL_STATUS = {
    0: "ok",
    -1: "invalid_request",
    -2: "allocation_failed",
    -3: "nonfinite_solution",
    -4: "diverged",
    -5: "output_buffer_too_small",
}


class _CFractionalLyapunovRequest(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("system_id", ctypes.c_int),
        ("execution_contract", ctypes.c_int),
        ("convolution_mode", ctypes.c_int),
        ("fft_block_size", ctypes.c_int),
        ("q", ctypes.c_double),
        ("h", ctypes.c_double),
        ("t_final", ctypes.c_double),
        ("t_burn", ctypes.c_double),
        ("reorthonormalization_time", ctypes.c_double),
        ("divergence_norm", ctypes.c_double),
        ("x0", ctypes.c_double * 4),
        ("parameters", ctypes.c_double * 8),
    ]


class _CFractionalLyapunovResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("status_code", ctypes.c_int),
        ("steps_completed", ctypes.c_int),
        ("convergence_rows", ctypes.c_int),
        ("exponents", ctypes.c_double * 4),
        ("final_state", ctypes.c_double * 4),
    ]


@dataclass
class NativeFractionalVariationalBackend:
    """Native C backend for extensive fractional variational LE calculations."""

    lib: Any
    build_metadata: dict[str, object]
    _cache = {}
    _cache_lock = threading.RLock()

    @classmethod
    def build(cls, output_name: str = "fractional_variational_lyapunov") -> "NativeFractionalVariationalBackend":
        native_cache = get_native_cache()
        result, lib = load_ctypes_library(
            C_SOURCE_ROOT / "fractional_variational_lyapunov_lib.c",
            native_cache / f"{output_name}{_shared_suffix()}",
            openmp=False,
            expected_symbols=(
                "fractional_lyapunov_abi_version",
                "fractional_lyapunov_rhs_jacobian",
                "fractional_lyapunov_run",
            ),
            expected_abi_version=1,
            abi_version_symbol="fractional_lyapunov_abi_version",
        )
        cache_key = str(result.path.resolve())
        with cls._cache_lock:
            if cache_key in cls._cache:
                return cls._cache[cache_key]
        lib.fractional_lyapunov_abi_version.argtypes = []
        lib.fractional_lyapunov_abi_version.restype = ctypes.c_int
        lib.fractional_lyapunov_rhs_jacobian.argtypes = [
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        lib.fractional_lyapunov_rhs_jacobian.restype = ctypes.c_int
        lib.fractional_lyapunov_run.argtypes = [
            ctypes.POINTER(_CFractionalLyapunovRequest),
            ctypes.POINTER(_CFractionalLyapunovResult),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
        ]
        lib.fractional_lyapunov_run.restype = ctypes.c_int
        if int(lib.fractional_lyapunov_abi_version()) != 1:
            raise RuntimeError("Unsupported fractional Lyapunov native ABI.")
        backend = cls(
            lib=lib,
            build_metadata={
                "compiler": result.compiler,
                "compile_command": list(result.command),
                "openmp_requested": result.openmp_requested,
                "openmp_active": result.openmp_active,
                "target_kind": result.target_kind,
            },
        )
        with cls._cache_lock:
            if cache_key in cls._cache:
                return cls._cache[cache_key]
            cls._cache[cache_key] = backend
        return backend

    @staticmethod
    def _parameter_vector(system_id: str, parameters: dict[str, float] | Any) -> np.ndarray:
        if system_id == "rabinovich_fabrikant":
            return np.asarray([parameters["a"], parameters["b"]], dtype=np.float64)
        if system_id == "lorenz":
            return np.asarray([parameters["sigma"], parameters["beta"], parameters["rho"]], dtype=np.float64)
        raise ValueError(f"Unsupported native fractional Lyapunov system: {system_id}")

    def rhs_jacobian(
        self,
        system_id: str,
        parameters: dict[str, float],
        state: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        params = np.zeros(8, dtype=np.float64)
        values = self._parameter_vector(system_id, parameters)
        params[: values.size] = values
        x = np.ascontiguousarray(state, dtype=np.float64)
        if x.shape != (3,):
            raise ValueError("Native fractional Lyapunov systems currently require a 3D state.")
        rhs = np.empty(3, dtype=np.float64)
        jacobian = np.empty(9, dtype=np.float64)
        rc = int(self.lib.fractional_lyapunov_rhs_jacobian(_FRACTIONAL_SYSTEM_IDS[system_id], params, x, rhs, jacobian))
        if rc != 0:
            raise RuntimeError(f"Native RHS/Jacobian evaluation failed with status {rc}.")
        return rhs, jacobian.reshape(3, 3)

    def run(self, request: FractionalLyapunovRequest) -> FractionalLyapunovResult:
        try:
            system_value = _FRACTIONAL_SYSTEM_IDS[request.system_id]
            contract_value = _FRACTIONAL_CONTRACT_IDS[request.execution_contract]
            convolution_value = _FRACTIONAL_CONVOLUTION_IDS[request.convolution_mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported native fractional Lyapunov selector: {exc.args[0]}") from exc

        x0 = _finite_state(request.x0, 3, "x0")
        params = np.ascontiguousarray(
            self._parameter_vector(request.system_id, request.parameters),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(params)):
            raise ValueError("parameters must contain only finite values.")
        q = _finite_scalar(request.q, "q", positive=True)
        if q >= 1.0:
            raise ValueError("q must satisfy 0 < q < 1 for this native backend.")
        h = _finite_scalar(request.h, "h", positive=True)
        t_final = _finite_scalar(request.t_final, "t_final", positive=True)
        t_burn = _finite_scalar(request.t_burn, "t_burn", non_negative=True)
        reorthonormalization_time = _finite_scalar(
            request.reorthonormalization_time,
            "reorthonormalization_time",
            positive=True,
        )
        divergence_norm = _finite_scalar(
            request.divergence_norm,
            "divergence_norm",
            non_negative=True,
        )
        if isinstance(request.fft_block_size, (bool, np.bool_)) or not isinstance(
            request.fft_block_size, (int, np.integer)
        ):
            raise TypeError("fft_block_size must be a positive integer.")
        fft_block_size = int(request.fft_block_size)
        if fft_block_size < 1 or fft_block_size > np.iinfo(np.int32).max:
            raise ValueError("fft_block_size must fit a positive C int.")

        final_steps = exact_fixed_step_count(
            h,
            t_final,
            caller="NativeFractionalVariationalBackend.run(t_final)",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        burn_steps = exact_fixed_step_count(
            h,
            t_burn,
            caller="NativeFractionalVariationalBackend.run(t_burn)",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        interval_steps = exact_fixed_step_count(
            h,
            reorthonormalization_time,
            caller="NativeFractionalVariationalBackend.run(reorthonormalization_time)",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        if final_steps < 1 or interval_steps < 1:
            raise ValueError("t_final and reorthonormalization_time need at least one step.")
        total_steps = final_steps + burn_steps
        if total_steps > _C_INT_MAX - 1:
            raise ValueError("The requested horizon exceeds the native C int step capacity.")
        max_rows = total_steps // interval_steps + 3
        if max_rows > _C_INT_MAX:
            raise ValueError("The convergence output exceeds the native buffer capacity.")
        checked_array_capacity(
            (max_rows,), np.float64,
            caller="NativeFractionalVariationalBackend convergence times",
        )
        checked_array_capacity(
            (max_rows, 3), np.float64,
            caller="NativeFractionalVariationalBackend convergence spectrum",
        )

        c_request = _CFractionalLyapunovRequest()
        c_request.abi_version = 1
        c_request.system_id = system_value
        c_request.execution_contract = contract_value
        c_request.convolution_mode = convolution_value
        c_request.fft_block_size = fft_block_size
        c_request.q = q
        c_request.h = h
        c_request.t_final = t_final
        c_request.t_burn = t_burn
        c_request.reorthonormalization_time = reorthonormalization_time
        c_request.divergence_norm = divergence_norm
        for index, value in enumerate(x0):
            c_request.x0[index] = float(value)
        for index, value in enumerate(params):
            c_request.parameters[index] = float(value)

        times = np.empty(max_rows, dtype=np.float64)
        convergence = np.empty((max_rows, 3), dtype=np.float64).reshape(-1)
        c_result = _CFractionalLyapunovResult()
        rc = int(self.lib.fractional_lyapunov_run(ctypes.byref(c_request), ctypes.byref(c_result), times, convergence, max_rows))
        status = _FRACTIONAL_STATUS.get(rc, f"native_error_{rc}")
        rows = int(c_result.convergence_rows)
        result = FractionalLyapunovResult(
            exponents=np.asarray(c_result.exponents[:3], dtype=float),
            final_state=np.asarray(c_result.final_state[:3], dtype=float),
            times=times[:rows].copy(),
            convergence=convergence[: rows * 3].reshape(rows, 3).copy(),
            status=status,
            steps_completed=int(c_result.steps_completed),
            execution_contract=request.execution_contract,
            convolution_mode=request.convolution_mode,
            metadata={**self.build_metadata, "abi_version": int(c_result.abi_version)},
        )
        if request.convergence_csv is not None and rows:
            csv_path = Path(request.convergence_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", "lambda_0", "lambda_1", "lambda_2"])
                for time_value, values in zip(result.times, result.convergence):
                    writer.writerow([time_value, *values])
        return result


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


@dataclass
class FractionalChuaBackend:
    """Wrapper for ``chua_frac_backend_lib.c``.

    Purpose:
        Integrate ``^C D_t^q x = F(x)`` with the repository EFORK finite-memory
        implementation and return trajectories for diagnostics.

    Validity warning:
        ``Lm`` is a finite-memory approximation.  It must be documented in any
        scientific result and should not be confused with full-history Caputo.
    """

    lib: Any
    _cache = {}
    _cache_lock = threading.RLock()

    @classmethod
    def build(cls, output_name: str = "chua_frac_backend") -> "FractionalChuaBackend":
        native_cache = get_native_cache()
        result, lib = load_ctypes_library(
            C_SOURCE_ROOT / "chua_frac_backend_lib.c",
            native_cache / f"{output_name}{_shared_suffix()}",
            openmp=False,
            expected_symbols=(
                "chua_frac_backend_abi_version",
                "set_frac_chua_params",
                "set_frac_chua_arctan_params",
                "set_frac_chua_model",
                "efork_rows",
                "integrate_chua_efork3",
                "compute_continuation_efork3",
            ),
            expected_abi_version=3,
            abi_version_symbol="chua_frac_backend_abi_version",
        )
        cache_key = str(result.path.resolve())
        with cls._cache_lock:
            if cache_key in cls._cache:
                return cls._cache[cache_key]
        lib.set_frac_chua_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.set_frac_chua_params.restype = None
        lib.set_frac_chua_arctan_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.set_frac_chua_arctan_params.restype = None
        lib.set_frac_chua_model.argtypes = [ctypes.c_int]
        lib.set_frac_chua_model.restype = None
        lib.efork_rows.argtypes = [ctypes.c_double, ctypes.c_double]
        lib.efork_rows.restype = ctypes.c_int
        lib.integrate_chua_efork3.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        lib.integrate_chua_efork3.restype = ctypes.c_int
        lib.compute_continuation_efork3.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.compute_continuation_efork3.restype = ctypes.c_int
        backend = cls(lib=lib)
        backend._transaction_lock = _library_transaction_lock(result.path)
        backend._thread_parameters = threading.local()
        backend._default_parameter_selection = (0, chua_nonsmooth_parameters())
        backend.set_nonsmooth_params(chua_nonsmooth_parameters())
        with cls._cache_lock:
            if cache_key in cls._cache:
                return cls._cache[cache_key]
            cls._cache[cache_key] = backend
            return backend

    def set_nonsmooth_params(self, params: ChuaParameters) -> None:
        """Set the C backend to the non-smooth Chua parameters."""

        selection = self._validated_parameter_selection(0, params)
        self._thread_parameters.selection = selection
        with self._transaction_lock:
            self._default_parameter_selection = selection
            self._apply_parameter_selection_unlocked(selection)

    def set_piecewise_params(self, params: ChuaParameters) -> None:
        """Compatibility alias for :meth:`set_nonsmooth_params`."""

        self.set_nonsmooth_params(params)

    def set_arctan_params(self, params: ChuaParameters) -> None:
        """Set the C backend to a smooth arctan Chua parameterization."""

        selection = self._validated_parameter_selection(1, params)
        self._thread_parameters.selection = selection
        with self._transaction_lock:
            self._default_parameter_selection = selection
            self._apply_parameter_selection_unlocked(selection)

    def set_params(self, params: ChuaParameters) -> None:
        """Dispatch parameter loading according to ``params.model``."""

        if params.model == "arctan":
            self.set_arctan_params(params)
        else:
            self.set_nonsmooth_params(params)

    @staticmethod
    def _validated_parameter_selection(
        model: int, params: ChuaParameters
    ) -> tuple[int, ChuaParameters]:
        values = [params.alpha, params.beta, params.gamma, params.m0, params.m1]
        if model == 1:
            values.extend([params.a1, params.a2, params.rho])
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError("Chua parameters must contain only finite values.")
        return model, params

    def _current_parameter_selection(self) -> tuple[int, ChuaParameters]:
        return getattr(
            self._thread_parameters,
            "selection",
            self._default_parameter_selection,
        )

    def _apply_parameter_selection_unlocked(
        self, selection: tuple[int, ChuaParameters]
    ) -> None:
        model, params = selection
        self.lib.set_frac_chua_model(model)
        self.lib.set_frac_chua_params(
            params.alpha, params.beta, params.gamma, params.m0, params.m1
        )
        if model == 1:
            self.lib.set_frac_chua_arctan_params(params.a1, params.a2, params.rho)

    def integrate_efork3(
        self,
        x0: Sequence[float],
        *,
        q: float,
        h: float,
        Lm: float,
        t_final: float,
        k: float = 0.0,
        eps: float = 1.0,
    ) -> np.ndarray:
        """Integrate one trajectory and return columns ``t,x,y,z``."""

        seed = _finite_state(x0, 3, "x0")
        q = _finite_scalar(q, "q")
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        h = _finite_scalar(h, "h", positive=True)
        Lm = _finite_scalar(Lm, "Lm", positive=True)
        t_final = _finite_scalar(t_final, "t_final", non_negative=True)
        k = _finite_scalar(k, "k")
        eps = _finite_scalar(eps, "eps")
        expected_steps = exact_fixed_step_count(
            h,
            t_final,
            caller="FractionalChuaBackend.integrate_efork3",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        rows = int(self.lib.efork_rows(t_final, h))
        if rows != expected_steps + 1:
            raise RuntimeError("efork_rows disagrees with the shared time-grid contract.")
        if rows <= 0:
            raise RuntimeError(f"efork_rows returned {rows}")
        out = _checked_empty(
            (rows, 4), np.float64,
            caller="FractionalChuaBackend.integrate_efork3 output",
        ).reshape(-1)
        with self._transaction_lock:
            self._apply_parameter_selection_unlocked(
                self._current_parameter_selection()
            )
            rc = int(
                self.lib.integrate_chua_efork3(
                    seed[0], seed[1], seed[2], q, h, Lm, t_final, k, eps,
                    out, out.size,
                )
            )
        if rc != 0:
            raise RuntimeError(f"integrate_chua_efork3 returned {rc}")
        trajectory = out.reshape((rows, 4))
        if float(trajectory[-1, 0]) > t_final:
            raise RuntimeError("native Chua EFORK trajectory exceeded t_final.")
        return trajectory

    def continue_efork3(
        self,
        x0: Sequence[float],
        *,
        lambda_values: Sequence[float] | None = None,
        eps_values: Sequence[float] | None = None,
        q: float,
        k: float,
        h: float,
        Lm: float,
        t_transient: float,
        t_keep: float,
        t_observe: float = 0.0,
        carry_memory: bool = True,
    ) -> dict[str, Any]:
        """Run public ``lambda`` continuation through the native C ABI.

        ``eps_values`` is retained only as a historical-reproduction input
        alias; official outputs expose ``lambda`` and record the internal
        mapping as metadata.
        """

        if lambda_values is not None and eps_values is not None:
            raise ValueError("provide lambda_values only; eps_values is a historical alias.")
        selected_values = lambda_values if lambda_values is not None else eps_values
        if selected_values is None:
            raise ValueError("lambda_values must contain the continuation stages.")
        eps = np.ascontiguousarray(selected_values, dtype=np.float64)
        if eps.ndim != 1 or eps.size == 0:
            raise ValueError("lambda_values must contain at least one continuation stage.")
        if eps.size > np.iinfo(np.int32).max or not np.all(np.isfinite(eps)):
            raise ValueError("lambda_values must be finite and within the native range.")
        seed = _finite_state(x0, 3, "x0")
        q = _finite_scalar(q, "q")
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        k = _finite_scalar(k, "k")
        h = _finite_scalar(h, "h", positive=True)
        Lm = _finite_scalar(Lm, "Lm", positive=True)
        t_transient = _finite_scalar(t_transient, "t_transient", non_negative=True)
        t_keep = _finite_scalar(t_keep, "t_keep", non_negative=True)
        t_observe = _finite_scalar(t_observe, "t_observe", non_negative=True)
        transient_steps = exact_fixed_step_count(
            h,
            t_transient,
            caller="FractionalChuaBackend.continue_efork3 transient",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        keep_steps = exact_fixed_step_count(
            h,
            t_keep,
            caller="FractionalChuaBackend.continue_efork3 keep",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        observation_steps = exact_fixed_step_count(
            h,
            t_observe,
            caller="FractionalChuaBackend.continue_efork3 observation",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        if transient_steps + keep_steps > _NATIVE_STEP_LIMIT:
            raise ValueError("continuation stage length exceeds the native step range.")
        _bounded_positive_ceil_count(
            Lm, h, caller="FractionalChuaBackend.continue_efork3 memory window"
        )
        keep_rows = int(self.lib.efork_rows(t_keep, h))
        if keep_rows != keep_steps + 1:
            raise RuntimeError("efork_rows disagrees for the keep horizon.")
        x_in = _checked_empty(
            (int(eps.size), 3), np.float64,
            caller="FractionalChuaBackend continuation input states",
        ).reshape(-1)
        x_transient = _checked_empty(
            (int(eps.size), 3), np.float64,
            caller="FractionalChuaBackend continuation transient states",
        ).reshape(-1)
        x_out = _checked_empty(
            (int(eps.size), 3), np.float64,
            caller="FractionalChuaBackend continuation output states",
        ).reshape(-1)
        history_in = _checked_empty(
            (int(eps.size),), np.int32,
            caller="FractionalChuaBackend continuation input history counts",
        )
        history_out = _checked_empty(
            (int(eps.size),), np.int32,
            caller="FractionalChuaBackend continuation output history counts",
        )
        traj = _checked_empty(
            (int(eps.size), keep_rows, 4), np.float64,
            caller="FractionalChuaBackend continuation trajectories",
        ).reshape(-1)
        observation_rows = int(self.lib.efork_rows(t_observe, h))
        if observation_rows != observation_steps + 1:
            raise RuntimeError("efork_rows disagrees for the observation horizon.")
        observation = _checked_empty(
            (observation_rows, 4), np.float64,
            caller="FractionalChuaBackend continuation observation",
        ).reshape(-1)
        if not isinstance(carry_memory, (bool, np.bool_)):
            raise TypeError("carry_memory must be boolean.")
        with self._transaction_lock:
            self._apply_parameter_selection_unlocked(
                self._current_parameter_selection()
            )
            rc = int(
                self.lib.compute_continuation_efork3(
                    eps, int(eps.size), seed, q, k, h, Lm,
                    t_transient, t_keep, int(carry_memory), 1,
                    x_in, x_transient, x_out, history_in, history_out, traj,
                    t_observe, observation,
                    x_in.size, x_transient.size, x_out.size, history_in.size,
                    traj.size, observation.size,
                )
            )
        if rc != 0:
            raise RuntimeError(f"compute_continuation_efork3 returned {rc}")
        return {
            "lambda": eps,
            "x_in": x_in.reshape((-1, 3)),
            "x_transient": x_transient.reshape((-1, 3)),
            "x_out": x_out.reshape((-1, 3)),
            "history_in_counts": history_in,
            "history_out_counts": history_out,
            "trajectories": traj.reshape((eps.size, keep_rows, 4)),
            "observation": observation.reshape((observation_rows, 4)),
            "provenance": {"mapping": {"public_parameter": "lambda", "internal_parameter": "epsilon"}},
        }


@dataclass
class FullHistoryABMBackend:
    """Native ABM backend for the non-smooth Chua system.

    :meth:`integrate` retains the complete Caputo history used by the Danca
    reference. :meth:`integrate_truncated` is a separate sliding restarted
    finite-memory approximation and must be labelled as such in comparisons.
    Continuation methods transport either the complete chronological Caputo
    history or the declared finite window across Lur'e deformation stages.
    """

    lib: Any

    @classmethod
    def build(cls, output_name: str = "chua_abm_full_history") -> "FullHistoryABMBackend":
        native_cache = get_native_cache()
        result, lib = load_ctypes_library(
            C_SOURCE_ROOT / "chua_abm_full_history_lib.c",
            native_cache / f"{output_name}{_shared_suffix()}",
            openmp=False,
            expected_symbols=(
                "chua_abm_abi_version",
                "set_abm_chua_params",
                "get_abm_chua_equilibria",
                "abm_rows",
                "integrate_chua_abm_full_history",
                "integrate_chua_abm_truncated_history",
                "compute_continuation_abm",
            ),
            expected_abi_version=3,
            abi_version_symbol="chua_abm_abi_version",
        )
        lib.set_abm_chua_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.set_abm_chua_params.restype = None
        lib.get_abm_chua_equilibria.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
        lib.get_abm_chua_equilibria.restype = None
        lib.abm_rows.argtypes = [ctypes.c_double, ctypes.c_double]
        lib.abm_rows.restype = ctypes.c_int
        lib.integrate_chua_abm_full_history.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        lib.integrate_chua_abm_full_history.restype = ctypes.c_int
        lib.integrate_chua_abm_truncated_history.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        lib.integrate_chua_abm_truncated_history.restype = ctypes.c_int
        lib.compute_continuation_abm.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.compute_continuation_abm.restype = ctypes.c_int
        backend = cls(lib=lib)
        backend._transaction_lock = _library_transaction_lock(result.path)
        backend._thread_parameters = threading.local()
        backend._default_parameters = chua_nonsmooth_parameters()
        backend.set_nonsmooth_params(chua_nonsmooth_parameters())
        return backend

    def set_nonsmooth_params(self, params: ChuaParameters) -> None:
        values = np.asarray(
            [params.alpha, params.beta, params.gamma, params.m0, params.m1],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Chua parameters must contain only finite values.")
        self._thread_parameters.params = params
        with self._transaction_lock:
            self._default_parameters = params
            self._apply_parameters_unlocked(params)

    def _current_parameters(self) -> ChuaParameters:
        return getattr(self._thread_parameters, "params", self._default_parameters)

    def _apply_parameters_unlocked(self, params: ChuaParameters) -> None:
        self.lib.set_abm_chua_params(
            params.alpha, params.beta, params.gamma, params.m0, params.m1
        )

    def equilibria(self) -> dict[str, np.ndarray]:
        out = np.empty(9, dtype=np.float64)
        with self._transaction_lock:
            self._apply_parameters_unlocked(self._current_parameters())
            self.lib.get_abm_chua_equilibria(out)
        return {"E0": out[0:3].copy(), "E+": out[3:6].copy(), "E-": out[6:9].copy()}

    def integrate(
        self,
        x0: Sequence[float],
        *,
        q: float,
        h: float,
        t_final: float,
    ) -> np.ndarray:
        """Integrate one full-history ABM trajectory as columns ``t,x,y,z``."""

        seed = _finite_state(x0, 3, "x0")
        q = _finite_scalar(q, "q")
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        h = _finite_scalar(h, "h", positive=True)
        t_final = _finite_scalar(t_final, "t_final", non_negative=True)
        steps = exact_fixed_step_count(
            h,
            t_final,
            caller="FullHistoryABMBackend.integrate",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        rows = int(self.lib.abm_rows(t_final, h))
        if rows != steps + 1:
            raise RuntimeError("abm_rows disagrees with the shared time-grid contract.")
        if rows <= 0:
            raise RuntimeError(f"abm_rows returned {rows}")
        out = _checked_empty(
            (rows, 4), np.float64,
            caller="FullHistoryABMBackend.integrate output",
        ).reshape(-1)
        with self._transaction_lock:
            self._apply_parameters_unlocked(self._current_parameters())
            rc = int(
                self.lib.integrate_chua_abm_full_history(
                    seed[0], seed[1], seed[2], q, h, t_final, out, out.size,
                )
            )
        if rc != 0:
            raise RuntimeError(f"integrate_chua_abm_full_history returned {rc}")
        trajectory = out.reshape((rows, 4))
        if float(trajectory[-1, 0]) > t_final:
            raise RuntimeError("native full-history ABM trajectory exceeded t_final.")
        return trajectory

    def integrate_truncated(
        self,
        x0: Sequence[float],
        *,
        q: float,
        h: float,
        Lm: float,
        t_final: float,
    ) -> np.ndarray:
        """Integrate with a sliding restarted ABM history window of length ``Lm``."""

        seed = _finite_state(x0, 3, "x0")
        q = _finite_scalar(q, "q")
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        h = _finite_scalar(h, "h", positive=True)
        Lm = _finite_scalar(Lm, "Lm", positive=True)
        t_final = _finite_scalar(t_final, "t_final", non_negative=True)
        steps = exact_fixed_step_count(
            h,
            t_final,
            caller="FullHistoryABMBackend.integrate_truncated",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        _bounded_positive_ceil_count(
            Lm, h, caller="FullHistoryABMBackend.integrate_truncated memory window"
        )
        rows = int(self.lib.abm_rows(t_final, h))
        if rows != steps + 1:
            raise RuntimeError("abm_rows disagrees with the shared time-grid contract.")
        if rows <= 0:
            raise RuntimeError(f"abm_rows returned {rows}")
        out = _checked_empty(
            (rows, 4), np.float64,
            caller="FullHistoryABMBackend.integrate_truncated output",
        ).reshape(-1)
        with self._transaction_lock:
            self._apply_parameters_unlocked(self._current_parameters())
            rc = int(
                self.lib.integrate_chua_abm_truncated_history(
                    seed[0], seed[1], seed[2], q, h, Lm, t_final, out, out.size,
                )
            )
        if rc != 0:
            raise RuntimeError(f"integrate_chua_abm_truncated_history returned {rc}")
        trajectory = out.reshape((rows, 4))
        if float(trajectory[-1, 0]) > t_final:
            raise RuntimeError("native truncated ABM trajectory exceeded t_final.")
        return trajectory

    def _continue_abm(
        self,
        x0: Sequence[float],
        *,
        lambda_values: Sequence[float],
        q: float,
        k: float,
        h: float,
        t_transient: float,
        t_keep: float,
        truncated_history: bool,
        Lm: float | None,
    ) -> dict[str, Any]:
        """Continue the Lur'e deformation while retaining declared history.

        The public parameter ``lambda`` equals the native deformation
        parameter ``epsilon``.  A full-history call represents a causal
        Caputo eta chain.  A truncated call uses the restarted sliding-window
        approximation of duration ``Lm``; it is not full-history Caputo.
        """

        values = np.ascontiguousarray(lambda_values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("lambda_values must contain at least one continuation stage.")
        if values.size > np.iinfo(np.int32).max or not np.all(np.isfinite(values)):
            raise ValueError("lambda_values must be finite and within the native range.")
        if not isinstance(truncated_history, (bool, np.bool_)):
            raise TypeError("truncated_history must be boolean.")
        seed = _finite_state(x0, 3, "x0")
        q = _finite_scalar(q, "q")
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        k = _finite_scalar(k, "k")
        h = _finite_scalar(h, "h", positive=True)
        t_transient = _finite_scalar(t_transient, "t_transient", non_negative=True)
        t_keep = _finite_scalar(t_keep, "t_keep", non_negative=True)
        if truncated_history and Lm is None:
            raise ValueError("Lm must be positive for truncated ABM continuation.")
        normalized_lm = (
            _finite_scalar(Lm, "Lm", positive=True) if truncated_history else 0.0
        )
        keep_steps = exact_fixed_step_count(
            h,
            t_keep,
            caller="FullHistoryABMBackend continuation keep",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        transient_steps = exact_fixed_step_count(
            h,
            t_transient,
            caller="FullHistoryABMBackend continuation transient",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        keep_rows = int(self.lib.abm_rows(t_keep, h))
        transient_rows = int(self.lib.abm_rows(t_transient, h))
        if keep_rows != keep_steps + 1 or transient_rows != transient_steps + 1:
            raise RuntimeError("abm_rows disagrees with the shared time-grid contract.")
        if keep_rows <= 0 or transient_rows <= 0:
            raise RuntimeError("abm_rows returned a non-positive stage length.")
        stage_steps = transient_steps + keep_steps
        if stage_steps > _NATIVE_STEP_LIMIT:
            raise ValueError("continuation stage length exceeds the native range.")
        total_steps = int(values.size) * stage_steps
        if total_steps > _NATIVE_STEP_LIMIT:
            raise ValueError("continuation horizon exceeds the native step range.")
        total_rows = total_steps + 1
        memory_steps = (
            _bounded_positive_ceil_count(
                normalized_lm,
                h,
                caller="FullHistoryABMBackend continuation memory window",
            )
            if truncated_history
            else 0
        )
        history_capacity = (
            memory_steps + 1
            if truncated_history
            else int(total_rows)
        )
        x_in = _checked_empty(
            (int(values.size), 3), np.float64,
            caller="FullHistoryABMBackend continuation input states",
        ).reshape(-1)
        x_transient = _checked_empty(
            (int(values.size), 3), np.float64,
            caller="FullHistoryABMBackend continuation transient states",
        ).reshape(-1)
        x_out = _checked_empty(
            (int(values.size), 3), np.float64,
            caller="FullHistoryABMBackend continuation output states",
        ).reshape(-1)
        history_in = _checked_empty(
            (int(values.size),), np.int32,
            caller="FullHistoryABMBackend continuation input history counts",
        )
        history_out = _checked_empty(
            (int(values.size),), np.int32,
            caller="FullHistoryABMBackend continuation output history counts",
        )
        trajectories = _checked_empty(
            (int(values.size), keep_rows, 4), np.float64,
            caller="FullHistoryABMBackend continuation trajectories",
        ).reshape(-1)
        final_history = _checked_empty(
            (history_capacity, 4), np.float64,
            caller="FullHistoryABMBackend continuation final history",
        ).reshape(-1)
        final_count = np.empty(1, dtype=np.int32)
        with self._transaction_lock:
            self._apply_parameters_unlocked(self._current_parameters())
            rc = int(
                self.lib.compute_continuation_abm(
                    values, int(values.size), seed, q, k, h, normalized_lm,
                    t_transient, t_keep, int(truncated_history),
                    x_in, x_transient, x_out, history_in, history_out,
                    trajectories, final_history, int(history_capacity), final_count,
                    x_in.size, x_transient.size, x_out.size, history_in.size,
                    trajectories.size, final_history.size,
                )
            )
        if rc != 0:
            raise RuntimeError(f"compute_continuation_abm returned {rc}")
        count = int(final_count[0])
        return {
            "lambda": values,
            "x_in": x_in.reshape((-1, 3)),
            "x_transient": x_transient.reshape((-1, 3)),
            "x_out": x_out.reshape((-1, 3)),
            "history_in_counts": history_in,
            "history_out_counts": history_out,
            "trajectories": trajectories.reshape((values.size, keep_rows, 4)),
            "final_history": final_history.reshape((-1, 4))[:count].copy(),
            "final_history_exact": True,
            "history_policy": "truncated_restarted_window" if truncated_history else "full_caputo_history",
            "provenance": {
                "mapping": {"public_parameter": "lambda", "internal_parameter": "epsilon"},
                "eta_boundary_policy": "right_continuous",
            },
        }

    def continue_full_history(
        self,
        x0: Sequence[float],
        *,
        lambda_values: Sequence[float],
        q: float,
        k: float,
        h: float,
        t_transient: float,
        t_keep: float,
    ) -> dict[str, Any]:
        """Continue with complete causal Caputo history across eta stages."""

        return self._continue_abm(
            x0,
            lambda_values=lambda_values,
            q=q,
            k=k,
            h=h,
            t_transient=t_transient,
            t_keep=t_keep,
            truncated_history=False,
            Lm=None,
        )

    def continue_truncated_history(
        self,
        x0: Sequence[float],
        *,
        lambda_values: Sequence[float],
        q: float,
        k: float,
        h: float,
        Lm: float,
        t_transient: float,
        t_keep: float,
    ) -> dict[str, Any]:
        """Continue with an explicit finite restarted memory window ``Lm``."""

        return self._continue_abm(
            x0,
            lambda_values=lambda_values,
            q=q,
            k=k,
            h=h,
            t_transient=t_transient,
            t_keep=t_keep,
            truncated_history=True,
            Lm=Lm,
        )


@dataclass
class BasinBackend:
    """Wrapper for ``chua_basin_lib.c`` classification routines."""

    lib: Any

    @classmethod
    def build(cls, output_name: str = "chua_basin_backend") -> "BasinBackend":
        native_cache = get_native_cache()
        result, lib = load_ctypes_library(
            C_SOURCE_ROOT / "chua_basin_lib.c",
            native_cache / f"{output_name}{_shared_suffix()}",
            openmp=False,
            expected_symbols=(
                "chua_basin_abi_version",
                "set_chua_params",
                "set_chua_arctan_params",
                "set_chua_model",
                "get_equilibria",
                "classify_basin_point",
            ),
            expected_abi_version=3,
            abi_version_symbol="chua_basin_abi_version",
        )
        lib.set_chua_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.set_chua_params.restype = None
        lib.set_chua_arctan_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.set_chua_arctan_params.restype = None
        lib.set_chua_model.argtypes = [ctypes.c_int]
        lib.set_chua_model.restype = None
        lib.get_equilibria.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
        lib.get_equilibria.restype = None
        lib.classify_basin_point.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
        ]
        lib.classify_basin_point.restype = ctypes.c_int
        backend = cls(lib=lib)
        backend._transaction_lock = _library_transaction_lock(result.path)
        backend._thread_parameters = threading.local()
        backend._default_parameter_selection = (0, chua_nonsmooth_parameters())
        backend.set_nonsmooth_params(chua_nonsmooth_parameters())
        return backend

    def set_nonsmooth_params(self, params: ChuaParameters) -> None:
        selection = self._validated_parameter_selection(0, params)
        self._thread_parameters.selection = selection
        with self._transaction_lock:
            self._default_parameter_selection = selection
            self._apply_parameter_selection_unlocked(selection)

    def set_piecewise_params(self, params: ChuaParameters) -> None:
        """Compatibility alias for :meth:`set_nonsmooth_params`."""

        self.set_nonsmooth_params(params)

    def set_arctan_params(self, params: ChuaParameters) -> None:
        """Set the basin backend to a smooth arctan Chua parameterization."""

        selection = self._validated_parameter_selection(1, params)
        self._thread_parameters.selection = selection
        with self._transaction_lock:
            self._default_parameter_selection = selection
            self._apply_parameter_selection_unlocked(selection)

    def set_params(self, params: ChuaParameters) -> None:
        """Dispatch parameter loading according to ``params.model``."""

        if params.model == "arctan":
            self.set_arctan_params(params)
        else:
            self.set_nonsmooth_params(params)

    @staticmethod
    def _validated_parameter_selection(
        model: int, params: ChuaParameters
    ) -> tuple[int, ChuaParameters]:
        values = [params.alpha, params.beta, params.gamma, params.m0, params.m1]
        if model == 1:
            values.extend([params.a1, params.a2, params.rho])
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError("Chua parameters must contain only finite values.")
        return model, params

    def _current_parameter_selection(self) -> tuple[int, ChuaParameters]:
        return getattr(
            self._thread_parameters, "selection", self._default_parameter_selection
        )

    def _apply_parameter_selection_unlocked(
        self, selection: tuple[int, ChuaParameters]
    ) -> None:
        model, params = selection
        self.lib.set_chua_model(model)
        self.lib.set_chua_params(
            params.alpha, params.beta, params.gamma, params.m0, params.m1
        )
        if model == 1:
            self.lib.set_chua_arctan_params(params.a1, params.a2, params.rho)

    def equilibria(self) -> dict[str, np.ndarray]:
        out = np.zeros(9, dtype=np.float64)
        with self._transaction_lock:
            self._apply_parameter_selection_unlocked(
                self._current_parameter_selection()
            )
            self.lib.get_equilibria(out)
        return {"E0": out[0:3].copy(), "E+": out[3:6].copy(), "E-": out[6:9].copy()}

    def classify_point(
        self,
        x0: Sequence[float],
        *,
        q: float,
        h: float,
        Lm: float,
        t_final: float,
        t_burn: float,
        divergence_norm: float = 120.0,
        r_bound: float = 60.0,
        equilibrium_tol: float = 1.0e-3,
        cap_win: int = 150,
        mean_x_gap: float = 0.75,
    ) -> int:
        seed = _finite_state(x0, 3, "x0")
        q = _finite_scalar(q, "q")
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        h = _finite_scalar(h, "h", positive=True)
        Lm = _finite_scalar(Lm, "Lm", positive=True)
        t_final = _finite_scalar(t_final, "t_final", positive=True)
        t_burn = _finite_scalar(t_burn, "t_burn", non_negative=True)
        if t_burn > t_final:
            raise ValueError("t_burn must not exceed t_final.")
        exact_fixed_step_count(
            h,
            t_final,
            caller="BasinBackend.classify_point",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        exact_fixed_step_count(
            h,
            t_burn,
            caller="BasinBackend.classify_point burn",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        divergence_norm = _finite_scalar(
            divergence_norm, "divergence_norm", positive=True
        )
        r_bound = _finite_scalar(r_bound, "r_bound", positive=True)
        equilibrium_tol = _finite_scalar(
            equilibrium_tol, "equilibrium_tol", positive=True
        )
        mean_x_gap = _finite_scalar(mean_x_gap, "mean_x_gap", positive=True)
        if isinstance(cap_win, (bool, np.bool_)) or not isinstance(cap_win, (int, np.integer)):
            raise TypeError("cap_win must be an integer.")
        cap_win = int(cap_win)
        if cap_win < 1 or cap_win > np.iinfo(np.int32).max:
            raise ValueError("cap_win must be between 1 and INT32_MAX.")
        with self._transaction_lock:
            self._apply_parameter_selection_unlocked(
                self._current_parameter_selection()
            )
            result = int(
                self.lib.classify_basin_point(
                    seed[0], seed[1], seed[2], q, h, Lm, t_final, t_burn,
                    divergence_norm, r_bound, equilibrium_tol, cap_win, mean_x_gap,
                )
            )
        if result < 0:
            raise RuntimeError(f"classify_basin_point returned error code {result}.")
        return result


@dataclass
class FractionalLyapunovBackend:
    """Runner for the native EFORK/Benettin finite-memory diagnostic."""

    executable: Path

    @classmethod
    def build(cls, output_name: str = "chua_frac_lyapunov_efork_benettin") -> "FractionalLyapunovBackend":
        native_cache = get_native_cache()
        suffix = ".exe" if sys.platform == "win32" else ""
        result = compile_c_target(
            C_SOURCE_ROOT / "chua_frac_lyapunov_efork_benettin.c",
            native_cache / f"{output_name}{suffix}",
            target_kind="executable",
            openmp=False,
        )
        return cls(executable=result.path)

    def run(
        self,
        x0: Sequence[float],
        *,
        params: ChuaParameters | None = None,
        q: float,
        h: float,
        Lm: float,
        t_burn: float,
        n_blocks: int,
        t_block: float,
        convergence_csv: str | Path,
    ) -> dict[str, Any]:
        """Execute the native diagnostic and return the reported exponents."""

        p = params or chua_nonsmooth_parameters()
        if p.model != "nonsmooth":
            raise ValueError(
                "FractionalLyapunovBackend supports only the nonsmooth Chua model."
            )
        parameter_values = np.asarray(
            [p.alpha, p.beta, p.gamma, p.m0, p.m1], dtype=np.float64
        )
        if not np.all(np.isfinite(parameter_values)):
            raise ValueError("Chua parameters must contain only finite values.")
        seed = _finite_state(x0, 3, "x0")
        q = _finite_scalar(q, "q")
        if not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        h = _finite_scalar(h, "h", positive=True)
        Lm = _finite_scalar(Lm, "Lm", positive=True)
        t_burn = _finite_scalar(t_burn, "t_burn", non_negative=True)
        t_block = _finite_scalar(t_block, "t_block", positive=True)
        if isinstance(n_blocks, (bool, np.bool_)) or not isinstance(
            n_blocks, (int, np.integer)
        ):
            raise TypeError("n_blocks must be an integer.")
        n_blocks = int(n_blocks)
        if n_blocks < 1 or n_blocks > np.iinfo(np.int32).max:
            raise ValueError("n_blocks must be between 1 and INT32_MAX.")
        burn_steps = exact_fixed_step_count(
            h,
            t_burn,
            caller="FractionalLyapunovBackend.run burn",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        block_steps = exact_fixed_step_count(
            h,
            t_block,
            caller="FractionalLyapunovBackend.run block",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        if block_steps < 1:
            raise ValueError("t_block must contain at least one fixed step.")
        if burn_steps + n_blocks * block_steps > _C_INT_MAX:
            raise ValueError("requested Lyapunov horizon exceeds the native step range.")
        env = os.environ.copy()
        env["CHUA_LE_CSV"] = str(Path(convergence_csv))
        cmd = [
            str(self.executable),
            str(float(seed[0])),
            str(float(seed[1])),
            str(float(seed[2])),
            str(float(p.alpha)),
            str(float(p.beta)),
            str(float(p.gamma)),
            str(float(p.m0)),
            str(float(p.m1)),
            str(float(q)),
            str(float(h)),
            str(float(Lm)),
            str(float(t_burn)),
            str(int(n_blocks)),
            str(float(t_block)),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        exponents: list[float] = []
        final_state: list[float] = []
        for line in result.stdout.splitlines():
            if line.startswith("# LE_frac_standard "):
                exponents = [float(value) for value in line.split()[2:5]]
            elif line.startswith("# final_state "):
                final_state = [float(value) for value in line.split()[2:5]]
        if len(exponents) != 3:
            raise RuntimeError("Native Lyapunov executable did not return three exponents.")
        if len(final_state) != 3:
            raise RuntimeError("Native Lyapunov executable did not return a 3D final state.")
        if not np.all(np.isfinite(np.asarray(exponents + final_state))):
            raise RuntimeError("Native Lyapunov executable returned non-finite values.")
        return {
            "exponents": np.asarray(exponents, dtype=float),
            "final_state": np.asarray(final_state, dtype=float),
            "stdout": result.stdout,
            "convergence_csv": str(Path(convergence_csv)),
        }


@dataclass
class GeneralFDEBackend:
    """Wrapper for general FDE solver in C.
    """
    lib: Any
    _cache = {}
    _cache_lock = threading.RLock()

    @classmethod
    def build(cls, output_name: str = "general_fde_solver") -> "GeneralFDEBackend":
        native_cache = get_native_cache()
        result, lib = load_ctypes_library(
            C_SOURCE_ROOT / "general_fde_solver.c",
            native_cache / f"{output_name}{_shared_suffix()}",
            openmp=False,
            expected_symbols=(
                "general_fde_abi_version",
                "general_fde_rows",
                "integrate_general_efork_c",
                "integrate_general_abm_c",
            ),
            expected_abi_version=3,
            abi_version_symbol="general_fde_abi_version",
        )
        cache_key = str(result.path.resolve())
        with cls._cache_lock:
            if cache_key in cls._cache:
                return cls._cache[cache_key]

        # Callback type: RhsCallback(double t, const double *x, double *f)
        cls.RHS_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
        lib.general_fde_rows.argtypes = [ctypes.c_double, ctypes.c_double]
        lib.general_fde_rows.restype = ctypes.c_int
        
        lib.integrate_general_efork_c.argtypes = [
            cls.RHS_CALLBACK,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        lib.integrate_general_efork_c.restype = ctypes.c_int

        lib.integrate_general_abm_c.argtypes = [
            cls.RHS_CALLBACK,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        lib.integrate_general_abm_c.restype = ctypes.c_int
        
        backend = cls(lib=lib)
        with cls._cache_lock:
            if cache_key in cls._cache:
                return cls._cache[cache_key]
            cls._cache[cache_key] = backend
        return backend

    def integrate(
        self,
        rhs: Any,
        x0: np.ndarray,
        q: float,
        h: float,
        t_final: float,
        divergence_norm: float = 120.0,
        integrator: str = "efork"
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if not callable(rhs):
            raise TypeError("rhs must be callable.")
        x0_raw = np.asarray(x0, dtype=np.float64)
        if x0_raw.ndim != 1 or x0_raw.size == 0:
            raise ValueError("x0 must have non-empty shape (dim,).")
        if x0_raw.size > np.iinfo(np.int32).max:
            raise ValueError("x0 dimension exceeds the native INT32 range.")
        if not np.all(np.isfinite(x0_raw)):
            raise ValueError("x0 must contain only finite values.")
        x0_arr = np.ascontiguousarray(x0_raw)
        dim = x0_arr.size
        q = _finite_scalar(q, "q")
        if not 0.0 < q < 1.0:
            raise ValueError("q must satisfy 0 < q < 1 for GeneralFDEBackend.")
        h = _finite_scalar(h, "h", positive=True)
        t_final = _finite_scalar(t_final, "t_final", non_negative=True)
        divergence_norm = _finite_scalar(
            divergence_norm, "divergence_norm", positive=True
        )
        nsteps = exact_fixed_step_count(
            h,
            t_final,
            caller="GeneralFDEBackend.integrate",
            max_steps=_NATIVE_STEP_LIMIT,
        )
        rows = nsteps + 1
        if hasattr(self.lib, "general_fde_rows"):
            native_rows = int(self.lib.general_fde_rows(t_final, h))
            if native_rows != rows:
                raise RuntimeError(
                    f"general_fde_rows disagrees with Python: {native_rows} != {rows}."
                )
        integrator_l = str(integrator).lower() if isinstance(integrator, str) else ""
        if integrator_l not in {"efork", "abm"}:
            raise ValueError("integrator must be exactly 'efork' or 'abm'.")
        
        # Output buffer for [t, x_0, x_1, ...]
        out = _checked_empty(
            (rows, dim + 1), np.float64,
            caller="GeneralFDEBackend output",
        ).reshape(-1)
        bound_rhs = bind_rhs(rhs)
        callback_errors: list[BaseException] = []
        
        # Construct C-compatible callback
        def c_rhs(t_val, x_ptr, f_ptr):
            x_arr = np.ctypeslib.as_array(x_ptr, shape=(dim,))
            f_arr = np.ctypeslib.as_array(f_ptr, shape=(dim,))
            if callback_errors:
                f_arr.fill(np.nan)
                return
            try:
                deriv = np.asarray(bound_rhs(t_val, x_arr), dtype=np.float64)
                if deriv.shape != (dim,):
                    raise ValueError(
                        f"rhs output shape must be ({dim},), got {deriv.shape}."
                    )
                if not np.all(np.isfinite(deriv)):
                    raise FloatingPointError("rhs returned non-finite values.")
                f_arr[:] = deriv
            except BaseException as exc:  # ctypes cannot propagate callback errors
                callback_errors.append(exc)
                f_arr.fill(np.nan)
                
        c_callback = self.RHS_CALLBACK(c_rhs)
        
        if integrator_l == "efork":
            rc = int(
                self.lib.integrate_general_efork_c(
                    c_callback,
                    x0_arr,
                    dim,
                    q,
                    h,
                    t_final,
                    divergence_norm,
                    out,
                    out.size,
                )
            )
        else: # abm
            rc = int(
                self.lib.integrate_general_abm_c(
                    c_callback,
                    x0_arr,
                    dim,
                    q,
                    h,
                    t_final,
                    divergence_norm,
                    out,
                    out.size,
                )
            )

        if callback_errors:
            raise callback_errors[0]

        if rc < 0:
            raise RuntimeError(f"General FDE solver in C returned error code: {rc}")
        if rc < 1 or rc > rows:
            raise RuntimeError(
                f"General FDE solver returned invalid row count {rc} for capacity {rows}."
            )
            
        # Re-shape output
        actual_rows = rc
        out_res = out.reshape((rows, dim + 1))[:actual_rows]
        status = "ok"
        if actual_rows < rows:
            status = "diverged"
        if len(out_res) and float(out_res[-1, 0]) > t_final:
            raise RuntimeError("General FDE solver exceeded t_final.")
            
        return out_res[:, 0], out_res[:, 1:], status

