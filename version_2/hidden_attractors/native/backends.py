"""Native C backend wrappers.

These wrappers centralize ctypes signatures and compilation policy.  They do
not introduce a new numerical method; they expose the existing C EFORK and basin
classifiers behind a Python API suitable for experiments and examples.
"""

from __future__ import annotations

import ctypes
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models.chua import ChuaParameters, chua_nonsmooth_parameters
from ..parallel import compile_c_target
from ..paths import NATIVE_CACHE, PACKAGE_ROOT
from .contracts import FractionalLyapunovRequest, FractionalLyapunovResult


C_SOURCE_ROOT = PACKAGE_ROOT / "native" / "csrc"

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

    @classmethod
    def build(cls, output_name: str = "fractional_variational_lyapunov") -> "NativeFractionalVariationalBackend":
        if output_name in cls._cache:
            return cls._cache[output_name]
        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
        result = compile_c_target(
            C_SOURCE_ROOT / "fractional_variational_lyapunov_lib.c",
            NATIVE_CACHE / f"{output_name}{_shared_suffix()}",
            target_kind="shared",
            openmp=False,
        )
        lib = ctypes.CDLL(str(result.path.resolve()))
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
        cls._cache[output_name] = backend
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

        x0 = np.asarray(request.x0, dtype=float)
        if x0.shape != (3,):
            raise ValueError("Native fractional Lyapunov systems currently require x0 with shape (3,).")
        params = self._parameter_vector(request.system_id, request.parameters)
        c_request = _CFractionalLyapunovRequest()
        c_request.abi_version = 1
        c_request.system_id = system_value
        c_request.execution_contract = contract_value
        c_request.convolution_mode = convolution_value
        c_request.fft_block_size = int(request.fft_block_size)
        c_request.q = float(request.q)
        c_request.h = float(request.h)
        c_request.t_final = float(request.t_final)
        c_request.t_burn = float(request.t_burn)
        c_request.reorthonormalization_time = float(request.reorthonormalization_time)
        c_request.divergence_norm = float(request.divergence_norm)
        for index, value in enumerate(x0):
            c_request.x0[index] = float(value)
        for index, value in enumerate(params):
            c_request.parameters[index] = float(value)

        interval = max(1, round(float(request.reorthonormalization_time) / float(request.h)))
        total_steps = round((float(request.t_final) + float(request.t_burn)) / float(request.h))
        max_rows = max(2, total_steps // interval + 3)
        times = np.empty(max_rows, dtype=np.float64)
        convergence = np.empty(max_rows * 3, dtype=np.float64)
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

    @classmethod
    def build(cls, output_name: str = "chua_frac_backend") -> "FractionalChuaBackend":
        if output_name in cls._cache:
            return cls._cache[output_name]
        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
        result = compile_c_target(
            C_SOURCE_ROOT / "chua_frac_backend_lib.c",
            NATIVE_CACHE / f"{output_name}{_shared_suffix()}",
            target_kind="shared",
            openmp=False,
        )
        lib = ctypes.CDLL(str(result.path.resolve()))
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
        ]
        lib.compute_continuation_efork3.restype = ctypes.c_int
        backend = cls(lib=lib)
        backend.set_nonsmooth_params(chua_nonsmooth_parameters())
        cls._cache[output_name] = backend
        return backend

    def set_nonsmooth_params(self, params: ChuaParameters) -> None:
        """Set the C backend to the non-smooth Chua parameters."""

        self.lib.set_frac_chua_model(0)
        self.lib.set_frac_chua_params(params.alpha, params.beta, params.gamma, params.m0, params.m1)

    def set_piecewise_params(self, params: ChuaParameters) -> None:
        """Compatibility alias for :meth:`set_nonsmooth_params`."""

        self.set_nonsmooth_params(params)

    def set_arctan_params(self, params: ChuaParameters) -> None:
        """Set the C backend to a smooth arctan Chua parameterization."""

        self.lib.set_frac_chua_model(1)
        self.lib.set_frac_chua_params(params.alpha, params.beta, params.gamma, params.m0, params.m1)
        self.lib.set_frac_chua_arctan_params(params.a1, params.a2, params.rho)

    def set_params(self, params: ChuaParameters) -> None:
        """Dispatch parameter loading according to ``params.model``."""

        if params.model == "arctan":
            self.set_arctan_params(params)
        else:
            self.set_nonsmooth_params(params)

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

        rows = int(self.lib.efork_rows(float(t_final), float(h)))
        if rows <= 0:
            raise RuntimeError(f"efork_rows returned {rows}")
        out = np.empty(rows * 4, dtype=np.float64)
        seed = np.asarray(x0, dtype=float)
        rc = int(
            self.lib.integrate_chua_efork3(
                float(seed[0]),
                float(seed[1]),
                float(seed[2]),
                float(q),
                float(h),
                float(Lm),
                float(t_final),
                float(k),
                float(eps),
                out,
            )
        )
        if rc != 0:
            raise RuntimeError(f"integrate_chua_efork3 returned {rc}")
        return out.reshape((rows, 4))

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
        if eps.size == 0:
            raise ValueError("lambda_values must contain at least one continuation stage.")
        seed = np.ascontiguousarray(x0, dtype=np.float64)
        keep_rows = int(self.lib.efork_rows(float(t_keep), float(h)))
        x_in = np.empty(eps.size * 3, dtype=np.float64)
        x_transient = np.empty(eps.size * 3, dtype=np.float64)
        x_out = np.empty(eps.size * 3, dtype=np.float64)
        history_in = np.empty(eps.size, dtype=np.int32)
        history_out = np.empty(eps.size, dtype=np.int32)
        traj = np.empty(eps.size * keep_rows * 4, dtype=np.float64)
        observation_rows = int(self.lib.efork_rows(float(t_observe), float(h)))
        observation = np.empty(observation_rows * 4, dtype=np.float64)
        rc = int(
            self.lib.compute_continuation_efork3(
                eps,
                int(eps.size),
                seed,
                float(q),
                float(k),
                float(h),
                float(Lm),
                float(t_transient),
                float(t_keep),
                int(bool(carry_memory)),
                1,
                x_in,
                x_transient,
                x_out,
                history_in,
                history_out,
                traj,
                float(t_observe),
                observation,
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
        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
        result = compile_c_target(
            C_SOURCE_ROOT / "chua_abm_full_history_lib.c",
            NATIVE_CACHE / f"{output_name}{_shared_suffix()}",
            target_kind="shared",
            openmp=False,
        )
        lib = ctypes.CDLL(str(result.path.resolve()))
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
        ]
        lib.compute_continuation_abm.restype = ctypes.c_int
        backend = cls(lib=lib)
        backend.set_nonsmooth_params(chua_nonsmooth_parameters())
        return backend

    def set_nonsmooth_params(self, params: ChuaParameters) -> None:
        self.lib.set_abm_chua_params(params.alpha, params.beta, params.gamma, params.m0, params.m1)

    def equilibria(self) -> dict[str, np.ndarray]:
        out = np.empty(9, dtype=np.float64)
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

        rows = int(self.lib.abm_rows(float(t_final), float(h)))
        if rows <= 0:
            raise RuntimeError(f"abm_rows returned {rows}")
        out = np.empty(rows * 4, dtype=np.float64)
        seed = np.asarray(x0, dtype=float)
        rc = int(
            self.lib.integrate_chua_abm_full_history(
                float(seed[0]),
                float(seed[1]),
                float(seed[2]),
                float(q),
                float(h),
                float(t_final),
                out,
            )
        )
        if rc != 0:
            raise RuntimeError(f"integrate_chua_abm_full_history returned {rc}")
        return out.reshape((rows, 4))

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

        rows = int(self.lib.abm_rows(float(t_final), float(h)))
        if rows <= 0:
            raise RuntimeError(f"abm_rows returned {rows}")
        out = np.empty(rows * 4, dtype=np.float64)
        seed = np.asarray(x0, dtype=float)
        rc = int(
            self.lib.integrate_chua_abm_truncated_history(
                float(seed[0]),
                float(seed[1]),
                float(seed[2]),
                float(q),
                float(h),
                float(Lm),
                float(t_final),
                out,
            )
        )
        if rc != 0:
            raise RuntimeError(f"integrate_chua_abm_truncated_history returned {rc}")
        return out.reshape((rows, 4))

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
        if values.size == 0:
            raise ValueError("lambda_values must contain at least one continuation stage.")
        if truncated_history and (Lm is None or float(Lm) <= 0.0):
            raise ValueError("Lm must be positive for truncated ABM continuation.")
        seed = np.ascontiguousarray(x0, dtype=np.float64)
        keep_rows = int(self.lib.abm_rows(float(t_keep), float(h)))
        transient_rows = int(self.lib.abm_rows(float(t_transient), float(h)))
        if keep_rows <= 0 or transient_rows <= 0:
            raise RuntimeError("abm_rows returned a non-positive stage length.")
        total_rows = 1 + values.size * ((transient_rows - 1) + (keep_rows - 1))
        history_capacity = (
            int(np.ceil(float(Lm) / float(h))) + 1
            if truncated_history
            else int(total_rows)
        )
        x_in = np.empty(values.size * 3, dtype=np.float64)
        x_transient = np.empty(values.size * 3, dtype=np.float64)
        x_out = np.empty(values.size * 3, dtype=np.float64)
        history_in = np.empty(values.size, dtype=np.int32)
        history_out = np.empty(values.size, dtype=np.int32)
        trajectories = np.empty(values.size * keep_rows * 4, dtype=np.float64)
        final_history = np.empty(history_capacity * 4, dtype=np.float64)
        final_count = np.empty(1, dtype=np.int32)
        rc = int(
            self.lib.compute_continuation_abm(
                values,
                int(values.size),
                seed,
                float(q),
                float(k),
                float(h),
                0.0 if Lm is None else float(Lm),
                float(t_transient),
                float(t_keep),
                int(bool(truncated_history)),
                x_in,
                x_transient,
                x_out,
                history_in,
                history_out,
                trajectories,
                final_history,
                int(history_capacity),
                final_count,
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
        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
        result = compile_c_target(
            C_SOURCE_ROOT / "chua_basin_lib.c",
            NATIVE_CACHE / f"{output_name}{_shared_suffix()}",
            target_kind="shared",
            openmp=False,
        )
        lib = ctypes.CDLL(str(result.path.resolve()))
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
        backend.set_nonsmooth_params(chua_nonsmooth_parameters())
        return backend

    def set_nonsmooth_params(self, params: ChuaParameters) -> None:
        self.lib.set_chua_model(0)
        self.lib.set_chua_params(params.alpha, params.beta, params.gamma, params.m0, params.m1)

    def set_piecewise_params(self, params: ChuaParameters) -> None:
        """Compatibility alias for :meth:`set_nonsmooth_params`."""

        self.set_nonsmooth_params(params)

    def set_arctan_params(self, params: ChuaParameters) -> None:
        """Set the basin backend to a smooth arctan Chua parameterization."""

        self.lib.set_chua_model(1)
        self.lib.set_chua_params(params.alpha, params.beta, params.gamma, params.m0, params.m1)
        self.lib.set_chua_arctan_params(params.a1, params.a2, params.rho)

    def set_params(self, params: ChuaParameters) -> None:
        """Dispatch parameter loading according to ``params.model``."""

        if params.model == "arctan":
            self.set_arctan_params(params)
        else:
            self.set_nonsmooth_params(params)

    def equilibria(self) -> dict[str, np.ndarray]:
        out = np.zeros(9, dtype=np.float64)
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
        seed = np.asarray(x0, dtype=float)
        return int(
            self.lib.classify_basin_point(
                float(seed[0]),
                float(seed[1]),
                float(seed[2]),
                float(q),
                float(h),
                float(Lm),
                float(t_final),
                float(t_burn),
                float(divergence_norm),
                float(r_bound),
                float(equilibrium_tol),
                int(cap_win),
                float(mean_x_gap),
            )
        )


@dataclass
class FractionalLyapunovBackend:
    """Runner for the native EFORK/Benettin finite-memory diagnostic."""

    executable: Path

    @classmethod
    def build(cls, output_name: str = "chua_frac_lyapunov_efork_benettin") -> "FractionalLyapunovBackend":
        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
        suffix = ".exe" if sys.platform == "win32" else ""
        result = compile_c_target(
            C_SOURCE_ROOT / "chua_frac_lyapunov_efork_benettin.c",
            NATIVE_CACHE / f"{output_name}{suffix}",
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
        seed = np.asarray(x0, dtype=float)
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

    @classmethod
    def build(cls, output_name: str = "general_fde_solver") -> "GeneralFDEBackend":
        if output_name in cls._cache:
            return cls._cache[output_name]
        NATIVE_CACHE.mkdir(parents=True, exist_ok=True)
        result = compile_c_target(
            C_SOURCE_ROOT / "general_fde_solver.c",
            NATIVE_CACHE / f"{output_name}{_shared_suffix()}",
            target_kind="shared",
            openmp=False,
        )
        lib = ctypes.CDLL(str(result.path.resolve()))
        
        # Callback type: RhsCallback(double t, const double *x, double *f)
        cls.RHS_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
        
        lib.integrate_general_efork_c.argtypes = [
            cls.RHS_CALLBACK,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
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
        ]
        lib.integrate_general_abm_c.restype = ctypes.c_int
        
        backend = cls(lib=lib)
        cls._cache[output_name] = backend
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
        x0_arr = np.asarray(x0, dtype=np.float64)
        dim = x0_arr.size
        nsteps = int(round(t_final / h))
        rows = nsteps + 1
        
        # Output buffer for [t, x_0, x_1, ...]
        out = np.empty(rows * (dim + 1), dtype=np.float64)
        
        # Construct C-compatible callback
        def c_rhs(t_val, x_ptr, f_ptr):
            x_arr = np.ctypeslib.as_array(x_ptr, shape=(dim,))
            f_arr = np.ctypeslib.as_array(f_ptr, shape=(dim,))
            try:
                deriv = np.asarray(rhs(t_val, x_arr), dtype=np.float64)
            except TypeError:
                deriv = np.asarray(rhs(x_arr), dtype=np.float64)
            for d in range(dim):
                f_arr[d] = deriv[d]
                
        c_callback = self.RHS_CALLBACK(c_rhs)
        
        if integrator == "efork":
            rc = int(
                self.lib.integrate_general_efork_c(
                    c_callback,
                    x0_arr,
                    dim,
                    q,
                    h,
                    t_final,
                    divergence_norm,
                    out
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
                    out
                )
            )
            
        if rc < 0:
            raise RuntimeError(f"General FDE solver in C returned error code: {rc}")
            
        # Re-shape output
        actual_rows = rc
        out_res = out.reshape((rows, dim + 1))[:actual_rows]
        status = "ok"
        if actual_rows < rows:
            status = "diverged"
            
        return out_res[:, 0], out_res[:, 1:], status

