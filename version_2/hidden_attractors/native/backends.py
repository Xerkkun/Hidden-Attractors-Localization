"""Native C backend wrappers.

These wrappers centralize ctypes signatures and compilation policy.  They do
not introduce a new numerical method; they expose the existing C EFORK and basin
classifiers behind a Python API suitable for experiments and examples.
"""

from __future__ import annotations

import ctypes
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


C_SOURCE_ROOT = PACKAGE_ROOT / "native" / "csrc"


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

    @classmethod
    def build(cls, output_name: str = "chua_frac_backend") -> "FractionalChuaBackend":
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
