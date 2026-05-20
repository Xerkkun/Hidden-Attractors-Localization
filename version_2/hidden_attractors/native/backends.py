"""Native C backend wrappers.

These wrappers centralize ctypes signatures and compilation policy.  They do
not introduce a new numerical method; they expose the existing C EFORK and basin
classifiers behind a Python API suitable for experiments and examples.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models.chua import ChuaParameters, chua_piecewise_parameters
from ..parallel import compile_c_target
from ..paths import NATIVE_CACHE, PACKAGE_ROOT


C_SOURCE_ROOT = PACKAGE_ROOT / "native" / "csrc"


def _shared_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


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
        backend = cls(lib=lib)
        backend.set_piecewise_params(chua_piecewise_parameters())
        return backend

    def set_piecewise_params(self, params: ChuaParameters) -> None:
        """Set the C backend to the piecewise Chua parameters."""

        self.lib.set_frac_chua_model(0)
        self.lib.set_frac_chua_params(params.alpha, params.beta, params.gamma, params.m0, params.m1)

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
        backend.set_piecewise_params(chua_piecewise_parameters())
        return backend

    def set_piecewise_params(self, params: ChuaParameters) -> None:
        self.lib.set_chua_model(0)
        self.lib.set_chua_params(params.alpha, params.beta, params.gamma, params.m0, params.m1)

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
