"""Optional native-C Grunwald-Letnikov history operators.

Stability: experimental

The native ABI evaluates weights and multicomponent GL convolutions without a
Python callback in the inner history loop.  Compilation is lazy and uses the
repository native cache.  Public convenience functions fall back to the
existing Numba kernel when a C compiler or loadable shared-library runtime is
unavailable, unless ``fallback=False`` is requested.

References
----------
I. Podlubny, *Fractional Differential Equations*, Academic Press, 1999,
ISBN 978-0-12-558840-9.
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17(3), 704-719, 1986, https://doi.org/10.1137/0517050.
"""

from __future__ import annotations

import ctypes
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np

from .._native_loading import load_shared_library as _load_shared_library
from ..parallel import compile_c_target
from ..paths import PACKAGE_ROOT, get_native_cache
from .contracts import normalize_fractional_orders


_C_SOURCE = PACKAGE_ROOT / "native" / "csrc" / "grunwald_letnikov_lib.c"
_ABI_VERSION = 1
_STATUS_MESSAGES = {
    0: "ok",
    -1: "null_pointer",
    -2: "invalid_shape",
    -3: "invalid_order",
    -4: "invalid_step",
    -5: "allocation_failed",
    -6: "nonfinite_input",
    -7: "invalid_mode",
    -8: "aliased_buffers",
    -9: "size_overflow",
}


class NativeGLBackendUnavailable(RuntimeError):
    """Raised when native execution is required but cannot be prepared."""


class NativeGLKernelError(RuntimeError):
    """Raised when the C ABI rejects a request or cannot finish it."""


@dataclass(frozen=True, slots=True)
class NativeGLBuildMetadata:
    """Build and provenance information for one native or fallback run."""

    available: bool
    backend: str
    abi_version: int | None
    kernel_id: str | None
    source_sha256: str
    library_path: str | None = None
    compiler: str | None = None
    compile_command: tuple[str, ...] = ()
    openmp_requested: bool = False
    openmp_active: bool = False
    runtime_library_directory: str | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class NativeGLResult:
    """Structured output from a native or Numba GL history operation."""

    values: np.ndarray
    orders: np.ndarray
    definition: str
    operation: str
    method: str
    step: float | None
    memory_policy: str
    history_window: int | None
    backend: str
    status: str
    build: NativeGLBuildMetadata
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NativeGLWeightsResult:
    """Structured result for recursively generated GL binomial weights."""

    values: np.ndarray
    order: float
    count: int
    backend: str
    status: str
    build: NativeGLBuildMetadata


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _source_sha256() -> str:
    return hashlib.sha256(_C_SOURCE.read_bytes()).hexdigest()


def _native_array_type():
    return np.ctypeslib.ndpointer(
        dtype=np.float64,
        ndim=1,
        flags="C_CONTIGUOUS",
    )


def _validate_definition(definition: str) -> tuple[str, bool]:
    normalized = str(definition).strip().lower()
    allowed = {
        "grunwald_letnikov",
        "riemann_liouville_gl",
        "caputo_shifted",
    }
    if normalized not in allowed:
        raise ValueError(f"definition must be one of {sorted(allowed)}.")
    return normalized, normalized == "caputo_shifted"


def _validate_samples(
    samples: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    array = np.asarray(samples, dtype=np.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(
            "samples must have shape (n_times,) or (n_times, dimension)."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("samples must contain only finite values.")
    normalized_orders = normalize_fractional_orders(orders, array.shape[1])
    return (
        np.ascontiguousarray(array, dtype=np.float64),
        np.ascontiguousarray(normalized_orders, dtype=np.float64),
        was_vector,
    )


def _validate_history_window(history_window: int | None) -> tuple[int, str]:
    if history_window is None:
        return 0, "full_history"
    window = int(history_window)
    if window < 1:
        raise ValueError("history_window must be a positive integer.")
    return window, "finite_window"


@dataclass(slots=True)
class NativeGrunwaldLetnikovBackend:
    """Loaded native GL ABI and its reproducible build metadata."""

    lib: Any
    build_metadata: NativeGLBuildMetadata

    _cache: ClassVar[dict[str, "NativeGrunwaldLetnikovBackend"]] = {}
    _default_backend: ClassVar["NativeGrunwaldLetnikovBackend | None"] = None

    @classmethod
    def build(
        cls,
        output_name: str | None = None,
    ) -> "NativeGrunwaldLetnikovBackend":
        """Compile or load the source-fingerprinted shared library."""

        if output_name is None and cls._default_backend is not None:
            return cls._default_backend
        source_hash = _source_sha256()
        selected_name = output_name or f"grunwald_letnikov_omp_{source_hash[:12]}"
        if selected_name in cls._cache:
            return cls._cache[selected_name]

        output_path = get_native_cache() / f"{selected_name}{_shared_suffix()}"
        result = compile_c_target(
            _C_SOURCE,
            output_path,
            target_kind="shared",
            openmp=True,
        )
        lib, runtime_directory = _load_shared_library(result.path, result.compiler)
        array_type = _native_array_type()

        lib.hafo_gl_abi_version.argtypes = []
        lib.hafo_gl_abi_version.restype = ctypes.c_int
        lib.hafo_gl_kernel_id.argtypes = []
        lib.hafo_gl_kernel_id.restype = ctypes.c_char_p
        lib.hafo_gl_openmp_enabled.argtypes = []
        lib.hafo_gl_openmp_enabled.restype = ctypes.c_int
        lib.hafo_gl_weights.argtypes = [
            ctypes.c_double,
            ctypes.c_size_t,
            array_type,
        ]
        lib.hafo_gl_weights.restype = ctypes.c_int
        lib.hafo_gl_convolution.argtypes = [
            array_type,
            ctypes.c_size_t,
            ctypes.c_size_t,
            array_type,
            ctypes.c_int,
            ctypes.c_size_t,
            array_type,
        ]
        lib.hafo_gl_convolution.restype = ctypes.c_int
        lib.hafo_gl_derivative.argtypes = [
            array_type,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            array_type,
            ctypes.c_int,
            ctypes.c_size_t,
            array_type,
        ]
        lib.hafo_gl_derivative.restype = ctypes.c_int

        abi_version = int(lib.hafo_gl_abi_version())
        if abi_version != _ABI_VERSION:
            raise RuntimeError(
                f"Unsupported native GL ABI {abi_version}; expected {_ABI_VERSION}."
            )
        raw_kernel_id = lib.hafo_gl_kernel_id()
        kernel_id = raw_kernel_id.decode("ascii") if raw_kernel_id else None
        openmp_active = bool(lib.hafo_gl_openmp_enabled())
        metadata = NativeGLBuildMetadata(
            available=True,
            backend="native_c",
            abi_version=abi_version,
            kernel_id=kernel_id,
            source_sha256=source_hash,
            library_path=str(result.path.resolve()),
            compiler=result.compiler,
            compile_command=tuple(str(item) for item in result.command),
            openmp_requested=result.openmp_requested,
            openmp_active=openmp_active,
            runtime_library_directory=runtime_directory,
        )
        backend = cls(lib=lib, build_metadata=metadata)
        cls._cache[selected_name] = backend
        if output_name is None:
            cls._default_backend = backend
        return backend

    @staticmethod
    def _check_status(status_code: int, operation: str) -> None:
        if status_code == 0:
            return
        reason = _STATUS_MESSAGES.get(status_code, f"unknown_status_{status_code}")
        raise NativeGLKernelError(
            f"Native GL {operation} failed with status {status_code} ({reason})."
        )

    def weights(self, order: float, count: int) -> np.ndarray:
        """Return GL weights through the native recurrence."""

        order = float(order)
        count = int(count)
        if not np.isfinite(order) or order <= 0.0 or order > 1.0:
            raise ValueError("order must be finite and lie in (0, 1].")
        if count < 0:
            raise ValueError("count must be non-negative.")
        output = np.empty(count, dtype=np.float64)
        status = int(self.lib.hafo_gl_weights(order, count, output))
        self._check_status(status, "weights")
        return output

    def apply(
        self,
        samples: np.ndarray,
        orders: np.ndarray,
        *,
        shift_initial: bool,
        history_window: int,
        step: float | None,
    ) -> np.ndarray:
        """Apply the C convolution, optionally scaled as a derivative."""

        n_times, dimension = samples.shape
        flattened = np.ascontiguousarray(samples.reshape(-1), dtype=np.float64)
        output = np.empty_like(flattened)
        if step is None:
            status = int(
                self.lib.hafo_gl_convolution(
                    flattened,
                    n_times,
                    dimension,
                    orders,
                    int(shift_initial),
                    history_window,
                    output,
                )
            )
            operation = "convolution"
        else:
            status = int(
                self.lib.hafo_gl_derivative(
                    flattened,
                    n_times,
                    dimension,
                    float(step),
                    orders,
                    int(shift_initial),
                    history_window,
                    output,
                )
            )
            operation = "derivative"
        self._check_status(status, operation)
        return output.reshape(n_times, dimension)


def _resolve_backend(
    fallback: bool,
) -> tuple[NativeGrunwaldLetnikovBackend | None, NativeGLBuildMetadata]:
    try:
        backend = NativeGrunwaldLetnikovBackend.build()
        return backend, backend.build_metadata
    except (OSError, RuntimeError) as exc:
        if not fallback:
            raise NativeGLBackendUnavailable(
                f"Native GL backend is unavailable: {exc}"
            ) from exc
        return None, NativeGLBuildMetadata(
            available=False,
            backend="numba_fallback",
            abi_version=None,
            kernel_id=None,
            source_sha256=_source_sha256(),
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def native_grunwald_letnikov_weights(
    order: float,
    count: int,
    *,
    fallback: bool = True,
) -> NativeGLWeightsResult:
    """Generate GL weights in C, falling back safely to Numba if requested."""

    backend, build = _resolve_backend(fallback)
    if backend is None:
        from .grunwald_letnikov import grunwald_letnikov_weights

        values = grunwald_letnikov_weights(order, count)
        selected_backend = "numba_fallback"
    else:
        values = backend.weights(order, count)
        selected_backend = "native_c"
    return NativeGLWeightsResult(
        values=values,
        order=float(order),
        count=int(count),
        backend=selected_backend,
        status="ok",
        build=build,
    )


def _apply_with_optional_native(
    samples: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    definition: str,
    history_window: int | None,
    step: float | None,
    fallback: bool,
) -> NativeGLResult:
    array, normalized_orders, was_vector = _validate_samples(samples, orders)
    normalized_definition, shift_initial = _validate_definition(definition)
    window_value, memory_policy = _validate_history_window(history_window)
    if step is not None:
        step = float(step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be a finite positive number.")

    backend, build = _resolve_backend(fallback)
    if backend is None:
        from .grunwald_letnikov import grunwald_letnikov_derivative

        fallback_result = grunwald_letnikov_derivative(
            array,
            1.0 if step is None else step,
            normalized_orders,
            definition=normalized_definition,
            history_window=history_window,
        )
        values = fallback_result.values
        selected_backend = "numba_fallback"
        method = (
            "gl_convolution_numba_fallback"
            if step is None
            else "gl_direct_numba_fallback"
        )
    else:
        values = backend.apply(
            array,
            normalized_orders,
            shift_initial=shift_initial,
            history_window=window_value,
            step=step,
        )
        selected_backend = "native_c"
        method = "gl_convolution_native_c" if step is None else "gl_direct_native_c"

    if was_vector:
        values = values[:, 0]
    operation = "convolution" if step is None else "derivative"
    return NativeGLResult(
        values=values,
        orders=normalized_orders,
        definition=normalized_definition,
        operation=operation,
        method=method,
        step=step,
        memory_policy=memory_policy,
        history_window=history_window,
        backend=selected_backend,
        status="finite_numerical_diagnostic",
        build=build,
        metadata={
            "storage_order": "row_major",
            "weight_recurrence": "w[k]=w[k-1]*(1-(order+1)/k)",
            "complexity": "O(dimension*n_times*effective_history)",
            "finite_window_changes_operator": history_window is not None,
        },
    )


def native_grunwald_letnikov_convolution(
    samples: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    definition: str = "grunwald_letnikov",
    history_window: int | None = None,
    fallback: bool = True,
) -> NativeGLResult:
    """Apply the unscaled multicomponent GL history convolution."""

    return _apply_with_optional_native(
        samples,
        orders,
        definition=definition,
        history_window=history_window,
        step=None,
        fallback=fallback,
    )


def native_grunwald_letnikov_derivative(
    samples: np.ndarray,
    step: float,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    definition: str = "grunwald_letnikov",
    history_window: int | None = None,
    fallback: bool = True,
) -> NativeGLResult:
    """Evaluate ``h**(-q)`` times the native GL history convolution."""

    return _apply_with_optional_native(
        samples,
        orders,
        definition=definition,
        history_window=history_window,
        step=step,
        fallback=fallback,
    )


__all__ = [
    "NativeGLBackendUnavailable",
    "NativeGLBuildMetadata",
    "NativeGLKernelError",
    "NativeGLResult",
    "NativeGLWeightsResult",
    "NativeGrunwaldLetnikovBackend",
    "native_grunwald_letnikov_convolution",
    "native_grunwald_letnikov_derivative",
    "native_grunwald_letnikov_weights",
]
