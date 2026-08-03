"""Optional native-C Grassberger--Procaccia correlation counts.

Stability: experimental

The native ABI counts unordered pairs for several radii without a Python
callback in the quadratic distance loop.  Compilation is lazy, source
fingerprinted, and follows HAFO's shared OpenMP policy.  If compilation or DLL
loading is unavailable, the public function can import the Numba reference
late from :mod:`hidden_attractors.analysis.correlation_dimension`.

This module returns raw q=2 pair counts.  Normalization, scaling-region
selection, and dimension estimation remain separate analysis decisions.

Reference
---------
P. Grassberger and I. Procaccia, "Measuring the strangeness of strange
attractors", Physica D 9 (1983), 189--208,
https://doi.org/10.1016/0167-2789(83)90298-1.
"""

from __future__ import annotations

import ctypes
import hashlib
import operator
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..parallel import compile_c_target
from ..paths import PACKAGE_ROOT, get_native_cache


_C_SOURCE = PACKAGE_ROOT / "native" / "csrc" / "correlation_sum.c"
_ABI_VERSION = 1
_KERNEL_ID = "hafo_correlation_sum_q2_v1"
_METRIC_CODES = {
    "euclidean": 0,
    "chebyshev": 1,
    "manhattan": 2,
}
_STATUS_MESSAGES = {
    0: "ok",
    -1: "null_pointer",
    -2: "invalid_shape",
    -3: "invalid_radius",
    -4: "invalid_metric",
    -5: "allocation_failed",
    -6: "nonfinite_input",
    -7: "aliased_buffers",
    -8: "size_overflow",
    -9: "count_overflow",
}


class NativeCorrelationBackendUnavailable(RuntimeError):
    """Raised when native correlation counting was required but unavailable."""


class NativeCorrelationKernelError(RuntimeError):
    """Raised when the native ABI rejects a request or violates its contract."""


@dataclass(frozen=True, slots=True)
class NativeCorrelationBuildMetadata:
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
class NativeCorrelationCountsResult:
    """Raw cumulative q=2 counts and their complete execution contract."""

    counts: np.ndarray
    eligible_pairs: int
    metric: str
    theiler_window: int
    backend: str
    status: str
    build: NativeCorrelationBuildMetadata

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        if counts.ndim != 1 or counts.size < 1 or counts.dtype != np.uint64:
            raise ValueError("counts must be a non-empty one-dimensional uint64 array.")
        immutable = (
            counts
            if counts.flags.c_contiguous and not counts.flags.writeable
            else np.array(counts, dtype=np.uint64, order="C", copy=True)
        )
        immutable.flags.writeable = False
        eligible = int(self.eligible_pairs)
        if eligible < 0:
            raise ValueError("eligible_pairs must be nonnegative.")
        if np.any(immutable[1:] < immutable[:-1]):
            raise ValueError("counts must be cumulative and nondecreasing.")
        if int(immutable[-1]) > eligible:
            raise ValueError("counts cannot exceed eligible_pairs.")
        object.__setattr__(self, "counts", immutable)
        object.__setattr__(self, "eligible_pairs", eligible)


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _source_sha256() -> str:
    return hashlib.sha256(_C_SOURCE.read_bytes()).hexdigest()


def _double_array_type() -> Any:
    return np.ctypeslib.ndpointer(
        dtype=np.float64,
        ndim=1,
        flags="C_CONTIGUOUS",
    )


def _uint64_array_type() -> Any:
    return np.ctypeslib.ndpointer(
        dtype=np.uint64,
        ndim=1,
        flags="C_CONTIGUOUS",
    )


def _load_shared_library(path: Path, compiler: str) -> tuple[Any, str | None]:
    """Load a DLL while making the compiler runtime visible on Windows."""

    runtime_directory: str | None = None
    directory_handle = None
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        compiler_path = shutil.which(compiler)
        if compiler_path is not None:
            runtime_directory = str(Path(compiler_path).resolve().parent)
            directory_handle = os.add_dll_directory(runtime_directory)
    try:
        return ctypes.CDLL(str(path.resolve())), runtime_directory
    finally:
        if directory_handle is not None:
            directory_handle.close()


def _validate_points(points: Any) -> np.ndarray:
    raw = np.asarray(points)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError("points must be real-valued float64-compatible data.")
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("points must be real-valued float64-compatible data.") from exc
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(
            "points must have shape (n_points, dimension) with n_points >= 2."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _validate_radii(radii: Any) -> np.ndarray:
    raw = np.asarray(radii)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError("radii must be real-valued float64-compatible data.")
    try:
        values = np.asarray(radii, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("radii must be real-valued float64-compatible data.") from exc
    if values.ndim != 1 or values.size < 1:
        raise ValueError("radii must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("radii must contain only finite values.")
    if np.any(values <= 0.0) or np.any(values[1:] <= values[:-1]):
        raise ValueError("radii must be positive and strictly increasing.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _validate_theiler_window(theiler_window: Any) -> int:
    if isinstance(theiler_window, (bool, np.bool_)):
        raise TypeError("theiler_window must be a nonnegative integer.")
    try:
        value = operator.index(theiler_window)
    except TypeError as exc:
        raise TypeError("theiler_window must be a nonnegative integer.") from exc
    if value < 0:
        raise ValueError("theiler_window must be a nonnegative integer.")
    return int(value)


def _validate_metric(metric: Any) -> tuple[str, int]:
    normalized = str(metric).strip().lower()
    try:
        return normalized, _METRIC_CODES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"metric must be one of {sorted(_METRIC_CODES)}."
        ) from exc


def _eligible_pairs(n_points: int, theiler_window: int) -> int:
    separated_points = max(0, n_points - theiler_window - 1)
    return separated_points * (separated_points + 1) // 2


@dataclass(slots=True)
class NativeCorrelationSumBackend:
    """Loaded native correlation-count ABI and its build metadata."""

    lib: Any
    build_metadata: NativeCorrelationBuildMetadata

    _cache: ClassVar[dict[str, "NativeCorrelationSumBackend"]] = {}
    _default_backend: ClassVar["NativeCorrelationSumBackend | None"] = None

    @classmethod
    def build(cls, output_name: str | None = None) -> "NativeCorrelationSumBackend":
        """Compile or load the source-fingerprinted shared library."""

        if output_name is None and cls._default_backend is not None:
            return cls._default_backend
        source_hash = _source_sha256()
        selected_name = output_name or f"correlation_sum_omp_{source_hash[:12]}"
        if selected_name in cls._cache:
            return cls._cache[selected_name]

        output_path = get_native_cache() / f"{selected_name}{_shared_suffix()}"
        compile_result = compile_c_target(
            _C_SOURCE,
            output_path,
            target_kind="shared",
            openmp=True,
        )
        lib, runtime_directory = _load_shared_library(
            compile_result.path,
            compile_result.compiler,
        )
        double_array = _double_array_type()
        uint64_array = _uint64_array_type()

        lib.hafo_correlation_abi_version.argtypes = []
        lib.hafo_correlation_abi_version.restype = ctypes.c_int
        lib.hafo_correlation_kernel_id.argtypes = []
        lib.hafo_correlation_kernel_id.restype = ctypes.c_char_p
        lib.hafo_correlation_openmp_enabled.argtypes = []
        lib.hafo_correlation_openmp_enabled.restype = ctypes.c_int
        lib.hafo_correlation_status.argtypes = [ctypes.c_int]
        lib.hafo_correlation_status.restype = ctypes.c_char_p
        lib.hafo_correlation_status_message.argtypes = [ctypes.c_int]
        lib.hafo_correlation_status_message.restype = ctypes.c_char_p
        lib.hafo_correlation_sum_counts.argtypes = [
            double_array,
            ctypes.c_size_t,
            ctypes.c_size_t,
            double_array,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            uint64_array,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.hafo_correlation_sum_counts.restype = ctypes.c_int

        abi_version = int(lib.hafo_correlation_abi_version())
        if abi_version != _ABI_VERSION:
            raise RuntimeError(
                f"Unsupported native correlation ABI {abi_version}; "
                f"expected {_ABI_VERSION}."
            )
        raw_kernel_id = lib.hafo_correlation_kernel_id()
        kernel_id = raw_kernel_id.decode("ascii") if raw_kernel_id else None
        if kernel_id != _KERNEL_ID:
            raise RuntimeError(
                f"Unexpected native correlation kernel_id {kernel_id!r}; "
                f"expected {_KERNEL_ID!r}."
            )
        openmp_active = bool(lib.hafo_correlation_openmp_enabled())
        build_metadata = NativeCorrelationBuildMetadata(
            available=True,
            backend="native_c",
            abi_version=abi_version,
            kernel_id=kernel_id,
            source_sha256=source_hash,
            library_path=str(compile_result.path.resolve()),
            compiler=compile_result.compiler,
            compile_command=tuple(str(item) for item in compile_result.command),
            openmp_requested=compile_result.openmp_requested,
            openmp_active=openmp_active,
            runtime_library_directory=runtime_directory,
        )
        backend = cls(lib=lib, build_metadata=build_metadata)
        cls._cache[selected_name] = backend
        if output_name is None:
            cls._default_backend = backend
        return backend

    def _check_status(self, status_code: int) -> None:
        if status_code == 0:
            return
        raw_message = self.lib.hafo_correlation_status(status_code)
        reason = (
            raw_message.decode("ascii")
            if raw_message
            else _STATUS_MESSAGES.get(status_code, f"unknown_status_{status_code}")
        )
        raise NativeCorrelationKernelError(
            "Native correlation-count kernel failed with "
            f"status {status_code} ({reason})."
        )

    def count(
        self,
        points: np.ndarray,
        radii: np.ndarray,
        *,
        theiler_window: int,
        metric_code: int,
    ) -> tuple[np.ndarray, int]:
        """Count eligible pairs through the native ABI."""

        n_points, dimension = points.shape
        flattened = np.ascontiguousarray(points.reshape(-1), dtype=np.float64)
        counts = np.zeros(radii.size, dtype=np.uint64)
        eligible_pairs = ctypes.c_uint64(0)
        status_code = int(
            self.lib.hafo_correlation_sum_counts(
                flattened,
                n_points,
                dimension,
                radii,
                radii.size,
                theiler_window,
                metric_code,
                counts,
                ctypes.byref(eligible_pairs),
            )
        )
        self._check_status(status_code)
        expected_eligible = _eligible_pairs(n_points, theiler_window)
        if int(eligible_pairs.value) != expected_eligible:
            raise NativeCorrelationKernelError(
                "Native eligible-pair count violated the ABI invariant: "
                f"{eligible_pairs.value} != {expected_eligible}."
            )
        if np.any(counts[1:] < counts[:-1]) or int(counts[-1]) > expected_eligible:
            raise NativeCorrelationKernelError(
                "Native cumulative counts violated monotonicity or eligibility bounds."
            )
        return counts, expected_eligible


def _resolve_backend(
    fallback: bool,
) -> tuple[NativeCorrelationSumBackend | None, NativeCorrelationBuildMetadata]:
    try:
        backend = NativeCorrelationSumBackend.build()
        return backend, backend.build_metadata
    except (OSError, RuntimeError) as exc:
        if not fallback:
            raise NativeCorrelationBackendUnavailable(
                f"Native correlation-count backend is unavailable: {exc}"
            ) from exc
        return None, NativeCorrelationBuildMetadata(
            available=False,
            backend="numba_fallback",
            abi_version=None,
            kernel_id=None,
            source_sha256=_source_sha256(),
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def native_correlation_sum_counts(
    points: Any,
    radii: Any,
    *,
    theiler_window: int = 0,
    metric: str = "euclidean",
    fallback: bool = True,
) -> NativeCorrelationCountsResult:
    """Return strict-radius Grassberger--Procaccia q=2 pair counts.

    For every supplied radius this counts pairs ``i < j`` satisfying both
    ``j - i > theiler_window`` and ``distance(points[i], points[j]) < radius``.
    Counts are cumulative because radii must be strictly increasing.  They are
    not normalized into a correlation sum by this low-level API.
    """

    if not isinstance(fallback, (bool, np.bool_)):
        raise TypeError("fallback must be Boolean.")
    point_values = _validate_points(points)
    radius_values = _validate_radii(radii)
    window = _validate_theiler_window(theiler_window)
    metric_name, metric_code = _validate_metric(metric)
    backend, build = _resolve_backend(bool(fallback))

    if backend is None:
        from .correlation_dimension import _correlation_counts_numba

        counts = np.asarray(
            _correlation_counts_numba(
                point_values,
                radius_values,
                window,
                metric_code,
            ),
            dtype=np.uint64,
        )
        if counts.shape != radius_values.shape:
            raise RuntimeError(
                "Numba correlation fallback returned an unexpected count shape."
            )
        eligible_pairs = _eligible_pairs(point_values.shape[0], window)
        selected_backend = "numba_fallback"
    else:
        counts, eligible_pairs = backend.count(
            point_values,
            radius_values,
            theiler_window=window,
            metric_code=metric_code,
        )
        selected_backend = "native_c"

    return NativeCorrelationCountsResult(
        counts=counts,
        eligible_pairs=eligible_pairs,
        metric=metric_name,
        theiler_window=window,
        backend=selected_backend,
        status="ok",
        build=build,
    )


__all__ = [
    "NativeCorrelationBackendUnavailable",
    "NativeCorrelationBuildMetadata",
    "NativeCorrelationCountsResult",
    "NativeCorrelationKernelError",
    "NativeCorrelationSumBackend",
    "native_correlation_sum_counts",
]
