"""Optional native-C ordinal-pattern counts for permutation entropy.

Stability: experimental

The native ABI constructs forward delay-embedding windows, maps their ordinal
patterns to zero-based lexicographic Lehmer ranks, and returns the raw uint64
histogram.  Entropy normalization and interpretation remain analysis-layer
decisions.  Compilation is lazy and source-fingerprinted, following HAFO's
shared OpenMP policy.  If the compiler or shared-library loader is unavailable,
the wrapper can import the Numba kernel from :mod:`.permutation_entropy` late.

Reference
---------
C. Bandt and B. Pompe, "Permutation Entropy: A Natural Complexity Measure for
Time Series", Physical Review Letters 88 (2002), 174102,
https://doi.org/10.1103/PhysRevLett.88.174102.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import operator
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numba import njit

from ..parallel import compile_c_target
from ..paths import PACKAGE_ROOT, get_native_cache


_C_SOURCE = PACKAGE_ROOT / "native" / "csrc" / "permutation_entropy.c"
_ABI_VERSION = 1
_KERNEL_ID = "hafo_permutation_entropy_counts_v1"
_MIN_EMBEDDING_DIMENSION = 2
_MAX_EMBEDDING_DIMENSION = 10
_TIE_POLICY_CODES = {
    "stable_index": 0,
    "omit": 1,
    "raise": 2,
}
_STATUS_MESSAGES = {
    0: "ok",
    -1: "null_pointer",
    -2: "invalid_shape",
    -3: "invalid_embedding_dimension",
    -4: "invalid_delay",
    -5: "invalid_tie_policy",
    -6: "invalid_counts_length",
    -7: "nonfinite_input",
    -8: "aliased_buffers",
    -9: "size_overflow",
    -10: "count_overflow",
    -11: "tied_window",
}


class NativePermutationBackendUnavailable(RuntimeError):
    """Raised when native ordinal counting was required but unavailable."""


class NativePermutationKernelError(RuntimeError):
    """Raised when the native ordinal-count ABI violates its contract."""


class NativePermutationTieError(NativePermutationKernelError, ValueError):
    """Raised when ``tie_policy='raise'`` encounters an ordinal tie."""

    def __init__(
        self,
        *,
        total_windows: int,
        valid_windows: int,
        tied_windows: int,
        first_tied_window: int | None = None,
    ) -> None:
        self.total_windows = int(total_windows)
        self.valid_windows = int(valid_windows)
        self.tied_windows = int(tied_windows)
        self.first_tied_window = (
            None if first_tied_window is None else int(first_tied_window)
        )
        location = (
            ""
            if self.first_tied_window is None
            else f"; first_tied_window={self.first_tied_window}"
        )
        super().__init__(
            "tie_policy='raise' rejected an ordinal window containing equal "
            f"values (tied_windows={self.tied_windows}{location})."
        )


@dataclass(frozen=True, slots=True)
class NativePermutationBuildMetadata:
    """Build and provenance metadata for one native or fallback execution."""

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
class NativePermutationCountsResult:
    """Raw ordinal-pattern counts and the complete embedding contract."""

    counts: np.ndarray
    embedding_dimension: int
    delay: int
    tie_policy: str
    total_windows: int
    valid_windows: int
    tied_windows: int
    backend: str
    status: str
    build: NativePermutationBuildMetadata

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        embedding_dimension = int(self.embedding_dimension)
        delay = int(self.delay)
        total_windows = int(self.total_windows)
        valid_windows = int(self.valid_windows)
        tied_windows = int(self.tied_windows)
        expected_patterns = math.factorial(embedding_dimension)

        if (
            counts.ndim != 1
            or counts.dtype != np.uint64
            or counts.size != expected_patterns
        ):
            raise ValueError(
                "counts must be a one-dimensional uint64 array of length m!."
            )
        immutable = (
            counts
            if counts.flags.c_contiguous and not counts.flags.writeable
            else np.array(counts, dtype=np.uint64, order="C", copy=True)
        )
        immutable.flags.writeable = False
        if not (
            _MIN_EMBEDDING_DIMENSION
            <= embedding_dimension
            <= _MAX_EMBEDDING_DIMENSION
        ):
            raise ValueError("embedding_dimension must be between 2 and 10.")
        if delay < 1:
            raise ValueError("delay must be a positive integer.")
        if self.tie_policy not in _TIE_POLICY_CODES:
            raise ValueError("tie_policy is not a supported canonical token.")
        if total_windows < 1:
            raise ValueError("total_windows must be positive.")
        if not (0 <= valid_windows <= total_windows):
            raise ValueError("valid_windows must lie between zero and total_windows.")
        if not (0 <= tied_windows <= total_windows):
            raise ValueError("tied_windows must lie between zero and total_windows.")
        if sum(int(value) for value in immutable) != valid_windows:
            raise ValueError("sum(counts) must equal valid_windows.")
        if self.tie_policy == "stable_index" and valid_windows != total_windows:
            raise ValueError("stable_index must count every embedding window.")
        if self.tie_policy == "omit" and valid_windows + tied_windows != total_windows:
            raise ValueError("omit must partition windows into valid and tied counts.")
        object.__setattr__(self, "counts", immutable)
        object.__setattr__(self, "embedding_dimension", embedding_dimension)
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "total_windows", total_windows)
        object.__setattr__(self, "valid_windows", valid_windows)
        object.__setattr__(self, "tied_windows", tied_windows)


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
    """Load a DLL while making a MinGW compiler runtime visible on Windows."""

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


def _validate_signal(signal: Any) -> np.ndarray:
    raw = np.asarray(signal)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError("signal must be real-valued float64-compatible data.")
    try:
        values = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("signal must be real-valued float64-compatible data.") from exc
    if values.ndim != 1 or values.size < 1:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _validate_index(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    try:
        validated = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if validated < minimum or (maximum is not None and validated > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}.")
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return int(validated)


def _validate_tie_policy(tie_policy: Any) -> tuple[str, int]:
    if not isinstance(tie_policy, str):
        raise TypeError("tie_policy must be a string token.")
    try:
        return tie_policy, _TIE_POLICY_CODES[tie_policy]
    except KeyError as exc:
        raise ValueError(
            "tie_policy must be 'stable_index', 'omit', or 'raise'."
        ) from exc


def _window_count(n_samples: int, embedding_dimension: int, delay: int) -> int:
    total = n_samples - (embedding_dimension - 1) * delay
    if total < 1:
        raise ValueError(
            "signal is too short for embedding_dimension and delay; at least "
            "one forward ordinal window is required."
        )
    return total


@njit(cache=True)
def _ordinal_pattern_counts_numba_fallback(
    signal: np.ndarray,
    embedding_dimension: int,
    delay: int,
    tie_policy_code: int,
) -> tuple[np.ndarray, int, int]:
    """Self-contained Numba fallback with the same Lehmer/tie contract."""

    pattern_count = 1
    for factor in range(2, embedding_dimension + 1):
        pattern_count *= factor
    counts = np.zeros(pattern_count, dtype=np.uint64)
    total_windows = signal.size - (embedding_dimension - 1) * delay
    tied_windows = 0
    first_tied_window = -1

    # Match the native raise policy: inspect all ties before producing counts.
    if tie_policy_code == 2:
        for start in range(total_windows):
            tied = False
            for left in range(embedding_dimension):
                left_value = signal[start + left * delay]
                for right in range(left + 1, embedding_dimension):
                    if left_value == signal[start + right * delay]:
                        tied = True
                        break
                if tied:
                    break
            if tied:
                if first_tied_window < 0:
                    first_tied_window = start
                tied_windows += 1
        if first_tied_window >= 0:
            return counts, tied_windows, first_tied_window

    for start in range(total_windows):
        tied = False
        for left in range(embedding_dimension):
            left_value = signal[start + left * delay]
            for right in range(left + 1, embedding_dimension):
                if left_value == signal[start + right * delay]:
                    tied = True
                    break
            if tied:
                break
        if tied:
            tied_windows += 1
            if tie_policy_code == 1:
                continue

        permutation = np.empty(embedding_dimension, dtype=np.int64)
        for index in range(embedding_dimension):
            permutation[index] = index
        for index in range(1, embedding_dimension):
            key = permutation[index]
            key_value = signal[start + key * delay]
            position = index
            while position > 0:
                previous = permutation[position - 1]
                previous_value = signal[start + previous * delay]
                key_precedes = key_value < previous_value or (
                    key_value == previous_value and key < previous
                )
                if not key_precedes:
                    break
                permutation[position] = previous
                position -= 1
            permutation[position] = key

        rank = 0
        for index in range(embedding_dimension - 1):
            smaller_to_right = 0
            for later in range(index + 1, embedding_dimension):
                if permutation[later] < permutation[index]:
                    smaller_to_right += 1
            factorial_weight = 1
            for factor in range(2, embedding_dimension - index):
                factorial_weight *= factor
            rank += smaller_to_right * factorial_weight
        counts[rank] += np.uint64(1)
    return counts, tied_windows, first_tied_window


@dataclass(slots=True)
class NativePermutationEntropyBackend:
    """Loaded native ordinal-pattern ABI and its build metadata."""

    lib: Any
    build_metadata: NativePermutationBuildMetadata

    _cache: ClassVar[dict[str, "NativePermutationEntropyBackend"]] = {}
    _default_backend: ClassVar["NativePermutationEntropyBackend | None"] = None

    @classmethod
    def build(
        cls,
        output_name: str | None = None,
    ) -> "NativePermutationEntropyBackend":
        """Compile or load the source-fingerprinted shared library."""

        if output_name is None and cls._default_backend is not None:
            return cls._default_backend
        source_hash = _source_sha256()
        selected_name = output_name or f"permutation_entropy_omp_{source_hash[:12]}"
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

        lib.hafo_permutation_abi_version.argtypes = []
        lib.hafo_permutation_abi_version.restype = ctypes.c_int
        lib.hafo_permutation_kernel_id.argtypes = []
        lib.hafo_permutation_kernel_id.restype = ctypes.c_char_p
        lib.hafo_permutation_openmp_enabled.argtypes = []
        lib.hafo_permutation_openmp_enabled.restype = ctypes.c_int
        lib.hafo_permutation_max_embedding_dimension.argtypes = []
        lib.hafo_permutation_max_embedding_dimension.restype = ctypes.c_size_t
        lib.hafo_permutation_pattern_count.argtypes = [ctypes.c_size_t]
        lib.hafo_permutation_pattern_count.restype = ctypes.c_uint64
        lib.hafo_permutation_status.argtypes = [ctypes.c_int]
        lib.hafo_permutation_status.restype = ctypes.c_char_p
        lib.hafo_permutation_status_message.argtypes = [ctypes.c_int]
        lib.hafo_permutation_status_message.restype = ctypes.c_char_p
        lib.hafo_permutation_entropy_counts.argtypes = [
            double_array,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            uint64_array,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.hafo_permutation_entropy_counts.restype = ctypes.c_int

        abi_version = int(lib.hafo_permutation_abi_version())
        if abi_version != _ABI_VERSION:
            raise RuntimeError(
                f"Unsupported native permutation ABI {abi_version}; "
                f"expected {_ABI_VERSION}."
            )
        raw_kernel_id = lib.hafo_permutation_kernel_id()
        kernel_id = raw_kernel_id.decode("ascii") if raw_kernel_id else None
        if kernel_id != _KERNEL_ID:
            raise RuntimeError(
                f"Unexpected native permutation kernel_id {kernel_id!r}; "
                f"expected {_KERNEL_ID!r}."
            )
        maximum = int(lib.hafo_permutation_max_embedding_dimension())
        if maximum != _MAX_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Native maximum embedding dimension {maximum} does not match "
                f"the Python contract {_MAX_EMBEDDING_DIMENSION}."
            )
        for dimension in range(_MIN_EMBEDDING_DIMENSION, maximum + 1):
            reported = int(lib.hafo_permutation_pattern_count(dimension))
            expected = math.factorial(dimension)
            if reported != expected:
                raise RuntimeError(
                    "Native factorial table violated the ABI invariant for "
                    f"m={dimension}: {reported} != {expected}."
                )

        build_metadata = NativePermutationBuildMetadata(
            available=True,
            backend="native_c",
            abi_version=abi_version,
            kernel_id=kernel_id,
            source_sha256=source_hash,
            library_path=str(compile_result.path.resolve()),
            compiler=compile_result.compiler,
            compile_command=tuple(str(item) for item in compile_result.command),
            openmp_requested=compile_result.openmp_requested,
            openmp_active=bool(lib.hafo_permutation_openmp_enabled()),
            runtime_library_directory=runtime_directory,
        )
        backend = cls(lib=lib, build_metadata=build_metadata)
        cls._cache[selected_name] = backend
        if output_name is None:
            cls._default_backend = backend
        return backend

    def _status_reason(self, status_code: int) -> str:
        raw_message = self.lib.hafo_permutation_status(status_code)
        return (
            raw_message.decode("ascii")
            if raw_message
            else _STATUS_MESSAGES.get(status_code, f"unknown_status_{status_code}")
        )

    def count(
        self,
        signal: np.ndarray,
        *,
        embedding_dimension: int,
        delay: int,
        tie_policy_code: int,
    ) -> tuple[np.ndarray, int, int, int]:
        """Count ordinal patterns through the native ABI."""

        counts = np.zeros(math.factorial(embedding_dimension), dtype=np.uint64)
        total_windows = ctypes.c_uint64(0)
        valid_windows = ctypes.c_uint64(0)
        tied_windows = ctypes.c_uint64(0)
        status_code = int(
            self.lib.hafo_permutation_entropy_counts(
                signal,
                signal.size,
                embedding_dimension,
                delay,
                tie_policy_code,
                counts,
                counts.size,
                ctypes.byref(total_windows),
                ctypes.byref(valid_windows),
                ctypes.byref(tied_windows),
            )
        )
        total = int(total_windows.value)
        valid = int(valid_windows.value)
        tied = int(tied_windows.value)
        if status_code == -11:
            raise NativePermutationTieError(
                total_windows=total,
                valid_windows=valid,
                tied_windows=tied,
            )
        if status_code != 0:
            raise NativePermutationKernelError(
                "Native ordinal-count kernel failed with "
                f"status {status_code} ({self._status_reason(status_code)})."
            )

        expected_total = _window_count(
            signal.size,
            embedding_dimension,
            delay,
        )
        if total != expected_total:
            raise NativePermutationKernelError(
                "Native total-window count violated the ABI invariant: "
                f"{total} != {expected_total}."
            )
        if valid != sum(int(value) for value in counts):
            raise NativePermutationKernelError(
                "Native valid-window count does not equal sum(counts)."
            )
        if not (0 <= tied <= total and 0 <= valid <= total):
            raise NativePermutationKernelError(
                "Native window counts violated total-window bounds."
            )
        return counts, total, valid, tied


def _resolve_backend(
    fallback: bool,
) -> tuple[NativePermutationEntropyBackend | None, NativePermutationBuildMetadata]:
    try:
        backend = NativePermutationEntropyBackend.build()
        return backend, backend.build_metadata
    except (OSError, RuntimeError) as exc:
        if not fallback:
            raise NativePermutationBackendUnavailable(
                f"Native permutation-count backend is unavailable: {exc}"
            ) from exc
        return None, NativePermutationBuildMetadata(
            available=False,
            backend="numba_fallback",
            abi_version=None,
            kernel_id=None,
            source_sha256=_source_sha256(),
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def native_permutation_counts(
    signal: Any,
    *,
    embedding_dimension: int,
    delay: int = 1,
    tie_policy: str = "stable_index",
    fallback: bool = True,
) -> NativePermutationCountsResult:
    """Return raw ordinal-pattern counts indexed by lexicographic Lehmer rank.

    Windows are forward embeddings ``signal[s + k * delay]``.  Under
    ``stable_index``, equal values retain increasing original-index order;
    ``omit`` excludes tied windows; and ``raise`` raises
    :class:`NativePermutationTieError` if any window contains equal values.
    """

    if not isinstance(fallback, (bool, np.bool_)):
        raise TypeError("fallback must be Boolean.")
    values = _validate_signal(signal)
    dimension = _validate_index(
        embedding_dimension,
        name="embedding_dimension",
        minimum=_MIN_EMBEDDING_DIMENSION,
        maximum=_MAX_EMBEDDING_DIMENSION,
    )
    lag = _validate_index(delay, name="delay", minimum=1)
    policy_name, policy_code = _validate_tie_policy(tie_policy)
    expected_total = _window_count(values.size, dimension, lag)
    backend, build = _resolve_backend(bool(fallback))

    if backend is None:
        counts_raw, tied_raw, first_tied_raw = (
            _ordinal_pattern_counts_numba_fallback(
            values,
            dimension,
            lag,
            policy_code,
            )
        )
        counts = np.asarray(counts_raw, dtype=np.uint64)
        tied = int(tied_raw)
        first_tied = int(first_tied_raw)
        valid = sum(int(value) for value in counts)
        if policy_code == _TIE_POLICY_CODES["raise"] and first_tied >= 0:
            raise NativePermutationTieError(
                total_windows=expected_total,
                valid_windows=expected_total - tied,
                tied_windows=tied,
                first_tied_window=first_tied,
            )
        total = expected_total
        selected_backend = "numba_fallback"
    else:
        counts, total, valid, tied = backend.count(
            values,
            embedding_dimension=dimension,
            delay=lag,
            tie_policy_code=policy_code,
        )
        selected_backend = "native_c"

    return NativePermutationCountsResult(
        counts=counts,
        embedding_dimension=dimension,
        delay=lag,
        tie_policy=policy_name,
        total_windows=total,
        valid_windows=valid,
        tied_windows=tied,
        backend=selected_backend,
        status="ok",
        build=build,
    )


# A longer alias keeps the low-level operation discoverable without changing
# the concise name consumed by the public analysis layer.
native_permutation_entropy_counts = native_permutation_counts


__all__ = [
    "NativePermutationBackendUnavailable",
    "NativePermutationBuildMetadata",
    "NativePermutationCountsResult",
    "NativePermutationEntropyBackend",
    "NativePermutationKernelError",
    "NativePermutationTieError",
    "native_permutation_counts",
    "native_permutation_entropy_counts",
]
