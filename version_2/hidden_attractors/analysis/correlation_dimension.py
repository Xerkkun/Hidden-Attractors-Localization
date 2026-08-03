"""Correlation-sum curves and explicit correlation-dimension fits.

Stability: experimental

The routines in this module consume finite sampled point sets.  They can be
applied to trajectories produced by integer- or fractional-order solvers, but
the result describes only the supplied projection or delay reconstruction.  In
particular, a projected fractional trajectory need not be a Markovian state of
the hereditary problem.

The implementation is independent of FractalDimensions.jl and pynamicalsys.
Their public APIs informed the capability comparison, while the mathematical
contract below follows the primary Grassberger--Procaccia definition and the
temporal-correlation exclusion discussed by Theiler.  A scaling interval is
never selected silently: callers must provide ``fit_radius_range`` explicitly.

References
----------
P. Grassberger and I. Procaccia, "Measuring the Strangeness of Strange
Attractors," Physica D 9 (1983), 189--208.
https://doi.org/10.1016/0167-2789(83)90298-1

J. Theiler, "Spurious Dimension from Correlation Algorithms Applied to Limited
Time-Series Data," Physical Review A 34 (1986), 2427--2432.
https://doi.org/10.1103/PhysRevA.34.2427

V. Deshmukh, E. Bradley, J. Garland, and J. D. Meiss, "Toward Automated
Extraction and Characterization of Scaling Regions in Dynamical Systems,"
Chaos 31 (2021), 123102.  https://doi.org/10.1063/5.0069365
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from operator import index as operator_index
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numba import njit


Metric = Literal["euclidean", "chebyshev", "manhattan"]
Backend = Literal["auto", "native_c", "numba", "python"]

CORRELATION_DIMENSION_REFERENCE_DOIS = (
    "10.1016/0167-2789(83)90298-1",
    "10.1103/PhysRevA.34.2427",
    "10.1063/5.0069365",
)
CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS = 131_072
CORRELATION_DIMENSION_EVIDENCE_SCOPE = "finite_sample_empirical_trajectory_diagnostic"
FRACTIONAL_STATE_CAVEAT = (
    "A projected fractional trajectory need not be a Markovian state because "
    "the governing derivative retains history; the supplied coordinates are "
    "not the complete hereditary state. D2 characterizes only the supplied "
    "projection or reconstruction."
)

_METRIC_CODES: dict[str, int] = {
    "euclidean": 0,
    "chebyshev": 1,
    "manhattan": 2,
}


def _readonly_array(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class CorrelationSumResult:
    """Exact finite correlation counts and normalized correlation sums."""

    radii: np.ndarray
    counts: np.ndarray
    correlation_sums: np.ndarray
    eligible_pairs: int
    sample_count: int
    feature_dimension: int
    theiler_window: int
    metric: str
    requested_backend: str
    backend: str
    sampling: str
    projection: str
    status: str
    evidence_scope: str = CORRELATION_DIMENSION_EVIDENCE_SCOPE
    fractional_state_caveat: str = FRACTIONAL_STATE_CAVEAT
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "radii", _readonly_array(self.radii, np.float64))
        object.__setattr__(self, "counts", _readonly_array(self.counts, np.uint64))
        object.__setattr__(
            self,
            "correlation_sums",
            _readonly_array(self.correlation_sums, np.float64),
        )
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CorrelationDimensionResult:
    """An explicitly selected log--log regression over a correlation curve."""

    curve: CorrelationSumResult
    slope: float
    intercept: float
    r_squared: float
    regression_standard_error: float
    fit_indices: np.ndarray
    log_radii: np.ndarray
    log_correlation_sums: np.ndarray
    local_slope_radii: np.ndarray
    local_slopes: np.ndarray
    fit_radius_range: tuple[float, float]
    minimum_points: int
    status: str
    evidence_scope: str = CORRELATION_DIMENSION_EVIDENCE_SCOPE
    fractional_state_caveat: str = FRACTIONAL_STATE_CAVEAT
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_indices", _readonly_array(self.fit_indices, np.int64))
        object.__setattr__(self, "log_radii", _readonly_array(self.log_radii, np.float64))
        object.__setattr__(
            self,
            "log_correlation_sums",
            _readonly_array(self.log_correlation_sums, np.float64),
        )
        object.__setattr__(
            self,
            "local_slope_radii",
            _readonly_array(self.local_slope_radii, np.float64),
        )
        object.__setattr__(self, "local_slopes", _readonly_array(self.local_slopes, np.float64))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))


def _as_points(points: np.ndarray) -> np.ndarray:
    raw = np.asarray(points)
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("points must not have Boolean dtype.")
    if np.iscomplexobj(raw):
        raise TypeError("points must be real-valued, not complex.")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("points must be a real numeric array.") from exc
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("points must have shape (n_samples,) or (n_samples, n_features).")
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("points must contain at least two non-empty sampled states.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_radii(radii: np.ndarray) -> np.ndarray:
    raw = np.asarray(radii)
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("radii must not have Boolean dtype.")
    if np.iscomplexobj(raw):
        raise TypeError("radii must be real-valued, not complex.")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("radii must be a real numeric array.") from exc
    if values.ndim != 1 or values.size < 1:
        raise ValueError("radii must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("radii must contain finite strictly positive values.")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("radii must be strictly increasing without duplicates.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _strict_integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not Boolean.")
    try:
        result = int(operator_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _normalize_metric(metric: str) -> tuple[str, int]:
    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")
    normalized = metric.strip()
    if normalized not in _METRIC_CODES:
        raise ValueError("metric must be 'euclidean', 'chebyshev', or 'manhattan'.")
    return normalized, _METRIC_CODES[normalized]


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string.")
    normalized = backend.strip()
    if normalized not in {"auto", "native_c", "numba", "python"}:
        raise ValueError("backend must be 'auto', 'native_c', 'numba', or 'python'.")
    return normalized


def _provenance_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty provenance description.")
    return value.strip()


def _eligible_pair_count(sample_count: int, theiler_window: int) -> int:
    remaining = sample_count - theiler_window
    return remaining * (remaining - 1) // 2


@njit(cache=True, nogil=True)
def _correlation_counts_numba(
    points: np.ndarray,
    radii: np.ndarray,
    theiler_window: int,
    metric_code: int,
) -> np.ndarray:
    """Count pairs in all radii with one binary search per eligible pair."""

    sample_count, dimension = points.shape
    radius_count = radii.size
    first_radius_bins = np.zeros(radius_count, dtype=np.uint64)
    for first in range(sample_count):
        start = first + theiler_window + 1
        for second in range(start, sample_count):
            if metric_code == 1:
                distance = 0.0
                for coordinate in range(dimension):
                    difference = abs(points[first, coordinate] - points[second, coordinate])
                    if difference > distance:
                        distance = difference
            else:
                distance = 0.0
                for coordinate in range(dimension):
                    difference = abs(points[first, coordinate] - points[second, coordinate])
                    if metric_code == 0:
                        distance += difference * difference
                    else:
                        distance += difference
                if metric_code == 0:
                    distance = np.sqrt(distance)

            lower = 0
            upper = radius_count
            while lower < upper:
                middle = (lower + upper) // 2
                if distance < radii[middle]:
                    upper = middle
                else:
                    lower = middle + 1
            if lower < radius_count:
                first_radius_bins[lower] += np.uint64(1)

    counts = np.empty(radius_count, dtype=np.uint64)
    running = np.uint64(0)
    for radius_index in range(radius_count):
        running += first_radius_bins[radius_index]
        counts[radius_index] = running
    return counts


def _correlation_counts_python(
    points: np.ndarray,
    radii: np.ndarray,
    theiler_window: int,
    metric_code: int,
) -> np.ndarray:
    """Transparent reference implementation of the finite pair count."""

    bins = np.zeros(radii.size, dtype=np.uint64)
    sample_count, dimension = points.shape
    for first in range(sample_count):
        for second in range(first + theiler_window + 1, sample_count):
            if metric_code == 0:
                distance = float(np.sqrt(np.sum((points[first] - points[second]) ** 2)))
            elif metric_code == 1:
                distance = float(np.max(np.abs(points[first] - points[second])))
            else:
                distance = float(np.sum(np.abs(points[first] - points[second])))
            first_radius = int(np.searchsorted(radii, distance, side="right"))
            if first_radius < radii.size:
                bins[first_radius] += np.uint64(1)
    return np.cumsum(bins, dtype=np.uint64)


def _serializable_native_build(build: object) -> object:
    if build is None:
        return None
    if is_dataclass(build) and not isinstance(build, type):
        return asdict(build)
    if isinstance(build, Mapping):
        return dict(build)
    return repr(build)


def _native_counts(
    points: np.ndarray,
    radii: np.ndarray,
    theiler_window: int,
    metric: str,
    *,
    fallback: bool,
) -> tuple[np.ndarray, str, Mapping[str, object]]:
    try:
        from .native_correlation_sum import native_correlation_sum_counts

        result = native_correlation_sum_counts(
            points,
            radii,
            theiler_window=theiler_window,
            metric=metric,
            fallback=fallback,
        )
        return (
            np.asarray(result.counts, dtype=np.uint64),
            str(result.backend),
            {
                "native_status": result.status,
                "native_build": _serializable_native_build(result.build),
            },
        )
    except (ImportError, OSError, RuntimeError) as exc:
        if not fallback:
            raise
        return (
            _correlation_counts_numba(
                points, radii, theiler_window, _METRIC_CODES[metric]
            ),
            "numba_fallback",
            {"fallback_reason": f"{type(exc).__name__}: {exc}"},
        )


def correlation_sum_curve(
    points: np.ndarray,
    radii: np.ndarray,
    *,
    theiler_window: int = 0,
    metric: str = "euclidean",
    backend: str = "auto",
    fallback: bool = True,
    sampling: str = "samples supplied in row order",
    projection: str = "coordinates supplied",
) -> CorrelationSumResult:
    """Return exact finite ``q=2`` correlation sums for increasing radii.

    Only unordered pairs ``i < j`` satisfying ``j - i > theiler_window`` are
    eligible.  A pair is counted at radius ``r`` only when its distance is
    strictly smaller than ``r``.  Thus

    ``C(r) = count(r) / ((N-w)*(N-w-1)/2)``.
    """

    values = _as_points(points)
    radius_values = _as_radii(radii)
    window = _strict_integer(theiler_window, "theiler_window", minimum=0)
    if window > values.shape[0] - 2:
        raise ValueError("theiler_window leaves no eligible unordered point pairs.")
    metric_name, metric_code = _normalize_metric(metric)
    requested_backend = _normalize_backend(backend)
    if not isinstance(fallback, (bool, np.bool_)):
        raise TypeError("fallback must be Boolean.")
    fallback_value = bool(fallback)
    sampling_text = _provenance_text(sampling, "sampling")
    projection_text = _provenance_text(projection, "projection")
    eligible_pairs = _eligible_pair_count(values.shape[0], window)

    backend_metadata: dict[str, object] = {
        "algorithm": "unordered_pair_binary_search_difference_bins",
        "strict_radius_comparison": True,
        "normalization": "eligible_unordered_pairs_after_theiler_exclusion",
        "complexity": "O(eligible_pairs*(feature_dimension+log2(n_radii)))",
        "working_memory": "O(n_radii) excluding native thread-local bins",
        "references": CORRELATION_DIMENSION_REFERENCE_DOIS,
        "sampling": sampling_text,
        "projection": projection_text,
        "evidence_boundary": (
            "finite empirical trajectory diagnostic; not proof of chaos or "
            "hiddenness and not the dimension of a complete fractional "
            "hereditary state"
        ),
        "fractional_state_caveat": FRACTIONAL_STATE_CAVEAT,
    }
    if requested_backend == "python":
        counts = _correlation_counts_python(values, radius_values, window, metric_code)
        actual_backend = "python"
    elif requested_backend == "numba":
        counts = _correlation_counts_numba(values, radius_values, window, metric_code)
        actual_backend = "numba"
    elif requested_backend == "native_c":
        counts, actual_backend, native_metadata = _native_counts(
            values,
            radius_values,
            window,
            metric_name,
            fallback=fallback_value,
        )
        backend_metadata.update(native_metadata)
    elif eligible_pairs >= CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS:
        counts, actual_backend, native_metadata = _native_counts(
            values,
            radius_values,
            window,
            metric_name,
            fallback=fallback_value,
        )
        backend_metadata.update(native_metadata)
        backend_metadata["auto_policy"] = "native_c_above_pair_threshold"
    else:
        counts = _correlation_counts_numba(values, radius_values, window, metric_code)
        actual_backend = "numba"
        backend_metadata["auto_policy"] = "numba_below_pair_threshold"
    backend_metadata["native_auto_min_pairs"] = CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS
    backend_metadata["eligible_pairs"] = eligible_pairs

    counts = np.asarray(counts, dtype=np.uint64)
    if counts.shape != radius_values.shape:
        raise RuntimeError("correlation-count backend returned an invalid shape.")
    if np.any(counts > np.uint64(eligible_pairs)) or np.any(counts[1:] < counts[:-1]):
        raise RuntimeError("correlation-count backend violated finite-count invariants.")
    correlation_sums = counts.astype(np.float64) / float(eligible_pairs)
    return CorrelationSumResult(
        radii=radius_values,
        counts=counts,
        correlation_sums=correlation_sums,
        eligible_pairs=eligible_pairs,
        sample_count=int(values.shape[0]),
        feature_dimension=int(values.shape[1]),
        theiler_window=window,
        metric=metric_name,
        requested_backend=requested_backend,
        backend=actual_backend,
        sampling=sampling_text,
        projection=projection_text,
        status="finite_sample_diagnostic",
        metadata=backend_metadata,
    )


def _fit_range(value: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        raw = value
    else:
        try:
            raw = np.asarray(tuple(value))
        except (TypeError, ValueError) as exc:
            raise TypeError("fit_radius_range must contain exactly two real values.") from exc
    if raw.ndim != 1 or raw.size != 2:
        raise ValueError("fit_radius_range must contain exactly two values.")
    try:
        original_values = tuple(value)
    except TypeError as exc:
        raise TypeError("fit_radius_range must contain exactly two real values.") from exc
    if any(isinstance(item, (bool, np.bool_)) for item in original_values):
        raise TypeError("fit_radius_range must contain real non-Boolean values.")
    if np.issubdtype(raw.dtype, np.bool_) or np.iscomplexobj(raw):
        raise TypeError("fit_radius_range must contain real non-Boolean values.")
    try:
        bounds = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("fit_radius_range must contain real numeric values.") from exc
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise ValueError("fit_radius_range must satisfy 0 < lower < upper with finite bounds.")
    return lower, upper


def _local_slopes(curve: CorrelationSumResult) -> tuple[np.ndarray, np.ndarray]:
    radii = curve.radii
    sums = curve.correlation_sums
    centers = np.sqrt(radii[:-1] * radii[1:])
    slopes = np.full(max(0, radii.size - 1), np.nan, dtype=np.float64)
    for index in range(slopes.size):
        if 0.0 < sums[index] < 1.0 and 0.0 < sums[index + 1] < 1.0:
            slopes[index] = (
                np.log(sums[index + 1]) - np.log(sums[index])
            ) / (np.log(radii[index + 1]) - np.log(radii[index]))
    return centers, slopes


def fit_correlation_dimension(
    curve: CorrelationSumResult,
    *,
    fit_radius_range: tuple[float, float],
    minimum_points: int = 3,
) -> CorrelationDimensionResult:
    """Fit ``log(C(r)) = intercept + D2*log(r)`` on an explicit range."""

    if not isinstance(curve, CorrelationSumResult):
        raise TypeError("curve must be a CorrelationSumResult.")
    bounds = _fit_range(fit_radius_range)
    minimum = _strict_integer(minimum_points, "minimum_points", minimum=3)
    usable = (
        (curve.radii >= bounds[0])
        & (curve.radii <= bounds[1])
        & (curve.correlation_sums > 0.0)
        & (curve.correlation_sums < 1.0)
    )
    indices = np.flatnonzero(usable)
    if indices.size < minimum:
        raise ValueError(
            "fit_radius_range contains fewer than minimum_points values with 0 < C(r) < 1."
        )
    x_values = np.log(curve.radii[indices])
    y_values = np.log(curve.correlation_sums[indices])
    x_centered = x_values - float(np.mean(x_values))
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        raise ValueError("selected log radii do not span a fit interval.")
    y_mean = float(np.mean(y_values))
    slope = float(np.dot(x_centered, y_values - y_mean) / denominator)
    intercept = float(y_mean - slope * float(np.mean(x_values)))
    residuals = y_values - (intercept + slope * x_values)
    residual_sum_squares = float(np.dot(residuals, residuals))
    total_centered = y_values - y_mean
    total_sum_squares = float(np.dot(total_centered, total_centered))
    r_squared = (
        float(1.0 - residual_sum_squares / total_sum_squares)
        if total_sum_squares > 0.0
        else float("nan")
    )
    regression_standard_error = float(
        np.sqrt(max(0.0, residual_sum_squares) / (indices.size - 2) / denominator)
    )
    local_radii, local_slopes = _local_slopes(curve)
    return CorrelationDimensionResult(
        curve=curve,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        regression_standard_error=regression_standard_error,
        fit_indices=indices,
        log_radii=x_values,
        log_correlation_sums=y_values,
        local_slope_radii=local_radii,
        local_slopes=local_slopes,
        fit_radius_range=bounds,
        minimum_points=minimum,
        status="explicit_scaling_range_regression",
        metadata={
            "fit_selection": "caller_supplied_inclusive_radius_range",
            "saturation_excluded": "requires 0 < C(r) < 1",
            "logarithm": "natural",
            "regression": "unweighted_ordinary_least_squares",
            "regression_standard_error_scope": (
                "line-fit diagnostic only; correlated radii and trajectory pairs "
                "prevent interpreting it as a complete uncertainty interval"
            ),
            "fit_point_count": int(indices.size),
            "evidence_boundary": (
                "explicit finite-sample regression; not proof of chaos or "
                "hiddenness and not an automatic scaling-region certificate"
            ),
            "fractional_state_caveat": FRACTIONAL_STATE_CAVEAT,
        },
    )


def estimate_correlation_dimension(
    points: np.ndarray,
    radii: np.ndarray,
    *,
    fit_radius_range: tuple[float, float],
    minimum_points: int = 3,
    theiler_window: int = 0,
    metric: str = "euclidean",
    backend: str = "auto",
    fallback: bool = True,
    sampling: str = "samples supplied in row order",
    projection: str = "coordinates supplied",
) -> CorrelationDimensionResult:
    """Build a correlation curve and fit an explicitly declared scale range."""

    curve = correlation_sum_curve(
        points,
        radii,
        theiler_window=theiler_window,
        metric=metric,
        backend=backend,
        fallback=fallback,
        sampling=sampling,
        projection=projection,
    )
    return fit_correlation_dimension(
        curve,
        fit_radius_range=fit_radius_range,
        minimum_points=minimum_points,
    )


__all__ = [
    "CORRELATION_DIMENSION_EVIDENCE_SCOPE",
    "CORRELATION_DIMENSION_REFERENCE_DOIS",
    "CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS",
    "CorrelationDimensionResult",
    "CorrelationSumResult",
    "FRACTIONAL_STATE_CAVEAT",
    "correlation_sum_curve",
    "estimate_correlation_dimension",
    "fit_correlation_dimension",
]
