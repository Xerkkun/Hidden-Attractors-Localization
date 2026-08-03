"""Delay-coordinate reconstruction for sampled dynamical trajectories.

Stability: experimental

The functions in this module operate only on sampled arrays, so the same API
can analyse trajectories produced by integer- or fractional-order solvers.
For a fractional differential equation (FDE), however, a finite delay vector
is an *empirical reconstruction of an observable*.  It does not prove that the
hereditary state of the FDE has a finite-dimensional diffeomorphic embedding.

The implementation is original and uses the following primary references:

* F. Takens, "Detecting strange attractors in turbulence", in *Dynamical
  Systems and Turbulence, Warwick 1980*, Lecture Notes in Mathematics 898,
  Springer, 1981, pp. 366--381, DOI: 10.1007/BFb0091924.
* A. M. Fraser and H. L. Swinney, "Independent coordinates for strange
  attractors from mutual information", *Physical Review A* 33 (1986),
  1134--1140, DOI: 10.1103/PhysRevA.33.1134.
* M. B. Kennel, R. Brown, and H. D. I. Abarbanel, "Determining embedding
  dimension for phase-space reconstruction using a geometrical
  construction", *Physical Review A* 45 (1992), 3403--3411,
  DOI: 10.1103/PhysRevA.45.3403.

``DelayEmbeddings.jl`` informed the capability selection, but no source code
was copied.  All results below are finite-sample numerical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy.spatial import cKDTree


FDE_RECONSTRUCTION_CAVEAT = (
    "For fractional differential equations this is an empirical observable "
    "reconstruction; it does not establish a finite-dimensional diffeomorphic "
    "embedding of the hereditary state."
)
EVIDENCE_SCOPE = "finite_sample_empirical_trajectory_diagnostic"
INDEX_LAG_CAVEAT = (
    "Delay estimators operate on sample-index lags. With irregular timestamps, "
    "a fixed index lag is not a constant physical-time delay."
)


@dataclass(frozen=True, slots=True)
class GeneralizedEmbeddingResult:
    """A generalized delay embedding and its complete alignment contract.

    ``projection[k] == (column, lag)`` means
    ``vectors[row, k] == trajectory[anchor_indices[row] - lag, column]``.
    Positive lags therefore refer to past samples and negative lags to future
    samples.  ``aligned_times`` always accompanies the anchor indices: it is
    copied from ``times``, constructed from ``dt``, or expressed in sample
    indices if neither is supplied.
    """

    vectors: np.ndarray
    anchor_indices: np.ndarray
    aligned_times: np.ndarray
    projection: tuple[tuple[int, int], ...]
    lag_times: tuple[float, ...] | None
    dt: float | None
    time_unit: str
    lag_unit: str
    time_source: str
    backend: str = "numpy"
    evidence_scope: str = EVIDENCE_SCOPE
    fractional_state_caveat: str = FDE_RECONSTRUCTION_CAVEAT

    @property
    def parameters(self) -> dict[str, Any]:
        """Serializable reconstruction parameters."""

        return {
            "projection": self.projection,
            "alignment": "trajectory[anchor_index - lag, column]",
            "dt": self.dt,
            "time_unit": self.time_unit,
            "lag_unit": self.lag_unit,
            "time_source": self.time_source,
        }


@dataclass(frozen=True, slots=True)
class DelayEstimateResult:
    """Selected delay and the score curve from which it was selected."""

    method: str
    lag_samples: int | None
    lag_time: float | None
    lags: np.ndarray
    scores: np.ndarray
    score_name: str
    selection: str
    status: str
    source_column: int
    projection: tuple[tuple[int, int], ...]
    parameters: dict[str, Any]
    dt: float | None
    time_unit: str
    time_source: str
    lag_unit: str = "samples"
    backend: str = "numpy"
    sampling_caveat: str = INDEX_LAG_CAVEAT
    evidence_scope: str = EVIDENCE_SCOPE
    fractional_state_caveat: str = FDE_RECONSTRUCTION_CAVEAT


@dataclass(frozen=True, slots=True)
class FNNDimensionResult:
    """False-nearest-neighbour statistics for one trial dimension."""

    dimension: int
    fraction: float
    false_neighbors: int
    valid_neighbors: int
    candidate_vectors: int
    projection: tuple[tuple[int, int], ...]
    extension_projection: tuple[tuple[int, int], ...]
    status: str


@dataclass(frozen=True, slots=True)
class FalseNearestNeighborsResult:
    """FNN dimension sweep with its numerical and sampling contract."""

    records: tuple[FNNDimensionResult, ...]
    selected_dimension: int | None
    selection_threshold: float
    delay_samples: int
    delay_time: float | None
    source_column: int
    projection_convention: str
    theiler_window: int
    rtol: float
    atol: float
    metric: str
    algorithm: str
    attractor_scale: float
    minimum_valid_neighbors: int
    dt: float | None
    time_unit: str
    time_source: str
    lag_unit: str = "samples"
    backend: str = "scipy.spatial.cKDTree"
    sampling_caveat: str = INDEX_LAG_CAVEAT
    evidence_scope: str = EVIDENCE_SCOPE
    fractional_state_caveat: str = FDE_RECONSTRUCTION_CAVEAT

    @property
    def dimensions(self) -> np.ndarray:
        """Trial dimensions as a newly allocated integer array."""

        return np.asarray([record.dimension for record in self.records], dtype=int)

    @property
    def fractions(self) -> np.ndarray:
        """FNN fractions as a newly allocated floating-point array."""

        return np.asarray([record.fraction for record in self.records], dtype=float)

    @property
    def parameters(self) -> dict[str, Any]:
        """Serializable FNN parameters, excluding the score records."""

        return {
            "selection_threshold": self.selection_threshold,
            "delay_samples": self.delay_samples,
            "delay_time": self.delay_time,
            "source_column": self.source_column,
            "projection_convention": self.projection_convention,
            "theiler_window": self.theiler_window,
            "rtol": self.rtol,
            "atol": self.atol,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "attractor_scale": self.attractor_scale,
            "minimum_valid_neighbors": self.minimum_valid_neighbors,
            "dt": self.dt,
            "time_unit": self.time_unit,
            "time_source": self.time_source,
            "lag_unit": self.lag_unit,
        }


def _as_trajectory(trajectory: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectory, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(
            "trajectory must have shape (n_samples,) or "
            "(n_samples, n_columns)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory must contain only finite values.")
    return values


def _positive_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _sampling_contract(
    n_samples: int,
    *,
    times: np.ndarray | None,
    dt: float | None,
    time_unit: str,
) -> tuple[np.ndarray, float | None, str, str]:
    unit = str(time_unit).strip()
    if not unit:
        raise ValueError("time_unit must be a non-empty string.")
    supplied_dt = _positive_float(dt, "dt")

    if times is None:
        if supplied_dt is None:
            return (
                np.arange(n_samples, dtype=float),
                None,
                "sample_index",
                "sample_indices",
            )
        return (
            np.arange(n_samples, dtype=float) * supplied_dt,
            supplied_dt,
            unit,
            "constructed_from_dt",
        )

    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1 or time_values.shape[0] != n_samples:
        raise ValueError("times must be one-dimensional with len(trajectory) entries.")
    if not np.all(np.isfinite(time_values)):
        raise ValueError("times must contain only finite values.")
    if n_samples > 1:
        increments = np.diff(time_values)
        if np.any(increments <= 0.0):
            raise ValueError("times must be strictly increasing.")
        inferred_dt = float(np.median(increments))
        uniform = bool(
            np.allclose(
                increments,
                inferred_dt,
                rtol=1.0e-10,
                atol=max(1.0e-14, abs(inferred_dt) * 1.0e-12),
            )
        )
    else:
        inferred_dt = supplied_dt if supplied_dt is not None else float("nan")
        uniform = supplied_dt is not None

    if supplied_dt is not None:
        if not uniform:
            raise ValueError("dt cannot describe non-uniform times.")
        if not np.isclose(
            supplied_dt,
            inferred_dt,
            rtol=1.0e-10,
            atol=max(1.0e-14, abs(inferred_dt) * 1.0e-12),
        ):
            raise ValueError("dt is inconsistent with the supplied times.")
        effective_dt = supplied_dt
    else:
        effective_dt = inferred_dt if uniform else None
    source = "supplied_uniform_times" if uniform else "supplied_irregular_times"
    return time_values, effective_dt, unit, source


def _integer(value: int, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def generalized_delay_embedding(
    trajectory: np.ndarray,
    projection: Sequence[tuple[int, int]],
    *,
    times: np.ndarray | None = None,
    dt: float | None = None,
    time_unit: str = "arbitrary_time",
) -> GeneralizedEmbeddingResult:
    """Build a generalized, possibly multivariate, delay embedding.

    Parameters
    ----------
    trajectory:
        Sample-major array.  A one-dimensional array is treated as one column.
    projection:
        Explicit ``(column, lag_in_samples)`` pairs.  For anchor ``i`` the
        corresponding coordinate is ``trajectory[i - lag, column]``.  Positive
        lags look backwards; negative lags look forwards.
    times, dt, time_unit:
        Optional sampling metadata.  Uniform ``times`` imply ``dt``; irregular
        ``times`` are accepted, but a single physical ``lag_times`` conversion
        is then intentionally unavailable.

    Returns
    -------
    GeneralizedEmbeddingResult
        Embedded vectors plus exactly aligned anchor indices and times.
    """

    values = _as_trajectory(trajectory)
    pairs: list[tuple[int, int]] = []
    for coordinate, pair in enumerate(projection):
        try:
            pair_length = len(pair)
        except TypeError as exc:
            raise ValueError(
                f"projection entry {coordinate} must be a (column, lag) pair."
            ) from exc
        if isinstance(pair, (str, bytes)) or pair_length != 2:
            raise ValueError(
                f"projection entry {coordinate} must be a (column, lag) pair."
            )
        column = _integer(pair[0], f"projection[{coordinate}].column", minimum=0)
        lag = _integer(pair[1], f"projection[{coordinate}].lag")
        if column >= values.shape[1]:
            raise ValueError(
                f"projection column {column} is outside trajectory width "
                f"{values.shape[1]}."
            )
        pairs.append((column, lag))
    if not pairs:
        raise ValueError("projection must contain at least one (column, lag) pair.")

    source_times, effective_dt, effective_unit, time_source = _sampling_contract(
        values.shape[0], times=times, dt=dt, time_unit=time_unit
    )
    lags = np.asarray([lag for _, lag in pairs], dtype=int)
    first_anchor = max(0, int(np.max(lags)))
    final_anchor_exclusive = min(values.shape[0], values.shape[0] + int(np.min(lags)))
    if final_anchor_exclusive <= first_anchor:
        raise ValueError("trajectory is too short for the requested projection.")
    anchors = np.arange(first_anchor, final_anchor_exclusive, dtype=int)
    embedded = np.empty((anchors.size, len(pairs)), dtype=float)
    for output_column, (source_column, lag) in enumerate(pairs):
        embedded[:, output_column] = values[anchors - lag, source_column]
    lag_times = (
        tuple(float(lag) * effective_dt for lag in lags)
        if effective_dt is not None
        else None
    )
    return GeneralizedEmbeddingResult(
        vectors=embedded,
        anchor_indices=anchors,
        aligned_times=np.asarray(source_times[anchors], dtype=float),
        projection=tuple(pairs),
        lag_times=lag_times,
        dt=effective_dt,
        time_unit=effective_unit,
        lag_unit="samples",
        time_source=time_source,
    )


def _observable(
    trajectory: np.ndarray,
    column: int,
) -> tuple[np.ndarray, int, int]:
    values = _as_trajectory(trajectory)
    selected = _integer(column, "column", minimum=0)
    if selected >= values.shape[1]:
        raise ValueError(
            f"column {selected} is outside trajectory width {values.shape[1]}."
        )
    return values[:, selected], selected, values.shape[0]


def _resolved_max_lag(n_samples: int, max_lag: int | None) -> int:
    if n_samples < 3:
        raise ValueError("at least three samples are required for delay estimation.")
    if max_lag is None:
        return max(1, min(n_samples // 2, n_samples - 1))
    result = _integer(max_lag, "max_lag", minimum=1)
    if result >= n_samples:
        raise ValueError("max_lag must be smaller than the number of samples.")
    return result


def _first_local_minimum(scores: np.ndarray) -> int | None:
    for index in range(1, scores.size - 1):
        if scores[index] < scores[index - 1] and scores[index] <= scores[index + 1]:
            return index
    return None


def estimate_delay_autocorrelation(
    trajectory: np.ndarray,
    *,
    column: int = 0,
    max_lag: int | None = None,
    criterion: Literal["first_zero_crossing", "first_local_minimum"] = (
        "first_zero_crossing"
    ),
    times: np.ndarray | None = None,
    dt: float | None = None,
    time_unit: str = "arbitrary_time",
) -> DelayEstimateResult:
    """Estimate a scalar-observable delay from its normalized autocorrelation.

    The returned autocorrelation uses the overlapping-products numerator
    divided by the total zero-lag sum of squares.  It is computed by a
    zero-padded FFT without circular wraparound.  Selection is either the first
    positive-to-nonpositive sample crossing or the first strict-left,
    non-strict-right local minimum.
    """

    series, selected_column, n_samples = _observable(trajectory, column)
    resolved_max_lag = _resolved_max_lag(n_samples, max_lag)
    if criterion not in {"first_zero_crossing", "first_local_minimum"}:
        raise ValueError(
            "criterion must be 'first_zero_crossing' or 'first_local_minimum'."
        )
    centered = series - float(np.mean(series))
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("autocorrelation delay is undefined for a constant series.")
    fft_length = 1 << (2 * n_samples - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    correlations = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_length)
    scores = np.asarray(correlations[: resolved_max_lag + 1] / denominator, dtype=float)
    scores[0] = 1.0
    lags = np.arange(resolved_max_lag + 1, dtype=int)

    selected_index: int | None = None
    if criterion == "first_zero_crossing":
        for index in range(1, scores.size):
            if scores[index] <= 0.0 < scores[index - 1]:
                selected_index = index
                break
    else:
        selected_index = _first_local_minimum(scores)
    selected_lag = int(lags[selected_index]) if selected_index is not None else None
    _, effective_dt, effective_unit, time_source = _sampling_contract(
        n_samples, times=times, dt=dt, time_unit=time_unit
    )
    return DelayEstimateResult(
        method="autocorrelation",
        lag_samples=selected_lag,
        lag_time=(
            float(selected_lag) * effective_dt
            if selected_lag is not None and effective_dt is not None
            else None
        ),
        lags=lags,
        scores=scores,
        score_name="normalized_autocorrelation",
        selection=criterion,
        status="selected" if selected_lag is not None else "criterion_not_found",
        source_column=selected_column,
        projection=(
            ((selected_column, 0), (selected_column, selected_lag))
            if selected_lag is not None
            else ((selected_column, 0),)
        ),
        parameters={
            "max_lag": resolved_max_lag,
            "normalization": "overlap_sum_divided_by_zero_lag_sum_of_squares",
            "fft_length": fft_length,
            "pair_counts": tuple(int(n_samples - lag) for lag in lags),
            "index_lag_caveat": INDEX_LAG_CAVEAT,
        },
        dt=effective_dt,
        time_unit=effective_unit,
        time_source=time_source,
        backend="numpy.fft",
    )


def estimate_delay_mutual_information(
    trajectory: np.ndarray,
    *,
    column: int = 0,
    max_lag: int | None = None,
    bins: int | str = "fd",
    fallback: Literal["global_minimum", "none"] = "global_minimum",
    minimum_pairs: int = 16,
    times: np.ndarray | None = None,
    dt: float | None = None,
    time_unit: str = "arbitrary_time",
) -> DelayEstimateResult:
    """Estimate delay with fixed-bin plug-in mutual information.

    One global set of histogram edges is computed from the full observable and
    reused at every lag, making the binning contract directly reproducible.
    This is not Fraser and Swinney's adaptive recursive partition; HAFO adopts
    their *first-minimum selection criterion* with a declared plug-in estimator.
    ``minimum_pairs`` prevents sparsely overlapped tail lags from becoming
    artificial minima.  If no local minimum exists, ``fallback`` either selects
    the global sampled minimum or reports that no criterion was found.  Mutual
    information is returned in nats.
    """

    series, selected_column, n_samples = _observable(trajectory, column)
    minimum_pair_count = _integer(minimum_pairs, "minimum_pairs", minimum=2)
    if minimum_pair_count >= n_samples:
        raise ValueError("minimum_pairs must be smaller than the number of samples.")
    resolved_max_lag = _resolved_max_lag(n_samples, max_lag)
    largest_supported_lag = n_samples - minimum_pair_count
    if resolved_max_lag > largest_supported_lag:
        if max_lag is not None:
            raise ValueError(
                "max_lag leaves fewer than minimum_pairs overlapping observations."
            )
        resolved_max_lag = largest_supported_lag
    if fallback not in {"global_minimum", "none"}:
        raise ValueError("fallback must be 'global_minimum' or 'none'.")
    if float(np.ptp(series)) <= np.finfo(float).tiny:
        raise ValueError("mutual-information delay is undefined for a constant series.")
    if isinstance(bins, (bool, np.bool_)):
        raise TypeError("bins must be an integer >= 2 or a NumPy binning rule.")
    if isinstance(bins, (int, np.integer)):
        if int(bins) < 2:
            raise ValueError("integer bins must be at least 2.")
        binning: int | str = int(bins)
    elif isinstance(bins, str) and bins.strip():
        binning = bins.strip()
    else:
        raise TypeError("bins must be an integer >= 2 or a NumPy binning rule.")
    try:
        edges = np.histogram_bin_edges(series, bins=binning)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid histogram binning rule {binning!r}.") from exc
    if edges.size < 3:
        raise ValueError("histogram binning must produce at least two bins.")

    lags = np.arange(1, resolved_max_lag + 1, dtype=int)
    pair_counts = n_samples - lags
    scores = np.empty(lags.size, dtype=float)
    for output_index, lag in enumerate(lags):
        counts, _, _ = np.histogram2d(
            series[:-lag], series[lag:], bins=(edges, edges)
        )
        total = float(np.sum(counts))
        probabilities = counts / total
        marginal_x = np.sum(probabilities, axis=1)
        marginal_y = np.sum(probabilities, axis=0)
        expected = marginal_x[:, None] * marginal_y[None, :]
        occupied = probabilities > 0.0
        scores[output_index] = float(
            np.sum(
                probabilities[occupied]
                * np.log(probabilities[occupied] / expected[occupied])
            )
        )

    marginal_counts, _ = np.histogram(series, bins=edges)
    marginal_probabilities = marginal_counts[marginal_counts > 0] / n_samples
    zero_lag_information = float(
        -np.sum(marginal_probabilities * np.log(marginal_probabilities))
    )
    full_curve = np.concatenate(([zero_lag_information], scores))
    full_selected_index = _first_local_minimum(full_curve)
    selected_index = (
        full_selected_index - 1
        if full_selected_index is not None and full_selected_index >= 1
        else None
    )
    if selected_index is not None:
        selection = "first_local_minimum"
        status = "selected"
    elif fallback == "global_minimum":
        selected_index = int(np.argmin(scores))
        selection = "global_minimum_fallback"
        status = "selected_by_fallback"
    else:
        selection = "first_local_minimum"
        status = "criterion_not_found"
    selected_lag = int(lags[selected_index]) if selected_index is not None else None
    _, effective_dt, effective_unit, time_source = _sampling_contract(
        n_samples, times=times, dt=dt, time_unit=time_unit
    )
    return DelayEstimateResult(
        method="fixed_bin_plugin_mutual_information",
        lag_samples=selected_lag,
        lag_time=(
            float(selected_lag) * effective_dt
            if selected_lag is not None and effective_dt is not None
            else None
        ),
        lags=lags,
        scores=scores,
        score_name="mutual_information_nats",
        selection=selection,
        status=status,
        source_column=selected_column,
        projection=(
            ((selected_column, 0), (selected_column, selected_lag))
            if selected_lag is not None
            else ((selected_column, 0),)
        ),
        parameters={
            "max_lag": resolved_max_lag,
            "binning": binning,
            "n_bins": int(edges.size - 1),
            "bin_edges": tuple(float(value) for value in edges),
            "edges_reused_for_all_lags": True,
            "estimator": "fixed_bin_plugin_not_adaptive_partition",
            "zero_lag_information": zero_lag_information,
            "minimum_pairs": minimum_pair_count,
            "pair_counts": tuple(int(value) for value in pair_counts),
            "fallback": fallback,
            "index_lag_caveat": INDEX_LAG_CAVEAT,
        },
        dt=effective_dt,
        time_unit=effective_unit,
        time_source=time_source,
        backend="numpy.histogram2d",
    )


def _metric_order(metric: str) -> tuple[str, float]:
    normalized = str(metric).strip().lower()
    aliases = {
        "euclidean": ("euclidean", 2.0),
        "l2": ("euclidean", 2.0),
        "manhattan": ("manhattan", 1.0),
        "cityblock": ("manhattan", 1.0),
        "l1": ("manhattan", 1.0),
        "chebyshev": ("chebyshev", float("inf")),
        "linf": ("chebyshev", float("inf")),
    }
    if normalized not in aliases:
        raise ValueError(
            "metric must be 'euclidean', 'manhattan', or 'chebyshev'."
        )
    return aliases[normalized]


def _distance(first: np.ndarray, second: np.ndarray, order: float) -> float:
    difference = np.abs(first - second)
    if np.isinf(order):
        return float(np.max(difference))
    return float(np.sum(difference**order) ** (1.0 / order))


def _nearest_allowed_neighbor(
    tree: cKDTree,
    points: np.ndarray,
    anchors: np.ndarray,
    query_index: int,
    *,
    theiler_window: int,
    order: float,
) -> tuple[int, float] | None:
    """Find a deterministic nearest neighbour outside the Theiler window."""

    count = points.shape[0]
    if count < 2:
        return None
    query_count = min(count, max(2, 2 * theiler_window + 2))
    distances, indices = tree.query(points[query_index], k=query_count, p=order)
    candidate_indices = np.atleast_1d(indices).astype(int)
    candidate_distances = np.atleast_1d(distances).astype(float)
    allowed = [
        (float(distance), int(index))
        for distance, index in zip(candidate_distances, candidate_indices)
        if index < count
        and abs(int(anchors[index]) - int(anchors[query_index])) > theiler_window
    ]
    if not allowed and query_count < count:
        distances, indices = tree.query(points[query_index], k=count, p=order)
        allowed = [
            (float(distance), int(index))
            for distance, index in zip(
                np.atleast_1d(distances), np.atleast_1d(indices)
            )
            if int(index) < count
            and abs(int(anchors[int(index)]) - int(anchors[query_index]))
            > theiler_window
        ]
    if not allowed:
        return None

    minimum_distance = min(item[0] for item in allowed)
    radius = np.nextafter(minimum_distance, float("inf"))
    tied = tree.query_ball_point(points[query_index], radius, p=order)
    exact_candidates: list[tuple[float, int]] = []
    for index in tied:
        if abs(int(anchors[index]) - int(anchors[query_index])) <= theiler_window:
            continue
        exact_candidates.append(
            (_distance(points[query_index], points[index], order), int(index))
        )
    if not exact_candidates:
        exact_candidates = allowed
    exact_distance, selected_index = min(
        exact_candidates, key=lambda item: (item[0], item[1])
    )
    return selected_index, float(exact_distance)


def false_nearest_neighbors(
    trajectory: np.ndarray,
    *,
    column: int = 0,
    delay: int = 1,
    min_dimension: int = 1,
    max_dimension: int = 10,
    theiler_window: int = 0,
    rtol: float = 10.0,
    atol: float = 2.0,
    metric: str = "euclidean",
    selection_threshold: float = 0.01,
    minimum_valid_neighbors: int = 20,
    times: np.ndarray | None = None,
    dt: float | None = None,
    time_unit: str = "arbitrary_time",
) -> FalseNearestNeighborsResult:
    """Estimate an embedding dimension with false nearest neighbours (FNN).

    For dimension ``m``, vectors use lags ``0, delay, ..., (m-1)*delay``
    and share anchors with the dimension-``m+1`` extension.  A neighbour is
    false if either Kennel--Brown--Abarbanel test is true:

    ``|delta_extra| / distance_m > rtol`` or
    ``distance_(m+1) / std(observable) > atol``.

    This attribution is exact for the Euclidean metric used in the original
    method.  Manhattan and Chebyshev select a declared generalized-Lp variant
    using the same threshold form.  The nearest neighbour must lie more than
    ``theiler_window`` samples away.
    Results with fewer than ``minimum_valid_neighbors`` comparisons are marked
    ``"insufficient_neighbors"`` and cannot select a dimension.  Ties are
    resolved by the smallest source anchor, making repeated runs deterministic.
    """

    series, selected_column, n_samples = _observable(trajectory, column)
    delay_value = _integer(delay, "delay", minimum=1)
    minimum_dimension = _integer(min_dimension, "min_dimension", minimum=1)
    maximum_dimension = _integer(max_dimension, "max_dimension", minimum=1)
    if maximum_dimension < minimum_dimension:
        raise ValueError("max_dimension must be at least min_dimension.")
    theiler_value = _integer(theiler_window, "theiler_window", minimum=0)
    minimum_neighbors = _integer(
        minimum_valid_neighbors, "minimum_valid_neighbors", minimum=1
    )
    rtol_value = _positive_float(rtol, "rtol")
    atol_value = _positive_float(atol, "atol")
    assert rtol_value is not None and atol_value is not None
    threshold = float(selection_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("selection_threshold must be finite and in [0, 1].")
    metric_name, order = _metric_order(metric)
    attractor_scale = float(np.std(series))
    if attractor_scale <= np.finfo(float).tiny:
        raise ValueError("FNN is undefined for a constant observable.")
    _, effective_dt, effective_unit, time_source = _sampling_contract(
        n_samples, times=times, dt=dt, time_unit=time_unit
    )

    records: list[FNNDimensionResult] = []
    selected_dimension: int | None = None
    for dimension in range(minimum_dimension, maximum_dimension + 1):
        base_projection = tuple(
            (selected_column, coordinate * delay_value)
            for coordinate in range(dimension)
        )
        extension_projection = base_projection + (
            (selected_column, dimension * delay_value),
        )
        first_anchor = dimension * delay_value
        if first_anchor >= n_samples:
            records.append(
                FNNDimensionResult(
                    dimension=dimension,
                    fraction=float("nan"),
                    false_neighbors=0,
                    valid_neighbors=0,
                    candidate_vectors=0,
                    projection=base_projection,
                    extension_projection=extension_projection,
                    status="insufficient_neighbors",
                )
            )
            continue
        anchors = np.arange(first_anchor, n_samples, dtype=int)
        points = np.column_stack(
            [series[anchors - coordinate * delay_value] for coordinate in range(dimension)]
        )
        extra = series[anchors - dimension * delay_value]
        tree = cKDTree(points)
        false_count = 0
        valid_count = 0
        for query_index in range(points.shape[0]):
            neighbor = _nearest_allowed_neighbor(
                tree,
                points,
                anchors,
                query_index,
                theiler_window=theiler_value,
                order=order,
            )
            if neighbor is None:
                continue
            neighbor_index, base_distance = neighbor
            extra_distance = abs(float(extra[query_index] - extra[neighbor_index]))
            if base_distance == 0.0:
                relative_extra = float("inf") if extra_distance > 0.0 else 0.0
            else:
                relative_extra = extra_distance / base_distance
            if np.isinf(order):
                extended_distance = max(base_distance, extra_distance)
            else:
                extended_distance = (
                    base_distance**order + extra_distance**order
                ) ** (1.0 / order)
            is_false = (
                relative_extra > rtol_value
                or extended_distance / attractor_scale > atol_value
            )
            false_count += int(is_false)
            valid_count += 1

        fraction = (
            float(false_count / valid_count) if valid_count else float("nan")
        )
        status = (
            "ok" if valid_count >= minimum_neighbors else "insufficient_neighbors"
        )
        records.append(
            FNNDimensionResult(
                dimension=dimension,
                fraction=fraction,
                false_neighbors=false_count,
                valid_neighbors=valid_count,
                candidate_vectors=int(points.shape[0]),
                projection=base_projection,
                extension_projection=extension_projection,
                status=status,
            )
        )
        if (
            selected_dimension is None
            and status == "ok"
            and fraction <= threshold
        ):
            selected_dimension = dimension

    return FalseNearestNeighborsResult(
        records=tuple(records),
        selected_dimension=selected_dimension,
        selection_threshold=threshold,
        delay_samples=delay_value,
        delay_time=(
            float(delay_value) * effective_dt if effective_dt is not None else None
        ),
        source_column=selected_column,
        projection_convention="(source_column, coordinate_index * delay_samples)",
        theiler_window=theiler_value,
        rtol=rtol_value,
        atol=atol_value,
        metric=metric_name,
        algorithm=(
            "kennel_brown_abarbanel_euclidean"
            if metric_name == "euclidean"
            else "generalized_lp_fnn_using_kba_threshold_form"
        ),
        attractor_scale=attractor_scale,
        minimum_valid_neighbors=minimum_neighbors,
        dt=effective_dt,
        time_unit=effective_unit,
        time_source=time_source,
    )


__all__ = [
    "DelayEstimateResult",
    "EVIDENCE_SCOPE",
    "FDE_RECONSTRUCTION_CAVEAT",
    "FNNDimensionResult",
    "FalseNearestNeighborsResult",
    "GeneralizedEmbeddingResult",
    "INDEX_LAG_CAVEAT",
    "estimate_delay_autocorrelation",
    "estimate_delay_mutual_information",
    "false_nearest_neighbors",
    "generalized_delay_embedding",
]
