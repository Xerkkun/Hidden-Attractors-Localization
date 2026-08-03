"""Recurrence plots and recurrence-quantification analysis (RQA).

Stability: experimental

These functions operate on sampled trajectories and are therefore shared by
integer- and fractional-order systems.  Their output is a finite numerical
diagnostic; it is not, by itself, proof of chaos or hiddenness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@dataclass(frozen=True, slots=True)
class RecurrenceQuantificationResult:
    """Structured RQA measures and the contract used to compute them."""

    recurrence_rate: float
    determinism: float
    laminarity: float
    trapping_time: float
    diagonal_entropy: float
    longest_diagonal: int
    longest_vertical: int
    radius: float
    min_diagonal: int
    min_vertical: int
    n_points: int
    exclude_diagonal: bool
    recurrence_matrix: np.ndarray | None
    status: str = "finite_numerical_diagnostic"


def delay_embedding(
    series: np.ndarray,
    dimension: int,
    delay: int,
) -> np.ndarray:
    """Construct a delay embedding from a scalar or multivariate time series."""

    values = np.asarray(series, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("series must have shape (n_times,) or (n_times, n_features).")
    if not np.all(np.isfinite(values)):
        raise ValueError("series must contain only finite values.")
    dimension = int(dimension)
    delay = int(delay)
    if dimension < 1 or delay < 1:
        raise ValueError("dimension and delay must be positive integers.")
    n_vectors = values.shape[0] - (dimension - 1) * delay
    if n_vectors < 1:
        raise ValueError("series is too short for the requested embedding.")
    embedded = np.empty((n_vectors, dimension * values.shape[1]), dtype=float)
    for coordinate in range(dimension):
        start = coordinate * values.shape[1]
        stop = start + values.shape[1]
        offset = coordinate * delay
        embedded[:, start:stop] = values[offset : offset + n_vectors]
    return embedded


@njit(cache=True, nogil=True, parallel=True)
def _recurrence_matrix_numba(points: np.ndarray, radius_squared: float) -> np.ndarray:
    count, dimension = points.shape
    matrix = np.empty((count, count), dtype=np.bool_)
    for row in prange(count):
        for column in range(count):
            squared_distance = 0.0
            for coordinate in range(dimension):
                difference = points[row, coordinate] - points[column, coordinate]
                squared_distance += difference * difference
            matrix[row, column] = squared_distance <= radius_squared
    return matrix


def recurrence_matrix(
    points: np.ndarray,
    radius: float,
    *,
    exclude_diagonal: bool = True,
    max_matrix_bytes: int = 512 * 1024 * 1024,
) -> np.ndarray:
    """Return a Boolean recurrence matrix using an explicit Euclidean radius."""

    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("points must contain at least two sampled states.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values.")
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative.")
    max_matrix_bytes = int(max_matrix_bytes)
    required_bytes = values.shape[0] * values.shape[0]
    if max_matrix_bytes < 1 or required_bytes > max_matrix_bytes:
        raise MemoryError(
            f"recurrence matrix needs approximately {required_bytes} bytes, "
            f"above max_matrix_bytes={max_matrix_bytes}."
        )
    matrix = _recurrence_matrix_numba(
        np.ascontiguousarray(values), radius * radius
    )
    if exclude_diagonal:
        np.fill_diagonal(matrix, False)
    return matrix


def _run_lengths(values: np.ndarray) -> np.ndarray:
    padded = np.concatenate(([False], np.asarray(values, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return stops - starts


def _diagonal_lengths(matrix: np.ndarray) -> np.ndarray:
    lengths: list[np.ndarray] = []
    count = matrix.shape[0]
    for offset in range(-(count - 1), count):
        line_lengths = _run_lengths(np.diagonal(matrix, offset=offset))
        if line_lengths.size:
            lengths.append(line_lengths)
    if not lengths:
        return np.empty(0, dtype=int)
    return np.concatenate(lengths)


def _vertical_lengths(matrix: np.ndarray) -> np.ndarray:
    lengths: list[np.ndarray] = []
    for column in range(matrix.shape[1]):
        line_lengths = _run_lengths(matrix[:, column])
        if line_lengths.size:
            lengths.append(line_lengths)
    if not lengths:
        return np.empty(0, dtype=int)
    return np.concatenate(lengths)


def recurrence_quantification(
    points: np.ndarray,
    radius: float,
    *,
    min_diagonal: int = 2,
    min_vertical: int = 2,
    exclude_diagonal: bool = True,
    return_matrix: bool = False,
    max_matrix_bytes: int = 512 * 1024 * 1024,
) -> RecurrenceQuantificationResult:
    """Compute standard RQA measures from sampled trajectory points."""

    min_diagonal = int(min_diagonal)
    min_vertical = int(min_vertical)
    if min_diagonal < 1 or min_vertical < 1:
        raise ValueError("minimum line lengths must be positive integers.")
    matrix = recurrence_matrix(
        points,
        radius,
        exclude_diagonal=exclude_diagonal,
        max_matrix_bytes=max_matrix_bytes,
    )
    count = matrix.shape[0]
    recurrent_points = int(np.count_nonzero(matrix))
    denominator = count * count - (count if exclude_diagonal else 0)
    recurrence_rate = recurrent_points / denominator if denominator else 0.0

    diagonal_lengths = _diagonal_lengths(matrix)
    selected_diagonals = diagonal_lengths[diagonal_lengths >= min_diagonal]
    diagonal_points = int(selected_diagonals.sum())
    determinism = diagonal_points / recurrent_points if recurrent_points else 0.0
    longest_diagonal = int(selected_diagonals.max()) if selected_diagonals.size else 0

    vertical_lengths = _vertical_lengths(matrix)
    selected_verticals = vertical_lengths[vertical_lengths >= min_vertical]
    vertical_points = int(selected_verticals.sum())
    laminarity = vertical_points / recurrent_points if recurrent_points else 0.0
    trapping_time = float(selected_verticals.mean()) if selected_verticals.size else 0.0
    longest_vertical = int(selected_verticals.max()) if selected_verticals.size else 0

    if selected_diagonals.size:
        _, frequencies = np.unique(selected_diagonals, return_counts=True)
        probabilities = frequencies / frequencies.sum()
        diagonal_entropy = float(-np.sum(probabilities * np.log(probabilities)))
    else:
        diagonal_entropy = 0.0

    return RecurrenceQuantificationResult(
        recurrence_rate=float(recurrence_rate),
        determinism=float(determinism),
        laminarity=float(laminarity),
        trapping_time=trapping_time,
        diagonal_entropy=diagonal_entropy,
        longest_diagonal=longest_diagonal,
        longest_vertical=longest_vertical,
        radius=float(radius),
        min_diagonal=min_diagonal,
        min_vertical=min_vertical,
        n_points=count,
        exclude_diagonal=bool(exclude_diagonal),
        recurrence_matrix=matrix if return_matrix else None,
    )


__all__ = [
    "RecurrenceQuantificationResult",
    "delay_embedding",
    "recurrence_matrix",
    "recurrence_quantification",
]
