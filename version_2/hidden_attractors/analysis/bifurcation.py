"""Post-processing helpers for parameter scans and bifurcation diagrams.

These helpers do not replace continuation packages such as PyDSTool. They take
trajectories produced by this project and extract observable values that can be
plotted as a bifurcation diagram for the current study systems.

Reference notes:
    This module implements trajectory post-processing only. Numerical
    continuation, branch tracking, and full bifurcation analysis should cite and
    delegate to a dedicated continuation package when used. See
    ``docs/code_reference_map.md`` and ``docs/external_tools.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OBSERVABLE_COLUMNS = {"t": 0, "x": 1, "y": 2, "z": 3}


@dataclass(frozen=True)
class BifurcationPoint:
    """One observable value associated with one parameter value.

    Attributes
    ----------
    parameter : float
        The bifurcation parameter value for this point.
    observable : float
        The observed state-component value (e.g. a local maximum).
    time : float
        Time at which the observable was recorded.
    index : int
        Row index in the post-transient trajectory tail.
    kind : str
        Extraction mode: ``'maxima'``, ``'minima'``, ``'both'``,
        or ``'sample'``.
    """

    parameter: float
    observable: float
    time: float
    index: int
    kind: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "parameter": self.parameter,
            "observable": self.observable,
            "time": self.time,
            "index": self.index,
            "kind": self.kind,
        }


def observable_column(observable: str | int) -> int:
    """Return the trajectory column index for an observable label.

    Parameters
    ----------
    observable : str or int
        Named column (``'t'``, ``'x'``, ``'y'``, ``'z'``) or an integer
        column index (non-negative).

    Returns
    -------
    col : int
        Zero-based column index.

    Raises
    ------
    ValueError
        If *observable* is an unknown string or a negative integer.

    Examples
    --------
    >>> from hidden_attractors.analysis.bifurcation import observable_column
    >>> observable_column('x')
    1
    >>> observable_column(3)
    3
    """

    if isinstance(observable, str):
        key = observable.strip().lower()
        if key not in OBSERVABLE_COLUMNS:
            raise ValueError(f"Unknown observable {observable!r}; use one of {sorted(OBSERVABLE_COLUMNS)} or an integer column.")
        return OBSERVABLE_COLUMNS[key]
    col = int(observable)
    if col < 0:
        raise ValueError("observable column must be non-negative")
    return col


def trajectory_tail(traj: np.ndarray, *, t_start: float | None = None) -> np.ndarray:
    """Return the post-transient rows of a trajectory.

    Parameters
    ----------
    traj : np.ndarray, shape (N, ≥2)
        Trajectory array with time in column 0.
    t_start : float or None, default None
        Transient cutoff.  If ``None``, the whole trajectory is returned.

    Returns
    -------
    tail : np.ndarray
        Rows with ``t >= t_start``.

    Raises
    ------
    ValueError
        If *traj* is not 2-D with at least two columns.
    """

    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("trajectory must be a 2D array with at least columns t and one observable")
    if t_start is None:
        return X
    return X[X[:, 0] >= float(t_start)]


def local_extrema(values: Sequence[float], *, mode: str = "maxima") -> np.ndarray:
    """Return indices of local extrema in a one-dimensional series.

    Parameters
    ----------
    values : sequence of float
        One-dimensional time series.
    mode : str, default 'maxima'
        Which extrema to find:
        - ``'maxima'``: local maxima.
        - ``'minima'``: local minima.
        - ``'both'``: local maxima and minima.
        - ``'sample'``: return all indices (no filtering).

    Returns
    -------
    indices : np.ndarray of int
        Row indices where the requested extrema occur.

    Raises
    ------
    ValueError
        If *mode* is not one of the four accepted values.
    """

    y = np.asarray(values, dtype=float)
    if y.size < 3:
        return np.arange(y.size, dtype=int)
    left = y[1:-1] - y[:-2]
    right = y[1:-1] - y[2:]
    if mode == "maxima":
        mask = (left >= 0.0) & (right >= 0.0)
    elif mode == "minima":
        mask = (left <= 0.0) & (right <= 0.0)
    elif mode == "both":
        mask = ((left >= 0.0) & (right >= 0.0)) | ((left <= 0.0) & (right <= 0.0))
    elif mode == "sample":
        return np.arange(y.size, dtype=int)
    else:
        raise ValueError("mode must be 'maxima', 'minima', 'both', or 'sample'")
    return np.where(mask)[0] + 1


def _scan_item(item: Mapping[str, Any] | tuple[float, np.ndarray], parameter_key: str) -> tuple[float, np.ndarray]:
    if isinstance(item, Mapping):
        if "trajectory" not in item:
            raise KeyError("scan mapping must contain a 'trajectory' entry")
        if parameter_key not in item:
            raise KeyError(f"scan mapping must contain parameter key {parameter_key!r}")
        return float(item[parameter_key]), np.asarray(item["trajectory"], dtype=float)
    parameter, trajectory = item
    return float(parameter), np.asarray(trajectory, dtype=float)


def bifurcation_points_from_trajectories(
    scans: Iterable[Mapping[str, Any] | tuple[float, np.ndarray]],
    *,
    parameter_key: str = "parameter",
    observable: str | int = "x",
    t_start: float | None = None,
    mode: str = "maxima",
    max_points_per_parameter: int = 250,
) -> list[BifurcationPoint]:
    """Extract bifurcation-diagram points from a parameter scan.

    Parameters
    ----------
    scans : iterable
        Each element is either:

        - A ``(parameter, trajectory)`` tuple.
        - A mapping with keys *parameter_key* and ``'trajectory'``.

    parameter_key : str, default 'parameter'
        Key used to read the parameter value from mapping items.
    observable : str or int, default 'x'
        Column to extract.  See :func:`observable_column`.
    t_start : float or None, default None
        Transient cutoff passed to :func:`trajectory_tail`.
    mode : str, default 'maxima'
        Extremum extraction mode passed to :func:`local_extrema`.
    max_points_per_parameter : int, default 250
        Maximum points retained per parameter value (uniformly subsampled).
        Set to 0 to disable the limit.

    Returns
    -------
    points : list[BifurcationPoint]
        All extracted :class:`BifurcationPoint` objects, one per extremum.

    Notes
    -----
    This function performs trajectory post-processing only.  Numerical
    continuation, branch tracking, and full bifurcation analysis require
    a dedicated continuation package such as PyDSTool.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.analysis.bifurcation import bifurcation_points_from_trajectories
    >>> t = np.linspace(0, 100, 10000)
    >>> traj = np.column_stack([t, np.sin(2*np.pi*0.5*t), np.zeros_like(t), np.zeros_like(t)])
    >>> pts = bifurcation_points_from_trajectories(
    ...     [(1.0, traj)], t_start=10.0, mode='maxima')
    >>> len(pts) > 0
    True
    """

    col = observable_column(observable)
    points: list[BifurcationPoint] = []
    for item in scans:
        parameter, trajectory = _scan_item(item, parameter_key)
        tail = trajectory_tail(trajectory, t_start=t_start)
        if tail.size == 0:
            continue
        if col >= tail.shape[1]:
            raise ValueError(f"observable column {col} exceeds trajectory width {tail.shape[1]}")
        idx = local_extrema(tail[:, col], mode=mode)
        if max_points_per_parameter > 0 and idx.size > max_points_per_parameter:
            select = np.linspace(0, idx.size - 1, int(max_points_per_parameter), dtype=int)
            idx = idx[select]
        for i in idx:
            points.append(
                BifurcationPoint(
                    parameter=parameter,
                    observable=float(tail[i, col]),
                    time=float(tail[i, 0]),
                    index=int(i),
                    kind=mode,
                )
            )
    return points


def bifurcation_summary(points: Sequence[BifurcationPoint]) -> dict[str, float | int]:
    """Return summary statistics for a set of bifurcation points.

    Parameters
    ----------
    points : sequence of BifurcationPoint
        Extracted bifurcation points, e.g. from
        :func:`bifurcation_points_from_trajectories`.

    Returns
    -------
    summary : dict[str, float | int]
        Keys: ``'n_points'``, ``'parameter_min'``, ``'parameter_max'``,
        ``'observable_min'``, ``'observable_max'``.
        If *points* is empty, only ``'n_points': 0`` is returned.

    Examples
    --------
    >>> from hidden_attractors.analysis.bifurcation import (
    ...     BifurcationPoint, bifurcation_summary)
    >>> pts = [BifurcationPoint(1.0, 0.5, 10.0, 0, 'maxima')]
    >>> bifurcation_summary(pts)['n_points']
    1
    """

    if not points:
        return {"n_points": 0}
    params = np.array([p.parameter for p in points], dtype=float)
    values = np.array([p.observable for p in points], dtype=float)
    return {
        "n_points": int(values.size),
        "parameter_min": float(np.min(params)),
        "parameter_max": float(np.max(params)),
        "observable_min": float(np.min(values)),
        "observable_max": float(np.max(values)),
    }
