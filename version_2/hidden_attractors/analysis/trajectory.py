"""Trajectory diagnostics for chaotic fractional systems.

Reference notes:
    - Hidden/self-excited interpretation follows Leonov--Kuznetsov hidden
      attractor terminology.
    - Lyapunov-style diagnostics, when delegated to external tools or native
      backends, should cite Benettin et al., "Lyapunov Characteristic
      Exponents for Smooth Dynamical Systems and for Hamiltonian Systems".
    - Cloud, section, FFT, and range metrics are local finite-time diagnostics,
      not proofs of hiddenness. See ``docs/code_reference_map.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from ..models.chua import ChuaParameters, chua_nonsmooth_parameters, equilibria_nonsmooth, rhs_nonsmooth
from ..systems.base import ChaoticSystem


@dataclass(frozen=True)
class RobustnessCase:
    """Numerical contract for one robustness trajectory.

    Attributes
    ----------
    case_id : str
        Human-readable label, e.g. ``'R0_base'`` or ``'R1_h_finer'``.
    q : float, default 0.9998
        Caputo fractional order.
    h : float, default 0.01
        Integration step size.
    Lm : float, default 10.0
        Memory length (truncation parameter for EFORK).
    t_final : float, default 1500.0
        Total integration time.
    t_burn : float, default 100.0
        Burn-in time discarded before recording.

    Notes
    -----
    For chaotic attractors, robustness means *geometric* similarity of
    invariant features (attractor shape, section clouds, FFT peak), not
    pointwise trajectory agreement.
    """

    case_id: str
    q: float = 0.9998
    h: float = 0.01
    Lm: float = 10.0
    t_final: float = 1500.0
    t_burn: float = 100.0

    def as_dict(self, baseline: "RobustnessCase | None" = None) -> Dict[str, float | str]:
        base = baseline or self
        return {
            "case_id": self.case_id,
            "q": self.q,
            "h": self.h,
            "Lm": self.Lm,
            "t_final": self.t_final,
            "t_burn": self.t_burn,
            "h_change_pct": 100.0 * (self.h - base.h) / base.h,
            "Lm_change_pct": 100.0 * (self.Lm - base.Lm) / base.Lm,
            "t_final_change_pct": 100.0 * (self.t_final - base.t_final) / base.t_final,
        }


def default_robustness_cases(q: float = 0.9998) -> list[RobustnessCase]:
    """Return the standard six-case h/Lm/time perturbation set.

    Parameters
    ----------
    q : float, default 0.9998
        Caputo fractional order shared by all cases.

    Returns
    -------
    cases : list[RobustnessCase]
        Six cases: base, finer step, coarser step, lower memory,
        higher memory, and longer integration.

    Examples
    --------
    >>> from hidden_attractors.analysis.trajectory import default_robustness_cases
    >>> cases = default_robustness_cases()
    >>> cases[0].case_id
    'R0_base'
    >>> len(cases)
    6
    """

    return [
        RobustnessCase("R0_base", q=q, h=0.01, Lm=10.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R1_h_finer", q=q, h=0.005, Lm=10.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R2_h_coarser", q=q, h=0.02, Lm=10.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R3_Lm_lower", q=q, h=0.01, Lm=5.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R4_Lm_higher", q=q, h=0.01, Lm=20.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R5_t_longer", q=q, h=0.01, Lm=10.0, t_final=3000.0, t_burn=200.0),
    ]


def component_fft(values: np.ndarray, h: float) -> Tuple[float, float]:
    """Return the dominant FFT frequency and normalised spectral entropy.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
        One-dimensional time series of a state component.
    h : float
        Sample step size (seconds).

    Returns
    -------
    peak_freq : float
        Frequency (Hz) at the spectral maximum.  ``nan`` if the series
        has fewer than 9 finite samples.
    entropy : float
        Normalised Shannon entropy of the power spectrum (0–1).
        ``nan`` if the series is too short or constant.
    """

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size <= 8:
        return float("nan"), float("nan")
    data = data - float(np.mean(data))
    if not np.any(np.abs(data) > 0.0):
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(data * np.hanning(data.size))) ** 2
    freq = np.fft.rfftfreq(data.size, d=float(h))
    if spec.size <= 1 or not np.any(spec[1:] > 0.0):
        return 0.0, 0.0
    idx = 1 + int(np.argmax(spec[1:]))
    prob = spec[1:] / max(float(np.sum(spec[1:])), 1e-300)
    entropy = -float(np.sum(prob * np.log(prob + 1e-300))) / max(math.log(prob.size), 1e-300)
    return float(freq[idx]), entropy


def state_view(traj: np.ndarray) -> np.ndarray:
    """Extract state columns from a trajectory array.

    Parameters
    ----------
    traj : np.ndarray
        Either a ``(N, d)`` pure-state array or a ``(N, d+1)`` array
        whose first column is time (i.e. ``(N, ≥4)`` is treated as
        ``t, states...``).

    Returns
    -------
    states : np.ndarray, shape (N, d)
        State-only columns.  Returns ``(0, 0)`` for non-2-D input.
    """

    X = np.asarray(traj, dtype=float)
    if X.ndim != 2:
        return np.empty((0, 0), dtype=float)
    if X.shape[1] >= 4:
        return X[:, 1:]
    return X


def min_distance_to_points(state: np.ndarray, points: Iterable[np.ndarray]) -> float:
    """Return the Euclidean distance from *state* to the nearest point in *points*.

    Parameters
    ----------
    state : np.ndarray, shape (d,)
        Query state vector.
    points : iterable of np.ndarray
        Reference points to measure distance to.

    Returns
    -------
    dist : float
        Minimum distance.  ``nan`` if *points* is empty.
    """

    s = np.asarray(state, dtype=float)
    pts = [np.asarray(point, dtype=float) for point in points]
    if not pts:
        return float("nan")
    return float(min(np.linalg.norm(s - point) for point in pts))


def system_equilibria(system: ChaoticSystem, parameters: Dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Return equilibria from a registered system, raising if none are defined.

    Parameters
    ----------
    system : ChaoticSystem
        System with an equilibrium provider.
    parameters : dict[str, Any] or None, default None
        Override parameters for the equilibrium computation.

    Returns
    -------
    equilibria : dict[str, np.ndarray]
        Maps labels to state vectors.

    Raises
    ------
    ValueError
        If ``system.equilibria`` is ``None``.
    """

    equilibria = system.equilibrium_points(parameters)
    if not equilibria:
        raise ValueError(f"{system.name} must define equilibria for hiddenness checks.")
    return equilibria


def classify_trajectory_against_equilibria(
    traj: np.ndarray,
    equilibria: Dict[str, np.ndarray],
    *,
    divergence_norm: float = 120.0,
    equilibrium_tol: float = 1.0e-3,
    t_start: float | None = None,
) -> Dict[str, Any]:
    """Classify a trajectory's boundedness and final proximity to equilibria.

    Parameters
    ----------
    traj : np.ndarray, shape (N, ≥4)
        Trajectory with columns ``(t, states...)``, or a pure state array.
    equilibria : dict[str, np.ndarray]
        Named equilibrium points to compare against.
    divergence_norm : float, default 120.0
        State norm above which the trajectory is declared diverged.
    equilibrium_tol : float, default 1e-3
        Distance threshold for declaring equilibrium contact.
    t_start : float or None, default None
        If set, only rows with ``t >= t_start`` are used for classification.

    Returns
    -------
    result : dict[str, Any]
        Keys: ``'bounded'``, ``'diverged'``, ``'equilibrium_hit'``,
        ``'closest_equilibrium'``, ``'closest_equilibrium_distance'``,
        ``'final_class'``, ``'final_norm'``, ``'max_norm'``.
    """

    X = np.asarray(traj, dtype=float)
    if t_start is not None and X.ndim == 2 and X.shape[1] >= 2:
        X = X[X[:, 0] >= float(t_start)]
    states = state_view(X)
    finite = bool(states.size > 0 and np.all(np.isfinite(states)))
    norms = np.linalg.norm(states, axis=1) if states.size else np.array([float("inf")])
    final = states[-1] if states.size else np.full(1, float("nan"))
    distances = {
        name: float(np.linalg.norm(final - np.asarray(eq, dtype=float)))
        for name, eq in equilibria.items()
        if np.asarray(eq, dtype=float).shape == final.shape
    }
    closest_eq = min(distances, key=distances.get) if distances else ""
    closest_dist = distances[closest_eq] if closest_eq else float("nan")
    diverged = bool((not finite) or float(np.max(norms)) > float(divergence_norm))
    equilibrium_hit = bool(finite and np.isfinite(closest_dist) and closest_dist <= float(equilibrium_tol))
    return {
        "bounded": bool(finite and not diverged),
        "diverged": diverged,
        "equilibrium_hit": equilibrium_hit,
        "closest_equilibrium": closest_eq,
        "closest_equilibrium_distance": closest_dist,
        "final_class": f"equilibrium_{closest_eq}" if equilibrium_hit else ("diverged" if diverged else "bounded_nontrivial"),
        "final_norm": float(np.linalg.norm(final)),
        "max_norm": float(np.max(norms)),
    }


def trajectory_metrics_for_system(
    traj: np.ndarray,
    *,
    system: ChaoticSystem | None = None,
    equilibria: Dict[str, np.ndarray] | None = None,
    h: float,
    t_start: float,
    divergence_norm: float = 120.0,
    equilibrium_tol: float = 1.0e-3,
) -> Dict[str, Any]:
    """Compute dimension-agnostic trajectory metrics for a registered system.

    Parameters
    ----------
    traj : np.ndarray, shape (N, d+1)
        Trajectory with time in column 0 and *d* state columns.
    system : ChaoticSystem or None, default None
        Used to obtain equilibria if *equilibria* is not supplied.
    equilibria : dict[str, np.ndarray] or None, default None
        Named equilibrium points.  If ``None``, fetched from *system*.
    h : float
        Integration step size for FFT.
    t_start : float
        Burn-in cutoff.
    divergence_norm : float, default 120.0
        State norm above which the trajectory is declared diverged.
    equilibrium_tol : float, default 1e-3
        Distance threshold for equilibrium contact.

    Returns
    -------
    metrics : dict[str, Any]
        Scalar diagnostics: classification flags, per-component ranges,
        variances, FFT peak, and spectral entropy.

    Raises
    ------
    ValueError
        If both *system* and *equilibria* are ``None``.
    """

    if equilibria is None:
        if system is None:
            raise ValueError("provide either system or equilibria.")
        equilibria = system_equilibria(system)
    X = np.asarray(traj, dtype=float)
    tail = tail_view(X, t_start=t_start)
    states = state_view(tail if tail.shape[0] else X)
    cls = classify_trajectory_against_equilibria(
        X,
        equilibria,
        divergence_norm=divergence_norm,
        equilibrium_tol=equilibrium_tol,
        t_start=t_start,
    )
    ranges = np.ptp(states, axis=0) if states.size else np.empty(0, dtype=float)
    variances = np.var(states, axis=0) if states.size else np.empty(0, dtype=float)
    peak, entropy = component_fft(states[:, 0], h) if states.shape[0] and states.shape[1] else (float("nan"), float("nan"))
    out: Dict[str, Any] = {
        **cls,
        "dimension": int(states.shape[1]) if states.ndim == 2 else 0,
        "fft_peak_component_0": peak,
        "psd_entropy_component_0": entropy,
    }
    for idx, value in enumerate(ranges):
        out[f"range_{idx}"] = float(value)
    for idx, value in enumerate(variances):
        out[f"var_{idx}_tail"] = float(value)
    return out


def trajectory_ranges(traj: np.ndarray) -> Dict[str, float]:
    """Compute coordinate ranges for the ``x``, ``y``, ``z`` columns.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory array with columns ``(t, x, y, z)`` or a pure state
        array.  Only the first three state columns are used.

    Returns
    -------
    ranges : dict[str, float]
        Keys ``'range_x'``, ``'range_y'``, ``'range_z'``.  Values are
        ``nan`` if the state is empty.
    """

    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else X
    if states.size == 0:
        return {"range_x": float("nan"), "range_y": float("nan"), "range_z": float("nan")}
    values = np.ptp(states, axis=0)
    return {"range_x": float(values[0]), "range_y": float(values[1]), "range_z": float(values[2])}


def tail_view(traj: np.ndarray, *, t_start: float) -> np.ndarray:
    """Return trajectory rows with ``t >= t_start``.

    Parameters
    ----------
    traj : np.ndarray, shape (N, ≥4)
        Full trajectory array with time in column 0.
    t_start : float
        Burn-in cutoff.  Rows with ``t < t_start`` are dropped.

    Returns
    -------
    tail : np.ndarray, shape (M, 4)
        Post-burn-in rows.  Returns empty ``(0, 4)`` if *traj* has
        fewer than 4 columns or is not 2-D.
    """

    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4:
        return np.empty((0, 4), dtype=float)
    return X[X[:, 0] >= float(t_start)]


def sample_rows(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Subsample rows evenly, preserving the first and last rows.

    Parameters
    ----------
    arr : np.ndarray, shape (N, ...)
        Array to subsample along axis 0.
    max_points : int
        Maximum number of rows to keep.

    Returns
    -------
    sampled : np.ndarray
        Subsampled array.  Returned unchanged if ``N <= max_points``.
    """

    X = np.asarray(arr)
    if X.shape[0] <= int(max_points):
        return X
    idx = np.linspace(0, X.shape[0] - 1, int(max_points)).astype(int)
    return X[idx]


def section_points(
    traj: np.ndarray,
    *,
    t_start: float,
    max_points: int,
    params: ChuaParameters | None = None,
) -> np.ndarray:
    """Compute upward ``x=0`` Poincaré-section points ``(y, z)``.

    Records the linearly interpolated ``(y, z)`` crossing whenever the
    trajectory passes through ``x=0`` from below with positive x-velocity.

    Parameters
    ----------
    traj : np.ndarray, shape (N, 4)
        Trajectory array with columns ``(t, x, y, z)``.
    t_start : float
        Only crossings after *t_start* are recorded.
    max_points : int
        Maximum number of section points to return.
    params : ChuaParameters or None, default None
        Chua parameters used to evaluate the x-velocity sign.
        Defaults to :func:`~hidden_attractors.models.chua.chua_nonsmooth_parameters`.

    Returns
    -------
    pts : np.ndarray, shape (K, 2)
        Poincaré section points ``(y, z)``.  Empty ``(0, 2)`` if none found.

    Notes
    -----
    Section clouds are more informative than raw trajectories for comparing
    chaotic attractors that cannot be compared pointwise.
    """

    p = params or chua_nonsmooth_parameters()
    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4 or X.shape[0] < 2:
        return np.empty((0, 2), dtype=float)
    pts: List[Tuple[float, float]] = []
    for k in range(1, X.shape[0]):
        if X[k, 0] < float(t_start):
            continue
        xp, x = X[k - 1, 1], X[k, 1]
        if xp < 0.0 <= x and rhs_nonsmooth(X[k, 1:4], p)[0] > 0.0:
            lam = (0.0 - xp) / ((x - xp) + 1e-300)
            y = X[k - 1, 2] + lam * (X[k, 2] - X[k - 1, 2])
            z = X[k - 1, 3] + lam * (X[k, 3] - X[k - 1, 3])
            pts.append((float(y), float(z)))
            if len(pts) >= int(max_points):
                break
    return np.asarray(pts, dtype=float)


def cloud_median_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric median nearest-neighbour distance between two point clouds.

    Parameters
    ----------
    a : np.ndarray, shape (M, d)
        First point cloud.
    b : np.ndarray, shape (N, d)
        Second point cloud.

    Returns
    -------
    dist : float
        Median of the union of one-way nearest-neighbour distances
        ``A→B`` and ``B→A``.  ``nan`` if either cloud is empty.

    Notes
    -----
    Uses a block-wise computation to bound peak memory to ``O(128 × N)``
    per block.
    """

    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.size == 0 or B.size == 0:
        return float("nan")

    def one_way(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        vals: List[np.ndarray] = []
        for i in range(0, P.shape[0], 128):
            block = P[i : i + 128]
            d = np.linalg.norm(block[:, None, :] - Q[None, :, :], axis=2)
            vals.append(np.min(d, axis=1))
        return np.concatenate(vals) if vals else np.empty(0)

    return float(np.median(np.concatenate([one_way(A, B), one_way(B, A)])))


def min_equilibrium_distance(state: np.ndarray, params: ChuaParameters | None = None) -> float:
    """Distance from *state* to the nearest non-smooth Chua equilibrium.

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        Query state vector.
    params : ChuaParameters or None, default None
        Chua parameters used to compute equilibria.
        Defaults to :func:`~hidden_attractors.models.chua.chua_nonsmooth_parameters`.

    Returns
    -------
    dist : float
        Minimum Euclidean distance to ``E0``, ``E+``, or ``E-``.
    """

    s = np.asarray(state, dtype=float)
    return float(min(np.linalg.norm(s - eq) for eq in equilibria_nonsmooth(params).values()))


def trajectory_metrics(
    traj: np.ndarray,
    *,
    h: float,
    t_start: float,
    divergence_norm: float = 120.0,
    equilibrium_tol: float = 1.0e-3,
    max_section_points: int = 300,
    max_cloud_points: int = 1000,
    reference: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute geometric and spectral diagnostics for one Chua trajectory.

    Parameters
    ----------
    traj : np.ndarray, shape (N, 4)
        Trajectory array with columns ``(t, x, y, z)``.
    h : float
        Integration step size used for FFT frequency axis.
    t_start : float
        Burn-in cutoff; only the tail ``t >= t_start`` is used for metrics.
    divergence_norm : float, default 120.0
        State norm threshold for declaring divergence.
    equilibrium_tol : float, default 1e-3
        Distance threshold for declaring equilibrium proximity.
    max_section_points : int, default 300
        Maximum Poincaré section points to collect.
    max_cloud_points : int, default 1000
        Maximum tail-cloud rows kept for subsequent comparison.
    reference : dict[str, Any] or None, default None
        Payload from a reference trajectory.  When supplied, relative
        distance metrics are computed against the reference cloud, section,
        and range vector.

    Returns
    -------
    metrics : dict[str, Any]
        Flat dictionary of scalar diagnostics including ``'bounded'``,
        ``'diverged'``, ``'equilibrium_like'``, ``'noncollapsed_variance'``,
        range / variance / FFT / section / cloud-distance entries.
    payload : dict[str, Any]
        Stores ``'tail_sample'``, ``'section'``, ``'range_vec'``, and
        ``'fft_peak'`` for passing as *reference* to subsequent cases.

    Notes
    -----
    All metrics are local finite-time diagnostics.  They do not constitute
    proof of hiddenness; that requires separate equilibrium-neighborhood
    controls documented in the hiddenness protocol.
    """

    X = np.asarray(traj, dtype=float)
    tail = tail_view(X, t_start=t_start)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3))
    tail_states = tail[:, 1:4] if tail.shape[0] else np.empty((0, 3))
    finite = bool(states.size > 0 and np.all(np.isfinite(states)))
    norms = np.linalg.norm(states, axis=1) if states.size else np.array([float("inf")])
    final = states[-1] if states.size else np.array([float("nan"), float("nan"), float("nan")])
    ranges = trajectory_ranges(tail if tail.shape[0] else X)
    var = np.var(tail_states, axis=0) if tail_states.size else np.array([float("nan"), float("nan"), float("nan")])
    peak, entropy = component_fft(tail[:, 1] if tail.shape[0] else X[:, 1], h)
    section = section_points(X, t_start=t_start, max_points=max_section_points)
    range_vec = np.array([ranges["range_x"], ranges["range_y"], ranges["range_z"]], dtype=float)
    payload = {
        "tail_sample": sample_rows(tail_states, max_cloud_points),
        "section": section,
        "range_vec": range_vec,
        "fft_peak": peak,
    }
    diverged = bool((not finite) or float(np.max(norms)) > float(divergence_norm))
    eq_like = bool(finite and min_equilibrium_distance(final) <= float(equilibrium_tol))
    noncollapsed = bool(np.nanmax(var) > 1.0e-6 and np.nanmax(range_vec) > 1.0e-2)
    metrics: Dict[str, Any] = {
        "bounded": bool(finite and not diverged),
        "diverged": diverged,
        "equilibrium_like": eq_like,
        "noncollapsed_variance": noncollapsed,
        "final_norm": float(np.linalg.norm(final)),
        "max_norm": float(np.max(norms)),
        **ranges,
        "var_x_tail": float(var[0]),
        "var_y_tail": float(var[1]),
        "var_z_tail": float(var[2]),
        "fft_peak": peak,
        "psd_entropy": entropy,
        "section_points": int(section.shape[0]) if section.ndim == 2 else 0,
    }
    if reference is not None:
        ref_range = np.asarray(reference["range_vec"], dtype=float)
        denom = max(float(np.linalg.norm(ref_range)), 1.0e-12)
        metrics["range_relative_distance"] = float(np.linalg.norm(range_vec - ref_range) / denom)
        ref_fft = float(reference.get("fft_peak", float("nan")))
        metrics["fft_relative_delta"] = float(abs(peak - ref_fft) / max(abs(ref_fft), 1.0e-12)) if math.isfinite(ref_fft) else float("nan")
        cloud = cloud_median_distance(payload["tail_sample"], np.asarray(reference["tail_sample"], dtype=float))
        secd = cloud_median_distance(payload["section"], np.asarray(reference["section"], dtype=float))
        metrics["cloud_median_distance"] = cloud
        metrics["cloud_median_distance_norm"] = cloud / denom if math.isfinite(cloud) else float("nan")
        metrics["section_median_distance"] = secd
        metrics["section_median_distance_norm"] = secd / denom if math.isfinite(secd) else float("nan")
    else:
        metrics.update(
            {
                "range_relative_distance": float("nan"),
                "fft_relative_delta": float("nan"),
                "cloud_median_distance": float("nan"),
                "cloud_median_distance_norm": float("nan"),
                "section_median_distance": float("nan"),
                "section_median_distance_norm": float("nan"),
            }
        )
    return metrics, payload
