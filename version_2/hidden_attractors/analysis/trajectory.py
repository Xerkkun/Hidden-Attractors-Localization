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

from ..models.chua import ChuaParameters, chua_piecewise_parameters, equilibria_piecewise, rhs_piecewise


@dataclass(frozen=True)
class RobustnessCase:
    """Numerical contract for one robustness trajectory.

    Purpose:
        Record exactly what changed when comparing candidate trajectories.

    Validity warning:
        For chaotic attractors, robustness means similarity of invariant or
        geometric features, not pointwise trajectory agreement.
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
    """Return the standard h/Lm/time perturbation set used in examples."""

    return [
        RobustnessCase("R0_base", q=q, h=0.01, Lm=10.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R1_h_finer", q=q, h=0.005, Lm=10.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R2_h_coarser", q=q, h=0.02, Lm=10.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R3_Lm_lower", q=q, h=0.01, Lm=5.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R4_Lm_higher", q=q, h=0.01, Lm=20.0, t_final=1500.0, t_burn=100.0),
        RobustnessCase("R5_t_longer", q=q, h=0.01, Lm=10.0, t_final=3000.0, t_burn=200.0),
    ]


def component_fft(values: np.ndarray, h: float) -> Tuple[float, float]:
    """Return dominant FFT frequency and normalized spectral entropy."""

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


def trajectory_ranges(traj: np.ndarray) -> Dict[str, float]:
    """Compute coordinate ranges for columns ``t,x,y,z`` or state arrays."""

    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else X
    if states.size == 0:
        return {"range_x": float("nan"), "range_y": float("nan"), "range_z": float("nan")}
    values = np.ptp(states, axis=0)
    return {"range_x": float(values[0]), "range_y": float(values[1]), "range_z": float(values[2])}


def tail_view(traj: np.ndarray, *, t_start: float) -> np.ndarray:
    """Return the part of a trajectory after ``t_start``."""

    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4:
        return np.empty((0, 4), dtype=float)
    return X[X[:, 0] >= float(t_start)]


def sample_rows(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Subsample rows evenly without changing endpoints."""

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
    """Compute upward ``x=0`` Poincare-section points ``(y,z)``.

    Purpose:
        Compare chaotic attractors through section clouds rather than pointwise
        trajectory agreement.
    """

    p = params or chua_piecewise_parameters()
    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4 or X.shape[0] < 2:
        return np.empty((0, 2), dtype=float)
    pts: List[Tuple[float, float]] = []
    for k in range(1, X.shape[0]):
        if X[k, 0] < float(t_start):
            continue
        xp, x = X[k - 1, 1], X[k, 1]
        if xp < 0.0 <= x and rhs_piecewise(X[k, 1:4], p)[0] > 0.0:
            lam = (0.0 - xp) / ((x - xp) + 1e-300)
            y = X[k - 1, 2] + lam * (X[k, 2] - X[k - 1, 2])
            z = X[k - 1, 3] + lam * (X[k, 3] - X[k - 1, 3])
            pts.append((float(y), float(z)))
            if len(pts) >= int(max_points):
                break
    return np.asarray(pts, dtype=float)


def cloud_median_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric median nearest-neighbor distance between two point clouds."""

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
    """Distance from one state to the nearest Chua equilibrium."""

    s = np.asarray(state, dtype=float)
    return float(min(np.linalg.norm(s - eq) for eq in equilibria_piecewise(params).values()))


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
    """Compute geometric/spectral diagnostics for one trajectory.

    Output:
        ``(metrics, payload)`` where payload stores tail cloud and section data
        for comparison with subsequent cases.
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
