from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import chua_initial_cond as chua
from extended_search_utils import fft_peak_and_entropy, min_distance_to_points, write_csv


SEED_FIELDS = [
    "seed_id",
    "x0",
    "y0",
    "z0",
    "min_dist_to_equilibria_initial",
    "min_dist_to_equilibria_trajectory",
    "final_class",
    "bounded",
    "diverged",
    "equilibrium_hit",
    "attractor_label",
    "lyap_max",
    "fft_peak",
    "psd_entropy",
    "t_final",
    "h",
    "q",
    "memory_length",
]


def lhs_points(bounds: Sequence[tuple[float, float]], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    d = len(bounds)
    u = (rng.random((n, d)) + np.arange(n).reshape(n, 1)) / max(n, 1)
    for j in range(d):
        rng.shuffle(u[:, j])
    out = np.empty_like(u)
    for j, (lo, hi) in enumerate(bounds):
        out[:, j] = float(lo) + u[:, j] * (float(hi) - float(lo))
    return out


def generate_seed_cloud(cfg: Dict[str, Any], equilibria: Sequence[np.ndarray]) -> np.ndarray:
    sc = cfg["seed_cloud"]
    bounds = [
        (float(sc["x_min"]), float(sc["x_max"])),
        (float(sc["y_min"]), float(sc["y_max"])),
        (float(sc["z_min"]), float(sc["z_max"])),
    ]
    pts = []
    n_axis = int(sc.get("grid_per_axis", 5))
    if n_axis > 1:
        xs = [np.linspace(lo, hi, n_axis) for lo, hi in bounds]
        XX, YY, ZZ = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
        pts.append(np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]))
    n_lhs = int(sc.get("lhs_count", 80))
    if n_lhs > 0:
        pts.append(lhs_points(bounds, n_lhs, int(cfg.get("random_seed", 123456789))))
    if not pts:
        return np.empty((0, 3), dtype=float)
    all_pts = np.vstack(pts)
    delta = float(sc.get("delta_eq", 0.35))
    mask = np.array([min_distance_to_points(p, equilibria) > delta for p in all_pts], dtype=bool)
    return all_pts[mask]


def classify_traj(traj: np.ndarray, x0: np.ndarray, equilibria: Sequence[np.ndarray], cfg: Dict[str, Any]) -> Dict[str, Any]:
    sc = cfg["seed_cloud"]
    states = traj[:, 1:4]
    norms = np.linalg.norm(states, axis=1)
    div_thr = float(sc.get("divergence_norm", 120.0))
    eps_eq = float(sc.get("equilibrium_tol", 0.05))
    t_burn = float(sc.get("t_burn", 40.0))
    burn_idx = int(np.searchsorted(traj[:, 0], t_burn))
    burn_idx = min(max(burn_idx, 0), max(0, len(traj) - 1))
    post = states[burn_idx:]
    initial_dist = min_distance_to_points(x0, equilibria)
    traj_dist = min_distance_to_points(post if post.size else states, equilibria)
    final_dist = min_distance_to_points(states[-1], equilibria)
    diverged = bool((not np.all(np.isfinite(states))) or np.nanmax(norms) > div_thr)
    equilibrium_hit = bool(np.isfinite(final_dist) and final_dist < eps_eq)
    bounded = not diverged
    variances = np.var(post, axis=0) if post.shape[0] > 3 else np.zeros(3)
    nontrivial = bool(np.max(variances) > float(sc.get("variance_tol", 1e-3)))
    if diverged:
        final_class = "divergent"
    elif equilibrium_hit:
        final_class = "equilibrium_convergence"
    elif bounded and nontrivial:
        final_class = "candidate_hidden_like" if traj_dist > float(sc.get("delta_eq", 0.35)) else "candidate_bounded_nontrivial"
    else:
        final_class = "bounded_small"
    fft = fft_peak_and_entropy(traj[burn_idx:] if burn_idx < len(traj) - 8 else traj, h=float(sc.get("h", cfg.get("h", 0.01))))
    return {
        "min_dist_to_equilibria_initial": initial_dist,
        "min_dist_to_equilibria_trajectory": traj_dist,
        "final_class": final_class,
        "bounded": bounded,
        "diverged": diverged,
        "equilibrium_hit": equilibrium_hit,
        "attractor_label": final_class,
        "lyap_max": "",
        "fft_peak": fft["fft_peak"],
        "psd_entropy": fft["psd_entropy"],
        "variance_x": float(variances[0]),
        "variance_y": float(variances[1]),
        "variance_z": float(variances[2]),
    }


def simulate_seed(x0: np.ndarray, p: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    sc = cfg["seed_cloud"]
    return chua.efork3_integrate(
        lambda x: chua.rhs_original(x, p),
        np.asarray(x0, dtype=float),
        qord=float(cfg["q"]),
        h=float(sc.get("h", cfg.get("h", 0.01))),
        Lm=float(sc.get("memory_length", cfg.get("Lm", 8.0))),
        t_f=float(sc.get("t_final", 100.0)),
    )


def plot_seed_classes(rows: Sequence[Dict[str, Any]], outdir: Path) -> None:
    if not rows:
        return
    plots = Path(outdir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    class_colors = {
        "equilibrium_convergence": "#111827",
        "candidate_bounded_nontrivial": "#2563eb",
        "candidate_hidden_like": "#dc2626",
        "divergent": "#9ca3af",
        "bounded_small": "#16a34a",
        "numerical_failure": "#f97316",
    }
    planes = [("x0", "y0", "seed_cloud_classes_xy.png"), ("x0", "z0", "seed_cloud_classes_xz.png"), ("y0", "z0", "seed_cloud_classes_yz.png")]
    for xkey, ykey, name in planes:
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        for cls in sorted({str(r["final_class"]) for r in rows}):
            sub = [r for r in rows if str(r["final_class"]) == cls]
            ax.scatter([float(r[xkey]) for r in sub], [float(r[ykey]) for r in sub], s=18, label=cls, color=class_colors.get(cls, "#6b7280"))
        ax.set_xlabel(xkey.replace("0", ""))
        ax.set_ylabel(ykey.replace("0", ""))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots / name, dpi=180)
        plt.close(fig)


def plot_candidate_attractor(seed_id: str, traj: np.ndarray, outdir: Path) -> str:
    plots = Path(outdir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    stride = max(1, traj.shape[0] // 4000)
    T = traj[::stride]
    ax.plot(T[:, 1], T[:, 2], T[:, 3], lw=0.55, color="#2563eb")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    path = plots / f"candidate_attractor_{seed_id}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def run_seed_cloud_search(cfg: Dict[str, Any], p: Dict[str, Any], equilibria: Sequence[np.ndarray], outdir: Path) -> List[Dict[str, Any]]:
    pts = generate_seed_cloud(cfg, equilibria)
    limit = int(cfg.get("seed_cloud", {}).get("max_seeds", 0))
    if limit > 0:
        pts = pts[:limit]
    rows: List[Dict[str, Any]] = []
    candidate_plots = 0
    for i, x0 in enumerate(pts):
        seed_id = f"seed_{i:05d}"
        try:
            traj = simulate_seed(x0, p, cfg)
            cls = classify_traj(traj, x0, equilibria, cfg)
            if cls["final_class"] in {"candidate_hidden_like", "candidate_bounded_nontrivial"} and candidate_plots < int(cfg.get("seed_cloud", {}).get("max_candidate_plots", 6)):
                plot_candidate_attractor(seed_id, traj, outdir)
                candidate_plots += 1
        except Exception as exc:
            cls = {
                "min_dist_to_equilibria_initial": min_distance_to_points(x0, equilibria),
                "min_dist_to_equilibria_trajectory": "",
                "final_class": "numerical_failure",
                "bounded": False,
                "diverged": False,
                "equilibrium_hit": False,
                "attractor_label": "numerical_failure",
                "lyap_max": "",
                "fft_peak": "",
                "psd_entropy": "",
                "notes": str(exc),
            }
        rows.append({
            "seed_id": seed_id,
            "x0": float(x0[0]),
            "y0": float(x0[1]),
            "z0": float(x0[2]),
            **cls,
            "t_final": float(cfg["seed_cloud"].get("t_final", 100.0)),
            "h": float(cfg["seed_cloud"].get("h", cfg.get("h", 0.01))),
            "q": float(cfg["q"]),
            "memory_length": float(cfg["seed_cloud"].get("memory_length", cfg.get("Lm", 8.0))),
        })
    write_csv(Path(outdir) / "seed_cloud_summary.csv", rows, SEED_FIELDS)
    plot_seed_classes(rows, outdir)
    # Basin-slice placeholders are the same direct cloud classes projected on
    # coordinate planes; full verified basins remain in the existing basin code.
    plots = Path(outdir) / "plots"
    for src, dst in [
        ("seed_cloud_classes_xy.png", "basin_slice_xy.png"),
        ("seed_cloud_classes_xz.png", "basin_slice_xz.png"),
    ]:
        s = plots / src
        if s.exists():
            (plots / dst).write_bytes(s.read_bytes())
    return rows
