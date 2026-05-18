from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import chua_initial_cond as chua
from extended_search_utils import write_csv
from harmonic_diagnostics import RHO_H_FIELDS, rho_h_diagnostic


BIASED_FIELDS = [
    "candidate_id",
    "A",
    "sigma0",
    "omega",
    "q",
    "mu",
    "N_re",
    "N_im",
    "W_re",
    "W_im",
    "residual_abs",
    "rho_H",
    "seed_x",
    "seed_y",
    "seed_z",
    "df_family",
    "machado_branch",
    "valid",
    "invalid_reason",
]


def latin_hypercube(bounds: Sequence[tuple[float, float]], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    d = len(bounds)
    u = (rng.random((n, d)) + np.arange(n).reshape(n, 1)) / max(n, 1)
    for j in range(d):
        rng.shuffle(u[:, j])
    out = np.empty_like(u)
    for j, (lo, hi) in enumerate(bounds):
        out[:, j] = float(lo) + u[:, j] * (float(hi) - float(lo))
    return out


def grid_points(cfg: Dict[str, Any]) -> np.ndarray:
    search = cfg["biased_search"]
    A = np.linspace(float(cfg["amplitude"]["A_min"]), float(cfg["amplitude"]["A_max"]), int(search.get("A_count", 16)))
    s0 = np.linspace(float(search["sigma0_min"]), float(search["sigma0_max"]), int(search.get("sigma0_count", 17)))
    w = np.linspace(float(cfg["frequency"]["omega_min"]), float(cfg["frequency"]["omega_max"]), int(search.get("omega_count", 24)))
    AA, SS, WW = np.meshgrid(A, s0, w, indexing="ij")
    return np.column_stack([AA.ravel(), SS.ravel(), WW.ravel()])


def sampled_search_points(cfg: Dict[str, Any]) -> np.ndarray:
    pts = []
    if cfg.get("biased_search", {}).get("use_grid", True):
        pts.append(grid_points(cfg))
    n_lhs = int(cfg.get("biased_search", {}).get("lhs_count", 0))
    if n_lhs > 0:
        bounds = [
            (float(cfg["amplitude"]["A_min"]), float(cfg["amplitude"]["A_max"])),
            (float(cfg["biased_search"]["sigma0_min"]), float(cfg["biased_search"]["sigma0_max"])),
            (float(cfg["frequency"]["omega_min"]), float(cfg["frequency"]["omega_max"])),
        ]
        pts.append(latin_hypercube(bounds, n_lhs, int(cfg.get("random_seed", 123456789))))
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.vstack(pts)


def evaluate_biased_candidate(
    candidate_id: str,
    df_family: str,
    A: float,
    sigma0: float,
    omega: float,
    q: float,
    p: Dict[str, Any],
    cfg: Dict[str, Any],
    mu: float | None = None,
) -> Dict[str, Any]:
    diag = rho_h_diagnostic(
        candidate_id=candidate_id,
        df_family=df_family,
        A=A,
        sigma0=sigma0,
        omega=omega,
        q=q,
        p=p,
        mu=mu,
        K=int(cfg["rho_H"].get("K", 10)),
        n_quad=int(cfg["rho_H"].get("n_quad", 4096)),
        threshold=float(cfg["rho_H"].get("threshold", 0.1)),
        machado_branch=int(cfg.get("machado", {}).get("branch", 0)),
        machado_eps=float(cfg.get("machado", {}).get("zero_eps", 1e-12)),
    )
    valid = not bool(diag.get("invalid_reason"))
    seed = np.array([np.nan, np.nan, np.nan], dtype=float)
    if valid:
        try:
            seed, _xbar, _V, _fourier = chua.reconstruct_biased_lure_seed(
                q, p, A, sigma0, omega,
                theta=float(cfg.get("biased_search", {}).get("seed_phase", 0.0)),
                K=int(cfg["rho_H"].get("K", 10)),
                n_quad=int(cfg["rho_H"].get("n_quad", 4096)),
            )
        except Exception as exc:
            valid = False
            diag["invalid_reason"] = str(exc)
    return {
        **{k: diag.get(k) for k in RHO_H_FIELDS if k in diag},
        "candidate_id": candidate_id,
        "df_family": df_family,
        "mu": "" if mu is None else float(mu),
        "seed_x": float(seed[0]),
        "seed_y": float(seed[1]),
        "seed_z": float(seed[2]),
        "machado_branch": int(cfg.get("machado", {}).get("branch", 0)),
        "valid": bool(valid),
        "invalid_reason": diag.get("invalid_reason", ""),
    }


def search_biased_candidates(cfg: Dict[str, Any], p: Dict[str, Any], outdir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    q = float(cfg["q"])
    points = sampled_search_points(cfg)
    rows: List[Dict[str, Any]] = []
    families = list(cfg.get("biased_search", {}).get("families", ["classical_biased", "machado_biased"]))
    mu_values = [float(v) for v in cfg.get("machado", {}).get("mu_values", [1.0])]
    max_rows = int(cfg.get("biased_search", {}).get("max_evaluations", 0))
    if max_rows > 0 and points.shape[0] > max_rows:
        points = points[:max_rows]

    idx = 0
    for A, sigma0, omega in points:
        for family in families:
            if family == "machado_biased":
                for mu in mu_values:
                    rows.append(evaluate_biased_candidate(f"biased_{idx:06d}_machado_mu_{mu:.5g}", family, A, sigma0, omega, q, p, cfg, mu=mu))
                    idx += 1
            else:
                rows.append(evaluate_biased_candidate(f"biased_{idx:06d}_classical", family, A, sigma0, omega, q, p, cfg, mu=None))
                idx += 1

    valid_rows = [r for r in rows if r.get("valid") and np.isfinite(float(r.get("residual_abs", np.nan)))]
    valid_rows.sort(key=lambda r: (float(r["residual_abs"]), float(r.get("rho_H", np.inf))))
    keep = int(cfg.get("biased_search", {}).get("keep_best", 40))
    selected = valid_rows[:keep]
    write_csv(Path(outdir) / "biased_df_candidates.csv", selected, BIASED_FIELDS)
    write_csv(Path(outdir) / "biased_df_all_evaluations.csv", rows, BIASED_FIELDS)
    return selected, rows
