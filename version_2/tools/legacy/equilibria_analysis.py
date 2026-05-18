from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import chua_initial_cond as chua
from extended_search_utils import write_csv


EQ_FIELDS = [
    "eq_id",
    "x",
    "y",
    "z",
    "region",
    "eig_1",
    "eig_2",
    "eig_3",
    "min_arg_margin",
    "matignon_stable",
    "nonsmooth_boundary",
]


def region_for_sigma(sigma: float, tol: float = 1e-10) -> str:
    if abs(abs(float(sigma)) - 1.0) <= tol:
        return "switching_boundary"
    if sigma < -1.0:
        return "left_saturation"
    if sigma > 1.0:
        return "right_saturation"
    return "central_linear"


def solve_piecewise_equilibria(p: Dict[str, Any]) -> Dict[str, np.ndarray]:
    beta = float(p["beta"])
    gamma = float(p["gamma"])
    m0 = float(p["m0"])
    m1 = float(p["m1"])
    A = m0 - m1
    eqs: Dict[str, np.ndarray] = {}

    def make_eq(x: float) -> np.ndarray:
        y = gamma / (beta + gamma) * x
        z = -beta / (beta + gamma) * x
        return np.array([x, y, z], dtype=float)

    # Region central: sat(x)=x. La ecuacion siempre contiene x=0 para el
    # caso oficial; se verifica la pertenencia a la region.
    E0 = make_eq(0.0)
    if abs(E0[0]) <= 1.0 + 1e-10:
        eqs["E0"] = E0

    den = (beta + gamma) * m1 + beta
    if abs(den) > 1e-14:
        xp = -((beta + gamma) * A) / den
        Ep = make_eq(xp)
        Em = make_eq(-xp)
        if Ep[0] > 1.0 - 1e-10:
            eqs["E+"] = Ep
        if Em[0] < -1.0 + 1e-10:
            eqs["E-"] = Em
    return eqs


def solve_equilibria(p: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if chua.chua_model(p) != "piecewise":
        # Fallback compatible con el helper existente para arctan.
        from run_hidden_verify_frac_hybrid import ChuaParams, chua_equilibria

        hp = ChuaParams(
            alpha_chua=float(p["alpha"]),
            beta=float(p["beta"]),
            gamma_chua=float(p["gamma"]),
            m0=float(p["m0"]),
            m1=float(p["m1"]),
            model=str(p.get("model", "arctan")),
            a1=float(p.get("a1", 0.4)),
            a2=float(p.get("a2", -1.5585)),
            rho=float(p.get("rho", 1.0)),
        )
        return chua_equilibria(hp)
    return solve_piecewise_equilibria(p)


def local_jacobian(p: Dict[str, Any], eq: np.ndarray) -> np.ndarray:
    P, b, r = chua.chua_matrices(p)
    sigma = float(np.asarray(eq, dtype=float) @ r)
    if chua.chua_model(p) == "piecewise":
        slope = float(chua.chua_gain_A(p)) if abs(sigma) < 1.0 - 1e-10 else 0.0
    else:
        rho = float(p.get("rho", 1.0))
        slope = float(p.get("a2", -1.5585)) * rho / (1.0 + (rho * sigma) ** 2)
    return np.asarray(P + slope * np.outer(b, r), dtype=float)


def analyze_equilibria(cfg: Dict[str, Any], p: Dict[str, Any], outdir: Path) -> List[Dict[str, Any]]:
    q = float(cfg["q"])
    theta = q * np.pi / 2.0
    rows: List[Dict[str, Any]] = []
    for eq_id, eq in solve_equilibria(p).items():
        eq = np.asarray(eq, dtype=float)
        if not np.all(np.isfinite(eq)):
            continue
        J = local_jacobian(p, eq)
        eig = np.linalg.eigvals(J)
        margins = [abs(np.angle(v)) - theta for v in eig]
        region = region_for_sigma(float(eq[0]))
        rows.append({
            "eq_id": eq_id,
            "x": float(eq[0]),
            "y": float(eq[1]),
            "z": float(eq[2]),
            "region": region,
            "eig_1": complex(eig[0]),
            "eig_2": complex(eig[1]),
            "eig_3": complex(eig[2]),
            "min_arg_margin": float(min(margins)),
            "matignon_stable": bool(all(m > 0.0 for m in margins)),
            "nonsmooth_boundary": bool(region == "switching_boundary"),
            "point": eq,
        })
    write_csv(Path(outdir) / "equilibria_summary.csv", rows, EQ_FIELDS)
    return rows
