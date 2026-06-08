from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

import chua_initial_cond as chua
from extended_search_utils import write_csv


RHO_H_FIELDS = [
    "candidate_id",
    "df_type",
    "df_family",
    "mu",
    "A",
    "sigma0",
    "omega",
    "q",
    "N_re",
    "N_im",
    "W_re",
    "W_im",
    "residual_re",
    "residual_im",
    "residual_abs",
    "rho_H",
    "K",
    "harmonic_energy_ratio",
    "accepted_by_rhoH",
]


def effective_N(base_N: complex, df_family: str, mu: float | None, branch: int, eps: float) -> complex:
    if "machado" in str(df_family):
        return complex(chua.machado_complex_power(base_N, float(mu if mu is not None else 1.0), branch=branch, eps=eps))
    return complex(base_N)


def rho_h_diagnostic(
    *,
    candidate_id: str,
    df_family: str,
    A: float,
    sigma0: float,
    omega: float,
    q: float,
    p: Dict[str, Any],
    mu: float | None = None,
    K: int = 10,
    n_quad: int = 4096,
    threshold: float = 0.1,
    machado_branch: int = 0,
    machado_eps: float = 1e-12,
) -> Dict[str, Any]:
    fourier = chua.fourier_coefficients_psi(A, sigma0, p, K=K, n_quad=n_quad)
    coeffs = fourier["coefficients"]
    Y1 = complex(coeffs[1]["Y"])
    base_N = Y1 / float(A)
    try:
        N_eff = effective_N(base_N, df_family, mu, machado_branch, machado_eps)
        valid = True
        invalid_reason = ""
    except Exception as exc:
        N_eff = complex(np.nan, np.nan)
        valid = False
        invalid_reason = str(exc)

    W1 = complex(chua.W_frac(omega, q, p))
    residual = 1.0 + W1 * N_eff if valid else complex(np.nan, np.nan)
    denom = abs(W1) * abs(Y1) + 1e-14
    higher = 0.0
    higher_energy = 0.0
    total_energy = abs(Y1) ** 2
    for k in range(2, int(K) + 1):
        Yk = complex(coeffs[k]["Y"])
        higher += abs(chua.W_frac(k * float(omega), q, p)) * abs(Yk)
        higher_energy += abs(Yk) ** 2
        total_energy += abs(Yk) ** 2
    rho_H = float(higher / denom)
    harmonic_energy_ratio = float(higher_energy / max(total_energy, 1e-300))
    return {
        "candidate_id": candidate_id,
        "df_type": "biased" if abs(float(sigma0)) > 1e-14 else "centered",
        "df_family": df_family,
        "mu": "" if mu is None else float(mu),
        "A": float(A),
        "sigma0": float(sigma0),
        "omega": float(omega),
        "q": float(q),
        "N_re": float(np.real(N_eff)),
        "N_im": float(np.imag(N_eff)),
        "W_re": float(np.real(W1)),
        "W_im": float(np.imag(W1)),
        "residual_re": float(np.real(residual)),
        "residual_im": float(np.imag(residual)),
        "residual_abs": float(abs(residual)) if valid else float("nan"),
        "rho_H": rho_H,
        "K": int(K),
        "harmonic_energy_ratio": harmonic_energy_ratio,
        "accepted_by_rhoH": bool(rho_H < float(threshold)),
        "invalid_reason": invalid_reason,
        "fourier": fourier,
        "base_N": base_N,
    }


def centered_candidates_from_nyquist(cfg: Dict[str, Any], p: Dict[str, Any]) -> List[Dict[str, Any]]:
    q = float(cfg["q"])
    raw = chua.find_omega_k_candidates(q, p, wmin=float(cfg["frequency"]["omega_min"]), wmax=float(cfg["frequency"]["omega_max"]), nscan=int(cfg["frequency"].get("nscan", chua.NSCAN)))
    out: List[Dict[str, Any]] = []
    for branch, pair in enumerate(raw):
        omega, k = float(pair[0]), float(pair[1])
        if chua.is_describing_gain_compatible(k, p):
            try:
                A = chua.solve_amplitude_from_k(k, p, amax=float(cfg["amplitude"]["A_max"]))
                out.append({"candidate_id": f"centered_classical_b{branch}", "df_family": "classical", "branch": branch, "A": A, "sigma0": 0.0, "omega": omega, "mu": None})
            except Exception:
                pass
        for mu in cfg.get("machado", {}).get("mu_values", []):
            if chua.is_machado_gain_compatible(k, p, float(mu)):
                try:
                    A = chua.solve_machado_amplitude_from_k(k, p, float(mu), amax=float(cfg["amplitude"]["A_max"]))
                    out.append({"candidate_id": f"centered_machado_b{branch}_mu_{float(mu):.5g}", "df_family": "machado", "branch": branch, "A": A, "sigma0": 0.0, "omega": omega, "mu": float(mu)})
                except Exception:
                    pass
    return out


def write_rho_outputs(rows: Sequence[Dict[str, Any]], outdir: Path) -> None:
    outdir = Path(outdir)
    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    public_rows = [{k: v for k, v in row.items() if k not in {"fourier", "base_N"}} for row in rows]
    write_csv(outdir / "rho_H_diagnostics.csv", public_rows, RHO_H_FIELDS)

    accepted = [r for r in rows if np.isfinite(float(r.get("rho_H", np.nan)))]
    if accepted:
        for xkey, name in [("A", "rho_H_vs_A.png"), ("omega", "rho_H_vs_omega.png")]:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for family in sorted({str(r["df_family"]) for r in accepted}):
                sub = [r for r in accepted if str(r["df_family"]) == family]
                ax.scatter([r[xkey] for r in sub], [r["rho_H"] for r in sub], s=18, label=family)
            ax.set_xlabel("A" if xkey == "A" else "omega")
            ax.set_ylabel("rho_H")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            from version_2.hidden_attractors.plotting.export import intercept_and_export_path
            intercept_and_export_path(fig, plots / name, 'attractor')
            plt.close(fig)


def plot_harmonic_spectrum(row: Dict[str, Any], outdir: Path) -> str:
    coeffs = row["fourier"]["coefficients"]
    ks = np.array(sorted(coeffs), dtype=int)
    vals = np.array([abs(coeffs[int(k)]["Y"]) for k in ks], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(ks, vals, color="#2563eb")
    ax.set_xlabel("k")
    ax.set_ylabel("|Y_k|")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = Path(outdir) / "plots" / f"harmonic_spectrum_candidate_{row['candidate_id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    from version_2.hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, path, 'attractor')
    plt.close(fig)
    return str(path)
