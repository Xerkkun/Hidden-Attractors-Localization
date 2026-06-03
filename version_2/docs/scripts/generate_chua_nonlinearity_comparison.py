from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures" / "chua_fractional_report"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = 0.4
    n = -1.1585
    rho = 1.0
    a1 = m
    a2 = n - m
    x = np.linspace(-6.0, 6.0, 2000)

    f_piecewise = m * x + 0.5 * (n - m) * (np.abs(x + 1.0) - np.abs(x - 1.0))
    f_arctan = a1 * x + a2 * np.arctan(rho * x)
    d_piecewise = np.where(np.abs(x) < 1.0, n, m)
    d_arctan = a1 + a2 * rho / (1.0 + (rho * x) ** 2)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), dpi=180)
    axes[0].plot(x, f_piecewise, lw=1.8, color="#245a9b", label="definida a trozos")
    axes[0].plot(x, f_arctan, lw=1.8, color="#b15f18", label="arctan suave")
    axes[0].axvline(-1.0, color="#777777", ls="--", lw=0.8)
    axes[0].axvline(1.0, color="#777777", ls="--", lw=0.8)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Caracteristica no lineal")
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].plot(x, d_piecewise, lw=1.8, color="#245a9b", label="pendiente a trozos")
    axes[1].plot(x, d_arctan, lw=1.8, color="#b15f18", label="pendiente arctan")
    axes[1].axvline(-1.0, color="#777777", ls="--", lw=0.8)
    axes[1].axvline(1.0, color="#777777", ls="--", lw=0.8)
    axes[1].axhline(m, color="#2f7d4f", ls=":", lw=1.1, label="pendiente exterior m")
    axes[1].axhline(n, color="#7a3b9f", ls=":", lw=1.1, label="pendiente interior n")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("pendiente")
    axes[1].set_title("Pendiente: saltos vs suavidad")
    axes[1].legend(frameon=True, fontsize=8)

    fig.suptitle("Comparacion de no linealidades Chua: trozos y arctan (Wu et al. 2023)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chua_nonlinearity_piecewise_vs_arctan.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "figure": "chua_nonlinearity_piecewise_vs_arctan.png",
        "article_convention": "f_arctan(x)=m*x+(n-m)*atan(x)",
        "piecewise_comparison": "f_pwl(x)=m*x+0.5*(n-m)*(|x+1|-|x-1|)",
        "m": m,
        "n": n,
        "a1": a1,
        "a2": a2,
        "rho": rho,
    }
    (OUT_DIR / "chua_nonlinearity_piecewise_vs_arctan_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
