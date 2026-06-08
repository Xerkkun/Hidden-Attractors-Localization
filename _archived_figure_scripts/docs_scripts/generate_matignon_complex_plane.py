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

    q_values = [1.0, 0.8, 0.6]
    colors = ["#b73535", "#245a9b", "#2f7d4f"]
    labels = [r"$q=1.0$", r"$q=0.8$", r"$q=0.6$"]

    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=180)
    radius = 1.0
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="#888888", lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.annotate(r"$\mathrm{Re}(\lambda)$", xy=(1.12, 0.0), xytext=(1.12, 0.06), ha="center", fontsize=10)
    ax.annotate(r"$\mathrm{Im}(\lambda)$", xy=(0.0, 1.12), xytext=(0.13, 1.08), ha="left", fontsize=10)

    q_fill = 0.8
    phi = q_fill * np.pi / 2.0
    angles = np.linspace(phi, 2.0 * np.pi - phi, 600)
    ax.fill(
        np.concatenate([[0.0], radius * np.cos(angles), [0.0]]),
        np.concatenate([[0.0], radius * np.sin(angles), [0.0]]),
        color="#245a9b",
        alpha=0.11,
        label=r"zona estable para $q=0.8$",
    )

    for q_value, color, label in zip(q_values, colors, labels):
        phi = q_value * np.pi / 2.0
        for sign in [1.0, -1.0]:
            angle = sign * phi
            ax.plot([0.0, radius * np.cos(angle)], [0.0, radius * np.sin(angle)], color=color, lw=1.7)
        ax.text(0.74 * np.cos(phi), 0.74 * np.sin(phi) + 0.035, label, color=color, fontsize=9, ha="center")
        ax.text(0.74 * np.cos(-phi), 0.74 * np.sin(-phi) - 0.07, label, color=color, fontsize=9, ha="center")

    stable = np.array([[-0.6, 0.55], [-0.6, -0.55], [-0.9, 0.0]])
    unstable = np.array([[0.22, 0.72], [0.22, -0.72]])
    ax.scatter(stable[:, 0], stable[:, 1], s=40, color="#245a9b", marker="o", label="autovalores admisibles")
    ax.scatter(unstable[:, 0], unstable[:, 1], s=52, color="#b73535", marker="x", label="autovalores no admisibles")

    ax.text(
        -0.95,
        0.92,
        r"estable si $|\arg(\lambda_i)|>q\pi/2$",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92},
    )
    ax.text(0.25, 0.10, "sector inestable", fontsize=9, color="#6b2b2b")
    ax.text(-0.82, -0.18, "sector estable", fontsize=9, color="#245a9b")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("parte real")
    ax.set_ylabel("parte imaginaria")
    ax.set_title("Criterio de Matignon para sistemas conmensurados de Caputo")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matignon_complex_plane.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "figure": "matignon_complex_plane.png",
        "criterion": "|arg(lambda_i)| > q*pi/2 for all eigenvalues lambda_i of A",
        "q_values_drawn": q_values,
        "filled_example_q": q_fill,
    }
    (OUT_DIR / "matignon_complex_plane_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
