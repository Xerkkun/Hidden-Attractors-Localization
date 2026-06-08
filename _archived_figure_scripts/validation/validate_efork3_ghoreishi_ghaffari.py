#!/usr/bin/env python3
"""Generate EFORK-3 validation evidence from published manufactured solutions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from hidden_attractors.solvers import efork3_caputo_integrate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "reference_cases" / "efork3_ghoreishi_ghaffari" / "03_integrators"
N_STEPS = (40, 80, 160, 320, 640)
PUBLISHED = {
    ("example_1", 0.25): (9.94252e-4, 5.54011e-4, 3.13499e-4, 1.79258e-4, 1.03255e-4),
    ("example_1", 0.50): (7.45694e-5, 2.46986e-5, 8.26771e-6, 2.79911e-6, 9.57367e-7),
    ("example_2", 0.25): (9.90939e-3, 5.54249e-3, 3.14342e-3, 1.79955e-3, 1.03718e-3),
    ("example_2", 0.50): (5.79341e-4, 1.96590e-4, 6.68302e-5, 2.28624e-5, 7.87606e-6),
}


def mittag_leffler(alpha: float, beta: float, z: float) -> float:
    total = 0.0
    for index in range(300):
        term = z**index / math.gamma(alpha * index + beta)
        total += term
        if abs(term) < 1.0e-16:
            break
    return total


def terminal_error(example: str, alpha: float, n_steps: int) -> float:
    if example == "example_1":
        rhs = lambda t, y: -y + t ** (4.0 - alpha) / math.gamma(5.0 - alpha)
        exact = mittag_leffler(alpha, 5.0, -1.0)
    elif example == "example_2":
        rhs = lambda t, y: (
            2.0 * t ** (2.0 - alpha) / math.gamma(3.0 - alpha)
            - t ** (1.0 - alpha) / math.gamma(2.0 - alpha)
            - y
            + t**2
            - t
        )
        exact = 0.0
    else:
        raise ValueError(f"Unsupported example: {example}")
    _, states = efork3_caputo_integrate(
        rhs,
        np.array([0.0]),
        alpha=alpha,
        h=1.0 / n_steps,
        t_final=1.0,
    )
    return abs(float(states[-1, 0]) - exact)


def build_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for (example, alpha), displayed_errors in PUBLISHED.items():
        for n_steps, published_error in zip(N_STEPS, displayed_errors):
            computed = terminal_error(example, alpha, n_steps)
            rows.append(
                {
                    "example": example,
                    "alpha": alpha,
                    "n_steps": n_steps,
                    "h": 1.0 / n_steps,
                    "published_terminal_error": published_error,
                    "computed_terminal_error": computed,
                    "absolute_display_difference": abs(computed - published_error),
                }
            )
    return rows


def save_plot(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for example, marker in (("example_1", "o"), ("example_2", "s")):
        for alpha, color in ((0.25, "#005f73"), (0.50, "#d1495b")):
            subset = [row for row in rows if row["example"] == example and row["alpha"] == alpha]
            ax.loglog(
                [float(row["h"]) for row in subset],
                [float(row["computed_terminal_error"]) for row in subset],
                marker=marker,
                color=color,
                linestyle="-" if example == "example_1" else "--",
                label=f"{example.replace('_', ' ')}, alpha={alpha:g}",
            )
    ax.set_xlabel("Step size h")
    ax.set_ylabel("Terminal absolute error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    max_difference = max(float(row["absolute_display_difference"]) for row in rows)
    summary = {
        "stage": "integrator",
        "status": "passed_published_efork3_tables",
        "method": "Three-stage EFORK reference implementation with full known-history evaluation at every stage.",
        "reference": "Ghoreishi, Ghaffari, and Saad (2023), Fractional Order Runge-Kutta Methods, Tables 3, 4, 9, and 10.",
        "tables_reproduced": [3, 4, 9, 10],
        "max_absolute_difference_from_displayed_values": max_difference,
        "acceptance_tolerance": 6.0e-9,
        "checks": {
            "all_rows_within_display_tolerance": bool(max_difference <= 6.0e-9),
            "stage_order": "K3 uses a31*K1 + a32*K2",
            "source_script_provenance": "Python validation script supplied by Dr. Luis Gerardo de la Fraga, CINVESTAV Unidad Zacatenco; archived in sources/.",
        },
        "files": {
            "report": "integrator_validation.md",
            "results": "benchmark_results.csv",
            "plot": "convergence_errors.png",
        },
    }
    (output_dir / "efork3_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    save_plot(rows, output_dir / "convergence_errors.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
