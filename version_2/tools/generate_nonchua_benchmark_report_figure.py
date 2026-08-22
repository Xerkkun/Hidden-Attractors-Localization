"""Generate the report figure for the executed non-Chua Lyapunov benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DK_CSV = ROOT / (
    "validation/chaos_validation/lyapunov_methods/F4_internal_validation/"
    "F4_3_fractional_published_dk2018/dk2018_published_results.csv"
)
FISCHER_CSV = ROOT / (
    "validation/chaos_validation/lyapunov_methods/"
    "fractional_cloned_dynamics_abm_gs_published/discrepancy_diagnostics/"
    "fischer2020_row_classification.csv"
)
OUT_DIR = ROOT / (
    "docs/master_report_geometric_topological/figures/nonchua_benchmarks"
)

COLORS = {
    "quantitative": "#2474A6",
    "sign": "#D49A28",
    "discrepancy": "#B84A4A",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification(status: str) -> str:
    if "failed" in status:
        return "discrepancy"
    if "quantitative" in status:
        return "quantitative"
    if "sign_pattern" in status:
        return "sign"
    return "discrepancy"


def _load() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with DK_CSV.open(encoding="utf-8", newline="") as stream:
        dk_rows = list(csv.DictReader(stream))
    with FISCHER_CSV.open(encoding="utf-8", newline="") as stream:
        fischer_rows = list(csv.DictReader(stream))
    if len(dk_rows) != 2 or len(fischer_rows) != 24:
        raise RuntimeError(
            f"Unexpected benchmark inventory: DK={len(dk_rows)}, "
            f"Fischer={len(fischer_rows)}"
        )
    return dk_rows, fischer_rows


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def main() -> None:
    dk_rows, fischer_rows = _load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.4,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.13, wspace=0.23, hspace=0.42)

    axis = axes[0, 0]
    x = np.arange(2)
    width = 0.23
    component_colors = ["#315B7D", "#648FAB", "#9BB8C9"]
    for component in range(3):
        errors = [json.loads(row["absolute_differences"])[component] for row in dk_rows]
        axis.bar(
            x + (component - 1) * width,
            errors,
            width,
            color=component_colors[component],
            label=rf"$|\Delta\lambda_{component + 1}|$",
        )
    axis.axhline(0.05, color="#555555", linestyle="--", linewidth=1.0)
    axis.text(1.46, 0.053, "tolerancia 0.05", ha="right", va="bottom", fontsize=8)
    axis.set_yscale("log")
    axis.set_ylim(2e-5, 0.5)
    axis.set_xticks(x, ["Lorenz\n$q=0.985$", "RF\n$q=0.999$"])
    axis.set_ylabel("diferencia absoluta")
    axis.set_title("(a) Método variacional de Danca--Kuznetsov")
    axis.legend(frameon=False, ncols=3, fontsize=8, loc="upper left")
    _style_axis(axis)

    systems = [
        ("jerk", "(b) Jerk exponencial"),
        ("financial", "(c) Sistema financiero"),
        ("four_wing", "(d) Sistema four-wing"),
    ]
    order_labels = ["C1", "C.9", "C.8", "C.7", "I.9", "I.8", "I.7", "I.6"]
    for axis, (system, title) in zip(axes.flat[1:], systems, strict=True):
        rows = [row for row in fischer_rows if row["system"] == system]
        rows.sort(key=lambda row: (0 if row["type"] == "Comm" else 1, int(row["row_index"])))
        values = [float(row["max_abs_error"]) for row in rows]
        colors = [COLORS[_classification(row["status"])] for row in rows]
        bars = axis.bar(np.arange(8), values, color=colors, width=0.72)
        axis.axhline(0.05, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_xticks(np.arange(8), order_labels)
        axis.set_xlabel("C: conmensurable; I: inconmensurable")
        axis.set_ylabel(r"máximo $|\Delta\lambda_i|$")
        axis.set_title(title)
        upper = max(values) * 1.20
        axis.set_ylim(0, max(0.08, upper))
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(upper * 0.012, 0.002),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.8,
            )
        _style_axis(axis)

    legend = [
        Patch(facecolor=COLORS["quantitative"], label="coincidencia cuantitativa"),
        Patch(facecolor=COLORS["sign"], label="coincidencia de signos"),
        Patch(facecolor=COLORS["discrepancy"], label="discrepancia"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncols=3,
        frameon=False,
        fontsize=8.8,
    )

    pdf_path = OUT_DIR / "lyapunov_validation_overview.pdf"
    png_path = OUT_DIR / "lyapunov_validation_overview.png"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Creator": "HAFO validation"},
    )
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    summary = {
        "schema_version": "1.0",
        "source_files": {
            str(DK_CSV.relative_to(ROOT)).replace("\\", "/"): _sha256(DK_CSV),
            str(FISCHER_CSV.relative_to(ROOT)).replace("\\", "/"): _sha256(FISCHER_CSV),
        },
        "row_counts": {"dk2018": len(dk_rows), "fischer2020": len(fischer_rows)},
        "fischer_classification": {
            system: {
                kind: sum(
                    1
                    for row in fischer_rows
                    if row["system"] == system and _classification(row["status"]) == kind
                )
                for kind in ("quantitative", "sign", "discrepancy")
            }
            for system, _ in systems
        },
        "outputs": {
            pdf_path.name: _sha256(pdf_path),
            png_path.name: _sha256(png_path),
        },
    }
    (OUT_DIR / "lyapunov_validation_overview_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
