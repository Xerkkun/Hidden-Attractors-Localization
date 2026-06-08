"""Plot summaries from ``validation_outputs`` CSV/JSON artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_OUTPUTS = ROOT / "validation_outputs"
FIGURES = VALIDATION_OUTPUTS / "figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?<=\d)\.e([+-]?\d+)", r".0e\1", text)
    text = re.sub(r"(?<=\d)\.E([+-]?\d+)", r".0E\1", text)
    return json.loads(text)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip().strip('"')
    if not text:
        return None
    constants = {
        "Pi": math.pi,
        "-Pi": -math.pi,
        "pi": math.pi,
        "-pi": -math.pi,
    }
    if text in constants:
        return constants[text]
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _system_dirs() -> list[Path]:
    return sorted(path for path in VALIDATION_OUTPUTS.iterdir() if path.is_dir() and path.name != "figures")


def _case_label(system_id: str) -> str:
    return system_id.replace("chua_", "").replace("_", "\n")


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


def plot_validation_test_matrix() -> None:
    systems: list[str] = []
    test_names: list[str] = []
    summaries: dict[str, dict[str, bool]] = {}

    for directory in _system_dirs():
        files = list(directory.glob("*_validation_summary.json"))
        if not files:
            continue
        summary = _read_json(files[0])
        system_id = str(summary.get("system_id", directory.name))
        systems.append(system_id)
        rows: dict[str, bool] = {}
        for test in summary.get("tests", []):
            name = str(test.get("name", "unknown"))
            rows[name] = bool(test.get("passed", False))
            if name not in test_names:
                test_names.append(name)
        summaries[system_id] = rows

    if not systems or not test_names:
        return

    matrix = np.full((len(systems), len(test_names)), np.nan)
    for i, system_id in enumerate(systems):
        for j, name in enumerate(test_names):
            if name in summaries[system_id]:
                matrix[i, j] = 1.0 if summaries[system_id][name] else 0.0

    fig, ax = plt.subplots(figsize=(max(7, len(test_names) * 1.1), 4.8))
    cmap = matplotlib.colors.ListedColormap(["#c2410c", "#15803d"])
    masked = np.ma.masked_invalid(matrix)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(systems)), [_case_label(s) for s in systems])
    ax.set_xticks(range(len(test_names)), test_names, rotation=35, ha="right")
    ax.set_title("Validation output test status")
    for i in range(len(systems)):
        for j in range(len(test_names)):
            if math.isnan(matrix[i, j]):
                ax.text(j, i, "NA", ha="center", va="center", color="#334155", fontsize=8)
            else:
                ax.text(j, i, "PASS" if matrix[i, j] else "FAIL", ha="center", va="center", color="white", fontsize=8)
    _save(fig, "validation_test_status_matrix")


def plot_numeric_coverage() -> None:
    rows: list[tuple[str, str, int, int]] = []
    for directory in _system_dirs():
        for path in sorted(directory.glob("*.csv")):
            records = _read_csv(path)
            total = 0
            numeric = 0
            for record in records:
                for value in record.values():
                    total += 1
                    if _as_float(value) is not None:
                        numeric += 1
            rows.append((directory.name, path.name.replace(f"{directory.name}_", "").replace(".csv", ""), numeric, total))

    if not rows:
        return

    labels = [f"{_case_label(system)}\n{file_label}" for system, file_label, _n, _t in rows]
    fractions = [numeric / total if total else 0.0 for _system, _file, numeric, total in rows]

    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.75), 5.5))
    colors = ["#2563eb" if frac >= 0.8 else "#f59e0b" if frac > 0 else "#b91c1c" for frac in fractions]
    ax.bar(range(len(rows)), fractions, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Numeric cell fraction")
    ax.set_title("Numeric coverage by validation CSV")
    ax.set_xticks(range(len(rows)), labels, rotation=70, ha="right", fontsize=8)
    ax.grid(axis="y", color="#cbd5e1", linestyle=":", linewidth=0.7)
    _save(fig, "numeric_coverage_by_csv")


def plot_seed_parameters() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    plotted = False
    width = 0.22
    metric_names = ["omega0", "a0", "k"]
    offsets = [-width, 0.0, width]
    labels: list[str] = []
    x_base: list[float] = []
    values_by_metric = {metric: [] for metric in metric_names}

    for directory in _system_dirs():
        path = directory / f"{directory.name}_seed_summary.csv"
        if not path.exists():
            continue
        for record in _read_csv(path):
            if str(record.get("status", "")).lower() != "ok":
                continue
            labels.append(f"{_case_label(directory.name)}\nbranch {record.get('branch', '')}")
            x_base.append(float(len(x_base)))
            for metric in metric_names:
                values_by_metric[metric].append(_as_float(record.get(metric)) or 0.0)

    if not labels:
        ax.text(0.5, 0.5, "No numeric seed rows with status=ok", ha="center", va="center", transform=ax.transAxes)
    else:
        plotted = True
        for metric, offset in zip(metric_names, offsets):
            ax.bar([x + offset for x in x_base], values_by_metric[metric], width=width, label=metric)
        ax.set_xticks(x_base, labels)
        ax.set_ylabel("Value")
        ax.set_title("Describing-function seed parameters")
        ax.legend()
        ax.grid(axis="y", color="#cbd5e1", linestyle=":", linewidth=0.7)
    _save(fig, "seed_parameters")

    if plotted:
        plot_seed_vectors()


def plot_seed_vectors() -> None:
    data: list[tuple[str, list[float], list[float]]] = []
    for directory in _system_dirs():
        path = directory / f"{directory.name}_seed_data.json"
        if not path.exists():
            continue
        records = _read_json(path)
        if not isinstance(records, list):
            continue
        for record in records:
            plus = record.get("seed_plus")
            minus = record.get("seed_minus")
            if isinstance(plus, list) and isinstance(minus, list):
                plus_vals = [_as_float(v) for v in plus]
                minus_vals = [_as_float(v) for v in minus]
                if all(v is not None for v in plus_vals + minus_vals):
                    data.append((f"{_case_label(directory.name)}\nbranch {record.get('branch', '')}", plus_vals, minus_vals))  # type: ignore[arg-type]

    if not data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharex=True)
    components = ["x", "y", "z"]
    x = np.arange(len(data))
    for comp_idx, ax in enumerate(axes):
        ax.bar(x - 0.18, [row[1][comp_idx] for row in data], width=0.36, label="seed_plus", color="#2563eb")
        ax.bar(x + 0.18, [row[2][comp_idx] for row in data], width=0.36, label="seed_minus", color="#dc2626")
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.set_title(f"Seed {components[comp_idx]}")
        ax.grid(axis="y", color="#cbd5e1", linestyle=":", linewidth=0.7)
    axes[0].set_xticks(x, [row[0] for row in data], rotation=25, ha="right")
    axes[0].legend()
    _save(fig, "seed_vectors")


def plot_equilibria_and_residuals() -> None:
    eq_rows: list[tuple[str, str, float, float, float, float]] = []
    for directory in _system_dirs():
        path = directory / f"{directory.name}_equilibria_residuals.csv"
        if not path.exists():
            continue
        for record in _read_csv(path):
            x = _as_float(record.get("x"))
            y = _as_float(record.get("y"))
            z = _as_float(record.get("z"))
            residual = _as_float(record.get("rhs_residual_norm"))
            if x is None or y is None or z is None or residual is None:
                continue
            eq_rows.append((directory.name, record.get("equilibrium", ""), x, y, z, residual))

    if not eq_rows:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for system, eq, x, y, z, _res in eq_rows:
        ax.scatter([x], [y], [z], s=70, label=f"{_case_label(system)} {eq}")
        ax.text(x, y, z, eq, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Numeric equilibria from validation_outputs")
    ax.legend(fontsize=7, loc="best")
    _save(fig, "numeric_equilibria_3d")

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
    labels = [f"{_case_label(system)}\n{eq}" for system, eq, *_rest in eq_rows]
    residuals = [max(res, 1e-30) for *_coords, res in eq_rows]
    ax2.bar(range(len(eq_rows)), residuals, color="#0f766e")
    ax2.set_yscale("log")
    ax2.set_ylabel("RHS residual norm")
    ax2.set_title("Equilibrium residuals")
    ax2.set_xticks(range(len(eq_rows)), labels, rotation=45, ha="right")
    ax2.grid(axis="y", color="#cbd5e1", linestyle=":", linewidth=0.7)
    _save(fig2, "equilibrium_residuals")


def plot_eigenvalues_matignon() -> None:
    rows: list[tuple[str, str, float, float, float | None]] = []
    q_values: list[float] = []
    for directory in _system_dirs():
        path = directory / f"{directory.name}_eigenvalues_matignon.csv"
        if not path.exists():
            continue
        for record in _read_csv(path):
            real = _as_float(record.get("real"))
            imag = _as_float(record.get("imag"))
            q = _as_float(record.get("q"))
            margin = _as_float(record.get("matignon_margin"))
            if real is None or imag is None:
                continue
            if q is not None:
                q_values.append(q)
            label = str(record.get("region") or record.get("equilibrium") or directory.name)
            rows.append((directory.name, label, real, imag, margin))

    if not rows:
        return

    q_ref = q_values[0] if q_values else 1.0
    fig, ax = plt.subplots(figsize=(7, 6))
    for system in sorted({row[0] for row in rows}):
        subset = [row for row in rows if row[0] == system]
        ax.scatter([row[2] for row in subset], [row[3] for row in subset], s=55, label=_case_label(system))
    radius = max(1.0, max(math.hypot(row[2], row[3]) for row in rows) * 1.15)
    angle = q_ref * math.pi / 2.0
    for sign in (1, -1):
        ax.plot([0, radius * math.cos(sign * angle)], [0, radius * math.sin(sign * angle)], color="#dc2626", linestyle="--", linewidth=1.1)
    ax.axhline(0, color="#475569", linewidth=0.8)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title(f"Numeric eigenvalues and Matignon boundary (q={q_ref:g})")
    ax.legend(fontsize=8)
    ax.grid(color="#cbd5e1", linestyle=":", linewidth=0.7)
    _save(fig, "eigenvalues_matignon_numeric")

    margin_rows = [row for row in rows if row[4] is not None]
    if margin_rows:
        fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
        labels = [f"{_case_label(row[0])}\n{row[1]} {idx + 1}" for idx, row in enumerate(margin_rows)]
        colors = ["#15803d" if (row[4] or 0.0) >= 0 else "#b91c1c" for row in margin_rows]
        ax2.bar(range(len(margin_rows)), [row[4] or 0.0 for row in margin_rows], color=colors)
        ax2.axhline(0, color="#334155", linewidth=0.8)
        ax2.set_ylabel("Matignon margin")
        ax2.set_title("Matignon margins")
        ax2.set_xticks(range(len(margin_rows)), labels, rotation=65, ha="right", fontsize=8)
        ax2.grid(axis="y", color="#cbd5e1", linestyle=":", linewidth=0.7)
        _save(fig2, "matignon_margins_numeric")


def plot_jacobian_heatmaps() -> None:
    for directory in _system_dirs():
        path = directory / f"{directory.name}_jacobians.csv"
        if not path.exists():
            continue
        records = _read_csv(path)
        numeric_records: list[tuple[str, np.ndarray]] = []
        for record in records:
            values = [_as_float(record.get(key)) for key in ("j11", "j12", "j13", "j21", "j22", "j23", "j31", "j32", "j33")]
            if any(value is None for value in values):
                continue
            label = str(record.get("region") or record.get("equilibrium") or "jacobian")
            numeric_records.append((label, np.asarray(values, dtype=float).reshape(3, 3)))  # type: ignore[arg-type]
        if not numeric_records:
            continue
        fig, axes = plt.subplots(1, len(numeric_records), figsize=(4.2 * len(numeric_records), 4.0), squeeze=False)
        max_abs = max(float(np.max(np.abs(matrix))) for _label, matrix in numeric_records)
        for ax, (label, matrix) in zip(axes[0], numeric_records):
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
            ax.set_title(label)
            ax.set_xticks(range(3), ["x", "y", "z"])
            ax.set_yticks(range(3), ["x", "y", "z"])
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{matrix[i, j]:.2g}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78)
        fig.suptitle(f"Jacobian matrices: {directory.name}")
        _save(fig, f"jacobians_{directory.name}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_validation_test_matrix()
    plot_numeric_coverage()
    plot_seed_parameters()
    plot_equilibria_and_residuals()
    plot_eigenvalues_matignon()
    plot_jacobian_heatmaps()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
