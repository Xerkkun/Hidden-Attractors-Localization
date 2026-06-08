from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION2 = ROOT / "version_2"
if str(VERSION2) not in sys.path:
    sys.path.insert(0, str(VERSION2))

from hidden_attractors.analysis.lyapunov_api import compute_lyapunov_spectrum
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import (
    ChuaParameters,
    equilibria_nonsmooth,
    jacobian_nonsmooth,
    rhs_nonsmooth,
)

ALPHA = 8.4562
BETA = 12.0732
GAMMA = 0.0052
M0 = -0.2
M1 = -1.2
Q = 0.9998
H = 0.01
CASE_ID = "m1_m1p2000_m0_m0p2000_branch_0"
DEFAULT_X0 = np.array(
    [-1.5774788811484806, -0.28498901376346114, 4.364931371054484],
    dtype=float,
)

VAL_DIR = (
    ROOT
    / "version_2"
    / "validation"
    / "outputs"
    / "candidate_chaos_hiddenness"
    / "danca2017_chua_fractional_saturation_candidate"
)
FIG_DIR = ROOT / "DF y NC Chua entero y fraccionario copy" / "Figs"
REFERENCE_TRAJECTORY = (
    ROOT
    / "outputs"
    / "saturation_search_seed0p9998_mem_full_sweep"
    / f"{CASE_ID}_trajectory.csv"
)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
    }
)


@dataclass(frozen=True)
class Profile:
    name: str
    lyapunov_t_final: float
    lyapunov_t_burn: float
    hiddenness_t_final: float
    hiddenness_t_transient: float
    radii: tuple[float, ...]
    samples_base: int
    samples_growth: int


_WORKER_PROFILE: Profile | None = None
_WORKER_EQUILIBRIA: dict[str, np.ndarray] | None = None
_WORKER_REFERENCE_TAIL: np.ndarray | None = None


PROFILES = {
    "smoke": Profile(
        name="smoke",
        lyapunov_t_final=20.0,
        lyapunov_t_burn=5.0,
        hiddenness_t_final=20.0,
        hiddenness_t_transient=5.0,
        radii=(1.0e-5, 1.0e-3),
        samples_base=4,
        samples_growth=0,
    ),
    "report": Profile(
        name="report",
        lyapunov_t_final=160.0,
        lyapunov_t_burn=40.0,
        hiddenness_t_final=300.0,
        hiddenness_t_transient=100.0,
        radii=(1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 1.0e-2),
        samples_base=100,
        samples_growth=50,
    ),
}


def _init_hiddenness_worker(
    profile: Profile,
    equilibria_items: list[tuple[str, list[float]]],
    reference_tail: np.ndarray,
) -> None:
    global _WORKER_PROFILE, _WORKER_EQUILIBRIA, _WORKER_REFERENCE_TAIL
    _WORKER_PROFILE = profile
    _WORKER_EQUILIBRIA = {
        name: np.asarray(point, dtype=float) for name, point in equilibria_items
    }
    _WORKER_REFERENCE_TAIL = np.asarray(reference_tail, dtype=float)


def _hiddenness_worker(task: tuple[str, float, int, list[float]]) -> dict[str, object]:
    if _WORKER_PROFILE is None or _WORKER_EQUILIBRIA is None or _WORKER_REFERENCE_TAIL is None:
        raise RuntimeError("hiddenness worker was not initialized")
    eq_name, radius, sample_index, x0_list = task
    x0 = np.asarray(x0_list, dtype=float)
    times, states, status, info = fractional_integrate(
        _rhs_time,
        x0,
        q=Q,
        h=H,
        t_final=_WORKER_PROFILE.hiddenness_t_final,
        method="abm",
        memory_mode="full",
        use_c_backend=True,
        allow_python_fallback=True,
        divergence_norm=120.0,
        equilibria=list(_WORKER_EQUILIBRIA.values()),
    )
    label, metrics = _classify_tail(
        np.asarray(states, dtype=float),
        status,
        _WORKER_EQUILIBRIA,
        _WORKER_REFERENCE_TAIL,
        _WORKER_PROFILE.hiddenness_t_transient,
    )
    row = {
        "profile": _WORKER_PROFILE.name,
        "equilibrium": eq_name,
        "radius": radius,
        "sample_index": sample_index,
        "x0": float(x0[0]),
        "y0": float(x0[1]),
        "z0": float(x0[2]),
        "label": label,
        "integration_status": status,
        "backend": info.get("backend", ""),
        "final_x": float(states[-1, 0]) if len(states) else math.nan,
        "final_y": float(states[-1, 1]) if len(states) else math.nan,
        "final_z": float(states[-1, 2]) if len(states) else math.nan,
    }
    row.update(metrics)
    return row


def _params() -> ChuaParameters:
    return ChuaParameters(
        model="nonsmooth",
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        m0=M0,
        m1=M1,
    )


def _rhs(state: np.ndarray) -> np.ndarray:
    return rhs_nonsmooth(np.asarray(state, dtype=float), _params())


def _jacobian(state: np.ndarray) -> np.ndarray:
    return jacobian_nonsmooth(np.asarray(state, dtype=float), _params())


def _rhs_time(_t: float, state: np.ndarray) -> np.ndarray:
    return _rhs(state)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _figure_name(name: str, profile: str) -> str:
    if profile == "report":
        return name
    return name.replace("chua_frac_ns_", f"chua_frac_ns_{profile}_", 1)


def _savefig(name: str, profile: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    profiled = _figure_name(name, profile)
    for ext in ("png", "pdf"):
        plt.savefig(
            FIG_DIR / f"{profiled}.{ext}",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )


def _load_reference_tail(t_transient: float) -> np.ndarray:
    data = np.loadtxt(REFERENCE_TRAJECTORY, delimiter=",", skiprows=1)
    return data[int(round(t_transient / H)) :, 1:]


def _uniform_ball(center: np.ndarray, radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(count, center.size))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scales = rng.random(count) ** (1.0 / center.size)
    return center + radius * directions * scales[:, None]


def _classify_tail(
    states: np.ndarray,
    status: str,
    equilibria: dict[str, np.ndarray],
    reference_tail: np.ndarray,
    t_transient: float,
) -> tuple[str, dict[str, float | str]]:
    if status not in {"ok", "converged_equilibrium_early"}:
        return "diverged", {"integration_status": status}
    if len(states) == 0 or not np.all(np.isfinite(states[-1])):
        return "diverged", {"integration_status": "nonfinite"}

    final_state = states[-1]
    distances = {
        name: float(np.linalg.norm(final_state - point))
        for name, point in equilibria.items()
    }
    closest_name = min(distances, key=distances.get)
    closest_distance = distances[closest_name]
    if closest_distance <= 5.0e-2:
        return "equilibrium", {
            "closest_equilibrium": closest_name,
            "closest_equilibrium_distance": closest_distance,
        }

    tail_start = int(round(t_transient / H))
    tail = states[tail_start:] if len(states) > tail_start else states
    if len(tail) < 10:
        return "no_target_match", {"reason": "short_tail"}

    ref_range = np.ptp(reference_tail, axis=0)
    ref_scale = float(max(np.linalg.norm(ref_range), 1.0e-9))
    tail_range = np.ptp(tail, axis=0)
    range_relative_distance = float(np.linalg.norm(tail_range - ref_range) / ref_scale)
    median_distance_norm = float(
        np.linalg.norm(np.median(tail, axis=0) - np.median(reference_tail, axis=0)) / ref_scale
    )
    collapsed_ratio = float(np.linalg.norm(tail_range) / ref_scale)
    metrics = {
        "closest_equilibrium": closest_name,
        "closest_equilibrium_distance": closest_distance,
        "range_relative_distance": range_relative_distance,
        "median_distance_norm": median_distance_norm,
        "collapsed_ratio": collapsed_ratio,
    }
    if (
        median_distance_norm <= 0.35
        and range_relative_distance <= 0.60
        and collapsed_ratio >= 0.15
    ):
        return "target_candidate", metrics
    return "no_target_match", metrics


def run_lyapunov(profile: Profile, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    method_specs = [
        {
            "method": "fractional_variational_abm_qr",
            "memory_mode": "full",
            "extra": {
                "history_aware_qr": True,
            },
        },
    ]

    for spec in method_specs:
        summary = compute_lyapunov_spectrum(
            rhs=_rhs,
            jacobian=_jacobian if spec["method"] == "fractional_variational_abm_qr" else None,
            x0=DEFAULT_X0,
            q=Q,
            method=spec["method"],
            h=H,
            t_final=profile.lyapunov_t_final,
            t_burn=profile.lyapunov_t_burn,
            reorthonormalization_time=0.1,
            memory_mode=spec["memory_mode"],
            div_threshold=120.0,
            **spec["extra"],
        )
        result = summary.result
        conv = np.asarray(result.convergence, dtype=float)
        times = np.asarray(result.times, dtype=float)
        for idx, exponent in enumerate(np.asarray(result.exponents, dtype=float)):
            rows.append(
                {
                    "profile": profile.name,
                    "method": result.method_id,
                    "exponent_index": idx + 1,
                    "exponent": float(exponent),
                    "status": result.status,
                }
            )
        for time_value, values in zip(times, conv):
            row = {
                "profile": profile.name,
                "method": result.method_id,
                "time": float(time_value),
            }
            for idx, value in enumerate(values):
                row[f"lambda_{idx + 1}"] = float(value)
            rows.append(row)

        summaries.append(
            {
                "method": result.method_id,
                "status": result.status,
                "exponents": np.asarray(result.exponents, dtype=float).tolist(),
                "max_exponent": float(np.nanmax(result.exponents)),
                "warnings": list(summary.warnings) + list(result.methodological_warnings),
                "request": summary.request_summary,
            }
        )

    _write_csv(output_dir / "lyapunov_results.csv", rows)
    with (output_dir / "lyapunov_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready({"profile": profile.name, "methods": summaries}), handle, indent=2)
    _plot_lyapunov_convergence(output_dir / "lyapunov_results.csv", profile.name)
    return {"profile": profile.name, "methods": summaries}


def _plot_lyapunov_convergence(path: Path, profile: str) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    method_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if "time" in row and row.get("time"):
            method_rows.setdefault(row["method"], []).append(row)
    fig, axes = plt.subplots(len(method_rows), 1, figsize=(7.0, 3.2 * max(len(method_rows), 1)), sharex=False)
    if len(method_rows) == 1:
        axes = [axes]
    for ax, (method, items) in zip(axes, method_rows.items()):
        times = np.array([float(row["time"]) for row in items], dtype=float)
        for idx in range(1, 4):
            key = f"lambda_{idx}"
            values = [float(row[key]) for row in items if row.get(key)]
            if len(values) == len(times):
                ax.plot(times, values, lw=0.9, label=rf"$\lambda_{idx}$")
        ax.axhline(0.0, color="red", lw=0.8, ls=":")
        ax.set_ylabel(method.replace("_", r"\_"))
        ax.grid(True, ls=":", lw=0.55)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    _savefig("chua_frac_ns_fig12_lyapunov_full_history", profile)
    plt.close(fig)


def run_hiddenness(profile: Profile, output_dir: Path, seed: int, workers: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    equilibria = equilibria_nonsmooth(_params())
    reference_tail = _load_reference_tail(profile.hiddenness_t_transient)
    rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    tasks: list[tuple[str, float, int, list[float]]] = []

    for eq_name, center in equilibria.items():
        for radius_index, radius in enumerate(profile.radii):
            count = profile.samples_base + radius_index * profile.samples_growth
            samples = _uniform_ball(center, radius, count, rng)
            plan_rows.append(
                {
                    "profile": profile.name,
                    "equilibrium": eq_name,
                    "radius": radius,
                    "sample_count": count,
                }
            )
            for sample_index, x0 in enumerate(samples):
                tasks.append((eq_name, radius, sample_index, x0.tolist()))

    workers = max(1, int(workers))
    if workers == 1:
        _init_hiddenness_worker(
            profile,
            [(name, point.tolist()) for name, point in equilibria.items()],
            reference_tail,
        )
        rows = [_hiddenness_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_hiddenness_worker,
            initargs=(
                profile,
                [(name, point.tolist()) for name, point in equilibria.items()],
                reference_tail,
            ),
        ) as executor:
            future_to_index = {
                executor.submit(_hiddenness_worker, task): index
                for index, task in enumerate(tasks)
            }
            ordered_rows: list[dict[str, object] | None] = [None] * len(tasks)
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                ordered_rows[index] = future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    print(
                        json.dumps(
                            {
                                "stage": "hiddenness",
                                "profile": profile.name,
                                "completed": completed,
                                "total": len(tasks),
                                "workers": workers,
                            }
                        ),
                        flush=True,
                    )
            rows = [row for row in ordered_rows if row is not None]

    _write_csv(output_dir / "ball_sampling_plan.csv", plan_rows)
    _write_csv(output_dir / "ball_sampling_results.csv", rows)
    decisions = _hiddenness_decisions(rows)
    _write_csv(output_dir / "hiddenness_decisions.csv", decisions)
    with (output_dir / "hiddenness_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(
                {
                    "profile": profile.name,
                    "q": Q,
                    "h": H,
                    "t_final": profile.hiddenness_t_final,
                    "t_transient": profile.hiddenness_t_transient,
                    "equilibria": {name: point.tolist() for name, point in equilibria.items()},
                    "decisions": decisions,
                }
            ),
            handle,
            indent=2,
        )
    _plot_hiddenness_spheres(rows, equilibria, profile.name)
    _plot_hiddenness_spheres_by_equilibrium(rows, equilibria, profile.name)
    _plot_hiddenness_heatmap(rows, profile.name)
    return {"profile": profile.name, "decisions": decisions}


def _hiddenness_decisions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    groups = sorted({(row["equilibrium"], row["radius"]) for row in rows})
    for eq_name, radius in groups:
        subset = [row for row in rows if row["equilibrium"] == eq_name and row["radius"] == radius]
        target_hits = sum(1 for row in subset if row["label"] == "target_candidate")
        out.append(
            {
                "equilibrium": eq_name,
                "radius": radius,
                "samples": len(subset),
                "target_hits": target_hits,
                "decision": "self_excited_contact_detected" if target_hits else "no_contact_detected",
            }
        )
    return out


def _plot_hiddenness_spheres(
    rows: list[dict[str, object]],
    equilibria: dict[str, np.ndarray],
    profile: str,
) -> None:
    colors = {
        "target_candidate": "#dc2626",
        "equilibrium": "#2563eb",
        "no_target_match": "#64748b",
        "diverged": "#111827",
    }
    fig = plt.figure(figsize=(7.0, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    for label, color in colors.items():
        subset = [row for row in rows if row["label"] == label]
        if not subset:
            continue
        ax.scatter(
            [row["x0"] for row in subset],
            [row["y0"] for row in subset],
            [row["z0"] for row in subset],
            s=14,
            color=color,
            alpha=0.82,
            label=label.replace("_", " "),
        )
    for eq_name, point in equilibria.items():
        ax.scatter([point[0]], [point[1]], [point[2]], s=42, color="gold", edgecolor="black")
        ax.text(point[0], point[1], point[2], eq_name, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="best")
    _savefig("chua_frac_ns_fig13_hiddenness_spherical_probes", profile)
    plt.close(fig)


def _plot_hiddenness_heatmap(rows: list[dict[str, object]], profile: str) -> None:
    equilibria = sorted({str(row["equilibrium"]) for row in rows})
    radii = sorted({float(row["radius"]) for row in rows})
    data = np.zeros((len(equilibria), len(radii)), dtype=float)
    for i, eq_name in enumerate(equilibria):
        for j, radius in enumerate(radii):
            subset = [
                row for row in rows
                if row["equilibrium"] == eq_name and float(row["radius"]) == radius
            ]
            data[i, j] = sum(1 for row in subset if row["label"] == "target_candidate")
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    im = ax.imshow(data, cmap="Reds", aspect="auto")
    ax.set_yticks(np.arange(len(equilibria)), labels=equilibria)
    ax.set_xticks(np.arange(len(radii)), labels=[f"{value:.0e}" for value in radii], rotation=35)
    ax.set_xlabel("radio")
    ax.set_ylabel("equilibrio")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="contactos con candidato")
    fig.tight_layout()
    _savefig("chua_frac_ns_fig14_hiddenness_contact_heatmap", profile)
    plt.close(fig)


def _plot_hiddenness_spheres_by_equilibrium(
    rows: list[dict[str, object]],
    equilibria: dict[str, np.ndarray],
    profile: str,
) -> None:
    colors = {
        "target_candidate": "#dc2626",
        "equilibrium": "#2563eb",
        "no_target_match": "#64748b",
        "diverged": "#111827",
    }
    suffixes = {"E0": "E0", "E+": "Ep", "E-": "Em"}
    radii = sorted({float(row["radius"]) for row in rows})
    u = np.linspace(0.0, 2.0 * math.pi, 28)
    v = np.linspace(0.0, math.pi, 14)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    for eq_name, center in equilibria.items():
        fig = plt.figure(figsize=(9.0, 6.0), facecolor="white")
        axes = [fig.add_subplot(2, 3, index + 1, projection="3d") for index in range(6)]
        for ax, radius in zip(axes, radii):
            subset = [
                row for row in rows
                if str(row["equilibrium"]) == eq_name and float(row["radius"]) == radius
            ]
            ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="0.35", linewidth=0.28, alpha=0.28)
            ax.scatter([0.0], [0.0], [0.0], s=28, marker="s", color="#facc15", edgecolor="black", linewidth=0.4)
            for label, color in colors.items():
                label_rows = [row for row in subset if row["label"] == label]
                if not label_rows:
                    continue
                xs = [(float(row["x0"]) - center[0]) / radius for row in label_rows]
                ys = [(float(row["y0"]) - center[1]) / radius for row in label_rows]
                zs = [(float(row["z0"]) - center[2]) / radius for row in label_rows]
                ax.scatter(xs, ys, zs, s=10, color=color, alpha=0.82, label=label.replace("_", " "))
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.08, 1.08)
            ax.set_zlim(-1.08, 1.08)
            ax.set_xlabel(r"$(x-x^*)/r$", labelpad=-2)
            ax.set_ylabel(r"$(y-y^*)/r$", labelpad=-2)
            ax.set_zlabel(r"$(z-z^*)/r$", labelpad=-2)
            ax.tick_params(labelsize=6, pad=0)
            ax.grid(True, ls=":", lw=0.45, alpha=0.7)
            ax.text2D(0.03, 0.92, f"r={radius:.0e}", transform=ax.transAxes, fontsize=8)
            ax.view_init(elev=22, azim=-55)

        handles: list[object] = []
        labels: list[str] = []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        if unique:
            fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=7)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        _savefig(f"chua_frac_ns_fig13_{suffixes[eq_name]}_hiddenness_spherical_3d", profile)
        plt.close(fig)


def _coerce_csv_value(value: str) -> object:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def render_hiddenness_figures_from_outputs(profile: Profile, output_dir: Path) -> dict[str, object]:
    rows_path = output_dir / "ball_sampling_results.csv"
    summary_path = output_dir / "hiddenness_run_summary.json"
    rows = [
        {key: _coerce_csv_value(value) for key, value in row.items()}
        for row in csv.DictReader(rows_path.open(newline="", encoding="utf-8"))
    ]
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        equilibria = {name: np.asarray(point, dtype=float) for name, point in data["equilibria"].items()}
    else:
        equilibria = equilibria_nonsmooth(_params())
    _plot_hiddenness_spheres(rows, equilibria, profile.name)
    _plot_hiddenness_spheres_by_equilibrium(rows, equilibria, profile.name)
    _plot_hiddenness_heatmap(rows, profile.name)
    return {
        "profile": profile.name,
        "source": str(rows_path),
        "figures": "rendered_from_existing_outputs",
    }


def render_all_figures_from_outputs(profile: Profile, output_dir: Path) -> dict[str, object]:
    rendered: dict[str, object] = {}
    lyapunov_path = output_dir / "lyapunov_results.csv"
    if lyapunov_path.exists():
        _plot_lyapunov_convergence(lyapunov_path, profile.name)
        rendered["lyapunov"] = str(lyapunov_path)
    hiddenness_path = output_dir / "ball_sampling_results.csv"
    if hiddenness_path.exists():
        rendered["hiddenness"] = render_hiddenness_figures_from_outputs(profile, output_dir)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--only", choices=("all", "lyapunov", "hiddenness", "figures"), default="all")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--output-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    out = args.output_dir / profile.name
    summary: dict[str, object] = {
        "case_id": CASE_ID,
        "profile": profile.name,
        "parameters": {
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "m0": M0,
            "m1": M1,
            "q": Q,
            "h": H,
        },
    }
    if args.only == "figures":
        summary["figures"] = render_all_figures_from_outputs(profile, out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "figure_render_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_ready(summary), handle, indent=2)
        print(json.dumps(_json_ready(summary), indent=2))
        return
    if args.only in {"all", "lyapunov"}:
        summary["lyapunov"] = run_lyapunov(profile, out)
    if args.only in {"all", "hiddenness"}:
        summary["hiddenness"] = run_hiddenness(profile, out, args.seed, args.workers)
    if args.only != "all":
        existing_path = out / "candidate_chaos_hiddenness_summary.json"
        if existing_path.exists():
            with existing_path.open(encoding="utf-8") as handle:
                existing_summary = json.load(handle)
            existing_summary.update(summary)
            summary = existing_summary
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate_chaos_hiddenness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)
    print(json.dumps(_json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
