from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DOCS_DIR = Path(__file__).resolve().parents[1]
VERSION_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = DOCS_DIR / "assets" / "figures" / "chua_fractional_report"
sys.path.insert(0, str(VERSION_ROOT))

from hidden_attractors.seed_generation.chua_arctan_wu2023 import format_arctan_wu2023_seed_report  # noqa: E402
from hidden_attractors.seed_generation.lure import find_lure_harmonic_seed  # noqa: E402
from hidden_attractors.systems import get_system  # noqa: E402


CASES = [
    {
        "id": "nonsmooth",
        "system_name": "chua-nonsmooth",
        "q_seed_integer": 1.0,
        "q_seed_fractional": 0.998,
        "typical_ic": [0.1, 0.1, 0.1],
        "title": "Chua no suave",
        "color": "#245a9b",
    },
    {
        "id": "arctan",
        "system_name": "chua-arctan",
        "q_seed_integer": 1.0,
        "q_seed_fractional": 0.95,
        "typical_ic": [0.1, 0.1, 0.1],
        "title": "Chua arctan",
        "color": "#7a3b9f",
    },
]


def nonlinear_rhs(system):
    params = system.parameters

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(system.rhs(np.asarray(state, dtype=float), params), dtype=float)

    return rhs


def homotopy_rhs(lure, gain: float, eta: float):
    matrix = np.asarray(lure.matrix, dtype=float)
    bvec = np.asarray(lure.input_vector, dtype=float)
    rvec = np.asarray(lure.output_vector, dtype=float)
    gain_value = float(gain)
    eta_value = float(eta)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        sigma = float(rvec @ x)
        nonlinear_feedback = float(lure.nonlinearity(sigma))
        linear_feedback = gain_value * sigma
        feedback = (1.0 - eta_value) * linear_feedback + eta_value * nonlinear_feedback
        return matrix @ x + bvec * feedback

    return rhs


def integrate(rhs, x0: np.ndarray, *, t_final: float, dt: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, t_final + 0.5 * dt, dt)
    sol = solve_ivp(rhs, (0.0, t_final), np.asarray(x0, dtype=float), t_eval=t_eval, method="DOP853", rtol=1.0e-9, atol=1.0e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def plot_attractor(case: dict, system, seed) -> tuple[str, str, dict]:
    t, states = integrate(nonlinear_rhs(system), seed.seed, t_final=180.0)
    mask = t >= 60.0
    tail = states[mask]
    idx = np.linspace(0, len(tail) - 1, min(7000, len(tail))).astype(int)

    fig = plt.figure(figsize=(7.0, 5.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[idx, 0], tail[idx, 1], tail[idx, 2], lw=0.45, color=case["color"])
    ax.scatter([seed.seed[0]], [seed.seed[1]], [seed.seed[2]], s=28, color="#b73535", label="semilla DF")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{case['title']}: atractor 3D con parametros tipicos")
    ax.legend(loc="upper left", fontsize=7)
    ax.view_init(elev=23, azim=-57)
    fig.tight_layout()
    attractor_name = f"chua_{case['id']}_typical_attractor_3d.png"
    fig.savefig(OUT_DIR / attractor_name, bbox_inches="tight")
    plt.close(fig)

    series_mask = (t >= 60.0) & (t <= 120.0)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.0), dpi=180, sharex=True)
    for ax, col, label, color in zip(
        axes,
        range(3),
        ["x(t)", "y(t)", "z(t)"],
        ["#245a9b", "#7a3b9f", "#b15f18"],
    ):
        ax.plot(t[series_mask] - 60.0, states[series_mask, col], lw=0.7, color=color)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("tiempo post-transitorio")
    axes[0].set_title(f"{case['title']}: series temporales")
    fig.tight_layout()
    series_name = f"chua_{case['id']}_typical_time_series.png"
    fig.savefig(OUT_DIR / series_name, bbox_inches="tight")
    plt.close(fig)

    info = {
        "status": "ok",
        "final_state": states[-1].tolist(),
        "post_burn_bounds": {"min": tail.min(axis=0).tolist(), "max": tail.max(axis=0).tolist()},
    }
    return attractor_name, series_name, info


def plot_continuation(case: dict, system, seed) -> tuple[str, dict]:
    etas = np.linspace(0.0, 1.0, 6)
    current = np.asarray(seed.seed, dtype=float)
    traces = []
    for eta in etas:
        t_final = 70.0 if eta == 0.0 else 110.0
        burn = 15.0 if eta == 0.0 else 40.0
        t, states = integrate(homotopy_rhs(system.lure, seed.gain, float(eta)), current, t_final=t_final)
        traces.append((float(eta), states[t >= burn]))
        current = states[-1]

    fig = plt.figure(figsize=(7.2, 5.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    for i, (eta, trace) in enumerate(traces):
        idx = np.linspace(0, len(trace) - 1, min(2500, len(trace))).astype(int)
        ax.plot(trace[idx, 0], trace[idx, 1], trace[idx, 2], lw=0.65, color=cmap(i / (len(traces) - 1)), label=fr"$\eta={eta:.1f}$")
    ax.scatter([seed.seed[0]], [seed.seed[1]], [seed.seed[2]], s=34, color="#b73535", label="semilla inicial")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{case['title']}: continuacion DF lineal a no lineal")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    continuation_name = f"chua_{case['id']}_df_continuation_3d.png"
    fig.savefig(OUT_DIR / continuation_name, bbox_inches="tight")
    plt.close(fig)
    return continuation_name, {"etas": etas.tolist(), "final_state": current.tolist()}


def seed_payload(seed, q_value: float) -> dict:
    return {
        "q": float(q_value),
        "omega": float(seed.omega),
        "gain": float(seed.gain),
        "amplitude": float(seed.amplitude),
        "seed": seed.seed.tolist(),
        "matched_eigenvalue": [float(seed.matched_eigenvalue.real), float(seed.matched_eigenvalue.imag)],
        "method": seed.method,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    summary: dict[str, object] = {}
    for case in CASES:
        system = get_system(case["system_name"])
        seed_integer = find_lure_harmonic_seed(q=case["q_seed_integer"], system=system.lure, wmin=0.01, wmax=20.0, nscan=8000)
        seed_fractional = find_lure_harmonic_seed(q=case["q_seed_fractional"], system=system.lure, wmin=0.01, wmax=20.0, nscan=8000)
        attractor_name, series_name, attractor_info = plot_attractor(case, system, seed_integer)
        continuation_name, continuation_info = plot_continuation(case, system, seed_integer)
        summary[case["id"]] = {
            "system_name": case["system_name"],
            "parameters": dict(system.parameters),
            "typical_initial_condition_yaml": case["typical_ic"],
            "displayed_attractor_initial_condition": "integer describing-function seed",
            "integer_seed": seed_payload(seed_integer, case["q_seed_integer"]),
            "fractional_seed": seed_payload(seed_fractional, case["q_seed_fractional"]),
            "figures": {
                "attractor_3d": attractor_name,
                "time_series": series_name,
                "continuation_3d": continuation_name,
            },
            "attractor_info": attractor_info,
            "continuation_info": continuation_info,
        }

    summary["arctan_wu2023_centered_q095"] = format_arctan_wu2023_seed_report(q=0.95, nscan=8000)
    (OUT_DIR / "chua_model_seed_and_continuation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
