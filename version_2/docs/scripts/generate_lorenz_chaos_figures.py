from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures" / "chua_fractional_report"


def lorenz_rhs(t: float, x: np.ndarray) -> np.ndarray:
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    return np.array(
        [
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ],
        dtype=float,
    )


def combined_lorenz_rhs(state: np.ndarray) -> np.ndarray:
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    x, y, z = state[:3]
    tangent = state[3:].reshape(3, 3)
    vector_field = np.array(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ],
        dtype=float,
    )
    jacobian = np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ],
        dtype=float,
    )
    return np.concatenate([vector_field, (jacobian @ tangent).ravel()])


def rk4_step(state: np.ndarray, h: float) -> np.ndarray:
    k1 = combined_lorenz_rhs(state)
    k2 = combined_lorenz_rhs(state + 0.5 * h * k1)
    k3 = combined_lorenz_rhs(state + 0.5 * h * k2)
    k4 = combined_lorenz_rhs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def estimate_lyapunov(x0: np.ndarray) -> np.ndarray:
    h = 0.01
    burn = 10.0
    total_time = 120.0
    state = np.concatenate([x0.copy(), np.eye(3).ravel()])
    log_diag = np.zeros(3)
    accumulated_time = 0.0
    for step in range(int(total_time / h)):
        state = rk4_step(state, h)
        tangent = state[3:].reshape(3, 3)
        q_mat, r_mat = np.linalg.qr(tangent)
        signs = np.sign(np.diag(r_mat))
        signs[signs == 0.0] = 1.0
        q_mat = q_mat * signs
        r_mat = signs[:, None] * r_mat
        state[3:] = q_mat.ravel()
        time = (step + 1) * h
        if time > burn:
            log_diag += np.log(np.maximum(np.abs(np.diag(r_mat)), 1.0e-300))
            accumulated_time += h
    return log_diag / accumulated_time


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x0 = np.array([1.0, 1.0, 1.0], dtype=float)
    dt = 0.01
    t_eval = np.arange(0.0, 80.0 + 0.5 * dt, dt)
    sol = solve_ivp(
        lorenz_rhs,
        (0.0, 80.0),
        x0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    states = sol.y.T
    tail_mask = t >= 20.0
    t_tail = t[tail_mask]
    states_tail = states[tail_mask]

    z_section = 27.0
    poincare_points: list[tuple[float, float]] = []
    for i in range(len(states_tail) - 1):
        z0 = states_tail[i, 2]
        z1 = states_tail[i + 1, 2]
        if z0 < z_section <= z1:
            alpha = (z_section - z0) / (z1 - z0)
            x_cross = states_tail[i, 0] + alpha * (states_tail[i + 1, 0] - states_tail[i, 0])
            y_cross = states_tail[i, 1] + alpha * (states_tail[i + 1, 1] - states_tail[i, 1])
            poincare_points.append((float(x_cross), float(y_cross)))
    poincare = np.asarray(poincare_points, dtype=float)

    lyapunov = estimate_lyapunov(x0)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    idx = np.linspace(0, len(states_tail) - 1, min(len(states_tail), 9000)).astype(int)
    ax.plot(states_tail[idx, 0], states_tail[idx, 1], states_tail[idx, 2], lw=0.45, color="#245a9b")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Sistema de Lorenz: atractor caotico clasico")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chaos_lorenz_phase_3d.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.4), dpi=180, sharex=True)
    series_mask = (t >= 20.0) & (t <= 55.0)
    for ax, col, label, color in zip(
        axes,
        range(3),
        ["x(t)", "y(t)", "z(t)"],
        ["#245a9b", "#7a3b9f", "#b15f18"],
    ):
        ax.plot(t[series_mask] - 20.0, states[series_mask, col], lw=0.75, color=color)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("tiempo despues del transitorio")
    axes[0].set_title("Series temporales post-transitorio")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chaos_lorenz_time_series.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    if len(poincare):
        ax.scatter(poincare[:, 0], poincare[:, 1], s=8, alpha=0.62, c="#245a9b", edgecolors="none")
    ax.set_xlabel("x en cruce z=27")
    ax.set_ylabel("y en cruce z=27")
    ax.set_title(f"Seccion de Poincare ascendente, z=27, N={len(poincare)}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chaos_lorenz_poincare.png", bbox_inches="tight")
    plt.close(fig)

    x_signal = states_tail[:, 0] - np.mean(states_tail[:, 0])
    freq, psd = welch(x_signal, fs=1.0 / dt, nperseg=4096, noverlap=2048, scaling="density")
    valid = freq > 0.0
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=180)
    ax.semilogy(freq[valid], psd[valid], lw=0.9, color="#245a9b")
    ax.set_xlim(0.0, 8.0)
    ax.set_xlabel("frecuencia")
    ax.set_ylabel("PSD de x(t)")
    ax.set_title("Espectro amplio post-transitorio")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chaos_lorenz_spectrum.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "system": "Lorenz 1963",
        "parameters": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "initial_condition": x0.tolist(),
        "integration": {"t_final": 80.0, "dt": dt, "burn_in": 20.0, "method": "scipy.solve_ivp DOP853"},
        "lyapunov_qr": {"exponents": lyapunov.tolist(), "sum": float(np.sum(lyapunov))},
        "poincare": {"section": "z=27 upward", "points": int(len(poincare))},
    }
    (OUT_DIR / "chaos_lorenz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
