from __future__ import annotations

import ast
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION2 = ROOT / "version_2"
if str(VERSION2) not in sys.path:
    sys.path.insert(0, str(VERSION2))

from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import (
    ChuaParameters,
    equilibria_nonsmooth,
    jacobian_nonsmooth,
)
from hidden_attractors.seed_generation.chua import (
    build_fractional_seed,
    chua_matrices,
    describing_function,
    find_omega_gain_candidates,
    psi_sigma,
    solve_amplitude_from_gain,
    transfer_function,
)
from hidden_attractors.systems.builtins import _chua_lure_system


ALPHA = 8.4562
BETA = 12.0732
GAMMA = 0.0052
M1 = -1.2
M0 = -0.2
Q_DYNAMICS = 0.9998
H = 0.01
T_FINAL = 300.0
T_TRANSIENT = 100.0
CASE_ID = "m1_m1p2000_m0_m0p2000_branch_0"
DANCA_M0 = -0.1768
DANCA_M1 = -1.1468

DOC_DIR = ROOT / "DF y NC Chua entero y fraccionario copy"
FIG_DIR = DOC_DIR / "Figs"
VAL_DIR = (
    ROOT
    / "version_2"
    / "validation"
    / "outputs"
    / "published_continuation_comparison"
    / "danca2017_chua_fractional_saturation_candidate"
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


def _params() -> ChuaParameters:
    return ChuaParameters(
        model="nonsmooth",
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        m0=M0,
        m1=M1,
    )


def _danca_params() -> ChuaParameters:
    return ChuaParameters(
        model="nonsmooth",
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        m0=DANCA_M0,
        m1=DANCA_M1,
    )


def _equilibria_report(params: ChuaParameters, q: float) -> dict[str, object]:
    qpi2 = float(q * math.pi / 2.0)
    rows: dict[str, object] = {}
    for name, point in equilibria_nonsmooth(params).items():
        eigvals = np.linalg.eigvals(jacobian_nonsmooth(point, params))
        min_abs_arg = float(np.min(np.abs(np.angle(eigvals))))
        rows[name] = {
            "point": np.asarray(point, dtype=float).tolist(),
            "jacobian_eigenvalues": [
                [float(np.real(value)), float(np.imag(value))]
                for value in eigvals
            ],
            "min_abs_arg": min_abs_arg,
            "matignon_threshold_qpi2": qpi2,
            "locally_asymptotically_stable_matignon": bool(min_abs_arg > qpi2),
        }
    return rows


def _equilibria_csv_rows(label: str, params: ChuaParameters, q: float) -> list[dict[str, object]]:
    rows = []
    report = _equilibria_report(params, q)
    for name, payload in report.items():
        point = payload["point"]
        eigvals = payload["jacobian_eigenvalues"]
        rows.append(
            {
                "parameter_set": label,
                "equilibrium": name,
                "x": point[0],
                "y": point[1],
                "z": point[2],
                "eigenvalues_re_im": eigvals,
                "min_abs_arg": payload["min_abs_arg"],
                "matignon_threshold_qpi2": payload["matignon_threshold_qpi2"],
                "locally_asymptotically_stable_matignon": payload[
                    "locally_asymptotically_stable_matignon"
                ],
            }
        )
    return rows


def _system() -> SimpleNamespace:
    payload = {
        "model": "nonsmooth",
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": GAMMA,
        "m0": M0,
        "m1": M1,
    }
    return SimpleNamespace(
        system_id="chua_fractional_saturation",
        name="chua_fractional_saturation",
        parameters=payload,
        lure=_chua_lure_system(payload),
    )


def _read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == CASE_ID:
                return row
    raise RuntimeError(f"{CASE_ID} not found in {path}")


def _read_trajectory(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _post_transient(traj: np.ndarray) -> np.ndarray:
    return traj[int(T_TRANSIENT / H) :]


def _dominant_frequency(traj: np.ndarray) -> dict[str, float]:
    tail = _post_transient(traj)
    x = tail[:, 1] - np.mean(tail[:, 1])
    freqs = np.fft.rfftfreq(len(x), d=H)
    mag = np.abs(np.fft.rfft(x))
    if len(mag) > 0:
        mag[0] = 0.0
    idx = int(np.argmax(mag))
    fd = float(freqs[idx])
    return {"fft_hz": fd, "fft_rad_s": float(2.0 * math.pi * fd)}


def _welch_frequency(traj: np.ndarray) -> dict[str, float]:
    tail = _post_transient(traj)
    x = tail[:, 1] - np.mean(tail[:, 1])
    nper = min(4096, len(x))
    if nper < 32:
        return {"psd_hz": 0.0, "psd_rad_s": 0.0}
    step = nper // 2
    window = np.hanning(nper)
    spec = None
    count = 0
    for start in range(0, len(x) - nper + 1, step):
        seg = x[start : start + nper] * window
        cur = np.abs(np.fft.rfft(seg)) ** 2
        spec = cur if spec is None else spec + cur
        count += 1
    assert spec is not None
    spec = spec / max(count, 1)
    spec[0] = 0.0
    freqs = np.fft.rfftfreq(nper, d=H)
    idx = int(np.argmax(spec))
    fd = float(freqs[idx])
    return {"psd_hz": fd, "psd_rad_s": float(2.0 * math.pi * fd)}


def _seed_report(q_seed: float) -> dict[str, object]:
    params = _params()
    pairs = find_omega_gain_candidates(q=q_seed, params=params)
    omega0, k = pairs[0]
    a0 = solve_amplitude_from_gain(k, params)
    seed, vector, matched = build_fractional_seed(q_seed, params, omega0, k, a0)
    pmat, bvec, rvec = chua_matrices(params)
    p0 = pmat + k * np.outer(bvec, rvec)
    eigvals = np.linalg.eigvals(p0)
    response = transfer_function(omega0, q_seed, params)
    nval = describing_function(a0, params)
    return {
        "q_seed": q_seed,
        "omega0": float(omega0),
        "k": float(k),
        "A0": float(a0),
        "seed": seed.tolist(),
        "modal_vector": [[float(np.real(v)), float(np.imag(v))] for v in vector],
        "matched_eigenvalue": [float(np.real(matched)), float(np.imag(matched))],
        "P0": p0.tolist(),
        "P0_eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eigvals],
        "W": [float(np.real(response)), float(np.imag(response))],
        "N_A": float(nval),
        "closure_residual_1_plus_WN": float(abs(1.0 + response * nval)),
    }


def _run_fractional_continuation(q_seed: float, memory_mode: str) -> list[dict[str, object]]:
    params = _params()
    seed_info = _seed_report(q_seed)
    system = _system()
    window_len = int(10.0 / H) if memory_mode == "window" else None
    return run_fractional_continuation(
        system=system,
        seed_x0=np.asarray(seed_info["seed"], dtype=float),
        k_gain=float(seed_info["k"]),
        lambda_values=np.linspace(0.0, 1.0, 11),
        h=H,
        memory_mode=memory_mode,
        memory_window_length=window_len,
        integrator="abm",
        use_c_backend=True,
        require_c_backend=False,
        allow_python_fallback=True,
        t_transient=30.0,
        t_keep=30.0,
        q=Q_DYNAMICS,
    )


def _rk4(rhs, x0: np.ndarray, t_final: float) -> np.ndarray:
    steps = int(round(t_final / H))
    out = np.empty((steps + 1, 4), dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    out[0, 0] = 0.0
    out[0, 1:] = x
    t = 0.0
    for i in range(1, steps + 1):
        k1 = rhs(t, x)
        k2 = rhs(t + 0.5 * H, x + 0.5 * H * k1)
        k3 = rhs(t + 0.5 * H, x + 0.5 * H * k2)
        k4 = rhs(t + H, x + H * k3)
        x = x + (H / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = i * H
        out[i, 0] = t
        out[i, 1:] = x
    return out


def _run_integer_continuation() -> tuple[list[dict[str, object]], np.ndarray]:
    params = _params()
    seed_info = _seed_report(1.0)
    pmat, bvec, rvec = chua_matrices(params)
    k = float(seed_info["k"])
    p0 = pmat + k * np.outer(bvec, rvec)
    x = np.asarray(seed_info["seed"], dtype=float)
    rows: list[dict[str, object]] = []
    for idx, eta in enumerate(np.linspace(0.0, 1.0, 11)):
        def rhs(_t, val, eta_f=float(eta)):
            sigma = float(rvec @ val)
            delta = psi_sigma(sigma, params) - k * sigma
            return p0 @ val + eta_f * bvec * delta

        traj = _rk4(rhs, x, 60.0)
        x = traj[-1, 1:].copy()
        rows.append(
            {
                "eta": float(eta),
                "step": idx,
                "x_out": x.tolist(),
                "max_norm": float(np.max(np.linalg.norm(traj[:, 1:], axis=1))),
            }
        )

    def rhs_original(_t, val):
        sigma = float(rvec @ val)
        return pmat @ val + bvec * psi_sigma(sigma, params)

    final_traj = _rk4(rhs_original, x, T_FINAL)
    return rows, final_traj


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


def _save_np_csv(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header="t,x,y,z", comments="")


def _savefig(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(
            FIG_DIR / f"{name}.{ext}",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )


def _sample_trajectory(traj: np.ndarray, max_points: int) -> np.ndarray:
    if len(traj) <= max_points:
        return traj
    idx = np.linspace(0, len(traj) - 1, max_points, dtype=int)
    return traj[idx]


def _plot_transfer(seed_frac: dict[str, object], seed_int: dict[str, object]) -> None:
    params = _params()
    ws = np.linspace(1.0e-4, 10.2, 1000)
    wq = np.array([transfer_function(float(w), Q_DYNAMICS, params) for w in ws])
    wi = np.array([transfer_function(float(w), 1.0, params) for w in ws])
    omega_frac = float(seed_frac["omega0"])
    omega_int = float(seed_int["omega0"])
    k_frac = float(seed_frac["k"])
    closure_real = -1.0 / k_frac

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), sharex=True)

    axes[0].plot(ws, wq.real, color="#0b57ff", lw=1.5, label=r"$\Re(W_q(i\omega))$")
    axes[0].plot(ws, wi.real, color="#0b57ff", lw=1.0, ls=":", alpha=0.75, label=r"$\Re(W(i\omega)), q=1$")
    axes[0].axhline(closure_real, color="orangered", lw=1.0, ls="--", label=r"$-1/k$")
    axes[0].axvline(omega_frac, color="0.15", lw=1.0, ls=":")
    axes[0].axvline(omega_int, color="0.55", lw=0.9, ls=":", alpha=0.8)
    axes[0].plot(
        [omega_frac],
        [float(np.real(transfer_function(omega_frac, Q_DYNAMICS, params)))],
        marker="o",
        ms=8,
        mfc="white",
        mec="#ef4444",
        mew=1.8,
        linestyle="None",
        label="cierre seleccionado",
    )
    axes[0].set_ylabel(r"$\Re(W_q(i\omega))$")
    axes[0].grid(True, ls=":", lw=0.6, alpha=0.75)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].plot(ws, wq.imag, color="#0b57ff", lw=1.5, label=r"$\Im(W_q(i\omega))$")
    axes[1].plot(ws, wi.imag, color="#0b57ff", lw=1.0, ls=":", alpha=0.75, label=r"$\Im(W(i\omega)), q=1$")
    axes[1].axhline(0.0, color="orangered", lw=1.0, ls="--", label="condicion de cruce")
    axes[1].axvline(omega_frac, color="0.15", lw=1.0, ls=":")
    axes[1].axvline(omega_int, color="0.55", lw=0.9, ls=":", alpha=0.8)
    axes[1].plot(
        [omega_frac],
        [float(np.imag(transfer_function(omega_frac, Q_DYNAMICS, params)))],
        marker="o",
        ms=8,
        mfc="white",
        mec="#ef4444",
        mew=1.8,
        linestyle="None",
        label="raiz seleccionada",
    )
    axes[1].set_xlabel(r"$\omega$ (rad/s)")
    axes[1].set_ylabel(r"$\Im(W_q(i\omega))$")
    axes[1].grid(True, ls=":", lw=0.6, alpha=0.75)
    axes[1].legend(fontsize=8, loc="best")

    fig.tight_layout()
    _savefig("chua_frac_ns_fig01_transfer_real_imag")
    plt.close(fig)

    fig_real, ax_real = plt.subplots(figsize=(7.2, 3.2))
    ax_real.plot(ws, wq.real, color="#0b57ff", lw=1.5, label=r"$\Re(W_q(i\omega))$")
    ax_real.plot(ws, wi.real, color="#0b57ff", lw=1.0, ls=":", alpha=0.75, label=r"$\Re(W(i\omega)), q=1$")
    ax_real.axhline(closure_real, color="orangered", lw=1.0, ls="--", label=r"$-1/k$")
    ax_real.axvline(omega_frac, color="0.15", lw=1.0, ls=":")
    ax_real.axvline(omega_int, color="0.55", lw=0.9, ls=":", alpha=0.8)
    ax_real.plot(
        [omega_frac],
        [float(np.real(transfer_function(omega_frac, Q_DYNAMICS, params)))],
        marker="o",
        ms=8,
        mfc="white",
        mec="#ef4444",
        mew=1.8,
        linestyle="None",
        label="cierre seleccionado",
    )
    ax_real.set_xlabel(r"$\omega$ (rad/s)")
    ax_real.set_ylabel(r"$\Re(W_q(i\omega))$")
    ax_real.grid(True, ls=":", lw=0.6, alpha=0.75)
    ax_real.legend(fontsize=8, loc="best")
    fig_real.tight_layout()
    _savefig("chua_frac_ns_fig01a_transfer_real")
    plt.close(fig_real)

    fig_imag, ax_imag = plt.subplots(figsize=(7.2, 3.2))
    ax_imag.plot(ws, wq.imag, color="#0b57ff", lw=1.5, label=r"$\Im(W_q(i\omega))$")
    ax_imag.plot(ws, wi.imag, color="#0b57ff", lw=1.0, ls=":", alpha=0.75, label=r"$\Im(W(i\omega)), q=1$")
    ax_imag.axhline(0.0, color="orangered", lw=1.0, ls="--", label="condicion de cruce")
    ax_imag.axvline(omega_frac, color="0.15", lw=1.0, ls=":")
    ax_imag.axvline(omega_int, color="0.55", lw=0.9, ls=":", alpha=0.8)
    ax_imag.plot(
        [omega_frac],
        [float(np.imag(transfer_function(omega_frac, Q_DYNAMICS, params)))],
        marker="o",
        ms=8,
        mfc="white",
        mec="#ef4444",
        mew=1.8,
        linestyle="None",
        label="raiz seleccionada",
    )
    ax_imag.set_xlabel(r"$\omega$ (rad/s)")
    ax_imag.set_ylabel(r"$\Im(W_q(i\omega))$")
    ax_imag.grid(True, ls=":", lw=0.6, alpha=0.75)
    ax_imag.legend(fontsize=8, loc="best")
    fig_imag.tight_layout()
    _savefig("chua_frac_ns_fig01b_transfer_imag")
    plt.close(fig_imag)


def _plot_continuation(frac_full_rows, final_traj: np.ndarray) -> None:
    fig = plt.figure(figsize=(7.1, 5.7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0.08, 0.86, len(frac_full_rows)))
    x_in_points = []

    for idx, (row, color) in enumerate(zip(frac_full_rows, colors)):
        eta = float(row.get("lambda_value", row.get("eta")))
        traj = np.asarray(row.get("trajectory", []), dtype=float)
        x_in = np.asarray(row["x_in"], dtype=float)
        x_in_points.append(x_in)
        if traj.ndim == 2 and traj.shape[1] >= 4 and len(traj) > 0:
            sample = _sample_trajectory(traj, 850)
            label = r"$\varepsilon=0$" if idx == 0 else (r"$\varepsilon=1$" if idx == len(frac_full_rows) - 1 else None)
            ax.plot(sample[:, 1], sample[:, 2], sample[:, 3], lw=0.7, color=color, alpha=0.75, label=label)
        ax.scatter([x_in[0]], [x_in[1]], [x_in[2]], s=16, color=color, edgecolor="k", linewidth=0.25)
        if idx in {0, len(frac_full_rows) - 1} or idx % 2 == 0:
            ax.text(x_in[0], x_in[1], x_in[2], f"{eta:.1f}", fontsize=6)

    x_in_arr = np.asarray(x_in_points, dtype=float)
    ax.plot(x_in_arr[:, 0], x_in_arr[:, 1], x_in_arr[:, 2], color="k", lw=1.0, ls=":", label="condiciones iniciales")

    tail = _sample_trajectory(_post_transient(final_traj), 1800)
    ax.plot(tail[:, 1], tail[:, 2], tail[:, 3], color="crimson", lw=0.55, alpha=0.9, label="atractor final")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="best")
    _savefig("chua_frac_ns_fig02_continuation_story")
    plt.close(fig)


def _plot_attractor(traj: np.ndarray, seed: np.ndarray, continuation_final: np.ndarray, suffix: str = "") -> None:
    tail = _post_transient(traj)
    sample = tail
    if len(sample) > 8000:
        idx = np.linspace(0, len(sample) - 1, 8000, dtype=int)
        sample = sample[idx]
    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sample[:, 1], sample[:, 2], sample[:, 3], lw=0.35)
    ax.scatter(
        [seed[0]], [seed[1]], [seed[2]],
        s=42, color="#facc15", edgecolor="black", linewidth=0.55, label="semilla efectiva"
    )
    ax.scatter(
        [continuation_final[0]], [continuation_final[1]], [continuation_final[2]],
        s=34, color="black", label="estado final de continuacion"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="best")
    _savefig(f"chua_frac_ns_fig03_final_attractor{suffix}")
    plt.close(fig)


def _plot_projections(traj: np.ndarray) -> None:
    tail = _post_transient(traj)
    sample = tail
    if len(sample) > 10000:
        idx = np.linspace(0, len(sample) - 1, 10000, dtype=int)
        sample = sample[idx]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    pairs = [(1, 2, "x", "y"), (1, 3, "x", "z"), (2, 3, "y", "z")]
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        ax.plot(sample[:, i], sample[:, j], lw=0.35)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, ls=":", lw=0.5)
    fig.tight_layout()
    _savefig("chua_frac_ns_fig03abc_final_attractor_projections")
    plt.close(fig)


def _plot_timeseries(traj: np.ndarray) -> None:
    tail = _post_transient(traj)
    sample = tail
    if len(sample) > 6000:
        idx = np.linspace(0, len(sample) - 1, 6000, dtype=int)
        sample = sample[idx]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.0), sharex=True)
    for ax, col, label in zip(axes, (1, 2, 3), ("x", "y", "z")):
        ax.plot(sample[:, 0], sample[:, col], lw=0.45)
        ax.set_ylabel(label)
        ax.grid(True, ls=":", lw=0.5)
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    _savefig("chua_frac_ns_fig04_time_series")
    plt.close(fig)


def _plot_spectrum(traj: np.ndarray, omega0: float) -> None:
    tail = _post_transient(traj)
    x = tail[:, 1] - np.mean(tail[:, 1])
    freqs = np.fft.rfftfreq(len(x), d=H)
    mag = np.abs(np.fft.rfft(x))
    mag[0] = 0.0
    plt.figure(figsize=(6.5, 4.0))
    omega = 2.0 * math.pi * freqs
    plt.plot(omega, mag / max(np.max(mag), 1.0), lw=0.7, label=r"FFT normalizada")
    plt.axvline(omega0, color="red", lw=1.0, ls="-", label=rf"$\omega_0={omega0:.4f}$")
    plt.xlim(0.0, 8.0)
    plt.xlabel(r"$\omega$ [rad/s]")
    plt.ylabel("FFT normalizada de x")
    plt.grid(True, ls=":", lw=0.6)
    plt.legend(fontsize=8, loc="best")
    _savefig("chua_frac_ns_fig11a_fft_x")
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    full_summary = _read_summary(ROOT / "outputs" / "saturation_search_seed0p9998_mem_full_sweep" / "summary.csv")
    win_summary = _read_summary(ROOT / "outputs" / "saturation_search_seed0p9998_mem_window_sweep" / "summary.csv")
    int_seed_summary = _read_summary(ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep" / "summary.csv")
    full_traj = _read_trajectory(
        ROOT
        / "outputs"
        / "saturation_search_seed0p9998_mem_full_sweep"
        / f"{CASE_ID}_trajectory.csv"
    )
    win_traj = _read_trajectory(
        ROOT
        / "outputs"
        / "saturation_search_seed0p9998_mem_window_sweep"
        / f"{CASE_ID}_trajectory.csv"
    )

    seed_frac = _seed_report(Q_DYNAMICS)
    seed_int = _seed_report(1.0)
    frac_full_steps = _run_fractional_continuation(Q_DYNAMICS, "full")
    frac_win_steps = _run_fractional_continuation(Q_DYNAMICS, "window")
    int_steps, int_traj = _run_integer_continuation()
    _save_np_csv(VAL_DIR / "integer_continuation_final_trajectory.csv", int_traj)
    _write_csv(VAL_DIR / "integer_continuation_steps.csv", int_steps)
    _write_csv(
        VAL_DIR / "fractional_full_continuation_steps.csv",
        [
            {
                "lambda_value": row["lambda_value"],
                "status": row["status"],
                "x_in": np.asarray(row["x_in"], dtype=float).tolist(),
                "x_out": np.asarray(row["x_out"], dtype=float).tolist(),
                "max_norm": row.get("max_norm", None),
            }
            for row in frac_full_steps
        ],
    )
    _write_csv(
        VAL_DIR / "fractional_window_continuation_steps.csv",
        [
            {
                "lambda_value": row["lambda_value"],
                "status": row["status"],
                "x_in": np.asarray(row["x_in"], dtype=float).tolist(),
                "x_out": np.asarray(row["x_out"], dtype=float).tolist(),
                "max_norm": row.get("max_norm", None),
            }
            for row in frac_win_steps
        ],
    )
    _write_csv(
        VAL_DIR / "equilibria_parameter_sets.csv",
        _equilibria_csv_rows("danca2017_reference", _danca_params(), Q_DYNAMICS)
        + _equilibria_csv_rows("selected_experiment", _params(), Q_DYNAMICS),
    )

    int_diag = classify_post_transient_periodicity(int_traj, h=H, config={"t_transient": T_TRANSIENT})
    full_freq = {**_dominant_frequency(full_traj), **_welch_frequency(full_traj)}
    win_freq = {**_dominant_frequency(win_traj), **_welch_frequency(win_traj)}
    int_freq = {**_dominant_frequency(int_traj), **_welch_frequency(int_traj)}
    target = _post_transient(full_traj)[-1, 1:].tolist()

    _plot_transfer(seed_frac, seed_int)
    _plot_continuation(frac_full_steps, full_traj)
    _plot_attractor(
        full_traj,
        np.asarray(seed_frac["seed"], dtype=float),
        np.asarray(frac_full_steps[-1]["x_out"], dtype=float),
    )
    _plot_projections(full_traj)
    _plot_timeseries(full_traj)
    _plot_spectrum(full_traj, float(seed_frac["omega0"]))

    report = {
        "case_id": CASE_ID,
        "date": "2026-06-07",
        "scope": "candidate_isolation_before_hiddenness_tests",
        "parameters": {
            "model": "nonsmooth",
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "m0": M0,
            "m1": M1,
            "q_dynamics": Q_DYNAMICS,
        },
        "reference_danca_parameters": {
            "model": "nonsmooth",
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "m0": DANCA_M0,
            "m1": DANCA_M1,
            "q_for_matignon_comparison": Q_DYNAMICS,
            "equilibria": _equilibria_report(_danca_params(), Q_DYNAMICS),
        },
        "selected_experiment_parameters": {
            "model": "nonsmooth",
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "m0": M0,
            "m1": M1,
            "q_for_matignon_comparison": Q_DYNAMICS,
            "equilibria": _equilibria_report(_params(), Q_DYNAMICS),
        },
        "planned_lyapunov_characterization": {
            "primary_method": "fractional_variational_abm_qr",
            "primary_memory_mode": "full",
            "secondary_method": "fractional_cloned_dynamics_abm_gs_published",
            "secondary_memory_protocol": "published_block_restart",
            "purpose": "finite_time_chaos_characterization",
            "positive_criterion": "at_least_one_positive_exponent_with_bounded_noncollapsed_trajectory",
            "scope_note": "Lyapunov exponents characterize chaos evidence; hiddenness is decided by equilibrium-neighborhood basin probes.",
        },
        "proposed_fractional_method": {
            "seed": seed_frac,
            "continuation": {
                "integrator": "ABM",
                "derivative": "Caputo",
                "q": Q_DYNAMICS,
                "h": H,
                "lambda_values": "0.0:0.1:1.0",
                "t_transient_per_step": 30.0,
                "t_keep_per_step": 30.0,
                "full_memory_summary_row": full_summary,
                "window_memory_summary_row": win_summary,
                "full_eta1_state": np.asarray(frac_full_steps[-1]["x_out"], dtype=float).tolist(),
                "full_initial_conditions": [
                    np.asarray(row["x_in"], dtype=float).tolist()
                    for row in frac_full_steps
                ],
                "window_eta1_state": np.asarray(frac_win_steps[-1]["x_out"], dtype=float).tolist(),
            },
            "final_simulation": {
                "integrator": "ABM",
                "memory_mode": "full",
                "h": H,
                "t_final": T_FINAL,
                "t_transient": T_TRANSIENT,
                "target_reference_state": target,
                "frequency": full_freq,
                "hiddenness_tests": "not_run",
                "lyapunov_tests": "not_run",
            },
        },
        "article_style_integer_comparison": {
            "reason": "Danca 2017 does not report a continuation algorithm; this artifact records the integer DF plus integer continuation control route.",
            "seed": seed_int,
            "continuation": {
                "integrator": "RK4",
                "derivative": "integer_ode",
                "h": H,
                "lambda_values": "0.0:0.1:1.0",
                "t_per_step": 60.0,
                "eta1_state": int_steps[-1]["x_out"],
            },
            "integer_final_simulation": {
                "h": H,
                "t_final": T_FINAL,
                "t_transient": T_TRANSIENT,
                "diagnostic_label": int_diag["candidate_label"],
                "frequency": int_freq,
            },
            "caputo_seed1_existing_row": int_seed_summary,
        },
        "figure_files": {
            "transfer_real": "Figs/chua_frac_ns_fig01a_transfer_real.pdf",
            "transfer_imag": "Figs/chua_frac_ns_fig01b_transfer_imag.pdf",
            "transfer_combined_legacy": "Figs/chua_frac_ns_fig01_transfer_real_imag.pdf",
            "continuation": "Figs/chua_frac_ns_fig02_continuation_story.pdf",
            "attractor": "Figs/chua_frac_ns_fig03_final_attractor.pdf",
            "projections": "Figs/chua_frac_ns_fig03abc_final_attractor_projections.pdf",
            "time_series": "Figs/chua_frac_ns_fig04_time_series.pdf",
            "fft_x": "Figs/chua_frac_ns_fig11a_fft_x.pdf",
        },
    }

    with (VAL_DIR / "candidate_method_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps({
        "validation_dir": str(VAL_DIR.relative_to(ROOT)),
        "figure_dir": str(FIG_DIR.relative_to(ROOT)),
        "case_id": CASE_ID,
        "fractional_verdict": full_summary["verdict"],
        "window_verdict": win_summary["verdict"],
        "integer_diagnostic": int_diag["candidate_label"],
        "target": target,
        "fft_rad_s": full_freq["fft_rad_s"],
        "psd_rad_s": full_freq["psd_rad_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
