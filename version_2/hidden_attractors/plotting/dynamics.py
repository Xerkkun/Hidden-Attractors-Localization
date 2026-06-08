"""Plotting functions for dynamical-system trajectories and scans."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.bifurcation import BifurcationPoint, observable_column
from ..analysis.lyapunov import LyapunovResult
from ..analysis.spectral import SpectrumResult, trajectory_component_spectra
from ..analysis.trajectory import sample_rows
from ..seed_generation import (
    HarmonicSeed,
    find_lure_omega_gain_candidates,
    lure_describing_function,
    lure_machado_describing_function,
    lure_transfer_function,
)
from ..systems.lure import LureSystem
from ..workflows.integer_lure import IntegerHiddennessProbe, IntegerLureContinuationStep


def _output_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_phase_space(
    trajectory: np.ndarray,
    output_path: str | Path,
    *,
    dims: Sequence[str | int] = ("x", "y", "z"),
    title: str | None = "Phase space",
    max_points: int = 5000,
    color_by_time: bool = True,
) -> str:
    """Plot a 2D or 3D phase-space view of a ``t,x,y,z`` trajectory."""

    X = sample_rows(np.asarray(trajectory, dtype=float), max_points)
    cols = [observable_column(dim) for dim in dims]
    if len(cols) not in {2, 3}:
        raise ValueError("dims must contain two or three observables")
    if max(cols) >= X.shape[1]:
        raise ValueError("trajectory does not contain all requested dimensions")

    path = _output_path(output_path)
    fig = plt.figure(figsize=(7.5, 6.5))
    colors = X[:, 0] if color_by_time and X.shape[1] > 0 else "#2563eb"
    if len(cols) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, cols[0]], X[:, cols[1]], X[:, cols[2]], c=colors, s=2.2, cmap="viridis", alpha=0.78)
        ax.set_zlabel(str(dims[2]))
    else:
        ax = fig.add_subplot(111)
        ax.scatter(X[:, cols[0]], X[:, cols[1]], c=colors, s=2.2, cmap="viridis", alpha=0.78)
    ax.set_xlabel(str(dims[0]))
    ax.set_ylabel(str(dims[1]))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "attractor")
    plt.close(fig)
    return str(path)


def plot_phase_projections(
    trajectory: np.ndarray,
    output_path: str | Path,
    *,
    title: str | None = "Phase projections",
    max_points: int = 5000,
) -> str:
    """Plot standard ``xy``, ``xz``, and ``yz`` projections."""

    X = sample_rows(np.asarray(trajectory, dtype=float), max_points)
    path = _output_path(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    specs = [(1, 2, "x", "y"), (1, 3, "x", "z"), (2, 3, "y", "z")]
    for ax, (a, b, xlabel, ylabel) in zip(axes, specs):
        ax.plot(X[:, a], X[:, b], lw=0.55, color="#2563eb", alpha=0.82)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(f"{xlabel}{ylabel}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "attractor")
    plt.close(fig)
    return str(path)


def plot_time_series(
    trajectory: np.ndarray,
    output_path: str | Path,
    *,
    columns: Sequence[str | int] = ("x", "y", "z"),
    title: str | None = "Time series",
    max_points: int = 6000,
) -> str:
    """Plot selected trajectory coordinates against time."""

    X = sample_rows(np.asarray(trajectory, dtype=float), max_points)
    path = _output_path(output_path)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for column in columns:
        col = observable_column(column)
        ax.plot(X[:, 0], X[:, col], lw=0.75, label=str(column))
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "time_series")
    plt.close(fig)
    return str(path)


def plot_bifurcation_diagram(
    points: Sequence[BifurcationPoint],
    output_path: str | Path,
    *,
    parameter_label: str = "parameter",
    observable_label: str = "observable",
    title: str = "Bifurcation diagram",
) -> str:
    """Plot extracted bifurcation points from a parameter scan."""

    path = _output_path(output_path)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if points:
        params = np.array([p.parameter for p in points], dtype=float)
        values = np.array([p.observable for p in points], dtype=float)
        ax.scatter(params, values, s=2.5, color="#111827", alpha=0.72)
    ax.set_xlabel(parameter_label)
    ax.set_ylabel(observable_label)
    ax.set_title(title)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "bifurcation")
    plt.close(fig)
    return str(path)


def _state_column(dim: str | int) -> int:
    if isinstance(dim, str):
        aliases = {"x": 1, "y": 2, "z": 3, "t": 0}
        return aliases.get(dim.lower(), int(dim))
    if int(dim) == 0:
        return 1
    return int(dim)


def plot_lure_nyquist_describing_function(
    system: LureSystem,
    seed: HarmonicSeed,
    output_path: str | Path,
    *,
    q: float = 1.0,
    method: str | None = None,
    mu: float | None = None,
    wmin: float = 1.0e-5,
    wmax: float = 50.0,
    amin: float = 1.0 + 1.0e-8,
    amax: float | None = None,
    title: str = "Lur'e Nyquist/DF closure",
) -> str:
    """Plot ``W_q(i omega)`` and ``-1/N(A)`` for any Lur'e system."""

    path = _output_path(output_path)
    omega = np.logspace(np.log10(float(wmin)), np.log10(float(wmax)), 2400)
    wvals = np.array([lure_transfer_function(float(w), q, system) for w in omega], dtype=complex)
    a_hi = float(amax) if amax is not None else max(50.0, 1.25 * float(seed.amplitude))
    amplitudes = np.linspace(float(amin), a_hi, 1800)
    mode = method or seed.method
    if mode == "machado":
        exponent = float(seed.mu if mu is None else mu)
        nvals = np.array([lure_machado_describing_function(float(a), system, exponent) for a in amplitudes], dtype=float)
    else:
        nvals = np.array([np.real(lure_describing_function(float(a), system)) for a in amplitudes], dtype=float)
    valid = np.isfinite(nvals) & (np.abs(nvals) > 1.0e-14)
    minus_inv = np.full_like(nvals, np.nan, dtype=float)
    minus_inv[valid] = -1.0 / nvals[valid]
    w0 = lure_transfer_function(seed.omega, q, system)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.plot(np.real(wvals), np.imag(wvals), lw=1.25, color="#0047ff", label=r"$W_q(i\omega)$")
    ax.plot(minus_inv, np.zeros_like(minus_inv), lw=1.1, color="#ff4a1a", label=r"$-1/N(A)$")
    ax.scatter([np.real(w0)], [np.imag(w0)], s=58, facecolors="none", edgecolors="#ef4444", linewidths=1.4, label="chosen closure")
    ax.scatter([-1.0 / seed.gain], [0.0], s=52, c="#ff4a1a", marker="x", linewidths=1.6)
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.axvline(0.0, color="#9ca3af", ls=":", lw=0.7)
    ax.set_xlabel(r"Re$(W_q(i\omega))$")
    ax.set_ylabel(r"Im$(W_q(i\omega))$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "nyquist")
    plt.close(fig)
    return str(path)


def plot_lure_transfer_components(
    system: LureSystem,
    seed: HarmonicSeed,
    output_path: str | Path,
    *,
    q: float = 1.0,
    wmin: float = 1.0e-4,
    wmax: float = 10.2,
    nscan: int = 5000,
    title: str | None = None,
) -> str:
    """Plot real and imaginary transfer-function closure conditions.

    The two panels expose the scalar checks used to choose the harmonic seed:
    ``Re(W_q(i omega_0)) = -1/k`` and ``Im(W_q(i omega_0)) = 0``.
    """

    path = _output_path(output_path)
    omega = np.linspace(float(wmin), float(wmax), int(nscan))
    wvals = np.array([lure_transfer_function(float(w), q, system) for w in omega], dtype=complex)
    w0 = lure_transfer_function(seed.omega, q, system)
    pairs = find_lure_omega_gain_candidates(
        q,
        system,
        wmin=wmin,
        wmax=wmax,
        nscan=nscan,
        compatible_only=False,
    )
    alternate_roots = [float(root) for root, _gain in pairs if not np.isclose(root, seed.omega)]

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True)
    if title:
        fig.suptitle(title)

    axes[0].plot(omega, np.real(wvals), lw=1.6, color="#2563eb", label=r"Re$(W_q(i\omega))$")
    axes[0].axhline(-1.0 / seed.gain, color="#ef4444", ls="--", lw=1.15, label=r"$-1/k$")
    axes[0].scatter([seed.omega], [np.real(w0)], s=48, facecolors="white", edgecolors="#ef4444", linewidths=1.4, zorder=4, label="selected closure")
    axes[0].set_ylabel(r"Re$(W_q(i\omega))$")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(omega, np.imag(wvals), lw=1.6, color="#2563eb", label=r"Im$(W_q(i\omega))$")
    axes[1].axhline(0.0, color="#ef4444", ls="--", lw=1.15, label="zero-crossing condition")
    axes[1].scatter([seed.omega], [np.imag(w0)], s=48, facecolors="white", edgecolors="#ef4444", linewidths=1.4, zorder=4, label="selected root")
    axes[1].set_xlabel(r"$\omega$ (rad/s)")
    axes[1].set_ylabel(r"Im$(W_q(i\omega))$")

    for ax in axes:
        ax.axvline(seed.omega, color="#374151", ls=":", lw=0.9)
        for index, root in enumerate(alternate_roots):
            label = "alternate root" if index == 0 else None
            ax.axvline(root, color="#9ca3af", ls=":", lw=0.8, label=label)
        ax.grid(True, color="#e5e7eb", lw=0.8)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "transfer")
    plt.close(fig)
    return str(path)


def plot_integer_lure_continuation(
    steps: Sequence[IntegerLureContinuationStep],
    output_path: str | Path,
    *,
    dims: Sequence[int] = (0, 1, 2),
    title: str | None = "Integer Lur'e continuation",
    max_points: int = 1500,
) -> str:
    """Plot epsilon-continuation trajectories for any integer Lur'e system."""

    if not steps:
        raise ValueError("steps must not be empty.")
    path = _output_path(output_path)
    fig = plt.figure(figsize=(7.4, 5.9))
    ax = fig.add_subplot(111, projection="3d")
    eps_values = np.array([step.epsilon for step in steps], dtype=float)
    eps_min = float(np.min(eps_values))
    eps_max = float(np.max(eps_values))
    cmap = plt.get_cmap("plasma")
    cols = [1 + int(dim) for dim in dims]
    xout = np.vstack([step.x_out for step in steps])
    ax.plot(xout[:, dims[0]], xout[:, dims[1]], xout[:, dims[2]], color="0.35", ls=":", lw=1.0, label="epsilon path")
    for step in steps:
        data = sample_rows(np.asarray(step.trajectory, dtype=float), max_points)
        color = cmap((float(step.epsilon) - eps_min) / max(1.0e-12, eps_max - eps_min))
        ax.plot(data[:, cols[0]], data[:, cols[1]], data[:, cols[2]], lw=0.8, color=color, alpha=0.92)
    ax.set_xlabel(f"x{dims[0]}")
    ax.set_ylabel(f"x{dims[1]}")
    ax.set_zlabel(f"x{dims[2]}")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "continuation")
    plt.close(fig)
    return str(path)


def plot_fractional_continuation_phase_story(
    steps: Sequence[dict],
    output_path: str | Path,
    *,
    final_trajectory: np.ndarray | None = None,
    seed_effective: Sequence[float] | None = None,
    continuation_final: Sequence[float] | None = None,
    max_step_points: int = 850,
    max_final_points: int = 1800,
) -> str:
    """Plot fractional continuation trajectories directly in phase space.

    The standard report figure overlays each continuation trajectory, marks
    every initial condition, and connects those initial conditions with a
    dotted line.  If a final physical trajectory is supplied, it is drawn in
    red after the continuation path.
    """

    if not steps:
        raise ValueError("steps must not be empty.")

    path = _output_path(output_path)
    fig = plt.figure(figsize=(7.1, 5.7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0.08, 0.86, len(steps)))
    x_in_points: list[np.ndarray] = []

    for idx, (row, color) in enumerate(zip(steps, colors)):
        eta_raw = row.get("lambda_value", row.get("eta", idx))
        eta = float(eta_raw)
        traj = np.asarray(row.get("trajectory", []), dtype=float)
        x_in = np.asarray(row.get("x_in", row.get("x0", row.get("seed", []))), dtype=float)
        if x_in.size >= 3:
            x_in = x_in[:3]
            x_in_points.append(x_in)
            ax.scatter([x_in[0]], [x_in[1]], [x_in[2]], s=16, color=color, edgecolor="k", linewidth=0.25)
            if idx in {0, len(steps) - 1} or idx % 2 == 0:
                ax.text(x_in[0], x_in[1], x_in[2], f"{eta:.1f}", fontsize=6)
        if traj.ndim == 2 and traj.shape[1] >= 4 and len(traj) > 0:
            sample = sample_rows(traj, max_step_points)
            label = r"$\varepsilon=0$" if idx == 0 else (r"$\varepsilon=1$" if idx == len(steps) - 1 else None)
            ax.plot(sample[:, 1], sample[:, 2], sample[:, 3], lw=0.7, color=color, alpha=0.75, label=label)

    if x_in_points:
        x_in_arr = np.asarray(x_in_points, dtype=float)
        ax.plot(x_in_arr[:, 0], x_in_arr[:, 1], x_in_arr[:, 2], color="k", lw=1.0, ls=":", label="condiciones iniciales")

    if final_trajectory is not None:
        data = np.asarray(final_trajectory, dtype=float)
        if data.ndim == 2 and data.shape[1] >= 4 and len(data) > 0:
            half_t = 0.5 * float(data[-1, 0])
            tail = data[data[:, 0] >= half_t]
            tail = sample_rows(tail, max_final_points)
            ax.plot(tail[:, 1], tail[:, 2], tail[:, 3], color="crimson", lw=0.55, alpha=0.9, label="atractor final")

    if seed_effective is not None:
        seed = np.asarray(seed_effective, dtype=float)[:3]
        ax.scatter([seed[0]], [seed[1]], [seed[2]], s=42, color="#facc15", edgecolor="black", linewidth=0.55, label="semilla efectiva")

    if continuation_final is not None:
        final = np.asarray(continuation_final, dtype=float)[:3]
        ax.scatter([final[0]], [final[1]], [final[2]], s=34, color="black", label="estado final de continuacion")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "continuation")
    plt.close(fig)
    return str(path)


def plot_phase_space_with_reference_points(
    trajectory: np.ndarray,
    output_path: str | Path,
    *,
    seed_effective: Sequence[float] | None = None,
    continuation_final: Sequence[float] | None = None,
    max_points: int = 8000,
) -> str:
    """Plot a phase-space trajectory with standard seed/final markers."""

    data = sample_rows(np.asarray(trajectory, dtype=float), max_points)
    path = _output_path(output_path)
    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(data[:, 1], data[:, 2], data[:, 3], lw=0.35)
    if seed_effective is not None:
        seed = np.asarray(seed_effective, dtype=float)[:3]
        ax.scatter([seed[0]], [seed[1]], [seed[2]], s=42, color="#facc15", edgecolor="black", linewidth=0.55, label="semilla efectiva")
    if continuation_final is not None:
        final = np.asarray(continuation_final, dtype=float)[:3]
        ax.scatter([final[0]], [final[1]], [final[2]], s=34, color="black", label="estado final de continuacion")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if seed_effective is not None or continuation_final is not None:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "attractor")
    plt.close(fig)
    return str(path)


def plot_integer_hiddenness_controls(
    target_trajectory: np.ndarray,
    probes: Sequence[IntegerHiddennessProbe],
    output_path: str | Path,
    *,
    dims: Sequence[int] = (0, 1, 2),
    title: str | None = "Integer hiddenness controls",
    max_target_points: int = 2500,
    max_probe_points: int = 180,
) -> str:
    """Plot target attractor and sampled equilibrium-neighborhood probes."""

    path = _output_path(output_path)
    cols = [1 + int(dim) for dim in dims]
    target = sample_rows(np.asarray(target_trajectory, dtype=float), max_target_points)
    fig = plt.figure(figsize=(7.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target[:, cols[0]], target[:, cols[1]], target[:, cols[2]], color="#15803d", lw=0.7, alpha=0.9, label="target")
    for probe in probes:
        data = sample_rows(np.asarray(probe.trajectory, dtype=float), max_probe_points)
        color = "#dc2626" if probe.target_hit else "#2563eb"
        alpha = 0.82 if probe.target_hit else 0.35
        ax.plot(data[:, cols[0]], data[:, cols[1]], data[:, cols[2]], color=color, lw=0.55, alpha=alpha)
    ax.set_xlabel(f"x{dims[0]}")
    ax.set_ylabel(f"x{dims[1]}")
    ax.set_zlabel(f"x{dims[2]}")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "sphere_test")
    plt.close(fig)
    return str(path)


def plot_spectrum(
    spectrum: SpectrumResult,
    output_path: str | Path,
    *,
    title: str | None = None,
    x_units: str = "rad/s",
    omega_marker: float | None = None,
    marker_label: str | None = None,
) -> str:
    """Plot one reusable FFT/PSD spectrum."""

    path = _output_path(output_path)
    x = spectrum.frequency_rad_s if x_units in {"rad/s", "omega"} else spectrum.frequency_hz
    xlabel = "angular frequency" if x_units in {"rad/s", "omega"} else "frequency [Hz]"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, spectrum.values, lw=0.95, color="#111827", label=spectrum.method)
    if omega_marker is not None and x_units in {"rad/s", "omega"}:
        label = marker_label or rf"$\omega_0={float(omega_marker):.4f}$"
        ax.axvline(float(omega_marker), color="red", lw=1.0, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(spectrum.method)
    if title:
        ax.set_title(title)
    if omega_marker is not None and x_units in {"rad/s", "omega"}:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "fft")
    plt.close(fig)
    return str(path)


def plot_trajectory_spectra(
    trajectory: np.ndarray,
    output_dir: str | Path,
    *,
    method: str = "fft",
    components: Sequence[int] | None = None,
    prefix: str = "spectrum",
) -> list[str]:
    """Write one FFT/PSD figure per trajectory component."""

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    spectra = trajectory_component_spectra(trajectory, components=components, method=method)
    paths: list[str] = []
    for spec in spectra:
        paths.append(plot_spectrum(spec, outdir / f"{prefix}_{method}_component_{spec.component}.png"))
    return paths


def plot_lyapunov_convergence(result: LyapunovResult, output_path: str | Path) -> str:
    """Plot Lyapunov convergence curves for any dimension."""

    path = _output_path(output_path)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if result.convergence.size and result.times.size:
        for idx in range(result.convergence.shape[1]):
            ax.plot(result.times, result.convergence[:, idx], lw=0.95, label=f"LE{idx}")
    else:
        ax.scatter(np.arange(result.exponents.size), result.exponents, color="#111827", s=28)
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("exponent")
    ax.set_title("Lyapunov convergence")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "lyapunov")
    plt.close(fig)
    return str(path)
