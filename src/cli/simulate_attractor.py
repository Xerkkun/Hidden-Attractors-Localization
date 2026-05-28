"""Direct attractor simulation CLI — Ruta B (ADM Wu2023 / ABM / EFORK).

Usage
-----
    python -m src.cli.simulate_attractor --config PATH_TO_YAML

or invoked via run_workflow when workflow_mode: simulate_attractor_only.

This mode:
  - Reads system parameters and initial conditions from YAML.
  - Integrates directly from each initial condition using the chosen method.
  - Saves CSV files (time series, post-burn attractor) and PNG figures.
  - Computes basic diagnostic metrics.
  - NEVER calls seed generation, describing function, or continuation.
  - NEVER sets hidden_verified = true.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Matplotlib non-interactive backend ─────────────────────────────────────
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── YAML loader ─────────────────────────────────────────────────────────────
import yaml


# ---------------------------------------------------------------------------
# Integrator dispatch
# ---------------------------------------------------------------------------

def _run_adm(params: dict, x0: np.ndarray, q: float, h: float, N: int,
             divergence_norm: float):
    from ..integrators.adm_wu2023 import adm_wu2023_integrate
    return adm_wu2023_integrate(
        params=params, x0=x0, q=q, h=h, N=N, divergence_norm=divergence_norm
    )


def _run_abm(params: dict, x0: np.ndarray, q: float, h: float, N: int,
             divergence_norm: float):
    from ..integrators.abm import caputo_abm_integrate
    from ..systems.chua_arctan import ChuaArctanSystem

    system = ChuaArctanSystem(
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        m=params["m"], n=params["n"], q=q
    )
    t_final = N * h
    t_arr, x_arr, status = caputo_abm_integrate(
        rhs=system.evaluate_rhs,
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        divergence_norm=divergence_norm,
        use_c_backend=True,
    )
    info = {
        "integrator": "abm",
        "integrator_class": "caputo_full_memory",
        "scientific_label": (
            "ABM (Adams-Bashforth-Moulton) with full Caputo memory. "
            "Rigorous Caputo fractional integration."
        ),
        "hidden_verified": False,
        "q": q, "h": h, "N": N,
        "steps_completed": len(t_arr) - 1,
        "t_final_reached": float(t_arr[-1]),
        "caputo_memory": "full — Caputo kernel from t=0",
    }
    return t_arr, x_arr, status, info


def _run_efork(params: dict, x0: np.ndarray, q: float, h: float, N: int,
               divergence_norm: float):
    from ..integrators.efork import efork_integrate
    from ..systems.chua_arctan import ChuaArctanSystem

    system = ChuaArctanSystem(
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        m=params["m"], n=params["n"], q=q
    )
    t_final = N * h
    t_arr, x_arr, status = efork_integrate(
        system=system,
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        divergence_norm=divergence_norm,
        use_c_backend=True,
    )
    info = {
        "integrator": "efork",
        "integrator_class": "caputo_full_memory",
        "scientific_label": (
            "EFORK-3 with full Caputo memory (Ghoreishi et al. 2023). "
            "Rigorous Caputo fractional integration."
        ),
        "hidden_verified": False,
        "q": q, "h": h, "N": N,
        "steps_completed": len(t_arr) - 1,
        "t_final_reached": float(t_arr[-1]),
        "caputo_memory": "full — Caputo kernel from t=0",
    }
    return t_arr, x_arr, status, info


def _run_rk4(params: dict, x0: np.ndarray, q: float, h: float, N: int,
             divergence_norm: float):
    from ..integrators.rk4 import rk4_integrate
    from ..systems.chua_arctan import ChuaArctanSystem

    system = ChuaArctanSystem(
        alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
        m=params["m"], n=params["n"], q=1.0  # Force q=1.0 for integer RK4
    )
    t_arr, x_arr, status, info = rk4_integrate(
        rhs=system.evaluate_rhs,
        x0=x0,
        h=h,
        N=N,
        divergence_norm=divergence_norm,
    )
    return t_arr, x_arr, status, info


def _run_integer_general(integrator_name: str):
    def run_func(params: dict, x0: np.ndarray, q: float, h: float, N: int,
                 divergence_norm: float):
        from ..integrators.general import integrate_general
        from ..systems.chua_arctan import ChuaArctanSystem

        system = ChuaArctanSystem(
            alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"],
            m=params["m"], n=params["n"], q=1.0
        )
        t_final = N * h
        t_arr, x_arr, status = integrate_general(
            rhs=system.evaluate_rhs,
            x0=x0,
            q=1.0,
            h=h,
            t_final=t_final,
            integrator=integrator_name,
            divergence_norm=divergence_norm,
            system=system,
        )
        info = {
            "integrator": integrator_name,
            "integrator_class": "integer_order_solver",
            "scientific_label": f"Integer-order {integrator_name} solver via integrate_general.",
            "hidden_verified": False,
            "q": 1.0, "h": h, "N": N,
            "steps_completed": len(t_arr) - 1,
            "t_final_reached": float(t_arr[-1]),
            "caputo_memory": "none - integer dynamics",
        }
        return t_arr, x_arr, status, info
    return run_func


INTEGRATOR_DISPATCH = {
    "adm_wu2023": _run_adm,
    "abm": _run_abm,
    "efork": _run_efork,
    "efork3": _run_efork,
    "rk4": _run_rk4,
    "heun": _run_integer_general("heun"),
    "efork_q1": _run_integer_general("efork_q1"),
}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _compute_diagnostics(times: np.ndarray, states: np.ndarray,
                          t_burn: float, h: float) -> Dict[str, Any]:
    """Compute basic diagnostic metrics on the post-burn portion of a trajectory."""
    n_burn = max(0, int(math.ceil(t_burn / h)))
    if n_burn >= len(times):
        n_burn = 0  # cannot burn more than we have

    tail_t = times[n_burn:]
    tail_x = states[n_burn:]

    if len(tail_x) < 4:
        return {
            "warning": "trajectory_too_short_for_diagnostics",
            "n_burn_steps": n_burn,
            "n_tail_steps": len(tail_x),
        }

    x, y, z = tail_x[:, 0], tail_x[:, 1], tail_x[:, 2]
    norms = np.linalg.norm(tail_x, axis=1)

    diag: Dict[str, Any] = {
        "n_burn_steps": n_burn,
        "n_tail_steps": len(tail_x),
        "t_burn_actual": float(tail_t[0]),
        "t_final": float(tail_t[-1]),
        "has_nan": bool(np.any(~np.isfinite(tail_x))),
        "range_x": [float(x.min()), float(x.max())],
        "range_y": [float(y.min()), float(y.max())],
        "range_z": [float(z.min()), float(z.max())],
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
    }

    if diag["has_nan"]:
        diag["classification"] = "invalid"
        return diag

    # ── Dominant frequency (FFT on x) ─────────────────────────────────────
    try:
        n_fft = len(x)
        fft_amp = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(n_fft, d=h)
        # Skip DC (index 0)
        dom_idx = int(np.argmax(fft_amp[1:]) + 1)
        dominant_freq = float(freqs[dom_idx])
        peak_ratio = float(fft_amp[dom_idx] / (fft_amp[1:].mean() + 1e-15))
        diag["dominant_frequency"] = dominant_freq
        diag["peak_ratio"] = peak_ratio

        # ── Spectral entropy ─────────────────────────────────────────────────
        psd = fft_amp[1:] ** 2
        psd_norm = psd / (psd.sum() + 1e-30)
        ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-30)))
        max_ent = math.log(len(psd_norm))
        diag["spectral_entropy"] = ent
        diag["spectral_entropy_normalized"] = float(ent / max_ent) if max_ent > 0 else 0.0
    except Exception as exc:
        diag["fft_warning"] = str(exc)
        peak_ratio = 1.0

    # ── 0–1 test (approximate) ───────────────────────────────────────────
    try:
        diag["zero_one_test"] = _zero_one_test_approx(x)
    except Exception:
        diag["zero_one_test"] = None

    # ── Preliminary classification ────────────────────────────────────────
    norm_max = diag["max_norm"]
    if norm_max > 1e6:
        cls = "diverged"
    elif diag["has_nan"]:
        cls = "invalid"
    elif diag.get("zero_one_test") is not None and diag["zero_one_test"] > 0.8:
        cls = "chaotic_like"
    elif diag.get("peak_ratio", 1.0) > 20.0 and diag.get("spectral_entropy_normalized", 1.0) < 0.3:
        cls = "periodic_like"
    elif diag.get("spectral_entropy_normalized", 0.0) > 0.6:
        cls = "chaotic_like"
    else:
        cls = "unclassified"

    diag["classification"] = cls
    return diag


def _zero_one_test_approx(x: np.ndarray, n_samples: int = 100) -> float:
    """Approximate 0-1 test for chaos (Gottwald & Melbourne 2004).

    Returns a value close to 0 for regular dynamics, close to 1 for chaotic.
    This is a simplified approximation using the mean-square displacement growth.
    """
    rng = np.random.default_rng(42)
    N = len(x)
    if N < 100:
        return float("nan")

    # Use at most n_samples random c values
    c_vals = rng.uniform(math.pi / 5.0, 4.0 * math.pi / 5.0, size=min(n_samples, 20))
    K_vals = []

    for c in c_vals:
        p = np.cumsum(x * np.cos(np.arange(N) * c))
        q_arr = np.cumsum(x * np.sin(np.arange(N) * c))
        n_max = N // 10
        if n_max < 5:
            continue
        ns = np.arange(1, n_max + 1)
        # Mean-square displacement
        M = np.array([
            np.mean((p[n:] - p[:-n]) ** 2 + (q_arr[n:] - q_arr[:-n]) ** 2)
            for n in ns
        ])
        # Regression slope against n
        if len(ns) < 3:
            continue
        slope = np.polyfit(np.log(ns + 1e-10), np.log(M + 1e-10), 1)[0]
        K_vals.append(float(np.clip(slope, 0.0, 2.0) / 1.0))  # normalised ~[0,1]

    if not K_vals:
        return float("nan")
    return float(np.median(K_vals))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(path: str, times: np.ndarray, states: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        for i in range(len(times)):
            writer.writerow([times[i], states[i, 0], states[i, 1], states[i, 2]])


def _plot_attractor(times: np.ndarray, states: np.ndarray,
                    t_burn: float, h: float,
                    label: str, output_dir: str, prefix: str,
                    system_id: str = "chua_fractional_arctan") -> None:
    n_burn = max(0, int(math.ceil(t_burn / h)))
    if n_burn >= len(times):
        n_burn = 0

    x = states[n_burn:, 0]
    y = states[n_burn:, 1]
    z = states[n_burn:, 2]

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── 3D ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="#10b981", linewidth=0.5, alpha=0.85)
    ax.set_title(f"ADM Attractor — {label}\n{system_id}", fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{prefix}_3d.png"), dpi=150)
    plt.close(fig)

    # ── 2D projections ───────────────────────────────────────────────────────
    for pname, u, v, xl, yl in [
        ("xy", x, y, "x", "y"),
        ("xz", x, z, "x", "z"),
        ("yz", y, z, "y", "z"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=150)
        ax2.plot(u, v, color="#10b981", linewidth=0.4, alpha=0.8)
        ax2.set_title(f"Projection {pname.upper()} — {label}", fontsize=10)
        ax2.set_xlabel(xl); ax2.set_ylabel(yl)
        ax2.grid(True, linestyle="--", linewidth=0.4, color="#cbd5e1")
        plt.tight_layout()
        fig2.savefig(os.path.join(fig_dir, f"{prefix}_{pname}.png"), dpi=150)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# Core workflow: simulate one IC with one integrator
# ---------------------------------------------------------------------------

def simulate_one(
    ic_label: str,
    x0: np.ndarray,
    integrator_name: str,
    params: Dict[str, Any],
    q: float,
    h: float,
    N: int,
    t_burn: float,
    divergence_norm: float,
    output_dir: str,
    save_timeseries: bool = True,
    save_attractor: bool = True,
    plot_3d: bool = True,
    plot_projections: bool = True,
    diagnostics_enabled: bool = True,
    system_id: str = "chua_fractional_arctan",
) -> Dict[str, Any]:
    """Integrate from x0 with the specified integrator and save all outputs."""

    fn = INTEGRATOR_DISPATCH.get(integrator_name)
    if fn is None:
        raise ValueError(
            f"Unknown integrator '{integrator_name}'. "
            f"Valid: {list(INTEGRATOR_DISPATCH.keys())}"
        )

    t0_wall = time.perf_counter()
    times, states, status, info = fn(params, x0, q, h, N, divergence_norm)
    elapsed = time.perf_counter() - t0_wall

    result: Dict[str, Any] = {
        "ic_label": ic_label,
        "x0": x0.tolist(),
        "integrator": integrator_name,
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "steps_completed": info.get("steps_completed", len(times) - 1),
        "t_final_reached": info.get("t_final_reached", float(times[-1])),
        "integrator_class": info.get("integrator_class", "unknown"),
        "scientific_label": info.get("scientific_label", ""),
        "hidden_verified": False,
    }

    os.makedirs(output_dir, exist_ok=True)
    prefix = ic_label

    # ── Time series CSV ──────────────────────────────────────────────────────
    if save_timeseries:
        ts_path = os.path.join(output_dir, f"{prefix}_timeseries.csv")
        _save_csv(ts_path, times, states)
        result["timeseries_csv"] = ts_path

    # ── Attractor CSV (post-burn) ────────────────────────────────────────────
    if save_attractor:
        n_burn = max(0, int(math.ceil(t_burn / h)))
        if n_burn < len(times):
            att_path = os.path.join(output_dir, f"{prefix}_attractor.csv")
            _save_csv(att_path, times[n_burn:], states[n_burn:])
            result["attractor_csv"] = att_path

    # ── Figures ──────────────────────────────────────────────────────────────
    if (plot_3d or plot_projections) and status in ("ok", "diverged"):
        try:
            _plot_attractor(
                times, states, t_burn, h,
                label=f"{ic_label} ({integrator_name})",
                output_dir=output_dir,
                prefix=prefix,
                system_id=system_id,
            )
            result["figures_dir"] = os.path.join(output_dir, "figures")
        except Exception as exc:
            result["plot_warning"] = str(exc)

    # ── Diagnostics ─────────────────────────────────────────────────────────
    if diagnostics_enabled and len(times) > 1:
        diag = _compute_diagnostics(times, states, t_burn, h)
        result["diagnostics"] = diag
        result["classification"] = diag.get("classification", "unknown")

    print(
        f"  [{ic_label}] integrator={integrator_name}  status={status}"
        f"  steps={result['steps_completed']}  t={result['t_final_reached']:.1f}"
        f"  class={result.get('classification', '?')}  [{elapsed:.1f}s]"
    )
    return result


# ---------------------------------------------------------------------------
# Main workflow runner
# ---------------------------------------------------------------------------

def run_simulate_attractor_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the simulate_attractor_only workflow.

    Parameters
    ----------
    config : Loaded and validated configuration dictionary.

    Returns
    -------
    summary : dict with all results and metadata.
    """
    system_id      = config.get("system_id", "chua_fractional_arctan")
    integrator     = config.get("integrator", "adm_wu2023")
    q              = float(config.get("q", 0.99))
    h              = float(config.get("h", 0.01))
    N              = int(config.get("N", int(math.ceil(float(config.get("t_final", 100.0)) / h))))
    t_burn         = float(config.get("t_burn", 50.0))
    div_norm       = float(config.get("divergence_norm", 120.0))
    output_dir     = config.get("output_dir", os.path.join("outputs", system_id, "adm"))
    sci_label      = config.get("scientific_label", "ADM simulation — Ruta B")

    save_ts   = bool(config.get("save_timeseries", True))
    save_att  = bool(config.get("save_attractor", True))
    plot3d    = bool(config.get("plot_3d", True))
    plot_proj = bool(config.get("plot_projections", True))
    diag_en   = bool(config.get("diagnostics", True))

    # ── System parameters ────────────────────────────────────────────────────
    params = {
        "alpha": float(config.get("alpha", 8.4562)),
        "beta":  float(config.get("beta",  12.0732)),
        "gamma": float(config.get("gamma", 0.0052)),
        "m":     float(config.get("m",     0.4)),
        "n":     float(config.get("n",     -1.1585)),
    }

    # ── Initial conditions ───────────────────────────────────────────────────
    ics_raw = config.get("initial_conditions", {})
    if not ics_raw:
        raise ValueError(
            "No initial_conditions found in config. "
            "Provide at least one entry under 'initial_conditions:'."
        )
    initial_conditions: Dict[str, np.ndarray] = {}
    for lbl, val in ics_raw.items():
        initial_conditions[lbl] = np.asarray(val, dtype=float)

    # ── Print header ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(" simulate_attractor_only - Ruta B")
    print("=" * 72)
    print(f"  workflow_mode    = simulate_attractor_only")
    print(f"  system           = {system_id}")
    print(f"  integrator       = {integrator}")
    print(f"  Seed search      = DISABLED")
    print(f"  Continuation     = DISABLED")
    print(f"  q                = {q}")
    print(f"  h                = {h}")
    print(f"  N                = {N}  (t_final = {N * h:.1f})")
    print(f"  t_burn           = {t_burn}")
    print(f"  divergence_norm  = {div_norm}")
    print(f"  output_dir       = {output_dir}")
    print(f"  Initial cond.    = {list(initial_conditions.keys())}")
    print(f"  Scientific label = {sci_label}")
    print("=" * 72)

    os.makedirs(output_dir, exist_ok=True)

    # ── Integrator class label ────────────────────────────────────────────────
    if integrator == "adm_wu2023":
        integrator_class = "adm_local_reproduction"
        memory_note = "Local ADM step - no Caputo history"
    elif integrator in {"rk4", "heun", "efork_q1"}:
        integrator_class = "integer_order_solver"
        memory_note = "none - integer dynamics"
    else:
        integrator_class = "caputo_full_memory"
        memory_note = "Full Caputo memory (kernel sum from t=0)"

    all_results: List[Dict[str, Any]] = []

    for ic_label, x0 in initial_conditions.items():
        ic_out_dir = output_dir  # all ICs in same folder (prefix distinguishes them)
        res = simulate_one(
            ic_label=ic_label,
            x0=x0,
            integrator_name=integrator,
            params=params,
            q=q, h=h, N=N,
            t_burn=t_burn,
            divergence_norm=div_norm,
            output_dir=ic_out_dir,
            save_timeseries=save_ts,
            save_attractor=save_att,
            plot_3d=plot3d,
            plot_projections=plot_proj,
            diagnostics_enabled=diag_en,
            system_id=system_id,
        )
        all_results.append(res)

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "workflow_mode": "simulate_attractor_only",
        "system_id": system_id,
        "integrator": integrator,
        "integrator_class": integrator_class,
        "memory_note": memory_note,
        "scientific_label": sci_label,
        "hidden_verified": False,
        "q": q,
        "h": h,
        "N": N,
        "t_burn": t_burn,
        "divergence_norm": div_norm,
        "params": params,
        "initial_conditions": {k: v.tolist() for k, v in initial_conditions.items()},
        "results": all_results,
        "n_completed": sum(1 for r in all_results if r["status"] == "ok"),
        "n_diverged": sum(1 for r in all_results if r["status"] == "diverged"),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved -> {summary_path}")

    # ── Final table ──────────────────────────────────────────────────────────
    print()
    print(f"{'IC label':<20} {'status':<12} {'class':<18} {'t_reached':<10}")
    print("-" * 62)
    for r in all_results:
        print(
            f"  {r['ic_label']:<18} {r['status']:<12}"
            f" {r.get('classification', '?'):<18}"
            f" {r.get('t_final_reached', 0):<10.1f}"
        )
    print("=" * 72)
    print("  NOTE: hidden_verified = false")
    print(f"  Integrator class: {integrator_class}")
    print(f"  {memory_note}")
    print("=" * 72)

    return summary


# ---------------------------------------------------------------------------
# YAML loader (minimal, no contracts from Ruta A)
# ---------------------------------------------------------------------------

def load_attractor_config(path: str) -> Dict[str, Any]:
    """Load YAML config for simulate_attractor_only mode."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Validate workflow_mode
    wm = cfg.get("workflow_mode", "simulate_attractor_only")
    if wm != "simulate_attractor_only":
        raise ValueError(
            f"This CLI expects workflow_mode=simulate_attractor_only, got '{wm}'. "
            "Use run_workflow for other modes."
        )

    # Set default output_dir
    if not cfg.get("output_dir"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        sid = cfg.get("system_id", "chua_fractional_arctan")
        cfg["output_dir"] = os.path.join("outputs", sid, f"adm_attractor_{ts}")

    return cfg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="simulate_attractor — direct integration from explicit ICs (Ruta B)."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory.")
    parser.add_argument("--integrator", type=str, default=None,
                        choices=["adm_wu2023", "abm", "efork", "efork3"],
                        help="Override integrator.")
    args = parser.parse_args(argv)

    config = load_attractor_config(args.config)

    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.integrator:
        config["integrator"] = args.integrator

    run_simulate_attractor_workflow(config)


if __name__ == "__main__":
    main()
