#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pipeline for the integer-order Chua hidden-attractor procedure.

The seed construction mirrors verifica_chua_entero.m:
    numeric Lure form -> W(iw) roots -> harmonic amplitude -> P0/A/S
    -> X0 = S [a0, 0, 0]^T.

Then Python performs numerical continuation in epsilon from the harmonic
linearization to the original integer-order Chua system. The expensive
hidden-attractor verification and basin classification are delegated to the C
backends already used by the fractional pipeline. Those backends are invoked
with q=1.0 for the integer run. The full integer pipeline uses the same
EFORK-3 operational process at q=1.0: Python stages use the explicit q=1
EFORK step, and the C verification/basin/Lyapunov helpers use the shared
EFORK-style backend interface at q=1.0.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import quad
from scipy.optimize import brentq

import run_hidden_verify_frac_hybrid as hv


ROOT = Path(__file__).resolve().parent
PROJECT_SLUG = "hidden_attractors_chua_integer"

REGISTERED_DLL_DIRS: set[str] = set()
DLL_SEARCH_HANDLES: list[Any] = []
BASIN_LIBRARY_CACHE = None

BASIN_CLASS_COLORS = {
    0: "#111827",
    1: "#7c3aed",
    2: "#ffd43b",
    3: "#4b5563",
    4: "#38bdf8",
}
BASIN_CLASS_LABELS = {
    0: "equilibrium",
    1: "hidden +",
    2: "hidden -",
    3: "divergent",
    4: "unknown",
}
BASIN_SEED_COLOR = "#ffd35a"
BIFURCATION_POS_COLOR = "#ff3b1f"
BIFURCATION_NEG_COLOR = "#d27bff"
LINEARIZED_COLOR = "#9467bd"
ORIGINAL_COLOR = "#d62728"
NYQUIST_W_COLOR = "#0047ff"
NYQUIST_DF_COLOR = "#ff4a1a"
WHITE_BG = "#ffffff"


def basin_cmap_norm() -> Tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap([BASIN_CLASS_COLORS[i] for i in range(5)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    return cmap, norm


def log(message: str) -> None:
    print(f"[chua-integer] {message}", flush=True)


def _dir_accepts_writes(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def resolve_runtime_root() -> Path:
    env_dir = os.environ.get("HIDDEN_ATTRACTORS_OUTPUT_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(ROOT / "chua_integer_outputs")
    candidates.append(Path(tempfile.gettempdir()) / PROJECT_SLUG)

    for candidate in candidates:
        if _dir_accepts_writes(candidate):
            return candidate
    raise PermissionError("Define HIDDEN_ATTRACTORS_OUTPUT_DIR to a writable directory.")


RUNTIME_ROOT = resolve_runtime_root()
OUTDIR = RUNTIME_ROOT / "final_pdf_figs"
HIDDEN_VERIFY_DIR = RUNTIME_ROOT / "hidden_verify"
NATIVE_DIR = RUNTIME_ROOT / "native"
OUTDIR.mkdir(parents=True, exist_ok=True)
HIDDEN_VERIFY_DIR.mkdir(parents=True, exist_ok=True)
NATIVE_DIR.mkdir(parents=True, exist_ok=True)


CONFIG: Dict[str, Any] = {
    "params": {
        "alpha_chua": 8.4562,
        "beta": 12.0732,
        "gamma_chua": 0.0052,
        "m0": -0.1768,
        "m1": -1.1468,
    },
    "branch_index": 0,
    "lure": {
        "tol_jacobian": 1e-7,
        "wmin": 1e-5,
        "wmax": 50.0,
        "nscan": 40000,
        "amin": 1.0 + 1e-8,
        "amax": 50.0,
        "ascan": 20000,
    },
    "continuation": {
        "eps_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "t_transient": 80.0,
        "t_keep": 80.0,
        "dt_output": 0.01,
        "rtol": 1e-9,
        "atol": 1e-11,
    },
    "final_attractor": {
        "t_burn": 120.0,
        "t_keep": 180.0,
        "dt_output": 0.01,
    },
    "comparison": {
        "t_final": 60.0,
        "dt_output": 0.01,
        "max_plot_points": 900,
    },
    "verify_hidden": {
        "h": 0.01,
        "Lm": 1.0,
        "TMAX_REF": 500.0,
        "TMAX_TEST": 500.0,
        "TBURN_REF": 120.0,
        "TBURN_TEST": 120.0,
        "R_DIV": 120.0,
        "EPS_EQ": 0.03,
        "CAP_WIN": 200,
        "SEC_TOL": 0.12,
        "MIN_SEC_MATCH": 20,
        "TEST_MAX_SEC": 100,
        "HIT_FRAC_REQ": 0.70,
        "RADII": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "NSAMPLES_PER_RADIUS": 24,
        "random_seed": 123456789,
    },
    "hidden_illustration": {
        "enabled": True,
        "max_probe_trajectories": 200,
        "max_points_per_probe": 240,
        "max_attractor_points": 3600,
        "t_total": 80.0,
        "dt_output": 0.02,
        "zoom_pad_fraction": 0.12,
        "random_seed": 246813579,
    },
    "basin": {
        "source": "chua_basin_lib.c",
        "openmp": True,
        "nx": 300,
        "ny": 300,
        "xmin": -8.0,
        "xmax": 8.0,
        "ymin": -8.0,
        "ymax": 8.0,
        "z0": 0.0,
        "q": 1.0,
        "h": 0.01,
        "Lm": 1.0,
        "TMAX": 180.0,
        "TBURN": 45.0,
        "R_DIV": 80.0,
        "R_BOUND": 30.0,
        "EPS_EQ": 0.03,
        "CAP_WIN": 150,
        "MEAN_X_GAP": 0.08,
    },
    "basin_planes": {
        "enabled": True,
        "nx": 300,
        "ny": 300,
        "nz": 300,
        "xlim": (-8.0, 8.0),
        "ylim": (-8.0, 8.0),
        "zlim": (-12.0, 12.0),
        "q": 1.0,
        "h": 0.01,
        "Lm": 1.0,
        "TMAX": 180.0,
        "TBURN": 45.0,
        "R_DIV": 80.0,
        "R_BOUND": 30.0,
        "EPS_EQ": 0.03,
        "CAP_WIN": 150,
        "MEAN_X_GAP": 0.08,
    },
    "basin_3d": {
        "enabled": True,
        "nx": 56,
        "ny": 56,
        "nz": 56,
        "xlim": (-8.0, 8.0),
        "ylim": (-8.0, 8.0),
        "zlim": (-12.0, 12.0),
        "q": 1.0,
        "h": 0.01,
        "Lm": 1.0,
        "TMAX": 120.0,
        "TBURN": 35.0,
        "R_DIV": 80.0,
        "R_BOUND": 30.0,
        "EPS_EQ": 0.03,
        "CAP_WIN": 120,
        "MEAN_X_GAP": 0.08,
        "max_points_per_class": 18000,
    },
    "bifurcation": {
        "enabled": True,
        "alpha_values": np.linspace(8.2, 8.7, 261).tolist(),
        "beta_values": np.linspace(11.5, 12.5, 261).tolist(),
        "t_total": 200.0,
        "t_burn": 100.0,
        "dt_output": 0.02,
        "div_threshold": 120.0,
        "max_peaks": 500,
        "progress_every": 5,
    },
    "spectral": {
        "enabled": True,
    },
    "psd": {
        "enabled": False,
        "component_names": ["x", "y", "z"],
        "primary_component_index": 0,
        "welch_window": "hann",
        "welch_scaling": "density",
        "nperseg_max": 4096,
        "noverlap_fraction": 0.5,
        "ignore_zero_bin": True,
    },
    "fft": {
        "component_names": ["x", "y", "z"],
        "primary_component_index": 0,
        "window": "hann",
        "ignore_zero_bin": True,
        "top_n": 5,
        "zoom_half_width_factor": 1.5,
    },
    "lyapunov": {
        "enabled": False,
        "strict": False,
        "source_c": ROOT / "chua_frac_lyapunov_efork_benettin.c",
        "exe": NATIVE_DIR / ("chua_integer_lyapunov_benettin.exe" if os.name == "nt" else "chua_integer_lyapunov_benettin"),
        "h": 0.01,
        "Lm": 1.0,
        "t_burn": 80.0,
        "n_blocks": 400,
        "t_block": 0.5,
    },
    "outputs": {
        "seed_json": RUNTIME_ROOT / "chua_integer_seed_summary.json",
        "cont_json": RUNTIME_ROOT / "chua_integer_continuation_summary.json",
        "summary_json": RUNTIME_ROOT / "chua_integer_pipeline_summary.json",
        "dist_csv": RUNTIME_ROOT / "seed_equilibrium_distances_integer.csv",
        "nyquist_pdf": OUTDIR / "fig01_nyquist_df.pdf",
        "nyquist_zoom_pdf": OUTDIR / "fig01b_nyquist_zoom_x.pdf",
        "cont_progress_x_pdf": OUTDIR / "fig02a_continuation_x.pdf",
        "cont_progress_y_pdf": OUTDIR / "fig02b_continuation_y.pdf",
        "cont_progress_z_pdf": OUTDIR / "fig02c_continuation_z.pdf",
        "continuation_story_pdf": OUTDIR / "fig02d_continuation_story.pdf",
        "final_attr_pdf": OUTDIR / "fig03_final_attractor.pdf",
        "final_attr_xy_pdf": OUTDIR / "fig03a_final_attractor_xy.pdf",
        "final_attr_xz_pdf": OUTDIR / "fig03b_final_attractor_xz.pdf",
        "final_attr_yz_pdf": OUTDIR / "fig03c_final_attractor_yz.pdf",
        "linear_vs_original_x_pdf": OUTDIR / "fig03d_linear_vs_original_x.pdf",
        "linear_vs_original_y_pdf": OUTDIR / "fig03e_linear_vs_original_y.pdf",
        "linear_vs_original_z_pdf": OUTDIR / "fig03f_linear_vs_original_z.pdf",
        "linear_vs_original_3d_pdf": OUTDIR / "fig03g_linear_vs_original_3d.pdf",
        "ref_section_pdf": OUTDIR / "fig04_reference_section.pdf",
        "probe_summary_pdf": OUTDIR / "fig05_probe_summary.pdf",
        "hidden_illustration_overview_pdf": OUTDIR / "fig05b_hiddenness_overview.pdf",
        "hidden_illustration_zoom_pdf": OUTDIR / "fig05c_hiddenness_zoom.pdf",
        "hidden_illustration_json": RUNTIME_ROOT / "chua_integer_hiddenness_illustration.json",
        "basin_pdf": OUTDIR / "fig06b_basin_overlay_zfinal.pdf",
        "basin_z0_pdf": OUTDIR / "fig06a_basin_overlay_z0.pdf",
        "basin_zfinal_pdf": OUTDIR / "fig06b_basin_overlay_zfinal.pdf",
        "basin_xy_pdf": OUTDIR / "fig10a_basin_xy_final_state.pdf",
        "basin_xz_pdf": OUTDIR / "fig10b_basin_xz_final_state.pdf",
        "basin_yz_pdf": OUTDIR / "fig10c_basin_yz_final_state.pdf",
        "basin_3d_pdf": OUTDIR / "fig12_basin_3d_volume.pdf",
        "bif_alpha_pdf": OUTDIR / "fig08_bifurcation_alpha.pdf",
        "bif_beta_pdf": OUTDIR / "fig09_bifurcation_beta.pdf",
        "bif_json": RUNTIME_ROOT / "chua_integer_bifurcation_summary.json",
        "fft_x_pdf": OUTDIR / "fig11a_fft_x.pdf",
        "fft_y_pdf": OUTDIR / "fig11b_fft_y.pdf",
        "fft_z_pdf": OUTDIR / "fig11c_fft_z.pdf",
        "psd_x_pdf": OUTDIR / "fig11d_psd_x.pdf",
        "psd_y_pdf": OUTDIR / "fig11e_psd_y.pdf",
        "psd_z_pdf": OUTDIR / "fig11f_psd_z.pdf",
        "spectral_json": RUNTIME_ROOT / "chua_integer_spectral_summary.json",
        "lyapunov_csv": RUNTIME_ROOT / "chua_integer_le_convergence.csv",
        "lyapunov_json": RUNTIME_ROOT / "chua_integer_lyapunov_summary.json",
        "lyapunov_pdf": OUTDIR / "fig13_lyapunov_convergence.pdf",
    },
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "si", "s"}


def synchronize_integer_runtime_contract(cfg: Dict[str, Any]) -> None:
    """Record the effective numerical contract shared by the integer pipeline.

    The integer case uses the same EFORK-3 operational method at q=1.0 across
    the pipeline. Python stages use the explicit q=1 EFORK step; C verification,
    basin and Lyapunov backends receive q=1.0 through the same shared backend
    interface used by the fractional pipeline.
    """
    cfg["runtime_contract"] = {
        "order": "integer",
        "q_passed_to_c_backends": 1.0,
        "numerical_method": "EFORK-3 explicit operational integrator at q=1.0",
        "python_solver": "internal EFORK-3 q=1.0 step",
        "continuation": {
            "eps_values": [float(v) for v in cfg["continuation"]["eps_values"]],
            "t_transient": float(cfg["continuation"]["t_transient"]),
            "t_keep": float(cfg["continuation"]["t_keep"]),
            "h": float(cfg["continuation"]["dt_output"]),
            "dt_output": float(cfg["continuation"]["dt_output"]),
            "rtol_atol_note": "not used by fixed-step EFORK q=1; kept only for backward-compatible config fields",
        },
        "final_attractor": {
            "t_burn": float(cfg["final_attractor"]["t_burn"]),
            "t_keep": float(cfg["final_attractor"]["t_keep"]),
            "h": float(cfg["final_attractor"]["dt_output"]),
            "dt_output": float(cfg["final_attractor"]["dt_output"]),
        },
        "comparison": {
            "t_final": float(cfg["comparison"]["t_final"]),
            "h": float(cfg["comparison"]["dt_output"]),
            "dt_output": float(cfg["comparison"]["dt_output"]),
            "max_plot_points": int(cfg["comparison"].get("max_plot_points", 900)),
        },
        "hidden_verification": {
            "backend": "chua_hidden_backend.c EFORK-style integrator at q=1.0",
            "h": float(cfg["verify_hidden"]["h"]),
            "Lm": float(cfg["verify_hidden"]["Lm"]),
            "TMAX_REF": float(cfg["verify_hidden"]["TMAX_REF"]),
            "TMAX_TEST": float(cfg["verify_hidden"]["TMAX_TEST"]),
            "TBURN_REF": float(cfg["verify_hidden"]["TBURN_REF"]),
            "TBURN_TEST": float(cfg["verify_hidden"]["TBURN_TEST"]),
        },
        "hidden_illustration": {
            "enabled": bool(cfg["hidden_illustration"].get("enabled", True)),
            "max_probe_trajectories": int(cfg["hidden_illustration"]["max_probe_trajectories"]),
            "t_total": float(cfg["hidden_illustration"]["t_total"]),
            "dt_output": float(cfg["hidden_illustration"]["dt_output"]),
            "source": "actual hidden_target_check CSV from hidden verification",
        },
        "basin": {
            "backend": "chua_basin_lib.c EFORK-style classifier at q=1.0",
            "z_cuts": [0.0, "final_state"],
            "grid": [int(cfg["basin"]["nx"]), int(cfg["basin"]["ny"])],
            "TMAX": float(cfg["basin"]["TMAX"]),
            "TBURN": float(cfg["basin"]["TBURN"]),
        },
        "basin_3d": {
            "enabled": bool(cfg["basin_3d"].get("enabled", False)),
            "grid": [int(cfg["basin_3d"]["nx"]), int(cfg["basin_3d"]["ny"]), int(cfg["basin_3d"]["nz"])],
        },
        "bifurcation": {
            "alpha_count": len(cfg["bifurcation"]["alpha_values"]),
            "beta_count": len(cfg["bifurcation"]["beta_values"]),
            "h": float(cfg["bifurcation"]["dt_output"]),
            "t_total": float(cfg["bifurcation"]["t_total"]),
            "t_burn": float(cfg["bifurcation"]["t_burn"]),
            "max_peaks": int(cfg["bifurcation"]["max_peaks"]),
        },
        "lyapunov": {
            "enabled": bool(cfg["lyapunov"].get("enabled", False)),
            "backend_c": str(cfg["lyapunov"]["source_c"]),
            "q": 1.0,
            "h": float(cfg["lyapunov"]["h"]),
            "Lm": float(cfg["lyapunov"]["Lm"]),
            "t_burn": float(cfg["lyapunov"]["t_burn"]),
            "n_blocks": int(cfg["lyapunov"]["n_blocks"]),
            "t_block": float(cfg["lyapunov"]["t_block"]),
        },
        "notes": [
            "All figures use white backgrounds.",
            "The integer pipeline uses one numerical process: EFORK-3 at q=1.0 for Python trajectories and C hiddenness, basins and Lyapunov.",
            "Basin cuts are kept separately for z=0 and z=final_state; 3D basin volume can be toggled with HIDDEN_ATTRACTORS_BASIN_3D.",
            "FFT is the primary spectral figure; Welch PSD is optional with HIDDEN_ATTRACTORS_PSD=1.",
            "Lyapunov exponents are optional and use the C Benettin backend with q=1.0.",
        ],
    }


def configure_runtime_profile(cfg: Dict[str, Any]) -> None:
    run_mode = os.environ.get("HIDDEN_ATTRACTORS_RUN_MODE", "balanced").strip().lower()
    if run_mode in {"quick", "test"}:
        run_mode = "fast"
    if run_mode not in {"fast", "balanced", "full"}:
        raise ValueError("HIDDEN_ATTRACTORS_RUN_MODE must be fast, balanced, or full.")
    cfg["run_mode"] = run_mode
    cfg["spectral"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_SPECTRAL", True)
    cfg["psd"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_PSD", False)
    cfg["basin_planes"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_BASIN_PLANES", True)
    cfg["basin_3d"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_BASIN_3D", bool(cfg["basin_3d"].get("enabled", True)))
    cfg["hidden_illustration"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_HIDDEN_ILLUSTRATION", True)
    cfg["lyapunov"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_LYAPUNOV", False)
    cfg["lyapunov"]["strict"] = _env_flag("HIDDEN_ATTRACTORS_LYAPUNOV_STRICT", False)

    if run_mode == "full":
        return
    if run_mode == "balanced":
        cfg["continuation"].update({
            "eps_values": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "t_transient": 30.0,
            "t_keep": 30.0,
        })
        cfg["final_attractor"].update({"t_burn": 60.0, "t_keep": 90.0})
        cfg["comparison"].update({"t_final": 35.0, "max_plot_points": 700})
        cfg["verify_hidden"].update({
            "TMAX_REF": 320.0,
            "TMAX_TEST": 320.0,
            "TBURN_REF": 80.0,
            "TBURN_TEST": 80.0,
            "NSAMPLES_PER_RADIUS": 24,
        })
        cfg["basin"].update({"nx": 180, "ny": 180, "TMAX": 120.0, "TBURN": 30.0})
        cfg["basin_planes"].update({"nx": 160, "ny": 160, "nz": 160, "TMAX": 120.0, "TBURN": 30.0})
        cfg["basin_3d"].update({"nx": 34, "ny": 34, "nz": 34, "TMAX": 75.0, "TBURN": 20.0, "CAP_WIN": 90, "max_points_per_class": 9000})
        cfg["hidden_illustration"].update({"t_total": 60.0, "max_probe_trajectories": 160, "max_points_per_probe": 220})
        cfg["bifurcation"].update({
            "alpha_values": np.linspace(8.2, 8.7, 221).tolist(),
            "beta_values": np.linspace(11.5, 12.5, 221).tolist(),
            "t_total": 140.0,
            "t_burn": 70.0,
            "div_threshold": 120.0,
            "max_peaks": 500,
            "progress_every": 5,
        })
        cfg["lyapunov"].update({"t_burn": 60.0, "n_blocks": 300, "t_block": 0.5})
        return

    cfg["lure"].update({"nscan": 10000, "ascan": 6000})
    cfg["continuation"].update({
        "eps_values": [0.25, 0.5, 0.75, 1.0],
        "t_transient": 8.0,
        "t_keep": 8.0,
        "dt_output": 0.02,
    })
    cfg["final_attractor"].update({"t_burn": 8.0, "t_keep": 12.0, "dt_output": 0.02})
    cfg["comparison"].update({"t_final": 12.0, "dt_output": 0.02, "max_plot_points": 450})
    cfg["verify_hidden"].update({
        "TMAX_REF": 100.0,
        "TMAX_TEST": 100.0,
        "TBURN_REF": 25.0,
        "TBURN_TEST": 25.0,
        "TEST_MAX_SEC": 35,
        "RADII": [1e-4, 1e-3, 1e-2],
        "NSAMPLES_PER_RADIUS": 8,
    })
    cfg["basin"].update({"nx": 70, "ny": 70, "TMAX": 45.0, "TBURN": 12.0, "CAP_WIN": 80})
    cfg["basin_planes"].update({"nx": 40, "ny": 40, "nz": 40, "TMAX": 35.0, "TBURN": 10.0, "CAP_WIN": 70})
    cfg["basin_3d"].update({"nx": 18, "ny": 18, "nz": 18, "TMAX": 24.0, "TBURN": 7.0, "CAP_WIN": 50, "max_points_per_class": 4000})
    cfg["hidden_illustration"].update({"t_total": 24.0, "max_probe_trajectories": 60, "max_points_per_probe": 140, "max_attractor_points": 1600})
    cfg["bifurcation"].update({
        "alpha_values": np.linspace(8.25, 8.65, 5).tolist(),
        "beta_values": np.linspace(11.7, 12.4, 5).tolist(),
        "t_total": 24.0,
        "t_burn": 8.0,
        "dt_output": 0.02,
        "div_threshold": 120.0,
        "max_peaks": 20,
        "progress_every": 1,
    })
    cfg["lyapunov"].update({"t_burn": 8.0, "n_blocks": 40, "t_block": 0.25})


configure_runtime_profile(CONFIG)
synchronize_integer_runtime_contract(CONFIG)


def chua_model(X: np.ndarray, psi_value: float, params: Dict[str, float]) -> np.ndarray:
    x, y, z = np.asarray(X, dtype=float)
    return np.array([
        params["alpha_chua"] * (y - x - params["m1"] * x - psi_value),
        x - y + z,
        -(params["beta"] * y + params["gamma_chua"] * z),
    ], dtype=float)


def sat(u: float | np.ndarray) -> float | np.ndarray:
    return np.clip(u, -1.0, 1.0)


def psi_pwl(sigma: float | np.ndarray, params: Dict[str, float]) -> float | np.ndarray:
    return (params["m0"] - params["m1"]) * sat(sigma)


def numeric_lure_matrices(params: Dict[str, float], eps_j: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X0 = np.zeros(3, dtype=float)
    P = np.zeros((3, 3), dtype=float)
    for j in range(3):
        ej = np.zeros(3, dtype=float)
        ej[j] = 1.0
        fp = chua_model(X0 + eps_j * ej, 0.0, params)
        fm = chua_model(X0 - eps_j * ej, 0.0, params)
        P[:, j] = (fp - fm) / (2.0 * eps_j)
    qp = chua_model(X0, eps_j, params)
    qm = chua_model(X0, -eps_j, params)
    qvec = (qp - qm) / (2.0 * eps_j)
    r = np.array([1.0, 0.0, 0.0], dtype=float)
    return P, qvec, r


def transfer_W(s: complex, P: np.ndarray, qvec: np.ndarray, r: np.ndarray) -> complex:
    return complex(r @ np.linalg.solve(P - s * np.eye(3), qvec))


def find_roots_positive(fun: Callable[[float], float], a: float, b: float, n: int) -> np.ndarray:
    grid = np.linspace(a, b, n)
    vals = np.array([fun(float(x)) for x in grid], dtype=float)
    roots: list[float] = []
    for j in range(n - 1):
        if not np.isfinite(vals[j]) or not np.isfinite(vals[j + 1]):
            continue
        if vals[j] == 0.0:
            root = float(grid[j])
        elif vals[j] * vals[j + 1] < 0.0:
            root = float(brentq(fun, float(grid[j]), float(grid[j + 1]), maxiter=500))
        else:
            continue
        if a < root < b and (not roots or min(abs(root - x) for x in roots) > 1e-7):
            roots.append(root)
    return np.array(sorted(roots), dtype=float)


def harmonic_phi(amplitude: float, k: float, params: Dict[str, float]) -> float:
    amp = float(amplitude)
    gain = float(params["m0"] - params["m1"])

    def integrand(theta: float) -> float:
        return (gain * float(sat(amp * math.cos(theta))) - float(k) * amp * math.cos(theta)) * math.cos(theta)

    theta0 = math.acos(1.0 / amp)
    pts = [0.0, theta0, math.pi - theta0, math.pi + theta0, 2.0 * math.pi - theta0, 2.0 * math.pi]
    pts = sorted(set(max(0.0, min(2.0 * math.pi, p)) for p in pts))
    val = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        val += quad(integrand, lo, hi, epsabs=1e-12, epsrel=1e-12, limit=100)[0]
    return float(val)


def describing_function_gain(amplitude: float, params: Dict[str, float]) -> float:
    amp = float(amplitude)
    gain = float(params["m0"] - params["m1"])
    if amp <= 1.0:
        return gain
    return (2.0 * gain / math.pi) * (math.asin(1.0 / amp) + math.sqrt(amp * amp - 1.0) / (amp * amp))


def canonical_params(P0: np.ndarray, qvec: np.ndarray, r: np.ndarray, A: np.ndarray) -> Tuple[float, float, float, float]:
    eye = np.eye(3)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    rows = []
    rhs = []
    for sample in [0.3, 0.7, 1.1, 1.7, 2.3, 3.1]:
        R = np.linalg.solve(A - sample * eye, eye)
        row = np.array([
            -(e3 @ R @ e3),
            e1 @ R @ e1,
            e1 @ R @ e2,
        ], dtype=complex)
        target = transfer_W(sample, P0, qvec, r)
        rows.extend([np.real(row), np.imag(row)])
        rhs.extend([float(np.real(target)), float(np.imag(target))])
    M = np.asarray(rows, dtype=float)
    y = np.asarray(rhs, dtype=float)
    h, b1, b2 = np.linalg.lstsq(M, y, rcond=None)[0]
    err = float(np.linalg.norm(M @ np.array([h, b1, b2]) - y))
    return float(h), float(b1), float(b2), err


def solve_S(P0: np.ndarray, A: np.ndarray, b: np.ndarray, c: np.ndarray, qvec: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float]:
    n = 3
    rows = []
    rhs = []

    def idx(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            row = np.zeros(n * n)
            for ell in range(n):
                row[idx(ell, j)] += P0[i, ell]
                row[idx(i, ell)] -= A[ell, j]
            rows.append(row)
            rhs.append(0.0)

    for i in range(n):
        row = np.zeros(n * n)
        for ell in range(n):
            row[idx(i, ell)] += b[ell]
        rows.append(row)
        rhs.append(qvec[i])

    for j in range(n):
        row = np.zeros(n * n)
        for ell in range(n):
            row[idx(ell, j)] += r[ell]
        rows.append(row)
        rhs.append(c[j])

    M = np.asarray(rows, dtype=float)
    y = np.asarray(rhs, dtype=float)
    svec = np.linalg.lstsq(M, y, rcond=None)[0]
    S = svec.reshape((n, n))
    return S, float(np.linalg.norm(M @ svec - y))


def build_integer_seed(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = cfg["params"]
    lure = cfg["lure"]
    P, qvec, r = numeric_lure_matrices(params, float(lure["tol_jacobian"]))
    eye = np.eye(3)

    testX = np.array([0.37, -0.21, 0.58], dtype=float)
    test_psi = -0.42
    err_lure = float(np.linalg.norm(P @ testX + qvec * test_psi - chua_model(testX, test_psi, params)))

    W = lambda omega: transfer_W(1j * omega, P, qvec, r)
    omega_roots = find_roots_positive(
        lambda om: float(np.imag(W(om))),
        float(lure["wmin"]),
        float(lure["wmax"]),
        int(lure["nscan"]),
    )
    if omega_roots.size == 0:
        raise RuntimeError("No positive roots of Im(W(i*omega)) were found.")
    k_roots = np.array([-1.0 / float(np.real(W(om))) for om in omega_roots], dtype=float)

    branch_index = int(cfg["branch_index"])
    if branch_index < 0 or branch_index >= len(omega_roots):
        raise ValueError("branch_index is out of range.")
    omega0 = float(omega_roots[branch_index])
    k = float(k_roots[branch_index])

    amp_roots = find_roots_positive(
        lambda amp: harmonic_phi(amp, k, params),
        float(lure["amin"]),
        float(lure["amax"]),
        int(lure["ascan"]),
    )
    if amp_roots.size == 0:
        raise RuntimeError("No harmonic amplitude root was found.")
    a0 = float(amp_roots[0])

    P0 = P + k * np.outer(qvec, r)
    eigP0 = np.linalg.eigvals(P0)
    idx_real = int(np.argmin(np.abs(np.imag(eigP0))))
    d = -float(np.real(eigP0[idx_real]))
    A = np.array([[0.0, -omega0, 0.0], [omega0, 0.0, 0.0], [0.0, 0.0, -d]], dtype=float)

    h_can, b1, b2, err_tf = canonical_params(P0, qvec, r, A)
    b = np.array([b1, b2, 1.0], dtype=float)
    c = np.array([1.0, 0.0, -h_can], dtype=float)
    S, err_s = solve_S(P0, A, b, c, qvec, r)

    Y0 = np.array([a0, 0.0, 0.0], dtype=float)
    X0 = S @ Y0

    err_dyn = float(np.linalg.norm(P0 @ S - S @ A, ord="fro"))
    err_b = float(np.linalg.norm(S @ b - qvec))
    err_c = float(np.linalg.norm(r @ S - c))
    phi0 = harmonic_phi(a0, k, params)
    df_phi0 = math.pi * a0 * (describing_function_gain(a0, params) - k)
    N_a0 = float(describing_function_gain(a0, params))
    W0 = complex(W(omega0))
    nyquist_df_residuals = {
        "convention": "W(s)=r(P-sI)^(-1)q; integer-order Nyquist uses s=i*omega and closure W(i*omega0)=-1/N(a0).",
        "omega0": omega0,
        "W0_real": float(np.real(W0)),
        "W0_imag": float(np.imag(W0)),
        "N_a0": N_a0,
        "minus_inv_N_a0": float(-1.0 / N_a0),
        "imag_W0_abs": float(abs(np.imag(W0))),
        "real_closure_abs": float(abs(np.real(W0) + 1.0 / k)),
        "df_gain_closure_abs": float(abs(N_a0 - k)),
        "total_complex_closure_abs": float(abs(W0 + 1.0 / N_a0)),
    }

    summary = {
        "params": params,
        "matlab_reference": "C:/Users/moren/Documents/MATLAB/verifica_chua_entero.m",
        "P": P.tolist(),
        "qvec": qvec.tolist(),
        "r": r.tolist(),
        "err_lure": err_lure,
        "omega_roots": omega_roots.tolist(),
        "k_roots": k_roots.tolist(),
        "chosen_branch": {
            "branch_index": branch_index,
            "omega0": omega0,
            "k": k,
            "a0": a0,
            "Phi_a0": phi0,
            "Phi_a0_closed_form_check": df_phi0,
            "X0": X0.tolist(),
        },
        "nyquist_df_residuals": nyquist_df_residuals,
        "P0": P0.tolist(),
        "eigP0": [[float(np.real(v)), float(np.imag(v))] for v in eigP0],
        "d": d,
        "canonical": {
            "h": h_can,
            "b1": b1,
            "b2": b2,
            "S": S.tolist(),
            "err_transfer": err_tf,
            "err_S": err_s,
            "err_dyn": err_dyn,
            "err_b": err_b,
            "err_c": err_c,
        },
        "status": {
            "lure_ok": err_lure < 1e-8,
            "amplitude_ok": abs(phi0) < 1e-7,
            "nyquist_df_ok": max(
                nyquist_df_residuals["imag_W0_abs"],
                nyquist_df_residuals["real_closure_abs"],
                nyquist_df_residuals["df_gain_closure_abs"],
                nyquist_df_residuals["total_complex_closure_abs"],
            ) < 1e-6,
            "canonical_ok": max(err_tf, err_s, err_dyn, err_b, err_c) < 1e-6,
        },
    }
    Path(cfg["outputs"]["seed_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "P": P,
        "qvec": qvec,
        "r": r,
        "W": W,
        "omega_roots": omega_roots,
        "k_roots": k_roots,
        "omega0": omega0,
        "k": k,
        "a0": a0,
        "P0": P0,
        "eigP0": eigP0,
        "d": d,
        "A": A,
        "h": h_can,
        "b1": b1,
        "b2": b2,
        "S": S,
        "X0": X0,
        "summary": summary,
    }


def rhs_original(x: np.ndarray, P: np.ndarray, qvec: np.ndarray, r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    sigma = float(r @ x)
    return P @ x + qvec * float(psi_pwl(sigma, params))


def rhs_original_direct(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    x1, x2, x3 = np.asarray(x, dtype=float)
    return np.array([
        params["alpha_chua"] * (x2 - x1 - params["m1"] * x1 - float(psi_pwl(x1, params))),
        x1 - x2 + x3,
        -params["beta"] * x2 - params["gamma_chua"] * x3,
    ], dtype=float)


def rhs_epsilon_family(x: np.ndarray, P0: np.ndarray, qvec: np.ndarray, r: np.ndarray, k: float, eps: float, params: Dict[str, float]) -> np.ndarray:
    sigma = float(r @ x)
    delta = float(psi_pwl(sigma, params)) - k * sigma
    return P0 @ x + float(eps) * qvec * delta


EFORK_Q1_A21 = 0.5
EFORK_Q1_A31 = 0.5
EFORK_Q1_A32 = -0.25
EFORK_Q1_W1 = 2.0 / 3.0
EFORK_Q1_W2 = 5.0 / 3.0
EFORK_Q1_W3 = -4.0 / 3.0


def efork_q1_step(rhs: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float) -> np.ndarray:
    """Advance one integer-order Chua step with the EFORK-3 q=1 formula.

    Mathematically this is the q=1 specialization of the EFORK method used by
    the fractional backends. The Caputo memory correction vanishes at q=1, but
    the stage coefficients are kept exactly as in the shared EFORK scheme so
    the integer pipeline and C backends follow one numerical process.
    """
    x = np.asarray(x, dtype=float)
    h = float(h)
    k1 = h * np.asarray(rhs(x), dtype=float)
    k2 = h * np.asarray(rhs(x + EFORK_Q1_A21 * k1), dtype=float)
    k3 = h * np.asarray(rhs(x + EFORK_Q1_A31 * k2 + EFORK_Q1_A32 * k1), dtype=float)
    return x + EFORK_Q1_W1 * k1 + EFORK_Q1_W2 * k2 + EFORK_Q1_W3 * k3


def efork_q1_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_final: float,
    h: float,
    *,
    div_threshold: float | None = None,
) -> Tuple[np.ndarray, str]:
    """Integrate a trajectory with the q=1 EFORK operational method.

    The returned array has columns [t, x, y, z]. `h` is the actual numerical
    step and also the output spacing, matching the integer pipeline figures.
    """
    h = float(h)
    t_final = float(t_final)
    if h <= 0.0:
        raise ValueError("EFORK q=1 requires h > 0.")
    if t_final < 0.0:
        raise ValueError("EFORK q=1 requires t_final >= 0.")
    x = np.asarray(x0, dtype=float).copy()
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise ValueError("Initial condition must be a finite vector in R^3.")
    n_steps = int(math.ceil(t_final / h))
    times = np.empty(n_steps + 1, dtype=float)
    states = np.empty((n_steps + 1, 3), dtype=float)
    times[0] = 0.0
    states[0] = x
    if n_steps == 0:
        return np.column_stack((times, states)), "ok"
    status = "ok"
    last_index = 0
    for n in range(n_steps):
        if div_threshold is not None and float(np.linalg.norm(x)) >= float(div_threshold):
            status = "diverged"
            break
        try:
            x_next = efork_q1_step(rhs, x, h)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            status = f"solver_exception:{exc}"
            break
        if not np.all(np.isfinite(x_next)):
            status = "nonfinite_solution"
            break
        x = np.asarray(x_next, dtype=float)
        last_index = n + 1
        times[last_index] = last_index * h
        states[last_index] = x
        if div_threshold is not None and float(np.linalg.norm(x)) >= float(div_threshold):
            status = "diverged"
            break
    traj = np.column_stack((times[: last_index + 1], states[: last_index + 1]))
    return traj, status


def integrate_ode(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_final: float,
    dt_output: float,
    rtol: float,
    atol: float,
) -> np.ndarray:
    _ = (rtol, atol)  # EFORK q=1 is fixed-step; tolerances are kept for API compatibility.
    traj, status = efork_q1_integrate(rhs, x0, t_final, dt_output)
    if status != "ok":
        raise RuntimeError(f"EFORK q=1 integration failed: {status}")
    return traj


def continuation_in_epsilon(cfg: Dict[str, Any], seed: Dict[str, Any]) -> List[Dict[str, Any]]:
    c = cfg["continuation"]
    params = cfg["params"]
    x_in = np.asarray(seed["X0"], dtype=float).copy()
    results: list[dict[str, Any]] = []
    log(f"Continuation: {len(c['eps_values'])} epsilon steps.")
    for eps in c["eps_values"]:
        rhs = lambda x, ee=float(eps): rhs_epsilon_family(x, seed["P0"], seed["qvec"], seed["r"], seed["k"], ee, params)
        transient = integrate_ode(rhs, x_in, c["t_transient"], c["dt_output"], c["rtol"], c["atol"])
        x_out = transient[-1, 1:4].astype(float)
        kept = integrate_ode(rhs, x_out, c["t_keep"], c["dt_output"], c["rtol"], c["atol"])
        results.append({"eps": float(eps), "x_in": x_in.copy(), "x_out": x_out.copy(), "traj": kept})
        x_in = x_out.copy()
        log(f"Continuation eps={float(eps):.3f}: x_out={x_out}")

    summary = {
        "eps_values": [float(x) for x in c["eps_values"]],
        "final_state_eps1": results[-1]["x_out"].tolist(),
        "states_by_step": [
            {"eps": r["eps"], "x_in": r["x_in"].tolist(), "x_out": r["x_out"].tolist()}
            for r in results
        ],
    }
    Path(cfg["outputs"]["cont_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def integrate_final_attractor(cfg: Dict[str, Any], seed: Dict[str, Any], x0: np.ndarray) -> Dict[str, Any]:
    fcfg = cfg["final_attractor"]
    c = cfg["continuation"]
    rhs = lambda x: rhs_original(x, seed["P"], seed["qvec"], seed["r"], cfg["params"])
    log(
        "Final attractor: "
        f"burn={fcfg['t_burn']}, keep={fcfg['t_keep']}, dt={fcfg['dt_output']}."
    )
    burn = integrate_ode(
        rhs,
        np.asarray(x0, dtype=float),
        float(fcfg["t_burn"]),
        float(fcfg["dt_output"]),
        float(c["rtol"]),
        float(c["atol"]),
    )
    target_seed = burn[-1, 1:4].astype(float)
    traj = integrate_ode(
        rhs,
        target_seed,
        float(fcfg["t_keep"]),
        float(fcfg["dt_output"]),
        float(c["rtol"]),
        float(c["atol"]),
    )
    return {
        "continuation_final_state": np.asarray(x0, dtype=float),
        "target_seed": target_seed,
        "traj": traj,
        "config": dict(fcfg),
    }


def style_white_axis(ax, *, grid: bool = True, grid_alpha: float = 0.25) -> None:
    ax.set_facecolor(WHITE_BG)
    for spine in getattr(ax, "spines", {}).values():
        spine.set_color("#111827")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#111827", direction="in")
    ax.xaxis.label.set_color("#111827")
    ax.yaxis.label.set_color("#111827")
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_color("#111827")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
                axis.pane.set_edgecolor("#d1d5db")
            except AttributeError:
                pass
    if grid:
        ax.grid(True, alpha=grid_alpha)
    else:
        ax.grid(False)


def downsample_for_plot(data: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.shape[0] <= int(max_points):
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, int(max_points), dtype=int)
    return arr[idx]


def sample_selected_results(results: List[Dict[str, Any]], max_curves: int = 6) -> List[Dict[str, Any]]:
    if len(results) <= int(max_curves):
        return results
    idx = np.linspace(0, len(results) - 1, int(max_curves), dtype=int)
    idx = list(dict.fromkeys(idx.tolist()))
    return [results[i] for i in idx]


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(WHITE_BG)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor=WHITE_BG)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=WHITE_BG)
    plt.close(fig)
    log(f"Saved {path.name} and {png_path.name}")


def plot_seed_figures(cfg: Dict[str, Any], seed: Dict[str, Any]) -> None:
    """Plot the integer-order Nyquist/describing-function closure.

    The integer-order Lure convention used here is
    W(s)=r(P-sI)^(-1)q. The plotted real-axis curve is -1/N(A), so a
    consistent harmonic seed satisfies W(i omega0)=-1/N(a0).
    """
    omega0 = seed["omega0"]
    k = seed["k"]
    a0 = seed["a0"]
    W = seed["W"]
    lure = cfg["lure"]
    omega = np.logspace(np.log10(float(lure["wmin"])), np.log10(float(lure["wmax"])), 5000)
    Wvals = np.array([W(float(w)) for w in omega], dtype=complex)
    W0 = W(omega0)
    Agrid = np.linspace(float(lure["amin"]), max(float(lure["amax"]), 1.25 * a0), 3000)
    Nvals = np.array([describing_function_gain(float(A), cfg["params"]) for A in Agrid], dtype=float)
    valid = np.isfinite(Nvals) & (np.abs(Nvals) > 1e-14)
    minus_invN = np.full_like(Nvals, np.nan, dtype=float)
    minus_invN[valid] = -1.0 / Nvals[valid]

    residuals = seed["summary"].get("nyquist_df_residuals", {})
    max_residual = max(
        float(residuals.get("imag_W0_abs", 0.0)),
        float(residuals.get("real_closure_abs", 0.0)),
        float(residuals.get("df_gain_closure_abs", 0.0)),
        float(residuals.get("total_complex_closure_abs", 0.0)),
    )
    log(
        "Nyquist/DF check: "
        f"omega0={omega0:.10g}, a0={a0:.10g}, "
        f"|W(iw0)+1/N(a0)|={float(residuals.get('total_complex_closure_abs', float('nan'))):.3e}."
    )
    if max_residual > 1e-5:
        raise RuntimeError(
            "Nyquist/DF closure is not numerically consistent; "
            f"largest residual={max_residual:.3e}."
        )

    roots = np.asarray(seed["omega_roots"], dtype=float)
    root_points = np.array([W(float(om)) for om in roots], dtype=complex)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.plot(np.real(Wvals), np.imag(Wvals), lw=1.35, color=NYQUIST_W_COLOR, label=r"$W(i\omega)$")
    ax.plot(minus_invN, np.zeros_like(minus_invN), lw=1.2, color=NYQUIST_DF_COLOR, label=r"$-1/N(A)$")
    if root_points.size:
        ax.scatter(np.real(root_points), np.imag(root_points), s=38, facecolors="none", edgecolors="#111827", linewidths=1.0, label="cruces Im W=0")
    ax.scatter([np.real(W0)], [np.imag(W0)], s=58, facecolors="none", edgecolors="#ef4444", linewidths=1.5, zorder=5, label=r"$W(i\omega_0)$")
    ax.scatter([-1.0 / k], [0.0], s=52, c=NYQUIST_DF_COLOR, marker="x", linewidths=1.6, zorder=5, label=r"$-1/N(a_0)$")
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.axvline(0.0, color="#9ca3af", ls=":", lw=0.7)
    ax.set_xlabel(r"Re$(W(i\omega))$")
    ax.set_ylabel(r"Im$(W(i\omega))$")
    style_white_axis(ax)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, cfg["outputs"]["nyquist_pdf"])

    x_center = float(np.real(W0))
    x_margin = max(0.03, 0.08 * max(1.0, abs(x_center)))
    zoom_mask = (np.real(Wvals) >= x_center - x_margin) & (np.real(Wvals) <= x_center + x_margin)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(np.real(Wvals), np.imag(Wvals), lw=1.35, color=NYQUIST_W_COLOR, label=r"$W(i\omega)$")
    ax.plot(minus_invN, np.zeros_like(minus_invN), lw=1.2, color=NYQUIST_DF_COLOR, label=r"$-1/N(A)$")
    ax.scatter([np.real(W0)], [np.imag(W0)], s=62, facecolors="none", edgecolors="#ef4444", linewidths=1.5, zorder=5, label="cierre elegido")
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.set_xlim(x_center - x_margin, x_center + x_margin)
    if np.any(zoom_mask):
        yvals = np.imag(Wvals[zoom_mask])
        ypad = max(0.015, 0.25 * float(np.max(np.abs(yvals))))
        ax.set_ylim(float(np.min(yvals) - ypad), float(np.max(yvals) + ypad))
    ax.set_xlabel(r"Re$(W(i\omega))$")
    ax.set_ylabel(r"Im$(W(i\omega))$")
    style_white_axis(ax)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, cfg["outputs"]["nyquist_zoom_pdf"])


def plot_continuation_figures(cfg: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    eps = np.array([r["eps"] for r in results], dtype=float)
    xin = np.vstack([r["x_in"] for r in results])
    xout = np.vstack([r["x_out"] for r in results])

    for i, name in enumerate(["x", "y", "z"]):
        fig, ax = plt.subplots(figsize=(6.7, 4.4))
        ax.plot(eps, xin[:, i], "o-", ms=4, lw=1.1, color=LINEARIZED_COLOR, label="entrada")
        ax.plot(eps, xout[:, i], "s--", ms=3.5, lw=1.0, color=ORIGINAL_COLOR, label="salida")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(name)
        style_white_axis(ax)
        ax.legend(loc="best", fontsize=8)
        save_figure(fig, cfg["outputs"][f"cont_progress_{name}_pdf"])


def plot_continuation_story(cfg: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    chosen = sample_selected_results(results, max_curves=6)
    eps_all = np.array([float(r["eps"]) for r in results], dtype=float)
    eps_min = float(np.min(eps_all))
    eps_max = float(np.max(eps_all))
    xin_path = np.array([np.asarray(r["x_in"], dtype=float) for r in results], dtype=float)
    xout_path = np.array([np.asarray(r["x_out"], dtype=float) for r in results], dtype=float)

    fig = plt.figure(figsize=(7.4, 5.9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("plasma")
    ax.plot(xin_path[:, 0], xin_path[:, 1], xin_path[:, 2], "k--", lw=1.0, label="entrada epsilon")
    ax.plot(xout_path[:, 0], xout_path[:, 1], xout_path[:, 2], color="0.45", ls=":", lw=1.0, label="salida epsilon")
    for r in chosen:
        eps = float(r["eps"])
        color = cmap((eps - eps_min) / max(1e-12, eps_max - eps_min))
        traj = np.asarray(r["traj"], dtype=float)
        ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], lw=0.85, color=color, alpha=0.95)
    first = np.asarray(results[0]["traj"], dtype=float)
    last = np.asarray(results[-1]["traj"], dtype=float)
    ax.plot(first[:, 1], first[:, 2], first[:, 3], lw=1.2, color=NYQUIST_W_COLOR, label="primer paso")
    ax.plot(last[:, 1], last[:, 2], last[:, 3], lw=1.4, color=ORIGINAL_COLOR, label="paso final")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    style_white_axis(ax, grid=True, grid_alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, cfg["outputs"]["continuation_story_pdf"])


def plot_final_attractor(cfg: Dict[str, Any], final_data: Dict[str, Any]) -> None:
    final = np.asarray(final_data["traj"], dtype=float)
    target_seed = np.asarray(final_data["target_seed"], dtype=float)
    continuation_final = np.asarray(final_data["continuation_final_state"], dtype=float)
    fig = plt.figure(figsize=(7.0, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(final[:, 1], final[:, 2], final[:, 3], lw=0.7, color=ORIGINAL_COLOR)
    ax.scatter(continuation_final[0], continuation_final[1], continuation_final[2], s=28, c="black", label="continuation final")
    ax.scatter(target_seed[0], target_seed[1], target_seed[2], s=42, c=BASIN_SEED_COLOR, edgecolors="black", label="post-burn seed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    style_white_axis(ax, grid=True, grid_alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, cfg["outputs"]["final_attr_pdf"])


def plot_final_attractor_planes(cfg: Dict[str, Any], final_data: Dict[str, Any]) -> None:
    final = np.asarray(final_data["traj"], dtype=float)
    plane_specs = [
        ("xy", 1, 2, "x", "y", cfg["outputs"]["final_attr_xy_pdf"]),
        ("xz", 1, 3, "x", "z", cfg["outputs"]["final_attr_xz_pdf"]),
        ("yz", 2, 3, "y", "z", cfg["outputs"]["final_attr_yz_pdf"]),
    ]
    for _name, col_u, col_v, xlabel, ylabel, path in plane_specs:
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ax.plot(final[:, col_u], final[:, col_v], lw=0.75, color=ORIGINAL_COLOR)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        style_white_axis(ax)
        save_figure(fig, path)


def plot_linear_vs_original(cfg: Dict[str, Any], seed: Dict[str, Any]) -> Dict[str, float]:
    comp = cfg["comparison"]
    c = cfg["continuation"]
    X0 = np.asarray(seed["X0"], dtype=float)
    lin = integrate_ode(lambda x: seed["P0"] @ x, X0, comp["t_final"], comp["dt_output"], c["rtol"], c["atol"])
    org = integrate_ode(lambda x: rhs_original(x, seed["P"], seed["qvec"], seed["r"], cfg["params"]), X0, comp["t_final"], comp["dt_output"], c["rtol"], c["atol"])
    lin_interp = np.column_stack([np.interp(org[:, 0], lin[:, 0], lin[:, j]) for j in range(1, 4)])
    err_abs = np.linalg.norm(org[:, 1:4] - lin_interp, axis=1)
    err_rel = err_abs / np.maximum(np.linalg.norm(org[:, 1:4], axis=1), 1e-12)
    max_plot_points = int(comp.get("max_plot_points", 900))
    lin_plot = downsample_for_plot(lin, max_plot_points)
    org_plot = downsample_for_plot(org, max_plot_points)

    figure_paths = {}
    for i, label in enumerate(["x", "y", "z"]):
        fig, ax = plt.subplots(figsize=(6.7, 4.4))
        ax.plot(org_plot[:, 0], org_plot[:, i + 1], lw=0.95, color=ORIGINAL_COLOR, label="original")
        ax.plot(lin_plot[:, 0], lin_plot[:, i + 1], "--", lw=0.95, color=LINEARIZED_COLOR, label="linealizada")
        ax.set_xlabel("t")
        ax.set_ylabel(f"{label}(t)")
        style_white_axis(ax)
        ax.legend(loc="best", fontsize=8)
        key = f"linear_vs_original_{label}_pdf"
        figure_paths[key] = str(cfg["outputs"][key])
        save_figure(fig, cfg["outputs"][key])

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(org_plot[:, 1], org_plot[:, 2], org_plot[:, 3], lw=0.85, color=ORIGINAL_COLOR, label="original")
    ax.plot(lin_plot[:, 1], lin_plot[:, 2], lin_plot[:, 3], "--", lw=0.85, color=LINEARIZED_COLOR, label="linealizada")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    style_white_axis(ax, grid=True, grid_alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    figure_paths["linear_vs_original_3d_pdf"] = str(cfg["outputs"]["linear_vs_original_3d_pdf"])
    save_figure(fig, cfg["outputs"]["linear_vs_original_3d_pdf"])
    return {
        "final_abs_error": float(err_abs[-1]),
        "final_rel_error": float(err_rel[-1]),
        "max_rel_error": float(np.max(err_rel)),
        "integration_points": int(org.shape[0]),
        "plotted_points": int(org_plot.shape[0]),
        "figures": figure_paths,
    }


def hidden_config_from_pipeline(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    hcfg = copy.deepcopy(hv.DEFAULT_CONFIG)
    hcfg["params"] = dict(cfg["params"])
    hcfg["frac_order"] = 1.0
    hcfg["target_seed"] = {"x": float(final_state[0]), "y": float(final_state[1]), "z": float(final_state[2])}
    hcfg["integration"] = {
        "h": float(cfg["verify_hidden"]["h"]),
        "Lm": float(cfg["verify_hidden"]["Lm"]),
        "TMAX_REF": float(cfg["verify_hidden"]["TMAX_REF"]),
        "TMAX_TEST": float(cfg["verify_hidden"]["TMAX_TEST"]),
        "TBURN_REF": float(cfg["verify_hidden"]["TBURN_REF"]),
        "TBURN_TEST": float(cfg["verify_hidden"]["TBURN_TEST"]),
    }
    hcfg["thresholds"] = {
        "R_DIV": float(cfg["verify_hidden"]["R_DIV"]),
        "EPS_EQ": float(cfg["verify_hidden"]["EPS_EQ"]),
        "CAP_WIN": int(cfg["verify_hidden"]["CAP_WIN"]),
        "SEC_TOL": float(cfg["verify_hidden"]["SEC_TOL"]),
        "MIN_SEC_MATCH": int(cfg["verify_hidden"]["MIN_SEC_MATCH"]),
        "TEST_MAX_SEC": int(cfg["verify_hidden"]["TEST_MAX_SEC"]),
        "HIT_FRAC_REQ": float(cfg["verify_hidden"]["HIT_FRAC_REQ"]),
    }
    hcfg["sampling"] = {
        "RADII": [float(r) for r in cfg["verify_hidden"]["RADII"]],
        "NSAMPLES_PER_RADIUS": int(cfg["verify_hidden"]["NSAMPLES_PER_RADIUS"]),
        "random_seed": int(cfg["verify_hidden"]["random_seed"]),
    }
    hcfg["files"] = {
        "summary_from_pipeline": str(cfg["outputs"]["summary_json"]),
        "csv_out": str(HIDDEN_VERIFY_DIR / "hidden_target_check_integer.csv"),
        "ref_csv_out": str(HIDDEN_VERIFY_DIR / "reference_section_integer.csv"),
        "summary_csv_out": str(HIDDEN_VERIFY_DIR / "summary_by_radius_integer.csv"),
        "json_out": str(HIDDEN_VERIFY_DIR / "hidden_target_summary_integer.json"),
        "fig_section": str(HIDDEN_VERIFY_DIR / "reference_section_integer.png"),
        "fig_probe": str(HIDDEN_VERIFY_DIR / "probe_summary_integer.png"),
    }
    hcfg["backend"] = {
        "source_c": str((ROOT / "chua_hidden_backend.c").resolve()),
        "exe": str((NATIVE_DIR / ("chua_hidden_backend_integer.exe" if os.name == "nt" else "chua_hidden_backend_integer")).resolve()),
        "compile": True,
        "openmp": True,
    }
    return hv.prepare_runtime_paths(hcfg, HIDDEN_VERIFY_DIR)


def plot_reference_section_pdf(ref: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(ref[:, 0], ref[:, 1], s=9, color=NYQUIST_W_COLOR)
    ax.set_xlabel("y on section x=0")
    ax.set_ylabel("z on section x=0")
    style_white_axis(ax)
    save_figure(fig, path)


def plot_probe_summary_pdf(summary_rows: List[Dict[str, str]], eq_names: Iterable[str], path: Path) -> None:
    classes = ["EQ", "DIV", "TARGET", "OTHER", "UNKNOWN"]
    colors = {
        "EQ": BASIN_CLASS_COLORS[0],
        "DIV": BASIN_CLASS_COLORS[3],
        "TARGET": BASIN_SEED_COLOR,
        "OTHER": BASIN_CLASS_COLORS[4],
        "UNKNOWN": "#9ca3af",
    }
    eq_names = list(eq_names)
    safe = lambda text: "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "eq"
    for eq in eq_names:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        rows = [r for r in summary_rows if r["equilibrium"] == eq]
        radii = [float(r["radius"]) for r in rows]
        for cname in classes:
            ax.plot(radii, [int(r[cname]) for r in rows], marker="o", ms=3, lw=1.0, color=colors[cname], label=cname)
        ax.set_xscale("log")
        ax.set_xlabel("radius")
        ax.set_ylabel(eq)
        style_white_axis(ax)
        ax.legend(loc="best", fontsize=8)
        out_path = path if len(eq_names) == 1 else path.with_name(f"{path.stem}_{safe(eq)}{path.suffix}")
        save_figure(fig, out_path)


def run_hidden_verification(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    hidden_cfg = hidden_config_from_pipeline(cfg, final_state)
    cfg_path = HIDDEN_VERIFY_DIR / "config_hidden_verify_integer.json"
    cfg_path.write_text(json.dumps(hidden_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    log(
        "Hidden verification C: "
        f"TREF={hidden_cfg['integration']['TMAX_REF']}, "
        f"TTEST={hidden_cfg['integration']['TMAX_TEST']}, "
        f"samples={len(hidden_cfg['sampling']['RADII']) * hidden_cfg['sampling']['NSAMPLES_PER_RADIUS']}."
    )
    hv.compile_backend(hidden_cfg)
    hv.run_backend(hidden_cfg)

    files = hidden_cfg["files"]
    ref = hv.load_reference_csv(files["ref_csv_out"])
    detail_rows = hv.load_csv_dicts(files["csv_out"])
    summary_rows = hv.load_csv_dicts(files["summary_csv_out"])
    eqs = hv.chua_equilibria(hv.ChuaParams(**hidden_cfg["params"]))
    hv.plot_reference_section(ref, files["fig_section"])
    hv.plot_probe_summary(summary_rows, [float(r) for r in hidden_cfg["sampling"]["RADII"]], list(eqs.keys()), files["fig_probe"])
    plot_reference_section_pdf(ref, cfg["outputs"]["ref_section_pdf"])
    plot_probe_summary_pdf(summary_rows, list(eqs.keys()), cfg["outputs"]["probe_summary_pdf"])

    summary = hv.build_summary_json(hidden_cfg, eqs, ref, summary_rows, detail_rows)
    summary["notes"] = [
        "Integer-order verification uses the same C backend interface as the fractional pipeline.",
        "For frac_order=1.0 the C backend is invoked through the shared EFORK-style operational integrator.",
        "Hiddenness is tested by sampling neighborhoods of equilibria and comparing section fingerprints.",
    ]
    Path(files["json_out"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


HIDDEN_PROBE_CLASS_COLORS = {
    "EQ": "#ef4444",
    "DIV": "#2563eb",
    "TARGET": "#f472b6",
    "OTHER": "#111827",
    "UNKNOWN": "#6b7280",
}
HIDDEN_PROBE_CLASS_LABELS = {
    "EQ": "equilibrio",
    "DIV": "diverge",
    "TARGET": "atractor objetivo",
    "OTHER": "otro destino",
    "UNKNOWN": "inconcluso",
}


def downsample_rows(values: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape[0] <= max_points:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, int(max_points), dtype=int)
    return arr[idx]


def hidden_probe_start(row: Dict[str, str]) -> np.ndarray:
    return np.array([float(row["x0"]), float(row["y0"]), float(row["z0"])], dtype=float)


def select_hidden_probe_rows(
    detail_rows: List[Dict[str, str]],
    max_count: int,
    random_seed: int,
) -> List[Dict[str, str]]:
    """Select a deterministic, class-stratified subset of hiddenness probes.

    The figure is evidence-linked: rows come from the same hiddenness CSV used
    to compute the quantitative report. Rare TARGET hits are deliberately kept
    when present, because they weaken a hidden-attractor claim and must remain
    visible.
    """
    rows = [r for r in detail_rows if r.get("class") in HIDDEN_PROBE_CLASS_COLORS]
    if len(rows) <= max_count:
        return rows

    rng = np.random.default_rng(int(random_seed))
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["class"], []).append(row)
    for cls_rows in groups.values():
        order = rng.permutation(len(cls_rows))
        cls_rows[:] = [cls_rows[int(i)] for i in order]

    selected: List[Dict[str, str]] = []
    nonempty = [cls for cls in ["TARGET", "EQ", "DIV", "OTHER", "UNKNOWN"] if groups.get(cls)]
    quota = max(1, max_count // max(1, len(nonempty)))
    cursors = {cls: 0 for cls in nonempty}
    for cls in nonempty:
        take = min(quota, len(groups[cls]), max_count - len(selected))
        selected.extend(groups[cls][:take])
        cursors[cls] = take

    while len(selected) < max_count:
        advanced = False
        for cls in nonempty:
            pos = cursors[cls]
            if pos < len(groups[cls]):
                selected.append(groups[cls][pos])
                cursors[cls] = pos + 1
                advanced = True
                if len(selected) >= max_count:
                    break
        if not advanced:
            break
    return selected


def integrate_integer_probe_trajectory(cfg: Dict[str, Any], x0: np.ndarray) -> np.ndarray:
    """Integrate one integer-order Chua probe with the same EFORK q=1 process.

    This auxiliary visualization does not replace the hiddenness classifier:
    the classification is still the C backend result. The trajectory is only
    drawn to make the sampled equilibrium neighborhoods geometrically visible.
    """
    vis = cfg["hidden_illustration"]
    t_final = float(vis["t_total"])
    dt_output = float(vis["dt_output"])
    threshold = float(cfg["verify_hidden"]["R_DIV"])
    traj, _status = efork_q1_integrate(
        lambda x: rhs_original_direct(x, cfg["params"]),
        np.asarray(x0, dtype=float),
        t_final,
        dt_output,
        div_threshold=threshold,
    )
    if traj.shape[0] == 0:
        return np.column_stack(([0.0], np.asarray(x0, dtype=float)[None, :]))
    return traj


def padded_limits(points: np.ndarray, pad_fraction: float = 0.15, min_pad: float = 0.5) -> List[Tuple[float, float]]:
    arr = np.asarray(points, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.size == 0:
        return [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    limits: List[Tuple[float, float]] = []
    for j in range(3):
        lo = float(np.min(arr[:, j]))
        hi = float(np.max(arr[:, j]))
        span = hi - lo
        pad = max(min_pad, pad_fraction * max(span, 1e-12))
        limits.append((lo - pad, hi + pad))
    return limits


def apply_3d_limits(ax: Any, limits: List[Tuple[float, float]]) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


def draw_hiddenness_scene(
    cfg: Dict[str, Any],
    path: Path,
    attractor_xyz: np.ndarray,
    target_seed: np.ndarray,
    equilibria: Dict[str, np.ndarray],
    probe_trajs: List[Tuple[Dict[str, str], np.ndarray]],
    limits: List[Tuple[float, float]],
    *,
    zoom: bool,
) -> None:
    fig = plt.figure(figsize=(7.1, 5.8))
    fig.patch.set_facecolor(WHITE_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(WHITE_BG)
    ax.plot(
        attractor_xyz[:, 0],
        attractor_xyz[:, 1],
        attractor_xyz[:, 2],
        lw=1.05 if zoom else 0.95,
        color="#00b050",
        alpha=0.98,
        label="atractor candidato",
        zorder=5,
    )
    for row, traj in probe_trajs:
        xyz = np.asarray(traj[:, 1:4], dtype=float)
        xyz = downsample_rows(xyz, int(cfg["hidden_illustration"]["max_points_per_probe"]))
        cls = row.get("class", "UNKNOWN")
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            lw=0.34 if not zoom else 0.42,
            alpha=0.30 if not zoom else 0.38,
            color=HIDDEN_PROBE_CLASS_COLORS.get(cls, "#6b7280"),
            zorder=2,
        )

    for name, eq in equilibria.items():
        eq = np.asarray(eq, dtype=float)
        if np.all(np.isfinite(eq)):
            ax.scatter(eq[0], eq[1], eq[2], s=34, c="#111827", edgecolors=WHITE_BG, linewidths=0.5, zorder=7)
            ax.text(eq[0], eq[1], eq[2], f" {name}", color="#111827", fontsize=8)
    ax.scatter(
        target_seed[0],
        target_seed[1],
        target_seed[2],
        s=70,
        c=BASIN_SEED_COLOR,
        marker="*",
        edgecolors="#111827",
        linewidths=0.6,
        zorder=8,
        label="semilla post-transitorio",
    )
    apply_3d_limits(ax, limits)
    ax.view_init(elev=22 if not zoom else 18, azim=-52 if not zoom else -38)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    style_white_axis(ax, grid=True, grid_alpha=0.18)
    legend_items = [
        Line2D([0], [0], color="#00b050", lw=1.4, label="atractor candidato"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=BASIN_SEED_COLOR, markeredgecolor="#111827", markersize=9, label="semilla"),
    ]
    used_classes = sorted({row.get("class", "UNKNOWN") for row, _ in probe_trajs})
    for cls in ["TARGET", "EQ", "DIV", "OTHER", "UNKNOWN"]:
        if cls in used_classes:
            legend_items.append(Line2D([0], [0], color=HIDDEN_PROBE_CLASS_COLORS[cls], lw=1.2, label=HIDDEN_PROBE_CLASS_LABELS[cls]))
    ax.legend(handles=legend_items, loc="best", fontsize=7.6)
    save_figure(fig, path)


def plot_hiddenness_illustration(
    cfg: Dict[str, Any],
    final_data: Dict[str, Any],
    hidden_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Create article-style hiddenness overview and zoom figures.

    The green orbit is the candidate attractor obtained after continuation and
    post-transient integration. Colored thin curves are re-integrated only for
    visualization from the actual initial conditions sampled by the hiddenness
    backend; their colors preserve the backend class labels. Therefore TARGET
    hits from equilibrium neighborhoods remain visible and should be interpreted
    conservatively, not hidden.
    """
    if hidden_summary is None:
        return {"status": "skipped", "reason": "hidden verification was skipped"}
    if not bool(cfg["hidden_illustration"].get("enabled", True)):
        return {"status": "skipped", "reason": "hidden illustration disabled"}

    files = hidden_summary.get("files", {})
    csv_path = files.get("csv_out")
    if not csv_path or not Path(csv_path).exists():
        return {"status": "skipped", "reason": "hidden verification detail CSV is missing"}

    detail_rows = hv.load_csv_dicts(csv_path)
    selected_rows = select_hidden_probe_rows(
        detail_rows,
        int(cfg["hidden_illustration"]["max_probe_trajectories"]),
        int(cfg["hidden_illustration"]["random_seed"]),
    )
    probe_trajs: List[Tuple[Dict[str, str], np.ndarray]] = []
    for row in selected_rows:
        try:
            traj = integrate_integer_probe_trajectory(cfg, hidden_probe_start(row))
        except Exception as exc:  # pragma: no cover - visualization should not abort science run
            row = dict(row)
            row["integration_error"] = str(exc)
            traj = np.column_stack(([0.0], hidden_probe_start(row)[None, :]))
        probe_trajs.append((row, traj))

    final = np.asarray(final_data["traj"], dtype=float)
    attractor_xyz = downsample_rows(final[:, 1:4], int(cfg["hidden_illustration"]["max_attractor_points"]))
    target_seed = np.asarray(final_data["target_seed"], dtype=float)
    equilibria = {
        name: np.asarray(value, dtype=float)
        for name, value in hidden_summary.get("equilibria", {}).items()
    }
    start_points = np.array([hidden_probe_start(r) for r in selected_rows], dtype=float) if selected_rows else np.empty((0, 3))
    eq_points = np.array([v for v in equilibria.values() if np.all(np.isfinite(v))], dtype=float) if equilibria else np.empty((0, 3))
    overview_basis = np.vstack([arr for arr in [attractor_xyz, target_seed[None, :], start_points, eq_points] if arr.size])
    overview_limits = padded_limits(overview_basis, pad_fraction=0.22, min_pad=0.75)
    zoom_limits = padded_limits(attractor_xyz, pad_fraction=float(cfg["hidden_illustration"]["zoom_pad_fraction"]), min_pad=0.35)

    draw_hiddenness_scene(
        cfg,
        Path(cfg["outputs"]["hidden_illustration_overview_pdf"]),
        attractor_xyz,
        target_seed,
        equilibria,
        probe_trajs,
        overview_limits,
        zoom=False,
    )
    draw_hiddenness_scene(
        cfg,
        Path(cfg["outputs"]["hidden_illustration_zoom_pdf"]),
        attractor_xyz,
        target_seed,
        equilibria,
        probe_trajs,
        zoom_limits,
        zoom=True,
    )

    total_counts: Dict[str, int] = {cls: 0 for cls in HIDDEN_PROBE_CLASS_COLORS}
    selected_counts: Dict[str, int] = {cls: 0 for cls in HIDDEN_PROBE_CLASS_COLORS}
    for row in detail_rows:
        cls = row.get("class", "UNKNOWN")
        if cls in total_counts:
            total_counts[cls] += 1
    for row in selected_rows:
        cls = row.get("class", "UNKNOWN")
        if cls in selected_counts:
            selected_counts[cls] += 1
    summary = {
        "status": "ok",
        "method": "visualization of hidden verification probes from backend CSV",
        "detail_csv": str(csv_path),
        "total_counts": total_counts,
        "selected_counts": selected_counts,
        "outputs": {
            "overview_pdf": str(cfg["outputs"]["hidden_illustration_overview_pdf"]),
            "overview_png": str(Path(cfg["outputs"]["hidden_illustration_overview_pdf"]).with_suffix(".png")),
            "zoom_pdf": str(cfg["outputs"]["hidden_illustration_zoom_pdf"]),
            "zoom_png": str(Path(cfg["outputs"]["hidden_illustration_zoom_pdf"]).with_suffix(".png")),
        },
        "interpretation": (
            "No TARGET hits were reported in the sampled equilibrium neighborhoods."
            if total_counts.get("TARGET", 0) == 0 else
            "TARGET hits were reported; interpret hiddenness conservatively and inspect the quantitative report."
        ),
    }
    Path(cfg["outputs"]["hidden_illustration_json"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def compile_shared_library(source: str, openmp: bool = True) -> Path:
    src = ROOT / source
    system = platform.system().lower()
    if system == "windows":
        ext = ".dll"
    elif system == "darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    out = NATIVE_DIR / f"chua_basin_integer{ext}"
    cmd = ["gcc", "-O3", "-shared"]
    if system != "windows":
        cmd.append("-fPIC")
    if openmp:
        cmd.insert(2, "-fopenmp")
    cmd += ["-o", str(out), str(src), "-lm"]
    log(f"Compiling basin C library: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=bool(openmp), text=True)
    except subprocess.CalledProcessError:
        if not openmp or "-fopenmp" not in cmd:
            raise
        fallback_cmd = [part for part in cmd if part != "-fopenmp"]
        log("OpenMP is not available in this compiler; retrying the basin library without -fopenmp.")
        subprocess.run(fallback_cmd, check=True)
    return out


def load_basin_library(cfg: Dict[str, Any]):
    global BASIN_LIBRARY_CACHE
    if BASIN_LIBRARY_CACHE is not None:
        return BASIN_LIBRARY_CACHE
    libpath = compile_shared_library(cfg["basin"]["source"], bool(cfg["basin"]["openmp"]))
    if platform.system().lower() == "windows" and hasattr(os, "add_dll_directory"):
        candidate_dirs = [libpath.parent]
        gcc_path = shutil.which("gcc")
        if gcc_path:
            candidate_dirs.append(Path(gcc_path).resolve().parent)
        for candidate in candidate_dirs:
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in REGISTERED_DLL_DIRS:
                DLL_SEARCH_HANDLES.append(os.add_dll_directory(candidate_str))
                REGISTERED_DLL_DIRS.add(candidate_str)
    lib = ctypes.CDLL(str(libpath.resolve()))
    lib.set_chua_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.set_chua_params.restype = None
    lib.get_equilibria.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
    lib.get_equilibria.restype = None
    lib.compute_basin_xy.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.compute_basin_xy.restype = ctypes.c_int
    lib.compute_basin_plane.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.compute_basin_plane.restype = ctypes.c_int
    BASIN_LIBRARY_CACHE = lib
    return lib


def compute_basin(cfg: Dict[str, Any], z0: float | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lib = load_basin_library(cfg)
    p = cfg["params"]
    b = cfg["basin"]
    lib.set_chua_params(float(p["alpha_chua"]), float(p["beta"]), float(p["gamma_chua"]), float(p["m0"]), float(p["m1"]))
    nx, ny = int(b["nx"]), int(b["ny"])
    z_plane = float(b["z0"] if z0 is None else z0)
    out = np.empty(nx * ny, dtype=np.int32)
    log(f"Basin C: grid={nx}x{ny}, z0={z_plane}, q=1.0, TMAX={b['TMAX']}.")
    rc = lib.compute_basin_xy(
        nx, ny, float(b["xmin"]), float(b["xmax"]), float(b["ymin"]), float(b["ymax"]),
        z_plane, 1.0, float(b["h"]), float(b["Lm"]), float(b["TMAX"]), float(b["TBURN"]),
        float(b["R_DIV"]), float(b["R_BOUND"]), float(b["EPS_EQ"]), int(b["CAP_WIN"]), float(b["MEAN_X_GAP"]), out,
    )
    if rc != 0:
        raise RuntimeError(f"compute_basin_xy returned {rc}")
    eq = np.empty(9, dtype=np.float64)
    lib.get_equilibria(eq)
    return out.reshape((ny, nx)), eq[0:3].copy(), eq[3:6].copy(), eq[6:9].copy()


def basin_class_counts(grid: np.ndarray) -> Dict[str, int]:
    arr = np.asarray(grid, dtype=np.int32)
    return {str(cls): int(np.count_nonzero(arr == cls)) for cls in range(5)}


def compute_basin_planes(cfg: Dict[str, Any], final_seed: np.ndarray) -> Dict[str, Any]:
    lib = load_basin_library(cfg)
    p = cfg["params"]
    b = cfg["basin_planes"]
    lib.set_chua_params(float(p["alpha_chua"]), float(p["beta"]), float(p["gamma_chua"]), float(p["m0"]), float(p["m1"]))

    nx, ny, nz = int(b["nx"]), int(b["ny"]), int(b["nz"])
    xvals = np.linspace(float(b["xlim"][0]), float(b["xlim"][1]), nx)
    yvals = np.linspace(float(b["ylim"][0]), float(b["ylim"][1]), ny)
    zvals = np.linspace(float(b["zlim"][0]), float(b["zlim"][1]), nz)
    seed = np.asarray(final_seed, dtype=float)
    planes = {
        "xy": {"plane_id": 0, "uvals": xvals, "vvals": yvals, "fixed": float(seed[2]), "xlabel": "x", "ylabel": "y"},
        "xz": {"plane_id": 1, "uvals": xvals, "vvals": zvals, "fixed": float(seed[1]), "xlabel": "x", "ylabel": "z"},
        "yz": {"plane_id": 2, "uvals": yvals, "vvals": zvals, "fixed": float(seed[0]), "xlabel": "y", "ylabel": "z"},
    }
    out: Dict[str, Any] = {}
    log(f"Basin cuts C: xy={nx}x{ny}, xz={nx}x{nz}, yz={ny}x{nz}, q=1.0.")
    for name, spec in planes.items():
        uvals = spec["uvals"]
        vvals = spec["vvals"]
        grid_out = np.empty(len(uvals) * len(vvals), dtype=np.int32)
        log(f"Basin cut {name}: starting.")
        rc = lib.compute_basin_plane(
            int(len(uvals)), int(len(vvals)),
            float(uvals[0]), float(uvals[-1]), float(vvals[0]), float(vvals[-1]),
            float(spec["fixed"]), int(spec["plane_id"]), 1.0, float(b["h"]), float(b["Lm"]),
            float(b["TMAX"]), float(b["TBURN"]), float(b["R_DIV"]), float(b["R_BOUND"]),
            float(b["EPS_EQ"]), int(b["CAP_WIN"]), float(b["MEAN_X_GAP"]), grid_out,
        )
        if rc != 0:
            raise RuntimeError(f"compute_basin_plane({name}) returned {rc}")
        out[name] = {
            **spec,
            "grid": grid_out.reshape((len(vvals), len(uvals))),
        }
        log(f"Basin cut {name}: done.")
    return out


def save_distances_csv(cfg: Dict[str, Any], final_state: np.ndarray, E0: np.ndarray, Ep: np.ndarray, Em: np.ndarray) -> None:
    lines = ["label,dist_3d,dist_xy\n"]
    for label, eq in [("E0", E0), ("E+", Ep), ("E-", Em)]:
        d3 = float(np.linalg.norm(final_state - eq))
        dxy = float(np.linalg.norm(final_state[:2] - eq[:2]))
        lines.append(f"{label},{d3:.16g},{dxy:.16g}\n")
    Path(cfg["outputs"]["dist_csv"]).write_text("".join(lines), encoding="utf-8")


def plot_basin_overlay(
    cfg: Dict[str, Any],
    basin: np.ndarray,
    E0: np.ndarray,
    Ep: np.ndarray,
    Em: np.ndarray,
    final_state: np.ndarray,
    path: Path | None = None,
    z0: float | None = None,
) -> None:
    b = cfg["basin"]
    cmap, norm = basin_cmap_norm()
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    fig.patch.set_facecolor(WHITE_BG)
    ax.set_facecolor(WHITE_BG)
    ax.imshow(
        basin,
        origin="lower",
        extent=[b["xmin"], b["xmax"], b["ymin"], b["ymax"]],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax.scatter(E0[0], E0[1], c="black", s=34, marker="o", zorder=4, label="E0")
    ax.scatter(Ep[0], Ep[1], c="white", edgecolors="black", s=48, marker="^", zorder=4, label="E+")
    ax.scatter(Em[0], Em[1], c="white", edgecolors="black", s=48, marker="v", zorder=4, label="E-")
    ax.scatter(final_state[0], final_state[1], c=BASIN_SEED_COLOR, edgecolors="black", s=95, marker="*", zorder=5, label="target")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if z0 is not None:
        ax.text(
            0.02, 0.98, f"z={z0:.4g}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#111827", "alpha": 0.82, "linewidth": 0.35},
        )
    class_handles = [
        Patch(facecolor=BASIN_CLASS_COLORS[i], edgecolor="black", linewidth=0.25, label=BASIN_CLASS_LABELS[i])
        for i in (1, 2, 3, 4, 0)
    ]
    point_handles, point_labels = ax.get_legend_handles_labels()
    ax.legend(class_handles + point_handles, [h.get_label() for h in class_handles] + point_labels, loc="upper right", fontsize=7)
    style_white_axis(ax, grid=False)
    save_figure(fig, path or cfg["outputs"]["basin_pdf"])


def plot_basin_planes(cfg: Dict[str, Any], planes: Dict[str, Any], final_seed: np.ndarray) -> None:
    cmap, norm = basin_cmap_norm()
    seed = np.asarray(final_seed, dtype=float)
    seed_by_plane = {"xy": (seed[0], seed[1]), "xz": (seed[0], seed[2]), "yz": (seed[1], seed[2])}
    output_by_plane = {
        "xy": cfg["outputs"]["basin_xy_pdf"],
        "xz": cfg["outputs"]["basin_xz_pdf"],
        "yz": cfg["outputs"]["basin_yz_pdf"],
    }
    for name in ("xy", "xz", "yz"):
        data = planes[name]
        uvals = np.asarray(data["uvals"], dtype=float)
        vvals = np.asarray(data["vvals"], dtype=float)
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        fig.patch.set_facecolor(WHITE_BG)
        ax.set_facecolor(WHITE_BG)
        ax.imshow(
            data["grid"],
            origin="lower",
            extent=[float(uvals[0]), float(uvals[-1]), float(vvals[0]), float(vvals[-1])],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="auto",
        )
        ax.scatter(*seed_by_plane[name], c=BASIN_SEED_COLOR, edgecolors="black", s=82, marker="*", zorder=5)
        ax.set_xlabel(data["xlabel"])
        ax.set_ylabel(data["ylabel"])
        ax.text(
            0.02, 0.98, f"fixed={data['fixed']:.4g}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#111827", "alpha": 0.82, "linewidth": 0.35},
        )
        class_handles = [
            Patch(facecolor=BASIN_CLASS_COLORS[i], edgecolor="black", linewidth=0.25, label=BASIN_CLASS_LABELS[i])
            for i in (1, 2, 3, 4, 0)
        ]
        ax.legend(handles=class_handles, loc="upper right", fontsize=7)
        style_white_axis(ax, grid=False)
        save_figure(fig, output_by_plane[name])


def compute_basin_volume(cfg: Dict[str, Any]) -> Dict[str, Any]:
    lib = load_basin_library(cfg)
    p = cfg["params"]
    b = cfg["basin_3d"]
    lib.set_chua_params(float(p["alpha_chua"]), float(p["beta"]), float(p["gamma_chua"]), float(p["m0"]), float(p["m1"]))

    nx, ny, nz = int(b["nx"]), int(b["ny"]), int(b["nz"])
    xvals = np.linspace(float(b["xlim"][0]), float(b["xlim"][1]), nx)
    yvals = np.linspace(float(b["ylim"][0]), float(b["ylim"][1]), ny)
    zvals = np.linspace(float(b["zlim"][0]), float(b["zlim"][1]), nz)
    volume = np.empty((nz, ny, nx), dtype=np.int32)
    total = nx * ny * nz
    log(f"Basin 3D C: {nx}x{ny}x{nz} = {total} trajectories, q=1.0.")

    for iz, z0 in enumerate(zvals, start=1):
        grid_out = np.empty(nx * ny, dtype=np.int32)
        rc = lib.compute_basin_plane(
            nx, ny,
            float(xvals[0]), float(xvals[-1]), float(yvals[0]), float(yvals[-1]),
            float(z0), 0, 1.0, float(b["h"]), float(b["Lm"]),
            float(b["TMAX"]), float(b["TBURN"]), float(b["R_DIV"]), float(b["R_BOUND"]),
            float(b["EPS_EQ"]), int(b["CAP_WIN"]), float(b["MEAN_X_GAP"]), grid_out,
        )
        if rc != 0:
            raise RuntimeError(f"compute_basin_plane(volume z={z0}) returned {rc}")
        volume[iz - 1] = grid_out.reshape((ny, nx))
        if iz == 1 or iz % max(1, nz // 10) == 0 or iz == nz:
            log(f"Basin 3D C: slice {iz}/{nz}.")

    return {"xvals": xvals, "yvals": yvals, "zvals": zvals, "volume": volume}


def plot_basin_volume_3d(cfg: Dict[str, Any], volume_data: Dict[str, Any], final_seed: np.ndarray) -> None:
    b = cfg["basin_3d"]
    xvals = np.asarray(volume_data["xvals"], dtype=float)
    yvals = np.asarray(volume_data["yvals"], dtype=float)
    zvals = np.asarray(volume_data["zvals"], dtype=float)
    volume = np.asarray(volume_data["volume"], dtype=np.int32)
    max_points = int(b.get("max_points_per_class", 18000))
    rng = np.random.default_rng(12345)

    fig = plt.figure(figsize=(8.2, 6.8))
    fig.patch.set_facecolor(WHITE_BG)
    ax = fig.add_subplot(111, projection="3d")
    draw_order = (3, 4, 1, 2, 0)
    alpha_by_class = {0: 0.45, 1: 0.75, 2: 0.75, 3: 0.08, 4: 0.20}
    size_by_class = {0: 2.0, 1: 3.0, 2: 3.0, 3: 1.0, 4: 1.4}

    for cls in draw_order:
        idx = np.argwhere(volume == cls)
        if idx.size == 0:
            continue
        if idx.shape[0] > max_points:
            keep = rng.choice(idx.shape[0], size=max_points, replace=False)
            idx = idx[keep]
        zs = zvals[idx[:, 0]]
        ys = yvals[idx[:, 1]]
        xs = xvals[idx[:, 2]]
        ax.scatter(
            xs, ys, zs,
            s=size_by_class[cls],
            c=BASIN_CLASS_COLORS[cls],
            alpha=alpha_by_class[cls],
            linewidths=0,
            depthshade=False,
            label=BASIN_CLASS_LABELS[cls],
        )

    seed = np.asarray(final_seed, dtype=float)
    ax.scatter(seed[0], seed[1], seed[2], s=100, c=BASIN_SEED_COLOR, edgecolors="black", marker="*", label="target seed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(float(xvals[0]), float(xvals[-1]))
    ax.set_ylim(float(yvals[0]), float(yvals[-1]))
    ax.set_zlim(float(zvals[0]), float(zvals[-1]))
    ax.view_init(elev=22, azim=-54)
    style_white_axis(ax, grid=True, grid_alpha=0.15)
    ax.legend(loc="upper left", fontsize=8, markerscale=3)
    save_figure(fig, cfg["outputs"]["basin_3d_pdf"])


def _signal_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name in {"hann", "hanning"}:
        return np.hanning(n)
    if name in {"none", "rect", "rectangular"}:
        return np.ones(n)
    raise ValueError(f"Unsupported spectral window: {name}")


def extract_spectral_series(final_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    traj = np.asarray(final_data["traj"], dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 4:
        raise ValueError("Final attractor trajectory must have columns t,x,y,z.")
    return {"t": traj[:, 0], "X": traj[:, 1:4]}


def top_spectrum_peaks(freq: np.ndarray, values: np.ndarray, top_n: int, start_idx: int) -> List[Dict[str, float]]:
    freq = np.asarray(freq, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size <= start_idx:
        return []
    order = np.argsort(values[start_idx:])[::-1][:max(1, int(top_n))] + start_idx
    return [
        {
            "frequency_hz": float(freq[idx]),
            "frequency_rad_s": float(2.0 * np.pi * freq[idx]),
            "value": float(values[idx]),
        }
        for idx in order
    ]


def infer_sample_step(t: np.ndarray, fallback: float) -> float:
    t = np.asarray(t, dtype=float)
    if t.size >= 2:
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            return float(np.median(diffs))
    return float(fallback)


def compute_fft_summary(spectral_data: Dict[str, np.ndarray], cfg: Dict[str, Any], h: float) -> Dict[str, Any]:
    X = np.asarray(spectral_data["X"], dtype=float)
    fft_cfg = cfg["fft"]
    comp_names = fft_cfg["component_names"]
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    summary_by_component: Dict[str, Any] = {}

    for j, name in enumerate(comp_names):
        signal = X[:, j] - np.mean(X[:, j])
        n = signal.size
        if n < 16:
            raise ValueError("The final trajectory is too short for FFT.")
        win = _signal_window(n, str(fft_cfg.get("window", "hann")))
        coherent_gain = float(np.sum(win) / n)
        yf = np.fft.rfft(signal * win)
        freq = np.fft.rfftfreq(n, d=float(h))
        amp = (2.0 / (n * max(coherent_gain, 1e-15))) * np.abs(yf)
        start_idx = 1 if fft_cfg.get("ignore_zero_bin", True) else 0
        kmax = int(np.argmax(amp[start_idx:]) + start_idx)
        top_peaks = top_spectrum_peaks(freq, amp, int(fft_cfg.get("top_n", 5)), start_idx)
        summary_by_component[name] = {
            "dominant_frequency_hz": float(freq[kmax]),
            "dominant_frequency_rad_s": float(2.0 * np.pi * freq[kmax]),
            "peak_amplitude": float(amp[kmax]),
            "n_samples": int(n),
            "window": str(fft_cfg.get("window", "hann")),
            "frequency_resolution_hz": float(freq[1] - freq[0]) if freq.size > 1 else float("nan"),
            "top_frequencies": top_peaks,
        }
        spectra[name] = (freq, amp)

    primary_name = comp_names[int(fft_cfg["primary_component_index"])]
    return {
        "spectra": spectra,
        "summary_by_component": summary_by_component,
        "primary_component": primary_name,
        "omega_d": summary_by_component[primary_name]["dominant_frequency_rad_s"],
        "f_d": summary_by_component[primary_name]["dominant_frequency_hz"],
    }


def _welch_component(
    signal: np.ndarray,
    h: float,
    nperseg_max: int,
    noverlap_fraction: float,
    window: str = "hann",
    scaling: str = "density",
) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    fs = 1.0 / float(h)
    n = len(signal)
    if n < 16:
        raise ValueError("The final trajectory is too short for Welch PSD.")

    nperseg = min(int(nperseg_max), n)
    noverlap = int(max(0, min(nperseg - 1, round(float(noverlap_fraction) * nperseg))))
    try:
        from scipy.signal import welch

        return welch(
            signal,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling=scaling,
            return_onesided=True,
        )
    except ImportError:
        win = _signal_window(nperseg, window)
        step = max(1, nperseg - noverlap)
        starts = list(range(0, n - nperseg + 1, step)) or [0]
        acc = None
        for start in starts:
            seg = signal[start:start + nperseg]
            if seg.size < nperseg:
                seg = np.pad(seg, (0, nperseg - seg.size))
            seg = seg - np.mean(seg)
            yf = np.fft.rfft(seg * win)
            pxx = (np.abs(yf) ** 2) / max(fs * np.sum(win ** 2), 1e-15)
            if pxx.size > 2:
                pxx[1:-1] *= 2.0
            if scaling != "density":
                pxx *= fs / max(nperseg, 1)
            acc = pxx if acc is None else acc + pxx
        freq = np.fft.rfftfreq(nperseg, d=float(h))
        return freq, acc / max(1, len(starts))


def compute_psd_summary(spectral_data: Dict[str, np.ndarray], cfg: Dict[str, Any], h: float) -> Dict[str, Any]:
    X = np.asarray(spectral_data["X"], dtype=float)
    psd_cfg = cfg["psd"]
    comp_names = psd_cfg["component_names"]
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    summary_by_component: Dict[str, Any] = {}
    for j, name in enumerate(comp_names):
        f, pxx = _welch_component(
            X[:, j],
            h=h,
            nperseg_max=int(psd_cfg["nperseg_max"]),
            noverlap_fraction=float(psd_cfg["noverlap_fraction"]),
            window=str(psd_cfg["welch_window"]),
            scaling=str(psd_cfg["welch_scaling"]),
        )
        start_idx = 1 if psd_cfg.get("ignore_zero_bin", True) else 0
        kmax = int(np.argmax(pxx[start_idx:]) + start_idx)
        summary_by_component[name] = {
            "dominant_frequency_hz": float(f[kmax]),
            "dominant_frequency_rad_s": float(2.0 * np.pi * f[kmax]),
            "peak_psd": float(pxx[kmax]),
            "n_samples": int(X[:, j].size),
        }
        spectra[name] = (f, pxx)

    primary_name = comp_names[int(psd_cfg["primary_component_index"])]
    return {
        "spectra": spectra,
        "summary_by_component": summary_by_component,
        "primary_component": primary_name,
        "omega_d": summary_by_component[primary_name]["dominant_frequency_rad_s"],
        "f_d": summary_by_component[primary_name]["dominant_frequency_hz"],
    }


def plot_fft_figures(cfg: Dict[str, Any], fft_data: Dict[str, Any], omega0: float) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    omega_focus = max(0.0, float(fft_data["omega_d"]))
    half_width = max(
        omega_focus * float(cfg["fft"].get("zoom_half_width_factor", 1.5)),
        float(omega0) * 0.35,
        1e-6,
    )
    xmin = max(0.0, omega_focus - half_width)
    xmax = omega_focus + half_width
    if np.isfinite(omega0):
        xmin = max(0.0, min(xmin, float(omega0) * 0.85))
        xmax = max(xmax, float(omega0) * 1.15)

    for name in cfg["fft"]["component_names"]:
        f, amp = fft_data["spectra"][name]
        omega = 2.0 * np.pi * f
        fig, ax = plt.subplots(figsize=(6.9, 4.6))
        ax.plot(omega, amp, lw=1.0, color=BIFURCATION_NEG_COLOR, label=f"FFT {name}")
        ax.axvline(omega0, color=NYQUIST_DF_COLOR, ls="--", lw=1.0, label=r"$\omega_0$ Nyquist/DF")
        for peak in fft_data["summary_by_component"][name].get("top_frequencies", [])[: int(cfg["fft"].get("top_n", 5))]:
            ax.scatter(peak["frequency_rad_s"], peak["value"], s=20, c=WHITE_BG, edgecolors="#222222", linewidths=0.45, zorder=4)
        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel(f"|FFT({name})|")
        ax.set_xlim(xmin, xmax)
        style_white_axis(ax, grid=True, grid_alpha=0.24)
        ax.legend(loc="best", fontsize=8)
        key = f"fft_{name}_pdf"
        paths[key] = str(cfg["outputs"][key])
        save_figure(fig, cfg["outputs"][key])
    return paths


def plot_psd_figures(cfg: Dict[str, Any], psd_data: Dict[str, Any], omega0: float) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for name in cfg["psd"]["component_names"]:
        f, pxx = psd_data["spectra"][name]
        fig, ax = plt.subplots(figsize=(6.9, 4.6))
        ax.semilogy(2.0 * np.pi * f, pxx, lw=1.0, color=NYQUIST_W_COLOR, label=f"PSD {name}")
        ax.axvline(omega0, color=NYQUIST_DF_COLOR, ls="--", lw=1.0, label=r"$\omega_0$ Nyquist/DF")
        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel(f"PSD({name})")
        style_white_axis(ax, grid=True, grid_alpha=0.24)
        ax.legend(loc="best", fontsize=8)
        key = f"psd_{name}_pdf"
        paths[key] = str(cfg["outputs"][key])
        save_figure(fig, cfg["outputs"][key])
    return paths


def run_spectral_analysis(cfg: Dict[str, Any], final_data: Dict[str, Any], seed: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("spectral", {}).get("enabled", True)):
        return {"enabled": False, "status": "skipped", "extra_figures": {}}
    spectral_data = extract_spectral_series(final_data)
    h = infer_sample_step(spectral_data["t"], float(final_data["config"].get("dt_output", 0.01)))
    omega0 = float(seed["omega0"])
    log("Spectral stage: direct FFT as the main spectrum; Welch PSD is optional.")
    fft_data = compute_fft_summary(spectral_data, cfg, h)
    fft_paths = plot_fft_figures(cfg, fft_data, omega0)
    omega_fft = float(fft_data["omega_d"])
    eta_fft = abs(omega_fft - omega0) / abs(omega0) if abs(omega0) > 0 else float("nan")

    psd_summary: Dict[str, Any] = {"enabled": False, "status": "skipped"}
    psd_paths: Dict[str, str] = {}
    if bool(cfg.get("psd", {}).get("enabled", False)):
        psd_data = compute_psd_summary(spectral_data, cfg, h)
        psd_paths = plot_psd_figures(cfg, psd_data, omega0)
        omega_psd = float(psd_data["omega_d"])
        eta_psd = abs(omega_psd - omega0) / abs(omega0) if abs(omega0) > 0 else float("nan")
        psd_summary = {
            "enabled": True,
            "omega_d_primary": omega_psd,
            "f_d_primary_hz": float(psd_data["f_d"]),
            "relative_mismatch_eta": float(eta_psd),
            "primary_component": psd_data["primary_component"],
            "summary_by_component": psd_data["summary_by_component"],
        }

    summary = {
        "enabled": True,
        "status": "ok",
        "sample_step": h,
        "omega0_nyquist_df": omega0,
        "fft_summary": {
            "omega_d_primary": omega_fft,
            "f_d_primary_hz": float(fft_data["f_d"]),
            "relative_mismatch_eta": float(eta_fft),
            "primary_component": fft_data["primary_component"],
            "summary_by_component": fft_data["summary_by_component"],
        },
        "psd_summary": psd_summary,
        "extra_figures": {**fft_paths, **psd_paths},
        "notes": [
            "The direct FFT is the primary spectral artifact and is zoomed around the dominant frequencies.",
            "Welch PSD is optional; it smooths power over windows and is not used as a proof of hiddenness.",
        ],
    }
    Path(cfg["outputs"]["spectral_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def compile_lyapunov_binary(cfg: Dict[str, Any]) -> Path:
    le_cfg = cfg["lyapunov"]
    src = Path(le_cfg["source_c"]).resolve()
    exe = Path(le_cfg["exe"]).resolve()
    exe.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"No existe el backend C de Lyapunov: {src}")
    cmd = ["gcc", "-O3", str(src), "-lm", "-o", str(exe)]
    log(f"Lyapunov C: compiling {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return exe


def parse_lyapunov_stdout(stdout: str) -> Tuple[List[float], List[float] | None]:
    le: List[float] | None = None
    final_state: List[float] | None = None
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] == "#" and parts[1] == "LE_frac_standard":
            le = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif len(parts) >= 5 and parts[0] == "#" and parts[1] == "final_state":
            final_state = [float(parts[2]), float(parts[3]), float(parts[4])]
    if le is None:
        raise RuntimeError("No se pudieron extraer los exponentes de Lyapunov de la salida del backend C.")
    return le, final_state


def plot_lyapunov_convergence(csv_path: Path, pdf_path: Path) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(data["time"], data["lambda1"], lw=1.2, color=BIFURCATION_POS_COLOR, label=r"$\lambda_1$")
    ax.plot(data["time"], data["lambda2"], lw=1.2, color=NYQUIST_W_COLOR, label=r"$\lambda_2$")
    ax.plot(data["time"], data["lambda3"], lw=1.2, color=BIFURCATION_NEG_COLOR, label=r"$\lambda_3$")
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("Lyapunov exponent estimate")
    style_white_axis(ax)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, pdf_path)


def run_lyapunov_estimate(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    le_cfg = cfg["lyapunov"]
    if not bool(le_cfg.get("enabled", False)):
        return {"enabled": False, "status": "skipped"}

    try:
        exe = compile_lyapunov_binary(cfg)
        csv_path = Path(cfg["outputs"]["lyapunov_csv"])
        env = dict(os.environ)
        env["CHUA_LE_CSV"] = str(csv_path)
        x0 = [float(v) for v in np.asarray(final_state, dtype=float)]
        p = cfg["params"]
        args = [
            str(exe),
            str(x0[0]), str(x0[1]), str(x0[2]),
            str(float(p["alpha_chua"])), str(float(p["beta"])), str(float(p["gamma_chua"])),
            str(float(p["m0"])), str(float(p["m1"])),
            "1.0",
            str(float(le_cfg["h"])), str(float(le_cfg["Lm"])),
            str(float(le_cfg["t_burn"])), str(int(le_cfg["n_blocks"])), str(float(le_cfg["t_block"])),
        ]
        log(
            "Lyapunov C: running "
            f"q=1.0, h={le_cfg['h']}, burn={le_cfg['t_burn']}, "
            f"blocks={le_cfg['n_blocks']}x{le_cfg['t_block']}."
        )
        proc = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
        le, reported_final_state = parse_lyapunov_stdout(proc.stdout)
        plot_lyapunov_convergence(csv_path, Path(cfg["outputs"]["lyapunov_pdf"]))
        summary = {
            "enabled": True,
            "status": "ok",
            "method": "EFORK-3 q=1.0 C backend chua_frac_lyapunov_efork_benettin.c",
            "interpretation": "Operational integer-order Benettin/variational estimate using the same EFORK q=1 numerical process as the integer pipeline.",
            "initial_state": x0,
            "lyapunov_exponents": le,
            "final_state_reported": reported_final_state,
            "config": {
                "q": 1.0,
                "h": float(le_cfg["h"]),
                "Lm": float(le_cfg["Lm"]),
                "t_burn": float(le_cfg["t_burn"]),
                "n_blocks": int(le_cfg["n_blocks"]),
                "t_block": float(le_cfg["t_block"]),
                "source_c": str(le_cfg["source_c"]),
                "exe": str(exe),
            },
            "outputs": {
                "csv": str(csv_path),
                "pdf": str(cfg["outputs"]["lyapunov_pdf"]),
            },
            "raw_stdout": proc.stdout,
            "notes": [
                "The C file name is fractional because it is shared with the fractional pipeline.",
                "Here the backend is called with q=1.0 for the integer-order Chua run.",
                "This is consistent with continuation, final-attractor, comparison, bifurcation and hiddenness-illustration trajectories, which also use EFORK q=1.",
                "The exponents are numerical evidence and should be interpreted together with attractor, basin and spectral figures.",
            ],
        }
        Path(cfg["outputs"]["lyapunov_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    except Exception as exc:
        summary = {
            "enabled": True,
            "status": "failed",
            "error": str(exc),
            "strict": bool(le_cfg.get("strict", False)),
        }
        Path(cfg["outputs"]["lyapunov_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if bool(le_cfg.get("strict", False)):
            raise
        log(f"Lyapunov C: failed but non-strict mode continues: {exc}")
        return summary


def local_maxima(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    if s.size < 3:
        return np.array([], dtype=float)
    idx = np.nonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1
    return s[idx]


def integrate_bifurcation_ode(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_final: float,
    dt_output: float,
    rtol: float,
    atol: float,
    div_threshold: float,
) -> Tuple[np.ndarray | None, str]:
    x0 = np.asarray(x0, dtype=float)
    if not np.all(np.isfinite(x0)) or np.linalg.norm(x0) >= div_threshold:
        return None, "bad_seed"

    def guarded_rhs(x: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 10.0 * div_threshold:
            raise FloatingPointError("state exceeded guarded divergence threshold")
        y = np.asarray(rhs(x), dtype=float)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError("rhs returned non-finite values")
        return y

    try:
        traj, status = efork_q1_integrate(
            guarded_rhs,
            x0,
            t_final,
            dt_output,
            div_threshold=div_threshold,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return None, f"solver_exception:{exc}"

    _ = (rtol, atol)  # EFORK q=1 is fixed-step; retained to keep the call contract explicit.
    if traj.size == 0:
        return None, "empty_solution"
    if not np.all(np.isfinite(traj)):
        return None, "nonfinite_solution"
    if status == "diverged":
        return traj, "diverged"
    if status != "ok":
        return traj, status
    return traj, "ok"


def compute_bifurcation_sweep(param_name: str, values: Iterable[float], seed_pos: np.ndarray, seed_neg: np.ndarray, seed: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    vals = np.asarray(list(values), dtype=float)
    bcfg = cfg["bifurcation"]
    c = cfg["continuation"]
    progress_every = max(1, int(bcfg.get("progress_every", max(1, len(vals) // 10))))
    pts_pos_x: list[float] = []
    pts_pos_y: list[float] = []
    pts_neg_x: list[float] = []
    pts_neg_y: list[float] = []
    current_pos = np.asarray(seed_pos, dtype=float).copy()
    current_neg = np.asarray(seed_neg, dtype=float).copy()
    failures: list[dict[str, Any]] = []
    log(f"Bifurcation {param_name}: {len(vals)} values x 2 seeds, t_total={bcfg['t_total']}.")
    for idx, value in enumerate(vals, start=1):
        if idx == 1 or idx % progress_every == 0 or idx == len(vals):
            log(f"Bifurcation {param_name}: {idx}/{len(vals)} = {value:.8g}")
        params = dict(cfg["params"])
        if param_name == "alpha":
            params["alpha_chua"] = float(value)
        elif param_name == "beta":
            params["beta"] = float(value)
        else:
            raise ValueError("param_name must be alpha or beta")

        for branch_name, current_seed, fallback_seed, xs, ys in (
            ("pos", current_pos, seed_pos, pts_pos_x, pts_pos_y),
            ("neg", current_neg, seed_neg, pts_neg_x, pts_neg_y),
        ):
            rhs = lambda x, pp=params: rhs_original_direct(x, pp)
            traj, status = integrate_bifurcation_ode(
                rhs,
                current_seed,
                float(bcfg["t_total"]),
                float(bcfg["dt_output"]),
                float(c["rtol"]),
                float(c["atol"]),
                float(bcfg["div_threshold"]),
            )
            if traj is None or status != "ok":
                failures.append({
                    "parameter": param_name,
                    "value": float(value),
                    "branch": branch_name,
                    "status": status,
                })
                current_seed[:] = np.asarray(fallback_seed, dtype=float)
                continue
            current_seed[:] = traj[-1, 1:4]
            tail = traj[traj[:, 0] >= float(bcfg["t_burn"])]
            if tail.size == 0:
                continue
            peaks = local_maxima(tail[:, 1])
            if peaks.size == 0:
                peaks = np.array([float(np.max(tail[:, 1]))], dtype=float)
            peaks = peaks[-int(bcfg["max_peaks"]):]
            xs.extend([float(value)] * len(peaks))
            ys.extend(peaks.tolist())
    return {
        "values": vals,
        "pos_x": np.asarray(pts_pos_x, dtype=float),
        "pos_y": np.asarray(pts_pos_y, dtype=float),
        "neg_x": np.asarray(pts_neg_x, dtype=float),
        "neg_y": np.asarray(pts_neg_y, dtype=float),
        "failures": failures,
    }


def plot_bifurcation(data: Dict[str, np.ndarray], xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(data["pos_x"], data["pos_y"], s=0.65, c=BIFURCATION_POS_COLOR, alpha=0.86, linewidths=0, rasterized=True, label="semilla +")
    ax.scatter(data["neg_x"], data["neg_y"], s=0.65, c=BIFURCATION_NEG_COLOR, alpha=0.86, linewidths=0, rasterized=True, label="semilla -")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$x_{\max}$")
    xs = np.concatenate([np.asarray(data["pos_x"], dtype=float), np.asarray(data["neg_x"], dtype=float)])
    ys = np.concatenate([np.asarray(data["pos_y"], dtype=float), np.asarray(data["neg_y"], dtype=float)])
    mask = np.isfinite(xs) & np.isfinite(ys)
    if np.any(mask):
        xmin, xmax = float(np.min(xs[mask])), float(np.max(xs[mask]))
        ymin, ymax = float(np.min(ys[mask])), float(np.max(ys[mask]))
        xpad = 0.015 * max(1e-12, xmax - xmin)
        ypad = 0.06 * max(1e-12, ymax - ymin)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
    style_white_axis(ax, grid=True, grid_alpha=0.22)
    ax.legend(loc="best", markerscale=4, fontsize=8)
    save_figure(fig, path)


def run_bifurcations(cfg: Dict[str, Any], final_seed: np.ndarray, seed: Dict[str, Any]) -> Dict[str, Any]:
    bcfg = cfg["bifurcation"]
    seed_pos = np.asarray(final_seed, dtype=float)
    seed_neg = -seed_pos
    alpha_data = compute_bifurcation_sweep("alpha", bcfg["alpha_values"], seed_pos, seed_neg, seed, cfg)
    plot_bifurcation(alpha_data, r"$\alpha$", cfg["outputs"]["bif_alpha_pdf"])
    beta_data = compute_bifurcation_sweep("beta", bcfg["beta_values"], seed_pos, seed_neg, seed, cfg)
    plot_bifurcation(beta_data, r"$\beta$", cfg["outputs"]["bif_beta_pdf"])
    alpha_failures = alpha_data.get("failures", [])
    beta_failures = beta_data.get("failures", [])
    if alpha_failures or beta_failures:
        log(
            "Bifurcation: skipped divergent/failed integrations "
            f"(alpha={len(alpha_failures)}, beta={len(beta_failures)})."
        )
    summary = {
        "config": {
            "alpha_values": [float(v) for v in bcfg["alpha_values"]],
            "beta_values": [float(v) for v in bcfg["beta_values"]],
            "t_total": float(bcfg["t_total"]),
            "t_burn": float(bcfg["t_burn"]),
            "dt_output": float(bcfg["dt_output"]),
            "div_threshold": float(bcfg["div_threshold"]),
            "max_peaks": int(bcfg["max_peaks"]),
        },
        "alpha_points": int(len(alpha_data["pos_x"]) + len(alpha_data["neg_x"])),
        "beta_points": int(len(beta_data["pos_x"]) + len(beta_data["neg_x"])),
        "alpha_failures": alpha_failures,
        "beta_failures": beta_failures,
        "figures": {
            "alpha": str(cfg["outputs"]["bif_alpha_pdf"]),
            "beta": str(cfg["outputs"]["bif_beta_pdf"]),
        },
    }
    Path(cfg["outputs"]["bif_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Integer-order Chua hidden-attractor pipeline.")
    parser.add_argument("--seed-only", action="store_true", help="Only run the MATLAB-equivalent seed checks and seed plots.")
    parser.add_argument("--skip-hidden", action="store_true", help="Skip C hiddenness verification.")
    parser.add_argument("--skip-basin", action="store_true", help="Skip C basin computation.")
    parser.add_argument("--skip-basin-planes", action="store_true", help="Skip C xy/xz/yz basin cuts.")
    parser.add_argument("--basin-3d", action="store_true", help="Force computation of the sampled 3D basin volume.")
    parser.add_argument("--skip-basin-3d", action="store_true", help="Skip the sampled 3D basin volume even if enabled by config.")
    parser.add_argument("--skip-bifurcation", action="store_true", help="Skip alpha and beta bifurcation diagrams.")
    args = parser.parse_args()

    skip_heavy = _env_flag("HIDDEN_ATTRACTORS_STYLE_ONLY", False)

    log(f"Output root: {RUNTIME_ROOT}")
    seed = build_integer_seed(CONFIG)
    plot_seed_figures(CONFIG, seed)
    comparison = plot_linear_vs_original(CONFIG, seed)

    if args.seed_only:
        summary = {
            "run_mode": CONFIG["run_mode"],
            "output_root": str(RUNTIME_ROOT),
            "runtime_contract": CONFIG.get("runtime_contract", {}),
            "seed_summary": str(CONFIG["outputs"]["seed_json"]),
            "comparison": comparison,
        }
        Path(CONFIG["outputs"]["summary_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log("Seed-only run complete.")
        return

    results = continuation_in_epsilon(CONFIG, seed)
    plot_continuation_figures(CONFIG, results)
    plot_continuation_story(CONFIG, results)
    continuation_final_state = np.asarray(results[-1]["x_out"], dtype=float)
    final_data = integrate_final_attractor(CONFIG, seed, continuation_final_state)
    plot_final_attractor(CONFIG, final_data)
    plot_final_attractor_planes(CONFIG, final_data)
    final_state = np.asarray(final_data["target_seed"], dtype=float)
    spectral_summary = run_spectral_analysis(CONFIG, final_data, seed)
    lyapunov_summary = run_lyapunov_estimate(CONFIG, final_state)

    hidden_summary = None
    hidden_illustration_summary = None
    if not args.skip_hidden and not skip_heavy:
        hidden_summary = run_hidden_verification(CONFIG, final_state)
        hidden_illustration_summary = plot_hiddenness_illustration(CONFIG, final_data, hidden_summary)

    basin_info = None
    basin_planes_info = None
    basin_3d_info = None
    if not args.skip_basin and not skip_heavy:
        z_attractor = float(final_state[2])
        basin_z0, E0, Ep, Em = compute_basin(CONFIG, z0=0.0)
        plot_basin_overlay(CONFIG, basin_z0, E0, Ep, Em, final_state, path=CONFIG["outputs"]["basin_z0_pdf"], z0=0.0)
        CONFIG["basin"]["z0"] = z_attractor
        basin, E0, Ep, Em = compute_basin(CONFIG, z0=z_attractor)
        save_distances_csv(CONFIG, final_state, E0, Ep, Em)
        plot_basin_overlay(CONFIG, basin, E0, Ep, Em, final_state, path=CONFIG["outputs"]["basin_zfinal_pdf"], z0=z_attractor)
        basin_info = {
            "z0": {
                "z": 0.0,
                "counts": basin_class_counts(basin_z0),
                "figure": str(CONFIG["outputs"]["basin_z0_pdf"]),
            },
            "z_attractor": {
                "z": z_attractor,
                "counts": basin_class_counts(basin),
                "figure": str(CONFIG["outputs"]["basin_zfinal_pdf"]),
            },
        }
        if CONFIG["basin_planes"].get("enabled", True) and not args.skip_basin_planes:
            basin_planes = compute_basin_planes(CONFIG, final_state)
            plot_basin_planes(CONFIG, basin_planes, final_state)
            basin_planes_info = {
                name: {
                    "shape": list(np.asarray(data["grid"]).shape),
                    "fixed": float(data["fixed"]),
                    "counts": {str(int(k)): int(v) for k, v in zip(*np.unique(data["grid"], return_counts=True))},
                    "figure": str(CONFIG["outputs"][f"basin_{name}_pdf"]),
                }
                for name, data in basin_planes.items()
            }
        if (args.basin_3d or CONFIG["basin_3d"].get("enabled", False)) and not args.skip_basin_3d:
            basin_volume = compute_basin_volume(CONFIG)
            plot_basin_volume_3d(CONFIG, basin_volume, final_state)
            vol = np.asarray(basin_volume["volume"], dtype=np.int32)
            unique, counts = np.unique(vol, return_counts=True)
            basin_3d_info = {
                "shape": list(vol.shape),
                "counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
                "figure": str(CONFIG["outputs"]["basin_3d_pdf"]),
                "config": {
                    key: value for key, value in CONFIG["basin_3d"].items()
                    if key not in {"enabled"}
                },
            }

    bifurcation_summary = None
    if CONFIG["bifurcation"].get("enabled", True) and not args.skip_bifurcation and not skip_heavy:
        bifurcation_summary = run_bifurcations(CONFIG, final_state, seed)

    pipeline_summary = {
        "run_mode": CONFIG["run_mode"],
        "output_root": str(RUNTIME_ROOT),
        "runtime_contract": CONFIG.get("runtime_contract", {}),
        "params": CONFIG["params"],
        "order": "integer",
        "frac_order_passed_to_c_backends": 1.0,
        "seed": seed["summary"],
        "continuation": {
            "final_state_eps1": continuation_final_state.tolist(),
            "summary_json": str(CONFIG["outputs"]["cont_json"]),
        },
        "final_attractor": {
            "target_seed_after_burn": final_state.tolist(),
            "config": final_data["config"],
            "trajectory_points": int(np.asarray(final_data["traj"]).shape[0]),
        },
        "linear_vs_original": comparison,
        "spectral": spectral_summary,
        "lyapunov": lyapunov_summary,
        "hidden_verification": hidden_summary,
        "hiddenness_illustration": hidden_illustration_summary,
        "basin_counts": basin_info,
        "basin_plane_counts": basin_planes_info,
        "basin_3d_counts": basin_3d_info,
        "bifurcation": bifurcation_summary,
        "figures_dir": str(OUTDIR),
    }
    Path(CONFIG["outputs"]["summary_json"]).write_text(json.dumps(pipeline_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("Pipeline complete.")
    print("Resumen:", CONFIG["outputs"]["summary_json"], flush=True)


if __name__ == "__main__":
    main()
