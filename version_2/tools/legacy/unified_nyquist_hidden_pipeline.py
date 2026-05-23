#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_nyquist_hidden_pipeline.py

Pipeline unificado para Chua fraccionario:

    Nyquist/funciÃ³n descriptiva -> semilla armÃ³nica -> continuaciÃ³n en epsilon
    -> verificaciÃ³n de ocultedad con backend C (EFORK + fingerprint seccional)
    -> corte operacional de cuenca en plano z=z0 -> grÃ¡ficas finales en PDF.

Este script NO modifica los archivos fuente adjuntos por la usuaria. Reutiliza:
- chua_initial_cond.py
- run_hidden_verify_frac_hybrid.py
- chua_basin_lib.c
- chua_hidden_backend.c

Referencias matemÃ¡ticas del pipeline implementado
-----------------------------------------------
[1] Gelb & Vander Velde, Multiple-Input Describing Functions and Nonlinear System Design.
    Base clÃ¡sica de funciÃ³n descriptiva y balance armÃ³nico.
[2] Leonov, Kuznetsov et al., trabajos sobre localizaciÃ³n de hidden attractors en Chua
    mediante harmonic linearization / describing function y continuaciÃ³n numÃ©rica.
[3] Danca (2018), Hidden chaotic attractors in fractional-order systems.
    DefiniciÃ³n computacional de hidden attractor y verificaciÃ³n con vecindades de equilibrios.
[4] ViguÃ© et al. (2019), Continuation of periodic solutions for systems with fractional derivatives.
    La formulaciÃ³n periÃ³dica exacta se hace en Weyl, no directamente en Caputo.
[5] Haacker et al. (2025), Hill-type stability analysis...
    El ansatz exponencial no cierra el lado estable en Caputo; la validaciÃ³n final debe ser causal.

Notas de interpretaciÃ³n
-----------------------
- La etapa Nyquist/DF produce una oscilaciÃ³n dominante candidata, no una prueba de ciclo lÃ­mite exacto de Caputo.
- La continuaciÃ³n en epsilon transporta esa semilla hacia el sistema objetivo.
- La verificaciÃ³n final de ocultedad se hace por integraciÃ³n causal y muestreo en vecindades de equilibrio.
- La secciÃ³n x=0, xdot>0 se usa como huella geomÃ©trica (fingerprint), no como mapa de retorno exacto.

Salida principal
----------------
Se crea la carpeta ./final_pdf_figs con PDFs separados, todos SIN tÃ­tulo y con nombres en ejes:
- fig01_nyquist_df.pdf
- fig02_continuation_progress.pdf
- fig03_final_attractor.pdf
- fig04_reference_section.pdf
- fig05_probe_summary.pdf
- fig06_basin_overlay.pdf

AdemÃ¡s se guardan:
- unified_pipeline_summary.json
- unified_continuation_summary.json
- seed_equilibrium_distances_unified.csv

No ejecuta nada por sÃ­ solo fuera de lo que le pidas como usuaria.
"""

from __future__ import annotations

import csv
import argparse
import copy
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import ctypes
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

from parallel_policy import (
    compile_c_target,
    force_single_openmp_thread_current_process,
    parallel_contract,
)


ROOT = Path(__file__).resolve().parent

PROJECT_SLUG = "hidden_attractors_fractional_order"


def _bootstrap_cli_options_into_env(argv: List[str] | None = None) -> None:
    """Compatibility bridge: explicit CLI options replace PowerShell env setup.

    The legacy script still reads its historical ``HIDDEN_ATTRACTORS_*`` keys in
    the configuration layer.  This bootstrap is intentionally kept at the very
    top so users and library wrappers can call the script with normal arguments
    while old environment-based runs remain reproducible.
    """

    parser = argparse.ArgumentParser(
        description="Unified fractional Chua pipeline. Prefer explicit options over HIDDEN_ATTRACTORS_* environment variables.",
        add_help=True,
    )
    parser.add_argument("--ignore-env", action="store_true", help="Clear existing HIDDEN_ATTRACTORS_* variables before applying CLI options.")
    value_options = {
        "output_dir": ("--output-dir", "HIDDEN_ATTRACTORS_OUTPUT_DIR"),
        "model": ("--model", "HIDDEN_ATTRACTORS_MODEL"),
        "run_mode": ("--run-mode", "HIDDEN_ATTRACTORS_RUN_MODE"),
        "q": ("--q", "HIDDEN_ATTRACTORS_FRAC_ORDER"),
        "q_values": ("--q-values", "HIDDEN_ATTRACTORS_Q_VALUES"),
        "h": ("--h", "HIDDEN_ATTRACTORS_H"),
        "memory_length": ("--memory-length", "HIDDEN_ATTRACTORS_LM"),
        "t_transient": ("--t-transient", "HIDDEN_ATTRACTORS_T_TRANSIENT"),
        "t_keep": ("--t-keep", "HIDDEN_ATTRACTORS_T_KEEP"),
        "basin_grid": ("--basin-grid", "HIDDEN_ATTRACTORS_BASIN_GRID"),
        "basin_planes_grid": ("--basin-planes-grid", "HIDDEN_ATTRACTORS_BASIN_PLANES_GRID"),
        "basin_z0": ("--basin-z0", "HIDDEN_ATTRACTORS_BASIN_Z0"),
        "basin_workers": ("--basin-workers", "HIDDEN_ATTRACTORS_BASIN_WORKERS"),
        "bif_workers": ("--bif-workers", "HIDDEN_ATTRACTORS_BIF_WORKERS"),
        "native_efork_workers": ("--native-efork-workers", "HIDDEN_ATTRACTORS_NATIVE_EFORK_WORKERS"),
        "verify_nsamples": ("--verify-nsamples", "HIDDEN_ATTRACTORS_VERIFY_NSAMPLES"),
        "verify_radii": ("--verify-radii", "HIDDEN_ATTRACTORS_VERIFY_RADII"),
        "machado_mu": ("--machado-mu", "HIDDEN_ATTRACTORS_MACHADO_MU"),
        "machado_mu_values": ("--machado-mu-values", "HIDDEN_ATTRACTORS_MACHADO_MU_VALUES"),
        "machado_sweep_max_candidates": ("--machado-sweep-max-candidates", "HIDDEN_ATTRACTORS_MACHADO_SWEEP_MAX_CANDIDATES"),
        "df_compare_branch_index": ("--df-compare-branch-index", "HIDDEN_ATTRACTORS_DF_COMPARE_BRANCH_INDEX"),
    }
    for dest, (flag, _) in value_options.items():
        parser.add_argument(flag, dest=dest)
    bool_options = {
        "spectral": "HIDDEN_ATTRACTORS_SPECTRAL",
        "psd": "HIDDEN_ATTRACTORS_PSD",
        "tisean": "HIDDEN_ATTRACTORS_TISEAN",
        "lyapunov": "HIDDEN_ATTRACTORS_LYAPUNOV",
        "lyapunov_strict": "HIDDEN_ATTRACTORS_LYAPUNOV_STRICT",
        "bifurcation": "HIDDEN_ATTRACTORS_BIFURCATION",
        "basin_planes": "HIDDEN_ATTRACTORS_BASIN_PLANES",
        "hidden_illustration": "HIDDEN_ATTRACTORS_HIDDEN_ILLUSTRATION",
        "style_only": "HIDDEN_ATTRACTORS_STYLE_ONLY",
        "native_efork": "HIDDEN_ATTRACTORS_NATIVE_EFORK",
    }
    for dest in bool_options:
        parser.add_argument(f"--{dest.replace('_', '-')}", dest=dest, default=None, action=argparse.BooleanOptionalAction)

    args, _ = parser.parse_known_args(argv)
    if args.ignore_env:
        for key in list(os.environ):
            if key.startswith("HIDDEN_ATTRACTORS_"):
                os.environ.pop(key, None)
    for dest, (_, env_name) in value_options.items():
        value = getattr(args, dest)
        if value is not None:
            os.environ[env_name] = str(value)
    for dest, env_name in bool_options.items():
        value = getattr(args, dest)
        if value is not None:
            os.environ[env_name] = "1" if bool(value) else "0"


_bootstrap_cli_options_into_env(sys.argv[1:])


# CONFIGURACION DE CARPETAS DE CORRIDA
# Cambia solo RUNS_BASE_DIR_DEFAULT si quieres elegir la carpeta base desde el
# codigo. Tambien puedes sobrescribirla sin editar el archivo con:
#   HIDDEN_ATTRACTORS_OUTPUT_DIR=C:\Users\moren\Desktop\mis_corridas
#
# Estructura usada:
#   RUNS_BASE_DIR / chua_nonsmooth / ...
#   RUNS_BASE_DIR / chua_arctan    / ...
RUNS_BASE_DIR_DEFAULT = ROOT
RUNS_MODEL_SUBDIR_ENABLED = True
RUNS_BASE_DIR_ENV = os.environ.get("HIDDEN_ATTRACTORS_OUTPUT_DIR")
OUTPUT_ROOT_SOURCE = "unresolved"


def normalize_chua_model(raw: Any) -> str:
    text = str(raw or "nonsmooth").strip().lower().replace("-", "_")
    aliases = {
        "piecewise": "nonsmooth",
        "pwl": "nonsmooth",
        "nonsmooth": "nonsmooth",
        "non_smooth": "nonsmooth",
        "no_suave": "nonsmooth",
        "tramos": "nonsmooth",
        "piecewise_linear": "nonsmooth",
        "atan": "arctan",
        "arc_tan": "arctan",
        "smooth": "arctan",
        "suave": "arctan",
    }
    text = aliases.get(text, text)
    if text not in {"nonsmooth", "arctan"}:
        raise ValueError("HIDDEN_ATTRACTORS_MODEL debe ser 'nonsmooth' o 'arctan'.")
    return text


CHUA_MODEL_KIND = normalize_chua_model(
    os.environ.get("HIDDEN_ATTRACTORS_MODEL", os.environ.get("HIDDEN_ATTRACTORS_CHUA_MODEL", "nonsmooth"))
)
CHUA_MODEL_SLUGS = {
    "nonsmooth": "chua_nonsmooth",
    "arctan": "chua_arctan",
}
CHUA_OUTPUT_SLUG = CHUA_MODEL_SLUGS[CHUA_MODEL_KIND]


def resolve_user_base_dir(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


RUNS_BASE_DIR = resolve_user_base_dir(RUNS_BASE_DIR_ENV or RUNS_BASE_DIR_DEFAULT)


def run_root_from_base(base_dir: Path) -> Path:
    return base_dir / CHUA_OUTPUT_SLUG if RUNS_MODEL_SUBDIR_ENABLED else base_dir


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
    global OUTPUT_ROOT_SOURCE
    configured_root = run_root_from_base(RUNS_BASE_DIR)
    if _dir_accepts_writes(configured_root):
        OUTPUT_ROOT_SOURCE = "HIDDEN_ATTRACTORS_OUTPUT_DIR" if RUNS_BASE_DIR_ENV else "RUNS_BASE_DIR_DEFAULT"
        return configured_root

    if RUNS_BASE_DIR_ENV:
        raise PermissionError(
            "No se pudo escribir en HIDDEN_ATTRACTORS_OUTPUT_DIR="
            f"{RUNS_BASE_DIR}. Corrige esa carpeta o quita la variable de entorno."
        )

    fallback_root = Path(tempfile.gettempdir()) / PROJECT_SLUG / CHUA_OUTPUT_SLUG
    if _dir_accepts_writes(fallback_root):
        OUTPUT_ROOT_SOURCE = "temp_fallback"
        return fallback_root

    raise PermissionError(
        "No se encontro un directorio con permisos de escritura para los resultados. "
        "Define RUNS_BASE_DIR_DEFAULT al inicio del script o HIDDEN_ATTRACTORS_OUTPUT_DIR."
    )


RUNTIME_ROOT = resolve_runtime_root()
HIDDEN_VERIFY_DIR = RUNTIME_ROOT / "hidden_verify"
NATIVE_DIR = RUNTIME_ROOT / "native"
OUTDIR = RUNTIME_ROOT / "final_pdf_figs"
for _runtime_dir in (RUNTIME_ROOT, HIDDEN_VERIFY_DIR, NATIVE_DIR, OUTDIR):
    _runtime_dir.mkdir(parents=True, exist_ok=True)

REGISTERED_DLL_DIRS = set()
DLL_SEARCH_HANDLES = []
BASIN_LIBRARY_CACHE = None
FRACTIONAL_BACKEND_CACHE = None


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    if minutes >= 1.0:
        return f"{int(minutes)}m {sec:04.1f}s"
    return f"{sec:.1f}s"


def output_layout_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "output_root_source": OUTPUT_ROOT_SOURCE,
        "runs_base_dir": str(RUNS_BASE_DIR),
        "model": cfg.get("model", {}),
        "run_root": str(RUNTIME_ROOT),
        "folders": {
            "figures": str(OUTDIR),
            "hidden_verification": str(HIDDEN_VERIFY_DIR),
            "native_backends": str(NATIVE_DIR),
            "q_sweep": str(cfg.get("q_sweep", {}).get("output_dir", RUNTIME_ROOT / "q_order_sweep")),
            "df_compare": str(cfg.get("df_compare", {}).get("output_dir", RUNTIME_ROOT / "df_seed_comparison")),
            "machado_sweep": str(cfg.get("machado_sweep", {}).get("output_dir", RUNTIME_ROOT / "machado_sweep")),
            "machado_sweep_fast": str(cfg.get("machado_sweep_fast", {}).get("output_dir", RUNTIME_ROOT / "machado_sweep_fast")),
        },
        "root_files": {
            "pipeline_summary": str(cfg["outputs"]["summary_json"]),
            "continuation_summary": str(cfg["outputs"]["cont_json"]),
            "basin_only_summary": str(cfg["outputs"]["basin_only_summary_json"]),
            "article_style_summary": str(cfg["outputs"]["article_style_summary_json"]),
            "hiddenness_illustration_summary": str(cfg["outputs"]["hidden_illustration_json"]),
            "psd_summary": str(cfg["outputs"]["psd_json"]),
            "lyapunov_summary": str(cfg["outputs"]["lyapunov_json"]),
        },
        "notes": [
            "Todos los archivos de una corrida deben quedar dentro de run_root.",
            "Las figuras finales van en final_pdf_figs.",
            "La prueba de ocultedad escribe config, CSV, JSON y PNG auxiliares en hidden_verify.",
            "Los binarios C compilados para esta corrida van en native.",
        ],
    }


def write_output_layout_files(cfg: Dict[str, Any]) -> Dict[str, str]:
    layout = output_layout_dict(cfg)
    json_path = RUNTIME_ROOT / "run_output_layout.json"
    txt_path = RUNTIME_ROOT / "CARPETAS_CORRIDA.txt"
    json_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "Carpetas de esta corrida",
        "=========================",
        f"fuente_configuracion: {layout['output_root_source']}",
        f"carpeta_base_corridas: {layout['runs_base_dir']}",
        f"carpeta_corrida: {layout['run_root']}",
        "",
        "Subcarpetas",
        "-----------",
    ]
    for name, path in layout["folders"].items():
        lines.append(f"{name}: {path}")
    lines.extend([
        "",
        "Archivos principales",
        "--------------------",
    ])
    for name, path in layout["root_files"].items():
        lines.append(f"{name}: {path}")
    lines.extend([
        "",
        "Regla: si una corrida es no suave, todo queda bajo chua_nonsmooth; "
        "si es arctan, todo queda bajo chua_arctan.",
    ])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "txt": str(txt_path)}


def log_output_layout(layout_files: Dict[str, str]) -> None:
    log("Carpetas de esta corrida:")
    log(f"  base_corridas={RUNS_BASE_DIR}")
    log(f"  corrida={RUNTIME_ROOT}")
    log(f"  figuras={OUTDIR}")
    log(f"  ocultedad={HIDDEN_VERIFY_DIR}")
    log(f"  nativos={NATIVE_DIR}")
    log(f"  mapa={layout_files['txt']}")


def validate_fractional_order(qord: float) -> float:
    qord = float(qord)
    if not np.isfinite(qord) or not (0.0 < qord <= 1.0):
        raise ValueError("El orden fraccionario q debe cumplir 0 < q <= 1.")
    return qord


BASIN_CLASS_COLORS = ["#111827", "#7c3aed", "#ffd43b", "#4b5563", "#38bdf8"]
BASIN_SEED_COLOR = "#ffd35a"
BIFURCATION_POS_COLOR = "#ff3b1f"
BIFURCATION_NEG_COLOR = "#d27bff"
LINEARIZED_COLOR = "#9467bd"
ORIGINAL_COLOR = "#d62728"
NYQUIST_W_COLOR = "#0047ff"
NYQUIST_DF_COLOR = "#ff4a1a"
WHITE_BG = "#ffffff"
BASIN_CLASS_LABELS = [
    "Equilibrio",
    "Candidato x>0",
    "Candidato x<0",
    "Divergente",
    "No clasificado",
]

DEFAULT_EPS_VALUES = [round(0.1 * i, 10) for i in range(1, 11)]


def basin_cmap_norm() -> Tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(BASIN_CLASS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    return cmap, norm


def add_basin_legend(ax, *, loc: str = "upper right", fontsize: int = 7):
    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.35, label=label)
        for color, label in zip(BASIN_CLASS_COLORS, BASIN_CLASS_LABELS)
    ]
    return ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.88)


def basin_class_label(class_id: int) -> str:
    idx = int(class_id)
    if 0 <= idx < len(BASIN_CLASS_LABELS):
        return BASIN_CLASS_LABELS[idx]
    return f"Clase {idx}"


CONFIG: Dict[str, Any] = {
    "model": {
        "kind": CHUA_MODEL_KIND,
        "output_slug": CHUA_OUTPUT_SLUG,
    },
    "params": {
        # Caso principal: Chua fraccionario no suave con saturacion.
        "alpha_chua": 8.4562,
        "beta": 12.0732,
        "gamma_chua": 0.0052,
        "m0": -0.1768,
        "m1": -1.1468,
        "a1": 0.4,
        "a2": -1.5585,
        "rho": 1.0,
    },
    "frac_order": 0.9998,
    "branch_index": 0,
    "continuation": {
        "eps_values": DEFAULT_EPS_VALUES.copy(),
        "h": 0.01,
        "Lm": 8.0,
        "t_transient": 150.0,
        "t_keep": 120.0,
        "memory_mode": "window",
        "memory_update_source": "observed",
    },
    "verify_hidden": {
        "config_path": "config_hidden_verify_frac.json",
        "override_backend_params": True,
        "h": 0.01,
        "Lm": 20.0,
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
        "NSAMPLES_PER_RADIUS": 40,
        "random_seed": 123456789,
    },
    "basin": {
        "source": "chua_basin_lib.c",
        "openmp": True,
        "parallel": True,
        "workers": max(1, min(8, os.cpu_count() or 1)),
        "nx": 360,
        "ny": 360,
        "xmin": -8.0,
        "xmax": 8.0,
        "ymin": -8.0,
        "ymax": 8.0,
        "z0": "final_state",
        "q": 0.9998,
        "h": 0.01,
        "Lm": 10.0,
        "TMAX": 270.0,
        "TBURN": 150.0,
        "R_DIV": 80.0,
        "R_BOUND": 30.0,
        "EPS_EQ": 0.03,
        "CAP_WIN": 150,
        "MEAN_X_GAP": 0.08,
    },
    "native_efork": {
        "enabled": True,
        "source": "chua_frac_backend_lib.c",
        "openmp": True,
        "workers": max(1, min(8, os.cpu_count() or 1)),
    },
    "outputs": {
        "nyquist_pdf": OUTDIR / "fig01_nyquist_df.pdf",
        "cont_progress_pdf": OUTDIR / "fig02_continuation_progress.pdf",
        "final_attr_pdf": OUTDIR / "fig03_final_attractor.pdf",
        "ref_section_pdf": OUTDIR / "fig04_reference_section.pdf",
        "probe_summary_pdf": OUTDIR / "fig05_probe_summary.pdf",
        "basin_pdf": OUTDIR / "fig06_basin_overlay.pdf",
        "cont_json": RUNTIME_ROOT / "unified_continuation_summary.json",
        "summary_json": RUNTIME_ROOT / "unified_pipeline_summary.json",
        "dist_csv": RUNTIME_ROOT / "seed_equilibrium_distances_unified.csv",
    },
}

CONFIG["article_style"] = {
    "enabled": True,
    "basin_planes_enabled": True,
    # Los cortes de cuenca se calculan con backend C; no hay fallback pesado en Python.
    "max_cont_curves": 6,
}

CONFIG["q_sweep"] = {
    "output_dir": RUNTIME_ROOT / "q_order_sweep",
    "q_values": [0.94, 0.955, 0.970, 0.980, 0.985, 0.990, 0.995, 0.9998, 1.0],
    "continue_on_error": True,
    "trajectory_csv": True,
}

CONFIG["df_compare"] = {
    "output_dir": RUNTIME_ROOT / "df_seed_comparison",
    "machado_mu_values": [0.5],
    "continue_on_error": True,
}

THETA_GRID = [float(v) for v in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)]

MACHADO_SWEEP = {
    0: {
        "mu_values": [0.50, 0.75, 1.00, 1.10, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00],
        "theta_values": THETA_GRID,
    },
    1: {
        "mu_values": [0.25, 0.50, 0.75, 0.90, 1.00, 1.05, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40],
        "theta_values": THETA_GRID,
    },
}

MACHADO_SWEEP_FAST = {
    0: {
        "mu_values": [0.75, 1.00, 1.25, 1.50, 2.00, 3.00],
        "theta_values": [0.0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3],
    },
    1: {
        "mu_values": [0.50, 0.75, 1.00, 1.10, 1.25, 1.40],
        "theta_values": [0.0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3],
    },
}

CONFIG["machado_sweep"] = {
    "output_dir": RUNTIME_ROOT / "machado_sweep",
    "grid": copy.deepcopy(MACHADO_SWEEP),
    "continue_on_error": True,
    "trajectory_csv": True,
    "max_candidates": None,
}

CONFIG["machado_sweep_fast"] = {
    "output_dir": RUNTIME_ROOT / "machado_sweep_fast",
    "grid": copy.deepcopy(MACHADO_SWEEP_FAST),
    "continue_on_error": True,
    "trajectory_csv": True,
    "max_candidates": None,
}

CONFIG["bifurcation"] = {
    "enabled": True,
    "q_values": np.linspace(0.80, 1.0, 241).tolist(),
    "alpha_values": np.linspace(8.2, 8.7, 261).tolist(),
    "beta_values": np.linspace(11.5, 12.5, 261).tolist(),
    "h": 0.01,
    "Lm": 8.0,
    "t_total": 270.0,
    "t_burn": 150.0,
    "max_peaks": 500,
    "div_threshold": 120.0,
    "progress_every": 5,
    "parallel": True,
    "workers": max(1, min(8, os.cpu_count() or 1)),
    "chunksize": 1,
    "seed_strategy": "continuation",
}

CONFIG["basin_python"] = {
    "nx": 180,
    "ny": 180,
    "xlim": (-20.0, 20.0),
    "ylim": (-20.0, 20.0),
    "zlim": (-30.0, 30.0),
    "h": 0.01,
    "Lm": 8.0,
    "t_total": 120.0,
    "t_burn": 60.0,
    "div_threshold": 120.0,
    "eq_tol": 0.15,
    "hidden_amp_threshold": 1.0,
    "progress_rows": 5,
}

CONFIG["spectral"] = {
    "enabled": True,
}

CONFIG["psd"] = {
    "enabled": False,
    "component_names": ["x", "y", "z"],
    "primary_component_index": 0,
    "welch_window": "hann",
    "welch_scaling": "density",
    "nperseg_max": 4096,
    "noverlap_fraction": 0.5,
    "ignore_zero_bin": True,
}

CONFIG["fft"] = {
    "component_names": ["x", "y", "z"],
    "primary_component_index": 0,
    "window": "hann",
    "ignore_zero_bin": True,
    "top_n": 5,
    "zoom_half_width_factor": 1.5,
    "nyquist_peak_window_fraction": 0.25,
}

CONFIG["tisean"] = {
    "enabled": False,
    "embedding_dim": 6,
    "delay": 10,
    "theiler_window": 50,
    "horizon": 100,
    "radius": 2.0,
    "commands": ["lyap_r", "lyap_k", "lyap_spec"],
}

CONFIG["lyapunov"] = {
    "enabled": False,
    "strict": False,
    "source_c": ROOT / "chua_frac_lyapunov_efork_benettin.c",
    "exe": NATIVE_DIR / ("chua_frac_lyapunov_efork_benettin.exe" if os.name == "nt" else "chua_frac_lyapunov_efork_benettin"),
    "h": 0.01,
    "Lm": 20.0,
    "t_burn": 100.0,
    "n_blocks": 500,
    "t_block": 0.5,
}

CONFIG["hidden_illustration"] = {
    "enabled": True,
    "max_probe_trajectories": 200,
    "max_points_per_probe": 240,
    "max_attractor_points": 3600,
    "t_total": 120.0,
    "zoom_pad_fraction": 0.12,
    "random_seed": 246813579,
}

CONFIG["outputs"].update({
    "transfer_reim_pdf": OUTDIR / "fig01a_transferencia_ReIm.pdf",
    "nyquist_zoom_pdf": OUTDIR / "fig01b_nyquist_zoom_x.pdf",
    "matignon_equilibria_pdf": OUTDIR / "fig04_matignon_equilibria.pdf",
    "linear_vs_original_pdf": OUTDIR / "fig03b_linearized_vs_original.pdf",
    "continuation_story_pdf": OUTDIR / "fig02b_continuation_story.pdf",
    "final_attr_planes_pdf": OUTDIR / "fig03c_final_attractor_planes.pdf",
    "final_attr_xy_pdf": OUTDIR / "fig03c_final_attractor_xy.pdf",
    "final_attr_xz_pdf": OUTDIR / "fig03d_final_attractor_xz.pdf",
    "final_attr_yz_pdf": OUTDIR / "fig03e_final_attractor_yz.pdf",
    "hidden_illustration_overview_pdf": OUTDIR / "fig05b_hiddenness_overview.pdf",
    "hidden_illustration_zoom_pdf": OUTDIR / "fig05c_hiddenness_zoom.pdf",
    "hidden_illustration_json": RUNTIME_ROOT / "hiddenness_illustration_summary.json",
    "bif_q_pdf": OUTDIR / "fig07_bifurcation_q.pdf",
    "bif_alpha_pdf": OUTDIR / "fig08_bifurcation_alpha.pdf",
    "bif_beta_pdf": OUTDIR / "fig09_bifurcation_beta.pdf",
    "basin_planes_pdf": OUTDIR / "fig10_basin_planes.pdf",
    "basin_xy_pdf": OUTDIR / "fig10a_basin_xy.pdf",
    "basin_xz_pdf": OUTDIR / "fig10b_basin_xz.pdf",
    "basin_yz_pdf": OUTDIR / "fig10c_basin_yz.pdf",
    "psd_pdf": OUTDIR / "fig11_psd_welch.pdf",
    "fft_pdf": OUTDIR / "fig11b_fft_spectrum.pdf",
    "fft_x_pdf": OUTDIR / "fig11a_fft_x.pdf",
    "fft_y_pdf": OUTDIR / "fig11b_fft_y.pdf",
    "fft_z_pdf": OUTDIR / "fig11c_fft_z.pdf",
    "psd_json": RUNTIME_ROOT / "unified_psd_summary.json",
    "ts_x_txt": RUNTIME_ROOT / "final_timeseries_x.txt",
    "ts_xyz_txt": RUNTIME_ROOT / "final_timeseries_xyz.txt",
    "tisean_cmds": RUNTIME_ROOT / "tisean_commands.txt",
    "lyapunov_csv": RUNTIME_ROOT / "chua_frac_le_convergence.csv",
    "lyapunov_json": RUNTIME_ROOT / "lyapunov_frac_summary.json",
    "lyapunov_pdf": OUTDIR / "fig12_lyapunov_convergence.pdf",
    "article_style_summary_json": RUNTIME_ROOT / "article_style_plots_summary.json",
    "basin_only_summary_json": RUNTIME_ROOT / "basin_only_summary.json",
})


ENV_ALIASES: Dict[str, Tuple[str, ...]] = {
    "HIDDEN_ATTRACTORS_FRAC_ORDER": (
        "HIDDEN_ATTRACTORS_Q",
        "HIDDEN_ATTRACTORS_QORD",
        "HIDDEN_ATTRACTORS_FRAC_ORDER_Q",
    ),
    "HIDDEN_ATTRACTORS_T_TRANSIENT": (
        "HIDDEN_ATTRACTORS_T_TRANSIENTE",
        "HIDDEN_ATTRACTORS_TRANSIENT",
        "HIDDEN_ATTRACTORS_TRANSIENTE",
        "HIDDEN_ATTRACTORS_TBURN",
        "HIDDEN_ATTRACTORS_T_BURN",
    ),
    "HIDDEN_ATTRACTORS_T_KEEP": (
        "HIDDEN_ATTRACTORS_KEEP",
        "HIDDEN_ATTRACTORS_TKEEP",
        "HIDDEN_ATTRACTORS_T_POST",
    ),
}

CANONICAL_ENV_NAMES: Tuple[str, ...] = (
    "HIDDEN_ATTRACTORS_OUTPUT_DIR",
    "HIDDEN_ATTRACTORS_MODEL",
    "HIDDEN_ATTRACTORS_CHUA_MODEL",
    "HIDDEN_ATTRACTORS_RUN_MODE",
    "HIDDEN_ATTRACTORS_Q_SWEEP",
    "HIDDEN_ATTRACTORS_Q_VALUES",
    "HIDDEN_ATTRACTORS_MACHADO_MU",
    "HIDDEN_ATTRACTORS_MACHADO_MU_VALUES",
    "HIDDEN_ATTRACTORS_MACHADO_SWEEP_MAX_CANDIDATES",
    "HIDDEN_ATTRACTORS_DF_COMPARE_BRANCH_INDEX",
    "HIDDEN_ATTRACTORS_FRAC_ORDER",
    "HIDDEN_ATTRACTORS_EPS_VALUES",
    "HIDDEN_ATTRACTORS_H",
    "HIDDEN_ATTRACTORS_LM",
    "HIDDEN_ATTRACTORS_T_TRANSIENT",
    "HIDDEN_ATTRACTORS_T_KEEP",
    "HIDDEN_ATTRACTORS_BASIN_T_KEEP",
    "HIDDEN_ATTRACTORS_BIF_T_KEEP",
    "HIDDEN_ATTRACTORS_BIFURCATION",
    "HIDDEN_ATTRACTORS_BASIN_GRID",
    "HIDDEN_ATTRACTORS_BASIN_NX",
    "HIDDEN_ATTRACTORS_BASIN_NY",
    "HIDDEN_ATTRACTORS_BASIN_PLANES_GRID",
    "HIDDEN_ATTRACTORS_BASIN_PLANES_NX",
    "HIDDEN_ATTRACTORS_BASIN_PLANES_NY",
    "HIDDEN_ATTRACTORS_BASIN_Z0",
    "HIDDEN_ATTRACTORS_BASIN_WORKERS",
    "HIDDEN_ATTRACTORS_BASIN_PARALLEL",
    "HIDDEN_ATTRACTORS_VERIFY_NSAMPLES",
    "HIDDEN_ATTRACTORS_VERIFY_RADII",
    "HIDDEN_ATTRACTORS_VERIFY_TEST_MAX_SEC",
    "HIDDEN_ATTRACTORS_STYLE_ONLY",
    "HIDDEN_ATTRACTORS_BASIN_PLANES",
    "HIDDEN_ATTRACTORS_SPECTRAL",
    "HIDDEN_ATTRACTORS_PSD",
    "HIDDEN_ATTRACTORS_TISEAN",
    "HIDDEN_ATTRACTORS_LYAPUNOV",
    "HIDDEN_ATTRACTORS_LYAPUNOV_STRICT",
    "HIDDEN_ATTRACTORS_BIF_WORKERS",
    "HIDDEN_ATTRACTORS_BIF_PARALLEL",
    "HIDDEN_ATTRACTORS_BIF_SEED_STRATEGY",
    "ALLOW_NO_OPENMP",
    "HIDDEN_ATTRACTORS_ALPHA_CHUA",
    "HIDDEN_ATTRACTORS_BETA",
    "HIDDEN_ATTRACTORS_GAMMA_CHUA",
    "HIDDEN_ATTRACTORS_M0",
    "HIDDEN_ATTRACTORS_M1",
    "HIDDEN_ATTRACTORS_A1",
    "HIDDEN_ATTRACTORS_A2",
    "HIDDEN_ATTRACTORS_RHO",
)


def _env_lookup(name: str, aliases: Iterable[str] = ()) -> Tuple[str | None, str]:
    if name in os.environ:
        return os.environ.get(name), name
    for alias in aliases:
        if alias in os.environ:
            return os.environ.get(alias), alias
    return None, name


def _env_flag(name: str, default: bool = False, aliases: Iterable[str] = ()) -> bool:
    raw, _ = _env_lookup(name, aliases)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "si", "s"}


def _env_float(name: str, default: float, aliases: Iterable[str] = ()) -> float:
    raw, source_name = _env_lookup(name, aliases)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{source_name} debe ser numerico.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{source_name} debe ser finito.")
    return value


def _env_positive_float(name: str, default: float, aliases: Iterable[str] = ()) -> float:
    value = _env_float(name, default, aliases=aliases)
    if value <= 0.0:
        raise ValueError(f"{name} debe ser positivo.")
    return value


def _env_nonnegative_float(name: str, default: float, aliases: Iterable[str] = ()) -> float:
    value = _env_float(name, default, aliases=aliases)
    if value < 0.0:
        raise ValueError(f"{name} debe ser no negativo.")
    return value


def _env_int(name: str, default: int, aliases: Iterable[str] = ()) -> int:
    raw, source_name = _env_lookup(name, aliases)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{source_name} debe ser entero.") from exc
    return value


def _env_min_int(name: str, default: int, minimum: int, aliases: Iterable[str] = ()) -> int:
    value = _env_int(name, default, aliases=aliases)
    if value < minimum:
        raise ValueError(f"{name} debe ser >= {minimum}.")
    return value


def _parse_grid_env(name: str, default_nx: int, default_ny: int, minimum: int = 2) -> Tuple[int, int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default_nx), int(default_ny)
    text = raw.strip().lower().replace("×", "x")
    parts = [part for part in re.split(r"[x,;\s]+", text) if part]
    if len(parts) == 1:
        try:
            nx = ny = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{name} debe ser entero o tener formato NxM, por ejemplo 160 o 160x120.") from exc
    elif len(parts) == 2:
        try:
            nx, ny = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"{name} debe usar enteros en formato NxM, por ejemplo 160x120.") from exc
    else:
        raise ValueError(f"{name} debe ser entero o tener formato NxM, por ejemplo 160 o 160x120.")
    if nx < minimum or ny < minimum:
        raise ValueError(f"{name} debe tener dimensiones >= {minimum}.")
    return nx, ny


def _parse_float_sequence(raw: str, name: str) -> List[float]:
    raw = raw.strip()
    if raw.lower().startswith("linspace:"):
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError(f"{name} con formato linspace debe ser linspace:inicio:fin:n")
        start = float(parts[1])
        stop = float(parts[2])
        count = int(parts[3])
        if count < 2:
            raise ValueError(f"{name} necesita al menos dos valores en modo linspace")
        return [float(v) for v in np.linspace(start, stop, count)]

    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} no contiene valores numericos")
    return values


def _parse_positive_float_sequence(raw: str, name: str) -> List[float]:
    values = _parse_float_sequence(raw, name)
    if any((not np.isfinite(v)) or v <= 0.0 for v in values):
        raise ValueError(f"{name} debe contener valores positivos.")
    return values


def validate_eps_values(values: Iterable[float], name: str = "HIDDEN_ATTRACTORS_EPS_VALUES") -> List[float]:
    """Valida la malla de continuacion en epsilon hasta el sistema objetivo."""
    seq = [float(v) for v in values]
    if not seq:
        raise ValueError(f"{name} no puede estar vacio.")
    if any((not np.isfinite(v)) or v <= 0.0 or v > 1.0 for v in seq):
        raise ValueError(f"{name} debe contener valores finitos con 0 < eps <= 1.")
    if any(seq[i] <= seq[i - 1] for i in range(1, len(seq))):
        raise ValueError(f"{name} debe estar estrictamente creciente.")
    if not np.isclose(seq[-1], 1.0):
        raise ValueError(f"{name} debe terminar en 1.0 para llegar al sistema objetivo.")
    return seq


def apply_continuation_env_overrides(cfg: Dict[str, Any]) -> None:
    eps_values_env = os.environ.get("HIDDEN_ATTRACTORS_EPS_VALUES")
    if not eps_values_env:
        return
    cfg["continuation"]["eps_values"] = validate_eps_values(
        _parse_float_sequence(eps_values_env, "HIDDEN_ATTRACTORS_EPS_VALUES")
    )


def default_bifurcation_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def default_basin_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def normalize_bifurcation_seed_strategy(raw: Any) -> str:
    text = str(raw or "continuation").strip().lower().replace("-", "_")
    if text in {"continued", "continue", "causal", "secuencial", "sequential"}:
        text = "continuation"
    if text in {"independent", "independiente", "parallel"}:
        text = "independent"
    if text not in {"continuation", "independent"}:
        raise ValueError("bifurcation.seed_strategy debe ser 'continuation' o 'independent'.")
    return text


def refresh_parallel_runtime_contract(cfg: Dict[str, Any]) -> None:
    contract = cfg.setdefault("runtime_contract", {})
    previous_parallelism = dict(contract.get("parallelism", {}))
    basin = cfg.get("basin", {})
    bif = cfg.get("bifurcation", {})
    native = cfg.get("native_efork", {})
    seed_strategy = normalize_bifurcation_seed_strategy(bif.get("seed_strategy", "continuation"))
    bif_workers = int(bif.get("workers", 1))
    basin_workers = int(basin.get("workers", 1))
    basin_openmp_active = bool(basin.get("openmp_active", basin.get("openmp", False)))
    native_openmp_active = bool(native.get("openmp_active", native.get("openmp", False)))
    hidden_contract = previous_parallelism.get("hidden_verification")
    if not hidden_contract or hidden_contract.get("backend_openmp_active") == "external_backend_config":
        hidden_contract = {
            "python_workers": 1,
            "omp_threads": int(cfg.get("verify_hidden", {}).get("max_threads", 1)),
            "backend_openmp_active": "external_backend_config",
            "seed_strategy": "independent_equilibrium_samples",
            "stage_kind": "embarrassingly_parallel",
        }
    contract["parallelism"] = {
        "continuation": parallel_contract(
            python_workers=1,
            omp_threads=1,
            backend_openmp_active=False,
            seed_strategy="causal_memory_chain",
            stage_kind="causal",
        ),
        "hidden_verification": hidden_contract,
        "basin": parallel_contract(
            python_workers=1,
            omp_threads=basin_workers if basin_openmp_active else 1,
            backend_openmp_active=basin_openmp_active,
            seed_strategy="independent_grid_points",
            stage_kind="embarrassingly_parallel",
        ),
        "bifurcation": parallel_contract(
            python_workers=1,
            omp_threads=bif_workers if seed_strategy == "independent" and native_openmp_active else 1,
            backend_openmp_active=native_openmp_active,
            seed_strategy=seed_strategy,
            stage_kind="causal" if seed_strategy == "continuation" else "embarrassingly_parallel",
        ),
    }


def configure_basin_parallelism(cfg: Dict[str, Any]) -> None:
    bcfg = cfg["basin"]
    raw_workers = os.environ.get("HIDDEN_ATTRACTORS_BASIN_WORKERS")
    if raw_workers is None:
        workers = int(bcfg.get("workers", default_basin_workers()))
    else:
        try:
            workers = int(raw_workers)
        except ValueError as exc:
            raise ValueError("HIDDEN_ATTRACTORS_BASIN_WORKERS debe ser un entero positivo") from exc
    bcfg["workers"] = max(1, workers)
    bcfg["parallel"] = _env_flag(
        "HIDDEN_ATTRACTORS_BASIN_PARALLEL",
        bool(bcfg.get("parallel", bcfg["workers"] > 1)),
    )
    bcfg["openmp"] = bool(bcfg.get("openmp", True)) and bool(bcfg["parallel"]) and int(bcfg["workers"]) > 1
    refresh_parallel_runtime_contract(cfg)


def configure_bifurcation_parallelism(cfg: Dict[str, Any]) -> None:
    bcfg = cfg["bifurcation"]
    strategy = normalize_bifurcation_seed_strategy(
        os.environ.get("HIDDEN_ATTRACTORS_BIF_SEED_STRATEGY", bcfg.get("seed_strategy", "continuation"))
    )
    bcfg["seed_strategy"] = strategy
    raw_workers = os.environ.get("HIDDEN_ATTRACTORS_BIF_WORKERS")
    if raw_workers is None:
        workers = int(bcfg.get("workers", default_bifurcation_workers()))
    else:
        try:
            workers = int(raw_workers)
        except ValueError as exc:
            raise ValueError("HIDDEN_ATTRACTORS_BIF_WORKERS debe ser un entero positivo") from exc
    bcfg["workers"] = max(1, workers)
    bcfg["parallel"] = _env_flag(
        "HIDDEN_ATTRACTORS_BIF_PARALLEL",
        bool(bcfg.get("parallel", bcfg["workers"] > 1)),
    )
    if strategy == "continuation":
        bcfg["workers"] = 1
        bcfg["parallel"] = False
        bcfg["continue_seed"] = 1
    else:
        bcfg["continue_seed"] = 0
        bcfg["parallel"] = bool(bcfg["parallel"]) and int(bcfg["workers"]) > 1
    refresh_parallel_runtime_contract(cfg)


def _apply_system_parameter_env_overrides(cfg: Dict[str, Any]) -> None:
    cfg["model"]["kind"] = normalize_chua_model(cfg.get("model", {}).get("kind", CHUA_MODEL_KIND))
    env_to_key = {
        "HIDDEN_ATTRACTORS_ALPHA_CHUA": "alpha_chua",
        "HIDDEN_ATTRACTORS_BETA": "beta",
        "HIDDEN_ATTRACTORS_GAMMA_CHUA": "gamma_chua",
        "HIDDEN_ATTRACTORS_M0": "m0",
        "HIDDEN_ATTRACTORS_M1": "m1",
        "HIDDEN_ATTRACTORS_A1": "a1",
        "HIDDEN_ATTRACTORS_A2": "a2",
        "HIDDEN_ATTRACTORS_RHO": "rho",
    }
    for env_name, key in env_to_key.items():
        if os.environ.get(env_name) is None:
            continue
        value = _env_float(env_name, float(cfg["params"][key]))
        if key in {"alpha_chua", "beta"} and value <= 0.0:
            raise ValueError(f"{env_name} debe ser positivo para este modelo de Chua.")
        if key == "rho" and value <= 0.0:
            raise ValueError("HIDDEN_ATTRACTORS_RHO debe ser positivo.")
        cfg["params"][key] = value


def apply_chua_model_defaults(cfg: Dict[str, Any]) -> None:
    model = normalize_chua_model(cfg.get("model", {}).get("kind", CHUA_MODEL_KIND))
    cfg["model"]["kind"] = model
    cfg["model"]["output_slug"] = CHUA_MODEL_SLUGS[model]
    if model == "arctan":
        cfg["params"].update({
            "alpha_chua": 8.4562,
            "beta": 12.0732,
            "gamma_chua": 0.0052,
            "a1": 0.4,
            "a2": -1.5585,
            "rho": 1.0,
            # m0/m1 se conservan solo para compatibilidad con archivos antiguos.
            "m0": -0.1768,
            "m1": -1.1468,
        })
        cfg["frac_order"] = 0.99
        cfg["branch_index"] = 0


def _stage_keep_time(total: float, burn: float) -> float:
    return max(0.0, float(total) - float(burn))


def _apply_basin_plane_env_override(cfg: Dict[str, Any]) -> None:
    raw = os.environ.get("HIDDEN_ATTRACTORS_BASIN_Z0")
    if raw is None:
        return
    text = raw.strip().lower()
    if text in {"final", "final_state", "seed", "auto"}:
        cfg["basin"]["z0"] = "final_state"
    else:
        cfg["basin"]["z0"] = _env_float("HIDDEN_ATTRACTORS_BASIN_Z0", 0.0)


def _apply_basin_resolution_env_overrides(cfg: Dict[str, Any]) -> None:
    """Permite cambiar la resolucion de cuencas sin tocar el codigo fuente."""
    b = cfg["basin"]
    bp = cfg["basin_python"]

    main_grid = os.environ.get("HIDDEN_ATTRACTORS_BASIN_GRID")
    if main_grid is not None and main_grid.strip():
        b["nx"], b["ny"] = _parse_grid_env("HIDDEN_ATTRACTORS_BASIN_GRID", int(b["nx"]), int(b["ny"]), 2)
    b["nx"] = _env_min_int("HIDDEN_ATTRACTORS_BASIN_NX", int(b["nx"]), 2)
    b["ny"] = _env_min_int("HIDDEN_ATTRACTORS_BASIN_NY", int(b["ny"]), 2)

    plane_grid = os.environ.get("HIDDEN_ATTRACTORS_BASIN_PLANES_GRID")
    if plane_grid is not None and plane_grid.strip():
        bp["nx"], bp["ny"] = _parse_grid_env("HIDDEN_ATTRACTORS_BASIN_PLANES_GRID", int(bp["nx"]), int(bp["ny"]), 2)
    bp["nx"] = _env_min_int("HIDDEN_ATTRACTORS_BASIN_PLANES_NX", int(bp["nx"]), 2)
    bp["ny"] = _env_min_int("HIDDEN_ATTRACTORS_BASIN_PLANES_NY", int(bp["ny"]), 2)


def synchronize_runtime_contract(cfg: Dict[str, Any]) -> None:
    """Propaga el contrato numerico comun a las etapas de una misma corrida."""
    cfg["frac_order"] = validate_fractional_order(cfg["frac_order"])
    h = _env_positive_float("HIDDEN_ATTRACTORS_H", float(cfg["continuation"]["h"]))
    Lm = _env_positive_float("HIDDEN_ATTRACTORS_LM", float(cfg["continuation"]["Lm"]))
    t_transient = _env_nonnegative_float(
        "HIDDEN_ATTRACTORS_T_TRANSIENT",
        float(cfg["continuation"]["t_transient"]),
        aliases=ENV_ALIASES["HIDDEN_ATTRACTORS_T_TRANSIENT"],
    )
    t_keep = _env_positive_float(
        "HIDDEN_ATTRACTORS_T_KEEP",
        float(cfg["continuation"]["t_keep"]),
        aliases=ENV_ALIASES["HIDDEN_ATTRACTORS_T_KEEP"],
    )

    basin_existing_keep = _stage_keep_time(cfg["basin"]["TMAX"], cfg["basin"]["TBURN"])
    if basin_existing_keep <= 0.0:
        basin_existing_keep = t_keep
    basin_keep_default = min(t_keep, basin_existing_keep)
    basin_keep = _env_positive_float("HIDDEN_ATTRACTORS_BASIN_T_KEEP", basin_keep_default)

    bif_existing_keep = _stage_keep_time(cfg["bifurcation"]["t_total"], cfg["bifurcation"]["t_burn"])
    if bif_existing_keep <= 0.0:
        bif_existing_keep = t_keep
    bif_keep_default = min(t_keep, bif_existing_keep)
    bif_keep = _env_positive_float("HIDDEN_ATTRACTORS_BIF_T_KEEP", bif_keep_default)

    cfg["continuation"].update({
        "h": h,
        "Lm": Lm,
        "t_transient": t_transient,
        "t_keep": t_keep,
    })
    cfg["verify_hidden"].update({
        "h": h,
        "Lm": Lm,
        "TBURN_REF": t_transient,
        "TBURN_TEST": t_transient,
        "TMAX_REF": t_transient + t_keep,
        "TMAX_TEST": t_transient + t_keep,
    })
    verify_radii_env = os.environ.get("HIDDEN_ATTRACTORS_VERIFY_RADII")
    if verify_radii_env:
        cfg["verify_hidden"]["RADII"] = _parse_positive_float_sequence(
            verify_radii_env,
            "HIDDEN_ATTRACTORS_VERIFY_RADII",
        )
    cfg["verify_hidden"]["NSAMPLES_PER_RADIUS"] = _env_min_int(
        "HIDDEN_ATTRACTORS_VERIFY_NSAMPLES",
        int(cfg["verify_hidden"]["NSAMPLES_PER_RADIUS"]),
        1,
    )
    cfg["verify_hidden"]["TEST_MAX_SEC"] = _env_min_int(
        "HIDDEN_ATTRACTORS_VERIFY_TEST_MAX_SEC",
        int(cfg["verify_hidden"]["TEST_MAX_SEC"]),
        1,
    )
    cfg["basin"].update({
        "q": cfg["frac_order"],
        "h": h,
        "Lm": Lm,
        "TBURN": t_transient,
        "TMAX": t_transient + basin_keep,
    })
    cfg["basin_python"].update({
        "h": h,
        "Lm": Lm,
        "t_burn": t_transient,
        "t_total": t_transient + basin_keep,
    })
    cfg["bifurcation"].update({
        "h": h,
        "Lm": Lm,
        "t_burn": t_transient,
        "t_total": t_transient + bif_keep,
    })
    cfg["lyapunov"].update({
        "h": h,
        "Lm": Lm,
        "t_burn": t_transient,
    })
    cfg["hidden_illustration"].update({
        "h": h,
        "Lm": Lm,
    })
    _apply_basin_resolution_env_overrides(cfg)
    _apply_basin_plane_env_override(cfg)
    cfg["runtime_contract"] = {
        "model": cfg["model"]["kind"],
        "output_slug": cfg["model"]["output_slug"],
        "q": float(cfg["frac_order"]),
        "h": h,
        "Lm": Lm,
        "t_transient": t_transient,
        "t_keep": t_keep,
        "basin_t_keep": basin_keep,
        "bifurcation_t_keep": bif_keep,
        "basin_grid": [int(cfg["basin"]["nx"]), int(cfg["basin"]["ny"])],
        "basin_planes_grid": [int(cfg["basin_python"]["nx"]), int(cfg["basin_python"]["ny"])],
        "basin_z0": cfg["basin"].get("z0", "final_state"),
        "basin_workers": int(cfg["basin"].get("workers", 1)),
        "hidden_illustration": {
            "enabled": bool(cfg["hidden_illustration"].get("enabled", True)),
            "source": "actual hidden_target_check CSV from hidden verification",
            "max_probe_trajectories": int(cfg["hidden_illustration"]["max_probe_trajectories"]),
            "t_total": float(cfg["hidden_illustration"]["t_total"]),
        },
        "note": (
            "continuation, hidden verification, basins, bifurcation and Lyapunov "
            "share q/h/Lm and the same transient; basins and bifurcations may use "
            "shorter post-transient windows."
        ),
    }
    refresh_parallel_runtime_contract(cfg)


def _format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"


def _format_numeric_sequence(values: Iterable[Any], *, max_items: int = 12) -> str:
    seq = [float(v) for v in values]
    if not seq:
        return "[]"
    if len(seq) <= max_items:
        return "[" + ", ".join(_format_float(v) for v in seq) + "]"
    return (
        f"{len(seq)} valores desde {_format_float(seq[0])} hasta {_format_float(seq[-1])} "
        f"(primeros: {', '.join(_format_float(v) for v in seq[:3])}; "
        f"ultimos: {', '.join(_format_float(v) for v in seq[-3:])})"
    )


def runtime_env_overrides() -> Tuple[List[Dict[str, str]], List[str]]:
    recognized = set(CANONICAL_ENV_NAMES)
    for aliases in ENV_ALIASES.values():
        recognized.update(aliases)

    rows: List[Dict[str, str]] = []
    seen_names = set()
    for canonical in CANONICAL_ENV_NAMES:
        names = (canonical, *ENV_ALIASES.get(canonical, ()))
        for name in names:
            if name not in os.environ:
                continue
            rows.append({
                "canonical": canonical,
                "source": name,
                "value": os.environ.get(name, ""),
            })
            seen_names.add(name)
            break

    unknown = [
        name for name in sorted(os.environ)
        if name.startswith("HIDDEN_ATTRACTORS_")
        and name not in recognized
        and name not in seen_names
    ]
    return rows, unknown


def log_runtime_contract(cfg: Dict[str, Any], *, header: str = "Contrato numerico efectivo") -> None:
    """Imprime los parametros que realmente gobiernan la corrida."""
    contract = cfg.get("runtime_contract", {})
    cont = cfg["continuation"]
    verify = cfg["verify_hidden"]
    basin = cfg["basin"]
    bif = cfg["bifurcation"]

    log(header + ":")
    log("  " f"modelo={cfg['model']['kind']}; salida={cfg['model']['output_slug']}.")
    log(
        "  "
        f"q={_format_float(contract.get('q', cfg['frac_order']))}; "
        f"h={_format_float(contract.get('h', cont['h']))}; "
        f"Lm={_format_float(contract.get('Lm', cont['Lm']))}; "
        f"t_transient={_format_float(contract.get('t_transient', cont['t_transient']))}; "
        f"t_keep={_format_float(contract.get('t_keep', cont['t_keep']))}."
    )
    log(
        "  "
        f"continuacion: eps_values={_format_numeric_sequence(cont['eps_values'])}; "
        f"memory_mode={cont.get('memory_mode', 'window')}; "
        f"memory_update_source={cont.get('memory_update_source', 'observed')}."
    )
    log(
        "  "
        f"hidden verification: TBURN={_format_float(verify['TBURN_REF'])}; "
        f"TMAX={_format_float(verify['TMAX_REF'])}; "
        f"radii={len(verify['RADII'])}; muestras/radio={int(verify['NSAMPLES_PER_RADIUS'])}."
    )
    log(
        "  "
        f"cuenca: q={_format_float(basin['q'])}; "
        f"TBURN={_format_float(basin['TBURN'])}; "
        f"TMAX={_format_float(basin['TMAX'])}; "
        f"t_keep={_format_float(contract.get('basin_t_keep', _stage_keep_time(basin['TMAX'], basin['TBURN'])))}; "
        f"malla={int(basin['nx'])}x{int(basin['ny'])}; "
        f"z0={contract.get('basin_z0', basin.get('z0', 'final_state'))}; "
        f"workers={int(basin.get('workers', 1))}; OpenMP={bool(basin.get('openmp', False))}."
    )
    log(
        "  "
        f"cuencas de planos: malla={int(cfg['basin_python']['nx'])}x{int(cfg['basin_python']['ny'])}; "
        f"habilitadas={bool(cfg.get('article_style', {}).get('basin_planes_enabled', True))}."
    )
    log(
        "  "
        f"bifurcacion: t_burn={_format_float(bif['t_burn'])}; "
        f"t_total={_format_float(bif['t_total'])}; "
        f"t_keep={_format_float(contract.get('bifurcation_t_keep', _stage_keep_time(bif['t_total'], bif['t_burn'])))}; "
        f"q_values={len(bif['q_values'])}; alpha_values={len(bif['alpha_values'])}; "
        f"beta_values={len(bif['beta_values'])}; "
        f"seed_strategy={normalize_bifurcation_seed_strategy(bif.get('seed_strategy', 'continuation'))}."
    )
    native = cfg.get("native_efork", {})
    log(
        "  "
        f"EFORK nativo C: habilitado={bool(native.get('enabled', True))}; "
        f"workers_configurados={int(native.get('workers', 1))}; "
        f"OpenMP_solicitado={bool(native.get('openmp', True))}; "
        f"OpenMP_activo={native.get('openmp_active', 'pendiente_compilacion')}."
    )
    refresh_parallel_runtime_contract(cfg)
    parallelism = cfg.get("runtime_contract", {}).get("parallelism", {})
    for stage_name in ("continuation", "hidden_verification", "basin", "bifurcation"):
        pc = parallelism.get(stage_name, {})
        log(
            "  "
            f"paralelismo {stage_name}: "
            f"python_workers={pc.get('python_workers', 1)}; "
            f"omp_threads={pc.get('omp_threads', 1)}; "
            f"backend_openmp_active={pc.get('backend_openmp_active', False)}; "
            f"seed_strategy={pc.get('seed_strategy', 'not_applicable')}; "
            f"stage_kind={pc.get('stage_kind', 'unknown')}."
        )
    t_keep = float(contract.get("t_keep", cont["t_keep"]))
    basin_keep = float(contract.get("basin_t_keep", _stage_keep_time(basin["TMAX"], basin["TBURN"])))
    bif_keep = float(contract.get("bifurcation_t_keep", _stage_keep_time(bif["t_total"], bif["t_burn"])))
    shorter_stages = []
    if basin_keep != t_keep:
        shorter_stages.append("cuenca")
    if bif_keep != t_keep:
        shorter_stages.append("bifurcacion")
    if shorter_stages:
        log(
            "  "
            f"aviso: {', '.join(shorter_stages)} usa una ventana post-transitoria distinta de t_keep; "
            "para igualarla usa HIDDEN_ATTRACTORS_BASIN_T_KEEP y HIDDEN_ATTRACTORS_BIF_T_KEEP."
        )
    if cfg.get("run_mode") == "q_sweep":
        q_values = cfg.get("q_sweep", {}).get("q_values", [])
        log(
            "  "
            f"q_sweep: valores={_format_numeric_sequence(q_values)}; "
            "omite verificacion de ocultedad, cuencas, bifurcaciones y espectro completo."
        )
    if cfg.get("run_mode") == "df_compare":
        mu_values = cfg.get("df_compare", {}).get("machado_mu_values", [])
        log(
            "  "
            f"df_compare: mu_values={_format_numeric_sequence(mu_values)}; "
            "ejecuta DF clasica y Machado con continuacion + ocultedad, omite cuencas y bifurcaciones."
        )
    if cfg.get("run_mode") in {"machado_sweep", "machado_sweep_fast"}:
        sweep_key = str(cfg.get("run_mode"))
        grid = cfg.get(sweep_key, {}).get("grid", {})
        count = sum(len(v.get("mu_values", [])) * len(v.get("theta_values", [])) for v in grid.values())
        max_candidates = cfg.get(sweep_key, {}).get("max_candidates")
        suffix = "" if max_candidates is None else f"; limite={max_candidates}"
        log(
            "  "
            f"{sweep_key}: ramas={list(grid.keys())}; candidatos teoricos={count}{suffix}; "
            "ejecuta continuacion + ocultedad por semilla, omite cuencas y bifurcaciones."
        )


def log_environment_overrides() -> None:
    rows, unknown = runtime_env_overrides()
    if rows:
        log("Overrides de entorno recibidos por Python:")
        for row in rows:
            alias_note = "" if row["source"] == row["canonical"] else f" -> {row['canonical']}"
            log(f"  {row['source']}{alias_note}={row['value']!r}")
    else:
        log(
            "Overrides de entorno recibidos por Python: ninguno. "
            "En zsh/macOS usa 'export VAR=valor' o 'VAR=valor python3 ...'."
        )
    if unknown:
        log(
            "Aviso: variables HIDDEN_ATTRACTORS_* no reconocidas: "
            + ", ".join(unknown)
            + ". Revisa nombres como HIDDEN_ATTRACTORS_T_TRANSIENT y HIDDEN_ATTRACTORS_T_KEEP."
        )


def q_sweep_contract_line(cfg: Dict[str, Any]) -> str:
    contract = cfg.get("runtime_contract", {})
    cont = cfg["continuation"]
    return (
        f"q={_format_float(contract.get('q', cfg['frac_order']))}; "
        f"h={_format_float(contract.get('h', cont['h']))}; "
        f"Lm={_format_float(contract.get('Lm', cont['Lm']))}; "
        f"t_transient={_format_float(contract.get('t_transient', cont['t_transient']))}; "
        f"t_keep={_format_float(contract.get('t_keep', cont['t_keep']))}; "
        f"eps_values={_format_numeric_sequence(cont['eps_values'])}"
    )


def configure_runtime_profile(cfg: Dict[str, Any]) -> None:
    apply_chua_model_defaults(cfg)
    _apply_system_parameter_env_overrides(cfg)
    frac_order_env, frac_order_source = _env_lookup(
        "HIDDEN_ATTRACTORS_FRAC_ORDER",
        ENV_ALIASES["HIDDEN_ATTRACTORS_FRAC_ORDER"],
    )
    if frac_order_env is not None:
        try:
            cfg["frac_order"] = validate_fractional_order(float(frac_order_env))
        except ValueError as exc:
            raise ValueError(f"{frac_order_source} debe cumplir 0 < q <= 1.") from exc
    cfg["frac_order"] = validate_fractional_order(cfg["frac_order"])
    cfg["basin"]["q"] = cfg["frac_order"]
    run_mode = os.environ.get("HIDDEN_ATTRACTORS_RUN_MODE", "balanced").strip().lower()
    if _env_flag("HIDDEN_ATTRACTORS_Q_SWEEP", False):
        run_mode = "q_sweep"
    if run_mode not in {"full", "balanced", "fast", "quick", "test", "q_sweep", "sweep", "basin_only", "df_compare", "machado_compare", "machado_sweep", "machado_sweep_fast"}:
        raise ValueError("HIDDEN_ATTRACTORS_RUN_MODE debe ser 'full', 'balanced', 'fast', 'q_sweep', 'basin_only', 'df_compare', 'machado_sweep' o 'machado_sweep_fast'")

    cfg["run_mode"] = (
        "fast" if run_mode in {"fast", "quick", "test"}
        else ("q_sweep" if run_mode == "sweep"
              else ("df_compare" if run_mode == "machado_compare"
                    else run_mode))
    )
    cfg["style_only"] = _env_flag("HIDDEN_ATTRACTORS_STYLE_ONLY", False)
    cfg["article_style"]["basin_planes_enabled"] = _env_flag("HIDDEN_ATTRACTORS_BASIN_PLANES", True)
    if "HIDDEN_ATTRACTORS_PYTHON_BASIN" in os.environ:
        raise ValueError(
            "HIDDEN_ATTRACTORS_PYTHON_BASIN fue retirado. "
            "Los cortes de cuenca pesados deben ejecutarse con backend C; usa --basin-planes/--no-basin-planes."
        )
    cfg["spectral"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_SPECTRAL", True)
    cfg["psd"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_PSD", False)
    cfg["tisean"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_TISEAN", False)
    cfg["hidden_illustration"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_HIDDEN_ILLUSTRATION", True)
    cfg["lyapunov"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_LYAPUNOV", False)
    cfg["lyapunov"]["strict"] = _env_flag("HIDDEN_ATTRACTORS_LYAPUNOV_STRICT", False)
    cfg["bifurcation"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_BIFURCATION", True)
    cfg["native_efork"]["enabled"] = _env_flag("HIDDEN_ATTRACTORS_NATIVE_EFORK", True)
    cfg["native_efork"]["workers"] = _env_min_int(
        "HIDDEN_ATTRACTORS_NATIVE_EFORK_WORKERS",
        int(cfg["native_efork"].get("workers", default_bifurcation_workers())),
        1,
    )
    configure_basin_parallelism(cfg)

    q_values_env = os.environ.get("HIDDEN_ATTRACTORS_Q_VALUES")
    if q_values_env:
        cfg["q_sweep"]["q_values"] = [
            validate_fractional_order(v) for v in _parse_float_sequence(q_values_env, "HIDDEN_ATTRACTORS_Q_VALUES")
        ]

    mu_values_env = os.environ.get("HIDDEN_ATTRACTORS_MACHADO_MU_VALUES")
    if mu_values_env:
        cfg["df_compare"]["machado_mu_values"] = _parse_positive_float_sequence(
            mu_values_env,
            "HIDDEN_ATTRACTORS_MACHADO_MU_VALUES",
        )
    elif os.environ.get("HIDDEN_ATTRACTORS_MACHADO_MU"):
        cfg["df_compare"]["machado_mu_values"] = [
            _env_positive_float("HIDDEN_ATTRACTORS_MACHADO_MU", float(cfg["df_compare"]["machado_mu_values"][0]))
        ]

    if os.environ.get("HIDDEN_ATTRACTORS_DF_COMPARE_BRANCH_INDEX") is not None:
        cfg["df_compare"]["branch_index"] = _env_min_int("HIDDEN_ATTRACTORS_DF_COMPARE_BRANCH_INDEX", int(cfg["branch_index"]), 0)
    else:
        cfg["df_compare"]["branch_index"] = int(cfg["branch_index"])
    if os.environ.get("HIDDEN_ATTRACTORS_MACHADO_SWEEP_MAX_CANDIDATES") is not None:
        max_sweep_candidates = _env_min_int(
            "HIDDEN_ATTRACTORS_MACHADO_SWEEP_MAX_CANDIDATES",
            0,
            1,
        )
        cfg["machado_sweep"]["max_candidates"] = max_sweep_candidates
        cfg["machado_sweep_fast"]["max_candidates"] = max_sweep_candidates

    if cfg["run_mode"] == "q_sweep":
        cfg["spectral"]["enabled"] = False
        cfg["article_style"]["enabled"] = False
        cfg["continuation"].update({
            "eps_values": DEFAULT_EPS_VALUES.copy(),
            "t_transient": 100.0,
            "t_keep": 100.0,
        })
        apply_continuation_env_overrides(cfg)
        synchronize_runtime_contract(cfg)
        configure_basin_parallelism(cfg)
        return

    if cfg["run_mode"] in {"machado_sweep", "machado_sweep_fast"}:
        cfg["spectral"]["enabled"] = False
        cfg["article_style"]["enabled"] = False
        cfg["bifurcation"]["enabled"] = False
        cfg["continuation"].update({
            "eps_values": DEFAULT_EPS_VALUES.copy(),
            "Lm": 8.0,
            "t_transient": 80.0,
            "t_keep": 100.0,
        })
        cfg["verify_hidden"].update({
            "Lm": 8.0,
            "TMAX_REF": 180.0,
            "TMAX_TEST": 180.0,
            "TBURN_REF": 80.0,
            "TBURN_TEST": 80.0,
            "TEST_MAX_SEC": 80,
            "RADII": [1e-4, 1e-3, 1e-2],
            "NSAMPLES_PER_RADIUS": 4,
        })
        apply_continuation_env_overrides(cfg)
        synchronize_runtime_contract(cfg)
        configure_basin_parallelism(cfg)
        return

    if cfg["run_mode"] == "df_compare":
        cfg["spectral"]["enabled"] = False
        cfg["article_style"]["enabled"] = False
        cfg["bifurcation"]["enabled"] = False
        cfg["continuation"].update({
            "eps_values": DEFAULT_EPS_VALUES.copy(),
            "Lm": 8.0,
            "t_transient": 100.0,
            "t_keep": 100.0,
        })
        cfg["verify_hidden"].update({
            "Lm": 16.0,
            "TMAX_REF": 320.0,
            "TMAX_TEST": 320.0,
            "TBURN_REF": 90.0,
            "TBURN_TEST": 90.0,
            "TEST_MAX_SEC": 80,
            "RADII": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "NSAMPLES_PER_RADIUS": 24,
        })
        apply_continuation_env_overrides(cfg)
        synchronize_runtime_contract(cfg)
        configure_basin_parallelism(cfg)
        return

    if cfg["run_mode"] == "basin_only":
        cfg["spectral"]["enabled"] = False
        cfg["article_style"]["enabled"] = False
        synchronize_runtime_contract(cfg)
        configure_basin_parallelism(cfg)
        return

    if cfg["run_mode"] == "full":
        apply_continuation_env_overrides(cfg)
        synchronize_runtime_contract(cfg)
        configure_basin_parallelism(cfg)
        configure_bifurcation_parallelism(cfg)
        return

    if cfg["run_mode"] == "balanced":
        # Perfil por defecto: corrida completa con suficiente detalle para figuras,
        # pero evitando los barridos y cuencas mas pesados del modo full.
        cfg["continuation"].update({
            "eps_values": DEFAULT_EPS_VALUES.copy(),
            "Lm": 8.0,
            "t_transient": 100.0,
            "t_keep": 100.0,
        })
        cfg["verify_hidden"].update({
            "Lm": 16.0,
            "TMAX_REF": 320.0,
            "TMAX_TEST": 320.0,
            "TBURN_REF": 90.0,
            "TBURN_TEST": 90.0,
            "TEST_MAX_SEC": 80,
            "RADII": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "NSAMPLES_PER_RADIUS": 24,
        })
        cfg["basin"].update({
            "nx": 260,
            "ny": 260,
            "Lm": 8.0,
            "TMAX": 200.0,
            "TBURN": 100.0,
            "CAP_WIN": 140,
        })
        cfg["bifurcation"].update({
            "q_values": np.linspace(0.80, 1.0, 201).tolist(),
            "alpha_values": np.linspace(8.2, 8.7, 221).tolist(),
            "beta_values": np.linspace(11.5, 12.5, 221).tolist(),
            "Lm": 7.0,
            "t_total": 200.0,
            "t_burn": 100.0,
            "max_peaks": 500,
            "progress_every": 5,
        })
        cfg["basin_python"].update({
            "nx": 140,
            "ny": 140,
            "Lm": 7.0,
            "t_total": 200.0,
            "t_burn": 100.0,
            "progress_rows": 6,
        })
        cfg["hidden_illustration"].update({
            "t_total": 120.0,
            "max_probe_trajectories": 160,
            "max_points_per_probe": 220,
        })
        cfg["psd"].update({
            "nperseg_max": 4096,
        })
        apply_continuation_env_overrides(cfg)
        synchronize_runtime_contract(cfg)
        configure_basin_parallelism(cfg)
        configure_bifurcation_parallelism(cfg)
        return

    # Perfil de prueba: conserva todas las etapas importantes, pero baja resolucion,
    # tiempos de integracion y muestras para iterar sin esperar horas.
    cfg["continuation"].update({
        "eps_values": DEFAULT_EPS_VALUES.copy(),
        "Lm": 5.0,
        "t_transient": 80.0,
        "t_keep": 100.0,
    })
    cfg["verify_hidden"].update({
        "Lm": 8.0,
        "TMAX_REF": 100.0,
        "TMAX_TEST": 100.0,
        "TBURN_REF": 25.0,
        "TBURN_TEST": 25.0,
        "TEST_MAX_SEC": 35,
        "RADII": [1e-4, 1e-3, 1e-2],
        "NSAMPLES_PER_RADIUS": 8,
    })
    cfg["basin"].update({
        "nx": 120,
        "ny": 120,
        "Lm": 6.0,
        "TMAX": 180.0,
        "TBURN": 80.0,
        "CAP_WIN": 80,
    })
    cfg["bifurcation"].update({
        "q_values": np.linspace(0.80, 1.0, 31).tolist(),
        "alpha_values": np.linspace(8.25, 8.65, 41).tolist(),
        "beta_values": np.linspace(11.7, 12.4, 41).tolist(),
        "Lm": 5.0,
        "t_total": 180.0,
        "t_burn": 80.0,
        "max_peaks": 200,
        "progress_every": 1,
    })
    cfg["basin_python"].update({
        "nx": 60,
        "ny": 60,
        "Lm": 5.0,
        "t_total": 180.0,
        "t_burn": 80.0,
        "progress_rows": 2,
    })
    cfg["hidden_illustration"].update({
        "t_total": 45.0,
        "max_probe_trajectories": 60,
        "max_points_per_probe": 140,
        "max_attractor_points": 1600,
    })
    cfg["psd"].update({
        "nperseg_max": 1024,
    })
    cfg["lyapunov"].update({
        "Lm": 8.0,
        "t_burn": 20.0,
        "n_blocks": 40,
        "t_block": 0.25,
    })
    apply_continuation_env_overrides(cfg)
    synchronize_runtime_contract(cfg)
    configure_basin_parallelism(cfg)
    configure_bifurcation_parallelism(cfg)


configure_runtime_profile(CONFIG)


def _load_module(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el mÃ³dulo {mod_name} desde {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


chua_ic = _load_module("chua_initial_cond_mod", ROOT / "chua_initial_cond.py")
hv = _load_module("run_hidden_verify_frac_hybrid_mod", ROOT / "run_hidden_verify_frac_hybrid.py")


def hidden_verify_dir_for_config(cfg: Dict[str, Any]) -> Path:
    path = Path(cfg.get("verify_hidden", {}).get("runtime_dir", HIDDEN_VERIFY_DIR))
    if not path.is_absolute():
        path = RUNTIME_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def native_dir_for_config(cfg: Dict[str, Any]) -> Path:
    path = Path(cfg.get("native_dir", NATIVE_DIR))
    if not path.is_absolute():
        path = RUNTIME_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_hidden_config_path(cfg: Dict[str, Any]) -> Path:
    cfg_path = Path(cfg["verify_hidden"]["config_path"])
    if not cfg_path.is_absolute():
        cfg_path = hidden_verify_dir_for_config(cfg) / cfg_path.name
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg_path


def pair_to_point(pair: Tuple[float, ...], qord: float, params: Dict[str, np.floating]) -> complex:
    if len(pair) >= 3:
        return complex(pair[2])
    omega0 = float(pair[0])
    return complex(chua_ic.W_frac(omega0, qord, params))


def chua_ic_params_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg["params"]
    return {
        "model": normalize_chua_model(cfg.get("model", {}).get("kind", "nonsmooth")),
        "alpha": np.float64(p["alpha_chua"]),
        "beta": np.float64(p["beta"]),
        "gamma": np.float64(p["gamma_chua"]),
        "m0": np.float64(p["m0"]),
        "m1": np.float64(p["m1"]),
        "a1": np.float64(p.get("a1", 0.4)),
        "a2": np.float64(p.get("a2", -1.5585)),
        "rho": np.float64(p.get("rho", 1.0)),
    }


def hidden_backend_params_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Mapeo Python -> backend C: alpha_chua/beta/gamma_chua/m0/m1/model son
    # los argumentos directos de chua_hidden_backend.c; frac_order se pasa
    # aparte como --frac_order y tambien alimenta cfg["basin"]["q"].
    p = cfg["params"]
    return {
        "model": normalize_chua_model(cfg.get("model", {}).get("kind", "nonsmooth")),
        "alpha_chua": float(p["alpha_chua"]),
        "beta": float(p["beta"]),
        "gamma_chua": float(p["gamma_chua"]),
        "m0": float(p["m0"]),
        "m1": float(p["m1"]),
        "a1": float(p.get("a1", 0.4)),
        "a2": float(p.get("a2", -1.5585)),
        "rho": float(p.get("rho", 1.0)),
    }


def ensure_hidden_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hidden_dir = hidden_verify_dir_for_config(cfg)
    native_dir = native_dir_for_config(cfg)
    cfg_path = resolve_hidden_config_path(cfg)
    if not cfg_path.exists():
        hv.save_default_config(str(cfg_path))
    with open(cfg_path, "r", encoding="utf-8") as f:
        hidden_cfg = json.load(f)
    hidden_cfg = hv.prepare_runtime_paths(hidden_cfg, hidden_dir)
    hidden_files = hidden_cfg["files"]
    for key, default_name in {
        "csv_out": "hidden_target_check_frac.csv",
        "ref_csv_out": "reference_section.csv",
        "summary_csv_out": "summary_by_radius.csv",
        "json_out": "hidden_target_summary.json",
        "fig_section": "reference_section.png",
        "fig_probe": "probe_summary.png",
    }.items():
        hidden_files[key] = str(hidden_dir / hv.portable_filename(hidden_files.get(key), default_name))
    hidden_files["summary_from_pipeline"] = str(Path(cfg["outputs"]["summary_json"]))

    hidden_backend = hidden_cfg["backend"]
    source_name = hv.portable_filename(hidden_backend.get("source_c"), "chua_hidden_backend.c")
    exe_name = hv.portable_filename(hidden_backend.get("exe"), "chua_hidden_backend")
    hidden_backend["source_c"] = str((ROOT / source_name).resolve())
    hidden_exe = native_dir / exe_name
    if os.name == "nt" and hidden_exe.suffix.lower() != ".exe":
        hidden_exe = hidden_exe.with_suffix(".exe")
    native_dir.mkdir(parents=True, exist_ok=True)
    hidden_backend["exe"] = str(hidden_exe)
    hidden_cfg["native_dir"] = str(native_dir)
    hidden_cfg["runtime_dir"] = str(hidden_dir)

    # Sobrescribir con parÃ¡metros y tiempos del pipeline unificado
    if cfg["verify_hidden"].get("override_backend_params", True):
        hidden_cfg["params"] = hidden_backend_params_from_config(cfg)
    hidden_cfg["frac_order"] = validate_fractional_order(cfg["frac_order"])
    hidden_cfg["integration"] = {
        "h": float(cfg["verify_hidden"]["h"]),
        "Lm": float(cfg["verify_hidden"]["Lm"]),
        "TMAX_REF": float(cfg["verify_hidden"]["TMAX_REF"]),
        "TMAX_TEST": float(cfg["verify_hidden"]["TMAX_TEST"]),
        "TBURN_REF": float(cfg["verify_hidden"]["TBURN_REF"]),
        "TBURN_TEST": float(cfg["verify_hidden"]["TBURN_TEST"]),
    }
    hidden_cfg["thresholds"] = {
        "R_DIV": float(cfg["verify_hidden"]["R_DIV"]),
        "EPS_EQ": float(cfg["verify_hidden"]["EPS_EQ"]),
        "CAP_WIN": int(cfg["verify_hidden"]["CAP_WIN"]),
        "SEC_TOL": float(cfg["verify_hidden"]["SEC_TOL"]),
        "MIN_SEC_MATCH": int(cfg["verify_hidden"]["MIN_SEC_MATCH"]),
        "TEST_MAX_SEC": int(cfg["verify_hidden"]["TEST_MAX_SEC"]),
        "HIT_FRAC_REQ": float(cfg["verify_hidden"]["HIT_FRAC_REQ"]),
    }
    hidden_cfg["sampling"] = {
        "RADII": [float(r) for r in cfg["verify_hidden"]["RADII"]],
        "NSAMPLES_PER_RADIUS": int(cfg["verify_hidden"]["NSAMPLES_PER_RADIUS"]),
        "random_seed": int(cfg["verify_hidden"]["random_seed"]),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(hidden_cfg, f, indent=2, ensure_ascii=False)
    return hidden_cfg


def continuation_in_epsilon_c(
    cfg: Dict[str, Any],
    qord: float,
    k0: float,
    xseed: np.ndarray,
    eps_values: np.ndarray,
) -> List[Dict[str, Any]]:
    lib = load_fractional_backend(cfg)
    configure_fractional_backend_for_case(lib, cfg)
    lib.set_frac_backend_workers(1)
    cont = cfg["continuation"]
    eps_arr = np.asarray(eps_values, dtype=np.float64)
    seed = np.asarray(xseed, dtype=np.float64)
    n_eps = int(eps_arr.size)
    keep_rows = int(lib.efork_rows(float(cont["t_keep"]), float(cont["h"])))
    if n_eps <= 0 or keep_rows <= 0:
        raise RuntimeError("Configuracion invalida para continuacion EFORK C.")

    x_in = np.empty(n_eps * 3, dtype=np.float64)
    x_trans = np.empty(n_eps * 3, dtype=np.float64)
    x_out = np.empty(n_eps * 3, dtype=np.float64)
    hist_in = np.zeros(n_eps, dtype=np.int32)
    hist_out = np.zeros(n_eps, dtype=np.int32)
    traj = np.empty(n_eps * keep_rows * 4, dtype=np.float64)
    memory_mode = 1 if str(cont.get("memory_mode", "window")).strip().lower() == "window" else 0
    update_source = 1 if str(cont.get("memory_update_source", "observed")).strip().lower() == "observed" else 0
    log(
        "Continuacion EFORK C: etapa causal con memoria; "
        "python_workers=1; omp_threads=1; seed_strategy=causal_memory_chain."
    )

    rc = int(lib.compute_continuation_efork3(
        eps_arr, n_eps, seed, float(qord), float(k0), float(cont["h"]), float(cont["Lm"]),
        float(cont["t_transient"]), float(cont["t_keep"]), memory_mode, update_source,
        x_in, x_trans, x_out, hist_in, hist_out, traj
    ))
    if rc != 0:
        raise RuntimeError(f"compute_continuation_efork3 devolvio error {rc}")

    results: List[Dict[str, Any]] = []
    traj_view = traj.reshape((n_eps, keep_rows, 4))
    for i, eps in enumerate(eps_arr):
        results.append({
            "eps": float(eps),
            "x_in": x_in[3 * i:3 * i + 3].copy(),
            "x_transient_out": x_trans[3 * i:3 * i + 3].copy(),
            "x_out": x_out[3 * i:3 * i + 3].copy(),
            "traj": traj_view[i].copy(),
            "memory_mode": str(cont.get("memory_mode", "window")),
            "memory_update_source": str(cont.get("memory_update_source", "observed")),
            "history_points_in": int(hist_in[i]),
            "history_points_out": int(hist_out[i]),
            "backend": "c_efork3",
        })
    return results


def compute_nyquist_seed_and_continuation(
    cfg: Dict[str, Any],
    df_method: str = "classic",
    machado_mu: float = 1.0,
    branch_index_override: int | None = None,
) -> Dict[str, Any]:
    log("Nyquist: buscando cruces y semilla armonica candidata.")
    frac_order = validate_fractional_order(cfg["frac_order"])
    df_method = str(df_method).strip().lower()
    if df_method in {"clasica", "classical"}:
        df_method = "classic"
    if df_method in {"fdf", "fractional"}:
        df_method = "machado"
    if df_method not in {"classic", "machado"}:
        raise ValueError("df_method debe ser 'classic' o 'machado'.")
    machado_mu = float(machado_mu)
    if df_method == "machado" and (not np.isfinite(machado_mu) or machado_mu <= 0.0):
        raise ValueError("machado_mu debe ser positivo.")
    # Ajustar parÃ¡metros del mÃ³dulo importado
    chua_ic.PARAMS = chua_ic_params_from_config(cfg)
    chua_ic.QORD = np.float64(frac_order)
    chua_ic.EPS_VALUES = np.array(cfg["continuation"]["eps_values"], dtype=np.float64)

    raw_pairs = chua_ic.find_omega_k_candidates(
        chua_ic.QORD,
        chua_ic.PARAMS,
        wmin=chua_ic.WMIN,
        wmax=chua_ic.WMAX,
        nscan=chua_ic.NSCAN,
    )
    if df_method == "classic":
        pairs = [pair for pair in raw_pairs if chua_ic.is_describing_gain_compatible(pair[1], chua_ic.PARAMS)]
    else:
        pairs = [pair for pair in raw_pairs if chua_ic.is_machado_gain_compatible(pair[1], chua_ic.PARAMS, machado_mu)]
    if len(pairs) == 0:
        raise RuntimeError("No se encontraron raÃ­ces para Im(W_q(iÏ‰))=0.")
    label = "clasica" if df_method == "classic" else f"Machado mu={machado_mu:g}"
    log(f"Nyquist ({label}): {len(pairs)} cruce(s) compatible(s) de {len(raw_pairs)} cruce(s) crudo(s).")

    branch_index = int(cfg["branch_index"] if branch_index_override is None else branch_index_override)
    if branch_index < 0 or branch_index >= len(pairs):
        raise ValueError("branch_index fuera de rango.")

    omega0, k0 = pairs[branch_index][:2]
    a0 = chua_ic.solve_amplitude_for_describing_method(k0, chua_ic.PARAMS, method=df_method, mu=machado_mu)
    xseed, v, eig_match = chua_ic.build_fractional_seed(chua_ic.QORD, chua_ic.PARAMS, omega0, k0, a0)
    W0 = chua_ic.W_frac(omega0, chua_ic.QORD, chua_ic.PARAMS)
    N_classic_a0 = float(chua_ic.N_sat(a0, chua_ic.PARAMS))
    N_effective_a0 = (
        float(chua_ic.N_sat_machado(a0, chua_ic.PARAMS, machado_mu))
        if df_method == "machado"
        else N_classic_a0
    )
    nyquist_df_residuals = {
        "W0_real": float(np.real(W0)),
        "W0_imag": float(np.imag(W0)),
        "k_from_W": float(-1.0 / np.real(W0)),
        "N_classic_a0": N_classic_a0,
        "N_effective_a0": N_effective_a0,
        "closure_N_minus_k": float(N_effective_a0 - k0),
    }
    log(
        "Continuacion: "
        f"{len(cfg['continuation']['eps_values'])} pasos epsilon; "
        f"t_transient={cfg['continuation']['t_transient']}, "
        f"t_keep={cfg['continuation']['t_keep']}, "
        f"memory_mode={cfg['continuation'].get('memory_mode', 'window')}, "
        f"memory_update_source={cfg['continuation'].get('memory_update_source', 'observed')}."
    )

    if bool(cfg.get("native_efork", {}).get("enabled", True)):
        results = continuation_in_epsilon_c(cfg, chua_ic.QORD, k0, xseed, chua_ic.EPS_VALUES)
        log("Continuacion C EFORK-3: terminada.")
    else:
        results = chua_ic.continuation_in_epsilon(
            p=chua_ic.PARAMS,
            qord=chua_ic.QORD,
            k=k0,
            x0_seed=xseed,
            eps_values=chua_ic.EPS_VALUES,
            h=float(cfg["continuation"]["h"]),
            Lm=float(cfg["continuation"]["Lm"]),
            t_transient=float(cfg["continuation"]["t_transient"]),
            t_keep=float(cfg["continuation"]["t_keep"]),
            memory_mode=cfg["continuation"].get("memory_mode", "window"),
            memory_update_source=cfg["continuation"].get("memory_update_source", "observed"),
        )
        log("Continuacion Python EFORK-3: terminada.")

    final_state = results[-1]["x_out"].tolist()
    summary = {
        "model": cfg["model"],
        "params": cfg["params"],
        "frac_order": frac_order,
        "runtime_contract": cfg.get("runtime_contract", {}),
        "describing_function": {
            "method": df_method,
            "machado_mu": float(machado_mu) if df_method == "machado" else None,
            "source": "Machado fractional describing function N_mu=N^mu" if df_method == "machado" else "classic describing function",
        },
        "raw_pairs": [
            {"omega0": float(p[0]), "k": float(p[1])} for p in raw_pairs
        ],
        "pairs": [
            {"omega0": float(p[0]), "k": float(p[1])} for p in pairs
        ],
        "chosen_branch": {
            "describing_function": df_method,
            "machado_mu": float(machado_mu) if df_method == "machado" else None,
            "branch_index": branch_index,
            "omega0": float(omega0),
            "k": float(k0),
            "a0": float(a0),
            "seed": np.asarray(xseed, dtype=float).tolist(),
            "eig_match": [float(np.real(eig_match)), float(np.imag(eig_match))],
            "eigvec_real": np.real(v).astype(float).tolist(),
            "eigvec_imag": np.imag(v).astype(float).tolist(),
            "nyquist_df_residuals": nyquist_df_residuals,
        },
        "continuation": {
            "backend": "c_efork3" if bool(cfg.get("native_efork", {}).get("enabled", True)) else "python",
            "eps_values": [float(x) for x in cfg["continuation"]["eps_values"]],
            "memory_mode": str(cfg["continuation"].get("memory_mode", "window")),
            "memory_update_source": str(cfg["continuation"].get("memory_update_source", "observed")),
            "h": float(cfg["continuation"]["h"]),
            "Lm": float(cfg["continuation"]["Lm"]),
            "t_transient": float(cfg["continuation"]["t_transient"]),
            "t_keep": float(cfg["continuation"]["t_keep"]),
            "final_state_eps1": final_state,
            "states_by_step": [
                {
                    "eps": float(r["eps"]),
                    "x_in": np.asarray(r["x_in"], dtype=float).tolist(),
                    "x_transient_out": np.asarray(r.get("x_transient_out", r["x_out"]), dtype=float).tolist(),
                    "x_out": np.asarray(r["x_out"], dtype=float).tolist(),
                    "history_points_in": int(r.get("history_points_in", 0)),
                    "history_points_out": int(r.get("history_points_out", 0)),
                }
                for r in results
            ],
        },
    }
    Path(cfg["outputs"]["cont_json"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["outputs"]["cont_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "pairs": pairs,
        "raw_pairs": raw_pairs,
        "describing_function": df_method,
        "machado_mu": float(machado_mu) if df_method == "machado" else None,
        "omega0": float(omega0),
        "k0": float(k0),
        "a0": float(a0),
        "xseed": np.asarray(xseed, dtype=float),
        "eigvec": np.asarray(v),
        "results": results,
        "final_state": np.asarray(final_state, dtype=float),
        "summary": summary,
    }


# ---------- plots: titles removed ----------
def save_pdf(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(WHITE_BG)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor=WHITE_BG)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=WHITE_BG)
    plt.close(fig)
    log(f"Guardado: {path.name} y {png_path.name}")


ARTICLE_DARK_BG = WHITE_BG
ARTICLE_DARK_AX = WHITE_BG
ARTICLE_DARK_GRID = "#d1d5db"
ARTICLE_DARK_TEXT = "#111827"


def article_dark_subplots(*args, **kwargs):
    fig, axs = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(WHITE_BG)
    return fig, axs


def style_dark_axis(ax, grid_alpha: float = 0.28) -> None:
    style_white_axis(ax, grid=True, grid_alpha=grid_alpha)


def dark_legend(ax, *args, **kwargs):
    return ax.legend(*args, **kwargs)


def all_bifurcation_points(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.concatenate([np.asarray(data["pos_x"], dtype=float), np.asarray(data["neg_x"], dtype=float)])
    ys = np.concatenate([np.asarray(data["pos_y"], dtype=float), np.asarray(data["neg_y"], dtype=float)])
    mask = np.isfinite(xs) & np.isfinite(ys)
    return xs[mask], ys[mask]


def nyquist_transfer_samples(cfg: Dict[str, Any], n: int = 5000) -> Dict[str, np.ndarray]:
    p = chua_ic.PARAMS
    qord = chua_ic.QORD
    omega = np.linspace(0.0, float(chua_ic.WMAX), int(n))
    omega[0] = max(float(chua_ic.WMIN), 1e-8)
    W = np.array([chua_ic.W_frac(w, qord, p) for w in omega], dtype=np.complex128)
    return {"omega": omega, "W": W}


def plot_transfer_re_im(cfg: Dict[str, Any], nyq_data: Dict[str, Any]):
    samples = nyquist_transfer_samples(cfg)
    omega = samples["omega"]
    W = samples["W"]
    omega0 = float(nyq_data["omega0"])
    k0 = float(nyq_data["k0"])
    W0 = complex(chua_ic.W_frac(omega0, chua_ic.QORD, chua_ic.PARAMS))
    pairs = [(float(pair[0]), complex(pair_to_point(pair, chua_ic.QORD, chua_ic.PARAMS))) for pair in nyq_data["pairs"]]

    fig, axs = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=True)
    axs[0].plot(omega, W.real, lw=1.25, color=NYQUIST_W_COLOR, label=r"Re$\{W\}$")
    axs[0].axhline(-1.0 / k0, color="#6b7280", ls="--", lw=0.95, label=r"$-1/k$")
    axs[0].scatter([omega0], [np.real(W0)], s=72, facecolors="none", edgecolors="#ff1f1f", linewidths=1.6, label=r"$\omega_0$")
    for om, z in pairs:
        axs[0].axvline(om, color="#9ca3af", ls=":", lw=0.85)
    axs[0].set_ylabel(r"Re$\{W(i\omega)\}$")
    style_white_axis(axs[0], grid=True, grid_alpha=0.24)
    axs[0].legend(loc="best", fontsize=8)

    axs[1].plot(omega, W.imag, lw=1.25, color=NYQUIST_W_COLOR, label=r"Im$\{W\}$")
    axs[1].axhline(0.0, color="#6b7280", ls="--", lw=0.95, label="0")
    axs[1].scatter([omega0], [np.imag(W0)], s=72, facecolors="none", edgecolors="#ff1f1f", linewidths=1.6, label=r"$\omega_0$")
    for om, z in pairs:
        axs[1].axvline(om, color="#9ca3af", ls=":", lw=0.85)
    axs[1].set_xlabel(r"$\omega$")
    axs[1].set_ylabel(r"Im$\{W(i\omega)\}$")
    style_white_axis(axs[1], grid=True, grid_alpha=0.24)
    axs[1].legend(loc="best", fontsize=8)
    save_pdf(fig, cfg["outputs"]["transfer_reim_pdf"])


def plot_clean_nyquist(cfg: Dict[str, Any], nyq_data: Dict[str, Any]):
    p = chua_ic.PARAMS
    qord = chua_ic.QORD
    omg = np.logspace(np.log10(chua_ic.WMIN), np.log10(chua_ic.WMAX), 4000)
    W = np.array([chua_ic.W_frac(w, qord, p) for w in omg], dtype=np.complex128)

    fig, ax = article_dark_subplots(figsize=(7.6, 5.7))
    ax.plot(W.real, W.imag, lw=1.7, color=NYQUIST_W_COLOR, label=r"$W(i\omega)$")
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.9)
    # marcar ramas
    for idx_pair, pair in enumerate(nyq_data["pairs"]):
        W0 = pair_to_point(pair, qord, p)
        if idx_pair == int(cfg["branch_index"]):
            ax.scatter(
                np.real(W0), np.imag(W0), s=82, marker="x",
                c="#ff1f1f", linewidths=2.0, zorder=4,
                label=r"$W(i\omega_0)=-1/k$",
            )
        else:
            ax.scatter(
                np.real(W0), np.imag(W0), s=62, marker="o",
                facecolors="none", edgecolors="#ff1f1f",
                linewidths=1.5, zorder=4, label="otro cruce",
            )
    ax.set_xlabel(r"Re$(W(i\omega))$")
    ax.set_ylabel(r"Im$(W(i\omega))$")
    ax.set_xlim(-10.0, 0.0)
    style_dark_axis(ax)
    dark_legend(ax, loc="best", fontsize=8)
    save_pdf(fig, cfg["outputs"]["nyquist_pdf"])


def plot_clean_continuation_progress(cfg: Dict[str, Any], results: List[Dict[str, Any]]):
    eps = np.array([r["eps"] for r in results], dtype=float)
    Xin = np.array([r["x_in"] for r in results], dtype=float)
    Xout = np.array([r["x_out"] for r in results], dtype=float)

    fig, axs = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        axs[i].plot(eps, Xin[:, i], "o-", lw=1.2, ms=4, color=LINEARIZED_COLOR, label="entrada")
        axs[i].plot(eps, Xout[:, i], "s--", lw=1.0, ms=3, color=ORIGINAL_COLOR, label="salida")
        axs[i].set_ylabel(labels[i])
        style_white_axis(axs[i])
        axs[i].legend(loc="best", fontsize=8)
    axs[-1].set_xlabel(r"$\varepsilon$")
    save_pdf(fig, cfg["outputs"]["cont_progress_pdf"])


def plot_clean_final_attractor(cfg: Dict[str, Any], results: List[Dict[str, Any]]):
    T = results[-1]["traj"]
    x_in_last = np.asarray(results[-1]["x_in"], dtype=float)

    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(T[:, 1], T[:, 2], T[:, 3], lw=0.9, color=ORIGINAL_COLOR, label="trayectoria final")
    ax.scatter(x_in_last[0], x_in_last[1], x_in_last[2], s=28, c=BASIN_SEED_COLOR, edgecolors="#111827", label="semilla")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    style_white_axis(ax, grid=True, grid_alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    save_pdf(fig, cfg["outputs"]["final_attr_pdf"])


def run_hidden_verify_with_seed(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    hidden_cfg = ensure_hidden_config(cfg)
    hidden_cfg["target_seed"] = {"x": float(final_state[0]), "y": float(final_state[1]), "z": float(final_state[2])}
    cfg_path = resolve_hidden_config_path(cfg)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(hidden_cfg, f, indent=2, ensure_ascii=False)

    def log_hidden_backend_config(prefix: str) -> None:
        log(
            f"{prefix}: "
            f"Lm={hidden_cfg['integration']['Lm']}, "
            f"TMAX_REF={hidden_cfg['integration']['TMAX_REF']}, "
            f"TMAX_TEST={hidden_cfg['integration']['TMAX_TEST']}, "
            f"muestras={len(hidden_cfg['sampling']['RADII']) * int(hidden_cfg['sampling']['NSAMPLES_PER_RADIUS'])}."
        )

    log_hidden_backend_config("Backend C ocultedad")
    log("Backend C ocultedad: compilando.")
    hv.compile_backend(hidden_cfg)
    hidden_openmp_active = bool(hidden_cfg.get("backend", {}).get("openmp_active", hidden_cfg.get("backend", {}).get("openmp", False)))
    hidden_threads = max(1, int(hidden_cfg.get("safety", {}).get("max_threads", 1))) if hidden_openmp_active else 1
    cfg.setdefault("runtime_contract", {}).setdefault("parallelism", {})["hidden_verification"] = parallel_contract(
        python_workers=1,
        omp_threads=hidden_threads,
        backend_openmp_active=hidden_openmp_active,
        seed_strategy="independent_equilibrium_samples",
        stage_kind="embarrassingly_parallel",
    )
    log(
        "Backend C ocultedad: contrato de paralelismo efectivo; "
        f"python_workers=1; omp_threads={hidden_threads}; "
        f"backend_openmp_active={hidden_openmp_active}; "
        "seed_strategy=independent_equilibrium_samples; stage_kind=embarrassingly_parallel."
    )
    log("Backend C ocultedad: ejecutando integraciones.")
    retry_used = False
    try:
        hv.run_backend(hidden_cfg)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "El backend C no pudo construir una referencia robusta del atractor objetivo con el contrato actual. "
            "Aumenta HIDDEN_ATTRACTORS_T_KEEP, HIDDEN_ATTRACTORS_T_TRANSIENT o HIDDEN_ATTRACTORS_LM, "
            "o revisa si la semilla final pertenece al atractor esperado."
        ) from exc
    log("Backend C ocultedad: procesando CSV y graficas auxiliares.")

    files = hidden_cfg["files"]
    ref = hv.load_reference_csv(files["ref_csv_out"])
    detail_rows = hv.load_csv_dicts(files["csv_out"])
    summary_rows = hv.load_csv_dicts(files["summary_csv_out"])
    eqs = hv.chua_equilibria(hv.ChuaParams(**hidden_cfg["params"]))

    hv.plot_reference_section(ref, files["fig_section"])
    hv.plot_probe_summary(
        summary_rows,
        [float(r) for r in hidden_cfg["sampling"]["RADII"]],
        list(eqs.keys()),
        files["fig_probe"],
    )

    summary = hv.build_summary_json(hidden_cfg, eqs, ref, summary_rows, detail_rows)
    summary["pipeline_hidden_retry_used"] = retry_used
    with open(files["json_out"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def load_reference_csv(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data


def load_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_clean_reference_section(cfg: Dict[str, Any], ref_csv_path: str):
    ref = load_reference_csv(ref_csv_path)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(ref[:, 0], ref[:, 1], s=8, color=NYQUIST_W_COLOR, label="seccion objetivo")
    ax.set_xlabel("y en secciÃ³n x=0")
    ax.set_ylabel("z en secciÃ³n x=0")
    style_white_axis(ax)
    ax.legend(loc="best", fontsize=8)
    save_pdf(fig, cfg["outputs"]["ref_section_pdf"])
    return ref


def plot_clean_probe_summary(cfg: Dict[str, Any], summary_csv_path: str, eq_names: List[str]):
    rows = load_csv_dicts(summary_csv_path)
    radii = [float(r) for r in cfg["verify_hidden"]["RADII"]]
    class_order = ["EQ", "DIV", "TARGET", "OTHER", "UNKNOWN"]

    fig, axs = plt.subplots(len(eq_names), 1, figsize=(7.2, 3.4 * len(eq_names)), sharex=True)
    if len(eq_names) == 1:
        axs = [axs]

    for ax, eq in zip(axs, eq_names):
        sub = [r for r in rows if r["equilibrium"] == eq]
        for cname in class_order:
            ys = []
            for rr in radii:
                val = 0
                for row in sub:
                    if math.isclose(float(row["radius"]), rr, rel_tol=0.0, abs_tol=1e-18):
                        val = int(row[cname])
                        break
                ys.append(val)
            ax.plot(radii, ys, marker="o", lw=1.1, ms=4, label=cname)
        ax.set_xscale("log")
        ax.set_ylabel(eq)
        style_white_axis(ax)
        ax.legend(loc="best", fontsize=8)
    axs[-1].set_xlabel("radio")
    save_pdf(fig, cfg["outputs"]["probe_summary_pdf"])
    return rows


# ---------- basin lib reused from plot_basin_from_c.py ----------
def _lib_filename() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "chua_basin.dll"
    if system == "darwin":
        return "libchua_basin.dylib"
    return "libchua_basin.so"


def compile_shared_library(source: str, openmp: bool = True):
    src = Path(source)
    if not src.is_absolute():
        src = ROOT / src
    src = src.resolve()
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = NATIVE_DIR / _lib_filename()
    if not src.exists():
        raise FileNotFoundError(f"No existe el archivo fuente: {src}")
    return compile_c_target(src, out, target_kind="shared", openmp=openmp, logger=log)


def _frac_backend_filename() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "chua_frac_backend.dll"
    if system == "darwin":
        return "libchua_frac_backend.dylib"
    return "libchua_frac_backend.so"


def compile_fractional_backend_library(cfg: Dict[str, Any]):
    backend = cfg.get("native_efork", {})
    src = Path(backend.get("source", "chua_frac_backend_lib.c"))
    if not src.is_absolute():
        src = ROOT / src
    src = src.resolve()
    out = NATIVE_DIR / _frac_backend_filename()
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"No existe el backend EFORK C: {src}")
    return compile_c_target(
        src,
        out,
        target_kind="shared",
        openmp=bool(backend.get("openmp", True)),
        logger=log,
    )


def load_fractional_backend(cfg: Dict[str, Any]):
    global FRACTIONAL_BACKEND_CACHE
    if not bool(cfg.get("native_efork", {}).get("enabled", True)):
        raise RuntimeError("El backend EFORK C esta deshabilitado.")
    if FRACTIONAL_BACKEND_CACHE is not None:
        return FRACTIONAL_BACKEND_CACHE
    compile_result = compile_fractional_backend_library(cfg)
    cfg.setdefault("native_efork", {})["openmp_active"] = bool(compile_result.openmp_active)
    cfg.setdefault("native_efork", {})["compile_command"] = compile_result.command
    refresh_parallel_runtime_contract(cfg)
    libpath = compile_result.path
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
    lib.set_frac_chua_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.set_frac_chua_params.restype = None
    lib.set_frac_chua_model.argtypes = [ctypes.c_int]
    lib.set_frac_chua_model.restype = None
    lib.set_frac_chua_arctan_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.set_frac_chua_arctan_params.restype = None
    lib.set_frac_backend_workers.argtypes = [ctypes.c_int]
    lib.set_frac_backend_workers.restype = None
    lib.efork_rows.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.efork_rows.restype = ctypes.c_int
    lib.integrate_chua_efork3.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.integrate_chua_efork3.restype = ctypes.c_int
    lib.compute_continuation_efork3.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.compute_continuation_efork3.restype = ctypes.c_int
    lib.compute_bifurcation_sweep_efork3.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.compute_bifurcation_sweep_efork3.restype = ctypes.c_int
    FRACTIONAL_BACKEND_CACHE = lib
    return lib


def configure_fractional_backend_for_case(lib: Any, cfg: Dict[str, Any]) -> None:
    p = cfg["params"]
    model = normalize_chua_model(cfg.get("model", {}).get("kind", "nonsmooth"))
    lib.set_frac_chua_params(float(p["alpha_chua"]), float(p["beta"]), float(p["gamma_chua"]), float(p["m0"]), float(p["m1"]))
    lib.set_frac_chua_model(1 if model == "arctan" else 0)
    lib.set_frac_chua_arctan_params(float(p.get("a1", 0.4)), float(p.get("a2", -1.5585)), float(p.get("rho", 1.0)))
    lib.set_frac_backend_workers(int(cfg.get("native_efork", {}).get("workers", 1)))


def configure_fractional_backend_from_params(lib: Any, p: Dict[str, Any], *, workers: int = 1) -> None:
    model = normalize_chua_model(p.get("model", "nonsmooth"))
    lib.set_frac_chua_params(float(p["alpha"]), float(p["beta"]), float(p["gamma"]), float(p["m0"]), float(p["m1"]))
    lib.set_frac_chua_model(1 if model == "arctan" else 0)
    lib.set_frac_chua_arctan_params(float(p.get("a1", 0.4)), float(p.get("a2", -1.5585)), float(p.get("rho", 1.0)))
    lib.set_frac_backend_workers(int(workers))


def load_basin_library(cfg: Dict[str, Any]):
    global BASIN_LIBRARY_CACHE
    if BASIN_LIBRARY_CACHE is not None:
        return BASIN_LIBRARY_CACHE
    compile_result = compile_shared_library(cfg["basin"]["source"], openmp=bool(cfg["basin"]["openmp"]))
    cfg.setdefault("basin", {})["openmp_active"] = bool(compile_result.openmp_active)
    cfg.setdefault("basin", {})["compile_command"] = compile_result.command
    refresh_parallel_runtime_contract(cfg)
    libpath = compile_result.path
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
    if hasattr(lib, "set_chua_model"):
        lib.set_chua_model.argtypes = [ctypes.c_int]
        lib.set_chua_model.restype = None
    if hasattr(lib, "set_chua_arctan_params"):
        lib.set_chua_arctan_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.set_chua_arctan_params.restype = None
    if hasattr(lib, "set_basin_workers"):
        lib.set_basin_workers.argtypes = [ctypes.c_int]
        lib.set_basin_workers.restype = None
    lib.get_equilibria.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
    lib.get_equilibria.restype = None
    lib.compute_basin_xy.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    ]
    lib.compute_basin_xy.restype = ctypes.c_int
    lib.compute_basin_plane.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    ]
    lib.compute_basin_plane.restype = ctypes.c_int
    lib.classify_basin_point.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.c_double
    ]
    lib.classify_basin_point.restype = ctypes.c_int
    BASIN_LIBRARY_CACHE = lib
    return lib


def effective_basin_z0(cfg: Dict[str, Any], final_state: np.ndarray) -> float:
    raw_z0 = cfg["basin"].get("z0", "final_state")
    if isinstance(raw_z0, str) and raw_z0.strip().lower() in {"final", "final_state", "seed", "auto"}:
        return float(np.asarray(final_state, dtype=float)[2])
    return float(raw_z0)


def configure_basin_library_for_case(lib: Any, cfg: Dict[str, Any]) -> None:
    p = cfg["params"]
    model = normalize_chua_model(cfg.get("model", {}).get("kind", "nonsmooth"))
    lib.set_chua_params(float(p["alpha_chua"]), float(p["beta"]), float(p["gamma_chua"]), float(p["m0"]), float(p["m1"]))
    if hasattr(lib, "set_chua_model"):
        lib.set_chua_model(1 if model == "arctan" else 0)
    if hasattr(lib, "set_chua_arctan_params"):
        lib.set_chua_arctan_params(float(p.get("a1", 0.4)), float(p.get("a2", -1.5585)), float(p.get("rho", 1.0)))
    if hasattr(lib, "set_basin_workers"):
        workers = int(cfg["basin"].get("workers", 1))
        active = bool(cfg["basin"].get("openmp_active", cfg["basin"].get("openmp", False)))
        lib.set_basin_workers(workers if active and bool(cfg["basin"].get("parallel", True)) else 1)


def nearest_basin_cell_class(cfg: Dict[str, Any], basin: np.ndarray, final_state: np.ndarray) -> Dict[str, Any]:
    b = cfg["basin"]
    nx, ny = int(b["nx"]), int(b["ny"])
    x = float(np.asarray(final_state, dtype=float)[0])
    y = float(np.asarray(final_state, dtype=float)[1])
    xfrac = (x - float(b["xmin"])) / max(float(b["xmax"]) - float(b["xmin"]), 1e-15)
    yfrac = (y - float(b["ymin"])) / max(float(b["ymax"]) - float(b["ymin"]), 1e-15)
    j = int(np.clip(round(xfrac * (nx - 1)), 0, nx - 1))
    i = int(np.clip(round(yfrac * (ny - 1)), 0, ny - 1))
    class_id = int(basin[i, j])
    return {
        "i": i,
        "j": j,
        "x_grid": float(b["xmin"]) + j * (float(b["xmax"]) - float(b["xmin"])) / max(nx - 1, 1),
        "y_grid": float(b["ymin"]) + i * (float(b["ymax"]) - float(b["ymin"])) / max(ny - 1, 1),
        "class_id": class_id,
        "class_label": basin_class_label(class_id),
    }


def classify_basin_seed_exact(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    lib = load_basin_library(cfg)
    configure_basin_library_for_case(lib, cfg)
    b = cfg["basin"]
    x, y, z = [float(v) for v in np.asarray(final_state, dtype=float)]
    class_id = int(lib.classify_basin_point(
        x, y, z,
        validate_fractional_order(cfg["frac_order"]), float(b["h"]), float(b["Lm"]),
        float(b["TMAX"]), float(b["TBURN"]), float(b["R_DIV"]), float(b["R_BOUND"]),
        float(b["EPS_EQ"]), int(b["CAP_WIN"]), float(b["MEAN_X_GAP"])
    ))
    return {
        "point": [x, y, z],
        "class_id": class_id,
        "class_label": basin_class_label(class_id),
    }


def compute_basin_seed_diagnostics(cfg: Dict[str, Any], basin: np.ndarray, final_state: np.ndarray) -> Dict[str, Any]:
    z0_eff = float(cfg["basin"].get("_last_z0_effective", effective_basin_z0(cfg, final_state)))
    seed = np.asarray(final_state, dtype=float)
    exact = classify_basin_seed_exact(cfg, seed)
    nearest = nearest_basin_cell_class(cfg, basin, seed)
    projected = not math.isclose(float(seed[2]), z0_eff, rel_tol=0.0, abs_tol=1e-10)
    warnings = []
    if projected:
        warnings.append(
            "La estrella se esta mostrando como proyeccion sobre un corte que no coincide con z de la semilla."
        )
    if exact["class_id"] != nearest["class_id"]:
        warnings.append(
            "La clase exacta de la semilla difiere de la clase de la celda mas cercana del corte de cuenca."
        )
    return {
        "basin_z0": z0_eff,
        "seed_projected_on_xy": projected,
        "seed_exact": exact,
        "nearest_grid_cell": nearest,
        "warnings": warnings,
    }


def compute_basin_and_eq(cfg: Dict[str, Any], final_state: np.ndarray):
    lib = load_basin_library(cfg)
    b = cfg["basin"]
    configure_basin_library_for_case(lib, cfg)
    nx, ny = int(b["nx"]), int(b["ny"])
    z0_eff = effective_basin_z0(cfg, final_state)
    b["_last_z0_effective"] = z0_eff
    log(
        "Cuenca C: "
        f"grid={nx}x{ny}, z0={z0_eff:.10g}, q={validate_fractional_order(cfg['frac_order'])}, "
        f"TMAX={b['TMAX']}, TBURN={b['TBURN']}, "
        f"workers={int(b.get('workers', 1))}, OpenMP={bool(b['openmp'])}, "
        f"OpenMP_activo={b.get('openmp_active', 'pendiente_compilacion')}, "
        f"modelo={cfg['model']['kind']}."
    )
    log(
        "Cuenca C xy: "
        f"iniciando {nx * ny} condiciones iniciales; "
        "el backend C reportara avance por filas."
    )
    out = np.empty(nx * ny, dtype=np.int32)
    started = time.perf_counter()
    rc = lib.compute_basin_xy(
        nx, ny, float(b["xmin"]), float(b["xmax"]), float(b["ymin"]), float(b["ymax"]),
        z0_eff, validate_fractional_order(cfg["frac_order"]), float(b["h"]), float(b["Lm"]), float(b["TMAX"]), float(b["TBURN"]),
        float(b["R_DIV"]), float(b["R_BOUND"]), float(b["EPS_EQ"]), int(b["CAP_WIN"]), float(b["MEAN_X_GAP"]), out
    )
    if rc != 0:
        raise RuntimeError(f"compute_basin_xy devolviÃ³ error {rc}")
    log(f"Cuenca C xy: terminada en {format_elapsed(time.perf_counter() - started)}.")
    basin = out.reshape((ny, nx))
    eq = np.empty(9, dtype=np.float64)
    lib.get_equilibria(eq)
    E0 = eq[0:3].copy(); Ep = eq[3:6].copy(); Em = eq[6:9].copy()
    return basin, E0, Ep, Em


def compute_basin_planes_c(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    lib = load_basin_library(cfg)
    bcfg = cfg["basin_python"]
    configure_basin_library_for_case(lib, cfg)

    nx, ny = int(bcfg["nx"]), int(bcfg["ny"])
    qord = validate_fractional_order(cfg.get("frac_order", chua_ic.QORD))
    plane_defs = {
        "xy": (0, np.linspace(*bcfg["xlim"], nx), np.linspace(*bcfg["ylim"], ny), float(final_state[2])),
        "xz": (1, np.linspace(*bcfg["xlim"], nx), np.linspace(*bcfg["zlim"], ny), float(final_state[1])),
        "yz": (2, np.linspace(*bcfg["ylim"], nx), np.linspace(*bcfg["zlim"], ny), float(final_state[0])),
    }
    seed_uv = {
        "xy": (float(final_state[0]), float(final_state[1])),
        "xz": (float(final_state[0]), float(final_state[2])),
        "yz": (float(final_state[1]), float(final_state[2])),
    }
    out: Dict[str, Any] = {}
    log(
        "Cuenca C por planos: "
        f"3 planos x {nx}x{ny}; t_total={bcfg['t_total']}, h={bcfg['h']}."
    )
    for plane, (plane_id, uvals, vvals, fixed) in plane_defs.items():
        started = time.perf_counter()
        log(
            f"Cuenca C plano {plane}: iniciando {nx * ny} condiciones iniciales "
            f"({nx}x{ny}, fijo={fixed:.8g}); el backend C reportara avance por filas."
        )
        grid_out = np.empty(nx * ny, dtype=np.int32)
        rc = lib.compute_basin_plane(
            nx, ny,
            float(uvals[0]), float(uvals[-1]), float(vvals[0]), float(vvals[-1]),
            fixed, int(plane_id), qord, float(bcfg["h"]), float(bcfg["Lm"]),
            float(bcfg["t_total"]), float(bcfg["t_burn"]),
            float(bcfg["div_threshold"]), float(cfg["basin"].get("R_BOUND", 30.0)),
            float(cfg["basin"].get("EPS_EQ", 0.03)), int(cfg["basin"].get("CAP_WIN", 150)),
            float(cfg["basin"].get("MEAN_X_GAP", 0.08)), grid_out
        )
        if rc != 0:
            raise RuntimeError(f"compute_basin_plane({plane}) devolvio error {rc}")
        out[plane] = {
            "uvals": uvals,
            "vvals": vvals,
            "grid": grid_out.reshape((ny, nx)),
            "fixed": fixed,
            "seed_uv": seed_uv[plane],
        }
        log(f"Cuenca C plano {plane}: terminado en {format_elapsed(time.perf_counter() - started)}.")
    return out


def dist3(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def dist2_xy(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a[:2] - b[:2]))


def save_distances_csv(cfg: Dict[str, Any], final_state: np.ndarray, E0, Ep, Em):
    lines = [
        "label,dist_3d,dist_xy\n",
        f"E0,{dist3(final_state, E0):.16g},{dist2_xy(final_state, E0):.16g}\n",
        f"E+,{dist3(final_state, Ep):.16g},{dist2_xy(final_state, Ep):.16g}\n",
        f"E-,{dist3(final_state, Em):.16g},{dist2_xy(final_state, Em):.16g}\n",
    ]
    Path(cfg["outputs"]["dist_csv"]).write_text("".join(lines), encoding="utf-8")


def plot_clean_basin_overlay(cfg: Dict[str, Any], basin: np.ndarray, E0, Ep, Em, final_state: np.ndarray):
    b = cfg["basin"]
    seed = np.asarray(final_state, dtype=float)
    z0_eff = float(b.get("_last_z0_effective", effective_basin_z0(cfg, seed)))
    seed_projected = not math.isclose(float(seed[2]), z0_eff, rel_tol=0.0, abs_tol=1e-10)
    cmap, norm = basin_cmap_norm()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_facecolor(BASIN_CLASS_COLORS[-1])
    ax.imshow(
        basin, origin="lower", extent=[b["xmin"], b["xmax"], b["ymin"], b["ymax"]],
        cmap=cmap, norm=norm, interpolation="nearest", aspect="equal"
    )
    ax.scatter(E0[0], E0[1], c="black", s=34, marker="o", zorder=4)
    ax.scatter(Ep[0], Ep[1], c="white", edgecolors="black", s=48, marker="^", zorder=4)
    ax.scatter(Em[0], Em[1], c="white", edgecolors="black", s=48, marker="v", zorder=4)
    seed_label = "estado final" if not seed_projected else "estado final (proy.)"
    ax.scatter(seed[0], seed[1], c=BASIN_SEED_COLOR, edgecolors="black", s=105, marker="*", zorder=5)
    ax.text(
        0.02, 0.98, f"corte z={z0_eff:.4g}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        zorder=6,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    style_white_axis(ax, grid=False)
    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.35, label=label)
        for color, label in zip(BASIN_CLASS_COLORS, BASIN_CLASS_LABELS)
    ]
    handles.append(
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor=BASIN_SEED_COLOR,
               markeredgecolor="black", markersize=10, label=seed_label)
    )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.88)
    save_pdf(fig, cfg["outputs"]["basin_pdf"])



def ensure_current_chua_params(cfg: Dict[str, Any]):
    chua_ic.PARAMS = chua_ic_params_from_config(cfg)
    chua_ic.QORD = np.float64(validate_fractional_order(cfg["frac_order"]))



def symmetric_seed(seed: np.ndarray) -> np.ndarray:
    return -np.asarray(seed, dtype=float)


def arctan_equilibrium_root(p: Dict[str, float]) -> float | None:
    beta = float(p["beta"])
    gamma = float(p["gamma"])
    a1 = float(p.get("a1", 0.4))
    a2 = float(p.get("a2", -1.5585))
    rho = float(p.get("rho", 1.0))
    linear_coeff = 1.0 + a1 - gamma / (beta + gamma)

    def f(x: float) -> float:
        return linear_coeff * x + a2 * math.atan(rho * x)

    xmax = 100.0
    xs = np.linspace(1e-8, xmax, 20000)
    vals = np.array([f(float(x)) for x in xs], dtype=float)
    for i in range(len(xs) - 1):
        if not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
            continue
        if vals[i] == 0.0:
            return float(xs[i])
        if vals[i] * vals[i + 1] < 0.0:
            lo, hi = float(xs[i]), float(xs[i + 1])
            flo, fhi = f(lo), f(hi)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if abs(fm) < 1e-14:
                    return mid
                if flo * fm <= 0.0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            return 0.5 * (lo + hi)
    return None


def chua_equilibria_for_params(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    if normalize_chua_model(p.get("model", "nonsmooth")) == "arctan":
        beta = float(p["beta"])
        gamma = float(p["gamma"])
        out = {"E0": np.array([0.0, 0.0, 0.0], dtype=float)}
        xp = arctan_equilibrium_root(p)
        if xp is not None and np.isfinite(xp):
            yp = gamma / (beta + gamma) * xp
            zp = -beta / (beta + gamma) * xp
            out["E+"] = np.array([xp, yp, zp], dtype=float)
            out["E-"] = -out["E+"]
        else:
            out["E+"] = np.array([np.nan, np.nan, np.nan], dtype=float)
            out["E-"] = np.array([np.nan, np.nan, np.nan], dtype=float)
        return out

    beta = float(p["beta"])
    gamma = float(p["gamma"])
    m0 = float(p["m0"])
    m1 = float(p["m1"])
    out = {"E0": np.array([0.0, 0.0, 0.0], dtype=float)}
    A = m0 - m1
    den = (beta + gamma) * m1 + beta
    if abs(den) > 1e-12:
        xeq = -((beta + gamma) * A) / den
        if abs(xeq) > 1.0:
            fx = m1 * xeq + A
            out["E+"] = np.array([xeq, xeq + fx, fx], dtype=float)
        else:
            out["E+"] = np.array([np.nan, np.nan, np.nan], dtype=float)
        out["E-"] = -out["E+"]
    else:
        out["E+"] = np.array([np.nan, np.nan, np.nan], dtype=float)
        out["E-"] = np.array([np.nan, np.nan, np.nan], dtype=float)
    return out


def local_jacobian_at_equilibrium(p: Dict[str, float], eq: np.ndarray) -> np.ndarray:
    if normalize_chua_model(p.get("model", "nonsmooth")) == "arctan":
        alpha = float(p["alpha"])
        beta = float(p["beta"])
        gamma = float(p["gamma"])
        a1 = float(p.get("a1", 0.4))
        a2 = float(p.get("a2", -1.5585))
        rho = float(p.get("rho", 1.0))
        xeq = float(np.asarray(eq, dtype=float)[0])
        slope = a1 + (a2 * rho) / (1.0 + (rho * xeq) ** 2)
        return np.array([
            [-alpha * (1.0 + slope), alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -beta, -gamma],
        ], dtype=float)

    params = {
        "alpha": np.float64(p["alpha"]),
        "beta": np.float64(p["beta"]),
        "gamma": np.float64(p["gamma"]),
        "m0": np.float64(p["m0"]),
        "m1": np.float64(p["m1"]),
    }
    P, qvec, r = chua_ic.chua_matrices(params)
    slope_correction = 0.0
    if abs(float(np.asarray(eq, dtype=float)[0])) < 1.0:
        slope_correction = float(params["m0"] - params["m1"])
    return np.asarray(P + slope_correction * np.outer(qvec, r), dtype=float)


def plot_matignon_equilibrium_spectrum(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ensure_current_chua_params(cfg)
    p = dict(chua_ic.PARAMS)
    qord = validate_fractional_order(cfg["frac_order"])
    theta = float(qord * np.pi / 2.0)
    theta_deg = float(np.degrees(theta))
    equilibria = chua_equilibria_for_params(p)
    spectra: Dict[str, Any] = {}
    eig_points: List[complex] = []

    for name, eq in equilibria.items():
        eq_arr = np.asarray(eq, dtype=float)
        if not np.all(np.isfinite(eq_arr)):
            continue
        J = local_jacobian_at_equilibrium(p, eq_arr)
        eigvals = np.linalg.eigvals(J)
        eig_points.extend([complex(v) for v in eigvals])
        spectra[name] = {
            "equilibrium": eq_arr.tolist(),
            "eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eigvals],
            "matignon_stable": [bool(abs(np.angle(v)) > theta) for v in eigvals],
        }

    arr = np.array(eig_points, dtype=np.complex128)
    if arr.size:
        max_abs = max(1.0, float(np.max(np.abs(np.concatenate([arr.real, arr.imag])))))
    else:
        max_abs = 1.0
    lim = 1.18 * max_abs

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.set_facecolor("#ecfdf5")
    unstable_sector = Wedge(
        (0.0, 0.0),
        2.2 * lim,
        -theta_deg,
        theta_deg,
        color="#fee2e2",
        alpha=0.72,
        zorder=0,
    )
    ax.add_patch(unstable_sector)
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.85)
    ax.axvline(0.0, color="#6b7280", ls="--", lw=0.85)

    def plot_ray(angle_rad: float, *, color: str, linestyle: str, linewidth: float) -> None:
        ray_len = 1.75 * lim
        ax.plot(
            [0.0, ray_len * np.cos(angle_rad)],
            [0.0, ray_len * np.sin(angle_rad)],
            color=color,
            ls=linestyle,
            lw=linewidth,
            zorder=2,
        )

    plot_ray(theta, color="#dc2626", linestyle="-", linewidth=1.2)
    plot_ray(-theta, color="#dc2626", linestyle="-", linewidth=1.2)

    markers = {"E0": "o", "E+": "^", "E-": "v"}
    colors = {"E0": "#111827", "E+": "#7c3aed", "E-": "#d97706"}
    text_offsets = {
        "E0": (0.035 * lim, 0.000 * lim),
        "E+": (-0.135 * lim, 0.050 * lim),
        "E-": (0.035 * lim, -0.055 * lim),
    }
    equilibrium_handles: List[Line2D] = []
    for name, info in spectra.items():
        eigvals = np.array([complex(re, im) for re, im in info["eigenvalues"]], dtype=np.complex128)
        eig_stable = [bool(x) for x in info["matignon_stable"]]
        eq_is_stable = bool(all(eig_stable))
        equilibrium_handles.append(
            Line2D(
                [0],
                [0],
                marker=markers.get(name, "s"),
                color="none",
                markerfacecolor=colors.get(name, NYQUIST_W_COLOR),
                markeredgecolor="#16a34a" if eq_is_stable else "#dc2626",
                markeredgewidth=1.4,
                markersize=7.0,
                label=f"{name}: {'compatible' if eq_is_stable else 'no compatible'}",
            )
        )
        for idx, val in enumerate(eigvals, start=1):
            is_stable = eig_stable[idx - 1]
            ax.scatter(
                np.real(val),
                np.imag(val),
                s=72,
                marker=markers.get(name, "s"),
                c=colors.get(name, NYQUIST_W_COLOR),
                edgecolors="#16a34a" if is_stable else "#dc2626",
                linewidths=1.3,
                zorder=4,
            )
            dx, dy = text_offsets.get(name, (0.035 * lim, 0.0))
            ax.text(np.real(val) + dx, np.imag(val) + dy, f"{name}.{idx}", fontsize=8, color="#111827", va="center")

    ax.text(
        0.03,
        0.97,
        rf"$q={qord:.3f}$, $\phi_M=q\pi/2={theta_deg:.2f}^\circ$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": WHITE_BG, "edgecolor": "#d1d5db", "alpha": 0.88},
    )
    ax.text(
        0.73,
        0.53,
        "sector\ninestable",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#991b1b",
    )
    ax.text(
        0.18,
        0.88,
        "region compatible\ncon Matignon",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#166534",
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Re$\{\lambda\}$")
    ax.set_ylabel(r"Im$\{\lambda\}$")
    style_white_axis(ax, grid=True, grid_alpha=0.22)
    ax.set_facecolor("#ecfdf5")
    legend_handles = [
        Patch(facecolor="#ecfdf5", edgecolor="#86efac", label=r"compatible: $|\arg(\lambda)|>q\pi/2$"),
        Patch(facecolor="#fee2e2", edgecolor="#fca5a5", label=r"inestable: $|\arg(\lambda)|\leq q\pi/2$"),
        Line2D([0], [0], color="#dc2626", lw=1.2, label=rf"frontera q actual ({theta_deg:.2f}$^\circ$)"),
    ] + equilibrium_handles
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7.4, framealpha=0.9)
    save_pdf(fig, cfg["outputs"]["matignon_equilibria_pdf"])
    return {
        "q": qord,
        "matignon_angle_rad": theta,
        "matignon_angle_deg": theta_deg,
        "equilibria": spectra,
        "figure": str(cfg["outputs"]["matignon_equilibria_pdf"]),
        "interpretation": "Estabilidad local fraccionaria requiere |arg(lambda)| > q*pi/2 para todos los autovalores.",
    }



def integrate_efork3_c(
    x0: np.ndarray,
    p: Dict[str, Any],
    qord: float,
    h: float,
    Lm: float,
    t_total: float,
    *,
    k: float = 0.0,
    eps: float = 1.0,
) -> np.ndarray:
    qord = validate_fractional_order(qord)
    lib = load_fractional_backend(CONFIG)
    configure_fractional_backend_from_params(
        lib,
        p,
        workers=int(CONFIG.get("native_efork", {}).get("workers", 1)),
    )
    rows = int(lib.efork_rows(float(t_total), float(h)))
    if rows <= 0:
        raise RuntimeError("El backend EFORK C devolvio un numero de filas invalido.")
    out = np.empty(rows * 4, dtype=np.float64)
    seed = np.asarray(x0, dtype=float)
    rc = int(lib.integrate_chua_efork3(
        float(seed[0]), float(seed[1]), float(seed[2]),
        float(qord), float(h), float(Lm), float(t_total), float(k), float(eps), out
    ))
    if rc != 0:
        raise RuntimeError(f"integrate_chua_efork3 devolvio error {rc}")
    return out.reshape((rows, 4))


def integrate_original(x0: np.ndarray, p: Dict[str, Any], qord: float, h: float, Lm: float, t_total: float) -> np.ndarray:
    return integrate_efork3_c(x0, p, qord, h, Lm, t_total, k=0.0, eps=1.0)


HIDDEN_PROBE_CLASS_COLORS = {
    "EQ": "#ef4444",
    "DIV": "#2563eb",
    "TARGET": BASIN_SEED_COLOR,
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
    """Select a deterministic subset of backend hiddenness probes.

    The subset is class-stratified so rare TARGET hits remain visible. That is
    essential for the scientific reading: such hits weaken a hiddenness claim
    and should not be averaged away in a clean-looking figure.
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


def style_white_axis(ax, *, grid: bool = True, grid_alpha: float = 0.22) -> None:
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
        ax.grid(True, color="#d1d5db", alpha=grid_alpha, linewidth=0.6)
    else:
        ax.grid(False)


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


def integrate_fractional_probe_trajectory(cfg: Dict[str, Any], x0: np.ndarray) -> np.ndarray:
    """Integrate one Chua no-suave probe with EFORK for the illustration.

    The color/class still comes from the C hiddenness backend. This integration
    supplies the geometry of the sampled initial condition in the same
    Caputo-EFORK numerical frame as the fractional pipeline; when native EFORK
    is enabled, it is routed through the C EFORK-3 backend.
    """
    ensure_current_chua_params(cfg)
    return integrate_original(
        np.asarray(x0, dtype=float),
        dict(chua_ic.PARAMS),
        validate_fractional_order(float(cfg["frac_order"])),
        float(cfg["verify_hidden"]["h"]),
        float(cfg["verify_hidden"]["Lm"]),
        float(cfg["hidden_illustration"]["t_total"]),
    )


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
    save_pdf(fig, path)


def plot_hiddenness_illustration(cfg: Dict[str, Any], nyq_data: Dict[str, Any], hidden_summary: Dict[str, Any] | None) -> Dict[str, Any]:
    """Create overview and zoom figures for hidden-attractor evidence.

    The green orbit is the continued candidate attractor. Colored probes start
    exactly at the sampled equilibrium-neighborhood initial conditions from
    `hidden_target_check_frac.csv`, and their labels preserve the backend
    classification. A TARGET color therefore means the hiddenness conclusion is
    not strong and must be reported conservatively.
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
            traj = integrate_fractional_probe_trajectory(cfg, hidden_probe_start(row))
        except Exception as exc:  # pragma: no cover - visualization should not abort the quantitative run
            row = dict(row)
            row["integration_error"] = str(exc)
            traj = np.column_stack(([0.0], hidden_probe_start(row)[None, :]))
        probe_trajs.append((row, traj))

    final = np.asarray(nyq_data["results"][-1]["traj"], dtype=float)
    attractor_xyz = downsample_rows(final[:, 1:4], int(cfg["hidden_illustration"]["max_attractor_points"]))
    target_seed = np.asarray(nyq_data["final_state"], dtype=float)
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
        "method": "EFORK visualization of hidden verification probes from backend CSV",
        "detail_csv": str(csv_path),
        "total_counts": total_counts,
        "selected_counts": selected_counts,
        "outputs": {
            "overview_pdf": str(cfg["outputs"]["hidden_illustration_overview_pdf"]),
            "zoom_pdf": str(cfg["outputs"]["hidden_illustration_zoom_pdf"]),
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



def integrate_linearized_family(x0: np.ndarray, p: Dict[str, Any], qord: float, k0: float, h: float, Lm: float, t_total: float) -> np.ndarray:
    return integrate_efork3_c(x0, p, qord, h, Lm, t_total, k=float(k0), eps=0.0)



def local_maxima(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    if s.size < 3:
        return np.array([], dtype=float)
    mask = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    idx = np.nonzero(mask)[0] + 1
    return s[idx]



def sample_selected_results(results: List[Dict[str, Any]], max_curves: int = 6) -> List[Dict[str, Any]]:
    if len(results) <= max_curves:
        return results
    idx = np.linspace(0, len(results) - 1, max_curves, dtype=int)
    idx = list(dict.fromkeys(idx.tolist()))
    return [results[i] for i in idx]



def classify_trajectory(traj: np.ndarray, eqs: Dict[str, np.ndarray], div_threshold: float, eq_tol: float, hidden_amp_threshold: float) -> int:
    xyz = np.asarray(traj[:, 1:4], dtype=float)
    radius = np.linalg.norm(xyz, axis=1)
    if np.any(~np.isfinite(radius)) or np.max(radius) > div_threshold:
        return 3
    tail = xyz[max(0, int(0.8 * len(xyz))):]
    center = np.mean(tail, axis=0)
    tail_std = np.std(tail, axis=0)
    for name in ("E0", "E+", "E-"):
        eq = eqs[name]
        if np.all(np.isfinite(eq)) and np.linalg.norm(center - eq) <= eq_tol and np.linalg.norm(tail_std) < 0.4:
            return 0
    xamp = np.max(tail[:, 0]) - np.min(tail[:, 0])
    if xamp < hidden_amp_threshold:
        return 0
    xmean = float(np.mean(tail[:, 0]))
    return 1 if xmean >= 0.0 else 2


def _bifurcation_param_setup(
    param_name: str,
    value: float,
    base_params: Dict[str, float],
    frac_order: float,
) -> Tuple[Dict[str, np.float64], float]:
    p: Dict[str, Any] = {}
    for k, v in base_params.items():
        p[k] = normalize_chua_model(v) if k == "model" else np.float64(v)
    qord = validate_fractional_order(frac_order)
    if param_name == "q":
        qord = validate_fractional_order(value)
    elif param_name == "alpha":
        p["alpha"] = np.float64(value)
    elif param_name == "beta":
        p["beta"] = np.float64(value)
    else:
        raise ValueError("param_name debe ser 'q', 'alpha' o 'beta'")
    return p, qord


def _bifurcation_peaks_from_seed(
    seed: Iterable[float],
    params: Dict[str, np.float64],
    qord: float,
    bcfg: Dict[str, float],
) -> List[float]:
    T = integrate_original(
        np.asarray(seed, dtype=float),
        params,
        qord=qord,
        h=float(bcfg["h"]),
        Lm=float(bcfg["Lm"]),
        t_total=float(bcfg["t_total"]),
    )
    tail = T[T[:, 0] >= float(bcfg["t_burn"])]
    if tail.size == 0:
        return []
    x = tail[:, 1]
    peaks = local_maxima(x)
    if peaks.size == 0:
        peaks = np.array([np.max(x)], dtype=float)
    peaks = peaks[-int(bcfg["max_peaks"]):]
    return [float(v) for v in peaks]


def _bifurcation_worker(task: Tuple[Any, ...]) -> Tuple[int, float, List[float], List[float]]:
    force_single_openmp_thread_current_process()
    idx, param_name, value, seed_pos, seed_neg, base_params, frac_order, bcfg = task
    params, qord = _bifurcation_param_setup(param_name, float(value), base_params, float(frac_order))
    pos_peaks = _bifurcation_peaks_from_seed(seed_pos, params, qord, bcfg)
    neg_peaks = _bifurcation_peaks_from_seed(seed_neg, params, qord, bcfg)
    return int(idx), float(value), pos_peaks, neg_peaks


def compute_bifurcation_sweep_parallel(
    param_name: str,
    values: Iterable[float],
    seed_pos: np.ndarray,
    seed_neg: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    vals = np.asarray(list(values), dtype=float)
    bcfg = cfg["bifurcation"]
    nvals = len(vals)
    workers = max(1, min(int(bcfg.get("workers", default_bifurcation_workers())), max(1, nvals)))
    chunksize = max(1, int(bcfg.get("chunksize", 1)))
    progress_every = max(1, int(bcfg.get("progress_every", max(1, nvals // 10))))
    base_params = {k: (normalize_chua_model(v) if k == "model" else float(v)) for k, v in chua_ic.PARAMS.items()}
    simple_bcfg = {
        "h": float(bcfg["h"]),
        "Lm": float(bcfg["Lm"]),
        "t_total": float(bcfg["t_total"]),
        "t_burn": float(bcfg["t_burn"]),
        "max_peaks": int(bcfg["max_peaks"]),
    }
    pos_seed = np.asarray(seed_pos, dtype=float).tolist()
    neg_seed = np.asarray(seed_neg, dtype=float).tolist()
    tasks = [
        (idx, param_name, float(v), pos_seed, neg_seed, base_params, float(cfg["frac_order"]), simple_bcfg)
        for idx, v in enumerate(vals)
    ]
    log(
        f"Bifurcacion {param_name}: modo paralelo con {workers} procesos; "
        "cada parametro usa las mismas semillas iniciales; "
        "OMP_NUM_THREADS=1 y OMP_THREAD_LIMIT=1 dentro de cada worker."
    )
    records: List[Tuple[int, float, List[float], List[float]]] = []
    done = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=force_single_openmp_thread_current_process) as pool:
        for record in pool.map(_bifurcation_worker, tasks, chunksize=chunksize):
            records.append(record)
            done += 1
            if done == 1 or done % progress_every == 0 or done == nvals:
                percent = 100.0 * done / max(1, nvals)
                log(
                    f"Bifurcacion {param_name}: paralelo {done}/{nvals} "
                    f"({percent:.1f}%, {format_elapsed(time.perf_counter() - started)})."
                )
    records.sort(key=lambda item: item[0])

    pts_pos_x: List[float] = []
    pts_pos_y: List[float] = []
    pts_neg_x: List[float] = []
    pts_neg_y: List[float] = []
    for _, v, pos_peaks, neg_peaks in records:
        pts_pos_x.extend([v] * len(pos_peaks))
        pts_pos_y.extend(pos_peaks)
        pts_neg_x.extend([v] * len(neg_peaks))
        pts_neg_y.extend(neg_peaks)
    log(
        f"Bifurcacion {param_name}: terminada en {format_elapsed(time.perf_counter() - started)}; "
        f"puntos guardados={len(pts_pos_x) + len(pts_neg_x)}."
    )
    return {
        "vals": vals,
        "pos_x": np.asarray(pts_pos_x, dtype=float),
        "pos_y": np.asarray(pts_pos_y, dtype=float),
        "neg_x": np.asarray(pts_neg_x, dtype=float),
        "neg_y": np.asarray(pts_neg_y, dtype=float),
    }


def compute_bifurcation_sweep_c(
    param_name: str,
    values: Iterable[float],
    seed_pos: np.ndarray,
    seed_neg: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    vals = np.ascontiguousarray(np.asarray(list(values), dtype=np.float64))
    bcfg = cfg["bifurcation"]
    nvals = int(vals.size)
    max_peaks = int(bcfg["max_peaks"])
    if nvals <= 0:
        return {
            "vals": vals,
            "pos_x": np.array([], dtype=float),
            "pos_y": np.array([], dtype=float),
            "neg_x": np.array([], dtype=float),
            "neg_y": np.array([], dtype=float),
        }
    param_type = {"q": 0, "alpha": 1, "beta": 2}.get(param_name)
    if param_type is None:
        raise ValueError("param_name debe ser 'q', 'alpha' o 'beta'")

    lib = load_fractional_backend(cfg)
    configure_fractional_backend_for_case(lib, cfg)
    seed_strategy = normalize_bifurcation_seed_strategy(bcfg.get("seed_strategy", "continuation"))
    workers = int(bcfg.get("workers", cfg.get("native_efork", {}).get("workers", 1)))
    if seed_strategy == "continuation":
        workers = 1
    lib.set_frac_backend_workers(workers)
    continue_seed = 1 if seed_strategy == "continuation" else 0
    pos_x = np.empty(nvals * max_peaks, dtype=np.float64)
    pos_y = np.empty(nvals * max_peaks, dtype=np.float64)
    neg_x = np.empty(nvals * max_peaks, dtype=np.float64)
    neg_y = np.empty(nvals * max_peaks, dtype=np.float64)
    pos_count = np.zeros(nvals, dtype=np.int32)
    neg_count = np.zeros(nvals, dtype=np.int32)
    seed_pos_arr = np.ascontiguousarray(np.asarray(seed_pos, dtype=np.float64))
    seed_neg_arr = np.ascontiguousarray(np.asarray(seed_neg, dtype=np.float64))

    log(
        f"Bifurcacion {param_name}: backend C EFORK-3; valores={nvals}; "
        f"python_workers=1; omp_threads={workers if not continue_seed else 1}; "
        f"backend_openmp_active={bool(cfg.get('native_efork', {}).get('openmp_active', cfg.get('native_efork', {}).get('openmp', False)))}; "
        f"seed_strategy={seed_strategy}; "
        f"stage_kind={'causal' if continue_seed else 'embarrassingly_parallel'}."
    )
    started = time.perf_counter()
    rc = int(lib.compute_bifurcation_sweep_efork3(
        int(param_type), vals, nvals, seed_pos_arr, seed_neg_arr,
        float(cfg["frac_order"]), float(bcfg["h"]), float(bcfg["Lm"]),
        float(bcfg["t_total"]), float(bcfg["t_burn"]), max_peaks, int(continue_seed),
        pos_x, pos_y, pos_count, neg_x, neg_y, neg_count
    ))
    if rc != 0:
        raise RuntimeError(f"compute_bifurcation_sweep_efork3({param_name}) devolvio error {rc}")

    pts_pos_x: List[float] = []
    pts_pos_y: List[float] = []
    pts_neg_x: List[float] = []
    pts_neg_y: List[float] = []
    for i, _v in enumerate(vals):
        pc = int(pos_count[i])
        nc = int(neg_count[i])
        if pc > 0:
            start = i * max_peaks
            pts_pos_x.extend(pos_x[start:start + pc].tolist())
            pts_pos_y.extend(pos_y[start:start + pc].tolist())
        if nc > 0:
            start = i * max_peaks
            pts_neg_x.extend(neg_x[start:start + nc].tolist())
            pts_neg_y.extend(neg_y[start:start + nc].tolist())
    log(
        f"Bifurcacion {param_name}: C terminado en {format_elapsed(time.perf_counter() - started)}; "
        f"puntos guardados={len(pts_pos_x) + len(pts_neg_x)}."
    )
    return {
        "vals": vals,
        "pos_x": np.asarray(pts_pos_x, dtype=float),
        "pos_y": np.asarray(pts_pos_y, dtype=float),
        "neg_x": np.asarray(pts_neg_x, dtype=float),
        "neg_y": np.asarray(pts_neg_y, dtype=float),
        "backend": "c_efork3",
    }



def compute_bifurcation_sweep(param_name: str, values: Iterable[float], seed_pos: np.ndarray, seed_neg: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    vals = np.asarray(list(values), dtype=float)
    bcfg = cfg["bifurcation"]
    nvals = len(vals)
    progress_every = max(1, int(bcfg.get("progress_every", max(1, nvals // 10))))
    seed_strategy = normalize_bifurcation_seed_strategy(bcfg.get("seed_strategy", "continuation"))
    log(
        f"Bifurcacion {param_name}: {nvals} valores x 2 semillas; "
        f"t_total={bcfg['t_total']}, h={bcfg['h']}; seed_strategy={seed_strategy}."
    )
    if bool(cfg.get("native_efork", {}).get("enabled", True)):
        return compute_bifurcation_sweep_c(param_name, vals, seed_pos, seed_neg, cfg)
    workers = int(bcfg.get("workers", 1))
    if seed_strategy == "independent" and bool(bcfg.get("parallel", False)) and workers > 1 and nvals > 1:
        return compute_bifurcation_sweep_parallel(param_name, vals, seed_pos, seed_neg, cfg)
    if seed_strategy == "independent":
        log(
            f"Bifurcacion {param_name}: modo independiente secuencial; "
            "workers=1 efectivo porque no hay paralelismo Python habilitado."
        )
    pts_pos_x: List[float] = []
    pts_pos_y: List[float] = []
    pts_neg_x: List[float] = []
    pts_neg_y: List[float] = []
    x0_pos = np.asarray(seed_pos, dtype=float).copy()
    x0_neg = np.asarray(seed_neg, dtype=float).copy()
    base_pos = x0_pos.copy()
    base_neg = x0_neg.copy()
    started = time.perf_counter()
    for idx, v in enumerate(vals, start=1):
        if idx == 1 or idx % progress_every == 0 or idx == nvals:
            percent = 100.0 * idx / max(1, nvals)
            log(
                f"Bifurcacion {param_name}: valor {idx}/{nvals} = {v:.8g} "
                f"({percent:.1f}%, {format_elapsed(time.perf_counter() - started)})."
            )
        p = dict(chua_ic.PARAMS)
        qord = validate_fractional_order(cfg["frac_order"])
        if param_name == "q":
            qord = validate_fractional_order(v)
        elif param_name == "alpha":
            p["alpha"] = np.float64(v)
        elif param_name == "beta":
            p["beta"] = np.float64(v)
        else:
            raise ValueError("param_name debe ser 'q', 'alpha' o 'beta'")
        seed_triplets = (
            (x0_pos if seed_strategy == "continuation" else base_pos, pts_pos_x, pts_pos_y),
            (x0_neg if seed_strategy == "continuation" else base_neg, pts_neg_x, pts_neg_y),
        )
        for current_seed, xs, ys in seed_triplets:
            T = integrate_original(current_seed, p, qord=qord, h=float(bcfg["h"]), Lm=float(bcfg["Lm"]), t_total=float(bcfg["t_total"]))
            if seed_strategy == "continuation":
                current_seed[:] = T[-1, 1:4]
            tail = T[T[:, 0] >= float(bcfg["t_burn"])]
            if tail.size == 0:
                continue
            x = tail[:, 1]
            peaks = local_maxima(x)
            if peaks.size == 0:
                peaks = np.array([np.max(x)], dtype=float)
            peaks = peaks[-int(bcfg["max_peaks"]):]
            xs.extend([v] * len(peaks))
            ys.extend(peaks.tolist())
    log(
        f"Bifurcacion {param_name}: terminada en {format_elapsed(time.perf_counter() - started)}; "
        f"puntos guardados={len(pts_pos_x) + len(pts_neg_x)}."
    )
    return {
        "vals": vals,
        "pos_x": np.asarray(pts_pos_x, dtype=float),
        "pos_y": np.asarray(pts_pos_y, dtype=float),
        "neg_x": np.asarray(pts_neg_x, dtype=float),
        "neg_y": np.asarray(pts_neg_y, dtype=float),
    }



def plot_article_style_bifurcation(data: Dict[str, np.ndarray], xlabel: str, ylabel: str, path: Path):
    xs_all, ys_all = all_bifurcation_points(data)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.scatter(data["pos_x"], data["pos_y"], s=0.65, c=BIFURCATION_POS_COLOR, alpha=0.86, linewidths=0, rasterized=True, label="semilla +")
    ax.scatter(data["neg_x"], data["neg_y"], s=0.65, c=BIFURCATION_NEG_COLOR, alpha=0.86, linewidths=0, rasterized=True, label="semilla -")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_white_axis(ax, grid=True, grid_alpha=0.22)
    ax.legend(loc="best", markerscale=4, fontsize=8)

    if xs_all.size:
        xmin, xmax = float(np.min(xs_all)), float(np.max(xs_all))
        ymin, ymax = float(np.min(ys_all)), float(np.max(ys_all))
        xpad = 0.015 * max(1e-12, xmax - xmin)
        ypad = 0.06 * max(1e-12, ymax - ymin)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    save_pdf(fig, path)



def plot_basin_planes(basin_data: Dict[str, Any], path: Path):
    cmap, norm = basin_cmap_norm()
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.6))
    labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
    panel_labels = {"xy": "(a)", "xz": "(b)", "yz": "(c)"}
    for ax, plane in zip(axs, ["xy", "xz", "yz"]):
        dat = basin_data[plane]
        ax.set_facecolor(BASIN_CLASS_COLORS[-1])
        ax.imshow(dat["grid"], origin="lower", interpolation="nearest", cmap=cmap, norm=norm, aspect="auto", extent=[dat["uvals"][0], dat["uvals"][-1], dat["vvals"][0], dat["vvals"][-1]])
        if "seed_uv" in dat:
            ax.scatter(dat["seed_uv"][0], dat["seed_uv"][1], c=BASIN_SEED_COLOR, edgecolors="black", s=75, marker="*", zorder=5)
        ax.set_xlabel(labels[plane][0])
        ax.set_ylabel(labels[plane][1])
        style_white_axis(ax, grid=False)
        ax.text(0.92, 1.03, panel_labels[plane], transform=ax.transAxes, ha="center", va="bottom", color="#b00000", fontweight="bold")
    add_basin_legend(axs[-1], loc="upper right", fontsize=7)
    save_pdf(fig, path)


def plot_single_basin_plane(dat: Dict[str, Any], plane: str, path: Path):
    cmap, norm = basin_cmap_norm()
    labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.set_facecolor(BASIN_CLASS_COLORS[-1])
    ax.imshow(dat["grid"], origin="lower", interpolation="nearest", cmap=cmap, norm=norm, aspect="auto", extent=[dat["uvals"][0], dat["uvals"][-1], dat["vvals"][0], dat["vvals"][-1]])
    if "seed_uv" in dat:
        ax.scatter(dat["seed_uv"][0], dat["seed_uv"][1], c=BASIN_SEED_COLOR, edgecolors="black", s=90, marker="*", zorder=5)
    ax.set_xlabel(labels[plane][0])
    ax.set_ylabel(labels[plane][1])
    style_white_axis(ax, grid=False)
    add_basin_legend(ax, loc="upper right", fontsize=7)
    save_pdf(fig, path)



def plot_nyquist_with_x_zoom(cfg: Dict[str, Any], nyq_data: Dict[str, Any]):
    p = chua_ic.PARAMS
    qord = chua_ic.QORD
    omg = np.logspace(np.log10(chua_ic.WMIN), np.log10(chua_ic.WMAX), 4000)
    W = np.array([chua_ic.W_frac(w, qord, p) for w in omg], dtype=np.complex128)
    aa = np.linspace(1.0 + 1e-6, 80.0, 2400)
    minus_invN = -1.0 / np.array([chua_ic.N_sat(a, p) for a in aa], dtype=float)
    pts = [pair_to_point(pair, qord, p) for pair in nyq_data["pairs"]]
    x_zoom_center = np.real(pts[int(cfg["branch_index"])] if pts else W[np.argmin(np.abs(W.imag))])
    x_margin = 0.08
    fig, axs = article_dark_subplots(1, 2, figsize=(11.0, 4.8))
    for ax in axs:
        ax.plot(W.real, W.imag, lw=1.7, color=NYQUIST_W_COLOR, label=r"$W(i\omega)$")
        ax.plot(minus_invN, np.zeros_like(minus_invN), lw=1.4, color=NYQUIST_DF_COLOR, label=r"$-1/N(a)$")
        for idx_pair, z in enumerate(pts):
            ax.scatter(
                np.real(z), np.imag(z), s=54, facecolors="none", edgecolors="#ff1f1f",
                linewidths=1.5, zorder=4, label="cruce Nyquist/DF" if idx_pair == 0 else "_nolegend_"
            )
        ax.axhline(0.0, color=ARTICLE_DARK_TEXT, lw=0.7, alpha=0.65)
        ax.set_xlabel(r"Re$(W(i\omega))$")
        ax.set_ylabel(r"Im$(W(i\omega))$")
        style_dark_axis(ax)
    dark_legend(axs[0], loc="best", fontsize=8)
    axs[1].set_xlim(x_zoom_center - x_margin, x_zoom_center + x_margin)
    ymask = (W.real >= x_zoom_center - x_margin) & (W.real <= x_zoom_center + x_margin)
    if np.any(ymask):
        yvals = W.imag[ymask]
        ypad = max(0.03, 0.15 * float(np.max(np.abs(yvals))))
        axs[1].set_ylim(float(np.min(yvals) - ypad), float(np.max(yvals) + ypad))
    axs[0].text(0.5, 1.02, "(a)", transform=axs[0].transAxes, color=ARTICLE_DARK_TEXT, ha="center", va="bottom", fontweight="bold")
    axs[1].text(0.5, 1.02, "(b)", transform=axs[1].transAxes, color=ARTICLE_DARK_TEXT, ha="center", va="bottom", fontweight="bold")
    save_pdf(fig, cfg["outputs"]["nyquist_zoom_pdf"])



def plot_linearized_vs_original(cfg: Dict[str, Any], nyq_data: Dict[str, Any]):
    final = np.asarray(nyq_data["results"][-1]["traj"], dtype=float)
    t = final[:, 0]
    eigvec = np.asarray(nyq_data["eigvec"], dtype=np.complex128)
    omega0 = float(nyq_data["omega0"])
    a0 = float(nyq_data["a0"])
    harmonic_xyz = a0 * np.real(np.exp(1j * omega0 * t)[:, None] * eigvec[None, :])
    Tlin = np.column_stack((t, harmonic_xyz))
    Tori = final
    fig = plt.figure(figsize=(10.6, 7.4))
    ax1 = fig.add_subplot(221)
    ax1.plot(Tlin[:, 0], Tlin[:, 1], lw=1.0, color=LINEARIZED_COLOR, label="Nyquist/DF")
    ax1.plot(Tori[:, 0], Tori[:, 1], lw=1.0, color=ORIGINAL_COLOR, label="original")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x(t)")
    style_white_axis(ax1, grid=True, grid_alpha=0.22)
    ax1.legend(loc="best", fontsize=8)
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot(Tlin[:, 1], Tlin[:, 2], Tlin[:, 3], lw=0.8, color=LINEARIZED_COLOR, label="Nyquist/DF")
    ax2.plot(Tori[:, 1], Tori[:, 2], Tori[:, 3], lw=0.8, color=ORIGINAL_COLOR, label="original")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    style_white_axis(ax2, grid=True, grid_alpha=0.18)
    ax2.legend(loc="best", fontsize=8)
    ax3 = fig.add_subplot(223)
    ax3.plot(Tlin[:, 1], Tlin[:, 2], lw=0.9, color=LINEARIZED_COLOR, label="Nyquist/DF")
    ax3.plot(Tori[:, 1], Tori[:, 2], lw=0.9, color=ORIGINAL_COLOR, label="original")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    style_white_axis(ax3, grid=True, grid_alpha=0.22)
    ax3.legend(loc="best", fontsize=8)
    ax4 = fig.add_subplot(224)
    ax4.plot(Tlin[:, 1], Tlin[:, 3], lw=0.9, color=LINEARIZED_COLOR, label="Nyquist/DF")
    ax4.plot(Tori[:, 1], Tori[:, 3], lw=0.9, color=ORIGINAL_COLOR, label="original")
    ax4.set_xlabel("x")
    ax4.set_ylabel("z")
    style_white_axis(ax4, grid=True, grid_alpha=0.22)
    ax4.legend(loc="best", fontsize=8)
    save_pdf(fig, cfg["outputs"]["linear_vs_original_pdf"])



def plot_continuation_story(cfg: Dict[str, Any], results: List[Dict[str, Any]]):
    chosen = sample_selected_results(results, max_curves=int(cfg["article_style"]["max_cont_curves"]))
    fig = plt.figure(figsize=(10.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("plasma")
    eps_all = np.array([float(r["eps"]) for r in results], dtype=float)
    eps_min = float(np.min(eps_all))
    eps_max = float(np.max(eps_all))
    xin_path = np.array([np.asarray(r["x_in"], dtype=float) for r in results], dtype=float)
    xout_path = np.array([np.asarray(r["x_out"], dtype=float) for r in results], dtype=float)
    ax.plot(xin_path[:, 0], xin_path[:, 1], xin_path[:, 2], "k--", lw=1.0, label="entrada epsilon")
    ax.plot(xout_path[:, 0], xout_path[:, 1], xout_path[:, 2], color="0.45", ls=":", lw=1.0, label="salida epsilon")
    for r in chosen:
        eps = float(r["eps"])
        color = cmap((eps - eps_min) / max(1e-12, eps_max - eps_min))
        T = np.asarray(r["traj"], dtype=float)
        ax.plot(T[:, 1], T[:, 2], T[:, 3], lw=1.0, color=color, alpha=0.95)
        xi = np.asarray(r["x_in"], dtype=float)
        xo = np.asarray(r["x_out"], dtype=float)
        ax.scatter(xi[0], xi[1], xi[2], s=25, c=[color], marker="o")
        ax.scatter(xo[0], xo[1], xo[2], s=18, c=[color], marker="s")
    first = np.asarray(results[0]["traj"], dtype=float)
    last = np.asarray(results[-1]["traj"], dtype=float)
    ax.plot(first[:, 1], first[:, 2], first[:, 3], lw=1.4, color=NYQUIST_W_COLOR, label="primer paso")
    ax.plot(last[:, 1], last[:, 2], last[:, 3], lw=1.6, color=ORIGINAL_COLOR, label="paso final")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    style_white_axis(ax, grid=True, grid_alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    save_pdf(fig, cfg["outputs"]["continuation_story_pdf"])



def plot_final_attractor_with_planes(cfg: Dict[str, Any], results: List[Dict[str, Any]]):
    T = np.asarray(results[-1]["traj"], dtype=float)
    plane_specs = [
        ("xy", 1, 2, "x", "y", cfg["outputs"]["final_attr_xy_pdf"]),
        ("xz", 1, 3, "x", "z", cfg["outputs"]["final_attr_xz_pdf"]),
        ("yz", 2, 3, "y", "z", cfg["outputs"]["final_attr_yz_pdf"]),
    ]
    for plane, col_u, col_v, xlabel, ylabel, path in plane_specs:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.plot(T[:, col_u], T[:, col_v], lw=0.75, color=ORIGINAL_COLOR, label=plane)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        style_white_axis(ax, grid=True, grid_alpha=0.22)
        ax.legend(loc="best", fontsize=8)
        save_pdf(fig, path)


def _signal_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name in {"hann", "hanning"}:
        return np.hanning(n)
    if name in {"none", "rect", "rectangular"}:
        return np.ones(n)
    raise ValueError(f"Ventana no soportada: {name}")


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
        raise ValueError("La serie es demasiado corta para Welch.")

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


def compute_psd_summary(results: List[Dict[str, Any]], h: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    spectral_data = extract_spectral_series(results)
    t = spectral_data["t"]
    X = spectral_data["X"]
    psd_cfg = cfg["psd"]
    comp_names = psd_cfg["component_names"]
    summary_by_component = {}
    spectra = {}

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
        fd = float(f[kmax])
        wd = float(2.0 * np.pi * fd)
        summary_by_component[name] = {
            "dominant_frequency_hz": fd,
            "dominant_frequency_rad_s": wd,
            "peak_psd": float(pxx[kmax]),
            "n_samples": int(len(X[:, j])),
        }
        spectra[name] = (f, pxx)

    primary_idx = int(psd_cfg["primary_component_index"])
    primary_name = comp_names[primary_idx]
    return {
        "t": t,
        "X": X,
        "spectra": spectra,
        "summary_by_component": summary_by_component,
        "primary_component": primary_name,
        "omega_d": summary_by_component[primary_name]["dominant_frequency_rad_s"],
        "f_d": summary_by_component[primary_name]["dominant_frequency_hz"],
    }


def extract_spectral_series(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    traj = np.asarray(results[-1]["traj"], dtype=float)
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


def compute_fft_summary(spectral_data: Dict[str, Any], h: float, cfg: Dict[str, Any], omega0: float | None = None) -> Dict[str, Any]:
    X = np.asarray(spectral_data["X"], dtype=float)
    fft_cfg = cfg["fft"]
    comp_names = fft_cfg["component_names"]
    spectra = {}
    summary_by_component = {}

    for j, name in enumerate(comp_names):
        signal = X[:, j] - np.mean(X[:, j])
        n = signal.size
        if n < 16:
            raise ValueError("La serie es demasiado corta para FFT.")
        win = _signal_window(n, str(fft_cfg.get("window", "hann")))
        coherent_gain = float(np.sum(win) / n)
        yf = np.fft.rfft(signal * win)
        freq = np.fft.rfftfreq(n, d=float(h))
        amp = (2.0 / (n * max(coherent_gain, 1e-15))) * np.abs(yf)
        start_idx = 1 if fft_cfg.get("ignore_zero_bin", True) else 0
        kmax = int(np.argmax(amp[start_idx:]) + start_idx)
        top_peaks = top_spectrum_peaks(freq, amp, int(fft_cfg.get("top_n", 5)), start_idx)
        omega = 2.0 * np.pi * freq
        nyquist_peak: Dict[str, float] | None = None
        nearest_nyquist: Dict[str, float] | None = None
        if omega0 is not None and np.isfinite(float(omega0)) and float(omega0) > 0.0:
            omega0_val = float(omega0)
            nearest_idx = int(np.argmin(np.abs(omega - omega0_val)))
            nearest_nyquist = {
                "frequency_hz": float(freq[nearest_idx]),
                "frequency_rad_s": float(omega[nearest_idx]),
                "value": float(amp[nearest_idx]),
                "relative_mismatch_eta": float(abs(omega[nearest_idx] - omega0_val) / omega0_val),
            }
            bin_width = float(omega[1] - omega[0]) if omega.size > 1 else 0.0
            half_width = max(
                3.0 * max(bin_width, 0.0),
                float(fft_cfg.get("nyquist_peak_window_fraction", 0.25)) * omega0_val,
            )
            band = np.nonzero((omega >= omega0_val - half_width) & (omega <= omega0_val + half_width))[0]
            band = band[band >= start_idx]
            if band.size:
                band_idx = int(band[np.argmax(amp[band])])
                nyquist_peak = {
                    "frequency_hz": float(freq[band_idx]),
                    "frequency_rad_s": float(omega[band_idx]),
                    "value": float(amp[band_idx]),
                    "relative_mismatch_eta": float(abs(omega[band_idx] - omega0_val) / omega0_val),
                    "search_half_width_rad_s": float(half_width),
                }
        summary_by_component[name] = {
            "dominant_frequency_hz": float(freq[kmax]),
            "dominant_frequency_rad_s": float(omega[kmax]),
            "peak_amplitude": float(amp[kmax]),
            "n_samples": int(n),
            "window": str(fft_cfg.get("window", "hann")),
            "frequency_resolution_hz": float(freq[1] - freq[0]) if freq.size > 1 else float("nan"),
            "top_frequencies": top_peaks,
            "nearest_bin_to_omega0": nearest_nyquist,
            "peak_near_omega0": nyquist_peak,
        }
        spectra[name] = (freq, amp)

    primary_idx = int(fft_cfg["primary_component_index"])
    primary_name = comp_names[primary_idx]
    return {
        "spectra": spectra,
        "summary_by_component": summary_by_component,
        "primary_component": primary_name,
        "omega_d": summary_by_component[primary_name]["dominant_frequency_rad_s"],
        "f_d": summary_by_component[primary_name]["dominant_frequency_hz"],
    }


def plot_clean_psd(cfg: Dict[str, Any], psd_data: Dict[str, Any], omega0: float):
    fig, axs = article_dark_subplots(3, 1, figsize=(7.4, 8.1), sharex=True)
    comp_names = cfg["psd"]["component_names"]
    for ax, name in zip(axs, comp_names):
        f, pxx = psd_data["spectra"][name]
        ax.semilogy(2.0 * np.pi * f, pxx, lw=1.2, color=NYQUIST_W_COLOR, label=f"PSD {name}")
        ax.axvline(omega0, color=NYQUIST_DF_COLOR, ls="--", lw=1.0, label=r"$\omega_0$ Nyquist/DF")
        ax.set_ylabel(f"PSD({name})")
        style_dark_axis(ax, grid_alpha=0.24)
        dark_legend(ax, loc="best", fontsize=8)
    axs[-1].set_xlabel(r"$\omega$ [rad/s]")
    save_pdf(fig, cfg["outputs"]["psd_pdf"])


def plot_clean_fft(cfg: Dict[str, Any], fft_data: Dict[str, Any], omega0: float) -> Dict[str, str]:
    comp_names = cfg["fft"]["component_names"]
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
    path_keys = ["fft_x_pdf", "fft_y_pdf", "fft_z_pdf"]
    outputs: Dict[str, str] = {}
    for name, path_key in zip(comp_names, path_keys):
        fig, ax = article_dark_subplots(figsize=(7.4, 4.9))
        f, amp = fft_data["spectra"][name]
        omega = 2.0 * np.pi * f
        ax.plot(omega, amp, lw=1.0, color=BIFURCATION_NEG_COLOR, label=f"FFT {name}")
        ax.axvline(omega0, color=NYQUIST_DF_COLOR, ls="--", lw=1.0, label=r"$\omega_0$ Nyquist/DF")
        near = fft_data["summary_by_component"][name].get("peak_near_omega0")
        if near:
            ax.scatter(
                near["frequency_rad_s"], near["value"],
                s=38, facecolors="none", edgecolors=NYQUIST_DF_COLOR, linewidths=1.1,
                zorder=5, label="pico cerca de Nyquist"
            )
        for peak in fft_data["summary_by_component"][name].get("top_frequencies", [])[: int(cfg["fft"].get("top_n", 5))]:
            ax.scatter(peak["frequency_rad_s"], peak["value"], s=18, c="#ffffff", edgecolors="#222222", linewidths=0.45, zorder=4)
        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel(f"|FFT({name})|")
        ax.set_xlim(xmin, xmax)
        style_dark_axis(ax, grid_alpha=0.24)
        dark_legend(ax, loc="best", fontsize=8)
        save_pdf(fig, cfg["outputs"][path_key])
        outputs[path_key] = str(cfg["outputs"][path_key])
    return outputs


def export_time_series_for_tisean(cfg: Dict[str, Any], psd_data: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("tisean", {}).get("enabled", False)):
        return {"enabled": False, "status": "skipped"}

    tisean_cfg = cfg["tisean"]
    t = np.asarray(psd_data["t"], dtype=float)
    X = np.asarray(psd_data["X"], dtype=float)
    np.savetxt(cfg["outputs"]["ts_x_txt"], X[:, 0], fmt="%.16e")
    np.savetxt(cfg["outputs"]["ts_xyz_txt"], np.column_stack((t, X)), fmt="%.16e")

    m = int(tisean_cfg["embedding_dim"])
    d = int(tisean_cfg["delay"])
    tw = int(tisean_cfg["theiler_window"])
    horizon = int(tisean_cfg["horizon"])
    radius = float(tisean_cfg["radius"])
    scalar = Path(cfg["outputs"]["ts_x_txt"]).name
    commands = [
        f"lyap_r -m {m} -d {d} -t {tw} -s {horizon} -r {radius:g} -o lyap_r_out.txt {scalar}",
        f"lyap_k -m {m} -d {d} -t {tw} -s {horizon} -r {radius:g} -o lyap_k_out.txt {scalar}",
        f"lyap_spec -m {m} -d {d} -t {tw} -r {radius:g} -o lyap_spec_out.txt {scalar}",
    ]
    selected = set(tisean_cfg.get("commands", ["lyap_r", "lyap_k", "lyap_spec"]))
    commands = [cmd for cmd in commands if cmd.split()[0] in selected]
    txt = "\n".join([
        "# Plantilla minima para TISEAN",
        f"# Archivo escalar principal: {scalar}",
        f"# Archivo multicanal con tiempo + estados: {Path(cfg['outputs']['ts_xyz_txt']).name}",
        "# Ajusta embedding, delay, Theiler window, horizonte y radio antes de reportar resultados.",
        "",
        *commands,
        "",
    ])
    Path(cfg["outputs"]["tisean_cmds"]).write_text(txt, encoding="utf-8")
    return {
        "enabled": True,
        "status": "exported",
        "config": {
            "embedding_dim": m,
            "delay": d,
            "theiler_window": tw,
            "horizon": horizon,
            "radius": radius,
            "commands": commands,
        },
        "outputs": {
            "ts_x_txt": str(cfg["outputs"]["ts_x_txt"]),
            "ts_xyz_txt": str(cfg["outputs"]["ts_xyz_txt"]),
            "tisean_cmds": str(cfg["outputs"]["tisean_cmds"]),
        },
    }


def _compile_lyapunov_binary(cfg: Dict[str, Any]) -> Path:
    le_cfg = cfg["lyapunov"]
    source = Path(le_cfg["source_c"])
    exe = Path(le_cfg["exe"])
    exe.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gcc", "-O3", str(source), "-lm", "-o", str(exe)]
    subprocess.run(cmd, check=True)
    return exe


def _plot_lyapunov_convergence(csv_path: Path, pdf_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(data["time"], data["lambda1"], lw=1.2, label=r"$\lambda_1$")
    ax.plot(data["time"], data["lambda2"], lw=1.2, label=r"$\lambda_2$")
    ax.plot(data["time"], data["lambda3"], lw=1.2, label=r"$\lambda_3$")
    ax.set_xlabel("t")
    ax.set_ylabel("Lyapunov exponent estimate")
    style_white_axis(ax)
    ax.legend()
    save_pdf(fig, pdf_path)


def run_lyapunov_estimate(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    le_cfg = cfg["lyapunov"]
    if not bool(le_cfg.get("enabled", False)):
        return {"enabled": False, "status": "skipped"}

    try:
        exe = _compile_lyapunov_binary(cfg)
        p = cfg["params"]
        env = dict(os.environ)
        env["CHUA_LE_CSV"] = str(cfg["outputs"]["lyapunov_csv"])
        args = [
            str(exe),
            str(float(final_state[0])), str(float(final_state[1])), str(float(final_state[2])),
            str(float(p["alpha_chua"])), str(float(p["beta"])), str(float(p["gamma_chua"])),
            str(float(p["m0"])), str(float(p["m1"])),
            str(validate_fractional_order(cfg["frac_order"])), str(float(le_cfg["h"])), str(float(le_cfg["Lm"])),
            str(float(le_cfg["t_burn"])), str(int(le_cfg["n_blocks"])), str(float(le_cfg["t_block"])),
        ]
        proc = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
        match = re.search(r"# LE_frac_standard\s+([Ee0-9+\-.]+)\s+([Ee0-9+\-.]+)\s+([Ee0-9+\-.]+)", proc.stdout)
        if not match:
            raise RuntimeError("No se pudieron extraer exponentes de Lyapunov de la salida del binario.")
        le = [float(match.group(i)) for i in range(1, 4)]
        final_match = re.search(r"# final_state\s+([Ee0-9+\-.]+)\s+([Ee0-9+\-.]+)\s+([Ee0-9+\-.]+)", proc.stdout)
        final_reported = [float(final_match.group(i)) for i in range(1, 4)] if final_match else None
        _plot_lyapunov_convergence(Path(cfg["outputs"]["lyapunov_csv"]), Path(cfg["outputs"]["lyapunov_pdf"]))
        summary = {
            "enabled": True,
            "status": "ok",
            "method": "C EFORK-3 + Benettin/Gram-Schmidt operational extension",
            "lyapunov_exponents": le,
            "final_state_reported": final_reported,
            "config": {
                "x0": [float(v) for v in final_state],
                "q": validate_fractional_order(cfg["frac_order"]),
                "h": float(le_cfg["h"]),
                "Lm": float(le_cfg["Lm"]),
                "t_burn": float(le_cfg["t_burn"]),
                "n_blocks": int(le_cfg["n_blocks"]),
                "t_block": float(le_cfg["t_block"]),
            },
            "outputs": {
                "csv": str(cfg["outputs"]["lyapunov_csv"]),
                "pdf": str(cfg["outputs"]["lyapunov_pdf"]),
            },
            "raw_stdout": proc.stdout,
        }
        Path(cfg["outputs"]["lyapunov_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    except Exception as exc:
        summary = {
            "enabled": True,
            "status": "failed",
            "error": str(exc),
            "outputs": {
                "csv": str(cfg["outputs"]["lyapunov_csv"]),
                "pdf": str(cfg["outputs"]["lyapunov_pdf"]),
            },
        }
        Path(cfg["outputs"]["lyapunov_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if bool(le_cfg.get("strict", False)):
            raise
        return summary


def run_spectral_analysis(cfg: Dict[str, Any], nyq_data: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("spectral", {}).get("enabled", True)):
        return {"enabled": False, "status": "skipped", "extra_figures": {}}

    log("Etapa espectral: FFT directa; PSD de Welch opcional.")
    h = float(cfg["continuation"]["h"])
    omega0 = float(nyq_data["omega0"])
    spectral_data = extract_spectral_series(nyq_data["results"])
    fft_data = compute_fft_summary(spectral_data, h=h, cfg=cfg, omega0=omega0)
    fft_figures = plot_clean_fft(cfg, fft_data, omega0=omega0)
    psd_data = None
    psd_summary: Dict[str, Any] = {"enabled": False, "status": "skipped"}
    extra_figures = dict(fft_figures)
    if bool(cfg.get("psd", {}).get("enabled", False)):
        psd_data = compute_psd_summary(nyq_data["results"], h=h, cfg=cfg)
        plot_clean_psd(cfg, psd_data, omega0=omega0)
        omega_psd = float(psd_data["omega_d"])
        eta_psd = abs(omega_psd - omega0) / abs(omega0) if abs(omega0) > 0 else float("nan")
        psd_summary = {
            "enabled": True,
            "status": "ok",
            "omega_d_primary": omega_psd,
            "f_d_primary_hz": float(psd_data["f_d"]),
            "relative_mismatch_eta": float(eta_psd),
            "primary_component": psd_data["primary_component"],
            "summary_by_component": psd_data["summary_by_component"],
        }
        extra_figures["psd_pdf"] = str(cfg["outputs"]["psd_pdf"])
    tisean_summary = export_time_series_for_tisean(cfg, spectral_data)
    lyapunov_summary = run_lyapunov_estimate(cfg, np.asarray(nyq_data["final_state"], dtype=float))

    omega_fft = float(fft_data["omega_d"])
    eta_fft = abs(omega_fft - omega0) / abs(omega0) if abs(omega0) > 0 else float("nan")
    summary = {
        "enabled": True,
        "status": "ok",
        "omega0_nyquist_df": omega0,
        "psd_summary": psd_summary,
        "fft_summary": {
            "omega_d_primary": omega_fft,
            "f_d_primary_hz": float(fft_data["f_d"]),
            "relative_mismatch_eta": float(eta_fft),
            "primary_component": fft_data["primary_component"],
            "summary_by_component": fft_data["summary_by_component"],
            "interpretation": (
                "omega_d_primary es el pico global de amplitud FFT. peak_near_omega0 reporta el pico local "
                "dentro de una ventana alrededor de la frecuencia Nyquist/DF; si ambos difieren, la trayectoria "
                "observada no esta dominada por la frecuencia armonica predicha."
            ),
        },
        "tisean_summary": tisean_summary,
        "lyapunov_summary": lyapunov_summary,
        "extra_figures": extra_figures,
        "outputs": {
            "psd_json": str(cfg["outputs"]["psd_json"]),
        },
        "notes": [
            "La FFT directa se usa como descriptor espectral principal y se centra alrededor de frecuencias dominantes.",
            "La PSD de Welch promedia FFTs por ventanas; suaviza la potencia espectral, pero no es una prueba de caos ni reemplaza la validacion de Caputo.",
            "TISEAN y Lyapunov operacional quedan desactivados por default y se activan manualmente con variables de entorno.",
        ],
    }
    if bool(cfg.get("lyapunov", {}).get("enabled", False)) and lyapunov_summary.get("status") == "ok":
        summary["extra_figures"]["lyapunov_pdf"] = str(cfg["outputs"]["lyapunov_pdf"])
    Path(cfg["outputs"]["psd_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

def _pdf_figure_map(cfg: Dict[str, Any], keys: Iterable[str]) -> Dict[str, str]:
    wanted = set(keys)
    return {
        k: str(v)
        for k, v in cfg["outputs"].items()
        if k in wanted and str(v).endswith(".pdf")
    }


def run_article_phase_figures(cfg: Dict[str, Any], nyq_data: Dict[str, Any]) -> Dict[str, Any]:
    ensure_current_chua_params(cfg)
    log("Figuras de fase: transferencia, Matignon, comparacion Nyquist/DF-original, continuacion y retratos finales.")
    plot_transfer_re_im(cfg, nyq_data)
    matignon_summary = plot_matignon_equilibrium_spectrum(cfg)
    plot_linearized_vs_original(cfg, nyq_data)
    plot_continuation_story(cfg, nyq_data["results"])
    plot_final_attractor_with_planes(cfg, nyq_data["results"])
    extra_keys = [
        "transfer_reim_pdf",
        "matignon_equilibria_pdf",
        "linear_vs_original_pdf",
        "continuation_story_pdf",
        "final_attr_xy_pdf",
        "final_attr_xz_pdf",
        "final_attr_yz_pdf",
    ]
    return {
        "status": "ok",
        "extra_figures": _pdf_figure_map(cfg, extra_keys),
        "matignon_equilibria": matignon_summary,
        "notes": [
            "Transferencia separada en Re/Im para mostrar la condicion Im(W)=0 y Re(W)=-1/k.",
            "Autovalores de la linealizacion local en equilibrios con sectores de Matignon.",
            "Comparacion de la aproximacion armonica Nyquist/DF contra el atractor original.",
            "Morfologia de la continuacion con trayectorias superpuestas.",
            "Retratos de fase xy/xz/yz del atractor final, guardados como figuras individuales.",
        ],
    }


def run_basin_plane_figures(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    seed_pos = np.asarray(final_state, dtype=float)
    basin_generated = False
    basin_backend = "none"
    extra_keys: List[str] = []
    if bool(cfg["article_style"].get("basin_planes_enabled", True)):
        basin_data = compute_basin_planes_c(cfg, seed_pos)
        plot_basin_planes(basin_data, cfg["outputs"]["basin_planes_pdf"])
        plot_single_basin_plane(basin_data["xy"], "xy", cfg["outputs"]["basin_xy_pdf"])
        plot_single_basin_plane(basin_data["xz"], "xz", cfg["outputs"]["basin_xz_pdf"])
        plot_single_basin_plane(basin_data["yz"], "yz", cfg["outputs"]["basin_yz_pdf"])
        basin_generated = True
        basin_backend = "c"
    else:
        log("Cuenca de 3 planos omitida. Activa --basin-planes si necesitas fig10.")
    if basin_generated:
        extra_keys = ["basin_planes_pdf", "basin_xy_pdf", "basin_xz_pdf", "basin_yz_pdf"]
    return {
        "status": "ok" if basin_generated else "skipped",
        "basin_planes_enabled": bool(cfg["article_style"].get("basin_planes_enabled", True)),
        "basin_planes_backend": basin_backend,
        "extra_figures": _pdf_figure_map(cfg, extra_keys),
        "notes": ["Cortes de cuenca en planos xy/xz/yz, combinados y separados."] if basin_generated else [],
    }


def run_bifurcation_figures(cfg: Dict[str, Any], final_state: np.ndarray) -> Dict[str, Any]:
    ensure_current_chua_params(cfg)
    seed_pos = np.asarray(final_state, dtype=float)
    seed_neg = symmetric_seed(seed_pos)
    started_all = time.perf_counter()
    log("Bifurcaciones: iniciando barrido en q.")
    q_data = compute_bifurcation_sweep("q", cfg["bifurcation"]["q_values"], seed_pos, seed_neg, cfg)
    plot_article_style_bifurcation(q_data, xlabel="q", ylabel=r"$x_{\max}$", path=cfg["outputs"]["bif_q_pdf"])
    log("Bifurcaciones: q terminado; iniciando barrido en alpha.")
    alpha_data = compute_bifurcation_sweep("alpha", cfg["bifurcation"]["alpha_values"], seed_pos, seed_neg, cfg)
    plot_article_style_bifurcation(alpha_data, xlabel=r"$\alpha$", ylabel=r"$x_{\max}$", path=cfg["outputs"]["bif_alpha_pdf"])
    log("Bifurcaciones: alpha terminado; iniciando barrido en beta.")
    beta_data = compute_bifurcation_sweep("beta", cfg["bifurcation"]["beta_values"], seed_pos, seed_neg, cfg)
    plot_article_style_bifurcation(beta_data, xlabel=r"$\beta$", ylabel=r"$x_{\max}$", path=cfg["outputs"]["bif_beta_pdf"])
    log(f"Bifurcaciones: terminadas en {format_elapsed(time.perf_counter() - started_all)}.")
    q_failures = q_data.get("failures", [])
    alpha_failures = alpha_data.get("failures", [])
    beta_failures = beta_data.get("failures", [])
    if q_failures or alpha_failures or beta_failures:
        log(
            "Bifurcacion: integraciones fallidas/divergentes omitidas "
            f"(q={len(q_failures)}, alpha={len(alpha_failures)}, beta={len(beta_failures)})."
        )
    extra_keys = ["bif_q_pdf", "bif_alpha_pdf", "bif_beta_pdf"]
    return {
        "status": "ok",
        "q_points": int(len(q_data["pos_x"]) + len(q_data["neg_x"])),
        "alpha_points": int(len(alpha_data["pos_x"]) + len(alpha_data["neg_x"])),
        "beta_points": int(len(beta_data["pos_x"]) + len(beta_data["neg_x"])),
        "q_failures": q_failures,
        "alpha_failures": alpha_failures,
        "beta_failures": beta_failures,
        "backend": {
            "q": q_data.get("backend", "python"),
            "alpha": alpha_data.get("backend", "python"),
            "beta": beta_data.get("backend", "python"),
        },
        "extra_figures": _pdf_figure_map(cfg, extra_keys),
        "notes": ["Diagramas de bifurcacion tipo x_max para q, alpha y beta."],
    }


def merge_article_style_summaries(
    cfg: Dict[str, Any],
    phase_summary: Dict[str, Any] | None = None,
    basin_planes_summary: Dict[str, Any] | None = None,
    bifurcation_summary: Dict[str, Any] | None = None,
    *,
    write: bool = True,
) -> Dict[str, Any]:
    phase_summary = phase_summary or {"status": "skipped", "extra_figures": {}}
    basin_planes_summary = basin_planes_summary or {"status": "skipped", "extra_figures": {}}
    bifurcation_summary = bifurcation_summary or {"status": "skipped", "extra_figures": {}}
    extra_figures: Dict[str, str] = {}
    for part in (phase_summary, basin_planes_summary, bifurcation_summary):
        extra_figures.update(part.get("extra_figures", {}))
    summary = {
        "enabled": bool(cfg.get("article_style", {}).get("enabled", True)),
        "extra_figures": extra_figures,
        "phase_figures": phase_summary,
        "basin_planes": basin_planes_summary,
        "bifurcation": bifurcation_summary,
        "generation_order": [
            "phase_figures",
            "spectral_figures",
            "hidden_verification_figures",
            "basin_figures",
            "bifurcation_figures",
        ],
        "notes": [
            "Las figuras ligeras se generan junto a su etapa numerica.",
            "Las cuencas se calculan despues de la verificacion oculta.",
            "Los diagramas de bifurcacion se calculan al final del pipeline.",
        ],
    }
    if write:
        with open(cfg["outputs"]["article_style_summary_json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def run_article_style_figures(cfg: Dict[str, Any], nyq_data: Dict[str, Any]) -> Dict[str, Any]:
    phase_summary = run_article_phase_figures(cfg, nyq_data)
    basin_planes_summary = run_basin_plane_figures(cfg, np.asarray(nyq_data["final_state"], dtype=float))
    bifurcation_summary = run_bifurcation_figures(cfg, np.asarray(nyq_data["final_state"], dtype=float))
    return merge_article_style_summaries(cfg, phase_summary, basin_planes_summary, bifurcation_summary)

    ensure_current_chua_params(cfg)
    log("Figuras estilo articulo: Nyquist zoom, comparacion lineal/original y retratos de fase.")
    plot_nyquist_with_x_zoom(cfg, nyq_data)
    plot_linearized_vs_original(cfg, nyq_data)
    plot_continuation_story(cfg, nyq_data["results"])
    plot_final_attractor_with_planes(cfg, nyq_data["results"])
    seed_pos = np.asarray(nyq_data["final_state"], dtype=float)
    seed_neg = symmetric_seed(seed_pos)
    q_data = compute_bifurcation_sweep("q", cfg["bifurcation"]["q_values"], seed_pos, seed_neg, cfg)
    plot_article_style_bifurcation(q_data, xlabel="q", ylabel=r"$x_{\max}$", path=cfg["outputs"]["bif_q_pdf"])
    alpha_data = compute_bifurcation_sweep("alpha", cfg["bifurcation"]["alpha_values"], seed_pos, seed_neg, cfg)
    plot_article_style_bifurcation(alpha_data, xlabel=r"$\alpha$", ylabel=r"$x_{\max}$", path=cfg["outputs"]["bif_alpha_pdf"])
    beta_data = compute_bifurcation_sweep("beta", cfg["bifurcation"]["beta_values"], seed_pos, seed_neg, cfg)
    plot_article_style_bifurcation(beta_data, xlabel=r"$\beta$", ylabel=r"$x_{\max}$", path=cfg["outputs"]["bif_beta_pdf"])
    basin_generated = False
    basin_backend = "none"
    if bool(cfg["article_style"].get("basin_planes_enabled", True)):
        basin_data = compute_basin_planes_c(cfg, seed_pos)
        plot_basin_planes(basin_data, cfg["outputs"]["basin_planes_pdf"])
        plot_single_basin_plane(basin_data["xy"], "xy", cfg["outputs"]["basin_xy_pdf"])
        plot_single_basin_plane(basin_data["xz"], "xz", cfg["outputs"]["basin_xz_pdf"])
        plot_single_basin_plane(basin_data["yz"], "yz", cfg["outputs"]["basin_yz_pdf"])
        basin_generated = True
        basin_backend = "c"
    else:
        log("Cuenca de 3 planos omitida. Activa --basin-planes si necesitas fig10.")
    extra_keys = [
        "nyquist_zoom_pdf",
        "linear_vs_original_pdf",
        "continuation_story_pdf",
        "final_attr_planes_pdf",
        "bif_q_pdf",
        "bif_alpha_pdf",
        "bif_beta_pdf",
    ]
    if basin_generated:
        extra_keys.append("basin_planes_pdf")
        extra_keys.extend(["basin_xy_pdf", "basin_xz_pdf", "basin_yz_pdf"])
    summary = {
        "extra_figures": {
            k: str(v)
            for k, v in cfg["outputs"].items()
            if str(v).endswith(".pdf") and k in set(extra_keys)
        },
        "basin_planes_enabled": bool(cfg["article_style"].get("basin_planes_enabled", True)),
        "basin_planes_backend": basin_backend,
        "notes": [
            "Nyquist adicional con acercamiento en eje x.",
            "Comparación linealizado (eps=0) vs original.",
            "Morfología de la continuación con trayectorias superpuestas.",
            "Retratos de fase 3D + xy/xz/yz del atractor final.",
            "Diagramas de bifurcación tipo x_max para q, alpha y beta, con panel de acercamiento.",
            "Cortes de cuenca en planos xy/xz/yz, combinados y separados.",
        ],
    }
    with open(cfg["outputs"]["article_style_summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def q_sweep_tag(qord: float) -> str:
    return f"{validate_fractional_order(qord):.5f}".replace(".", "p")


def trajectory_shape_diagnostics(traj: np.ndarray) -> Dict[str, float]:
    X = np.asarray(traj[:, 1:4], dtype=float)
    if X.shape[0] < 4:
        return {
            "range_x": float("nan"),
            "range_y": float("nan"),
            "range_z": float("nan"),
            "cov_ratio_mid_to_max": float("nan"),
            "cov_ratio_min_to_max": float("nan"),
        }
    centered = X - np.mean(X, axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
    denom = eigvals[0] if eigvals[0] > 0.0 else float("nan")
    ranges = np.ptp(X, axis=0)
    return {
        "range_x": float(ranges[0]),
        "range_y": float(ranges[1]),
        "range_z": float(ranges[2]),
        "cov_lambda1": float(eigvals[0]),
        "cov_lambda2": float(eigvals[1]),
        "cov_lambda3": float(eigvals[2]),
        "cov_ratio_mid_to_max": float(eigvals[1] / denom) if np.isfinite(denom) else float("nan"),
        "cov_ratio_min_to_max": float(eigvals[2] / denom) if np.isfinite(denom) else float("nan"),
    }


def save_nyquist_df_samples(cfg: Dict[str, Any], path: Path) -> None:
    p = chua_ic.PARAMS
    qord = chua_ic.QORD
    path.parent.mkdir(parents=True, exist_ok=True)
    omg = np.logspace(np.log10(chua_ic.WMIN), np.log10(chua_ic.WMAX), 4000)
    W = np.array([chua_ic.W_frac(w, qord, p) for w in omg], dtype=np.complex128)
    aa = np.linspace(1.0 + 1e-6, cfg["continuation"].get("amax", 100.0), 2500)
    minus_invN = -1.0 / np.array([chua_ic.N_sat(a, p) for a in aa], dtype=float)
    nrows = max(len(omg), len(aa))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["omega", "Re_W", "Im_W", "amplitude", "minus_inv_N"])
        for i in range(nrows):
            writer.writerow([
                f"{omg[i]:.16e}" if i < len(omg) else "",
                f"{W.real[i]:.16e}" if i < len(W) else "",
                f"{W.imag[i]:.16e}" if i < len(W) else "",
                f"{aa[i]:.16e}" if i < len(aa) else "",
                f"{minus_invN[i]:.16e}" if i < len(minus_invN) else "",
            ])


def save_q_sweep_trajectory(q_dir: Path, tag: str, traj: np.ndarray, write_csv: bool) -> Dict[str, str]:
    q_dir.mkdir(parents=True, exist_ok=True)
    npz_path = q_dir / f"final_attractor_q_{tag}.npz"
    np.savez_compressed(npz_path, t=traj[:, 0], x=traj[:, 1], y=traj[:, 2], z=traj[:, 3])
    outputs = {"final_attractor_npz": str(npz_path)}
    if write_csv:
        csv_path = q_dir / f"final_attractor_q_{tag}.csv"
        np.savetxt(csv_path, traj, delimiter=",", header="t,x,y,z", comments="", fmt="%.16e")
        outputs["final_attractor_csv"] = str(csv_path)
    return outputs


def q_sweep_outputs(base_cfg: Dict[str, Any], q_dir: Path, tag: str) -> Dict[str, Path]:
    outputs = dict(base_cfg["outputs"])
    outputs.update({
        "nyquist_pdf": q_dir / f"fig01_nyquist_df_q_{tag}.pdf",
        "cont_progress_pdf": q_dir / f"fig02_continuation_progress_q_{tag}.pdf",
        "final_attr_pdf": q_dir / f"fig03_final_attractor_q_{tag}.pdf",
        "cont_json": q_dir / f"continuation_summary_q_{tag}.json",
        "summary_json": q_dir / f"q_summary_{tag}.json",
    })
    return outputs


def df_compare_slug(method: str, mu: float | None = None) -> str:
    method = str(method).strip().lower()
    if method == "classic":
        return "classic"
    tag = f"{float(mu):.5f}".replace(".", "p")
    return f"machado_mu_{tag}"


def df_compare_outputs(base_cfg: Dict[str, Any], candidate_dir: Path, slug: str) -> Dict[str, Path]:
    outputs = dict(base_cfg["outputs"])
    outputs.update({
        "nyquist_pdf": candidate_dir / f"fig01_nyquist_df_{slug}.pdf",
        "cont_progress_pdf": candidate_dir / f"fig02_continuation_progress_{slug}.pdf",
        "final_attr_pdf": candidate_dir / f"fig03_final_attractor_{slug}.pdf",
        "ref_section_pdf": candidate_dir / f"fig04_reference_section_{slug}.pdf",
        "probe_summary_pdf": candidate_dir / f"fig05_probe_summary_{slug}.pdf",
        "cont_json": candidate_dir / f"continuation_summary_{slug}.json",
        "summary_json": candidate_dir / f"df_candidate_summary_{slug}.json",
    })
    return outputs


def save_df_candidate_trajectory(candidate_dir: Path, slug: str, traj: np.ndarray) -> Dict[str, str]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    npz_path = candidate_dir / f"final_attractor_{slug}.npz"
    csv_path = candidate_dir / f"final_attractor_{slug}.csv"
    np.savez_compressed(npz_path, t=traj[:, 0], x=traj[:, 1], y=traj[:, 2], z=traj[:, 3])
    np.savetxt(csv_path, traj, delimiter=",", header="t,x,y,z", comments="", fmt="%.16e")
    return {"final_attractor_npz": str(npz_path), "final_attractor_csv": str(csv_path)}


def save_df_candidate_samples(cfg: Dict[str, Any], path: Path, method: str, mu: float | None) -> None:
    p = chua_ic.PARAMS
    qord = chua_ic.QORD
    path.parent.mkdir(parents=True, exist_ok=True)
    omg = np.logspace(np.log10(chua_ic.WMIN), np.log10(chua_ic.WMAX), 4000)
    W = np.array([chua_ic.W_frac(w, qord, p) for w in omg], dtype=np.complex128)
    aa = np.linspace(1.0 + 1e-6, cfg["continuation"].get("amax", 100.0), 2500)
    classic_N = np.array([chua_ic.N_sat(a, p) for a in aa], dtype=float)
    if method == "machado":
        effective_N = np.array([chua_ic.N_sat_machado(a, p, float(mu)) for a in aa], dtype=float)
    else:
        effective_N = classic_N
    nrows = max(len(omg), len(aa))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["omega", "Re_W", "Im_W", "amplitude", "N_classic", "N_effective"])
        for i in range(nrows):
            writer.writerow([
                f"{omg[i]:.16e}" if i < len(omg) else "",
                f"{W.real[i]:.16e}" if i < len(W) else "",
                f"{W.imag[i]:.16e}" if i < len(W) else "",
                f"{aa[i]:.16e}" if i < len(aa) else "",
                f"{classic_N[i]:.16e}" if i < len(classic_N) else "",
                f"{effective_N[i]:.16e}" if i < len(effective_N) else "",
            ])


def phase_tag(theta: float) -> str:
    return f"{float(theta):.5f}".replace("-", "m").replace(".", "p")


def mu_tag(mu: float) -> str:
    return f"{float(mu):.5f}".replace("-", "m").replace(".", "p")


def continue_seed_candidate(cfg: Dict[str, Any], qord: float, k0: float, xseed: np.ndarray) -> List[Dict[str, Any]]:
    chua_ic.EPS_VALUES = np.array(cfg["continuation"]["eps_values"], dtype=np.float64)
    if bool(cfg.get("native_efork", {}).get("enabled", True)):
        return continuation_in_epsilon_c(cfg, qord, k0, xseed, chua_ic.EPS_VALUES)
    return chua_ic.continuation_in_epsilon(
        p=chua_ic.PARAMS,
        qord=qord,
        k=k0,
        x0_seed=xseed,
        eps_values=chua_ic.EPS_VALUES,
        h=float(cfg["continuation"]["h"]),
        Lm=float(cfg["continuation"]["Lm"]),
        t_transient=float(cfg["continuation"]["t_transient"]),
        t_keep=float(cfg["continuation"]["t_keep"]),
        memory_mode=cfg["continuation"].get("memory_mode", "window"),
        memory_update_source=cfg["continuation"].get("memory_update_source", "observed"),
    )


def continuation_summary_from_results(
    cfg: Dict[str, Any],
    qord: float,
    branch_index: int,
    omega0: float,
    k0: float,
    a0: float,
    mu: float,
    theta: float,
    xseed: np.ndarray,
    v: np.ndarray,
    eig_match: complex,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    final_state = np.asarray(results[-1]["x_out"], dtype=float)
    W0 = chua_ic.W_frac(omega0, qord, chua_ic.PARAMS)
    N_classic_a0 = float(chua_ic.N_sat(a0, chua_ic.PARAMS))
    N_effective_a0 = float(chua_ic.N_sat_machado(a0, chua_ic.PARAMS, mu))
    return {
        "model": cfg["model"],
        "params": cfg["params"],
        "frac_order": float(qord),
        "runtime_contract": cfg.get("runtime_contract", {}),
        "describing_function": {
            "method": "machado",
            "machado_mu": float(mu),
            "theta": float(theta),
            "source": "Machado fractional describing function N_mu=N^mu with phase sweep",
        },
        "chosen_branch": {
            "describing_function": "machado",
            "machado_mu": float(mu),
            "theta": float(theta),
            "branch_index": int(branch_index),
            "omega0": float(omega0),
            "k": float(k0),
            "a0": float(a0),
            "seed": np.asarray(xseed, dtype=float).tolist(),
            "eig_match": [float(np.real(eig_match)), float(np.imag(eig_match))],
            "eigvec_real": np.real(v).astype(float).tolist(),
            "eigvec_imag": np.imag(v).astype(float).tolist(),
            "nyquist_df_residuals": {
                "W0_real": float(np.real(W0)),
                "W0_imag": float(np.imag(W0)),
                "k_from_W": float(-1.0 / np.real(W0)),
                "N_classic_a0": N_classic_a0,
                "N_effective_a0": N_effective_a0,
                "closure_N_minus_k": float(N_effective_a0 - k0),
            },
        },
        "continuation": {
            "backend": "c_efork3" if bool(cfg.get("native_efork", {}).get("enabled", True)) else "python",
            "eps_values": [float(x) for x in cfg["continuation"]["eps_values"]],
            "memory_mode": str(cfg["continuation"].get("memory_mode", "window")),
            "memory_update_source": str(cfg["continuation"].get("memory_update_source", "observed")),
            "h": float(cfg["continuation"]["h"]),
            "Lm": float(cfg["continuation"]["Lm"]),
            "t_transient": float(cfg["continuation"]["t_transient"]),
            "t_keep": float(cfg["continuation"]["t_keep"]),
            "final_state_eps1": final_state.tolist(),
            "states_by_step": [
                {
                    "eps": float(r["eps"]),
                    "x_in": np.asarray(r["x_in"], dtype=float).tolist(),
                    "x_transient_out": np.asarray(r.get("x_transient_out", r["x_out"]), dtype=float).tolist(),
                    "x_out": np.asarray(r["x_out"], dtype=float).tolist(),
                    "history_points_in": int(r.get("history_points_in", 0)),
                    "history_points_out": int(r.get("history_points_out", 0)),
                }
                for r in results
            ],
        },
    }


def df_candidate_sort_key(record: Dict[str, Any]) -> Tuple[int, int, float, float]:
    status_penalty = 0 if record.get("status") == "ok" else 1
    hits = record.get("total_target_hits")
    hit_score = int(hits) if hits is not None else 10**9
    range_x = float(record.get("range_x", float("nan")))
    if not np.isfinite(range_x):
        range_x = -1.0
    seed_norm = float(record.get("seed_norm", float("nan")))
    if not np.isfinite(seed_norm):
        seed_norm = 10**9
    return (status_penalty, hit_score, -range_x, seed_norm)


def run_describing_function_comparison(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara la DF clasica contra la extension tipo Machado.

    Cada candidato ejecuta: Nyquist/semilla -> continuacion en epsilon ->
    verificacion de ocultedad con el backend C. No calcula cuencas ni
    bifurcaciones para mantener aislada la comparacion de semillas.
    """
    ensure_current_chua_params(cfg)
    if normalize_chua_model(cfg["model"]["kind"]) != "nonsmooth":
        raise ValueError("df_compare/Machado esta implementado para el Chua no suave.")

    root = Path(cfg["df_compare"]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    branch_index = int(cfg["df_compare"].get("branch_index", cfg["branch_index"]))
    methods: List[Tuple[str, float | None]] = [("classic", None)]
    for mu in cfg["df_compare"].get("machado_mu_values", [0.5]):
        methods.append(("machado", float(mu)))

    log(
        "Comparacion DF: "
        f"{len(methods)} candidato(s); branch_index={branch_index}; salida={root}."
    )

    records: List[Dict[str, Any]] = []
    for idx, (method, mu) in enumerate(methods, start=1):
        slug = df_compare_slug(method, mu)
        candidate_dir = root / slug
        candidate_dir.mkdir(parents=True, exist_ok=True)
        log(f"Comparacion DF {idx}/{len(methods)}: {slug}")

        local_cfg = copy.deepcopy(cfg)
        local_cfg["outputs"] = df_compare_outputs(cfg, candidate_dir, slug)
        local_cfg["verify_hidden"]["runtime_dir"] = candidate_dir / "hidden_verify"
        local_cfg["verify_hidden"]["config_path"] = candidate_dir / "hidden_verify" / "config_hidden_verify_frac.json"
        local_cfg["native_dir"] = candidate_dir / "native"
        synchronize_runtime_contract(local_cfg)

        record: Dict[str, Any] = {
            "method": method,
            "machado_mu": float(mu) if mu is not None else None,
            "slug": slug,
            "status": "failed",
            "output_dir": str(candidate_dir),
        }
        try:
            nyq_data = compute_nyquist_seed_and_continuation(
                local_cfg,
                df_method=method,
                machado_mu=float(mu) if mu is not None else 1.0,
                branch_index_override=branch_index,
            )
            plot_clean_nyquist(local_cfg, nyq_data)
            plot_clean_continuation_progress(local_cfg, nyq_data["results"])
            plot_clean_final_attractor(local_cfg, nyq_data["results"])

            samples_csv = candidate_dir / f"nyquist_df_samples_{slug}.csv"
            save_df_candidate_samples(local_cfg, samples_csv, method, mu)
            traj = np.asarray(nyq_data["results"][-1]["traj"], dtype=float)
            traj_outputs = save_df_candidate_trajectory(candidate_dir, slug, traj)
            diagnostics = trajectory_shape_diagnostics(traj)

            hidden_summary = run_hidden_verify_with_seed(local_cfg, nyq_data["final_state"])
            plot_clean_reference_section(local_cfg, hidden_summary["files"]["ref_csv_out"])
            eq_names = list(hidden_summary["equilibria"].keys())
            plot_clean_probe_summary(local_cfg, hidden_summary["files"]["summary_csv_out"], eq_names)

            record.update({
                "status": "ok",
                "omega0": float(nyq_data["omega0"]),
                "k0": float(nyq_data["k0"]),
                "a0": float(nyq_data["a0"]),
                "seed": np.asarray(nyq_data["xseed"], dtype=float).tolist(),
                "seed_norm": float(np.linalg.norm(nyq_data["xseed"])),
                "final_state_eps1": np.asarray(nyq_data["final_state"], dtype=float).tolist(),
                "total_target_hits": int(hidden_summary.get("total_target_hits", 0)),
                "hiddenness_status": hidden_summary.get("hiddenness_status"),
                "reference_points": int(hidden_summary.get("reference_points", 0)),
                "continuation_summary_json": str(local_cfg["outputs"]["cont_json"]),
                "candidate_summary_json": str(local_cfg["outputs"]["summary_json"]),
                "nyquist_df_csv": str(samples_csv),
                "hidden_summary_json": str(hidden_summary["files"]["json_out"]),
                "outputs": {
                    "nyquist_pdf": str(local_cfg["outputs"]["nyquist_pdf"]),
                    "continuation_pdf": str(local_cfg["outputs"]["cont_progress_pdf"]),
                    "final_attractor_pdf": str(local_cfg["outputs"]["final_attr_pdf"]),
                    "reference_section_pdf": str(local_cfg["outputs"]["ref_section_pdf"]),
                    "probe_summary_pdf": str(local_cfg["outputs"]["probe_summary_pdf"]),
                    **traj_outputs,
                },
                "chosen_branch": nyq_data["summary"]["chosen_branch"],
                "shape_diagnostics": diagnostics,
                **diagnostics,
            })
            candidate_summary = {
                "model": local_cfg["model"],
                "params": local_cfg["params"],
                "frac_order": local_cfg["frac_order"],
                "runtime_contract": local_cfg.get("runtime_contract", {}),
                "candidate": record,
                "nyquist_and_continuation": nyq_data["summary"],
                "hidden_verification": hidden_summary,
                "notes": [
                    "Candidato evaluado dentro de run_mode=df_compare.",
                    "La extension de Machado usa N_mu(a)=N(a)^mu; mu modifica la amplitud, no la dinamica de Caputo.",
                    "La prueba de ocultedad se basa en muestreo numerico alrededor de equilibrios.",
                ],
            }
            Path(local_cfg["outputs"]["summary_json"]).write_text(
                json.dumps(candidate_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            record["error"] = str(exc)
            Path(local_cfg["outputs"]["summary_json"]).write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if not bool(cfg["df_compare"].get("continue_on_error", True)):
                raise
            log(f"Comparacion DF {slug}: fallo controlado: {exc}")
        records.append(record)

    best = min(records, key=df_candidate_sort_key) if records else None
    seed_distances: List[Dict[str, Any]] = []
    ok_records = [r for r in records if r.get("status") == "ok" and isinstance(r.get("seed"), list)]
    for i in range(len(ok_records)):
        for j in range(i + 1, len(ok_records)):
            si = np.asarray(ok_records[i]["seed"], dtype=float)
            sj = np.asarray(ok_records[j]["seed"], dtype=float)
            seed_distances.append({
                "left": ok_records[i]["slug"],
                "right": ok_records[j]["slug"],
                "euclidean_distance": float(np.linalg.norm(si - sj)),
            })

    csv_path = root / "df_seed_comparison_summary.csv"
    fields = [
        "slug", "method", "machado_mu", "status", "omega0", "k0", "a0",
        "seed_norm", "total_target_hits", "hiddenness_status",
        "range_x", "range_y", "range_z", "output_dir", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    summary = {
        "model": cfg["model"],
        "params": cfg["params"],
        "frac_order": cfg["frac_order"],
        "run_mode": "df_compare",
        "machado_mu_values": [float(v) for v in cfg["df_compare"].get("machado_mu_values", [])],
        "branch_index": branch_index,
        "output_layout": output_layout_dict(cfg),
        "output_root": str(root),
        "summary_csv": str(csv_path),
        "records": records,
        "seed_distances": seed_distances,
        "best_candidate": best,
        "selection_rule": "menor total_target_hits; empate por mayor rango_x y menor norma de semilla",
        "notes": [
            "La DF clasica es el caso mu=1.",
            "La alternativa tipo Machado usa la formulacion N_mu(a)=N(a)^mu descrita en docs/reporte_unificado_chua_fraccionario.tex y en Tenreiro Machado.",
            "Si best_candidate tiene total_target_hits > 0, el candidato no queda soportado como oculto bajo esta muestra.",
        ],
    }
    summary_path = root / "df_seed_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Comparacion DF completada.", flush=True)
    print("Directorio:", root, flush=True)
    print("Resumen CSV:", csv_path, flush=True)
    print("Resumen JSON:", summary_path, flush=True)
    if best:
        print(
            "Mejor candidato:",
            best.get("slug"),
            "TARGET=",
            best.get("total_target_hits"),
            "status=",
            best.get("hiddenness_status"),
            flush=True,
        )
    return summary


def run_machado_sweep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Barrido de semillas Machado:
        X_seed(mu,j,theta) = a_mu,j Re(v_j exp(i theta)).

    Descarta automaticamente pares (j, mu) que no cumplen 0 < k_j < Delta^mu.
    Para cada semilla admisible ejecuta continuacion y prueba de ocultedad.
    """
    ensure_current_chua_params(cfg)
    if normalize_chua_model(cfg["model"]["kind"]) != "nonsmooth":
        raise ValueError("machado_sweep esta implementado exclusivamente para el Chua no suave.")

    run_mode = str(cfg.get("run_mode", "machado_sweep_fast"))
    sweep_key = "machado_sweep" if run_mode == "machado_sweep" else "machado_sweep_fast"
    fallback_grid = MACHADO_SWEEP if sweep_key == "machado_sweep" else MACHADO_SWEEP_FAST
    root = Path(cfg[sweep_key]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    grid = cfg[sweep_key].get("grid", fallback_grid)
    max_candidates = cfg[sweep_key].get("max_candidates")
    max_candidates = None if max_candidates is None else int(max_candidates)

    qord = validate_fractional_order(cfg["frac_order"])
    chua_ic.PARAMS = chua_ic_params_from_config(cfg)
    chua_ic.QORD = np.float64(qord)
    raw_pairs = chua_ic.find_omega_k_candidates(
        chua_ic.QORD,
        chua_ic.PARAMS,
        wmin=chua_ic.WMIN,
        wmax=chua_ic.WMAX,
        nscan=chua_ic.NSCAN,
    )
    delta = float(chua_ic.chua_gain_A(chua_ic.PARAMS))

    total_theoretical = sum(len(v.get("mu_values", [])) * len(v.get("theta_values", [])) for v in grid.values())
    log(
        f"{sweep_key}: "
        f"ramas={list(grid.keys())}; candidatos teoricos={total_theoretical}; "
        f"pares Nyquist crudos={len(raw_pairs)}; salida={root}."
    )

    records: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    executed = 0

    for branch_key in sorted(grid, key=lambda x: int(x)):
        branch_index = int(branch_key)
        branch_cfg = grid[branch_key]
        if branch_index < 0 or branch_index >= len(raw_pairs):
            discarded.append({
                "branch_index": branch_index,
                "reason": "branch_not_available",
                "available_pairs": len(raw_pairs),
            })
            continue

        omega0, k0 = raw_pairs[branch_index][:2]
        for mu in [float(v) for v in branch_cfg.get("mu_values", [])]:
            admissible_bound = float(delta ** mu)
            if not chua_ic.is_machado_gain_compatible(k0, chua_ic.PARAMS, mu):
                discarded.append({
                    "branch_index": branch_index,
                    "omega0": float(omega0),
                    "k0": float(k0),
                    "mu": float(mu),
                    "delta_power_mu": admissible_bound,
                    "reason": "inadmissible_gain",
                    "condition": "0 < k_j < Delta^mu",
                })
                continue

            a0 = chua_ic.solve_machado_amplitude_from_k(k0, chua_ic.PARAMS, mu)
            for theta in [float(v) for v in branch_cfg.get("theta_values", [])]:
                if max_candidates is not None and executed >= max_candidates:
                    log(f"{sweep_key}: limite de {max_candidates} candidatos alcanzado.")
                    break

                slug = f"branch_{branch_index}_mu_{mu_tag(mu)}_theta_{phase_tag(theta)}"
                candidate_dir = root / f"branch_{branch_index}" / f"mu_{mu_tag(mu)}" / f"theta_{phase_tag(theta)}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                log(f"Machado sweep {executed + 1}: {slug}")

                local_cfg = copy.deepcopy(cfg)
                local_cfg["outputs"] = df_compare_outputs(cfg, candidate_dir, slug)
                local_cfg["verify_hidden"]["runtime_dir"] = candidate_dir / "hidden_verify"
                local_cfg["verify_hidden"]["config_path"] = candidate_dir / "hidden_verify" / "config_hidden_verify_frac.json"
                local_cfg["native_dir"] = candidate_dir / "native"
                synchronize_runtime_contract(local_cfg)
                chua_ic.PARAMS = chua_ic_params_from_config(local_cfg)
                chua_ic.QORD = np.float64(qord)

                record: Dict[str, Any] = {
                    "slug": slug,
                    "method": "machado",
                    "branch_index": branch_index,
                    "mu": float(mu),
                    "theta": float(theta),
                    "omega0": float(omega0),
                    "k0": float(k0),
                    "delta_power_mu": admissible_bound,
                    "status": "failed",
                    "output_dir": str(candidate_dir),
                }
                try:
                    xseed, v, eig_match = chua_ic.build_fractional_seed(
                        qord,
                        chua_ic.PARAMS,
                        omega0,
                        k0,
                        a0,
                        theta=theta,
                    )
                    results = continue_seed_candidate(local_cfg, qord, k0, xseed)
                    final_state = np.asarray(results[-1]["x_out"], dtype=float)
                    traj = np.asarray(results[-1]["traj"], dtype=float)
                    traj_outputs = save_df_candidate_trajectory(candidate_dir, slug, traj)
                    diagnostics = trajectory_shape_diagnostics(traj)
                    cont_summary = continuation_summary_from_results(
                        local_cfg,
                        qord,
                        branch_index,
                        omega0,
                        k0,
                        a0,
                        mu,
                        theta,
                        xseed,
                        v,
                        eig_match,
                        results,
                    )
                    Path(local_cfg["outputs"]["cont_json"]).write_text(
                        json.dumps(cont_summary, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    hidden_summary = run_hidden_verify_with_seed(local_cfg, final_state)
                    record.update({
                        "status": "ok",
                        "a0": float(a0),
                        "seed": np.asarray(xseed, dtype=float).tolist(),
                        "seed_norm": float(np.linalg.norm(xseed)),
                        "final_state_eps1": final_state.tolist(),
                        "total_target_hits": int(hidden_summary.get("total_target_hits", 0)),
                        "hiddenness_status": hidden_summary.get("hiddenness_status"),
                        "reference_points": int(hidden_summary.get("reference_points", 0)),
                        "continuation_summary_json": str(local_cfg["outputs"]["cont_json"]),
                        "candidate_summary_json": str(local_cfg["outputs"]["summary_json"]),
                        "hidden_summary_json": str(hidden_summary["files"]["json_out"]),
                        "outputs": traj_outputs,
                        "shape_diagnostics": diagnostics,
                        **diagnostics,
                    })
                    candidate_summary = {
                        "model": local_cfg["model"],
                        "params": local_cfg["params"],
                        "frac_order": local_cfg["frac_order"],
                        "runtime_contract": local_cfg.get("runtime_contract", {}),
                        "candidate": record,
                        "nyquist_and_continuation": cont_summary,
                        "hidden_verification": hidden_summary,
                        "notes": [
                            f"Candidato evaluado dentro de run_mode={sweep_key}.",
                            "Semilla construida como a_mu,j Re(v_j exp(i theta)).",
                            "Los pares no admisibles se descartan si no cumplen 0 < k_j < Delta^mu.",
                        ],
                    }
                    Path(local_cfg["outputs"]["summary_json"]).write_text(
                        json.dumps(candidate_summary, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    record["error"] = str(exc)
                    Path(local_cfg["outputs"]["summary_json"]).write_text(
                        json.dumps(record, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    if not bool(cfg[sweep_key].get("continue_on_error", True)):
                        raise
                    log(f"Machado sweep {slug}: fallo controlado: {exc}")
                records.append(record)
                executed += 1

            if max_candidates is not None and executed >= max_candidates:
                break
        if max_candidates is not None and executed >= max_candidates:
            break

    csv_path = root / f"{sweep_key}_summary.csv"
    fields = [
        "slug", "status", "branch_index", "mu", "theta", "omega0", "k0", "delta_power_mu",
        "a0", "seed_norm", "total_target_hits", "hiddenness_status",
        "range_x", "range_y", "range_z", "output_dir", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    best = min(records, key=df_candidate_sort_key) if records else None
    summary = {
        "model": cfg["model"],
        "params": cfg["params"],
        "frac_order": cfg["frac_order"],
        "run_mode": sweep_key,
        "grid": {
            str(k): {
                "mu_values": [float(v) for v in val.get("mu_values", [])],
                "theta_values": [float(v) for v in val.get("theta_values", [])],
            }
            for k, val in grid.items()
        },
        "raw_pairs": [{"branch_index": i, "omega0": float(p[0]), "k": float(p[1])} for i, p in enumerate(raw_pairs)],
        "delta": delta,
        "output_layout": output_layout_dict(cfg),
        "output_root": str(root),
        "summary_csv": str(csv_path),
        "records": records,
        "discarded": discarded,
        "best_candidate": best,
        "selection_rule": "menor total_target_hits; empate por mayor rango_x y menor norma de semilla",
        "notes": [
            "No calcula bifurcaciones ni cuencas.",
            "Cada registro ok incluye continuacion y verificacion de ocultedad.",
            "Si best_candidate tiene total_target_hits > 0, el candidato no queda soportado como oculto bajo esta muestra.",
        ],
    }
    summary_path = root / f"{sweep_key}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{sweep_key} completado.", flush=True)
    print("Directorio:", root, flush=True)
    print("Resumen CSV:", csv_path, flush=True)
    print("Resumen JSON:", summary_path, flush=True)
    if best:
        print(
            "Mejor candidato:",
            best.get("slug"),
            "TARGET=",
            best.get("total_target_hits"),
            "status=",
            best.get("hiddenness_status"),
            flush=True,
        )
    return summary


def run_fractional_order_sweep(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta un barrido ligero solo en el orden fraccionario q.

    Propósito matemático:
    Comparar cómo cambia la predicción por función descriptiva/Weyl y la
    continuación numérica al variar q, manteniendo fijos los parámetros de
    Chua. Esto sirve para seleccionar valores candidatos antes de ejecutar la
    verificación completa de ocultedad y cuencas.

    Ecuaciones usadas:
    Para cada q se evalúa W_q(i omega) usando (i omega)^q en la rama principal,
    se resuelve la condición de función descriptiva y se integra la familia en
    epsilon con EFORK y memoria truncada.

    Salida:
    Un JSON y un CSV maestros, más una carpeta por q con Nyquist, continuación,
    atractor final y trayectoria guardada.

    Advertencias sobre validez:
    El barrido no declara caos ni ocultedad. Un atractor delgado puede indicar
    comportamiento casi periódico, pero se requiere verificación posterior con
    integración refinada, espectro/Lyapunov operacional y cuencas.
    """
    ensure_current_chua_params(cfg)
    sweep_cfg = cfg["q_sweep"]
    q_values = [validate_fractional_order(q) for q in sweep_cfg["q_values"]]
    root = Path(sweep_cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    log(
        "Barrido q: "
        f"{len(q_values)} valores; "
        f"memory_mode={cfg['continuation'].get('memory_mode', 'window')}; "
        f"memory_update_source={cfg['continuation'].get('memory_update_source', 'observed')}; "
        f"salida={root}."
    )

    for idx, qord in enumerate(q_values, start=1):
        tag = q_sweep_tag(qord)
        q_dir = root / f"q_{tag}"
        q_dir.mkdir(parents=True, exist_ok=True)
        log(f"Barrido q {idx}/{len(q_values)}: q={qord:.6f}")

        local_cfg = copy.deepcopy(cfg)
        local_cfg["frac_order"] = float(qord)
        local_cfg["basin"]["q"] = float(qord)
        local_cfg["outputs"] = q_sweep_outputs(cfg, q_dir, tag)
        synchronize_runtime_contract(local_cfg)
        log(f"Barrido q {idx}/{len(q_values)} contrato efectivo: {q_sweep_contract_line(local_cfg)}")

        record: Dict[str, Any] = {
            "q": float(qord),
            "tag": tag,
            "status": "failed",
            "output_dir": str(q_dir),
        }
        try:
            nyq_data = compute_nyquist_seed_and_continuation(local_cfg)
            plot_clean_nyquist(local_cfg, nyq_data)
            plot_clean_continuation_progress(local_cfg, nyq_data["results"])
            plot_clean_final_attractor(local_cfg, nyq_data["results"])

            nyq_csv = q_dir / f"nyquist_df_samples_q_{tag}.csv"
            save_nyquist_df_samples(local_cfg, nyq_csv)

            traj = np.asarray(nyq_data["results"][-1]["traj"], dtype=float)
            traj_outputs = save_q_sweep_trajectory(
                q_dir,
                tag,
                traj,
                write_csv=bool(sweep_cfg.get("trajectory_csv", True)),
            )
            diagnostics = trajectory_shape_diagnostics(traj)
            chosen = nyq_data["summary"]["chosen_branch"]
            final_state = np.asarray(nyq_data["final_state"], dtype=float)

            record.update({
                "status": "ok",
                "omega0": float(nyq_data["omega0"]),
                "k0": float(nyq_data["k0"]),
                "a0": float(nyq_data["a0"]),
                "seed_x": float(nyq_data["xseed"][0]),
                "seed_y": float(nyq_data["xseed"][1]),
                "seed_z": float(nyq_data["xseed"][2]),
                "final_x": float(final_state[0]),
                "final_y": float(final_state[1]),
                "final_z": float(final_state[2]),
                "branch_index": int(local_cfg["branch_index"]),
                "memory_mode": str(local_cfg["continuation"].get("memory_mode", "window")),
                "memory_update_source": str(local_cfg["continuation"].get("memory_update_source", "observed")),
                "runtime_contract": local_cfg.get("runtime_contract", {}),
                "continuation_summary_json": str(local_cfg["outputs"]["cont_json"]),
                "nyquist_pdf": str(local_cfg["outputs"]["nyquist_pdf"]),
                "continuation_pdf": str(local_cfg["outputs"]["cont_progress_pdf"]),
                "final_attractor_pdf": str(local_cfg["outputs"]["final_attr_pdf"]),
                "nyquist_df_csv": str(nyq_csv),
                "chosen_branch": chosen,
                "shape_diagnostics": diagnostics,
                **diagnostics,
                **traj_outputs,
            })
            q_summary = {
                "model": local_cfg["model"],
                "params": local_cfg["params"],
                "q": float(qord),
                "runtime_contract": local_cfg.get("runtime_contract", {}),
                "status": "ok",
                "chosen_branch": chosen,
                "final_state_eps1": final_state.tolist(),
                "outputs": {
                    "nyquist_pdf": str(local_cfg["outputs"]["nyquist_pdf"]),
                    "continuation_pdf": str(local_cfg["outputs"]["cont_progress_pdf"]),
                    "final_attractor_pdf": str(local_cfg["outputs"]["final_attr_pdf"]),
                    "continuation_summary_json": str(local_cfg["outputs"]["cont_json"]),
                    "nyquist_df_csv": str(nyq_csv),
                    **traj_outputs,
                },
                "shape_diagnostics": diagnostics,
                "notes": [
                    "Resultado de barrido ligero: no incluye verificacion de ocultedad ni cuencas.",
                    "La semilla proviene de funcion descriptiva/Weyl y se valida aqui solo por integracion EFORK causal.",
                ],
            }
            Path(local_cfg["outputs"]["summary_json"]).write_text(
                json.dumps(q_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            record["error"] = str(exc)
            (q_dir / f"q_summary_{tag}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if not bool(sweep_cfg.get("continue_on_error", True)):
                raise
            log(f"Barrido q={qord:.6f}: fallo controlado: {exc}")

        records.append(record)

    csv_path = root / "q_sweep_summary.csv"
    csv_fields = [
        "q", "status", "omega0", "k0", "a0",
        "seed_x", "seed_y", "seed_z",
        "final_x", "final_y", "final_z",
        "range_x", "range_y", "range_z",
        "cov_ratio_mid_to_max", "cov_ratio_min_to_max",
        "memory_mode", "memory_update_source", "output_dir", "nyquist_pdf", "final_attractor_pdf",
        "final_attractor_npz", "final_attractor_csv", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    summary = {
        "model": cfg["model"],
        "params": cfg["params"],
        "run_mode": "q_sweep",
        "q_values": q_values,
        "output_layout": output_layout_dict(cfg),
        "output_root": str(root),
        "summary_csv": str(csv_path),
        "records": records,
        "notes": [
            "Este barrido evita bifurcaciones, verificacion oculta y cuencas para explorar q rapidamente.",
            "cov_ratio_min_to_max pequeño indica geometria muy delgada, pero no prueba periodicidad por si solo.",
            "Para una corrida completa de un q elegido usa HIDDEN_ATTRACTORS_FRAC_ORDER=<q> y HIDDEN_ATTRACTORS_RUN_MODE=balanced o full.",
        ],
    }
    summary_path = root / "q_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Barrido de q completado.", flush=True)
    print("Directorio:", root, flush=True)
    print("Resumen CSV:", csv_path, flush=True)
    print("Resumen JSON:", summary_path, flush=True)
    return summary


def load_previous_pipeline_summary_for_basin_only(cfg: Dict[str, Any]) -> Tuple[Path, Dict[str, Any], np.ndarray]:
    summary_path = Path(cfg["outputs"]["summary_json"])
    cont_path = Path(cfg["outputs"]["cont_json"])
    if not summary_path.exists() and not cont_path.exists():
        raise FileNotFoundError(
            f"No existe {summary_path} ni {cont_path}. Ejecuta primero una corrida balanced o full."
        )

    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    for path in (summary_path, cont_path):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                candidates.append((path, json.load(f)))
    if not candidates:
        raise FileNotFoundError("No se encontro ningun resumen previo para basin_only.")

    if (
        summary_path.exists()
        and cont_path.exists()
        and bool(candidates[0][1].get("style_only", False))
    ):
        log(
            "Modo basin_only: unified_pipeline_summary.json es style_only; "
            "se usa unified_continuation_summary.json para recuperar la continuacion completa."
        )
        candidates = [candidates[1], candidates[0]]

    last_error: Exception | None = None
    for selected_path, summary in candidates:
        try:
            if "final_state_eps1" not in summary and isinstance(summary.get("continuation"), dict):
                nested_final_state = summary["continuation"].get("final_state_eps1")
                if nested_final_state is not None:
                    summary = dict(summary)
                    summary["final_state_eps1"] = nested_final_state
            required = ["params", "runtime_contract", "final_state_eps1"]
            missing = [key for key in required if key not in summary]
            if missing:
                raise ValueError(
                    f"{selected_path} no contiene {', '.join(missing)}."
                )
            if "frac_order" not in summary and "q" not in summary.get("runtime_contract", {}):
                raise ValueError(
                    f"{selected_path} no contiene frac_order ni runtime_contract.q."
                )
            final_state = np.asarray(summary["final_state_eps1"], dtype=float)
            if final_state.shape != (3,) or not np.all(np.isfinite(final_state)):
                raise ValueError(f"final_state_eps1 en {selected_path} debe ser un vector finito de dimension 3.")
            return selected_path, summary, final_state
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        "Ningun resumen previo contiene los campos necesarios para basin_only. "
        "Ejecuta primero una corrida balanced o full actualizada."
    ) from last_error


def apply_basin_only_contract_from_summary(cfg: Dict[str, Any], summary: Dict[str, Any]) -> None:
    params = summary["params"]
    contract = summary["runtime_contract"]
    frac_order = validate_fractional_order(float(summary.get("frac_order", contract.get("q"))))
    summary_model = summary.get("model", {})
    if isinstance(summary_model, dict):
        summary_model = summary_model.get("kind", contract.get("model", cfg["model"]["kind"]))
    model = normalize_chua_model(summary_model or contract.get("model", cfg["model"]["kind"]))
    cfg["model"]["kind"] = model
    cfg["model"]["output_slug"] = CHUA_MODEL_SLUGS[model]

    cfg["params"].update({
        "alpha_chua": float(params["alpha_chua"]),
        "beta": float(params["beta"]),
        "gamma_chua": float(params["gamma_chua"]),
        "m0": float(params.get("m0", cfg["params"]["m0"])),
        "m1": float(params.get("m1", cfg["params"]["m1"])),
        "a1": float(params.get("a1", cfg["params"].get("a1", 0.4))),
        "a2": float(params.get("a2", cfg["params"].get("a2", -1.5585))),
        "rho": float(params.get("rho", cfg["params"].get("rho", 1.0))),
    })
    cfg["frac_order"] = frac_order

    h = float(contract.get("h", cfg["continuation"]["h"]))
    Lm = float(contract.get("Lm", cfg["continuation"]["Lm"]))
    t_transient = float(contract.get("t_transient", cfg["continuation"]["t_transient"]))
    t_keep = float(contract.get("t_keep", cfg["continuation"]["t_keep"]))
    basin_keep_default = float(contract.get("basin_t_keep", t_keep))
    basin_keep = _env_positive_float("HIDDEN_ATTRACTORS_BASIN_T_KEEP", basin_keep_default)
    bif_keep = float(contract.get("bifurcation_t_keep", t_keep))

    basin_z0 = contract.get("basin_z0", cfg["basin"].get("z0", "final_state"))
    cfg["basin"]["z0"] = basin_z0

    cfg["continuation"].update({
        "h": h,
        "Lm": Lm,
        "t_transient": t_transient,
        "t_keep": t_keep,
    })
    cfg["verify_hidden"].update({
        "h": h,
        "Lm": Lm,
        "TBURN_REF": t_transient,
        "TBURN_TEST": t_transient,
        "TMAX_REF": t_transient + t_keep,
        "TMAX_TEST": t_transient + t_keep,
    })
    cfg["basin"].update({
        "q": frac_order,
        "h": h,
        "Lm": Lm,
        "TBURN": t_transient,
        "TMAX": t_transient + basin_keep,
    })
    cfg["basin_python"].update({
        "h": h,
        "Lm": Lm,
        "t_burn": t_transient,
        "t_total": t_transient + basin_keep,
    })
    cfg["bifurcation"].update({
        "h": h,
        "Lm": Lm,
        "t_burn": t_transient,
        "t_total": t_transient + bif_keep,
    })
    cfg["lyapunov"].update({
        "h": h,
        "Lm": Lm,
        "t_burn": t_transient,
    })
    _apply_basin_resolution_env_overrides(cfg)
    _apply_basin_plane_env_override(cfg)
    cfg["runtime_contract"] = {
        "model": cfg["model"]["kind"],
        "output_slug": cfg["model"]["output_slug"],
        "q": frac_order,
        "h": h,
        "Lm": Lm,
        "t_transient": t_transient,
        "t_keep": t_keep,
        "basin_t_keep": basin_keep,
        "bifurcation_t_keep": bif_keep,
        "basin_grid": [int(cfg["basin"]["nx"]), int(cfg["basin"]["ny"])],
        "basin_planes_grid": [int(cfg["basin_python"]["nx"]), int(cfg["basin_python"]["ny"])],
        "basin_z0": cfg["basin"].get("z0", "final_state"),
        "basin_workers": int(cfg["basin"].get("workers", 1)),
        "note": (
            "basin_only loaded q/h/Lm/transient from the selected previous summary; "
            "only basin resolution, basin z0 and basin post-transient window are intended overrides."
        ),
    }
    refresh_parallel_runtime_contract(cfg)


def run_basin_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    summary_path, previous_summary, final_state = load_previous_pipeline_summary_for_basin_only(cfg)
    apply_basin_only_contract_from_summary(cfg, previous_summary)
    log(f"Modo basin_only: fuente={summary_path}")
    log(f"Modo basin_only: final_state_eps1={final_state.tolist()}")
    log_runtime_contract(cfg, header="Contrato numerico efectivo para basin_only")

    ensure_current_chua_params(cfg)
    log("basin_only: regenerando fig06_basin_overlay.pdf.")
    basin, E0, Ep, Em = compute_basin_and_eq(cfg, final_state)
    basin_diagnostics = compute_basin_seed_diagnostics(cfg, basin, final_state)
    plot_clean_basin_overlay(cfg, basin, E0, Ep, Em, final_state)

    final_pdf_keys = ["basin_pdf"]
    planes_summary: Dict[str, Any] = {
        "enabled": bool(cfg.get("article_style", {}).get("basin_planes_enabled", True)),
        "generated": False,
        "backend": "none",
    }
    if bool(cfg.get("article_style", {}).get("basin_planes_enabled", True)):
        log("basin_only: regenerando fig10_basin_planes.pdf y cortes xy/xz/yz.")
        basin_planes = compute_basin_planes_c(cfg, final_state)
        plot_basin_planes(basin_planes, cfg["outputs"]["basin_planes_pdf"])
        plot_single_basin_plane(basin_planes["xy"], "xy", cfg["outputs"]["basin_xy_pdf"])
        plot_single_basin_plane(basin_planes["xz"], "xz", cfg["outputs"]["basin_xz_pdf"])
        plot_single_basin_plane(basin_planes["yz"], "yz", cfg["outputs"]["basin_yz_pdf"])
        final_pdf_keys.extend(["basin_planes_pdf", "basin_xy_pdf", "basin_xz_pdf", "basin_yz_pdf"])
        planes_summary.update({
            "generated": True,
            "backend": "c",
            "grid": [int(cfg["basin_python"]["nx"]), int(cfg["basin_python"]["ny"])],
        })
    else:
        log("basin_only: HIDDEN_ATTRACTORS_BASIN_PLANES=0; se omite fig10.")

    summary = {
        "run_mode": "basin_only",
        "source_summary_json": str(summary_path),
        "model": cfg["model"],
        "params": cfg["params"],
        "frac_order": cfg["frac_order"],
        "runtime_contract": cfg.get("runtime_contract", {}),
        "output_layout": output_layout_dict(cfg),
        "output_root": str(RUNTIME_ROOT),
        "final_state_eps1": final_state.tolist(),
        "basin_diagnostics": basin_diagnostics,
        "basin_planes": planes_summary,
        "final_pdfs": {k: str(cfg["outputs"][k]) for k in final_pdf_keys},
        "notes": [
            "Modo basin_only: reutiliza final_state_eps1 del resumen previo seleccionado y no recalcula Nyquist/DF ni continuacion.",
            "No ejecuta hidden verification, bifurcaciones, espectro ni Lyapunov.",
        ],
    }
    Path(cfg["outputs"]["basin_only_summary_json"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Basin-only completado. PDFs regenerados:", flush=True)
    for k, v in summary["final_pdfs"].items():
        print(f"- {k}: {v}", flush=True)
    print("Resumen basin_only:", cfg["outputs"]["basin_only_summary_json"], flush=True)
    return summary


def main():
    log(f"Inicio pipeline; modo={CONFIG.get('run_mode', 'full')}; salida={RUNTIME_ROOT}")
    log_environment_overrides()
    layout_files = write_output_layout_files(CONFIG)
    log_output_layout(layout_files)
    if CONFIG.get("run_mode") == "basin_only":
        run_basin_only(CONFIG)
        return
    if CONFIG.get("run_mode") == "df_compare":
        log_runtime_contract(CONFIG)
        run_describing_function_comparison(CONFIG)
        return
    if CONFIG.get("run_mode") in {"machado_sweep", "machado_sweep_fast"}:
        log_runtime_contract(CONFIG)
        run_machado_sweep(CONFIG)
        return
    log_runtime_contract(CONFIG)
    if CONFIG.get("run_mode") == "q_sweep":
        run_fractional_order_sweep(CONFIG)
        return
    if CONFIG.get("run_mode") == "fast":
        log("Modo fast activo: menos puntos, integraciones mas cortas y cuenca C de menor resolucion.")

    ensure_current_chua_params(CONFIG)
    log("Etapa 1/8: Nyquist/DF, continuacion numerica y figuras base.")
    nyq_data = compute_nyquist_seed_and_continuation(CONFIG)
    plot_clean_nyquist(CONFIG, nyq_data)
    plot_clean_continuation_progress(CONFIG, nyq_data["results"])
    plot_clean_final_attractor(CONFIG, nyq_data["results"])

    phase_summary: Dict[str, Any] = {"status": "skipped", "extra_figures": {}}
    basin_planes_summary: Dict[str, Any] = {"status": "skipped", "extra_figures": {}}
    bifurcation_summary: Dict[str, Any] = {"status": "skipped", "extra_figures": {}}
    article_summary = merge_article_style_summaries(
        CONFIG,
        phase_summary,
        basin_planes_summary,
        bifurcation_summary,
        write=False,
    )
    if bool(CONFIG.get("article_style", {}).get("enabled", True)):
        log("Etapa 2/8: figuras ligeras de la fase actual.")
        phase_summary = run_article_phase_figures(CONFIG, nyq_data)
        article_summary = merge_article_style_summaries(
            CONFIG,
            phase_summary,
            basin_planes_summary,
            bifurcation_summary,
            write=False,
        )

    log("Etapa 3/8: analisis espectral.")
    spectral_summary = run_spectral_analysis(CONFIG, nyq_data)

    if bool(CONFIG.get("style_only", False)):
        article_summary = merge_article_style_summaries(
            CONFIG,
            phase_summary,
            basin_planes_summary,
            bifurcation_summary,
            write=True,
        )
        log("Modo style_only activo: se omiten verificacion oculta, cuencas y bifurcaciones.")
        base_pdf_keys = ["nyquist_pdf", "cont_progress_pdf", "final_attr_pdf"]
        article_pdf_keys = list(article_summary.get("extra_figures", {}).keys())
        spectral_pdf_keys = list(spectral_summary.get("extra_figures", {}).keys())
        final_pdf_keys = base_pdf_keys + [k for k in article_pdf_keys if k not in base_pdf_keys]
        final_pdf_keys += [k for k in spectral_pdf_keys if k not in final_pdf_keys]
        pipeline_summary = {
            "model": CONFIG["model"],
            "params": CONFIG["params"],
            "frac_order": CONFIG["frac_order"],
            "run_mode": CONFIG.get("run_mode", "full"),
            "style_only": True,
            "runtime_contract": CONFIG.get("runtime_contract", {}),
            "output_layout": output_layout_dict(CONFIG),
            "output_root": str(RUNTIME_ROOT),
            "chosen_branch": nyq_data["summary"]["chosen_branch"],
            "final_state_eps1": nyq_data["summary"]["continuation"]["final_state_eps1"],
            "article_style": article_summary,
            "spectral": spectral_summary,
            "final_pdfs": {k: str(CONFIG["outputs"][k]) for k in final_pdf_keys},
            "notes": [
                "Modo de prueba visual: no ejecuta verificacion de ocultedad, cuencas ni bifurcaciones.",
                "Usa HIDDEN_ATTRACTORS_STYLE_ONLY=0 para la corrida completa.",
            ],
        }
        with open(CONFIG["outputs"]["summary_json"], "w", encoding="utf-8") as f:
            json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
        print("Pipeline visual completado. PDFs generados:", flush=True)
        print("Directorio de salida:", RUNTIME_ROOT, flush=True)
        for k, v in pipeline_summary["final_pdfs"].items():
            print(f"- {k}: {v}", flush=True)
        print("Resumen:", CONFIG["outputs"]["summary_json"], flush=True)
        return

    log("Etapa 4/8: verificacion de atractor oculto con backend C.")
    hidden_summary = run_hidden_verify_with_seed(CONFIG, nyq_data["final_state"])
    log("Etapa 5/8: graficas de seccion de referencia y resumen de sondas.")
    plot_clean_reference_section(CONFIG, hidden_summary["files"]["ref_csv_out"])
    eq_names = list(hidden_summary["equilibria"].keys())
    plot_clean_probe_summary(CONFIG, hidden_summary["files"]["summary_csv_out"], eq_names)
    hidden_illustration_summary = plot_hiddenness_illustration(CONFIG, nyq_data, hidden_summary)

    log("Etapa 6/8: cuencas de atraccion con backend C.")
    basin, E0, Ep, Em = compute_basin_and_eq(CONFIG, nyq_data["final_state"])
    basin_diagnostics = compute_basin_seed_diagnostics(CONFIG, basin, nyq_data["final_state"])
    save_distances_csv(CONFIG, nyq_data["final_state"], E0, Ep, Em)
    plot_clean_basin_overlay(CONFIG, basin, E0, Ep, Em, nyq_data["final_state"])
    if bool(CONFIG.get("article_style", {}).get("enabled", True)):
        basin_planes_summary = run_basin_plane_figures(CONFIG, nyq_data["final_state"])
        article_summary = merge_article_style_summaries(
            CONFIG,
            phase_summary,
            basin_planes_summary,
            bifurcation_summary,
            write=False,
        )

    if (
        bool(CONFIG.get("article_style", {}).get("enabled", True))
        and bool(CONFIG.get("bifurcation", {}).get("enabled", True))
    ):
        log("Etapa 7/8: diagramas de bifurcacion al final.")
        bifurcation_summary = run_bifurcation_figures(CONFIG, nyq_data["final_state"])
    elif bool(CONFIG.get("article_style", {}).get("enabled", True)):
        log("Etapa 7/8: bifurcacion omitida por HIDDEN_ATTRACTORS_BIFURCATION=0.")
    article_summary = merge_article_style_summaries(
        CONFIG,
        phase_summary,
        basin_planes_summary,
        bifurcation_summary,
        write=True,
    )

    base_pdf_keys = [
        "nyquist_pdf",
        "cont_progress_pdf",
        "final_attr_pdf",
        "ref_section_pdf",
        "probe_summary_pdf",
        "hidden_illustration_overview_pdf",
        "hidden_illustration_zoom_pdf",
        "basin_pdf",
    ]
    article_pdf_keys = list(article_summary.get("extra_figures", {}).keys())
    spectral_pdf_keys = list(spectral_summary.get("extra_figures", {}).keys())
    final_pdf_keys = base_pdf_keys + [k for k in article_pdf_keys if k not in base_pdf_keys]
    final_pdf_keys += [k for k in spectral_pdf_keys if k not in final_pdf_keys]

    log("Etapa 8/8: escritura de resumen JSON.")
    pipeline_summary = {
        "model": CONFIG["model"],
        "params": CONFIG["params"],
        "frac_order": CONFIG["frac_order"],
        "run_mode": CONFIG.get("run_mode", "full"),
        "runtime_contract": CONFIG.get("runtime_contract", {}),
        "output_layout": output_layout_dict(CONFIG),
        "output_root": str(RUNTIME_ROOT),
        "chosen_branch": nyq_data["summary"]["chosen_branch"],
        "final_state_eps1": nyq_data["summary"]["continuation"]["final_state_eps1"],
        "hidden_verification": hidden_summary,
        "hiddenness_illustration": hidden_illustration_summary,
        "basin_diagnostics": basin_diagnostics,
        "article_style": article_summary,
        "spectral": spectral_summary,
        "final_pdfs": {k: str(CONFIG["outputs"][k]) for k in final_pdf_keys},
        "notes": [
            "Nyquist/DF produce una semilla oscilatoria candidata.",
            "La continuaciÃ³n en epsilon transporta la semilla al sistema objetivo.",
            "La ocultedad se verifica por integraciÃ³n causal y muestreo desde vecindades de equilibrio.",
            "Las figuras finales se guardan en PDF, separadas, con nombres en ejes y sin tÃ­tulo.",
            "Las figuras adicionales estilo artÃ­culo se integraron dentro del mismo pipeline.",
        ],
    }
    with open(CONFIG["outputs"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)

    print("Pipeline completado. PDFs finales:", flush=True)
    print("Directorio de salida:", RUNTIME_ROOT, flush=True)
    for k, v in pipeline_summary["final_pdfs"].items():
        print(f"- {k}: {v}", flush=True)
    print("Resumen:", CONFIG["outputs"]["summary_json"], flush=True)


if __name__ == "__main__":
    main()
