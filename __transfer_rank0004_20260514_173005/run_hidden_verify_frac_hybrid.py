#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_hidden_verify_frac_hybrid.py

Versión híbrida:
- C ejecuta la parte pesada: integración EFORK, memoria fraccionaria, muestreo,
  clasificación y construcción de la nube de referencia.
- Python se queda con: configuración, llamada al backend, lectura de CSV,
  gráficas y ensamblado del JSON final.

Archivos que produce:
- hidden_target_check_frac.csv
- reference_section.csv
- summary_by_radius.csv
- hidden_target_summary.json
- reference_section.png
- probe_summary.png
"""

from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path, PureWindowsPath
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT_SLUG = "hidden_attractors_fractional_order"
DEFAULT_MAX_WORK_UNITS = 4_000_000_000


# CONFIGURACION DE CARPETAS DE CORRIDA
# Misma regla que unified_nyquist_hidden_pipeline.py:
#   RUNS_BASE_DIR / chua_piecewise / hidden_verify
#   RUNS_BASE_DIR / chua_arctan    / hidden_verify
RUNS_BASE_DIR_DEFAULT = ROOT
RUNS_MODEL_SUBDIR_ENABLED = True
RUNS_BASE_DIR_ENV = os.environ.get("HIDDEN_ATTRACTORS_OUTPUT_DIR")


def normalize_chua_model(raw) -> str:
    text = str(raw or "piecewise").strip().lower().replace("-", "_")
    aliases = {
        "pwl": "piecewise",
        "nonsmooth": "piecewise",
        "non_smooth": "piecewise",
        "no_suave": "piecewise",
        "tramos": "piecewise",
        "piecewise_linear": "piecewise",
        "atan": "arctan",
        "arc_tan": "arctan",
        "smooth": "arctan",
        "suave": "arctan",
    }
    text = aliases.get(text, text)
    if text not in {"piecewise", "arctan"}:
        raise ValueError("HIDDEN_ATTRACTORS_MODEL debe ser 'piecewise' o 'arctan'.")
    return text


CHUA_MODEL_KIND = normalize_chua_model(
    os.environ.get("HIDDEN_ATTRACTORS_MODEL", os.environ.get("HIDDEN_ATTRACTORS_CHUA_MODEL", "piecewise"))
)
CHUA_MODEL_SLUGS = {
    "piecewise": "chua_piecewise",
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
    configured_root = run_root_from_base(RUNS_BASE_DIR)
    if _dir_accepts_writes(configured_root):
        return configured_root

    if RUNS_BASE_DIR_ENV:
        raise PermissionError(
            "No se pudo escribir en HIDDEN_ATTRACTORS_OUTPUT_DIR="
            f"{RUNS_BASE_DIR}. Corrige esa carpeta o quita la variable de entorno."
        )

    fallback_root = Path(tempfile.gettempdir()) / PROJECT_SLUG / CHUA_OUTPUT_SLUG
    if _dir_accepts_writes(fallback_root):
        return fallback_root

    raise PermissionError(
        "No se encontró un directorio con permisos de escritura para los resultados. "
        "Define RUNS_BASE_DIR_DEFAULT al inicio del script o HIDDEN_ATTRACTORS_OUTPUT_DIR."
    )


RUNTIME_ROOT = resolve_runtime_root()
HIDDEN_VERIFY_DIR = RUNTIME_ROOT / "hidden_verify"
NATIVE_DIR = RUNTIME_ROOT / "native"
DEFAULT_CONFIG_PATH = HIDDEN_VERIFY_DIR / "config_hidden_verify_frac.json"


DEFAULT_CONFIG = {
    "params": {
        "model": "piecewise",
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
    "target_seed": {
        "x": 5.85176778548633,
        "y": 0.370408600306164,
        "z": -8.36097293442065,
    },
    "integration": {
        "h": 0.01,
        "Lm": 5.0,
        "TMAX_REF": 140.0,
        "TMAX_TEST": 140.0,
        "TBURN_REF": 70.0,
        "TBURN_TEST": 70.0,
    },
    "thresholds": {
        "R_DIV": 120.0,
        "EPS_EQ": 0.03,
        "CAP_WIN": 200,
        "SEC_TOL": 0.12,
        "MIN_SEC_MATCH": 12,
        "TEST_MAX_SEC": 60,
        "HIT_FRAC_REQ": 0.70,
    },
    "sampling": {
        "RADII": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "NSAMPLES_PER_RADIUS": 6,
        "random_seed": 123456789,
        "EQ_FILTER": "",
    },
    "files": {
        "summary_from_pipeline": "unified_pipeline_summary.json",
        "csv_out": "hidden_target_check_frac.csv",
        "ref_csv_out": "reference_section.csv",
        "summary_csv_out": "summary_by_radius.csv",
        "json_out": "hidden_target_summary.json",
        "fig_section": "reference_section.png",
        "fig_probe": "probe_summary.png",
    },
    "backend": {
        "source_c": "chua_hidden_backend.c",
        "exe": "./chua_hidden_backend",
        "compile": True,
        "openmp": True,
    },
    "safety": {
        "profile": "safe",
        "max_threads": 2,
        "timeout_sec": 1800,
        "max_work_units": DEFAULT_MAX_WORK_UNITS,
        "sync_from_pipeline_summary": True,
    },
}


@dataclass
class ChuaParams:
    alpha_chua: float
    beta: float
    gamma_chua: float
    m0: float
    m1: float
    model: str = "piecewise"
    a1: float = 0.4
    a2: float = -1.5585
    rho: float = 1.0


def resolve_config_path(path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = HIDDEN_VERIFY_DIR / cfg_path.name
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg_path


def portable_filename(path_value: str | Path | None, default_name: str) -> str:
    """Extrae el nombre de archivo aceptando rutas POSIX o Windows heredadas.

    Propósito matemático/computacional:
    Mantener reproducibles las corridas de verificación de ocultación cuando una
    configuración se mueve entre sistemas operativos. No modifica ecuaciones ni
    parámetros del integrador EFORK; solo evita que una ruta absoluta antigua
    sea interpretada como parte del nombre del backend o de los archivos CSV.

    Ecuaciones usadas:
    No aplica. Es una normalización de rutas previa a la compilación y a la
    escritura de resultados.

    Parámetros de entrada:
    path_value:
        Ruta o nombre de archivo escrito con separadores POSIX "/" o Windows "\\".
    default_name:
        Nombre usado si path_value está vacío.

    Salida:
    Nombre final del archivo, sin directorios.

    Advertencias sobre validez:
    Si se pasa una ruta absoluta de Windows en una máquina POSIX, se conserva
    únicamente el nombre del archivo y se reubica dentro del directorio runtime
    del proyecto.
    """
    raw = str(default_name if path_value is None else path_value).strip()
    if not raw:
        raw = default_name
    native_name = Path(raw).name
    windows_name = PureWindowsPath(raw).name
    if "\\" in raw and windows_name:
        return windows_name
    return native_name or windows_name or default_name


def _contains_windows_path_fragment(path_value: str | Path | None) -> bool:
    raw = "" if path_value is None else str(path_value)
    return "\\" in raw or ":\\" in raw or ":/" in raw


def _contains_windows_absolute_fragment(path_value: str | Path | None) -> bool:
    raw = "" if path_value is None else str(path_value)
    return PureWindowsPath(raw).is_absolute() or ":\\" in raw or ":/" in raw


def _resolve_project_source_path(path_value: str | Path | None, default_name: str) -> Path:
    raw = str(default_name if path_value is None else path_value).strip()
    if not raw:
        raw = default_name

    path = Path(raw).expanduser()
    windows_path = PureWindowsPath(raw)
    if path.is_absolute() and (path.exists() or not _contains_windows_path_fragment(raw)):
        return path
    if os.name != "nt" and _contains_windows_absolute_fragment(raw):
        return ROOT / portable_filename(raw, default_name)
    if "\\" in raw:
        parts = [
            part for part in windows_path.parts
            if part not in ("\\", "/") and not part.endswith(":")
        ]
        if parts:
            return ROOT.joinpath(*parts)
    if path.is_absolute():
        return ROOT / portable_filename(raw, default_name)
    return ROOT / path


def _resolve_project_input_path(path_value: str | Path | None, default_name: str) -> Path:
    raw = str(default_name if path_value is None else path_value).strip()
    if not raw:
        raw = default_name
    path = Path(raw).expanduser()
    if path.is_absolute() and (os.name == "nt" or not _contains_windows_path_fragment(raw)):
        return path
    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute():
        return RUNTIME_ROOT / portable_filename(raw, default_name)
    if "\\" in raw:
        parts = [
            part for part in windows_path.parts
            if part not in ("\\", "/") and not part.endswith(":")
        ]
        if parts:
            return RUNTIME_ROOT.joinpath(*parts)
    return RUNTIME_ROOT / path


def _resolve_output_path(path_value: str | Path | None, base_dir: Path, default_name: str) -> Path:
    raw = str(default_name if path_value is None else path_value).strip()
    if not raw:
        raw = default_name
    path = Path(raw).expanduser()
    if path.is_absolute() and not _contains_windows_path_fragment(raw):
        return path
    return base_dir / portable_filename(raw, default_name)


def prepare_runtime_paths(cfg: Dict, runtime_dir: Path | None = None) -> Dict:
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else HIDDEN_VERIFY_DIR
    native_dir = Path(cfg.get("native_dir", NATIVE_DIR))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)

    files = cfg.setdefault("files", {})
    files["csv_out"] = str(_resolve_output_path(files.get("csv_out"), runtime_dir, "hidden_target_check_frac.csv"))
    files["ref_csv_out"] = str(_resolve_output_path(files.get("ref_csv_out"), runtime_dir, "reference_section.csv"))
    files["summary_csv_out"] = str(_resolve_output_path(files.get("summary_csv_out"), runtime_dir, "summary_by_radius.csv"))
    files["json_out"] = str(_resolve_output_path(files.get("json_out"), runtime_dir, "hidden_target_summary.json"))
    files["fig_section"] = str(_resolve_output_path(files.get("fig_section"), runtime_dir, "reference_section.png"))
    files["fig_probe"] = str(_resolve_output_path(files.get("fig_probe"), runtime_dir, "probe_summary.png"))

    summary_from_pipeline = files.get("summary_from_pipeline")
    if summary_from_pipeline:
        summary_path = _resolve_project_input_path(summary_from_pipeline, "summary.json")
        files["summary_from_pipeline"] = str(summary_path.resolve())

    backend = cfg.setdefault("backend", {})
    source_c = _resolve_project_source_path(backend.get("source_c"), "chua_hidden_backend.c")

    exe = _resolve_output_path(backend.get("exe"), native_dir, "chua_hidden_backend")
    if os.name == "nt" and exe.suffix.lower() != ".exe":
        exe = exe.with_suffix(".exe")
    exe.parent.mkdir(parents=True, exist_ok=True)

    backend["source_c"] = str(source_c.resolve())
    backend["exe"] = str(exe)
    cfg["runtime_dir"] = str(runtime_dir.resolve())
    return cfg


def save_default_config(path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    cfg_path = resolve_config_path(path)
    cfg = prepare_runtime_paths(copy.deepcopy(DEFAULT_CONFIG))
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict:
    cfg_path = resolve_config_path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return prepare_runtime_paths(json.load(f))


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "si"}


def env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def env_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def env_string(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def env_float_list(name: str, default: List[float] | None = None) -> List[float] | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def env_target_seed(name: str, default: Dict[str, float] | None = None) -> Dict[str, float] | None:
    values = env_float_list(name, None)
    if values is None:
        return default
    if len(values) != 3:
        raise ValueError(f"{name} debe tener exactamente 3 componentes: x,y,z.")
    return {"x": values[0], "y": values[1], "z": values[2]}


def get_summary_q(summary: Dict) -> float | None:
    for key in ("frac_order", "q"):
        if key in summary:
            return float(summary[key])
    contract = summary.get("runtime_contract", {})
    if isinstance(contract, dict) and "q" in contract:
        return float(contract["q"])
    return None


def sync_integration_from_contract(cfg: Dict, summary: Dict) -> None:
    contract = summary.get("runtime_contract", {})
    if not isinstance(contract, dict):
        return
    integ = cfg.setdefault("integration", {})
    h = contract.get("h")
    lm = contract.get("Lm")
    t_transient = contract.get("t_transient")
    t_keep = contract.get("t_keep")
    if h is not None:
        integ["h"] = float(h)
    if lm is not None:
        integ["Lm"] = float(lm)
    if t_transient is not None:
        t_transient = float(t_transient)
        integ["TBURN_REF"] = t_transient
        integ["TBURN_TEST"] = t_transient
        if t_keep is not None:
            t_total = t_transient + float(t_keep)
            integ["TMAX_REF"] = t_total
            integ["TMAX_TEST"] = t_total


def maybe_override_target_from_pipeline(cfg: Dict) -> Dict:
    safety = cfg.setdefault("safety", {})
    if not safety.get("sync_from_pipeline_summary", True):
        cfg.setdefault("diagnostics", {})["pipeline_summary_sync"] = "disabled"
        return cfg

    path = cfg["files"].get("summary_from_pipeline")
    if not path:
        return cfg
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        applied = []
        if "final_state_eps1" in s and len(s["final_state_eps1"]) == 3:
            cfg["target_seed"] = {
                "x": float(s["final_state_eps1"][0]),
                "y": float(s["final_state_eps1"][1]),
                "z": float(s["final_state_eps1"][2]),
            }
            applied.append("target_seed")
        summary_q = get_summary_q(s)
        if summary_q is not None:
            old_q = float(cfg.get("frac_order", summary_q))
            cfg["frac_order"] = summary_q
            if abs(old_q - summary_q) > 1e-12:
                applied.append(f"frac_order {old_q:g}->{summary_q:g}")
            else:
                applied.append("frac_order")
        sync_integration_from_contract(cfg, s)
        if "runtime_contract" in s:
            applied.append("integration_contract")
        if "params" in s:
            params = s["params"]
            needed = {"alpha_chua", "beta", "gamma_chua", "m0", "m1"}
            alt_needed = {"alpha", "beta", "gamma", "m0", "m1"}
            summary_model = s.get("model", {})
            if isinstance(summary_model, dict):
                summary_model = summary_model.get("kind", "piecewise")
            common_extra = {
                "model": str(params.get("model", summary_model or "piecewise")),
                "a1": float(params.get("a1", 0.4)),
                "a2": float(params.get("a2", -1.5585)),
                "rho": float(params.get("rho", 1.0)),
            }
            if needed.issubset(params.keys()):
                cfg["params"] = {
                    "alpha_chua": float(params["alpha_chua"]),
                    "beta": float(params["beta"]),
                    "gamma_chua": float(params["gamma_chua"]),
                    "m0": float(params["m0"]),
                    "m1": float(params["m1"]),
                    **common_extra,
                }
                applied.append("params")
            elif alt_needed.issubset(params.keys()):
                cfg["params"] = {
                    "alpha_chua": float(params["alpha"]),
                    "beta": float(params["beta"]),
                    "gamma_chua": float(params["gamma"]),
                    "m0": float(params["m0"]),
                    "m1": float(params["m1"]),
                    **common_extra,
                }
                applied.append("params")
        cfg.setdefault("diagnostics", {})["pipeline_summary_sync"] = {
            "path": str(path),
            "applied": applied,
        }
    else:
        cfg.setdefault("diagnostics", {})["pipeline_summary_sync"] = {
            "path": str(path),
            "applied": [],
            "warning": "summary_from_pipeline no existe; se usa la semilla de la configuracion",
        }
    return cfg


def chua_equilibria(p: ChuaParams) -> Dict[str, np.ndarray]:
    E0 = np.array([0.0, 0.0, 0.0], dtype=float)
    A = p.m0 - p.m1
    eqs = {"E0": E0}
    if str(getattr(p, "model", "piecewise")).strip().lower() in {"arctan", "atan", "smooth"}:
        coeff = 1.0 + p.a1 - p.gamma_chua / (p.beta + p.gamma_chua)

        def f(x: float) -> float:
            return coeff * x + p.a2 * np.arctan(p.rho * x)

        xs = np.linspace(1e-8, 100.0, 20000)
        vals = np.array([f(float(x)) for x in xs], dtype=float)
        xp = None
        for i in range(len(xs) - 1):
            if vals[i] * vals[i + 1] < 0.0:
                lo, hi = float(xs[i]), float(xs[i + 1])
                flo = f(lo)
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    fm = f(mid)
                    if abs(fm) < 1e-14:
                        lo = hi = mid
                        break
                    if flo * fm <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                        flo = fm
                xp = 0.5 * (lo + hi)
                break
        if xp is not None:
            yp = p.gamma_chua / (p.beta + p.gamma_chua) * xp
            zp = -p.beta / (p.beta + p.gamma_chua) * xp
            eqs["E+"] = np.array([xp, yp, zp], dtype=float)
            eqs["E-"] = -eqs["E+"]
        return eqs

    den = (p.beta + p.gamma_chua) * p.m1 + p.beta
    if abs(den) < 1e-14:
        return eqs
    xp = -((p.beta + p.gamma_chua) * A) / den
    if abs(xp) > 1.0:
        fp = p.m1 * xp + A
        eqs["E+"] = np.array([xp, xp + fp, fp], dtype=float)
        eqs["E-"] = -eqs["E+"]
    return eqs


def apply_runtime_profile(cfg: Dict, profile: str | None) -> Dict:
    safety = cfg.setdefault("safety", {})
    selected = (profile or os.environ.get("HIDDEN_VERIFY_PROFILE") or safety.get("profile") or "safe").strip().lower()
    safety["profile"] = selected
    if selected in {"config", "none"}:
        return cfg

    integ = cfg.setdefault("integration", {})
    thr = cfg.setdefault("thresholds", {})
    sampling = cfg.setdefault("sampling", {})

    if selected == "smoke":
        integ["Lm"] = min(float(integ.get("Lm", 3.0)), 3.0)
        integ["TMAX_REF"] = min(float(integ.get("TMAX_REF", 80.0)), 80.0)
        integ["TMAX_TEST"] = min(float(integ.get("TMAX_TEST", 80.0)), 80.0)
        integ["TBURN_REF"] = min(float(integ.get("TBURN_REF", 40.0)), 40.0)
        integ["TBURN_TEST"] = min(float(integ.get("TBURN_TEST", 40.0)), 40.0)
        thr["MIN_SEC_MATCH"] = min(int(thr.get("MIN_SEC_MATCH", 6)), 6)
        thr["TEST_MAX_SEC"] = min(int(thr.get("TEST_MAX_SEC", 24)), 24)
        sampling["RADII"] = [float(r) for r in sampling.get("RADII", [])[:3] or [1e-5, 1e-4, 1e-3]]
        sampling["NSAMPLES_PER_RADIUS"] = min(int(sampling.get("NSAMPLES_PER_RADIUS", 2)), 2)
        safety["timeout_sec"] = min(int(safety.get("timeout_sec", 300)), 300)
        safety["max_threads"] = min(int(safety.get("max_threads", 1)), 1)
    elif selected == "safe":
        integ["Lm"] = min(float(integ.get("Lm", 5.0)), 5.0)
        integ["TMAX_REF"] = min(float(integ.get("TMAX_REF", 140.0)), 140.0)
        integ["TMAX_TEST"] = min(float(integ.get("TMAX_TEST", 140.0)), 140.0)
        integ["TBURN_REF"] = min(float(integ.get("TBURN_REF", 70.0)), 70.0)
        integ["TBURN_TEST"] = min(float(integ.get("TBURN_TEST", 70.0)), 70.0)
        thr["MIN_SEC_MATCH"] = min(int(thr.get("MIN_SEC_MATCH", 12)), 12)
        thr["TEST_MAX_SEC"] = min(int(thr.get("TEST_MAX_SEC", 60)), 60)
        sampling["NSAMPLES_PER_RADIUS"] = min(int(sampling.get("NSAMPLES_PER_RADIUS", 6)), 6)
        safety["timeout_sec"] = min(int(safety.get("timeout_sec", 1800)), 1800)
        safety["max_threads"] = min(int(safety.get("max_threads", 2)), 2)
    elif selected == "balanced":
        safety["timeout_sec"] = int(safety.get("timeout_sec", 3600))
        safety["max_threads"] = min(int(safety.get("max_threads", 4)), 4)
    elif selected == "full":
        safety["timeout_sec"] = int(safety.get("timeout_sec", 7200))
        safety["max_threads"] = min(int(safety.get("max_threads", 4)), 4)
    else:
        raise ValueError(f"Perfil no reconocido: {selected}. Usa smoke, safe, balanced, full o config.")

    return cfg


def apply_env_overrides(cfg: Dict) -> Dict:
    integ = cfg.setdefault("integration", {})
    sampling = cfg.setdefault("sampling", {})
    safety = cfg.setdefault("safety", {})
    files = cfg.setdefault("files", {})

    overrides = {
        "HIDDEN_VERIFY_FRAC_ORDER": ("root", "frac_order", env_float),
        "HIDDEN_VERIFY_TMAX_REF": ("integration", "TMAX_REF", env_float),
        "HIDDEN_VERIFY_TMAX_TEST": ("integration", "TMAX_TEST", env_float),
        "HIDDEN_VERIFY_TBURN_REF": ("integration", "TBURN_REF", env_float),
        "HIDDEN_VERIFY_TBURN_TEST": ("integration", "TBURN_TEST", env_float),
        "HIDDEN_VERIFY_H": ("integration", "h", env_float),
        "HIDDEN_VERIFY_LM": ("integration", "Lm", env_float),
        "HIDDEN_VERIFY_NSAMPLES": ("sampling", "NSAMPLES_PER_RADIUS", env_int),
        "HIDDEN_VERIFY_RADII": ("sampling", "RADII", env_float_list),
        "HIDDEN_VERIFY_EQ_FILTER": ("sampling", "EQ_FILTER", env_string),
        "HIDDEN_VERIFY_THREADS": ("safety", "max_threads", env_int),
        "HIDDEN_VERIFY_TIMEOUT_SEC": ("safety", "timeout_sec", env_int),
        "HIDDEN_VERIFY_MAX_WORK_UNITS": ("safety", "max_work_units", env_int),
        "HIDDEN_VERIFY_SUMMARY_FROM_PIPELINE": ("files", "summary_from_pipeline", env_string),
    }
    targets = {"root": cfg, "integration": integ, "sampling": sampling, "safety": safety, "files": files}
    for env_name, (section, key, parser) in overrides.items():
        value = parser(env_name, None)
        if value is not None:
            targets[section][key] = value
    target_seed = env_target_seed("HIDDEN_VERIFY_TARGET_SEED", None)
    if target_seed is not None:
        cfg["target_seed"] = target_seed
    if env_flag("HIDDEN_VERIFY_NO_PIPELINE_SYNC", False):
        safety["sync_from_pipeline_summary"] = False
    return cfg


def estimate_backend_work(cfg: Dict, eq_count: int | None = None) -> Dict:
    integ = cfg["integration"]
    sampling = cfg["sampling"]
    h = float(integ["h"])
    lm = float(integ["Lm"])
    if h <= 0.0:
        raise ValueError("h debe ser positivo.")
    if lm <= 0.0:
        raise ValueError("Lm debe ser positivo.")
    n_ref = int(math.ceil(float(integ["TMAX_REF"]) / h))
    n_test = int(math.ceil(float(integ["TMAX_TEST"]) / h))
    nu = max(1, int(math.ceil(lm / h)))
    if eq_count is None:
        eq_count = 3
    n_radii = len(sampling.get("RADII", []))
    nsamples = int(sampling["NSAMPLES_PER_RADIUS"])
    test_trajectories = int(eq_count) * n_radii * nsamples
    # Each EFORK step evaluates three truncated-memory convolutions.
    work_units = 3 * (n_ref * nu + test_trajectories * n_test * nu)
    return {
        "eq_count": int(eq_count),
        "n_radii": int(n_radii),
        "nsamples_per_radius": int(nsamples),
        "test_trajectories": int(test_trajectories),
        "steps_ref": int(n_ref),
        "steps_test": int(n_test),
        "memory_window_steps": int(nu),
        "work_units": int(work_units),
    }


def selected_equilibrium_names(eqs: Dict[str, np.ndarray], sampling: Dict) -> List[str]:
    eq_filter = str(sampling.get("EQ_FILTER", "") or "").strip()
    if not eq_filter or eq_filter.lower() == "all":
        return list(eqs.keys())
    requested = {item.strip() for item in eq_filter.split(",") if item.strip()}
    selected = [name for name in eqs.keys() if name in requested]
    missing = sorted(requested - set(selected))
    if missing:
        raise ValueError(f"EQ_FILTER contiene equilibrios no validos: {', '.join(missing)}")
    if not selected:
        raise ValueError("EQ_FILTER no dejo ningun equilibrio activo.")
    return selected


def validate_config(cfg: Dict, estimate: Dict, allow_heavy: bool) -> None:
    integ = cfg["integration"]
    thr = cfg["thresholds"]
    sampling = cfg["sampling"]
    safety = cfg.setdefault("safety", {})

    for key in ("TMAX_REF", "TMAX_TEST", "TBURN_REF", "TBURN_TEST", "h", "Lm"):
        if float(integ[key]) <= 0.0:
            raise ValueError(f"integration.{key} debe ser positivo.")
    if float(integ["TBURN_REF"]) >= float(integ["TMAX_REF"]):
        raise ValueError("TBURN_REF debe ser menor que TMAX_REF.")
    if float(integ["TBURN_TEST"]) >= float(integ["TMAX_TEST"]):
        raise ValueError("TBURN_TEST debe ser menor que TMAX_TEST.")
    if int(thr["MIN_SEC_MATCH"]) > int(thr["TEST_MAX_SEC"]):
        raise ValueError("MIN_SEC_MATCH no puede ser mayor que TEST_MAX_SEC.")
    if int(sampling["NSAMPLES_PER_RADIUS"]) <= 0:
        raise ValueError("NSAMPLES_PER_RADIUS debe ser positivo.")
    if not sampling.get("RADII"):
        raise ValueError("sampling.RADII no puede estar vacio.")

    max_work = int(safety.get("max_work_units", DEFAULT_MAX_WORK_UNITS))
    if estimate["work_units"] > max_work and not allow_heavy:
        msg = (
            "La corrida fue bloqueada antes de arrancar el backend C porque es pesada.\n"
            f"Trabajo estimado: {estimate['work_units']:,} unidades de memoria fraccionaria; "
            f"limite: {max_work:,}.\n"
            "Usa --profile smoke/safe para una prueba, reduce TMAX/Lm/muestras, "
            "o ejecuta con --force si de verdad quieres esa carga."
        )
        raise RuntimeError(msg)


def print_run_plan(cfg: Dict, estimate: Dict) -> None:
    safety = cfg.get("safety", {})
    integ = cfg["integration"]
    sampling = cfg["sampling"]
    print("Contrato efectivo de verificacion:", flush=True)
    print(f"- perfil: {safety.get('profile', 'config')}", flush=True)
    print(f"- q={float(cfg['frac_order']):g}, h={float(integ['h']):g}, Lm={float(integ['Lm']):g}", flush=True)
    print(
        f"- TMAX_REF/TEST={float(integ['TMAX_REF']):g}/{float(integ['TMAX_TEST']):g}, "
        f"TBURN_REF/TEST={float(integ['TBURN_REF']):g}/{float(integ['TBURN_TEST']):g}",
        flush=True,
    )
    print(
        f"- radios={len(sampling.get('RADII', []))}, muestras/radio={int(sampling['NSAMPLES_PER_RADIUS'])}, "
        f"trayectorias de prueba={estimate['test_trajectories']}",
        flush=True,
    )
    eq_filter = str(sampling.get("EQ_FILTER", "") or "").strip()
    if eq_filter:
        print(f"- filtro de equilibrios: {eq_filter}", flush=True)
    print(
        f"- ventana={estimate['memory_window_steps']} pasos, trabajo estimado={estimate['work_units']:,}",
        flush=True,
    )
    print(f"- hilos OpenMP maximos: {int(safety.get('max_threads', 1))}", flush=True)
    sync = cfg.get("diagnostics", {}).get("pipeline_summary_sync")
    if sync:
        print(f"- sync pipeline: {sync}", flush=True)


def normalize_backend_paths(cfg: dict):
    runtime_dir = cfg.get("runtime_dir")
    prepare_runtime_paths(cfg, Path(runtime_dir) if runtime_dir else None)
    backend = cfg["backend"]

    src = Path(backend["source_c"])
    exe = Path(backend["exe"])

    # En Windows, forzar extensión .exe
    if os.name == "nt" and exe.suffix.lower() != ".exe":
        exe = exe.with_suffix(".exe")

    backend["source_c"] = str(src)
    backend["exe"] = str(exe)

    return src, exe


def compile_backend(cfg: dict) -> None:
    backend = cfg["backend"]
    src, exe = normalize_backend_paths(cfg)

    if not backend.get("compile", True) and exe.exists():
        return

    if not src.exists():
        raise FileNotFoundError(f"No se encontró el backend en C: {src}")

    cmd = ["gcc", "-O3", str(src), "-lm", "-o", str(exe)]
    if backend.get("openmp", True):
        cmd.insert(2, "-fopenmp")

    print("Compilando backend en C...")
    try:
        subprocess.run(cmd, check=True, capture_output=bool(backend.get("openmp", True)), text=True)
    except subprocess.CalledProcessError:
        if "-fopenmp" not in cmd:
            raise
        fallback_cmd = [part for part in cmd if part != "-fopenmp"]
        print("OpenMP no está disponible en este compilador; reintentando backend C sin -fopenmp.")
        subprocess.run(fallback_cmd, check=True)


def build_backend_command(cfg: dict):
    normalize_backend_paths(cfg)

    p = cfg["params"]
    integ = cfg["integration"]
    thr = cfg["thresholds"]
    samp = cfg["sampling"]
    files = cfg["files"]

    backend_exe = str(Path(cfg["backend"]["exe"]).resolve())
    frac_order = float(cfg["frac_order"])
    if not (0.0 < frac_order <= 1.0):
        raise ValueError("El orden fraccionario q debe cumplir 0 < q <= 1.")

    # Mapeo al backend C:
    # alpha_chua -> --alpha_chua, gamma_chua -> --gamma_chua,
    # frac_order/q -> --frac_order, y model -> --model.
    radii_txt = ",".join(f"{float(r):.17g}" for r in samp["RADII"])
    target_seed_txt = ",".join(
        f"{float(cfg['target_seed'][k]):.17g}" for k in ("x", "y", "z")
    )

    cmd = [
        backend_exe,
        "--alpha_chua", str(p["alpha_chua"]),
        "--beta", str(p["beta"]),
        "--gamma_chua", str(p["gamma_chua"]),
        "--m0", str(p["m0"]),
        "--m1", str(p["m1"]),
        "--model", str(p.get("model", "piecewise")),
        "--a1", str(p.get("a1", 0.4)),
        "--a2", str(p.get("a2", -1.5585)),
        "--rho", str(p.get("rho", 1.0)),
        "--frac_order", str(frac_order),
        "--target_seed", target_seed_txt,
        "--h", str(integ["h"]),
        "--Lm", str(integ["Lm"]),
        "--TMAX_REF", str(integ["TMAX_REF"]),
        "--TMAX_TEST", str(integ["TMAX_TEST"]),
        "--TBURN_REF", str(integ["TBURN_REF"]),
        "--TBURN_TEST", str(integ["TBURN_TEST"]),
        "--R_DIV", str(thr["R_DIV"]),
        "--EPS_EQ", str(thr["EPS_EQ"]),
        "--CAP_WIN", str(thr["CAP_WIN"]),
        "--SEC_TOL", str(thr["SEC_TOL"]),
        "--MIN_SEC_MATCH", str(thr["MIN_SEC_MATCH"]),
        "--TEST_MAX_SEC", str(thr["TEST_MAX_SEC"]),
        "--HIT_FRAC_REQ", str(thr["HIT_FRAC_REQ"]),
        "--radii", radii_txt,
        "--nsamples", str(samp["NSAMPLES_PER_RADIUS"]),
        "--random_seed", str(samp["random_seed"]),
        "--csv_out", files["csv_out"],
        "--ref_out", files["ref_csv_out"],
        "--summary_csv_out", files["summary_csv_out"],
    ]
    return cmd


def run_backend(cfg: dict) -> None:
    cmd = build_backend_command(cfg)
    safety = cfg.get("safety", {})
    max_threads = max(1, int(safety.get("max_threads", 1)))
    timeout = int(safety.get("timeout_sec", 0) or 0)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(max_threads)
    env["OMP_THREAD_LIMIT"] = str(max_threads)
    eq_filter = str(cfg.get("sampling", {}).get("EQ_FILTER", "") or "").strip()
    if eq_filter:
        env["HIDDEN_VERIFY_EQ_FILTER"] = eq_filter
    print("Ejecutando backend numerico en C...", flush=True)
    print(f"OMP_NUM_THREADS={max_threads}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env, timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"El backend supero el timeout de {timeout} s. "
            "Reduce el perfil o usa HIDDEN_VERIFY_TIMEOUT_SEC/--force para una corrida mas larga."
        ) from exc

def load_reference_csv(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data


def load_csv_dicts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_reference_section(ref: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(ref[:, 0], ref[:, 1], s=10, label="seccion objetivo")
    plt.xlabel("y en sección x=0")
    plt.ylabel("z en sección x=0")
    plt.title("Nube de referencia del atractor objetivo")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_probe_summary(summary_rows: List[Dict], radii: List[float], eq_names: List[str], path: str) -> None:
    class_order = ["EQ", "DIV", "TARGET", "OTHER", "UNKNOWN"]
    fig, axs = plt.subplots(len(eq_names), 1, figsize=(10, 4 * len(eq_names)), sharex=True)
    if len(eq_names) == 1:
        axs = [axs]

    for ax, eq in zip(axs, eq_names):
        sub = [r for r in summary_rows if r["equilibrium"] == eq]
        for cname in class_order:
            ys = []
            for rr in radii:
                val = 0
                for row in sub:
                    if math.isclose(float(row["radius"]), rr, rel_tol=0.0, abs_tol=1e-18):
                        val = int(row[cname])
                        break
                ys.append(val)
            ax.plot(radii, ys, marker="o", label=cname)
        ax.set_xscale("log")
        ax.set_ylabel(eq)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axs[-1].set_xlabel("radio")
    fig.suptitle("Resumen de clasificación por vecindad de equilibrio")
    fig.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build_summary_json(cfg: Dict, eqs: Dict[str, np.ndarray], ref: np.ndarray,
                       summary_rows: List[Dict], detail_rows: List[Dict]) -> Dict:
    p = ChuaParams(**cfg["params"])
    frac_order = float(cfg["frac_order"])
    if not (0.0 < frac_order <= 1.0):
        raise ValueError("El orden fraccionario q debe cumplir 0 < q <= 1.")
    total_target_hits = sum(int(r["TARGET"]) for r in summary_rows)
    hiddenness_status = "supported_by_sample" if total_target_hits == 0 else "not_supported_by_sample"
    final_msg = (
        "No se detectaron trayectorias que cayeran en el atractor objetivo desde las vecindades muestreadas de los equilibrios."
        if total_target_hits == 0 else
        f"Se detectaron {total_target_hits} trayectorias clasificadas como TARGET."
    )

    return {
        "params": asdict(p),
        "frac_order": frac_order,
        "target_seed": [
            float(cfg["target_seed"]["x"]),
            float(cfg["target_seed"]["y"]),
            float(cfg["target_seed"]["z"]),
        ],
        "equilibria": {k: v.tolist() for k, v in eqs.items()},
        "reference_points": int(ref.shape[0]),
        "total_target_hits": int(total_target_hits),
        "hiddenness_status": hiddenness_status,
        "final_message": final_msg,
        "safety": cfg.get("safety", {}),
        "diagnostics": cfg.get("diagnostics", {}),
        "summary_by_radius": [
            {
                "equilibrium": row["equilibrium"],
                "radius": float(row["radius"]),
                "EQ": int(row["EQ"]),
                "DIV": int(row["DIV"]),
                "TARGET": int(row["TARGET"]),
                "OTHER": int(row["OTHER"]),
                "UNKNOWN": int(row["UNKNOWN"]),
            }
            for row in summary_rows
        ],
        "files": cfg["files"],
        "notes": [
            "La parte numéricamente costosa corre en C: EFORK, memoria truncada, muestreo y clasificación.",
            "Python sólo coordina, grafica y arma el resumen final.",
            "La sección se usa como huella geométrica del atractor objetivo.",
            "Para una verificación más fuerte, aumenta tiempos, memoria y muestras.",
        ],
        "n_detail_rows": len(detail_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compila y ejecuta la verificacion hibrida de ocultedad para Chua fraccionario."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Ruta al JSON de configuracion.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "safe", "balanced", "full", "config", "none"],
        default=None,
        help="Perfil de costo. smoke/safe recortan tiempos y muestras para no saturar la maquina.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Muestra el contrato efectivo sin ejecutar C.")
    parser.add_argument("--write-config", action="store_true", help="Escribe una configuracion default segura y termina.")
    parser.add_argument("--force", action="store_true", help="Permite corridas por encima del limite de trabajo estimado.")
    parser.add_argument(
        "--no-pipeline-sync",
        action="store_true",
        help="No tomar semilla, q, parametros ni contrato desde summary_from_pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    if args.write_config:
        save_default_config(cfg_path)
        print(f"Se escribio {cfg_path}.", flush=True)
        return

    if not os.path.exists(cfg_path):
        save_default_config(cfg_path)
        print(f"Se creo {cfg_path}. Revisalo y vuelve a ejecutar.", flush=True)
        return

    cfg = load_config(cfg_path)
    cfg = apply_env_overrides(cfg)
    if args.no_pipeline_sync:
        cfg.setdefault("safety", {})["sync_from_pipeline_summary"] = False
    cfg = maybe_override_target_from_pipeline(cfg)
    cfg = apply_runtime_profile(cfg, args.profile)
    cfg = apply_env_overrides(cfg)
    cfg = prepare_runtime_paths(cfg)

    p = ChuaParams(**cfg["params"])
    eqs = chua_equilibria(p)
    active_eq_names = selected_equilibrium_names(eqs, cfg.get("sampling", {}))
    estimate = estimate_backend_work(cfg, eq_count=len(active_eq_names))
    cfg.setdefault("diagnostics", {})["backend_work_estimate"] = estimate
    cfg.setdefault("diagnostics", {})["active_equilibria"] = active_eq_names
    allow_heavy = args.force or env_flag("HIDDEN_VERIFY_ALLOW_HEAVY", False)
    validate_config(cfg, estimate, allow_heavy=allow_heavy)

    print(f"Directorio de salida: {cfg['runtime_dir']}", flush=True)
    print_run_plan(cfg, estimate)
    if args.dry_run:
        print("Dry-run: no se ejecuto el backend C.", flush=True)
        return

    compile_backend(cfg)
    run_backend(cfg)

    files = cfg["files"]
    ref = load_reference_csv(files["ref_csv_out"])
    detail_rows = load_csv_dicts(files["csv_out"])
    summary_rows = load_csv_dicts(files["summary_csv_out"])

    plot_reference_section(ref, files["fig_section"])
    plot_probe_summary(summary_rows, [float(r) for r in cfg["sampling"]["RADII"]], active_eq_names, files["fig_probe"])

    summary = build_summary_json(cfg, eqs, ref, summary_rows, detail_rows)
    with open(files["json_out"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Resumen final:", flush=True)
    print(summary["final_message"], flush=True)
    print("\nArchivos generados:", flush=True)
    print(f"- {files['csv_out']}", flush=True)
    print(f"- {files['ref_csv_out']}", flush=True)
    print(f"- {files['summary_csv_out']}", flush=True)
    print(f"- {files['json_out']}", flush=True)
    print(f"- {files['fig_section']}", flush=True)
    print(f"- {files['fig_probe']}", flush=True)


if __name__ == "__main__":
    main()
