from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - only used without PyYAML
        raise RuntimeError(
            f"No se pudo leer {path}. Instala PyYAML o usa JSON valido dentro del archivo .yaml."
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("La configuracion debe ser un diccionario.")
    return data


def timestamped_output_dir(cfg: Dict[str, Any]) -> Path:
    root = Path(cfg.get("outputs", {}).get("root", "outputs/extended_search"))
    if bool(cfg.get("outputs", {}).get("timestamped", True)):
        root = root / time.strftime("%Y%m%d_%H%M%S")
    (root / "plots").mkdir(parents=True, exist_ok=True)
    return root


def chua_ic_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg["params"]
    return {
        "model": cfg.get("model", "piecewise"),
        "alpha": np.float64(p["alpha_chua"]),
        "beta": np.float64(p["beta"]),
        "gamma": np.float64(p["gamma_chua"]),
        "m0": np.float64(p["m0"]),
        "m1": np.float64(p["m1"]),
        "a1": np.float64(p.get("a1", 0.4)),
        "a2": np.float64(p.get("a2", -1.5585)),
        "rho": np.float64(p.get("rho", 1.0)),
    }


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return ";".join(str(float(x)) if np.issubdtype(arr.dtype, np.number) else str(x) for x in arr.ravel())
    if isinstance(value, complex):
        return f"{value.real:.16g}{value.imag:+.16g}j"
    return value


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def fft_peak_and_entropy(traj: np.ndarray, h: float, component: int = 0) -> Dict[str, float]:
    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[0] < 8:
        return {"fft_peak": float("nan"), "psd_entropy": float("nan")}
    data = X[:, 1 + int(component)] if X.shape[1] >= 4 else X[:, int(component)]
    data = data - np.mean(data)
    n = int(data.size)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(data * win)) ** 2
    freq = np.fft.rfftfreq(n, d=float(h))
    if spec.size <= 1 or not np.any(spec[1:] > 0.0):
        return {"fft_peak": 0.0, "psd_entropy": 0.0}
    k = 1 + int(np.argmax(spec[1:]))
    prob = spec[1:] / max(float(np.sum(spec[1:])), 1e-300)
    entropy = -float(np.sum(prob * np.log(prob + 1e-300))) / max(math.log(prob.size), 1e-300)
    return {"fft_peak": float(freq[k]), "psd_entropy": entropy}


def min_distance_to_points(points: np.ndarray, refs: Iterable[np.ndarray]) -> float:
    refs_arr = [np.asarray(r, dtype=float) for r in refs if np.all(np.isfinite(r))]
    if not refs_arr:
        return float("nan")
    P = np.asarray(points, dtype=float)
    if P.ndim == 1:
        return float(min(np.linalg.norm(P - r) for r in refs_arr))
    return float(min(np.min(np.linalg.norm(P - r.reshape(1, 3), axis=1)) for r in refs_arr))


def trajectory_ranges(traj: np.ndarray) -> Dict[str, float]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else X
    if states.size == 0:
        return {"range_x": float("nan"), "range_y": float("nan"), "range_z": float("nan")}
    ranges = np.ptp(states, axis=0)
    return {"range_x": float(ranges[0]), "range_y": float(ranges[1]), "range_z": float(ranges[2])}


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, complex):
        return [float(obj.real), float(obj.imag)]
    return obj
