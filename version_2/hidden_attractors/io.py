"""Filesystem and serialization helpers for reproducible numerical runs."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a timestamp for non-overwriting output folders."""

    return time.strftime(fmt)


def safe_name(text: str) -> str:
    """Return a filesystem-safe identifier while preserving readability."""

    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))


def json_safe(obj: Any) -> Any:
    """Convert NumPy/scalar objects into values accepted by ``json.dumps``."""

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


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Write JSON metadata for a run, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return ";".join(str(float(x)) if np.issubdtype(value.dtype, np.number) else str(x) for x in value.ravel())
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, complex):
        return f"{value.real:.16g}{value.imag:+.16g}j"
    return value


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    """Write tabular run artifacts with stable field ordering."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: List[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


def append_csv(path: str | Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    """Append one row to a CSV artifact, writing the header if needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read a CSV artifact into dictionaries; return an empty list if missing."""

    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_trajectory_csv(path: str | Path, columns: Sequence[str] = ("t", "x", "y", "z")) -> np.ndarray:
    """Load a trajectory CSV into the package convention ``t,x,y,z``.

    The project's generated trajectory files normally use a header with the
    columns ``t,x,y,z``. Headerless numeric CSV files are also accepted when
    they already follow that column order.
    """

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)

    with target.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line:
        raise ValueError(f"trajectory CSV is empty: {target}")
    has_header = any(ch.isalpha() or ch == "_" for ch in first_line)
    if has_header:
        data = np.genfromtxt(target, delimiter=",", names=True, dtype=float)
        names = data.dtype.names or ()
        missing = [name for name in columns if name not in names]
        if missing:
            raise ValueError(f"trajectory CSV is missing columns: {missing}")
        return np.column_stack([np.asarray(data[name], dtype=float) for name in columns])

    raw = np.genfromtxt(target, delimiter=",", dtype=float)
    X = np.asarray(raw, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < len(columns):
        raise ValueError(f"trajectory CSV must contain at least {len(columns)} columns")
    return X[:, : len(columns)]
