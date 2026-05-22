"""Filesystem and serialization helpers for reproducible numerical runs.

Stability: stable
    JSON/CSV read-write helpers and :func:`load_trajectory_csv`.  Signatures
    are fixed; new helpers may be added without breaking existing calls.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a timestamp string for non-overwriting output folder names.

    Parameters
    ----------
    fmt : str, default '%Y%m%d_%H%M%S'
        :func:`time.strftime` format string.

    Returns
    -------
    stamp : str
        Formatted local time, e.g. ``'20260522_143000'``.

    Examples
    --------
    >>> from hidden_attractors.io import timestamp
    >>> len(timestamp())  # default format has 15 chars
    15
    """

    return time.strftime(fmt)


def safe_name(text: str) -> str:
    """Return a filesystem-safe version of *text* with readability preserved.

    Replaces any character that is not alphanumeric, ``'_'``, or ``'-'``
    with an underscore.

    Parameters
    ----------
    text : str
        Arbitrary string (e.g. a candidate ID or parameter label).

    Returns
    -------
    name : str
        String containing only ``[A-Za-z0-9_-]``.

    Examples
    --------
    >>> from hidden_attractors.io import safe_name
    >>> safe_name('q=0.9998 / run #3')
    'q_0_9998___run__3'
    """

    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))


def json_safe(obj: Any) -> Any:
    """Recursively convert *obj* to types accepted by :func:`json.dumps`.

    Handles NumPy scalars, arrays, complex numbers, dicts, lists, and tuples.
    All other types are returned unchanged (and will raise at serialisation
    time if they are not JSON-serialisable).

    Parameters
    ----------
    obj : Any
        Python object to convert.

    Returns
    -------
    safe : Any
        JSON-serialisable representation of *obj*.
        - ``np.ndarray`` → ``list``
        - ``np.generic`` → Python scalar via ``.item()``
        - ``complex`` → ``[real, imag]``
        - ``dict`` / ``list`` / ``tuple`` → recursively processed.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.io import json_safe
    >>> json_safe(np.array([1.0, 2.0]))
    [1.0, 2.0]
    >>> json_safe({'a': np.float32(3.14)})  # doctest: +ELLIPSIS
    {'a': 3.14...}
    """

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
    """Write *data* as JSON, creating parent directories as needed.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Parent directories are created if absent.
    data : dict[str, Any]
        Mapping to serialise.  All values must be passable through
        :func:`json_safe`.

    Returns
    -------
    None

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from hidden_attractors.io import write_json, read_json
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / 'meta.json'
    ...     write_json(p, {'q': 0.9998})
    ...     read_json(p)['q']
    0.9998
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    """Read a JSON object from *path*.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    data : dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """

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
    """Write a sequence of row dicts to a CSV file with stable field ordering.

    Parameters
    ----------
    path : str or Path
        Destination CSV path.  Parent directories are created if absent.
    rows : sequence of dict[str, Any]
        Rows to write; each dict maps field names to values.
    fields : sequence of str or None, default None
        Column order.  If ``None``, fields are inferred from row keys in
        insertion order (union across all rows).

    Returns
    -------
    None

    Notes
    -----
    NumPy arrays, scalars, lists, and complex numbers are converted via
    :func:`_csv_value` before writing.
    """

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
    """Append one row to a CSV file, writing the header if the file is new.

    Parameters
    ----------
    path : str or Path
        CSV file path.  Created with header if absent.
    row : dict[str, Any]
        Single row to append.
    fields : sequence of str
        Column names; also controls the header written on first creation.

    Returns
    -------
    None
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of string dicts.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    rows : list[dict[str, str]]
        One dict per data row; all values are strings as returned by
        :class:`csv.DictReader`.  Returns ``[]`` if *path* does not exist.
    """

    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_trajectory_csv(path: str | Path, columns: Sequence[str] = ("t", "x", "y", "z")) -> np.ndarray:
    """Load a trajectory CSV into the package-standard column layout.

    Accepts files with a ``t,x,y,z`` header or headerless numeric CSVs
    already ordered as ``(t, x, y, z)``.

    Parameters
    ----------
    path : str or Path
        Path to the trajectory CSV file.
    columns : sequence of str, default ('t', 'x', 'y', 'z')
        Column names to extract, in order.  Used for header-based lookup.

    Returns
    -------
    traj : np.ndarray, shape (N, len(columns))
        Trajectory array with the requested columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is empty, missing required header columns, or has
        fewer columns than requested when headerless.

    Examples
    --------
    >>> import numpy as np, tempfile, pathlib
    >>> from hidden_attractors.io import write_csv, load_trajectory_csv
    >>> rows = [{'t': i*0.01, 'x': 0.0, 'y': 0.0, 'z': 0.0} for i in range(5)]
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / 'traj.csv'
    ...     write_csv(p, rows)
    ...     arr = load_trajectory_csv(p)
    ...     arr.shape
    (5, 4)
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
