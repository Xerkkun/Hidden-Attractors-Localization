"""Portable records for user-supplied attractor-candidate outputs.

Stability: experimental
    Candidate schemas may gain optional metadata fields.  Loaders require an
    explicit JSON source and never inspect a repository validation tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .io import read_json


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(frozen=True)
class CandidateRecord:
    """Numerical-attractor candidate used by verification workflows.

    Attributes
    ----------
    candidate_id : str
        User-defined unique identifier.
    route : str
        Seed or workflow family recorded by the source file.
    q : float
        Caputo fractional order used during seed search.
    robust_start : np.ndarray, shape (d,)
        State vector from the continuation run used as the robustness seed.
    seed : np.ndarray, shape (d,)
        Harmonic-balance seed state that initiated the continuation.
    mu : float or None
        Optional exponent parameter recorded by the source workflow.
    theta : float or None
        Optional phase angle recorded by the source workflow.
    A : float or None
        Oscillation amplitude from the describing-function solution.
    sigma0 : float or None
        Bias offset for biased-DF candidates; ``None`` otherwise.
    omega : float or None
        Angular frequency from the DF scan.
    rho_H : float or None
        Harmonic balance residual norm.
    residual_abs : float or None
        Absolute DF equation residual.
    source : str, default ''
        Filesystem path to the CSV/JSON file this record was loaded from.

    Notes
    -----
    A record only captures seed metadata and one continuation endpoint.
    Hiddenness and robustness must be established by separate numerical
    tests using the workflow modules.
    """

    candidate_id: str
    route: str
    q: float
    robust_start: np.ndarray
    seed: np.ndarray
    mu: float | None = None
    theta: float | None = None
    A: float | None = None
    sigma0: float | None = None
    omega: float | None = None
    rho_H: float | None = None
    residual_abs: float | None = None
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "route": self.route,
            "q": self.q,
            "mu": "" if self.mu is None else self.mu,
            "theta": "" if self.theta is None else self.theta,
            "A": "" if self.A is None else self.A,
            "sigma0": "" if self.sigma0 is None else self.sigma0,
            "omega": "" if self.omega is None else self.omega,
            "rho_H": "" if self.rho_H is None else self.rho_H,
            "residual_abs": "" if self.residual_abs is None else self.residual_abs,
            "seed": self.seed.tolist(),
            "robust_start": self.robust_start.tolist(),
            "source": self.source,
        }


def _vec(value: Sequence[Any] | None) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    return np.asarray([_float(v) for v in value], dtype=float)


def _selection_path(source_dir: str | Path) -> Path:
    source = Path(source_dir)
    return source if source.suffix.lower() == ".json" else source / "selected_candidates.json"


def _record_from_selected(row: Dict[str, Any], source: Path) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=str(row["candidate_id"]),
        route=str(row.get("method", row.get("route", ""))),
        q=_float(row.get("q")),
        robust_start=_vec(row.get("robust_start")),
        seed=_vec(row.get("seed")),
        mu=None if row.get("mu", "") in {"", None} else _float(row.get("mu")),
        theta=None if row.get("theta", "") in {"", None} else _float(row.get("theta")),
        A=None if row.get("A", "") in {"", None} else _float(row.get("A")),
        sigma0=None if row.get("sigma0", "") in {"", None} else _float(row.get("sigma0")),
        omega=None if row.get("omega", "") in {"", None} else _float(row.get("omega")),
        rho_H=None if row.get("rho_H", "") in {"", None} else _float(row.get("rho_H")),
        residual_abs=None if row.get("residual_abs", "") in {"", None} else _float(row.get("residual_abs")),
        source=str(source),
    )


def load_final_candidate_records(
    source_dir: str | Path,
) -> List[CandidateRecord]:
    """Load candidate records from an explicitly supplied JSON file or folder.

    No checkout-relative fallback is used, so this function behaves identically
    in a source tree and an installed wheel.
    """

    selection = _selection_path(source_dir)
    payload = read_json(selection)
    rows = payload.get("selected_candidates", payload.get("candidates", []))
    if not rows:
        raise FileNotFoundError(f"No candidate records were found in {selection}")
    return [_record_from_selected(row, selection) for row in rows]
