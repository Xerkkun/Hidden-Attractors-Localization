"""Candidate records and loaders for final-project analyses.

Stability: stable
    :class:`CandidateRecord` and :func:`load_final_candidate_records` are the
    primary user-facing API for loading reference outputs.  Signatures are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .io import read_csv_rows, read_json
from .paths import PROJECT_ROOT


PROMOTED_SELECTION = PROJECT_ROOT / "validation" / "04_candidates" / "selected_candidates.json"


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
        Unique identifier of the form ``'branch_0_mu_4p00000_theta_0p00000'``
        or ``'lure_biased_q_0p99980_rank_0001'``.
    route : str
        Seed-generation method: ``'Machado_FDF'`` or ``'Lure_rank_0001'``.
    q : float
        Caputo fractional order used during seed search.
    robust_start : np.ndarray, shape (3,)
        State vector from the continuation run used as the robustness seed.
    seed : np.ndarray, shape (3,)
        Harmonic-balance seed state that initiated the continuation.
    mu : float or None
        Machado exponent; ``None`` for classical-DF candidates.
    theta : float or None
        Phase angle from the Machado DF solution; ``None`` if not applicable.
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
        return np.array([float("nan"), float("nan"), float("nan")], dtype=float)
    return np.asarray([_float(v) for v in value], dtype=float)


def load_lure_survivor(
    source_dir: str | Path,
    candidate_id: str,
) -> CandidateRecord:
    """Load a Lur'e continuation survivor from the final q=0.9998 run.

    Parameters
    ----------
    source_dir : str or Path
        Directory produced by the biased-Lur'e multi-parameter sweep.
        Defaults to the project-canonical output folder.
    candidate_id : str, default 'lure_biased_q_0p99980_rank_0001'
        Key matching ``candidate_id`` in ``biased_lure_candidates.csv``
        and ``continuation_survivors.csv``.

    Returns
    -------
    record : CandidateRecord
        Frozen record with seed, robust-start, and DF parameters.

    Raises
    ------
    FileNotFoundError
        If *candidate_id* is absent from the CSV files in *source_dir*.
    """

    root = Path(source_dir)
    candidates = read_csv_rows(root / "biased_lure_candidates.csv")
    survivors = {row["candidate_id"]: row for row in read_csv_rows(root / "continuation_survivors.csv")}
    for row in candidates:
        if row.get("candidate_id") != candidate_id or candidate_id not in survivors:
            continue
        surv = survivors[candidate_id]
        return CandidateRecord(
            candidate_id=candidate_id,
            route="Lure_rank_0001",
            q=_float(row.get("q"), 0.9998),
            mu=None,
            theta=None,
            A=_float(row.get("A")),
            sigma0=_float(row.get("sigma0")),
            omega=_float(row.get("omega")),
            rho_H=_float(row.get("rho_H")),
            residual_abs=_float(row.get("residual_abs")),
            seed=np.array([_float(row.get("seed_x")), _float(row.get("seed_y")), _float(row.get("seed_z"))], dtype=float),
            robust_start=np.array([_float(surv.get("final_x")), _float(surv.get("final_y")), _float(surv.get("final_z"))], dtype=float),
            source=str(root / "continuation_survivors.csv"),
        )
    raise FileNotFoundError(f"No se encontro {candidate_id} en {root}")


def load_machado_candidate(candidate_id: str, targeted_path: str | Path, corrida1_path: str | Path) -> CandidateRecord:
    """Load one Machado/FDF candidate from the final targeted verification outputs.

    Parameters
    ----------
    candidate_id : str
        Key of the form ``'branch_0_mu_4p00000_theta_0p00000'`` matching
        entries in ``machado_targeted_summary.json`` and
        ``corrida1_summary.json``.

    Returns
    -------
    record : CandidateRecord
        Frozen record with Machado exponent, phase, amplitude, and
        robust-start coordinates.

    Raises
    ------
    FileNotFoundError
        If *candidate_id* is absent from the targeted verification JSON.
    """

    targeted_path = Path(targeted_path)
    corrida1_path = Path(corrida1_path)
    targeted = read_json(targeted_path)
    corrida1 = read_json(corrida1_path)
    ref_by_id = {row["candidate_id"]: row for row in targeted.get("reference_attractor", [])}
    cand_by_id = {row["candidate_id"]: row for row in corrida1.get("candidates", [])}
    ref = ref_by_id.get(candidate_id)
    cand = cand_by_id.get(candidate_id, {})
    if ref is None:
        raise FileNotFoundError(f"No se encontro {candidate_id} en {targeted_path}")
    return CandidateRecord(
        candidate_id=candidate_id,
        route="Machado_FDF",
        q=_float(ref.get("q"), 0.9998),
        mu=_float(ref.get("mu")),
        theta=_float(ref.get("theta")),
        A=_float(cand.get("A")),
        sigma0=None,
        omega=_float(cand.get("omega")),
        rho_H=None,
        residual_abs=None,
        seed=_vec(cand.get("seed") if isinstance(cand.get("seed"), list) else None),
        robust_start=np.array([_float(ref.get("final_x")), _float(ref.get("final_y")), _float(ref.get("final_z"))], dtype=float),
        source=str(targeted_path),
    )


def _selection_path(source_dir: str | Path | None) -> Path:
    if source_dir is None:
        return PROMOTED_SELECTION
    source = Path(source_dir)
    return source if source.suffix.lower() == ".json" else source / "selected_candidates.json"


def _record_from_selected(row: Dict[str, Any], source: Path) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=str(row["candidate_id"]),
        route=str(row.get("method", row.get("route", ""))),
        q=_float(row.get("q"), 0.9998),
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
    source_dir: str | Path | None = None,
) -> List[CandidateRecord]:
    """Return the three candidates promoted from the current validated run.

    Historical outputs are not an implicit fallback.  A promoted
    ``selected_candidates.json`` must exist under the validation tree or in
    an explicitly provided current run directory.
    """

    selection = _selection_path(source_dir)
    payload = read_json(selection)
    status = payload.get("selection_status")
    if status is not None and status != "promoted_for_hiddenness":
        raise FileNotFoundError(
            f"La selección actual no está promovida para ocultedad ({status}) en {selection}"
        )
    rows = payload.get("selected_candidates", payload.get("candidates", []))
    if len(rows) < 3:
        raise FileNotFoundError(f"No hay una terna promovida de candidatos actuales en {selection}")
    return [_record_from_selected(row, selection) for row in rows[:3]]
