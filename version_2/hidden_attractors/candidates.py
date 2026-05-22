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
from .paths import OUTPUTS, PROJECT_ROOT


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
    source_dir: str | Path = OUTPUTS / "lure_biased_multiparam_q09998_20260515_195444",
    candidate_id: str = "lure_biased_q_0p99980_rank_0001",
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


def load_machado_candidate(candidate_id: str) -> CandidateRecord:
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

    targeted_path = OUTPUTS / "extended_search" / "machado_targeted_verification_lm10_20260515_182252" / "machado_targeted_summary.json"
    corrida1_path = OUTPUTS / "extended_search" / "corrida1" / "corrida1_summary.json"
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


def load_final_candidate_records(
    source_dir: str | Path = OUTPUTS / "lure_biased_multiparam_q09998_20260515_195444",
) -> List[CandidateRecord]:
    """Return the three final candidates used in comparative analyses.

    Loads two Machado/FDF candidates and one biased-Lur'e survivor from the
    project canonical output directories.

    Parameters
    ----------
    source_dir : str or Path
        Root directory of the biased-Lur'e multi-parameter sweep.
        Defaults to the project-canonical folder.

    Returns
    -------
    records : list[CandidateRecord]
        Three :class:`CandidateRecord` objects in the following order:

        1. Machado branch 0, μ=4.0, θ=0.
        2. Machado branch 0, μ=2.0, θ≈3.927.
        3. Lur'e biased survivor rank 0001.

    Examples
    --------
    >>> from hidden_attractors.candidates import load_final_candidate_records
    >>> records = load_final_candidate_records()  # doctest: +SKIP
    >>> len(records)
    3
    """

    return [
        load_machado_candidate("branch_0_mu_4p00000_theta_0p00000"),
        load_machado_candidate("branch_0_mu_2p00000_theta_3p92699"),
        load_lure_survivor(source_dir, "lure_biased_q_0p99980_rank_0001"),
    ]
