"""Operational hiddenness verification contract and validation states."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Sequence
import numpy as np


class HiddennessVerificationStatus(str, Enum):
    """Enumeration of strict hiddenness verification states."""

    NOT_RUN = "NOT_RUN"
    INCOMPLETE_PROTOCOL = "INCOMPLETE_PROTOCOL"
    SEED_NOT_AVAILABLE = "SEED_NOT_AVAILABLE"
    CANDIDATE_NOT_AVAILABLE = "CANDIDATE_NOT_AVAILABLE"
    SELF_EXCITED_CONTACT_DETECTED = "SELF_EXCITED_CONTACT_DETECTED"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    HIDDEN_COMPATIBLE = "HIDDEN_COMPATIBLE"
    HIDDEN_VERIFIED = "HIDDEN_VERIFIED"


def is_radius_close(r1: float, r2: float, rtol: float = 1e-12, atol: float = 1e-15) -> bool:
    """Tolerant floating-point comparison for radii."""
    return abs(r1 - r2) <= (atol + rtol * abs(r2))


def verify_hiddenness_contract(
    equilibria: Dict[str, np.ndarray],
    sphere_summary_records: List[Dict[str, Any]],
    probe_runs: List[Dict[str, Any]],
    required_radii: Sequence[float],
    require_all_equilibria: bool = True,
    allow_numerical_failures: bool = False,
    require_candidate_attractor: bool = True,
    seed_reached_attractor: bool = True,
    min_ref_tail_points: int = 1000,
    min_probe_tail_points: int = 200,
    ref_tail_size: int = 1000,
    target_match_metric: str = "nn_percentile",
    target_match_tol: float = 0.5,
    target_match_nn_percentile: float = 90.0,
) -> Dict[str, Any]:
    """Verify operational hiddenness condition: B(A) ∩ U_epsilon(X_i*) = empty.

    Parameters
    ----------
    equilibria : Dict[str, np.ndarray]
        All equilibria calculated for the system.
    sphere_summary_records : List[Dict[str, Any]]
        Summary records for tested (equilibrium, radius) pairs.
    probe_runs : List[Dict[str, Any]]
        List of all integrated probe runs.
    required_radii : Sequence[float]
        The set of radii required to declare hidden_verified.
    require_all_equilibria : bool, default True
        If True, vecindades of ALL equilibria must be tested.
    allow_numerical_failures : bool, default False
        If False, any numerical failure in probe runs blocks hidden_verified.
    require_candidate_attractor : bool, default True
        If True, a candidate attractor must have been reached by the seed.
    seed_reached_attractor : bool, default True
        Whether the final attractor simulation succeeded.
    min_ref_tail_points : int, default 1000
        Minimum points required in the reference attractor tail.
    min_probe_tail_points : int, default 200
        Minimum points required in the persistent probe tails.
    ref_tail_size : int, default 1000
        Actual size of the reference attractor tail.
    target_match_metric : str, default "nn_percentile"
        Metric used to evaluate matches against the target attractor.
    target_match_tol : float, default 0.5
    target_match_nn_percentile : float, default 90.0

    Returns
    -------
    report : Dict[str, Any]
        Dictionary matching the formal X_i* verification contract schema.
    """
    failed_reqs = []

    # 1. Base Candidate Availability Check
    if require_candidate_attractor and not seed_reached_attractor:
        failed_reqs.append("Seed did not reach the target attractor candidate.")
        return {
            "hiddenness_status": HiddennessVerificationStatus.CANDIDATE_NOT_AVAILABLE.value,
            "hidden_verified": False,
            "hidden_compatible": False,
            "self_excited_contact_detected": False,
            "protocol_complete": False,
            "failed_requirements": failed_reqs,
            "methodological_note": _get_methodological_note(),
            # Expose empty details to avoid key errors
            "equilibria_required": list(equilibria.keys()),
            "equilibria_tested": [],
            "missing_equilibria": list(equilibria.keys()),
            "required_radii": list(required_radii),
            "radii_tested_by_equilibrium": {},
            "missing_radii_by_equilibrium": {eq: list(required_radii) for eq in equilibria},
            "target_hits_total": 0,
            "numerical_failures_total": 0,
            "other_attractor_total": 0,
            "divergence_total": 0,
            "stable_equilibrium_total": 0,
            "samples_total": 0,
        }

    # Validate ref tail length
    if ref_tail_size < min_ref_tail_points:
        failed_reqs.append(
            f"Reference attractor tail has fewer than min_ref_tail_points ({ref_tail_size} < {min_ref_tail_points})."
        )

    # Validate target match metric documentation
    valid_metrics = {"centroid_distance", "bbox_overlap", "nn_percentile"}
    if target_match_metric not in valid_metrics:
        failed_reqs.append(f"Target match metric {target_match_metric!r} is not documented/supported.")

    # 2. Extract tested equilibria and radii
    radii_tested_by_eq: Dict[str, List[float]] = {eq: [] for eq in equilibria}
    
    # We scan summary records or probe runs to populate tested radii
    for record in sphere_summary_records:
        eq_name = record.get("equilibrium")
        radius = record.get("radius")
        if eq_name in radii_tested_by_eq and radius is not None:
            # Avoid duplicate floats
            if not any(is_radius_close(radius, r) for r in radii_tested_by_eq[eq_name]):
                radii_tested_by_eq[eq_name].append(float(radius))

    # Fallback to probe runs if summary records are empty
    if not any(radii_tested_by_eq.values()):
        for run in probe_runs:
            eq_name = run.get("equilibrium")
            radius = run.get("radius")
            if eq_name in radii_tested_by_eq and radius is not None:
                if not any(is_radius_close(radius, r) for r in radii_tested_by_eq[eq_name]):
                    radii_tested_by_eq[eq_name].append(float(radius))

    # Determine tested/missing equilibria
    equilibria_required = list(equilibria.keys())
    # An equilibrium is considered tested if at least one radius was tested
    equilibria_tested = [eq for eq, rads in radii_tested_by_eq.items() if len(rads) > 0]
    missing_equilibria = [eq for eq in equilibria_required if eq not in equilibria_tested]

    # Required/missing radii by equilibrium
    missing_radii_by_eq: Dict[str, List[float]] = {}
    for eq in equilibria_required:
        tested = radii_tested_by_eq.get(eq, [])
        missing = []
        for req in required_radii:
            if not any(is_radius_close(req, t) for t in tested):
                missing.append(float(req))
        missing_radii_by_eq[eq] = missing

    # 3. Calculate totals from probe runs or summary records
    target_hits_total = 0
    numerical_failures_total = 0
    other_attractor_total = 0
    divergence_total = 0
    stable_equilibrium_total = 0
    samples_total = 0

    if probe_runs:
        for run in probe_runs:
            dest = run.get("destination")
            samples_total += 1
            if dest == "target_attractor":
                target_hits_total += 1
            elif dest == "numerical_failure":
                numerical_failures_total += 1
            elif dest == "other_attractor":
                other_attractor_total += 1
            elif dest == "divergence":
                divergence_total += 1
            elif dest == "stable_equilibrium":
                stable_equilibrium_total += 1

            # Validate probe tail length for persistent dynamics
            if dest in ("target_attractor", "other_attractor"):
                traj = run.get("trajectory")
                # If trajectory is available as numpy array or sequence, check its size
                if traj is not None:
                    # In run_sphere_probe_sweep, trajectory contains full or plotted points.
                    # We check if trajectory_available and length of points in tail is checked.
                    # To be robust, if a run dictionary says length, or if we evaluate its shape:
                    if hasattr(traj, "shape"):
                        t_len = traj.shape[0]
                    else:
                        t_len = len(traj)
                    if t_len > 0 and t_len < min_probe_tail_points:
                        msg = f"Probe run sample tail has fewer than min_probe_tail_points ({t_len} < {min_probe_tail_points})."
                        if msg not in failed_reqs:
                            failed_reqs.append(msg)
    else:
        # Fallback to summary records
        for rec in sphere_summary_records:
            samples_total += rec.get("samples", 0)
            target_hits_total += rec.get("TARGET", 0)
            numerical_failures_total += rec.get("FAIL", 0)
            other_attractor_total += rec.get("OTHER", 0)
            divergence_total += rec.get("DIV", 0)
            stable_equilibrium_total += rec.get("EQ", 0)

    # 4. Check requirements for complete protocol
    protocol_complete = True
    
    # Require all equilibria
    if require_all_equilibria:
        if missing_equilibria:
            protocol_complete = False
            failed_reqs.append(f"Missing neighborhood tests for equilibria: {missing_equilibria}")
    else:
        # If strict_all_equilibria is False, but we didn't test all equilibria, protocol is incomplete
        if len(equilibria_tested) < len(equilibria_required):
            protocol_complete = False
            failed_reqs.append("Not all system equilibria were tested (strict_all_equilibria=False).")

    # Require all radii for required/tested equilibria
    for eq in equilibria_tested:
        if missing_radii_by_eq[eq]:
            protocol_complete = False
            failed_reqs.append(f"Equilibrium {eq} is missing required radii: {missing_radii_by_eq[eq]}")

    # 5. Apply verification decision rules
    hidden_verified = False
    hidden_compatible = False
    self_excited_contact_detected = False

    if target_hits_total > 0:
        hiddenness_status = HiddennessVerificationStatus.SELF_EXCITED_CONTACT_DETECTED.value
        self_excited_contact_detected = True
        failed_reqs.append(f"Self-excited contact detected: {target_hits_total} probe trajectory hits on target attractor.")
    elif not protocol_complete:
        hiddenness_status = HiddennessVerificationStatus.INCOMPLETE_PROTOCOL.value
        # If there are no target hits and seed reached attractor, it is hidden_compatible
        if seed_reached_attractor:
            hidden_compatible = True
    elif numerical_failures_total > 0 and not allow_numerical_failures:
        hiddenness_status = HiddennessVerificationStatus.NUMERICAL_FAILURE.value
        failed_reqs.append(f"Numerical failures detected: {numerical_failures_total} probe integrations failed.")
        # Still compatible if zero target hits
        hidden_compatible = True
    elif len(failed_reqs) > 0:
        # Catch-all for failed requirements (like tail length limits)
        hiddenness_status = HiddennessVerificationStatus.INCOMPLETE_PROTOCOL.value
        hidden_compatible = True
    else:
        hiddenness_status = HiddennessVerificationStatus.HIDDEN_VERIFIED.value
        hidden_verified = True
        hidden_compatible = True

    return {
        "hiddenness_status": hiddenness_status,
        "hidden_verified": hidden_verified,
        "hidden_compatible": hidden_compatible,
        "self_excited_contact_detected": self_excited_contact_detected,
        "protocol_complete": protocol_complete,
        "equilibria_required": equilibria_required,
        "equilibria_tested": equilibria_tested,
        "missing_equilibria": missing_equilibria,
        "required_radii": list(required_radii),
        "radii_tested_by_equilibrium": {eq: list(rads) for eq, rads in radii_tested_by_eq.items()},
        "missing_radii_by_equilibrium": missing_radii_by_eq,
        "target_hits_total": target_hits_total,
        "numerical_failures_total": numerical_failures_total,
        "other_attractor_total": other_attractor_total,
        "divergence_total": divergence_total,
        "stable_equilibrium_total": stable_equilibrium_total,
        "samples_total": samples_total,
        "failed_requirements": failed_reqs,
        "methodological_note": _get_methodological_note(),
    }


def _get_methodological_note() -> str:
    """Return the formal warning required by the verification contract."""
    return (
        "La declaración hidden_verified es operacional y depende de los radios, direcciones, "
        "tiempos de integración, métrica de comparación, tolerancias e integrador usados. "
        "La ausencia de contactos con las vecindades ensayadas no constituye una prueba matemática "
        "global de ocultedad, pero sí una verificación numérica estricta bajo el contrato especificado."
    )
