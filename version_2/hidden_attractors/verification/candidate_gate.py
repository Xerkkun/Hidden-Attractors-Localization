"""Central evidence gate for auditable hidden-attractor candidates."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from hidden_attractors.reproducibility import (
    DEFAULT_TOLERANCES,
    extract_run_metadata,
    validate_hiddenness_promotion_metadata,
)


CHAOS_EVIDENCE_LEVELS = (
    "strong_chaos_evidence",
    "chaotic_dynamics_supported",
    "chaos_evidence_inconclusive",
    "regular_or_periodic_candidate",
    "unbounded_or_diverged",
)
HIDDENNESS_EVIDENCE_LEVELS = (
    "hiddenness_supported_under_tested_neighborhoods",
    "compatible_with_hiddenness_under_tested_radii",
    "self_excited_contact_detected",
    "hiddenness_inconclusive",
    "candidate_not_reproducible",
    "numerical_failure",
    "candidate_rejected",
)
LEGACY_HIDDENNESS_ALIASES = {
    "hidden_verified": "hiddenness_supported_under_tested_neighborhoods",
    "hidden_verified_only_if_full_protocol_passed": "hiddenness_supported_under_tested_neighborhoods",
    "rejected_self_excited_contact": "self_excited_contact_detected",
    "compatible_in_all_tested_solver_memory_cases": "compatible_with_hiddenness_under_tested_radii",
}
SEED_METHODS = {
    "df_nyquist",
    "describing_function",
    "published_ic",
    "continuation",
    "manual_traced",
    "machado_df",
}


def normalize_hiddenness_label(label: str) -> str:
    """Map legacy labels to the frozen sampled-neighborhood vocabulary."""

    return LEGACY_HIDDENNESS_ALIASES.get(str(label), str(label))


def normalize_candidate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical evidence dictionary with stable section defaults."""

    normalized = deepcopy(evidence)
    for key in (
        "equilibria",
        "matignon",
        "seed",
        "continuation",
        "trajectory",
        "robustness",
        "hiddenness",
        "lyapunov",
        "zero_one",
        "spectrum",
        "poincare",
    ):
        value = normalized.get(key)
        normalized[key] = dict(value) if isinstance(value, Mapping) else {}

    metadata = extract_run_metadata(normalized)
    normalized["run_metadata"] = metadata
    normalized.pop("reproducibility_metadata", None)

    hiddenness = normalized["hiddenness"]
    if "target_hits_from_equilibria" not in hiddenness:
        hiddenness["target_hits_from_equilibria"] = hiddenness.get("target_hits", 0)
    if "tested_radii" not in hiddenness:
        hiddenness["tested_radii"] = hiddenness.get("radii", [])
    if "basin_intersection_detected" not in hiddenness:
        hiddenness["basin_intersection_detected"] = hiddenness.get("basin_intersection", False)

    continuation = normalized["continuation"]
    if not continuation and metadata:
        continuation.update(metadata.get("continuation", {}))

    tolerances = dict(DEFAULT_TOLERANCES)
    if metadata and isinstance(metadata.get("tolerances"), Mapping):
        tolerances.update(metadata["tolerances"])
    if isinstance(normalized.get("tolerances"), Mapping):
        tolerances.update(normalized["tolerances"])
    normalized["tolerances"] = tolerances
    return normalized


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close_to_one(value: Any) -> bool:
    return _finite(value) and math.isclose(float(value), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12)


def _checked_conditions(evidence: dict[str, Any]) -> dict[str, bool]:
    equilibria = evidence["equilibria"]
    matignon = evidence["matignon"]
    seed = evidence["seed"]
    continuation = evidence["continuation"]
    trajectory = evidence["trajectory"]
    robustness = evidence["robustness"]
    hiddenness = evidence["hiddenness"]
    tested_radii = tuple(float(value) for value in hiddenness.get("tested_radii", ()))
    required_radii = tuple(float(value) for value in hiddenness.get("required_radii", tested_radii))
    residual_tol = float(evidence["tolerances"]["equilibrium_residual_tol"])
    finite_fraction = trajectory.get("finite_fraction", 1.0)
    min_post_transient = int(trajectory.get("minimum_post_transient_length", 1))
    continuation_used = bool(continuation.get("used", False))
    continuation_mode = continuation.get("continuation_mode", "none")
    caputo_memory_declared = (
        continuation.get("memory_window_propagated") is True
        or continuation_mode in {"paper_style", "block_restart"}
        or not continuation_used
    )
    metadata_errors = validate_hiddenness_promotion_metadata(evidence.get("run_metadata"))
    return {
        "equilibria_all_found": equilibria.get("all_found") is True,
        "equilibria_residual_within_tolerance": _finite(equilibria.get("max_residual"))
        and float(equilibria["max_residual"]) <= residual_tol,
        "matignon_all_classified": matignon.get("all_classified") is True,
        "matignon_q_recorded": _finite(matignon.get("q")),
        "seed_localized": seed.get("localized") is True,
        "seed_method_supported": seed.get("method") in SEED_METHODS,
        "seed_source_traceable": bool(str(seed.get("source", "")).strip()),
        "continuation_reaches_target": not continuation_used or _close_to_one(continuation.get("final_eta")),
        "continuation_eta_path_recorded": not continuation_used or bool(continuation.get("eta_path")),
        "continuation_memory_declared": caputo_memory_declared,
        "trajectory_bounded": trajectory.get("bounded") is True,
        "trajectory_nontrivial": trajectory.get("nontrivial") is True,
        "trajectory_finite_fraction_acceptable": _finite(finite_fraction) and float(finite_fraction) >= 0.99,
        "trajectory_post_transient_sufficient": int(trajectory.get("post_transient_length", 0)) >= min_post_transient,
        "robustness_tested_h": robustness.get("tested_h") is True,
        "robustness_tested_memory": robustness.get("tested_memory") is True,
        "robustness_tested_t_final": robustness.get("tested_t_final") is True,
        "robustness_tested_integrator": robustness.get("tested_integrator") is True,
        "robustness_consistent": robustness.get("consistent") is True,
        "hiddenness_tested_all_equilibria": hiddenness.get("tested_all_equilibria") is True,
        "hiddenness_tested_radii_recorded": bool(tested_radii),
        "hiddenness_required_radii_tested": bool(required_radii)
        and all(any(math.isclose(required, tested, rel_tol=1.0e-12, abs_tol=1.0e-15) for tested in tested_radii) for required in required_radii),
        "hiddenness_zero_equilibrium_contacts": int(hiddenness.get("target_hits_from_equilibria", 0)) == 0,
        "hiddenness_no_basin_intersection": hiddenness.get("basin_intersection_detected") is False,
        "hiddenness_basin_controls_complete": hiddenness.get("basin_controls_complete", True) is True,
        "reproducibility_metadata_complete": not metadata_errors,
    }


def missing_candidate_conditions(evidence: dict[str, Any]) -> list[str]:
    """List unmet conditions for the strongest sampled-neighborhood label."""

    checked = _checked_conditions(normalize_candidate_evidence(evidence))
    return [key for key, passed in checked.items() if not passed]


def classify_chaos_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Classify finite-time chaos evidence using the frozen positive vocabulary."""

    normalized = normalize_candidate_evidence(evidence)
    trajectory = normalized["trajectory"]
    lyapunov = normalized["lyapunov"]
    zero_one = normalized["zero_one"]
    spectrum = normalized["spectrum"]
    poincare = normalized["poincare"]
    tolerances = normalized["tolerances"]

    bounded = trajectory.get("bounded") is True
    nontrivial = trajectory.get("nontrivial") is True
    lambda_max = lyapunov.get("lambda_max")
    positive = _finite(lambda_max) and float(lambda_max) > float(tolerances["lyapunov_positive_tol"])
    nonpositive = _finite(lambda_max) and not positive
    k_value = zero_one.get("K", zero_one.get("kappa"))
    zero_one_chaotic = _finite(k_value) and float(k_value) >= float(tolerances["zero_one_chaos_threshold"])
    zero_one_regular = _finite(k_value) and float(k_value) <= float(tolerances["zero_one_regular_threshold"])
    spectrum_label = spectrum.get("label")
    spectral_chaotic = spectrum_label == "broadband_spectrum"
    spectral_regular = spectrum_label in {"dominant_periodic_peak", "quasiperiodic_candidate"}
    poincare_label = poincare.get("label")
    poincare_chaotic = poincare_label in {"scattered_section", "complex_section", "nontrivial_section"}
    poincare_regular = poincare_label in {"point_like", "finite_set_like", "curve_like"}
    method_controlled = lyapunov.get("method_status") in {
        "validated",
        "implemented_with_documented_controls",
        "published_reference_partially_reproduced",
        "internal_controls_passed",
    }
    complementary_chaos = zero_one_chaotic or spectral_chaotic or poincare_chaotic
    regular_support = zero_one_regular or spectral_regular or poincare_regular
    conflicts = []
    if positive and zero_one_regular:
        conflicts.append("positive_lyapunov_vs_regular_zero_one")
    if positive and spectral_regular:
        conflicts.append("positive_lyapunov_vs_regular_spectrum")
    if positive and poincare_regular:
        conflicts.append("positive_lyapunov_vs_regular_poincare")
    if not bounded:
        level = "unbounded_or_diverged"
    elif positive and conflicts:
        level = "chaos_evidence_inconclusive"
    elif positive and method_controlled and complementary_chaos:
        level = "strong_chaos_evidence"
    elif positive:
        level = "chaotic_dynamics_supported"
    elif nonpositive and regular_support:
        level = "regular_or_periodic_candidate"
    else:
        level = "chaos_evidence_inconclusive"
    return {
        "chaos_evidence_level": level,
        "lyapunov_support": "positive" if positive else "nonpositive" if nonpositive else "not_available",
        "zero_one_support": "chaotic" if zero_one_chaotic else "regular" if zero_one_regular else "not_available_or_intermediate",
        "spectral_support": "chaotic" if spectral_chaotic else "regular" if spectral_regular else "not_available_or_inconclusive",
        "boundedness_support": "bounded_nontrivial" if bounded and nontrivial else "unbounded_or_trivial",
        "poincare_support": "chaotic" if poincare_chaotic else "regular" if poincare_regular else "not_available_or_inconclusive",
        "diagnostic_conflicts": conflicts,
        "recommended_interpretation": {
            "strong_chaos_evidence": "strong numerical evidence of chaos",
            "chaotic_dynamics_supported": "chaotic dynamics supported by finite-time diagnostics",
            "chaos_evidence_inconclusive": "inconclusive chaos evidence",
            "regular_or_periodic_candidate": "regular/periodic dynamics supported",
            "unbounded_or_diverged": "trajectory is unbounded or diverged",
        }[level],
    }


def evaluate_candidate_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate candidate promotion, hiddenness level, and chaos evidence."""

    normalized = normalize_candidate_evidence(evidence)
    checked = _checked_conditions(normalized)
    missing = [key for key, passed in checked.items() if not passed]
    chaos = classify_chaos_evidence(normalized)
    hiddenness = normalized["hiddenness"]
    numerical_failures = int(hiddenness.get("numerical_failures", 0))
    target_hits = int(hiddenness.get("target_hits_from_equilibria", 0))
    metadata_missing = not checked["reproducibility_metadata_complete"]

    if chaos["chaos_evidence_level"] == "unbounded_or_diverged":
        verdict = "candidate_rejected"
    elif target_hits > 0 or hiddenness.get("basin_intersection_detected") is True:
        verdict = "self_excited_contact_detected"
    elif numerical_failures > 0:
        verdict = "numerical_failure"
    elif not checked["hiddenness_tested_all_equilibria"] or not checked["hiddenness_tested_radii_recorded"]:
        verdict = "hiddenness_inconclusive"
    elif all(checked.values()):
        verdict = "hiddenness_supported_under_tested_neighborhoods"
    elif metadata_missing and normalized.get("run_metadata") is None:
        verdict = "candidate_not_reproducible"
    else:
        verdict = "compatible_with_hiddenness_under_tested_radii"
    warnings = []
    if metadata_missing:
        warnings.extend(validate_hiddenness_promotion_metadata(normalized.get("run_metadata")))
    warnings.extend(chaos["diagnostic_conflicts"])
    return {
        "verdict": verdict,
        "hiddenness_evidence_level": verdict,
        "evidence_level": verdict,
        **chaos,
        "checked_conditions": checked,
        "missing_conditions": missing,
        "warnings": list(dict.fromkeys(warnings)),
        "promotion_allowed": verdict == "hiddenness_supported_under_tested_neighborhoods",
    }


__all__ = [
    "CHAOS_EVIDENCE_LEVELS",
    "HIDDENNESS_EVIDENCE_LEVELS",
    "LEGACY_HIDDENNESS_ALIASES",
    "classify_chaos_evidence",
    "evaluate_candidate_gate",
    "missing_candidate_conditions",
    "normalize_candidate_evidence",
    "normalize_hiddenness_label",
]
