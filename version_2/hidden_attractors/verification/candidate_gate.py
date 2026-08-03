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


from .status_labels import CANONICAL_ATTRACTOR_STATUS, normalize_attractor_status

CHAOS_EVIDENCE_LEVELS = (
    "strong_chaos_evidence",
    "chaotic_dynamics_supported",
    "chaos_evidence_inconclusive",
    "regular_or_periodic_candidate",
    "unbounded_or_diverged",
)
HIDDENNESS_EVIDENCE_LEVELS = tuple(CANONICAL_ATTRACTOR_STATUS)

SEED_METHODS = {
    "df_nyquist",
    "describing_function",
    "published_ic",
    "continuation",
    "manual_traced",
    "machado_df",
}


def normalize_hiddenness_label(label: str) -> str:
    """Map legacy labels to the canonical status vocabulary."""

    return normalize_attractor_status(label)


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
        hiddenness["target_hits_from_equilibria"] = hiddenness.get("target_hits")
    if "tested_radii" not in hiddenness:
        hiddenness["tested_radii"] = hiddenness.get("radii", [])
    if "basin_intersection_detected" not in hiddenness:
        hiddenness["basin_intersection_detected"] = hiddenness.get("basin_intersection")

    continuation = normalized["continuation"]
    if not continuation and metadata:
        continuation.update(metadata.get("continuation", {}))

    tolerances = dict(DEFAULT_TOLERANCES)
    if metadata and isinstance(metadata.get("tolerances"), Mapping):
        tolerances.update(metadata["tolerances"])
    if isinstance(normalized.get("tolerances"), Mapping):
        tolerances.update(normalized["tolerances"])
    normalized["tolerances"] = tolerances

    trajectory = normalized["trajectory"]
    if "post_transient_length" not in trajectory and _finite(trajectory.get("post_transient_rows")):
        trajectory["post_transient_length"] = int(trajectory["post_transient_rows"])
    if "bounded" not in trajectory:
        boundedness_status = trajectory.get("boundedness_status", trajectory.get("status"))
        if boundedness_status in {"bounded", "bounded_nontrivial", "ok"}:
            trajectory["bounded"] = True
        elif boundedness_status in {"unbounded", "diverged", "nonfinite"}:
            trajectory["bounded"] = False
    if "nontrivial" not in trajectory and _finite(trajectory.get("variance_max")):
        trajectory["nontrivial"] = float(trajectory["variance_max"]) > float(
            tolerances["nontrivial_variance_tol"]
        )

    lyapunov = normalized["lyapunov"]
    if "lambda_max" not in lyapunov:
        if _finite(lyapunov.get("largest_exponent")):
            lyapunov["lambda_max"] = lyapunov["largest_exponent"]
        else:
            exponents = lyapunov.get("exponents")
            try:
                exponent_values = list(exponents) if exponents is not None else []
            except TypeError:
                exponent_values = []
            if exponent_values and all(_finite(value) for value in exponent_values):
                lyapunov["lambda_max"] = max(float(value) for value in exponent_values)

    zero_one = normalized["zero_one"]
    if "K" not in zero_one and _finite(zero_one.get("K_global_median")):
        zero_one["K"] = zero_one["K_global_median"]

    spectrum = normalized["spectrum"]
    if "label" not in spectrum:
        label = spectrum.get("state_global", spectrum.get("state"))
        if isinstance(label, str):
            spectrum["label"] = label

    poincare = normalized["poincare"]
    if "label" not in poincare:
        raw_label = poincare.get("interpretation_label", poincare.get("state"))
        poincare_aliases = {
            "point_like_or_fixed_return": "point_like",
            "finite_set_like": "finite_set_like",
            "curve_like": "curve_like",
            "cloud_like": "complex_section",
            "dispersed_cloud_like": "scattered_section",
            "no_crossings": "inconclusive",
            "insufficient_crossings": "inconclusive",
        }
        if isinstance(raw_label, str):
            poincare["label"] = poincare_aliases.get(raw_label, raw_label)
    return normalized


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_integer(value: Any) -> bool:
    return _finite(value) and float(value) >= 0.0 and float(value).is_integer()


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
    target_hits = hiddenness.get("target_hits_from_equilibria")
    numerical_failures = hiddenness.get("numerical_failures")
    tested_radii = tuple(float(value) for value in hiddenness.get("tested_radii", ()))
    required_radii = tuple(float(value) for value in hiddenness.get("required_radii", ()))
    residual_tol = float(evidence["tolerances"]["equilibrium_residual_tol"])
    finite_fraction = trajectory.get("finite_fraction")
    min_post_transient = int(trajectory.get("minimum_post_transient_length", 1))
    continuation_used = bool(continuation.get("used", False))
    continuation_mode = continuation.get("continuation_mode", "none")
    caputo_memory_declared = (
        continuation.get("memory_window_propagated") is True
        or continuation_mode in {"paper_style", "block_restart"}
        or not continuation_used
    )
    metadata_errors = validate_hiddenness_promotion_metadata(evidence.get("run_metadata"))
    q_value = matignon.get("q")
    if not _finite(q_value) and isinstance(evidence.get("run_metadata"), Mapping):
        numerical = evidence["run_metadata"].get("numerical_contract", {})
        if isinstance(numerical, Mapping):
            q_value = numerical.get("q")
    integer_q1 = _close_to_one(q_value)
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
        "continuation_memory_declared": integer_q1 or caputo_memory_declared,
        "trajectory_bounded": trajectory.get("bounded") is True,
        "trajectory_nontrivial": trajectory.get("nontrivial") is True,
        "trajectory_finite_fraction_acceptable": _finite(finite_fraction) and float(finite_fraction) >= 0.99,
        "trajectory_post_transient_sufficient": int(trajectory.get("post_transient_length", 0)) >= min_post_transient,
        "robustness_tested_h": robustness.get("tested_h") is True,
        "robustness_memory_requirement_satisfied": integer_q1
        or robustness.get("tested_memory") is True,
        "robustness_tested_t_final": robustness.get("tested_t_final") is True,
        "robustness_tested_integrator": robustness.get("tested_integrator") is True,
        "robustness_consistent": robustness.get("consistent") is True,
        "hiddenness_tested_all_equilibria": hiddenness.get("tested_all_equilibria") is True,
        "hiddenness_tested_radii_recorded": bool(tested_radii),
        "hiddenness_required_radii_tested": bool(required_radii)
        and all(any(math.isclose(required, tested, rel_tol=1.0e-12, abs_tol=1.0e-15) for tested in tested_radii) for required in required_radii),
        "hiddenness_equilibrium_radius_coverage_complete": hiddenness.get(
            "coverage_by_equilibrium_radius_complete"
        )
        is True,
        "hiddenness_target_contacts_recorded": _nonnegative_integer(target_hits),
        "hiddenness_zero_equilibrium_contacts": _nonnegative_integer(target_hits)
        and float(target_hits) == 0.0,
        "hiddenness_no_basin_intersection": hiddenness.get("basin_intersection_detected") is False,
        "hiddenness_basin_controls_complete": hiddenness.get("basin_controls_complete") is True,
        "hiddenness_numerical_failures_recorded": _nonnegative_integer(numerical_failures),
        "hiddenness_no_numerical_failures": _nonnegative_integer(numerical_failures)
        and float(numerical_failures) == 0.0,
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
    positive = (
        bounded
        and nontrivial
        and _finite(lambda_max)
        and float(lambda_max) > float(tolerances["lyapunov_positive_tol"])
    )
    nonpositive = _finite(lambda_max) and not positive
    k_value = zero_one.get("K", zero_one.get("kappa"))
    zero_one_applicable = zero_one.get("gate_applicable", True) is not False
    zero_one_chaotic = zero_one_applicable and _finite(k_value) and float(k_value) >= float(tolerances["zero_one_chaos_threshold"])
    zero_one_regular = zero_one_applicable and _finite(k_value) and float(k_value) <= float(tolerances["zero_one_regular_threshold"])
    spectrum_label = spectrum.get("label")
    spectrum_applicable = spectrum.get("gate_applicable", True) is not False
    spectral_chaotic = spectrum_applicable and spectrum_label == "broadband_spectrum"
    spectral_regular = spectrum_applicable and spectrum_label in {"dominant_periodic_peak", "quasiperiodic_candidate"}
    poincare_label = poincare.get("label")
    poincare_applicable = poincare.get("gate_applicable", True) is not False
    poincare_chaotic = poincare_applicable and poincare_label in {"scattered_section", "complex_section", "nontrivial_section"}
    poincare_regular = poincare_applicable and poincare_label in {"point_like", "finite_set_like", "curve_like"}
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
    raw_numerical_failures = hiddenness.get("numerical_failures")
    raw_target_hits = hiddenness.get("target_hits_from_equilibria")
    numerical_failures = int(raw_numerical_failures) if _nonnegative_integer(raw_numerical_failures) else None
    target_hits = int(raw_target_hits) if _nonnegative_integer(raw_target_hits) else None
    metadata_missing = not checked["reproducibility_metadata_complete"]

    if chaos["chaos_evidence_level"] == "unbounded_or_diverged":
        verdict = "rejected"
    elif (target_hits is not None and target_hits > 0) or hiddenness.get("basin_intersection_detected") is True:
        verdict = "self_excited"
    elif numerical_failures is None or target_hits is None:
        verdict = "inconclusive"
    elif numerical_failures > 0:
        verdict = "inconclusive"
    elif not checked["hiddenness_tested_all_equilibria"] or not checked["hiddenness_tested_radii_recorded"]:
        verdict = "inconclusive"
    elif all(checked.values()):
        verdict = "hidden_under_tested_neighborhoods"
    elif metadata_missing and normalized.get("run_metadata") is None:
        verdict = "inconclusive"
    else:
        verdict = "compatible_with_hiddenness"
    warnings = []
    if metadata_missing:
        warnings.extend(validate_hiddenness_promotion_metadata(normalized.get("run_metadata")))
    warnings.extend(chaos["diagnostic_conflicts"])
    hiddenness_promotion_allowed = verdict == "hidden_under_tested_neighborhoods"
    chaotic_hidden_promotion_allowed = bool(
        hiddenness_promotion_allowed
        and chaos["chaos_evidence_level"] == "strong_chaos_evidence"
        and normalized["trajectory"].get("bounded") is True
        and normalized["trajectory"].get("nontrivial") is True
        and not chaos["diagnostic_conflicts"]
    )
    return {
        "attractor_status": verdict,
        "verdict": verdict,
        "hiddenness_evidence_level": verdict,
        "evidence_level": verdict,
        **chaos,
        "checked_conditions": checked,
        "missing_conditions": missing,
        "warnings": list(dict.fromkeys(warnings)),
        "promotion_allowed": hiddenness_promotion_allowed,
        "hiddenness_promotion_allowed": hiddenness_promotion_allowed,
        "chaotic_hidden_promotion_allowed": chaotic_hidden_promotion_allowed,
        "hidden_chaos_status": (
            "chaotic_hidden_under_tested_neighborhoods"
            if chaotic_hidden_promotion_allowed
            else "hiddenness_only_not_chaotic"
            if hiddenness_promotion_allowed
            else "not_promoted"
        ),
    }


__all__ = [
    "CHAOS_EVIDENCE_LEVELS",
    "HIDDENNESS_EVIDENCE_LEVELS",
    "classify_chaos_evidence",
    "evaluate_candidate_gate",
    "missing_candidate_conditions",
    "normalize_candidate_evidence",
    "normalize_hiddenness_label",
]
