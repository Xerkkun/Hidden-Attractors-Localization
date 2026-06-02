"""Phase E: Published Continuation Comparison.

Compares the paper-style integration strategy reported (or inferred) from
published articles against the Caputo-aware history_window_transport strategy
available in the library.

What this phase does NOT do
---------------------------
- Does NOT verify hiddenness (hidden_verified is never set).
- Does NOT certify chaos (chaos_certified_by_this_pipeline: false).
- Does NOT declare hidden_verified or chaos_verified.
- Does NOT invent omega0, k, a0, seed, attractor ranges, or continuation
  paths if the article does not report them.

Article-specific rules
----------------------
Kuznetsov 2017 (q=1):
    Integer-order system. No Caputo memory, no history_window_transport.
    paper_style_initial_condition_integration only.

Danca 2017 (q=0.9998):
    No IC/seed reported. All comparison modes disabled.
    Outputs published_data_missing.

Wu 2023 (q=0.99):
    IC reported. k=null so deformed Lure is unavailable.
    original_system_strategy_comparison with paper_style_last_point_restart
    vs. caputo_aware_history_window_transport.

Allowed overall_status values
------------------------------
- published_continuation_reproduced
    Only if paper.reports_continuation == true AND path was reproduced.
- published_initial_condition_reintegrated
- published_seed_reintegrated
- published_paper_style_comparison_performed
- published_comparison_partial_original_only
- published_comparison_inconclusive
- published_data_missing
- published_continuation_not_reported

References
----------
- Kuznetsov et al. (2017): doi:10.1016/j.ifacol.2017.08.470
- Danca (2017): doi:10.1007/s11071-017-3472-7
- Wu et al. (2023): doi:10.1016/j.rinp.2023.106866
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional YAML import
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Reuse Phase C+D classifiers and metrics — no re-implementation
# ---------------------------------------------------------------------------
from validation.python.fractional_memory_validation import (
    classify_fractional_trajectory,
    compute_memory_metrics,
)

# ---------------------------------------------------------------------------
# Reuse Phase D comparison helpers
# ---------------------------------------------------------------------------
from validation.python.continuation_memory_validation import (
    extract_history,
    restart_from_last_point,
    continue_with_history_window,
    classify_continuation_segment,
    compute_segment_metrics,
    compare_restart_vs_history,
)

# ---------------------------------------------------------------------------
# Mandatory no-claim metadata
# ---------------------------------------------------------------------------
_NO_CLAIM: Dict[str, Any] = {
    "hiddenness_certified_by_this_pipeline": False,
    "chaos_certified_by_this_pipeline": False,
    "no_hidden_verified_claim": True,
    "pointwise_comparison_used": False,
}

_ALLOWED_OVERALL_STATUSES = {
    "published_continuation_reproduced",
    "published_initial_condition_reintegrated",
    "published_seed_reintegrated",
    "published_paper_style_comparison_performed",
    "published_comparison_partial_original_only",
    "published_comparison_inconclusive",
    "published_data_missing",
    "published_continuation_not_reported",
}

_ALLOWED_PAPER_STYLE_VS_HISTORY = {
    "paper_style_and_history_consistent",
    "paper_style_differs_from_history",
    "paper_style_restart_artifact_possible",
    "paper_style_result_differs_from_caputo_history_transport",
    "comparison_inconclusive",
    "comparison_not_applicable",
}

_ALLOWED_REINTEGRATION_STATUSES = {
    "paper_initial_condition_reintegrated",
    "paper_seed_reintegrated",
    "paper_style_comparison_performed",
    "paper_continuation_reproduced",
    "paper_does_not_report_continuation",
    "paper_data_missing",
    "paper_reproduction_partial",
    "paper_reproduction_inconclusive",
}

_ALLOWED_ARTICLE_SCOPES = {
    "paper_reports_continuation",
    "paper_does_not_report_continuation",
    "paper_reports_memory_transport",
    "paper_does_not_report_memory_transport",
    "paper_data_missing",
}


# ===========================================================================
# 1. load_published_continuation_config
# ===========================================================================

def load_published_continuation_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a published continuation comparison YAML config.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    config : dict

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    ValueError
        If required fields are missing or invalid.
    """
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load published continuation configs. "
            "Install with: pip install pyyaml"
        )

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        config = _yaml.safe_load(fh)

    if config is None:
        raise ValueError(f"Empty YAML file: {path}")

    required = [
        "case_id", "reference_id", "system_id",
        "paper", "dynamics", "published_data",
        "comparison_modes", "no_claims",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required fields in {path}: {missing}")

    # Validate no_claims
    nc = config["no_claims"]
    if nc.get("hiddenness_certified_by_this_pipeline") is not False:
        raise ValueError("no_claims.hiddenness_certified_by_this_pipeline must be false")
    if nc.get("chaos_certified_by_this_pipeline") is not False:
        raise ValueError("no_claims.chaos_certified_by_this_pipeline must be false")
    if nc.get("no_hidden_verified_claim") is not True:
        raise ValueError("no_claims.no_hidden_verified_claim must be true")

    # Validate comparison_policy.pointwise_comparison_used
    policy = config.get("comparison_policy", {})
    if policy.get("pointwise_comparison_used", False) is True:
        raise ValueError(
            "comparison_policy.pointwise_comparison_used must be false"
        )

    return config


# ===========================================================================
# 2. classify_article_continuation_scope
# ===========================================================================

def classify_article_continuation_scope(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify what the article reports regarding continuation.

    Parameters
    ----------
    config : dict
        Validated config from load_published_continuation_config.

    Returns
    -------
    scope : dict
        Fields:
        - paper_reports_continuation : bool
        - paper_does_not_report_continuation : bool
        - paper_reports_memory_transport : bool
        - paper_does_not_report_memory_transport : bool
        - paper_data_missing : bool
        - paper_style_strategy : str
          'paper_style_last_point_restart' | 'paper_style_initial_condition_integration'
          | 'none'
    """
    paper = config.get("paper", {})
    pub = config.get("published_data", {})

    reports_continuation = paper.get("reports_continuation", False)
    reports_memory_transport = paper.get("reports_memory_transport", False)
    reports_initial_condition = paper.get("reports_initial_condition", False)
    reports_seed = paper.get("reports_seed", False)

    # Determine if data is missing
    ics = pub.get("initial_conditions")
    seed = pub.get("seed", {})
    seed_source = seed.get("seed_source", "missing") if isinstance(seed, dict) else "missing"
    has_ic = (ics is not None and len(ics) > 0)
    has_seed = (seed_source not in ("missing", None))

    paper_data_missing = (not has_ic) and (not has_seed)

    # Continuation scope
    paper_reports_cont = bool(reports_continuation)
    paper_does_not_report_cont = not paper_reports_cont

    # Memory transport scope
    if reports_memory_transport is None or reports_memory_transport is False:
        paper_reports_mem = False
        paper_does_not_report_mem = True
    else:
        paper_reports_mem = True
        paper_does_not_report_mem = False

    # Paper-style strategy
    modes = config.get("comparison_modes", {})
    if modes.get("paper_style_initial_condition_integration", False) and has_ic:
        paper_style_strategy = "paper_style_initial_condition_integration"
    elif modes.get("paper_style_last_point_restart", False) and (has_ic or has_seed):
        paper_style_strategy = "paper_style_last_point_restart"
    else:
        paper_style_strategy = "none"

    return {
        "paper_reports_continuation": paper_reports_cont,
        "paper_does_not_report_continuation": paper_does_not_report_cont,
        "paper_reports_memory_transport": paper_reports_mem,
        "paper_does_not_report_memory_transport": paper_does_not_report_mem,
        "paper_data_missing": paper_data_missing,
        "paper_style_strategy": paper_style_strategy,
    }


# ===========================================================================
# 3. resolve_initial_conditions
# ===========================================================================

def resolve_initial_conditions(
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Resolve initial conditions from published data.

    Returns a list of IC dicts:
        {
          "ic_id": str,
          "x0": list[float] or None,
          "source": str  # one of the IC source labels
        }

    Source labels:
    - "paper_reported_initial_condition"
    - "paper_reported_seed"
    - "library_reconstructed_seed"
    - "missing"

    Rules
    -----
    - Published ICs (initial_conditions) → "paper_reported_initial_condition"
    - Published seeds (seed_plus/seed_minus != null and seed_source != "missing")
      with source = "paper_reported_seed" → "paper_reported_seed"
    - If seed_source = "library_reconstructed_seed" → "library_reconstructed_seed"
    - Otherwise → "missing"
    """
    pub = config.get("published_data", {})
    ics_raw = pub.get("initial_conditions")
    seed_raw = pub.get("seed", {})

    result: List[Dict[str, Any]] = []

    # Published ICs
    if isinstance(ics_raw, dict):
        for ic_key, ic_val in ics_raw.items():
            if isinstance(ic_val, list) and len(ic_val) > 0:
                result.append({
                    "ic_id": ic_key,
                    "x0": ic_val,
                    "source": "paper_reported_initial_condition",
                })
    elif isinstance(ics_raw, list) and len(ics_raw) > 0:
        result.append({
            "ic_id": "x0_paper",
            "x0": ics_raw,
            "source": "paper_reported_initial_condition",
        })

    # Seed entries
    if isinstance(seed_raw, dict):
        seed_source = seed_raw.get("seed_source", "missing")
        for seed_key in ("seed_plus", "seed_minus"):
            sv = seed_raw.get(seed_key)
            if sv is not None and isinstance(sv, list) and len(sv) > 0:
                if seed_source == "paper_reported_seed":
                    src = "paper_reported_seed"
                elif seed_source == "library_reconstructed_seed":
                    src = "library_reconstructed_seed"
                else:
                    src = "missing"
                result.append({
                    "ic_id": seed_key,
                    "x0": sv,
                    "source": src,
                })

    if not result:
        result.append({
            "ic_id": "ic_0",
            "x0": None,
            "source": "missing",
        })

    return result


# ===========================================================================
# 4. run_paper_style_integration
# ===========================================================================

def _get_system_rhs(system_id: str):
    """Return (rhs, system_obj) for a given system_id."""
    from hidden_attractors.systems import get_system

    _ID_MAP = {
        "chua_integer_saturation": "chua-nonsmooth",
        "chua_fractional_saturation": "chua-nonsmooth",
        "chua_fractional_arctan": "chua-arctan",
    }
    registry_name = _ID_MAP.get(system_id)
    if registry_name is None:
        raise ValueError(
            f"Unknown system_id '{system_id}'. "
            f"Supported: {list(_ID_MAP.keys())}"
        )
    system_obj = get_system(registry_name)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        return system_obj.evaluate(np.asarray(x, dtype=float))

    return rhs, system_obj


def run_paper_style_integration(
    config: Dict[str, Any],
    ic: Dict[str, Any],
    fast: bool = False,
) -> Dict[str, Any]:
    """Run a single paper-style integration from a given IC.

    For q == 1: uses integer-order integrator (EFORK_Q1 / RK4).
    For q < 1: uses ABM Caputo with memory_mode="full",
               history_times=None (fresh start — paper_style_last_point_restart).

    Does NOT transport history (paper-style means last_point_restart or
    initial_condition_integration as reported by the article).

    Returns
    -------
    result : dict
        Keys: ic_id, ic_source, times, states, int_status, dynamic_class,
        rho_attractor, rho_max, range_vector, final_state,
        bounded_nontrivial, chaotic_candidate_by_geometry,
        strategy, q
    """
    dyn = config.get("dynamics", {})
    q = float(dyn.get("q", 1.0))
    h = float(dyn.get("h", 0.05))
    divergence_norm = 120.0

    if fast and "fast_test" in config:
        t_final = float(config["fast_test"]["t_final"])
        t_burn = float(config["fast_test"]["t_burn"])
    else:
        t_final = float(dyn.get("t_final", 100.0))
        t_burn = float(dyn.get("t_burn", 50.0))

    x0 = ic.get("x0")
    ic_id = ic.get("ic_id", "unknown")
    ic_source = ic.get("source", "missing")

    if x0 is None:
        return {
            "ic_id": ic_id,
            "ic_source": ic_source,
            "times": np.array([]),
            "states": np.array([]).reshape(0, 3),
            "int_status": "skipped_missing_ic",
            "dynamic_class": "inconclusive",
            "rho_attractor": float("nan"),
            "rho_max": float("nan"),
            "range_vector": [float("nan"), float("nan"), float("nan")],
            "final_state": [float("nan"), float("nan"), float("nan")],
            "bounded_nontrivial": False,
            "chaotic_candidate_by_geometry": False,
            "strategy": "none",
            "q": q,
        }

    x0_arr = np.asarray(x0, dtype=float)
    system_id = config.get("system_id", "")
    rhs, system_obj = _get_system_rhs(system_id)

    if q == 1.0:
        # Integer-order: use RK4
        from hidden_attractors.integrations.rk4 import rk4_integrate
        N = int(math.ceil(t_final / h))
        times, states, int_status, _ = rk4_integrate(
            rhs=rhs,
            x0=x0_arr,
            h=h,
            N=N,
            divergence_norm=divergence_norm,
        )
        strategy = "paper_style_initial_condition_integration_integer_rk4"
    else:
        # Fractional: use ABM Caputo, no history transport
        from hidden_attractors.integrations.abm import caputo_abm_integrate
        times, states, int_status = caputo_abm_integrate(
            rhs=rhs,
            x0=x0_arr,
            q=q,
            h=h,
            t_final=t_final,
            divergence_norm=divergence_norm,
            memory_mode="full",
            history_times=None,
            history_states=None,
            use_c_backend=True,
        )
        strategy = "paper_style_initial_condition_integration_abm"

    # Classify
    dyn_class = classify_fractional_trajectory(
        times=times,
        states=states,
        t_burn=t_burn,
        divergence_norm=divergence_norm,
        collapse_variance_tolerance=1e-8,
        min_range_tolerance=1e-5,
    )

    # Apply chaotic geometry heuristic if bounded
    chaotic_candidate = False
    if dyn_class == "bounded_nontrivial" and len(times) > 0:
        mask = times >= t_burn
        post = states[mask]
        if post.shape[0] >= 10:
            var_per_dim = np.var(post, axis=0)
            rho_full = float(np.sqrt(np.mean(np.sum(
                (post - np.mean(post, axis=0)) ** 2, axis=1
            ))))
            significant_dims = int(np.sum(var_per_dim > 0.01 * np.max(var_per_dim)))
            if significant_dims >= 2 and rho_full > 0.5:
                chaotic_candidate = True

    metrics = compute_memory_metrics(times, states, t_burn)

    return {
        "ic_id": ic_id,
        "ic_source": ic_source,
        "times": times,
        "states": states,
        "int_status": int_status,
        "dynamic_class": dyn_class,
        "rho_attractor": metrics["rho_attractor"],
        "rho_max": metrics["rho_max"],
        "range_vector": metrics["range_vector"],
        "final_state": metrics["final_state"],
        "bounded_nontrivial": dyn_class in (
            "bounded_nontrivial", "periodic_candidate", "chaotic_candidate_by_geometry"
        ),
        "chaotic_candidate_by_geometry": chaotic_candidate,
        "strategy": strategy,
        "q": q,
    }


# ===========================================================================
# 5. run_published_or_inferred_continuation
# ===========================================================================

def run_published_or_inferred_continuation(
    config: Dict[str, Any],
    ic: Dict[str, Any],
    fast: bool = False,
) -> Dict[str, Any]:
    """Run the comparison: paper-style vs Caputo-aware strategy.

    Rules:
    - If q == 1: no history_window_transport (not applicable).
    - If k == null: deformed_lure_continuation is unavailable.
    - If original_system_strategy_comparison is true:
        run last_point_restart and history_window_transport on original system.

    Reuses Phase D functions from continuation_memory_validation.

    Returns
    -------
    result : dict
        Contains 'paper_style_result', 'history_result' (if applicable),
        'deformed_lure_continuation_status', 'original_system_comparison_performed'.
    """
    dyn = config.get("dynamics", {})
    q = float(dyn.get("q", 1.0))
    h = float(dyn.get("h", 0.01))
    divergence_norm = 120.0
    system_id = config.get("system_id", "")
    modes = config.get("comparison_modes", {})
    cont_cfg = config.get("continuation", {})
    policy = config.get("comparison_policy", {})

    if fast and "fast_test" in config:
        t_final = float(config["fast_test"]["t_final"])
        t_burn = float(config["fast_test"]["t_burn"])
    else:
        t_final = float(dyn.get("t_final", 100.0))
        t_burn = float(dyn.get("t_burn", 50.0))

    x0 = ic.get("x0")
    if x0 is None:
        return {
            "paper_style_result": None,
            "history_result": None,
            "deformed_lure_continuation_status": "continuation_auxiliary_unavailable",
            "original_system_comparison_performed": False,
            "status": "paper_data_missing",
        }

    x0_arr = np.asarray(x0, dtype=float)
    pub = config.get("published_data", {})
    k_val = pub.get("k")

    # Deformed Lure: k is required
    deformed_lure_status = "continuation_auxiliary_unavailable"
    if modes.get("deformed_lure_continuation", False) and k_val is not None:
        deformed_lure_status = "deformed_lure_available_but_not_run_by_phase_e"

    # q == 1: no history transport
    if q == 1.0:
        return {
            "paper_style_result": None,  # Handled by run_paper_style_integration
            "history_result": None,
            "deformed_lure_continuation_status": deformed_lure_status,
            "original_system_comparison_performed": False,
            "status": "q1_no_history_transport",
        }

    # Original system strategy comparison (q < 1, k=null)
    run_original = modes.get("original_system_strategy_comparison", False)
    run_history = modes.get("caputo_aware_history_window_transport", False)

    if not run_original:
        return {
            "paper_style_result": None,
            "history_result": None,
            "deformed_lure_continuation_status": deformed_lure_status,
            "original_system_comparison_performed": False,
            "status": "no_comparison_modes_enabled",
        }

    rhs, system_obj = _get_system_rhs(system_id)

    def rhs_original(t: float, x: np.ndarray) -> np.ndarray:
        return system_obj.evaluate(np.asarray(x, dtype=float))

    # Paper-style: last_point_restart (fresh Caputo start, no history)
    paper_style_seg = restart_from_last_point(
        rhs_eta=rhs_original,
        x_last=x0_arr,
        q=q,
        h=h,
        t_final=t_final,
        divergence_norm=divergence_norm,
    )
    paper_times = paper_style_seg["times"]
    paper_states = paper_style_seg["states"]
    paper_dyn = classify_fractional_trajectory(
        times=paper_times, states=paper_states, t_burn=t_burn,
        divergence_norm=divergence_norm,
        collapse_variance_tolerance=1e-8, min_range_tolerance=1e-5,
    )
    paper_metrics = compute_memory_metrics(paper_times, paper_states, t_burn)

    paper_result = {
        "strategy": "paper_style_last_point_restart",
        "int_status": paper_style_seg["status"],
        "dynamic_class": paper_dyn,
        "rho_attractor": paper_metrics["rho_attractor"],
        "rho_max": paper_metrics["rho_max"],
        "range_vector": paper_metrics["range_vector"],
        "final_state": paper_metrics["final_state"],
        "final_state_x": paper_metrics["final_state"][0] if len(paper_metrics["final_state"]) > 0 else float("nan"),
        "final_state_y": paper_metrics["final_state"][1] if len(paper_metrics["final_state"]) > 1 else float("nan"),
        "final_state_z": paper_metrics["final_state"][2] if len(paper_metrics["final_state"]) > 2 else float("nan"),
        "range_x": paper_metrics["range_vector"][0] if len(paper_metrics["range_vector"]) > 0 else float("nan"),
        "range_y": paper_metrics["range_vector"][1] if len(paper_metrics["range_vector"]) > 1 else float("nan"),
        "range_z": paper_metrics["range_vector"][2] if len(paper_metrics["range_vector"]) > 2 else float("nan"),
        "availability": "available",
    }

    history_result = None
    if run_history:
        # Caputo-aware: run paper-style first, then continue with history window
        window_entries = cont_cfg.get("history_windows", [{"M": 256}])
        M = int(window_entries[-1]["M"]) if window_entries else 256

        # First do the paper-style integration to get a history
        from hidden_attractors.integrations.abm import caputo_abm_integrate
        init_times, init_states, _ = caputo_abm_integrate(
            rhs=rhs_original,
            x0=x0_arr,
            q=q,
            h=h,
            t_final=t_final,
            divergence_norm=divergence_norm,
            memory_mode="full",
            history_times=None,
            history_states=None,
            use_c_backend=True,
        )

        # Extract history and continue
        hist_t, hist_x = extract_history(init_times, init_states, M)
        hist_seg = continue_with_history_window(
            rhs_eta_new=rhs_original,
            history_times=hist_t,
            history_states=hist_x,
            q=q,
            h=h,
            t_final=t_final,
            M=M,
            divergence_norm=divergence_norm,
        )
        hist_times = hist_seg["times"]
        hist_states = hist_seg["states"]
        hist_dyn = classify_fractional_trajectory(
            times=hist_times, states=hist_states, t_burn=t_burn,
            divergence_norm=divergence_norm,
            collapse_variance_tolerance=1e-8, min_range_tolerance=1e-5,
        )
        hist_metrics = compute_memory_metrics(hist_times, hist_states, t_burn)

        history_result = {
            "strategy": "caputo_aware_history_window_transport",
            "M": M,
            "int_status": hist_seg["status"],
            "dynamic_class": hist_dyn,
            "rho_attractor": hist_metrics["rho_attractor"],
            "rho_max": hist_metrics["rho_max"],
            "range_vector": hist_metrics["range_vector"],
            "final_state": hist_metrics["final_state"],
            "final_state_x": hist_metrics["final_state"][0] if len(hist_metrics["final_state"]) > 0 else float("nan"),
            "final_state_y": hist_metrics["final_state"][1] if len(hist_metrics["final_state"]) > 1 else float("nan"),
            "final_state_z": hist_metrics["final_state"][2] if len(hist_metrics["final_state"]) > 2 else float("nan"),
            "range_x": hist_metrics["range_vector"][0] if len(hist_metrics["range_vector"]) > 0 else float("nan"),
            "range_y": hist_metrics["range_vector"][1] if len(hist_metrics["range_vector"]) > 1 else float("nan"),
            "range_z": hist_metrics["range_vector"][2] if len(hist_metrics["range_vector"]) > 2 else float("nan"),
            "availability": "available",
        }

    return {
        "paper_style_result": paper_result,
        "history_result": history_result,
        "deformed_lure_continuation_status": deformed_lure_status,
        "original_system_comparison_performed": True,
        "status": "ok",
    }


# ===========================================================================
# 6. compare_to_published_result
# ===========================================================================

def compare_to_published_result(
    config: Dict[str, Any],
    run_result: Dict[str, Any],
) -> str:
    """Determine paper-style reintegration status.

    Parameters
    ----------
    config : dict
    run_result : dict
        Output of run_paper_style_integration.

    Returns
    -------
    status : str
        One of _ALLOWED_REINTEGRATION_STATUSES.
    """
    paper = config.get("paper", {})
    reports_continuation = paper.get("reports_continuation", False)

    # Guard: never return paper_continuation_reproduced if not reported
    if reports_continuation is False:
        # Classify based on what was available
        ic_source = run_result.get("ic_source", "missing")
        dyn_class = run_result.get("dynamic_class", "inconclusive")
        bounded = run_result.get("bounded_nontrivial", False)
        int_status = run_result.get("int_status", "skipped_missing_ic")

        if int_status == "skipped_missing_ic":
            return "paper_data_missing"

        if ic_source == "paper_reported_initial_condition" and bounded:
            return "paper_initial_condition_reintegrated"
        elif ic_source in ("paper_reported_seed", "library_reconstructed_seed") and bounded:
            return "paper_seed_reintegrated"
        elif bounded:
            return "paper_style_comparison_performed"
        elif dyn_class in ("diverged", "nan_detected", "collapsed_to_equilibrium", "too_short"):
            return "paper_reproduction_inconclusive"
        else:
            return "paper_reproduction_inconclusive"

    elif reports_continuation is True:
        # Hypothetically: if paper reports continuation and we reproduced it
        dyn_class = run_result.get("dynamic_class", "inconclusive")
        bounded = run_result.get("bounded_nontrivial", False)
        if bounded:
            return "paper_continuation_reproduced"
        else:
            return "paper_reproduction_inconclusive"

    return "paper_reproduction_inconclusive"


# ===========================================================================
# 7. compare_paper_style_vs_history
# ===========================================================================

def compare_paper_style_vs_history(
    config: Dict[str, Any],
    paper_style_result: Optional[Dict[str, Any]],
    history_result: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Compare paper-style last_point_restart vs history_window_transport.

    Parameters
    ----------
    config : dict
    paper_style_result : dict or None
    history_result : dict or None

    Returns
    -------
    status : str
        One of _ALLOWED_PAPER_STYLE_VS_HISTORY.
    comparison_dict : dict
    """
    if paper_style_result is None or history_result is None:
        return "comparison_not_applicable", {}

    policy = config.get("comparison_policy", {})
    fs_tol = float(policy.get("final_state_relative_tolerance", 0.50))
    rho_tol = float(policy.get("rho_relative_tolerance", 0.35))
    range_tol = float(policy.get("range_relative_tolerance", 0.35))
    eps = 1e-15

    ps_class = paper_style_result.get("dynamic_class", "inconclusive")
    hs_class = history_result.get("dynamic_class", "inconclusive")
    class_changed = (ps_class != hs_class)

    fs_ps = np.array([
        paper_style_result.get("final_state_x", float("nan")),
        paper_style_result.get("final_state_y", float("nan")),
        paper_style_result.get("final_state_z", float("nan")),
    ], dtype=float)
    fs_hs = np.array([
        history_result.get("final_state_x", float("nan")),
        history_result.get("final_state_y", float("nan")),
        history_result.get("final_state_z", float("nan")),
    ], dtype=float)

    fs_rel_dist = float("nan")
    if np.all(np.isfinite(fs_ps)) and np.all(np.isfinite(fs_hs)):
        fs_rel_dist = float(np.linalg.norm(fs_hs - fs_ps)) / (float(np.linalg.norm(fs_ps)) + eps)

    rho_ps = float(paper_style_result.get("rho_attractor", float("nan")))
    rho_hs = float(history_result.get("rho_attractor", float("nan")))
    rho_rel_diff = float("nan")
    if math.isfinite(rho_ps) and math.isfinite(rho_hs):
        rho_rel_diff = abs(rho_hs - rho_ps) / (rho_ps + eps)

    range_ps = np.array([
        paper_style_result.get("range_x", float("nan")),
        paper_style_result.get("range_y", float("nan")),
        paper_style_result.get("range_z", float("nan")),
    ], dtype=float)
    range_hs = np.array([
        history_result.get("range_x", float("nan")),
        history_result.get("range_y", float("nan")),
        history_result.get("range_z", float("nan")),
    ], dtype=float)
    range_rel_diff = float("nan")
    if np.all(np.isfinite(range_ps)) and np.all(np.isfinite(range_hs)):
        rn_ps = float(np.linalg.norm(range_ps))
        range_rel_diff = float(np.linalg.norm(range_hs - range_ps)) / (rn_ps + eps)

    exceeded = []
    if math.isfinite(fs_rel_dist) and fs_rel_dist > fs_tol:
        exceeded.append(f"final_state_relative_distance={fs_rel_dist:.4f} > {fs_tol}")
    if math.isfinite(rho_rel_diff) and rho_rel_diff > rho_tol:
        exceeded.append(f"rho_relative_difference={rho_rel_diff:.4f} > {rho_tol}")
    if math.isfinite(range_rel_diff) and range_rel_diff > range_tol:
        exceeded.append(f"range_relative_difference={range_rel_diff:.4f} > {range_tol}")

    comparison_dict: Dict[str, Any] = {
        "paper_style_dynamic_class": ps_class,
        "history_dynamic_class": hs_class,
        "class_changed": class_changed,
        "final_state_relative_distance": fs_rel_dist,
        "rho_relative_difference": rho_rel_diff,
        "range_relative_difference": range_rel_diff,
        "warnings": exceeded,
    }

    if class_changed and exceeded:
        status = "paper_style_restart_artifact_possible"
    elif class_changed:
        status = "paper_style_differs_from_history"
    elif exceeded:
        status = "paper_style_result_differs_from_caputo_history_transport"
    elif not math.isfinite(fs_rel_dist):
        status = "comparison_inconclusive"
    else:
        status = "paper_style_and_history_consistent"

    comparison_dict["status"] = status
    comparison_dict["warning"] = len(exceeded) > 0 or class_changed

    return status, comparison_dict


# ===========================================================================
# 8. run_published_continuation_case
# ===========================================================================

def run_published_continuation_case(
    config_path: str | Path,
    output_dir: str | Path,
    fast: bool = False,
    save_trajectories: bool = False,
) -> Dict[str, Any]:
    """Run the published continuation comparison for one YAML config.

    Parameters
    ----------
    config_path : str or Path
    output_dir : str or Path
    fast : bool
    save_trajectories : bool

    Returns
    -------
    summary : dict
        Contains published_continuation_summary fields.
    """
    config = load_published_continuation_config(config_path)
    case_id = config["case_id"]
    reference_id = config["reference_id"]
    system_id = config["system_id"]

    scope = classify_article_continuation_scope(config)
    ics = resolve_initial_conditions(config)

    out_root = Path(output_dir) / case_id
    out_root.mkdir(parents=True, exist_ok=True)

    if save_trajectories:
        traj_dir = out_root / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
    else:
        traj_dir = None

    modes = config.get("comparison_modes", {})
    q = float(config.get("dynamics", {}).get("q", 1.0))

    # -----------------------------------------------------------------------
    # Track per-IC results
    # -----------------------------------------------------------------------
    paper_style_rows: List[Dict[str, Any]] = []
    vs_history_rows: List[Dict[str, Any]] = []
    missing_data_records: List[Dict[str, Any]] = []

    # Aggregate state
    any_bounded = False
    any_chaotic_candidate = False
    best_dyn_class = "inconclusive"
    paper_style_reintegration_status = "paper_data_missing"
    paper_style_vs_history_status = "comparison_not_applicable"

    for ic in ics:
        ic_id = ic["ic_id"]
        ic_source = ic["source"]

        if ic["x0"] is None:
            missing_data_records.append({
                "ic_id": ic_id,
                "source": ic_source,
                "reason": "no_published_ic_or_seed",
            })
            continue

        # ------------------------------------------------------------------
        # Paper-style integration
        # ------------------------------------------------------------------
        do_paper_style = (
            modes.get("paper_style_initial_condition_integration", False)
            or (modes.get("paper_style_last_point_restart", False) and q < 1.0)
        )

        ps_result: Optional[Dict[str, Any]] = None
        if do_paper_style:
            ps_result = run_paper_style_integration(config, ic, fast=fast)
            reint_status = compare_to_published_result(config, ps_result)

            row = {
                "case_id": case_id,
                "ic_id": ic_id,
                "ic_source": ic_source,
                "strategy": ps_result["strategy"],
                "int_status": ps_result["int_status"],
                "dynamic_class": ps_result["dynamic_class"],
                "rho_attractor": ps_result["rho_attractor"],
                "rho_max": ps_result["rho_max"],
                "range_x": ps_result["range_vector"][0] if len(ps_result["range_vector"]) > 0 else float("nan"),
                "range_y": ps_result["range_vector"][1] if len(ps_result["range_vector"]) > 1 else float("nan"),
                "range_z": ps_result["range_vector"][2] if len(ps_result["range_vector"]) > 2 else float("nan"),
                "final_state_x": ps_result["final_state"][0] if len(ps_result["final_state"]) > 0 else float("nan"),
                "final_state_y": ps_result["final_state"][1] if len(ps_result["final_state"]) > 1 else float("nan"),
                "final_state_z": ps_result["final_state"][2] if len(ps_result["final_state"]) > 2 else float("nan"),
                "bounded_nontrivial": ps_result["bounded_nontrivial"],
                "chaotic_candidate_by_geometry": ps_result["chaotic_candidate_by_geometry"],
                "reintegration_status": reint_status,
            }
            paper_style_rows.append(row)

            if ps_result["bounded_nontrivial"]:
                any_bounded = True
                paper_style_reintegration_status = reint_status
                best_dyn_class = ps_result["dynamic_class"]
            if ps_result["chaotic_candidate_by_geometry"]:
                any_chaotic_candidate = True

            if save_trajectories and traj_dir is not None:
                tms = ps_result.get("times")
                sts = ps_result.get("states")
                if tms is not None and len(tms) > 0:
                    np.save(traj_dir / f"{ic_id}_paper_style_times.npy", tms)
                    np.save(traj_dir / f"{ic_id}_paper_style_states.npy", sts)

        # ------------------------------------------------------------------
        # Caputo-aware comparison (q < 1 only)
        # ------------------------------------------------------------------
        do_comparison = (
            q < 1.0
            and modes.get("original_system_strategy_comparison", False)
        )

        if do_comparison:
            cont_result = run_published_or_inferred_continuation(config, ic, fast=fast)
            ps_for_cmp = cont_result.get("paper_style_result")
            hs_for_cmp = cont_result.get("history_result")

            vs_status, vs_dict = compare_paper_style_vs_history(
                config, ps_for_cmp, hs_for_cmp
            )
            paper_style_vs_history_status = vs_status

            row2 = {
                "case_id": case_id,
                "ic_id": ic_id,
                "ic_source": ic_source,
                "paper_style_strategy": ps_for_cmp["strategy"] if ps_for_cmp else "none",
                "history_strategy": hs_for_cmp["strategy"] if hs_for_cmp else "none",
                "paper_style_dynamic_class": vs_dict.get("paper_style_dynamic_class", "inconclusive"),
                "history_dynamic_class": vs_dict.get("history_dynamic_class", "inconclusive"),
                "class_changed": vs_dict.get("class_changed", False),
                "final_state_relative_distance": vs_dict.get("final_state_relative_distance", float("nan")),
                "rho_relative_difference": vs_dict.get("rho_relative_difference", float("nan")),
                "range_relative_difference": vs_dict.get("range_relative_difference", float("nan")),
                "status": vs_status,
                "warning": vs_dict.get("warning", False),
            }
            vs_history_rows.append(row2)

            # Update dynamic class tracking from paper-style
            if ps_for_cmp and ps_for_cmp.get("dynamic_class") in (
                "bounded_nontrivial", "periodic_candidate", "chaotic_candidate_by_geometry"
            ):
                any_bounded = True
                best_dyn_class = ps_for_cmp["dynamic_class"]

            if save_trajectories and traj_dir is not None and ps_for_cmp:
                # trajectories are internal to run_published_or_inferred_continuation
                pass  # Trajectories are not returned for memory efficiency

    # -----------------------------------------------------------------------
    # Determine overall_status
    # -----------------------------------------------------------------------
    paper = config.get("paper", {})
    reports_continuation = paper.get("reports_continuation", False)
    k_val = config.get("published_data", {}).get("k")
    all_missing = all(ic["x0"] is None for ic in ics)

    if all_missing:
        overall_status = "published_data_missing"
    elif not reports_continuation and not any_bounded:
        if paper_style_reintegration_status == "paper_data_missing" and not paper_style_rows:
            overall_status = "published_data_missing"
        elif not paper_style_rows and not vs_history_rows:
            overall_status = "published_continuation_not_reported"
        else:
            overall_status = "published_comparison_inconclusive"
    elif not reports_continuation and any_bounded:
        if modes.get("original_system_strategy_comparison", False) and k_val is None:
            overall_status = "published_comparison_partial_original_only"
        elif modes.get("paper_style_initial_condition_integration", False):
            if paper_style_reintegration_status == "paper_initial_condition_reintegrated":
                overall_status = "published_initial_condition_reintegrated"
            elif paper_style_reintegration_status == "paper_seed_reintegrated":
                overall_status = "published_seed_reintegrated"
            else:
                overall_status = "published_paper_style_comparison_performed"
        else:
            overall_status = "published_paper_style_comparison_performed"
    elif reports_continuation is True and any_bounded:
        # Only if paper truly reports continuation AND we reproduced it
        if paper_style_reintegration_status == "paper_continuation_reproduced":
            overall_status = "published_continuation_reproduced"
        else:
            overall_status = "published_paper_style_comparison_performed"
    else:
        overall_status = "published_comparison_inconclusive"

    assert overall_status in _ALLOWED_OVERALL_STATUSES, (
        f"Invalid overall_status: '{overall_status}'"
    )
    # Guard: reports_continuation == false → must not be published_continuation_reproduced
    if not reports_continuation:
        assert overall_status != "published_continuation_reproduced", (
            "published_continuation_reproduced cannot be set when "
            "paper.reports_continuation is false."
        )

    # -----------------------------------------------------------------------
    # Write output files
    # -----------------------------------------------------------------------
    _write_csv(out_root / "paper_style_runs.csv", paper_style_rows)
    _write_csv(out_root / "paper_style_vs_history.csv", vs_history_rows)

    missing_data = {
        "case_id": case_id,
        "missing_ics": missing_data_records,
        "published_missing_values": config.get("published_data", {}).get("missing_values", []),
    }
    with (out_root / "published_data_missing.json").open("w", encoding="utf-8") as fh:
        json.dump(missing_data, fh, indent=2)

    summary: Dict[str, Any] = {
        "stage": "published_continuation_comparison",
        "case_id": case_id,
        "reference_id": reference_id,
        "system_id": system_id,
        "paper_reports_continuation": scope["paper_reports_continuation"],
        "paper_reports_memory_transport": scope["paper_reports_memory_transport"],
        "paper_does_not_report_continuation": scope["paper_does_not_report_continuation"],
        "paper_style_strategy": scope["paper_style_strategy"],
        "caputo_aware_strategy": (
            "history_window_transport"
            if modes.get("caputo_aware_history_window_transport", False)
            else "not_applicable"
        ),
        "overall_status": overall_status,
        "published_data_status": (
            "published_data_missing" if all_missing else "published_data_available"
        ),
        "paper_style_reintegration_status": paper_style_reintegration_status,
        "paper_style_vs_history_status": paper_style_vs_history_status,
        "dynamic_class_detected": best_dyn_class,
        "chaotic_dynamics_candidate_detected": any_chaotic_candidate,
        "hiddenness_certified_by_this_pipeline": False,
        "chaos_certified_by_this_pipeline": False,
        "no_hidden_verified_claim": True,
        "pointwise_comparison_used": False,
    }

    with (out_root / "published_continuation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ===========================================================================
# 9. run_all_published_continuation_comparisons
# ===========================================================================

def run_all_published_continuation_comparisons(
    config_dir: str | Path,
    output_dir: str | Path,
    fast: bool = False,
    save_trajectories: bool = False,
) -> List[Dict[str, Any]]:
    """Run published continuation comparison for all YAML configs in a directory.

    Parameters
    ----------
    config_dir : str or Path
    output_dir : str or Path
    fast : bool
    save_trajectories : bool

    Returns
    -------
    summaries : list of dict
    """
    config_dir = Path(config_dir)
    yaml_files = sorted(config_dir.glob("*.yaml"))

    if not yaml_files:
        raise FileNotFoundError(
            f"No *.yaml files found in: {config_dir}"
        )

    summaries = []
    for yaml_path in yaml_files:
        if yaml_path.name == "README.md":
            continue
        print(f"[published_continuation_comparison] Running: {yaml_path.name}")
        try:
            summary = run_published_continuation_case(
                config_path=yaml_path,
                output_dir=output_dir,
                fast=fast,
                save_trajectories=save_trajectories,
            )
            summaries.append(summary)
            print(
                f"  -> case_id={summary['case_id']}, "
                f"overall_status={summary['overall_status']}"
            )
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {yaml_path.name}: {exc}")
            traceback.print_exc()
            summaries.append({
                "stage": "published_continuation_comparison",
                "yaml_file": str(yaml_path),
                "overall_status": "published_comparison_inconclusive",
                "error": str(exc),
                "hiddenness_certified_by_this_pipeline": False,
                "chaos_certified_by_this_pipeline": False,
                "no_hidden_verified_claim": True,
            })

    return summaries


# ===========================================================================
# CSV writer helper
# ===========================================================================

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
