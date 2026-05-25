from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import chua_initial_cond as chua

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hidden_attractors.models.chua import ChuaParameters  # noqa: E402
from hidden_attractors.native.backends import FractionalChuaBackend, FullHistoryABMBackend  # noqa: E402


COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def _centered_signal(signal: Sequence[float]) -> np.ndarray:
    data = np.asarray(signal, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return data
    return data - np.mean(data)


def _spectrum(signal: Sequence[float], h: float) -> tuple[np.ndarray, np.ndarray]:
    data = _centered_signal(signal)
    if data.size < 8 or not np.any(np.abs(data) > 0.0):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    power = np.abs(np.fft.rfft(data * np.hanning(data.size))) ** 2
    omega = 2.0 * np.pi * np.fft.rfftfreq(data.size, d=float(h))
    return omega[1:], power[1:]


def spectral_entropy(signal: Sequence[float], h: float) -> float:
    """Return normalized spectral entropy of a centered trajectory component."""

    _omega, power = _spectrum(signal, h)
    total = float(np.sum(power))
    if power.size < 2 or total <= 0.0:
        return 0.0
    probabilities = power / total
    return -float(np.sum(probabilities * np.log(probabilities + 1.0e-300))) / math.log(power.size)


def dominant_power_ratio(signal: Sequence[float], h: float) -> float:
    """Return the power fraction carried by the dominant nonzero frequency."""

    _omega, power = _spectrum(signal, h)
    total = float(np.sum(power))
    return float(np.max(power) / total) if power.size and total > 0.0 else 0.0


def dominant_frequency(signal: Sequence[float], h: float) -> float:
    """Return the dominant angular frequency in radians per unit time."""

    omega, power = _spectrum(signal, h)
    return float(omega[int(np.argmax(power))]) if power.size else 0.0


def windowed_frequency_stability(signal: Sequence[float], h: float, n_windows: int = 3) -> float:
    """Return the maximum relative drift of dominant angular frequencies."""

    data = np.asarray(signal, dtype=float).ravel()
    windows = [chunk for chunk in np.array_split(data, max(int(n_windows), 1)) if chunk.size >= 8]
    frequencies = [dominant_frequency(chunk, h) for chunk in windows]
    if len(frequencies) < 2 or any(freq <= 0.0 for freq in frequencies):
        return float("inf")
    drift = 0.0
    for index, left in enumerate(frequencies):
        for right in frequencies[index + 1 :]:
            drift = max(drift, abs(left - right) / max(left, right, 1.0e-12))
    return float(drift)


def poincare_section_cluster_count(traj: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Count repeated early Poincare crossings using a normalized greedy clustering."""

    values = np.asarray(traj, dtype=float)
    states = values[:, 1:4] if values.ndim == 2 and values.shape[1] >= 4 else values
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 3:
        return {"section_points": 0, "section_clusters": 0, "poincare_repetitive": False}
    crossing = states[:, 0] - float(np.mean(states[:, 0]))
    points: List[np.ndarray] = []
    for index in range(1, states.shape[0]):
        if crossing[index - 1] < 0.0 <= crossing[index] and states[index, 1] > states[index - 1, 1]:
            alpha = -crossing[index - 1] / max(crossing[index] - crossing[index - 1], 1.0e-300)
            points.append(states[index - 1, 1:3] + alpha * (states[index, 1:3] - states[index - 1, 1:3]))
    minimum = int(config.get("poincare_min_points", 6))
    if len(points) < minimum:
        return {"section_points": len(points), "section_clusters": len(points), "poincare_repetitive": False}
    point_array = np.asarray(points, dtype=float)
    scale = max(float(np.linalg.norm(np.ptp(point_array, axis=0))), 1.0e-12)
    tolerance = float(config.get("poincare_cluster_tol", 0.05)) * scale
    clusters: List[np.ndarray] = []
    for point in point_array:
        if not clusters or min(float(np.linalg.norm(point - center)) for center in clusters) > tolerance:
            clusters.append(point)
    max_clusters = int(config.get("poincare_max_clusters", 3))
    return {
        "section_points": len(points),
        "section_clusters": len(clusters),
        "poincare_repetitive": bool(len(clusters) <= max_clusters),
    }


def post_transient_segment(traj: np.ndarray, h: float, config: Dict[str, Any]) -> np.ndarray:
    """Return only the observation segment used for periodicity diagnostics."""

    values = np.asarray(traj, dtype=float)
    t_transient = float(config.get("t_transient", 0.0))
    if values.ndim != 2 or values.shape[0] == 0 or t_transient <= 0.0:
        return values
    if values.shape[1] >= 4:
        segment = values[values[:, 0] >= t_transient - 0.5 * float(h)].copy()
        if segment.size:
            segment[:, 0] -= segment[0, 0]
        return segment
    start = int(math.ceil(t_transient / max(float(h), 1.0e-300)))
    return values[start:, :].copy()


def classify_early_periodicity(traj: np.ndarray, h: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """Classify post-transient narrowband dynamics before continuation.

    A dominant FFT peak alone is not an exclusion criterion.  A component is
    periodic only when it additionally has low entropy and high concentration,
    or stable frequency across windows and sufficiently high concentration.
    """

    full_values = np.asarray(traj, dtype=float)
    values = post_transient_segment(full_values, h, config)
    states = values[:, 1:4] if values.ndim == 2 and values.shape[1] >= 4 else values
    components = [str(name) for name in config.get("components", ["x", "y", "z"])]
    entropy_min = float(config.get("entropy_min", 0.25))
    dominant_max = float(config.get("dominant_ratio_max", 0.65))
    relaxed_dominant = float(config.get("relaxed_dominant_ratio", 0.45))
    drift_max = float(config.get("freq_drift_max", 0.05))
    n_windows = int(config.get("n_windows", 3))
    min_range = float(config.get("min_range", 0.01))
    divergence_norm = float(config.get("divergence_norm", 120.0))
    component_rows: List[Dict[str, Any]] = []
    periodic_components: List[str] = []
    max_ratio = 0.0
    finite = bool(states.ndim == 2 and states.size > 0 and np.all(np.isfinite(states)))
    diverged = bool(finite and np.max(np.linalg.norm(states, axis=1)) > divergence_norm)
    nontrivial = bool(finite and np.max(np.ptp(states, axis=0)) >= min_range)
    if finite and not diverged:
        for name in components:
            index = COMPONENT_INDEX.get(name)
            if index is None or index >= states.shape[1]:
                continue
            signal = states[:, index]
            entropy = spectral_entropy(signal, h)
            ratio = dominant_power_ratio(signal, h)
            omega = dominant_frequency(signal, h)
            drift = windowed_frequency_stability(signal, h, n_windows=n_windows)
            low_entropy_narrowband = bool(entropy < entropy_min and ratio > dominant_max)
            stable_narrowband = bool(drift < drift_max and ratio > relaxed_dominant)
            periodic = bool(nontrivial and (low_entropy_narrowband or stable_narrowband))
            max_ratio = max(max_ratio, ratio)
            if periodic:
                periodic_components.append(name)
            component_rows.append(
                {
                    "component": name,
                    "spectral_entropy": entropy,
                    "dominant_power_ratio": ratio,
                    "dominant_omega": omega,
                    "relative_frequency_drift": drift,
                    "low_entropy_narrowband": low_entropy_narrowband,
                    "stable_narrowband": stable_narrowband,
                    "periodic_component": periodic,
                }
            )
    section = poincare_section_cluster_count(values, config) if finite and not diverged else {
        "section_points": 0,
        "section_clusters": 0,
        "poincare_repetitive": False,
    }
    required = 2 if bool(config.get("require_two_components", True)) else 1
    spectral_periodic = len(periodic_components) >= required
    poincare_periodic = bool(
        nontrivial
        and section["poincare_repetitive"]
        and max_ratio > relaxed_dominant
    )
    periodic = bool(finite and not diverged and nontrivial and (spectral_periodic or poincare_periodic))
    status = "periodic_post_transient" if periodic else (
        "divergent_post_transient" if diverged else "nonperiodic_post_transient"
    )
    return {
        "early_periodicity_status": status,
        "periodicity_status": status,
        "t_transient": float(config.get("t_transient", 0.0)),
        "analysis_rows": int(values.shape[0]) if values.ndim == 2 else 0,
        "periodic_early": periodic,
        "periodic_post_transient": periodic,
        "diverged_early": diverged,
        "diverged_post_transient": diverged,
        "nontrivial_early": nontrivial,
        "nontrivial_post_transient": nontrivial,
        "periodic_components": ";".join(periodic_components),
        "n_periodic_components": len(periodic_components),
        "component_metrics": component_rows,
        **section,
    }


def _configured_cases(settings: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    matrix = settings.get("matrix", {})
    if bool(matrix.get("enabled", False)):
        cases = [dict(case) for case in matrix.get("cases", [])]
        if not cases:
            raise ValueError("early_periodicity_filter.matrix.enabled requires at least one case.")
        return str(matrix.get("primary_case_id", cases[0]["case_id"])), cases
    backend = str(settings.get("backend", "native_c_efork3"))
    return "legacy_primary", [
        {
            "case_id": "legacy_primary",
            "solver": "python_legacy" if backend == "python_legacy" else "efork3",
            "memory_policy": "truncated",
            "memory_length": float(settings.get("memory_length", 8.0)),
        }
    ]


def _integrate_case(
    x0: np.ndarray,
    case: Dict[str, Any],
    *,
    settings: Dict[str, Any],
    p: Dict[str, Any],
    q: float,
    h: float,
    t_final: float,
    efork_backend: FractionalChuaBackend | None,
    abm_backend: FullHistoryABMBackend | None,
) -> tuple[np.ndarray, str, float | str, str]:
    solver = str(case.get("solver", "")).lower()
    memory_policy = str(case.get("memory_policy", "truncated")).lower()
    configured_lm = float(case.get("memory_length", settings.get("memory_length", 8.0)))
    if solver == "efork3":
        if efork_backend is None:
            raise RuntimeError("EFORK matrix case requested without a native EFORK backend.")
        lm = t_final if memory_policy == "full_history" else configured_lm
        traj = efork_backend.integrate_efork3(x0, q=q, h=h, Lm=lm, t_final=t_final)
        policy = "full_history_over_simulated_horizon" if memory_policy == "full_history" else "finite_memory_window"
        return traj, "chua_frac_backend_lib.c", lm, policy
    if solver == "abm":
        if abm_backend is None:
            raise RuntimeError("ABM matrix case requested without a native ABM backend.")
        if memory_policy == "full_history":
            return (
                abm_backend.integrate(x0, q=q, h=h, t_final=t_final),
                "chua_abm_full_history_lib.c",
                "",
                "full_caputo_history_no_finite_memory_truncation",
            )
        return (
            abm_backend.integrate_truncated(x0, q=q, h=h, Lm=configured_lm, t_final=t_final),
            "chua_abm_full_history_lib.c",
            configured_lm,
            "sliding_restarted_finite_memory_window",
        )
    if solver == "python_legacy":
        return (
            chua.efork3_integrate(
                lambda x: chua.rhs_original(x, p),
                x0,
                qord=q,
                h=h,
                Lm=configured_lm,
                t_f=t_final,
            ),
            "chua_initial_cond.efork3_integrate",
            configured_lm,
            "finite_memory_window",
        )
    raise ValueError(f"Unsupported early-periodicity solver: {solver}")


def run_early_periodicity_filter(
    seed_rows: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run the post-transient solver/memory matrix and gate on its primary case."""

    settings = cfg.get("early_periodicity_filter", {})
    if not bool(settings.get("enabled", True)):
        return {
            "kept_seeds": [dict(row) for row in seed_rows],
            "rejected_seeds": [],
            "diagnostics": [],
            "summary": {"enabled": False, "n_seed_bank_total": len(seed_rows), "n_rejected_periodic_post_transient": 0},
        }
    q = float(cfg["q"])
    h = float(settings.get("h", 0.01))
    t_transient = float(settings.get("t_transient", 0.0))
    observation_time = float(settings.get("observation_time", settings.get("t_final", 120.0) - t_transient))
    if t_transient < 0.0 or observation_time <= 0.0:
        raise ValueError("Periodicidad requiere t_transient >= 0 y observation_time > 0.")
    t_final = t_transient + observation_time
    discard = bool(settings.get("discard_if_periodic", True))
    gate_before_continuation = bool(settings.get("gate_before_continuation", False))
    if gate_before_continuation and not bool(settings.get("historical_reproduction_mode", False)):
        raise ValueError(
            "gate_before_continuation is deprecated for official runs; "
            "use diagnostic soft_precheck or set historical_reproduction_mode=true "
            "only to reproduce an archived route."
        )
    primary_case_id, cases = _configured_cases(settings)
    case_ids = {str(case["case_id"]) for case in cases}
    if primary_case_id not in case_ids:
        raise ValueError("primary_case_id must identify one configured periodicity matrix case.")

    parameters = cfg.get("params", {})
    native_params = ChuaParameters(
        model="nonsmooth",
        alpha=float(parameters.get("alpha_chua", 8.4562)),
        beta=float(parameters.get("beta", 12.0732)),
        gamma=float(parameters.get("gamma_chua", 0.0052)),
        m0=float(parameters.get("m0", -0.1768)),
        m1=float(parameters.get("m1", -1.1468)),
    )
    needs_efork = any(str(case.get("solver", "")).lower() == "efork3" for case in cases)
    needs_abm = any(str(case.get("solver", "")).lower() == "abm" for case in cases)
    efork_backend: FractionalChuaBackend | None = None
    abm_backend: FullHistoryABMBackend | None = None
    if needs_efork:
        efork_backend = FractionalChuaBackend.build(output_name=f"post_transient_periodicity_efork3_q09998_{os.getpid()}")
        efork_backend.set_nonsmooth_params(native_params)
    if needs_abm:
        abm_backend = FullHistoryABMBackend.build(output_name=f"post_transient_periodicity_abm_q09998_{os.getpid()}")
        abm_backend.set_nonsmooth_params(native_params)

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    case_counts = {
        str(case["case_id"]): {"n_tested": 0, "n_periodic_post_transient": 0, "n_nonperiodic_post_transient": 0, "n_failed_or_divergent": 0}
        for case in cases
    }
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    contract = {
        "q": q,
        "h": h,
        "t_transient": t_transient,
        "observation_time": observation_time,
        "primary_case_id": primary_case_id,
        "case_ids": [str(case["case_id"]) for case in cases],
    }

    def account_row(row: Dict[str, Any]) -> None:
        case_id = str(row.get("periodicity_case_id", ""))
        if case_id not in case_counts:
            return
        status = str(row.get("periodicity_status", ""))
        if status == "periodic_post_transient":
            case_counts[case_id]["n_tested"] += 1
            case_counts[case_id]["n_periodic_post_transient"] += 1
        elif status == "nonperiodic_post_transient":
            case_counts[case_id]["n_tested"] += 1
            case_counts[case_id]["n_nonperiodic_post_transient"] += 1
        elif status:
            case_counts[case_id]["n_failed_or_divergent"] += 1

    completed_seed_ids: set[str] = set()
    if checkpoint is not None and resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("contract") != contract:
            raise ValueError("Periodicity checkpoint contract does not match the requested matrix.")
        diagnostics = [dict(row) for row in saved.get("diagnostics", [])]
        for row in diagnostics:
            account_row(row)
            if str(row.get("periodicity_case_id", "")) == primary_case_id:
                completed_seed_ids.add(str(row.get("seed_id", "")))
                if str(row.get("candidate_status", "")).startswith("rejected_"):
                    rejected.append(row)
                else:
                    kept.append(row)

    def save_checkpoint() -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps({"contract": contract, "diagnostics": diagnostics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for source in seed_rows:
        base_row = dict(source)
        if str(base_row.get("seed_id", "")) in completed_seed_ids:
            continue
        if str(base_row.get("valid_seed", "")).lower() not in {"true", "1", "yes"} and base_row.get("valid_seed") is not True:
            base_row["early_periodicity_status"] = "invalid_seed"
            diagnostics.append(base_row)
            save_checkpoint()
            continue
        x0 = np.asarray([float(base_row["x0"]), float(base_row["y0"]), float(base_row["z0"])], dtype=float)
        primary_row: Dict[str, Any] | None = None
        for case in cases:
            row = dict(base_row)
            case_id = str(case["case_id"])
            row["periodicity_case_id"] = case_id
            row["periodicity_solver"] = str(case.get("solver", ""))
            row["periodicity_memory_policy"] = str(case.get("memory_policy", ""))
            row["early_h"] = h
            row["early_t_transient"] = t_transient
            row["early_observation_time"] = observation_time
            row["early_t_final"] = t_final
            row["hiddenness_status"] = "not_tested"
            try:
                traj, backend_label, lm, history_contract = _integrate_case(
                    x0,
                    case,
                    settings=settings,
                    p=p,
                    q=q,
                    h=h,
                    t_final=t_final,
                    efork_backend=efork_backend,
                    abm_backend=abm_backend,
                )
                classification = classify_early_periodicity(traj, h, {**settings, "t_transient": t_transient})
                row.update({key: value for key, value in classification.items() if key != "component_metrics"})
                row["early_memory_length"] = lm
                row["early_backend"] = backend_label
                row["history_contract"] = history_contract
                row["efork_stage"] = "K3=h^q*f(x_n+a31*K1+a32*K2)" if str(case.get("solver", "")).lower() == "efork3" else ""
                row["component_metrics_json"] = json.dumps(classification["component_metrics"], ensure_ascii=False)
            except Exception as exc:
                row["early_periodicity_status"] = "post_transient_integration_failed"
                row["periodicity_status"] = "post_transient_integration_failed"
                row["notes"] = f"Post-transient periodicity integration failed: {exc}"
            diagnostics.append(row)
            account_row(row)
            if case_id == primary_case_id:
                primary_row = row

        if primary_row is None:
            continue
        status = str(primary_row.get("periodicity_status", "post_transient_integration_failed"))
        if gate_before_continuation and discard and status != "nonperiodic_post_transient":
            if status == "periodic_post_transient":
                primary_row["candidate_status"] = "rejected_periodic_post_transient"
                primary_row["notes"] = (
                    "Discarded before continuation because the primary full-history EFORK "
                    "post-transient observation remains persistently narrowband."
                )
            else:
                primary_row["candidate_status"] = "rejected_dynamic_post_transient"
                primary_row["notes"] = "Discarded before continuation because the primary post-transient integration was not usable."
            rejected.append(primary_row)
        elif status == "periodic_post_transient":
            primary_row["early_periodicity_status"] = "pre_continuation_periodic"
            primary_row["notes"] = (
                "Direct integration of the harmonic seed is periodic before continuation; "
                "this is diagnostic only because the seed must first be transported through "
                "the Lure continuation to the nonlinear target system."
            )
            kept.append(primary_row)
        elif status != "nonperiodic_post_transient":
            primary_row["early_periodicity_status"] = "unusable_precontinuation_diagnostic"
            primary_row["notes"] = (
                "Direct pre-continuation integration was not usable; retained only for "
                "traceability and must not be promoted without a usable continuation output."
            )
            kept.append(primary_row)
        else:
            primary_row["early_periodicity_status"] = "nonperiodic_post_transient"
            primary_row["notes"] = (
                "Direct pre-continuation integration is nonperiodic in the primary full-history "
                "EFORK case; this prioritizes but does not validate the seed."
            )
            kept.append(primary_row)
        completed_seed_ids.add(str(base_row.get("seed_id", "")))
        save_checkpoint()

    n_periodic_primary = case_counts[primary_case_id]["n_periodic_post_transient"]
    summary = {
        "enabled": True,
        "q": q,
        "h": h,
        "t_transient": t_transient,
        "observation_time": observation_time,
        "t_final": t_final,
        "primary_case_id": primary_case_id,
        "checkpoint_path": str(checkpoint) if checkpoint is not None else "",
        "matrix_cases": cases,
        "matrix_case_counts": case_counts,
        "efork_stage": "K3=h^q*f(x_n+a31*K1+a32*K2)",
        "n_seed_bank_total": len(seed_rows),
        "n_rejected_periodic_post_transient": n_periodic_primary,
        "n_rejected_by_primary_post_transient_gate": len(rejected),
        "gate_before_continuation": gate_before_continuation,
        "n_nonperiodic_post_transient_seeds_for_continuation": sum(
            str(row.get("early_periodicity_status", "")) == "nonperiodic_post_transient" for row in kept
        ),
        "n_seeds_released_to_continuation": len(kept),
        "n_nonperiodic_seeds_for_continuation": len(kept),
        "decision": (
            "No seed is dynamically usable for continuation."
            if not kept
            else (
                "Pre-continuation dynamics recorded diagnostically; harmonic seeds retained for Lure continuation."
                if not gate_before_continuation
                else "Post-transient nonperiodic seeds retained for continuation."
            )
        ),
        "interpretation": (
            "La función descriptiva clásica, sesgada o Machado/FDF sólo propone semillas. "
            "La integración directa de una semilla armónica antes de la continuación se registra "
            "como diagnóstico, pero no la descarta: la periodicidad decisoria debe evaluarse en "
            "la salida del sistema objetivo después de la continuación. Ninguna celda prueba ocultedad."
        ),
    }
    return {"kept_seeds": kept, "rejected_seeds": rejected, "diagnostics": diagnostics, "summary": summary}
