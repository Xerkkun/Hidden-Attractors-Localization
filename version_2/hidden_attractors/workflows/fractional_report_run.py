"""Fresh q=0.9998 report run using native C for dynamic evidence.

The describing-function scan is intentionally lightweight Python algebra and
quadrature.  Every integration-dependent quantity promoted into validation is
computed through the corrected native EFORK backends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..analysis.trajectory import trajectory_metrics
from ..io import read_csv_rows, read_json, safe_name, timestamp, write_csv, write_json
from ..models.chua import equilibria_nonsmooth
from ..native.backends import FractionalChuaBackend, FullHistoryABMBackend
from ..paths import OUTPUTS, PROJECT_ROOT, RUNTIME_CACHE
from ..plotting.dynamics import plot_phase_projections, plot_phase_space, plot_time_series


Q = 0.9998
EFORK_STAGE = "K3 = a31*K1 + a32*K2"
FULL_HISTORY_POLICY = "full_caputo_history_no_finite_memory_truncation"
CONTINUATION_POLICY = "full_generated_homotopy_history_carried_without_truncation"
FINITE_WINDOW_POLICY = "finite_caputo_history_window"
FINITE_CONTINUATION_POLICY = "finite_terminal_window_carried_across_homotopy"
APERIODIC_RETURN_THRESHOLD = 0.05
PARAMETERS = {
    "model": "nonsmooth",
    "alpha": 8.4562,
    "beta": 12.0732,
    "gamma": 0.0052,
    "m0": -0.1768,
    "m1": -1.1468,
}


def _configure_runtime() -> None:
    cache = RUNTIME_CACHE / "fractional_report"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))


def _legacy_modules() -> tuple[Any, Any, Any, Any]:
    legacy = PROJECT_ROOT / "tools" / "legacy"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    import chua_initial_cond as chua  # type: ignore
    from biased_describing_function import search_biased_candidates  # type: ignore
    from extended_search_utils import chua_ic_params, load_config  # type: ignore
    from harmonic_diagnostics import centered_candidates_from_nyquist, rho_h_diagnostic  # type: ignore

    return chua, search_biased_candidates, chua_ic_params, (load_config, centered_candidates_from_nyquist, rho_h_diagnostic)


def _method_label(df_family: str) -> str:
    labels = {
        "classical": "DF centrada",
        "classical_biased": "Lur'e sesgada",
        "machado": "Machado centrada",
        "machado_biased": "Machado sesgada",
    }
    return labels.get(str(df_family), str(df_family))


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _short_seed(row: dict[str, Any]) -> np.ndarray:
    if "seed" in row:
        return np.asarray(row["seed"], dtype=float)
    return np.asarray([row["seed_x"], row["seed_y"], row["seed_z"]], dtype=float)


def _full_history_horizon(t_final: float, h: float) -> float:
    """Return an EFORK storage horizon that cannot truncate a run from t=0."""

    return float(t_final) + float(h)


def _dominant_period_return_ratio(
    trajectory: np.ndarray,
    *,
    h: float,
    t_start: float,
    dominant_frequency: float,
) -> tuple[float, int]:
    """Return normalized closure error at one dominant sampled period.

    Small values identify nearly closed thin traces and reject promotion as a
    chaotic candidate.  Passing this exclusion gate is not, by itself,
    evidence of chaos.
    """

    tail = trajectory[trajectory[:, 0] >= t_start, 1:4]
    if tail.shape[0] < 4 or not math.isfinite(dominant_frequency) or dominant_frequency <= 0.0:
        return float("nan"), 0
    lag = max(1, int(round(1.0 / (float(h) * float(dominant_frequency)))))
    if lag >= tail.shape[0]:
        return float("nan"), lag
    scale = float(np.sqrt(np.mean(np.sum((tail - tail.mean(axis=0)) ** 2, axis=1))))
    if not math.isfinite(scale) or scale <= 0.0:
        return float("nan"), lag
    closure = float(np.sqrt(np.mean(np.sum((tail[lag:] - tail[:-lag]) ** 2, axis=1))))
    return closure / scale, lag


def generate_lightweight_df_pool(outdir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate fresh DF seeds without long trajectory integrations."""

    started = time.time()
    chua, search_biased_candidates, chua_ic_params, imported = _legacy_modules()
    load_config, centered_candidates_from_nyquist, rho_h_diagnostic = imported
    cfg = load_config(PROJECT_ROOT / "configs" / "chua_fractional_nonsmooth.yaml")
    cfg["rho_H"]["n_quad"] = 1024
    cfg["rho_H"]["K"] = 10
    cfg["biased_search"]["lhs_count"] = 24
    cfg["biased_search"]["keep_best"] = 12
    p = chua_ic_params(cfg)
    chua.PARAMS = p
    chua.QORD = np.float64(Q)

    rows: list[dict[str, Any]] = []
    for source in centered_candidates_from_nyquist(cfg, p):
        diag = rho_h_diagnostic(
            candidate_id=f"current_{source['candidate_id']}",
            df_family=str(source["df_family"]),
            A=float(source["A"]),
            sigma0=0.0,
            omega=float(source["omega"]),
            q=Q,
            p=p,
            mu=None if source.get("mu") in {"", None} else float(source["mu"]),
            K=10,
            n_quad=1024,
            threshold=float(cfg["rho_H"]["threshold"]),
        )
        W = complex(chua.W_frac(float(source["omega"]), Q, p))
        gain = float(-1.0 / np.real(W))
        seed, _v, _eig = chua.build_fractional_seed(Q, p, float(source["omega"]), gain, float(source["A"]))
        rows.append(
            {
                **{key: value for key, value in diag.items() if key not in {"fourier", "base_N"}},
                "candidate_id": f"current_{source['candidate_id']}",
                "method": _method_label(str(source["df_family"])),
                "gain_k": gain,
                "theta": "",
                "seed_x": float(seed[0]),
                "seed_y": float(seed[1]),
                "seed_z": float(seed[2]),
                "analytical_backend": "python_lightweight_df",
            }
        )

    biased, _all_rows = search_biased_candidates(cfg, p, outdir / "df_scan")
    for source in biased:
        row = dict(source)
        row["candidate_id"] = "current_" + str(source["candidate_id"])
        row["method"] = _method_label(str(source["df_family"]))
        row["gain_k"] = float(source["N_re"])
        row["theta"] = ""
        row["analytical_backend"] = "python_lightweight_df"
        rows.append(row)

    rows.sort(key=lambda row: (_finite(row.get("residual_abs"), 1e99), _finite(row.get("rho_H"), 1e99)))
    screened = rows
    write_csv(outdir / "df_candidate_pool.csv", rows)
    metadata = {
        "method": "Python ligero: barrido algebraico y cuadratura DF, sin integración temporal",
        "elapsed_sec": time.time() - started,
        "n_candidates": len(rows),
        "n_screened_in_c": len(screened),
        "q": Q,
        "quadrature_points": 1024,
        "harmonics": 10,
    }
    write_json(outdir / "df_generation_metadata.json", metadata)
    return screened, metadata


def _write_trajectory(path: Path, traj: np.ndarray, max_rows: int = 5000) -> None:
    data = np.asarray(traj, dtype=float)
    if data.shape[0] > max_rows:
        idx = np.linspace(0, data.shape[0] - 1, max_rows).astype(int)
        data = data[idx]
    write_csv(path, [{"t": row[0], "x": row[1], "y": row[2], "z": row[3]} for row in data], ["t", "x", "y", "z"])


def _read_trajectory(path: Path) -> np.ndarray:
    return np.asarray(
        [[float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])] for row in read_csv_rows(path)],
        dtype=float,
    )


def _continued_observation(
    backend: FractionalChuaBackend,
    row: dict[str, Any],
    *,
    h: float,
    memory_length: float,
    t_final: float,
    full_history: bool,
) -> tuple[dict[str, np.ndarray], float, str, str]:
    eps = np.linspace(0.0, 1.0, 5)
    stage_time = 4.0 + 6.0
    horizon = (
        _full_history_horizon(float(eps.size) * stage_time + t_final, h)
        if full_history
        else memory_length
    )
    memory_policy = FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY
    continuation_policy = CONTINUATION_POLICY if full_history else FINITE_CONTINUATION_POLICY
    cont = backend.continue_efork3(
        _short_seed(row),
        eps_values=eps,
        q=Q,
        k=float(row["gain_k"]),
        h=h,
        Lm=horizon,
        t_transient=4.0,
        t_keep=6.0,
        t_observe=t_final,
    )
    return cont, horizon, memory_policy, continuation_policy


def screen_candidates_with_c(
    outdir: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    h: float,
    memory_length: float,
    t_final: float,
    full_history: bool,
) -> list[dict[str, Any]]:
    suffix = "full" if full_history else "window"
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_efork_{suffix}")
    continuation_rows: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    for row in candidates:
        seed = _short_seed(row)
        cont, trajectory_horizon, memory_policy, continuation_policy = _continued_observation(
            backend, row, h=h, memory_length=memory_length, t_final=t_final, full_history=full_history
        )
        for index, eta in enumerate(cont["epsilon"]):
            state = cont["x_out"][index]
            continuation_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "method": row["method"],
                    "epsilon": float(eta),
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "z": float(state[2]),
                    "history_in": int(cont["history_in_counts"][index]),
                    "history_out": int(cont["history_out_counts"][index]),
                    "backend": "chua_frac_backend_lib.c",
                    "memory_policy": continuation_policy,
                    "history_horizon": trajectory_horizon,
                }
            )
        start = cont["x_out"][-1]
        traj = cont["observation"]
        metric, _payload = trajectory_metrics(traj, h=h, t_start=0.5 * t_final)
        continuation_eligible = bool(
            metric["bounded"]
            and metric["noncollapsed_variance"]
            and not metric["equilibrium_like"]
            and _finite(metric.get("range_x"), 0.0) >= 0.25
        )
        return_ratio, return_lag = _dominant_period_return_ratio(
            traj,
            h=h,
            t_start=0.5 * t_final,
            dominant_frequency=_finite(metric.get("fft_peak")),
        )
        aperiodic_gate = bool(
            continuation_eligible
            and math.isfinite(return_ratio)
            and return_ratio >= APERIODIC_RETURN_THRESHOLD
        )
        item = {
            **row,
            **metric,
            "q": Q,
            "seed": seed.tolist(),
            "robust_start": start.tolist(),
            "backend": "chua_frac_backend_lib.c",
            "efork_stage": EFORK_STAGE,
            "h": h,
            "memory_length": trajectory_horizon,
            "memory_policy": memory_policy,
            "continuation_memory_policy": continuation_policy,
            "continuation_history_horizon": trajectory_horizon,
            "t_final": t_final,
            "continuation_eligible": continuation_eligible,
            "dominant_period_return_ratio": return_ratio,
            "dominant_period_return_lag": return_lag,
            "aperiodic_threshold": APERIODIC_RETURN_THRESHOLD,
            "aperiodic_gate": aperiodic_gate,
            "eligible": aperiodic_gate,
        }
        screened.append(item)
        _write_trajectory(outdir / "trajectories" / f"{safe_name(str(row['candidate_id']))}.csv", traj)
    screened.sort(
        key=lambda item: (
            not bool(item["eligible"]),
            _finite(item.get("residual_abs"), 1e99),
            _finite(item.get("rho_H"), 1e99),
            -_finite(item.get("range_x"), -1e99),
        )
    )
    write_csv(outdir / "continuation_paths.csv", continuation_rows)
    write_csv(outdir / "candidate_dynamic_screen.csv", screened)
    return screened


def select_top_three(
    outdir: Path,
    screened: Sequence[dict[str, Any]],
    run_id: str,
    provenance: dict[str, Any],
    *,
    branch_id: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    for row in screened:
        if not bool(row["eligible"]):
            continue
        seed = np.asarray(row["seed"], dtype=float)
        start = np.asarray(row["robust_start"], dtype=float)
        if any(
            np.linalg.norm(seed - np.asarray(old["seed"], dtype=float)) <= 1.0e-6
            and np.linalg.norm(start - np.asarray(old["robust_start"], dtype=float)) <= 1.0e-6
            for old in unique_rows
        ):
            continue
        unique_rows.append(row)
        if len(unique_rows) == 3:
            break
    selection_policy = (
        "distinct EFORK C trajectories passing bounded nontrivial screening and "
        f"dominant-period return ratio >= {APERIODIC_RETURN_THRESHOLD}; "
        "then residual_abs, rho_H and range_x"
    )
    if len(unique_rows) < 3:
        write_json(
            outdir / "selected_candidates.json",
            {
                "run_id": run_id,
                **provenance,
                "q": Q,
                "branch_id": branch_id,
                "selection_status": "insufficient_aperiodic_candidates",
                "selection_policy": selection_policy,
                "selected_candidates": [],
            },
        )
        write_csv(outdir / "selected_candidates.csv", [])
        raise RuntimeError(
            "No se obtuvieron tres candidatos no triviales que superen el filtro "
            "de retorno casi periódico; no procede ejecutar ocultedad."
        )
    for rank, row in enumerate(unique_rows, 1):
        selected.append(
            {
                "rank": rank,
                "branch_id": branch_id,
                "candidate_id": row["candidate_id"],
                "method": row["method"],
                "q": Q,
                "mu": row.get("mu", ""),
                "theta": row.get("theta", ""),
                "A": row.get("A", ""),
                "sigma0": row.get("sigma0", ""),
                "omega": row.get("omega", ""),
                "gain_k": row.get("gain_k", ""),
                "rho_H": row.get("rho_H", ""),
                "residual_abs": row.get("residual_abs", ""),
                "seed": row["seed"],
                "robust_start": row["robust_start"],
                "h": row["h"],
                "memory_length": row["memory_length"],
                "memory_policy": row["memory_policy"],
                "continuation_memory_policy": row["continuation_memory_policy"],
                "t_final": row["t_final"],
                "range_x": row.get("range_x", ""),
                "fft_peak": row.get("fft_peak", ""),
                "psd_entropy": row.get("psd_entropy", ""),
                "dominant_period_return_ratio": row.get("dominant_period_return_ratio", ""),
                "aperiodic_threshold": APERIODIC_RETURN_THRESHOLD,
                "aperiodic_gate": bool(row["aperiodic_gate"]),
                "eligible": bool(row["eligible"]),
                "backend": row["backend"],
                "efork_stage": EFORK_STAGE,
                "run_id": run_id,
                "repository_commit": provenance["repository_commit"],
                "working_tree_diff_sha256": provenance["working_tree_diff_sha256"],
            }
        )
    write_json(
        outdir / "selected_candidates.json",
        {
            "run_id": run_id,
            **provenance,
            "q": Q,
            "branch_id": branch_id,
            "selection_status": "promoted_for_hiddenness",
            "selection_policy": selection_policy,
            "selected_candidates": selected,
        },
    )
    write_csv(outdir / "selected_candidates.csv", selected)
    return selected


def run_dynamic_evidence(
    outdir: Path,
    selected: Sequence[dict[str, Any]],
    *,
    h: float,
    memory_length: float,
    t_final: float,
    full_history: bool,
    trajectory_source: Path,
) -> dict[str, Any]:
    memory_policy = FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY
    metric_rows: list[dict[str, Any]] = []
    lyapunov_rows: list[dict[str, Any]] = []
    for row in selected:
        traj = _read_trajectory(trajectory_source / f"{safe_name(str(row['candidate_id']))}.csv")
        metric, _payload = trajectory_metrics(traj, h=h, t_start=t_final / 2.0)
        metric_rows.append({"candidate_id": row["candidate_id"], **metric, "backend": "chua_frac_backend_lib.c", "memory_policy": memory_policy, "history_horizon": row["memory_length"]})
        path = outdir / "trajectories" / f"selected_{int(row['rank'])}.csv"
        _write_trajectory(path, traj)
        if int(row["rank"]) == 1:
            plot_phase_space(traj, outdir / "phase_3d.png", title="Candidato mejor clasificado, EFORK-3 C")
            plot_phase_projections(traj, outdir / "projections.png", title="Proyecciones del candidato mejor clasificado")
            plot_time_series(traj, outdir / "time_series.png", title="Serie temporal del candidato mejor clasificado")
            _plot_spectrum(traj, h, outdir / "spectrum_x.png")
        lyapunov_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "status": "pending_causal_history_lyapunov_backend",
                "reason": "El backend Lyapunov actual reinicia la historia y no se promueve para candidatos obtenidos por continuacion causal.",
                "memory_policy": memory_policy,
            }
        )
    write_csv(outdir / "trajectory_metrics.csv", metric_rows)
    write_csv(outdir / "lyapunov_summary.csv", lyapunov_rows)
    write_csv(outdir / "fft_summary.csv", [{"candidate_id": row["candidate_id"], "fft_peak": row["fft_peak"]} for row in metric_rows])
    write_csv(outdir / "psd_summary.csv", [{"candidate_id": row["candidate_id"], "psd_entropy": row["psd_entropy"]} for row in metric_rows])
    return {"trajectory_metrics": metric_rows, "lyapunov": lyapunov_rows}


def _plot_spectrum(traj: np.ndarray, h: float, output: Path) -> None:
    import matplotlib.pyplot as plt

    tail = traj[traj[:, 0] >= 0.5 * float(traj[-1, 0]), 1]
    data = tail - np.mean(tail)
    spectrum = np.abs(np.fft.rfft(data * np.hanning(data.size))) ** 2
    freq = np.fft.rfftfreq(data.size, h)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.semilogy(freq[1:], np.maximum(spectrum[1:], 1e-30), color="#2563eb")
    ax.set_xlabel("Frecuencia")
    ax.set_ylabel("Potencia FFT")
    ax.set_title("Espectro de x(t), trayectoria EFORK-3 C")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run_robustness_evidence(outdir: Path, selected: Sequence[dict[str, Any]], *, full_history: bool, memory_length: float) -> dict[str, Any]:
    suffix = "full" if full_history else "window"
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_robustness_{suffix}")
    if full_history:
        cases = [("R0_base", 0.02, 80.0, None), ("R1_h_fino", 0.01, 60.0, None), ("R2_h_fino_largo", 0.01, 80.0, None), ("R3_tiempo", 0.02, 120.0, None)]
        memory_policy = FULL_HISTORY_POLICY
    else:
        cases = [("R0_base", 0.02, 80.0, memory_length), ("R1_h_fino", 0.01, 60.0, memory_length), ("R2_memoria", 0.02, 80.0, memory_length * 1.5), ("R3_tiempo", 0.02, 120.0, memory_length)]
        memory_policy = FINITE_WINDOW_POLICY
    rows: list[dict[str, Any]] = []
    best_trajs: list[np.ndarray] = []
    best_labels: list[str] = []
    for cand in selected:
        reference: dict[str, Any] | None = None
        for case_id, h, tfinal, configured_lm in cases:
            branch_lm = memory_length if full_history else float(configured_lm)
            cont, lm, _policy, _continuation_policy = _continued_observation(
                backend, cand, h=h, memory_length=branch_lm, t_final=tfinal, full_history=full_history
            )
            traj = cont["observation"]
            metric, payload = trajectory_metrics(traj, h=h, t_start=tfinal / 2.0, reference=reference)
            rows.append({"candidate_id": cand["candidate_id"], "case_id": case_id, "q": Q, "h": h, "history_horizon": lm, "memory_policy": memory_policy, "t_final": tfinal, **metric})
            if reference is None:
                reference = payload
            if int(cand["rank"]) == 1:
                best_trajs.append(traj)
                best_labels.append(case_id)
    write_csv(outdir / "robustness_overlay_metrics.csv", rows)
    _plot_robustness_overlay(best_trajs, best_labels, outdir / "overlay_3d.png")
    summary = {
        "status": "completed",
        "method": "C EFORK trajectory comparison under its stated memory contract",
        "backend": "chua_frac_backend_lib.c",
        "efork_stage": EFORK_STAGE,
        "rows": len(rows),
        "memory_policy": memory_policy,
        "notes": "This measures finite-time geometric persistence; hiddenness is evaluated separately with the same memory contract.",
    }
    write_json(outdir / "robustness_summary.json", summary)
    return summary


def _plot_robustness_overlay(trajectories: Sequence[np.ndarray], labels: Sequence[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for traj, label in zip(trajectories, labels):
        tail = traj[traj[:, 0] >= 0.5 * traj[-1, 0]]
        idx = np.linspace(0, tail.shape[0] - 1, min(tail.shape[0], 1200)).astype(int)
        data = tail[idx]
        ax.plot(data[:, 1], data[:, 2], data[:, 3], lw=0.65, alpha=0.78, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Robustez geométrica, trayectorias C")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run_hiddenness_evidence(
    outdir: Path,
    selected: Sequence[dict[str, Any]],
    *,
    full_history: bool,
    memory_length: float,
    trajectory_source: Path,
) -> dict[str, Any]:
    suffix = "full" if full_history else "window"
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_hiddenness_{suffix}")
    equilibria = equilibria_nonsmooth()
    directions = np.eye(3).tolist() + (-np.eye(3)).tolist()
    radii = [1.0e-4, 1.0e-3]
    h = 0.02
    t_final = 30.0
    t_burn = 15.0
    history_horizon = _full_history_horizon(t_final, h) if full_history else memory_length
    memory_policy = FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY
    plan: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for cand in selected:
        reference_traj = _read_trajectory(trajectory_source / f"{safe_name(str(cand['candidate_id']))}.csv")
        reference_traj = reference_traj[reference_traj[:, 0] >= float(reference_traj[-1, 0]) - t_final].copy()
        reference_traj[:, 0] -= reference_traj[0, 0]
        _reference_metric, reference = trajectory_metrics(reference_traj, h=h, t_start=t_burn)
        for eq_id, center in equilibria.items():
            for radius in radii:
                for sample_id, direction in enumerate(directions):
                    x0 = center + radius * np.asarray(direction, dtype=float)
                    item = {
                        "candidate_id": cand["candidate_id"],
                        "equilibrium_id": eq_id,
                        "radius": radius,
                        "sample_id": sample_id,
                        "x0": float(x0[0]),
                        "y0": float(x0[1]),
                        "z0": float(x0[2]),
                    }
                    plan.append(item)
                    traj = backend.integrate_efork3(x0, q=Q, h=h, Lm=history_horizon, t_final=t_final)
                    metrics, _payload = trajectory_metrics(traj, h=h, t_start=t_burn, reference=reference)
                    tail_end = traj[-1, 1:4]
                    nearest = min(float(np.linalg.norm(tail_end - eq)) for eq in equilibria.values())
                    target_hit = bool(
                        metrics["bounded"]
                        and metrics["noncollapsed_variance"]
                        and _finite(metrics.get("cloud_median_distance_norm"), 1e99) <= 0.35
                        and _finite(metrics.get("range_relative_distance"), 1e99) <= 0.60
                    )
                    label = "target_candidate" if target_hit else ("equilibrium" if nearest <= 1.0e-3 else "no_target_match")
                    raw.append(
                        {
                            **item,
                            "class_label": label,
                            "target_hit": target_hit,
                            "cloud_median_distance_norm": metrics["cloud_median_distance_norm"],
                            "range_relative_distance": metrics["range_relative_distance"],
                            "fft_relative_delta": metrics["fft_relative_delta"],
                            "nearest_equilibrium_distance": nearest,
                            "backend": "chua_frac_backend_lib.c",
                            "memory_policy": memory_policy,
                        }
                    )
    write_csv(outdir / "sphere_plan.csv", plan)
    write_csv(outdir / "sphere_raw.csv", raw)
    decisions: list[dict[str, Any]] = []
    contract_label = "historial completo" if full_history else "ventana finita"
    for cand in selected:
        rows = [row for row in raw if row["candidate_id"] == cand["candidate_id"]]
        hits = sum(1 for row in rows if bool(row["target_hit"]))
        decisions.append(
            {
                "candidate_id": cand["candidate_id"],
                "tested_trajectories": len(rows),
                "target_hits": hits,
                "hiddenness_status": "not_supported_by_tested_neighborhoods" if hits else "compatible_under_tested_neighborhoods",
                "contract_note": f"Una coincidencia refuta ocultedad bajo el contrato EFORK de {contract_label}; cero impactos no es demostracion.",
            }
        )
    write_csv(outdir / "sphere_decision.csv", decisions)
    summary = {
        "status": "completed",
        "backend": "chua_frac_backend_lib.c",
        "efork_stage": EFORK_STAGE,
        "contract": {"q": Q, "h": h, "history_horizon": history_horizon, "memory_policy": memory_policy, "t_final": t_final, "t_burn": t_burn, "radii": radii, "directions": "axis_six"},
        "decisions": decisions,
        "basins_and_bifurcations": "pending",
    }
    write_json(outdir / "hiddenness_run_summary.json", summary)
    run_strict_refinement(outdir, selected, full_history=full_history, memory_length=memory_length, trajectory_source=trajectory_source)
    return summary


def run_strict_refinement(
    outdir: Path,
    selected: Sequence[dict[str, Any]],
    *,
    full_history: bool,
    memory_length: float,
    trajectory_source: Path,
) -> None:
    """Reintegrate target hits and compare them with native-C references."""

    suffix = "full" if full_history else "window"
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_strict_refinement_{suffix}")
    raw = read_csv_rows(outdir / "sphere_raw.csv")
    rows: list[dict[str, Any]] = []
    h = 0.02
    t_final = 30.0
    history_horizon = _full_history_horizon(t_final, h) if full_history else memory_length
    memory_policy = FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY
    fields = [
        "candidate_id",
        "equilibrium_id",
        "radius",
        "sample_id",
        "cloud_median_distance_norm",
        "range_relative_distance",
        "fft_relative_delta",
        "matched_candidate_reference",
        "backend",
        "memory_policy",
        "status",
        "input_target_hits",
    ]
    for cand in selected:
        reference_traj = _read_trajectory(trajectory_source / f"{safe_name(str(cand['candidate_id']))}.csv")
        reference_traj = reference_traj[reference_traj[:, 0] >= float(reference_traj[-1, 0]) - t_final].copy()
        reference_traj[:, 0] -= reference_traj[0, 0]
        _metric, reference = trajectory_metrics(reference_traj, h=h, t_start=15.0)
        hits = [
            row for row in raw
            if row["candidate_id"] == cand["candidate_id"] and str(row.get("target_hit", "")).lower() == "true"
        ][:6]
        for hit in hits:
            start = [float(hit["x0"]), float(hit["y0"]), float(hit["z0"])]
            traj = backend.integrate_efork3(start, q=Q, h=h, Lm=history_horizon, t_final=t_final)
            metrics, _payload = trajectory_metrics(traj, h=h, t_start=15.0, reference=reference)
            match = bool(
                _finite(metrics.get("cloud_median_distance_norm"), 1e99) <= 0.35
                and _finite(metrics.get("range_relative_distance"), 1e99) <= 0.60
            )
            rows.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "equilibrium_id": hit["equilibrium_id"],
                    "radius": hit["radius"],
                    "sample_id": hit["sample_id"],
                    "cloud_median_distance_norm": metrics["cloud_median_distance_norm"],
                    "range_relative_distance": metrics["range_relative_distance"],
                    "fft_relative_delta": metrics["fft_relative_delta"],
                    "matched_candidate_reference": match,
                    "backend": "chua_frac_backend_lib.c",
                    "memory_policy": memory_policy,
                    "status": "refined_target_hit",
                    "input_target_hits": 1,
                }
            )
    if not rows:
        rows.append(
            {
                "candidate_id": "all_selected_candidates",
                "matched_candidate_reference": False,
                "backend": "chua_frac_backend_lib.c",
                "memory_policy": memory_policy,
                "status": "not_executed_no_target_hits",
                "input_target_hits": 0,
            }
        )
    write_csv(outdir / "strict_refinement_summary.csv", rows, fields)


def run_danca_abm_control(outdir: Path) -> dict[str, Any]:
    """Fresh ABM full-history control for the Danca-reported route only."""

    backend = FullHistoryABMBackend.build(output_name="danca_abm_full_history_control")
    seed = np.asarray([3.039383584794975, -0.2416862069577155, -6.873467365218827], dtype=float)
    h = 0.05
    t_final = 80.0
    t_burn = 40.0
    reference_traj = backend.integrate(seed, q=Q, h=h, t_final=t_final)
    reference_metric, reference = trajectory_metrics(reference_traj, h=h, t_start=t_burn)
    _write_trajectory(outdir / "danca_reference_abm_full_history.csv", reference_traj)
    equilibria = backend.equilibria()
    directions = np.eye(3).tolist() + (-np.eye(3)).tolist()
    rows: list[dict[str, Any]] = []
    for eq_id in ("E+", "E-"):
        center = equilibria[eq_id]
        for sample_id, direction in enumerate(directions):
            x0 = center + 1.0e-2 * np.asarray(direction, dtype=float)
            traj = backend.integrate(x0, q=Q, h=h, t_final=t_final)
            metrics, _payload = trajectory_metrics(traj, h=h, t_start=t_burn, reference=reference)
            target_hit = bool(
                metrics["bounded"]
                and metrics["noncollapsed_variance"]
                and _finite(metrics.get("cloud_median_distance_norm"), 1e99) <= 0.35
                and _finite(metrics.get("range_relative_distance"), 1e99) <= 0.60
            )
            rows.append(
                {
                    "equilibrium_id": eq_id,
                    "delta": 1.0e-2,
                    "sample_id": sample_id,
                    "x0": x0.tolist(),
                    "target_hit": target_hit,
                    "cloud_median_distance_norm": metrics["cloud_median_distance_norm"],
                    "range_relative_distance": metrics["range_relative_distance"],
                    "backend": "chua_abm_full_history_lib.c",
                    "memory_policy": FULL_HISTORY_POLICY,
                }
            )
    write_csv(outdir / "danca_abm_hiddenness_controls.csv", rows)
    hits = sum(1 for row in rows if row["target_hit"])
    summary = {
        "status": "completed_exploratory_current_run",
        "scope": "Danca-reported route only; excluded from ranking of new EFORK candidates",
        "candidate_seed_source": "Danca ABM replication locator from prior method setup; all reported metrics are newly recomputed",
        "backend": "chua_abm_full_history_lib.c",
        "method": "Caputo ABM PECE with full history and no finite-memory truncation",
        "contract": {"q": Q, "h": h, "t_final": t_final, "t_burn": t_burn, "delta": 1.0e-2, "tested_directions_per_unstable_equilibrium": 6},
        "reference_metrics": reference_metric,
        "tested_trajectories": len(rows),
        "target_hits": hits,
        "decision": "not_supported_by_current_abm_control" if hits else "compatible_under_current_abm_control",
    }
    write_json(outdir / "abm_replication_summary.json", summary)
    return summary


def _copy_files(source: Path, destination: Path, names: Sequence[str]) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in names:
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)
            copied[name] = name
    return copied


def promote_validation(
    run_root: Path,
    run_id: str,
    provenance: dict[str, Any],
    df_metadata: dict[str, Any],
    branch_results: dict[str, dict[str, Any]],
    danca_summary: dict[str, Any],
) -> None:
    validation = PROJECT_ROOT / "validation"
    candidate_dir = validation / "04_candidates"
    dynamic_dir = validation / "05_dynamic_analysis"
    hidden_dir = validation / "06_hiddenness"
    robust_dir = validation / "07_robustness"
    literature_dir = validation / "08_literature_comparison"
    full = branch_results["full_history"]
    window = branch_results["finite_window"]
    full_root = run_root / "full_history"
    window_root = run_root / "finite_window"
    _copy_files(run_root, candidate_dir, ["df_candidate_pool.csv", "df_generation_metadata.json"])
    _copy_files(full_root, candidate_dir, ["selected_candidates.json", "selected_candidates.csv", "candidate_dynamic_screen.csv", "continuation_paths.csv"])
    for source_name, target_name in (
        ("selected_candidates.json", "selected_candidates_finite_window.json"),
        ("selected_candidates.csv", "selected_candidates_finite_window.csv"),
        ("candidate_dynamic_screen.csv", "candidate_dynamic_screen_finite_window.csv"),
        ("continuation_paths.csv", "continuation_paths_finite_window.csv"),
    ):
        shutil.copy2(window_root / source_name, candidate_dir / target_name)
    _copy_files(full_root / "dynamic", dynamic_dir, ["trajectory_metrics.csv", "fft_summary.csv", "psd_summary.csv", "lyapunov_summary.csv", "phase_3d.png", "projections.png", "time_series.png", "spectrum_x.png"])
    for name in ("trajectory_metrics.csv", "fft_summary.csv", "psd_summary.csv", "lyapunov_summary.csv", "phase_3d.png", "projections.png", "time_series.png", "spectrum_x.png"):
        shutil.copy2(window_root / "dynamic" / name, dynamic_dir / f"finite_window_{name}")
    _copy_files(full_root / "hiddenness", hidden_dir, ["sphere_plan.csv", "sphere_raw.csv", "sphere_decision.csv", "strict_refinement_summary.csv", "hiddenness_run_summary.json"])
    for name in ("sphere_plan.csv", "sphere_raw.csv", "sphere_decision.csv", "strict_refinement_summary.csv", "hiddenness_run_summary.json"):
        shutil.copy2(window_root / "hiddenness" / name, hidden_dir / f"finite_window_{name}")
    _copy_files(full_root / "robustness", robust_dir, ["robustness_overlay_metrics.csv", "robustness_summary.json", "overlay_3d.png"])
    for name in ("robustness_overlay_metrics.csv", "robustness_summary.json", "overlay_3d.png"):
        shutil.copy2(window_root / "robustness" / name, robust_dir / f"finite_window_{name}")
    _copy_files(run_root / "danca_abm_control", literature_dir, ["danca_abm_hiddenness_controls.csv", "danca_reference_abm_full_history.csv", "abm_replication_summary.json"])
    write_csv(candidate_dir / "q_sweep_summary.csv", [{"q": Q, "status": "fixed_order_current_run", "run_id": run_id}])
    combined = list(full["selected"]) + list(window["selected"])
    write_csv(candidate_dir / "df_compare_summary.csv", [{"branch_id": row["branch_id"], "method": row["method"], "candidate_id": row["candidate_id"], "residual_abs": row["residual_abs"], "rho_H": row["rho_H"]} for row in combined])
    write_csv(candidate_dir / "machado_sweep_summary.csv", [{"branch_id": row["branch_id"], "candidate_id": row["candidate_id"], "method": row["method"], "mu": row.get("mu", ""), "selected": True} for row in combined if "Machado" in str(row["method"])])
    candidate_md = rf"""# Selección de candidatos fraccionarios actuales

Se ejecutó la corrida `{run_id}` para el Chua no suave con \(q=0.9998\). La
etapa DF utiliza cuadratura Python de corta duración ({df_metadata['elapsed_sec']:.3f} s);
la continuación y la clasificación dinámica que determinan cada selección se
ejecutaron con `chua_frac_backend_lib.c` y el estadio corregido `{EFORK_STAGE}`.

Se producen dos ternas nuevas: `selected_candidates.csv` corresponde a EFORK
con historial completo transportado, mientras que
`selected_candidates_finite_window.csv` corresponde a EFORK con ventana
finita \(L_m=8\). No se utilizaron salidas históricas para sus valores
numéricos.
"""
    (candidate_dir / "candidate_selection.md").write_text(candidate_md, encoding="utf-8")
    write_json(candidate_dir / "candidate_selection_summary.json", {"stage": "candidate_selection", "status": "passed", "run_id": run_id, **provenance, "files": {"full_history_selection": "selected_candidates.json", "full_history_table": "selected_candidates.csv", "finite_window_selection": "selected_candidates_finite_window.json", "finite_window_table": "selected_candidates_finite_window.csv"}, "python_lightweight_df": df_metadata, "dynamic_backend": "chua_frac_backend_lib.c", "efork_stage": EFORK_STAGE, "branches": ["full_history", "finite_window"]})
    dynamic_md = f"""# Análisis dinámico fraccionario actual

Las trayectorias de ambas ternas fueron calculadas dentro del tramo de
observación de su continuación causal EFORK-3 C para `{run_id}`. El espectro
se obtiene como posprocesamiento ligero de esas trayectorias C.

Los exponentes de Lyapunov permanecen pendientes: el backend existente
reinicia por bloques y no se promueve como evidencia de una trayectoria que
transporta historia de continuación.
"""
    (dynamic_dir / "dynamic_analysis.md").write_text(dynamic_md, encoding="utf-8")
    write_json(dynamic_dir / "dynamic_analysis_summary.json", {"stage": "dynamic_analysis", "status": "passed", "run_id": run_id, "files": {"full_history_metrics": "trajectory_metrics.csv", "finite_window_metrics": "finite_window_trajectory_metrics.csv", "full_history_phase": "phase_3d.png", "finite_window_phase": "finite_window_phase_3d.png"}, "backends": ["chua_frac_backend_lib.c"], "lyapunov": "pending_causal_history_backend", "efork_stage": EFORK_STAGE})
    hidden_md = rf"""# Validación operacional de ocultedad

Para `{run_id}` se sondearon esferas discretas alrededor de cada equilibrio
mediante integración C `chua_frac_backend_lib.c`. El ensayo se realizó por
separado bajo EFORK con historial completo y bajo EFORK con ventana
\(L_m=8\). Si una condición inicial próxima a un equilibrio coincide con la
referencia candidata, la etiqueta de atractor oculto queda descartada para
ese contrato.

La ausencia de impactos sólo es evidencia compatible con ocultedad bajo las
vecindades muestreadas. Los impactos detectados se reintegran mediante C y se
comparan contra la referencia candidata en `strict_refinement_summary.csv`.
Las cuencas completas y los diagramas de bifurcación permanecen pendientes.
"""
    (hidden_dir / "hiddenness_validation.md").write_text(hidden_md, encoding="utf-8")
    write_json(hidden_dir / "hiddenness_validation_summary.json", {"stage": "hiddenness", "status": "passed", "run_id": run_id, "files": {"full_history_decision": "sphere_decision.csv", "finite_window_decision": "finite_window_sphere_decision.csv", "full_history_strict": "strict_refinement_summary.csv", "finite_window_strict": "finite_window_strict_refinement_summary.csv"}, "branches": {"full_history": full["hiddenness"], "finite_window": window["hiddenness"]}})
    robust_md = rf"""# Validación de robustez actual

La robustez se evaluó para cada rama mediante continuaciones EFORK C
independientes. La rama de historial completo varía \(h\) y horizonte sin
truncar historia; la rama de ventana finita incluye además variación de
\(L_m\).

Esta etapa verifica persistencia geométrica bajo perturbaciones numéricas; no
clasifica ocultedad.
"""
    (robust_dir / "robustness_validation.md").write_text(robust_md, encoding="utf-8")
    write_json(robust_dir / "robustness_validation_summary.json", {"stage": "robustness", "status": "passed", "run_id": run_id, "files": {"full_history_metrics": "robustness_overlay_metrics.csv", "finite_window_metrics": "finite_window_robustness_overlay_metrics.csv", "full_history_overlay": "overlay_3d.png", "finite_window_overlay": "finite_window_overlay_3d.png"}, "branches": {"full_history": full["robustness"], "finite_window": window["robustness"]}})
    write_csv(literature_dir / "danca_comparison_summary.csv", [{"method": danca_summary["method"], "scope": danca_summary["scope"], "tested_trajectories": danca_summary["tested_trajectories"], "target_hits": danca_summary["target_hits"], "decision": danca_summary["decision"]}])
    literature_md = f"""# Control independiente Danca ABM

La ruta publicada por Danca se ensayó en una etapa separada mediante ABM C de
historial completo. Este control no interviene en el ranking EFORK de los
nuevos candidatos. En la corrida `{run_id}` se ejecutaron
{danca_summary['tested_trajectories']} sondeos locales y se observaron
{danca_summary['target_hits']} coincidencias operacionales.
"""
    (literature_dir / "literature_comparison.md").write_text(literature_md, encoding="utf-8")
    write_json(literature_dir / "literature_comparison_summary.json", {"stage": "literature_comparison", "status": "passed", "run_id": run_id, "files": {"report": "literature_comparison.md", "table": "danca_comparison_summary.csv", "abm": "abm_replication_summary.json"}, "danca_abm_control": danca_summary})
    manifest_path = validation / "00_manifest" / "validation_manifest.json"
    manifest = read_json(manifest_path)
    manifest["validation_id"] = run_id
    manifest["repository_commit"] = provenance["repository_commit"]
    manifest["working_tree_dirty"] = provenance["working_tree_dirty"]
    manifest["working_tree_diff_sha256"] = provenance["working_tree_diff_sha256"]
    manifest["stages"].update({
        "candidate_selection": "04_candidates/candidate_selection_summary.json",
        "dynamic_analysis": "05_dynamic_analysis/dynamic_analysis_summary.json",
        "hiddenness": "06_hiddenness/hiddenness_validation_summary.json",
        "robustness": "07_robustness/robustness_validation_summary.json",
        "literature_comparison": "08_literature_comparison/literature_comparison_summary.json",
    })
    manifest["pending_stages"] = ["basins", "bifurcations"]
    manifest["final_report"] = {"status": "current_report_run_completed_without_basins_or_bifurcations", "run_id": run_id}
    write_json(manifest_path, manifest)


def repository_provenance() -> dict[str, Any]:
    commit = subprocess.run(["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", str(PROJECT_ROOT.parent), "diff", "--binary", "--", "version_2"], check=True, capture_output=True).stdout
    status = subprocess.run(["git", "-C", str(PROJECT_ROOT.parent), "status", "--short"], check=True, capture_output=True, text=True).stdout.strip()
    implemented_sources = [
        Path(__file__),
        PROJECT_ROOT / "hidden_attractors" / "native" / "backends.py",
        PROJECT_ROOT / "hidden_attractors" / "native" / "csrc" / "chua_frac_backend_lib.c",
        PROJECT_ROOT / "hidden_attractors" / "native" / "csrc" / "chua_abm_full_history_lib.c",
        PROJECT_ROOT / "hidden_attractors" / "native" / "csrc" / "chua_hidden_backend.c",
    ]
    implementation_hash = hashlib.sha256()
    for source in implemented_sources:
        implementation_hash.update(source.read_bytes())
    return {
        "repository_commit": commit,
        "working_tree_dirty": bool(status),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "implementation_sources_sha256": implementation_hash.hexdigest(),
    }


def run_efork_branch(
    root: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    branch_id: str,
    full_history: bool,
    args: argparse.Namespace,
    run_id: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir()
    screened = screen_candidates_with_c(
        root,
        candidates,
        h=args.h,
        memory_length=args.memory_length,
        t_final=args.t_final,
        full_history=full_history,
    )
    selected = select_top_three(root, screened, run_id, provenance, branch_id=branch_id)
    dynamic_dir = root / "dynamic"
    hidden_dir = root / "hiddenness"
    robust_dir = root / "robustness"
    dynamic_dir.mkdir()
    hidden_dir.mkdir()
    robust_dir.mkdir()
    dynamic = run_dynamic_evidence(
        dynamic_dir,
        selected,
        h=args.h,
        memory_length=args.memory_length,
        t_final=args.t_final,
        full_history=full_history,
        trajectory_source=root / "trajectories",
    )
    robustness = run_robustness_evidence(
        robust_dir,
        selected,
        full_history=full_history,
        memory_length=args.memory_length,
    )
    hiddenness = run_hiddenness_evidence(
        hidden_dir,
        selected,
        full_history=full_history,
        memory_length=args.memory_length,
        trajectory_source=root / "trajectories",
    )
    return {
        "branch_id": branch_id,
        "memory_policy": FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY,
        "selected": selected,
        "dynamic": dynamic,
        "robustness": robustness,
        "hiddenness": hiddenness,
    }


def run(args: argparse.Namespace) -> Path:
    _configure_runtime()
    run_id = args.run_id or f"chua_fractional_nonsmooth_q09998_efork3_{timestamp()}"
    root = OUTPUTS / run_id
    root.mkdir(parents=True, exist_ok=False)
    provenance = repository_provenance()
    screened_pool, df_metadata = generate_lightweight_df_pool(root)
    branch_results = {
        "full_history": run_efork_branch(
            root / "full_history", screened_pool, branch_id="full_history", full_history=True, args=args, run_id=run_id, provenance=provenance
        ),
        "finite_window": run_efork_branch(
            root / "finite_window", screened_pool, branch_id="finite_window", full_history=False, args=args, run_id=run_id, provenance=provenance
        ),
    }
    danca_dir = root / "danca_abm_control"
    danca_dir.mkdir()
    danca_summary = run_danca_abm_control(danca_dir)
    write_json(
        root / "run_metadata.json",
        {
            "run_id": run_id,
            **provenance,
            "q": Q,
            "params": PARAMETERS,
            "efork_stage": EFORK_STAGE,
            "factorial_design": {
                "describing_function_methods": ["DF centrada", "Lur'e sesgada", "Machado centrada", "Machado sesgada"],
                "efork_memory_branches": ["full_history", "finite_window"],
                "finite_window_length": args.memory_length,
            },
            "python_lightweight_df": df_metadata,
            "native_backends": ["chua_frac_backend_lib.c", "chua_abm_full_history_lib.c"],
            "branches": branch_results,
            "danca_abm_control": danca_summary,
            "basins": "pending",
            "bifurcations": "pending",
        },
    )
    promote_validation(root, run_id, provenance, df_metadata, branch_results, danca_summary)
    return root


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate current fractional Chua report evidence with corrected native EFORK backends.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--memory-length", type=float, default=8.0, help="Ventana Lm usada solamente en la rama finite_window.")
    parser.add_argument("--t-final", type=float, default=80.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    output = run(make_parser().parse_args(argv))
    print(str(output))


if __name__ == "__main__":
    main()
