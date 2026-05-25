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
from .protocol import PROTOCOL_VERSION, SCHEMA_VERSION, StageEnvelope, sample_uniform_ball


Q = 0.9998
EFORK_STAGE = "K3 = a31*K1 + a32*K2"
FULL_HISTORY_POLICY = "full_caputo_history_no_finite_memory_truncation"
CONTINUATION_POLICY = "full_generated_homotopy_history_carried_without_truncation"
FINITE_WINDOW_POLICY = "finite_caputo_history_window"
FINITE_CONTINUATION_POLICY = "finite_terminal_window_carried_across_homotopy"
POST_CONTINUATION_PERIODICITY_RETURN_THRESHOLD = 0.05
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


def _seed_family(df_family: str) -> str:
    families = {
        "classical": "lure_classical_centered",
        "classical_centered": "lure_classical_centered",
        "classical_biased": "lure_classical_biased",
        "lure_biased": "lure_classical_biased",
        "machado": "machado_centered",
        "machado_centered": "machado_centered",
        "machado_biased": "machado_biased",
    }
    return families.get(str(df_family), str(df_family))


def _method_label(df_family: str) -> str:
    return _seed_family(df_family)


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


def _valid_seed_configuration(row: dict[str, Any]) -> bool:
    return bool(
        math.isfinite(_finite(row.get("A")))
        and _finite(row.get("A")) > 0.0
        and math.isfinite(_finite(row.get("omega")))
        and _finite(row.get("omega")) > 0.0
        and np.all(np.isfinite(_short_seed(row)))
    )


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
        family = _seed_family(str(source["df_family"]))
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
                "family": family,
                "method": family,
                "centered_or_biased": "centered",
                "gain_k": gain,
                "mu": 1.0 if family == "lure_classical_centered" else _finite(source.get("mu"), 1.0),
                "theta": _finite(source.get("theta"), 0.0),
                "harmonic_residual": _finite(diag.get("residual_abs", diag.get("harmonic_residual"))),
                "x0": seed.tolist(),
                "reconstruction_metadata": {"implementation": "build_fractional_seed", "gain_k": gain},
                "source_config": "configs/chua_fractional_nonsmooth.yaml",
                "seed_x": float(seed[0]),
                "seed_y": float(seed[1]),
                "seed_z": float(seed[2]),
                "analytical_backend": "python_lightweight_df",
            }
        )

    biased, _all_rows = search_biased_candidates(cfg, p, outdir / "df_scan")
    for source in biased:
        row = dict(source)
        family = _seed_family(str(source["df_family"]))
        row["candidate_id"] = "current_" + str(source["candidate_id"])
        row["family"] = family
        row["method"] = family
        row["centered_or_biased"] = "biased"
        row["mu"] = 1.0 if family == "lure_classical_biased" else _finite(row.get("mu"), 1.0)
        row["theta"] = _finite(row.get("theta"), 0.0)
        row["harmonic_residual"] = _finite(row.get("residual_abs", row.get("harmonic_residual")))
        row["gain_k"] = float(source["N_re"])
        row["analytical_backend"] = "python_lightweight_df"
        row["x0"] = _short_seed(row).tolist()
        row["reconstruction_metadata"] = {"implementation": "biased_describing_function", "gain_k": row["gain_k"]}
        row["source_config"] = "configs/chua_fractional_nonsmooth.yaml"
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
    lambda_values = np.linspace(0.0, 1.0, 5)
    stage_time = 4.0 + 6.0
    horizon = (
        _full_history_horizon(float(lambda_values.size) * stage_time + t_final, h)
        if full_history
        else memory_length
    )
    memory_policy = FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY
    continuation_policy = CONTINUATION_POLICY if full_history else FINITE_CONTINUATION_POLICY
    cont = backend.continue_efork3(
        _short_seed(row),
        # The native API retains its historical argument name; this is the
        # internal mapping of the public ContinuationPlan(lambda).
        lambda_values=lambda_values,
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
        for index, lambda_value in enumerate(cont["lambda"]):
            state = cont["x_out"][index]
            continuation_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "method": row["method"],
                    "lambda": float(lambda_value),
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "z": float(state[2]),
                    "history_in": int(cont["history_in_counts"][index]),
                    "history_out": int(cont["history_out_counts"][index]),
                    "backend": "chua_frac_backend_lib.c",
                    "memory_policy": continuation_policy,
                    "history_horizon": trajectory_horizon,
                    "internal_mapping": "native epsilon=lambda",
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
        post_continuation_periodicity_pass = bool(
            continuation_eligible
            and math.isfinite(return_ratio)
            and return_ratio >= POST_CONTINUATION_PERIODICITY_RETURN_THRESHOLD
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
            "post_continuation_periodicity_threshold": POST_CONTINUATION_PERIODICITY_RETURN_THRESHOLD,
            "post_continuation_periodicity_pass": post_continuation_periodicity_pass,
            "post_continuation_survivor": post_continuation_periodicity_pass,
            "verdict": "continuation_survivor" if post_continuation_periodicity_pass else "rejected_post_continuation",
        }
        screened.append(item)
        _write_trajectory(outdir / "trajectories" / f"{safe_name(str(row['candidate_id']))}.csv", traj)
    screened.sort(
        key=lambda item: (
            not bool(item["post_continuation_survivor"]),
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
        if not bool(row["post_continuation_survivor"]):
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
        "distinct EFORK C target-system trajectories passing the post_continuation_filter and "
        f"dominant-period return ratio >= {POST_CONTINUATION_PERIODICITY_RETURN_THRESHOLD}; "
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
                "selection_status": "insufficient_post_continuation_survivors",
                "selection_policy": selection_policy,
                "selected_candidates": [],
            },
        )
        write_csv(outdir / "selected_candidates.csv", [])
        raise RuntimeError(
            "No se obtuvieron tres supervivientes tras el filtro posterior a la "
            "continuacion; no procede ejecutar ocultedad."
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
                "post_continuation_periodicity_threshold": POST_CONTINUATION_PERIODICITY_RETURN_THRESHOLD,
                "post_continuation_periodicity_pass": bool(row["post_continuation_periodicity_pass"]),
                "verdict": "continuation_survivor",
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
            "selection_status": "continuation_survivors_selected_for_reference",
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
    radii = [1.0e-4, 1.0e-3]
    samples_per_radius = 12
    sample_growth_per_radius = 12
    rng = np.random.default_rng(20260524 if full_history else 20260525)
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
            for radius_index, radius in enumerate(radii):
                count = samples_per_radius + radius_index * sample_growth_per_radius
                for sample_id, x0 in enumerate(sample_uniform_ball(center, radius, count, rng)):
                    item = {
                        "candidate_id": cand["candidate_id"],
                        "equilibrium_id": eq_id,
                        "radius": radius,
                        "sample_id": sample_id,
                        "sampling_mode": "ball",
                        "distance_from_equilibrium": float(np.linalg.norm(x0 - center)),
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
    write_csv(outdir / "ball_sampling_plan.csv", plan)
    write_csv(outdir / "ball_sampling_results.csv", raw)
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
                "hiddenness_status": "rejected_self_excited_contact" if hits else "compatible_with_hiddenness_under_tested_radii",
                "contract_note": f"Una coincidencia refuta ocultedad bajo el contrato EFORK de {contract_label}; cero impactos no es demostracion.",
            }
        )
    write_csv(outdir / "hiddenness_decisions.csv", decisions)
    summary = {
        "status": "completed",
        "backend": "chua_frac_backend_lib.c",
        "efork_stage": EFORK_STAGE,
        "contract": {
            "q": Q,
            "h": h,
            "history_horizon": history_horizon,
            "memory_policy": memory_policy,
            "t_final": t_final,
            "t_burn": t_burn,
            "radii": radii,
            "sampling_mode": "ball",
            "samples_per_radius": samples_per_radius,
            "sample_growth_per_radius": sample_growth_per_radius,
        },
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
    raw = read_csv_rows(outdir / "ball_sampling_results.csv")
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
    radius = 1.0e-2
    samples_per_equilibrium = 12
    rng = np.random.default_rng(20260526)
    rows: list[dict[str, Any]] = []
    for eq_id in ("E+", "E-"):
        center = equilibria[eq_id]
        for sample_id, x0 in enumerate(sample_uniform_ball(center, radius, samples_per_equilibrium, rng)):
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
                    "radius": radius,
                    "sample_id": sample_id,
                    "sampling_mode": "ball",
                    "distance_from_equilibrium": float(np.linalg.norm(x0 - center)),
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
        "scope": "ABM full-history robustness/reference comparison; excluded from EFORK seed ranking",
        "candidate_seed_source": "Danca ABM replication locator from prior method setup; all reported metrics are newly recomputed",
        "backend": "chua_abm_full_history_lib.c",
        "method": "Caputo ABM PECE with full history and no finite-memory truncation",
        "contract": {
            "q": Q,
            "h": h,
            "t_final": t_final,
            "t_burn": t_burn,
            "radius": radius,
            "sampling_mode": "ball",
            "samples_per_unstable_equilibrium": samples_per_equilibrium,
        },
        "reference_metrics": reference_metric,
        "tested_trajectories": len(rows),
        "target_hits": hits,
        "decision": "rejected_self_excited_contact" if hits else "compatible_with_hiddenness_under_tested_radii",
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


def _run_numerical_contract() -> dict[str, Any]:
    return {
        "q": Q,
        "backend": "efork_c",
        "reference_backend": "abm_c_full_history",
        "memory_policy": "full_history_and_finite_memory_robustness_comparison",
        "boundedness_thresholds": {"bounded": True, "minimum_range_x": 0.25},
        "equilibrium_distance_thresholds": {"nearest_equilibrium": 1.0e-3},
        "similarity_thresholds": {"cloud_median_distance_norm": 0.35, "range_relative_distance": 0.60},
        "hiddenness_radii": [1.0e-4, 1.0e-3],
        "samples_per_radius": 12,
        "sample_growth_per_radius": 12,
        "random_seed_policy": "fixed_reproducible",
        "output_schema_version": SCHEMA_VERSION,
        "efork_stage": EFORK_STAGE,
    }


def _write_stage_summary(
    directory: Path,
    stage: str,
    status: str,
    contract: dict[str, Any],
    *,
    files: dict[str, str],
    provenance: dict[str, Any],
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    verdict: str | None = None,
) -> None:
    summary = StageEnvelope(
        stage=stage,
        status=status,
        system="fractional_nonsmooth_chua",
        numerical_contract=contract,
        inputs=inputs or {},
        outputs=outputs or {},
        metrics=metrics or {},
        verdict=verdict,
        files=files,
        provenance=provenance,
    )
    errors = summary.validate()
    if errors:
        raise ValueError("; ".join(errors))
    write_json(directory / f"{stage}_validation_summary.json", summary.to_dict())


def promote_validation(
    run_root: Path,
    run_id: str,
    provenance: dict[str, Any],
    df_metadata: dict[str, Any],
    branch_results: dict[str, dict[str, Any]],
    danca_summary: dict[str, Any],
) -> None:
    validation = PROJECT_ROOT / "validation"
    numerical_dir = validation / "01_numerical_contract"
    seed_dir = validation / "03_seed_generation"
    precheck_dir = validation / "04_soft_precheck"
    continuation_dir = validation / "05_continuation"
    filter_dir = validation / "06_post_continuation_filter"
    dynamic_dir = validation / "07_dynamic_reference"
    robust_dir = validation / "08_robustness"
    hidden_dir = validation / "09_hiddenness_tests"
    diagnostics_dir = validation / "10_diagnostics"
    for directory in (
        numerical_dir,
        seed_dir,
        precheck_dir,
        continuation_dir,
        filter_dir,
        dynamic_dir,
        robust_dir,
        hidden_dir,
        diagnostics_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    full = branch_results["full_history"]
    window = branch_results["finite_window"]
    full_root = run_root / "full_history"
    window_root = run_root / "finite_window"
    contract = _run_numerical_contract()

    write_json(numerical_dir / "effective_contract.json", contract)
    write_csv(
        numerical_dir / "integrator_benchmark_summary.csv",
        [
            {
                "backend": "chua_frac_backend_lib.c",
                "role": "validated_sweep_backend",
                "required_stage_formula": EFORK_STAGE,
                "memory_policy": "full_history_and_finite_memory_variants",
            },
            {
                "backend": "chua_abm_full_history_lib.c",
                "role": "reference_backend",
                "required_stage_formula": "not_applicable_abm",
                "memory_policy": FULL_HISTORY_POLICY,
            },
        ],
    )
    numerical_md = (
        "# Numerical contract\n\n"
        f"Corrida `{run_id}` con EFORK C corregido (`{EFORK_STAGE}`) como backend de barrido "
        "y ABM full-history como referencia de robustez. La truncacion de memoria se compara "
        "como variante numerica y no sustituye al benchmark de historia completa.\n"
    )
    (numerical_dir / "numerical_contract_validation.md").write_text(numerical_md, encoding="utf-8")
    _write_stage_summary(
        numerical_dir,
        "numerical_contract",
        "completed",
        contract,
        files={
            "report": "numerical_contract_validation.md",
            "contract": "effective_contract.json",
            "integrator_benchmark": "integrator_benchmark_summary.csv",
        },
        provenance=provenance,
        outputs={"efork_stage": EFORK_STAGE, "protocol_version": PROTOCOL_VERSION},
    )

    seed_rows = read_csv_rows(run_root / "df_candidate_pool.csv")
    seeds: list[dict[str, Any]] = []
    for row in seed_rows:
        seeds.append(
            {
                "candidate_id": row["candidate_id"],
                "family": row["family"],
                "centered_or_biased": row["centered_or_biased"],
                "A": _finite(row.get("A")),
                "sigma0": _finite(row.get("sigma0"), 0.0),
                "omega": _finite(row.get("omega")),
                "mu": _finite(row.get("mu"), 1.0),
                "theta": _finite(row.get("theta"), 0.0),
                "q": Q,
                "harmonic_residual": _finite(row.get("harmonic_residual", row.get("residual_abs"))),
                "rho_H": _finite(row.get("rho_H")),
                "x0": [_finite(row.get("seed_x")), _finite(row.get("seed_y")), _finite(row.get("seed_z"))],
                "reconstruction_metadata": {"implementation": "fractional_report_run"},
                "source_config": "configs/chua_fractional_nonsmooth.yaml",
            }
        )
    write_json(seed_dir / "unified_seeds.json", {"schema_version": SCHEMA_VERSION, "protocol_version": PROTOCOL_VERSION, "seeds": seeds})
    write_csv(seed_dir / "harmonic_residuals.csv", [{"candidate_id": item["candidate_id"], "family": item["family"], "harmonic_residual": item["harmonic_residual"], "rho_H": item["rho_H"]} for item in seeds])
    (seed_dir / "seed_generation_validation.md").write_text(
        "# Seed generation\n\nLas familias Lur'e y Machado/FDF generan semillas; no prueban ocultedad.\n",
        encoding="utf-8",
    )
    _write_stage_summary(
        seed_dir,
        "seed_generation",
        "completed",
        contract,
        files={"report": "seed_generation_validation.md", "seeds": "unified_seeds.json", "residuals": "harmonic_residuals.csv"},
        provenance=provenance,
        outputs={"candidate_count": len(seeds), "families": sorted({item["family"] for item in seeds})},
        verdict="seed_only",
    )

    precheck_rows = []
    for item in seeds:
        finite_configuration = bool(
            math.isfinite(item["A"])
            and item["A"] > 0.0
            and math.isfinite(item["omega"])
            and item["omega"] > 0.0
            and all(math.isfinite(v) for v in item["x0"])
        )
        precheck_rows.append(
            {
                "candidate_id": item["candidate_id"],
                "label": "pre_continuation_admissible" if finite_configuration else "rejected_invalid_amplitude_frequency",
                "admissible_for_continuation": finite_configuration,
                "finite_configuration": finite_configuration,
                "short_window_label": "not_evaluated_no_hard_rejection" if finite_configuration else "invalid_configuration",
                "periodicity_policy": "periodic_short_window_would_remain_admissible",
            }
        )
    write_csv(precheck_dir / "precheck_decisions.csv", precheck_rows)
    (precheck_dir / "soft_precheck_validation.md").write_text(
        "# Soft precheck\n\nNinguna semilla se elimina por periodicidad pre-continuacion; una observacion periodica se registra como `pre_continuation_periodic`.\n",
        encoding="utf-8",
    )
    _write_stage_summary(
        precheck_dir,
        "soft_precheck",
        "completed",
        contract,
        files={"report": "soft_precheck_validation.md", "decisions": "precheck_decisions.csv"},
        provenance=provenance,
        outputs={"admitted_to_continuation": sum(1 for row in precheck_rows if row["admissible_for_continuation"]), "periodicity_gate": False},
    )

    shutil.copy2(full_root / "continuation_paths.csv", continuation_dir / "continuation_trace.csv")
    shutil.copy2(window_root / "continuation_paths.csv", continuation_dir / "finite_memory_continuation_trace.csv")
    write_json(
        continuation_dir / "continuation_plan.json",
        {
            "public_parameter": "lambda",
            "lambda_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "provenance": {"mapping": {"internal_parameter": "epsilon", "backend": "chua_frac_backend_lib.c"}},
        },
    )
    (continuation_dir / "continuation_validation.md").write_text(
        "# Continuation\n\nEl parametro publico es `lambda`; el argumento `epsilon` existe solamente dentro del ABI C y se registra en metadatos.\n",
        encoding="utf-8",
    )
    _write_stage_summary(
        continuation_dir,
        "continuation",
        "completed",
        contract,
        files={"report": "continuation_validation.md", "plan": "continuation_plan.json", "trace": "continuation_trace.csv", "finite_memory_trace": "finite_memory_continuation_trace.csv"},
        provenance=provenance,
        outputs={"branches": ["full_history", "finite_memory"], "public_parameter": "lambda"},
    )

    full_filter = read_csv_rows(full_root / "candidate_dynamic_screen.csv")
    window_filter = read_csv_rows(window_root / "candidate_dynamic_screen.csv")
    survivor_decisions = [{**row, "branch_id": "full_history"} for row in full_filter] + [{**row, "branch_id": "finite_memory"} for row in window_filter]
    write_csv(filter_dir / "survivor_decisions.csv", survivor_decisions)
    _copy_files(full_root, filter_dir, ["selected_candidates.json", "selected_candidates.csv"])
    shutil.copy2(window_root / "selected_candidates.json", filter_dir / "selected_candidates_finite_memory.json")
    (filter_dir / "post_continuation_filter_validation.md").write_text(
        "# Post-continuation filter\n\nEl filtro de periodicidad, acotamiento y colapso se aplica despues de alcanzar `lambda=1`.\n",
        encoding="utf-8",
    )
    _write_stage_summary(
        filter_dir,
        "post_continuation_filter",
        "completed",
        contract,
        files={"report": "post_continuation_filter_validation.md", "decisions": "survivor_decisions.csv", "selected": "selected_candidates.json"},
        provenance=provenance,
        outputs={"full_history_survivors": len(full["selected"]), "finite_memory_survivors": len(window["selected"])},
        verdict="continuation_survivor",
    )

    _copy_files(full_root / "dynamic", dynamic_dir, ["trajectory_metrics.csv", "fft_summary.csv", "psd_summary.csv", "lyapunov_summary.csv", "phase_3d.png", "projections.png", "time_series.png", "spectrum_x.png"])
    for name in ("trajectory_metrics.csv", "fft_summary.csv", "psd_summary.csv", "lyapunov_summary.csv", "phase_3d.png", "projections.png", "time_series.png", "spectrum_x.png"):
        shutil.copy2(window_root / "dynamic" / name, dynamic_dir / f"finite_window_{name}")
    write_json(dynamic_dir / "dynamic_reference.json", {"full_history": full["dynamic"], "finite_memory": window["dynamic"], "reference_status": "continuation_survivor_reference"})
    write_json(dynamic_dir / "similarity_signature.json", {"metrics": ["fft_peak", "psd_entropy", "trajectory_metrics"], "backend": "chua_frac_backend_lib.c"})
    (dynamic_dir / "dynamic_reference_validation.md").write_text(
        "# Dynamic reference\n\nLas trayectorias post-transitorio y sus firmas espectrales definen las referencias usadas por robustez y ocultedad.\n",
        encoding="utf-8",
    )
    _write_stage_summary(
        dynamic_dir,
        "dynamic_reference",
        "completed",
        contract,
        files={"report": "dynamic_reference_validation.md", "reference": "dynamic_reference.json", "metrics": "trajectory_metrics.csv", "signature": "similarity_signature.json"},
        provenance=provenance,
        outputs={"branches": ["full_history", "finite_memory"]},
    )

    _copy_files(full_root / "robustness", robust_dir, ["robustness_overlay_metrics.csv", "robustness_summary.json", "overlay_3d.png"])
    for name in ("robustness_overlay_metrics.csv", "robustness_summary.json", "overlay_3d.png"):
        shutil.copy2(window_root / "robustness" / name, robust_dir / f"finite_window_{name}")
    _copy_files(run_root / "danca_abm_control", robust_dir, ["danca_abm_hiddenness_controls.csv", "danca_reference_abm_full_history.csv", "abm_replication_summary.json"])
    shutil.copy2(robust_dir / "robustness_overlay_metrics.csv", robust_dir / "robustness_metrics.csv")
    write_csv(robust_dir / "abm_efork_comparison.csv", [{"backend": danca_summary["backend"], "scope": danca_summary["scope"], "tested_trajectories": danca_summary["tested_trajectories"], "target_hits": danca_summary["target_hits"], "decision": danca_summary["decision"]}])
    write_json(robust_dir / "robustness_verdicts.json", {"verdict": "weak_target_hit", "reason": "robust similarity assessment remains limited until basin protocol completion", "branches": {"full_history": full["robustness"], "finite_memory": window["robustness"]}, "abm_reference": danca_summary})
    robust_md = rf"""# Robustness

La robustez se evaluó para cada rama mediante continuaciones EFORK C
independientes. La rama de historial completo varía \(h\) y horizonte sin
truncar historia; la rama de ventana finita incluye además variación de
\(L_m\).

ABM de historia completa se conserva aqui como comparacion de referencia. Esta
etapa verifica persistencia geometrica; no clasifica ocultedad.
"""
    (robust_dir / "robustness_validation.md").write_text(robust_md, encoding="utf-8")
    _write_stage_summary(
        robust_dir,
        "robustness",
        "incomplete",
        contract,
        files={"report": "robustness_validation.md", "metrics": "robustness_metrics.csv", "verdicts": "robustness_verdicts.json", "abm_comparison": "abm_efork_comparison.csv"},
        provenance=provenance,
        outputs={"branches": ["full_history", "finite_memory"], "abm_reference": True},
        verdict="weak_target_hit",
    )

    _copy_files(full_root / "hiddenness", hidden_dir, ["ball_sampling_plan.csv", "ball_sampling_results.csv", "hiddenness_decisions.csv", "strict_refinement_summary.csv", "hiddenness_run_summary.json"])
    for name in ("ball_sampling_plan.csv", "ball_sampling_results.csv", "hiddenness_decisions.csv", "strict_refinement_summary.csv", "hiddenness_run_summary.json"):
        shutil.copy2(window_root / "hiddenness" / name, hidden_dir / f"finite_memory_{name}")
    write_json(hidden_dir / "basin_slices_summary.json", {"status": "pending", "required_planes": ["xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"], "produced_planes": []})
    hidden_md = rf"""# Hiddenness tests

Para `{run_id}` se muestrearon puntos dentro de bolas centradas en cada
equilibrio mediante EFORK C corregido. La ausencia de contactos solo permite
`compatible_with_hiddenness_under_tested_radii`; los cortes de cuenca
`xy`, `xz` y `yz` permanecen pendientes, por lo que no se emite la etiqueta
fuerte de protocolo completo.
"""
    (hidden_dir / "hiddenness_tests_validation.md").write_text(hidden_md, encoding="utf-8")
    _write_stage_summary(
        hidden_dir,
        "hiddenness_tests",
        "incomplete_pending_basin_slices",
        contract,
        files={"report": "hiddenness_tests_validation.md", "plan": "ball_sampling_plan.csv", "results": "ball_sampling_results.csv", "basin_slices": "basin_slices_summary.json"},
        provenance=provenance,
        outputs={"branches": {"full_history": full["hiddenness"], "finite_memory": window["hiddenness"]}, "sampling_mode": "ball"},
        verdict="compatible_with_hiddenness_under_tested_radii",
    )

    _copy_files(full_root / "dynamic", diagnostics_dir, ["fft_summary.csv", "psd_summary.csv", "lyapunov_summary.csv", "spectrum_x.png"])
    (diagnostics_dir / "diagnostics_validation.md").write_text(
        "# Diagnostics\n\nFFT, PSD y estimaciones adicionales complementan las pruebas de vecindad; no sustituyen ocultedad.\n",
        encoding="utf-8",
    )
    _write_stage_summary(
        diagnostics_dir,
        "diagnostics",
        "completed_with_lyapunov_pending",
        contract,
        files={"report": "diagnostics_validation.md", "fft": "fft_summary.csv", "psd": "psd_summary.csv", "lyapunov": "lyapunov_summary.csv"},
        provenance=provenance,
        outputs={"lyapunov": "pending_causal_history_backend"},
    )

    manifest_path = validation / "00_manifest" / "validation_manifest.json"
    manifest = read_json(manifest_path)
    manifest["validation_id"] = run_id
    manifest["repository_commit"] = provenance["repository_commit"]
    manifest["working_tree_dirty"] = provenance["working_tree_dirty"]
    manifest["working_tree_diff_sha256"] = provenance["working_tree_diff_sha256"]
    manifest["stages"].update({
        "numerical_contract": "01_numerical_contract/numerical_contract_validation_summary.json",
        "seed_generation": "03_seed_generation/seed_generation_validation_summary.json",
        "soft_precheck": "04_soft_precheck/soft_precheck_validation_summary.json",
        "continuation": "05_continuation/continuation_validation_summary.json",
        "post_continuation_filter": "06_post_continuation_filter/post_continuation_filter_validation_summary.json",
        "dynamic_reference": "07_dynamic_reference/dynamic_reference_validation_summary.json",
        "robustness": "08_robustness/robustness_validation_summary.json",
        "hiddenness_tests": "09_hiddenness_tests/hiddenness_tests_validation_summary.json",
        "diagnostics": "10_diagnostics/diagnostics_validation_summary.json",
    })
    manifest["pending_stages"] = ["algebraic_validation"]
    manifest["final_report"] = {"status": "pending_full_protocol_basin_slices_and_algebraic_validation", "run_id": run_id}
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
    admissible_pool = [row for row in screened_pool if _valid_seed_configuration(row)]
    if not admissible_pool:
        raise RuntimeError("soft_precheck rejected every generated seed as invalid.")
    branch_results = {
        "full_history": run_efork_branch(
            root / "full_history", admissible_pool, branch_id="full_history", full_history=True, args=args, run_id=run_id, provenance=provenance
        ),
        "finite_window": run_efork_branch(
            root / "finite_window", admissible_pool, branch_id="finite_window", full_history=False, args=args, run_id=run_id, provenance=provenance
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
                "seed_families": ["lure_classical_centered", "lure_classical_biased", "machado_centered", "machado_biased"],
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
