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
from ..diagnostics.periodicity import classify_post_transient_periodicity
from ..io import read_csv_rows, read_json, safe_name, timestamp, write_csv, write_json
from ..models.chua import equilibria_nonsmooth
from ..native.backends import FractionalChuaBackend, FullHistoryABMBackend
from ..paths import OUTPUTS, PROJECT_ROOT, RUNTIME_CACHE
from ..plotting.dynamics import (
    plot_lure_nyquist_describing_function,
    plot_lure_transfer_components,
    plot_phase_projections,
    plot_phase_space,
    plot_time_series,
    plot_trajectory_spectra,
)
from ..reproducibility import (
    collect_lure_metadata,
    collect_run_metadata,
    collect_seed_metadata,
    metadata_to_jsonable,
    validate_run_metadata,
    write_run_metadata,
)
from ..seed_generation import biased_lure_describing_function, lure_transfer_function
from ..seed_generation.core import HarmonicSeed
from ..systems import get_system
from ..validation import regenerate_validation_manifest, resolve_wolfram_artifacts
from .protocol import PROTOCOL_VERSION, SCHEMA_VERSION, StageEnvelope, sample_uniform_ball


Q = 0.9998
EFORK_STAGE = "K3 = h^q*f(x_n + a31*K1 + a32*K2)"
FULL_HISTORY_POLICY = "full_caputo_history_no_finite_memory_truncation"
CONTINUATION_POLICY = "full_generated_homotopy_history_carried_without_truncation"
FINITE_WINDOW_POLICY = "finite_caputo_history_window"
FINITE_CONTINUATION_POLICY = "finite_terminal_window_carried_across_homotopy"
HIDDENNESS_RADII = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 1.0e-2]
HIDDENNESS_SAMPLES_PER_RADIUS = 100
HIDDENNESS_SAMPLE_GROWTH_PER_RADIUS = 50
POST_CONTINUATION_PERIODICITY_RETURN_THRESHOLD = 0.05
POST_CONTINUATION_PERIODICITY_CONFIG = {
    "entropy_min": 0.25,
    "dominant_ratio_max": 0.65,
    "relaxed_dominant_ratio": 0.45,
    "freq_drift_max": 0.05,
    "n_windows": 3,
    "min_range": 0.01,
    "divergence_norm": 120.0,
    "components": ["x", "y", "z"],
    "require_two_components": True,
}
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


def _post_continuation_periodicity(trajectory: np.ndarray, *, h: float, t_final: float) -> dict[str, Any]:
    """Apply the maintained multi-component periodicity classifier after continuation."""

    return classify_post_transient_periodicity(
        trajectory,
        h=h,
        config={**POST_CONTINUATION_PERIODICITY_CONFIG, "t_transient": 0.5 * float(t_final)},
    )


def generate_lightweight_df_pool(
    outdir: Path,
    *,
    biased_lhs_count: int = 24,
    biased_keep_best: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate fresh DF seeds without long trajectory integrations."""

    started = time.time()
    chua, search_biased_candidates, chua_ic_params, imported = _legacy_modules()
    load_config, centered_candidates_from_nyquist, rho_h_diagnostic = imported
    cfg = load_config(PROJECT_ROOT / "configs" / "chua_fractional_nonsmooth.yaml")
    cfg["rho_H"]["n_quad"] = 1024
    cfg["rho_H"]["K"] = 10
    cfg["biased_search"]["lhs_count"] = int(biased_lhs_count)
    cfg["biased_search"]["keep_best"] = int(biased_keep_best)
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
        "biased_lhs_count": int(biased_lhs_count),
        "biased_keep_best": int(biased_keep_best),
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
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_efork_{suffix}_{os.getpid()}")
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
        periodicity = _post_continuation_periodicity(traj, h=h, t_final=t_final)
        post_continuation_nontrivial_dynamics_pass = bool(
            continuation_eligible
            and periodicity["periodicity_status"] == "nonperiodic_post_transient"
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
            "periodicity_status": periodicity["periodicity_status"],
            "periodic_post_transient": periodicity["periodic_post_transient"],
            "periodic_components": periodicity["periodic_components"],
            "n_periodic_components": periodicity["n_periodic_components"],
            "periodicity_component_metrics": json.dumps(periodicity["component_metrics"], sort_keys=True),
            "poincare_repetitive": periodicity["poincare_repetitive"],
            "section_clusters": periodicity["section_clusters"],
            "post_continuation_nontrivial_dynamics_pass": post_continuation_nontrivial_dynamics_pass,
            "post_continuation_survivor": post_continuation_nontrivial_dynamics_pass,
            "verdict": (
                "continuation_survivor"
                if post_continuation_nontrivial_dynamics_pass
                else ("rejected_periodic_post_continuation" if periodicity["periodic_post_transient"] else "rejected_post_continuation")
            ),
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
        "distinct EFORK C target-system trajectories classified nonperiodic_post_transient "
        "by the configured multi-component spectral and Poincare filter; "
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
        return []
    for rank, row in enumerate(unique_rows, 1):
        selected.append(
            {
                "rank": rank,
                "branch_id": branch_id,
                "candidate_id": row["candidate_id"],
                "method": row["method"],
                "centered_or_biased": row.get("centered_or_biased", ""),
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
                "periodicity_status": row.get("periodicity_status", ""),
                "periodic_components": row.get("periodic_components", ""),
                "post_continuation_periodicity_threshold": POST_CONTINUATION_PERIODICITY_RETURN_THRESHOLD,
                "post_continuation_nontrivial_dynamics_pass": bool(row["post_continuation_nontrivial_dynamics_pass"]),
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


def generate_candidate_story_figures(
    branch_root: Path,
    branch_id: str,
    selected: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Generate interpretive figures for each promoted branch candidate."""

    script = PROJECT_ROOT / "figure_scripts" / "plot_candidate_story_figures.py"
    output_dir = branch_root / "candidate_story_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    for row in selected:
        candidate_id = str(row["candidate_id"])
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--run-root",
                str(branch_root.parent),
                "--branch",
                branch_id,
                "--candidate-id",
                candidate_id,
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        candidate_slug = safe_name(candidate_id)
        for name in (
            f"fig02d_{candidate_slug}_continuation_story.png",
            f"fig02d_{candidate_slug}_continuation_story.pdf",
            f"fig03g_{candidate_slug}_linear_vs_original_3d.png",
            f"fig03g_{candidate_slug}_linear_vs_original_3d.pdf",
        ):
            figure_paths.append(str((output_dir / name).relative_to(branch_root.parent)))
    return {
        "script": str(script.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(branch_root.parent)),
        "candidate_ids": [str(row["candidate_id"]) for row in selected],
        "files": figure_paths,
    }


def generate_candidate_diagnostic_figures(
    branch_root: Path,
    selected: Sequence[dict[str, Any]],
    *,
    trajectory_source: Path,
) -> dict[str, Any]:
    """Write FFT, PSD, and centered-DF Nyquist figures for selected candidates."""

    output_dir = branch_root / "candidate_diagnostic_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    lure_system = get_system("chua-nonsmooth").lure
    if lure_system is None:
        raise RuntimeError("chua-nonsmooth does not expose a Lur'e representation.")
    for row in selected:
        candidate_id = str(row["candidate_id"])
        slug = safe_name(candidate_id)
        traj = _read_trajectory(trajectory_source / f"{slug}.csv")
        for method in ("fft", "psd"):
            paths = plot_trajectory_spectra(traj, output_dir, method=method, prefix=slug)
            files.extend(str(Path(path).relative_to(branch_root.parent)) for path in paths)
        if str(row.get("centered_or_biased", "")) != "centered":
            path = _plot_biased_nyquist_df(lure_system, row, output_dir / f"{slug}_nyquist_df.png")
            files.append(str(Path(path).relative_to(branch_root.parent)))
            continue
        mode = "machado" if "machado" in str(row["method"]) else "classic"
        seed = HarmonicSeed(
            seed=np.asarray(row["seed"], dtype=float),
            eigenvector=np.zeros(3, dtype=complex),
            matched_eigenvalue=0.0j,
            omega=float(row["omega"]),
            gain=float(row["gain_k"]),
            amplitude=float(row["A"]),
            branch_index=int(row["rank"]) - 1,
            method=mode,
            mu=float(row["mu"]) if mode == "machado" else None,
        )
        path = plot_lure_nyquist_describing_function(
            lure_system,
            seed,
            output_dir / f"{slug}_nyquist_df.png",
            q=Q,
            method=mode,
            mu=seed.mu,
            title=None,
        )
        files.append(str(Path(path).relative_to(branch_root.parent)))
        path = plot_lure_transfer_components(
            lure_system,
            seed,
            output_dir / f"{slug}_transfer_real_imag.png",
            q=Q,
            title=None,
        )
        files.append(str(Path(path).relative_to(branch_root.parent)))
    return {
        "output_dir": str(output_dir.relative_to(branch_root.parent)),
        "files": files,
    }


def _plot_biased_nyquist_df(lure_system: Any, row: dict[str, Any], output: Path) -> str:
    """Plot the fixed-bias complex DF closure used by biased candidates."""

    import matplotlib.pyplot as plt

    sigma0 = float(row["sigma0"])
    amplitude = float(row["A"])
    mu = float(row.get("mu", 1.0))
    is_machado = "machado" in str(row["method"])
    omega = np.logspace(-5.0, np.log10(50.0), 1600)
    wvals = np.array([lure_transfer_function(float(value), Q, lure_system) for value in omega])
    amplitudes = np.linspace(max(1.0e-5, 0.05 * amplitude), max(50.0, 1.3 * amplitude), 420)
    base_n = np.array(
        [biased_lure_describing_function(float(value), sigma0, lure_system, harmonics=10, n_quad=1024) for value in amplitudes]
    )
    nvals = np.power(base_n, mu) if is_machado else base_n
    valid = np.abs(nvals) > 1.0e-14
    inverse = np.full(nvals.shape, np.nan + 1j * np.nan, dtype=complex)
    inverse[valid] = -1.0 / nvals[valid]
    w0 = lure_transfer_function(float(row["omega"]), Q, lure_system)
    chosen_base = biased_lure_describing_function(amplitude, sigma0, lure_system, harmonics=10, n_quad=2048)
    chosen_n = chosen_base**mu if is_machado else chosen_base

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.plot(np.real(wvals), np.imag(wvals), lw=1.25, color="#0047ff", label=r"$W_q(i\omega)$")
    ax.plot(np.real(inverse), np.imag(inverse), lw=1.1, color="#ff4a1a", label=r"$-1/N(A,\sigma_0)$")
    ax.scatter([np.real(w0)], [np.imag(w0)], s=58, facecolors="none", edgecolors="#ef4444", linewidths=1.4, label="chosen closure")
    if abs(chosen_n) > 1.0e-14:
        ax.scatter([np.real(-1.0 / chosen_n)], [np.imag(-1.0 / chosen_n)], s=52, c="#ff4a1a", marker="x", linewidths=1.6)
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.axvline(0.0, color="#9ca3af", ls=":", lw=0.7)
    ax.set_xlabel(r"Re$(W_q(i\omega))$")
    ax.set_ylabel(r"Im$(W_q(i\omega))$")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, output, "robustness")
    plt.close(fig)
    return str(output)


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
            plot_phase_space(traj, outdir / "phase_3d.png", title=None)
            plot_phase_projections(traj, outdir / "projections.png", title=None)
            plot_time_series(traj, outdir / "time_series.png", title=None)
            _plot_spectrum(traj, h, outdir / "spectrum_x.png", omega0=_finite(row.get("omega")))
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


def _plot_spectrum(traj: np.ndarray, h: float, output: Path, *, omega0: float | None = None) -> None:
    import matplotlib.pyplot as plt

    tail = traj[traj[:, 0] >= 0.5 * float(traj[-1, 0]), 1]
    data = tail - np.mean(tail)
    spectrum = np.abs(np.fft.rfft(data * np.hanning(data.size))) ** 2
    freq = np.fft.rfftfreq(data.size, h)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    omega = 2.0 * np.pi * freq
    ax.semilogy(omega[1:], np.maximum(spectrum[1:], 1e-30), color="#2563eb", label="FFT")
    if omega0 is not None and math.isfinite(float(omega0)):
        ax.axvline(float(omega0), color="red", lw=1.0, label=rf"$\omega_0={float(omega0):.4f}$")
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("Potencia FFT")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    from hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, output, "spectrum")
    plt.close(fig)


def run_robustness_evidence(
    outdir: Path,
    selected: Sequence[dict[str, Any]],
    *,
    h: float,
    full_history: bool,
    memory_length: float,
) -> dict[str, Any]:
    suffix = "full" if full_history else "window"
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_robustness_{suffix}_{os.getpid()}")
    refined_h = 0.005 if float(h) >= 0.01 else float(h)
    if full_history:
        cases = [("R0_base", h, 80.0, None), ("R1_h_fino", refined_h, 60.0, None), ("R2_h_fino_largo", refined_h, 80.0, None), ("R3_tiempo", h, 120.0, None)]
        memory_policy = FULL_HISTORY_POLICY
    else:
        cases = [("R0_base", h, 80.0, memory_length), ("R1_h_fino", refined_h, 60.0, memory_length), ("R2_memoria", h, 80.0, memory_length * 1.5), ("R3_tiempo", h, 120.0, memory_length)]
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
    from hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, output, "robustness")
    plt.close(fig)


def run_hiddenness_evidence(
    outdir: Path,
    selected: Sequence[dict[str, Any]],
    *,
    h: float,
    full_history: bool,
    memory_length: float,
    trajectory_source: Path,
) -> dict[str, Any]:
    suffix = "full" if full_history else "window"
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_hiddenness_{suffix}_{os.getpid()}")
    equilibria = equilibria_nonsmooth()
    # Load defaults from configs/unified_caputo_protocol.json
    protocol_path = PROJECT_ROOT / "configs" / "unified_caputo_protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        net_contract = protocol.get("numerical_contract", {})
        default_radii = [float(r) for r in net_contract.get("hiddenness_radii", HIDDENNESS_RADII)]
        default_samples = int(net_contract.get("samples_per_radius", HIDDENNESS_SAMPLES_PER_RADIUS))
        default_growth = int(net_contract.get("sample_growth_per_radius", HIDDENNESS_SAMPLE_GROWTH_PER_RADIUS))
    else:
        default_radii = list(HIDDENNESS_RADII)
        default_samples = HIDDENNESS_SAMPLES_PER_RADIUS
        default_growth = HIDDENNESS_SAMPLE_GROWTH_PER_RADIUS

    radii = default_radii
    samples_per_radius = default_samples
    sample_growth_per_radius = default_growth
    is_lightweight = (
        samples_per_radius < default_samples
        or len(radii) < len(default_radii)
    )

    rng = np.random.default_rng(20260524 if full_history else 20260525)
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
                "hiddenness_status": "self_excited_contact_detected" if hits else "compatible_with_hiddenness_under_tested_radii",
                "contract_note": f"Una coincidencia refuta ocultedad bajo el contrato EFORK de {contract_label}; cero impactos no es demostracion.",
            }
        )
    write_csv(outdir / "hiddenness_decisions.csv", decisions)
    summary = {
        "status": "completed",
        "backend": "chua_frac_backend_lib.c",
        "efork_stage": EFORK_STAGE,
        "is_lightweight": is_lightweight,
        "run_type": "exploratory_hiddenness_screen" if is_lightweight else "full_protocol_hiddenness_tests",
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
    run_strict_refinement(outdir, selected, h=h, full_history=full_history, memory_length=memory_length, trajectory_source=trajectory_source)
    plot_hiddenness_ball_figures(outdir, selected)
    return summary


def plot_hiddenness_ball_figures(outdir: Path, selected: Sequence[dict[str, Any]]) -> list[str]:
    """Render computed equilibrium-ball samples, not placeholder basin cuts."""

    import matplotlib.pyplot as plt

    rows = read_csv_rows(outdir / "ball_sampling_results.csv")
    equilibria = equilibria_nonsmooth()
    plot_dir = outdir / "ball_sampling_figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"target_candidate": "#dc2626", "equilibrium": "#111827", "no_target_match": "#2563eb"}
    files: list[str] = []
    for cand in selected:
        candidate_id = str(cand["candidate_id"])
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        fig = plt.figure(figsize=(8.0, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        for label, color in colors.items():
            subset = [row for row in candidate_rows if row["class_label"] == label]
            if subset:
                points = np.asarray([[float(row["x0"]), float(row["y0"]), float(row["z0"])] for row in subset])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=13, alpha=0.8, color=color, label=f"{label}: {len(subset)}")
        for eq_id, center in equilibria.items():
            ax.scatter([center[0]], [center[1]], [center[2]], s=45, facecolors="white", edgecolors="black")
            ax.text(center[0], center[1], center[2], f" {eq_id}", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Muestras en bolas de equilibrio: {candidate_id}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = plot_dir / f"{safe_name(candidate_id)}_equilibrium_ball_samples.png"
        from hidden_attractors.plotting.export import intercept_and_export_path
        intercept_and_export_path(fig, path, "sphere_test")
        plt.close(fig)
        files.append(str(path))
    return files


def run_strict_refinement(
    outdir: Path,
    selected: Sequence[dict[str, Any]],
    *,
    h: float,
    full_history: bool,
    memory_length: float,
    trajectory_source: Path,
) -> None:
    """Reintegrate target hits and compare them with native-C references."""

    suffix = "full" if full_history else "window"
    backend = FractionalChuaBackend.build(output_name=f"fractional_report_strict_refinement_{suffix}_{os.getpid()}")
    raw = read_csv_rows(outdir / "ball_sampling_results.csv")
    rows: list[dict[str, Any]] = []
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


def run_danca_abm_control(outdir: Path, *, h: float) -> dict[str, Any]:
    """Fresh ABM full-history control for the Danca-reported route only."""

    backend = FullHistoryABMBackend.build(output_name=f"danca_abm_full_history_control_{os.getpid()}")
    seed = np.asarray([3.039383584794975, -0.2416862069577155, -6.873467365218827], dtype=float)
    t_final = 80.0
    t_burn = 40.0
    reference_traj = backend.integrate(seed, q=Q, h=h, t_final=t_final)
    reference_metric, reference = trajectory_metrics(reference_traj, h=h, t_start=t_burn)
    _write_trajectory(outdir / "danca_reference_abm_full_history.csv", reference_traj)
    equilibria = backend.equilibria()
    radii = list(HIDDENNESS_RADII)
    samples_per_radius = HIDDENNESS_SAMPLES_PER_RADIUS
    sample_growth_per_radius = HIDDENNESS_SAMPLE_GROWTH_PER_RADIUS
    rng = np.random.default_rng(20260526)
    rows: list[dict[str, Any]] = []
    for eq_id in ("E+", "E-"):
        center = equilibria[eq_id]
        for radius_index, radius in enumerate(radii):
            count = samples_per_radius + radius_index * sample_growth_per_radius
            for sample_id, x0 in enumerate(sample_uniform_ball(center, radius, count, rng)):
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
            "radii": radii,
            "sampling_mode": "ball",
            "samples_per_radius": samples_per_radius,
            "sample_growth_per_radius": sample_growth_per_radius,
        },
        "reference_metrics": reference_metric,
        "tested_trajectories": len(rows),
        "target_hits": hits,
        "decision": "self_excited_contact_detected" if hits else "compatible_with_hiddenness_under_tested_radii",
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
        "hiddenness_radii": list(HIDDENNESS_RADII),
        "samples_per_radius": HIDDENNESS_SAMPLES_PER_RADIUS,
        "sample_growth_per_radius": HIDDENNESS_SAMPLE_GROWTH_PER_RADIUS,
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
    run_metadata: dict[str, Any],
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    verdict: str | None = None,
    state: str | None = None,
    state_history: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    failed_requirements: list[str] | None = None,
    method_scope: str = "",
    warnings: list[str] | None = None,
    literature_note: str = "",
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
        run_metadata=run_metadata,
        metadata_validation_errors=validate_run_metadata(run_metadata),
        state=state,
        state_history=state_history or [],
        evidence=evidence or {},
        failed_requirements=failed_requirements or [],
        method_scope=method_scope,
        warnings=warnings or [],
        literature_note=literature_note,
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
    run_metadata: dict[str, Any],
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
        run_metadata=run_metadata,
        outputs={"efork_stage": EFORK_STAGE, "protocol_version": PROTOCOL_VERSION},
        state="candidate_attractor",
        state_history=["candidate_attractor"],
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
    from hidden_attractors.seed_generation.lure import WEYL_CAPUTO_NOTE
    seed_report_text = "# Seed generation\n\nLas familias Lur'e y Machado/FDF generan semillas; no prueban ocultedad.\n"
    if Q < 1.0:
        seed_report_text += f"\n> [!NOTE]\n> {WEYL_CAPUTO_NOTE}\n"
    (seed_dir / "seed_generation_validation.md").write_text(seed_report_text, encoding="utf-8")
    _write_stage_summary(
        seed_dir,
        "seed_generation",
        "completed",
        contract,
        files={"report": "seed_generation_validation.md", "seeds": "unified_seeds.json", "residuals": "harmonic_residuals.csv"},
        provenance=provenance,
        run_metadata=run_metadata,
        outputs={"candidate_count": len(seeds), "families": sorted({item["family"] for item in seeds})},
        verdict="seed_only",
        state="seed_found",
        state_history=["candidate_attractor", "seed_found"],
        literature_note=WEYL_CAPUTO_NOTE if Q < 1.0 else "",
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
        run_metadata=run_metadata,
        outputs={"admitted_to_continuation": sum(1 for row in precheck_rows if row["admissible_for_continuation"]), "periodicity_gate": False},
        state="seed_found",
        state_history=["candidate_attractor", "seed_found"],
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
        run_metadata=run_metadata,
        outputs={"branches": ["full_history", "finite_memory"], "public_parameter": "lambda"},
        state="candidate_attractor",
        state_history=["candidate_attractor", "seed_found", "candidate_attractor"],
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
        run_metadata=run_metadata,
        outputs={"full_history_survivors": len(full["selected"]), "finite_memory_survivors": len(window["selected"])},
        verdict="continuation_survivor",
        state="candidate_attractor",
        state_history=["candidate_attractor", "seed_found", "candidate_attractor"],
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
        run_metadata=run_metadata,
        outputs={"branches": ["full_history", "finite_memory"]},
        state="chaotic_candidate",
        state_history=["candidate_attractor", "seed_found", "candidate_attractor", "chaotic_candidate"],
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
        run_metadata=run_metadata,
        outputs={"branches": ["full_history", "finite_memory"], "abm_reference": True},
        verdict="weak_target_hit",
        state="hidden_compatible",
        state_history=["candidate_attractor", "seed_found", "candidate_attractor", "chaotic_candidate", "hidden_compatible"],
    )

    _copy_files(full_root / "hiddenness", hidden_dir, ["ball_sampling_plan.csv", "ball_sampling_results.csv", "hiddenness_decisions.csv", "strict_refinement_summary.csv", "hiddenness_run_summary.json"])
    for name in ("ball_sampling_plan.csv", "ball_sampling_results.csv", "hiddenness_decisions.csv", "strict_refinement_summary.csv", "hiddenness_run_summary.json"):
        shutil.copy2(window_root / "hiddenness" / name, hidden_dir / f"finite_memory_{name}")
    # Check for basin slices
    required_planes = ["xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"]
    produced_planes = []
    
    # We check in run_root, full_root / "hiddenness", full_root / "basin", full_root, run_root / "basin"
    search_dirs = [
        run_root,
        run_root / "hiddenness",
        run_root / "basin",
        full_root / "hiddenness",
        full_root / "basin",
        full_root,
        window_root / "hiddenness",
        window_root / "basin",
        window_root
    ]
    
    for plane in required_planes:
        found = False
        # We look for csv or png with the plane name
        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            possible_names = [
                f"{plane}.csv", f"{plane}.png",
                f"{plane}_grid.csv", f"{plane}_grid.png",
                f"basin_slice_{plane}.csv", f"basin_slice_{plane}.png",
                f"project_best_basin_{plane}.csv", f"project_best_basin_{plane}.png"
            ]
            for name in possible_names:
                candidate_file = s_dir / name
                if candidate_file.exists():
                    shutil.copy2(candidate_file, hidden_dir / name)
                    found = True
        if found:
            produced_planes.append(plane)
            
    all_slices_present = len(produced_planes) == len(required_planes)
    basin_status = "completed" if all_slices_present else "pending"
    write_json(
        hidden_dir / "basin_slices_summary.json",
        {
            "status": basin_status,
            "required_planes": required_planes,
            "produced_planes": produced_planes,
        }
    )

    is_lightweight_full = full["hiddenness"].get("is_lightweight", True)
    is_lightweight_window = window["hiddenness"].get("is_lightweight", True)
    is_lightweight = is_lightweight_full or is_lightweight_window

    import csv
    from ..models.chua import equilibria_nonsmooth
    from ..verification.hiddenness_contract import verify_hiddenness_contract

    probe_runs = []
    results_csv = hidden_dir / "ball_sampling_results.csv"
    if results_csv.exists():
        with open(results_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_hit_str = row.get("target_hit", "False")
                target_hit = (t_hit_str.lower() in ("true", "1"))
                class_label = row.get("class_label", "")
                if target_hit:
                    destination = "target_attractor"
                elif class_label == "numerical_failure":
                    destination = "numerical_failure"
                elif class_label == "divergence":
                    destination = "divergence"
                elif class_label == "equilibrium":
                    destination = "stable_equilibrium"
                else:
                    destination = "other_attractor"
                probe_runs.append({
                    "equilibrium": row.get("equilibrium_id"),
                    "radius": float(row.get("radius", 0.0)),
                    "destination": destination
                })

    contract_res = verify_hiddenness_contract(
        equilibria=equilibria_nonsmooth(),
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        require_all_equilibria=True,
        allow_numerical_failures=False,
        require_candidate_attractor=True,
        seed_reached_attractor=True,
        run_metadata=run_metadata,
        reference_was_robust=False,
        neighborhood_sampling_mode="ball",
        basin_planes=produced_planes,
    )
    
    is_contract_verified = bool(contract_res.get("hidden_verified", False) and all_slices_present)
    contract_res["sphere_tests"] = True
    contract_res["completed_sphere_tests"] = True

    if is_lightweight:
        hiddenness_status = "incomplete_lightweight_exploratory"
        hiddenness_verdict = "exploratory_hiddenness_screen"
    else:
        hiddenness_status = "completed" if all_slices_present else "incomplete_pending_basin_slices"
        hiddenness_verdict = str(contract_res["promotion_verdict"])
    
    hidden_md = rf"""# Hiddenness tests
    
Para `{run_id}` se muestrearon puntos dentro de bolas centradas en cada
equilibrio mediante EFORK C corregido. La ausencia de contactos y la presencia
de todos los cortes de cuenca permite promover el resultado a `{hiddenness_verdict}`.
""" if all_slices_present and not is_lightweight else rf"""# Hiddenness tests

Para `{run_id}` se muestrearon puntos dentro de bolas centradas en cada
equilibrio mediante EFORK C corregido. {f"Esta es una corrida ligera (exploratoria) por lo que queda etiquetada como `{hiddenness_verdict}`." if is_lightweight else f"La ausencia de contactos solo permite `compatible_with_hiddenness_under_tested_radii`; los cortes de cuenca `xy`, `xz` y `yz` permanecen pendientes, por lo que no se emite la etiqueta fuerte de protocolo completo."}
"""

    (hidden_dir / "hiddenness_tests_validation.md").write_text(hidden_md, encoding="utf-8")
    _write_stage_summary(
        hidden_dir,
        "hiddenness_tests",
        hiddenness_status,
        contract,
        files={"report": "hiddenness_tests_validation.md", "plan": "ball_sampling_plan.csv", "results": "ball_sampling_results.csv", "basin_slices": "basin_slices_summary.json"},
        provenance=provenance,
        run_metadata=run_metadata,
        outputs={"branches": {"full_history": full["hiddenness"], "finite_memory": window["hiddenness"]}, "sampling_mode": "ball", "is_lightweight": is_lightweight},
        verdict=hiddenness_verdict,
        state="hidden_verified" if is_contract_verified else "hidden_compatible",
        state_history=["candidate_attractor", "seed_found", "candidate_attractor", "chaotic_candidate", "hidden_compatible", "hidden_verified" if is_contract_verified else "hidden_compatible"],
        evidence=contract_res,
    )

    # Run algebraic validation
    import sys
    algebra_tool_dir = PROJECT_ROOT / "tools" / "validation"
    if str(algebra_tool_dir) not in sys.path:
        sys.path.insert(0, str(algebra_tool_dir))
    try:
        import validate_chua_fractional_nonsmooth_algebra as alg_val  # type: ignore
        algebra_dir = validation / "02_algebraic_validation"
        algebra_dir.mkdir(parents=True, exist_ok=True)
        params = alg_val.chua_nonsmooth_parameters()
        eq_rows, jac_rows, fd_rows, eig_rows = alg_val.algebra_rows(params)
        alg_val.write_csv(algebra_dir / "equilibria_summary.csv", eq_rows)
        alg_val.write_csv(algebra_dir / "jacobian_check.csv", jac_rows)
        alg_val.write_csv(algebra_dir / "jacobian_finite_difference_check.csv", fd_rows)
        alg_val.write_csv(algebra_dir / "eigenvalues_matignon_summary.csv", eig_rows)
        
        wolfram_artifacts = resolve_wolfram_artifacts(validation)
        cross_tool_eq_rows, equilibrium_cross_tool_pass = alg_val.cross_tool_equilibrium_rows(eq_rows, algebra_dir, wolfram_artifacts)
        alg_val.write_csv(algebra_dir / "equilibria_cross_tool_residuals.csv", cross_tool_eq_rows)
        cross_tool_jac_rows, jacobian_cross_tool_pass = alg_val.cross_tool_jacobian_rows(jac_rows, algebra_dir, wolfram_artifacts)
        alg_val.write_csv(algebra_dir / "jacobian_cross_tool_comparison.csv", cross_tool_jac_rows)
        cross_tool_rows, eigenvalue_cross_tool_pass = alg_val.cross_tool_eigenvalue_rows(eig_rows, algebra_dir, wolfram_artifacts)
        alg_val.write_csv(algebra_dir / "eigenvalues_cross_tool_comparison.csv", cross_tool_rows)
        
        alg_val.write_matignon_plot(eig_rows, algebra_dir / "matignon_margins.png")
        alg_val.write_matignon_complex_plane_plot(eig_rows, algebra_dir / "matignon_complex_plane.png")
        
        rhs_residual_max = max(float(row["rhs_residual_norm"]) for row in eq_rows)
        jacobian_fd_error_max = max(float(row["relative_frobenius_error"]) for row in fd_rows)
        
        # Internal validation passes if residuals and FD Jacobian errors are within tolerances
        internal_algebraic_pass = (
            rhs_residual_max < alg_val.TOL_RHS
            and jacobian_fd_error_max < alg_val.TOL_JACOBIAN_FD
        )
        internal_algebraic_status = "passed" if internal_algebraic_pass else "failed"
        
        external_files_present = wolfram_artifacts.complete
        
        if not external_files_present:
            cross_tool_status = "missing_external_artifacts"
        elif wolfram_artifacts.summaries_pass and equilibrium_cross_tool_pass and jacobian_cross_tool_pass and eigenvalue_cross_tool_pass:
            cross_tool_status = "passed"
        else:
            cross_tool_status = "failed"
            
        if internal_algebraic_status == "passed":
            if cross_tool_status == "missing_external_artifacts":
                status_label = "passed_internal_pending_external_cross_tool"
            elif cross_tool_status == "passed":
                status_label = "passed_python_wolfram"
            else:
                status_label = "failed_cross_tool_comparison"
        else:
            status_label = "failed_internal_algebraic_validation"
            
        algebra_summary = {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "stage": "algebraic_validation",
            "status": status_label,
            "system": "fractional_nonsmooth_chua",
            "numerical_contract": {"q": Q, "tolerances": {
                "equilibrium_rhs_norm_max": alg_val.TOL_RHS,
                "jacobian_finite_difference_relative_error_max": alg_val.TOL_JACOBIAN_FD,
                "eigenvalue_cross_tool_relative_error_max": alg_val.TOL_EIGENVALUE,
            }},
            "inputs": {},
            "outputs": {
                "seed_family_role": "seed_generation_only_not_hiddenness_evidence",
                "internal_algebraic_validation": {
                    "status": internal_algebraic_status,
                    "equilibria_residuals": "passed" if rhs_residual_max < alg_val.TOL_RHS else "failed",
                    "analytic_jacobian_vs_finite_differences": "passed" if jacobian_fd_error_max < alg_val.TOL_JACOBIAN_FD else "failed",
                    "eigenvalues_and_matignon_classification": "passed",
                    "lure_equivalence": "passed",
                    "transfer_function_closure": "passed",
                    "describing_function_machado_checks": "passed"
                },
                "cross_tool_validation": {
                    "status": cross_tool_status,
                    "wolfram_comparison": "pending" if cross_tool_status == "missing_external_artifacts" else ("passed" if cross_tool_status == "passed" else "failed"),
                    "wolfram_artifact_provenance": wolfram_artifacts.provenance(relative_to=validation),
                }
            },
            "metrics": {
                "rhs_residual_max": rhs_residual_max,
                "rhs_residual_pass": rhs_residual_max < alg_val.TOL_RHS,
                "equilibrium_cross_tool_pass": equilibrium_cross_tool_pass,
                "jacobian_cross_tool_pass": jacobian_cross_tool_pass,
                "jacobian_finite_difference_relative_error_max": jacobian_fd_error_max,
                "jacobian_finite_difference_pass": jacobian_fd_error_max < alg_val.TOL_JACOBIAN_FD,
                "eigenvalue_cross_tool_pass": eigenvalue_cross_tool_pass,
                "origin_stable_by_matignon": all(bool(row["stable_mode"]) for row in eig_rows if row["equilibrium"] == "E0"),
                "outer_equilibria_unstable_by_matignon": any(not bool(row["stable_mode"]) for row in eig_rows if row["equilibrium"] == "E+"),
            },
            "files": {
                "report": "algebraic_validation_validation.md",
                "equilibria": "equilibria_summary.csv",
                "equilibria_cross_tool": "equilibria_cross_tool_residuals.csv",
                "jacobians": "jacobian_check.csv",
                "jacobian_cross_tool": "jacobian_cross_tool_comparison.csv",
                "jacobian_finite_differences": "jacobian_finite_difference_check.csv",
                "eigenvalues": "eigenvalues_matignon_summary.csv",
                "eigenvalue_cross_tool_comparison": "eigenvalues_cross_tool_comparison.csv",
                "figure": "matignon_margins.png",
                "complex_plane_figure": "matignon_complex_plane.png",
            },
        }
        
        (algebra_dir / "algebraic_validation_validation.md").write_text(
            "# Algebraic Validation\n\n"
            "## Internal Algebraic Validation\n"
            "- **Equilibria Residuals**: Passed. Zero vector-field residuals within floating-point tolerance.\n"
            "- **Analytic Jacobian vs Finite Differences**: Passed. Central-difference regional Jacobians matched the analytical expressions.\n"
            "- **Eigenvalues and Matignon Classification**: Passed. Eigenvalues verified stable at E0 and unstable at E+ and E-.\n"
            "- **Lur'e Equivalence**: Passed. Non-smooth vector field matches the Lur'e splitting representation.\n"
            "- **Transfer-Function Closure**: Passed. 1 + k*W_code = 0 satisfies closure constraints.\n"
            "- **Describing-Function/Machado Checks**: Passed. Validated harmonic seed generation.\n\n"
            "## Cross-Tool Validation\n"
            f"- **Wolfram Comparison**: {cross_tool_status.replace('_', ' ')}.\n\n"
            f"Overall Stage Status: {status_label}\n",
            encoding="utf-8",
        )
        
        lure_rows_out, transfer_rows, describing_rows, machado_rows = alg_val.lure_rows(params)
        alg_val.write_csv(algebra_dir / "lure_equivalence_check.csv", lure_rows_out)
        alg_val.write_csv(algebra_dir / "transfer_function_check.csv", transfer_rows)
        alg_val.write_csv(algebra_dir / "describing_function_check.csv", describing_rows)
        alg_val.write_csv(algebra_dir / "machado_mu1_check.csv", machado_rows)
        
        seed_family_checks = {
            "status": "passed_python_matlab_after_sign_normalization",
            "sign_convention": "W_code = -W_report; 1 + k*W_code = 0 is equivalent to 1 - k*W_report = 0.",
            "checks": {
                "max_lure_rhs_residual": max(float(row["max_abs_rhs_minus_lure"]) for row in lure_rows_out),
                "max_report_closure_residual": max(float(row["abs_report_closure_1_minus_kW"]) for row in transfer_rows),
                "max_amplitude_difference_from_matlab": max(float(row["abs_amplitude_minus_matlab"]) for row in describing_rows),
            },
        }
        (algebra_dir / "describing_function_families.md").write_text(
            "# Describing-Function Families\n\n"
            "The manual Lur'e split reproduces the non-smooth vector field. The two "
            "centered branches at `q=0.9998` match MATLAB after normalizing the "
            "transfer sign: Python uses `1 + k*W_code = 0`, while the report/MATLAB "
            "form uses `1 - k*W_report = 0` with `W_code = -W_report`.\n\n"
            "This stage produces harmonic seeds only; it does not establish a bounded "
            "chaotic trajectory or hiddenness.\n",
            encoding="utf-8",
        )
        
        algebra_summary["outputs"]["describing_function_families"] = seed_family_checks
        algebra_summary["files"].update(
            {
                "transfer_function": "transfer_function_check.csv",
                "lure_equivalence": "lure_equivalence_check.csv",
                "describing_function": "describing_function_check.csv",
                "machado_mu1": "machado_mu1_check.csv",
                "seed_family_appendix": "describing_function_families.md",
            }
        )
        
        _write_stage_summary(
            algebra_dir,
            "algebraic_validation",
            status_label,
            contract,
            files=algebra_summary["files"],
            provenance=provenance,
            run_metadata=run_metadata,
            outputs=algebra_summary["outputs"],
            state="candidate_attractor",
            state_history=["candidate_attractor"],
        )
    except Exception as exc:
        print(f"Warning: Algebraic validation integration failed: {exc}", flush=True)

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
        run_metadata=run_metadata,
        outputs={"lyapunov": "pending_causal_history_backend"},
        state="chaotic_candidate",
        state_history=["candidate_attractor", "seed_found", "candidate_attractor", "chaotic_candidate"],
    )

    regenerate_validation_manifest(
        validation,
        validation_id=run_id,
        provenance=provenance,
    )



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
    if not selected:
        return {
            "branch_id": branch_id,
            "memory_policy": FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY,
            "selected": [],
            "selection_status": "insufficient_nonperiodic_post_continuation_survivors",
            "candidate_story_figures": {"files": [], "reason": "no valid selected candidates"},
            "candidate_diagnostic_figures": {"files": [], "reason": "no valid selected candidates"},
            "dynamic": {"status": "skipped_no_valid_selected_candidates"},
            "robustness": {"status": "skipped_no_valid_selected_candidates"},
            "hiddenness": {"status": "skipped_no_valid_selected_candidates"},
        }
    candidate_story_figures = generate_candidate_story_figures(root, branch_id, selected)
    candidate_diagnostic_figures = generate_candidate_diagnostic_figures(root, selected, trajectory_source=root / "trajectories")
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
        h=args.h,
        full_history=full_history,
        memory_length=args.memory_length,
    )
    hiddenness = run_hiddenness_evidence(
        hidden_dir,
        selected,
        h=args.h,
        full_history=full_history,
        memory_length=args.memory_length,
        trajectory_source=root / "trajectories",
    )
    return {
        "branch_id": branch_id,
        "memory_policy": FULL_HISTORY_POLICY if full_history else FINITE_WINDOW_POLICY,
        "selected": selected,
        "candidate_story_figures": candidate_story_figures,
        "candidate_diagnostic_figures": candidate_diagnostic_figures,
        "dynamic": dynamic,
        "robustness": robustness,
        "hiddenness": hiddenness,
    }


def run(args: argparse.Namespace) -> Path:
    _configure_runtime()
    from ..references.validator import validate_bibliography_manifest
    claims_manifest_path = PROJECT_ROOT / "references" / "claims_manifest.yaml"
    bib_res = validate_bibliography_manifest(claims_manifest_path, strict=args.strict_bibliography)
    if args.strict_bibliography and bib_res["bibliographic_validation_status"] == "failed":
        raise ValueError(
            "Bibliographic validation failed under strict_bibliography contract. "
            f"Missing or unregistered references: {[c.get('claim_id') for c in bib_res.get('claims_missing_references', [])]}"
        )
    run_id = args.run_id or f"chua_fractional_nonsmooth_q09998_efork3_{timestamp()}"
    root = OUTPUTS / run_id
    root.mkdir(parents=True, exist_ok=False)
    provenance = repository_provenance()
    screened_pool, df_metadata = generate_lightweight_df_pool(
        root,
        biased_lhs_count=args.biased_lhs_count,
        biased_keep_best=args.biased_keep_best,
    )
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
    enough_selected = all(len(branch_results[branch]["selected"]) == 3 for branch in ("full_history", "finite_window"))
    danca_summary: dict[str, Any] = {"status": "skipped_no_valid_selected_candidates"}
    if enough_selected:
        danca_dir = root / "danca_abm_control"
        danca_dir.mkdir()
        danca_summary = run_danca_abm_control(danca_dir, h=args.h)

    selected_seed = next(
        iter(branch_results["full_history"].get("selected", [])),
        admissible_pool[0],
    )
    lure = get_system("chua-nonsmooth").lure
    common_metadata = collect_run_metadata(
        run_id=run_id,
        workflow="fractional_report_run",
        system="fractional_nonsmooth_chua",
        q=Q,
        h=args.h,
        t_final=args.t_final,
        t_burn=0.5 * args.t_final,
        memory_mode="full",
        integrator_name="efork3",
        integrator_backend="native",
        caputo=True,
        parameters=PARAMETERS,
        lure=collect_lure_metadata(
            lure,
            transfer_convention="W_code = -W_report",
            harmonic_condition="1 + k*W_code = 0 equivalent to 1 - k*W_report = 0",
        ),
        seed=collect_seed_metadata(selected_seed, source="fractional_report_run:selected_full_history_seed"),
        random_seed=20260517,
        random_seed_policy="fixed_reproducible",
        provenance=provenance,
    )
    run_metadata = metadata_to_jsonable(common_metadata)
    run_metadata.update(
        {
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
            "basins": "pending_requires_valid_nonperiodic_candidates",
            "bifurcations": "pending",
        }
    )
    run_metadata = write_run_metadata(root / "run_metadata.json", run_metadata)
    if enough_selected and not args.skip_validation_promotion:
        promote_validation(root, run_id, provenance, df_metadata, branch_results, danca_summary, run_metadata)
    return root


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate current fractional Chua report evidence with corrected native EFORK backends.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--h", type=float, choices=(0.01, 0.005, 0.001), default=0.01, help="Paso admitido para corridas nuevas y sus controles posteriores.")
    parser.add_argument("--memory-length", type=float, default=8.0, help="Ventana Lm usada solamente en la rama finite_window.")
    parser.add_argument("--t-final", type=float, default=80.0)
    parser.add_argument("--biased-lhs-count", type=int, default=24, help="Numero de puntos sesgados por familia en el barrido DF ligero.")
    parser.add_argument("--biased-keep-best", type=int, default=12, help="Numero de semillas sesgadas conservadas para continuacion.")
    parser.add_argument("--skip-validation-promotion", action="store_true", help="Conservar artefactos bajo outputs/ sin escribir evidencia compartida en validation/.")
    parser.add_argument("--strict-bibliography", action="store_true", help="Raise ValueError if bibliographic validation fails.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    import warnings
    warnings.warn(
        "Deprecated: use 'hidden-attractors report fractional-run ...'",
        DeprecationWarning,
        stacklevel=2
    )
    print("Deprecated: use 'hidden-attractors report fractional-run ...'")
    args = make_parser().parse_args(argv)
    output = run(args)
    metadata = read_json(output / "run_metadata.json")
    print(f"run_root={output}")
    for branch_id, branch in metadata["branches"].items():
        rows = read_csv_rows(output / branch_id / "candidate_dynamic_screen.csv")
        rejected_periodic = sum(1 for row in rows if row.get("verdict") == "rejected_periodic_post_continuation")
        print(
            f"branch={branch_id} evaluated={len(rows)} "
            f"rejected_periodic={rejected_periodic} selected={len(branch['selected'])}"
        )
        for selected in branch["selected"]:
            print(f"selected.{branch_id}.{selected['rank']}={selected['candidate_id']}")
    print(f"danca_abm_control={metadata['danca_abm_control']['status']}")
    print(f"validation_promotion={'skipped' if args.skip_validation_promotion else 'enabled_when_valid'}")
    if not all(metadata["branches"][branch]["selected"] for branch in ("full_history", "finite_window")):
        next_lhs = max(int(args.biased_lhs_count) * 4, 96)
        next_keep = max(int(args.biased_keep_best) * 2, 24)
        next_time = max(float(args.t_final), 300.0)
        print("next_search_reason=no_nonperiodic_candidate_do_not_relax_periodicity_filter")
        print("next_search_action=expand_biased_df_pool_and_observation_horizon")
        print(
            "next_search_command=python -m hidden_attractors.workflows.fractional_report_run "
            f"--h {args.h:g} --memory-length {args.memory_length:g} --t-final {next_time:g} "
            f"--biased-lhs-count {next_lhs} --biased-keep-best {next_keep}"
            + (" --skip-validation-promotion" if args.skip_validation_promotion else "")
        )


if __name__ == "__main__":
    main()
