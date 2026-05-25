"""Danca reference-attractor solver/memory controls plus strict refinement.

The module name is retained as an installed compatibility alias. New plans
sample inside equilibrium-centred balls after re-integrating the Danca-located
reference seed through ABM and EFORK3, both without memory truncation and with
an explicitly labelled finite-memory truncation.

Mathematical warning:
    The coarse labels are finite-time basin diagnostics.  A target hit from an
    equilibrium ball is evidence against hiddenness under the tested
    numerical contract; absence of such hits is compatibility evidence, not a
    proof.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..basins import class_label
from ..io import append_csv, read_csv_rows, read_json, safe_name, write_csv, write_json
from ..models.chua import ChuaParameters
from ..native.backends import FractionalChuaBackend, FullHistoryABMBackend
from ..parallel import force_single_openmp_thread_current_process, force_single_openmp_thread_env
from ..paths import PROJECT_ROOT
from ..plotting.dynamics import plot_phase_projections, plot_phase_space, plot_time_series, plot_trajectory_spectra
from ..seed_generation import lure_transfer_function
from ..systems import get_system
from .protocol import sample_uniform_ball


LEGACY_ROOT = PROJECT_ROOT / "tools" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from danca2017_chua_abm_replication import (  # noqa: E402
    DancaChuaConfig,
    classify_trajectory,
)
from equilibria_analysis import solve_equilibria  # noqa: E402


ROOT_OUTPUTS = PROJECT_ROOT.parent / "outputs"
DEFAULT_DANCA_SOURCE = ROOT_OUTPUTS / "danca2017_chua_abm_20260515_182354"
FULL_HISTORY_POLICY = "full_caputo_history_no_finite_memory_truncation"
FINITE_MEMORY_POLICY = "finite_caputo_history_window"

RAW_FIELDS = [
    "case_index",
    "candidate_id",
    "solver_case_id",
    "solver",
    "backend",
    "history_policy",
    "memory_length",
    "q",
    "equilibrium_id",
    "radius",
    "sample_id",
    "sampling_mode",
    "distance_from_equilibrium",
    "batch_100",
    "x0",
    "y0",
    "z0",
    "h",
    "t_final",
    "t_burn",
    "class_id",
    "class_label",
    "target_hit",
    "danca_class",
    "bounded",
    "final_norm",
    "closest_equilibrium",
    "closest_equilibrium_distance",
    "range_norm_tail",
    "range_x_tail",
    "range_y_tail",
    "range_z_tail",
    "mean_x_tail",
    "mean_y_tail",
    "mean_z_tail",
    "elapsed_sec",
    "status",
]

SUMMARY_FIELDS = [
    "candidate_id",
    "solver_case_id",
    "solver",
    "history_policy",
    "memory_length",
    "equilibrium_id",
    "radius",
    "n_executed",
    "n_target_hits",
    "target_hit_fraction",
    "n_equilibrium",
    "n_target_positive",
    "n_target_negative",
    "n_infinity",
    "n_unknown",
    "n_numerical_failure",
]

REFERENCE_FIELDS = [
    "candidate_id",
    "solver_case_id",
    "solver",
    "backend",
    "history_policy",
    "memory_length",
    "seed_source",
    "x0",
    "y0",
    "z0",
    "q",
    "h",
    "t_final",
    "t_burn",
    "class",
    "bounded",
    "target_hit",
    "final_norm",
    "closest_equilibrium",
    "closest_equilibrium_distance",
    "range_norm_tail",
    "range_x_tail",
    "range_y_tail",
    "range_z_tail",
    "status",
]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _danca_config(source_dir: Path) -> DancaChuaConfig:
    raw = read_json(source_dir / "run_config.json")
    return DancaChuaConfig(
        q=float(raw["q"]),
        h=float(raw["h"]),
        t_final=float(raw["t_final"]),
        transient=float(raw["transient"]),
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        gamma_chua=float(raw["gamma_chua"]),
        m0=float(raw["m0"]),
        m1=float(raw["m1"]),
        delta=float(raw["delta"]),
        equilibrium_tol=float(raw["equilibrium_tol"]),
        divergence_norm=float(raw["divergence_norm"]),
        nontrivial_range_tol=float(raw["nontrivial_range_tol"]),
        local_samples_per_unstable_eq=int(raw["local_samples_per_unstable_eq"]),
        figure_local_trajectories=int(raw["figure_local_trajectories"]),
        rng_seed=int(raw["rng_seed"]),
        store_stride=int(raw.get("store_stride", 1)),
    )


def _coarse_danca_class(row: dict[str, Any], *, mean_x_gap: float) -> int:
    klass = str(row.get("class", ""))
    if str(row.get("status", "")) != "ok":
        return 5
    if klass == "infinity":
        return 3
    if klass.startswith("equilibrium_"):
        return 0
    if klass == "bounded_nontrivial":
        mean_x = _float(row.get("mean_x_tail", 0.0), 0.0)
        if mean_x > float(mean_x_gap):
            return 1
        if mean_x < -float(mean_x_gap):
            return 2
        return 4
    return 4


def _effective_config(source: DancaChuaConfig, args: argparse.Namespace) -> DancaChuaConfig:
    """Return the fresh numerical contract while retaining Danca parameters."""

    return DancaChuaConfig(
        q=source.q,
        h=float(args.h),
        t_final=float(args.t_final) if float(args.t_final) > 0.0 else source.t_final,
        transient=float(args.t_burn) if float(args.t_burn) >= 0.0 else source.transient,
        alpha=source.alpha,
        beta=source.beta,
        gamma_chua=source.gamma_chua,
        m0=source.m0,
        m1=source.m1,
        delta=source.delta,
        equilibrium_tol=source.equilibrium_tol,
        divergence_norm=source.divergence_norm,
        nontrivial_range_tol=source.nontrivial_range_tol,
        local_samples_per_unstable_eq=source.local_samples_per_unstable_eq,
        figure_local_trajectories=source.figure_local_trajectories,
        rng_seed=source.rng_seed,
        store_stride=source.store_stride,
    )


def _runtime_config(cfg: dict[str, Any]) -> DancaChuaConfig:
    source = _danca_config(Path(cfg["danca_source_dir"]))
    contract = cfg["contract"]
    return DancaChuaConfig(
        q=float(contract["q"]),
        h=float(contract["h"]),
        t_final=float(contract["t_final"]),
        transient=float(contract["t_burn"]),
        alpha=source.alpha,
        beta=source.beta,
        gamma_chua=source.gamma_chua,
        m0=source.m0,
        m1=source.m1,
        delta=source.delta,
        equilibrium_tol=float(contract["equilibrium_tol"]),
        divergence_norm=float(contract["divergence_norm"]),
        nontrivial_range_tol=float(contract["nontrivial_range_tol"]),
        local_samples_per_unstable_eq=source.local_samples_per_unstable_eq,
        figure_local_trajectories=source.figure_local_trajectories,
        rng_seed=source.rng_seed,
        store_stride=source.store_stride,
    )


def _located_reference_seed(source_dir: Path) -> dict[str, Any]:
    """Load the seed previously located by an untruncated ABM reference run."""

    summary = read_json(source_dir / "danca_reference_summary.json")
    policy = str(summary.get("history_policy", "")).lower()
    if "full caputo history" not in policy or "no finite-memory truncation" not in policy:
        raise ValueError("Danca reference seed must originate from an ABM full-history run without truncation.")
    seed = summary.get("best_seed", {})
    x0 = seed.get("x0")
    if not isinstance(x0, list) or len(x0) != 3:
        raise ValueError("danca_reference_summary.json does not contain a located best_seed.x0.")
    return {
        "candidate_id": "danca2017_reference",
        "source_summary": str(source_dir / "danca_reference_summary.json"),
        "source_method": summary.get("method", ""),
        "source_history_policy": summary.get("history_policy", ""),
        "seed_id": seed.get("seed_id", ""),
        "seed_source": seed.get("source", ""),
        "x0": [float(value) for value in x0],
    }


def _solver_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    memory_length = float(args.memory_length)
    return [
        {
            "solver_case_id": "abm_full_history",
            "solver": "abm",
            "backend": "chua_abm_full_history_lib.c",
            "history_policy": FULL_HISTORY_POLICY,
            "memory_length": None,
            "reference_role": "single_seed_accreditation_and_comparison",
        },
        {
            "solver_case_id": f"abm_truncated_Lm{safe_name(f'{memory_length:g}')}",
            "solver": "abm",
            "backend": "chua_abm_full_history_lib.c",
            "history_policy": FINITE_MEMORY_POLICY,
            "memory_length": memory_length,
            "reference_role": "memory_truncation_comparison",
        },
        {
            "solver_case_id": "efork_full_history",
            "solver": "efork3",
            "backend": "chua_frac_backend_lib.c",
            "history_policy": FULL_HISTORY_POLICY,
            "memory_length": None,
            "reference_role": "integrator_comparison_without_truncation",
        },
        {
            "solver_case_id": f"efork_truncated_Lm{safe_name(f'{memory_length:g}')}",
            "solver": "efork3",
            "backend": "chua_frac_backend_lib.c",
            "history_policy": FINITE_MEMORY_POLICY,
            "memory_length": memory_length,
            "reference_role": "integrator_and_memory_truncation_comparison",
        },
    ]


def _native_params(dcfg: DancaChuaConfig) -> ChuaParameters:
    return ChuaParameters(
        model="nonsmooth",
        alpha=dcfg.alpha,
        beta=dcfg.beta,
        gamma=dcfg.gamma_chua,
        m0=dcfg.m0,
        m1=dcfg.m1,
    )


def _case_integrator(case: dict[str, Any], dcfg: DancaChuaConfig) -> Any:
    """Build the native integrator for one solver/memory cell."""

    case_id = safe_name(str(case["solver_case_id"]))
    if str(case["solver"]) == "abm":
        backend = FullHistoryABMBackend.build(output_name=f"danca_{case_id}_{os.getpid()}")
        backend.set_nonsmooth_params(_native_params(dcfg))
        if case["history_policy"] == FULL_HISTORY_POLICY:
            return lambda seed: backend.integrate(seed, q=dcfg.q, h=dcfg.h, t_final=dcfg.t_final)
        return lambda seed: backend.integrate_truncated(
            seed,
            q=dcfg.q,
            h=dcfg.h,
            Lm=float(case["memory_length"]),
            t_final=dcfg.t_final,
        )
    backend = FractionalChuaBackend.build(output_name=f"danca_{case_id}_{os.getpid()}")
    backend.set_nonsmooth_params(_native_params(dcfg))
    lm = dcfg.t_final if case["history_policy"] == FULL_HISTORY_POLICY else float(case["memory_length"])
    return lambda seed: backend.integrate_efork3(seed, q=dcfg.q, h=dcfg.h, Lm=lm, t_final=dcfg.t_final)


def _write_trajectory(path: Path, trajectory: np.ndarray) -> None:
    rows = [
        {"t": float(row[0]), "x": float(row[1]), "y": float(row[2]), "z": float(row[3])}
        for row in np.asarray(trajectory, dtype=float)
    ]
    write_csv(path, rows, ["t", "x", "y", "z"])


def _first_harmonic_reconstruction(trajectory: np.ndarray) -> np.ndarray:
    tail = np.asarray(trajectory, dtype=float)[int(0.15 * len(trajectory)) :, :]
    states = tail[:, 1:4]
    centered = states - np.mean(states, axis=0)
    coeffs = np.fft.rfft(centered, axis=0)
    dominant = int(np.argmax(np.abs(coeffs[1:, 0])) + 1)
    keep = np.zeros_like(coeffs)
    keep[0, :] = coeffs[0, :]
    keep[dominant, :] = coeffs[dominant, :]
    recon = np.fft.irfft(keep, n=states.shape[0], axis=0) + np.mean(states, axis=0)
    return np.column_stack((tail[:, 0], recon))


def _plot_reference_story(trajectory: np.ndarray, case_id: str, output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    slug = safe_name(case_id)
    tail = trajectory[int(0.15 * len(trajectory)) :, :]
    early = trajectory[: max(50, min(len(trajectory), len(trajectory) // 5)), :]
    reconstruction = _first_harmonic_reconstruction(trajectory)
    paths: list[str] = []
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(early[:, 1], early[:, 2], early[:, 3], color="blue", lw=2.0, label="inicio")
    ax.plot(tail[:, 1], tail[:, 2], tail[:, 3], color="red", lw=1.2, label="post-transitorio")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title=f"Danca reference: {case_id}")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig02d_{slug}_reference_story.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None)
        paths.append(str(path))
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[:, 1], tail[:, 2], tail[:, 3], color="red", lw=1.2, label="original")
    ax.plot(reconstruction[:, 1], reconstruction[:, 2], reconstruction[:, 3], "--", color="purple", lw=1.1, label="primer armonico")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title=f"Danca reference: {case_id}")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig03g_{slug}_harmonic_vs_original_3d.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_system_nyquist(q: float, case_id: str, output: Path) -> str:
    import matplotlib.pyplot as plt

    lure_system = get_system("chua-nonsmooth").lure
    if lure_system is None:
        raise RuntimeError("chua-nonsmooth does not expose a Lur'e representation.")
    omega = np.logspace(-5.0, np.log10(50.0), 2400)
    values = np.array([lure_transfer_function(float(w), q, lure_system) for w in omega])
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.plot(np.real(values), np.imag(values), lw=1.25, color="#0047ff", label=r"$W_q(i\omega)$")
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.axvline(0.0, color="#9ca3af", ls=":", lw=0.7)
    ax.set_xlabel(r"Re$(W_q(i\omega))$")
    ax.set_ylabel(r"Im$(W_q(i\omega))$")
    ax.set_title(f"Nyquist del sistema: {case_id}")
    ax.text(0.02, 0.02, "Sin cierre DF: semilla Danca no armonica", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return str(output)


def generate_reference_figures(root: Path, case_trajectories: dict[str, np.ndarray], q: float) -> dict[str, Any]:
    """Write the dynamic, spectral, and Nyquist diagnostics for each case."""

    output_dir = root / "candidate_diagnostic_figures"
    story_dir = root / "candidate_story_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    story_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for case_id, trajectory in case_trajectories.items():
        slug = safe_name(case_id)
        files.extend(_plot_reference_story(trajectory, case_id, story_dir))
        files.append(plot_phase_space(trajectory, output_dir / f"{slug}_phase_3d.png", title=f"Danca reference: {case_id}"))
        files.append(plot_phase_projections(trajectory, output_dir / f"{slug}_projections.png", title=f"Danca reference: {case_id}"))
        files.append(plot_time_series(trajectory, output_dir / f"{slug}_time_series.png", title=f"Danca reference: {case_id}"))
        for method in ("fft", "psd"):
            files.extend(plot_trajectory_spectra(trajectory, output_dir, method=method, prefix=slug))
        files.append(_plot_system_nyquist(q, case_id, output_dir / f"{slug}_nyquist_system.png"))
    return {
        "status": "generated",
        "note": "Nyquist represents the Lur'e system only; the Danca-located seed has no describing-function closure metadata.",
        "files": [str(Path(path).relative_to(root)) for path in files],
    }


def verify_reference(outdir: str | Path) -> Path:
    """Accredit the located Danca seed once with untruncated ABM and compare cases."""

    root = Path(outdir)
    cfg = read_json(root / "danca_abm_sphere_config.json")
    dcfg = _runtime_config(cfg)
    eqs = solve_equilibria(dcfg.params())
    seed = np.asarray(cfg["reference"]["x0"], dtype=float)
    case_trajectories: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        integrate = _case_integrator(case, dcfg)
        trajectory = integrate(seed)
        case_id = str(case["solver_case_id"])
        case_trajectories[case_id] = trajectory
        _write_trajectory(root / "reference_trajectories" / f"{safe_name(case_id)}.csv", trajectory)
        classified = classify_trajectory(trajectory, dcfg, eqs)
        return {
            "candidate_id": cfg["candidate_id"],
            **case,
            "seed_source": cfg["reference"]["source_summary"],
            "x0": float(seed[0]),
            "y0": float(seed[1]),
            "z0": float(seed[2]),
            "q": dcfg.q,
            "h": dcfg.h,
            "t_final": dcfg.t_final,
            "t_burn": dcfg.transient,
            **classified,
            "status": "ok",
        }

    accreditation_case = next(case for case in cfg["solver_cases"] if case["solver_case_id"] == "abm_full_history")
    accreditation = evaluate(accreditation_case)
    rows.append(accreditation)
    passed = accreditation["class"] == "bounded_nontrivial" and bool(accreditation["bounded"])
    if passed:
        for case in cfg["solver_cases"]:
            if case["solver_case_id"] != "abm_full_history":
                rows.append(evaluate(case))
        figures = generate_reference_figures(root, case_trajectories, dcfg.q)
    else:
        figures = {"status": "skipped_failed_abm_full_history_reference", "files": []}
    write_csv(root / "danca_reference_case_metrics.csv", rows, REFERENCE_FIELDS)
    summary = {
        "status": "passed_abm_full_history_reference" if passed else "failed_abm_full_history_reference",
        "verification_stage": "single_abm_full_history_numerical_continuation_check",
        "paper_seed_disclosure": "Danca does not publish the initial condition; this seed is loaded from the recorded ABM localization artifact.",
        "reference": cfg["reference"],
        "accreditation_case_id": "abm_full_history",
        "case_metrics_csv": "danca_reference_case_metrics.csv",
        "figures": figures,
    }
    write_json(root / "danca_reference_verification.json", summary)
    if not passed:
        raise RuntimeError("Danca-located seed was not reproduced as bounded_nontrivial by untruncated ABM.")
    return root / "danca_reference_verification.json"


def _require_verified_reference(root: Path, *, wait: bool = False, poll_sec: float = 30.0) -> None:
    path = root / "danca_reference_verification.json"
    while wait and not path.exists():
        time.sleep(float(poll_sec))
    if not path.exists():
        raise RuntimeError("Run the reference verification stage before equilibrium-ball controls.")
    if read_json(path).get("status") != "passed_abm_full_history_reference":
        raise RuntimeError("Equilibrium-ball controls require a passed untruncated ABM reference verification.")


def make_plan(outdir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.danca_source_dir).resolve()
    dcfg = _effective_config(_danca_config(source_dir), args)
    reference = _located_reference_seed(source_dir)
    solver_cases = _solver_cases(args)
    eqs = solve_equilibria(dcfg.params())
    radii = [float(item) for item in str(args.radii).split(",") if item.strip()]
    eq_names = [item for item in str(args.equilibria).split(",") if item.strip()]
    rng = np.random.default_rng(int(args.seed))
    base_rows: list[dict[str, Any]] = []
    for eq_id in eq_names:
        center = np.asarray(eqs[eq_id], dtype=float)
        for radius_index, radius in enumerate(radii):
            count = int(args.samples_per_radius) + radius_index * int(args.sample_growth_per_radius)
            for sample_id, x0 in enumerate(sample_uniform_ball(center, float(radius), count, rng)):
                base_rows.append(
                    {
                        "candidate_id": "danca2017_reference",
                        "q": dcfg.q,
                        "equilibrium_id": eq_id,
                        "radius": float(radius),
                        "sample_id": sample_id,
                        "sampling_mode": "ball",
                        "distance_from_equilibrium": float(np.linalg.norm(x0 - center)),
                        "batch_100": int(sample_id // 100 + 1),
                        "x0": float(x0[0]),
                        "y0": float(x0[1]),
                        "z0": float(x0[2]),
                    }
                )
    rows: list[dict[str, Any]] = []
    for case in solver_cases:
        for sample in base_rows:
            rows.append({"case_index": len(rows), **case, **sample})
    write_csv(root / "danca_sphere_plan.csv", rows)
    for name in ("run_config.json", "danca_reference_summary.json"):
        src = source_dir / name
        if src.exists():
            write_json(root / name, read_json(src))
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "danca_reference_abm_efork_memory_matrix_ball_controls",
        "danca_source_dir": str(source_dir),
        "candidate_id": "danca2017_reference",
        "reference": reference,
        "reference_accreditation": {
            "required_case_id": "abm_full_history",
            "verification_stage": "single_abm_full_history_numerical_continuation_check",
            "required_class": "bounded_nontrivial",
        },
        "solver_cases": solver_cases,
        "equilibria": {key: val.tolist() for key, val in eqs.items()},
        "tested_equilibria": eq_names,
        "radii": radii,
        "samples_per_radius": int(args.samples_per_radius),
        "sample_growth_per_radius": int(args.sample_growth_per_radius),
        "sampling_mode": "ball",
        "chunks": int(args.chunks),
        "random_seed": int(args.seed),
        "classification": {
            "mean_x_gap": float(args.mean_x_gap),
            "target_hit": "class_id in {1,2}; bounded_nontrivial with |mean_x_tail| <= mean_x_gap is unknown",
        },
        "contract": {
            "q": dcfg.q,
            "h": dcfg.h,
            "t_final": dcfg.t_final,
            "t_burn": dcfg.transient,
            "history_policy": "solver_case_matrix_full_history_and_finite_memory",
            "truncated_memory_length": float(args.memory_length),
            "equilibrium_tol": dcfg.equilibrium_tol,
            "divergence_norm": dcfg.divergence_norm,
            "nontrivial_range_tol": dcfg.nontrivial_range_tol,
            "store_stride": dcfg.store_stride,
        },
        "stages": [
            "reference_seed_accreditation_abm_full_history",
            "reference_solver_memory_comparison_and_figures",
            "equilibrium_ball_hiddenness_controls",
            "aggregate_hiddenness_by_solver_memory_case",
            "optional_strict_unknown_refinement",
        ],
        "planned_rows": len(rows),
        "planned_rows_per_solver_case": len(base_rows),
        "chain_strict_unknown_refinement": bool(args.chain_strict_unknown_refinement),
        "strict_refine_chunks": int(args.strict_refine_chunks),
    }
    write_json(root / "danca_abm_sphere_config.json", cfg)
    return cfg


def run_chunk(outdir: str | Path, chunk_id: int, chunks: int, *, wait_for_reference: bool = False) -> Path:
    force_single_openmp_thread_current_process()
    root = Path(outdir)
    _require_verified_reference(root, wait=wait_for_reference)
    cfg = read_json(root / "danca_abm_sphere_config.json")
    dcfg = _runtime_config(cfg)
    eqs = solve_equilibria(dcfg.params())
    cases = {str(case["solver_case_id"]): case for case in cfg["solver_cases"]}
    integrators: dict[str, Any] = {}
    plan = read_csv_rows(root / "danca_sphere_plan.csv")
    path = root / f"danca_sphere_raw_chunk_{int(chunk_id):03d}.csv"
    if path.exists():
        path.unlink()
    done = root / f"danca_sphere_raw_chunk_{int(chunk_id):03d}.done"
    if done.exists():
        done.unlink()
    rows = 0
    for item in plan:
        idx = int(float(item["case_index"]))
        if idx % int(chunks) != int(chunk_id):
            continue
        seed = np.array([_float(item["x0"]), _float(item["y0"]), _float(item["z0"])], dtype=float)
        solver_case_id = str(item["solver_case_id"])
        case = cases[solver_case_id]
        if solver_case_id not in integrators:
            integrators[solver_case_id] = _case_integrator(case, dcfg)
        started = time.time()
        try:
            traj = integrators[solver_case_id](seed)
            cls = classify_trajectory(traj, dcfg, eqs)
            tail = traj[traj[:, 0] >= dcfg.transient, 1:4]
            mean_tail = np.mean(tail, axis=0) if tail.size else np.array([np.nan, np.nan, np.nan])
            row = {
                **item,
                **cls,
                "danca_class": cls.get("class", ""),
                "mean_x_tail": float(mean_tail[0]),
                "mean_y_tail": float(mean_tail[1]),
                "mean_z_tail": float(mean_tail[2]),
                "status": "ok",
                "elapsed_sec": time.time() - started,
                "backend": case["backend"],
                "history_policy": case["history_policy"],
                "memory_length": case["memory_length"],
                "h": dcfg.h,
                "t_final": dcfg.t_final,
                "t_burn": dcfg.transient,
            }
            cid = _coarse_danca_class(row, mean_x_gap=float(cfg["classification"]["mean_x_gap"]))
            row["class_id"] = cid
            row["class_label"] = class_label(cid)
            row["target_hit"] = cid in (1, 2)
        except Exception as exc:
            row = {
                **item,
                "status": "exception",
                "error": repr(exc),
                "elapsed_sec": time.time() - started,
                "class_id": 5,
                "class_label": class_label(5),
                "target_hit": False,
                "danca_class": "numerical_failure",
                "backend": case["backend"],
                "history_policy": case["history_policy"],
                "memory_length": case["memory_length"],
                "h": dcfg.h,
                "t_final": dcfg.t_final,
                "t_burn": dcfg.transient,
            }
        append_csv(path, row, RAW_FIELDS)
        rows += 1
        if rows % 25 == 0:
            print(f"danca solver/memory ball chunk {chunk_id}: {rows} rows", flush=True)
    write_json(done, {"chunk_id": int(chunk_id), "rows": rows, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return path


def _plot_ball_control_figures(root: Path, cfg: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir = root / "ball_sampling_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"target_positive": "#dc2626", "target_negative": "#dc2626", "equilibrium": "#111827", "unknown": "#2563eb", "infinity": "#f59e0b", "numerical_failure": "#6b7280"}
    files: list[str] = []
    for case in cfg["solver_cases"]:
        case_id = str(case["solver_case_id"])
        selected = [row for row in rows if str(row.get("solver_case_id", "")) == case_id]
        fig = plt.figure(figsize=(8.0, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        for label, color in colors.items():
            samples = [row for row in selected if str(row.get("class_label", "")) == label]
            if samples:
                xyz = np.asarray([[_float(row["x0"]), _float(row["y0"]), _float(row["z0"])] for row in samples], dtype=float)
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=7, alpha=0.58, c=color, label=label)
        for eq_id, center in cfg["equilibria"].items():
            point = np.asarray(center, dtype=float)
            ax.scatter([point[0]], [point[1]], [point[2]], c="#000000", marker="x", s=38)
            ax.text(point[0], point[1], point[2], eq_id, fontsize=8)
        ax.set_xlabel("x0")
        ax.set_ylabel("y0")
        ax.set_zlabel("z0")
        ax.set_title(f"Controles esfericos Danca: {case_id}")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        path = output_dir / f"{safe_name(case_id)}_ball_controls.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        files.append(str(path.relative_to(root)))
    return files


def aggregate(outdir: str | Path, *, wait: bool = False, poll_sec: float = 60.0) -> Path:
    root = Path(outdir)
    cfg = read_json(root / "danca_abm_sphere_config.json")
    chunks = int(cfg["chunks"])
    while wait:
        if all((root / f"danca_sphere_raw_chunk_{idx:03d}.done").exists() for idx in range(chunks)):
            break
        time.sleep(float(poll_sec))
    rows: list[dict[str, str]] = []
    for idx in range(chunks):
        rows.extend(read_csv_rows(root / f"danca_sphere_raw_chunk_{idx:03d}.csv"))
    rows.sort(key=lambda row: int(float(row.get("case_index", 0))))
    write_csv(root / "danca_sphere_raw.csv", rows, RAW_FIELDS)

    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("solver_case_id", "")), str(row.get("equilibrium_id", "")), _float(row.get("radius")))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (case_id, eq_id, radius), sub in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        counts = Counter(row.get("class_label", "") for row in sub)
        targets = counts.get("target_positive", 0) + counts.get("target_negative", 0)
        n = len(sub)
        summary_rows.append(
            {
                "candidate_id": cfg["candidate_id"],
                "solver_case_id": case_id,
                "solver": sub[0].get("solver", ""),
                "history_policy": sub[0].get("history_policy", ""),
                "memory_length": sub[0].get("memory_length", ""),
                "equilibrium_id": eq_id,
                "radius": radius,
                "n_executed": n,
                "n_target_hits": targets,
                "target_hit_fraction": float(targets / max(n, 1)),
                "n_equilibrium": counts.get("equilibrium", 0),
                "n_target_positive": counts.get("target_positive", 0),
                "n_target_negative": counts.get("target_negative", 0),
                "n_infinity": counts.get("infinity", 0),
                "n_unknown": counts.get("unknown", 0),
                "n_numerical_failure": counts.get("numerical_failure", 0),
            }
        )
    write_csv(root / "danca_sphere_cumulative_summary.csv", summary_rows, SUMMARY_FIELDS)
    case_decisions: list[dict[str, Any]] = []
    for case in cfg["solver_cases"]:
        case_id = str(case["solver_case_id"])
        selected = [row for row in summary_rows if row["solver_case_id"] == case_id]
        targets = sum(int(row["n_target_hits"]) for row in selected)
        unknown = sum(int(row["n_unknown"]) for row in selected)
        executed = sum(int(row["n_executed"]) for row in selected)
        case_decisions.append(
            {
                **case,
                "tested_ball_trajectories": executed,
                "total_target_hits": targets,
                "total_unknown": unknown,
                "hiddenness_status": "rejected_self_excited_contact" if targets > 0 else "compatible_with_hiddenness_under_tested_radii",
            }
        )
    total_targets = sum(int(case["total_target_hits"]) for case in case_decisions)
    total_unknown = sum(int(case["total_unknown"]) for case in case_decisions)
    decision = {
        "candidate_id": cfg["candidate_id"],
        "tested_ball_trajectories": len(rows),
        "planned_rows": int(cfg["planned_rows"]),
        "total_target_hits": total_targets,
        "total_unknown": total_unknown,
        "case_decisions": case_decisions,
        "hiddenness_status": "rejected_in_at_least_one_solver_memory_case" if total_targets > 0 else "compatible_in_all_tested_solver_memory_cases",
        "notes": "Each case is interpreted independently. A target hit from any equilibrium ball refutes hiddenness under that tested solver-memory contract.",
    }
    write_json(root / "danca_sphere_decision.json", decision)
    ball_figures = _plot_ball_control_figures(root, cfg, rows)
    summary = {
        "status": "ok" if len(rows) == int(cfg["planned_rows"]) else "partial",
        "raw_rows": len(rows),
        "planned_rows": int(cfg["planned_rows"]),
        "summary_csv": str(root / "danca_sphere_cumulative_summary.csv"),
        "decision_json": str(root / "danca_sphere_decision.json"),
        "ball_sampling_figures": ball_figures,
        "decision": decision,
    }
    write_json(root / "danca_abm_sphere_summary.json", summary)
    return root / "danca_abm_sphere_summary.json"


def refine_after_aggregate(outdir: str | Path, *, wait: bool = True, poll_sec: float = 60.0) -> Path:
    root = Path(outdir)
    cfg = read_json(root / "danca_abm_sphere_config.json")
    while wait and not (root / "danca_abm_sphere_summary.json").exists():
        time.sleep(float(poll_sec))
    if len(cfg.get("solver_cases", [])) > 1:
        write_json(
            root / "strict_unknown_refinement_launch.json",
            {
                "status": "skipped_solver_memory_matrix",
                "reason": "The existing strict-target adapter is single-backend ABM only; it must not relabel EFORK3 or truncated-memory rows under a different contract.",
                "hiddenness_stage_completed": "equilibrium_ball_hiddenness_controls",
            },
        )
        return root / "strict_unknown_refinement_launch.json"
    if not bool(cfg.get("chain_strict_unknown_refinement", True)):
        write_json(root / "strict_unknown_refinement_launch.json", {"status": "skipped"})
        return root / "strict_unknown_refinement_launch.json"
    refined_dir = root / "strict_unknown_refinement"
    script = PROJECT_ROOT / "tools" / "cli" / "strict_target_refinement.py"
    cmd = [
        sys.executable,
        str(script),
        "--job",
        "launch",
        "--mode",
        "sphere-danca",
        "--source-dir",
        str(root),
        "--source-csv",
        str(root / "danca_sphere_raw.csv"),
        "--source-labels",
        "unknown",
        "--output-dir",
        str(refined_dir),
        "--chunks",
        str(int(cfg.get("strict_refine_chunks", 4))),
    ]
    logs = root / "logs"
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=force_single_openmp_thread_env(os.environ.copy()),
        stdout=(logs / "strict_unknown_refinement_launch.out").open("ab"),
        stderr=(logs / "strict_unknown_refinement_launch.err").open("ab"),
        start_new_session=True,
        close_fds=True,
    )
    manifest = {"status": "launched_strict_unknown_refinement", "pid": proc.pid, "cmd": cmd, "output_dir": str(refined_dir)}
    write_json(root / "strict_unknown_refinement_launch.json", manifest)
    return root / "strict_unknown_refinement_launch.json"


def launch(outdir: str | Path, args: argparse.Namespace) -> Path:
    root = Path(outdir)
    cfg = make_plan(root, args)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    script = Path(args.script_path).resolve()
    launched: list[dict[str, Any]] = []
    cmd = [sys.executable, str(script), "--job", "reference", "--output-dir", str(root)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=(logs / "reference.out").open("ab"),
        stderr=(logs / "reference.err").open("ab"),
        start_new_session=True,
        close_fds=True,
    )
    launched.append({"job": "reference", "pid": proc.pid, "cmd": cmd})
    for idx in range(int(args.chunks)):
        cmd = [sys.executable, str(script), "--job", "chunk", "--output-dir", str(root), "--chunk-id", str(idx), "--chunks", str(args.chunks), "--wait"]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=(logs / f"chunk_{idx:03d}.out").open("ab"),
            stderr=(logs / f"chunk_{idx:03d}.err").open("ab"),
            start_new_session=True,
            close_fds=True,
        )
        launched.append({"job": f"chunk_{idx:03d}", "pid": proc.pid, "cmd": cmd})
    for job in ("aggregate", "refine-after-aggregate"):
        cmd = [sys.executable, str(script), "--job", job, "--output-dir", str(root), "--wait"]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=(logs / f"{job}.out").open("ab"),
            stderr=(logs / f"{job}.err").open("ab"),
            start_new_session=True,
            close_fds=True,
        )
        launched.append({"job": job, "pid": proc.pid, "cmd": cmd})
    manifest = {"output_dir": str(root), "planned_rows": cfg["planned_rows"], "chunks": int(cfg["chunks"]), "launched": launched}
    write_json(root / "launch_manifest.json", manifest)
    print(manifest, flush=True)
    return root / "launch_manifest.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Danca reference control matrix: ABM/EFORK3 with full and finite memory plus equilibrium-ball hiddenness tests.")
    parser.add_argument("--job", choices=["launch", "reference", "chunk", "aggregate", "refine-after-aggregate"], default="launch")
    parser.add_argument("--output-dir", default=str(ROOT_OUTPUTS / "danca_abm_sphere_controls_20260520"))
    parser.add_argument("--danca-source-dir", default=str(DEFAULT_DANCA_SOURCE))
    parser.add_argument("--h", type=float, choices=(0.01, 0.005, 0.001), default=0.01)
    parser.add_argument("--memory-length", type=float, default=40.0, help="Lm para los casos ABM y EFORK3 de memoria truncada.")
    parser.add_argument("--t-final", type=float, default=-1.0, help="Si es positivo sustituye el horizonte del artefacto Danca fuente.")
    parser.add_argument("--t-burn", type=float, default=-1.0, help="Si no es negativo sustituye el transitorio del artefacto Danca fuente.")
    parser.add_argument("--equilibria", default="E0,E+,E-")
    parser.add_argument("--radii", default="1e-5,3e-5,1e-4,3e-4,1e-3,1e-2")
    parser.add_argument("--samples-per-radius", type=int, default=100)
    parser.add_argument("--sample-growth-per-radius", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--mean-x-gap", type=float, default=0.75)
    parser.add_argument("--chain-strict-unknown-refinement", action="store_true", default=True)
    parser.add_argument("--no-chain-strict-unknown-refinement", dest="chain_strict_unknown_refinement", action="store_false")
    parser.add_argument("--strict-refine-chunks", type=int, default=4)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--script-path", default=str(PROJECT_ROOT / "tools" / "cli" / "danca_abm_sphere_controls.py"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    outdir = Path(args.output_dir).resolve()
    if args.job == "launch":
        launch(outdir, args)
    elif args.job == "reference":
        verify_reference(outdir)
    elif args.job == "chunk":
        run_chunk(outdir, int(args.chunk_id), int(args.chunks), wait_for_reference=bool(args.wait))
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))
    elif args.job == "refine-after-aggregate":
        refine_after_aggregate(outdir, wait=bool(args.wait))


if __name__ == "__main__":
    main()
