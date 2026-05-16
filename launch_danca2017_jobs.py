#!/usr/bin/env python3
"""Launch Danca/project verification jobs as macOS launchd agents.

The goal is to keep long numerical jobs alive independently of the interactive
terminal or SSH session.  Each job is a separate process.  Python workers use
process-based parallelism and nested OpenMP is forced to one thread per worker.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from parallel_policy import force_single_openmp_thread_env


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable or "python3"


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_lure_timestamped_config(stamp: str, work_dir: Path) -> Path:
    canonical = ROOT / "outputs" / "lure_biased_multiparam_q09998"
    outdir = ROOT / "outputs" / f"lure_biased_multiparam_q09998_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in [
        "biased_lure_candidates.csv",
        "biased_lure_seed_bank.csv",
        "continuation_summary.csv",
        "continuation_paths.csv",
        "continuation_survivors.csv",
        "search_run_metadata.json",
    ]:
        copy_if_exists(canonical / name, outdir / name)
    cfg = {
        "model": "piecewise",
        "q": 0.9998,
        "frac_order": 0.9998,
        "enforce_q_consistency": True,
        "params": {
            "alpha_chua": 8.4562,
            "beta": 12.0732,
            "gamma_chua": 0.0052,
            "m0": -0.1768,
            "m1": -1.1468,
            "a1": 0.4,
            "a2": -1.5585,
            "rho": 1.0,
        },
        "outputs": {"root": str(outdir)},
        "search": {
            "A_min": 0.5,
            "A_max": 8.0,
            "omega_min": 1.2,
            "omega_max": 3.0,
            "sigma0_min": -4.0,
            "sigma0_max": 4.0,
            "n_samples": 5000,
            "quadrature_points": 4096,
            "K_rhoH": 20,
            "local_refine_top": 100,
            "residual_keep": 0.05,
            "rhoH_keep": 0.30,
            "residual_priority": 0.02,
            "rhoH_priority": 0.15,
            "random_seed": 20260514,
            "source_hint_manifest": "outputs/lure_route/lure_candidates_manifest.csv",
        },
        "seeds": {
            "phases": [
                0.0,
                0.7853981633974483,
                1.5707963267948966,
                2.356194490192345,
                3.141592653589793,
                3.9269908169872414,
                4.71238898038469,
                5.497787143782138,
            ]
        },
        "continuation": {
            "enabled": False,
            "routes": ["C1"],
            "max_candidates": 6,
            "max_seeds_per_candidate": 1,
            "eta_start": 0.0,
            "eta_target": 1.0,
            "eta_steps": 9,
            "q_fixed": 0.9998,
            "h": 0.01,
            "memory_length": 20,
            "t_block": 200,
            "n_blocks": 8,
            "survivor_h": 0.01,
            "survivor_memory_length": 40,
            "survivor_t_final": 1500,
            "smooth_width": 0.2,
            "divergence_norm": 120,
            "equilibrium_tol": 0.001,
            "nontrivial_range": 0.01,
        },
        "early_equilibrium_filter": {
            "enabled": False,
            "rho": 1.0e-5,
            "h": 0.01,
            "memory_length": 40,
            "t_final": 1500,
        },
        "robustness": {
            "enabled": False,
            "cases": {
                "R0": {"h": 0.01, "memory_length": 40, "t_final": 3000},
                "R1": {"h": 0.005, "memory_length": 40, "t_final": 3000},
                "R2": {"h": 0.005, "memory_length": 80, "t_final": 6000},
            },
        },
        "cost_guard": {
            "max_simulations_without_force": 2000,
            "max_estimated_hours_without_force": 12,
        },
    }
    path = work_dir / "lure_biased_multiparam_q09998_timestamped_config.json"
    write_json(path, cfg)
    return path


def make_plist(
    *,
    label: str,
    argv: List[str],
    stdout_path: Path,
    stderr_path: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "Label": label,
        "ProgramArguments": argv,
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "EnvironmentVariables": env,
    }


def launch_plist(plist_path: Path, label: str, *, dry_run: bool) -> Dict[str, Any]:
    domain = f"gui/{os.getuid()}"
    if dry_run:
        return {"label": label, "plist": str(plist_path), "launched": False, "dry_run": True}
    bootstrap = subprocess.run(
        ["launchctl", "bootstrap", domain, str(plist_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if bootstrap.returncode != 0:
        return {
            "label": label,
            "plist": str(plist_path),
            "launched": False,
            "returncode": bootstrap.returncode,
            "stdout": bootstrap.stdout,
            "stderr": bootstrap.stderr,
        }
    kick = subprocess.run(
        ["launchctl", "kickstart", "-k", f"{domain}/{label}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "label": label,
        "plist": str(plist_path),
        "launched": kick.returncode == 0,
        "bootstrap_stdout": bootstrap.stdout,
        "bootstrap_stderr": bootstrap.stderr,
        "kickstart_returncode": kick.returncode,
        "kickstart_stdout": kick.stdout,
        "kickstart_stderr": kick.stderr,
    }


def launch_popen(job: Dict[str, Any], *, dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        return {"label": job["label"], "launched": False, "dry_run": True, "backend": "popen"}
    stdout_path = Path(job["stdout"])
    stderr_path = Path(job["stderr"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    out = stdout_path.open("ab")
    err = stderr_path.open("ab")
    proc = subprocess.Popen(
        job["argv"],
        cwd=str(ROOT),
        stdout=out,
        stderr=err,
        env=job["env"],
        start_new_session=True,
        close_fds=True,
    )
    return {
        "label": job["label"],
        "launched": True,
        "backend": "popen",
        "pid": proc.pid,
    }


def build_jobs(args: argparse.Namespace, stamp: str, work_dir: Path, logs_dir: Path) -> List[Dict[str, Any]]:
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    env["HIDDEN_ATTRACTORS_FRAC_ORDER"] = "0.9998"
    env["HIDDEN_ATTRACTORS_MODEL"] = "piecewise"
    cache_root = ROOT / ".runtime_cache"
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg_cache").mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg_cache")

    danca_out = ROOT / "outputs" / f"danca2017_chua_abm_{stamp}"
    machado_out = ROOT / "outputs" / "extended_search" / f"machado_targeted_verification_lm10_{stamp}"
    machado_cfg = work_dir / "machado_targeted_verification_lm10_config.json"
    write_json(
        machado_cfg,
        {
            "q": 0.9998,
            "source_summary": "runs_machado_sweep_fast/chua_piecewise/machado_sweep/machado_sweep_summary.json",
            "previous_corrida1_dir": "outputs/extended_search/corrida1",
            "output_dir": str(machado_out),
            "memory_contract": {
                "max_memory_length": 10,
                "notes": "Hardware-oriented run: every EFORK stage must keep Lm <= 10.",
            },
            "candidates": [
                "branch_0_mu_4p00000_theta_0p00000",
                "branch_0_mu_2p00000_theta_3p92699",
            ],
            "reference_attractor": {"enabled": True, "h": 0.01, "memory_length": 10, "t_final": 3000},
            "targeted_equilibrium_controls": {
                "enabled": True,
                "equilibria": ["E-", "E0", "E+"],
                "radii": [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
                "h": 0.01,
                "memory_length": 10,
                "t_final": 3000,
            },
            "reproduction": {
                "enabled": True,
                "h": 0.005,
                "memory_length_values": [1, 10],
                "t_final": 3000,
            },
            "attractor_robustness": {
                "enabled": True,
                "cases": {
                    "Lm1_h010": {"h": 0.01, "memory_length": 1, "t_final": 3000},
                    "Lm5_h010": {"h": 0.01, "memory_length": 5, "t_final": 3000},
                    "Lm10_h010": {"h": 0.01, "memory_length": 10, "t_final": 3000},
                    "Lm10_h005": {"h": 0.005, "memory_length": 10, "t_final": 3000},
                },
            },
            "classification": {
                "divergence_norm": 120.0,
                "equilibrium_radius": 1.0e-3,
                "section_tolerance": 0.12,
                "min_section_matches": 20,
                "hit_fraction_required": 0.70,
                "tail_fraction": 0.20,
                "nontrivial_range_tol": 1.0e-2,
            },
            "cost_guard": {
                "max_trajectories_without_force": 1000,
                "stop_after_first_target_hit_in_controls": True,
                "stop_after_first_reproduced_target": True,
            },
        },
    )
    lure_cfg = prepare_lure_timestamped_config(stamp, work_dir)

    jobs = [
        {
            "name": "danca_abm_fig3_hiddenness",
            "label": f"com.hiddenattractors.danca2017.abm.{stamp}",
            "argv": [
                PYTHON,
                str(ROOT / "danca2017_chua_abm_replication.py"),
                "--output-dir",
                str(danca_out),
                "--job",
                "all",
                "--workers",
                str(args.workers),
                "--h",
                str(args.h),
                "--t-final",
                str(args.t_final),
                "--transient",
                str(args.transient),
                "--local-samples-per-unstable-eq",
                str(args.local_samples_per_unstable_eq),
                "--figure-local-trajectories",
                str(args.figure_local_trajectories),
                "--store-stride",
                str(args.store_stride),
            ],
            "output_dir": str(danca_out),
        },
        {
            "name": "machado_lm10_targeted_robustness",
            "label": f"com.hiddenattractors.danca2017.machado_lm10.{stamp}",
            "argv": [
                PYTHON,
                str(ROOT / "machado_targeted_verification.py"),
                "--candidate-id",
                "all",
                "--config",
                str(machado_cfg),
                "--output-dir",
                str(machado_out),
                "--max-trajectories",
                str(args.machado_max_trajectories),
                "--force",
            ],
            "output_dir": str(machado_out),
        },
        {
            "name": "lure_rank0003_rank0005_missing_controls",
            "label": f"com.hiddenattractors.danca2017.lure_missing.{stamp}",
            "argv": [
                PYTHON,
                str(ROOT / "lure_biased_multiparam_continuation.py"),
                "--config",
                str(lure_cfg),
                "--post-continuation-only",
                "--survivor-id",
                "lure_biased_q_0p99980_rank_0003",
                "--survivor-id",
                "lure_biased_q_0p99980_rank_0005",
                "--execute-early-filter",
                "--execute-robustness",
                "--force",
            ],
            "output_dir": str(ROOT / "outputs" / f"lure_biased_multiparam_q09998_{stamp}"),
        },
    ]

    if args.only:
        wanted = set(args.only)
        jobs = [job for job in jobs if job["name"] in wanted]

    for job in jobs:
        job["stdout"] = str(logs_dir / f"{job['name']}.out")
        job["stderr"] = str(logs_dir / f"{job['name']}.err")
        job["env"] = env
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Danca 2017 replication and project verification jobs with launchd.")
    parser.add_argument("--backend", choices=["launchd", "popen"], default="launchd")
    parser.add_argument("--only", action="append", default=[], help="Launch only a named job; can be repeated.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--t-final", type=float, default=500.0)
    parser.add_argument("--transient", type=float, default=250.0)
    parser.add_argument("--local-samples-per-unstable-eq", type=int, default=100)
    parser.add_argument("--figure-local-trajectories", type=int, default=80)
    parser.add_argument("--store-stride", type=int, default=1, help="Use 1 for Danca full saved trajectory; ABM always retains full history internally.")
    parser.add_argument("--machado-max-trajectories", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = timestamp()
    work_dir = ROOT / "logs" / f"danca2017_launch_{stamp}"
    logs_dir = work_dir / "job_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, stamp, work_dir, logs_dir)
    results: List[Dict[str, Any]] = []
    for job in jobs:
        plist_path = work_dir / f"{job['name']}.plist"
        plist = make_plist(
            label=job["label"],
            argv=job["argv"],
            stdout_path=Path(job["stdout"]),
            stderr_path=Path(job["stderr"]),
            env=job["env"],
        )
        with plist_path.open("wb") as f:
            plistlib.dump(plist, f)
        if args.backend == "launchd":
            launch_result = launch_plist(plist_path, job["label"], dry_run=args.dry_run)
        else:
            launch_result = launch_popen(job, dry_run=args.dry_run)
        results.append({k: v for k, v in job.items() if k != "env"} | launch_result)
    manifest = {
        "stamp": stamp,
        "dry_run": bool(args.dry_run),
        "backend": args.backend,
        "policy": "Independent OS processes; Danca ABM keeps full Caputo history, no finite-memory truncation; Python workers use processes when available and OMP_NUM_THREADS=1.",
        "jobs": results,
    }
    manifest_path = work_dir / "launch_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
