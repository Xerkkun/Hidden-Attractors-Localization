#!/usr/bin/env python3
"""Reproducibility lanes for Wu2023 arctan Chua and the c590 audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
VERSION2 = EXAMPLE_DIR.parents[1]
ROOT = VERSION2.parent
for path in (VERSION2, ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from hidden_attractors.models.chua import chua_parameters
from hidden_attractors.plotting.generate_publication_figures import generate_all_publication_figures
from hidden_attractors.seed_generation.chua_arctan_wu2023 import format_arctan_wu2023_seed_report
from hidden_attractors.validation.chua_arctan_wu2023 import write_algebra_validation
from hidden_attractors.workflows.centered_lure_df import run_centered_lure_df_workflow
from tools.arctan_hidden_screen import make_parser as make_screen_parser, run as run_screen
from tools.chua_candidate_extended_hiddenness import make_parser as make_hiddenness_parser, run as run_hiddenness

CONFIG_PATH = EXAMPLE_DIR / "reproducibility.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [EXAMPLE_DIR / path, VERSION2 / path, ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (EXAMPLE_DIR / path).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_published(cfg: dict[str, Any], *, run_trajectories: bool = False) -> None:
    lane = cfg["published_wu2023"]
    params = chua_parameters(model="arctan", **{k: lane["parameters"][k] for k in ("alpha", "beta", "gamma", "a1", "a2", "rho")})
    outdir = resolve(lane["output_dir"])
    numerical = lane["numerical_contract"]
    seed_cfg = lane["seed_generation"]
    seed_report = format_arctan_wu2023_seed_report(
        q=float(numerical["q"]),
        params=params,
        nscan=20000,
        transfer_mode=str(seed_cfg["transfer_mode"]),
    )
    seed_report["configuration"] = str(resolve(lane["source_config"]).relative_to(VERSION2)).replace("\\", "/")
    seed_report["interpretation"] = lane["interpretation"]
    write_json(outdir / "02_lure_df" / "centered_seeds.json", seed_report)
    algebra = write_algebra_validation(outdir / "01_algebra" / "chua_arctan_wu2023_algebra.json")
    write_json(outdir / "validation_summary.json", {
        "case_id": cfg["case_id"],
        "published_lane": lane,
        "seed_report": "02_lure_df/centered_seeds.json",
        "algebra_status": algebra["status"],
        "interpretation": "published_reproduction_not_hiddenness_evidence",
    })
    if run_trajectories:
        subprocess.run([sys.executable, str(EXAMPLE_DIR / "run_validation.py"), "--run-trajectories"], check=True)
    print(f"published_output={outdir}")


def run_search(cfg: dict[str, Any], *, quick: bool = False) -> None:
    lane = cfg["proposed_c590"]
    screen = dict(lane["screen"])
    if quick:
        screen.update({"t_final": 20.0, "t_burn": 10.0, "zero_one_samples": "200", "max_cases": 1})
    argv = [
        "--output-dir", str(resolve(screen["output_dir"])),
        "--q-values", str(screen["q_values"]),
        "--alpha-values", str(screen["alpha_values"]),
        "--beta-values", str(screen["beta_values"]),
        "--gamma", str(screen["gamma"]),
        "--a1", str(screen["a1"]),
        "--a2", str(screen["a2"]),
        "--rho", str(screen["rho"]),
        "--seed", ",".join(str(value) for value in lane["seed"]),
        "--seed-strategy", str(screen["seed_strategy"]),
        "--h", str(screen["h"]),
        "--t-final", str(screen["t_final"]),
        "--t-burn", str(screen["t_burn"]),
        "--integrator", str(screen["integrator"]),
        "--memory-mode", str(screen["memory_mode"]),
        "--zero-one-samples", str(screen["zero_one_samples"]),
        "--max-cases", str(screen["max_cases"]),
    ]
    summary = run_screen(make_screen_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, default=str))


def run_continuation(cfg: dict[str, Any], *, quick: bool = False) -> None:
    wf = dict(cfg["proposed_centered_df"]["workflow_config"])
    wf["output_dir"] = str(resolve(wf["output_dir"]))
    if quick:
        wf["grid_size_omega"] = 160
        wf["grid_size_amplitude"] = 120
        wf["continuation"]["eta_values"] = [0.001, 0.03, 0.2, 1.0]
        wf["continuation"]["t_transient"] = 2.0
        wf["continuation"]["t_keep"] = 2.0
        wf["final_simulation"]["t_final"] = 10.0
        wf["final_simulation"]["t_burn"] = 5.0
    result = run_centered_lure_df_workflow(wf)
    write_json(Path(wf["output_dir"]) / "reproducibility_result.json", result)
    print(f"continuation_output={wf['output_dir']}")


def run_verification(cfg: dict[str, Any], *, quick: bool = False) -> None:
    lane = cfg["proposed_c590"]
    verify = dict(lane["verification"])
    if quick:
        verify.update({
            "radii": "1e-5,1e-4",
            "directions_per_radius_list": "6,6",
            "workers": 1,
            "checkpoint_name": "smoke_hiddenness_checkpoint.jsonl",
            "matrix_name": "smoke_hiddenness_matrix.json",
            "summary_name": "smoke_candidate_summary.json",
            "rows_name": "smoke_hiddenness_rows.csv",
            "config_name": "smoke_hiddenness_run_config.json",
        })
    argv = [
        "--candidate-dir", str(resolve(lane["candidate_dir"])),
        "--radii", str(verify["radii"]),
        "--directions-per-radius-list", str(verify["directions_per_radius_list"]),
        "--h", str(verify["h"]),
        "--t-final", str(verify["t_final"]),
        "--t-burn", str(verify["t_burn"]),
        "--workers", str(verify["workers"]),
        "--checkpoint-name", str(verify["checkpoint_name"]),
        "--matrix-name", str(verify["matrix_name"]),
        "--summary-name", str(verify["summary_name"]),
        "--rows-name", str(verify["rows_name"]),
        "--config-name", str(verify["config_name"]),
        "--resume",
    ]
    if not quick:
        argv.append("--promote")
    result = run_hiddenness(make_hiddenness_parser().parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


def run_figures(cfg: dict[str, Any]) -> None:
    candidate_dir = resolve(cfg["proposed_c590"]["candidate_dir"])
    generate_all_publication_figures(str(candidate_dir), {})
    print(f"figures_source={candidate_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["published", "search", "continuation", "verification", "figures"],
        default=None,
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--all", action="store_true", help="include the long hiddenness verification lane")
    parser.add_argument("--run-published-trajectories", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    steps = args.steps or (["published", "search", "continuation", "verification", "figures"] if args.all else ["published", "search", "continuation", "figures"])
    if "published" in steps:
        run_published(cfg, run_trajectories=args.run_published_trajectories)
    if "search" in steps:
        run_search(cfg, quick=args.quick)
    if "continuation" in steps:
        run_continuation(cfg, quick=args.quick)
    if "verification" in steps:
        run_verification(cfg, quick=args.quick)
    if "figures" in steps:
        run_figures(cfg)


if __name__ == "__main__":
    main()

