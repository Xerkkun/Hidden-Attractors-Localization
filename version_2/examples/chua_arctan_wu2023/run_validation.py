#!/usr/bin/env python3
"""Run Wu2023 algebra and optional reported-initial-condition trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.adm_wu2023 import adm_wu2023_integrate_from_config
from hidden_attractors.validation.chua_arctan_wu2023 import write_algebra_validation


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "chua_arctan_wu2023_caputo.json"


def _load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _run_reported_trajectories(config: dict, output_dir: Path) -> dict:
    numerical = config["numerical_contract"]
    q = float(numerical["q"])
    h = float(numerical["h"])
    t_final = float(numerical["t_final"])
    integrator_config = {
        **config["parameters"],
        "q": q,
        "h": h,
        "N": int(numerical["N"]),
        "t_final": t_final,
        "divergence_norm": float(numerical.get("divergence_norm", 120.0)),
    }
    settings = dict(config["post_continuation_periodicity"])
    settings["t_transient"] = t_final * float(settings["discard_transient_fraction"])
    results = {}
    for name, state in config["initial_conditions_reported"].items():
        times, states, status, info = adm_wu2023_integrate_from_config(
            integrator_config,
            np.asarray(state, dtype=float),
            label=name,
        )
        trajectory = np.column_stack((times, states))
        periodicity = classify_post_transient_periodicity(trajectory, h=h, config=settings)
        np.savetxt(
            output_dir / f"{name}_trajectory.csv",
            trajectory,
            delimiter=",",
            header="t,x,y,z",
            comments="",
        )
        results[name] = {
            "initial_condition": state,
            "trajectory": f"03_reported_initial_conditions/{name}_trajectory.csv",
            "integration_status": status,
            "integrator": info["integrator"],
            "backend": info["integrator_class"],
            "N": int(numerical["N"]),
            "memory_policy": numerical["memory_policy"],
            "memory_length": numerical["memory_length"],
            "caputo_history_accumulated": numerical["caputo_history_accumulated"],
            "adm_metadata": info,
            "periodicity": periodicity,
            "attractor_status": "not_tested",
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-trajectories", action="store_true")
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Deprecated for Wu2023 ADM reproduction; ADM has no accumulated Caputo history.",
    )
    args = parser.parse_args()
    if args.full_history:
        parser.error("--full-history is not valid for Wu2023 ADM reproduction.")
    config = _load_config()
    output_dir = args.output_dir or ROOT / config["validation_output_root"]
    algebra_path = output_dir / "01_algebra" / "chua_arctan_wu2023_algebra.json"
    algebra = write_algebra_validation(algebra_path)
    result = {
        "case_id": config["case_id"],
        "configuration": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "algebra": str(algebra_path.relative_to(ROOT)).replace("\\", "/"),
        "algebra_status": algebra["status"],
        "attractor_status": "not_tested",
        "hiddenness_requirement": "Must test neighborhoods of E0, E+ and E- under a robust target reference.",
    }
    if args.run_trajectories:
        dynamics_dir = output_dir / "03_reported_initial_conditions"
        dynamics_dir.mkdir(parents=True, exist_ok=True)
        result["reported_initial_conditions"] = _run_reported_trajectories(config, dynamics_dir)
        labels = {
            row["periodicity"]["candidate_label"]
            for row in result["reported_initial_conditions"].values()
        }
        result["dynamic_status"] = (
            "rejected_periodic_reported_initial_conditions"
            if labels <= {"regular_periodic_rejected", "thin_periodic_rejected"}
            else "pending_robustness_and_hiddenness_controls"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
