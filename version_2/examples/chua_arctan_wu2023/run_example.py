#!/usr/bin/env python3
"""Reproduce the fixed Wu2023 arctan-Chua validation record."""

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
from hidden_attractors.seed_generation.chua_arctan_wu2023 import (
    format_arctan_wu2023_seed_report,
)
from hidden_attractors.validation.chua_arctan_wu2023 import (
    write_algebra_validation,
)

CONFIG_PATH = EXAMPLE_DIR / "reproducibility.yaml"


def load_config() -> dict[str, Any]:
    """Load the fixed bibliographic validation contract."""

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve(path_text: str | Path) -> Path:
    """Resolve a repository-relative validation path."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [EXAMPLE_DIR / path, VERSION2 / path, ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (EXAMPLE_DIR / path).resolve()


def write_json(path: Path, payload: Any) -> None:
    """Write one deterministic validation artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_published(
    cfg: dict[str, Any],
    *,
    run_trajectories: bool = False,
) -> None:
    """Rebuild the recorded Wu2023 algebra and seed checks."""

    lane = cfg["published_wu2023"]
    params = chua_parameters(
        model="arctan",
        **{
            key: lane["parameters"][key]
            for key in ("alpha", "beta", "gamma", "a1", "a2", "rho")
        },
    )
    outdir = resolve(lane["output_dir"])
    numerical = lane["numerical_contract"]
    seed_cfg = lane["seed_generation"]
    seed_report = format_arctan_wu2023_seed_report(
        q=float(numerical["q"]),
        params=params,
        nscan=20_000,
        transfer_mode=str(seed_cfg["transfer_mode"]),
    )
    seed_report["configuration"] = str(
        resolve(lane["source_config"]).relative_to(VERSION2)
    ).replace("\\", "/")
    seed_report["interpretation"] = lane["interpretation"]
    write_json(outdir / "02_lure_df" / "centered_seeds.json", seed_report)
    algebra = write_algebra_validation(
        outdir / "01_algebra" / "chua_arctan_wu2023_algebra.json"
    )
    write_json(
        outdir / "validation_summary.json",
        {
            "case_id": cfg["case_id"],
            "published_lane": lane,
            "seed_report": "02_lure_df/centered_seeds.json",
            "algebra_status": algebra["status"],
            "interpretation": (
                "published_bibliographic_reproduction_not_hiddenness_evidence"
            ),
        },
    )
    if run_trajectories:
        subprocess.run(
            [
                sys.executable,
                str(EXAMPLE_DIR / "run_validation.py"),
                "--run-trajectories",
            ],
            check=True,
        )
    print(f"published_output={outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-published-trajectories",
        action="store_true",
        help="Also execute the fixed reported-initial-condition checks.",
    )
    args = parser.parse_args()
    run_published(
        load_config(),
        run_trajectories=args.run_published_trajectories,
    )


if __name__ == "__main__":
    main()
