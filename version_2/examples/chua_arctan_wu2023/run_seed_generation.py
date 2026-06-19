#!/usr/bin/env python3
"""Generate centered Wu2023 arctan Lur'e seeds without claiming hiddenness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hidden_attractors.seed_generation.chua_arctan_wu2023 import format_arctan_wu2023_seed_report


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "chua_arctan_wu2023_caputo.json"
DEFAULT_OUTPUT = (
    ROOT
    / "validation"
    / "reference_cases"
    / "fractional_chua_arctan_wu2023"
    / "02_lure_df"
    / "centered_seeds.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nscan", type=int, default=20000)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seed_cfg = config["seed_generation"]
    report = format_arctan_wu2023_seed_report(
        q=float(config["numerical_contract"]["q"]),
        nscan=args.nscan,
        transfer_mode=seed_cfg["default_transfer_mode"],
    )
    report["configuration"] = str(CONFIG.relative_to(ROOT)).replace("\\", "/")
    report["attractor_status"] = "candidate"
    report["interpretation"] = "seed_generation_only_not_hiddenness_evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"seed_report={args.output}")
    print(f"branches={len(report['branches'])}")


if __name__ == "__main__":
    main()
