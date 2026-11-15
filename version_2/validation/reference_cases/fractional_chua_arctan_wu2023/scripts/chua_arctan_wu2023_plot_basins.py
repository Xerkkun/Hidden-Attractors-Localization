#!/usr/bin/env python3
"""Run traceable Wu2023 finite-destination slices with the declared method.

The default contract dispatches the actual local ADM implementation. Each
initial value problem, map matrix, plot and effective contract is retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hidden_attractors.validation.wu2023_basin_maps import run_basin_maps

CONFIG = ROOT / "validation" / "reference_cases" / "fractional_chua_arctan_wu2023" / "config.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Explicit JSON numerical and optional basin_map contract.")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output-dir", type=Path, help="New or empty run directory; existing artifacts are never overwritten.")
    outputs.add_argument("--output", type=Path, help="Compatibility naming alias: file stem becomes a new run directory containing map_xy.png (and requested other planes).")
    parser.add_argument("--planes", nargs="+", choices=["xy", "xz", "yz"], help="Override basin_map.planes; fixed coordinate is explicit in the contract.")
    parser.add_argument("--points", type=int, help="Points on each slice axis.")
    parser.add_argument("--limit", type=float, help="Axis interval [-limit, limit].")
    parser.add_argument("--fixed-coordinate", type=float)
    parser.add_argument("--t-burn", type=float)
    parser.add_argument("--t-final", type=float, help="Use a separate numerical contract to change T; this compatibility flag only checks equality.")
    args = parser.parse_args(argv)
    output = args.output_dir or (args.output.with_suffix("") if args.output else ROOT / "outputs" / "examples" / "chua_arctan_wu2023_finite_maps")
    overrides = {name: value for name, value in {
        "planes": args.planes, "points": args.points, "limit": args.limit,
        "fixed_coordinate": args.fixed_coordinate, "t_burn": args.t_burn,
    }.items() if value is not None}
    try:
        if args.t_final is not None:
            declared = json.loads(args.config.read_text(encoding="utf-8"))["numerical_contract"]["t_final"]
            if args.t_final != declared:
                raise ValueError("--t-final must equal the numerical contract; change h/N/T together in a separate --config file.")
        manifest = run_basin_maps(args.config, output, overrides=overrides)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": str(output.resolve()), "method": manifest["method"],
                      "points_total": manifest["points_total"], "class_counts": manifest["class_counts"],
                      "hidden_verified": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
