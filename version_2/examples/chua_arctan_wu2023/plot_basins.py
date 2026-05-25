#!/usr/bin/env python3
"""Render an exploratory Wu2023 basin slice; this is not hiddenness verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hidden_attractors.models import chua_arctan_wu2023_parameters
from hidden_attractors.native.backends import BasinBackend


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "chua_arctan_wu2023_caputo.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "examples" / "chua_arctan_wu2023_basin_xy.png")
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument("--limit", type=float, default=1.2)
    parser.add_argument("--t-final", type=float, default=100.0)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    numerical = config["numerical_contract"]
    backend = BasinBackend.build(output_name="chua_arctan_wu2023_basin")
    backend.set_arctan_params(chua_arctan_wu2023_parameters())
    axis = np.linspace(-args.limit, args.limit, args.points)
    labels = np.zeros((args.points, args.points), dtype=int)
    for iy, y0 in enumerate(axis):
        for ix, x0 in enumerate(axis):
            labels[iy, ix] = backend.classify_point(
                [x0, y0, 0.0],
                q=float(numerical["q"]),
                h=float(numerical["h"]),
                Lm=float(numerical["memory_length"]),
                t_final=args.t_final,
                t_burn=args.t_final / 2.0,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(labels, origin="lower", extent=[-args.limit, args.limit, -args.limit, args.limit], interpolation="nearest")
    ax.set_xlabel("x(0)")
    ax.set_ylabel("y(0)")
    ax.set_title("Wu2023 arctan Chua: exploratory xy basin slice (z0=0)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"figure={args.output}")
    print("hidden_verified=false; neighborhood tests around E0, E+ and E- remain required")


if __name__ == "__main__":
    main()
