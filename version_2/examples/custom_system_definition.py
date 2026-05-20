#!/usr/bin/env python3
"""Register and inspect a user-defined chaotic system."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.systems import ChaoticSystem, get_system, register_system


def lorenz_rhs(state: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    x, y, z = state
    sigma = float(p["sigma"])
    rho = float(p["rho"])
    beta = float(p["beta"])
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=float)


def lorenz_equilibria(p: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rho = float(p["rho"])
    beta = float(p["beta"])
    if rho <= 1.0:
        return {"E0": np.zeros(3, dtype=float)}
    a = float(np.sqrt(beta * (rho - 1.0)))
    return {
        "E0": np.zeros(3, dtype=float),
        "E+": np.array([a, a, rho - 1.0], dtype=float),
        "E-": np.array([-a, -a, rho - 1.0], dtype=float),
    }


def main() -> None:
    register_system(
        ChaoticSystem(
            name="lorenz63",
            dimension=3,
            rhs=lorenz_rhs,
            equilibria=lorenz_equilibria,
            parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
            description="Classic Lorenz 63 system registered by a user script.",
            tags=("lorenz", "integer-order", "example"),
        ),
        replace=True,
    )
    system = get_system("lorenz63")
    print(f"name={system.name}")
    print(f"rhs={system.evaluate([1.0, 1.0, 1.0]).tolist()}")
    for name, point in system.equilibrium_points().items():
        print(f"{name}={point.tolist()}")


if __name__ == "__main__":
    main()
