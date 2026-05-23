#!/usr/bin/env python3
"""Compute Chua equilibria and verify they are vector-field zeros."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors import chua_nonsmooth_parameters
from hidden_attractors.models import equilibria_nonsmooth, rhs_nonsmooth


def main() -> None:
    params = chua_nonsmooth_parameters()
    for name, point in equilibria_nonsmooth(params).items():
        residual_norm = float(np.linalg.norm(rhs_nonsmooth(point, params)))
        print(f"{name}: point={point.tolist()} residual_norm={residual_norm:.3e}")


if __name__ == "__main__":
    main()
