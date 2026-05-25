#!/usr/bin/env python3
"""Compatibility CLI for Danca ABM full-history equilibrium-ball controls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.workflows.danca_abm_sphere_controls import main


if __name__ == "__main__":
    main()
