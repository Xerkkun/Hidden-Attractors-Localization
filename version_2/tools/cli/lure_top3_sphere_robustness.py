#!/usr/bin/env python3
"""Compatibility CLI for top-candidate sphere controls and robustness.

The reusable implementation lives in
``hidden_attractors.workflows.sphere_controls``.  The CLI name and output
artifact names are preserved for reproducibility with previous runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.workflows.sphere_controls import main


if __name__ == "__main__":
    main()
