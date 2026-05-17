#!/usr/bin/env python3
"""Create a robustness-overlay configuration without launching long jobs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.io import timestamp
from hidden_attractors.paths import OUTPUTS
from hidden_attractors.workflows.robustness_overlay import make_config


def main() -> None:
    outdir = OUTPUTS / f"example_robustness_overlay_config_{timestamp()}"
    cfg = make_config(outdir)
    print(f"config={outdir / 'robustness_overlay_config.json'}")
    print(f"candidates={len(cfg['candidates'])} cases={len(cfg['cases'])}")


if __name__ == "__main__":
    main()
