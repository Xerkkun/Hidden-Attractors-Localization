#!/usr/bin/env python3
"""Regenerate summary/plots for an existing robustness-overlay output folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.workflows.robustness_overlay import aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Folder containing robustness_overlay_config.json and metrics_*.csv files.")
    args = parser.parse_args()
    summary = aggregate(Path(args.output_dir), wait=False)
    print(summary)


if __name__ == "__main__":
    main()
