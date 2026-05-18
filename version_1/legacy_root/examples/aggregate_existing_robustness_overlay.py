#!/usr/bin/env python3
"""Compatibility wrapper for version_2/examples/aggregate_existing_robustness_overlay.py."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runpy.run_path(
        str(ROOT / "version_2" / "examples" / "aggregate_existing_robustness_overlay.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
