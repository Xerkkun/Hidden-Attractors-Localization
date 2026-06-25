#!/usr/bin/env python3
"""Run finite equilibrium-neighborhood checks for the nonsmooth Chua example."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="also run the long extended hiddenness matrix")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    steps = ["3", "4"] if args.extended else ["3"]
    subprocess.run([sys.executable, str(here / "run_example.py"), "--steps", *steps], check=True)
