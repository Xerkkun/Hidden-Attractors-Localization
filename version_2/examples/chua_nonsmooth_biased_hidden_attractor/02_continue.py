#!/usr/bin/env python3
"""Run the continuation-producing biased DF stage for the nonsmooth Chua example."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "run_example.py"), "--steps", "2"], check=True)
