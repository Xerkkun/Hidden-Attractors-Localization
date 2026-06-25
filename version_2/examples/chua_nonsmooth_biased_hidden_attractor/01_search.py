#!/usr/bin/env python3
"""Run the centered and biased DF searches for the nonsmooth Chua report example."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "run_example.py"), "--steps", "1", "2"], check=True)
