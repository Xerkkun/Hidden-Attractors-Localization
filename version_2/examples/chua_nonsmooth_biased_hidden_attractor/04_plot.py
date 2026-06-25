#!/usr/bin/env python3
"""Generate the summary gallery for the nonsmooth Chua report example."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "run_example.py"), "--steps", "5"], check=True)
