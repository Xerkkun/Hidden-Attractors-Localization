#!/usr/bin/env python3
"""Compatibility CLI for the validation evidence contract checker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.validation_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
