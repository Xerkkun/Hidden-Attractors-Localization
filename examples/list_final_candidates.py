#!/usr/bin/env python3
"""List the final candidates loaded through the package API."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors import load_final_candidate_records


def main() -> None:
    for record in load_final_candidate_records():
        print(
            f"{record.candidate_id} | route={record.route} | "
            f"q={record.q:.4f} | start={record.robust_start.tolist()} | "
            f"seed={record.seed.tolist()}"
        )


if __name__ == "__main__":
    main()
