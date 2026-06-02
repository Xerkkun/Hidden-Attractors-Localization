#!/usr/bin/env python3
"""CLI for conservative F4 internal Lyapunov validation assembly."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from f4_internal_lyapunov_validation import (  # noqa: E402
    DK2018_SUMMARY,
    FISCHER2020_SUMMARY,
    run_selected_stages,
)


ALL_STAGES = ["F4_1", "F4_2", "F4_3", "F4_4"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all F4 assembly stages.")
    parser.add_argument("--stage", action="append", choices=ALL_STAGES, help="Run one F4 stage; repeat as needed.")
    parser.add_argument("--fast", action="store_true", help="Use short internal controls only.")
    parser.add_argument("--use-existing", action="store_true", help="Reuse existing DK2018 and Fischer published artifacts.")
    args = parser.parse_args()

    stages = ALL_STAGES if args.all or not args.stage else args.stage
    if not args.fast and os.environ.get("RUN_F4_LONG") != "1":
        raise RuntimeError("Non-fast F4 assembly requires RUN_F4_LONG=1.")
    required_existing_artifacts = []
    if "F4_3" in stages:
        required_existing_artifacts.append(DK2018_SUMMARY)
    if "F4_4" in stages:
        required_existing_artifacts.append(FISCHER2020_SUMMARY)
    use_existing = args.use_existing or all(path.exists() for path in required_existing_artifacts)
    if "F4_3" in stages and not use_existing and os.environ.get("RUN_PUBLISHED_LYAPUNOV") != "1":
        raise RuntimeError("F4.3 requires --use-existing or RUN_PUBLISHED_LYAPUNOV=1.")
    if "F4_4" in stages and not use_existing and os.environ.get("RUN_PUBLISHED_CLONED") != "1":
        raise RuntimeError("F4.4 requires --use-existing or RUN_PUBLISHED_CLONED=1.")
    if any(stage in stages for stage in ("F4_3", "F4_4")) and not use_existing:
        raise RuntimeError(
            "Run the dedicated opt-in published runner first, then assemble F4 with --use-existing."
        )

    summary = run_selected_stages(stages, fast=args.fast, use_existing=use_existing)
    print(f"F4 status: {summary['status']}")
    print("Validation promotion: false")


if __name__ == "__main__":
    main()
