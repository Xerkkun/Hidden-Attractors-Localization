#!/usr/bin/env python3
"""run_published_reproduction.py

Runner script for the published_case_reproduction validation layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root and validation directory to path
here = Path(__file__).resolve().parent
repo_root = here.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

from published_reproduction import run_case_reproduction, run_all_published_cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproduction checks for published chaotic systems cases."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", help="Run all registered cases."
    )
    group.add_argument(
        "--case", type=str, help="Path to a single case YAML configuration file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation/outputs/published_cases",
        help="Directory to save the output reports.",
    )
    parser.add_argument(
        "--run-dynamics",
        action="store_true",
        help="Run dynamic simulations (Caputo/integer trajectories). Skipping them is default.",
    )

    args = parser.parse_args()
    
    # Target output path
    out_dir = repo_root / args.output_dir

    try:
        if args.all:
            print("Running reproduction for all registered cases...")
            results = run_all_published_cases(out_dir, run_dynamics=args.run_dynamics)
            for case_id, res in results.items():
                print(f"  - Case {case_id}: {res['statuses']}")
            print("All cases executed successfully.")
        else:
            case_path = Path(args.case)
            if not case_path.is_absolute():
                case_path = repo_root / case_path
            
            if not case_path.exists():
                print(f"Error: Case file not found at {case_path}")
                return 1
            
            print(f"Running reproduction for case: {case_path.name}...")
            res = run_case_reproduction(case_path, out_dir, run_dynamics=args.run_dynamics)
            print(f"Case {res['case_id']} finished with statuses: {res['statuses']}")
            
        return 0
    except Exception as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
