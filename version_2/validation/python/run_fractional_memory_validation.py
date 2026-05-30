"""Runner for fractional memory validation.

Usage
-----
Run all cases (full simulation):
    python validation/python/run_fractional_memory_validation.py --all

Run all cases in fast mode:
    python validation/python/run_fractional_memory_validation.py --all --fast

Run a single case:
    python validation/python/run_fractional_memory_validation.py \\
        --case validation/fractional_memory_validation/chua_fractional_saturation_memory.yaml

Run with trajectory saving:
    python validation/python/run_fractional_memory_validation.py --all --save-trajectories

Custom output directory:
    python validation/python/run_fractional_memory_validation.py --all \\
        --output-dir validation/outputs/fractional_memory_validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root on sys.path so that both validation.python and
# hidden_attractors packages are importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.python.fractional_memory_validation import (
    run_all_fractional_memory_validations,
    run_fractional_memory_validation,
)

_DEFAULT_CONFIG_DIR = (
    _REPO_ROOT / "validation" / "fractional_memory_validation"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT / "validation" / "outputs" / "fractional_memory_validation"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_fractional_memory_validation",
        description=(
            "Fractional memory validation for Caputo-based Chua systems.\n\n"
            "Compares full Caputo memory to finite-window approximations and "
            "reports sensitivity metrics. Does NOT certify hidden attractors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Run all *_memory.yaml configs in the config directory.",
    )
    mode_group.add_argument(
        "--case",
        metavar="PATH",
        type=str,
        help="Path to a single YAML config file to run.",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help=(
            f"Root output directory (default: {_DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--config-dir",
        metavar="DIR",
        type=str,
        default=str(_DEFAULT_CONFIG_DIR),
        help=(
            f"Directory containing *_memory.yaml configs "
            f"(default: {_DEFAULT_CONFIG_DIR}). Only used with --all."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use fast_test time parameters (short t_final) for smoke testing. "
            "Results will be less representative but run quickly."
        ),
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save trajectory arrays (.npy) under <output_dir>/<case_id>/trajectories/.",
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fast = args.fast
    save_traj = args.save_trajectories

    if fast:
        print("[run_fractional_memory_validation] Fast mode: using fast_test parameters.")

    print(f"[run_fractional_memory_validation] Output directory: {output_dir}")
    print("[run_fractional_memory_validation] pointwise_comparison_used: false")
    print("[run_fractional_memory_validation] hiddenness_certified_by_this_pipeline: false")
    print()

    exit_code = 0

    if args.all:
        config_dir = Path(args.config_dir)
        print(f"[run_fractional_memory_validation] Config directory: {config_dir}")
        print()
        try:
            summaries = run_all_fractional_memory_validations(
                config_dir=config_dir,
                output_dir=output_dir,
                fast=fast,
                save_trajectories=save_traj,
            )
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        # Print summary table
        print()
        print("=" * 70)
        print(" FRACTIONAL MEMORY VALIDATION - ALL CASES SUMMARY")
        print("=" * 70)
        for s in summaries:
            cid = s.get("case_id", s.get("yaml_file", "?"))
            ost = s.get("overall_status", "?")
            warns = s.get("automatic_warnings", [])
            err = s.get("error", None)
            print(f"  case_id : {cid}")
            print(f"  status  : {ost}")
            if err:
                print(f"  error   : {err}")
            if warns:
                for w in warns:
                    print(f"  WARNING : {w}")
            print()
            if ost in ("memory_validation_failed", "memory_validation_inconclusive"):
                exit_code = 1

        print("=" * 70)
        print(f"  hiddenness_certified_by_this_pipeline : false")
        print(f"  no_hidden_verified_claim              : true")
        print(f"  pointwise_comparison_used             : false")
        print("=" * 70)
        print()

    else:
        # Single case
        case_path = Path(args.case)
        print(f"[run_fractional_memory_validation] Case: {case_path}")
        print()
        try:
            summary = run_fractional_memory_validation(
                config_path=case_path,
                output_dir=output_dir,
                fast=fast,
                save_trajectories=save_traj,
            )
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        print("=" * 70)
        print(" FRACTIONAL MEMORY VALIDATION — SINGLE CASE SUMMARY")
        print("=" * 70)
        print(json.dumps(summary, indent=2))
        print("=" * 70)

        ost = summary.get("overall_status", "")
        if ost in ("memory_validation_failed", "memory_validation_inconclusive"):
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
