"""Runner script for the integrator method validation phase.

Usage
-----
Run all methods:

    python validation/python/run_integrator_method_validation.py --all

Run only ABM:

    python validation/python/run_integrator_method_validation.py --method ABM

Run only RK4:

    python validation/python/run_integrator_method_validation.py --method RK4

Options
-------
--all                Run all methods (ABM and RK4).
--method NAME        Run only the specified method (ABM | RK4).
--output-dir PATH    Output directory (default: validation/outputs/integrator_method_validation).
--fast               Use fewer h values for faster iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_repo_to_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integrator method validation (ABM Caputo and RK4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all methods (ABM and RK4).",
    )
    group.add_argument(
        "--method",
        metavar="NAME",
        choices=["ABM", "RK4"],
        help="Run only the specified method (ABM | RK4).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default="validation/outputs/integrator_method_validation",
        help=(
            "Root output directory "
            "(default: validation/outputs/integrator_method_validation)."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer h values for faster iteration.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _add_repo_to_path()
    args = parse_args(argv)

    root = _repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    from validation.python.integrator_method_validation import run_integrator_method_validation

    if args.method:
        methods = [args.method]
    else:
        methods = ["ABM", "RK4"]

    print(f"Running integrator method validation for: {methods}")
    summary = run_integrator_method_validation(
        output_dir=output_dir,
        methods=methods,
        fast=args.fast,
    )

    # Print summary
    print()
    print("=" * 70)
    print("Integrator Method Validation Summary")
    print("=" * 70)
    for method, info in summary.get("methods", {}).items():
        status = info.get("status", "unknown")
        tests = info.get("tests", {})
        print(f"  [{status}] {method}")
        for test_name, test_status in tests.items():
            print(f"    {test_name}: {test_status}")
    print("-" * 70)
    print(f"  hiddenness_certified_by_this_pipeline: {summary.get('hiddenness_certified_by_this_pipeline')}")
    print(f"  no_hidden_verified_claim: {summary.get('no_hidden_verified_claim')}")
    print("=" * 70)
    print(f"Outputs written to: {output_dir}")
    print()

    # Return 0 if all methods passed or are inconclusive (not failed)
    all_ok = all(
        info.get("status") not in ("method_validation_failed",)
        for info in summary.get("methods", {}).values()
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
