"""Runner script for the integrator crosscheck validation phase.

Usage
-----
Run all cases (fast mode, suitable for CI):

    python validation/python/run_integrator_crosscheck.py --all --fast

Run a specific case:

    python validation/python/run_integrator_crosscheck.py \\
        --case validation/integrator_crosscheck/chua_fractional_saturation.yaml \\
        --fast

Full run saving trajectories and figures:

    python validation/python/run_integrator_crosscheck.py \\
        --all --save-trajectories --make-figures

Options
-------
--all                Run all YAML crosscheck configurations found in the
                     validation/integrator_crosscheck/ directory.
--case PATH          Run a single YAML crosscheck configuration.
--output-dir PATH    Root output directory (default: validation/outputs/integrator_crosscheck).
--fast               Use fast_test settings from each YAML (short t_final).
--save-trajectories  Save trajectory arrays as .npy files under trajectories/.
--make-figures       Generate phase-space figures (requires matplotlib).
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
        description="Run integrator crosscheck validation cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all YAML configurations in validation/integrator_crosscheck/.",
    )
    group.add_argument(
        "--case",
        metavar="PATH",
        help="Path to a single YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default="validation/outputs/integrator_crosscheck",
        help="Root output directory (default: validation/outputs/integrator_crosscheck).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast_test settings from each YAML (short t_final).",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        dest="save_trajectories",
        help="Save trajectory arrays as .npy files under trajectories/.",
    )
    parser.add_argument(
        "--make-figures",
        action="store_true",
        dest="make_figures",
        help="Generate phase-space figures (requires matplotlib).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _add_repo_to_path()

    args = parse_args(argv)
    root = _repo_root()

    # Resolve output directory relative to repo root if not absolute
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    from validation.python.integrator_crosscheck import (
        run_integrator_crosscheck_case,
        run_all_integrator_crosschecks,
    )

    results: dict = {}

    if args.run_all:
        crosscheck_dir = root / "validation" / "integrator_crosscheck"
        print(f"Running all crosscheck cases in {crosscheck_dir} ...")
        results = run_all_integrator_crosschecks(
            crosscheck_dir=crosscheck_dir,
            output_dir=output_dir,
            fast=args.fast,
            save_trajectories=args.save_trajectories,
            make_figures=args.make_figures,
        )
    elif args.case:
        case_path = Path(args.case)
        if not case_path.is_absolute():
            case_path = root / case_path
        print(f"Running crosscheck case: {case_path.name} ...")
        summary = run_integrator_crosscheck_case(
            config_path=case_path,
            output_dir=output_dir,
            fast=args.fast,
            save_trajectories=args.save_trajectories,
            make_figures=args.make_figures,
        )
        results[summary["case_id"]] = summary
    else:
        # Default: run all
        crosscheck_dir = root / "validation" / "integrator_crosscheck"
        print(f"No --all or --case specified; running all in {crosscheck_dir} ...")
        results = run_all_integrator_crosschecks(
            crosscheck_dir=crosscheck_dir,
            output_dir=output_dir,
            fast=args.fast,
            save_trajectories=args.save_trajectories,
            make_figures=args.make_figures,
        )

    # Print summary
    print()
    print("=" * 70)
    print("Integrator Crosscheck Summary")
    print("=" * 70)
    all_passed = True
    for case_id, summary in results.items():
        status = summary.get("overall_status", "unknown")
        h_sens = summary.get("h_sensitivity", "?")
        mem_sens = summary.get("memory_sensitivity", "?")
        certified = summary.get("hiddenness_certified_by_this_pipeline", "?")
        no_claim = summary.get("no_hidden_verified_claim", "?")
        print(
            f"  [{status}] {case_id}\n"
            f"    h_sensitivity={h_sens}, memory_sensitivity={mem_sens}\n"
            f"    hiddenness_certified_by_this_pipeline={certified}, "
            f"no_hidden_verified_claim={no_claim}"
        )
        if status in ("crosscheck_failed",):
            all_passed = False
    print("=" * 70)
    print(f"Outputs written to: {output_dir}")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
