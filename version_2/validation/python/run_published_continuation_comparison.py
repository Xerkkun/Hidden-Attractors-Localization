"""Runner for published continuation comparison (Phase E).

Usage
-----
Run all cases (full simulation):
    python validation/python/run_published_continuation_comparison.py --all

Run all cases in fast mode:
    python validation/python/run_published_continuation_comparison.py --all --fast

Run a single case:
    python validation/python/run_published_continuation_comparison.py \\
        --case validation/published_continuation_comparison/wu2023_chua_fractional_arctan.yaml

Run with trajectory saving:
    python validation/python/run_published_continuation_comparison.py --all --save-trajectories

Custom output directory:
    python validation/python/run_published_continuation_comparison.py --all \\
        --output-dir validation/outputs/published_continuation_comparison
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

from validation.python.published_continuation_comparison import (
    run_published_continuation_case,
    run_all_published_continuation_comparisons,
)

_DEFAULT_CONFIG_DIR = (
    _REPO_ROOT / "validation" / "published_continuation_comparison"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT / "validation" / "outputs" / "published_continuation_comparison"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_published_continuation_comparison",
        description=(
            "Phase E: Published Continuation Comparison.\n\n"
            "Compares paper-style integration strategy (inferred from published "
            "articles) against the Caputo-aware history_window_transport strategy "
            "available in the library.\n\n"
            "Does NOT certify hidden attractors or chaos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Run all *.yaml configs in the config directory.",
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
            f"Directory containing *.yaml configs "
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
        print("[run_published_continuation_comparison] Fast mode: using fast_test parameters.")

    print(f"[run_published_continuation_comparison] Output directory: {output_dir}")
    print("[run_published_continuation_comparison] hiddenness_certified_by_this_pipeline: false")
    print("[run_published_continuation_comparison] chaos_certified_by_this_pipeline: false")
    print("[run_published_continuation_comparison] no_hidden_verified_claim: true")
    print()

    exit_code = 0

    if args.all:
        config_dir = Path(args.config_dir)
        print(f"[run_published_continuation_comparison] Config directory: {config_dir}")
        print()
        try:
            summaries = run_all_published_continuation_comparisons(
                config_dir=config_dir,
                output_dir=output_dir,
                fast=fast,
                save_trajectories=save_traj,
            )
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        print()
        print("=" * 70)
        print(" PUBLISHED CONTINUATION COMPARISON - ALL CASES SUMMARY")
        print("=" * 70)
        for s in summaries:
            cid = s.get("case_id", s.get("yaml_file", "?"))
            ost = s.get("overall_status", "?")
            err = s.get("error", None)
            print(f"  case_id        : {cid}")
            print(f"  overall_status : {ost}")
            print(f"  paper_reports_continuation  : {s.get('paper_reports_continuation', '?')}")
            print(f"  paper_style_strategy        : {s.get('paper_style_strategy', '?')}")
            print(f"  paper_style_reintegration   : {s.get('paper_style_reintegration_status', '?')}")
            print(f"  paper_style_vs_history      : {s.get('paper_style_vs_history_status', '?')}")
            print(f"  dynamic_class_detected      : {s.get('dynamic_class_detected', '?')}")
            print(f"  chaotic_candidate_detected  : {s.get('chaotic_dynamics_candidate_detected', '?')}")
            if err:
                print(f"  error          : {err}")
                exit_code = 1
            print()

        print("=" * 70)
        print("  hiddenness_certified_by_this_pipeline : false")
        print("  chaos_certified_by_this_pipeline      : false")
        print("  no_hidden_verified_claim              : true")
        print("  pointwise_comparison_used             : false")
        print("=" * 70)
        print()

    else:
        # Single case
        case_path = Path(args.case)
        print(f"[run_published_continuation_comparison] Case: {case_path}")
        print()
        try:
            summary = run_published_continuation_case(
                config_path=case_path,
                output_dir=output_dir,
                fast=fast,
                save_trajectories=save_traj,
            )
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        print("=" * 70)
        print(" PUBLISHED CONTINUATION COMPARISON — SINGLE CASE SUMMARY")
        print("=" * 70)
        print(json.dumps(summary, indent=2))
        print("=" * 70)

        ost = summary.get("overall_status", "")
        if ost == "published_comparison_inconclusive" and summary.get("error"):
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
