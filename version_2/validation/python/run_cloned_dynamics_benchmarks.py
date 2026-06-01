#!/usr/bin/env python3
"""Run Fischer 2020 cloned-dynamics benchmark rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloned_dynamics_benchmarks import load_benchmark_case, run_benchmark_row, write_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run reduced smoke integrations.")
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=PROJECT_ROOT / "validation" / "lyapunov_benchmarks" / "fractional_cloned_dynamics_abm_gs_published",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "validation" / "outputs" / "lyapunov_benchmarks" / "fractional_cloned_dynamics_abm_gs_published",
    )
    parser.add_argument(
        "--official-summary-dir",
        type=Path,
        help="Also write the conservative tracked summary to this directory.",
    )
    args = parser.parse_args()
    rows = []
    executions = []
    global_parameters = []
    for path in sorted(args.benchmarks_dir.glob("*.yaml")):
        case = load_benchmark_case(path)
        global_parameters.append(
            {
                "case_file": path.name,
                "system": case["system"]["kind"],
                "integration": case["integration"],
            }
        )
        for row_index, row in enumerate(case["expected"]["rows"]):
            record, result = run_benchmark_row(
                case,
                row,
                fast=args.fast,
                case_file=path.name,
                row_index=row_index,
            )
            rows.append(record)
            executions.append((record, result))
    summary = write_results(
        args.output_dir,
        rows,
        executions=executions,
        fast=args.fast,
        command=" ".join(sys.argv),
        global_parameters=global_parameters,
        official_summary_dir=args.official_summary_dir,
    )
    print(f"Recorded {summary['rows_total']} rows with status: {summary['status']}")


if __name__ == "__main__":
    main()
