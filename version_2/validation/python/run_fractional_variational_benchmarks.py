#!/usr/bin/env python3
"""Runner script for fractional variational ABM-QR Lyapunov benchmarks (F2.1)."""

import os
import sys
import json
import argparse
import glob
from typing import List, Dict, Any

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from validation.python.fractional_variational_benchmarks import (
    load_benchmark_case,
    run_benchmark_case
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run fractional variational ABM-QR benchmarks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all benchmarks in the default directory.")
    group.add_argument("--case", type=str, help="Path to a specific benchmark YAML file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("validation", "outputs", "lyapunov_benchmarks", "fractional_variational_abm_qr"),
        help="Directory to write output files."
    )
    parser.add_argument("--fast", action="store_true", help="Run in fast mode for unit testing (reduced integration time).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Resolve list of files to process
    yaml_files: List[str] = []
    default_benchmarks_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lyapunov_benchmarks", "fractional_variational_abm_qr")
    )

    if args.all:
        pattern = os.path.join(default_benchmarks_dir, "*.yaml")
        yaml_files = sorted(glob.glob(pattern))
    elif args.case:
        yaml_files = [os.path.abspath(args.case)]

    if not yaml_files:
        print(f"No benchmark YAML files found.")
        sys.exit(1)

    print(f"Found {len(yaml_files)} benchmark case(s) to process.")

    # 2. Run cases
    results: List[Dict[str, Any]] = []
    missing_data_reports: List[Dict[str, Any]] = []

    for path in yaml_files:
        print(f"Loading case: {os.path.basename(path)}...", end="", flush=True)
        try:
            case_data = load_benchmark_case(path)
            print(" Done. Running...", end="", flush=True)
            res = run_benchmark_case(case_data, fast=args.fast, output_dir=args.output_dir)
            print(f" Done. Status: {res['status']}")
            results.append(res)
            
            if res["status"] == "published_reference_data_missing":
                missing_data_reports.append({
                    "case_id": res["case_id"],
                    "missing_fields": res.get("missing_fields", [])
                })
        except Exception as exc:
            print(f" Failed with exception: {exc}")
            results.append({
                "case_id": os.path.splitext(os.path.basename(path))[0],
                "benchmark_type": "unknown",
                "status": "synthetic_benchmark_failed",
                "computed_exponents": None,
                "message": f"Exception raised during load/execution: {exc}"
            })

    # 3. Create outputs
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # A. benchmark_cases.csv
    import csv
    csv_path = os.path.join(out_dir, "benchmark_cases.csv")
    if results:
        headers = list(results[0].keys())
        processed_results = []
        for r in results:
            item = r.copy()
            if "computed_exponents" in item and item["computed_exponents"] is not None:
                item["computed_exponents"] = str(item["computed_exponents"])
            else:
                item["computed_exponents"] = ""
            processed_results.append(item)
            
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(processed_results)
    print(f"Saved case details to {csv_path}")


    # B. missing_reference_data.json
    missing_path = os.path.join(out_dir, "missing_reference_data.json")
    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(missing_data_reports, f, indent=2)
    print(f"Saved missing data report to {missing_path}")

    # C. Determine overall status and build summary
    # Status lists
    all_statuses = [r["status"] for r in results]
    
    any_failed = any("failed" in s for s in all_statuses)
    any_missing = any(s == "published_reference_data_missing" for s in all_statuses)
    
    synthetic_results = [r for r in results if r["benchmark_type"] == "synthetic"]
    synthetic_passed = all(r["status"] == "synthetic_benchmark_passed" for r in synthetic_results) if synthetic_results else True
    
    published_results = [r for r in results if r["benchmark_type"] == "published"]
    published_passed = all(
        r["status"] in ("published_benchmark_passed_quantitative", "published_benchmark_passed_qualitative")
        for r in published_results
    ) if published_results else True

    if any_failed:
        global_status = "fractional_variational_abm_qr_validation_failed"
    elif any_missing and synthetic_passed:
        global_status = "fractional_variational_abm_qr_published_pending"
    elif synthetic_passed and published_passed and published_results:
        global_status = "fractional_variational_abm_qr_published_validated"
    elif synthetic_passed and not published_results:
        global_status = "fractional_variational_abm_qr_synthetic_validated"
    else:
        global_status = "fractional_variational_abm_qr_validation_inconclusive"

    summary = {
        "global_status": global_status,
        "method_id": "fractional_variational_abm_qr",
        "cases_run": len(results),
        "synthetic_cases_passed": sum(1 for r in results if r["status"] == "synthetic_benchmark_passed"),
        "published_cases_passed_quantitative": sum(1 for r in results if r["status"] == "published_benchmark_passed_quantitative"),
        "published_cases_passed_qualitative": sum(1 for r in results if r["status"] == "published_benchmark_passed_qualitative"),
        "published_cases_pending_missing_data": sum(1 for r in results if r["status"] == "published_reference_data_missing"),
        "failures": sum(1 for r in results if "failed" in r["status"]),
        "inconclusive": sum(1 for r in results if r["status"] == "benchmark_inconclusive")
    }

    summary_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark summary to {summary_path}")

    # Explicitly print summary
    print("\nBenchmark Execution Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
