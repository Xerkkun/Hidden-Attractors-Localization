"""CLI commands for Lyapunov exponent analysis.

Stability: internal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence
import numpy as np

from ..workflows.lyapunov import run_lyapunov_workflow
from ..workflows.config_loader import load_config, apply_cli_overrides
from ..analysis.spectral import infer_step


def compute_lyapunov(argv: Sequence[str] | None = None) -> None:
    """Compute Lyapunov exponents of a system from a configuration or preset."""
    parser = argparse.ArgumentParser(description="Compute Lyapunov exponents workflow")
    parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("-p", "--preset", type=str, help="Select a built-in config preset")
    args, extra_args = parser.parse_known_args(argv)

    from .run import find_example_config, parse_dynamic_overrides
    
    if args.preset:
        from .run import PRESETS
        filename = PRESETS.get(args.preset)
        if not filename:
            print(f"Error: Preset '{args.preset}' not recognized. Available: {list(PRESETS.keys())}")
            sys.exit(1)
        config_path = find_example_config(filename)
    elif args.config:
        config_path = Path(args.config)
    else:
        print("Error: Must provide --config (-c) or --preset (-p).")
        sys.exit(1)

    try:
        config = load_config(config_path)
        overrides = parse_dynamic_overrides(extra_args)
        if overrides:
            config = apply_cli_overrides(config, overrides)
        
        # Run workflow
        run_lyapunov_workflow(config)
    except Exception as e:
        print(f"Lyapunov computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def trajectory_lyapunov_spectrum(argv: Sequence[str] | None = None) -> None:
    """Estimate trajectory-based Lyapunov exponent spectrum using time-series analysis."""
    parser = argparse.ArgumentParser(description="Estimate trajectory-based Lyapunov exponent")
    parser.add_argument("-t", "--trajectory", type=str, required=True, help="Path to trajectory CSV")
    parser.add_argument("--observable", default="x", help="State coordinate to use for estimation")
    args = parser.parse_args(argv)

    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        print(f"Error: Trajectory file '{trajectory_path}' does not exist.")
        sys.exit(1)

    # Load CSV
    times = []
    signal = []
    
    try:
        with open(trajectory_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["t"]))
                signal.append(float(row[args.observable]))
    except Exception as e:
        print(f"Error reading trajectory file: {e}")
        sys.exit(1)

    if len(signal) < 100:
        print("Error: Signal is too short for estimation. Need at least 100 points.")
        sys.exit(1)

    t_arr = np.array(times)
    sig_arr = np.array(signal)
    
    # Try using nolds
    try:
        from ..integrations.external_tools import compute_complexity_measures
        h = infer_step(t_arr, fallback=0.01)
        measures = compute_complexity_measures(sig_arr, backend="nolds", sample_rate=1.0/h, measures=["lyapunov_rosenstein"])
        val = measures.get("lyapunov_rosenstein")
        
        print("\n" + "="*80)
        print(" TRAJECTORY LYAPUNOV SPECTRUM ESTIMATION (nolds/Rosenstein) ")
        print("="*80)
        print(f"| Trajectory File  | {trajectory_path.name:<48} |")
        print(f"| Observable       | {args.observable:<48} |")
        print(f"| Signal Length    | {len(signal):<48} |")
        print(f"| Estimated LLE    | {val:<48.6f} |")
        print(f"| Chaos Indicator  | {'positive_largest_exponent' if val > 0.0 else 'nonpositive':<48} |")
        print(f"| Warning          | {'finite_time_lyapunov_estimate (supporting diagnostic only)'} |")
        print("="*80 + "\n")
        
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Lyapunov trajectory estimation failed: {e}")
        sys.exit(1)


def validate_lyapunov(argv: Sequence[str] | None = None) -> None:
    """Validate Lyapunov results JSON summary against mathematical criteria."""
    parser = argparse.ArgumentParser(description="Validate Lyapunov summary JSON")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to lyapunov_summary.json")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Summary file '{input_path}' does not exist.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Validation criteria:
    # 1. Must have analysis_type == "lyapunov"
    # 2. Status must becompleted
    # 3. Must warn if q < 1.0 that it's finite time estimate
    # 4. Must check check compatibility of fractional order and method.
    print(f"Validating Lyapunov summary: {input_path}")
    
    errors = []
    if summary.get("analysis_type") != "lyapunov":
        errors.append("Invalid analysis_type. Expected 'lyapunov'.")
    if summary.get("status") != "completed":
        errors.append(f"Calculation status is not completed: {summary.get('status')}")
    
    q = summary.get("q")
    method = summary.get("method")
    if q is not None and method is not None:
        if q < 1.0 and method == "integer_qr_benettin":
            errors.append(f"Method {method} is incompatible with fractional order q={q}")
            
    warnings = summary.get("warnings", [])
    if q is not None and q < 1.0:
        has_finite_time_warn = any("finite_time" in w.lower() or "estimate" in w.lower() for w in warnings)
        if not has_finite_time_warn:
            errors.append("Missing warning regarding finite-time estimate for fractional system.")

    if errors:
        print("Validation: FAILED")
        for err in errors:
            print(f"  - ERROR: {err}")
        sys.exit(1)
    else:
        print("Validation: PASSED")
        sys.exit(0)
