"""CLI commands for 0-1 chaos testing and time-series diagnostics.

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

from ..diagnostics.zero_one import run_zero_one_diagnostic, run_zero_one_from_config
from ..workflows.config_loader import load_config, apply_cli_overrides


def run_zero_one(argv: Sequence[str] | None = None) -> None:
    """Run 0-1 chaos-test diagnostic on a trajectory or configuration."""
    parser = argparse.ArgumentParser(description="Run 0-1 chaos-test diagnostic")
    parser.add_argument("-t", "--trajectory", type=str, help="Path to trajectory CSV")
    parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--observable", default="x", help="Observable state coordinate (x, y, z)")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    args, extra_args = parser.parse_known_args(argv)

    if args.trajectory:
        trajectory_path = Path(args.trajectory)
        if not trajectory_path.exists():
            print(f"Error: Trajectory file '{trajectory_path}' does not exist.")
            sys.exit(1)
            
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
            
        t_arr = np.array(times)
        # We need the full states to reconstruct the coordinate selection
        # For simplicity, we create a 2D array of states where the observable index is aligned
        states_arr = np.zeros((len(signal), 3))
        coord_map = {"x": 0, "y": 1, "z": 2}
        idx = coord_map.get(args.observable.lower(), 0)
        states_arr[:, idx] = np.array(signal)
        
        # Parse extra args
        n_c = 100
        c_min = 0.1
        c_max = 3.04159
        seed = 12345
        
        # Simple override parser for extra args
        from .run import parse_dynamic_overrides
        overrides = parse_dynamic_overrides(extra_args)
        if "n_c_values" in overrides:
            n_c = int(overrides["n_c_values"])
        if "c_min" in overrides:
            c_min = float(overrides["c_min"])
        if "c_max" in overrides:
            c_max = float(overrides["c_max"])
        if "seed" in overrides:
            seed = int(overrides["seed"])
            
        output_dir = Path(args.output_dir or "outputs")
        run_zero_one_diagnostic(
            t_arr,
            states_arr,
            args.observable,
            output_dir,
            n_c=n_c,
            c_min=c_min,
            c_max=c_max,
            seed=seed,
            system_id="unknown_trajectory",
        )
        
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file '{config_path}' does not exist.")
            sys.exit(1)
            
        try:
            config = load_config(config_path)
            from .run import parse_dynamic_overrides
            overrides = parse_dynamic_overrides(extra_args)
            
            # CLI override for observable takes precedence
            if args.observable:
                if "zero_one" not in config:
                    config["zero_one"] = {}
                config["zero_one"]["observable"] = args.observable
                
            if args.output_dir:
                config["output_dir"] = args.output_dir
                
            if overrides:
                config = apply_cli_overrides(config, overrides)
                
            run_zero_one_from_config(config)
        except Exception as e:
            print(f"0-1 test configuration run failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Error: Must provide either --trajectory (-t) or --config (-c).")
        sys.exit(1)


def inspect_zero_one(argv: Sequence[str] | None = None) -> None:
    """Inspect zero_one summary JSON."""
    parser = argparse.ArgumentParser(description="Inspect 0-1 test summary JSON")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to zero_one_summary.json")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Summary file '{input_path}' does not exist.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\n" + "="*80)
    print(f" 0-1 CHAOS TEST DIAGNOSTIC SUMMARY: {input_path.name} ")
    print("="*80)
    print(f"| Observable       | {summary.get('observable', 'N/A'):<48} |")
    print(f"| Samples          | {str(summary.get('n_samples', 'N/A')):<48} |")
    print(f"| c-values Count   | {str(summary.get('n_c_values', 'N/A')):<48} |")
    print(f"| K Median         | {str(summary.get('K_median', 'N/A')):<48} |")
    print(f"| Classification   | {summary.get('classification', 'N/A').upper():<48} |")
    
    warnings = summary.get("warnings", [])
    if warnings:
        print("-"*80)
        print(" Warnings & Scope Cautions:")
        for w in warnings:
            print(f"  - {w}")
    print("="*80 + "\n")
