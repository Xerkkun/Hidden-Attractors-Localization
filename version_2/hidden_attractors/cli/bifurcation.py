"""CLI commands for bifurcation diagrams and parameter sweeps.

Stability: internal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

from ..workflows.bifurcation import run_bifurcation_workflow
from ..workflows.config_loader import load_config, apply_cli_overrides
from ..plotting.bifurcation import plot_bifurcation_diagram_styled
from ..analysis.bifurcation import BifurcationPoint


def run_bifurcation(argv: Sequence[str] | None = None) -> None:
    """Run bifurcation parameter sweep workflow."""
    parser = argparse.ArgumentParser(description="Run bifurcation parameter sweep workflow")
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
        run_bifurcation_workflow(config)
    except Exception as e:
        print(f"Bifurcation sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def plot_bifurcation(argv: Sequence[str] | None = None) -> None:
    """Plot bifurcation diagram from CSV data."""
    parser = argparse.ArgumentParser(description="Plot bifurcation diagram from CSV data")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to bifurcation_data.csv")
    parser.add_argument("-o", "--output", type=str, help="Path to output plot file")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    # Read CSV
    points = []
    param_name = "parameter"
    observable_name = "observable"
    
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            param_name = row.get("parameter_name", "parameter")
            observable_name = row.get("coordinate", "observable")
            points.append(
                BifurcationPoint(
                    parameter=float(row["parameter_value"]),
                    observable=float(row["coordinate_value"]),
                    time=float(row.get("t", 0.0)),
                    index=int(row.get("index", 0)),
                    kind=row.get("sample_type", "maxima"),
                )
            )

    output_path = Path(args.output) if args.output else input_path.parent / "bifurcation_plot.png"
    
    # Render plot
    plot_bifurcation_diagram_styled(
        points,
        output_path,
        parameter_label=param_name,
        observable_label=f"{observable_name} (local extrema)",
    )
    print(f"Plot saved successfully -> {output_path}")


def inspect_bifurcation(argv: Sequence[str] | None = None) -> None:
    """Inspect bifurcation summary JSON."""
    parser = argparse.ArgumentParser(description="Inspect bifurcation summary JSON")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to bifurcation_summary.json")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Summary file '{input_path}' does not exist.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\n" + "="*80)
    print(f" BIFURCATION DIAGRAM SUMMARY: {input_path.name} ")
    print("="*80)
    print(f"| System ID        | {summary.get('system_id', 'N/A'):<48} |")
    print(f"| Parameter Swept  | {summary.get('parameter_swept', 'N/A'):<48} |")
    print(f"| Integrator       | {summary.get('integrator', 'N/A'):<48} |")
    print(f"| Memory Mode      | {summary.get('memory_mode', 'N/A'):<48} |")
    print(f"| Order (q_base)   | {str(summary.get('q_base', 'N/A')):<48} |")
    print(f"| Swept Points     | {str(summary.get('n_swept_points', 'N/A')):<48} |")
    print(f"| Successful Runs  | {str(summary.get('n_success', 'N/A')):<48} |")
    print(f"| Failed Runs      | {str(summary.get('n_failed', 'N/A')):<48} |")
    
    stats = summary.get("stats", {})
    if stats:
        print("-"*80)
        print(" Extrema Statistics:")
        for k, v in stats.items():
            print(f"  - {k}: {v}")
    print("="*80 + "\n")
