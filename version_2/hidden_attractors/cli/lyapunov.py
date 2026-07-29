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
from ..analysis import estimate_time_series_lyapunov
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
    """Estimate Lyapunov diagnostics from a scalar trajectory observable."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Rosenstein LLE, Eckmann spectrum, and Kaplan-Yorke "
            "dimension from one scalar time series"
        )
    )
    parser.add_argument("-t", "--trajectory", type=str, required=True, help="Path to trajectory CSV")
    parser.add_argument("--observable", default="x", help="State coordinate to use for estimation")
    parser.add_argument("--window-start", type=int, default=0, help="First retained CSV sample")
    parser.add_argument(
        "--window-length",
        type=int,
        default=4096,
        help=(
            "Number of retained samples (default: 4096, limiting the "
            "quadratic-memory Rosenstein calculation)"
        ),
    )
    parser.add_argument("--time-unit", default="trajectory_time")
    parser.add_argument("--rosenstein-emb-dim", type=int, default=10)
    parser.add_argument("--rosenstein-lag", type=int)
    parser.add_argument("--rosenstein-min-tsep", type=int)
    parser.add_argument("--rosenstein-min-neighbors", type=int, default=20)
    parser.add_argument("--rosenstein-trajectory-len", type=int, default=20)
    parser.add_argument(
        "--rosenstein-fit",
        choices=("RANSAC", "poly"),
        default="poly",
    )
    parser.add_argument("--rosenstein-fit-offset", type=int, default=0)
    parser.add_argument("--eckmann-emb-dim", type=int, default=9)
    parser.add_argument("--eckmann-matrix-dim", type=int, default=3)
    parser.add_argument("--eckmann-min-neighbors", type=int)
    parser.add_argument("--eckmann-min-tsep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairwise-mib", type=int, default=256)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable result",
    )
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

    if args.window_start < 0:
        print("Error: --window-start must be non-negative.")
        sys.exit(1)
    if args.window_length is not None and args.window_length <= 0:
        print("Error: --window-length must be positive.")
        sys.exit(1)

    stop = (
        None
        if args.window_length is None
        else args.window_start + args.window_length
    )
    times = times[args.window_start:stop]
    signal = signal[args.window_start:stop]
    if len(signal) < 100:
        print("Error: Retained signal is too short. Need at least 100 points.")
        sys.exit(1)

    t_arr = np.array(times)
    sig_arr = np.array(signal)

    try:
        h = infer_step(t_arr)
        diffs = np.diff(t_arr)
        if not np.all(np.isfinite(diffs)) or not np.allclose(
            diffs,
            h,
            rtol=1e-6,
            atol=max(1e-12, abs(h) * 1e-9),
        ):
            raise ValueError(
                "trajectory time column is not uniformly sampled in the retained window"
            )

        result = estimate_time_series_lyapunov(
            sig_arr,
            sample_interval=h,
            time_unit=args.time_unit,
            observable=args.observable,
            rosenstein_emb_dim=args.rosenstein_emb_dim,
            rosenstein_lag=args.rosenstein_lag,
            rosenstein_min_tsep=args.rosenstein_min_tsep,
            rosenstein_min_neighbors=args.rosenstein_min_neighbors,
            rosenstein_trajectory_len=args.rosenstein_trajectory_len,
            rosenstein_fit=args.rosenstein_fit,
            rosenstein_fit_offset=args.rosenstein_fit_offset,
            eckmann_emb_dim=args.eckmann_emb_dim,
            eckmann_matrix_dim=args.eckmann_matrix_dim,
            eckmann_min_neighbors=args.eckmann_min_neighbors,
            eckmann_min_tsep=args.eckmann_min_tsep,
            random_seed=args.seed,
            max_pairwise_matrix_bytes=args.max_pairwise_mib * 1024 * 1024,
        )

        payload = {
            "analysis_type": "time_series_lyapunov",
            "status": "completed",
            "trajectory": str(trajectory_path),
            "window_start": args.window_start,
            "window_length": len(signal),
            **result.to_dict(),
        }
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        spectrum = ", ".join(f"{value:.9g}" for value in result.spectrum)
        print("\n" + "="*80)
        print(" SCALAR TIME-SERIES LYAPUNOV DIAGNOSTICS (nolds) ")
        print("="*80)
        print(f"| Trajectory File  | {trajectory_path.name:<48} |")
        print(f"| Observable       | {args.observable:<48} |")
        print(f"| Signal Length    | {len(signal):<48} |")
        print(f"| Sampling interval| {h:<48.9g} |")
        print(
            f"| Rosenstein LLE   | {result.largest_exponent:<35.9g} "
            f"{result.exponent_unit:<12} |"
        )
        print(f"| Eckmann spectrum | {spectrum:<48} |")
        print(f"| Kaplan-Yorke D   | {result.kaplan_yorke_dimension:<48.9g} |")
        print(f"| KY status        | {result.kaplan_yorke_status:<48} |")
        print(f"| Evidence status  | {result.evidence_status:<48} |")
        print(f"| Warning          | {'supporting diagnostic only; not a chaos proof':<48} |")
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
