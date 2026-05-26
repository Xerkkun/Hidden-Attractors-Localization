import argparse
import os
import sys
import csv
import time
from typing import Any, Dict, List
from ..workflows.configs import load_and_validate_config
from ..workflows.centered_lure_df_workflow import run_centered_lure_df_workflow

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(argv: List[str] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
        
    parser = argparse.ArgumentParser(description="Centered Lur'e Describing Function Workflow runner.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.")
    parser.add_argument("--system", type=str, help="Override system_id.")
    parser.add_argument("--transfer-mode", type=str, choices=["integer", "fractional"], help="Override transfer_mode.")
    parser.add_argument("--continuation-mode", type=str, choices=["integer", "fractional"], help="Override continuation_mode.")
    parser.add_argument("--integrator", type=str, choices=["abm", "efork"], help="Override integrator.")
    parser.add_argument("--memory-mode", type=str, choices=["full", "window"], help="Override memory_mode.")
    parser.add_argument("--memory-window-length", type=int, help="Override memory_window_length.")
    parser.add_argument("--workers", type=int, help="Override workers.")
    parser.add_argument("--run-hiddenness-tests", type=str2bool, help="Override run_hiddenness_tests.")
    parser.add_argument("--run-basin-slices", type=str2bool, help="Override run_basin_slices.")
    parser.add_argument("--plot-enabled", type=str2bool, help="Override plot_enabled.")
    parser.add_argument("--preset", type=str, choices=["basic_chua_three", "abm_chua_three"], help="Run a predefined preset.")
    
    args = parser.parse_args(argv)
    
    if args.preset in {"basic_chua_three", "abm_chua_three"}:
        print(f"\n=== RUNNING PRESET: {args.preset} ===\n")
        # Define paths to the three configs
        # Note: configs are created under configs/examples/ in the workspace root
        configs_to_run = [
            "configs/examples/chua_integer_centered_lure_df.yaml",
            "configs/examples/chua_fractional_centered_lure_df.yaml",
            "configs/examples/chua_arctan_fractional_centered_lure_df.yaml"
        ]
        
        summaries = []
        for c_path in configs_to_run:
            if not os.path.exists(c_path):
                print(f"ERROR: Default config not found at: {c_path}")
                continue
            print(f"\n--- Running: {c_path} ---")
            try:
                config = load_and_validate_config(c_path)
                
                # Apply preset overrides
                if args.preset == "abm_chua_three":
                    config["integrator"] = "abm"
                    config["run_hiddenness_tests"] = True
                    
                # Apply any CLI overrides to each preset run
                if args.workers is not None:
                    config["workers"] = args.workers
                if args.integrator is not None:
                    config["integrator"] = args.integrator
                if args.plot_enabled is not None:
                    config["plot_enabled"] = args.plot_enabled
                if args.run_hiddenness_tests is not None:
                    config["run_hiddenness_tests"] = args.run_hiddenness_tests
                if args.run_basin_slices is not None:
                    config["run_basin_slices"] = args.run_basin_slices
                    
                sum_res = run_centered_lure_df_workflow(config)
                summaries.append(sum_res)
            except Exception as e:
                print(f"Execution failed for {c_path}: {e}")
                
        # Print a unified final table of results
        if summaries:
            print("\n" + "="*95)
            print(" PRESET EXECUTION UNIFIED SUMMARY ")
            print("="*95)
            headers = ["system_id", "q", "transfer_mode", "integrator", "omega0", "amplitude_a0", "k", "final_class", "status"]
            print(f"| {' | '.join(f'{h:<15}' for h in headers)} |")
            print(f"|{'-'*93}|")
            for s in summaries:
                row = []
                for h in headers:
                    val = s.get(h, "N/A")
                    if isinstance(val, float):
                        val_str = f"{val:.4f}"
                    else:
                        val_str = str(val)
                    row.append(f"{val_str:<15}")
                print(f"| {' | '.join(row)} |")
            print("="*95 + "\n")
            
            # Save unified summary CSV in outputs/
            os.makedirs("outputs", exist_ok=True)
            unified_csv_path = f"outputs/preset_{args.preset}_summary.csv"
            with open(unified_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for s in summaries:
                    writer.writerow([s.get(h, "N/A") for h in headers])
            print(f"Unified preset summary saved to: {unified_csv_path}\n")
        return
        
    # Standard single config execution
    if not args.config:
        parser.error("either --config or --preset basic_chua_three/abm_chua_three is required.")
        
    config = load_and_validate_config(args.config)
    
    # Apply CLI overrides
    if args.system is not None:
        config["system_id"] = args.system
    if args.transfer_mode is not None:
        config["transfer_mode"] = args.transfer_mode
    if args.continuation_mode is not None:
        config["continuation_mode"] = args.continuation_mode
    if args.integrator is not None:
        config["integrator"] = args.integrator
    if args.memory_mode is not None:
        config["memory_mode"] = args.memory_mode
    if args.memory_window_length is not None:
        config["memory_window_length"] = args.memory_window_length
    if args.workers is not None:
        config["workers"] = args.workers
    if args.run_hiddenness_tests is not None:
        config["run_hiddenness_tests"] = args.run_hiddenness_tests
    if args.run_basin_slices is not None:
        config["run_basin_slices"] = args.run_basin_slices
    if args.plot_enabled is not None:
        config["plot_enabled"] = args.plot_enabled
        
    run_centered_lure_df_workflow(config)

if __name__ == "__main__":
    main()
