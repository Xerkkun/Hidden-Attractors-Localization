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

def parse_list_str(v):
    if not v:
        return []
    return [x.strip() for x in v.split(",")]

def parse_list_float(v):
    if not v:
        return []
    return [float(x.strip()) for x in v.split(",")]

def parse_equilibria_selection(v):
    if not v:
        return "all"
    if v.lower() == "all":
        return "all"
    return [x.strip() for x in v.split(",")]

def main(argv: List[str] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
        
    parser = argparse.ArgumentParser(description="Centered Lur'e Describing Function Workflow CLI Runner.")
    
    # Config and Preset selects
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.")
    parser.add_argument("--preset", type=str, choices=["chua_integer", "chua_fractional", "chua_arctan", "basic_chua_three"], help="Select a preset to run.")
    
    # Core Overrides
    parser.add_argument("--system-id", "--system", type=str, help="Override system_id.")
    parser.add_argument("--q", type=float, help="Override order q.")
    parser.add_argument("--transfer-mode", type=str, choices=["integer", "fractional"], help="Override transfer_mode.")
    parser.add_argument("--seed-strategy", type=str, choices=["k_phi", "imw_gain", "nyquist_df"], help="Override seed_strategy.")
    parser.add_argument("--seed-construction", type=str, choices=["modal", "closed_form_integer"], help="Override seed_construction.")
    parser.add_argument("--branch-index", type=int, help="Override branch_index.")
    parser.add_argument("--continuation-mode", type=str, choices=["integer", "fractional"], help="Override continuation_mode.")
    parser.add_argument("--dynamics-mode", type=str, choices=["integer", "fractional", "system"], help="Override dynamics_mode.")
    parser.add_argument("--integrator", type=str, choices=["abm", "efork"], help="Override integrator.")
    parser.add_argument("--memory-mode", type=str, choices=["full", "window"], help="Override memory_mode.")
    parser.add_argument("--memory-window-length", type=int, help="Override memory_window_length.")
    parser.add_argument("--h", type=float, help="Override h.")
    parser.add_argument("--t-final", type=float, help="Override t_final.")
    parser.add_argument("--t-burn", type=float, help="Override t_burn.")
    parser.add_argument("--workers", type=int, help="Override workers.")
    
    # Search / Grid Overrides
    parser.add_argument("--omega-min", type=float, help="Override omega_min.")
    parser.add_argument("--omega-max", type=float, help="Override omega_max.")
    parser.add_argument("--amplitude-min", type=float, help="Override amplitude_min.")
    parser.add_argument("--amplitude-max", type=float, help="Override amplitude_max.")
    parser.add_argument("--grid-size-omega", type=int, help="Override grid_size_omega.")
    parser.add_argument("--grid-size-amplitude", type=int, help="Override grid_size_amplitude.")
    parser.add_argument("--df-residual-tol", type=float, help="Override df_residual_tol.")
    parser.add_argument("--root-refinement", type=str2bool, help="Override root_refinement.")
    
    # Phase/Test Switches
    parser.add_argument("--run-hiddenness-tests", type=str2bool, help="Override run_hiddenness_tests.")
    parser.add_argument("--run-basin-slices", type=str2bool, help="Override run_basin_slices.")
    parser.add_argument("--run-sphere-tests", type=str2bool, help="Override run_sphere_tests.")
    parser.add_argument("--run-robustness", type=str2bool, help="Override run_robustness.")
    parser.add_argument("--plot-enabled", type=str2bool, help="Override plot_enabled.")
    
    # Sphere Overrides
    parser.add_argument("--sphere-equilibria", type=parse_equilibria_selection, help="Override sphere_tests equilibrium_selection (e.g. 'all' or 'E0,E+').")
    parser.add_argument("--sphere-radii", type=parse_list_float, help="Override sphere_tests radii (comma-separated, e.g. '1e-5,1e-4').")
    parser.add_argument("--sphere-samples-initial", type=int, help="Override sphere_tests samples_initial.")
    parser.add_argument("--sphere-samples-growth-factor", type=float, help="Override sphere_tests samples_growth_factor.")
    parser.add_argument("--sphere-t-final", type=float, help="Override sphere_tests t_final.")
    parser.add_argument("--sphere-t-burn", type=float, help="Override sphere_tests t_burn.")
    
    # Basin Overrides
    parser.add_argument("--basin-planes", type=parse_list_str, help="Override basin planes (comma-separated, e.g. 'xy,xz').")
    parser.add_argument("--basin-grid-n", type=int, help="Override basin grid_n.")
    parser.add_argument("--basin-x-interval", type=parse_list_float, help="Override basin x_interval (comma-separated min,max, e.g. '-10,10').")
    parser.add_argument("--basin-y-interval", type=parse_list_float, help="Override basin y_interval (comma-separated min,max).")
    parser.add_argument("--basin-z-interval", type=parse_list_float, help="Override basin z_interval (comma-separated min,max).")
    parser.add_argument("--basin-equilibria", type=parse_equilibria_selection, help="Override basin equilibrium_selection.")
    parser.add_argument("--basin-around-equilibria", type=str2bool, help="Override basin around_equilibria.")
    parser.add_argument("--basin-local-radius", type=float, help="Override basin local_radius.")
    parser.add_argument("--basin-t-final", type=float, help="Override basin t_final.")
    parser.add_argument("--basin-t-burn", type=float, help="Override basin t_burn.")
    
    args = parser.parse_args(argv)
    
    # Determine base configs to run
    configs_to_run = []
    
    preset_mapping = {
        "chua_integer": "configs/examples/chua_integer_centered_lure_df.yaml",
        "chua_fractional": "configs/examples/chua_fractional_centered_lure_df.yaml",
        "chua_arctan": "configs/examples/chua_arctan_fractional_centered_lure_df.yaml"
    }
    
    if args.preset == "basic_chua_three":
        print(f"\n=== EJECUTANDO PRESET SECUENCIAL: basic_chua_three ===\n")
        configs_to_run = [
            ("configs/examples/chua_integer_centered_lure_df.yaml", "Chua Integer"),
            ("configs/examples/chua_fractional_centered_lure_df.yaml", "Chua Fractional"),
            ("configs/examples/chua_arctan_fractional_centered_lure_df.yaml", "Chua Arctan")
        ]
    elif args.preset in preset_mapping:
        print(f"\n=== EJECUTANDO PRESET: {args.preset} ===\n")
        configs_to_run = [(preset_mapping[args.preset], args.preset)]
    elif args.config:
        configs_to_run = [(args.config, "User Config")]
    else:
        parser.error("Debe proporcionar --config o --preset (chua_integer, chua_fractional, chua_arctan, basic_chua_three).")
        
    summaries = []
    
    for c_path, name in configs_to_run:
        if not os.path.exists(c_path):
            print(f"ERROR: No se encontró el archivo de configuración en: {c_path}")
            continue
            
        print(f"\n--- Iniciando Ejecución: {name} ({c_path}) ---")
        try:
            # Load YAML configuration and populate defaults
            config = load_and_validate_config(c_path)
            
            # Apply general CLI overrides
            if args.system_id is not None:
                config["system_id"] = args.system_id
            if args.q is not None:
                config["q"] = args.q
            if args.transfer_mode is not None:
                config["transfer_mode"] = args.transfer_mode
            if args.seed_strategy is not None:
                config["seed_strategy"] = args.seed_strategy
            if args.seed_construction is not None:
                config["seed_construction"] = args.seed_construction
            if args.branch_index is not None:
                config["branch_index"] = args.branch_index
            if args.continuation_mode is not None:
                config["continuation_mode"] = args.continuation_mode
            if args.dynamics_mode is not None:
                config["dynamics_mode"] = args.dynamics_mode
            if args.integrator is not None:
                config["integrator"] = args.integrator
            if args.memory_mode is not None:
                config["memory_mode"] = args.memory_mode
            if args.memory_window_length is not None:
                config["memory_window_length"] = args.memory_window_length
            if args.h is not None:
                config["h"] = args.h
            if args.t_final is not None:
                config["t_final"] = args.t_final
            if args.t_burn is not None:
                config["t_burn"] = args.t_burn
            if args.workers is not None:
                config["workers"] = args.workers
                
            # Search / Grid Overrides
            if args.omega_min is not None:
                config["omega_min"] = args.omega_min
            if args.omega_max is not None:
                config["omega_max"] = args.omega_max
            if args.amplitude_min is not None:
                config["amplitude_min"] = args.amplitude_min
            if args.amplitude_max is not None:
                config["amplitude_max"] = args.amplitude_max
            if args.grid_size_omega is not None:
                config["grid_size_omega"] = args.grid_size_omega
            if args.grid_size_amplitude is not None:
                config["grid_size_amplitude"] = args.grid_size_amplitude
            if args.df_residual_tol is not None:
                config["df_residual_tol"] = args.df_residual_tol
            if args.root_refinement is not None:
                config["root_refinement"] = args.root_refinement
                
            # Phase / plot switches
            if args.run_hiddenness_tests is not None:
                config["run_hiddenness_tests"] = args.run_hiddenness_tests
            if args.run_basin_slices is not None:
                config["run_basin_slices"] = args.run_basin_slices
            if args.run_sphere_tests is not None:
                config["run_sphere_tests"] = args.run_sphere_tests
            if args.run_robustness is not None:
                config["run_robustness"] = args.run_robustness
            if args.plot_enabled is not None:
                config["plot_enabled"] = args.plot_enabled
                
            # Apply sphere nested overrides
            if any(x is not None for x in [args.sphere_equilibria, args.sphere_radii, args.sphere_samples_initial, args.sphere_samples_growth_factor, args.sphere_t_final, args.sphere_t_burn]):
                st = config["sphere_tests"]
                if args.sphere_equilibria is not None:
                    st["equilibrium_selection"] = args.sphere_equilibria
                if args.sphere_radii:
                    st["radii"] = args.sphere_radii
                if args.sphere_samples_initial is not None:
                    st["samples_initial"] = args.sphere_samples_initial
                if args.sphere_samples_growth_factor is not None:
                    st["samples_growth_factor"] = args.sphere_samples_growth_factor
                if args.sphere_t_final is not None:
                    st["t_final"] = args.sphere_t_final
                if args.sphere_t_burn is not None:
                    st["t_burn"] = args.sphere_t_burn
                    
            # Apply basin nested overrides
            if any(x is not None for x in [args.basin_planes, args.basin_grid_n, args.basin_x_interval, args.basin_y_interval, args.basin_z_interval, args.basin_equilibria, args.basin_around_equilibria, args.basin_local_radius, args.basin_t_final, args.basin_t_burn]):
                b = config["basin"]
                if args.basin_planes:
                    b["planes"] = args.basin_planes
                if args.basin_grid_n is not None:
                    b["grid_n"] = args.basin_grid_n
                if args.basin_x_interval:
                    b["x_interval"] = args.basin_x_interval
                if args.basin_y_interval:
                    b["y_interval"] = args.basin_y_interval
                if args.basin_z_interval:
                    b["z_interval"] = args.basin_z_interval
                if args.basin_equilibria is not None:
                    b["equilibrium_selection"] = args.basin_equilibria
                if args.basin_around_equilibria is not None:
                    b["around_equilibria"] = args.basin_around_equilibria
                if args.basin_local_radius is not None:
                    b["local_radius"] = args.basin_local_radius
                if args.basin_t_final is not None:
                    b["t_final"] = args.basin_t_final
                if args.basin_t_burn is not None:
                    b["t_burn"] = args.basin_t_burn
            
            # Execute workflow
            sum_res = run_centered_lure_df_workflow(config)
            summaries.append(sum_res)
        except Exception as e:
            print(f"La ejecución falló para {c_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # Unified output summary table for multiple runs (such as basic_chua_three)
    if len(summaries) > 1:
        print("\n" + "="*115)
        print(" PRESET SECUENCIAL basic_chua_three - RESUMEN UNIFICADO ")
        print("="*115)
        headers = ["system_id", "q", "transfer_mode", "integrator", "omega0", "amplitude_a0", "k", "final_class", "status"]
        
        # Display format helper
        def _fmt(val):
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)
            
        print(f"| {' | '.join(f'{h:<16}' if h != 'system_id' else f'{h:<26}' for h in headers)} |")
        print(f"|{'-'*113}|")
        for s in summaries:
            row = [
                f"{s.get('system_id', 'N/A'):<26}",
                f"{_fmt(s.get('q', 'N/A')):<16}",
                f"{s.get('transfer_mode', 'N/A'):<16}",
                f"{s.get('integrator', 'N/A'):<16}",
                f"{_fmt(s.get('omega0', 'N/A')):<16}",
                f"{_fmt(s.get('amplitude_a0', 'N/A')):<16}",
                f"{_fmt(s.get('k', 'N/A')):<16}",
                f"{s.get('final_class', 'N/A'):<16}",
                f"{s.get('status', 'N/A'):<16}"
            ]
            print(f"| {' | '.join(row)} |")
        print("="*115 + "\n")
        
        # Save unified preset summary CSV
        os.makedirs("outputs", exist_ok=True)
        unified_csv_path = f"outputs/preset_{args.preset}_summary.csv"
        with open(unified_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for s in summaries:
                writer.writerow([s.get(h, "N/A") for h in headers])
        print(f"Resumen unificado guardado en: {unified_csv_path}\n")

if __name__ == "__main__":
    main()
