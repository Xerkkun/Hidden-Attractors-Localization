"""Primary command dispatcher for the unified hidden-attractors CLI.

Stability: internal
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

# Import subcommand dispatchers
from .inspect import list_candidates, systems, workflow_requirements
from .validate import validate_contract, validate_bibliography
from .protocol import run_protocol_stage
from .hiddenness import sphere_controls, strict_target_refinement as hid_str_ref
from .basin import refined, strict_target_refinement as bas_str_ref
from .robustness import overlay
from .published import danca_abm_sphere_controls
from .report import fractional_run
from .bifurcation import run_bifurcation, plot_bifurcation, inspect_bifurcation
from .lyapunov import compute_lyapunov, trajectory_lyapunov_spectrum, validate_lyapunov
from .chaos_test import run_zero_one, inspect_zero_one


GROUPS = {
    "run": None,
    "init": None,
    "inspect-config": None,
    "inspect": ["candidates", "systems", "workflow-requirements"],
    "validate": ["contract", "bibliography"],
    "protocol": ["generate-seeds", "soft-precheck", "continue", "filter-survivors", "build-reference", "robustness", "hiddenness", "diagnostics"],
    "hiddenness": ["sphere-controls", "strict-target-refinement"],
    "basin": ["refined", "strict-target-refinement"],
    "robustness": ["overlay"],
    "bifurcation": ["run", "plot", "inspect"],
    "lyapunov": ["compute", "spectrum", "validate"],
    "chaos-test": ["zero-one", "inspect"],
    "published": ["danca-abm-sphere-controls"],
    "report": ["fractional-run"],
    "seed": ["lure-centered", "lure-biased", "machado-centered", "machado-biased"],
    "continuation": ["run", "multiparameter"],
}


def dispatch(group: str, cmd: str | None, argv: Sequence[str]) -> None:
    """Route the command to the appropriate subcommand logic."""
    if group == "run":
        # run command in run.py
        from .run import run_cmd
        # run_cmd expects parsed args and extra_args.
        # Let's parse args for run subcommand
        parser = argparse.ArgumentParser(prog="hidden-attractors run")
        parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
        parser.add_argument("-p", "--preset", type=str, help="Select a built-in config preset or 'basic_chua_three'")
        args, extra_args = parser.parse_known_args(argv)
        run_cmd(args, extra_args)
        
    elif group == "init":
        from .run import init_cmd
        parser = argparse.ArgumentParser(prog="hidden-attractors init")
        parser.add_argument("-e", "--example", type=str, help="Name of a specific example preset to extract")
        args = parser.parse_args(argv)
        init_cmd(args)
        
    elif group == "inspect-config":
        from .run import inspect_config_cmd
        parser = argparse.ArgumentParser(prog="hidden-attractors inspect-config")
        parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
        parser.add_argument("-p", "--preset", type=str, help="Select a built-in config preset")
        args, extra_args = parser.parse_known_args(argv)
        inspect_config_cmd(args, extra_args)
        
    elif group == "inspect":
        if cmd == "candidates":
            list_candidates()
        elif cmd == "systems":
            systems(argv)
        elif cmd == "workflow-requirements":
            workflow_requirements(argv)
            
    elif group == "validate":
        if cmd == "contract":
            validate_contract(argv)
        elif cmd == "bibliography":
            validate_bibliography(argv)
            
    elif group == "protocol":
        run_protocol_stage(cmd, argv)
        
    elif group == "hiddenness":
        if cmd == "sphere-controls":
            sphere_controls(argv)
        elif cmd == "strict-target-refinement":
            hid_str_ref(argv)
            
    elif group == "basin":
        if cmd == "refined":
            refined(argv)
        elif cmd == "strict-target-refinement":
            bas_str_ref(argv)
            
    elif group == "robustness":
        if cmd == "overlay":
            overlay(argv)
            
    elif group == "published":
        if cmd == "danca-abm-sphere-controls":
            danca_abm_sphere_controls(argv)
            
    elif group == "report":
        if cmd == "fractional-run":
            fractional_run(argv)
            
    elif group == "bifurcation":
        if cmd == "run":
            run_bifurcation(argv)
        elif cmd == "plot":
            plot_bifurcation(argv)
        elif cmd == "inspect":
            inspect_bifurcation(argv)
            
    elif group == "lyapunov":
        if cmd == "compute":
            compute_lyapunov(argv)
        elif cmd == "spectrum":
            trajectory_lyapunov_spectrum(argv)
        elif cmd == "validate":
            validate_lyapunov(argv)
            
    elif group == "chaos-test":
        if cmd == "zero-one":
            run_zero_one(argv)
        elif cmd == "inspect":
            inspect_zero_one(argv)
            
    elif group == "seed":
        if cmd == "lure-centered":
            from .seed import lure_centered
            lure_centered(argv)
        elif cmd == "lure-biased":
            from .seed import lure_biased
            lure_biased(argv)
        elif cmd in ("machado-centered", "machado-biased"):
            print("Machado/FDF seed generation is planned but not implemented in this release.")
            sys.exit(1)
            
    elif group == "continuation":
        if cmd == "run":
            from .continuation import run_scalar_continuation
            run_scalar_continuation(argv)
        elif cmd == "multiparameter":
            from .continuation import run_multiparameter_continuation
            run_multiparameter_continuation(argv)
            
    else:
        print(f"Unknown command group: {group}")
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    # Intercept leaf-level --help / -h to show correct subcommand help
    if len(argv) >= 2 and argv[0] in GROUPS and argv[1] not in ("-h", "--help"):
        if "-h" in argv[2:] or "--help" in argv[2:]:
            dispatch(argv[0], argv[1], argv[2:])
            return

    parser = argparse.ArgumentParser(
        prog="hidden-attractors",
        description="Unified interface for fractional-order hidden attractor analysis.",
        allow_abbrev=False
    )
    
    subparsers = parser.add_subparsers(dest="group", required=True, help="Command group to execute")
    
    # 1. Direct commands
    subparsers.add_parser("run", add_help=False, help="Run an experiment configuration")
    subparsers.add_parser("init", add_help=False, help="Copy template configs to the current directory")
    subparsers.add_parser("inspect-config", add_help=False, help="Preview the normalized configuration")
    
    # 2. Grouped commands
    inspect_parser = subparsers.add_parser("inspect", help="Inspect candidates, systems, or workflows")
    inspect_sub = inspect_parser.add_subparsers(dest="cmd", required=True)
    inspect_sub.add_parser("candidates", help="List final candidate records")
    inspect_sub.add_parser("systems", help="Inspect registered chaotic systems")
    inspect_sub.add_parser("workflow-requirements", help="Inspect reusable workflow requirements")
    
    validate_parser = subparsers.add_parser("validate", help="Validate contracts or bibliography")
    validate_sub = validate_parser.add_subparsers(dest="cmd", required=True)
    validate_sub.add_parser("contract", help="Validate numerical validation evidence contract")
    validate_sub.add_parser("bibliography", help="Validate claims bibliography manifest")
    
    protocol_parser = subparsers.add_parser("protocol", help="Official Caputo protocol stages")
    protocol_sub = protocol_parser.add_subparsers(dest="cmd", required=True)
    for p_cmd in GROUPS["protocol"]:
        protocol_sub.add_parser(p_cmd, help=f"Run protocol {p_cmd} stage")
        
    hiddenness_parser = subparsers.add_parser("hiddenness", help="Hiddenness verification workflows")
    hiddenness_sub = hiddenness_parser.add_subparsers(dest="cmd", required=True)
    hiddenness_sub.add_parser("sphere-controls", help="Run sphere controls validation workflow")
    hiddenness_sub.add_parser("strict-target-refinement", help="Run strict target refinement workflow")
    
    basin_parser = subparsers.add_parser("basin", help="Basin of attraction workflows")
    basin_sub = basin_parser.add_subparsers(dest="cmd", required=True)
    basin_sub.add_parser("refined", help="Run refined basin workflow")
    basin_sub.add_parser("strict-target-refinement", help="Run strict target refinement workflow for basins")
    
    robustness_parser = subparsers.add_parser("robustness", help="Robustness workflows")
    robustness_sub = robustness_parser.add_subparsers(dest="cmd", required=True)
    robustness_sub.add_parser("overlay", help="Run robustness overlay workflow")
    
    bif_parser = subparsers.add_parser("bifurcation", help="Bifurcation sweep workflows and plots")
    bif_sub = bif_parser.add_subparsers(dest="cmd", required=True)
    bif_sub.add_parser("run", help="Run parameter sweep bifurcation workflow")
    bif_sub.add_parser("plot", help="Plot bifurcation diagram from CSV data")
    bif_sub.add_parser("inspect", help="Inspect bifurcation summary JSON")
    
    lyap_parser = subparsers.add_parser("lyapunov", help="Lyapunov exponent calculation and convergence")
    lyap_sub = lyap_parser.add_subparsers(dest="cmd", required=True)
    lyap_sub.add_parser("compute", help="Compute Lyapunov exponents workflow")
    lyap_sub.add_parser("spectrum", help="Estimate trajectory-based Lyapunov exponent")
    lyap_sub.add_parser("validate", help="Validate Lyapunov summary JSON")
    
    chaos_parser = subparsers.add_parser("chaos-test", help="Supporting chaos time-series diagnostics")
    chaos_sub = chaos_parser.add_subparsers(dest="cmd", required=True)
    chaos_sub.add_parser("zero-one", help="Run 0-1 chaos-test diagnostic")
    chaos_sub.add_parser("inspect", help="Inspect 0-1 chaos-test summary JSON")
    
    pub_parser = subparsers.add_parser("published", help="Replicate/validate published workflows")
    pub_sub = pub_parser.add_subparsers(dest="cmd", required=True)
    pub_sub.add_parser("danca-abm-sphere-controls", help="Run published Danca ABM sphere controls")
    
    rep_parser = subparsers.add_parser("report", help="Report generation and publication figures")
    rep_sub = rep_parser.add_subparsers(dest="cmd", required=True)
    rep_sub.add_parser("fractional-run", help="Run fractional report run workflow")
    
    seed_parser = subparsers.add_parser("seed", help="Seed generation routes")
    seed_sub = seed_parser.add_subparsers(dest="cmd", required=True)
    seed_sub.add_parser("lure-centered", help="Centered Lur'e seed generation")
    seed_sub.add_parser("lure-biased", help="Biased Lur'e seed generation")
    seed_sub.add_parser("machado-centered", help="Machado centered seed generation (planned)")
    seed_sub.add_parser("machado-biased", help="Machado biased seed generation (planned)")
    
    cont_parser = subparsers.add_parser("continuation", help="Numerical continuation routes")
    cont_sub = cont_parser.add_subparsers(dest="cmd", required=True)
    cont_sub.add_parser("run", help="Run scalar continuation")
    cont_sub.add_parser("multiparameter", help="Run multiparameter continuation")

    args, extra_args = parser.parse_known_args(argv)
    dispatch(args.group, args.cmd, extra_args)


if __name__ == "__main__":
    main()
