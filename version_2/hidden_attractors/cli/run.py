"""Official command-line interface for hidden_attractors.

Stability: internal
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from hidden_attractors.workflows.config_loader import (
    load_config,
    save_effective_config,
    apply_cli_overrides,
)
from hidden_attractors.workflows.simple_runner import run_simple_workflow


PRESETS = {
    "chua_integer": "chua_integer_centered_lure_df.yaml",
    "chua_fractional": "chua_fractional_centered_lure_df.yaml",
    "chua_arctan": "chua_arctan_fractional_centered_lure_df.yaml",
    "chua_arctan_only_integer": "chua_arctan_attractor_only_integer.yaml",
    "chua_arctan_only_fractional": "chua_arctan_attractor_only_fractional.yaml",
    "chua_bifurcation": "chua_fractional_bifurcation.yaml",
    "chua_basin": "chua_fractional_basin.yaml",
    "chua_full_protocol": "chua_full_protocol.yaml",
}


def find_example_config(filename: str) -> Path:
    """Resolve the template configuration path dynamically, copying to cache if not physically present."""
    from hidden_attractors.paths import get_example_config_resource, RUNTIME_CACHE
    import importlib.resources
    import shutil

    # Try resolving via importlib resources
    try:
        ref = get_example_config_resource(filename)
        if ref.exists():
            with importlib.resources.as_file(ref) as p:
                if p.exists():
                    if "temp" in str(p).lower() or "tmp" in str(p).lower() or "site-packages" in str(p).lower() or ".egg" in str(p).lower():
                        cache_dir = RUNTIME_CACHE / "configs"
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        dest = cache_dir / filename
                        shutil.copy2(p, dest)
                        return dest
                    return p
    except Exception:
        pass

    # 2. Cwd subfolder
    p2 = Path.cwd() / "configs" / "examples" / filename
    if p2.exists():
        return p2
        
    # 3. Parent workspace version_2
    p3 = Path.cwd() / "version_2" / "configs" / "examples" / filename
    if p3.exists():
        return p3
        
    raise FileNotFoundError(
        f"Example configuration template '{filename}' not found."
    )


def parse_dynamic_overrides(extra_args: List[str]) -> Dict[str, Any]:
    """Parse dynamic double-dashed --key=val or --key val arguments as dictionary overrides."""
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            if "=" in arg:
                key, val = arg[2:].split("=", 1)
            else:
                key = arg[2:]
                if i + 1 < len(extra_args) and not extra_args[i+1].startswith("-"):
                    val = extra_args[i+1]
                    i += 1
                else:
                    val = True
            
            # Smart type conversion for values
            if isinstance(val, str):
                if val.lower() in ("true", "yes", "on"):
                    val = True
                elif val.lower() in ("false", "no", "off"):
                    val = False
                elif val.lower() == "none":
                    val = None
                else:
                    try:
                        if "." in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        if "," in val:
                            try:
                                val = [float(x.strip()) for x in val.split(",")]
                            except ValueError:
                                val = [x.strip() for x in val.split(",")]
            
            overrides[key] = val
        i += 1
    return overrides


def run_cmd(args: argparse.Namespace, extra_args: List[str]) -> None:
    """Execute the run subcommand."""
    presets_to_run = []
    
    if args.preset == "basic_chua_three":
        presets_to_run = ["chua_integer", "chua_fractional", "chua_arctan"]
    elif args.preset:
        presets_to_run = [args.preset]
    elif not args.config:
        print("Error: Must provide --config (-c) or --preset (-p).")
        sys.exit(1)

    summaries = []
    overrides = parse_dynamic_overrides(extra_args)
    
    if presets_to_run:
        for p_name in presets_to_run:
            filename = PRESETS.get(p_name)
            if not filename:
                print(f"Error: Preset '{p_name}' not recognized. Available: {list(PRESETS.keys())}")
                sys.exit(1)
            try:
                config_path = find_example_config(filename)
                print(f"\n--- Running Preset: {p_name} ({config_path}) ---")
                config = load_config(config_path)
                if overrides:
                    config = apply_cli_overrides(config, overrides)
                
                res = run_simple_workflow(config)
                summaries.append(res)
            except Exception as e:
                print(f"Preset execution failed for {p_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        config_path = Path(args.config)
        try:
            config = load_config(config_path)
            if overrides:
                config = apply_cli_overrides(config, overrides)
            
            res = run_simple_workflow(config)
            summaries.append(res)
        except Exception as e:
            print(f"Config execution failed for {config_path}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    # Print unified summary table if more than one preset was run
    if len(summaries) > 1:
        print("\n" + "="*115)
        print(" SEQUENTIAL RUNS UNIFIED SUMMARY ")
        print("="*115)
        headers = ["system_id", "q", "transfer_mode", "integrator", "omega0", "amplitude_a0", "k", "final_class", "status"]
        
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


def init_cmd(args: argparse.Namespace) -> None:
    """Execute the init subcommand."""
    from hidden_attractors.paths import list_packaged_example_configs, get_example_config_resource
    import importlib.resources
    import shutil
    
    yaml_files = list_packaged_example_configs()

    if not yaml_files:
        print("Error: Could not find templates directory.")
        sys.exit(1)

    if args.example:
        filename = PRESETS.get(args.example, args.example)
        if not filename.endswith(".yaml"):
            filename += ".yaml"
            
        if filename not in yaml_files:
            print(f"Error: Template for example '{args.example}' ({filename}) not found.")
            sys.exit(1)
            
        dest_file = Path.cwd() / filename
        ref = get_example_config_resource(filename)
        with importlib.resources.as_file(ref) as local_file:
            shutil.copy2(local_file, dest_file)
        print(f"Copied template to {dest_file}")
    else:
        dest_dir = Path.cwd() / "configs" / "examples"
        dest_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for name in yaml_files:
            ref = get_example_config_resource(name)
            with importlib.resources.as_file(ref) as local_file:
                shutil.copy2(local_file, dest_dir / name)
            count += 1
        print(f"Copied {count} example configuration files to {dest_dir}")


def inspect_config_cmd(args: argparse.Namespace, extra_args: List[str]) -> None:
    """Execute the inspect-config subcommand."""
    if args.preset:
        filename = PRESETS.get(args.preset)
        if not filename:
            print(f"Error: Preset '{args.preset}' not recognized. Available: {list(PRESETS.keys())}")
            sys.exit(1)
        try:
            config_path = find_example_config(filename)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
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
        
        print("\n" + "="*80)
        print(f" EFFECTIVE CONFIGURATION FOR: {config_path} ")
        print("="*80)
        
        for k in sorted(config.keys()):
            val = config[k]
            if isinstance(val, dict):
                print(f"| {k:<25} |")
                for nk, nv in sorted(val.items()):
                    print(f"|   {nk:<23} | {str(nv):<48} |")
            else:
                print(f"| {k:<25} | {str(val):<48} |")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Error inspecting config: {e}")
        sys.exit(1)


def validate_bibliography_cmd(args: argparse.Namespace) -> None:
    """Execute the validate-bibliography subcommand."""
    from hidden_attractors.references.validator import validate_bibliography_manifest, write_traceability_matrix_markdown
    import json
    
    strict = bool(args.strict)
    manifest_path = args.manifest
    
    print(f"Validating bibliography manifest from: {manifest_path} (strict={strict})")
    
    try:
        res = validate_bibliography_manifest(manifest_path, strict=strict)
        
        # Output JSON if requested
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Overall Validation Status: {res['bibliographic_validation_status'].upper()}")
            print(f"Total Claims: {res['claims_total']}")
            print(f"Valid Claims: {res['claims_valid']}")
            if res["warnings"]:
                print("\nWarnings:")
                for w in res["warnings"]:
                    print(f"  - {w}")
            if res["claims_missing_references"]:
                print("\nClaims missing references (failed):")
                for c in res["claims_missing_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('text')}")
            if res["claims_with_unregistered_references"]:
                print("\nClaims with unregistered references (failed):")
                for c in res["claims_with_unregistered_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('references')}")
            if res["claims_with_insufficient_references"]:
                print("\nClaims with insufficient references (failed):")
                for c in res["claims_with_insufficient_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('references')}")
                    
        # Write markdown output if requested
        if args.markdown_output:
            write_traceability_matrix_markdown(res, args.markdown_output)
            print(f"\nTraceability matrix written to: {args.markdown_output}")
            
        if res["bibliographic_validation_status"] == "FAILED" and strict:
            sys.exit(1)
            
    except Exception as e:
        print(f"Bibliography validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entry point."""
    from .main import main as main_dispatcher
    main_dispatcher(argv)


if __name__ == "__main__":
    main()
