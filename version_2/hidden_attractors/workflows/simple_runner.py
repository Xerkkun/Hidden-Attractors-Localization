"""Dispatcher for hidden-attractor workflow stages.

Stability: experimental
"""

from __future__ import annotations

import os
import sys
import importlib
from typing import Any, Dict

from .config_loader import load_config, save_effective_config
from .attractor_only import run_attractor_only_workflow
from .bifurcation import run_bifurcation_workflow
from .basin_runner import run_basin_workflow


def run_simple_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the configured workflow stages in order or dispatch to sub-workflows.

    Parameters
    ----------
    config : dict
        Fully normalized configuration dict from config_loader.

    Returns
    -------
    dict
        Execution summary/results dictionary.
    """
    # 1. Dispatch to specialized workflows if designated stages are active
    if config.get("run_attractor_only"):
        print("[simple_runner] Stage run_attractor_only is enabled -> running attractor_only workflow")
        return run_attractor_only_workflow(config)

    if config.get("run_bifurcation"):
        print("[simple_runner] Stage run_bifurcation is enabled -> running bifurcation workflow")
        return run_bifurcation_workflow(config)

    if config.get("run_basin_slices"):
        print("[simple_runner] Stage run_basin_slices is enabled -> running basin runner workflow")
        return run_basin_workflow(config)

    # 2. Otherwise run the standard seed search / continuation / validation workflow
    # Import legacy centered_lure_df_workflow runner from src/
    try:
        # Check if src is in path or needs insertion
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if workspace_root not in sys.path:
            sys.path.insert(1, workspace_root)
        
        legacy_module = importlib.import_module("src.workflows.centered_lure_df_workflow")
        run_centered_lure_df_workflow = legacy_module.run_centered_lure_df_workflow
    except ImportError as e:
        raise ImportError(
            f"Could not import legacy centered_lure_df_workflow. "
            f"Make sure workspace root '{workspace_root}' is in sys.path. "
            f"Original error: {e}"
        )

    print("[simple_runner] Dispatching to centered_lure_df_workflow")
    # Save the effective configuration inside output_dir
    save_effective_config(config)
    return run_centered_lure_df_workflow(config)
