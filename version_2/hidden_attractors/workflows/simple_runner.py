"""Dispatcher for hidden-attractor workflow stages.

Stability: experimental
"""

from __future__ import annotations

from typing import Any, Dict

from .config_loader import save_effective_config
from .attractor_only import run_attractor_only_workflow
from .bifurcation import run_bifurcation_workflow
from .basin_runner import run_basin_workflow
from .centered_lure_df import run_centered_lure_df_workflow


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
    if config.get("run_attractor_only"):
        print("[simple_runner] Stage run_attractor_only is enabled -> running attractor_only workflow")
        return run_attractor_only_workflow(config)

    if config.get("run_bifurcation") and not config.get("run_seed_search"):
        print("[simple_runner] Stage run_bifurcation is enabled (exclusive) -> running bifurcation workflow")
        return run_bifurcation_workflow(config)

    if config.get("run_basin_slices") and not config.get("run_seed_search"):
        print("[simple_runner] Stage run_basin_slices is enabled (exclusive) -> running basin runner workflow")
        return run_basin_workflow(config)

    print("[simple_runner] Dispatching to centered_lure_df_workflow")
    save_effective_config(config)
    return run_centered_lure_df_workflow(config)
