# -*- coding: utf-8 -*-
import fnmatch
import os
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]

PROHIBITED_ACTIVE_PATTERNS = [
    "scratch_*.py",
    "step[0-9]_*.py",
    "generate_*_plots*.py",
    "search_*_candidates*.py",
    "compare_solvers_*.py",
]

# We also keep a few specific historic ones for absolute safety
PROHIBITED_SPECIFIC_PATHS = [
    "scratch/replot_biased_lyapunov.py",
    "scratch/compute_biased_lyapunov_two_methods.py",
    "scratch/generate_biased_candidate_report_assets.py",
    "generate_all_plots_and_summary.py",
    "plot_chaotic_candidates.py",
    "search_saturation_candidates.py",
    "search_arctan_fractional.py",
    "compare_solvers_saturation.py",
]

ACTIVE_DIRS = [
    ".",
    "version_2",
    "version_2/examples",
    "version_2/tools/cli",
]

EXCLUDED_DIRS = [
    "version_2/tools/legacy",
    "_archived_figure_scripts",
    "version_2/tests",
    "version_2/tests/hygiene",
]

@pytest.mark.hygiene
def test_no_loose_or_duplicate_scripts():
    """Verify that none of the prohibited script patterns exist in active directories."""
    violations = []

    # 1. Check specific hardcoded prohibited paths
    for rel_path in PROHIBITED_SPECIFIC_PATHS:
        full_path = ROOT_DIR / rel_path
        if full_path.exists():
            violations.append(f"Specific prohibited file exists: {rel_path}")

    # 2. Check general glob patterns in active directories
    for active_dir_rel in ACTIVE_DIRS:
        active_path = ROOT_DIR / active_dir_rel
        if not active_path.exists():
            continue
            
        # List files directly under this directory (non-recursive to avoid traversing into excluded subdirectories)
        for item in active_path.iterdir():
            if not item.is_file():
                continue
                
            rel_to_root = item.relative_to(ROOT_DIR).as_posix()
            
            # Check exclusions
            is_excluded = False
            for excl in EXCLUDED_DIRS:
                if rel_to_root.startswith(excl + "/"):
                    is_excluded = True
                    break
            if is_excluded:
                continue
                
            # Check patterns
            for pat in PROHIBITED_ACTIVE_PATTERNS:
                if fnmatch.fnmatch(item.name, pat):
                    violations.append(f"File '{rel_to_root}' matches prohibited pattern '{pat}'")
                    
    assert not violations, "Found prohibited active scripts in repository:\n" + "\n".join(violations)
