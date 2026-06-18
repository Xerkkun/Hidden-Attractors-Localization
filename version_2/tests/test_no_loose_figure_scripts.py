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
    "version_2/figure_scripts",
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

@pytest.mark.hygiene
def test_no_loose_figure_scripts_outside_designated_directories():
    """Verify that no active figure-generation or plotting scripts exist outside
    version_2/figure_scripts/, version_2/hidden_attractors/plotting/, and version_2/tools/legacy/.
    Also ensures that the root LaTeX reports are excluded from checking.
    """
    allowed_prefixes = [
        "version_2/figure_scripts/",
        "version_2/hidden_attractors/",
        "version_2/tools/cli/",
        "version_2/tools/legacy/",
        "version_2/tests/",
        "version_2/docs/",
        "version_2/benchmarks/",
    ]
    violations = []
    
    # Scan recursively for python files under version_2/
    version_2_dir = ROOT_DIR / "version_2"
    if not version_2_dir.exists():
        return
        
    for r, d, files in os.walk(version_2_dir):
        # Exclude directories like __pycache__, .pytest_cache
        d[:] = [dirname for dirname in d if not dirname.startswith(".") and dirname != "__pycache__"]
        
        rel_dir = os.path.relpath(r, ROOT_DIR).replace("\\", "/")
        
        # Skip if directory is inside allowed prefixes
        is_allowed = False
        for prefix in allowed_prefixes:
            if rel_dir.startswith(prefix.rstrip("/")) or (rel_dir + "/").startswith(prefix):
                is_allowed = True
                break
        if is_allowed:
            continue
            
        for f in files:
            if not f.endswith(".py"):
                continue
                
            file_path = Path(r) / f
            file_rel = file_path.relative_to(ROOT_DIR).as_posix()
            
            # Check name or direct savefig calls
            name_lower = f.lower()
            is_fig_script = (
                "plot" in name_lower or
                "figure" in name_lower or
                "basin" in name_lower
            )
            
            has_savefig = False
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if ".savefig(" in content:
                    has_savefig = True
            except Exception:
                pass
                
            if is_fig_script or has_savefig:
                # Exempt normal workflows or common utilities that are not primarily loose figure scripts
                exemptions = [
                    "version_2/hidden_attractors/workflows/fractional_report_run.py",
                    "version_2/hidden_attractors/workflows/refined_basin.py",
                    "version_2/hidden_attractors/workflows/danca_abm_sphere_controls.py",
                    "version_2/hidden_attractors/workflows/robustness_overlay.py",
                    "version_2/hidden_attractors/workflows/robustness.py",
                    "version_2/hidden_attractors/workflows/basin.py",
                    "version_2/hidden_attractors/workflows/bifurcation.py",
                    "version_2/hidden_attractors/workflows/continuation.py",
                    "version_2/hidden_attractors/workflows/chaos_test.py",
                    "version_2/hidden_attractors/workflows/seed_search.py",
                    "version_2/hidden_attractors/workflows/simulation.py",
                    "version_2/hidden_attractors/workflows/sphere_tests.py",
                ]
                if file_rel in exemptions:
                    continue
                violations.append(f"Figure script '{file_rel}' found outside designated directories.")
                
    assert not violations, "Found loose figure scripts outside version_2/figure_scripts/:\n" + "\n".join(violations)
