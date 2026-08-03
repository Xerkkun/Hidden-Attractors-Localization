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
    "generate_*_report_assets*.py",
    "plot_*_candidates*.py",
    "search_*_candidates*.py",
    "search_*_fractional*.py",
    "compare_solvers_*.py",
]

ACTIVE_DIRS = [
    ".",
    "version_2",
    "version_2/examples",
    "version_2/tools/cli",
]

EXCLUDED_DIRS = [
    "version_2/tools/legacy",
    "version_2/tests",
    "version_2/tests/hygiene",
    "version_2/figure_scripts",
]

@pytest.mark.hygiene
def test_no_loose_or_duplicate_scripts():
    """Verify that none of the prohibited script patterns exist in active directories."""
    violations = []

    # Check only generic filename patterns in active directories.
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
        "version_2/examples/",
        "version_2/tools/legacy/",
        "version_2/validation/",
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
        excluded_names = {"__pycache__", "build", "local_reports"}
        if Path(r) == version_2_dir:
            # version_2/tmp is an ignored staging area, not active source.
            excluded_names.add("tmp")
        d[:] = [
            dirname
            for dirname in d
            if not dirname.startswith(".")
            and dirname not in excluded_names
        ]
        
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
                violations.append(f"Figure script '{file_rel}' found outside designated directories.")
                
    assert not violations, "Found loose figure scripts outside version_2/figure_scripts/:\n" + "\n".join(violations)

