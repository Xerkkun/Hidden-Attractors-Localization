# -*- coding: utf-8 -*-
import fnmatch
import os
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

PROHIBITED_ACTIVE_PATTERNS = [
    "scratch_*.py",
    "step*.py",
    "generate_*_plots*.py",
    "generate_*plots*.py",
    "generate_*figures*.py",
    "search_*candidates*.py",
    "search_*_candidates*.py",
    "compare_*solvers*.py",
    "compare_solvers_*.py",
    "*_old.py",
    "*_backup.py",
    "*_tmp.py",
    "test_manual_*.py",
]

EXCLUDED_PATHS = [
    "tools/legacy",
    "_archive",
    "_archived_figure_scripts",
]

EXEMPTED_FILES = [
    "hidden_attractors/plotting/generate_publication_figures.py",
    "tests/test_manual_manifest.py",
    "tests/test_manual_manifest_consistency.py",
    "tests/test_user_manual.py",
    "tests/test_user_manual_claims_consistency.py",
    "tests/test_manual_claims_consistency.py",
    "tests/test_manual_cli_consistency.py",
    "tests/test_manual_freeze_audit_reference.py",
]

@pytest.mark.hygiene
def test_no_loose_active_scripts():
    """Verify that no prohibited script patterns exist in active directories,

    except in explicitly legacy/archive folders or exempted files.
    """
    violations = []
    
    # We scan version_2/ recursively
    for r, d, files in os.walk(ROOT_DIR):
        rel_dir = os.path.relpath(r, ROOT_DIR).replace("\\", "/")
        
        # Skip if directory is inside excluded paths
        if rel_dir != "." and any(rel_dir.startswith(excl) for excl in EXCLUDED_PATHS):
            continue
            
        for f in files:
            file_rel_path = os.path.relpath(os.path.join(r, f), ROOT_DIR).replace("\\", "/")
            
            # Check exclusions again at file level (e.g. if the parent dir check was not enough)
            if any(file_rel_path.startswith(excl + "/") for excl in EXCLUDED_PATHS):
                continue
                
            if file_rel_path in EXEMPTED_FILES:
                continue
                
            # Check pattern matches
            for pat in PROHIBITED_ACTIVE_PATTERNS:
                if fnmatch.fnmatch(f, pat):
                    violations.append(
                        f"Active file '{file_rel_path}' matches prohibited pattern '{pat}'."
                    )
                    break
                    
    assert not violations, "Found prohibited active scripts in repository:\n" + "\n".join(violations)


@pytest.mark.hygiene
def test_no_legacy_script_as_recommended_command():
    """Fail if any legacy script from tools/legacy/ is recommended in markdown documentation

    as an execution command (preceded by python, powershell, etc.).
    """
    legacy_dir = ROOT_DIR / "tools/legacy"
    if not legacy_dir.exists():
        return
        
    legacy_scripts = [f.name for f in legacy_dir.iterdir() if f.is_file() and f.suffix in (".py", ".ps1")]
    
    # Scan markdown files
    markdown_violations = []
    command_prefixes = [r"python", r"python3", r"powershell", r"pwsh", r"cmd", r"\./", r"bash", r"sh"]
    
    # We compile regexes for each legacy script to find command recommendations
    regexes = {}
    for script in legacy_scripts:
        # Matches e.g. python tools/legacy/foo.py or ./foo.py
        pattern = rf"(?:{'|'.join(command_prefixes)})\s+.*?\b{re.escape(script)}\b"
        regexes[script] = re.compile(pattern, re.IGNORECASE)
        
    for r, d, files in os.walk(ROOT_DIR):
        rel_dir = os.path.relpath(r, ROOT_DIR).replace("\\", "/")
        if any(rel_dir.startswith(excl) for excl in EXCLUDED_PATHS):
            continue
            
        for f in files:
            if not f.endswith(".md"):
                continue
                
            md_path = Path(r) / f
            try:
                content = md_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
                
            for line_no, line in enumerate(content.splitlines(), 1):
                for script, rx in regexes.items():
                    if rx.search(line):
                        rel_md = md_path.relative_to(ROOT_DIR).as_posix()
                        markdown_violations.append(
                            f"{rel_md}:{line_no}: Found legacy script '{script}' in recommended command: '{line.strip()}'"
                        )
                        
    assert not markdown_violations, (
        "Legacy scripts must not be promoted as recommended execution commands in documentation:\n"
        + "\n".join(markdown_violations)
    )
