from __future__ import annotations

import os
import sys
import pytest
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_no_src_dependency():
    """Verify that no active python files under version_2 import or refer to 'src' package."""
    version_2_dir = workspace_root / "version_2"
    dirs_to_check = [version_2_dir / "hidden_attractors", version_2_dir / "tests"]

    forbidden_patterns = [
        "import src",
        "from src",
        'importlib.import_module("src',
        "importlib.import_module('src",
        '__import__("src',
        "__import__('src",
    ]

    for check_dir in dirs_to_check:
        assert check_dir.exists(), f"{check_dir} does not exist"
        
        for root, _, files in os.walk(check_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                if file in ("test_no_src_dependency.py", "test_workflow_smoke.py"):
                    continue
                    
                path = Path(root) / file
                
                # Read line by line
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    
                in_docstring = False
                docstring_char = None
                
                for line_num, line in enumerate(lines, 1):
                    stripped_line = line.strip()
                    
                    # Track docstrings
                    if not in_docstring:
                        if '"""' in stripped_line:
                            if stripped_line.count('"""') % 2 == 1:
                                in_docstring = True
                                docstring_char = '"""'
                            else:
                                continue  # Single-line docstring, skip checking
                        elif "'''" in stripped_line:
                            if stripped_line.count("'''") % 2 == 1:
                                in_docstring = True
                                docstring_char = "'''"
                            else:
                                continue  # Single-line docstring, skip checking
                    else:
                        if docstring_char in stripped_line:
                            in_docstring = False
                        continue  # Skip checking lines inside docstring
                        
                    # Strip comments
                    if "#" in line:
                        code_part = line.split("#", 1)[0].strip()
                    else:
                        code_part = stripped_line
                        
                    if not code_part:
                        continue
                        
                    # Check forbidden patterns
                    for pattern in forbidden_patterns:
                        if pattern in code_part:
                            pytest.fail(f"Forbidden legacy import/dependency found in {path.name}:{line_num}: '{line.strip()}'")
