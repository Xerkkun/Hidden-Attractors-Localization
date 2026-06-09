import os
import re
from pathlib import Path
import pytest

@pytest.mark.hygiene
def test_all_tests_are_inventoried():
    """Verify that every test file in tests/ is listed in docs/tests_inventory.md."""
    tests_dir = Path(__file__).resolve().parent
    inventory_path = tests_dir / ".." / "docs" / "tests_inventory.md"
    assert inventory_path.exists(), "docs/tests_inventory.md does not exist"
    
    # Read inventory content and extract files
    inventory_content = inventory_path.read_text(encoding="utf-8")
    inventoried_files = set()
    # Matches markdown table rows like | tests/cli/test_continuation_order_cli.py | ...
    for line in inventory_content.splitlines():
        if line.strip().startswith("|") and "test_" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 1:
                # Find the one that starts with tests/
                for p in parts:
                    if p.startswith("tests/") and p.endswith(".py"):
                        inventoried_files.add(p)
                        
    # Find all test files on disk
    disk_files = set()
    for root, dirs, files in os.walk(tests_dir):
        if '__pycache__' in root:
            continue
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                rel_path = Path(root).relative_to(tests_dir.parent) / file
                disk_files.add(rel_path.as_posix())
                
    missing_from_inventory = disk_files - inventoried_files
    assert not missing_from_inventory, (
        f"The following test files are on disk but not listed in docs/tests_inventory.md:\n"
        f"{missing_from_inventory}"
    )

@pytest.mark.hygiene
def test_all_markers_are_registered():
    """Verify that all pytest markers used in tests are registered in pyproject.toml."""
    tests_dir = Path(__file__).resolve().parent
    pyproject_path = tests_dir / ".." / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml does not exist"
    
    # Read pyproject.toml to extract markers
    pyproject_content = pyproject_path.read_text(encoding="utf-8")
    registered_markers = set()
    # Find all lines in the markers list
    in_markers = False
    for line in pyproject_content.splitlines():
        line = line.strip()
        if line.startswith("markers = ["):
            in_markers = True
            continue
        if in_markers and line == "]":
            in_markers = False
            continue
        if in_markers:
            # Matches "marker_name: description" or "marker_name"
            m = re.match(r'^["\']([^:"\']+)', line)
            if m:
                registered_markers.add(m.group(1).strip())
                
    # Also add standard built-in pytest markers that might be used
    registered_markers.update([
        "skip", "skipif", "xfail", "parametrize", "usefixtures", "filterwarnings", "tryfirst", "trylast"
    ])
    
    # Collect all used markers in test files
    used_markers = set()
    for root, dirs, files in os.walk(tests_dir):
        if '__pycache__' in root:
            continue
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                filepath = Path(root) / file
                content = filepath.read_text(encoding="utf-8")
                # Split content into lines and find matches only in active code lines (not comments)
                for line in content.splitlines():
                    if line.strip().startswith("#"):
                        continue
                    matches = re.findall(r'@pytest\.mark\.(\w+)', line)
                    used_markers.update(matches)
                
    unregistered_markers = used_markers - registered_markers
    assert not unregistered_markers, (
        f"The following markers are used in tests but not registered in pyproject.toml:\n"
        f"{unregistered_markers}"
    )
