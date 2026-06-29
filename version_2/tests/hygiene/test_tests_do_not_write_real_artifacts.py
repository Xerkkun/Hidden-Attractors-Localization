import os
import re
from pathlib import Path

import pytest


@pytest.mark.hygiene
def test_no_writing_to_real_paths():
    """Verify that no fast unit/contract test writes to real repository output directories."""
    tests_dir = Path(__file__).resolve().parents[2]

    dangerous_path_patterns = [
        r"version_2[/\\]library_figures",
        r"version_2[/\\]validation_outputs",
        r"version_2[/\\]outputs",
        r"ROOT\s*/\s*[\"']validation[\"']",
        r"ROOT\s*/\s*[\"']library_figures[\"']",
        r"ROOT_DIR\s*/\s*[\"']version_2[\"']\s*/\s*[\"']library_figures[\"']",
        r"[\"']outputs/",
        r"[\"']artifacts/",
    ]

    writing_patterns = [
        r"\.open\([\"'][wa]",
        r"open\(.*,[\s]*[\"'][wa][\"']",
        r"write_text",
        r"write_bytes",
        r"export_figure",
        r"savefig",
        r"\.mkdir\(",
        r"shutil\.copy",
        r"shutil\.rmtree",
    ]

    violations = []

    for root, dirs, files in os.walk(tests_dir):
        if "__pycache__" in root:
            continue
        for file in files:
            if not (file.startswith("test_") and file.endswith(".py")):
                continue
            if file == Path(__file__).name:
                continue

            filepath = Path(root) / file
            content = filepath.read_text(encoding="utf-8")

            has_dangerous_path = any(re.search(pattern, content) for pattern in dangerous_path_patterns)
            has_writing = any(re.search(pattern, content) for pattern in writing_patterns)

            if has_dangerous_path and has_writing:
                uses_isolation = "tmp_path" in content or "monkeypatch" in content
                is_slow_or_int = "@pytest.mark.slow" in content or "@pytest.mark.integration" in content
                documents_readonly_promoted_artifact = "read_text" in content and not any(
                    token in content for token in ("write_text", "write_bytes", "savefig", "export_figure")
                )

                if not (uses_isolation or is_slow_or_int or documents_readonly_promoted_artifact):
                    violations.append(
                        f"{filepath.relative_to(tests_dir.parent).as_posix()}: "
                        "mentions real artifact paths and writes to disk without tmp_path, "
                        "monkeypatch, or an integration/slow boundary."
                    )

    assert not violations, "Found tests writing to real paths without isolation:\n" + "\n".join(violations)


@pytest.mark.hygiene
def test_tests_inventory_has_no_pending_real_output_refactors():
    inventory_path = Path(__file__).resolve().parents[2] / "docs" / "tests_inventory.md"
    rows = [line for line in inventory_path.read_text(encoding="utf-8").splitlines() if line.startswith("| tests/")]
    violations = []
    for line in rows:
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 10:
            continue
        file_name, _objective, _category, _speed, _kind, _deps, writes_real_outputs, _long_sims, action, justification = columns[:10]
        if writes_real_outputs == "si" or writes_real_outputs == "sí":
            violations.append(f"{file_name}: inventory still marks real output writes")
        if action == "refactorizar" and "outputs reales" in justification.lower():
            violations.append(f"{file_name}: inventory still asks for output-write refactor")

    assert not violations, "tests_inventory.md has unresolved real-output refactors:\n" + "\n".join(violations)