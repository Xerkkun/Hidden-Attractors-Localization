import os
import re
from pathlib import Path
import pytest

@pytest.mark.hygiene
def test_no_writing_to_real_paths():
    """Verify that no fast unit/contract test writes to real repository output directories."""
    tests_dir = Path(__file__).resolve().parents[2]
    
    dangerous_paths = [
        "version_2/library_figures",
        "version_2/validation",
        "version_2/outputs",
        "outputs/",
        "artifacts/",
    ]
    
    writing_patterns = [
        r"\.open\([\"'][wa]",
        r"open\(.*,[\s]*[\"'][wa][\"']",
        r"write_text",
        r"write_bytes",
        r"export_figure",
        r"savefig",
        r"\.mkdir\(",
        r"shutil\.copy"
    ]
    
    violations = []
    
    for root, dirs, files in os.walk(tests_dir):
        if '__pycache__' in root:
            continue
        for file in files:
            if not (file.startswith("test_") and file.endswith(".py")):
                continue
            if file == Path(__file__).name:
                continue
                
            filepath = Path(root) / file
            content = filepath.read_text(encoding="utf-8")
            
            # Check if it mentions any dangerous paths
            has_dangerous_path = any(path in content for path in dangerous_paths)
            
            # Check if it has any writing actions
            has_writing = any(re.search(pat, content) for pat in writing_patterns)
            
            if has_dangerous_path and has_writing:
                # Check if it uses tmp_path or monkeypatch
                uses_mocking = "tmp_path" in content or "monkeypatch" in content
                
                # Check if it is marked as slow or integration
                is_slow_or_int = "@pytest.mark.slow" in content or "@pytest.mark.integration" in content
                
                if not (uses_mocking or is_slow_or_int):
                    violations.append(
                        f"{filepath.relative_to(tests_dir.parent).as_posix()}: "
                        f"mentions real paths and writes to disk, but does not use 'tmp_path', "
                        f"'monkeypatch', or '@pytest.mark.slow' / '@pytest.mark.integration'."
                    )
                    
    assert not violations, "Found tests writing to real paths without proper mocking or slow/integration markers:\n" + "\n".join(violations)
