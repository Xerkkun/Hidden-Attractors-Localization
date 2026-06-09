from pathlib import Path
import re
import importlib
import sys

ROOT = Path(__file__).resolve().parents[2]
VERSION2 = ROOT / "version_2"

ACTIVE_DIRS = [
    VERSION2 / "hidden_attractors",
    VERSION2 / "examples",
    VERSION2 / "configs",
    VERSION2 / "validation",
    VERSION2 / "tests",
]

FORBIDDEN_FUNCTIONAL_PATTERNS = [
    r"import\s+_archived_figure_scripts",
    r"from\s+_archived_figure_scripts",
    r"Path\(['\"]_archived_figure_scripts",
    r"open\(['\"]_archived_figure_scripts",
    r"_archived_figure_scripts/",
    r"_archived_figure_scripts\\\\",
]

DOC_ALLOWED_NOTE = "intentionally excluded"

def test_archived_figure_scripts_is_gitignored():
    gitignore = ROOT / ".gitignore"
    assert gitignore.exists(), "Root .gitignore is missing"
    text = gitignore.read_text(encoding="utf-8")
    assert "_archived_figure_scripts/" in text

def test_active_code_does_not_depend_on_archived_scripts():
    violations = []
    exts = {".py", ".yaml", ".yml", ".json", ".md", ".tex"}

    for base in ACTIVE_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_dir() or path.suffix not in exts:
                continue
            # Skip checking this test file itself to prevent self-matching on the patterns
            if path.name == "test_archive_independence.py":
                continue
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")

            for pat in FORBIDDEN_FUNCTIONAL_PATTERNS:
                if re.search(pat, text):
                    # Allow purely explanatory docs only if explicitly marked as non-functional.
                    if path.suffix == ".md" and DOC_ALLOWED_NOTE in text:
                        continue
                    violations.append(f"{rel}: matches {pat}")

    assert not violations, "Active files depend on archived scripts:\n" + "\n".join(violations)

def test_package_import_does_not_require_archive():
    if str(VERSION2) not in sys.path:
        sys.path.insert(0, str(VERSION2))
    importlib.import_module("hidden_attractors")
