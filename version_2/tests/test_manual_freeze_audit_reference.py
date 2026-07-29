from __future__ import annotations

import pytest

from tests.helpers.test_documentation_text import ROOT, active_doc_paths, read
from tests.test_manual_manifest import load_manifest


OBSOLETE_COUNT_KEYWORDS = (
    "156 tests",
    "156 unit tests",
    "156 pruebas",
    "suite de 156",
    "156 passed",
)


@pytest.mark.hygiene
def test_validation_audit_is_kept_out_of_public_manifest() -> None:
    violations: list[str] = []
    for path in active_doc_paths():
        content = read(path).lower()
        for keyword in OBSOLETE_COUNT_KEYWORDS:
            if keyword in content:
                violations.append(f"{path.name} mentions obsolete count {keyword!r}")

    assert not violations, "\n".join(violations)
    assert (ROOT / "validation" / "freeze_audit").is_dir()

    manifest = load_manifest()
    assert "freeze_audit" not in manifest
    assert "excluded from PyPI distributions" in manifest["publication_boundary"]["validation_records"]
