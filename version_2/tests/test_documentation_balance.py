from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

MAIN_DOCUMENTS = [
    WORKSPACE / "README.md",
    WORKSPACE / "RELEASE_NOTES.md",
    WORKSPACE / "CHANGELOG.md",
    WORKSPACE / "codemeta.json",
    WORKSPACE / ".zenodo.json",
    ROOT / "README.md",
    ROOT / "USER_MANUAL.md",
    ROOT / "THESIS_CLAIMS.md",
    ROOT / "docs" / "quick_start.md",
    ROOT / "docs" / "getting_started.md",
    ROOT / "docs" / "examples.md",
    ROOT / "docs" / "examples_index.md",
    ROOT / "docs" / "hiddenness_verification.md",
    ROOT / "docs" / "scientific_scope.md",
    ROOT / "docs" / "systems.md",
    ROOT / "docs" / "unified_report.md",
    ROOT / "docs" / "validation_evidence.md",
    ROOT / "docs" / "validation_methodology.md",
    ROOT / "release_package" / "README_RELEASE.md",
    ROOT / "release_package" / "PROGRAM_SUMMARY.md",
    ROOT / "release_package" / "ARCTAN_C590_PROMOTION_BOUNDARY.md",
    ROOT / "release_package" / "reproducibility_checklist.md",
    ROOT / "examples" / "README.md",
    ROOT / "examples" / "chua_nonsmooth_biased_hidden_attractor" / "README.md",
    ROOT / "examples" / "chua_arctan_wu2023" / "README.md",
    ROOT / "validation" / "chua_fractional_arctan" / "README.md",
    ROOT / "validation" / "chua_fractional_arctan_c590" / "README.md",
]

BALANCE_REQUIRED_DOCUMENTS = [
    WORKSPACE / "README.md",
    WORKSPACE / "RELEASE_NOTES.md",
    WORKSPACE / "CHANGELOG.md",
    WORKSPACE / "codemeta.json",
    WORKSPACE / ".zenodo.json",
    ROOT / "README.md",
    ROOT / "USER_MANUAL.md",
    ROOT / "THESIS_CLAIMS.md",
    ROOT / "docs" / "quick_start.md",
    ROOT / "docs" / "getting_started.md",
    ROOT / "docs" / "examples.md",
    ROOT / "docs" / "examples_index.md",
    ROOT / "docs" / "scientific_scope.md",
    ROOT / "release_package" / "README_RELEASE.md",
    ROOT / "release_package" / "PROGRAM_SUMMARY.md",
    ROOT / "examples" / "README.md",
]

PROHIBITED_PHRASES = [
    "main promoted result",
    "central validation case",
    "release centered on arctan",
    "release is centered on arctan",
    "key contribution is arctan",
    "the key contribution is arctan",
    "arctan validates the library",
]

INTEGER_CHUA = re.compile(r"\binteger\b.{0,80}\bchua\b|\bchua\b.{0,80}\binteger\b", re.I | re.S)
NONSMOOTH_CHUA = re.compile(
    r"non[- ]?smooth|nonsmooth|saturation|biased[- ]?df|biased describing function|\bbdf\b",
    re.I,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


@pytest.mark.hygiene
def test_documentation_avoids_arctan_centrality_phrases() -> None:
    violations: list[str] = []
    for path in MAIN_DOCUMENTS:
        text = _read(path).lower()
        for phrase in PROHIBITED_PHRASES:
            if phrase in text:
                violations.append(f"{path.relative_to(WORKSPACE)} contains {phrase!r}")
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
def test_main_arctan_documents_also_name_integer_and_nonsmooth_lanes() -> None:
    violations: list[str] = []
    for path in BALANCE_REQUIRED_DOCUMENTS:
        text = _read(path)
        if "arctan" not in text.lower():
            continue
        if not INTEGER_CHUA.search(text) or not NONSMOOTH_CHUA.search(text):
            violations.append(
                f"{path.relative_to(WORKSPACE)} mentions arctan without balanced integer/non-smooth Chua context"
            )
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
def test_macro_radius_contacts_are_not_documented_as_automatic_self_excited() -> None:
    violations: list[str] = []
    for path in MAIN_DOCUMENTS:
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            lower = line.lower()
            if "macro" in lower and "radius" in lower and "self-excited" in lower:
                if "not" not in lower and "extended" not in lower and "local" not in lower:
                    violations.append(f"{path.relative_to(WORKSPACE)}:{lineno}: {line}")
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
def test_candidate_rejected_language_is_tied_to_a_recorded_contract() -> None:
    contract_terms = ("local contract", "recorded contract", "tested contract", "recorded local", "under_local_contract")
    violations: list[str] = []
    for path in MAIN_DOCUMENTS:
        text = _read(path)
        for match in re.finditer(r"candidate[_ -]rejected|candidate rejected", text, re.I):
            window = text[max(0, match.start() - 160) : match.end() + 160].lower()
            if not any(term in window for term in contract_terms):
                violations.append(f"{path.relative_to(WORKSPACE)} uses candidate rejection without contract context")
    assert not violations, "\n".join(violations)


@pytest.mark.hygiene
def test_docs_define_local_neighborhoods_versus_extended_spherical_audits() -> None:
    corpus = "\n".join(_read(path).lower() for path in MAIN_DOCUMENTS)
    assert "local neighborhoods versus extended spherical audits" in corpus
    assert "a contact detected on a sphere of large radius" in corpus
    assert "large-radius spherical probes are reported as extended basin-geometry audits" in corpus