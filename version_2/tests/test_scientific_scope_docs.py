import pytest
"""Regression checks for the documented scientific scope."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "docs" / "scientific_scope.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.hygiene
def test_scientific_scope_page_is_linked() -> None:
    assert SCOPE.exists()
    assert "## Scientific Scope" in _read(ROOT / "README.md")
    assert "docs/scientific_scope.md" in _read(ROOT / "README.md")
    assert "scientific_scope.md" in _read(ROOT / "docs" / "index.md")
    assert "Scientific Scope: scientific_scope.md" in _read(ROOT / "mkdocs.yml")


@pytest.mark.scientific_contract
def test_scope_declares_supported_boundary_and_evidence_layers() -> None:
    text = _read(SCOPE)
    lowered = text.lower()
    for keyword in ("caputo", "commensurate", "lur'e", "scalar", "out of scope", "visual"):
        assert keyword in lowered
        
    required_terms = [
        "df", "nyquist", "seed", "not", "hiddenness proof",
        "continuation", "transports", "abm", "efork", "caputo",
        "matignon", "equilibria", "neighborhood", "basin", "promote"
    ]
    for term in required_terms:
        assert term in lowered


@pytest.mark.literature_traceability
def test_scope_has_required_literature_rows() -> None:
    text = _read(SCOPE)
    assert "| Article | System / object | Order | Method in article | Implemented / documented coverage | What the library extends | Library modules / evidence |" in text
    for author in ("Kuznetsov", "Danca", "Wu", "Machado", "Matignon", "Diethelm", "Caputo", "Guan", "Ghoreishi"):
        assert author in text
    rows = [line for line in text.splitlines() if line.startswith("| ")][1:]
    assert len(rows) >= 20


@pytest.mark.scientific_contract
def test_legacy_hidden_verified_appears_only_in_alias_note() -> None:
    text = _read(SCOPE)
    assert text.count("hidden_verified") == 1
    assert "Legacy `hidden_verified`" in text
    assert "`hiddenness_supported_under_tested_neighborhoods`" in text
    assert "`compatible_with_hiddenness_under_tested_radii`" in text


@pytest.mark.literature_traceability
def test_confirmed_reference_metadata_is_consistent() -> None:
    bib = _read(ROOT / "docs" / "references.bib")
    registry = _read(ROOT / "hidden_attractors" / "references" / "registry.py")
    kuz_case = _read(ROOT / "validation" / "published_cases" / "kuznetsov2017_chua_integer.yaml")
    assert "10.1016/j.ifacol.2017.08.470" in bib
    assert "10.1016/j.ifacol.2017.08.470" in registry
    assert "10.1016/j.ifacol.2017.08.470" in kuz_case
    assert "10.1007/s11071-017-3472-7" in bib
    assert "10.1016/j.sigpro.2014.05.012" in bib
    assert "10.1016/j.rinp.2023.106866" in registry
