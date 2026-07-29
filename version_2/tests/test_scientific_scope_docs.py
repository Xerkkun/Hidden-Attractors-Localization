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


@pytest.mark.scientific_contract
def test_scope_names_every_top_level_characterization_entry_point() -> None:
    text = _read(SCOPE)
    expected = {
        "bifurcation_points_from_trajectories",
        "bifurcation_summary",
        "compute_boundedness_metrics",
        "compute_fft_psd",
        "compute_lyapunov_spectrum",
        "compute_trajectory_metrics",
        "detect_poincare_crossings",
        "estimate_time_series_lyapunov",
        "integer_system_lyapunov_exponents",
        "kaplan_yorke_dimension",
        "trajectory_metrics",
        "trajectory_metrics_for_system",
        "validate_lyapunov_method_request",
        "zero_one_test",
    }
    assert all(f"`{name}`" in text for name in expected)
    assert (
        "`hidden_attractors.analysis.integer_qr_benettin_lyapunov_exponents`"
        in text
    )


@pytest.mark.scientific_contract
def test_scope_documents_direct_system_and_parameter_access() -> None:
    text = _read(SCOPE)
    expected = {
        "get_system",
        "list_systems",
        "register_system",
        "requirements_for",
        "check_system_capability",
        "chua_parameters",
        "chua_nonsmooth_parameters",
        "chua_arctan_wu2023_parameters",
        "equilibria_nonsmooth",
        "equilibria_arctan",
        "jacobian_nonsmooth",
        "jacobian_arctan",
        "rhs_nonsmooth",
        "rhs_arctan",
    }
    assert all(f"`{name}`" in text for name in expected)


@pytest.mark.literature_traceability
def test_scope_has_required_literature_rows() -> None:
    text = _read(SCOPE)
    assert "Published reference coverage" in text
    assert "validation/published_reference_coverage.json" in text
    assert "partial reference implementation" in text
    assert "| Article | System / object |" not in text


@pytest.mark.scientific_contract
def test_legacy_hidden_verified_appears_only_in_alias_note() -> None:
    text = _read(SCOPE)
    assert "hidden_verified" not in text
    assert "`hidden_under_tested_neighborhoods`" in text
    assert "`compatible_with_hiddenness`" in text


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
