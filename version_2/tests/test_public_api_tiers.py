from __future__ import annotations

from pathlib import Path

import hidden_attractors as ha
import hidden_attractors.analysis as analysis


ROOT = Path(__file__).resolve().parents[1]


def test_top_level_exports_are_partitioned_by_public_tier() -> None:
    stability_helpers = {
        "EXPERIMENTAL",
        "INTERNAL",
        "LEGACY",
        "STABLE",
        "api_tier",
        "assert_tier",
        "get_tier",
        "PUBLIC_API_STABLE",
        "PUBLIC_API_EXPERIMENTAL",
        "PUBLIC_API_TIERS",
    }
    stable = set(ha.PUBLIC_API_STABLE)
    experimental = set(ha.PUBLIC_API_EXPERIMENTAL)
    exported = set(ha.__all__)

    assert stable
    assert experimental
    assert stable.isdisjoint(experimental)
    assert exported == stability_helpers | stable | experimental
    assert ha.PUBLIC_API_TIERS[ha.STABLE] == ha.PUBLIC_API_STABLE
    assert ha.PUBLIC_API_TIERS[ha.EXPERIMENTAL] == ha.PUBLIC_API_EXPERIMENTAL


def test_every_declared_public_symbol_reports_its_runtime_tier() -> None:
    for name in ha.PUBLIC_API_STABLE:
        assert ha.get_tier(getattr(ha, name)) == ha.STABLE, name
    for name in ha.PUBLIC_API_EXPERIMENTAL:
        assert ha.get_tier(getattr(ha, name)) == ha.EXPERIMENTAL, name


def test_compatibility_aliases_and_internals_are_not_public_tier_exports() -> None:
    blocked = {
        "chua_piecewise_parameters",
        "equilibria_piecewise",
        "jacobian_piecewise",
        "rhs_piecewise",
        "cli",
        "native",
        "parallel",
        "paths",
        "legacy",
    }
    exported = set(ha.__all__)
    tiered = set(ha.PUBLIC_API_STABLE) | set(ha.PUBLIC_API_EXPERIMENTAL)

    assert exported.isdisjoint(blocked)
    assert tiered.isdisjoint(blocked)


def test_api_stability_docs_explain_top_level_export_boundary() -> None:
    text = (ROOT / "docs" / "api_stability.md").read_text(encoding="utf-8")
    assert "PUBLIC_API_STABLE" in text
    assert "PUBLIC_API_EXPERIMENTAL" in text
    assert "does not treat every exported name as equally stable" in text
    assert "Compatibility aliases" in text


def test_analysis_wildcard_surface_matches_tested_top_level_contract() -> None:
    expected = {
        name
        for name in ha.PUBLIC_API_EXPERIMENTAL
        if hasattr(analysis, name)
    }
    expected.add("integer_qr_benettin_lyapunov_exponents")
    assert set(analysis.__all__) == expected
