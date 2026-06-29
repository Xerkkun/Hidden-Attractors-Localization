from __future__ import annotations

from pathlib import Path

import hidden_attractors as ha


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