from __future__ import annotations

import json
from pathlib import Path

import hidden_attractors as ha
from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS
from hidden_attractors.fractional.contracts import FRACTIONAL_METHODS


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "validation" / "software_audit" / "current_function_closure.json"


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_fractional_method_closure_is_total_and_matches_registry() -> None:
    data = _matrix()
    records = data["fractional_methods"]
    assert isinstance(records, dict)
    assert set(records) == set(FRACTIONAL_METHODS)
    assert all(
        method.implementation_status != "experimental"
        for method in FRACTIONAL_METHODS.values()
    )

    implemented_closures = {
        "validated_promoted",
        "validated_promoted_operator_under_research_gated_derivative",
    }
    for name, method in FRACTIONAL_METHODS.items():
        record = records[name]
        assert record["implementation_status"] == method.implementation_status
        if method.implementation_status == "implemented":
            assert record["closure_status"] in implemented_closures
        elif method.implementation_status == "planned":
            assert record["closure_status"] == "planned_preserved"
        elif method.implementation_status == "theoretical_only":
            assert record["closure_status"] == "theoretical_boundary_preserved"
        else:
            raise AssertionError(f"Method {name!r} lacks a terminal closure state.")


def test_every_fractional_evidence_profile_resolves_to_real_files() -> None:
    data = _matrix()
    profiles = data["evidence_profiles"]
    records = data["fractional_methods"]
    assert isinstance(profiles, dict)
    assert isinstance(records, dict)
    for name, record in records.items():
        profile = profiles[record["evidence_profile"]]
        assert profile["tests"], name
        assert profile["docs"], name
        for category in ("tests", "docs", "oracles"):
            for relative in profile[category]:
                assert (ROOT / relative).is_file(), (name, category, relative)


def test_recent_analysis_capabilities_are_promoted_and_test_anchored() -> None:
    promoted = _matrix()["promoted_analysis_api"]
    assert isinstance(promoted, dict)
    for name, test_path in promoted.items():
        assert name in ha.PUBLIC_API_EXPERIMENTAL
        assert name in ha.__all__
        assert ha.get_tier(getattr(ha, name)) == ha.EXPERIMENTAL
        assert (ROOT / test_path).is_file()


def test_lyapunov_methods_have_terminal_verdicts_and_discrepancies_are_quarantined() -> None:
    verdicts = _matrix()["lyapunov_terminal_verdicts"]
    assert isinstance(verdicts, dict)
    assert set(LYAPUNOV_METHODS) <= set(verdicts)
    for method_id, info in LYAPUNOV_METHODS.items():
        verdict = verdicts[method_id]["closure_status"]
        if info.benchmark_status == "recorded_benchmark_discrepancy":
            assert verdict == "quarantined_reproduction_discrepancy"
        else:
            assert verdict in {
                "validated_promoted",
                "synthetic_validated_experimental",
            }


def test_removed_integer_scheme_is_absent_from_active_sources() -> None:
    removed_name = "".join(("he", "un"))
    excluded_roots = {
        ROOT / "site",
        ROOT / "local_reports" / "retired_public_docs",
        ROOT / "validation" / "freeze_audit",
        ROOT / "validation" / "source_snapshots",
    }
    suffixes = {".py", ".md", ".tex", ".yaml", ".yml"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(root == path or root in path.parents for root in excluded_roots):
            continue
        if removed_name in path.read_text(encoding="utf-8", errors="ignore").lower():
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
