"""Tests for published_continuation_comparison phase (Phase E).

Tests verify:
1. YAML configs exist and have correct schema.
2. No hidden_verified or chaos_verified claims in any output.
3. Article scope classification logic.
4. published_continuation_reproduced is NEVER set when reports_continuation=false.
5. Wu k=null does not invent k or run deformed Lure continuation.
6. Kuznetsov q=1 does not use history_window_transport.
7. IC resolution for Wu (paper_reported_initial_condition) and Danca (missing).
8. compare_paper_style_vs_history returns correct status on large differences.
9. Smoke tests for Wu and Danca (fast mode).
10. Summary schema validation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.python.published_continuation_comparison import (
    load_published_continuation_config,
    classify_article_continuation_scope,
    resolve_initial_conditions,
    run_paper_style_integration,
    compare_to_published_result,
    compare_paper_style_vs_history,
    run_published_continuation_case,
    run_all_published_continuation_comparisons,
    _ALLOWED_OVERALL_STATUSES,
    _ALLOWED_PAPER_STYLE_VS_HISTORY,
    _ALLOWED_REINTEGRATION_STATUSES,
    _NO_CLAIM,
)

# ---------------------------------------------------------------------------
# Paths to YAML configs
# ---------------------------------------------------------------------------
_CONFIG_DIR = REPO_ROOT / "validation" / "published_continuation_comparison"
_KUZNETSOV_YAML = _CONFIG_DIR / "kuznetsov2017_chua_integer.yaml"
_DANCA_YAML = _CONFIG_DIR / "danca2017_chua_fractional_saturation.yaml"
_WU_YAML = _CONFIG_DIR / "wu2023_chua_fractional_arctan.yaml"


# ===========================================================================
# 1. test_configs_exist
# ===========================================================================

class TestConfigsExist:
    """Verify that all three required YAML config files exist."""

    def test_kuznetsov_exists(self) -> None:
        assert _KUZNETSOV_YAML.exists(), f"Missing: {_KUZNETSOV_YAML}"

    def test_danca_exists(self) -> None:
        assert _DANCA_YAML.exists(), f"Missing: {_DANCA_YAML}"

    def test_wu_exists(self) -> None:
        assert _WU_YAML.exists(), f"Missing: {_WU_YAML}"


# ===========================================================================
# 2. test_no_hidden_or_chaos_claims
# ===========================================================================

class TestNoHiddenOrChaosClaims:
    """All YAML configs must have no_claims correctly set to false/true."""

    @pytest.mark.parametrize("yaml_path", [_KUZNETSOV_YAML, _DANCA_YAML, _WU_YAML])
    def test_yaml_no_claims(self, yaml_path: Path) -> None:
        config = load_published_continuation_config(yaml_path)
        nc = config["no_claims"]
        assert nc["hiddenness_certified_by_this_pipeline"] is False
        assert nc["chaos_certified_by_this_pipeline"] is False
        assert nc["no_hidden_verified_claim"] is True

    @pytest.mark.parametrize("yaml_path", [_KUZNETSOV_YAML, _DANCA_YAML, _WU_YAML])
    def test_no_claim_module_constant(self, yaml_path: Path) -> None:
        """The _NO_CLAIM constant must always have correct values."""
        assert _NO_CLAIM["hiddenness_certified_by_this_pipeline"] is False
        assert _NO_CLAIM["chaos_certified_by_this_pipeline"] is False
        assert _NO_CLAIM["no_hidden_verified_claim"] is True

    @pytest.mark.parametrize("yaml_path", [_KUZNETSOV_YAML, _DANCA_YAML, _WU_YAML])
    def test_comparison_policy_no_pointwise(self, yaml_path: Path) -> None:
        config = load_published_continuation_config(yaml_path)
        policy = config.get("comparison_policy", {})
        assert policy.get("pointwise_comparison_used", False) is False


# ===========================================================================
# 3. test_article_scope_classification
# ===========================================================================

class TestArticleScopeClassification:
    """Test classify_article_continuation_scope for each case."""

    def test_kuznetsov_scope(self) -> None:
        config = load_published_continuation_config(_KUZNETSOV_YAML)
        scope = classify_article_continuation_scope(config)
        assert scope["paper_does_not_report_continuation"] is True
        assert scope["paper_reports_continuation"] is False

    def test_danca_scope_no_continuation(self) -> None:
        config = load_published_continuation_config(_DANCA_YAML)
        scope = classify_article_continuation_scope(config)
        assert scope["paper_does_not_report_continuation"] is True
        assert scope["paper_reports_continuation"] is False

    def test_wu_scope_no_continuation(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        scope = classify_article_continuation_scope(config)
        assert scope["paper_does_not_report_continuation"] is True
        assert scope["paper_reports_continuation"] is False

    def test_danca_scope_no_memory_transport(self) -> None:
        config = load_published_continuation_config(_DANCA_YAML)
        scope = classify_article_continuation_scope(config)
        assert scope["paper_does_not_report_memory_transport"] is True
        assert scope["paper_reports_memory_transport"] is False

    def test_wu_scope_no_memory_transport(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        scope = classify_article_continuation_scope(config)
        assert scope["paper_does_not_report_memory_transport"] is True

    def test_danca_data_missing(self) -> None:
        config = load_published_continuation_config(_DANCA_YAML)
        scope = classify_article_continuation_scope(config)
        assert scope["paper_data_missing"] is True

    def test_wu_data_not_missing(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        scope = classify_article_continuation_scope(config)
        # Wu has x0_plus and x0_minus in initial_conditions
        assert scope["paper_data_missing"] is False


# ===========================================================================
# 4. test_no_paper_continuation_reproduced_when_not_reported
# ===========================================================================

class TestNoPaperContinuationReproducedWhenNotReported:
    """When paper.reports_continuation is false, overall_status must never
    be published_continuation_reproduced."""

    def test_danca_cannot_reproduce_continuation(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_DANCA_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["overall_status"] != "published_continuation_reproduced"

    def test_wu_cannot_reproduce_continuation(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_WU_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["overall_status"] != "published_continuation_reproduced"

    def test_kuznetsov_cannot_reproduce_continuation(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_KUZNETSOV_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["overall_status"] != "published_continuation_reproduced"

    def test_compare_to_published_with_false_continuation_never_reproduces(self) -> None:
        """compare_to_published_result must never return paper_continuation_reproduced
        when reports_continuation is false."""
        config = {
            "paper": {"reports_continuation": False},
            "published_data": {},
        }
        run_result = {
            "ic_source": "paper_reported_initial_condition",
            "dynamic_class": "bounded_nontrivial",
            "bounded_nontrivial": True,
            "int_status": "ok",
        }
        status = compare_to_published_result(config, run_result)
        assert status != "paper_continuation_reproduced"
        assert status in _ALLOWED_REINTEGRATION_STATUSES


# ===========================================================================
# 5. test_wu_k_null_does_not_invent_continuation
# ===========================================================================

class TestWuKNullDoesNotInventContinuation:
    """Wu 2023 has k=null; deformed Lure must not be constructed."""

    def test_wu_k_is_null(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        k = config["published_data"].get("k")
        assert k is None

    def test_wu_deformed_lure_disabled(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        modes = config["comparison_modes"]
        assert modes.get("deformed_lure_continuation", False) is False

    def test_wu_original_system_comparison_enabled(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        modes = config["comparison_modes"]
        assert modes.get("original_system_strategy_comparison", False) is True

    def test_wu_no_omega0_no_a0(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        pub = config["published_data"]
        assert pub.get("omega0") is None
        assert pub.get("a0") is None

    def test_wu_summary_does_not_invent_k(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_WU_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        # Should not be the "full" continuation (needs k)
        assert "published_continuation_reproduced" != summary["overall_status"]
        # Status must be in allowed set
        assert summary["overall_status"] in _ALLOWED_OVERALL_STATUSES


# ===========================================================================
# 6. test_kuznetsov_q1_does_not_use_history_transport
# ===========================================================================

class TestKuznetsovQ1NoHistoryTransport:
    """Kuznetsov q=1: history_window_transport must not be applied."""

    def test_kuznetsov_q_is_one(self) -> None:
        config = load_published_continuation_config(_KUZNETSOV_YAML)
        q = float(config["dynamics"]["q"])
        assert q == 1.0

    def test_kuznetsov_history_transport_disabled(self) -> None:
        config = load_published_continuation_config(_KUZNETSOV_YAML)
        modes = config["comparison_modes"]
        assert modes.get("caputo_aware_history_window_transport", False) is False

    def test_kuznetsov_derivative_is_integer(self) -> None:
        config = load_published_continuation_config(_KUZNETSOV_YAML)
        assert config["dynamics"]["derivative"] == "integer"

    def test_kuznetsov_summary_no_history_strategy(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_KUZNETSOV_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        # Caputo-aware strategy must be not_applicable for q=1
        assert summary["caputo_aware_strategy"] == "not_applicable"


# ===========================================================================
# 7. test_resolve_initial_conditions_wu
# ===========================================================================

class TestResolveInitialConditionsWu:
    """Wu 2023 ICs must resolve to paper_reported_initial_condition."""

    def test_wu_ics_resolved(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        ics = resolve_initial_conditions(config)
        sources = {ic["ic_id"]: ic["source"] for ic in ics}
        assert "x0_plus" in sources
        assert "x0_minus" in sources
        assert sources["x0_plus"] == "paper_reported_initial_condition"
        assert sources["x0_minus"] == "paper_reported_initial_condition"

    def test_wu_x0_plus_values(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        ics = resolve_initial_conditions(config)
        ic_plus = next(ic for ic in ics if ic["ic_id"] == "x0_plus")
        assert ic_plus["x0"] == [13.8, 0.7093, -19.8768]

    def test_wu_x0_minus_values(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        ics = resolve_initial_conditions(config)
        ic_minus = next(ic for ic in ics if ic["ic_id"] == "x0_minus")
        assert ic_minus["x0"] == [-13.8, -0.7093, 19.8768]

    def test_wu_no_ic_source_is_missing(self) -> None:
        config = load_published_continuation_config(_WU_YAML)
        ics = resolve_initial_conditions(config)
        for ic in ics:
            if ic["x0"] is not None:
                assert ic["source"] != "missing"


# ===========================================================================
# 8. test_resolve_initial_conditions_danca_missing
# ===========================================================================

class TestResolveInitialConditionsDancaMissing:
    """Danca 2017 has no published IC or seed; must resolve to missing."""

    def test_danca_ics_missing(self) -> None:
        config = load_published_continuation_config(_DANCA_YAML)
        ics = resolve_initial_conditions(config)
        for ic in ics:
            assert ic["source"] in ("missing", "library_reconstructed_seed")
            # Must not claim paper_reported_initial_condition
            assert ic["source"] != "paper_reported_initial_condition"

    def test_danca_x0_is_none(self) -> None:
        config = load_published_continuation_config(_DANCA_YAML)
        ics = resolve_initial_conditions(config)
        # All should have x0=None or be flagged as missing
        for ic in ics:
            if ic["source"] == "missing":
                assert ic["x0"] is None


# ===========================================================================
# 9. test_compare_paper_style_vs_history_warning
# ===========================================================================

class TestComparePaperStyleVsHistoryWarning:
    """Test that large final_state_relative_distance triggers proper status."""

    def _make_result(
        self,
        dyn_class: str = "bounded_nontrivial",
        fs_x: float = 1.0, fs_y: float = 1.0, fs_z: float = 1.0,
        rho: float = 2.0,
        rx: float = 10.0, ry: float = 10.0, rz: float = 10.0,
    ) -> Dict[str, Any]:
        return {
            "dynamic_class": dyn_class,
            "final_state_x": fs_x,
            "final_state_y": fs_y,
            "final_state_z": fs_z,
            "rho_attractor": rho,
            "range_x": rx, "range_y": ry, "range_z": rz,
            "availability": "available",
        }

    def _policy_config(self) -> Dict[str, Any]:
        return {
            "comparison_policy": {
                "pointwise_comparison_used": False,
                "final_state_relative_tolerance": 0.50,
                "rho_relative_tolerance": 0.35,
                "range_relative_tolerance": 0.35,
            }
        }

    def test_large_final_state_distance_triggers_differs(self) -> None:
        config = self._policy_config()
        paper_result = self._make_result(fs_x=1.0, fs_y=1.0, fs_z=1.0, rho=2.0)
        hist_result = self._make_result(fs_x=10.0, fs_y=10.0, fs_z=10.0, rho=2.0)

        status, comp = compare_paper_style_vs_history(config, paper_result, hist_result)
        assert status in (
            "paper_style_differs_from_history",
            "paper_style_restart_artifact_possible",
            "paper_style_result_differs_from_caputo_history_transport",
        )
        assert comp["warning"] is True

    def test_class_change_triggers_differs(self) -> None:
        config = self._policy_config()
        paper_result = self._make_result(dyn_class="collapsed_to_equilibrium", rho=0.0)
        hist_result = self._make_result(dyn_class="bounded_nontrivial", rho=2.0)

        status, comp = compare_paper_style_vs_history(config, paper_result, hist_result)
        assert status in (
            "paper_style_differs_from_history",
            "paper_style_restart_artifact_possible",
        )
        assert comp["class_changed"] is True

    def test_none_result_returns_not_applicable(self) -> None:
        config = self._policy_config()
        status, comp = compare_paper_style_vs_history(config, None, None)
        assert status == "comparison_not_applicable"
        assert comp == {}

    def test_consistent_results(self) -> None:
        config = self._policy_config()
        paper_result = self._make_result(fs_x=1.0, fs_y=1.0, fs_z=1.0, rho=2.0)
        hist_result = self._make_result(fs_x=1.01, fs_y=1.01, fs_z=1.01, rho=2.01)

        status, comp = compare_paper_style_vs_history(config, paper_result, hist_result)
        assert status == "paper_style_and_history_consistent"
        assert comp["warning"] is False


# ===========================================================================
# 10. test_run_wu_fast_smoke
# ===========================================================================

class TestRunWuFastSmoke:
    """Smoke test for Wu 2023 in fast mode."""

    def test_wu_fast_output_exists(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_WU_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        case_id = summary["case_id"]
        case_out = tmp_path / case_id
        assert (case_out / "published_continuation_summary.json").exists()
        assert (case_out / "paper_style_runs.csv").exists()
        assert (case_out / "paper_style_vs_history.csv").exists()
        assert (case_out / "published_data_missing.json").exists()

    def test_wu_fast_overall_status_allowed(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_WU_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["overall_status"] in _ALLOWED_OVERALL_STATUSES

    def test_wu_fast_no_hidden_certified(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_WU_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["hiddenness_certified_by_this_pipeline"] is False
        assert summary["chaos_certified_by_this_pipeline"] is False
        assert summary["no_hidden_verified_claim"] is True
        assert "hidden_verified" not in summary
        assert "chaos_verified" not in summary

    def test_wu_fast_summary_json_no_claims(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_WU_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        json_path = tmp_path / summary["case_id"] / "published_continuation_summary.json"
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "hidden_verified" not in data
        assert "chaos_verified" not in data
        assert data["hiddenness_certified_by_this_pipeline"] is False
        assert data["chaos_certified_by_this_pipeline"] is False
        assert data["no_hidden_verified_claim"] is True

    def test_wu_fast_not_published_continuation_reproduced(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_WU_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["overall_status"] != "published_continuation_reproduced"
        assert summary["paper_reports_continuation"] is False


# ===========================================================================
# 11. test_run_saturation_fast_smoke
# ===========================================================================

class TestRunSaturationFastSmoke:
    """Smoke test for Danca 2017 (saturation) in fast mode."""

    def test_danca_fast_output_exists(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_DANCA_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        case_id = summary["case_id"]
        case_out = tmp_path / case_id
        assert (case_out / "published_continuation_summary.json").exists()
        assert (case_out / "published_data_missing.json").exists()

    def test_danca_fast_is_data_missing_or_not_reported(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_DANCA_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        # Danca has no IC; status must be data_missing or continuation_not_reported
        allowed = {
            "published_data_missing",
            "published_continuation_not_reported",
            "published_comparison_inconclusive",
        }
        assert summary["overall_status"] in allowed

    def test_danca_fast_no_hidden_certified(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_DANCA_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["hiddenness_certified_by_this_pipeline"] is False
        assert summary["chaos_certified_by_this_pipeline"] is False
        assert summary["no_hidden_verified_claim"] is True

    def test_danca_fast_not_reproduced(self, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=_DANCA_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["overall_status"] != "published_continuation_reproduced"


# ===========================================================================
# 12. test_summary_schema
# ===========================================================================

class TestSummarySchema:
    """Verify that published_continuation_summary.json has required fields."""

    REQUIRED_FIELDS = [
        "stage",
        "case_id",
        "reference_id",
        "system_id",
        "paper_reports_continuation",
        "paper_reports_memory_transport",
        "paper_style_strategy",
        "caputo_aware_strategy",
        "overall_status",
        "published_data_status",
        "paper_style_reintegration_status",
        "paper_style_vs_history_status",
        "dynamic_class_detected",
        "chaotic_dynamics_candidate_detected",
        "hiddenness_certified_by_this_pipeline",
        "chaos_certified_by_this_pipeline",
        "no_hidden_verified_claim",
        "pointwise_comparison_used",
    ]

    @pytest.mark.parametrize("yaml_path", [_KUZNETSOV_YAML, _DANCA_YAML, _WU_YAML])
    def test_summary_has_required_fields(self, yaml_path: Path, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=yaml_path,
            output_dir=tmp_path,
            fast=True,
        )
        for field in self.REQUIRED_FIELDS:
            assert field in summary, f"Missing field '{field}' in summary for {yaml_path.name}"

    @pytest.mark.parametrize("yaml_path", [_KUZNETSOV_YAML, _DANCA_YAML, _WU_YAML])
    def test_summary_json_has_required_fields(self, yaml_path: Path, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=yaml_path,
            output_dir=tmp_path,
            fast=True,
        )
        json_path = tmp_path / summary["case_id"] / "published_continuation_summary.json"
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"JSON missing field '{field}' for {yaml_path.name}"

    @pytest.mark.parametrize("yaml_path", [_KUZNETSOV_YAML, _DANCA_YAML, _WU_YAML])
    def test_summary_stage_field(self, yaml_path: Path, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=yaml_path,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["stage"] == "published_continuation_comparison"

    @pytest.mark.parametrize("yaml_path", [_KUZNETSOV_YAML, _DANCA_YAML, _WU_YAML])
    def test_summary_overall_status_in_allowed(self, yaml_path: Path, tmp_path: Path) -> None:
        summary = run_published_continuation_case(
            config_path=yaml_path,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["overall_status"] in _ALLOWED_OVERALL_STATUSES, (
            f"unexpected overall_status='{summary['overall_status']}' "
            f"for {yaml_path.name}"
        )
