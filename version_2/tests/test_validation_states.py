"""Unit tests for validation states and global report coherence checks."""

from __future__ import annotations

import pytest

from hidden_attractors.validation.states import AttractorValidationState
from hidden_attractors.workflows.protocol import validate_global_report_coherence


def test_validation_state_enum():
    """Verify that AttractorValidationState enum contains all required states."""
    assert AttractorValidationState.CANDIDATE == "candidate"
    assert AttractorValidationState.HIDDEN_UNDER_TESTED_NEIGHBORHOODS == "hidden_under_tested_neighborhoods"
    assert AttractorValidationState.COMPATIBLE_WITH_HIDDENNESS == "compatible_with_hiddenness"
    assert AttractorValidationState.SELF_EXCITED == "self_excited"
    assert AttractorValidationState.NONCHAOTIC == "nonchaotic"
    assert AttractorValidationState.DIVERGED == "diverged"
    assert AttractorValidationState.INCONCLUSIVE == "inconclusive"
    assert AttractorValidationState.REJECTED == "rejected"
    assert AttractorValidationState.NOT_TESTED == "not_tested"


def test_coherence_seed_found_not_hidden():
    """Validate that seed_found cannot be labeled as hidden."""
    report = {
        "state": "seed_found",
        "verdict": "hidden_verified",
    }
    with pytest.raises(ValueError, match="cannot be labeled as hidden"):
        validate_global_report_coherence(report)


def test_coherence_chaotic_candidate_requires_evidence():
    """Validate that chaotic_candidate requires chaos evidence."""
    report = {
        "state": "chaotic_candidate",
        "metrics": {},
    }
    with pytest.raises(ValueError, match="requires chaos test evidence"):
        validate_global_report_coherence(report)

    # Adding evidence makes it pass
    report["metrics"]["zero_one_kappa"] = 0.95
    validate_global_report_coherence(report)


def test_coherence_hidden_verified_requires_evidence(valid_run_metadata):
    """Validate that hidden_verified requires metadata plus sphere and basin evidence."""
    report = {
        "state": "hidden_verified",
        "stage_statuses": {"hiddenness_tests": "incomplete"},
    }
    with pytest.raises(ValueError, match="requires complete reproducibility metadata"):
        validate_global_report_coherence(report)

    report["run_metadata"] = valid_run_metadata
    with pytest.raises(ValueError, match="requires completed sphere and basin-neighborhood evidence"):
        validate_global_report_coherence(report)

    # Adding evidence makes it pass
    report["evidence"] = {"completed_sphere_tests": True, "completed_basin_tests": True}
    validate_global_report_coherence(report)
