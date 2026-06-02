"""Unit tests for validation states and global report coherence checks."""

from __future__ import annotations

import pytest

from hidden_attractors.validation.states import AttractorValidationState
from hidden_attractors.workflows.protocol import validate_global_report_coherence


def test_validation_state_enum():
    """Verify that AttractorValidationState enum contains all required states."""
    assert AttractorValidationState.SEED_FOUND == "seed_found"
    assert AttractorValidationState.CANDIDATE_ATTRACTOR == "candidate_attractor"
    assert AttractorValidationState.CHAOTIC_CANDIDATE == "chaotic_candidate"
    assert AttractorValidationState.HIDDEN_COMPATIBLE == "hidden_compatible"
    assert AttractorValidationState.HIDDEN_VERIFIED == "hidden_verified"
    assert AttractorValidationState.REJECTED == "rejected"
    assert AttractorValidationState.FAILED_NUMERICALLY == "failed_numerically"


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


def test_coherence_hidden_verified_requires_evidence():
    """Validate that hidden_verified requires sphere or basin evidence."""
    report = {
        "state": "hidden_verified",
        "stage_statuses": {"hiddenness_tests": "incomplete"},
    }
    with pytest.raises(ValueError, match="requires evidence of completed sphere_tests"):
        validate_global_report_coherence(report)

    # Adding evidence makes it pass
    report["stage_statuses"]["hiddenness_tests"] = "completed"
    validate_global_report_coherence(report)
