"""Validation state definitions for the chaotic attractor localization workflow."""

from __future__ import annotations

from enum import Enum


class AttractorValidationState(str, Enum):
    """Enumeration of strict validation pipeline states."""

    CANDIDATE = "candidate"
    HIDDEN_UNDER_TESTED_NEIGHBORHOODS = "hidden_under_tested_neighborhoods"
    COMPATIBLE_WITH_HIDDENNESS = "compatible_with_hiddenness"
    SELF_EXCITED = "self_excited"
    NONCHAOTIC = "nonchaotic"
    DIVERGED = "diverged"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"
    NOT_TESTED = "not_tested"
