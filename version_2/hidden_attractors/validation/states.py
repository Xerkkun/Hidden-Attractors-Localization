"""Validation state definitions for the chaotic attractor localization workflow."""

from __future__ import annotations

from enum import Enum


class AttractorValidationState(str, Enum):
    """Enumeration of strict validation pipeline states."""

    SEED_FOUND = "seed_found"
    CANDIDATE_ATTRACTOR = "candidate_attractor"
    CHAOTIC_CANDIDATE = "chaotic_candidate"
    HIDDEN_COMPATIBLE = "hidden_compatible"
    HIDDEN_VERIFIED = "hidden_verified"
    REJECTED = "rejected"
    FAILED_NUMERICALLY = "failed_numerically"
