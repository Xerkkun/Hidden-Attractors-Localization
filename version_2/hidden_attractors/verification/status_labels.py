"""Canonical attractor classification status labels and normalization."""

from __future__ import annotations

from typing import Any

CANONICAL_ATTRACTOR_STATUS = {
    "candidate",
    "hidden_under_tested_neighborhoods",
    "compatible_with_hiddenness",
    "self_excited",
    "nonchaotic",
    "diverged",
    "inconclusive",
    "rejected",
    "not_tested",
}

LEGACY_STATUS_MAP = {
    "hidden_verified": "hidden_under_tested_neighborhoods",
    "hiddenness_supported_under_tested_neighborhoods": "hidden_under_tested_neighborhoods",
    "compatible_with_hiddenness_under_tested_radii": "compatible_with_hiddenness",
    "self_excited_contact_detected": "self_excited",
    "candidate_rejected": "rejected",
    "hiddenness_inconclusive": "inconclusive",
    "chaos_verified": "candidate",
    "rejected_self_excited_contact": "self_excited",
    "compatible_in_all_tested_solver_memory_cases": "compatible_with_hiddenness",
    "hidden_verified_only_if_full_protocol_passed": "hidden_under_tested_neighborhoods",
    "numerical_failure": "inconclusive",
    "candidate_not_reproducible": "inconclusive",
    "seed_found": "candidate",
    "candidate_attractor": "candidate",
    "chaotic_candidate": "candidate",
    "hidden_compatible": "compatible_with_hiddenness",
    "failed_numerically": "inconclusive",
    "not_run": "not_tested",
}

def normalize_attractor_status(label: str) -> str:
    """Normalize old status labels to canonical ones."""
    if not isinstance(label, str):
        label = str(label)
    normalized = label.strip().lower()
    if normalized in CANONICAL_ATTRACTOR_STATUS:
        return normalized
    return LEGACY_STATUS_MAP.get(normalized, normalized)
