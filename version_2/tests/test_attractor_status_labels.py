"""Tests verifying the unified canonical attractor status labels and normalization."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from hidden_attractors.verification.status_labels import (
    CANONICAL_ATTRACTOR_STATUS,
    LEGACY_STATUS_MAP,
    normalize_attractor_status,
)
from hidden_attractors.verification.candidate_gate import evaluate_candidate_gate
from hidden_attractors.workflows.protocol import StageEnvelope

ROOT = Path(__file__).resolve().parents[1]


def test_normalization():
    """Verify that normalize_attractor_status correctly maps legacy labels."""
    # Canonical ones should remain unchanged
    for status in CANONICAL_ATTRACTOR_STATUS:
        assert normalize_attractor_status(status) == status

    # Legacy ones should map to canonical ones
    for legacy, canonical in LEGACY_STATUS_MAP.items():
        assert normalize_attractor_status(legacy) == canonical
        assert normalize_attractor_status(legacy.upper()) == canonical

    # Unknown labels remain unchanged or get mapped
    assert normalize_attractor_status("unknown_status") == "unknown_status"


def test_no_legacy_labels_in_canonical():
    """Ensure no legacy labels are in the canonical set."""
    legacy_labels = {
        "hidden_verified",
        "chaos_verified",
        "hiddenness_supported_under_tested_neighborhoods",
        "compatible_with_hiddenness_under_tested_radii",
        "self_excited_contact_detected",
        "candidate_rejected",
        "seed_found",
    }
    intersection = CANONICAL_ATTRACTOR_STATUS.intersection(legacy_labels)
    assert not intersection, f"Legacy labels found in canonical set: {intersection}"


def test_candidate_gate_outputs():
    """Verify that evaluate_candidate_gate does not return legacy labels as attractor_status."""
    # A dummy minimal evidence dict that should evaluate to inconclusive
    evidence = {
        "equilibria": {"all_found": False},
        "hiddenness": {"numerical_failures": 0},
    }
    result = evaluate_candidate_gate(evidence)
    assert "attractor_status" in result
    status = result["attractor_status"]
    assert status in CANONICAL_ATTRACTOR_STATUS
    assert status != "chaos_verified"


def test_stage_envelope_normalization():
    """Verify StageEnvelope normalizes verdict, state, and attractor_status."""
    envelope = StageEnvelope(
        stage="seed_generation",
        status="completed",
        system="test_system",
        numerical_contract={},
        verdict="seed_found",
        state="seed_found",
    )
    assert envelope.verdict == "candidate"
    assert envelope.state == "candidate"
    assert envelope.attractor_status == "candidate"

    envelope_hidden = StageEnvelope(
        stage="hiddenness_tests",
        status="completed",
        system="test_system",
        numerical_contract={},
        verdict="hidden_verified",
    )
    assert envelope_hidden.verdict == "hidden_under_tested_neighborhoods"
    assert envelope_hidden.attractor_status == "hidden_under_tested_neighborhoods"


def test_documentation_has_no_legacy_section():
    """Verify that documentation files do not contain a 'Current vs legacy labels' section."""
    docs_to_check = [
        ROOT / "docs" / "freeze_status.md",
        ROOT / "docs" / "scientific_scope.md",
    ]
    for doc in docs_to_check:
        if doc.exists():
            content = doc.read_text(encoding="utf-8")
            assert "Current vs legacy labels" not in content
            assert "Legacy labels" not in content
