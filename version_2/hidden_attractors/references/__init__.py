"""Bibliographic validation and traceability package."""

from .registry import REFERENCE_REGISTRY
from .claims import ClaimType, CLAIM_REFERENCE_MATRIX
from .validator import validate_claim_references, write_traceability_matrix_markdown, validate_bibliography_manifest

__all__ = [
    "REFERENCE_REGISTRY",
    "ClaimType",
    "CLAIM_REFERENCE_MATRIX",
    "validate_claim_references",
    "write_traceability_matrix_markdown",
    "validate_bibliography_manifest",
]
