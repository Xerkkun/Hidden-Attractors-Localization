"""Contracts for complete hidden-attractor workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from ..systems.base import ChaoticSystem
from .protocol import NumericalContract


@dataclass(frozen=True)
class SeedResult:
    """Candidate seed produced by a describing-function stage."""

    seed: np.ndarray
    method: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuationResult:
    """Output of a numerical continuation stage."""

    trajectory: np.ndarray
    survived: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HiddennessResult:
    """Evidence summary from equilibrium-neighborhood controls."""

    hidden_candidate_allowed: bool
    target_hits_from_equilibria: int
    tested_equilibria: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SeedGenerator(Protocol):
    def __call__(self, system: ChaoticSystem, contract: NumericalContract) -> SeedResult:
        """Generate a classical describing-function seed."""


class MachadoSeedGenerator(Protocol):
    def __call__(self, system: ChaoticSystem, contract: NumericalContract, *, mu: float) -> SeedResult:
        """Generate a Machado-family describing-function seed."""


class ContinuationFunction(Protocol):
    def __call__(self, system: ChaoticSystem, seed: SeedResult, contract: NumericalContract) -> ContinuationResult:
        """Continue a candidate seed under the selected numerical contract."""


class HiddennessVerifier(Protocol):
    def __call__(self, system: ChaoticSystem, candidate: ContinuationResult, contract: NumericalContract) -> HiddennessResult:
        """Run equilibrium-neighborhood hiddenness controls."""


class BasinClassifier(Protocol):
    def __call__(self, system: ChaoticSystem, candidate: ContinuationResult, contract: NumericalContract) -> Mapping[str, Any]:
        """Classify basin evidence or return a documented replacement criterion."""


class ReportWriter(Protocol):
    def __call__(self, system: ChaoticSystem, evidence: Mapping[str, Any], contract: NumericalContract) -> Mapping[str, Any]:
        """Write or return reproducible workflow artifacts."""


@dataclass(frozen=True)
class FullWorkflowContract:
    """Required hooks for a system to run the full analysis protocol."""

    seed_generator: SeedGenerator
    machado_seed_generator: MachadoSeedGenerator
    continuation: ContinuationFunction
    hiddenness_verifier: HiddennessVerifier
    basin_classifier: BasinClassifier
    report_writer: ReportWriter


def validate_full_workflow_system(system: ChaoticSystem, workflow: FullWorkflowContract) -> None:
    """Validate that ``system`` exposes the mandatory full-workflow pieces."""

    if system.lure is None:
        raise ValueError(
            f"{system.name} cannot run the full protocol without a manual Lur'e form "
            "D^q x = A x + b psi(c^T x)."
        )
    if system.equilibria is None:
        raise ValueError(f"{system.name} must provide equilibria for hiddenness controls.")
    required = (
        workflow.seed_generator,
        workflow.machado_seed_generator,
        workflow.continuation,
        workflow.hiddenness_verifier,
        workflow.basin_classifier,
        workflow.report_writer,
    )
    if any(item is None for item in required):
        raise ValueError("full workflow requires seed, Machado, continuation, hiddenness, basin, and report hooks.")


__all__ = [
    "BasinClassifier",
    "ContinuationFunction",
    "ContinuationResult",
    "FullWorkflowContract",
    "HiddennessResult",
    "HiddennessVerifier",
    "MachadoSeedGenerator",
    "NumericalContract",
    "ReportWriter",
    "SeedGenerator",
    "SeedResult",
    "validate_full_workflow_system",
]
