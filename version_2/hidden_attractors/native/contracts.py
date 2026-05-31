"""Backend contracts for reusable native numerical engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class BackendBuildSpec:
    """Native build inputs for a system-specific backend."""

    source_files: tuple[Path, ...]
    output_path: Path
    compiler_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegrationRequest:
    """Common integration request for integer and fractional backends."""

    x0: np.ndarray
    t_final: float
    h: float
    q: float = 1.0
    memory_length: float = 1.0
    parameters: Mapping[str, float] | None = None


@dataclass(frozen=True)
class IntegrationResult:
    """Common integration result returned by backend adapters."""

    trajectory: np.ndarray
    status: str
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class FractionalLyapunovRequest:
    """Native-only request for extensive fractional Lyapunov calculations."""

    system_id: str
    x0: np.ndarray
    parameters: Mapping[str, float]
    q: float
    h: float
    t_final: float
    reorthonormalization_time: float
    t_burn: float = 0.0
    execution_contract: str = "fixed_lower_limit_full_history_qr"
    convolution_mode: str = "fft_block"
    fft_block_size: int = 256
    divergence_norm: float = 0.0
    convergence_csv: Path | None = None


@dataclass(frozen=True)
class FractionalLyapunovResult:
    """Spectrum and provenance returned by the native fractional backend."""

    exponents: np.ndarray
    final_state: np.ndarray
    times: np.ndarray
    convergence: np.ndarray
    status: str
    steps_completed: int
    execution_contract: str
    convolution_mode: str
    metadata: Mapping[str, object]


class NativeIntegrationBackend(Protocol):
    """Minimal protocol expected from reusable native integration backends."""

    def integrate(self, request: IntegrationRequest) -> IntegrationResult:
        """Run one integration request and return stored trajectory samples."""


class NativeLyapunovBackend(Protocol):
    """Protocol for backend Lyapunov estimators."""

    def lyapunov(self, request: IntegrationRequest, *, t_burn: float, blocks: int) -> Sequence[float]:
        """Return Lyapunov exponent estimates for the requested system."""


__all__ = [
    "BackendBuildSpec",
    "IntegrationRequest",
    "IntegrationResult",
    "FractionalLyapunovRequest",
    "FractionalLyapunovResult",
    "NativeIntegrationBackend",
    "NativeLyapunovBackend",
]
