"""Extensible chaotic-system registry.

Stability: stable
    System dataclasses, registry API (``register_system``, ``get_system``,
    ``list_systems``), and capability checks.  Signatures are fixed.

Users can register new systems with :func:`register_system` and then reuse the
same analysis and workflow-discovery entry points used by built-in systems.
"""

from .base import ChaoticSystem, SystemRegistry, get_system, list_systems, register_system
from .builtins import register_builtin_systems
from .lure import LureSystem
from .requirements import CapabilityReport, Requirement, check_system_capability, known_workflows, requirements_for
from .fischer_benchmarks import FISCHER_BENCHMARKS, get_fischer_benchmark

register_builtin_systems()

__all__ = [
    "CapabilityReport",
    "ChaoticSystem",
    "LureSystem",
    "Requirement",
    "SystemRegistry",
    "check_system_capability",
    "get_system",
    "known_workflows",
    "list_systems",
    "register_builtin_systems",
    "register_system",
    "requirements_for",
    "FISCHER_BENCHMARKS",
    "get_fischer_benchmark",
]
