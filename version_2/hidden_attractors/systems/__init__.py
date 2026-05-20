"""Extensible chaotic-system registry.

Users can register new systems with :func:`register_system` and then reuse the
same analysis and workflow-discovery entry points used by built-in systems.
"""

from .base import ChaoticSystem, SystemRegistry, get_system, list_systems, register_system
from .builtins import register_builtin_systems
from .lure import LureSystem

register_builtin_systems()

__all__ = [
    "ChaoticSystem",
    "LureSystem",
    "SystemRegistry",
    "get_system",
    "list_systems",
    "register_builtin_systems",
    "register_system",
]
