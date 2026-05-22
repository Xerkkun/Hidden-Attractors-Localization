"""Optional integrations with external dynamical-systems libraries.

Stability: experimental
    Adapters for ``nolds``, ``antropy``, and similar optional backends.  The
    adapter interface may change as new tools are integrated.
"""

from .external_tools import (
    EXTERNAL_TOOLS,
    ExternalTool,
    available_complexity_backends,
    compute_complexity_measures,
    external_tool_report,
    require_external,
)

__all__ = [
    "EXTERNAL_TOOLS",
    "ExternalTool",
    "available_complexity_backends",
    "compute_complexity_measures",
    "external_tool_report",
    "require_external",
]
