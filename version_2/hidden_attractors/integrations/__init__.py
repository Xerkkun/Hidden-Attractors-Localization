"""Optional integrations with external dynamical-systems libraries."""

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
