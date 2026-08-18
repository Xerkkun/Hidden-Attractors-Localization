"""Private extraction helpers for dynamical-system order declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _safe_attribute(system: object, name: str) -> Any:
    try:
        return getattr(system, name, None)
    except Exception:  # A best-effort metadata probe must not mask later routes.
        return None


def declared_system_order(system: object) -> Any:
    """Return the first declared order without coercing its representation."""

    for attribute in ("q", "order", "fractional_order"):
        value = _safe_attribute(system, attribute)
        if value is not None:
            return value
    for attribute in ("metadata", "parameters", "params"):
        mapping = _safe_attribute(system, attribute)
        if isinstance(mapping, Mapping) and mapping.get("q") is not None:
            return mapping["q"]
    return None


def infer_system_order(system: object) -> float | None:
    """Best-effort conversion of a declared order to ``float``."""

    for attribute in ("q", "order", "fractional_order"):
        value = _safe_attribute(system, attribute)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError, OverflowError):
                pass
    for attribute in ("metadata", "parameters", "params"):
        mapping = _safe_attribute(system, attribute)
        if isinstance(mapping, Mapping) and mapping.get("q") is not None:
            try:
                return float(mapping["q"])
            except (TypeError, ValueError, OverflowError):
                pass
    return None
