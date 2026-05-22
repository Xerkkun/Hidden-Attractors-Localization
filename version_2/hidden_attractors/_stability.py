"""Stability-tier constants and decorator for the hidden-attractors-fo API.

Stability: internal
    This module is consumed by package __init__ files and docstrings.
    It does not belong to any user-facing tier itself.

Usage
-----
Module authors annotate their public symbols with the decorator::

    from hidden_attractors._stability import api_tier, STABLE

    @api_tier(STABLE)
    class ChuaParameters: ...

Consumers can introspect any symbol::

    import hidden_attractors.models as m
    m.ChuaParameters.__api_tier__   # -> 'stable'

The four tier strings are also re-exported from the top-level package so
users can write ``hidden_attractors.STABLE`` without importing this module.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

__all__ = [
    "STABLE",
    "EXPERIMENTAL",
    "INTERNAL",
    "LEGACY",
    "api_tier",
    "assert_tier",
    "get_tier",
]

# ── Tier constants ──────────────────────────────────────────────────────────

STABLE: str = "stable"
"""Stable tier: signatures are fixed; breaking changes require a version bump
and a deprecation cycle."""

EXPERIMENTAL: str = "experimental"
"""Experimental tier: API is useful and tested but may evolve.
Changes will be noted in a changelog entry."""

INTERNAL: str = "internal"
"""Internal tier: consumed by workflows and backends; not part of the
user-facing surface.  May change without notice."""

LEGACY: str = "legacy"
"""Legacy tier: frozen compatibility facade over historical scripts.
No new features will be added.  Will not be ported to new APIs."""

_VALID_TIERS = frozenset({STABLE, EXPERIMENTAL, INTERNAL, LEGACY})

_F = TypeVar("_F", bound=Any)


# ── Decorator ───────────────────────────────────────────────────────────────

def api_tier(tier: str) -> Callable[[_F], _F]:
    """Stamp ``__api_tier__`` on a callable or class.

    Parameters
    ----------
    tier : str
        One of :data:`STABLE`, :data:`EXPERIMENTAL`, :data:`INTERNAL`,
        or :data:`LEGACY`.

    Returns
    -------
    decorator : callable
        A no-op decorator that sets ``obj.__api_tier__ = tier`` and
        returns *obj* unchanged.

    Raises
    ------
    ValueError
        If *tier* is not one of the four recognised tier strings.

    Examples
    --------
    >>> from hidden_attractors._stability import api_tier, STABLE
    >>> @api_tier(STABLE)
    ... def my_function(): pass
    >>> my_function.__api_tier__
    'stable'
    """
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"Unknown tier {tier!r}.  Must be one of: {sorted(_VALID_TIERS)}"
        )

    def decorator(obj: _F) -> _F:
        obj.__api_tier__ = tier  # type: ignore[attr-defined]
        return obj

    return decorator


# ── Introspection helpers ───────────────────────────────────────────────────

def get_tier(obj: Any) -> str | None:
    """Return the tier string stamped on *obj*, or ``None`` if absent.

    Parameters
    ----------
    obj : Any
        Any Python object, typically a function or class decorated with
        :func:`api_tier`.

    Returns
    -------
    tier : str or None
        One of ``'stable'``, ``'experimental'``, ``'internal'``,
        ``'legacy'``, or ``None`` if no tier has been stamped.

    Examples
    --------
    >>> from hidden_attractors._stability import get_tier, api_tier, STABLE
    >>> @api_tier(STABLE)
    ... def f(): pass
    >>> get_tier(f)
    'stable'
    >>> get_tier(len) is None
    True
    """
    return getattr(obj, "__api_tier__", None)


def assert_tier(obj: Any, expected: str) -> None:
    """Raise :exc:`AssertionError` if *obj* does not carry the expected tier.

    Intended for tests that pin a symbol's stability contract.

    Parameters
    ----------
    obj : Any
        Object to inspect; must have been decorated with :func:`api_tier`.
    expected : str
        The tier string that *obj* is expected to have.

    Raises
    ------
    AssertionError
        If ``obj.__api_tier__ != expected`` or if ``__api_tier__`` is absent.

    Examples
    --------
    >>> from hidden_attractors._stability import assert_tier, STABLE, api_tier
    >>> @api_tier(STABLE)
    ... def f(): pass
    >>> assert_tier(f, STABLE)  # passes silently
    """
    actual = get_tier(obj)
    if actual != expected:
        name = getattr(obj, "__name__", repr(obj))
        raise AssertionError(
            f"{name!r} has tier {actual!r}, expected {expected!r}."
        )
