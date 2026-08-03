"""Internal, signature-safe right-hand-side binding helpers.

The numerical kernels support a small compatibility surface for vector fields,
but callback exceptions must never be used to guess which signature was meant.
This module inspects the callable once, binds the most informative supported
signature, and then lets exceptions raised by the vector field propagate
unchanged.
"""

from __future__ import annotations

from inspect import Signature, signature
from typing import Any, Callable


_DUMMY_TIME = object()
_DUMMY_STATE = object()
_DUMMY_PARAMETERS = object()


def _accepts_positional(signature_value: Signature, *arguments: object) -> bool:
    try:
        signature_value.bind(*arguments)
    except TypeError:
        return False
    return True


def bind_rhs(
    rhs: Callable[..., Any],
    parameters: Any = None,
) -> Callable[[float, Any], Any]:
    """Bind a supported RHS signature without executing the callback.

    Supported conventions, in deterministic priority order, are
    ``rhs(t, state, parameters)``, ``rhs(t, state)``, and ``rhs(state)``.
    The three-argument form is selected only when ``parameters`` is not
    ``None``.  A two-argument callable always means ``rhs(t, state)``; the
    ambiguous legacy interpretation ``rhs(state, parameters)`` is deliberately
    not guessed.  Users of that convention can wrap it explicitly in a
    canonical time-aware callable.

    Signature inspection happens once.  A ``TypeError`` raised later by the
    body of ``rhs`` is therefore a solver callback error, not an arity signal.
    """

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
    try:
        rhs_signature = signature(rhs)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "rhs must expose an inspectable positional signature; wrap opaque "
            "callables as rhs(time, state) or rhs(time, state, parameters)."
        ) from exc

    if parameters is not None and _accepts_positional(
        rhs_signature,
        _DUMMY_TIME,
        _DUMMY_STATE,
        _DUMMY_PARAMETERS,
    ):

        def bound(time: float, state: Any) -> Any:
            return rhs(time, state, parameters)

        mode = "time_state_parameters"
    elif _accepts_positional(rhs_signature, _DUMMY_TIME, _DUMMY_STATE):

        def bound(time: float, state: Any) -> Any:
            return rhs(time, state)

        mode = "time_state"
    elif _accepts_positional(rhs_signature, _DUMMY_STATE):

        def bound(time: float, state: Any) -> Any:
            del time
            return rhs(state)

        mode = "state"
    else:
        expected = (
            "rhs(time, state, parameters), rhs(time, state), or rhs(state)"
            if parameters is not None
            else "rhs(time, state) or rhs(state)"
        )
        raise TypeError(f"rhs does not support a recognized signature; expected {expected}.")

    setattr(bound, "__hafo_rhs_signature__", mode)
    return bound


__all__ = ["bind_rhs"]
