"""Safe expression-defined dynamical systems.

Stability: experimental

Declarative equations are parsed with :mod:`ast` and evaluated by a small
arithmetic interpreter with an explicit function allowlist. No user-supplied
Python statements, attribute access, imports, or builtins are executed.
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .base import ChaoticSystem, SystemKind


class ExpressionValidationError(ValueError):
    """Raised when a declarative system contains unsafe or invalid syntax."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "cos": np.cos,
    "cosh": np.cosh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "max": max,
    "min": min,
    "sign": np.sign,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _identifier(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name.isidentifier():
        raise ExpressionValidationError(f"{label} '{value}' is not a valid identifier.")
    if name in _FUNCTIONS or name in _CONSTANTS:
        raise ExpressionValidationError(f"{label} '{name}' is reserved.")
    return name


def _parse_expression(expression: str, allowed_names: set[str]) -> ast.Expression:
    text = str(expression).strip()
    if not text:
        raise ExpressionValidationError("equations cannot be empty.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionValidationError(f"invalid equation '{text}': {exc.msg}.") from exc

    operator_nodes = tuple(_BINARY_OPERATORS) + tuple(_UNARY_OPERATORS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.Constant)):
            continue
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY_OPERATORS:
                raise ExpressionValidationError(f"operator {type(node.op).__name__} is not allowed.")
            continue
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPERATORS:
                raise ExpressionValidationError(f"operator {type(node.op).__name__} is not allowed.")
            continue
        if isinstance(node, operator_nodes):
            continue
        if isinstance(node, ast.Name):
            if node.id not in allowed_names and node.id not in _FUNCTIONS and node.id not in _CONSTANTS:
                raise ExpressionValidationError(f"unknown symbol '{node.id}'.")
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ExpressionValidationError("only documented mathematical functions are allowed.")
            if node.keywords:
                raise ExpressionValidationError("keyword arguments are not allowed in equations.")
            continue
        raise ExpressionValidationError(f"syntax element {type(node).__name__} is not allowed.")
    return tree


def _evaluate(node: ast.AST, environment: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, environment)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionValidationError("only real numeric constants are allowed.")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in environment:
            return float(environment[node.id])
        if node.id in _CONSTANTS:
            return float(_CONSTANTS[node.id])
        raise ExpressionValidationError(f"unknown symbol '{node.id}'.")
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS[type(node.op)]
        return float(operation(_evaluate(node.left, environment), _evaluate(node.right, environment)))
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS[type(node.op)]
        return float(operation(_evaluate(node.operand, environment)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        values = [_evaluate(arg, environment) for arg in node.args]
        return float(_FUNCTIONS[node.func.id](*values))
    raise ExpressionValidationError(f"cannot evaluate syntax element {type(node).__name__}.")


@dataclass(frozen=True)
class ExpressionSystemDefinition:
    """Serializable no-code definition for a continuous flow or discrete map."""

    name: str
    variables: tuple[str, ...]
    equations: tuple[str, ...]
    parameters: Mapping[str, float] = field(default_factory=dict)
    initial_state: tuple[float, ...] = field(default_factory=tuple)
    kind: SystemKind = "flow"
    description: str = ""
    reference: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ExpressionValidationError("name cannot be empty.")
        if self.kind not in {"flow", "map"}:
            raise ExpressionValidationError("kind must be 'flow' or 'map'.")
        variables = tuple(_identifier(item, label="variable") for item in self.variables)
        if not variables:
            raise ExpressionValidationError("at least one variable is required.")
        if len(set(variables)) != len(variables):
            raise ExpressionValidationError("variable names must be unique.")
        parameter_names = tuple(_identifier(item, label="parameter") for item in self.parameters)
        if set(variables) & set(parameter_names):
            raise ExpressionValidationError("variables and parameters must use distinct names.")
        if len(self.equations) != len(variables):
            raise ExpressionValidationError("one equation is required per variable.")
        if self.initial_state and len(self.initial_state) != len(variables):
            raise ExpressionValidationError("initial_state must match the number of variables.")
        parameter_values = np.asarray(list(self.parameters.values()), dtype=float)
        if parameter_values.size and not np.all(np.isfinite(parameter_values)):
            raise ExpressionValidationError("parameter values must be finite.")
        if self.initial_state and not np.all(np.isfinite(np.asarray(self.initial_state, dtype=float))):
            raise ExpressionValidationError("initial_state must contain finite values.")
        allowed_names = set(variables) | set(parameter_names)
        for expression in self.equations:
            _parse_expression(expression, allowed_names)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExpressionSystemDefinition":
        """Build a validated definition from JSON/YAML-compatible data."""

        return cls(
            name=str(value.get("name", "")).strip(),
            variables=tuple(str(item).strip() for item in value.get("variables", ())),
            equations=tuple(str(item).strip() for item in value.get("equations", ())),
            parameters={str(k).strip(): float(v) for k, v in dict(value.get("parameters", {})).items()},
            initial_state=tuple(float(item) for item in value.get("initial_state", ())),
            kind=str(value.get("kind", "flow")).strip().lower(),
            description=str(value.get("description", "")),
            reference=dict(value.get("reference", {})),
            metadata=dict(value.get("metadata", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return JSON/YAML-compatible data for persistence."""

        return {
            "schema": "hidden-attractors-expression-system/v1",
            "name": self.name,
            "kind": self.kind,
            "variables": list(self.variables),
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "equations": list(self.equations),
            "initial_state": [float(item) for item in self.initial_state],
            "description": self.description,
            "reference": dict(self.reference),
            "metadata": dict(self.metadata),
        }


def compile_expression_system(definition: ExpressionSystemDefinition | Mapping[str, Any]) -> ChaoticSystem:
    """Compile a declarative definition into a reusable :class:`ChaoticSystem`."""

    spec = definition if isinstance(definition, ExpressionSystemDefinition) else ExpressionSystemDefinition.from_mapping(definition)
    allowed_names = set(spec.variables) | set(spec.parameters)
    trees = tuple(_parse_expression(expression, allowed_names) for expression in spec.equations)

    def rhs(state: Sequence[float] | np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
        values = np.asarray(state, dtype=float)
        if values.shape != (len(spec.variables),):
            raise ValueError(f"{spec.name} expects state shape ({len(spec.variables)},).")
        environment = {name: float(values[index]) for index, name in enumerate(spec.variables)}
        for key, default in spec.parameters.items():
            environment[key] = float(parameters.get(key, default))
        with np.errstate(all="raise"):
            try:
                result = np.asarray([_evaluate(tree, environment) for tree in trees], dtype=float)
            except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as exc:
                raise FloatingPointError(f"equation evaluation failed: {exc}") from exc
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("equation evaluation produced a non-finite state.")
        return result

    metadata = dict(spec.metadata)
    metadata.update({"definition_schema": "hidden-attractors-expression-system/v1", "source": "safe_expression"})
    return ChaoticSystem(
        name=spec.name,
        dimension=len(spec.variables),
        rhs=rhs,
        parameters=dict(spec.parameters),
        description=spec.description,
        tags=("user-defined", "expression", spec.kind),
        kind=spec.kind,
        state_names=tuple(spec.variables),
        initial_state=tuple(spec.initial_state),
        reference=dict(spec.reference),
        metadata=metadata,
    )


__all__ = ["ExpressionSystemDefinition", "ExpressionValidationError", "compile_expression_system"]
