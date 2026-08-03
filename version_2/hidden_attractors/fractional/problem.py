"""Explicit problem contract for integer-compatible fractional dynamics.

Stability: experimental

The problem object separates derivative, orders, lower terminal, initial data,
kernel parameters, memory policy, and numerical method.  This is the migration
path away from treating a scalar ``q`` as a complete fractional model.

References
----------
Operator and method records resolve through :mod:`.references`; every result
contains the corresponding stable reference keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .._rhs import bind_rhs as bind_rhs_signature
from ._log_grid import stable_log_ratio, uniform_step_grid_metrics
from .contracts import (
    FractionalDerivativeDefinition,
    FractionalMethodDefinition,
    get_fractional_derivative,
    get_fractional_method,
    normalize_fractional_orders,
    validate_fractional_method,
)


_INITIAL_CONDITION_KINDS = frozenset(
    {
        "classical",
        "discrete_sample",
        "fractional_integral",
        "classical_compatibility_constrained",
        "model_specific",
        "classical_local",
    }
)

_DEFAULT_INITIAL_CONDITION_KIND = {
    "caputo": "classical",
    "grunwald_letnikov": "discrete_sample",
    "riemann_liouville": "fractional_integral",
    "caputo_fabrizio": "classical_compatibility_constrained",
    "atangana_baleanu_caputo": "classical_compatibility_constrained",
    "tempered_caputo": "classical",
    "tempered_riemann_liouville": "fractional_integral",
    "variable_order_caputo": "classical",
    "caputo_variable_type3": "classical",
    "variable_order_grunwald_letnikov": "discrete_sample",
    "caputo_distributed_order": "classical",
    "distributed_order": "model_specific",
    "conformable": "classical_local",
    "hadamard_riemann_liouville": "fractional_integral",
    "caputo_hadamard": "classical",
}

_ABC_KERNEL_PARAMETER_NAMES = frozenset({"normalization", "normalization_name"})
_ABC_METHOD_OPTION_NAMES = frozenset(
    {
        "compatibility_tolerance",
        "startup_tolerance",
        "startup_max_iterations",
    }
)
_TEMPERED_CAPUTO_KERNEL_PARAMETER_NAMES = frozenset({"tempering"})
_VO_TYPE3_KERNEL_PARAMETER_NAMES = frozenset(
    {"order_function", "order_function_name"}
)
_VO_TYPE3_METHOD_OPTION_NAMES = frozenset(
    {
        "corrector_atol",
        "corrector_rtol",
        "corrector_max_iterations",
        "on_nonconvergence",
        "initial_regularity",
        "compatibility_tolerance",
    }
)
_DISTRIBUTED_ORDER_CAPUTO_KERNEL_PARAMETER_NAMES = frozenset(
    {
        "order_weights",
        "weight_semantics",
        "density_values",
        "normalization",
        "order_quadrature_name",
    }
)
_DISTRIBUTED_ORDER_CAPUTO_METHOD_OPTION_NAMES = frozenset(
    {
        "corrector_atol",
        "corrector_rtol",
        "corrector_max_iterations",
        "on_nonconvergence",
        "initial_regularity",
        "compatibility_tolerance",
    }
)


def _normalize_distributed_order_nodes(value: Any) -> np.ndarray:
    """Validate order-space nodes independently of the state dimension."""

    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise TypeError(
            "caputo_distributed_order orders must be real order nodes, not "
            "Boolean or complex."
        )
    raw = np.asarray(value)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError(
            "caputo_distributed_order orders must be real order nodes, not "
            "Boolean or complex."
        )
    try:
        nodes = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "caputo_distributed_order orders must be a real one-dimensional "
            "sequence of order nodes."
        ) from exc
    if nodes.size < 1:
        raise ValueError(
            "caputo_distributed_order orders must contain at least one node."
        )
    if not np.all(np.isfinite(nodes)) or np.any(nodes <= 0.0) or np.any(nodes > 1.0):
        raise ValueError(
            "caputo_distributed_order order nodes must be finite and lie in (0, 1]."
        )
    return np.ascontiguousarray(nodes, dtype=float)


def _json_compatible_copy(value: Any, *, path: str) -> Any:
    """Return a detached JSON/YAML value or reject an executable object."""

    if isinstance(value, np.generic):
        return _json_compatible_copy(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _json_compatible_copy(value.tolist(), path=path)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise TypeError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            copied[key] = _json_compatible_copy(item, path=f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible_copy(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains non-serializable value {type(value).__name__}; "
        "store an importable identifier in configuration and bind the callable in code."
    )


def _metadata_copy(value: Any, *, path: str) -> Any:
    """Return JSON-oriented provenance, representing executable leaves safely."""

    try:
        return _json_compatible_copy(value, path=path)
    except TypeError:
        if isinstance(value, Mapping):
            return {
                str(key): _metadata_copy(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                _metadata_copy(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        return repr(value)


@dataclass(frozen=True, slots=True)
class FractionalProblem:
    """Complete numerical contract for a fractional initial/history problem.

    ``allow_experimental`` is an execution opt-in, not an evidence upgrade.
    Planned, research-gated, and theoretical-only methods remain non-executable.
    For ``caputo_distributed_order``, ``orders`` is the order-space node rule
    and is intentionally independent of the number of state components.
    """

    derivative: str
    method: str
    orders: float | Sequence[float] | np.ndarray
    initial_state: Sequence[float] | np.ndarray
    step: float
    t_span: tuple[float, float]
    grid_coordinate: str = "physical_time"
    memory_policy: str = "full_history"
    history_window: int | None = None
    lower_terminal: float | None = None
    initial_condition_kind: str = "auto"
    kernel_parameters: Mapping[str, Any] = field(default_factory=dict)
    method_options: Mapping[str, Any] = field(default_factory=dict)
    allow_experimental: bool = False
    problem_id: str | None = None

    def __post_init__(self) -> None:
        derivative = str(self.derivative).strip().lower()
        method = str(self.method).strip().lower()
        derivative_info = get_fractional_derivative(derivative)
        method_info = get_fractional_method(method)
        if not isinstance(self.allow_experimental, (bool, np.bool_)):
            raise TypeError("allow_experimental must be Boolean.")
        allow_experimental = bool(self.allow_experimental)

        state = np.asarray(self.initial_state, dtype=float).reshape(-1)
        if state.size < 1 or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain at least one finite value.")
        if derivative == "caputo_distributed_order":
            orders = _normalize_distributed_order_nodes(self.orders)
        else:
            orders = normalize_fractional_orders(self.orders, state.size)
        start, stop = (float(value) for value in self.t_span)
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise ValueError("t_span must contain finite values with stop > start.")
        step = float(self.step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive.")
        grid_coordinate = str(self.grid_coordinate).strip().lower()
        if grid_coordinate not in {
            "physical_time",
            "log_t_over_lower_terminal",
            "conformable_clock",
        }:
            raise ValueError(
                "grid_coordinate must be 'physical_time', "
                "'log_t_over_lower_terminal', or 'conformable_clock'."
            )
        if grid_coordinate == "log_t_over_lower_terminal":
            if start <= 0.0:
                raise ValueError("A logarithmic grid requires t_span[0] > 0.")
            coordinate_duration = float(stable_log_ratio(stop, start))
        elif grid_coordinate == "conformable_clock":
            if not np.all(orders == orders[0]):
                raise ValueError(
                    "A single conformable clock currently requires a commensurate order."
                )
            coordinate_duration = float((stop - start) ** orders[0] / orders[0])
        else:
            coordinate_duration = stop - start
        nearest_steps, grid_residual, grid_tolerance = uniform_step_grid_metrics(
            coordinate_duration,
            step,
        )
        if nearest_steps < 1 or grid_residual > grid_tolerance:
            raise ValueError(
                "t_span must contain an integer number of fixed steps/increments in "
                f"grid_coordinate={grid_coordinate!r}."
            )
        lower_terminal = start if self.lower_terminal is None else float(self.lower_terminal)
        if not np.isfinite(lower_terminal) or lower_terminal > start:
            raise ValueError("lower_terminal must be finite and no later than t_span[0].")
        if lower_terminal < start:
            raise NotImplementedError(
                "A lower terminal before t_span[0] requires an explicit prehistory contract."
            )
        if derivative in {"hadamard_riemann_liouville", "caputo_hadamard"} and (
            lower_terminal <= 0.0
        ):
            raise ValueError("Hadamard-family problems require lower_terminal > 0.")
        if grid_coordinate == "log_t_over_lower_terminal":
            if derivative != "caputo_hadamard":
                raise ValueError(
                    "The logarithmic solver grid is currently implemented only "
                    "for derivative='caputo_hadamard'."
                )
            if method != "caputo_hadamard_abm_pece":
                raise ValueError(
                    "grid_coordinate='log_t_over_lower_terminal' currently "
                    "requires method='caputo_hadamard_abm_pece'."
                )
            if lower_terminal != start:
                raise ValueError(
                    "The Caputo--Hadamard solver requires lower_terminal == t_span[0]."
                )
        elif grid_coordinate == "conformable_clock":
            if derivative != "conformable":
                raise ValueError(
                    "grid_coordinate='conformable_clock' currently requires "
                    "derivative='conformable'."
                )
            if method != "conformable_rk4_clock":
                raise ValueError(
                    "grid_coordinate='conformable_clock' currently requires "
                    "method='conformable_rk4_clock'."
                )
            if lower_terminal != start:
                raise ValueError(
                    "The conformable-clock solver requires "
                    "lower_terminal == t_span[0]."
                )
        elif method == "caputo_hadamard_abm_pece":
            raise ValueError(
                "caputo_hadamard_abm_pece requires "
                "grid_coordinate='log_t_over_lower_terminal'; step is log_step."
            )
        elif method == "conformable_rk4_clock":
            raise ValueError(
                "conformable_rk4_clock requires "
                "grid_coordinate='conformable_clock'; step is clock_step."
            )

        initial_kind = str(self.initial_condition_kind).strip().lower()
        if initial_kind == "auto":
            initial_kind = _DEFAULT_INITIAL_CONDITION_KIND[derivative]
        if initial_kind not in _INITIAL_CONDITION_KINDS:
            raise ValueError(f"Unknown initial_condition_kind: {initial_kind!r}.")
        expected_kind = _DEFAULT_INITIAL_CONDITION_KIND[derivative]
        if initial_kind != expected_kind:
            raise ValueError(
                f"Derivative {derivative!r} requires initial_condition_kind "
                f"{expected_kind!r}; received {initial_kind!r}."
            )

        memory_policy = str(self.memory_policy).strip().lower()
        kernel_parameters = dict(self.kernel_parameters)
        if derivative == "caputo_variable_type3":
            order_function = kernel_parameters.get("order_function")
            if not callable(order_function):
                raise TypeError(
                    "caputo_variable_type3 requires callable "
                    "kernel_parameters['order_function'] as a prescribed "
                    "alpha(time) schedule."
                )
            if not np.all(orders == orders[0]):
                raise ValueError(
                    "caputo_variable_type3 requires a common nominal initial "
                    "order in FractionalProblem.orders."
                )
            if "order_function_name" not in kernel_parameters:
                raise ValueError(
                    "caputo_variable_type3 requires "
                    "kernel_parameters['order_function_name']."
                )
            if not str(kernel_parameters["order_function_name"]).strip():
                raise ValueError("order_function_name must not be empty.")
            order_mode = "variable"
        elif derivative == "caputo_distributed_order":
            order_mode = "distributed"
        else:
            order_mode = (
                "commensurate" if np.all(orders == orders[0]) else "componentwise"
            )
        validate_fractional_method(
            derivative,
            method,
            order_mode=order_mode,
            memory_policy=memory_policy,
            require_implemented=False,
        )
        if method == "abc_predictor_corrector" and float(orders[0]) >= 1.0:
            raise ValueError("abc_predictor_corrector requires 0 < order < 1.")
        if method == "vo_caputo_type3_l1" and float(orders[0]) >= 1.0:
            raise ValueError("vo_caputo_type3_l1 requires 0 < nominal order < 1.")
        if (
            method == "tempered_caputo_abm_pece_transform"
            and float(orders[0]) >= 1.0
        ):
            raise ValueError(
                "tempered_caputo_abm_pece_transform requires 0 < order < 1."
            )
        normalized_history_window: int | None
        if memory_policy == "finite_window":
            if isinstance(self.history_window, (bool, np.bool_)):
                raise ValueError("finite_window requires history_window >= 2 samples.")
            try:
                normalized_history_window = operator.index(self.history_window)
            except TypeError as exc:
                raise ValueError(
                    "finite_window requires an integer history_window >= 2 samples."
                ) from exc
            if normalized_history_window < 2:
                raise ValueError("finite_window requires history_window >= 2 samples.")
            normalized_history_window = int(normalized_history_window)
        elif self.history_window is not None:
            raise ValueError("history_window is only valid with memory_policy='finite_window'.")
        else:
            normalized_history_window = None

        if derivative == "tempered_caputo":
            raw_tempering = kernel_parameters.get("tempering", np.nan)
            if isinstance(raw_tempering, (bool, np.bool_)) or np.iscomplexobj(
                raw_tempering
            ):
                raise TypeError(
                    "tempered_caputo requires a real kernel_parameters['tempering']."
                )
            tempering = float(raw_tempering)
            if not np.isfinite(tempering) or tempering < 0.0:
                raise ValueError("tempered_caputo requires kernel_parameters['tempering'] >= 0.")
            kernel_parameters["tempering"] = tempering
        if derivative == "variable_order_caputo" and "order_function" not in kernel_parameters:
            raise ValueError("variable_order_caputo requires kernel_parameters['order_function'].")
        if derivative == "caputo_distributed_order":
            if "order_weights" not in kernel_parameters:
                raise ValueError(
                    "caputo_distributed_order requires explicit "
                    "kernel_parameters['order_weights']."
                )
            semantics = str(
                kernel_parameters.get("weight_semantics", "nonnegative_mass")
            ).strip().lower()
            if semantics not in {
                "nonnegative_mass",
                "nonnegative_quadrature_density",
            }:
                raise ValueError(
                    "caputo_distributed_order requires nonnegative_mass or "
                    "nonnegative_quadrature_density semantics."
                )
            normalization = str(
                kernel_parameters.get("normalization", "none")
            ).strip().lower()
            from .distributed_order import _order_measure

            _order_measure(
                orders,
                kernel_parameters["order_weights"],
                kernel_parameters.get("density_values"),
                semantics,
                normalization,
            )
            kernel_parameters["weight_semantics"] = semantics
            kernel_parameters["normalization"] = normalization
            quadrature_name = str(
                kernel_parameters.get(
                    "order_quadrature_name",
                    "explicit_nodes_and_declared_weights",
                )
            ).strip()
            if not quadrature_name:
                raise ValueError("order_quadrature_name must not be empty.")
            kernel_parameters["order_quadrature_name"] = quadrature_name
        if derivative == "distributed_order" and not (
            "order_density" in kernel_parameters or "order_quadrature" in kernel_parameters
        ):
            raise ValueError(
                "distributed_order requires an order_density or order_quadrature kernel parameter."
            )

        object.__setattr__(self, "derivative", derivative)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "orders", tuple(float(value) for value in orders))
        object.__setattr__(self, "initial_state", tuple(float(value) for value in state))
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "t_span", (start, stop))
        object.__setattr__(self, "grid_coordinate", grid_coordinate)
        object.__setattr__(self, "memory_policy", memory_policy)
        object.__setattr__(self, "lower_terminal", lower_terminal)
        object.__setattr__(self, "initial_condition_kind", initial_kind)
        object.__setattr__(self, "history_window", normalized_history_window)
        object.__setattr__(self, "kernel_parameters", MappingProxyType(kernel_parameters))
        object.__setattr__(self, "method_options", MappingProxyType(dict(self.method_options)))
        object.__setattr__(self, "allow_experimental", allow_experimental)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FractionalProblem":
        """Build a validated problem from JSON/YAML-compatible input."""

        return cls(
            derivative=str(value.get("derivative", "")),
            method=str(value.get("method", "")),
            orders=value.get("orders", ()),
            initial_state=value.get("initial_state", ()),
            step=float(value.get("step", np.nan)),
            t_span=tuple(value.get("t_span", (0.0, np.nan))),
            grid_coordinate=str(value.get("grid_coordinate", "physical_time")),
            memory_policy=str(value.get("memory_policy", "full_history")),
            history_window=value.get("history_window"),
            lower_terminal=value.get("lower_terminal"),
            initial_condition_kind=str(value.get("initial_condition_kind", "auto")),
            kernel_parameters=dict(value.get("kernel_parameters", {})),
            method_options=dict(value.get("method_options", {})),
            allow_experimental=value.get("allow_experimental", False),
            problem_id=value.get("problem_id"),
        )

    @property
    def dimension(self) -> int:
        """Number of state components."""

        return len(self.initial_state)

    @property
    def duration(self) -> float:
        """Requested duration in physical time."""

        return self.t_span[1] - self.t_span[0]

    @property
    def coordinate_duration(self) -> float:
        """Duration in the explicitly selected integration coordinate."""

        if self.grid_coordinate == "log_t_over_lower_terminal":
            return float(stable_log_ratio(self.t_span[1], self.t_span[0]))
        if self.grid_coordinate == "conformable_clock":
            order = float(self.orders[0])
            return float(self.duration**order / order)
        return self.duration

    @property
    def n_steps(self) -> int:
        """Exact fixed-step count validated by the problem contract."""

        return int(round(self.coordinate_duration / self.step))

    @property
    def order_mode(self) -> str:
        """Return the explicit state/order-space mode."""

        if self.derivative == "caputo_variable_type3":
            return "variable"
        if self.derivative == "caputo_distributed_order":
            return "distributed"
        values = np.asarray(self.orders, dtype=float)
        return "commensurate" if np.all(values == values[0]) else "componentwise"

    @property
    def derivative_definition(self) -> FractionalDerivativeDefinition:
        """Resolved derivative registry entry."""

        return get_fractional_derivative(self.derivative)

    @property
    def method_definition(self) -> FractionalMethodDefinition:
        """Resolved numerical-method registry entry."""

        return get_fractional_method(self.method)

    @property
    def reference_keys(self) -> tuple[str, ...]:
        """Deduplicated references defining the operator and method."""

        return tuple(
            dict.fromkeys(
                self.derivative_definition.references + self.method_definition.references
            )
        )

    def validate_executable(self) -> None:
        """Reject registry-only and non-opted-in experimental problems."""

        derivative_status = self.derivative_definition.implementation_status
        method_status = self.method_definition.implementation_status
        forbidden = {"planned", "research_required", "theoretical_only"}
        if derivative_status in forbidden:
            raise NotImplementedError(
                f"Derivative {self.derivative!r} is {derivative_status} and cannot execute."
            )
        if method_status in forbidden:
            raise NotImplementedError(
                f"Method {self.method!r} is {method_status} and cannot execute."
            )
        if self.method_definition.execution_kind != "solver":
            raise NotImplementedError(
                f"Method {self.method!r} is a sampled operator, not a trajectory solver."
            )
        if "experimental" in {derivative_status, method_status} and not self.allow_experimental:
            raise PermissionError(
                "Experimental fractional execution requires allow_experimental=True."
            )
        if self.method_options:
            allowed_method_options = (
                _ABC_METHOD_OPTION_NAMES
                if self.method == "abc_predictor_corrector"
                else (
                _VO_TYPE3_METHOD_OPTION_NAMES
                    if self.method == "vo_caputo_type3_l1"
                    else (
                        _DISTRIBUTED_ORDER_CAPUTO_METHOD_OPTION_NAMES
                        if self.method == "distributed_order_caputo_l1"
                        else None
                    )
                )
            )
            if allowed_method_options is None:
                raise NotImplementedError(
                    f"Method {self.method!r} does not yet consume method_options; "
                    "execution is rejected to prevent silently ignored numerical settings."
                )
            unknown_method_options = set(self.method_options) - allowed_method_options
            if unknown_method_options:
                names = ", ".join(sorted(str(name) for name in unknown_method_options))
                raise ValueError(
                    f"{self.method} received unsupported method_options: "
                    f"{names}."
                )
        if self.kernel_parameters:
            if self.method == "abc_predictor_corrector":
                allowed_kernel_parameters = _ABC_KERNEL_PARAMETER_NAMES
            elif self.method == "tempered_caputo_abm_pece_transform":
                allowed_kernel_parameters = _TEMPERED_CAPUTO_KERNEL_PARAMETER_NAMES
            elif self.method == "vo_caputo_type3_l1":
                allowed_kernel_parameters = _VO_TYPE3_KERNEL_PARAMETER_NAMES
            elif self.method == "distributed_order_caputo_l1":
                allowed_kernel_parameters = (
                    _DISTRIBUTED_ORDER_CAPUTO_KERNEL_PARAMETER_NAMES
                )
            else:
                allowed_kernel_parameters = None
            if allowed_kernel_parameters is None:
                raise NotImplementedError(
                    f"Derivative {self.derivative!r} does not yet consume kernel_parameters "
                    "through this solver; execution is rejected to prevent a mismatched kernel."
                )
            unknown_kernel_parameters = (
                set(self.kernel_parameters) - allowed_kernel_parameters
            )
            if unknown_kernel_parameters:
                names = ", ".join(
                    sorted(str(name) for name in unknown_kernel_parameters)
                )
                raise ValueError(
                    f"{self.method} received unsupported kernel_parameters: "
                    f"{names}."
                )

    def as_metadata(self) -> dict[str, Any]:
        """Return a JSON-oriented numerical contract without executable callables."""

        kernel_metadata_input = dict(self.kernel_parameters)
        order_function_record: dict[str, str] | None = None
        if self.derivative == "caputo_variable_type3":
            kernel_metadata_input.pop("order_function", None)
            order_function_record = {
                "binding": "runtime_callable_not_serialized",
                "name": str(self.kernel_parameters["order_function_name"]),
            }
        kernel_parameters = _metadata_copy(
            kernel_metadata_input,
            path="kernel_parameters",
        )
        if order_function_record is not None:
            kernel_parameters["order_function"] = order_function_record
        method_options = _metadata_copy(self.method_options, path="method_options")
        return {
            "problem_id": self.problem_id,
            "derivative": self.derivative,
            "kernel_family": self.derivative_definition.kernel_family,
            "method": self.method,
            "orders": list(self.orders),
            "initial_state": list(self.initial_state),
            "order_mode": self.order_mode,
            "initial_condition_kind": self.initial_condition_kind,
            "lower_terminal": self.lower_terminal,
            "t_span": list(self.t_span),
            "step": self.step,
            "grid_coordinate": self.grid_coordinate,
            "coordinate_duration": self.coordinate_duration,
            "memory_policy": self.memory_policy,
            "history_window": self.history_window,
            "kernel_parameters": kernel_parameters,
            "method_options": method_options,
            "allow_experimental": self.allow_experimental,
            "reference_keys": list(self.reference_keys),
            "claims": "finite_numerical_trajectory_only",
        }

    def to_mapping(self) -> dict[str, Any]:
        """Return the executable JSON/YAML problem definition."""

        metadata = self.as_metadata()
        metadata.pop("claims")
        metadata.pop("kernel_family")
        metadata.pop("order_mode")
        metadata.pop("reference_keys")
        metadata.pop("coordinate_duration")
        metadata["schema"] = "hidden-attractors-fractional-problem/v1"
        metadata["kernel_parameters"] = _json_compatible_copy(
            self.kernel_parameters,
            path="kernel_parameters",
        )
        metadata["method_options"] = _json_compatible_copy(
            self.method_options,
            path="method_options",
        )
        return metadata


@dataclass(frozen=True, slots=True)
class FractionalProblemResult:
    """Trajectory and the complete problem contract used to generate it."""

    times: np.ndarray
    states: np.ndarray
    status: str
    backend: str
    problem: FractionalProblem
    metadata: Mapping[str, Any]

    @property
    def trajectory(self) -> np.ndarray:
        """Return conventional ``time,state...`` columns."""

        return np.column_stack((self.times, self.states))

    @property
    def coordinate_times(self) -> np.ndarray:
        """Return physical times or ``log(t/a)`` according to the problem grid."""

        if self.problem.grid_coordinate == "log_t_over_lower_terminal":
            return np.asarray(
                stable_log_ratio(self.times, float(self.problem.lower_terminal)),
                dtype=float,
            )
        if self.problem.grid_coordinate == "conformable_clock":
            from .conformable_solver import conformable_clock_from_time

            return np.asarray(
                conformable_clock_from_time(
                    self.times,
                    float(self.problem.orders[0]),
                    float(self.problem.lower_terminal),
                ),
                dtype=float,
            )
        return self.times


def _bind_rhs(rhs: Callable, parameters: Any) -> Callable[[float, np.ndarray], np.ndarray]:
    signature_bound_rhs = bind_rhs_signature(rhs, parameters)

    def bound(time: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(signature_bound_rhs(time, state), dtype=float)

    return bound


def solve_fractional_problem(
    problem: FractionalProblem,
    rhs: Callable,
    parameters: Any = None,
    *,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> FractionalProblemResult:
    """Solve one validated problem through an existing HAFO numerical lane."""

    if not isinstance(problem, FractionalProblem):
        raise TypeError("problem must be a FractionalProblem.")
    if not isinstance(use_acceleration, (bool, np.bool_)):
        raise TypeError("use_acceleration must be Boolean.")
    if not isinstance(allow_python_fallback, (bool, np.bool_)):
        raise TypeError("allow_python_fallback must be Boolean.")
    if divergence_norm is not None:
        divergence_norm = float(divergence_norm)
        if not np.isfinite(divergence_norm) or divergence_norm <= 0.0:
            raise ValueError("divergence_norm must be finite and positive, or None.")
    problem.validate_executable()
    start = problem.t_span[0]
    state0 = np.asarray(problem.initial_state, dtype=float)
    orders = np.asarray(problem.orders, dtype=float)
    backend_info: dict[str, Any] = {}

    if problem.method == "gl_explicit_discrete":
        from .gl_solver import integrate_gl_explicit

        initialization = (
            "caputo_shifted" if problem.derivative == "caputo" else "discrete_gl"
        )
        result = integrate_gl_explicit(
            rhs,
            state0,
            orders,
            parameters,
            step=problem.step,
            n_steps=problem.n_steps,
            t0=start,
            initialization=initialization,
            history_window=problem.history_window,
            use_acceleration=bool(use_acceleration),
            divergence_norm=divergence_norm,
        )
        times, states, status, backend = result.times, result.states, result.status, result.backend
        backend_info = {
            "implementation": result.backend,
            "used_acceleration": result.backend == "numba",
        }
    elif problem.method == "caputo_abm_pece":
        bound_rhs = _bind_rhs(rhs, parameters)
        if problem.order_mode == "componentwise":
            if problem.memory_policy != "block_restart":
                raise NotImplementedError(
                    "Component-wise Caputo ABM currently requires memory_policy='block_restart'."
                )
            from ..integrations.abm_fractional import integrate_fractional_abm

            times, states, status = integrate_fractional_abm(
                bound_rhs,
                state0,
                orders,
                problem.step,
                problem.n_steps,
                memory_protocol="block_restart",
                divergence_norm=divergence_norm,
            )
            times = times + start
            backend = "python_numpy"
            backend_info = {
                "implementation": "componentwise_block_restart",
                "used_c_backend": False,
                "rhs_source": "python_native",
            }
        else:
            from ..integrations.fractional_c import fractional_integrate

            if orders[0] == 1.0:
                raise ValueError("Caputo ABM requires q < 1; use an integer-order method at q=1.")
            memory_mode = "full" if problem.memory_policy == "full_history" else "window"
            if problem.memory_policy not in {"full_history", "finite_window"}:
                raise NotImplementedError(
                    "The monolithic Caputo ABM path supports full_history or finite_window."
                )

            def shifted_rhs(local_time: float, state: np.ndarray) -> np.ndarray:
                return bound_rhs(start + local_time, state)

            effective_divergence = (
                float("inf") if divergence_norm is None else divergence_norm
            )
            times, states, status, info = fractional_integrate(
                rhs=shifted_rhs,
                x0=state0,
                q=float(orders[0]),
                h=problem.step,
                t_final=float(
                    np.nextafter(problem.n_steps * problem.step, -np.inf)
                ),
                method="abm",
                memory_mode=memory_mode,
                memory_window_length=problem.history_window,
                use_c_backend=bool(use_acceleration),
                divergence_norm=effective_divergence,
                return_history=True,
                allow_python_fallback=bool(allow_python_fallback),
                early_stop_config={"enabled": False},
            )
            times = times + start
            backend_info = {
                str(key): value
                for key, value in info.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
            backend_info["effective_divergence_norm"] = effective_divergence
            backend = "native_c" if info.get("used_c_backend") else "python_numpy"
    elif problem.method == "tempered_caputo_abm_pece_transform":
        from .tempered_caputo_solver import integrate_tempered_caputo_abm

        result = integrate_tempered_caputo_abm(
            rhs,
            state0,
            float(orders[0]),
            parameters,
            tempering=float(problem.kernel_parameters["tempering"]),
            lower_terminal=float(problem.lower_terminal),
            upper_terminal=problem.t_span[1],
            step=problem.step,
            memory_policy=problem.memory_policy,
            history_window=problem.history_window,
            use_acceleration=bool(use_acceleration),
            allow_python_fallback=bool(allow_python_fallback),
            divergence_norm=divergence_norm,
        )
        times, states, status, backend = (
            result.times,
            result.states,
            result.status,
            result.backend,
        )
        backend_info = dict(result.solver_info)
        backend_info.update(
            {
                "tempering": result.tempering,
                "actual_upper_terminal": result.actual_upper_terminal,
                "memory_policy": result.memory_policy,
                "history_window": result.history_window,
            }
        )
    elif problem.method == "vo_caputo_type3_l1":
        from .variable_order_caputo_type3 import (
            integrate_variable_order_caputo_type3_l1,
        )

        kernel_parameters = dict(problem.kernel_parameters)
        method_options = dict(problem.method_options)
        result = integrate_variable_order_caputo_type3_l1(
            rhs,
            state0,
            parameters,
            step=problem.step,
            n_steps=problem.n_steps,
            lower_terminal=float(problem.lower_terminal),
            order_function=kernel_parameters["order_function"],
            order_function_name=kernel_parameters.get("order_function_name"),
            declared_initial_order=float(problem.orders[0]),
            corrector_atol=method_options.get("corrector_atol", 1.0e-12),
            corrector_rtol=method_options.get("corrector_rtol", 1.0e-10),
            corrector_max_iterations=method_options.get(
                "corrector_max_iterations",
                50,
            ),
            on_nonconvergence=method_options.get("on_nonconvergence", "raise"),
            initial_regularity=method_options.get("initial_regularity", "unknown"),
            compatibility_tolerance=method_options.get(
                "compatibility_tolerance",
                1.0e-10,
            ),
            use_acceleration=bool(use_acceleration),
            allow_python_fallback=bool(allow_python_fallback),
            divergence_norm=divergence_norm,
        )
        times, states, status, backend = (
            result.times,
            result.states,
            result.status,
            result.backend,
        )
        backend_info = dict(result.solver_info)
        backend_info.update(
            {
                "order_function_name": result.order_function_name,
                "evaluated_orders": result.orders.tolist(),
                "evaluated_order_samples": int(result.orders.size),
                "actual_upper_terminal": result.actual_upper_terminal,
                "memory_policy": result.memory_policy,
            }
        )
    elif problem.method == "distributed_order_caputo_l1":
        from .distributed_order_caputo_solver import (
            integrate_distributed_order_caputo_l1,
        )

        kernel_parameters = dict(problem.kernel_parameters)
        method_options = dict(problem.method_options)
        result = integrate_distributed_order_caputo_l1(
            rhs,
            state0,
            parameters,
            order_nodes=orders,
            order_weights=kernel_parameters["order_weights"],
            step=problem.step,
            n_steps=problem.n_steps,
            lower_terminal=float(problem.lower_terminal),
            weight_semantics=kernel_parameters.get(
                "weight_semantics",
                "nonnegative_mass",
            ),
            density_values=kernel_parameters.get("density_values"),
            normalization=kernel_parameters.get("normalization", "none"),
            order_quadrature_name=kernel_parameters.get(
                "order_quadrature_name",
                "explicit_nodes_and_declared_weights",
            ),
            corrector_atol=method_options.get("corrector_atol", 1.0e-12),
            corrector_rtol=method_options.get("corrector_rtol", 1.0e-10),
            corrector_max_iterations=method_options.get(
                "corrector_max_iterations",
                50,
            ),
            on_nonconvergence=method_options.get("on_nonconvergence", "raise"),
            initial_regularity=method_options.get(
                "initial_regularity",
                "unknown",
            ),
            compatibility_tolerance=method_options.get(
                "compatibility_tolerance",
                1.0e-10,
            ),
            use_acceleration=bool(use_acceleration),
            allow_python_fallback=bool(allow_python_fallback),
            divergence_norm=divergence_norm,
        )
        times, states, status, backend = (
            result.times,
            result.states,
            result.status,
            result.backend,
        )
        backend_info = dict(result.solver_info)
        backend_info.update(
            {
                "effective_order_weights": result.effective_weights.tolist(),
                "l1_current_coefficients": result.l1_coefficients.tolist(),
                "actual_upper_terminal": result.actual_upper_terminal,
                "memory_policy": result.memory_policy,
            }
        )
    elif problem.method == "abc_predictor_corrector":
        from .abc_solver import integrate_abc_predictor_corrector

        kernel_parameters = dict(problem.kernel_parameters)
        method_options = dict(problem.method_options)
        result = integrate_abc_predictor_corrector(
            rhs,
            state0,
            float(orders[0]),
            parameters,
            step=problem.step,
            n_steps=problem.n_steps,
            lower_terminal=float(problem.lower_terminal),
            normalization=kernel_parameters.get("normalization", 1.0),
            normalization_name=kernel_parameters.get("normalization_name"),
            compatibility_tolerance=method_options.get(
                "compatibility_tolerance",
                1.0e-12,
            ),
            startup_tolerance=method_options.get("startup_tolerance", 1.0e-12),
            startup_max_iterations=method_options.get(
                "startup_max_iterations",
                100,
            ),
            use_acceleration=bool(use_acceleration),
            allow_python_fallback=bool(allow_python_fallback),
            divergence_norm=divergence_norm,
        )
        times, states, status, backend = (
            result.times,
            result.states,
            result.status,
            result.backend,
        )
        backend_info = dict(result.solver_info)
        backend_info.update(
            {
                "normalization_value": result.normalization_value,
                "normalization_description": result.normalization_description,
                "compatibility_residual": result.compatibility_residual,
                "compatibility_tolerance": result.compatibility_tolerance,
                "startup_iterations": result.startup_iterations,
                "startup_tolerance": result.startup_tolerance,
                "startup_max_iterations": result.startup_max_iterations,
            }
        )
    elif problem.method == "caputo_hadamard_abm_pece":
        from .caputo_hadamard_solver import integrate_caputo_hadamard_abm

        result = integrate_caputo_hadamard_abm(
            rhs,
            state0,
            float(orders[0]),
            parameters,
            lower_terminal=float(problem.lower_terminal),
            upper_terminal=problem.t_span[1],
            log_step=problem.step,
            use_acceleration=bool(use_acceleration),
            allow_python_fallback=bool(allow_python_fallback),
            divergence_norm=divergence_norm,
        )
        times, states, status, backend = (
            result.times,
            result.states,
            result.status,
            result.backend,
        )
        backend_info = dict(result.solver_info)
    elif problem.method == "conformable_rk4_clock":
        from .conformable_solver import integrate_conformable_rk4

        result = integrate_conformable_rk4(
            rhs,
            state0,
            float(orders[0]),
            parameters,
            lower_terminal=float(problem.lower_terminal),
            upper_terminal=problem.t_span[1],
            clock_step=problem.step,
            use_acceleration=bool(use_acceleration),
            allow_python_fallback=bool(allow_python_fallback),
            divergence_norm=divergence_norm,
        )
        times, states, status, backend = (
            result.times,
            result.states,
            result.status,
            result.backend,
        )
        backend_info = dict(result.solver_info)
    elif problem.method == "efork3":
        if problem.order_mode != "commensurate":
            raise NotImplementedError("EFORK-3 currently requires a commensurate order.")
        if problem.memory_policy not in {"full_history", "finite_window"}:
            raise NotImplementedError("EFORK-3 facade supports full_history or finite_window.")
        bound_rhs = _bind_rhs(rhs, parameters)

        def shifted_rhs(local_time: float, state: np.ndarray) -> np.ndarray:
            return bound_rhs(start + local_time, state)

        memory_mode = (
            "full" if problem.memory_policy == "full_history" else "window"
        )
        if orders[0] < 1.0:
            from ..integrations.fractional_c import fractional_integrate

            effective_divergence = (
                float("inf") if divergence_norm is None else divergence_norm
            )
            times, states, status, info = fractional_integrate(
                rhs=shifted_rhs,
                x0=state0,
                q=float(orders[0]),
                h=problem.step,
                t_final=problem.duration,
                method="efork3",
                memory_mode=memory_mode,
                memory_window_length=problem.history_window,
                use_c_backend=bool(use_acceleration),
                divergence_norm=effective_divergence,
                return_history=True,
                allow_python_fallback=bool(allow_python_fallback),
                early_stop_config={"enabled": False},
            )
            backend_info = {
                str(key): value
                for key, value in info.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
            backend_info["effective_divergence_norm"] = effective_divergence
            backend = "native_c" if info.get("used_c_backend") else "python_numpy"
        else:
            from ..integrations.selector import integrate

            times, states, status = integrate(
                shifted_rhs,
                state0,
                float(orders[0]),
                problem.step,
                problem.duration,
                integrator="efork3",
                memory_mode=memory_mode,
                memory_window_length=problem.history_window,
                divergence_norm=divergence_norm,
                use_c_backend=bool(use_acceleration),
                allow_python_fallback=bool(allow_python_fallback),
                early_stop_config={"enabled": False},
            )
            backend = "python_efork_q1"
            backend_info = {
                "implementation": "python_efork_q1",
                "used_c_backend": False,
                "rhs_source": "python_native",
            }
        times = times + start
    else:
        raise NotImplementedError(
            f"Method {problem.method!r} is an operator or registry entry without a solver dispatcher."
        )

    metadata = problem.as_metadata()
    metadata["backend"] = backend
    metadata["backend_info"] = _metadata_copy(
        backend_info,
        path="backend_info",
    )
    metadata["status"] = str(status)
    metadata["divergence_norm"] = divergence_norm
    metadata["rhs_parameters"] = _metadata_copy(
        parameters,
        path="rhs_parameters",
    )
    return FractionalProblemResult(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=float),
        status=str(status),
        backend=backend,
        problem=problem,
        metadata=MappingProxyType(metadata),
    )


def solve_fractional_system(
    problem: FractionalProblem,
    system: Any,
    parameters: Mapping[str, Any] | None = None,
    *,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> FractionalProblemResult:
    """Solve a registered, built-in, or expression-defined HAFO flow.

    This adapter is the no-code bridge used by Toolbox Chaos.  It preserves the
    same :class:`FractionalProblemResult` contract as direct callable use.
    """

    from ..systems import ChaoticSystem, get_system

    model = get_system(system) if isinstance(system, str) else system
    if not isinstance(model, ChaoticSystem):
        raise TypeError("system must be a ChaoticSystem or registered system name.")
    if model.kind != "flow":
        raise ValueError(
            "FractionalProblem describes a continuous-memory flow; fractional "
            "difference equations require a separate problem type."
        )
    if model.dimension != problem.dimension:
        raise ValueError(
            f"system dimension {model.dimension} does not match problem dimension {problem.dimension}."
        )
    active_parameters = dict(model.parameters)
    if parameters:
        active_parameters.update(parameters)

    def system_rhs(time: float, state: np.ndarray, values: Mapping[str, Any]) -> np.ndarray:
        del time
        return model.evaluate(state, values)

    result = solve_fractional_problem(
        problem,
        system_rhs,
        active_parameters,
        use_acceleration=use_acceleration,
        allow_python_fallback=allow_python_fallback,
        divergence_norm=divergence_norm,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "system_name": model.name,
            "system_kind": model.kind,
            "system_parameters": active_parameters,
            "adapter": "hafo_system",
        }
    )
    return FractionalProblemResult(
        times=result.times,
        states=result.states,
        status=result.status,
        backend=result.backend,
        problem=result.problem,
        metadata=MappingProxyType(metadata),
    )


__all__ = [
    "FractionalProblem",
    "FractionalProblemResult",
    "solve_fractional_problem",
    "solve_fractional_system",
]
