"""Reusable integer-order workflows for systems in Lur'e form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..analysis.trajectory import classify_trajectory_against_equilibria, cloud_median_distance, sample_rows
from ..seed_generation import HarmonicSeed, build_lure_linearized_matrix, find_lure_harmonic_seed
from ..solvers.integer import efork_q1_integrate
from ..systems.base import ChaoticSystem
from ..systems.lure import LureSystem


@dataclass(frozen=True)
class IntegerLureContinuationStep:
    """One epsilon-continuation step from a Lur'e harmonic seed."""

    epsilon: float
    x_in: np.ndarray
    x_out: np.ndarray
    trajectory: np.ndarray
    status: str


@dataclass(frozen=True)
class IntegerHiddennessProbe:
    """One equilibrium-neighborhood probe for an integer-order system."""

    equilibrium: str
    radius: float
    sample_id: int
    x0: np.ndarray
    status: str
    final_class: str
    target_hit: bool
    cloud_distance: float
    cloud_distance_norm: float
    trajectory: np.ndarray
    metrics: Mapping[str, Any]


def require_lure(system: ChaoticSystem | LureSystem) -> LureSystem:
    """Return a Lur'e representation or raise a workflow-facing error."""

    if isinstance(system, LureSystem):
        return system
    if system.lure is None:
        raise ValueError(f"{system.name} must provide a manual Lur'e form for this workflow.")
    return system.lure


def require_equilibria(system: ChaoticSystem, equilibria: Mapping[str, Sequence[float]] | None = None) -> dict[str, np.ndarray]:
    """Return explicit or registered equilibria for hiddenness controls."""

    if equilibria is not None:
        out = {str(name): np.asarray(value, dtype=float) for name, value in equilibria.items()}
    else:
        out = system.equilibrium_points()
    if not out:
        raise ValueError(f"{system.name} must provide equilibria for hiddenness controls.")
    return out


def integer_lure_seed(
    system: ChaoticSystem | LureSystem,
    *,
    branch_index: int = 0,
    method: str = "classic",
    mu: float = 1.0,
    theta: float = 0.0,
    wmin: float = 1.0e-5,
    wmax: float = 50.0,
    nscan: int = 40_000,
) -> HarmonicSeed:
    """Build an integer-order Lur'e seed using ``s=i*omega``.

    ``method`` may be ``"classic"`` or ``"machado"``.  The selected system must
    provide the corresponding describing-function relation.
    """

    if method not in {"classic", "machado"}:
        raise ValueError("method must be 'classic' or 'machado'.")
    return find_lure_harmonic_seed(
        q=1.0,
        system=require_lure(system),
        branch_index=branch_index,
        method=method,  # type: ignore[arg-type]
        mu=mu,
        theta=theta,
        wmin=wmin,
        wmax=wmax,
        nscan=nscan,
    )


def integer_lure_original_rhs(lure: LureSystem):
    """Return ``x -> A x + b psi(c^T x)`` for an integer-order Lur'e system."""

    return lambda x: lure.evaluate(np.asarray(x, dtype=float))


def integer_lure_epsilon_rhs(lure: LureSystem, gain: float, epsilon: float):
    """Return the epsilon-family RHS used for continuation to the original system."""

    k = float(gain)
    eps = float(epsilon)
    p0 = build_lure_linearized_matrix(lure, k)
    bvec = np.asarray(lure.input_vector, dtype=float)
    cvec = np.asarray(lure.output_vector, dtype=float)

    def rhs(x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=float)
        sigma = float(cvec @ state)
        delta = float(lure.nonlinearity(sigma)) - k * sigma
        return p0 @ state + eps * bvec * delta

    return rhs


def integrate_integer_lure(
    system: ChaoticSystem | LureSystem,
    x0: Sequence[float] | np.ndarray,
    *,
    t_final: float,
    h: float,
    div_threshold: float | None = None,
) -> tuple[np.ndarray, str]:
    """Integrate an integer-order Lur'e system from ``x0``."""

    lure = require_lure(system)
    return efork_q1_integrate(
        integer_lure_original_rhs(lure),
        np.asarray(x0, dtype=float),
        t_final=t_final,
        h=h,
        div_threshold=div_threshold,
    )


def continue_integer_lure_seed(
    system: ChaoticSystem | LureSystem,
    seed: HarmonicSeed,
    *,
    eps_values: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    t_transient: float = 80.0,
    t_keep: float = 80.0,
    h: float = 0.01,
    div_threshold: float | None = None,
) -> list[IntegerLureContinuationStep]:
    """Run epsilon continuation from a harmonic seed to the original system."""

    lure = require_lure(system)
    x_in = np.asarray(seed.seed, dtype=float).copy()
    steps: list[IntegerLureContinuationStep] = []
    for eps in eps_values:
        rhs = integer_lure_epsilon_rhs(lure, seed.gain, float(eps))
        transient, transient_status = efork_q1_integrate(
            rhs,
            x_in,
            t_final=t_transient,
            h=h,
            div_threshold=div_threshold,
        )
        if transient_status != "ok":
            steps.append(
                IntegerLureContinuationStep(
                    epsilon=float(eps),
                    x_in=x_in.copy(),
                    x_out=transient[-1, 1:].copy(),
                    trajectory=transient,
                    status=transient_status,
                )
            )
            break
        x_mid = transient[-1, 1:].copy()
        kept, kept_status = efork_q1_integrate(
            rhs,
            x_mid,
            t_final=t_keep,
            h=h,
            div_threshold=div_threshold,
        )
        x_out = kept[-1, 1:].copy()
        steps.append(
            IntegerLureContinuationStep(
                epsilon=float(eps),
                x_in=x_in.copy(),
                x_out=x_out,
                trajectory=kept,
                status=kept_status,
            )
        )
        if kept_status != "ok":
            break
        x_in = x_out
    return steps


def final_integer_lure_attractor(
    system: ChaoticSystem | LureSystem,
    x0: Sequence[float] | np.ndarray,
    *,
    t_burn: float = 120.0,
    t_keep: float = 180.0,
    h: float = 0.01,
    div_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Burn in from ``x0`` and return ``(target_seed, kept_trajectory, status)``."""

    lure = require_lure(system)
    rhs = integer_lure_original_rhs(lure)
    burn, burn_status = efork_q1_integrate(rhs, np.asarray(x0, dtype=float), t_final=t_burn, h=h, div_threshold=div_threshold)
    if burn_status != "ok":
        return burn[-1, 1:].copy(), burn, burn_status
    target_seed = burn[-1, 1:].copy()
    kept, kept_status = efork_q1_integrate(rhs, target_seed, t_final=t_keep, h=h, div_threshold=div_threshold)
    return target_seed, kept, kept_status


def _tail_states(trajectory: np.ndarray, *, t_start: float, max_points: int) -> np.ndarray:
    data = np.asarray(trajectory, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        return np.empty((0, 0), dtype=float)
    tail = data[data[:, 0] >= float(t_start)]
    if tail.size == 0:
        tail = data
    return sample_rows(tail[:, 1:], max_points)


def _random_unit_vectors(dimension: int, count: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(int(count), int(dimension)))
    norms = np.linalg.norm(raw, axis=1)
    norms[norms == 0.0] = 1.0
    return raw / norms[:, None]


def run_integer_lure_hiddenness_controls(
    system: ChaoticSystem,
    target_trajectory: np.ndarray,
    *,
    equilibria: Mapping[str, Sequence[float]] | None = None,
    radii: Sequence[float] = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2),
    samples_per_radius: int = 24,
    t_final: float = 500.0,
    t_burn: float = 120.0,
    h: float = 0.01,
    div_threshold: float = 120.0,
    equilibrium_tol: float = 1.0e-3,
    target_cloud_tol: float = 0.08,
    max_cloud_points: int = 1000,
    random_seed: int = 123456789,
) -> list[IntegerHiddennessProbe]:
    """Run integer-order hiddenness controls from equilibrium neighborhoods.

    A TARGET hit means that a post-burn probe from an equilibrium neighborhood
    reaches a bounded nontrivial trajectory whose tail cloud is close to the
    reference attractor tail.  Any TARGET hit weakens or blocks a hiddenness
    claim under the tested numerical contract.
    """

    lure = require_lure(system)
    eqs = require_equilibria(system, equilibria)
    target_cloud = _tail_states(target_trajectory, t_start=t_burn, max_points=max_cloud_points)
    if target_cloud.size == 0:
        raise ValueError("target_trajectory does not contain usable state samples.")
    target_scale = max(float(np.linalg.norm(np.ptp(target_cloud, axis=0))), 1.0e-12)
    rng = np.random.default_rng(int(random_seed))
    probes: list[IntegerHiddennessProbe] = []
    sample_id = 0
    rhs = integer_lure_original_rhs(lure)
    for eq_name, eq in eqs.items():
        eq_arr = np.asarray(eq, dtype=float)
        if eq_arr.shape != (lure.dimension,):
            raise ValueError(f"equilibrium {eq_name} has shape {eq_arr.shape}, expected ({lure.dimension},).")
        for radius in radii:
            directions = _random_unit_vectors(lure.dimension, int(samples_per_radius), rng)
            for direction in directions:
                x0 = eq_arr + float(radius) * direction
                traj, status = efork_q1_integrate(
                    rhs,
                    x0,
                    t_final=t_final,
                    h=h,
                    div_threshold=div_threshold,
                )
                cls = classify_trajectory_against_equilibria(
                    traj,
                    eqs,
                    divergence_norm=div_threshold,
                    equilibrium_tol=equilibrium_tol,
                    t_start=t_burn,
                )
                probe_cloud = _tail_states(traj, t_start=t_burn, max_points=max_cloud_points)
                cloud = cloud_median_distance(probe_cloud, target_cloud) if probe_cloud.size else float("nan")
                cloud_norm = cloud / target_scale if np.isfinite(cloud) else float("nan")
                final_class = str(cls["final_class"])
                target_hit = bool(
                    status == "ok"
                    and final_class == "bounded_nontrivial"
                    and np.isfinite(cloud_norm)
                    and cloud_norm <= float(target_cloud_tol)
                )
                probes.append(
                    IntegerHiddennessProbe(
                        equilibrium=eq_name,
                        radius=float(radius),
                        sample_id=sample_id,
                        x0=x0,
                        status=status,
                        final_class=final_class,
                        target_hit=target_hit,
                        cloud_distance=float(cloud),
                        cloud_distance_norm=float(cloud_norm),
                        trajectory=traj,
                        metrics=cls,
                    )
                )
                sample_id += 1
    return probes


def summarize_integer_hiddenness_controls(probes: Sequence[IntegerHiddennessProbe]) -> dict[str, Any]:
    """Summarize integer hiddenness controls for reports and CSV exports."""

    total = len(probes)
    target_hits = sum(1 for probe in probes if probe.target_hit)
    by_equilibrium: dict[str, dict[str, int]] = {}
    for probe in probes:
        row = by_equilibrium.setdefault(probe.equilibrium, {"n": 0, "target_hits": 0, "equilibrium_hits": 0, "diverged": 0})
        row["n"] += 1
        row["target_hits"] += int(probe.target_hit)
        row["equilibrium_hits"] += int(str(probe.final_class).startswith("equilibrium_"))
        row["diverged"] += int(probe.final_class == "diverged" or probe.status != "ok")
    return {
        "n_probes": total,
        "target_hits": target_hits,
        "hidden_candidate_allowed": bool(total > 0 and target_hits == 0),
        "by_equilibrium": by_equilibrium,
    }


__all__ = [
    "IntegerLureContinuationStep",
    "IntegerHiddennessProbe",
    "continue_integer_lure_seed",
    "final_integer_lure_attractor",
    "integer_lure_epsilon_rhs",
    "integer_lure_original_rhs",
    "integer_lure_seed",
    "integrate_integer_lure",
    "require_lure",
    "run_integer_lure_hiddenness_controls",
    "summarize_integer_hiddenness_controls",
]
