"""Shared reproducibility metadata for auditable numerical runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata as importlib_metadata
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

from .io import write_json
from .paths import PROJECT_ROOT


REPRODUCIBILITY_SCHEMA_VERSION = "1.0"
CONSERVATIVE_HIDDENNESS_LABEL = "compatible_with_hiddenness_under_tested_radii"
DEFAULT_TOLERANCES = {
    "equilibrium_residual_tol": 1.0e-8,
    "matignon_tol": 1.0e-12,
    "target_match_tol": 0.5,
    "boundedness_norm": 120.0,
    "nontrivial_variance_tol": 1.0e-8,
    "lyapunov_positive_tol": 0.02,
    "zero_one_chaos_threshold": 0.7,
    "zero_one_regular_threshold": 0.3,
    "spectral_peak_dominance_threshold": 0.8,
}
TOLERANCE_FIELDS = tuple(DEFAULT_TOLERANCES)


@dataclass(frozen=True)
class LureMetadata:
    """Lure decomposition used to construct a seed."""

    matrix: Any
    input_vector: Any
    output_vector: Any
    scalar_nonlinearity: str
    transfer_convention: str
    harmonic_condition: str


@dataclass(frozen=True)
class SeedMetadata:
    """Seed identity and construction data."""

    candidate_id: str
    family: str
    x0: Any
    source: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NumericalMetadata:
    """Numerical contract needed to reproduce one integration."""

    q: float
    h: float
    t_final: float
    t_burn: float
    memory: Mapping[str, Any]
    integrator: Mapping[str, Any]


@dataclass(frozen=True)
class ContinuationMetadata:
    """Continuation path and Caputo-memory propagation policy."""

    used: bool = False
    eta_path: Sequence[float] = ()
    continuation_mode: str = "none"
    memory_window_propagated: bool | None = None
    final_eta: float | None = None


@dataclass(frozen=True)
class ToleranceMetadata:
    """Numerical tolerances used by decision gates."""

    equilibrium_residual_tol: float = DEFAULT_TOLERANCES["equilibrium_residual_tol"]
    matignon_tol: float = DEFAULT_TOLERANCES["matignon_tol"]
    target_match_tol: float = DEFAULT_TOLERANCES["target_match_tol"]
    boundedness_norm: float = DEFAULT_TOLERANCES["boundedness_norm"]
    nontrivial_variance_tol: float = DEFAULT_TOLERANCES["nontrivial_variance_tol"]
    lyapunov_positive_tol: float = DEFAULT_TOLERANCES["lyapunov_positive_tol"]
    zero_one_chaos_threshold: float = DEFAULT_TOLERANCES["zero_one_chaos_threshold"]
    zero_one_regular_threshold: float = DEFAULT_TOLERANCES["zero_one_regular_threshold"]
    spectral_peak_dominance_threshold: float = DEFAULT_TOLERANCES["spectral_peak_dominance_threshold"]


@dataclass(frozen=True)
class SoftwareMetadata:
    """Software provenance for one integration."""

    python_version: str
    platform: str
    package_version: str
    numpy_version: str
    scipy_version: str
    git_commit: str
    working_tree_dirty: bool
    git_diff_sha256: str | None


@dataclass(frozen=True)
class RunMetadata:
    """Complete reproducibility envelope for a numerical run."""

    schema_version: str
    run_id: str
    workflow: str
    system: str
    created_at_utc: str
    numerical_contract: NumericalMetadata | Mapping[str, Any]
    software: SoftwareMetadata | Mapping[str, Any]
    continuation: ContinuationMetadata | Mapping[str, Any]
    tolerances: ToleranceMetadata | Mapping[str, Any]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    lure: LureMetadata | Mapping[str, Any] | None = None
    seed: SeedMetadata | Mapping[str, Any] | None = None
    random_seed: int | None = None
    random_seed_policy: str = "not_applicable"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dirty_tree_sha256(status: str) -> str:
    """Hash tracked diffs plus untracked file contents for source-tree audit."""

    digest = sha256()
    digest.update(status.encode("utf-8"))
    try:
        diff = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "diff", "--binary", "--no-ext-diff", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        digest.update(diff)
        untracked = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        repository_root = Path(_git_output("rev-parse", "--show-toplevel"))
        for relative in sorted(item for item in untracked.split(b"\0") if item):
            digest.update(relative)
            path = repository_root / relative.decode("utf-8", errors="surrogateescape")
            if path.is_file():
                digest.update(path.read_bytes())
    except (OSError, subprocess.SubprocessError):
        digest.update(b"git-diff-unavailable")
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return importlib_metadata.version("hidden-attractors-fo")
    except importlib_metadata.PackageNotFoundError:
        return "source-tree"


def collect_software_metadata() -> SoftwareMetadata:
    """Collect software provenance without requiring an installed package."""

    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    dirty = status not in {"", "unknown"}
    diff_hash = _dirty_tree_sha256(status) if dirty else None
    return SoftwareMetadata(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_version=_package_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        git_commit=commit,
        working_tree_dirty=dirty,
        git_diff_sha256=diff_hash,
    )


def collect_lure_metadata(
    lure: Any,
    *,
    transfer_convention: str,
    harmonic_condition: str,
) -> LureMetadata:
    """Serialise a maintained ``LureSystem`` without serialising callbacks."""

    nonlinearity = getattr(lure, "nonlinearity", None)
    callback_name = getattr(nonlinearity, "__qualname__", getattr(nonlinearity, "__name__", "unknown"))
    return LureMetadata(
        matrix=getattr(lure, "matrix", None),
        input_vector=getattr(lure, "input_vector", None),
        output_vector=getattr(lure, "output_vector", None),
        scalar_nonlinearity=str(callback_name),
        transfer_convention=str(transfer_convention),
        harmonic_condition=str(harmonic_condition),
    )


def collect_seed_metadata(
    seed: Mapping[str, Any] | None,
    *,
    source: str,
) -> SeedMetadata | None:
    """Normalise a workflow seed record into the shared metadata schema."""

    if not seed:
        return None
    x0 = seed.get("x0", seed.get("seed", seed.get("robust_start")))
    if x0 is None and all(key in seed for key in ("seed_x", "seed_y", "seed_z")):
        x0 = [seed["seed_x"], seed["seed_y"], seed["seed_z"]]
    return SeedMetadata(
        candidate_id=str(seed.get("candidate_id", "")),
        family=str(seed.get("family", seed.get("method", ""))),
        x0=x0,
        source=str(source),
        parameters={
            key: seed[key]
            for key in ("A", "sigma0", "omega", "mu", "theta", "gain_k", "k")
            if key in seed
        },
    )


def _normalise_memory_mode(mode: str) -> str:
    aliases = {
        "full": "full",
        "full_history": "full",
        "full_caputo": "full",
        "finite_window": "finite_window",
        "window": "finite_window",
        "short_memory": "finite_window",
    }
    return aliases.get(str(mode).strip().lower(), str(mode).strip().lower())


def collect_run_metadata(
    *,
    run_id: str,
    workflow: str,
    system: str,
    q: float,
    h: float,
    t_final: float,
    t_burn: float,
    memory_mode: str,
    integrator_name: str,
    integrator_backend: str,
    caputo: bool,
    M: int | None = None,
    memory_window_steps: int | None = None,
    memory_window_time: float | None = None,
    is_full_caputo: bool | None = None,
    parameters: Mapping[str, Any] | None = None,
    lure: LureMetadata | Mapping[str, Any] | None = None,
    seed: SeedMetadata | Mapping[str, Any] | None = None,
    random_seed: int | None = None,
    random_seed_policy: str = "not_applicable",
    provenance: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    continuation: ContinuationMetadata | Mapping[str, Any] | None = None,
    tolerances: ToleranceMetadata | Mapping[str, Any] | None = None,
) -> RunMetadata:
    """Build the common metadata envelope used by maintained workflows."""

    mode = _normalise_memory_mode(memory_mode)
    if is_full_caputo is None:
        is_full_caputo = mode == "full"
    memory = {
        "mode": mode,
        "M": M,
        "memory_window_steps": memory_window_steps,
        "memory_window_time": memory_window_time,
        "is_full_caputo": bool(is_full_caputo),
    }
    integrator = {
        "name": str(integrator_name),
        "backend": str(integrator_backend),
        "caputo": bool(caputo),
    }
    continuation_payload = metadata_to_jsonable(ContinuationMetadata())
    if continuation is not None:
        continuation_payload.update(metadata_to_jsonable(continuation))
    tolerance_payload = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        tolerance_payload.update(metadata_to_jsonable(tolerances))
    return RunMetadata(
        schema_version=REPRODUCIBILITY_SCHEMA_VERSION,
        run_id=str(run_id),
        workflow=str(workflow),
        system=str(system),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        numerical_contract=NumericalMetadata(
            q=float(q),
            h=float(h),
            t_final=float(t_final),
            t_burn=float(t_burn),
            memory=memory,
            integrator=integrator,
        ),
        software=collect_software_metadata(),
        continuation=continuation_payload,
        tolerances=tolerance_payload,
        parameters=dict(parameters or {}),
        lure=lure,
        seed=seed,
        random_seed=random_seed,
        random_seed_policy=str(random_seed_policy),
        provenance=dict(provenance or {}),
        extra=dict(extra or {}),
    )


def metadata_to_jsonable(value: Any) -> Any:
    """Convert dataclasses, numpy values and paths to JSON-safe values."""

    if is_dataclass(value):
        return metadata_to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): metadata_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [metadata_to_jsonable(item) for item in value]
    return value


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_mapping(
    metadata: Mapping[str, Any],
    key: str,
    errors: list[str],
) -> Mapping[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


def extract_run_metadata(container: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Read either metadata alias and return the canonical JSON-ready payload."""

    if not isinstance(container, Mapping):
        return None
    candidate = container.get("run_metadata", container.get("reproducibility_metadata"))
    if candidate is None and "numerical_contract" in container and "software" in container:
        candidate = container
    if not isinstance(candidate, Mapping):
        return None
    return metadata_to_jsonable(candidate)


def validate_run_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return every missing or malformed base reproducibility field."""

    errors: list[str] = []
    if not isinstance(metadata, Mapping):
        return ["metadata must be an object"]
    for key in ("schema_version", "run_id", "workflow", "system", "created_at_utc"):
        if not str(metadata.get(key, "")).strip():
            errors.append(f"{key} is required")

    numerical = _require_mapping(metadata, "numerical_contract", errors)
    for key in ("q", "h", "t_final", "t_burn"):
        if not _is_finite_number(numerical.get(key)):
            errors.append(f"numerical_contract.{key} must be finite")
    if _is_finite_number(numerical.get("q")) and not 0 < float(numerical["q"]) <= 1:
        errors.append("numerical_contract.q must satisfy 0 < q <= 1")
    if _is_finite_number(numerical.get("h")) and float(numerical["h"]) <= 0:
        errors.append("numerical_contract.h must be positive")
    if _is_finite_number(numerical.get("t_final")) and float(numerical["t_final"]) <= 0:
        errors.append("numerical_contract.t_final must be positive")
    if (
        _is_finite_number(numerical.get("t_burn"))
        and _is_finite_number(numerical.get("t_final"))
        and not 0 <= float(numerical["t_burn"]) < float(numerical["t_final"])
    ):
        errors.append("numerical_contract.t_burn must satisfy 0 <= t_burn < t_final")

    memory = _require_mapping(numerical, "memory", errors)
    if memory.get("mode") not in {"full", "finite_window"}:
        errors.append("numerical_contract.memory.mode must be full or finite_window")
    for key in ("M", "memory_window_steps", "memory_window_time", "is_full_caputo"):
        if key not in memory:
            errors.append(f"numerical_contract.memory.{key} is required")
    if not isinstance(memory.get("is_full_caputo"), bool):
        errors.append("numerical_contract.memory.is_full_caputo must be boolean")

    integrator = _require_mapping(numerical, "integrator", errors)
    if not str(integrator.get("name", "")).strip():
        errors.append("numerical_contract.integrator.name is required")
    if integrator.get("backend") not in {"python", "native", "unknown"}:
        errors.append("numerical_contract.integrator.backend must be python, native or unknown")
    if not isinstance(integrator.get("caputo"), bool):
        errors.append("numerical_contract.integrator.caputo must be boolean")

    if not isinstance(metadata.get("parameters"), Mapping):
        errors.append("parameters must be an object")

    tolerances = _require_mapping(metadata, "tolerances", errors)
    for key in TOLERANCE_FIELDS:
        if not _is_finite_number(tolerances.get(key)):
            errors.append(f"tolerances.{key} must be finite")

    continuation = _require_mapping(metadata, "continuation", errors)
    if not isinstance(continuation.get("used"), bool):
        errors.append("continuation.used must be boolean")
    if continuation.get("continuation_mode") not in {
        "integer",
        "fractional",
        "none",
        "paper_style",
        "unknown",
    }:
        errors.append("continuation.continuation_mode is invalid")
    eta_path = continuation.get("eta_path")
    if not isinstance(eta_path, (list, tuple)):
        errors.append("continuation.eta_path must be an array")
    elif continuation.get("used") and not eta_path:
        errors.append("continuation.eta_path is required when continuation.used=true")
    elif any(not _is_finite_number(value) for value in eta_path):
        errors.append("continuation.eta_path values must be finite")
    if continuation.get("used") and not _is_finite_number(continuation.get("final_eta")):
        errors.append("continuation.final_eta must be finite when continuation.used=true")
    if continuation.get("memory_window_propagated") not in {True, False, None}:
        errors.append("continuation.memory_window_propagated must be boolean or null")

    software = _require_mapping(metadata, "software", errors)
    for key in ("python_version", "platform", "package_version", "numpy_version", "scipy_version", "git_commit"):
        if not str(software.get(key, "")).strip():
            errors.append(f"software.{key} is required")
    if not isinstance(software.get("working_tree_dirty"), bool):
        errors.append("software.working_tree_dirty must be boolean")
    if software.get("working_tree_dirty") and not software.get("git_diff_sha256"):
        errors.append("software.git_diff_sha256 is required for a dirty worktree")
    random_seed_policy = str(metadata.get("random_seed_policy", "")).strip()
    if not random_seed_policy:
        errors.append("random_seed_policy is required")
    if random_seed_policy == "fixed_reproducible" and not isinstance(metadata.get("random_seed"), int):
        errors.append("random_seed must be an integer when random_seed_policy=fixed_reproducible")
    return errors


def validate_hiddenness_promotion_metadata(metadata: dict[str, Any] | None) -> list[str]:
    """Validate metadata required for a strong sampled-neighborhood promotion."""

    if metadata is None:
        return ["run_metadata is required for a strong candidate promotion"]
    jsonable = metadata_to_jsonable(metadata)
    errors = validate_run_metadata(jsonable)
    numerical = jsonable.get("numerical_contract", {})
    memory = numerical.get("memory", {}) if isinstance(numerical, Mapping) else {}
    integrator = numerical.get("integrator", {}) if isinstance(numerical, Mapping) else {}
    if memory.get("is_full_caputo") is not True:
        errors.append("strong candidate promotion requires numerical_contract.memory.is_full_caputo=true")
    if integrator.get("backend") == "unknown":
        errors.append("strong candidate promotion requires a known integrator backend")
    software = jsonable.get("software", {})
    if isinstance(software, Mapping) and software.get("git_commit") == "unknown":
        errors.append("strong candidate promotion requires a known software.git_commit")
    if jsonable.get("random_seed_policy") != "fixed_reproducible" or not isinstance(jsonable.get("random_seed"), int):
        errors.append("strong candidate promotion requires an integer random_seed with random_seed_policy=fixed_reproducible")

    lure = _require_mapping(jsonable, "lure", errors)
    for key in (
        "matrix",
        "input_vector",
        "output_vector",
        "scalar_nonlinearity",
        "transfer_convention",
        "harmonic_condition",
    ):
        if lure.get(key) in (None, "", []):
            errors.append(f"lure.{key} is required for a strong candidate promotion")

    seed = _require_mapping(jsonable, "seed", errors)
    for key in ("candidate_id", "family", "x0", "source"):
        if seed.get(key) in (None, "", []):
            errors.append(f"seed.{key} is required for a strong candidate promotion")
    return list(dict.fromkeys(errors))


def write_run_metadata(path: str | Path, metadata: RunMetadata | Mapping[str, Any]) -> dict[str, Any]:
    """Write an auditable JSON metadata file and return its serialised payload."""

    payload = metadata_to_jsonable(metadata)
    payload["metadata_validation_errors"] = validate_run_metadata(payload)
    write_json(Path(path), payload)
    return payload
