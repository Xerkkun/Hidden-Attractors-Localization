"""Read-only audit of a completed full MAVPD integer hidden-chaos run.

The validator intentionally uses only the Python standard library.  It does
not rerun the dynamical system and it never writes into the run or repository;
instead, it independently checks the finite numerical claim against the
completed ledger, scientific-source snapshot, numerical CSV/JSON evidence, and
immutable ``by_run`` figure copies.  ``--require-active-promotion`` additionally
binds a newly promoted staging run to the mutable ``current``/``by_export``
copies and active global JSON/CSV manifests.  ``--scientific-source-root`` may
point at the immutable source snapshot used by the launcher, while
``--figure-store-root`` may point at the configured ``library_figures`` store.
``--repo-root`` continues to identify and confine the staged run.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


CASE_ID = "mavpd_integer_hidden_chaos"
SCIENTIFIC_SOURCE_FIXED_FILES = (
    "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py",
    "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/reproducibility.yaml",
    "validation/wolfram/cases/mavpd_integer.wl",
)
DIRECT_PHASES = (
    "contract",
    "search",
    "diagnostics",
    "hiddenness",
    "candidate_gate_and_figures",
    "manifest",
    "global_figure_promotion",
)
RESUMED_PHASES = (
    "contract",
    "search",
    "diagnostics",
    "hiddenness_resumed",
    "candidate_gate_and_figures_resumed",
    "manifest_resumed",
    "global_figure_promotion",
)
RESUMED_PHASE_TAIL = (
    "hiddenness_resumed",
    "candidate_gate_and_figures_resumed",
    "manifest_resumed",
    "global_figure_promotion",
)
DIRECT_TIMING_PHASES = (
    "contract",
    "search",
    "diagnostics",
    "hiddenness",
    "candidate_gate",
    "total",
)
RESUMED_TIMING_PHASES = (
    "contract",
    "search",
    "diagnostics",
    "hiddenness_resumed",
    "candidate_gate_resumed",
    "total_recorded_phase_seconds",
)
FIGURE_IDS = (
    "00_nyquist_direct_seed",
    "03_continuation_screen",
    "04_candidate_phase_portraits",
    "04_candidate_time_series",
    "05_lyapunov_convergence",
    "05_poincare_section",
    "05_normalized_fft_power",
    "07_hiddenness_outcomes",
)
REQUIRED_LEDGER_ARTIFACTS = frozenset(
    {
        "00_system_contract.json",
        "01_direct_seed_and_lambda_continuation.json",
        "02_parameter_continuation.csv",
        "03_candidate_screening.csv",
        "03_candidate_screening_probes.csv",
        "03_candidate_screening_contract.json",
        "03_candidate_selection.json",
        "04_candidate_trajectory.csv",
        "05_chaos_diagnostics.json",
        "05_lyapunov_convergence.csv",
        "05_poincare_section.csv",
        "05_zero_one_stride_sensitivity.csv",
        "05_zero_one_return_map.json",
        "05_normalized_fft_power.csv",
        "06_equilibrium_stability.json",
        "07_hiddenness_probes.csv",
        "07_hiddenness_summary.json",
        "07_hiddenness_initial_conditions.csv",
        "08_robustness_matrix.json",
        "09_candidate_gate.json",
        "phase_timings.csv",
        "run_manifest.json",
        "figures/figure_manifest.json",
        "figures/global_promotion_receipt.json",
        *(
            f"figures/{figure_id}.{suffix}"
            for figure_id in FIGURE_IDS
            for suffix in ("png", "pdf")
        ),
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_PATTERN = re.compile(
    rf"^{CASE_ID}-full-(?P<timestamp>[0-9]{{8}}T[0-9]{{12}}Z)-"
    r"(?P<source>[0-9a-f]{12})-(?P<nonce>[0-9a-f]{12})$"
)
EXPECTED_EQUATIONS = (
    "y1'=delta*gamma*y1+delta*y2-delta*y1^3",
    "y2'=y1-xi*y2-y3",
    "y3'=rho*y2",
)
EXPECTED_BASE_PARAMETERS = {"delta": 100.0, "gamma": 0.1, "rho": 200.0, "xi": 3.1}
EXPECTED_GATE_TOLERANCES = {
    "boundedness_norm": 120.0,
    "equilibrium_residual_tol": 1.0e-8,
    "lyapunov_positive_tol": 0.02,
    "matignon_tol": 1.0e-12,
    "nontrivial_variance_tol": 1.0e-8,
    "spectral_peak_dominance_threshold": 0.8,
    "target_match_tol": 0.5,
    "zero_one_chaos_threshold": 0.7,
    "zero_one_regular_threshold": 0.3,
}
EXPECTED_SCREEN_OFFSETS = (0.002, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015)
EXPECTED_SELECTION_RULE = (
    "largest tested Hopf offset with Lyapunov status ok, finite-time LLE > 0.5, "
    "zero sampled E0 contacts, zero ambiguous probes, and zero numerical failures; "
    "no transition boundary claimed"
)
EXPECTED_GATE_CONDITIONS = frozenset(
    {
        "equilibria_all_found",
        "equilibria_residual_within_tolerance",
        "matignon_all_classified",
        "matignon_q_recorded",
        "seed_localized",
        "seed_method_supported",
        "seed_source_traceable",
        "continuation_reaches_target",
        "continuation_eta_path_recorded",
        "continuation_memory_declared",
        "trajectory_bounded",
        "trajectory_nontrivial",
        "trajectory_finite_fraction_acceptable",
        "trajectory_post_transient_sufficient",
        "robustness_tested_h",
        "robustness_memory_requirement_satisfied",
        "robustness_tested_t_final",
        "robustness_tested_integrator",
        "robustness_consistent",
        "hiddenness_tested_all_equilibria",
        "hiddenness_tested_radii_recorded",
        "hiddenness_required_radii_tested",
        "hiddenness_equilibrium_radius_coverage_complete",
        "hiddenness_target_contacts_recorded",
        "hiddenness_zero_equilibrium_contacts",
        "hiddenness_no_basin_intersection",
        "hiddenness_basin_controls_complete",
        "hiddenness_numerical_failures_recorded",
        "hiddenness_no_numerical_failures",
        "reproducibility_metadata_complete",
    }
)
FIGURE_DATA_SOURCES = {
    "00_nyquist_direct_seed": ["00_system_contract.json"],
    "03_continuation_screen": ["03_candidate_screening.csv"],
    "04_candidate_phase_portraits": ["04_candidate_trajectory.csv"],
    "04_candidate_time_series": ["04_candidate_trajectory.csv"],
    "05_lyapunov_convergence": ["05_lyapunov_convergence.csv"],
    "05_poincare_section": ["05_poincare_section.csv"],
    "05_normalized_fft_power": ["05_normalized_fft_power.csv"],
    "07_hiddenness_outcomes": ["07_hiddenness_probes.csv"],
}
GLOBAL_MANIFEST_FIELDS = (
    "figure_id",
    "caption_key",
    "kind",
    "source_script",
    "source_function",
    "data_sources",
    "run_id",
    "system_id",
    "q",
    "parameters",
    "integrator",
    "memory_mode",
    "t_final",
    "t_burn",
    "pdf_path",
    "png_path",
    "metadata_path",
    "created_at",
    "git_commit",
    "export_targets",
)


class Audit:
    """Accumulate independent audit checks without hiding later failures."""

    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.checks: dict[str, dict[str, Any]] = {}

    def require(self, check: str, condition: bool, message: str) -> bool:
        entry = self.checks.setdefault(check, {"ok": True, "assertions": 0})
        entry["assertions"] += 1
        if not condition:
            entry["ok"] = False
            self.errors.append({"check": check, "message": message})
        return bool(condition)

    def fail(self, check: str, message: str) -> None:
        self.require(check, False, message)


def _phase_sequence_kind(phases: tuple[str, ...]) -> tuple[str, ...] | None:
    """Classify direct or legitimately resumed append-only phase histories."""

    if phases == DIRECT_PHASES:
        return DIRECT_PHASES
    # Resume is allowed after diagnostics.  The runner's append-only ledger may
    # retain any already recorded direct post-diagnostics prefix that the
    # resumed phases supersede (for example, an earlier ``hiddenness`` phase).
    for prefix_length in range(3, len(DIRECT_PHASES)):
        if phases == DIRECT_PHASES[:prefix_length] + RESUMED_PHASE_TAIL:
            return RESUMED_PHASES
    return None


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deterministic_unit_directions_3d(count: int) -> tuple[tuple[float, float, float], ...]:
    """Reconstruct the workflow's deterministic 3-D sphere-direction contract."""

    axes = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    if count == 6:
        return axes
    if count < 6:
        return ()
    extra_count = count - 6
    golden = math.pi * (3.0 - math.sqrt(5.0))
    extra: list[tuple[float, float, float]] = []
    for index in range(extra_count):
        z_value = 1.0 - (2.0 * index + 1.0) / extra_count
        radius = math.sqrt(max(0.0, 1.0 - z_value * z_value))
        angle = golden * index
        extra.append((radius * math.cos(angle), radius * math.sin(angle), z_value))
    return axes + tuple(extra)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        value[key] = item
    return value


def _assert_json_numbers_finite(value: Any, *, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_numbers_finite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_numbers_finite(item, location=f"{location}[{index}]")


def _loads_json_strict(text: str) -> Any:
    payload = json.loads(
        text,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_object_without_duplicate_keys,
    )
    _assert_json_numbers_finite(payload)
    return payload


def _read_json(audit: Audit, path: Path, label: str) -> Any | None:
    if not audit.require("required_files", path.is_file(), f"missing {label}: {path}"):
        return None
    if not audit.require("required_files", not path.is_symlink(), f"{label} must not be a symlink: {path}"):
        return None
    try:
        return _loads_json_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        audit.fail("json_parse", f"cannot parse {label} at {path}: {error}")
        return None


def _read_csv(audit: Audit, path: Path, label: str) -> list[dict[str, str]]:
    if not audit.require("required_files", path.is_file(), f"missing {label}: {path}"):
        return []
    if not audit.require("required_files", not path.is_symlink(), f"{label} must not be a symlink: {path}"):
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ())
            if not fieldnames or any(not field for field in fieldnames) or len(fieldnames) != len(set(fieldnames)):
                audit.fail("csv_parse", f"{label} has an empty or duplicate CSV header: {fieldnames}")
                return []
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                audit.fail("csv_parse", f"{label} has rows wider or shorter than its header")
                return []
            return rows
    except (OSError, UnicodeError, csv.Error) as error:
        audit.fail("csv_parse", f"cannot parse {label} at {path}: {error}")
        return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _bool_text(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    parsed = _float(value)
    if parsed is None or parsed < 0.0 or not parsed.is_integer():
        return None
    return int(parsed)


def _close(left: Any, right: Any, *, atol: float = 1.0e-12, rtol: float = 1.0e-10) -> bool:
    left_value = _float(left)
    right_value = _float(right)
    return bool(
        left_value is not None
        and right_value is not None
        and math.isclose(left_value, right_value, rel_tol=rtol, abs_tol=atol)
    )


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def _finite_vector(value: Any, length: int) -> tuple[float, ...] | None:
    items = _sequence(value)
    if len(items) != length:
        return None
    parsed = tuple(_float(item) for item in items)
    if any(item is None for item in parsed):
        return None
    return tuple(float(item) for item in parsed if item is not None)


def _complex_number(value: Any) -> complex | None:
    item = _mapping(value)
    real = _float(item.get("real"))
    imag = _float(item.get("imag"))
    if real is None or imag is None:
        return None
    return complex(real, imag)


def _mavpd_transfer_response(
    omega: float,
    *,
    delta: float = 100.0,
    gamma: float = 0.1,
    rho: float = 200.0,
    xi: float = 3.1,
) -> complex:
    """Evaluate c^T(A-i*omega*I)^(-1)b for the declared MAVPD Lur'e form."""

    s_value = complex(0.0, float(omega))
    quadratic = s_value * s_value + xi * s_value + rho
    denominator = (delta * gamma - s_value) * quadratic + delta * s_value
    if denominator == 0.0:
        raise ZeroDivisionError("MAVPD transfer function is singular at the recorded frequency")
    return -delta * quadratic / denominator


def _mavpd_direct_seed_pairs() -> tuple[tuple[float, float], ...]:
    """Recompute the two positive integer transfer roots and their gains."""

    delta = EXPECTED_BASE_PARAMETERS["delta"]
    rho = EXPECTED_BASE_PARAMETERS["rho"]
    xi = EXPECTED_BASE_PARAMETERS["xi"]
    # Im(N(i*w) conj(D(i*w))) / w = 0 reduces to this quadratic in u=w^2.
    linear = 2.0 * rho - delta - xi * xi
    constant = -rho * (delta - rho)
    discriminant = linear * linear - 4.0 * constant
    if discriminant <= 0.0:
        raise ValueError("MAVPD direct-transfer polynomial has no two real u roots")
    u_values = ((linear - math.sqrt(discriminant)) / 2.0, (linear + math.sqrt(discriminant)) / 2.0)
    pairs: list[tuple[float, float]] = []
    for u_value in u_values:
        if u_value <= 0.0:
            continue
        omega = math.sqrt(u_value)
        response = _mavpd_transfer_response(omega)
        if abs(response.imag) > 1.0e-8 or abs(response.real) <= 1.0e-14:
            raise ValueError("MAVPD direct-transfer root does not close on the real axis")
        pairs.append((omega, -1.0 / response.real))
    return tuple(sorted(pairs, key=lambda item: item[1]))


def _mavpd_characteristic_coefficients(
    *,
    linear_y1: float,
    delta: float,
    rho: float,
    xi: float,
) -> tuple[float, float, float]:
    """Return coefficients of det(lambda*I-J) for the MAVPD Jacobian form."""

    trace = linear_y1 - xi
    pair_sum = rho - delta - linear_y1 * xi
    determinant = linear_y1 * rho
    return -trace, pair_sum, -determinant


def _characteristic_residual(value: complex, coefficients: Sequence[float]) -> float:
    """Evaluate a relative cubic residual without overflowing on hostile input."""

    a1, a2, a3 = coefficients
    scale = max(
        1.0,
        abs(value),
        abs(a1),
        math.sqrt(abs(a2)),
        abs(a3) ** (1.0 / 3.0),
    )
    inverse = 1.0 / scale
    scaled_value = value * inverse
    terms = (
        scaled_value * scaled_value * scaled_value,
        (a1 * inverse) * scaled_value * scaled_value,
        ((a2 * inverse) * inverse) * scaled_value,
        ((a3 * inverse) * inverse) * inverse,
    )
    denominator = inverse * inverse * inverse + sum(abs(term) for term in terms)
    if not math.isfinite(denominator) or denominator <= 0.0:
        return math.inf
    result = abs(sum(terms)) / denominator
    return result if math.isfinite(result) else math.inf


def _audit_characteristic_spectrum(
    audit: Audit,
    *,
    check: str,
    label: str,
    values: Sequence[complex | None],
    coefficients: Sequence[float],
) -> tuple[complex, ...] | None:
    if len(values) != 3 or any(value is None for value in values):
        audit.fail(check, f"{label} must contain three finite eigenvalues")
        return None
    spectrum = tuple(value for value in values if value is not None)
    for value in spectrum:
        audit.require(
            check,
            _characteristic_residual(value, coefficients) <= 1.0e-10,
            f"{label} contains a value that is not a root of its Jacobian polynomial",
        )
    a1, a2, a3 = coefficients
    vieta = (
        (sum(spectrum), complex(-a1, 0.0), "sum"),
        (
            spectrum[0] * spectrum[1] + spectrum[0] * spectrum[2] + spectrum[1] * spectrum[2],
            complex(a2, 0.0),
            "pair sum",
        ),
        (spectrum[0] * spectrum[1] * spectrum[2], complex(-a3, 0.0), "product"),
    )
    for actual, expected, name in vieta:
        scale = 1.0 + abs(expected)
        audit.require(check, abs(actual - expected) / scale <= 1.0e-10, f"{label} violates Vieta {name}")
    return spectrum


def _same_json_semantics(actual: Any, expected: Any) -> bool:
    """Compare JSON-like values without Python's bool/int equality aliasing."""

    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _close(actual, expected, atol=1.0e-14, rtol=1.0e-12)
    if isinstance(expected, str) or expected is None:
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _same_json_semantics(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _same_json_semantics(left, right) for left, right in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _parse_json_cell(audit: Audit, raw: Any, *, check: str, label: str) -> Any | None:
    if not isinstance(raw, str):
        audit.fail(check, f"{label} must be JSON text")
        return None
    try:
        return _loads_json_strict(raw)
    except (ValueError, json.JSONDecodeError) as error:
        audit.fail(check, f"invalid embedded JSON in {label}: {error}")
        return None


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _best_effort_path_text(raw: Any) -> str:
    """Render a CLI path without allowing error reporting itself to fail."""

    try:
        return str(Path(raw).expanduser().resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        try:
            return str(raw)
        except (TypeError, ValueError):
            return repr(raw)


def _safe_relative_path(raw: Any, *, parent: Path) -> tuple[Path | None, str | None]:
    if not isinstance(raw, str) or not raw:
        return None, "path must be a non-empty string"
    candidate = Path(raw)
    if (
        candidate.is_absolute()
        or bool(candidate.drive)
        or bool(candidate.root)
        or candidate.anchor
        or ".." in candidate.parts
        or candidate.as_posix() != raw
        or any(":" in part for part in candidate.parts)
    ):
        return None, f"path is not canonical repository-relative POSIX text: {raw!r}"
    try:
        resolved = (parent / candidate).resolve()
    except (OSError, RuntimeError) as error:
        return None, f"path cannot be resolved safely: {raw!r}: {error}"
    if not _path_is_within(resolved, parent):
        return None, f"resolved path escapes {parent}: {raw!r} -> {resolved}"
    return resolved, None


def _resolve_recorded_path(
    raw: Any,
    *,
    relative_base: Path,
    confined_to: Path,
) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    try:
        if path.is_absolute():
            resolved = path.resolve()
        else:
            if (
                bool(path.drive)
                or bool(path.root)
                or ".." in path.parts
                or "\\" in raw
                or path.as_posix() != raw
                or any(":" in part for part in path.parts)
            ):
                return None
            resolved = (relative_base / path).resolve()
    except (OSError, RuntimeError):
        return None
    return resolved if _path_is_within(resolved, confined_to) else None


def _collect_key_values(value: Any, keys: frozenset[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys:
                found.append(item)
            found.extend(_collect_key_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_key_values(item, keys))
    return found


def _scientific_source_snapshot(repo_root: Path) -> dict[str, Any]:
    hidden_root = repo_root / "hidden_attractors"
    relative_files = set(SCIENTIFIC_SOURCE_FIXED_FILES)
    if hidden_root.is_dir():
        relative_files.update(
            path.relative_to(repo_root).as_posix()
            for path in hidden_root.rglob("*.py")
        )
    file_hashes: dict[str, str] = {}
    for relative in sorted(relative_files):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"scientific source is missing: {relative}")
        file_hashes[relative] = _sha256_file(path)
    bundle_material = "".join(
        f"{relative}\0{digest}\n" for relative, digest in sorted(file_hashes.items())
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "bundle_sha256": sha256(bundle_material).hexdigest(),
        "files": file_hashes,
    }


def _audit_status_and_ledger(
    audit: Audit,
    run_dir: Path,
    status: Mapping[str, Any],
) -> tuple[str | None, str | None, tuple[str, ...] | None, int]:
    run_id = status.get("run_id") if isinstance(status.get("run_id"), str) else None
    config_sha = status.get("config_sha256") if isinstance(status.get("config_sha256"), str) else None
    phases = tuple(str(item) for item in _sequence(status.get("completed_phases")))
    phase_kind: tuple[str, ...] | None = None

    audit.require("status", status.get("status") == "complete", "run_status.status must be 'complete'")
    audit.require("status", status.get("quick_mode") is False, "run_status.quick_mode must be false for full evidence")
    audit.require("status", bool(run_id), "run_status.run_id must be a non-empty string")
    run_match = RUN_ID_PATTERN.fullmatch(run_id or "")
    audit.require(
        "status",
        run_match is not None,
        f"run_id does not match the exact full-run schema: {run_id!r}",
    )
    source_snapshot = _mapping(status.get("scientific_source_snapshot"))
    source_bundle = source_snapshot.get("bundle_sha256")
    if run_match is not None:
        audit.require(
            "status",
            isinstance(source_bundle, str)
            and run_match.group("source") == source_bundle[:12],
            "run_id source prefix differs from scientific_source_snapshot.bundle_sha256",
        )
    audit.require(
        "status",
        bool(config_sha and SHA256_PATTERN.fullmatch(config_sha)),
        "run_status.config_sha256 must be a lowercase SHA-256 digest",
    )
    phase_kind = _phase_sequence_kind(phases)
    if phase_kind is None:
        audit.fail(
            "status_phases",
            f"completed_phases must equal a supported direct or append-only resumed sequence; got {list(phases)}",
        )
    audit.require(
        "status_phases",
        status.get("last_completed_phase") == "global_figure_promotion",
        "last_completed_phase must be global_figure_promotion",
    )
    audit.require("status", bool(status.get("completed_at_utc")), "completed_at_utc is missing")

    ledger = status.get("artifacts")
    if not isinstance(ledger, Mapping):
        audit.fail("ledger", "run_status.artifacts must be a mapping")
        return run_id, config_sha, phase_kind, 0
    missing = sorted(REQUIRED_LEDGER_ARTIFACTS - set(ledger))
    audit.require("ledger", not missing, f"required ledger artifacts are missing: {missing}")
    validated = 0
    for relative, expected in ledger.items():
        relative_text = str(relative)
        path, path_error = _safe_relative_path(relative_text, parent=run_dir)
        if not audit.require("ledger", path is not None, f"unsafe ledger path: {path_error}"):
            continue
        assert path is not None
        if not audit.require("ledger", path.is_file(), f"ledger artifact is missing: {relative_text}"):
            continue
        if not audit.require("ledger", not path.is_symlink(), f"ledger artifact is a symlink: {relative_text}"):
            continue
        if not audit.require(
            "ledger",
            isinstance(expected, str) and bool(SHA256_PATTERN.fullmatch(expected)),
            f"ledger digest is not lowercase SHA-256 for {relative_text}",
        ):
            continue
        try:
            actual = _sha256_file(path)
        except OSError as error:
            audit.fail("ledger", f"cannot hash {relative_text}: {error}")
            continue
        if audit.require(
            "ledger",
            actual == expected,
            f"ledger hash mismatch for {relative_text}: expected {expected}, got {actual}",
        ):
            validated += 1
    return run_id, config_sha, phase_kind, validated


def _audit_sources(
    audit: Audit,
    repo_root: Path,
    recorded_snapshot: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        current = _scientific_source_snapshot(repo_root)
    except (OSError, ValueError) as error:
        audit.fail("scientific_sources", str(error))
        return None
    audit.require(
        "scientific_sources",
        recorded_snapshot.get("algorithm") == "sha256",
        "recorded scientific source algorithm must be sha256",
    )
    recorded_files = recorded_snapshot.get("files")
    audit.require(
        "scientific_sources",
        isinstance(recorded_files, Mapping),
        "recorded scientific source files must be a mapping",
    )
    if isinstance(recorded_files, Mapping):
        audit.require(
            "scientific_sources",
            dict(recorded_files) == current["files"],
            "current scientific source file set or one of its hashes differs from the run",
        )
        try:
            recorded_material = "".join(
                f"{relative}\0{digest}\n"
                for relative, digest in sorted(recorded_files.items())
            ).encode("utf-8")
            rebuilt_recorded = sha256(recorded_material).hexdigest()
        except (AttributeError, TypeError) as error:
            audit.fail("scientific_sources", f"cannot recompute recorded source bundle: {error}")
        else:
            audit.require(
                "scientific_sources",
                rebuilt_recorded == recorded_snapshot.get("bundle_sha256"),
                "recorded source bundle does not match its recorded per-file hashes",
            )
    audit.require(
        "scientific_sources",
        current["bundle_sha256"] == recorded_snapshot.get("bundle_sha256"),
        (
            "current scientific source bundle differs from the completed run: "
            f"recorded={recorded_snapshot.get('bundle_sha256')}, current={current['bundle_sha256']}"
        ),
    )
    return current


def _audit_snapshot_manifests(
    audit: Audit,
    *,
    run_dir: Path,
    scientific_source_root: Path,
    recorded_snapshot: Mapping[str, Any],
) -> list[str]:
    checked: list[str] = []
    candidates = []
    for candidate in (
        scientific_source_root / "snapshot_manifest.json",
        run_dir / "snapshot_manifest.json",
    ):
        resolved = candidate.resolve()
        if resolved not in candidates and candidate.exists():
            candidates.append(resolved)
    for path in candidates:
        payload = _read_json(audit, path, f"scientific snapshot manifest {path}")
        if not isinstance(payload, Mapping):
            audit.fail("snapshot_manifest", f"snapshot manifest must be an object: {path}")
            continue
        nested = payload.get("scientific_source_snapshot")
        snapshot = _mapping(nested) if isinstance(nested, Mapping) else _mapping(payload)
        raw_files = snapshot.get("files")
        normalized_files: dict[str, str] = {}
        if not isinstance(raw_files, Mapping):
            audit.fail("snapshot_manifest", f"snapshot manifest files must be an object: {path}")
        else:
            for relative, raw_entry in raw_files.items():
                source_path, path_error = _safe_relative_path(relative, parent=scientific_source_root)
                audit.require(
                    "snapshot_manifest",
                    source_path is not None,
                    f"unsafe snapshot-manifest source path {relative!r}: {path_error}",
                )
                if isinstance(raw_entry, str):
                    digest = raw_entry
                    size_bytes = None
                elif isinstance(raw_entry, Mapping):
                    digest = raw_entry.get("sha256")
                    size_bytes = _integer(raw_entry.get("size_bytes"))
                    audit.require(
                        "snapshot_manifest",
                        set(raw_entry) == {"sha256", "size_bytes"},
                        f"verbose snapshot entry has missing or unexpected fields: {relative!r}",
                    )
                    audit.require(
                        "snapshot_manifest",
                        size_bytes is not None,
                        f"snapshot size_bytes must be a nonnegative integer: {relative!r}",
                    )
                else:
                    digest = None
                    size_bytes = None
                    audit.fail("snapshot_manifest", f"invalid snapshot file entry: {relative!r}")
                audit.require(
                    "snapshot_manifest",
                    isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest) is not None,
                    f"snapshot file digest is not lowercase SHA-256: {relative!r}",
                )
                if isinstance(digest, str):
                    normalized_files[str(relative)] = digest
                if size_bytes is not None and source_path is not None:
                    try:
                        actual_size = source_path.stat().st_size
                    except OSError as error:
                        audit.fail("snapshot_manifest", f"cannot stat snapshot source {relative!r}: {error}")
                    else:
                        audit.require(
                            "snapshot_manifest",
                            actual_size == size_bytes,
                            f"snapshot size differs for {relative!r}: manifest={size_bytes}, actual={actual_size}",
                        )
        projected = {
            "algorithm": snapshot.get("algorithm"),
            "bundle_sha256": snapshot.get("bundle_sha256"),
            "files": normalized_files,
        }
        audit.require(
            "snapshot_manifest",
            projected == dict(recorded_snapshot),
            f"snapshot_manifest.json differs from run_status.scientific_source_snapshot: {path}",
        )
        if "file_count" in snapshot:
            audit.require(
                "snapshot_manifest",
                _integer(snapshot.get("file_count")) == len(normalized_files),
                f"snapshot manifest file_count is inconsistent: {path}",
            )
        checked.append(str(path))
    return checked


def _audit_identity(
    audit: Audit,
    artifacts: Sequence[tuple[str, Any]],
    *,
    expected_run_id: str | None,
    expected_config_sha: str | None,
    expected_source_bundle: str | None,
) -> None:
    run_values: list[tuple[str, Any]] = []
    config_values: list[tuple[str, Any]] = []
    source_values: list[tuple[str, Any]] = []
    for label, payload in artifacts:
        run_values.extend((label, value) for value in _collect_key_values(payload, frozenset({"run_id"})))
        config_values.extend(
            (label, value) for value in _collect_key_values(payload, frozenset({"config_sha256"}))
        )
        source_values.extend(
            (label, value)
            for value in _collect_key_values(
                payload,
                frozenset({"bundle_sha256", "scientific_source_bundle_sha256"}),
            )
        )
    audit.require("identity", bool(run_values), "no run_id values were found in audited artifacts")
    audit.require("identity", bool(config_values), "no config_sha256 values were found in audited artifacts")
    audit.require("identity", bool(source_values), "no scientific source bundle values were found")
    for label, value in run_values:
        audit.require("identity", value == expected_run_id, f"run_id mismatch in {label}: {value!r}")
    for label, value in config_values:
        audit.require(
            "identity",
            value == expected_config_sha,
            f"config_sha256 mismatch in {label}: {value!r}",
        )
    for label, value in source_values:
        audit.require(
            "identity",
            value == expected_source_bundle,
            f"scientific source bundle mismatch in {label}: {value!r}",
        )


def _audit_claim_scope(
    audit: Audit,
    contract: Mapping[str, Any],
    selection: Mapping[str, Any],
    gate_file: Mapping[str, Any],
    manifest: Mapping[str, Any],
    hidden_summary: Mapping[str, Any],
) -> None:
    gate = _mapping(gate_file.get("gate"))
    evidence = _mapping(gate_file.get("evidence"))
    audit.require(
        "joint_gate",
        gate.get("chaotic_hidden_promotion_allowed") is True,
        "joint chaotic-hidden promotion gate did not pass",
    )
    audit.require(
        "joint_gate",
        gate.get("hidden_chaos_status") == "chaotic_hidden_under_tested_neighborhoods",
        f"unexpected finite hidden-chaos label: {gate.get('hidden_chaos_status')!r}",
    )
    audit.require(
        "joint_gate",
        gate.get("verdict") == "hidden_under_tested_neighborhoods",
        f"unexpected hiddenness verdict: {gate.get('verdict')!r}",
    )
    for key in ("missing_conditions", "warnings", "diagnostic_conflicts"):
        audit.require("joint_gate", gate.get(key) == [], f"gate.{key} must be an empty list")
    audit.require(
        "joint_gate",
        _same_json_semantics(manifest.get("candidate_gate"), gate),
        "run_manifest.candidate_gate differs from 09_candidate_gate.json",
    )

    frequency_expectations = (
        ("contract.frequency_grid_used_for_seed", contract.get("frequency_grid_used_for_seed"), False),
        ("contract.fallback_frequency_scan_used", contract.get("fallback_frequency_scan_used"), False),
        ("selection.no_frequency_sweep", selection.get("no_frequency_sweep"), True),
        ("manifest.frequency_grid_used_for_search", manifest.get("frequency_grid_used_for_search"), False),
        (
            "gate.evidence.run_metadata.provenance.frequency_grid_used_for_search",
            _mapping(_mapping(_mapping(evidence.get("run_metadata")).get("provenance"))).get(
                "frequency_grid_used_for_search"
            ),
            False,
        ),
    )
    for label, actual, expected in frequency_expectations:
        audit.require("no_frequency_sweep", actual is expected, f"{label} must be {expected!r}; got {actual!r}")
    for label, payload in (
        ("contract", contract),
        ("selection", selection),
        ("gate", gate_file),
        ("manifest", manifest),
    ):
        for value in _collect_key_values(
            payload,
            frozenset(
                {
                    "frequency_grid_used",
                    "frequency_grid_used_for_seed",
                    "frequency_grid_used_for_search",
                    "fallback_frequency_scan_used",
                }
            ),
        ):
            audit.require("no_frequency_sweep", value is False, f"truthy frequency-search flag in {label}")

    audit.require(
        "finite_scope",
        manifest.get("global_proof_claimed") is False,
        "run_manifest must explicitly set global_proof_claimed=false",
    )
    audit.require(
        "finite_scope",
        hidden_summary.get("sampled_hiddenness_status") == "hidden_under_tested_neighborhoods",
        "hiddenness summary must use the finite sampled-neighborhood label",
    )
    for family in ("main", "targeted_E0_unstable_direction"):
        family_summary = _mapping(hidden_summary.get(family))
        audit.require(
            "finite_scope",
            family_summary.get("finite_sample_only") is True,
            f"{family}.finite_sample_only must be true",
        )
        audit.require(
            "finite_scope",
            family_summary.get("global_hiddenness_proved") is False,
            f"{family}.global_hiddenness_proved must be false",
        )


def _mavpd_high_hopf_boundary(*, xi: float = 2.85, delta: float = 100.0, rho: float = 200.0) -> float:
    coefficient_b = xi * xi - delta
    discriminant = coefficient_b * coefficient_b - 4.0 * xi * xi * (rho - delta)
    if discriminant <= 0.0:
        raise ValueError("MAVPD high-Hopf quadratic has no distinct real roots")
    g_values = (
        (-coefficient_b - math.sqrt(discriminant)) / (2.0 * xi),
        (-coefficient_b + math.sqrt(discriminant)) / (2.0 * xi),
    )
    return max(g_values) / (2.0 * delta)


def _audit_parameter_mapping(
    audit: Audit,
    value: Any,
    expected: Mapping[str, float],
    *,
    check: str,
    label: str,
) -> dict[str, float]:
    mapping = _mapping(value)
    audit.require(check, set(mapping) == set(expected), f"{label} must contain exactly {sorted(expected)}")
    parsed: dict[str, float] = {}
    for key, expected_value in expected.items():
        actual = _float(mapping.get(key))
        audit.require(check, actual is not None, f"{label}.{key} must be finite numeric evidence")
        if actual is not None:
            parsed[key] = actual
            audit.require(
                check,
                math.isclose(actual, expected_value, rel_tol=1.0e-12, abs_tol=1.0e-14),
                f"{label}.{key} must be {expected_value}, got {actual}",
            )
    return parsed


def _audit_contract_semantics(audit: Audit, contract: Mapping[str, Any]) -> dict[str, Any]:
    check = "system_contract"
    audit.require(check, contract.get("source_doi") == "10.3390/math11030591", "wrong MAVPD source DOI")
    audit.require(check, contract.get("source_scope") == "published_model_equations_only", "wrong source_scope")
    audit.require(check, contract.get("candidate_parameter_set_published") is False, "candidate tuple must not be marked published")
    audit.require(check, contract.get("report_values_used_as_search_input") is False, "report values must not be search inputs")
    audit.require(check, contract.get("mathematica_validation") == "validation/wolfram/cases/mavpd_integer.wl", "wrong Mathematica validation source")
    audit.require(check, _close(contract.get("q"), 1.0, rtol=0.0), "contract q must equal 1")
    audit.require(check, tuple(_sequence(contract.get("equations"))) == EXPECTED_EQUATIONS, "contract equations differ from MAVPD")
    base = _audit_parameter_mapping(
        audit,
        contract.get("base_parameters"),
        EXPECTED_BASE_PARAMETERS,
        check=check,
        label="base_parameters",
    )
    jacobian_residual = _float(contract.get("jacobian_residual"))
    audit.require(
        check,
        jacobian_residual is not None and 0.0 <= jacobian_residual <= 1.0e-7,
        f"invalid Jacobian residual: {contract.get('jacobian_residual')!r}",
    )

    lure = _mapping(contract.get("lure"))
    expected_matrix = ((10.0, 100.0, 0.0), (1.0, -3.1, -1.0), (0.0, 200.0, 0.0))
    matrix = _sequence(lure.get("A"))
    matrix_ok = len(matrix) == 3 and all(
        _finite_vector(row, 3) is not None
        and all(_close(actual, expected, atol=1.0e-13) for actual, expected in zip(_sequence(row), expected_row))
        for row, expected_row in zip(matrix, expected_matrix)
    )
    audit.require(check, matrix_ok, "contract Lur'e matrix A differs from the declared transformation")
    audit.require(check, _finite_vector(lure.get("b"), 3) == (-100.0, 0.0, 0.0), "contract Lur'e b is wrong")
    audit.require(check, _finite_vector(lure.get("c"), 3) == (1.0, 0.0, 0.0), "contract Lur'e c is wrong")
    audit.require(check, lure.get("psi") == "sigma^3", "contract nonlinearity must be sigma^3")
    audit.require(check, lure.get("describing_function") == "N(a)=3*a^2/4", "wrong describing function")
    field_residual = _float(lure.get("max_field_residual"))
    audit.require(check, field_residual is not None and 0.0 <= field_residual <= 1.0e-10, "Lur'e field residual is invalid")

    expected_base_equilibria = {
        "E0": (0.0, 0.0, 0.0),
        "E+": (math.sqrt(0.1), 0.0, math.sqrt(0.1)),
        "E-": (-math.sqrt(0.1), 0.0, -math.sqrt(0.1)),
    }
    equilibrium_rows = [row for row in _sequence(contract.get("equilibria")) if isinstance(row, Mapping)]
    equilibrium_index = {str(row.get("name")): row for row in equilibrium_rows}
    audit.require(check, len(equilibrium_rows) == 3 and set(equilibrium_index) == set(expected_base_equilibria), "contract must record E0, E+, E-")
    for name, expected_state in expected_base_equilibria.items():
        row = equilibrium_index.get(name, {})
        state = _finite_vector(row.get("state"), 3)
        audit.require(
            check,
            state is not None and all(_close(left, right, atol=1.0e-13) for left, right in zip(state, expected_state)),
            f"wrong base equilibrium state for {name}",
        )
        residual = _float(row.get("rhs_residual"))
        audit.require(check, residual is not None and 0.0 <= residual <= 1.0e-8, f"invalid equilibrium residual for {name}")
        eigenvalues = _sequence(row.get("eigenvalues"))
        audit.require(check, len(eigenvalues) == 3, f"{name} must have three eigenvalues")
        parsed_eigenvalues = tuple(_complex_number(item) for item in eigenvalues)
        audit.require(check, all(value is not None for value in parsed_eigenvalues), f"non-finite eigenvalue for {name}")
        linear_y1 = 10.0 - 300.0 * expected_state[0] * expected_state[0]
        coefficients = _mavpd_characteristic_coefficients(
            linear_y1=linear_y1,
            delta=100.0,
            rho=200.0,
            xi=3.1,
        )
        _audit_characteristic_spectrum(
            audit,
            check=check,
            label=f"base-system Jacobian spectrum for {name}",
            values=parsed_eigenvalues,
            coefficients=coefficients,
        )

    seed_rows = [row for row in _sequence(contract.get("direct_seed_records")) if isinstance(row, Mapping)]
    seed_index: dict[int, Mapping[str, Any]] = {}
    for row in seed_rows:
        branch_value = _integer(row.get("branch_index"))
        audit.require(check, branch_value in {0, 1}, f"invalid direct seed branch_index: {row.get('branch_index')!r}")
        if branch_value is not None:
            audit.require(check, branch_value not in seed_index, f"duplicate direct seed branch_index: {branch_value}")
            seed_index[branch_value] = row
    audit.require(check, len(seed_rows) == 2 and set(seed_index) == {0, 1}, "direct seed records must contain branches 0 and 1")
    expected_seed_pairs = _mavpd_direct_seed_pairs()
    audit.require(check, len(expected_seed_pairs) == 2, "equation-derived direct route must have two admissible branches")
    seed_omegas: dict[int, float] = {}
    seed_vectors: dict[int, tuple[float, float, float]] = {}
    for branch in (0, 1):
        row = _mapping(seed_index.get(branch))
        omega = _float(row.get("omega0"))
        gain = _float(row.get("k"))
        amplitude = _float(row.get("a0"))
        seed = _finite_vector(row.get("seed"), 3)
        expected_omega, expected_gain = expected_seed_pairs[branch]
        audit.require(check, omega is not None and omega > 0.0, f"invalid omega0 for branch {branch}")
        audit.require(check, gain is not None and gain > 0.0, f"invalid k for branch {branch}")
        audit.require(check, amplitude is not None and amplitude > 0.0, f"invalid a0 for branch {branch}")
        audit.require(check, seed is not None, f"invalid seed vector for branch {branch}")
        audit.require(check, omega is not None and _close(omega, expected_omega, atol=1.0e-9), f"omega0 was not derived from the direct transfer polynomial for branch {branch}")
        audit.require(check, gain is not None and _close(gain, expected_gain, atol=1.0e-10), f"gain was not derived from W(i omega) for branch {branch}")
        if gain is not None and amplitude is not None:
            audit.require(check, _close(gain, 0.75 * amplitude * amplitude, atol=1.0e-12), f"k != 3*a0^2/4 for branch {branch}")
        audit.require(check, _close(row.get("phase"), 0.0, rtol=0.0), f"branch {branch} phase must be zero")
        audit.require(check, row.get("method") == "classic", f"branch {branch} must use the classic describing function")
        audit.require(check, row.get("frequency_grid_used") is False, f"branch {branch} used a frequency grid")
        audit.require(check, row.get("published_table_used") is False, f"branch {branch} claims a published table")
        audit.require(check, row.get("search_route") == "direct_integer_transfer", f"wrong route for branch {branch}")
        matched = _complex_number(row.get("matched_eigenvalue"))
        audit.require(
            check,
            matched is not None
            and abs(matched.real) <= 1.0e-9
            and omega is not None
            and _close(matched.imag, omega, atol=1.0e-9),
            f"matched harmonic eigenvalue is inconsistent for branch {branch}",
        )
        eigenvector_items = _sequence(row.get("eigenvector"))
        eigenvector = tuple(_complex_number(item) for item in eigenvector_items)
        audit.require(check, len(eigenvector) == 3 and all(value is not None for value in eigenvector), f"invalid eigenvector for branch {branch}")
        if len(eigenvector) == 3 and all(value is not None for value in eigenvector) and omega is not None and gain is not None:
            vector = tuple(value for value in eigenvector if value is not None)
            audit.require(check, abs(vector[0] - 1.0) <= 1.0e-10, f"branch {branch} eigenvector is not normalized by c^T v=1")
            linear_y1 = 10.0 - 100.0 * gain
            residual_vector = (
                (linear_y1 - 1j * omega) * vector[0] + 100.0 * vector[1],
                vector[0] + (-3.1 - 1j * omega) * vector[1] - vector[2],
                200.0 * vector[1] - 1j * omega * vector[2],
            )
            residual_scale = 1.0 + max(abs(value) for value in vector) * (200.0 + omega)
            audit.require(
                check,
                max(abs(value) for value in residual_vector) / residual_scale <= 1.0e-10,
                f"branch {branch} eigenvector does not satisfy (A+kbc^T-i omega I)v=0",
            )
            if amplitude is not None and seed is not None:
                expected_seed = tuple(amplitude * value.real for value in vector)
                audit.require(
                    check,
                    all(_close(left, right, atol=1.0e-10) for left, right in zip(seed, expected_seed)),
                    f"branch {branch} seed is not a0*Re(v) at phase zero",
                )
        if omega is not None:
            seed_omegas[branch] = omega
        if seed is not None:
            seed_vectors[branch] = seed
    if len(seed_omegas) == 2:
        audit.require(check, seed_omegas[0] < seed_omegas[1], "branch ordering must follow increasing harmonic frequency")

    transfer_rows = [row for row in _sequence(contract.get("transfer_checks")) if isinstance(row, Mapping)]
    transfer_index: dict[int, Mapping[str, Any]] = {}
    for row in transfer_rows:
        branch_value = _integer(row.get("branch_index"))
        audit.require(check, branch_value in {0, 1}, f"invalid transfer-check branch_index: {row.get('branch_index')!r}")
        if branch_value is not None:
            audit.require(check, branch_value not in transfer_index, f"duplicate transfer-check branch_index: {branch_value}")
            transfer_index[branch_value] = row
    audit.require(check, len(transfer_rows) == 2 and set(transfer_index) == {0, 1}, "transfer checks must cover branches 0 and 1")
    for branch in (0, 1):
        row = _mapping(transfer_index.get(branch))
        for field in ("imaginary_residual", "closure_residual", "describing_function_residual"):
            residual = _float(row.get(field))
            audit.require(check, residual is not None and 0.0 <= residual <= 1.0e-9, f"invalid {field} for branch {branch}")
        if branch in seed_omegas:
            audit.require(check, _close(row.get("omega0"), seed_omegas[branch]), f"transfer omega mismatch for branch {branch}")
        transfer_value = _complex_number(row.get("W_iomega"))
        seed_gain = _float(_mapping(seed_index.get(branch)).get("k"))
        recomputed_transfer = _mavpd_transfer_response(seed_omegas[branch]) if branch in seed_omegas else None
        audit.require(
            check,
            transfer_value is not None
            and seed_gain is not None
            and recomputed_transfer is not None
            and abs(transfer_value - recomputed_transfer) <= 1.0e-9
            and _close(transfer_value.real, -1.0 / seed_gain, atol=1.0e-9)
            and abs(transfer_value.imag) <= 1.0e-9,
            f"transfer closure value W(i omega) is inconsistent for branch {branch}",
        )

    expected_hopf = _mavpd_high_hopf_boundary()
    hopf = _mapping(contract.get("hopf_boundary_at_xi_target"))
    audit.require(check, hopf.get("values_derived_from_equations") is True, "Hopf/Routh values must be equation-derived")
    audit.require(check, _close(hopf.get("selected_high_gamma_boundary"), expected_hopf), "wrong high Hopf/Routh boundary")
    boundaries = tuple(_float(value) for value in _sequence(hopf.get("gamma_boundaries")))
    audit.require(check, len(boundaries) == 2 and all(value is not None for value in boundaries), "two finite Hopf/Routh boundaries are required")
    coefficient_b = 2.85 * 2.85 - 100.0
    discriminant = coefficient_b * coefficient_b - 4.0 * 2.85 * 2.85 * (200.0 - 100.0)
    expected_boundaries = tuple(
        sorted(
            (
                (-coefficient_b - math.sqrt(discriminant)) / (2.0 * 2.85) / 200.0,
                (-coefficient_b + math.sqrt(discriminant)) / (2.0 * 2.85) / 200.0,
            )
        )
    )
    audit.require(
        check,
        len(boundaries) == 2
        and all(value is not None for value in boundaries)
        and all(_close(left, right, atol=1.0e-12) for left, right in zip(sorted(float(value) for value in boundaries if value is not None), expected_boundaries)),
        "Hopf/Routh boundaries do not match the equation-derived quadratic",
    )
    routh_residual = _float(hopf.get("routh_hurwitz_residual_at_boundary"))
    audit.require(check, routh_residual is not None and abs(routh_residual) <= 1.0e-8, "invalid Routh-Hurwitz residual")
    return {
        "base_parameters": base,
        "base_equilibria": expected_base_equilibria,
        "seed_omegas": seed_omegas,
        "seed_vectors": seed_vectors,
        "hopf_boundary": expected_hopf,
    }


def _screen_row_values(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "hopf_offset": _float(row.get("hopf_offset")),
        "gamma": _float(row.get("gamma")),
        "equilibrium_stability_margin": _float(row.get("equilibrium_stability_margin")),
        "lambda_1": _float(row.get("lambda_1")),
        "lambda_2": _float(row.get("lambda_2")),
        "lambda_3": _float(row.get("lambda_3")),
        "lyapunov_status": row.get("lyapunov_status"),
        "E0_probe_count": _integer(row.get("E0_probe_count")),
        "E0_target_hits": _integer(row.get("E0_target_hits")),
        "E0_ambiguous": _integer(row.get("E0_ambiguous")),
        "eligible_hidden_chaos_screen": _bool_text(row.get("eligible_hidden_chaos_screen")),
    }


def _audit_route_and_selection(
    audit: Audit,
    route: Mapping[str, Any],
    continuation_rows: Sequence[Mapping[str, str]],
    screen_rows: Sequence[Mapping[str, str]],
    screen_probe_rows: Sequence[Mapping[str, str]],
    screen_contract: Mapping[str, Any],
    selection: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contract_info: Mapping[str, Any],
) -> dict[str, Any]:
    check = "candidate_lineage"
    audit.require(check, route.get("primary_route") == "direct_integer_lure", "01 primary_route must be direct_integer_lure")
    audit.require(check, route.get("frequency_grid_used") is False, "01 records a frequency-grid search")
    audit.require(check, route.get("alternative_triggered") is True, "alternative must be triggered after direct failure")
    audit.require(
        check,
        route.get("trigger") == "neither direct base branch passes the declared finite-time LLE chaos screen",
        "01 alternative trigger text differs from the declared rule",
    )
    branches = [row for row in _sequence(route.get("branches")) if isinstance(row, Mapping)]
    branch_index: dict[int, Mapping[str, Any]] = {}
    for row in branches:
        branch_value = _integer(row.get("branch_index"))
        audit.require(check, branch_value in {0, 1}, f"invalid route branch_index: {row.get('branch_index')!r}")
        if branch_value is not None:
            audit.require(check, branch_value not in branch_index, f"duplicate route branch_index: {branch_value}")
            branch_index[branch_value] = row
    audit.require(check, len(branches) == 2 and set(branch_index) == {0, 1}, "01 must record exactly branches 0 and 1")
    successful: list[tuple[float, int]] = []
    for branch in (0, 1):
        row = _mapping(branch_index.get(branch))
        lyapunov = _mapping(row.get("lyapunov"))
        exponents = _finite_vector(lyapunov.get("exponents"), 3)
        status = lyapunov.get("status")
        audit.require(check, isinstance(status, str) and bool(status), f"base branch {branch} lacks a Lyapunov status")
        audit.require(check, lyapunov.get("method") == "integer_dop853_variational_qr", f"base branch {branch} uses the wrong Lyapunov method")
        audit.require(check, lyapunov.get("finite_time_local") is True, f"base branch {branch} Lyapunov evidence is not finite-time local")
        audit.require(check, lyapunov.get("does_not_prove_chaos_alone") is True, f"base branch {branch} Lyapunov evidence overclaims")
        metadata = _mapping(lyapunov.get("metadata"))
        audit.require(check, metadata.get("solver_method") == "DOP853", f"base branch {branch} Lyapunov solver is wrong")
        audit.require(check, metadata.get("solver") == "scipy.integrate.solve_ivp", f"base branch {branch} solver implementation is wrong")
        audit.require(check, metadata.get("jacobian_source") == "analytic", f"base branch {branch} did not use the analytic Jacobian")
        audit.require(check, metadata.get("jacobian_eps") is None, f"base branch {branch} unexpectedly used a finite-difference Jacobian epsilon")
        audit.require(check, _integer(metadata.get("dimension")) == 3, f"base branch {branch} Lyapunov dimension is wrong")
        audit.require(check, _close(metadata.get("div_threshold"), 50.0), f"base branch {branch} divergence threshold is wrong")
        audit.require(check, _close(metadata.get("max_step"), 0.02), f"base branch {branch} max_step is wrong")
        audit.require(check, _close(metadata.get("rtol"), 2.0e-10), f"base branch {branch} rtol is wrong")
        audit.require(check, _close(metadata.get("atol"), 2.0e-12), f"base branch {branch} atol is wrong")
        audit.require(check, _close(metadata.get("qr_interval"), 0.5), f"base branch {branch} QR interval is wrong")
        audit.require(check, _close(metadata.get("t_burn_requested"), 100.0), f"base branch {branch} requested burn is wrong")
        audit.require(check, _close(metadata.get("t_accumulate_requested"), 250.0), f"base branch {branch} requested horizon is wrong")
        burn_completed = _float(metadata.get("t_burn_completed"))
        accumulation_completed = _float(metadata.get("t_accumulate_completed"))
        qr_segments = _integer(metadata.get("qr_segments"))
        audit.require(check, burn_completed is not None and 0.0 <= burn_completed <= 100.0, f"base branch {branch} has invalid completed burn")
        audit.require(check, accumulation_completed is not None and 0.0 <= accumulation_completed <= 250.0, f"base branch {branch} has invalid completed accumulation")
        audit.require(check, qr_segments is not None and 0 <= qr_segments <= 500, f"base branch {branch} has invalid QR segment count")
        audit.require(check, accumulation_completed is not None and _close(lyapunov.get("t_accumulate"), accumulation_completed), f"base branch {branch} recorded horizon differs from metadata")
        if status == "ok":
            audit.require(check, exponents is not None, f"successful base branch {branch} lacks finite Lyapunov exponents")
            audit.require(check, _close(burn_completed, 100.0), f"successful base branch {branch} burn is incomplete")
            audit.require(check, _close(accumulation_completed, 250.0), f"successful base branch {branch} accumulation is incomplete")
            audit.require(check, qr_segments == 500, f"successful base branch {branch} QR segment count is incomplete")
            audit.require(check, _finite_vector(lyapunov.get("final_state"), 3) is not None, f"successful base branch {branch} final Lyapunov state is invalid")
        if exponents is not None:
            audit.require(check, _close(lyapunov.get("sum_exponents"), sum(exponents), atol=1.0e-10), f"base branch {branch} exponent sum is inconsistent")
        derived_chaotic = bool(status == "ok" and exponents is not None and exponents[0] > 0.02)
        audit.require(check, row.get("chaotic_at_base") is derived_chaotic, f"base branch {branch} chaotic_at_base is not derived from lambda_1")
        audit.require(check, derived_chaotic is False, f"base branch {branch} unexpectedly qualified as chaotic")
        audit.require(check, _integer(row.get("lambda_nodes_completed")) == 21, f"branch {branch} did not complete 21 lambda nodes")
        audit.require(check, _close(row.get("final_lambda"), 1.0, rtol=0.0), f"branch {branch} did not reach lambda=1")
        audit.require(check, _finite_vector(row.get("final_state"), 3) is not None, f"branch {branch} final state is invalid")
        omega = _float(row.get("omega0"))
        audit.require(check, omega is not None and _close(omega, _mapping(contract_info.get("seed_omegas")).get(branch)), f"branch {branch} omega differs from contract")
        if omega is not None and status == "ok":
            successful.append((omega, branch))
    audit.require(check, bool(successful), "no successful direct base branch is available for the alternative route")
    expected_source_branch = min(successful)[1] if successful else None
    route_source = _integer(route.get("alternative_source_branch_index"))
    selection_source = _integer(selection.get("alternative_source_branch_index"))
    audit.require(check, route_source == expected_source_branch, "alternative source is not the lowest-frequency successful branch")
    audit.require(check, selection_source == expected_source_branch, "selection source branch differs from route")
    audit.require(check, selection.get("primary_route_chaos_screen_failed_before_alternative") is True, "selection does not record primary-route failure")
    audit.require(
        check,
        selection.get("parameter_provenance")
        == "gamma_selected_by_local_continuation_at_declared_xi_endpoint_not_a_published_parameter_tuple",
        "selection candidate-parameter provenance is wrong",
    )
    audit.require(
        check,
        selection.get("xi_endpoint_provenance")
        == "declared_local_continuation_endpoint_not_selected_by_candidate_screen",
        "selection xi provenance is wrong",
    )
    expected_rule = "lowest direct harmonic frequency among successful direct base branches"
    audit.require(check, route.get("alternative_source_branch_selection_rule") == expected_rule, "wrong source-branch rule in 01")
    audit.require(check, selection.get("alternative_source_branch_selection_rule") == expected_rule, "wrong source-branch rule in selection")

    audit.require(check, screen_contract.get("finite_sample_only") is True, "screening contract must be finite-sample only")
    audit.require(
        check,
        screen_contract.get("candidate_states_source") == "02_parameter_continuation.csv x_out at each exact Hopf-offset node",
        "screening candidate-state provenance is wrong",
    )
    audit.require(check, screen_contract.get("probe_initial_conditions") == "03_candidate_screening_probes.csv", "wrong screening probe provenance")
    expected_screen_contract = {
        "lyapunov": {
            "t_burn": 100.0,
            "t_accumulate": 250.0,
            "qr_interval": 0.5,
            "rtol": 2.0e-10,
            "atol": 2.0e-12,
            "max_step": 0.02,
            "positive_threshold": 0.5,
        },
        "reference": {
            "duration": 360.0,
            "burn": 160.0,
            "sample_step": 0.05,
            "max_step": 0.03,
            "safety_factor": 3.0,
            "max_points": 600,
        },
        "probes": {
            "equilibrium_names": ["E0"],
            "radii": [1.0e-7, 1.0e-4],
            "directions": 6,
            "sampling_mode": "sphere",
            "t_burn": 500.0,
            "t_keep": 100.0,
            "sample_step": 0.05,
            "rtol": 2.0e-10,
            "atol": 2.0e-12,
            "max_step": 0.03,
            "equilibrium_tol": 1.0e-6,
            "equilibrium_tail_span_tol": 1.0e-5,
        },
    }
    audit.require(
        check,
        _same_json_semantics(screen_contract.get("contract"), expected_screen_contract),
        "03_candidate_screening_contract.json does not contain the exact declared finite screening contract",
    )

    parsed_continuation: list[dict[str, Any]] = []
    previous_out: tuple[float, ...] | None = None
    previous_stage: str | None = None
    stage_indices: dict[str, list[int]] = {"xi": [], "gamma": []}
    for index, row in enumerate(continuation_rows, start=2):
        stage = row.get("stage", "")
        node_index = _integer(row.get("node_index"))
        xi = _float(row.get("xi"))
        gamma = _float(row.get("gamma"))
        x_in = _finite_vector(_parse_json_cell(audit, row.get("x_in"), check=check, label=f"02 row {index} x_in"), 3)
        x_out = _finite_vector(_parse_json_cell(audit, row.get("x_out"), check=check, label=f"02 row {index} x_out"), 3)
        audit.require(check, stage in {"xi", "gamma"}, f"invalid continuation stage at row {index}")
        audit.require(check, node_index is not None and xi is not None and gamma is not None, f"non-finite continuation coordinates at row {index}")
        audit.require(check, x_in is not None and x_out is not None, f"non-finite continuation state at row {index}")
        audit.require(check, row.get("status") == "ok", f"continuation row {index} is not ok")
        audit.require(check, _bool_text(row.get("system_rebuilt_at_node")) is True, f"system was not rebuilt at row {index}")
        audit.require(check, _bool_text(row.get("lure_rebuilt_at_node")) is True, f"Lur'e form was not rebuilt at row {index}")
        if node_index is not None and stage in stage_indices:
            stage_indices[stage].append(node_index)
        if previous_out is not None and x_in is not None:
            audit.require(
                check,
                all(_close(left, right, atol=1.0e-11) for left, right in zip(previous_out, x_in)),
                f"continuation state chain breaks at row {index}",
            )
        if previous_stage == "gamma" and stage == "xi":
            audit.fail(check, "xi rows cannot follow gamma rows")
        if x_out is not None:
            previous_out = x_out
        previous_stage = stage
        parsed_continuation.append({"stage": stage, "node_index": node_index, "xi": xi, "gamma": gamma, "x_in": x_in, "x_out": x_out})
    audit.require(check, bool(parsed_continuation), "02_parameter_continuation.csv has no rows")
    for stage, indices in stage_indices.items():
        audit.require(check, indices == list(range(len(indices))) and bool(indices), f"{stage} node_index values are not contiguous from zero")
    xi_rows = [row for row in parsed_continuation if row["stage"] == "xi"]
    gamma_rows = [row for row in parsed_continuation if row["stage"] == "gamma"]
    audit.require(check, len(xi_rows) == 25, f"xi continuation must contain 25 nodes, got {len(xi_rows)}")
    if xi_rows:
        expected_xi = tuple(3.09 - 0.01 * index for index in range(25))
        audit.require(
            check,
            len(xi_rows) == len(expected_xi)
            and all(_close(row["xi"], expected, atol=1.0e-12) for row, expected in zip(xi_rows, expected_xi)),
            "xi continuation does not follow every declared 0.01 node from 3.09 to 2.85",
        )
        audit.require(check, all(_close(row["gamma"], 0.1) for row in xi_rows), "gamma must remain 0.1 along xi continuation")
        audit.require(check, all(xi_rows[i]["xi"] > xi_rows[i + 1]["xi"] for i in range(len(xi_rows) - 1)), "xi continuation is not strictly decreasing")
        source_final_state = _finite_vector(_mapping(branch_index.get(expected_source_branch)).get("final_state"), 3)
        audit.require(
            check,
            source_final_state is not None
            and xi_rows[0]["x_in"] is not None
            and all(_close(left, right, atol=1.0e-11) for left, right in zip(source_final_state, xi_rows[0]["x_in"])),
            "first xi continuation state is not the selected direct branch endpoint",
        )
    audit.require(check, bool(gamma_rows), "gamma continuation has no rows")
    if gamma_rows:
        audit.require(check, all(_close(row["xi"], 2.85) for row in gamma_rows), "xi must remain 2.85 along gamma continuation")
        audit.require(check, all(gamma_rows[i]["gamma"] < gamma_rows[i + 1]["gamma"] for i in range(len(gamma_rows) - 1)), "gamma continuation is not strictly increasing")

    parsed_screens = [_screen_row_values(row) for row in screen_rows]
    offsets = tuple(item["hopf_offset"] for item in parsed_screens)
    audit.require(
        check,
        len(parsed_screens) == len(EXPECTED_SCREEN_OFFSETS)
        and all(_close(left, right) for left, right in zip(offsets, EXPECTED_SCREEN_OFFSETS)),
        f"screen offsets must equal {list(EXPECTED_SCREEN_OFFSETS)}, got {list(offsets)}",
    )
    hopf_boundary = float(contract_info["hopf_boundary"])
    largest_gamma = hopf_boundary + max(EXPECTED_SCREEN_OFFSETS)
    base_gamma_nodes: list[float] = []
    node_index = 1
    while 0.1 + 0.0025 * node_index < largest_gamma - 1.0e-14:
        base_gamma_nodes.append(0.1 + 0.0025 * node_index)
        node_index += 1
    expected_gamma_nodes = sorted(base_gamma_nodes + [hopf_boundary + value for value in EXPECTED_SCREEN_OFFSETS])
    audit.require(
        check,
        len(gamma_rows) == len(expected_gamma_nodes)
        and all(_close(row["gamma"], expected, atol=1.0e-11) for row, expected in zip(gamma_rows, expected_gamma_nodes)),
        f"gamma continuation must contain the exact {len(expected_gamma_nodes)} base/offset nodes",
    )
    screen_by_offset: dict[float, dict[str, Any]] = {}
    for item in parsed_screens:
        finite_core = all(item[key] is not None for key in ("hopf_offset", "gamma", "equilibrium_stability_margin", "lambda_1", "lambda_2", "lambda_3"))
        audit.require(check, finite_core, f"screen row has non-finite values: {item}")
        if item["hopf_offset"] is not None:
            audit.require(check, _close(item["gamma"], hopf_boundary + item["hopf_offset"]), "screen gamma != gamma_H + offset")
        if item["hopf_offset"] is not None:
            screen_by_offset[float(item["hopf_offset"])] = item

    # Screening probes independently establish every screen eligibility count.
    probes_by_offset: dict[float, list[Mapping[str, str]]] = {value: [] for value in EXPECTED_SCREEN_OFFSETS}
    thresholds_by_offset: dict[float, list[float]] = {value: [] for value in EXPECTED_SCREEN_OFFSETS}
    for index, row in enumerate(screen_probe_rows, start=2):
        offset = _radius_key(row.get("hopf_offset"), EXPECTED_SCREEN_OFFSETS)
        audit.require(check, offset is not None, f"unexpected screening-probe offset at row {index}")
        if offset is None:
            continue
        probes_by_offset[offset].append(row)
        audit.require(check, _close(row.get("gamma"), float(contract_info["hopf_boundary"]) + offset), f"screening probe gamma differs from its offset at row {index}")
        audit.require(check, row.get("contract") == "candidate_screen_E0", f"wrong screening probe contract at row {index}")
        audit.require(check, row.get("equilibrium") == "E0", f"screening probe is not centered on E0 at row {index}")
        audit.require(check, row.get("sampling_mode") == "sphere", f"screening probe is not sphere-sampled at row {index}")
        audit.require(check, bool(row.get("status")), f"screening probe lacks status at row {index}")
        audit.require(check, _bool_text(row.get("target_hit")) is not None, f"invalid screening target flag at row {index}")
        audit.require(check, _bool_text(row.get("ambiguous")) is not None, f"invalid screening ambiguous flag at row {index}")
        classification = row.get("target_classification")
        reference_threshold = _float(row.get("reference_acceptance_threshold"))
        target_distance = _float(row.get("target_distance_norm"))
        audit.require(check, reference_threshold is not None and reference_threshold > 0.0, f"invalid screening reference threshold at row {index}")
        audit.require(check, target_distance is not None and target_distance >= 0.0, f"invalid screening target distance at row {index}")
        if reference_threshold is not None:
            thresholds_by_offset[offset].append(reference_threshold)
        expected_classification = None
        if reference_threshold is not None and target_distance is not None:
            ambiguity_upper = reference_threshold + reference_threshold / 3.0
            if target_distance <= reference_threshold:
                expected_classification = "same_attractor_under_calibrated_cloud_test"
            elif target_distance < ambiguity_upper:
                expected_classification = "inconclusive"
            else:
                expected_classification = "different_from_target_under_calibrated_cloud_test"
            audit.require(
                check,
                classification == expected_classification,
                f"screening classification disagrees with the calibrated threshold/margin at row {index}",
            )
        expected_hit = expected_classification == "same_attractor_under_calibrated_cloud_test"
        expected_ambiguous = expected_classification == "inconclusive"
        audit.require(check, _bool_text(row.get("target_hit")) is expected_hit, f"screening target_hit disagrees with classification at row {index}")
        audit.require(check, _bool_text(row.get("ambiguous")) is expected_ambiguous, f"screening ambiguous flag disagrees with classification at row {index}")
        destination = row.get("destination", "")
        audit.require(
            check,
            destination.startswith("equilibrium_") or destination == classification,
            f"screening destination disagrees with classification at row {index}",
        )
    for offset, rows in probes_by_offset.items():
        audit.require(check, len(rows) == 12, f"screen offset {offset:g} must have 12 E0 probes")
        thresholds = thresholds_by_offset[offset]
        audit.require(
            check,
            len(thresholds) == len(rows)
            and bool(thresholds)
            and all(_close(value, thresholds[0], atol=1.0e-14) for value in thresholds),
            f"screen offset {offset:g} does not use one consistent calibrated reference threshold",
        )
        parsed_ids = [_integer(row.get("sample_id")) for row in rows]
        audit.require(
            check,
            all(value is not None for value in parsed_ids)
            and sorted(int(value) for value in parsed_ids if value is not None) == list(range(12)),
            f"screen offset {offset:g} sample IDs must be 0..11",
        )
        cells: dict[float, set[str]] = {1.0e-7: set(), 1.0e-4: set()}
        expected_directions = _deterministic_unit_directions_3d(6)
        for row in rows:
            radius = _radius_key(row.get("radius"), tuple(cells))
            direction = row.get("direction_id", "")
            if radius is not None:
                cells[radius].add(direction)
            x0 = _finite_vector(_parse_json_cell(audit, row.get("x0"), check=check, label=f"screen offset {offset:g} x0"), 3)
            audit.require(
                check,
                x0 is not None and radius is not None and _close(math.sqrt(sum(value * value for value in x0)), radius, atol=1.0e-12),
                f"screening x0 is not on its declared E0 sphere at offset {offset:g}",
            )
            parsed_direction = _integer(direction)
            if x0 is not None and radius is not None and parsed_direction is not None and 1 <= parsed_direction <= 6:
                observed_direction = tuple(value / radius for value in x0)
                expected_direction = expected_directions[parsed_direction - 1]
                audit.require(
                    check,
                    all(_close(left, right, atol=2.0e-9) for left, right in zip(observed_direction, expected_direction)),
                    f"screening direction {direction} differs from the deterministic workflow contract at offset {offset:g}",
                )
        for radius, directions in cells.items():
            audit.require(check, directions == {str(value) for value in range(1, 7)}, f"incomplete screening cell {offset:g}/{radius:g}")

    eligible_rows: list[dict[str, Any]] = []
    for offset, item in screen_by_offset.items():
        rows = probes_by_offset.get(offset, [])
        target_hits = sum(_bool_text(row.get("target_hit")) is True for row in rows)
        ambiguous = sum(_bool_text(row.get("ambiguous")) is True for row in rows)
        failures = sum(row.get("status") != "ok" for row in rows)
        audit.require(check, item.get("E0_probe_count") == len(rows), f"screen probe count mismatch at offset {offset:g}")
        audit.require(check, item.get("E0_target_hits") == target_hits, f"screen target-hit count mismatch at offset {offset:g}")
        audit.require(check, item.get("E0_ambiguous") == ambiguous, f"screen ambiguous count mismatch at offset {offset:g}")
        recomputed = bool(
            item.get("lyapunov_status") == "ok"
            and item.get("lambda_1") is not None
            and float(item["lambda_1"]) > 0.5
            and target_hits == 0
            and ambiguous == 0
            and failures == 0
        )
        audit.require(check, item.get("eligible_hidden_chaos_screen") is recomputed, f"screen eligibility is inconsistent at offset {offset:g}")
        if recomputed:
            eligible_rows.append(item)
    audit.require(check, bool(eligible_rows), "no screen row is eligible")
    selected_expected = max(eligible_rows, key=lambda item: float(item["hopf_offset"])) if eligible_rows else {}
    selected = _mapping(selection.get("selected"))
    for key in ("hopf_offset", "gamma", "equilibrium_stability_margin", "lambda_1", "lambda_2", "lambda_3"):
        audit.require(check, _close(selected.get(key), selected_expected.get(key)), f"selected.{key} differs from the largest eligible screen row")
    audit.require(check, selected.get("lyapunov_status") == selected_expected.get("lyapunov_status"), "selected.lyapunov_status differs from its screen row")
    for key in ("E0_probe_count", "E0_target_hits", "E0_ambiguous"):
        audit.require(check, _integer(selected.get(key)) == selected_expected.get(key), f"selected.{key} differs from its screen row")
    audit.require(check, selected.get("eligible_hidden_chaos_screen") is selected_expected.get("eligible_hidden_chaos_screen"), "selected eligibility differs from its screen row")
    audit.require(check, selection.get("selection_rule") == EXPECTED_SELECTION_RULE, "selection rule is not the declared largest-eligible rule")
    audit.require(check, _close(selection.get("hopf_boundary"), hopf_boundary), "selection Hopf boundary is wrong")
    selected_gamma = _float(selected.get("gamma"))
    selected_state = _finite_vector(selection.get("selected_candidate_initial_state"), 3)
    audit.require(check, selected_state is not None, "selected candidate initial state must be a finite 3-vector")
    selected_nodes = [row for row in gamma_rows if selected_gamma is not None and _close(row["gamma"], selected_gamma, atol=1.0e-11)]
    audit.require(check, len(selected_nodes) == 1, f"selected gamma must identify one continuation node, got {len(selected_nodes)}")
    if len(selected_nodes) == 1 and selected_state is not None:
        audit.require(
            check,
            selected_nodes[0]["x_out"] is not None
            and all(_close(left, right, atol=1.0e-12) for left, right in zip(selected_state, selected_nodes[0]["x_out"])),
            "selected state differs from its continuation x_out",
        )

    expected_candidate = {
        "delta": 100.0,
        "gamma": float(selected_gamma) if selected_gamma is not None else float("nan"),
        "rho": 200.0,
        "xi": 2.85,
    }
    candidate_parameters = _mapping(manifest.get("candidate_parameters"))
    for key, expected in expected_candidate.items():
        audit.require(check, _close(candidate_parameters.get(key), expected), f"manifest candidate parameter {key} is inconsistent")
    audit.require(check, _close(manifest.get("hopf_boundary"), hopf_boundary), "manifest Hopf boundary differs from selection")
    audit.require(check, _close(manifest.get("selected_hopf_offset"), selected.get("hopf_offset")), "manifest Hopf offset differs from selection")
    audit.require(check, _integer(manifest.get("alternative_source_branch_index")) == expected_source_branch, "manifest source branch differs from route")
    audit.require(check, manifest.get("alternative_source_branch_selection_rule") == expected_rule, "manifest source-branch rule is wrong")
    audit.require(
        check,
        manifest.get("candidate_parameter_provenance")
        == "gamma_selected_at_declared_xi_continuation_endpoint_not_a_published_parameter_tuple",
        "manifest candidate-parameter provenance is wrong",
    )
    audit.require(
        check,
        manifest.get("xi_endpoint_provenance")
        == "declared_local_continuation_endpoint_not_selected_by_candidate_screen",
        "manifest xi provenance is wrong",
    )
    return {
        "hopf_boundary": hopf_boundary,
        "selected": selected,
        "selected_gamma": selected_gamma,
        "selected_state": selected_state,
        "candidate_parameters": expected_candidate,
        "source_branch_index": expected_source_branch,
        "theoretical_harmonic_seed": _mapping(contract_info.get("seed_vectors")).get(expected_source_branch),
    }


def _radius_key(value: Any, expected: Sequence[float]) -> float | None:
    parsed = _float(value)
    if parsed is None:
        return None
    for candidate in expected:
        if math.isclose(parsed, candidate, rel_tol=0.0, abs_tol=max(1.0e-15, abs(candidate) * 1.0e-12)):
            return candidate
    return None


def _audit_probe_csvs(
    audit: Audit,
    probes: Sequence[Mapping[str, str]],
    initial_conditions: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    *,
    candidate_equilibria: Mapping[str, tuple[float, float, float]],
    candidate_stability: Any,
    reference_threshold: Any,
    reference_ambiguity_margin: Any,
) -> dict[str, int]:
    specifications = {
        "main_3x3xN": {
            "equilibria": ("E0", "E+", "E-"),
            "radii": (1.0e-5, 1.0e-3, 1.0e-2),
            "directions": 12,
            "count": 108,
        },
        "targeted_E0_unstable": {
            "equilibria": ("E0",),
            "radii": (1.0e-7, 1.0e-4),
            "directions": 2,
            "count": 4,
        },
    }
    calibrated_threshold = _float(reference_threshold)
    calibrated_margin = _float(reference_ambiguity_margin)
    audit.require(
        "probe_outcomes",
        calibrated_threshold is not None and calibrated_threshold > 0.0,
        "hiddenness probes lack a valid calibrated reference threshold",
    )
    audit.require(
        "probe_outcomes",
        calibrated_margin is not None and calibrated_margin > 0.0,
        "hiddenness probes lack a valid calibrated ambiguity margin",
    )
    audit.require("probe_coverage", len(probes) == 112, f"expected 112 probe rows (108+4), got {len(probes)}")
    audit.require(
        "probe_coverage",
        len(initial_conditions) == 112,
        f"expected 112 initial-condition rows (108+4), got {len(initial_conditions)}",
    )
    required_probe_columns = {
        "contract",
        "sample_id",
        "equilibrium",
        "radius",
        "direction_id",
        "sampling_mode",
        "x0",
        "status",
        "destination",
        "target_classification",
        "target_distance_norm",
        "target_hit",
        "ambiguous",
        "tail_span",
        "closest_equilibrium",
        "closest_equilibrium_distance",
    }
    required_ic_columns = {
        "contract",
        "sample_id",
        "equilibrium",
        "radius",
        "direction_id",
        "y1",
        "y2",
        "y3",
    }
    if probes:
        audit.require(
            "probe_csv_schema",
            required_probe_columns <= set(probes[0]),
            f"07_hiddenness_probes.csv lacks columns: {sorted(required_probe_columns - set(probes[0]))}",
        )
    if initial_conditions:
        audit.require(
            "probe_csv_schema",
            required_ic_columns <= set(initial_conditions[0]),
            (
                "07_hiddenness_initial_conditions.csv lacks columns: "
                f"{sorted(required_ic_columns - set(initial_conditions[0]))}"
            ),
        )

    probe_keys: dict[tuple[str, str, str, float, str], Mapping[str, str]] = {}
    family_counts: dict[str, int] = {name: 0 for name in specifications}
    cells: dict[tuple[str, str, float], list[Mapping[str, str]]] = {}
    sample_ids: dict[str, list[int]] = {name: [] for name in specifications}
    allowed_destinations = {
        "equilibrium_E0",
        "equilibrium_E+",
        "equilibrium_E-",
        "different_from_target_under_calibrated_cloud_test",
    }
    for index, row in enumerate(probes, start=2):
        family = row.get("contract", "")
        spec = specifications.get(family)
        if not audit.require("probe_coverage", spec is not None, f"unknown probe contract at CSV row {index}: {family!r}"):
            continue
        assert spec is not None
        equilibrium = row.get("equilibrium", "")
        radius = _radius_key(row.get("radius"), spec["radii"])
        direction = row.get("direction_id", "")
        audit.require(
            "probe_coverage",
            equilibrium in spec["equilibria"],
            f"unexpected equilibrium at probe row {index}: {equilibrium!r}",
        )
        audit.require("probe_coverage", radius is not None, f"unexpected radius at probe row {index}: {row.get('radius')!r}")
        audit.require("probe_outcomes", row.get("status") == "ok", f"non-ok probe status at row {index}: {row.get('status')!r}")
        audit.require("probe_outcomes", row.get("sampling_mode") == "sphere", f"wrong sampling mode at row {index}")
        audit.require(
            "probe_outcomes",
            row.get("target_classification") == "different_from_target_under_calibrated_cloud_test",
            f"wrong target classification at probe row {index}",
        )
        audit.require(
            "probe_outcomes",
            _bool_text(row.get("target_hit")) is False,
            f"target hit or invalid target_hit flag at probe row {index}",
        )
        audit.require(
            "probe_outcomes",
            _bool_text(row.get("ambiguous")) is False,
            f"ambiguous or invalid ambiguous flag at probe row {index}",
        )
        audit.require(
            "probe_outcomes",
            row.get("destination", "") in allowed_destinations,
            f"invalid non-target destination at probe row {index}: {row.get('destination')!r}",
        )
        target_distance = _float(row.get("target_distance_norm"))
        tail_span = _float(row.get("tail_span"))
        closest_distance = _float(row.get("closest_equilibrium_distance"))
        closest = row.get("closest_equilibrium", "")
        audit.require("probe_outcomes", target_distance is not None and target_distance >= 0.0, f"invalid target distance at row {index}")
        if calibrated_threshold is not None and calibrated_margin is not None and target_distance is not None:
            audit.require(
                "probe_outcomes",
                target_distance >= calibrated_threshold + calibrated_margin,
                f"non-target probe lies inside the calibrated acceptance/ambiguity band at row {index}",
            )
        audit.require("probe_outcomes", tail_span is not None and tail_span >= 0.0, f"invalid tail span at row {index}")
        audit.require("probe_outcomes", closest_distance is not None and closest_distance >= 0.0, f"invalid equilibrium distance at row {index}")
        audit.require("probe_outcomes", closest in candidate_equilibria, f"unknown closest equilibrium at row {index}: {closest!r}")
        destination = row.get("destination", "")
        if destination.startswith("equilibrium_"):
            destination_name = destination.removeprefix("equilibrium_")
            audit.require("probe_outcomes", destination_name == closest, f"equilibrium destination/closest mismatch at row {index}")
            audit.require(
                "probe_outcomes",
                closest_distance is not None and closest_distance <= 1.0e-6 + 1.0e-12,
                f"equilibrium destination lies outside equilibrium tolerance at row {index}",
            )
            audit.require(
                "probe_outcomes",
                tail_span is not None and tail_span <= 1.0e-5 + 1.0e-12,
                f"equilibrium destination tail is not collapsed at row {index}",
            )
        sample = row.get("sample_id", "")
        parsed_sample = _integer(sample)
        if parsed_sample is not None:
            sample_ids[family].append(parsed_sample)
        else:
            audit.fail("probe_coverage", f"non-integer sample_id at probe row {index}: {sample!r}")
        if radius is None:
            continue
        key = (family, sample, equilibrium, radius, direction)
        audit.require("probe_coverage", key not in probe_keys, f"duplicate probe identity at row {index}: {key}")
        probe_keys[key] = row
        cells.setdefault((family, equilibrium, radius), []).append(row)
        family_counts[family] += 1

    for family, spec in specifications.items():
        audit.require(
            "probe_coverage",
            family_counts[family] == spec["count"],
            f"{family} requires {spec['count']} rows, got {family_counts[family]}",
        )
        audit.require(
            "probe_coverage",
            sorted(sample_ids[family]) == list(range(spec["count"])),
            f"{family} sample_id values must be exactly 0..{spec['count'] - 1}",
        )
        required_directions = {str(value) for value in range(1, spec["directions"] + 1)}
        for equilibrium in spec["equilibria"]:
            for radius in spec["radii"]:
                rows = cells.get((family, equilibrium, radius), [])
                directions = {row.get("direction_id", "") for row in rows}
                audit.require(
                    "probe_coverage",
                    len(rows) == spec["directions"],
                    f"incomplete cell {family}/{equilibrium}/{radius:g}: {len(rows)} rows",
                )
                audit.require(
                    "probe_coverage",
                    directions == required_directions,
                    f"wrong direction IDs in {family}/{equilibrium}/{radius:g}: {sorted(directions)}",
                )

    ic_keys: dict[tuple[str, str, str, float, str], Mapping[str, str]] = {}
    for index, row in enumerate(initial_conditions, start=2):
        family = row.get("contract", "")
        spec = specifications.get(family)
        if not audit.require("probe_coverage", spec is not None, f"unknown initial-condition contract at row {index}"):
            continue
        assert spec is not None
        radius = _radius_key(row.get("radius"), spec["radii"])
        if radius is None:
            audit.fail("probe_coverage", f"unexpected initial-condition radius at row {index}: {row.get('radius')!r}")
            continue
        key = (family, row.get("sample_id", ""), row.get("equilibrium", ""), radius, row.get("direction_id", ""))
        audit.require("probe_coverage", key not in ic_keys, f"duplicate initial-condition identity at row {index}: {key}")
        ic_keys[key] = row
    audit.require(
        "probe_coverage",
        set(ic_keys) == set(probe_keys),
        "probe and initial-condition CSV identity sets differ",
    )
    for key in set(ic_keys) & set(probe_keys):
        row = ic_keys[key]
        coordinates = [_float(row.get(axis)) for axis in ("y1", "y2", "y3")]
        x0 = _parse_json_cell(
            audit,
            probe_keys[key].get("x0", ""),
            check="probe_initial_conditions",
            label=f"probe x0 for {key}",
        )
        finite_x0 = _finite_vector(x0, 3)
        matches = (
            finite_x0 is not None
            and all(value is not None for value in coordinates)
            and all(
                math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-14)
                for left, right in zip(finite_x0, coordinates)
            )
        )
        audit.require("probe_initial_conditions", matches, f"x0 differs from initial-condition row for {key}")
        if matches:
            family, _sample, equilibrium, radius, direction = key
            center = candidate_equilibria.get(equilibrium)
            assert center is not None
            displacement = tuple(float(value) - center[index] for index, value in enumerate(coordinates))
            norm = math.sqrt(sum(value * value for value in displacement))
            audit.require(
                "probe_geometry",
                math.isclose(norm, radius, rel_tol=1.0e-9, abs_tol=max(1.0e-13, radius * 1.0e-9)),
                f"x0 is not on the declared sphere for {key}: norm={norm}",
            )

    main_directions: dict[str, tuple[float, float, float]] = {}
    targeted_directions: dict[str, tuple[float, float, float]] = {}
    for key, row in ic_keys.items():
        family, _sample, equilibrium, radius, direction = key
        center = candidate_equilibria.get(equilibrium)
        coordinates = _finite_vector([row.get(axis) for axis in ("y1", "y2", "y3")], 3)
        if center is None or coordinates is None or radius <= 0.0:
            continue
        unit = tuple((coordinates[index] - center[index]) / radius for index in range(3))
        target = main_directions if family == "main_3x3xN" else targeted_directions
        if direction in target:
            audit.require(
                "probe_geometry",
                all(_close(left, right, atol=2.0e-9) for left, right in zip(unit, target[direction])),
                f"direction {direction} is not reused consistently in {family}",
            )
        else:
            target[direction] = unit
    audit.require("probe_geometry", set(main_directions) == {str(value) for value in range(1, 13)}, "main direction basis is incomplete")
    audit.require("probe_geometry", set(targeted_directions) == {"1", "2"}, "targeted direction pair is incomplete")
    for label, vector in {**{f"main/{k}": v for k, v in main_directions.items()}, **{f"targeted/{k}": v for k, v in targeted_directions.items()}}.items():
        audit.require("probe_geometry", _close(math.sqrt(sum(value * value for value in vector)), 1.0, atol=2.0e-9), f"{label} is not unit length")
    expected_main_directions = _deterministic_unit_directions_3d(12)
    for direction_id, expected_direction in enumerate(expected_main_directions, start=1):
        observed_direction = main_directions.get(str(direction_id))
        audit.require(
            "probe_geometry",
            observed_direction is not None
            and all(_close(left, right, atol=2.0e-9) for left, right in zip(observed_direction, expected_direction)),
            f"main direction {direction_id} differs from the deterministic workflow contract",
        )
    main_vectors = list(main_directions.items())
    for index, (left_label, left) in enumerate(main_vectors):
        for right_label, right in main_vectors[index + 1 :]:
            separation = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
            audit.require(
                "probe_geometry",
                separation > 1.0e-8,
                f"main directions {left_label} and {right_label} are duplicates",
            )
    if set(targeted_directions) == {"1", "2"}:
        audit.require(
            "probe_geometry",
            all(_close(left, -right, atol=2.0e-9) for left, right in zip(targeted_directions["1"], targeted_directions["2"])),
            "targeted E0 directions are not antipodal",
        )
        recorded_direction = _finite_vector(_mapping(summary.get("targeted_E0_unstable_direction")).get("unstable_direction"), 3)
        audit.require("probe_geometry", recorded_direction is not None, "targeted summary lacks a finite unstable direction")
        if recorded_direction is not None:
            recorded_norm = math.sqrt(sum(value * value for value in recorded_direction))
            audit.require("probe_geometry", _close(recorded_norm, 1.0, atol=2.0e-9), "recorded unstable direction is not normalized")
            alignment = abs(sum(left * right for left, right in zip(recorded_direction, targeted_directions["1"])))
            audit.require("probe_geometry", _close(alignment, 1.0, atol=2.0e-8), "targeted probes do not follow the recorded unstable eigendirection")
            targeted_summary = _mapping(summary.get("targeted_E0_unstable_direction"))
            unstable_eigenvalue = _complex_number(targeted_summary.get("unstable_eigenvalue"))
            stability_rows = [row for row in _sequence(candidate_stability) if isinstance(row, Mapping)]
            e0_stability = next((row for row in stability_rows if row.get("equilibrium") == "E0"), {})
            e0_spectrum = tuple(_complex_number(item) for item in _sequence(_mapping(e0_stability).get("eigenvalues")))
            valid_e0_spectrum = tuple(value for value in e0_spectrum if value is not None)
            max_real = max((value.real for value in valid_e0_spectrum), default=float("nan"))
            audit.require(
                "probe_geometry",
                unstable_eigenvalue is not None
                and abs(unstable_eigenvalue.imag) <= 1.0e-9
                and unstable_eigenvalue.real > 0.0
                and _close(unstable_eigenvalue.real, max_real, atol=1.0e-9),
                "targeted unstable eigenvalue is not the maximum-real-part E0 Jacobian eigenvalue",
            )
            positive_state = candidate_equilibria.get("E+")
            gamma = positive_state[0] * positive_state[0] if positive_state is not None else None
            if unstable_eigenvalue is not None and gamma is not None:
                linear_y1 = 100.0 * gamma
                direction = tuple(recorded_direction)
                jacobian_action = (
                    linear_y1 * direction[0] + 100.0 * direction[1],
                    direction[0] - 2.85 * direction[1] - direction[2],
                    200.0 * direction[1],
                )
                residual = tuple(
                    jacobian_action[index] - unstable_eigenvalue * direction[index]
                    for index in range(3)
                )
                scale = 1.0 + (200.0 + abs(unstable_eigenvalue)) * max(abs(value) for value in direction)
                audit.require(
                    "probe_geometry",
                    max(abs(value) for value in residual) / scale <= 1.0e-10,
                    "recorded targeted direction is not an eigenvector of the candidate E0 Jacobian",
                )

    coverage = _mapping(summary.get("coverage_by_equilibrium_radius"))
    audit.require("probe_summary", coverage.get("complete") is True, "summary coverage.complete must be true")
    summary_specs = (
        ("main", "main_3x3xN"),
        ("targeted_E0_unstable_direction", "targeted_E0_unstable"),
    )
    for summary_name, family in summary_specs:
        spec = specifications[family]
        family_summary = _mapping(summary.get(summary_name))
        family_coverage = _mapping(coverage.get(summary_name))
        expected_cells = len(spec["equilibria"]) * len(spec["radii"])
        audit.require(
            "probe_summary",
            _integer(family_summary.get("n_probes")) == spec["count"],
            f"summary {summary_name}.n_probes must be {spec['count']}",
        )
        expected_equilibria = list(spec["equilibria"])
        audit.require("probe_summary", family_summary.get("required_equilibria") == expected_equilibria, f"summary {summary_name}.required_equilibria is wrong")
        audit.require("probe_summary", family_summary.get("tested_equilibria") == expected_equilibria, f"summary {summary_name}.tested_equilibria is wrong")
        audit.require("probe_summary", family_summary.get("tested_all_required_equilibria") is True, f"summary {summary_name} did not test all required equilibria")
        if summary_name == "main":
            audit.require("probe_summary", family_summary.get("declared_equilibria") == ["E0", "E+", "E-"], "main declared equilibria are wrong")
            audit.require("probe_summary", family_summary.get("tested_all_declared_equilibria") is True, "main did not test all declared equilibria")
        by_equilibrium = _mapping(family_summary.get("by_equilibrium"))
        audit.require("probe_summary", set(by_equilibrium) == set(spec["equilibria"]), f"summary {summary_name}.by_equilibrium is incomplete")
        for equilibrium in spec["equilibria"]:
            recorded = _mapping(by_equilibrium.get(equilibrium))
            source_rows = [row for row in probes if row.get("contract") == family and row.get("equilibrium") == equilibrium]
            expected_counts = {
                "n": len(source_rows),
                "target_hits": sum(_bool_text(row.get("target_hit")) is True for row in source_rows),
                "ambiguous": sum(_bool_text(row.get("ambiguous")) is True for row in source_rows),
                "numerical_failures": sum(row.get("status") != "ok" for row in source_rows),
                "equilibrium_destinations": sum(row.get("destination", "").startswith("equilibrium_") for row in source_rows),
            }
            for field, expected_value in expected_counts.items():
                audit.require("probe_summary", _integer(recorded.get(field)) == expected_value, f"summary {summary_name}/{equilibrium}.{field} is inconsistent")
        for field in ("target_hits", "ambiguous", "numerical_failures"):
            audit.require(
                "probe_summary",
                _integer(family_summary.get(field)) == 0,
                f"summary {summary_name}.{field} must be zero",
            )
        audit.require("probe_summary", family_coverage.get("complete") is True, f"coverage {summary_name} incomplete")
        audit.require(
            "probe_summary",
            _integer(family_coverage.get("expected_cells")) == expected_cells,
            f"coverage {summary_name}.expected_cells must be {expected_cells}",
        )
        audit.require(
            "probe_summary",
            _integer(family_coverage.get("expected_per_cell")) == spec["directions"],
            f"coverage {summary_name}.expected_per_cell must be {spec['directions']}",
        )
        recorded_cells = _sequence(family_coverage.get("cells"))
        audit.require(
            "probe_summary",
            len(recorded_cells) == expected_cells,
            f"coverage {summary_name} must record exactly {expected_cells} cells",
        )
        seen_cells: set[tuple[str, float]] = set()
        for cell in recorded_cells:
            cell_map = _mapping(cell)
            radius = _radius_key(cell_map.get("radius"), spec["radii"])
            key = (str(cell_map.get("equilibrium")), radius) if radius is not None else None
            audit.require(
                "probe_summary",
                key is not None and key[0] in spec["equilibria"] and key not in seen_cells,
                f"invalid or duplicate summary cell in {summary_name}: {cell_map}",
            )
            if key is not None:
                seen_cells.add(key)
            for field in ("expected", "recorded", "completed", "unique_direction_ids"):
                audit.require(
                    "probe_summary",
                    _integer(cell_map.get(field)) == spec["directions"],
                    f"{summary_name} cell {key} has {field}={cell_map.get(field)!r}",
                )
            audit.require(
                "probe_summary",
                cell_map.get("complete") is True,
                f"{summary_name} cell {key} is not complete",
            )
    audit.require("probe_summary", _integer(summary.get("n_probes")) == 112, "top-level hiddenness n_probes must be 112")
    for field in ("target_hits", "ambiguous", "numerical_failures"):
        audit.require("probe_summary", _integer(summary.get(field)) == 0, f"top-level hiddenness {field} must be integer zero")
    audit.require("probe_summary", summary.get("finite_sample_only") is True, "top-level hiddenness finite_sample_only must be true")
    audit.require("probe_summary", summary.get("global_hiddenness_proved") is False, "top-level hiddenness global_hiddenness_proved must be false")
    return family_counts


def _audit_stability_semantics(
    audit: Audit,
    stability_payload: Any,
    candidate_parameters: Mapping[str, Any],
) -> dict[str, tuple[float, float, float]]:
    check = "equilibrium_stability"
    gamma = _float(candidate_parameters.get("gamma"))
    delta = _float(candidate_parameters.get("delta"))
    rho = _float(candidate_parameters.get("rho"))
    xi = _float(candidate_parameters.get("xi"))
    audit.require(check, gamma is not None and gamma > 0.0, "candidate gamma must be finite and positive")
    audit.require(check, delta == 100.0 and rho == 200.0 and xi == 2.85, "candidate stability parameters are not the declared MAVPD tuple")
    amplitude = math.sqrt(gamma) if gamma is not None and gamma > 0.0 else float("nan")
    expected_states = {
        "E0": (0.0, 0.0, 0.0),
        "E+": (amplitude, 0.0, amplitude),
        "E-": (-amplitude, 0.0, -amplitude),
    }
    rows = [row for row in _sequence(stability_payload) if isinstance(row, Mapping)]
    indexed = {str(row.get("equilibrium")): row for row in rows}
    audit.require(check, len(rows) == 3 and set(indexed) == set(expected_states), "06 must contain exactly E0, E+, E-")
    for name, expected_state in expected_states.items():
        row = _mapping(indexed.get(name))
        state = _finite_vector(row.get("state"), 3)
        audit.require(
            check,
            state is not None and all(_close(left, right, atol=1.0e-11) for left, right in zip(state, expected_state)),
            f"wrong candidate equilibrium state for {name}",
        )
        residual = _float(row.get("rhs_residual"))
        audit.require(check, residual is not None and 0.0 <= residual <= 1.0e-8, f"invalid candidate equilibrium residual for {name}")
        parsed_eigenvalues = tuple(_complex_number(item) for item in _sequence(row.get("eigenvalues")))
        linear_y1 = (
            delta * gamma - 3.0 * delta * expected_state[0] * expected_state[0]
            if delta is not None and gamma is not None
            else float("nan")
        )
        coefficients = _mavpd_characteristic_coefficients(
            linear_y1=linear_y1,
            delta=delta if delta is not None else float("nan"),
            rho=rho if rho is not None else float("nan"),
            xi=xi if xi is not None else float("nan"),
        )
        spectrum = _audit_characteristic_spectrum(
            audit,
            check=check,
            label=f"candidate Jacobian spectrum for {name}",
            values=parsed_eigenvalues,
            coefficients=coefficients,
        )
        spectral = _float(row.get("spectral_abscissa"))
        if spectrum is not None:
            audit.require(check, spectral is not None and _close(spectral, max(value.real for value in spectrum)), f"wrong spectral abscissa for {name}")
        expected_label = "unstable" if name == "E0" else "locally_asymptotically_stable"
        audit.require(check, row.get("stability") == expected_label, f"wrong stability classification for {name}")
        audit.require(
            check,
            spectral is not None and (spectral > 0.0 if name == "E0" else spectral < 0.0),
            f"candidate equilibrium {name} does not have the sign required by {expected_label}",
        )
    return expected_states


def _kaplan_yorke_dimension(exponents: Sequence[float]) -> float:
    ordered = sorted((float(value) for value in exponents), reverse=True)
    total = 0.0
    for index, value in enumerate(ordered):
        if total + value < 0.0:
            return float(index) + total / abs(value)
        total += value
    return float(len(ordered))


def _audit_dynamics_semantics(
    audit: Audit,
    *,
    trajectory_rows: Sequence[Mapping[str, str]],
    diagnostics: Mapping[str, Any],
    lyapunov_rows: Sequence[Mapping[str, str]],
    poincare_rows: Sequence[Mapping[str, str]],
    stride_rows: Sequence[Mapping[str, str]],
    return_map_payload: Any,
    spectrum_rows: Sequence[Mapping[str, str]],
    robustness: Mapping[str, Any],
    manifest: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    check = "finite_dynamics"
    required_trajectory_columns = {"time", "y1", "y2", "y3"}
    if trajectory_rows:
        audit.require(check, set(trajectory_rows[0]) == required_trajectory_columns, "04 trajectory has the wrong columns")
    trajectory: list[tuple[float, float, float, float]] = []
    for index, row in enumerate(trajectory_rows, start=2):
        values = tuple(_float(row.get(field)) for field in ("time", "y1", "y2", "y3"))
        audit.require(check, all(value is not None for value in values), f"non-finite trajectory row {index}")
        if all(value is not None for value in values):
            trajectory.append(tuple(float(value) for value in values if value is not None))
    audit.require(check, len(trajectory) == 45001, f"full trajectory must contain exactly 45001 finite rows, got {len(trajectory)}")
    if trajectory:
        audit.require(check, _close(trajectory[0][0], 0.0, atol=1.0e-14, rtol=0.0), "trajectory must start at t=0")
        audit.require(check, _close(trajectory[-1][0], 900.0, atol=1.0e-8), "full trajectory must end at t=900")
        audit.require(check, all(trajectory[index][0] < trajectory[index + 1][0] for index in range(len(trajectory) - 1)), "trajectory times are not strictly increasing")
        audit.require(
            check,
            len(trajectory) == 45001
            and all(_close(row[0], 0.02 * index, atol=2.0e-10, rtol=0.0) for index, row in enumerate(trajectory)),
            "trajectory sampling grid must be exactly dt=0.02 on [0,900]",
        )
        selected_state = lineage.get("selected_state")
        audit.require(
            check,
            isinstance(selected_state, tuple)
            and all(_close(left, right, atol=1.0e-12) for left, right in zip(trajectory[0][1:], selected_state)),
            "trajectory initial state differs from selected continuation state",
        )

    candidate_parameters = _mapping(diagnostics.get("candidate_parameters"))
    for key, expected in _mapping(lineage.get("candidate_parameters")).items():
        audit.require(check, _close(candidate_parameters.get(key), expected), f"diagnostic candidate parameter {key} is inconsistent")
    audit.require(check, diagnostics.get("finite_time_only") is True, "diagnostics must be explicitly finite-time only")

    calibration = _mapping(diagnostics.get("reference_calibration"))
    threshold = _float(calibration.get("acceptance_threshold"))
    ambiguity_margin = _float(calibration.get("ambiguity_margin"))
    scale = _float(calibration.get("scale"))
    within_distances = tuple(_float(value) for value in _sequence(calibration.get("within_reference_distances")))
    negative_distances = tuple(_float(value) for value in _sequence(calibration.get("negative_control_distances")))
    audit.require(check, calibration.get("status") == "calibrated", "attractor-reference calibration status is not calibrated")
    audit.require(check, threshold is not None and threshold > 0.0, "reference acceptance threshold is invalid")
    audit.require(check, ambiguity_margin is not None and ambiguity_margin > 0.0, "reference ambiguity margin is invalid")
    audit.require(check, scale is not None and scale > 0.0, "reference attractor scale is invalid")
    audit.require(check, _integer(calibration.get("max_points")) == 1000, "reference calibration max_points must be 1000")
    audit.require(check, len(within_distances) == 3 and all(value is not None and value >= 0.0 for value in within_distances), "reference calibration needs three finite within-window distances")
    audit.require(check, len(negative_distances) == 3 and all(value is not None and value >= 0.0 for value in negative_distances), "reference calibration needs three finite negative-control distances")
    if len(within_distances) == 3 and all(value is not None for value in within_distances) and threshold is not None and ambiguity_margin is not None:
        ordered_within = sorted(float(value) for value in within_distances if value is not None)
        baseline = 0.1 * ordered_within[1] + 0.9 * ordered_within[2]
        baseline = max(baseline, 10.0 * sys.float_info.epsilon)
        audit.require(check, _close(threshold, 3.0 * baseline, atol=1.0e-12), "reference threshold is not the declared 0.95-quantile times safety factor 3")
        audit.require(check, _close(ambiguity_margin, max(0.25 * threshold, baseline), atol=1.0e-12), "reference ambiguity margin is inconsistent")
        audit.require(check, max(ordered_within) <= threshold, "within-reference distance exceeds acceptance threshold")
        if len(negative_distances) == 3 and all(value is not None for value in negative_distances):
            audit.require(
                check,
                min(float(value) for value in negative_distances if value is not None) > threshold + ambiguity_margin,
                "negative-control cloud overlaps the reference acceptance band",
            )

    boundedness = _mapping(diagnostics.get("boundedness"))
    audit.require(check, boundedness.get("boundedness_status") == "bounded_candidate", "trajectory is not a bounded candidate")
    finite_fraction = _float(boundedness.get("finite_fraction"))
    audit.require(check, finite_fraction is not None and _close(finite_fraction, 1.0, rtol=0.0), "boundedness finite_fraction must equal the finite trajectory fraction")
    audit.require(check, _integer(boundedness.get("nonfinite_count")) == 0, "boundedness reports non-finite samples")
    post_trajectory = [row for row in trajectory if row[0] >= 300.0]
    post_rows = len(post_trajectory)
    audit.require(check, _integer(boundedness.get("post_transient_rows")) == post_rows and post_rows == 30001, "post-transient row count is inconsistent")
    post_norms = [math.sqrt(sum(value * value for value in row[1:])) for row in post_trajectory]
    observed_max_norm = max(post_norms, default=float("nan"))
    audit.require(check, _close(boundedness.get("max_norm"), observed_max_norm, atol=1.0e-8), "boundedness max_norm differs from trajectory")
    audit.require(check, _close(boundedness.get("R_observed"), observed_max_norm, atol=1.0e-8), "boundedness R_observed differs from trajectory")
    if post_norms:
        sorted_norms = sorted(post_norms)
        middle = len(sorted_norms) // 2
        median_norm = (
            sorted_norms[middle]
            if len(sorted_norms) % 2
            else 0.5 * (sorted_norms[middle - 1] + sorted_norms[middle])
        )
        window = max(2, int(round(len(post_norms) * 0.2)))
        early_mean = sum(post_norms[:window]) / window
        late_mean = sum(post_norms[-window:]) / window
        growth_ratio = late_mean / max(early_mean, sys.float_info.epsilon)
        derived_norms = {
            "min_norm": min(post_norms),
            "mean_norm": sum(post_norms) / len(post_norms),
            "median_norm": median_norm,
            "final_norm": post_norms[-1],
            "norm_growth_ratio": growth_ratio,
        }
        for field, expected in derived_norms.items():
            audit.require(check, _close(boundedness.get(field), expected, atol=1.0e-8), f"boundedness {field} differs from trajectory")
    audit.require(check, _close(boundedness.get("burn_time"), 300.0), "boundedness burn_time is wrong")
    audit.require(check, boundedness.get("norm") == "euclidean", "boundedness norm must be euclidean")
    for field in ("boundedness_proves_chaos", "chaos_certified_by_boundedness", "hiddenness_certified_by_boundedness"):
        audit.require(check, boundedness.get(field) is False, f"boundedness field {field} overclaims evidence")
    divergence_radius = _float(boundedness.get("divergence_radius"))
    audit.require(check, divergence_radius is not None and _close(divergence_radius, 50.0) and observed_max_norm < divergence_radius, "trajectory reaches/exceeds the configured divergence radius 50")
    coordinate_min = tuple(min(row[index] for row in post_trajectory) for index in (1, 2, 3)) if post_trajectory else ()
    coordinate_max = tuple(max(row[index] for row in post_trajectory) for index in (1, 2, 3)) if post_trajectory else ()
    coordinate_span = tuple(upper - lower for lower, upper in zip(coordinate_min, coordinate_max))
    for field, expected in (("coordinate_min", coordinate_min), ("coordinate_max", coordinate_max), ("coordinate_span", coordinate_span)):
        recorded = _finite_vector(boundedness.get(field), 3)
        audit.require(
            check,
            recorded is not None and all(_close(left, right, atol=1.0e-8) for left, right in zip(recorded, expected)),
            f"boundedness {field} differs from the post-transient trajectory",
        )
    audit.require(check, len(coordinate_span) == 3 and max(coordinate_span) > 1.0e-4, "trajectory is trivial or has invalid span")

    lyapunov_outputs: dict[str, tuple[float, float, float]] = {}
    lyapunov_contracts = (
        ("lyapunov", 300.0, 1200.0, 0.01, 2.0e-12, 2.0e-14, 2400),
        ("lyapunov_control", 200.0, 600.0, 0.02, 2.0e-10, 2.0e-12, 1200),
    )
    for name, expected_burn, expected_accumulation, expected_step, expected_rtol, expected_atol, expected_segments in lyapunov_contracts:
        payload = _mapping(diagnostics.get(name))
        exponents = _finite_vector(payload.get("exponents"), 3)
        audit.require(check, payload.get("status") == "ok", f"{name} status is not ok")
        audit.require(check, payload.get("method") == "integer_dop853_variational_qr", f"{name} method is wrong")
        audit.require(check, payload.get("finite_time_local") is True, f"{name} must be finite-time local")
        audit.require(check, payload.get("does_not_prove_chaos_alone") is True, f"{name} overclaims its evidence")
        audit.require(check, exponents is not None and exponents[0] > 0.02, f"{name} lacks a positive largest exponent")
        audit.require(check, _close(payload.get("t_accumulate"), expected_accumulation), f"{name} top-level accumulation horizon is wrong")
        if exponents is not None:
            lyapunov_outputs[name] = exponents
            audit.require(check, _close(payload.get("sum_exponents"), sum(exponents), atol=1.0e-10), f"{name} sum_exponents is inconsistent")
        metadata = _mapping(payload.get("metadata"))
        audit.require(check, metadata.get("solver_method") == "DOP853", f"{name} solver method is wrong")
        audit.require(check, metadata.get("solver") == "scipy.integrate.solve_ivp", f"{name} solver implementation is wrong")
        audit.require(check, metadata.get("jacobian_source") == "analytic", f"{name} did not use the analytic Jacobian")
        audit.require(check, metadata.get("jacobian_eps") is None, f"{name} unexpectedly used a finite-difference Jacobian epsilon")
        audit.require(check, _integer(metadata.get("dimension")) == 3, f"{name} Lyapunov dimension is wrong")
        audit.require(check, _close(metadata.get("div_threshold"), 50.0), f"{name} divergence threshold is wrong")
        audit.require(check, _close(metadata.get("max_step"), expected_step), f"{name} max_step is wrong")
        audit.require(check, _close(metadata.get("rtol"), expected_rtol), f"{name} rtol is wrong")
        audit.require(check, _close(metadata.get("atol"), expected_atol), f"{name} atol is wrong")
        audit.require(check, _close(metadata.get("qr_interval"), 0.5), f"{name} QR interval is wrong")
        audit.require(check, _integer(metadata.get("qr_segments")) == expected_segments, f"{name} QR segment count is wrong")
        audit.require(check, _close(metadata.get("t_burn_requested"), expected_burn), f"{name} requested burn is wrong")
        audit.require(check, _close(metadata.get("t_burn_completed"), expected_burn), f"{name} burn is incomplete")
        audit.require(check, _close(metadata.get("t_accumulate_requested"), expected_accumulation), f"{name} requested horizon is wrong")
        audit.require(check, _close(metadata.get("t_accumulate_completed"), expected_accumulation), f"{name} accumulation is incomplete")
    strict = lyapunov_outputs.get("lyapunov")
    control = lyapunov_outputs.get("lyapunov_control")
    if strict is not None:
        manifest_exponents = _finite_vector(manifest.get("lyapunov_exponents"), 3)
        audit.require(
            check,
            manifest_exponents is not None and all(_close(left, right, atol=1.0e-11) for left, right in zip(strict, manifest_exponents)),
            "manifest Lyapunov spectrum differs from diagnostics",
        )
        convergence: list[tuple[float, float, float, float]] = []
        for index, row in enumerate(lyapunov_rows, start=2):
            values = tuple(_float(row.get(field)) for field in ("time", "lambda_1", "lambda_2", "lambda_3"))
            audit.require(check, all(value is not None for value in values), f"non-finite Lyapunov convergence row {index}")
            if all(value is not None for value in values):
                convergence.append(tuple(float(value) for value in values if value is not None))
        audit.require(check, len(convergence) == 2400, f"Lyapunov convergence must contain 2400 QR records, got {len(convergence)}")
        if convergence:
            audit.require(check, all(convergence[index][0] < convergence[index + 1][0] for index in range(len(convergence) - 1)), "Lyapunov convergence times are not increasing")
            audit.require(
                check,
                len(convergence) == 2400
                and all(_close(row[0], 0.5 * (index + 1), atol=1.0e-9, rtol=0.0) for index, row in enumerate(convergence)),
                "Lyapunov convergence times do not match qr_interval=0.5 through t=1200",
            )
            audit.require(
                check,
                all(_close(left, right, atol=1.0e-10) for left, right in zip(convergence[-1][1:], strict)),
                "last Lyapunov convergence row differs from final spectrum",
            )
        expected_ky = _kaplan_yorke_dimension(strict)
        audit.require(check, _close(diagnostics.get("kaplan_yorke_dimension"), expected_ky, atol=1.0e-10), "Kaplan-Yorke dimension is inconsistent")
        mean_divergence = _float(diagnostics.get("mean_vector_field_divergence"))
        divergence_residual = _float(diagnostics.get("lyapunov_sum_minus_mean_divergence"))
        audit.require(check, mean_divergence is not None and divergence_residual is not None, "divergence consistency values must be finite")
        if mean_divergence is not None and divergence_residual is not None:
            parameters = _mapping(lineage.get("candidate_parameters"))
            delta = _float(parameters.get("delta"))
            gamma = _float(parameters.get("gamma"))
            xi = _float(parameters.get("xi"))
            recomputed_divergence = (
                sum(delta * gamma - xi - 3.0 * delta * row[1] * row[1] for row in post_trajectory) / len(post_trajectory)
                if delta is not None and gamma is not None and xi is not None and post_trajectory
                else None
            )
            audit.require(
                check,
                recomputed_divergence is not None and _close(mean_divergence, recomputed_divergence, atol=1.0e-9),
                "mean vector-field divergence differs from the post-transient trajectory",
            )
            audit.require(check, _close(divergence_residual, sum(strict) - mean_divergence, atol=1.0e-9), "Lyapunov/divergence residual is inconsistent")

    return_rows = [row for row in _sequence(return_map_payload) if isinstance(row, Mapping)]
    return_coordinates = tuple(_integer(row.get("coordinate")) for row in return_rows)
    audit.require(
        check,
        len(return_rows) == 2
        and all(value is not None for value in return_coordinates)
        and set(return_coordinates) == {0, 2},
        "return-map 0-1 evidence must contain integer coordinates 0 and 2",
    )
    return_k: list[float] = []
    for row in return_rows:
        k_value = _float(row.get("K"))
        audit.require(check, k_value is not None and -1.0 - 1.0e-12 <= k_value <= 1.0 + 1.0e-12, f"invalid return-map K for coordinate {row.get('coordinate')}")
        expected_state = (
            "zero_one_chaotic_candidate"
            if k_value is not None and k_value > 0.8
            else "zero_one_regular_candidate"
            if k_value is not None and k_value < 0.2
            else "zero_one_inconclusive"
        )
        audit.require(check, row.get("state") == expected_state, "return-map 0-1 state differs from its own K statistic")
        audit.require(check, _close(row.get("K_median"), k_value), "return-map K_median differs from K")
        k_mean = _float(row.get("K_mean"))
        k_min = _float(row.get("K_min"))
        k_max = _float(row.get("K_max"))
        k_std = _float(row.get("K_std"))
        audit.require(
            check,
            k_value is not None
            and k_mean is not None
            and k_min is not None
            and k_max is not None
            and k_std is not None
            and -1.0 - 1.0e-12 <= k_min <= k_value <= k_max <= 1.0 + 1.0e-12
            and -1.0 - 1.0e-12 <= k_mean <= 1.0 + 1.0e-12
            and k_std >= 0.0,
            "return-map 0-1 summary statistics are inconsistent",
        )
        audit.require(check, _integer(row.get("c_values_count")) == 64, "return-map 0-1 must use 64 c values")
        audit.require(check, _integer(row.get("random_seed")) == 20260802, "return-map 0-1 random seed is wrong")
        audit.require(check, row.get("detrend") is True and row.get("normalize") is True, "return-map 0-1 preprocessing flags are wrong")
        audit.require(check, row.get("zero_one_alone_does_not_certify_chaos") is True, "0-1 result overclaims certification")
        audit.require(check, row.get("chaos_certified_by_zero_one") is False, "0-1 result overclaims chaos")
        audit.require(check, row.get("hiddenness_certified_by_zero_one") is False, "0-1 result overclaims hiddenness")
        audit.require(check, _integer(row.get("signal_length")) == len(poincare_rows), "return-map signal_length differs from the Poincare section")
        if k_value is not None:
            return_k.append(k_value)
    diagnostic_return = _sequence(_mapping(diagnostics.get("zero_one")).get("return_map_results"))
    audit.require(check, list(diagnostic_return) == return_rows, "diagnostic return-map results differ from 05_zero_one_return_map.json")
    audit.require(check, len(stride_rows) == 10, "flow 0-1 stride table must contain 10 declared strides")
    expected_strides = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)
    normalized_stride_rows: list[dict[str, Any]] = []
    for expected_stride, row in zip(expected_strides, stride_rows):
        audit.require(check, row.get("series") == "flow_y1" and _integer(row.get("stride")) == expected_stride, f"wrong flow stride row for {expected_stride}")
        samples = _integer(row.get("samples"))
        k_value = _float(row.get("K"))
        expected_samples = (len(post_trajectory) + expected_stride - 1) // expected_stride
        audit.require(check, samples == expected_samples, f"flow stride {expected_stride} sample count is inconsistent")
        audit.require(check, _close(row.get("effective_sample_step"), 0.02 * expected_stride), f"flow stride {expected_stride} effective sample step is wrong")
        if row.get("state") == "insufficient_samples":
            audit.require(check, samples is not None and samples < 100 and k_value is None, f"invalid insufficient-sample row at stride {expected_stride}")
        else:
            audit.require(check, samples is not None and samples >= 100 and k_value is not None, f"invalid finite flow 0-1 row at stride {expected_stride}")
            expected_state = (
                "zero_one_chaotic_candidate"
                if k_value is not None and k_value > 0.8
                else "zero_one_regular_candidate"
                if k_value is not None and k_value < 0.2
                else "zero_one_inconclusive"
            )
            audit.require(check, row.get("state") == expected_state, f"flow stride {expected_stride} state differs from K")
        normalized_stride_rows.append(
            {
                "series": row.get("series"),
                "stride": _integer(row.get("stride")),
                "effective_sample_step": _float(row.get("effective_sample_step")),
                "samples": samples,
                "K": k_value,
                "state": row.get("state"),
            }
        )
    diagnostic_stride_rows = [row for row in _sequence(_mapping(diagnostics.get("zero_one")).get("flow_stride_sensitivity")) if isinstance(row, Mapping)]
    normalized_diagnostic_strides = [
        {
            "series": row.get("series"),
            "stride": _integer(row.get("stride")),
            "effective_sample_step": _float(row.get("effective_sample_step")),
            "samples": _integer(row.get("samples")),
            "K": _float(row.get("K")),
            "state": row.get("state"),
        }
        for row in diagnostic_stride_rows
    ]
    audit.require(
        check,
        normalized_diagnostic_strides == normalized_stride_rows,
        "diagnostic flow-stride evidence differs from 05_zero_one_stride_sensitivity.csv",
    )

    poincare = _mapping(diagnostics.get("poincare"))
    crossing_count = _integer(poincare.get("crossing_count"))
    audit.require(check, crossing_count == len(poincare_rows) and crossing_count is not None and crossing_count >= 100, "Poincare crossing count is inconsistent or insufficient")
    poincare_points: list[tuple[float, float]] = []
    for index, row in enumerate(poincare_rows, start=2):
        values = tuple(_float(row.get(field)) for field in ("time", "y1", "y3"))
        audit.require(check, all(value is not None for value in values), f"non-finite Poincare row {index}")
        if all(value is not None for value in values):
            poincare_points.append((float(values[1]), float(values[2])))
    poincare_points_bounded = all(abs(value) < 50.0 for point in poincare_points for value in point)
    audit.require(check, poincare_points_bounded, "Poincare points exceed the candidate trajectory divergence bound")
    audit.require(check, _integer(poincare.get("retained_after_burn")) == len(poincare_points), "Poincare retained-after-burn count is inconsistent")
    if len(poincare_points) >= 2:
        means = tuple(sum(point[index] for point in poincare_points) / len(poincare_points) for index in (0, 1))
        covariance = tuple(
            tuple(
                sum((point[i] - means[i]) * (point[j] - means[j]) for point in poincare_points)
                / (len(poincare_points) - 1)
                for j in (0, 1)
            )
            for i in (0, 1)
        )
        minima = tuple(min(point[index] for point in poincare_points) for index in (0, 1))
        maxima = tuple(max(point[index] for point in poincare_points) for index in (0, 1))
        bounding_box = _mapping(poincare.get("bounding_box"))
        for index in (0, 1):
            bounds = _mapping(bounding_box.get(f"coordinate_{index}"))
            audit.require(check, _close(bounds.get("minimum"), minima[index], atol=1.0e-10), f"Poincare coordinate_{index} minimum is wrong")
            audit.require(check, _close(bounds.get("maximum"), maxima[index], atol=1.0e-10), f"Poincare coordinate_{index} maximum is wrong")
        recorded_centroid = _finite_vector(poincare.get("centroid"), 2)
        audit.require(check, recorded_centroid is not None and all(_close(left, right, atol=1.0e-10) for left, right in zip(recorded_centroid, means)), "Poincare centroid differs from CSV")
        recorded_covariance = _sequence(poincare.get("covariance"))
        covariance_ok = len(recorded_covariance) == 2 and all(
            _finite_vector(row, 2) is not None
            and all(_close(left, right, atol=1.0e-10) for left, right in zip(_finite_vector(row, 2) or (), expected))
            for row, expected in zip(recorded_covariance, covariance)
        )
        audit.require(check, covariance_ok, "Poincare covariance differs from CSV")
        centered_squares_0 = sum((point[0] - means[0]) ** 2 for point in poincare_points)
        centered_squares_1 = sum((point[1] - means[1]) ** 2 for point in poincare_points)
        centered_cross = sum((point[0] - means[0]) * (point[1] - means[1]) for point in poincare_points)
        gram_trace = centered_squares_0 + centered_squares_1
        gram_discriminant = math.sqrt(
            max(
                0.0,
                (centered_squares_0 - centered_squares_1) ** 2 + 4.0 * centered_cross * centered_cross,
            )
        )
        largest_gram_eigenvalue = 0.5 * (gram_trace + gram_discriminant)
        gram_determinant = max(0.0, centered_squares_0 * centered_squares_1 - centered_cross * centered_cross)
        smallest_gram_eigenvalue = (
            gram_determinant / largest_gram_eigenvalue
            if largest_gram_eigenvalue > 0.0
            else 0.0
        )
        second_singular_value = math.sqrt(max(0.0, smallest_gram_eigenvalue))
        rank_two = second_singular_value > 1.0e-8
        expected_rank = 2 if rank_two else 1 if gram_trace > 1.0e-16 else 0
        audit.require(check, _integer(poincare.get("rank_estimate")) == expected_rank, "Poincare rank estimate differs from the declared matrix-rank tolerance")
        if poincare_points_bounded:
            duplicate_tolerance = 1.0e-6
            unique_count = len(
                {
                    (
                        round(point[0] / duplicate_tolerance),
                        round(point[1] / duplicate_tolerance),
                    )
                    for point in poincare_points
                }
            )
            duplicate_fraction = 1.0 - unique_count / len(poincare_points)
            audit.require(check, _close(poincare.get("duplicate_fraction"), duplicate_fraction, atol=1.0e-12), "Poincare duplicate fraction is inconsistent")
            nearest_distances: list[float] = []
            for point_index, point in enumerate(poincare_points):
                nearest_distances.append(
                    min(
                        math.hypot(point[0] - other[0], point[1] - other[1])
                        for other_index, other in enumerate(poincare_points)
                        if other_index != point_index
                    )
                )
            nearest_summary = {
                "minimum": min(nearest_distances),
                "median": _median(nearest_distances),
                "mean": sum(nearest_distances) / len(nearest_distances),
                "maximum": max(nearest_distances),
            }
            recorded_nearest = _mapping(poincare.get("nearest_neighbor_stats"))
            for field, expected in nearest_summary.items():
                audit.require(check, _close(recorded_nearest.get(field), expected, atol=1.0e-10), f"Poincare nearest-neighbor {field} is inconsistent")
            spread = math.hypot(maxima[0] - minima[0], maxima[1] - minima[1])
            nearest_median = nearest_summary["median"]
            expected_label = (
                "insufficient_crossings"
                if len(poincare_points) < 3
                else "point_like_or_fixed_return"
                if spread <= duplicate_tolerance
                else "finite_set_like"
                if duplicate_fraction >= 0.5
                else "curve_like"
                if expected_rank <= 1
                else "dispersed_cloud_like"
                if nearest_median is not None and spread > 0.0 and nearest_median / spread > 0.25
                else "cloud_like"
            )
            audit.require(check, poincare.get("interpretation_label") == expected_label, "Poincare interpretation label differs from its geometry summary")
    section_metadata = _mapping(poincare.get("section_metadata"))
    expected_section_metadata = {
        "section_variable": 1,
        "section_index": 1,
        "section_value": 0.0,
        "direction": "positive",
        "direction_rule": "rhs(section_crossing)[section_variable] has requested sign",
        "derivative_mode": "integer_rhs",
        "interpolation": "linear",
        "min_crossing_separation": 0.05,
        "burn_time": 300.0,
        "caputo_geometric_crossing": False,
        "exact_poincare_map": False,
        "sampled_linear_interpolation": True,
        "classical_integer_section_interpretation": True,
        "uses_classical_rhs_direction": True,
    }
    for field, expected in expected_section_metadata.items():
        audit.require(
            check,
            _same_json_semantics(section_metadata.get(field), expected),
            f"wrong Poincare section metadata field {field}",
        )

    reconstructed_crossings: list[tuple[float, float, float]] = []
    filtered_by_separation = 0
    for left, right in zip(trajectory, trajectory[1:]):
        left_y2 = left[2]
        right_y2 = right[2]
        delta_y2 = right_y2 - left_y2
        if not (left_y2 < 0.0 <= right_y2 and delta_y2 > 0.0):
            continue
        theta = -left_y2 / delta_y2
        crossing_time = left[0] + theta * (right[0] - left[0])
        if crossing_time < 300.0:
            continue
        crossing_y1 = left[1] + theta * (right[1] - left[1])
        crossing_y3 = left[3] + theta * (right[3] - left[3])
        if crossing_y1 - crossing_y3 <= 0.0:
            continue
        if reconstructed_crossings and crossing_time - reconstructed_crossings[-1][0] < 0.05:
            filtered_by_separation += 1
            continue
        reconstructed_crossings.append((crossing_time, crossing_y1, crossing_y3))
    audit.require(
        check,
        _integer(section_metadata.get("filtered_by_min_crossing_separation")) == filtered_by_separation,
        "Poincare filtered-crossing count differs from the trajectory reconstruction",
    )
    observed_crossings = [
        (_float(row.get("time")), _float(row.get("y1")), _float(row.get("y3")))
        for row in poincare_rows
    ]
    crossing_match = len(observed_crossings) == len(reconstructed_crossings)
    if crossing_match:
        crossing_match = all(
            all(observed is not None and _close(observed, expected, atol=1.0e-9) for observed, expected in zip(observed_row, expected_row))
            for observed_row, expected_row in zip(observed_crossings, reconstructed_crossings)
        )
    audit.require(check, crossing_match, "05_poincare_section.csv is not the declared crossing reconstruction of 04_candidate_trajectory.csv")

    spectrum = _mapping(diagnostics.get("spectrum"))
    audit.require(check, spectrum.get("gate_applicable") is False, "FFT must be supporting-only")
    audit.require(check, spectrum.get("normalized_fft_power_not_welch_psd") is True, "FFT must explicitly state that it is not Welch PSD")
    spectrum_counts = {"x": 0, "y": 0, "z": 0}
    for index, row in enumerate(spectrum_rows, start=2):
        coordinate = row.get("coordinate", "")
        frequency = _float(row.get("frequency"))
        power = _float(row.get("normalized_fft_power"))
        audit.require(check, coordinate in spectrum_counts, f"wrong FFT coordinate at row {index}")
        audit.require(check, frequency is not None and frequency >= 0.0 and power is not None and power >= 0.0, f"non-finite/negative FFT row {index}")
        if coordinate in spectrum_counts:
            spectrum_counts[coordinate] += 1
    audit.require(check, all(count >= 2 for count in spectrum_counts.values()), f"FFT coordinates are incomplete: {spectrum_counts}")

    efork = _mapping(diagnostics.get("efork_crosscheck"))
    target_match = _mapping(efork.get("target_match"))
    audit.require(check, efork.get("status") == "ok", "EFORK cross-check did not finish successfully")
    audit.require(check, _close(efork.get("h"), 0.002), "full EFORK step must be 0.002")
    audit.require(check, _close(efork.get("t_final"), 600.0), "full EFORK horizon must be 600")
    audit.require(check, _close(efork.get("tail_burn_fraction"), 0.5), "EFORK tail burn fraction must be 0.5")
    audit.require(check, target_match.get("classification") == "same_attractor_under_calibrated_cloud_test", "EFORK does not match the strict attractor cloud")
    audit.require(check, target_match.get("finite_sample_only") is True, "EFORK match must be finite-sample only")
    efork_distances = tuple(_float(value) for value in _sequence(target_match.get("distances_norm")))
    finite_efork_distances = tuple(value for value in efork_distances if value is not None)
    efork_distance = _float(target_match.get("distance_norm"))
    audit.require(
        check,
        len(efork_distances) == 3
        and len(finite_efork_distances) == 3
        and all(value is not None and value >= 0.0 for value in efork_distances)
        and efork_distance is not None
        and efork_distance >= 0.0
        and _close(efork_distance, _median(finite_efork_distances), atol=1.0e-12),
        "EFORK target distance is not the median of the three reference-cloud distances",
    )
    audit.require(check, _close(target_match.get("acceptance_threshold"), threshold, atol=1.0e-12), "EFORK acceptance threshold differs from the calibrated reference")
    audit.require(
        check,
        threshold is not None
        and ambiguity_margin is not None
        and _close(target_match.get("ambiguity_upper_bound"), threshold + ambiguity_margin, atol=1.0e-12),
        "EFORK ambiguity upper bound differs from the calibrated reference",
    )
    audit.require(check, target_match.get("calibration_status") == "calibrated", "EFORK target match does not use a calibrated reference")
    if threshold is not None and efork_distance is not None:
        audit.require(check, efork_distance <= threshold, "EFORK same-attractor classification lies outside the calibrated threshold")

    expected_robustness = {
        "tested_h": True,
        "tested_memory": False,
        "memory_applicable": False,
        "tested_t_final": True,
        "tested_integrator": True,
        "integrator_match": True,
        "consistent": True,
    }
    audit.require(
        check,
        _same_json_semantics(robustness, expected_robustness),
        f"08 robustness matrix differs from derived full contract: {robustness}",
    )
    if strict is not None and control is not None:
        audit.require(check, strict[0] > 0.02 and control[0] > 0.02, "strict/control Lyapunov robustness is not positive")
    return {
        "trajectory": trajectory,
        "strict_exponents": strict,
        "control_exponents": control,
        "return_k": return_k,
        "poincare": poincare,
        "boundedness": boundedness,
        "robustness": expected_robustness,
        "efork": efork,
        "reference_threshold": threshold,
        "reference_ambiguity_margin": ambiguity_margin,
    }


def _audit_gate_metadata(
    audit: Audit,
    metadata: Mapping[str, Any],
    lineage: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> bool:
    check = "joint_gate_semantics"
    requirements: list[tuple[bool, str]] = []

    def require(condition: bool, message: str) -> None:
        requirements.append((bool(condition), message))
        audit.require(check, bool(condition), message)

    for key in ("run_id", "created_at_utc"):
        require(isinstance(metadata.get(key), str) and bool(metadata.get(key).strip()), f"gate run_metadata.{key} is missing")
    require(_close(metadata.get("schema_version"), 1.0, rtol=0.0), "gate metadata schema_version is wrong")
    require(metadata.get("workflow") == "integer_hidden_chaos_search", "gate metadata workflow is wrong")
    require(metadata.get("system") == "modified-van-der-pol-duffing", "gate metadata system is wrong")

    numerical = _mapping(metadata.get("numerical_contract"))
    expected_numerical = {"q": 1.0, "h": 0.02, "t_final": 900.0, "t_burn": 300.0}
    for key, expected in expected_numerical.items():
        require(_close(numerical.get(key), expected), f"gate numerical_contract.{key} is wrong")
    require(
        _same_json_semantics(
            numerical.get("memory"),
            {
                "mode": "not_applicable",
                "M": None,
                "memory_window_steps": None,
                "memory_window_time": None,
                "is_full_caputo": False,
            },
        ),
        "gate numerical memory contract is wrong",
    )
    require(
        _same_json_semantics(
            numerical.get("integrator"),
            {"name": "DOP853", "backend": "python", "caputo": False},
        ),
        "gate numerical integrator contract is wrong",
    )

    candidate_parameters = _mapping(lineage.get("candidate_parameters"))
    require(
        _same_json_semantics(metadata.get("parameters"), candidate_parameters),
        "gate run_metadata.parameters differs from the selected candidate",
    )
    extra = _mapping(metadata.get("extra"))
    require(
        _same_json_semantics(extra.get("candidate_parameters"), candidate_parameters),
        "gate run_metadata.extra candidate parameters differ",
    )
    lure = _mapping(metadata.get("lure"))
    candidate_gamma = _float(candidate_parameters.get("gamma"))
    expected_lure_matrix = [
        [100.0 * candidate_gamma if candidate_gamma is not None else float("nan"), 100.0, 0.0],
        [1.0, -2.85, -1.0],
        [0.0, 200.0, 0.0],
    ]
    require(_same_json_semantics(lure.get("matrix"), expected_lure_matrix), "gate metadata Lur'e matrix is wrong")
    require(_same_json_semantics(lure.get("input_vector"), [-100.0, 0.0, 0.0]), "gate metadata Lur'e input vector is wrong")
    require(_same_json_semantics(lure.get("output_vector"), [1.0, 0.0, 0.0]), "gate metadata Lur'e output vector is wrong")
    require(isinstance(lure.get("scalar_nonlinearity"), str) and bool(lure.get("scalar_nonlinearity").strip()), "gate metadata scalar nonlinearity is missing")
    require(lure.get("transfer_convention") == "c^T (P-s I)^(-1) b; direct polynomial roots", "gate transfer convention is wrong")
    require(lure.get("harmonic_condition") == "Im W(i omega)=0 and Re W(i omega)=-1/k", "gate harmonic condition is wrong")

    seed = _mapping(metadata.get("seed"))
    require(seed.get("candidate_id") == "mavpd-chaos-continuation-endpoint", "gate candidate seed ID is wrong")
    require(seed.get("family") == "continuation_endpoint_from_theoretical_lure_seed", "gate seed family is wrong")
    require(seed.get("source") == "selected_parameter_continuation_endpoint", "gate seed source is wrong")
    selected_state = lineage.get("selected_state")
    recorded_x0 = _finite_vector(seed.get("x0"), 3)
    require(
        isinstance(selected_state, tuple)
        and recorded_x0 is not None
        and all(_close(left, right, atol=1.0e-11) for left, right in zip(recorded_x0, selected_state)),
        "gate seed x0 differs from the selected continuation endpoint",
    )
    seed_parameters = _mapping(seed.get("parameters"))
    source_branch_index = _integer(lineage.get("source_branch_index"))
    require(source_branch_index in {0, 1}, "selected alternative source branch is invalid")
    require(_integer(seed_parameters.get("source_branch_index")) == source_branch_index, "gate theoretical seed source branch is wrong")
    recorded_theoretical_seed = _finite_vector(seed_parameters.get("theoretical_harmonic_seed"), 3)
    expected_theoretical_seed = _finite_vector(lineage.get("theoretical_harmonic_seed"), 3)
    require(
        recorded_theoretical_seed is not None
        and expected_theoretical_seed is not None
        and all(_close(left, right, atol=1.0e-10) for left, right in zip(recorded_theoretical_seed, expected_theoretical_seed)),
        "gate theoretical harmonic seed differs from the selected direct branch",
    )
    selected_pair_index = source_branch_index if source_branch_index in {0, 1} else 0
    expected_omega, expected_gain = _mavpd_direct_seed_pairs()[selected_pair_index]
    require(_close(seed_parameters.get("omega0"), expected_omega, atol=1.0e-9), "gate theoretical omega0 is wrong")
    require(_close(seed_parameters.get("k"), expected_gain, atol=1.0e-10), "gate theoretical gain is wrong")
    require(_close(seed_parameters.get("a0"), math.sqrt(expected_gain / 0.75), atol=1.0e-10), "gate theoretical amplitude is wrong")

    expected_eta = [index / 20.0 for index in range(21)]
    continuation = _mapping(metadata.get("continuation"))
    require(continuation.get("used") is True, "gate metadata continuation.used must be true")
    require(continuation.get("continuation_mode") == "integer", "gate metadata continuation mode is wrong")
    require(continuation.get("memory_window_propagated") is None, "gate integer continuation must not declare memory propagation")
    require(_close(continuation.get("final_eta"), 1.0), "gate metadata continuation did not reach eta=1")
    eta_path = _sequence(continuation.get("eta_path"))
    require(
        len(eta_path) == 21 and all(_close(left, right, atol=1.0e-14) for left, right in zip(eta_path, expected_eta)),
        "gate metadata eta path is not the exact 21-node lambda path",
    )

    software = _mapping(metadata.get("software"))
    for key in ("python_version", "platform", "package_version", "numpy_version", "scipy_version", "git_commit"):
        value = software.get(key)
        require(isinstance(value, str) and bool(value.strip()) and (key != "git_commit" or value != "unknown"), f"gate software.{key} is missing or unknown")
    require(isinstance(software.get("working_tree_dirty"), bool), "gate software working_tree_dirty must be boolean")
    if software.get("working_tree_dirty") is True:
        require(
            isinstance(software.get("git_diff_sha256"), str)
            and SHA256_PATTERN.fullmatch(software.get("git_diff_sha256")) is not None,
            "gate dirty-tree software metadata lacks a valid diff digest",
        )
    require(_integer(metadata.get("random_seed")) == 20260802, "gate random seed is wrong")
    require(metadata.get("random_seed_policy") == "fixed_reproducible", "gate random seed policy is wrong")

    tolerances = _mapping(metadata.get("tolerances"))
    require(
        set(tolerances) == set(EXPECTED_GATE_TOLERANCES),
        "gate metadata tolerances have missing or unexpected fields",
    )
    for key, expected in EXPECTED_GATE_TOLERANCES.items():
        require(
            _close(tolerances.get(key), expected, atol=1.0e-15, rtol=1.0e-12),
            f"gate metadata tolerance {key} differs from the frozen full-run contract",
        )
    for software_key, runtime_key in (
        ("python_version", "python_version"),
        ("platform", "platform"),
        ("numpy_version", "numpy_version"),
        ("scipy_version", "scipy_version"),
    ):
        require(
            software.get(software_key) == runtime_environment.get(runtime_key),
            f"gate software.{software_key} differs from the recorded runtime environment",
        )
    provenance = _mapping(metadata.get("provenance"))
    require(provenance.get("source_doi") == "10.3390/math11030591", "gate metadata source DOI is wrong")
    require(provenance.get("source_scope") == "published_model_equations_only", "gate metadata source scope is wrong")
    require(provenance.get("candidate_parameter_set") == "gamma_selected_at_declared_xi_continuation_endpoint", "gate candidate-parameter provenance is wrong")
    require(provenance.get("candidate_parameter_set_published") is False, "gate metadata marks the candidate tuple as published")
    require(provenance.get("frequency_grid_used_for_search") is False, "gate metadata records a frequency sweep")
    require(provenance.get("alternative_triggered_after_direct_chaos_screen_failure") is True, "gate metadata omits direct-route failure")
    return all(condition for condition, _message in requirements)


def _audit_gate_semantics(
    audit: Audit,
    gate_file: Mapping[str, Any],
    hidden_summary: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    dynamics: Mapping[str, Any],
    robustness: Mapping[str, Any],
    lineage: Mapping[str, Any],
    stability_payload: Any,
    runtime_environment: Mapping[str, Any],
) -> None:
    check = "joint_gate_semantics"
    gate = _mapping(gate_file.get("gate"))
    evidence = _mapping(gate_file.get("evidence"))
    return_k = list(dynamics.get("return_k") or [])
    expected_k = 0.5 * sum(return_k) if len(return_k) == 2 else None
    zero_one_support = (
        "chaotic"
        if expected_k is not None and expected_k >= 0.7
        else "regular"
        if expected_k is not None and expected_k <= 0.3
        else "not_available_or_intermediate"
    )
    poincare_label = _mapping(dynamics.get("poincare")).get("interpretation_label")
    poincare_support = (
        "chaotic"
        if poincare_label in {"cloud_like", "dispersed_cloud_like"}
        else "regular"
        if poincare_label in {"point_like_or_fixed_return", "finite_set_like", "curve_like"}
        else "not_available_or_inconclusive"
    )
    complementary_chaos = zero_one_support == "chaotic" or poincare_support == "chaotic"
    regular_conflict = zero_one_support == "regular" or poincare_support == "regular"
    audit.require(
        check,
        complementary_chaos and not regular_conflict,
        "recomputed gate policy lacks complementary chaotic support or contains a regularity conflict",
    )
    expected_labels = {
        "attractor_status": "hidden_under_tested_neighborhoods",
        "verdict": "hidden_under_tested_neighborhoods",
        "hiddenness_evidence_level": "hidden_under_tested_neighborhoods",
        "evidence_level": "hidden_under_tested_neighborhoods",
        "chaos_evidence_level": "strong_chaos_evidence",
        "lyapunov_support": "positive",
        "zero_one_support": zero_one_support,
        "spectral_support": "not_available_or_inconclusive",
        "boundedness_support": "bounded_nontrivial",
        "poincare_support": poincare_support,
        "hidden_chaos_status": "chaotic_hidden_under_tested_neighborhoods",
    }
    for key, expected in expected_labels.items():
        audit.require(check, gate.get(key) == expected, f"gate.{key} must be {expected!r}")
    for key in ("promotion_allowed", "hiddenness_promotion_allowed", "chaotic_hidden_promotion_allowed"):
        audit.require(check, gate.get(key) is True, f"gate.{key} must be true")
    audit.require(check, gate.get("quick_smoke_only") is not True, "full gate is marked quick-smoke")
    audit.require(check, gate.get("scientific_promotion_allowed") is not False, "full gate disables scientific promotion")
    checked = _mapping(gate.get("checked_conditions"))
    run_metadata = _mapping(evidence.get("run_metadata"))
    metadata_complete = _audit_gate_metadata(audit, run_metadata, lineage, runtime_environment)
    equilibria = _mapping(evidence.get("equilibria"))
    matignon = _mapping(evidence.get("matignon"))
    seed = _mapping(evidence.get("seed"))
    continuation = _mapping(evidence.get("continuation"))
    evidence_trajectory = _mapping(evidence.get("trajectory"))
    evidence_robustness = _mapping(evidence.get("robustness"))
    hidden = _mapping(evidence.get("hiddenness"))
    tolerances = _mapping(evidence.get("tolerances"))
    audit.require(
        check,
        _same_json_semantics(tolerances, _mapping(run_metadata.get("tolerances"))),
        "gate evidence tolerances differ from run_metadata tolerances",
    )
    residual = _float(equilibria.get("max_residual"))
    residual_tolerance = _float(tolerances.get("equilibrium_residual_tol"))
    q_value = _float(matignon.get("q"))
    continuation_used = continuation.get("used") is True
    eta_path = _sequence(continuation.get("eta_path"))
    finite_fraction_gate = _float(evidence_trajectory.get("finite_fraction"))
    post_transient = _integer(
        evidence_trajectory.get("post_transient_length", evidence_trajectory.get("post_transient_rows"))
    )
    minimum_post = _integer(evidence_trajectory.get("minimum_post_transient_length"))
    target_hits = _integer(hidden.get("target_hits_from_equilibria"))
    numerical_failures = _integer(hidden.get("numerical_failures"))
    tested_radii = tuple(_float(value) for value in _sequence(hidden.get("tested_radii")))
    required_radii = tuple(_float(value) for value in _sequence(hidden.get("required_radii")))
    recomputed_conditions = {
        "equilibria_all_found": equilibria.get("all_found") is True,
        "equilibria_residual_within_tolerance": residual is not None
        and residual_tolerance is not None
        and 0.0 <= residual <= residual_tolerance,
        "matignon_all_classified": matignon.get("all_classified") is True,
        "matignon_q_recorded": q_value is not None,
        "seed_localized": seed.get("localized") is True,
        "seed_method_supported": seed.get("method") == "continuation",
        "seed_source_traceable": isinstance(seed.get("source"), str) and bool(seed.get("source").strip()),
        "continuation_reaches_target": not continuation_used or _close(continuation.get("final_eta"), 1.0),
        "continuation_eta_path_recorded": not continuation_used
        or (bool(eta_path) and all(_float(value) is not None for value in eta_path)),
        "continuation_memory_declared": _close(q_value, 1.0)
        or continuation.get("memory_window_propagated") is True
        or continuation.get("continuation_mode") in {"paper_style", "block_restart"}
        or not continuation_used,
        "trajectory_bounded": evidence_trajectory.get("bounded") is True,
        "trajectory_nontrivial": evidence_trajectory.get("nontrivial") is True,
        "trajectory_finite_fraction_acceptable": finite_fraction_gate is not None and finite_fraction_gate >= 0.99,
        "trajectory_post_transient_sufficient": post_transient is not None
        and minimum_post is not None
        and post_transient >= minimum_post,
        "robustness_tested_h": evidence_robustness.get("tested_h") is True,
        "robustness_memory_requirement_satisfied": _close(q_value, 1.0)
        or evidence_robustness.get("tested_memory") is True,
        "robustness_tested_t_final": evidence_robustness.get("tested_t_final") is True,
        "robustness_tested_integrator": evidence_robustness.get("tested_integrator") is True,
        "robustness_consistent": evidence_robustness.get("consistent") is True,
        "hiddenness_tested_all_equilibria": hidden.get("tested_all_equilibria") is True,
        "hiddenness_tested_radii_recorded": bool(tested_radii) and all(value is not None for value in tested_radii),
        "hiddenness_required_radii_tested": bool(required_radii)
        and all(value is not None for value in tested_radii + required_radii)
        and all(
            any(_close(required, tested, atol=1.0e-15) for tested in tested_radii)
            for required in required_radii
        ),
        "hiddenness_equilibrium_radius_coverage_complete": hidden.get("coverage_by_equilibrium_radius_complete") is True,
        "hiddenness_target_contacts_recorded": target_hits is not None,
        "hiddenness_zero_equilibrium_contacts": target_hits == 0,
        "hiddenness_no_basin_intersection": hidden.get("basin_intersection_detected") is False,
        "hiddenness_basin_controls_complete": hidden.get("basin_controls_complete") is True,
        "hiddenness_numerical_failures_recorded": numerical_failures is not None,
        "hiddenness_no_numerical_failures": numerical_failures == 0,
        "reproducibility_metadata_complete": metadata_complete,
    }
    audit.require(check, set(recomputed_conditions) == EXPECTED_GATE_CONDITIONS, "internal gate-condition schema differs from the expected workflow")
    audit.require(check, set(checked) == EXPECTED_GATE_CONDITIONS, "gate.checked_conditions has a missing or unexpected condition")
    for condition, recomputed in recomputed_conditions.items():
        audit.require(check, recomputed is True, f"recomputed gate condition did not pass: {condition}")
        audit.require(check, checked.get(condition) is recomputed, f"gate condition is forged/inconsistent: {condition}")

    stability_rows = [row for row in _sequence(stability_payload) if isinstance(row, Mapping)]
    stability_residuals = [_float(row.get("rhs_residual")) for row in stability_rows]
    expected_max_residual = (
        max(value for value in stability_residuals if value is not None)
        if len(stability_residuals) == 3 and all(value is not None for value in stability_residuals)
        else None
    )
    audit.require(check, equilibria.get("all_found") is (len(stability_rows) == 3), "gate equilibria.all_found differs from 06")
    audit.require(
        check,
        expected_max_residual is not None and _close(equilibria.get("max_residual"), expected_max_residual, atol=1.0e-12),
        "gate maximum equilibrium residual differs from 06",
    )
    audit.require(check, matignon.get("all_classified") is True and _close(matignon.get("q"), 1.0), "gate Matignon q=1 classification is wrong")
    audit.require(check, seed.get("localized") is True and seed.get("method") == "continuation", "gate seed localization/method is wrong")
    audit.require(check, seed.get("source") == "candidate_initial_state_is_selected_parameter_continuation_endpoint", "gate seed source is wrong")
    audit.require(check, seed.get("theoretical_seed_source") == "direct_integer_transfer_from_declared_equations", "gate theoretical seed source is wrong")
    metadata_continuation = _mapping(run_metadata.get("continuation"))
    audit.require(
        check,
        _same_json_semantics(continuation, metadata_continuation),
        "gate evidence continuation differs from run_metadata continuation",
    )

    boundedness = _mapping(dynamics.get("boundedness"))
    audit.require(check, evidence_trajectory.get("bounded") is True and evidence_trajectory.get("nontrivial") is True, "gate trajectory is not bounded/nontrivial")
    audit.require(check, _close(evidence_trajectory.get("finite_fraction"), boundedness.get("finite_fraction")), "gate finite_fraction differs from diagnostics")
    audit.require(check, _integer(evidence_trajectory.get("post_transient_rows")) == _integer(boundedness.get("post_transient_rows")), "gate post-transient count differs from diagnostics")

    evidence_lyapunov = _mapping(evidence.get("lyapunov"))
    strict = dynamics.get("strict_exponents")
    evidence_exponents = _finite_vector(evidence_lyapunov.get("exponents"), 3)
    audit.require(
        check,
        isinstance(strict, tuple)
        and evidence_exponents is not None
        and all(_close(left, right, atol=1.0e-11) for left, right in zip(strict, evidence_exponents)),
        "gate Lyapunov spectrum differs from diagnostics",
    )
    audit.require(check, evidence_lyapunov.get("method_status") == "internal_controls_passed", "gate Lyapunov controls did not pass")

    evidence_zero_one = _mapping(evidence.get("zero_one"))
    audit.require(check, expected_k is not None and _close(evidence_zero_one.get("K"), expected_k), "gate 0-1 K differs from return-map median")
    zero_one_threshold = _float(tolerances.get("zero_one_chaos_threshold"))
    expected_zero_one_state = (
        "zero_one_chaotic_candidate"
        if expected_k is not None and zero_one_threshold is not None and expected_k >= zero_one_threshold
        else "zero_one_inconclusive"
    )
    audit.require(
        check,
        _close(zero_one_threshold, 0.7)
        and evidence_zero_one.get("state") == expected_zero_one_state
        and evidence_zero_one.get("gate_applicable") is True,
        "gate 0-1 classification is wrong",
    )
    evidence_spectrum = _mapping(evidence.get("spectrum"))
    audit.require(check, evidence_spectrum.get("gate_applicable") is False, "gate incorrectly makes FFT decisive")
    audit.require(check, evidence_spectrum.get("state_global") == _mapping(diagnostics.get("spectrum")).get("state_global"), "gate FFT state differs from diagnostics")
    evidence_poincare = _mapping(evidence.get("poincare"))
    audit.require(check, evidence_poincare.get("gate_applicable") is True, "Poincare evidence must be applicable")
    audit.require(check, _integer(evidence_poincare.get("crossing_count")) == _integer(_mapping(dynamics.get("poincare")).get("crossing_count")), "gate Poincare count differs from diagnostics")
    poincare_without_gate_flag = {key: value for key, value in evidence_poincare.items() if key != "gate_applicable"}
    # The gate persists the numerical Poincare summary but not the procedural
    # section metadata.  That metadata is audited exhaustively against the
    # reconstructed crossings in ``_audit_dynamics`` above.
    expected_gate_poincare = {
        key: value
        for key, value in _mapping(dynamics.get("poincare")).items()
        if key != "section_metadata"
    }
    audit.require(
        check,
        _same_json_semantics(poincare_without_gate_flag, expected_gate_poincare),
        "gate Poincare numerical summary differs from diagnostics",
    )

    audit.require(
        check,
        _same_json_semantics(evidence_robustness, robustness),
        "gate robustness differs from 08_robustness_matrix.json",
    )
    audit.require(check, hidden.get("tested_all_equilibria") is True, "gate hiddenness does not cover all equilibria")
    audit.require(check, tuple(_float(value) for value in _sequence(hidden.get("tested_radii"))) == (1.0e-5, 1.0e-3, 1.0e-2), "gate tested radii are wrong")
    audit.require(check, tuple(_float(value) for value in _sequence(hidden.get("required_radii"))) == (1.0e-5, 1.0e-3, 1.0e-2), "gate required radii are wrong")
    audit.require(check, _integer(hidden.get("target_hits_from_equilibria")) == _integer(hidden_summary.get("target_hits")) == 0, "gate target-hit count is inconsistent")
    audit.require(check, hidden.get("basin_intersection_detected") is False, "gate reports a basin intersection")
    audit.require(check, hidden.get("basin_controls_complete") is True and hidden.get("coverage_by_equilibrium_radius_complete") is True, "gate basin coverage is incomplete")
    audit.require(check, _integer(hidden.get("numerical_failures")) == 0, "gate hiddenness has numerical failures")


def _has_png_signature(path: Path) -> bool:
    try:
        payload = path.read_bytes()
    except OSError:
        return False
    return payload.startswith(b"\x89PNG\r\n\x1a\n") and b"IEND" in payload[-64:]


def _has_pdf_signature(path: Path) -> bool:
    try:
        payload = path.read_bytes()
    except OSError:
        return False
    return payload.startswith(b"%PDF-") and b"%%EOF" in payload[-2048:]


def _audit_figures(
    audit: Audit,
    *,
    figure_store_root: Path,
    run_dir: Path,
    local_manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    run_id: str | None,
    source_bundle: str | None,
    candidate_parameters: Mapping[str, Any],
    require_active_promotion: bool,
) -> tuple[list[Mapping[str, Any]], int]:
    global_root = figure_store_root.resolve()
    rows = [row for row in _sequence(local_manifest.get("figures")) if isinstance(row, Mapping)]
    ids = [str(row.get("figure_id")) for row in rows]
    audit.require("figures", len(rows) == 8, f"local figure manifest must contain 8 rows, got {len(rows)}")
    audit.require("figures", set(ids) == set(FIGURE_IDS) and len(ids) == len(set(ids)), f"wrong local figure IDs: {ids}")
    local_digests: dict[tuple[str, str], str] = {}
    for row in rows:
        figure_id = str(row.get("figure_id"))
        audit.require("figures", row.get("run_id") == run_id, f"local figure {figure_id} has wrong run_id")
        audit.require(
            "figures",
            row.get("global_promotion_requested") is True and row.get("promoted_to_global_manifest") is True,
            f"local figure {figure_id} is not recorded as globally promoted",
        )
        metadata = _mapping(row.get("metadata"))
        audit.require(
            "figures",
            metadata.get("quick_smoke_only") is False,
            f"local figure {figure_id} is marked quick-only",
        )
        audit.require(
            "figures",
            metadata.get("scientific_source_bundle_sha256") == source_bundle,
            f"local figure {figure_id} has wrong source bundle",
        )
        for suffix in ("png", "pdf"):
            path_key = f"local_{suffix}"
            hash_key = f"{suffix}_sha256"
            raw_path = row.get(path_key)
            expected_relative = f"figures/{figure_id}.{suffix}"
            path, path_error = _safe_relative_path(raw_path, parent=run_dir)
            safe = path is not None and raw_path == expected_relative
            rendered_path_error = path_error if path_error is not None else repr(raw_path)
            if not audit.require("figures", safe, f"unsafe/noncanonical {path_key} for {figure_id}: {rendered_path_error}"):
                continue
            assert path is not None
            if not audit.require("figures", path.is_file() and not path.is_symlink(), f"missing local {suffix} for {figure_id}"):
                continue
            expected = row.get(hash_key)
            actual = _sha256_file(path)
            hash_ok = audit.require(
                "figures",
                isinstance(expected, str) and SHA256_PATTERN.fullmatch(expected) is not None and actual == expected,
                f"local {suffix} hash mismatch for {figure_id}",
            )
            signature_ok = _has_png_signature(path) if suffix == "png" else _has_pdf_signature(path)
            audit.require("figures", signature_ok, f"local {figure_id}.{suffix} is not a valid-signature {suffix.upper()} file")
            if hash_ok and signature_ok:
                local_digests[(figure_id, suffix)] = actual
            expected_central = global_root / "by_run" / str(run_id) / suffix / f"{figure_id}.{suffix}"
            central = _resolve_recorded_path(
                _mapping(row.get("central_paths")).get(suffix),
                relative_base=global_root.parent,
                confined_to=global_root,
            )
            audit.require(
                "global_figures",
                central == expected_central,
                f"central path mismatch for {figure_id}.{suffix}: {central}",
            )
            global_paths = [expected_central]
            if require_active_promotion:
                global_paths.extend(
                    (
                        global_root / "current" / suffix / f"{figure_id}.{suffix}",
                        global_root
                        / "by_export"
                        / "mavpd_integer_hidden_chaos_report"
                        / suffix
                        / f"{figure_id}.{suffix}",
                    )
                )
            for global_path in global_paths:
                if not audit.require(
                    "global_figures",
                    global_path.is_file() and not global_path.is_symlink(),
                    f"missing promoted figure: {global_path}",
                ):
                    continue
                audit.require(
                    "global_figures",
                    _sha256_file(global_path) == actual,
                    f"promoted hash mismatch: {global_path}",
                )
        metadata_path = global_root / "by_run" / str(run_id) / "metadata" / f"{figure_id}.json"
        global_metadata = _read_json(audit, metadata_path, f"global metadata for {figure_id}")
        if global_metadata is not None:
            audit.require(
                "global_figures",
                _same_json_semantics(global_metadata, metadata),
                f"global metadata differs from local manifest for {figure_id}",
            )
        expected_parameters = EXPECTED_BASE_PARAMETERS if figure_id.startswith("00_") else candidate_parameters
        expected_integrator = "display_only" if figure_id.startswith("00_") else "DOP853"
        audit.require("figures", metadata.get("caption_key") == f"fig_{figure_id}", f"wrong caption key for {figure_id}")
        audit.require(
            "figures",
            metadata.get("source_script") == SCIENTIFIC_SOURCE_FIXED_FILES[0],
            f"wrong source script for {figure_id}",
        )
        audit.require("figures", metadata.get("source_function") == "run_contract/run_diagnostics/run_hiddenness", f"wrong source function for {figure_id}")
        audit.require("figures", metadata.get("data_sources") == FIGURE_DATA_SOURCES.get(figure_id), f"wrong data sources for {figure_id}")
        audit.require("figures", metadata.get("system_id") == "modified_van_der_pol_duffing", f"wrong system ID for {figure_id}")
        audit.require("figures", _close(metadata.get("q"), 1.0, rtol=0.0), f"wrong q for {figure_id}")
        audit.require("figures", metadata.get("integrator") == expected_integrator, f"wrong integrator for {figure_id}")
        audit.require("figures", metadata.get("memory_mode") == "not_applicable_integer_q1", f"wrong memory mode for {figure_id}")
        audit.require("figures", _close(metadata.get("t_final"), 900.0), f"wrong t_final for {figure_id}")
        audit.require("figures", _close(metadata.get("t_burn"), 300.0), f"wrong t_burn for {figure_id}")
        recorded_parameters = _mapping(metadata.get("parameters"))
        audit.require(
            "figures",
            set(recorded_parameters) == set(expected_parameters)
            and all(_close(recorded_parameters.get(key), value) for key, value in expected_parameters.items()),
            f"wrong parameters for {figure_id}",
        )

    audit.require("receipt", receipt.get("status") == "committed", "global promotion receipt is not committed")
    audit.require("receipt", receipt.get("run_id") == run_id, "global promotion receipt has wrong run_id")
    audit.require(
        "receipt",
        receipt.get("scientific_source_bundle_sha256") == source_bundle,
        "global promotion receipt has wrong scientific source bundle",
    )
    audit.require("receipt", _integer(receipt.get("figure_count")) == 8, "global promotion receipt figure_count must be 8")
    receipt_ids = list(_sequence(receipt.get("figure_ids")))
    audit.require("receipt", receipt_ids == list(FIGURE_IDS), "global promotion receipt has wrong/duplicate/out-of-order figure IDs")

    manifest_json = global_root / "manifests" / "figure_manifest.json"
    manifest_csv = global_root / "manifests" / "figure_manifest.csv"
    receipt_path_values = list(_sequence(receipt.get("global_manifest_paths")))
    receipt_paths = {
        _resolve_recorded_path(
            raw,
            relative_base=global_root.parent,
            confined_to=global_root,
        )
        for raw in receipt_path_values
    }
    audit.require(
        "receipt",
        len(receipt_path_values) == 2 and receipt_paths == {manifest_json, manifest_csv},
        f"receipt global_manifest_paths must identify the active JSON/CSV pair: {receipt_paths}",
    )
    if not require_active_promotion:
        return [], len(local_digests)

    global_payload = _read_json(audit, manifest_json, "global figure JSON manifest")
    global_rows = global_payload if isinstance(global_payload, list) else []
    active = [row for row in global_rows if isinstance(row, Mapping) and row.get("figure_id") in FIGURE_IDS]
    audit.require(
        "global_manifest",
        len(active) == 8 and {row.get("figure_id") for row in active} == set(FIGURE_IDS),
        "active global JSON manifest must contain exactly the 8 MAVPD figure IDs",
    )
    for row in active:
        figure_id = str(row.get("figure_id"))
        audit.require("global_manifest", row.get("run_id") == run_id, f"global manifest run_id mismatch for {figure_id}")
        audit.require(
            "global_manifest",
            row.get("kind") == "mavpd_integer_hidden_chaos",
            f"global manifest kind mismatch for {figure_id}",
        )
        audit.require(
            "global_manifest",
            row.get("export_targets") == ["mavpd_integer_hidden_chaos_report"],
            f"global manifest export_targets mismatch for {figure_id}",
        )
        for suffix in ("png", "pdf"):
            path = _resolve_recorded_path(
                row.get(f"{suffix}_path"),
                relative_base=global_root.parent,
                confined_to=global_root,
            )
            expected = global_root / "by_run" / str(run_id) / suffix / f"{figure_id}.{suffix}"
            audit.require("global_manifest", path == expected, f"global manifest {suffix}_path mismatch for {figure_id}")
            if path and path.is_file() and (figure_id, suffix) in local_digests:
                audit.require(
                    "global_manifest",
                    _sha256_file(path) == local_digests[(figure_id, suffix)],
                    f"global manifest target hash mismatch for {figure_id}.{suffix}",
                )
        metadata_path = _resolve_recorded_path(
            row.get("metadata_path"),
            relative_base=global_root.parent,
            confined_to=global_root,
        )
        expected_metadata = global_root / "by_run" / str(run_id) / "metadata" / f"{figure_id}.json"
        audit.require(
            "global_manifest",
            metadata_path == expected_metadata,
            f"global manifest metadata_path mismatch for {figure_id}",
        )

    csv_rows = _read_csv(audit, manifest_csv, "global figure CSV manifest")
    active_csv = [row for row in csv_rows if row.get("figure_id") in FIGURE_IDS]
    audit.require(
        "global_manifest",
        len(active_csv) == 8 and {row.get("figure_id") for row in active_csv} == set(FIGURE_IDS),
        "active global CSV manifest must contain exactly the 8 MAVPD figure IDs",
    )
    for row in active_csv:
        audit.require(
            "global_manifest",
            row.get("run_id") == run_id,
            f"global CSV manifest run_id mismatch for {row.get('figure_id')}",
        )
        audit.require(
            "global_manifest",
            set(row) == set(GLOBAL_MANIFEST_FIELDS),
            f"global CSV manifest columns differ for {row.get('figure_id')}",
        )
    json_by_id = {str(row.get("figure_id")): row for row in active}
    csv_by_id = {str(row.get("figure_id")): row for row in active_csv}
    for figure_id in FIGURE_IDS:
        json_row = json_by_id.get(figure_id)
        csv_row = csv_by_id.get(figure_id)
        if json_row is None or csv_row is None:
            continue
        for field in GLOBAL_MANIFEST_FIELDS:
            value = json_row.get(field, "")
            expected_text = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            audit.require(
                "global_manifest",
                csv_row.get(field) == expected_text,
                f"global JSON/CSV manifest mismatch for {figure_id}.{field}",
            )
    return active, len(local_digests)


def _audit_timings(
    audit: Audit,
    timing_rows: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    phase_kind: tuple[str, ...] | None,
) -> dict[str, float]:
    expected_phases = RESUMED_TIMING_PHASES if phase_kind == RESUMED_PHASES else DIRECT_TIMING_PHASES
    phases = tuple(row.get("phase", "") for row in timing_rows)
    audit.require(
        "timings",
        phases == expected_phases,
        f"phase_timings.csv must contain exact ordered phases {list(expected_phases)}, got {list(phases)}",
    )
    parsed: dict[str, float] = {}
    for row in timing_rows:
        phase = row.get("phase", "")
        seconds = _float(row.get("seconds"))
        audit.require("timings", seconds is not None and seconds >= 0.0, f"invalid duration for {phase}: {row.get('seconds')!r}")
        expected_timing_source = (
            "sum_of_persisted_and_resumed_phase_timers_not_wall_clock"
            if phase == "total_recorded_phase_seconds"
            else "perf_counter"
        )
        audit.require(
            "timings",
            row.get("timing_source") == expected_timing_source,
            f"timing_source for {phase} must be {expected_timing_source!r}",
        )
        if seconds is not None:
            parsed[phase] = seconds

    manifest_rows = [row for row in _sequence(manifest.get("timings")) if isinstance(row, Mapping)]
    audit.require("timings", len(manifest_rows) == len(timing_rows), "manifest and CSV timing row counts differ")
    for csv_row, manifest_row in zip(timing_rows, manifest_rows):
        csv_seconds = _float(csv_row.get("seconds"))
        manifest_seconds = _float(manifest_row.get("seconds"))
        audit.require(
            "timings",
            csv_row.get("phase") == manifest_row.get("phase")
            and csv_row.get("timing_source") == manifest_row.get("timing_source")
            and csv_seconds is not None
            and manifest_seconds is not None
            and math.isclose(csv_seconds, manifest_seconds, rel_tol=1.0e-12, abs_tol=1.0e-12),
            f"manifest timing differs from CSV for phase {csv_row.get('phase')}",
        )

    promotion_seconds = _float(receipt.get("seconds"))
    gate_phase = "candidate_gate_resumed" if phase_kind == RESUMED_PHASES else "candidate_gate"
    total_phase = "total_recorded_phase_seconds" if phase_kind == RESUMED_PHASES else "total"
    audit.require(
        "timings",
        promotion_seconds is not None and promotion_seconds > 0.0,
        f"receipt promotion duration must be positive, got {receipt.get('seconds')!r}",
    )
    if promotion_seconds is not None:
        audit.require(
            "timings",
            parsed.get(gate_phase, -1.0) >= promotion_seconds,
            f"{gate_phase} does not include the promotion duration",
        )
        audit.require(
            "timings",
            parsed.get(total_phase, -1.0) >= promotion_seconds,
            f"{total_phase} does not include the promotion duration",
        )
    if total_phase in parsed:
        component_sum = sum(value for phase, value in parsed.items() if phase != total_phase)
        tolerance = max(1.0e-9, abs(component_sum) * 1.0e-10)
        audit.require(
            "timings",
            parsed[total_phase] + tolerance >= component_sum,
            f"{total_phase}={parsed[total_phase]} is smaller than component sum {component_sum}",
        )
    manifest_promotion = _mapping(manifest.get("global_figure_promotion"))
    audit.require(
        "timings",
        _same_json_semantics(manifest_promotion, receipt),
        "run_manifest.global_figure_promotion differs from the committed receipt",
    )
    return parsed


def validate_run(
    run_dir: str | Path,
    repo_root: str | Path | None = None,
    scientific_source_root: str | Path | None = None,
    *,
    figure_store_root: str | Path | None = None,
    require_active_promotion: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable, read-only validation summary."""

    audit = Audit()
    run_path = Path(run_dir).expanduser().resolve()
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = (
        Path(scientific_source_root).expanduser().resolve()
        if scientific_source_root is not None
        else root
    )
    figure_root = (
        Path(figure_store_root).expanduser().resolve()
        if figure_store_root is not None
        else (root / "outputs" / "library_figures").resolve()
    )
    summary: dict[str, Any] = {
        "validator": "mavpd_integer_hidden_chaos_run",
        "validation_scope": "finite_numerical_hidden_chaos_under_tested_neighborhoods",
        "run_dir": str(run_path),
        "repo_root": str(root),
        "scientific_source_root": str(source_root),
        "figure_store_root": str(figure_root),
        "active_promotion_required": bool(require_active_promotion),
        "read_only": True,
    }
    audit.require("paths", root.is_dir(), f"repository root does not exist: {root}")
    audit.require("paths", source_root.is_dir(), f"scientific source root does not exist: {source_root}")
    audit.require("paths", figure_root.is_dir(), f"figure store root does not exist: {figure_root}")
    audit.require("paths", run_path.is_dir(), f"run directory does not exist: {run_path}")
    audit.require("paths", _path_is_within(run_path, root), f"run directory must resolve within repository root: {run_path}")
    if (
        not root.is_dir()
        or not source_root.is_dir()
        or not figure_root.is_dir()
        or not run_path.is_dir()
        or not _path_is_within(run_path, root)
    ):
        summary.update(ok=False, checks=audit.checks, errors=audit.errors)
        return summary

    status_obj = _read_json(audit, run_path / "run_status.json", "run status")
    manifest_obj = _read_json(audit, run_path / "run_manifest.json", "run manifest")
    contract_obj = _read_json(audit, run_path / "00_system_contract.json", "system contract")
    route_obj = _read_json(audit, run_path / "01_direct_seed_and_lambda_continuation.json", "direct route and lambda continuation")
    screen_contract_obj = _read_json(audit, run_path / "03_candidate_screening_contract.json", "candidate screening contract")
    selection_obj = _read_json(audit, run_path / "03_candidate_selection.json", "candidate selection")
    diagnostics_obj = _read_json(audit, run_path / "05_chaos_diagnostics.json", "chaos diagnostics")
    return_map_obj = _read_json(audit, run_path / "05_zero_one_return_map.json", "return-map 0-1 diagnostics")
    stability_obj = _read_json(audit, run_path / "06_equilibrium_stability.json", "equilibrium stability")
    hidden_obj = _read_json(audit, run_path / "07_hiddenness_summary.json", "hiddenness summary")
    robustness_obj = _read_json(audit, run_path / "08_robustness_matrix.json", "robustness matrix")
    gate_obj = _read_json(audit, run_path / "09_candidate_gate.json", "candidate gate")
    local_figures_obj = _read_json(audit, run_path / "figures" / "figure_manifest.json", "local figure manifest")
    receipt_obj = _read_json(
        audit,
        run_path / "figures" / "global_promotion_receipt.json",
        "global promotion receipt",
    )
    essentials = (
        status_obj,
        manifest_obj,
        contract_obj,
        route_obj,
        screen_contract_obj,
        selection_obj,
        diagnostics_obj,
        hidden_obj,
        robustness_obj,
        gate_obj,
        local_figures_obj,
        receipt_obj,
    )
    if not all(isinstance(item, Mapping) for item in essentials):
        audit.fail("required_files", "one or more required JSON artifacts are absent or not objects")
        summary.update(ok=False, checks=audit.checks, errors=audit.errors)
        return summary
    if not isinstance(return_map_obj, list) or not isinstance(stability_obj, list):
        audit.fail("required_files", "return-map and equilibrium-stability artifacts must be JSON arrays")
        summary.update(ok=False, checks=audit.checks, errors=audit.errors)
        return summary

    status = _mapping(status_obj)
    manifest = _mapping(manifest_obj)
    contract = _mapping(contract_obj)
    route = _mapping(route_obj)
    screen_contract = _mapping(screen_contract_obj)
    selection = _mapping(selection_obj)
    diagnostics = _mapping(diagnostics_obj)
    hidden_summary = _mapping(hidden_obj)
    robustness = _mapping(robustness_obj)
    gate_file = _mapping(gate_obj)
    local_figures = _mapping(local_figures_obj)
    receipt = _mapping(receipt_obj)

    run_id, config_sha, phase_kind, ledger_count = _audit_status_and_ledger(audit, run_path, status)
    audit.require("case", manifest.get("case_id") == CASE_ID, f"wrong manifest case_id: {manifest.get('case_id')!r}")
    audit.require("case", contract.get("case_id") == CASE_ID, f"wrong contract case_id: {contract.get('case_id')!r}")
    audit.require(
        "status",
        (
            phase_kind == DIRECT_PHASES
            and manifest.get("quick_mode") is False
            and manifest.get("resumed_from_persisted_phase_record") is not True
        )
        or (
            phase_kind == RESUMED_PHASES
            and manifest.get("resumed_from_persisted_phase_record") is True
            and manifest.get("quick_mode") is not True
        ),
        "run manifest mode must agree exactly with the direct/resumed status phase sequence",
    )
    status_runtime = _mapping(status.get("runtime_environment"))
    manifest_runtime = _mapping(manifest.get("runtime_environment"))
    required_runtime = {"python_version", "python_implementation", "platform", "numpy_version", "scipy_version"}
    audit.require(
        "runtime",
        _same_json_semantics(status_runtime, manifest_runtime),
        "status and manifest runtime environments differ",
    )
    audit.require("runtime", set(status_runtime) == required_runtime, "runtime environment has missing or unexpected fields")
    for key in required_runtime:
        audit.require("runtime", isinstance(status_runtime.get(key), str) and bool(status_runtime.get(key).strip()), f"runtime {key} must be a non-empty string")

    recorded_snapshot = _mapping(status.get("scientific_source_snapshot"))
    current_snapshot = _audit_sources(audit, source_root, recorded_snapshot)
    source_bundle = recorded_snapshot.get("bundle_sha256") if isinstance(recorded_snapshot.get("bundle_sha256"), str) else None
    snapshot_manifests = _audit_snapshot_manifests(
        audit,
        run_dir=run_path,
        scientific_source_root=source_root,
        recorded_snapshot=recorded_snapshot,
    )

    local_rows = [row for row in _sequence(local_figures.get("figures")) if isinstance(row, Mapping)]
    gate_metadata = _mapping(_mapping(gate_file.get("evidence")).get("run_metadata"))
    required_identity = (
        ("manifest.run_id", manifest.get("run_id"), run_id),
        ("gate.evidence.run_metadata.run_id", gate_metadata.get("run_id"), run_id),
        ("receipt.run_id", receipt.get("run_id"), run_id),
        ("manifest.config_sha256", manifest.get("config_sha256"), config_sha),
    )
    for label, actual, expected in required_identity:
        audit.require("identity", actual == expected and expected is not None, f"{label} mismatch: {actual!r}")
    gate_snapshot = _mapping(
        _mapping(_mapping(_mapping(gate_file.get("evidence")).get("run_metadata")).get("provenance")).get(
            "scientific_source_snapshot"
        )
    )
    for label, snapshot in (
        ("manifest", _mapping(manifest.get("scientific_source_snapshot"))),
        ("contract", _mapping(contract.get("scientific_source_snapshot"))),
        ("selection", _mapping(selection.get("scientific_source_snapshot"))),
        ("gate run metadata", gate_snapshot),
    ):
        audit.require("identity", snapshot == recorded_snapshot, f"{label} scientific source snapshot differs from status")
    _audit_identity(
        audit,
        (
            ("status", status),
            ("manifest", manifest),
             ("contract", contract),
            ("route", route),
            ("selection", selection),
            ("diagnostics", diagnostics),
            ("gate", gate_file),
            ("local figures", local_figures),
            ("receipt", receipt),
        ),
        expected_run_id=run_id,
        expected_config_sha=config_sha,
        expected_source_bundle=source_bundle,
    )
    _audit_claim_scope(audit, contract, selection, gate_file, manifest, hidden_summary)

    continuation_rows = _read_csv(audit, run_path / "02_parameter_continuation.csv", "parameter continuation")
    screen_rows = _read_csv(audit, run_path / "03_candidate_screening.csv", "candidate screening")
    screen_probe_rows = _read_csv(audit, run_path / "03_candidate_screening_probes.csv", "candidate screening probes")
    trajectory_rows = _read_csv(audit, run_path / "04_candidate_trajectory.csv", "candidate trajectory")
    lyapunov_rows = _read_csv(audit, run_path / "05_lyapunov_convergence.csv", "Lyapunov convergence")
    poincare_rows = _read_csv(audit, run_path / "05_poincare_section.csv", "Poincare section")
    stride_rows = _read_csv(audit, run_path / "05_zero_one_stride_sensitivity.csv", "0-1 stride sensitivity")
    spectrum_rows = _read_csv(audit, run_path / "05_normalized_fft_power.csv", "normalized FFT power")

    contract_info = _audit_contract_semantics(audit, contract)
    lineage = _audit_route_and_selection(
        audit,
        route,
        continuation_rows,
        screen_rows,
        screen_probe_rows,
        screen_contract,
        selection,
        manifest,
        contract_info,
    )
    candidate_equilibria = _audit_stability_semantics(
        audit,
        stability_obj,
        _mapping(lineage.get("candidate_parameters")),
    )
    dynamics = _audit_dynamics_semantics(
        audit,
        trajectory_rows=trajectory_rows,
        diagnostics=diagnostics,
        lyapunov_rows=lyapunov_rows,
        poincare_rows=poincare_rows,
        stride_rows=stride_rows,
        return_map_payload=return_map_obj,
        spectrum_rows=spectrum_rows,
        robustness=robustness,
        manifest=manifest,
        lineage=lineage,
    )

    probe_rows = _read_csv(audit, run_path / "07_hiddenness_probes.csv", "hiddenness probes")
    initial_rows = _read_csv(
        audit,
        run_path / "07_hiddenness_initial_conditions.csv",
        "hiddenness initial conditions",
    )
    probe_counts = _audit_probe_csvs(
        audit,
        probe_rows,
        initial_rows,
        hidden_summary,
        candidate_equilibria=candidate_equilibria,
        candidate_stability=stability_obj,
        reference_threshold=dynamics.get("reference_threshold"),
        reference_ambiguity_margin=dynamics.get("reference_ambiguity_margin"),
    )
    _audit_gate_semantics(
        audit,
        gate_file,
        hidden_summary,
        diagnostics,
        dynamics,
        robustness,
        lineage,
        stability_obj,
        status_runtime,
    )

    global_rows, figure_hashes = _audit_figures(
        audit,
        figure_store_root=figure_root,
        run_dir=run_path,
        local_manifest=local_figures,
        receipt=receipt,
        run_id=run_id,
        source_bundle=source_bundle,
        candidate_parameters=_mapping(lineage.get("candidate_parameters")),
        require_active_promotion=bool(require_active_promotion),
    )
    audit.require(
        "figures",
        _same_json_semantics(manifest.get("figures"), local_rows),
        "run_manifest.figures differs from figures/figure_manifest.json",
    )
    audit.require(
        "probe_summary",
        _same_json_semantics(manifest.get("hiddenness"), hidden_summary),
        "run_manifest.hiddenness differs from 07_hiddenness_summary.json",
    )
    timing_rows = _read_csv(audit, run_path / "phase_timings.csv", "phase timings")
    timing_values = _audit_timings(audit, timing_rows, manifest, receipt, phase_kind)

    summary.update(
        {
            "ok": not audit.errors,
            "run_id": run_id,
            "config_sha256": config_sha,
            "scientific_source_bundle_sha256": source_bundle,
            "current_source_bundle_matches": bool(
                current_snapshot
                and current_snapshot.get("bundle_sha256") == source_bundle
            ),
            "scientific_source_files_validated": len(_mapping(current_snapshot).get("files", {})),
            "snapshot_manifests_validated": snapshot_manifests,
            "ledger_artifacts_validated": ledger_count,
            "completed_phases": list(_sequence(status.get("completed_phases"))),
            "probe_counts": probe_counts,
            "total_probes": len(probe_rows),
            "local_figure_pairs": len(local_rows),
            "local_figure_hashes_validated": figure_hashes,
            "global_manifest_entries": len(global_rows),
            "active_promotion_checked": bool(require_active_promotion),
            "promotion_seconds": _float(receipt.get("seconds")),
            "timings_seconds": timing_values,
            "frequency_sweep_used": False if audit.checks.get("no_frequency_sweep", {}).get("ok") else None,
            "global_hiddenness_proved": False,
            "checks": audit.checks,
            "errors": audit.errors,
        }
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="completed full MAVPD run directory")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root; defaults to the root containing this validator",
    )
    parser.add_argument(
        "--scientific-source-root",
        type=Path,
        default=None,
        help=(
            "root whose scientific sources must match the run snapshot; defaults to --repo-root. "
            "Use this for runs launched from an immutable source snapshot distinct from the "
            "repository that contains the staged run"
        ),
    )
    parser.add_argument(
        "--figure-store-root",
        type=Path,
        default=None,
        help=(
            "absolute or relative path to the configured library_figures store; defaults strictly "
            "to --repo-root/outputs/library_figures. All recorded global figure paths are confined "
            "to its resolved absolute path"
        ),
    )
    parser.add_argument(
        "--require-active-promotion",
        action="store_true",
        help=(
            "also require current/, by_export/, and the active JSON/CSV global manifests to identify "
            "this run. By default the audit is immutable/history-safe and validates only by_run plus "
            "the committed receipt"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = validate_run(
            args.run_dir,
            args.repo_root,
            args.scientific_source_root,
            figure_store_root=args.figure_store_root,
            require_active_promotion=args.require_active_promotion,
        )
    except Exception as error:  # fail closed while preserving machine-readable output
        root_argument = args.repo_root if args.repo_root is not None else Path(__file__).parents[2]
        source_argument = (
            args.scientific_source_root
            if args.scientific_source_root is not None
            else root_argument
        )
        figure_argument = (
            args.figure_store_root
            if args.figure_store_root is not None
            else Path(root_argument) / "outputs" / "library_figures"
        )
        summary = {
            "validator": "mavpd_integer_hidden_chaos_run",
            "ok": False,
            "read_only": True,
            "run_dir": _best_effort_path_text(args.run_dir),
            "repo_root": _best_effort_path_text(root_argument),
            "scientific_source_root": _best_effort_path_text(source_argument),
            "figure_store_root": _best_effort_path_text(figure_argument),
            "active_promotion_required": bool(args.require_active_promotion),
            "checks": {"internal_validation": {"ok": False, "assertions": 1}},
            "errors": [{"check": "internal_validation", "message": f"{type(error).__name__}: {error}"}],
        }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if summary.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
