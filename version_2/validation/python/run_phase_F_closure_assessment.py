#!/usr/bin/env python3
"""Write the conservative Phase F closure assessment artifacts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for item in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from hidden_attractors.analysis.phase_f_closure import (  # noqa: E402
    PHASE_F_CLOSURE_RULES,
    assess_phase_f_closure,
    build_phase_f_closure_matrix,
)
from f5_diagnostics_common import write_csv, write_json  # noqa: E402


CHAOS_VALIDATION_ROOT = PROJECT_ROOT / "validation" / "chaos_validation"
OUTPUT_ROOT = CHAOS_VALIDATION_ROOT / "phase_F_closure"
SUMMARY_PATH = OUTPUT_ROOT / "phase_F_closure_summary.json"
MATRIX_PATH = OUTPUT_ROOT / "phase_F_closure_matrix.csv"
DECISION_PATH = OUTPUT_ROOT / "phase_F_closure_decision.md"
RULES_PATH = OUTPUT_ROOT / "phase_F_closure_rules.json"
SCOPE_STATEMENT_PATH = OUTPUT_ROOT / "phase_F_diagnostic_scope_statement.md"
ACCEPTED_FRACTIONAL_POLICY = OUTPUT_ROOT / "accepted_fractional_lyapunov_validation_policy.md"
FISCHER_RESOLUTION = OUTPUT_ROOT / "fischer2020_discrepancy_resolution.md"
F4_SUMMARY = (
    CHAOS_VALIDATION_ROOT
    / "lyapunov_methods"
    / "F4_internal_validation"
    / "f4_internal_validation_summary.json"
)
F5_SUMMARY = CHAOS_VALIDATION_ROOT / "dynamics_diagnostics" / "f5_diagnostics_summary.json"
DK2018_SUMMARY = (
    CHAOS_VALIDATION_ROOT
    / "lyapunov_methods"
    / "fractional_variational_dk2018_block_restart_abm_gs_published"
    / "validation_summary.json"
)
FISCHER2020_SUMMARY = (
    CHAOS_VALIDATION_ROOT
    / "lyapunov_methods"
    / "fractional_cloned_dynamics_abm_gs_published"
    / "validation_summary.json"
)
F6_SUMMARY = CHAOS_VALIDATION_ROOT / "integrated_chaos_validator" / "integrated_chaos_summary.json"
F7_SUMMARY = CHAOS_VALIDATION_ROOT / "method_comparison" / "method_comparison_summary.json"


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            raise
        return None


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _decision_markdown(summary: dict[str, Any]) -> str:
    routes = summary["closure_routes"]
    criteria = summary["criteria"]
    return f"""# Phase F closure decision

## Decision

Phase F is structurally complete but not strictly closed as chaos validation.

```text
status: {summary["status"]}
strict_chaos_validation_closed: false
structured_diagnostics_closed: true
chaos_verified: false
hiddenness_verified: false
```

## Closure Routes

| Route | Status | Reason |
|---|---|---|
| A: `fractional_variational_abm_qr` | `{routes["route_A_fractional_variational_abm_qr"]["status"]}` | `{routes["route_A_fractional_variational_abm_qr"]["reason"]}` |
| B: Fischer 2020 cloned dynamics | `{routes["route_B_fractional_cloned_dynamics"]["status"]}` | `{routes["route_B_fractional_cloned_dynamics"]["reason"]}` |
| C: diagnostic scope closure | `{routes["route_C_diagnostic_scope_closure"]["status"]}` | `{routes["route_C_diagnostic_scope_closure"]["reason"]}` |

## Documented Assessments

Route A is labeled
`{routes["route_A_fractional_variational_abm_qr"]["status"]}`. The local
full-history QR implementation has F4 internal controls and sensitivity
evidence. The DK2018 long block-restart reproduction remains a separate
contract and does not promote full-history QR.

Route B is labeled
`{routes["route_B_fractional_cloned_dynamics"]["status"]}`. The Fischer 2020
published GS lane records `{routes["route_B_fractional_cloned_dynamics"]["rows_total"]}`
rows: `{routes["route_B_fractional_cloned_dynamics"]["rows_passed_quantitative"]}`
quantitative passes, `{routes["route_B_fractional_cloned_dynamics"]["rows_passed_sign_pattern"]}`
sign-pattern passes, and `{routes["route_B_fractional_cloned_dynamics"]["rows_failed"]}`
discrepancy rows. Bounded sensitivity sweeps were completed for the current
scope, while the discrepancies remain explicit.

## Strict Closure Boundary

Rigorous assessments were executed for the fractional Lyapunov lanes. They are
recorded as documented evidence, not discarded as failed work. Strict chaos
validation remains outside the current evidence scope because no accepted
fractional Lyapunov method has been applied to each fractional candidate:

```text
valid_fractional_lyapunov_method_per_candidate: {str(criteria["valid_fractional_lyapunov_method_per_candidate"]).lower()}
```

F4 internal controls, published reproduction attempts, sensitivity sweeps, F5
standardized diagnostics, and optional F6/F7 integration outputs remain
reproducible numerical evidence. They do not certify mathematical chaos or
hiddenness.
"""


def run() -> dict[str, Any]:
    """Read Phase F evidence and write a non-certifying closure assessment."""

    f4 = _read_json(F4_SUMMARY)
    f5 = _read_json(F5_SUMMARY)
    dk2018 = _read_json(DK2018_SUMMARY)
    fischer2020 = _read_json(FISCHER2020_SUMMARY)
    f6 = _read_json(F6_SUMMARY, required=False)
    f7 = _read_json(F7_SUMMARY, required=False)
    payload = assess_phase_f_closure(
        f4_summary=f4,
        f5_summary=f5,
        dk2018_summary=dk2018,
        fischer2020_summary=fischer2020,
        f6_summary=f6,
        f7_summary=f7,
        accepted_fractional_policy_exists=ACCEPTED_FRACTIONAL_POLICY.exists(),
        fischer_resolution_exists=FISCHER_RESOLUTION.exists(),
        diagnostic_scope_statement_exists=SCOPE_STATEMENT_PATH.exists(),
    )
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["inputs"] = {
        "f4_summary": _relative(F4_SUMMARY),
        "f5_summary": _relative(F5_SUMMARY),
        "dk2018_summary": _relative(DK2018_SUMMARY),
        "fischer2020_summary": _relative(FISCHER2020_SUMMARY),
        "lyapunov_registry": "hidden_attractors/analysis/lyapunov_methods.py",
        "f6_summary": _relative(F6_SUMMARY) if f6 is not None else None,
        "f7_summary": _relative(F7_SUMMARY) if f7 is not None else None,
        "accepted_fractional_lyapunov_validation_policy": (
            _relative(ACCEPTED_FRACTIONAL_POLICY)
            if ACCEPTED_FRACTIONAL_POLICY.exists()
            else None
        ),
        "fischer2020_discrepancy_resolution": (
            _relative(FISCHER_RESOLUTION) if FISCHER_RESOLUTION.exists() else None
        ),
        "diagnostic_scope_statement": _relative(SCOPE_STATEMENT_PATH),
    }
    write_json(SUMMARY_PATH, payload)
    write_json(RULES_PATH, PHASE_F_CLOSURE_RULES)
    write_csv(MATRIX_PATH, build_phase_f_closure_matrix(payload))
    DECISION_PATH.write_text(_decision_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    summary = run()
    print(f"Phase F status: {summary['status']}")
    print("Strict chaos validation closed: false")
    print("Structured diagnostics closed: true")
    print("Chaos verified: false")
    print("Hiddenness verified: false")


if __name__ == "__main__":
    main()
