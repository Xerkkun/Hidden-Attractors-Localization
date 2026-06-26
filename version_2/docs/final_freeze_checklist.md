# Final Freeze Checklist

The library is frozen as an auditable framework for hidden-attractor candidate
localization, Caputo numerical simulation, finite-time chaos evidence, and
sampled-neighborhood hiddenness assessment for commensurate Caputo systems
compatible with scalar Lur'e form.

## Checklist

- [x] Algebraic tests pass.
- [x] Integrator tests pass.
- [x] Reproducibility metadata tests pass.
- [x] `candidate_gate` tests pass.
- [x] The `regression` marker is available.
- [x] Phase F is frozen as a finite-time evidence layer.
- [x] Hiddenness states are normalized.
- [x] Active documentation centralizes methodological cautions.
- [x] The JSON audit blocks obsolete strong labels in new outputs.

## Commands

```bash
python -m pytest tests/test_scientific_software_algebra.py -q
python -m pytest tests/test_scientific_software_integrators.py -q
python -m pytest tests/test_scientific_software_hiddenness.py -q
python -m pytest tests/test_reproducibility_metadata.py -q
python -m pytest tests/test_candidate_gate.py -q
python -m pytest tests/test_freeze_audit_json_outputs.py -q
python -m pytest -m regression -q
python -m pytest tests -q
```

## Chaos states

- `strong_chaos_evidence`
- `chaotic_dynamics_supported`
- `chaos_evidence_inconclusive`
- `regular_or_periodic_candidate`
- `unbounded_or_diverged`

## Hiddenness states

- `hiddenness_supported_under_tested_neighborhoods`
- `compatible_with_hiddenness_under_tested_radii`
- `self_excited_contact_detected`
- `hiddenness_inconclusive`
- `candidate_not_reproducible`
- `numerical_failure`

## G2. Final Freeze Consistency Pass

- [x] Normalize case integration and method comparison vocabularies across F6/F7.
- [x] Remove obsolete `chaos_verified` and `hidden_verified` keys from new summaries and reports.
- [x] Separate `CURRENT_FINAL_LABELS` and `LEGACY_FINAL_LABELS`, normalizing legacy verdicts immediately in `StageEnvelope`.
- [x] Link `HiddennessTestResult` with candidate evidence payload, executing fallback checks and demoting verdicts when necessary.
- [x] Update `candidate_gate.py` to output both `hiddenness_evidence_level` and `chaos_evidence_level`.
- [x] Create and run `run_final_freeze_audit.py` to generate `validation/freeze_audit/final_freeze_pytest_summary.json` with `freeze_ready=true`.

## G3. Final Freeze Cleanup and Published-Validation Coverage Audit

- [x] Published validation coverage matrix is complete.
- [x] Every extracted published case is classified.
- [x] Missing published values are not invented.



