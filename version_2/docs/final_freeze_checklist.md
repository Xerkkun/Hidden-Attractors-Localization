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
