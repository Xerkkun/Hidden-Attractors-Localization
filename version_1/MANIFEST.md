# Version 1 manifest

V1 agrupa los scripts historicos en `version_1/legacy_root/`. Esa carpeta es
una copia de la raiz anterior del proyecto para conservar contratos de rutas,
nombres de salida y dependencias locales.

## Pipelines principales

- `legacy_root/unified_nyquist_hidden_pipeline.py`
- `legacy_root/unified_chua_integer_pipeline.py`
- `legacy_root/run_hidden_verify_frac_hybrid.py`
- `legacy_root/run_extended_search.py`
- `legacy_root/validate_chua_piecewise_case.py`

## Busqueda, continuacion y verificacion

- `machado_targeted_verification.py`
- `corrida1_refined_verification.py`
- `multiparameter_continuation.py`
- `lure_biased_multiparam_search.py`
- `lure_biased_multiparam_continuation.py`
- `lure_refined_route.py`
- `lure_candidate_manifest.py`
- `lure_robustness_and_control_tests.py`
- `lure_adaptive_contact_test.py`
- `lure_rhoH_diagnostics.py`
- `seed_cloud_search.py`

## Diagnosticos y replicas

- `danca2017_chua_abm_replication.py`
- `chua_basin_comparison_h001.py`
- `chua_initial_cond.py`
- `formal_nyquist_df_chua.py`
- `harmonic_diagnostics.py`
- `equilibria_analysis.py`
- `debug_branch1_failures.py`
- `audit_and_homogenize_q.py`
- `biased_describing_function.py`

## Backends nativos y soporte

- `chua_basin_lib.c`
- `chua_frac_backend_lib.c`
- `chua_frac_lyapunov_efork_benettin.c`
- `chua_hidden_backend.c`
- `parallel_policy.py`
- `extended_search_utils.py`

## Compatibilidad hacia V2

Estos scripts conservan nombres historicos pero ya delegan en
`hidden_attractors.workflows`:

- `robustness_overlay_c_trajectories.py`
- `lure_top3_sphere_robustness.py`
- `refine_project_basin_classification.py`

## Regla

No agregar ejemplos ni analisis nuevos a V1. Agregarlos en `version_2/` y mover
la logica reusable a `version_2/hidden_attractors/`.
