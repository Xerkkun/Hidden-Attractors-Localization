# Validation Evidence

The validation workflow separates machine-readable evidence from scientific
interpretation. Keep this separation even when a stage is small.

## File Roles

| File type | Role |
|-----------|------|
| JSON | Traceability, run status, parameters, tolerances, software versions, seeds, and paths to generated evidence. |
| CSV | Numerical result tables that need filtering, comparison, or aggregation. |
| PNG/PDF | Visual evidence such as phase portraits, basin cuts, spectra, margins, and overlays. |
| MD | Short human explanation for one validation stage. |
| TEX/PDF | Final defensible report that cites stage summaries and selected evidence. |

JSON files do not replace the report. They record what a program needs to
reload a run. CSV files do not replace the JSON summary. They store tabular
measurements.

## Canonical Layout

Generated validation evidence belongs under `validation/`. The interpretive
documentation belongs under `docs/`. Reusable run contracts belong under
`configs/`.

The root contract below remains the fractional non-smooth Chua package. A
verified baseline for a different order or parameter set must not be mixed
into that tree. The integer Chua `q=1` baseline is therefore promoted under
`validation/reference_cases/chua_integer_q1/` and checked with
`configs/validation_chua_integer_q1.json`.

```text
version_2/
  configs/
    validation_contract.json

  validation/
    00_manifest/
      validation_manifest.json
      environment.json
      software_versions.json

    01_algebra/
      algebra_validation.md
      equilibria_summary.csv
      jacobian_check.csv
      eigenvalues_matignon_summary.csv
      matignon_margins.png
      algebra_validation_summary.json

    02_lure_df/
      lure_df_validation.md
      lure_equivalence_check.csv
      transfer_function_check.csv
      describing_function_check.csv
      machado_mu1_check.csv
      lure_df_validation_summary.json

    03_integrators/
      integrator_validation.md
      manufactured_solution_convergence.csv
      q1_limit_vs_solve_ivp.csv
      abm_vs_efork_short_time.csv
      memory_sensitivity.csv
      integrator_validation_summary.json

    04_candidates/
      candidate_selection.md
      q_sweep_summary.csv
      df_compare_summary.csv
      machado_sweep_summary.csv
      selected_candidates.json
      candidate_selection_summary.json

    05_dynamic_analysis/
      dynamic_analysis.md
      trajectory_metrics.csv
      fft_summary.csv
      psd_summary.csv
      lyapunov_summary.csv
      phase_3d.png
      projections.png
      dynamic_analysis_summary.json

    06_hiddenness/
      hiddenness_validation.md
      sphere_plan.csv
      sphere_raw.csv
      sphere_decision.csv
      basin_xy.csv
      basin_xy.png
      refined_basin_summary.json
      hiddenness_validation_summary.json

    07_robustness/
      robustness_validation.md
      robustness_overlay_metrics.csv
      robustness_summary.json
      overlay_3d.png
      robustness_validation_summary.json

    08_literature_comparison/
      literature_comparison.md
      danca_comparison_summary.csv
      abm_replication_summary.json
      literature_comparison_summary.json

    final_validation_report.tex
    final_validation_report.pdf
```

Each stage should follow the same local pattern:

```text
stage_directory/
  stage_validation.md
  numerical_results.csv
  figures.png
  stage_validation_summary.json
```

## Stage Summary JSON

Use one summary JSON per stage. Do not create one JSON file for every small
calculation unless the raw artifact is too large or structured for CSV.
Each stage in `configs/validation_contract.json` has both an ordered `id` and a
plain `slug`. The `id` names the directory, while the `slug` generates names
such as `<slug>_validation.md` and `<slug>_validation_summary.json`.

```json
{
  "id": "01_algebra",
  "slug": "algebra",
  "summary": "algebra_validation_summary.json",
  "example_summary": {
    "stage": "algebra",
    "status": "passed",
    "system": "fractional_nonsmooth_chua",
    "parameters": {
      "alpha": 8.4562,
      "beta": 12.0732,
      "gamma": 0.0052,
      "m0": -0.1768,
      "m1": -1.1468,
      "q": 0.9998
    },
    "checks": {
      "equilibria_residual_max": 2.1e-14,
      "jacobian_finite_difference_error_max": 4.8e-8,
      "matignon_margins_computed": true
    },
    "tolerances": {
      "equilibria_residual": 1e-10,
      "jacobian_error": 1e-6
    },
    "files": {
      "equilibria_summary": "equilibria_summary.csv",
      "jacobian_check": "jacobian_check.csv",
      "matignon_plot": "matignon_margins.png"
    }
  }
}
```

## Global Manifest

The global manifest in `validation/00_manifest/validation_manifest.json` should
record the validation identity, code version, environment, main system, global
parameters, and pointers to stage summaries.

```json
{
  "validation_id": "chua_fractional_validation_2026_05",
  "repository_commit": "...",
  "package_version": "0.1.0",
  "python_version": "...",
  "platform": "...",
  "main_system": "fractional nonsmooth Chua",
  "main_parameters": {},
  "stages": {
    "algebra": "01_algebra/algebra_validation_summary.json",
    "lure_df": "02_lure_df/lure_df_validation_summary.json",
    "integrators": "03_integrators/integrator_validation_summary.json",
    "candidates": "04_candidates/candidate_selection_summary.json",
    "dynamic_analysis": "05_dynamic_analysis/dynamic_analysis_summary.json",
    "hiddenness": "06_hiddenness/hiddenness_validation_summary.json",
    "robustness": "07_robustness/robustness_validation_summary.json",
    "literature_comparison": "08_literature_comparison/literature_comparison_summary.json"
  }
}
```

## Reporting Rule

Use JSON for traceability and validation status, CSV for numerical tables,
PNG/PDF for visual evidence, MD for short stage interpretation, and TEX/PDF for
the final scientific report.

## Contract Checker

After promoting real evidence into `validation/`, run:

```bash
hidden-attractors-check-validation
```

The command reads `configs/validation_contract.json` and verifies required
directories, required files, minimum JSON fields, manifest stage paths,
non-empty CSV tables, declared figure files, and the final report status. The
template-only tree is expected to fail until the stage artifacts are generated
or copied into place.

The integer-order reference case can be checked independently:

```bash
hidden-attractors-check-validation \
  --contract configs/validation_chua_integer_q1.json \
  --validation-root validation/reference_cases/chua_integer_q1
```

Its current package registers the theoretical report, library-produced
artifacts, a locally executed MATLAB reproduction, and a locally executed
Wolfram Language symbolic derivation. The Wolfram source intentionally omits
numerical parameter evaluation, so it supports algebraic traceability and is
not counted as an independent numerical trajectory reproduction. The Guan--Xie
comparison now includes its displayed Example 6 values for `omega0`, `k`,
`a0`, and the initial point, with relative differences stored in
`08_literature_comparison/paper_numeric_comparison.csv`.

The integration-dependent integer artifacts were regenerated after the
EFORK-3 third stage was aligned with the published ordering
`K3 = a31*K1 + a32*K2`. The report remains a registered algebraic and seed
source; current continuation, Lyapunov, spectrum, basin, and hiddenness
values come from the corrected rerun.

## Fractional Non-Smooth Chua Algebra Audit

The main fractional case now has a promoted algebra/Lur'e audit for
`q=0.9998`. Its generated outputs are stored in `validation/01_algebra/` and
`validation/02_lure_df/`; the interpretive report is
[`fractional_chua_algebra_validation.md`](fractional_chua_algebra_validation.md).

| Check | Result |
|---|---|
| Equilibria | MATLAB and Python reproduce `E0` and the symmetric outer pair at `x = +/-6.588307886539`. |
| Regional stability | Matignon classifies `E0` as stable and `E+`, `E-` as unstable. |
| Harmonic branches | MATLAB and Python reproduce two branches: `(omega, k) = (2.040286051079, 0.210022792962)` and `(3.244926730975, 0.956945404928)`. |
| Transfer convention | `W_code = -W_report`; therefore `1 + k W_code = 0` and `1 - k W_report = 0` are the same check. |
| Symbolic source | The supplied Wolfram file verifies the symbolic identities after its protected identifier `Tr` is renamed. |

Danca (2017), Section 3.3, is the direct published reference for this exact
non-smooth model, parameter set, order, and neighborhood-based hiddenness
experiment. Petras (2008) confirms the fractional PWL Chua model family and
physical derivation but uses different parameters and orders. Sene (2021)
supports the reporting methodology for fractional Chua stability and
dynamical diagnostics, but its model omits the `-gamma z` term; it is not a
numerical reference for Danca's case.

Only algebra and Lur'e/DF seed generation are marked complete here. Solver
convergence, independent integration, dynamic diagnostics, and hiddenness
controls remain separate required stages.

The numerical-method benchmark is stored separately from any chaotic-system
claim:

```bash
hidden-attractors-check-validation \
  --contract configs/validation_efork3_ghoreishi_ghaffari.json \
  --validation-root validation/reference_cases/efork3_ghoreishi_ghaffari
```

This reference case reproduces the three-stage EFORK terminal errors in
Ghoreishi, Ghaffari, and Saad (2023), Tables 3, 4, 9, and 10. The archived
Python scripts supplied by Dr. Luis Gerardo de la Fraga (CINVESTAV Unidad
Zacatenco) are registered as implementation provenance.
