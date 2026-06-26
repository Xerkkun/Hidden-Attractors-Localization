# Published Validation Coverage

This document outlines the coverage status of the published reference cases against the extracted truth data recorded in `docs/published_validation_data_extraction_v1.json`.

| Article | Case | Published data | Coverage status | What is validated | What remains reference-only | Release blocking |
| ------- | ---- | -------------- | --------------- | ----------------- | --------------------------- | ---------------- |
| Kuznetsov 2017 | `kuznetsov2017_case_18_hidden_chaotic` | Yes | `executable_regression` | Complete parameters, equilibria, and DF seed | N/A | No |
| Kuznetsov 2017 | `kuznetsov2017_case_21_hidden_chaotic_branch` | Yes | `future_extension` | N/A | Chaotic branch parameters, equilibria, and DF data | No |
| Kuznetsov 2017 | `kuznetsov2017_case_21_hidden_periodic_branch` | Yes | `future_extension` | N/A | Periodic branch parameters, equilibria, and DF data | No |
| Danca 2017 | `chua_fractional_saturation` | Yes | `partial_reference_implementation` | Fractional Chua saturation equations, parameters, Matignon stability, and neighborhood controls | Full reproduction not claimed: missing omega0, k, a0, seed coordinates, exact hidden-attractor IC, and Lyapunov exponents | No |
| Danca 2017 | `generalized_lorenz_fractional` | Yes | `reference_data_only` | N/A | Equations, parameters, equilibria, and eigenspectra | No |
| Danca 2017 | `rabinovich_fabrikant_fractional` | Yes | `reference_data_only` | N/A | Equations, parameters, equilibria, and eigenspectra | No |
| Wu 2023 | `wu2023_chua_fractional_arctan` | Yes | `partial_reference_implementation` | Equations, parameters, equilibria, and initial conditions | Full reproduction not claimed: missing omega0, k, a0, exact seed coordinates, and incomplete data for full independent seed/attractor reproduction | No |
| Danca & Kuznetsov 2018 | `DK2018_RF_q0999` | Yes | `diagnostic_comparison_with_discrepancies` | Equations, parameters, and comparison run data | Exact quantitative LE agreement (comparison run with documented block-restart discrepancies) | No |
| Danca & Kuznetsov 2018 | `DK2018_Lorenz_q0985` | Yes | `diagnostic_comparison_with_discrepancies` | Equations, parameters, and comparison run data | Exact quantitative LE agreement (comparison run with documented block-restart discrepancies) | No |
| Danca & Kuznetsov 2018 | `DK2018_4D_nonsmooth_q098` | Yes | `reference_data_only` | N/A | 4D nonsmooth equations, parameters, and qualitative LE estimates | No |
| Fischer 2020 | `jerk_system` | Yes | `diagnostic_comparison_with_discrepancies` | jerk_system equations, commensurate parameters, and comparison run data | Exact LCE/0-1 table quantitative values (comparison run with documented discrepancies) | No |
| Fischer 2020 | `financial_system` | Yes | `diagnostic_comparison_with_discrepancies` | financial_system equations, commensurate parameters, and comparison run data | Exact LCE/0-1 table quantitative values (comparison run with documented discrepancies) | No |
| Fischer 2020 | `four_wing_system` | Yes | `diagnostic_comparison_with_discrepancies` | four_wing_system equations, commensurate parameters, and comparison run data | Exact LCE/0-1 table quantitative values (comparison run with documented discrepancies) | No |

## Coverage Vocabulary Definitions

- **`executable_regression`**: The case has complete numerical values and is verified by an active test suite regression check.
- **`partial_reference_implementation`**: The model or part of the reference case is implemented, but the published article lacks enough numerical information or the repository uses a different documented numerical contract; therefore no full quantitative reproduction claim is made.
- **`reference_data_only`**: The data has been extracted from the publication for documentation/benchmark purposes but the system or model is not built-in to the current release.
- **`diagnostic_comparison_with_discrepancies`**: Run comparisons are implemented to audit the method behavior, but the results exhibit documented mathematical, procedural, or numerical discrepancies compared to the published values.
- **`future_extension`**: Reserved for post-release validation branches or future coverage extensions.
