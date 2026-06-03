# Published Validation Coverage

This document outlines the coverage status of the published reference cases against the extracted truth data recorded in `docs/published_validation_data_extraction_v1.json`.

| Article | Case | Published data | Coverage status | What is validated | What remains reference-only | Release blocking |
| ------- | ---- | -------------- | --------------- | ----------------- | --------------------------- | ---------------- |
| Kuznetsov 2017 | `kuznetsov2017_case_18_hidden_chaotic` | Yes | `executable_regression` | Complete parameters, equilibria, and DF seed | N/A | No |
| Kuznetsov 2017 | `kuznetsov2017_case_21_hidden_chaotic_branch` | Yes | `future_extension` | N/A | Chaotic branch parameters, equilibria, and DF data | No |
| Kuznetsov 2017 | `kuznetsov2017_case_21_hidden_periodic_branch` | Yes | `future_extension` | N/A | Periodic branch parameters, equilibria, and DF data | No |
| Danca 2017 | `chua_fractional_saturation` | Yes | `implemented_partial_reproduction` | Fractional Chua saturation equations, parameters, Matignon stability, and neighborhood controls | Exact hidden attractor seed/ICs and Lyapunov exponents (not reported in paper) | No |
| Danca 2017 | `generalized_lorenz_fractional` | Yes | `reference_data_only` | N/A | Equations, parameters, equilibria, and eigenspectra | No |
| Danca 2017 | `rabinovich_fabrikant_fractional` | Yes | `reference_data_only` | N/A | Equations, parameters, equilibria, and eigenspectra | No |
| Wu 2023 | `wu2023_chua_fractional_arctan` | Yes | `implemented_partial_reproduction` | Equations, parameters, equilibria, and initial conditions | Sweep plots, complexity maps, and DSP implementation | No |
| Danca & Kuznetsov 2018 | `DK2018_RF_q0999` | Yes | `implemented_partial_reproduction` | Equations, parameters, and comparison run data | Exact quantitative LE agreement (reproduced with documented block-restart discrepancies) | No |
| Danca & Kuznetsov 2018 | `DK2018_Lorenz_q0985` | Yes | `implemented_partial_reproduction` | Equations, parameters, and comparison run data | Exact quantitative LE agreement (reproduced with documented block-restart discrepancies) | No |
| Danca & Kuznetsov 2018 | `DK2018_4D_nonsmooth_q098` | Yes | `reference_data_only` | N/A | 4D nonsmooth equations, parameters, and qualitative LE estimates | No |
| Fischer 2020 | `jerk_system` | Yes | `implemented_partial_reproduction` | jer_system equations, commensurate parameters, and comparison run data | Exact LCE/0-1 table quantitative values (reproduced with documented discrepancies) | No |
| Fischer 2020 | `financial_system` | Yes | `implemented_partial_reproduction` | financial_system equations, commensurate parameters, and comparison run data | Exact LCE/0-1 table quantitative values (reproduced with documented discrepancies) | No |
| Fischer 2020 | `four_wing_system` | Yes | `implemented_partial_reproduction` | four_wing_system equations, commensurate parameters, and comparison run data | Exact LCE/0-1 table quantitative values (reproduced with documented discrepancies) | No |

## Coverage Vocabulary Definitions

- **`executable_regression`**: The case has complete numerical values and is verified by an active test suite regression check.
- **`implemented_partial_reproduction`**: The model or parts of the reference case are implemented and verified, but details are partially missing from the source text (e.g. initial conditions/seeds/Lyapunov parameters) or verified with documented mathematical/numerical discrepancies.
- **`reference_data_only`**: The data has been extracted from the publication for documentation/benchmark purposes but the system or model is not built-in to the current release.
- **`future_extension`**: Reserved for post-release validation branches or future coverage extensions.
