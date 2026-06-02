# Bibliographic Traceability Matrix

**Overall Status**: `WARNING`
**Claims Validated**: 12 / 12

## Claims Traceability Table

| Claim ID | Claim Type | Strong Claim | Required References | Provided References | Status |
|---|---|---|---|---|---|
| hidden_definition | HIDDEN_ATTRACTOR_DEFINITION | Yes | `leonov_kuznetsov_hidden_definition`, `kuznetsov_2017_chua_df` | `leonov_kuznetsov_hidden_definition`, `kuznetsov_2017_chua_df` | ✅ PASS |
| self_excited_vs_hidden | SELF_EXCITED_DEFINITION | Yes | `leonov_kuznetsov_hidden_definition` | `leonov_kuznetsov_hidden_definition` | ✅ PASS |
| df_chua_localization | DESCRIBING_FUNCTION_CHUA_LOCALIZATION | Yes | `kuznetsov_2017_chua_df` | `kuznetsov_2017_chua_df` | ✅ PASS |
| df_is_heuristic | DESCRIBING_FUNCTION_IS_HEURISTIC | Yes | `kuznetsov_2017_chua_df` | `kuznetsov_2017_chua_df` | ✅ PASS |
| caputo_memory | WEYL_CAPUTO_BRIDGE | Yes | `danca_2017_fractional_hidden` | `danca_2017_fractional_hidden` | ✅ PASS |
| caputo_abm_integration | CAPUTO_ABM_INTEGRATION | Yes | `diethelm_ford_freed_abm_caputo`, `danca_2017_fractional_hidden` | `diethelm_ford_freed_abm_caputo`, `danca_2017_fractional_hidden` | ✅ PASS |
| matignon_stability | FRACTIONAL_MATIGNON_STABILITY | Yes | `matignon_fractional_stability`, `danca_2017_fractional_hidden` | `matignon_fractional_stability`, `danca_2017_fractional_hidden` | ✅ PASS |
| danca_fractional_hiddenness | NONSMOOTH_CHUA_LIPSCHITZ_ABM | Yes | `danca_2017_fractional_hidden` | `danca_2017_fractional_hidden` | ✅ PASS |
| machado_fdf_describing | MACHADO_FDF | Yes | `machado_2015_fractional_describing_functions` | `machado_2015_fractional_describing_functions` | ✅ PASS |
| alternative_localization_review | ALTERNATIVE_LOCALIZATION_METHODS | Yes | `guan_xie_2025_review` | `guan_xie_2025_review` | ✅ PASS |
| alternative_methods_list | ALTERNATIVE_LOCALIZATION_METHODS | Yes | `guan_xie_2025_review` | `guan_xie_2025_review` | ✅ PASS |
| wu_chua_arctan | FRACTIONAL_CHUA_ARCTAN_WU2023 | Yes | `wu_2023_fractional_chua_arctan` | `wu_2023_fractional_chua_arctan` | ✅ PASS |

## Registered References

### `leonov_kuznetsov_hidden_definition` - Leonov-Kuznetsov hidden attractor definition
- **Authors**: Leonov, Gennady A. and Kuznetsov, Nikolay V.
- **Year**: 2013
- **Title**: *Hidden Attractors in Dynamical Systems: From Hidden Oscillations in Hilbert-Kolmogorov, Aizerman, and Kalman Problems to Hidden Chaotic Attractor in Chua Circuits*
- **Journal/Venue**: International Journal of Bifurcation and Chaos
- **DOI**: `10.1142/S0218127413300024`
- **URL**: https://doi.org/10.1142/S0218127413300024
- **Topics**: hidden_attractor_definition, self_excited_vs_hidden
- **Verification Status**: `verified`

### `kuznetsov_2017_chua_df` - Kuznetsov Describing Function for Chua Circuit
- **Authors**: Kuznetsov, Nikolay V. and Kuznetsova, Olga A. and Leonov, Gennady A. and Mokaev, Tatyana N. and Stankevich, Nikolay V.
- **Year**: 2017
- **Title**: *Hidden attractors localization in Chua circuit via the describing function method*
- **Journal/Venue**: IFAC-PapersOnLine
- **DOI**: `10.1016/j.ifacol.2017.08.470`
- **URL**: https://doi.org/10.1016/j.ifacol.2017.08.470
- **Topics**: describing_function_chua, hidden_attractor_localization
- **Verification Status**: `verified`

### `diethelm_ford_freed_abm_caputo` - Diethelm-Ford-Freed ABM Predictor-Corrector Method
- **Authors**: Diethelm, Kai and Ford, Neville J. and Freed, Alan D.
- **Year**: 2002
- **Title**: *A Predictor-Corrector Approach for the Numerical Solution of Fractional Differential Equations*
- **Journal/Venue**: Nonlinear Dynamics
- **DOI**: `10.1023/A:1016592217141`
- **URL**: https://doi.org/10.1023/A:1016592217141
- **Topics**: caputo_abm_integration, fractional_differential_equations
- **Verification Status**: `external_reference_pending`

### `danca_2017_fractional_hidden` - Danca Hidden Chaotic Attractors in Fractional-Order Systems
- **Authors**: Danca, Marius-F.
- **Year**: 2017
- **Title**: *Hidden Chaotic Attractors in Fractional-Order Systems*
- **Journal/Venue**: Nonlinear Dynamics
- **DOI**: `10.1007/s11071-017-3472-7`
- **URL**: https://doi.org/10.1007/s11071-017-3472-7
- **Topics**: fractional_hidden_attractors, fractional_stability, fractional_chua
- **Verification Status**: `verified`

### `matignon_fractional_stability` - Matignon Fractional Stability Results
- **Authors**: Matignon, Denis
- **Year**: 1996
- **Title**: *Stability Results for Fractional Differential Equations with Applications to Control Processing*
- **Journal/Venue**: Computational Engineering in Systems Applications
- **DOI**: `None`
- **URL**: https://www.researchgate.net/publication/229007412_Stability_results_for_fractional_differential_equations_with_applications_to_signal_processing
- **Topics**: fractional_matignon_stability, fractional_stability
- **Verification Status**: `external_reference_pending`

### `machado_2015_fractional_describing_functions` - Machado Fractional Describing Functions
- **Authors**: Machado, J. Tenreiro
- **Year**: 2015
- **Title**: *Fractional order describing functions*
- **Journal/Venue**: Signal Processing
- **DOI**: `10.1016/j.sigpro.2014.05.012`
- **URL**: https://doi.org/10.1016/j.sigpro.2014.05.012
- **Topics**: machado_fdf, describing_functions
- **Verification Status**: `external_reference_pending`

### `guan_xie_2025_review` - Guan-Xie Review on Hidden Attractor Localization
- **Authors**: Guan, Xinqi and Xie, Yong
- **Year**: 2025
- **Title**: *A Review on Methods for Localization of Hidden Attractors*
- **Journal/Venue**: Nonlinear Dynamics
- **DOI**: `10.1007/s11071-025-11327-5`
- **URL**: https://doi.org/10.1007/s11071-025-11327-5
- **Topics**: alternative_localization_methods, hidden_attractors_review
- **Verification Status**: `verified`

### `wu_2023_fractional_chua_arctan` - Wu et al. Fractional Chua with Arctan Nonlinearity
- **Authors**: Wu, Xianming and Fu, Longxiang and He, Shaobo and Yao, Zhao and Wang, Huihai and Han, Jiayu
- **Year**: 2023
- **Title**: *Hidden attractors in a new fractional-order Chua system with arctan nonlinearity and its DSP implementation*
- **Journal/Venue**: Results in Physics
- **DOI**: `10.1016/j.rinp.2023.106866`
- **URL**: https://doi.org/10.1016/j.rinp.2023.106866
- **Topics**: wu_fractional_chua_arctan, fractional_chua
- **Verification Status**: `verified`
