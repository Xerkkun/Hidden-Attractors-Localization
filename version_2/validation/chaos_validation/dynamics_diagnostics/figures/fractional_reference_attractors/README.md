# Fractional Reference Attractor Figures

These figures are generated from validation diagnostics, not from the Lure/EFORK experiment outputs.

## danca2017_chua_fractional_saturation_q09998:diagnostic_X0

- Source: Danca 2017
- q: 0.9998
- h: 0.01
- t_final: 500.0
- t_burn: 250.0
- integrator: ABM
- backend: native_full_history_abm
- memory_policy: full_history
- seed_scope: diagnostic_only_not_reported_by_article
- sampled rows plotted: 5000
- figures: danca2017_chua_fractional_saturation_q09998_diagnostic_X0_phase3d, danca2017_chua_fractional_saturation_q09998_diagnostic_X0_projections, danca2017_chua_fractional_saturation_q09998_diagnostic_X0_timeseries

Danca-specific note: the article does not disclose the exact hidden-attractor initial condition or DF seed parameters in the local reproduction metadata. This plotted trajectory uses the repository diagnostic seed and native full-history ABM contract.

## wu2023_chua_fractional_arctan_q099:x0_minus

- Source: Wu et al. 2023
- q: 0.99
- h: 0.01
- t_final: 100.0
- t_burn: 50.0
- integrator: ADM_WU2023
- backend: adm_local_reproduction
- memory_policy: none_local_adm
- seed_scope: paper_reported_initial_condition
- sampled rows plotted: 5000
- figures: wu2023_chua_fractional_arctan_q099_x0_minus_phase3d, wu2023_chua_fractional_arctan_q099_x0_minus_projections, wu2023_chua_fractional_arctan_q099_x0_minus_timeseries

## wu2023_chua_fractional_arctan_q099:x0_plus

- Source: Wu et al. 2023
- q: 0.99
- h: 0.01
- t_final: 100.0
- t_burn: 50.0
- integrator: ADM_WU2023
- backend: adm_local_reproduction
- memory_policy: none_local_adm
- seed_scope: paper_reported_initial_condition
- sampled rows plotted: 5000
- figures: wu2023_chua_fractional_arctan_q099_x0_plus_phase3d, wu2023_chua_fractional_arctan_q099_x0_plus_projections, wu2023_chua_fractional_arctan_q099_x0_plus_timeseries
