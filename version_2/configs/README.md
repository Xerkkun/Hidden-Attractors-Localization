# Configurations

`unified_caputo_protocol.json` is the only configuration contract for new
hidden-attractor candidate studies. Historical YAML files remain temporarily
only when an existing adapter or recorded artifact requires them.

When adding a new config, include enough context to identify:

- model and parameter set;
- fractional order `q`;
- step size `h`;
- memory length `Lm`;
- output directory policy;
- whether the run is exploratory, verification, or reporting.

`validation_contract.json` defines the promoted evidence order:
`numerical_contract`, `algebraic_validation`, `seed_generation`,
`soft_precheck`, `continuation`, `post_continuation_filter`,
`dynamic_reference`, `robustness`, `hiddenness_tests`, `diagnostics`.
All stage summaries must use the official JSON envelope.
