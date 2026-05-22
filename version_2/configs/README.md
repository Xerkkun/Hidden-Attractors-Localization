# Configurations

Workflow configuration files live here. They are examples of the numerical
contracts used in the current project outputs.

When adding a new config, include enough context to identify:

- model and parameter set;
- fractional order `q`;
- step size `h`;
- memory length `Lm`;
- output directory policy;
- whether the run is exploratory, verification, or reporting.

`validation_contract.json` defines the promoted validation evidence layout. It
does not contain numerical results; it records the expected manifest, stage
summary, table, figure, and report names used under `validation/`.
