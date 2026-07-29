# Public configuration surface

Runnable research-case templates are intentionally not stored here. Concrete
case contracts belong to validation records only.

PyPI distributes one non-runnable structural example at
`hidden_attractors/configs/examples/workflow_contract.yaml`. It contains no
system selection, candidate, campaign parameter, or scientific claim. Copy it
with `hidden-attractors init --example workflow_contract`, then provide and
archive every numerical choice explicitly.

`validation_contract.json` defines the recorded evidence order:
`numerical_contract`, `algebraic_validation`, `seed_generation`,
`soft_precheck`, `continuation`, `post_continuation_filter`,
`dynamic_reference`, `robustness`, `hiddenness_tests`, `diagnostics`.
