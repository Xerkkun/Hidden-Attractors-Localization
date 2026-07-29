# F6 Integrated Chaos Validator

F6 combines F5 diagnostics, Lyapunov method metadata, available case-specific
finite-time local spectra, and optional F4 internal validation metadata.

The output is a conservative diagnostic integration layer. It does not certify
chaos or hiddenness. Missing F4 artifacts are reported as
`f4_internal_validation_unavailable_or_not_evaluated` without preventing report
generation.
