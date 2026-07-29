# Versioning Policy

`hidden-attractors-fo` follows semantic versioning for the installed Python
package and unified `hidden-attractors` command.

- Patch releases preserve documented behavior and correct defects.
- Minor releases may add documented APIs while preserving the stable tier.
- Major releases may change stable interfaces and must identify those changes
  in the release notes.

The [API Stability Tiers](api_stability.md) page is the authoritative boundary
between stable and experimental symbols. Recorded validation artifacts keep
their original numerical contract and provenance; a software version change
does not relabel earlier evidence or turn a diagnostic into a claim of chaos
or hiddenness.
