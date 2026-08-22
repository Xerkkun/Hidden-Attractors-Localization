# Release Notes

## 1.2.0

`hidden-attractors-fo` 1.2.0 is the release-candidate identity for the current
validated source. It retains the stable 1.x API contract while promoting only
execution paths that already existed and now have complete public contracts,
tests, and finite-evidence boundaries. It also hardens native input handling,
same-named submodule imports, relocated validation runs, and the exact
tag-to-commit publication guard.

No `v1.2.0` tag or public artifact is implied by these local metadata records.
Publication remains a separate, protected workflow action.

The unified CLI now includes `hidden-attractors update`. It checks stable PyPI
releases by default, never treats a lower public version as a reason to
downgrade a newer local checkout, and invokes pip only after interactive
confirmation or `--yes`.

On Windows, `--yes` does not invoke pip from an active installed
`hidden-attractors.exe`; the command exits with an explanation and the exact
version-pinned `sys.executable -m pip ...` command to run from a new prompt.
This avoids claiming that a running launcher can replace itself.

Release ordering: publish and independently verify `hidden-attractors-fo`
1.2.0 on PyPI before publishing a Toolbox Chaos release that advertises or
requires the 1.2.0 HAFO surface.

## 1.1.0

`hidden-attractors-fo` 1.1.0 is a public-library and reproducibility release.
It fully integrates scalar time-series Lyapunov diagnostics, hardens runtime
paths for installed wheels, and narrows the source distribution to supported
user-facing material.

The distributed comprehensive example is the integer-order Chua Lur'e
reference/control. It exercises seed construction, continuation, integration,
sampled equilibrium-neighborhood controls, and structured outputs. All
decisions remain finite numerical evidence rather than global mathematical
proofs.

The wheel and sdist intentionally omit internal investigations, project plans,
ordinary run products, and the large validation data tree. Exact validation
records are preserved in the matching release tag and archived snapshot.

License: MIT.

Archived DOI: `10.17605/OSF.IO/ZGK74`.
