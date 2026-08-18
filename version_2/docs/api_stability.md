# API Stability Tiers

`hidden-attractors-fo` uses four formal stability tiers to communicate
how much users can rely on each part of the API.  Every sub-module starts
with a `Stability: <tier>` line in its docstring, and every public symbol
that has been explicitly annotated carries `__api_tier__` as an attribute.

---

## Top-Level Export Boundary

The top-level package intentionally exports both stable and experimental symbols for convenience, but it does not treat every exported name as equally stable. The machine-readable boundary is available as:

```python
import hidden_attractors as ha

ha.PUBLIC_API_STABLE
ha.PUBLIC_API_EXPERIMENTAL
ha.PUBLIC_API_TIERS
```

`PUBLIC_API_STABLE` contains model, system-registry, basin-label, and IO entry
points with stable guarantees. `PUBLIC_API_EXPERIMENTAL` contains independent
diagnostics, seed generation, workflow specs, and workflow runners that are
tested and documented but may evolve with changelog notes.

Compatibility aliases such as old piecewise Chua names are importable only to read historical runs. They are intentionally excluded from `__all__` and from the stable tier lists. Internal implementation modules such as `hidden_attractors.cli`, `native`, `parallel`, and `paths` are also not promoted by top-level export.

---
## The Four Tiers

### stable

> **Guarantee**: Signatures and return types will not change between minor
> versions.  If a breaking change is ever necessary it will be preceded by a
> deprecation cycle and a version bump.

| Module | Contents |
| --- | --- |
| `hidden_attractors.models` | `ChuaParameters`, vector field helpers, equilibria |
| `hidden_attractors.systems` | `ChaoticSystem`, `LureSystem`, registry API |
| `hidden_attractors.basins` | `CLASS_LABELS`, `class_label`, `is_target_class` |
| `hidden_attractors.io` | JSON/CSV read-write, `load_trajectory_csv` |

---

### experimental

> **Guarantee**: The API is useful and tested.  Function signatures may gain
> new *optional* keyword arguments.  Positional arguments and return types
> will not change without a changelog entry.  Renaming or removing a symbol
> will be announced at least one release in advance.

| Module | Contents |
| --- | --- |
| `hidden_attractors.analysis` | Common integer/fractional trajectory contracts, Bandt--Pompe permutation entropy, integer `q=1` SALI/GALI/LDI and CLV diagnostics, Lyapunov, scalar-time-series Lyapunov, Kaplan--Yorke dimension, correlation dimension, delay reconstruction, basic and advanced recurrence, basin uncertainty, spectral, bifurcation, boundedness, Poincare, 0-1, and trajectory metrics |
| `hidden_attractors.fractional` | Fractional definitions, problem contracts, sampled operators, ordinary/Hadamard/tempered BDF CQ, distributed-order solvers, and the semantic multi-term Caputo L1 facade |
| `hidden_attractors.seed_generation` | Harmonic-balance seeds - Chua-specific and generic Lur'e |
| `hidden_attractors.seed_generation.core` | Shared dataclasses, `validate_fractional_order` |
| `hidden_attractors.seed_generation.chua` | Implemented DF helpers and `find_harmonic_seed` |
| `hidden_attractors.seed_generation.lure` | Lur'e DF, `find_lure_harmonic_seed` |
| `hidden_attractors.solvers` | Fractional solver interfaces, EFORK wrapper |
| `hidden_attractors.plotting` | Phase-space and time-series plot helpers |
| `hidden_attractors.integrations` | Optional external-tool adapters (`nolds`, `antropy`), also re-exported through the top-level experimental tier |
| `hidden_attractors.workflows` | Workflow specs, contracts, integer Lur'e pipeline |

The module-qualified entries above are public experimental surfaces even when
their individual functions are not re-exported at package top level. In that
case `get_tier(function)` may return `None`; the module docstring and this table
are the current stability declaration. Promotion to the top-level symbol tier
requires a synchronized `PUBLIC_API_EXPERIMENTAL` and `__all__` change.

Delay reconstruction, basic/advanced recurrence, basin uncertainty, and the
complexity adapters have completed that synchronized top-level promotion. The
[API Reference](api_reference.md) is the complete public symbol index; each
diagnostic retains its finite-data scientific limits.

The top-level CLV surface comprises `CovariantQRHistoryResult`,
`CovariantLyapunovResult`, `CovariantAngleResult`,
`integer_covariant_vectors_from_qr_history`,
`integer_flow_covariant_lyapunov_vectors`,
`integer_map_covariant_lyapunov_vectors`,
`integer_system_covariant_lyapunov_vectors`, and
`covariant_lyapunov_angles`. Its experimental contract is restricted to
memoryless integer dynamics with `q=1`; fractional CLV is
`research_required`, not an experimental executable facade.

---

### internal

> **Guarantee**: None beyond "it works for the workflows that use it."
> Advanced users may depend on these modules, but changes will not be
> announced.  Pin to a specific commit if you rely on internals.

| Module | Contents |
| --- | --- |
| `hidden_attractors.native` | `FractionalChuaBackend`, `BasinBackend` (ctypes wrappers) |
| `hidden_attractors.parallel` | C compilation helpers, OpenMP flags, process-pool policy |
| `hidden_attractors.paths` | Repository path constants |
| `hidden_attractors.cli` | Console-script entry points (Internal implementation; the unified console command `hidden-attractors` is the official public command surface) |
| `hidden_attractors._stability` | Tier constants and `api_tier` decorator (this module) |

---

### legacy

> **Guarantee**: Compatibility annotation only.

The `LEGACY` constant remains available for tier introspection of historical
symbol aliases. It does not correspond to an importable
`hidden_attractors.legacy` module or to an installed legacy command surface.

---

## Introspecting a Symbol's Tier

```python
import hidden_attractors as ha

# Tier constants
print(ha.STABLE)        # 'stable'
print(ha.EXPERIMENTAL)  # 'experimental'
print(ha.INTERNAL)      # 'internal'
print(ha.LEGACY)        # 'legacy'

# Check a class
print(ha.ChuaParameters.__api_tier__)   # 'stable'

# get_tier on any object (returns None if not annotated)
print(ha.get_tier(ha.ChuaParameters))   # 'stable'

# assert_tier raises AssertionError if the tier is wrong (useful in tests)
ha.assert_tier(ha.ChuaParameters, ha.STABLE)   # passes silently
ha.assert_tier(ha.find_harmonic_seed, ha.STABLE)  # raises AssertionError
```

---

## Upgrade-Path Guidance

When an `experimental` symbol changes in a way that affects your code:

1. **Read the changelog** for the relevant release - experimental changes
   are always documented.
2. **Check the new signature** with `help()` or the API reference.
3. If you pin to a specific version in `pyproject.toml` or `requirements.txt`,
   update the pin after reviewing the changelog.

When an `internal` symbol you relied on changes:

1. Check whether an equivalent `stable` or `experimental` API now covers
   your use case.
2. If not, open an issue - it may warrant promotion to `experimental`.

---

## Annotating New Symbols

Package authors should use the `@api_tier` decorator when adding new public
symbols:

```python
from hidden_attractors._stability import api_tier, STABLE, EXPERIMENTAL

@api_tier(STABLE)
class MyNewModel:
    ...

@api_tier(EXPERIMENTAL)
def my_new_diagnostic(traj, *, threshold=0.01):
    ...
```

Module-level tier is declared in the docstring:

```python
"""My new module.

Stability: experimental
    Short explanation of what is and is not guaranteed.
"""
```
