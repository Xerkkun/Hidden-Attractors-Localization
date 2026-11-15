# Full-history L1 and stable PECE diagnostics

`l1_implicit` and `abm_stable` are independent, maintained diagnostic methods
for generic `rhs(t, x)` or `rhs(x)` systems with a finite state vector and
**0 < q < 1**. They preserve the complete Caputo history with lower limit zero.
They never stop on an equilibrium capture, truncate a window, or restart a
history. Existing `abm`, `efork3` and Fischer reference-rail protocols keep
their numerical meaning.

L1 discretizes the Caputo derivative using state increments with weights
`b[k]=(k+1)^(1-q)-k^(1-q)`. It solves each implicit nonlinear step with Newton.
An optional analytic Jacobian is checked for shape and finiteness before use;
otherwise a central finite difference with an explicit relative step is used.
Failure returns `newton_failure` and only the accepted steps. The normalized
residual is `||x-base-Gamma(2-q)*h^q*f||_inf/(1+||x||_inf)`.

Stable PECE discretizes the equivalent Volterra integral, with one predictor
and one corrector evaluation at each step. It uses the same analytical ABM
weights as the direct PECE scheme, generated with `log1p`/`expm1` and convergent
binomial remainders instead of adjacent large-power subtractions. Both methods
use compensated Kahan history sums, optionally compiled with Numba without
`fastmath`. States and stored weights use binary64. Metadata records the
platform's intermediate `longdouble` precision.

Both require O(N^2*d) work and O(N*d) storage. The comparison CLI checks an
explicit per-run history-term budget (default 50,000,000) before integration.
Raising that budget authorizes more work, without changing the method.
The reported elapsed time includes a cold Numba compilation when present;
it is not a controlled performance benchmark.

```python
import numpy as np
from hidden_attractors.integrations import integrate

diagnostics = {}
t, x, status = integrate(
    lambda t, x: -x, np.ones(2), q=0.5, h=0.01, t_final=1.0,
    integrator="l1_implicit", jacobian=lambda t, x: -np.eye(2),
    solver_options={"newton_tol": 2e-14, "newton_maxiter": 20,
                    "fd_relative_step": np.sqrt(np.finfo(float).eps),
                    "use_numba": True},
    diagnostics=diagnostics, divergence_norm=120.0,
)
```

The selector retains the `(t, x, status)` return contract. The optional mutable
`diagnostics` records method, order, step, requested/reached horizon, memory,
backend, work counts, RHS/Jacobian evaluations, Newton residuals, and failure
details with the uniform `caputo_diagnostic_v1` schema. `solver_options`,
`jacobian` and `diagnostics` are rejected for historical methods rather than
silently ignored. Unknown diagnostic options and enabled early-stop settings
are also rejected. Finite norm-threshold crossings remain distinct from a
nonfinite solution or callback/linear-solve failure.
The norm threshold is checked at t=0 before evaluating the initial RHS. An
initial state already outside the bound returns only that row with `diverged`,
zero RHS evaluations, and `full_horizon_reached=false`; equality is not a
crossing.

CLI for any registered continuous-time system:

```powershell
python -m hidden_attractors.cli.main integrate compare `
  --system chua-nonsmooth --x0 0.01 0.02 0.03 --q 0.9 `
  --h 0.02 0.01 0.005 --t-final 1.0 --output-dir outputs/diagnostico_nuevo
```

The installed entry point is `hidden-attractors integrate compare` with the
same arguments. `--parameters '{"alpha": 8.4562}'` overrides the registered
parameter mapping. `--finite-difference-jacobian` forces the declared central
difference when a registered analytic Jacobian is available. `--no-numba`
selects the pure Python reduction. `--methods l1_implicit` runs a single method.

Each run creates a contract before numerical work, complete NPZ trajectories,
`summary.json` with diagnostics and same-step method differences, SHA-256
hashes of trajectories and relevant source files, and an owned run marker.
An existing nonempty output directory is rejected before integration; failed
runs retain their partial evidence and require a new output directory.

Controls cover the half-order exact solution `erfcx(sqrt(t))`, manufactured
forcing and step refinement, exact constant-force PECE, the L1 limit as
`q -> 1-` against implicit Euler, analytic versus finite-difference Jacobians,
full-history direct PECE agreement, dimensional failures before assignment,
and identical compiled/Python compensated reductions. Exactly `q=1` remains
the responsibility of the existing integer solvers.

These comparisons quantify finite-horizon numerical sensitivity. They do not
automatically identify attractors, prove chaos, validate a robust basin, or
promote the diagnostic experiments to asymptotic hiddenness evidence. The
sealed P02/P07 campaigns and their source/metadata remain unchanged.
