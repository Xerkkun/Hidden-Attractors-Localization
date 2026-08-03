# Tempered BDF Convolution Quadrature

## Scope and status

`tempered_convolution_quadrature` is an **experimental sampled operator** for
left tempered Riemann--Liouville (RL) and exponentially conjugated Caputo
derivatives with $0<q\leq1$. It supports BDF1 and BDF2, scalar or
componentwise $q_i$ and $\lambda_i$, and direct Python, direct Numba, or
offline FFT evaluation.

It is not an FDE time-stepper. It does not solve an implicit CQ equation,
apply a short-memory approximation, prove a convergence theorem, or establish
chaos, attraction, stability, or hiddenness from a trajectory.

## Continuous operators fixed by HAFO

Let $\tau=t-a$, $\lambda\geq0$, and

$$
I_{a+}^{\beta,\lambda}x(t)=
\frac{1}{\Gamma(\beta)}\int_a^t
(t-s)^{\beta-1}e^{-\lambda(t-s)}x(s)\,ds.
$$

HAFO uses the unnormalised exponential-conjugation convention

$$
D_{a+}^{q,\lambda}x(t)=
e^{-\lambda\tau}D_{a+}^{q}
\left[e^{\lambda(\cdot-a)}x\right](t)
$$

for tempered RL, and

$$
{}^CD_{a+}^{q,\lambda}x(t)=
e^{-\lambda\tau}{}^CD_{a+}^{q}
\left[e^{\lambda(\cdot-a)}x\right](t)
$$

for conjugated tempered Caputo. For $0<q\leq1$, their relation is

$$
{}^CD_{a+}^{q,\lambda}x(t)=
D_{a+}^{q,\lambda}
\left[x(t)-e^{-\lambda(t-a)}x(a)\right].
$$

The anchor is therefore $e^{-\lambda(t-a)}x(a)$, not the physical constant
$x(a)$. A physical constant is generally **not** annihilated when
$\lambda>0$; the exponential anchor is. For $m-1<q<m$, the general formula
requires the complete tempered initial jet
$c_j=(D+\lambda)^j x(a+)$, $j=0,\ldots,m-1$. The current API deliberately
stops at $q\leq1$, where one point value suffices.

No term $-\lambda^q x$ is subtracted. A normalised tempered generator with
that correction is a different operator and must receive a separate contract.

## BDF1/BDF2 algebra

The BDF generating polynomials are

$$
\delta_1(\zeta)=1-\zeta,
\qquad
\delta_2(\zeta)=\frac32-2\zeta+\frac12\zeta^2.
$$

Ordinary Lubich weights satisfy

$$
\delta_p(\zeta)^q=\sum_{k\geq0}\omega_k^{(q,p)}\zeta^k.
$$

Exponential conjugation gives the tempered generating function

$$
\delta_p(e^{-\lambda h}\zeta)^q
=\sum_{k\geq0}
e^{-\lambda kh}\omega_k^{(q,p)}\zeta^k.
$$

Thus the RL value at $t_n=a+nh$ is

$$
\mathcal D_{h,p}^{q,\lambda}x_n=
h^{-q}\sum_{k=0}^{n}
\omega_k^{(q,p)}e^{-\lambda kh}x_{n-k}.
$$

For conjugated Caputo, HAFO applies the exact discrete initial correction

$$
{}^C\mathcal D_{h,p}^{q,\lambda}x_n=
h^{-q}\left[
\sum_{k=0}^{n}\omega_k^{(q,p)}e^{-\lambda kh}x_{n-k}
-x_0e^{-\lambda nh}\sum_{k=0}^{n}\omega_k^{(q,p)}
\right].
$$

This formula is evaluated without constructing
$e^{+\lambda(t-a)}x(t)$. Only non-growing damping factors are formed, so a
large $\lambda kh$ may safely underflow an already unrepresentable history
term to zero instead of overflowing a transformed state.

### Starting convention

BDF2 uses the direct terminal-truncated convolution from its first sample.
HAFO does not silently insert a BDF1 start and does not yet implement
high-order starting corrections. The Caputo anchor above is part of the
operator definition; it must not be confused with numerical corrections for
nonsmooth initial data. A BDF2 calculation can lose second-order convergence
when the conjugated solution has the usual initial singularity.

## Discretisations that remain distinct

| Formula or name | Meaning | HAFO state |
|---|---|---|
| $\delta_p(e^{-\lambda h}\zeta)^q$ | exact discrete exponential conjugation used here | `experimental` |
| $[\delta_p(\zeta)/h+\lambda]^q$ | direct CQ of the shifted Laplace symbol | `tempered_symbol_shift_cq`, `planned` |
| shifted CQ-$\theta$ | evaluation at $t_{n-\theta}$, not tempering | not implemented |
| $D^{q,\lambda}x-\lambda^q x$ | normalised tempered generator | not implemented; separate definition required |
| recurrent fast multistep history | exact local window plus real recurrent compression from Guo et al. | `tempered_fast_multistep_history`, `experimental` |

The first two symbols are consistent approximations to the same continuous
shift under suitable assumptions, but their finite weights, error constants,
startup behaviour, and discrete anchors differ. They are methods, not
interchangeable computational backends. The recurrent route likewise remains
separate: it supports real-axis FBDF1 and second-order GNGF2, not fractional
BDF2, and calibrates compression error on the complete finite nonlocal weight
grid. See [Fast Recurrent Tempered Multistep History](tempered_fast_multistep_history.md).

## Public API

```python
import numpy as np
from hidden_attractors.fractional import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    tempered_convolution_quadrature,
)

times = np.linspace(0.0, 1.0, 513)
q = np.array([0.58, 0.84])
lam = np.array([0.35, 1.10])
samples = np.column_stack((
    np.exp(-lam[0] * times) * (0.8 + 1.2 * times**3),
    np.exp(-lam[1] * times) * (-0.3 + 0.7 * times**4),
))

result = tempered_convolution_quadrature(
    samples,
    q,
    tempering=lam,
    bdf_order=2,
    definition="tempered_caputo",
    times=times,
    lower_terminal=0.0,
    initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
    backend="fft",
)

print(result.values.shape)
print(result.tempering_convention)
print(result.normalization_correction)
```

For raw tempered RL, use
`definition="tempered_riemann_liouville"` together with
`TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION`. That token acknowledges an
operator evaluation; it does not turn a classical point value into RL initial
data.

The result retains both `base_weights` and the actually applied damped
`weights`, plus the uniform grid, component orders and tempering parameters,
BDF formula, backend, complexity, starting convention, references, and
overflow/underflow metadata.

## Backends and complexity

| Backend | Work | Extra working storage | Intended use |
|---|---:|---:|---|
| `python` | $O(dN^2)$ | $O(dN)$ | transparent reference and small fixtures |
| `numba` | $O(dN^2)$ | $O(dN)$ | direct compiled reference for small/medium histories |
| `fft` | $O(dN\log N)$ | $O(dN)$ | complete trajectory already available in memory |

The FFT path zero-pads to a linear convolution. It is not online,
streaming, oblivious, SOE, or $O(1)$-memory history compression. NumPy's FFT
already enters an optimised native implementation, so adding a second FFT in C
would duplicate the algorithm without improving asymptotic complexity.

The separately validated recurrent fast-history method is now available as
`tempered_fast_multistep_history`. It uses an exact local contribution and an
explicit finite-grid compression tolerance, but it is not an FFT backend of
this BDF1/BDF2 operator. Julia does not provide a known algorithmic advantage
for an inner per-step convolution and would add cross-language cost per call;
it remains useful as an external batch comparator if an equivalent published
solver is identified.

## Reductions and overlap checks

- $\lambda=0$ delegates to `lubich_convolution_quadrature` and preserves its
  arithmetic result exactly.
- BDF1 tempered RL agrees with
  `tempered_grunwald_letnikov_derivative` up to floating-point evaluation of
  the same weights.
- $q=1$ reduces to the terminal-truncated BDF approximation of
  $(D+\lambda)x$, not to $Dx$.
- `integrate_tempered_caputo_abm` is a distinct FDE solver. It is neither
  called by this operator nor used as its exact oracle.

## Reproducible nonlinear manufactured example

The example constructs

$$
x_i(t)=e^{-\lambda_i\tau}
\left(x_{0,i}+a_i\tau^{\beta_i}\right)
$$

and a nonlinear right-hand side
$F_i(t,x_i)=x_i^2+g_i(t)$ whose analytic left side is

$$
e^{-\lambda_i\tau}a_i
\frac{\Gamma(\beta_i+1)}
{\Gamma(\beta_i+1-q_i)}
\tau^{\beta_i-q_i}.
$$

Run it with:

```powershell
python examples/tempered_convolution_quadrature.py
```

With 512 intervals and FFT-BDF2, the retained run reported endpoint absolute
errors $4.22\times10^{-6}$ and $5.53\times10^{-6}$. These are finite-grid
manufactured residuals, not a theorem or validation of a general nonlinear
FDE solver.

## Verification surface

The local tests cover:

- exact software reduction at $\lambda=0$;
- direct conjugation identity for BDF1/BDF2 and all three backends;
- componentwise $(q_i,\lambda_i)$;
- manufactured refinement;
- $q=1$, terminal startup, and strict parameter validation;
- safe damping underflow and explicit overflow errors;
- the exponential Caputo anchor and the nonzero physical-constant case;
- BDF1 overlap with the existing tempered GL operator.

The Wolfram case is independent: it constructs BDF coefficients from a
recurrence and a separate high-precision factor expansion, evaluates direct
damped weights and explicit conjugation, and checks scalar/vector fixtures,
$\lambda=0$, $q=1$, and manufactured endpoint convergence without reading
HAFO source or report data. Passing remains finite algebraic/numerical evidence,
not a stability or convergence theorem.

The retained Wolfram run passed 18/18 assertions. The independent Python
reconstruction differed by at most $1.3085\times10^{-14}$; the required
public-core comparison differed by at most $4.441\times10^{-15}$. The
portable files are under
`validation/outputs/wolfram/tempered_convolution_quadrature_verified/`.

## Measured backend decision

`benchmarks/bench_tempered_convolution_quadrature.py` measures the complete
public call after an excluded Numba warm-up. Its retained Windows run used
three repetitions for each of 12 combinations: three workloads
$(64,2)$, $(256,3)$, and $(768,4)$, both definitions, and BDF1/BDF2.
Numba matched Python exactly; the largest FFT--Python difference was
$1.95\times10^{-13}$.

FFT had the lowest median in all 12 finite cases. Relative to warmed Numba,
its median advantage was modest: approximately $1.09$--$1.11$ for the small
cases, $1.01$--$1.10$ for the medium cases, and $1.23$--$1.30$ for the largest
cases. Python was $2.68$--$2.93$ times slower at $N=64$,
$45.4$--$47.8$ times slower at $N=256$, and $312$--$359$ times slower than
Numba at $N=768$.

This evidence supports retaining Numba direct and FFT batch; it does **not**
measure a C or Julia candidate and therefore cannot justify either. A native C
route is admitted only after profiling exposes a residual production
bottleneck and an independently verified implementation beats inter-run noise
end to end. The retained JSON is
`validation/outputs/benchmarks/tempered_convolution_quadrature_backends_20260803.json`.

## Primary sources and SciSpace evidence

SciSpace was queried with full questions about tempered RL/Caputo CQ,
exponential conjugation, BDF1/BDF2, starting corrections, stability, and recent
improvements. The decisive primary sources were then checked through their DOI
records:

- Lubich, foundational fractional multistep CQ,
  [DOI 10.1137/0517050](https://doi.org/10.1137/0517050).
- Chen and Deng, explicit substantial/tempered weights
  $e^{-k\lambda h}\omega_k$,
  [DOI 10.1051/m2an/2014037](https://doi.org/10.1051/m2an/2014037).
- Sabzikar, Meerschaert, and Chen, tempered operator definitions,
  [DOI 10.1016/j.jcp.2014.04.024](https://doi.org/10.1016/j.jcp.2014.04.024).
- Jin, Li, and Zhou, high-order BDF starting corrections,
  [DOI 10.1137/17M1118816](https://doi.org/10.1137/17M1118816).
- Guo, Zeng, Turner, Burrage, and Karniadakis, tempered multistep weights and
  fast algorithms,
  [DOI 10.1137/18M1230153](https://doi.org/10.1137/18M1230153).
- Li, Deng, and Zhao, tempered IVP formulation and predictor--corrector,
  [DOI 10.3934/dcdsb.2019026](https://doi.org/10.3934/dcdsb.2019026).

Recent 2025--2026 results located by SciSpace chiefly extend corrected L1,
nonuniform meshes, or fast evaluation. They strengthen the backlog but do not
replace the verified BDF conjugation weights. An older method is retained when
its algebra and convergence framework remain the appropriate foundation.
