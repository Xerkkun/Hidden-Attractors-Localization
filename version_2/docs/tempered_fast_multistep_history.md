# Fast Recurrent Tempered Multistep History

## Scope and status

`tempered_fast_multistep_history` is an **implemented sampled operator** with an
**experimental public API** for left tempered Riemann--Liouville (RL) and exponentially conjugated Caputo
derivatives with $0<q\leq1$. It evaluates the recent history with exact
multistep coefficients and compresses only older samples into stable real
recurrent states. Orders $q_i$ and tempering parameters $\sigma_i$ may be
scalar or componentwise.

It is not an FDE solver, an implicit CQ time-stepper, a short-memory
truncation, or an FFT backend. Applying it to an existing trajectory does not
make that trajectory a solution of a fractional equation and does not prove
chaos, attraction, stability, or hiddenness.

## Continuous convention

Let $t_n=a+nh$, $\sigma\geq0$, and $0<q\leq1$. HAFO uses exponential
conjugation,

$$
D_{a+}^{q,\sigma}u(t)=e^{-\sigma(t-a)}D_{a+}^{q}
\left[e^{\sigma(\cdot-a)}u\right](t),
$$

and its Caputo counterpart

$$
{}^CD_{a+}^{q,\sigma}u(t)=e^{-\sigma(t-a)}{}^CD_{a+}^{q}
\left[e^{\sigma(\cdot-a)}u\right](t).
$$

Consequently,

$$
{}^CD_{a+}^{q,\sigma}u(t)=D_{a+}^{q,\sigma}
\left[u(t)-e^{-\sigma(t-a)}u(a)\right].
$$

No term $-\sigma^q u$ is inserted. The shifted-symbol formula
$[\delta(\zeta)/h+\sigma]^q$ and a normalised tempered generator are distinct
contracts.

## Supported multistep generators

The untempered coefficients satisfy

$$
\Omega_q(\zeta)=\sum_{\ell\geq0}\omega_\ell^{(q)}\zeta^\ell.
$$

HAFO exposes two real Fast Method II generators:

$$
\begin{aligned}
\text{FBDF1:}\quad
\Omega_q(\zeta)&=(1-\zeta)^q,\\
\text{GNGF2:}\quad
\Omega_q(\zeta)&=(1-\zeta)^q
\left(1+\frac q2(1-\zeta)\right).
\end{aligned}
$$

GNGF2 is the second-order generalized Newton--Gregory formula used in the
primary paper. It is **not** silently called fractional BDF2. For the BDF2
generator, the real-axis factor $F_\omega(-\lambda)$ crosses a fractional
power of a negative number, so a real-only implementation needs a separate
branch analysis. At $q=1$, GNGF2 does reduce exactly to

$$
(1-\zeta)\left(1+\frac12(1-\zeta)\right)
=\frac32-2\zeta+\frac12\zeta^2,
$$

the ordinary BDF2 polynomial. Direct and FFT fractional BDF2 remain available
through `tempered_convolution_quadrature`; they are a different numerical
route, not a backend of this recurrence.

The damped sampled RL operator is

$$
\mathcal D_{h}^{q,\sigma}u_n=
h^{-q}\sum_{\ell=0}^{n}
\omega_\ell^{(q)}e^{-\sigma h\ell}u_{n-\ell}.
$$

For conjugated Caputo, HAFO subtracts the exact discrete anchor,

$$
{}^C\mathcal D_{h}^{q,\sigma}u_n=h^{-q}\left[
\sum_{\ell=0}^{n}\omega_\ell^{(q)}e^{-\sigma h\ell}u_{n-\ell}
-u_0e^{-\sigma hn}\sum_{\ell=0}^{n}\omega_\ell^{(q)}
\right].
$$

The partial sum in the anchor is evaluated by the exact coefficient
recurrence; it is not compressed.

## Real-axis representation and its sign

For $0<q<1$ and every compressed lag, Fast Method II represents an untempered
coefficient by a real integral and applies a trapezoidal rule after the change
$r=e^x$. In dimensionless form HAFO uses

$$
\omega_\ell^{(q)}\approx
\widehat\omega_\ell^{(q)}=
\sum_{j=0}^{Q-1}a_j(1+r_j)^{-\ell-1},
\qquad r_j=h\lambda_j>0.
$$

For FBDF1 the quadrature density contains

$$
-\frac{\sin(\pi q)}{\pi}r^{1+q},
$$

and for GNGF2 it additionally contains the signed real factor
$1-qr/2$. The minus sign is required for derivative weights. Indeed, for
$\ell\geq1$,

$$
\omega_\ell^{(q)}=
\frac{\Gamma(\ell-q)}{\Gamma(-q)\Gamma(\ell+1)}
=-\frac{\sin(\pi q)}{\pi}
\int_0^\infty r^q(1+r)^{-\ell-1}\,dr.
$$

The equality follows from the beta integral and
$\Gamma(q+1)\Gamma(-q)=-\pi/\sin(\pi q)$. It also makes the expected
$\omega_\ell^{(q)}<0$ explicit. An intermediate displayed integral in the
published source carries the opposite sign, while its subsequent definition
of the real integrand carries the negative sign. HAFO fixes the ambiguity by
the coefficient identity above and verifies it independently.

## Exact local window and stable recurrence

Choose an exact local lag $n_0$ (the source uses $n_0=50$ in its GNGF2
examples). HAFO splits

$$
\mathcal D_h^{q,\sigma}u_n=L_{n,n_0}+H_{n,n_0},
$$

where lags $0,\ldots,n_0$ are summed with exact coefficients and only lags
$n_0+1,\ldots,n$ use the quadrature approximation.

After scaling the paper's state by $h$, the dimensionless recurrent state is

$$
y_m^{(j)}=
\frac{e^{-\sigma h}}{1+r_j}
\left(y_{m-1}^{(j)}+u_{m-1}\right),
\qquad y_0^{(j)}=0.
$$

The multiplier lies in $(0,1]$ and HAFO never constructs
$e^{+\sigma(t-a)}u(t)$. For $n>n_0$, the old-history contribution is

$$
H_{n,n_0}\approx h^{-q}e^{-n_0\sigma h}
\sum_{j=0}^{Q-1}a_j(1+r_j)^{-n_0-1}
y_{n-n_0}^{(j)}.
$$

This gives $O(d(Q+n_0)N)$ recurrent batch work and
$O(d(Q+n_0))$ active history storage, excluding the supplied samples and the
returned $O(dN)$ output. For $q=1$, the finite FBDF1 or BDF2-limit stencil is
evaluated exactly and no quadrature states are needed.

## Finite-grid tolerance contract

The source's infinite trapezoidal-rule theorem depends on analyticity-strip
constants that are not generally available from an arbitrary sampled signal.
HAFO therefore makes a narrower, executable promise. For each component it
checks **every** compressed coefficient on the requested finite grid:

$$
\varepsilon_{L^1,i}=
\frac{\sum_{\ell=n_0+1}^{N-1}
|\widehat\omega_{\ell,i}-\omega_{\ell,i}|}
{\sum_{\ell=n_0+1}^{N-1}|\omega_{\ell,i}|}.
$$

With `quadrature_points=None`, nested real-axis grids
$Q=65,129,257,\ldots$ are refined until
`epsilon_L1 <= relative_tolerance` for every component or
`max_quadrature_points` is reached; the cap must therefore be at least 65 when
automatic compression is required. An explicit `quadrature_points` value is
accepted only if it already meets the same check. `tail_cutoff` selects the
finite real interval from relative level sets of the first and last
compressed lags.

The result also reports

$$
E_{\mathrm{op},i}=h^{-q_i}\|u_i\|_\infty
\sum_{\ell=n_0+1}^{N-1}
|\widehat\omega_{\ell,i}-\omega_{\ell,i}|.
$$

This is an a posteriori compression bound for the finite sampled operator, up
to floating-point accumulation. It is not a separate certified tail bound,
not the analytic infinite-trapezoid estimate, not the multistep
discretisation error, and not an FDE solution-error bound. No high-order
starting corrections are implemented.

## Public API

```python
import numpy as np
from hidden_attractors.fractional import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    tempered_fast_multistep_history,
)

times = np.linspace(0.0, 4.0, 801)
samples = np.column_stack((x, y, z))

result = tempered_fast_multistep_history(
    samples,
    orders=[0.48, 0.67, 0.83],
    tempering=[0.10, 0.35, 0.70],
    multistep_method="gngf2",
    definition="tempered_caputo",
    times=times,
    lower_terminal=0.0,
    initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
    local_history_steps=50,
    quadrature_points=None,
    relative_tolerance=1.0e-9,
    tail_cutoff=1.0e-20,
    max_quadrature_points=2049,
    backend="numba",
)

print(result.values.shape)
print(result.quadrature_points)
print(result.l1_relative_weight_error)
print(result.operator_absolute_error_bound)
```

Use `TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION` with
`definition="tempered_riemann_liouville"`. The explicit token acknowledges
that raw RL evaluation does not infer fractional IVP data from a classical
point value.

`TemperedFastHistoryResult` retains the exact local weights, real quadrature
nodes and signed weights, final recurrent state, componentwise calibration
errors, requested tolerance, complexity, startup and anchor conventions,
references, and evidence scope. `backend` accepts only `"python"` and
`"numba"`; FFT cannot preserve this online recurrence contract.

## Chua sampled-history example

Run:

```powershell
python examples/tempered_fast_history_chua.py
```

The script obtains 801 uniformly sampled states from the integer-order
non-smooth Chua system using DOP853, applies componentwise tempered-Caputo
GNGF2 as post-processing, and compares the result with an independent direct
$O(dN^2)$ convolution. Its test requires the calibrated relative coefficient
error to remain below $10^{-9}$ and the fast/direct maximum difference below
$5\times10^{-10}$.

The source trajectory remains an integer $q=1$ trajectory. This example is
operator-parity evidence only; it is not a fractional Chua solve and carries
no chaos, attraction, basin, or hiddenness conclusion.

## Independent Wolfram verification

`validation/wolfram/cases/tempered_fast_multistep_history.wl` reconstructs
the FBDF1/GNGF2 coefficients, beta/reflection sign, real recurrent history,
tempered Caputo anchor, and $q=1$ GNGF2 reduction at 80-digit precision. It
does not import the HAFO implementation. The retained run passed 13/13
assertions; its largest internal fast/direct residual was
$1.72700092683619\times10^{-26}$ and the largest anchor residual was
$1.90289948718204\times10^{-26}$.

The independent Python reconstruction differed from that Wolfram artifact by
at most $8.881784197001252\times10^{-15}$. A separately required call through
the public HAFO core differed by at most
$1.2434497875801753\times10^{-14}$, using 129 nodes in all four retained
FBDF1/GNGF2 by RL/Caputo cases. The portable evidence is under
`validation/reference_cases/tempered_fast_multistep_history/`.

These results validate finite algebra and cross-implementation consistency.
They are not a general convergence, stability, long-time dynamics, or
hidden-attractor theorem.

## Local verification surface

The focused suite covers direct FBDF1/GNGF2 parity, Python/Numba equality,
componentwise orders and tempering, the exact Caputo anchor, and the integer
FBDF1/BDF2 limits. Fifteen additional edge tests exercise `N=n0+1` and
`N=n0+2`, impulses at the compression boundary, isolated $h^{-q}$ scaling,
orders near zero and one, strong-tempering underflow, a nonzero lower terminal,
mixed GNGF2 components, insufficient explicit `Q`, exhaustion of the automatic
cap, the Caputo compression bound, and a finite manufactured refinement. The
manufactured test separates compression from discretisation and records only
that error decreases on its fixture; it does not assert a universal order.

The Wolfram comparator labels its independent float64 route as a direct
convolution, reconstructs the anchor from the retained samples, and keeps
cross-implementation differences separate from reported L1 calibration
metrics.

## Measured backend decision

`benchmarks/bench_tempered_fast_history.py` parity-gates the complete public
Python/Numba calls against direct and offline FFT baselines before recording
timings. Automatic `Q` selection and JIT warm-up are measured separately; each
repeated fast timing still includes validation, exact local weights,
finite-grid calibration at the fixed validated `Q`, recurrence construction,
allocation, and returned values.

The retained historical Windows run used three repetitions for RL/Caputo crossed with
FBDF1/GNGF2 and three workloads. It selected `Q=65` for `(N,d,n0)=(128,2,16)`
and `(512,3,32)`, and `Q=129` for `(2048,4,50)`. Python and Numba fast results
were bitwise equal. The largest direct/FFT difference was
$5.0004\times10^{-12}$; the largest fast/direct difference was
$4.4384\times10^{-8}$ and remained below the separately recorded compression
bound plus floating-point accumulation margin.

The evaluator's analytical active-history storage was 3,392, 5,472, and
14,016 bytes, versus 4,096, 24,576, and 131,072 bytes for complete base plus
tempered weight arrays. Thus the measured storage advantage increased from
about $1.21\times$ to $9.35\times$ over these workloads. It excludes the common
input and returned output and is not a resident-memory measurement.

Fast Numba did not have the lowest median in this retained finite batch run:
the offline FFT baselines won all twelve cases. That result does
not invalidate the recurrent memory contract, because FFT stores a complete
weight history and is not streaming. No native-C or Julia implementation with
the identical local window, tolerance calibration, anchor, and metadata was
implemented or timed. The evidence-based decision is therefore
`insufficient_evidence_to_add_native_c_or_julia`, not a claim that either
language can never help. C is reconsidered only if representative
HAFO/Toolbox profiling identifies the recurrence as an end-to-end bottleneck;
Julia remains a possible pinned whole-batch comparator, never a per-step call.

The historical portable artifact is
`validation/outputs/benchmarks/tempered_fast_history_backends_20260803_v2.json`
(SHA-256
`325D9C4BF2254C6FEA9C295080157345485B1A5426329A9DE0F7C3D949417D4F`).
Its embedded `script_sha256` identifies an earlier script revision and does not
match the current checkout. The recorded figures therefore describe that run;
they are not a current performance claim until the benchmark is repeated.

## Toolbox Chaos bridge

The sibling Toolbox Chaos engine exposes a lazy
`tempered_fast_multistep_history` bridge with the exact HAFO signature and
returns the typed HAFO result unchanged. `engine_capability(name)` retrieves
the original HAFO capability object rather than duplicating a status table.
The bridge keeps imports optional, reports absence explicitly, and rejects an
FFT request for this recurrent API. The four focused tests plus the complete
existing hidden-engine integration file passed 28/28 together. This is software-integration
evidence; it does not turn the operator into a solver or validate dynamics.

## Primary sources and SciSpace evidence

SciSpace was used to locate and compare fast tempered fractional multistep
methods, their real-axis recurrence, starting assumptions, and later fast
history alternatives. The implemented formulas were then checked against the
primary publication and its source:

- Guo, Zeng, Turner, Burrage, and Karniadakis, Fast Methods I/II and tempered
  FBDF/GNGF formulas,
  [DOI 10.1137/18M1230153](https://doi.org/10.1137/18M1230153).
- Lubich, fractional linear multistep calculus,
  [DOI 10.1137/0517050](https://doi.org/10.1137/0517050).
- Trefethen and Weideman, exponentially convergent trapezoidal rule,
  [DOI 10.1137/130932132](https://doi.org/10.1137/130932132).

The method remains useful even though its foundations are not recent: later
fast Caputo, SOE, corrected-L1, and nonuniform-mesh algorithms solve different
contracts and do not invalidate this verified recurrent multistep operator.
