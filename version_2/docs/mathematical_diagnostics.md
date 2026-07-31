# Mathematical Diagnostics and Ready-to-Use Numerical Methods

This page documents calculations that can be used on a dynamical system,
trajectory, or scalar time series without running a hidden-attractor search.
Every returned classification is finite-time or finite-data evidence. None of
the functions below independently proves asymptotic boundedness, chaos,
stability, periodicity, or hiddenness.

For uniformly sampled data, write

$$
t_n=t_0+n\Delta t,\qquad
X_n=(x_{n,1},\ldots,x_{n,d})^\mathsf{T},
\qquad n=0,\ldots,N-1.
$$

When a transient cutoff is supplied, the calculations use
$\mathcal I=\{n:t_n\ge t_{\mathrm{burn}}\}$.

## Trajectory and boundedness metrics

`compute_trajectory_metrics` calculates the component range and population
variance over the selected tail:

$$
R_j=\max_{n\in\mathcal I}x_{n,j}-\min_{n\in\mathcal I}x_{n,j},
\qquad
s_j^2=\frac{1}{|\mathcal I|}
\sum_{n\in\mathcal I}(x_{n,j}-\bar x_j)^2.
$$

If equilibria $E_\ell$ are supplied, it also reports

$$
d_{\min}(X_{N-1})=\min_\ell\lVert X_{N-1}-E_\ell\rVert_2.
$$

`compute_boundedness_metrics` uses either
$r_n=\lVert X_n\rVert_2$ or $r_n=\lVert X_n\rVert_\infty$. With
$m=\max\{2,\operatorname{round}(\gamma N)\}$, it forms

$$
G=
\frac{\operatorname{mean}(r_{N-m},\ldots,r_{N-1})}
{\max\{\operatorname{mean}(r_0,\ldots,r_{m-1}),
\epsilon_{\mathrm{mach}}\}}.
$$

The current policy labels a trajectory `unbounded_candidate` if its declared
divergence radius is exceeded or $G\ge100$. It labels it
`bounded_candidate` when the observed maximum is finite and $G<10$.
Intermediate values are inconclusive. These thresholds are library policy,
not stability theorems.

```python
from hidden_attractors import (
    compute_boundedness_metrics,
    compute_trajectory_metrics,
    trajectory_metrics_for_system,
)

trajectory_metrics = compute_trajectory_metrics(
    times,
    states,
    equilibria=equilibria,
    t_start=40.0,
)
system_metrics = trajectory_metrics_for_system(
    trajectory,
    system=system,
    h=0.01,
    t_start=40.0,
)
boundedness = compute_boundedness_metrics(
    times,
    states,
    burn_time=40.0,
    divergence_radius=120.0,
)
```

`trajectory_metrics_for_system` is a facade for the same equations. It accepts
a $(t,X)$ matrix and obtains equilibria from `system`, or accepts an explicit
`equilibria` dictionary. With `has_time=False`, it constructs $t_n=nh$; when
the matrix already contains time, `h` does not alter those samples. One of the
two equilibrium sources is required. The function performs neither seed
search nor hiddenness testing.

```python
trajectory_metrics_for_system(
    traj,
    *,
    system=None,
    equilibria=None,
    h,
    t_start,
    divergence_norm=120.0,
    equilibrium_tol=1.0e-3,
    has_time=True,
)
```

## DFT, FFT, spectral entropy, and segmented periodograms

After centering or detrending and applying a window $w_n$, NumPy evaluates

$$
Y_k=\sum_{n=0}^{N-1}w_n y_n
\exp\left(-2\pi i\frac{kn}{N}\right),
\qquad
f_k=\frac{k}{N\Delta t}.
$$

`fft_spectrum` returns the one-sided amplitude convention

$$
A_k=\frac{|Y_k|}{\sum_n w_n}.
$$

`compute_fft_psd` forms $P_k=|Y_k|^2$ and, when requested, normalizes
$p_k=P_k/\sum_j P_j$. The normalized spectral entropy is

$$
H_{\mathrm{spec}}=
-\frac{\sum_{k:p_k>0}p_k\log p_k}{\log K}.
$$

The dominant-bin ratio and the minimum fraction of sorted bins that contains
90 percent of the power support the library's conservative spectral labels.
Those labels do not certify chaos.

For `psd_welch`, segment $r$ of length $L$ is centered and multiplied by a
Hann window. The implementation returns

$$
\widehat S_{\mathrm{code}}(f_k)=
\frac{1}{M}\sum_{r=1}^{M}
c_k\frac{\Delta t\left|
\sum_{n=0}^{L-1}w_ny_{r,n}e^{-2\pi i kn/L}
\right|^2}
{\sum_{n=0}^{L-1}w_n^2}.
$$

Here $c_k=2$ for interior positive-frequency bins and $c_k=1$ for DC and,
for even segment length, Nyquist. This is the standard one-sided Welch density
scaling. If the signal has units $U$ and time has units $T$, the result has
units $U^2/\mathrm{Hz}=U^2T$. The implementation is tested against
`scipy.signal.welch` with the same explicit window, overlap, and detrending.

```python
from hidden_attractors import compute_fft_psd
from hidden_attractors.analysis import fft_spectrum, psd_welch

summary = compute_fft_psd(
    times,
    states[:, 0],
    burn_time=40.0,
    window="hann",
    normalize_power=True,
)
fft = fft_spectrum(states[:, 0], h=times[1] - times[0])
welch = psd_welch(
    states[:, 0],
    h=times[1] - times[0],
    nperseg=512,
    overlap=0.5,
)
```

## Numerical Poincare sections

For the section $x_s=c$, a positive sampled crossing satisfies

$$
x_{n,s}<c\le x_{n+1,s},
\qquad x_{n+1,s}-x_{n,s}>0,
$$

and a negative crossing satisfies

$$
x_{n,s}>c\ge x_{n+1,s},
\qquad x_{n+1,s}-x_{n,s}<0.
$$

Linear interpolation gives

$$
\theta_n=\frac{c-x_{n,s}}{x_{n+1,s}-x_{n,s}},\qquad
t_\star=t_n+\theta_n(t_{n+1}-t_n),
$$

$$
X_\star=X_n+\theta_n(X_{n+1}-X_n).
$$

In `integer_rhs` mode, the requested sign is also checked against
$F_s(t_\star,X_\star)$. In `geometric_fractional` mode, only the orientation
of the sampled segment is used. Both modes remain sampled, linearly
interpolated sections, so `exact_poincare_map` is always `False`.

```python
from hidden_attractors import detect_poincare_crossings

section = detect_poincare_crossings(
    times,
    states,
    section_variable="x",
    section_value=0.0,
    direction="positive",
    derivative_mode="geometric_fractional",
    burn_time=40.0,
)
```

For an integer ODE, pass `derivative_mode="integer_rhs"` and the corresponding
`rhs` callable. For Caputo data, the result is a geometric section of the
stored trajectory, not an exact classical return map.

## Gottwald--Melbourne 0--1 test

For a scalar observable $\phi_j$ and
$c\in(\pi/5,4\pi/5)$, the implementation forms

$$
p_c(n)=\sum_{j=1}^{n}\phi_j\cos(jc),\qquad
q_c(n)=\sum_{j=1}^{n}\phi_j\sin(jc).
$$

The mean-square displacement at lag $\ell$ is

$$
M_c(\ell)=\frac{1}{N-\ell}
\sum_{j=1}^{N-\ell}
\left(
[p_c(j+\ell)-p_c(j)]^2+
[q_c(j+\ell)-q_c(j)]^2
\right).
$$

After removing

$$
V_{\mathrm{osc}}(\ell)=
\bar\phi^2\frac{1-\cos(\ell c)}{1-\cos c},
\qquad
D_c(\ell)=M_c(\ell)-V_{\mathrm{osc}}(\ell),
$$

the statistic is

$$
K_c=\operatorname{corr}\bigl(
(1,\ldots,L),(D_c(1),\ldots,D_c(L))
\bigr).
$$

The reported value is the median across reproducibly sampled values of $c$.
The library uses $K>0.8$ for a chaotic candidate, $K<0.2$ for a regular
candidate, and otherwise reports an inconclusive result.

```python
from hidden_attractors import zero_one_test

result_01 = zero_one_test(
    states[:, 0],
    n_c=100,
    random_seed=12345,
    detrend=True,
    normalize=True,
)
```

Noise, trends, oversampling, and transients can bias the statistic.

## Bifurcation-diagram post-processing

`bifurcation_points_from_trajectories` does not continue branches. For each
supplied parameter value, it discards the transient and can retain local
maxima satisfying

$$
y_n-y_{n-1}\ge0,\qquad y_n-y_{n+1}\ge0.
$$

The `minima`, `both`, and `sample` modes change this selection rule.
For a nonempty extracted set $\mathcal B$, `bifurcation_summary` reports only

$$
n_{\mathcal B}=|\mathcal B|,\qquad
\mu_{\min}=\min_{b\in\mathcal B}\mu_b,\qquad
\mu_{\max}=\max_{b\in\mathcal B}\mu_b,
$$

$$
y_{\min}=\min_{b\in\mathcal B}y_b,\qquad
y_{\max}=\max_{b\in\mathcal B}y_b.
$$

For an empty set it returns only `n_points=0`; the summary does not locate
bifurcations. Its public signature is `bifurcation_summary(points)`.

```python
from hidden_attractors import (
    bifurcation_points_from_trajectories,
    bifurcation_summary,
)
from hidden_attractors.plotting import plot_bifurcation_diagram

points = bifurcation_points_from_trajectories(
    scans,
    observable="x",
    t_start=40.0,
    mode="maxima",
)
point_summary = bifurcation_summary(points)
plot_bifurcation_diagram(
    points,
    "bifurcation.png",
    parameter_label="mu",
    observable_label="local maxima of x",
)
```

The plotted point cloud does not automatically identify bifurcation points,
branch stability, or Floquet multipliers.

## Equation-based Lyapunov spectra

### Integer QR--Benettin route

For $\dot X=F(X)$, the continuous variational equation is

$$
\dot\Phi(t)=J(X(t))\Phi(t),\qquad
J(X)=\frac{\partial F}{\partial X},\qquad
\Phi(0)=I.
$$

At reorthonormalization time $k$, write $\Phi_k=Q_kR_k$. The finite-time
estimate is

$$
\widehat\lambda_i(T)=
\frac{1}{T}\sum_{k=1}^{m}\log |(R_k)_{ii}|.
$$

The exact discretization used by `integer_qr_benettin` is

$$
X_{n+1}=\operatorname{EFORK}_{q=1}(F,X_n,h),\qquad
\Phi_{n+1}=\Phi_n+hJ(X_n)\Phi_n.
$$

Thus the state uses the three-stage `efork_q1_step`, while the tangent basis
uses explicit Euler before QR. Step refinement is required. The method has
exact linear controls and internal cross-checks, but no current quantitative
published-spectrum reproduction claim.

`integer_qr_benettin_lyapunov_exponents` is the canonical direct facade for
$F$, $J$, and $X_0$; it reuses the same numerical core and is not a second
method. It enforces the integer contract $|q-1|\le10^{-9}$.
`integer_system_lyapunov_exponents` adapts a `ChaoticSystem`: it uses
`system.evaluate`, consumes `jacobian_matrix` when an analytic Jacobian is
declared, and otherwise applies centered finite differences,

$$
J_{ij}(X)\approx
\frac{F_i(X+\varepsilon e_j)-F_i(X-\varepsilon e_j)}
{2\varepsilon}.
$$

The system facade rejects a declared order when
$|q_{\mathrm{system}}-1|>10^{-9}$. Objects with no detectable order remain
accepted for compatibility, so acceptance alone is not evidence that a model
is integer order.

```python
integer_qr_benettin_lyapunov_exponents(
    rhs,
    jacobian,
    x0,
    *,
    h,
    t_final,
    t_burn=0.0,
    reorthonormalize_every=10,
    jacobian_eps=1.0e-6,
    div_threshold=None,
    q=1.0,
)
integer_system_lyapunov_exponents(
    system,
    x0,
    *,
    h,
    t_final,
    t_burn=0.0,
    reorthonormalize_every=10,
    jacobian_eps=1.0e-6,
    div_threshold=None,
)
```

```python
from hidden_attractors import (
    integer_qr_benettin_lyapunov_exponents,
    integer_system_lyapunov_exponents,
)

direct = integer_qr_benettin_lyapunov_exponents(
    rhs,
    jacobian,
    x0,
    h=0.01,
    t_final=20.0,
    t_burn=2.0,
    q=1.0,
)
registered = integer_system_lyapunov_exponents(
    system,
    x0,
    h=0.01,
    t_final=20.0,
    t_burn=2.0,
)
```

### Fractional variational ABM--QR route

For $0<q<1$, the extended Caputo system is

$$
{}^CD_t^qX=F(X),\qquad
{}^CD_t^q\Phi=J(X)\Phi,\qquad
\Phi(0)=I.
$$

With history-aware QR, after $\Phi_k=Q_kR_k$ the implementation transforms
each stored variational block:

$$
\Phi_j\leftarrow\Phi_jR_k^{-1},\qquad
G_j\leftarrow
\begin{bmatrix}F(X_j)\\J(X_j)\Phi_j\end{bmatrix},
\qquad j\le k.
$$

The original--variational Caputo system follows the published foundation.
Transforming the entire stored history after QR is a project-specific
extension, not a literal reproduction of the cited algorithm.

### Cloned-dynamics routes

The Jacobian-free routes initialize $d$ clones:

$$
X^{(i)}_0=X_0+\delta e_i,\qquad i=1,\ldots,d.
$$

At the end of each block,

$$
V_k=\left[
X^{(1)}_k-X^{(0)}_k,\ldots,
X^{(d)}_k-X^{(0)}_k
\right]
$$

is orthogonalized by modified Gram--Schmidt or QR. If $\rho_{k,i}$ are the
resulting norms,

$$
\widehat\lambda_i=
\frac{1}{K t_{\mathrm{clone}}}
\sum_{k=1}^{K}\log\frac{\rho_{k,i}}{\delta}.
$$

Both cloned routes restart fractional ABM history at each block. The returned
metadata therefore records `effective_memory_protocol="published_block_restart"`.
Older protocol names are retained only as `requested_memory_protocol`
compatibility aliases; QR versus Gram--Schmidt is the effective difference.
They are not full-memory Caputo estimators.

| Method identifier | Order | Jacobian | Evidence status |
| --- | --- | --- | --- |
| `integer_qr_benettin` | `q=1` | Analytic or centered finite differences | Exact linear controls and internal cross-checks |
| `fractional_variational_abm_qr` | `0<q<1` | Analytic or centered finite differences | Synthetic validation only |
| `fractional_cloned_dynamics_abm_gs_published` | `0<q<=1` | No | Published reproduction lane with a recorded benchmark discrepancy |
| `fractional_cloned_dynamics_abm_qr` | `0<q<=1` | No | Numerical-comparison variant |

### Explicit method-validation contract

`validate_lyapunov_method_request(request)` checks a
`LyapunovComputationRequest` without running an integrator and returns exactly

$$
(\mathrm{ok},\mathrm{status},\mathrm{warnings}).
$$

A compatible request returns `(True, "compatible", warnings)`. An ordinary
incompatibility is represented by `ok=False`, not by an exception. The request
must select exactly one of `system` or `rhs` as its vector-field source:

$$
\mathbf 1_{\{\mathrm{system}\ne\mathrm{None}\}}+
\mathbf 1_{\{\mathrm{rhs}\ne\mathrm{None}\}}=1.
$$

The generic numerical domain is

$$
X_0\in\mathbb R^d,\ d\ge1,\qquad
h>0,\quad T_{\mathrm{final}}>0,\quad T_{\mathrm{burn}}\ge0,\quad
\varepsilon_J>0,
$$

with finite $X_0$, $q$, times, $h$, and $\varepsilon_J$. When supplied,
`reorthonormalize_every` must be a positive integer. `memory_window` is
inspected only for the variational route with `memory_mode="window"`, where it
must also be a positive integer. Supplied `reorthonormalization_time` and
`div_threshold` values must be finite and positive.

The method-specific gates mirror the live registry:

- integer QR--Benettin requires `q=1` and
  `memory_mode="not_applicable"`;
- fractional variational ABM--QR requires `0<q<1`, memory mode `full` or
  `window`, and checks a declared system order against the requested `q`;
- both cloned-dynamics routes require `0<q<=1` and accept
  `not_applicable`, `published_block_restart`, or the compatibility alias
  `experimental_qr_block_restart`.

Missing analytic Jacobians in the variational methods add
`analytic_jacobian_missing_finite_difference_used`; cloned routes add
`cloned_dynamics_no_jacobian_required`. These are advisories, not
certifications of chaos, hiddenness, or fractional validity.

The current incompatibility statuses are `unknown_method`,
`invalid_parameter`, `method_not_valid_for_fractional_caputo`,
`memory_mode_not_applicable_for_integer_method`,
`method_not_valid_for_integer_or_out_of_range_q`,
`memory_mode_must_be_full_or_window_for_fractional_method`,
`request_q_does_not_match_system_q`,
`method_not_valid_for_out_of_range_q`, and
`memory_mode_must_be_block_restart_for_cloned_dynamics`.

```python
from hidden_attractors import (
    LyapunovComputationRequest,
    validate_lyapunov_method_request,
)

request = LyapunovComputationRequest(
    system=system,
    rhs=None,
    jacobian=None,
    x0=x0,
    q=1.0,
    method="integer_qr_benettin",
    h=0.01,
    t_final=20.0,
    t_burn=2.0,
    reorthonormalize_every=10,
    memory_mode="not_applicable",
)
ok, status, warnings = validate_lyapunov_method_request(request)
```

```python
import numpy as np
from hidden_attractors import compute_lyapunov_spectrum

rhs = lambda x: np.array([-x[0], -2.0 * x[1]])
jac = lambda x: np.array([[-1.0, 0.0], [0.0, -2.0]])

summary = compute_lyapunov_spectrum(
    rhs=rhs,
    jacobian=jac,
    x0=np.array([1.0, 1.0]),
    q=1.0,
    method="integer_qr_benettin",
    h=0.01,
    t_final=20.0,
    t_burn=2.0,
)
print(summary.result.exponents)
```

## Lyapunov reconstruction from a scalar time series

For a scalar series $s_n$, delay reconstruction forms

$$
Y_i=(s_i,s_{i+\tau},\ldots,s_{i+(m-1)\tau})^\mathsf{T}.
$$

Rosenstein's route follows the average logarithmic separation of temporally
separated neighbors:

$$
d(k)=
\left\langle
\log\lVert Y_{i+k}-Y_{j(i)+k}\rVert_2
\right\rangle_i,\qquad
\widehat\lambda_{\max}=
\frac{1}{\Delta t}\frac{\mathrm d d(k)}{\mathrm d k}.
$$

Eckmann's route fits local maps

$$
Y_{i+1}-Y_{j+1}\approx A_i(Y_i-Y_j)
$$

and obtains a finite reconstructed spectrum by matrix products and
reorthogonalization. If
$\lambda_1\ge\cdots\ge\lambda_m$, the Kaplan--Yorke dimension is

$$
D_{\mathrm{KY}}=
j+\frac{\sum_{i=1}^{j}\lambda_i}{|\lambda_{j+1}|},
\qquad
j=\max\left\{r:\sum_{i=1}^{r}\lambda_i\ge0\right\}.
$$

```python
from hidden_attractors import estimate_time_series_lyapunov

result = estimate_time_series_lyapunov(
    signal,
    sample_interval=0.01,
    time_unit="s",
    observable="x",
    rosenstein_emb_dim=8,
    eckmann_emb_dim=9,
    eckmann_matrix_dim=3,
)
print(result.largest_exponent, result.exponent_unit)
print(result.spectrum, result.kaplan_yorke_dimension)
```

The result records sampling, embedding, neighbor, fit, backend-version, and
memory-limit metadata. It is dependent on the observable and reconstruction
parameters.

## Optional complexity backends

`compute_complexity_measures` delegates to installed `nolds` or `antropy`
versions. Depending on the backend, it exposes:

- sample entropy,
  $\operatorname{SampEn}(m,r)=-\log(A_{m+1}/B_m)$;
- correlation dimension,
  $D_2=\lim_{r\to0}\mathrm d\log C(r)/\mathrm d\log r$;
- Rosenstein's largest exponent and the Hurst exponent;
- detrended fluctuation analysis, $F(s)\propto s^\alpha$;
- normalized permutation and spectral entropy; and
- Higuchi fractal dimension.

```python
from hidden_attractors.integrations import (
    available_complexity_backends,
    compute_complexity_measures,
)

print(available_complexity_backends())
complexity = compute_complexity_measures(
    signal,
    backend="auto",
    sample_rate=100.0,
    measures=[
        "sample_entropy",
        "permutation_entropy",
        "spectral_entropy",
        "dfa",
    ],
)
```

The exact defaults and corrections belong to the recorded backend version.

## Ready-to-use integrators

For $\dot X=F(t,X)$, Heun uses

$$
K_1=F(t_n,X_n),\qquad
K_2=F(t_n+h,X_n+hK_1),
$$

$$
X_{n+1}=X_n+\frac h2(K_1+K_2).
$$

Classical RK4 uses

$$
\begin{aligned}
K_1&=F(t_n,X_n),\\
K_2&=F(t_n+h/2,X_n+hK_1/2),\\
K_3&=F(t_n+h/2,X_n+hK_2/2),\\
K_4&=F(t_n+h,X_n+hK_3),\\
X_{n+1}&=X_n+\frac h6(K_1+2K_2+2K_3+K_4).
\end{aligned}
$$

For commensurate Caputo order $0<q<1$, the full-history ABM predictor is

$$
X_{n+1}^{P}=X_0+
\frac{h^q}{\Gamma(q+1)}
\sum_{j=0}^{n}
\left[(n+1-j)^q-(n-j)^q\right]F_j,
$$

followed by the Adams--Moulton corrector

$$
X_{n+1}=X_0+\frac{h^q}{\Gamma(q+2)}
\left[
F(t_{n+1},X_{n+1}^{P})+
\sum_{j=0}^{n}a_{j,n+1}F_j
\right],
$$

with the weights implemented in `integrations/abm.py`. `memory_mode="window"`
changes the lower history anchor and is not equivalent to full Caputo memory.

EFORK-3 uses

$$
\begin{aligned}
K_1&=h^q\widetilde F(t_n,X_n),\\
K_2&=h^q\widetilde F(t_n+c_2h,X_n+a_{21}K_1),\\
K_3&=h^q\widetilde F(t_n+c_3h,
X_n+a_{31}K_1+a_{32}K_2),\\
X_{n+1}&=X_n+w_1K_1+w_2K_2+w_3K_3,
\end{aligned}
$$

where $\widetilde F$ includes the discrete Caputo-history term and the
coefficients are computed from $\Gamma(1+q)$, $\Gamma(1+2q)$, and
$\Gamma(1+3q)$. At $q=1$, the selector uses the distinct three-stage
`EFORK_Q1` coefficient limit.

```python
from hidden_attractors.integrations.selector import integrate

times, states, status = integrate(
    rhs,
    x0,
    q=1.0,
    h=0.01,
    t_final=20.0,
    integrator="rk4",       # or "heun", "efork_q1"
)

times_q, states_q, status_q = integrate(
    rhs,
    x0,
    q=0.98,
    h=0.01,
    t_final=20.0,
    integrator="abm",       # or "efork3"
    memory_mode="full",
)
```

`adm_wu2023_integrate` is a specialized fourth-order local Adomian-series
reproduction for the Wu arctan model:

$$
X_{n+1}=X_n+
\sum_{r=1}^{4}C^{(r)}
\frac{h^{rq}}{\Gamma(rq+1)}.
$$

Each update uses only $X_n$; it contains no Caputo history convolution and is
not equivalent to ABM or EFORK-3.

```python
import numpy as np
from hidden_attractors.integrations import adm_wu2023_integrate

times, states, status, info = adm_wu2023_integrate(
    params={
        "alpha": 10.0,
        "beta": 14.87,
        "gamma": 0.0,
        "a1": -1.27,
        "a2": 1.08,
        "rho": 1.0,
    },
    x0=np.array([0.1, 0.0, 0.0]),
    q=0.99,
    h=0.005,
    N=2000,
)
```

The generic selector intentionally rejects `integrator="adm_wu2023"` because
that method requires model-specific parameters and a step-count contract.

## Primary references

- Cooley and Tukey, “An Algorithm for the Machine Calculation of Complex
  Fourier Series,” 1965,
  [DOI 10.1090/S0025-5718-1965-0178586-1](https://doi.org/10.1090/S0025-5718-1965-0178586-1).
- Welch, “The Use of Fast Fourier Transform for the Estimation of Power
  Spectra,” 1967,
  [DOI 10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901).
- Shannon, “A Mathematical Theory of Communication,” 1948,
  [DOI 10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x).
- Henon, “On the Numerical Computation of Poincare Maps,” 1982,
  [DOI 10.1016/0167-2789(82)90034-3](https://doi.org/10.1016/0167-2789(82)90034-3).
- Gottwald and Melbourne, “A New Test for Chaos in Deterministic Systems,”
  2004,
  [DOI 10.1098/rspa.2003.1183](https://doi.org/10.1098/rspa.2003.1183),
  and “On the Implementation of the 0--1 Test for Chaos,” 2009,
  [DOI 10.1137/080718851](https://doi.org/10.1137/080718851).
- Benettin et al., “Lyapunov Characteristic Exponents for Smooth Dynamical
  Systems,” parts 1 and 2, 1980,
  [DOI 10.1007/BF02128236](https://doi.org/10.1007/BF02128236) and
  [DOI 10.1007/BF02128237](https://doi.org/10.1007/BF02128237).
- Danca and Kuznetsov, “Matlab Code for Lyapunov Exponents of
  Fractional-Order Systems,” 2018,
  [DOI 10.1142/S0218127418500670](https://doi.org/10.1142/S0218127418500670).
- Fischer, Zourmba, and Mohamadou, “Lyapunov Exponents Spectrum Estimation of
  Fractional Order Nonlinear Systems Using Cloned Dynamics,” 2020,
  [DOI 10.1016/j.apnum.2020.03.027](https://doi.org/10.1016/j.apnum.2020.03.027).
- Rosenstein, Collins, and De Luca, “A Practical Method for Calculating Largest
  Lyapunov Exponents from Small Data Sets,” 1993,
  [DOI 10.1016/0167-2789(93)90009-P](https://doi.org/10.1016/0167-2789(93)90009-P).
- Eckmann et al., “Liapunov Exponents from Time Series,” 1986,
  [DOI 10.1103/PhysRevA.34.4971](https://doi.org/10.1103/PhysRevA.34.4971).
- Frederickson et al., “The Liapunov Dimension of Strange Attractors,” 1983,
  [DOI 10.1016/0022-0396(83)90011-6](https://doi.org/10.1016/0022-0396(83)90011-6).
- Diethelm, Ford, and Freed, “A Predictor-Corrector Approach for the Numerical
  Solution of Fractional Differential Equations,” 2002,
  [DOI 10.1023/A:1016592219341](https://doi.org/10.1023/A:1016592219341).
- Ghoreishi, Ghaffari, and Saad, “Fractional Order Runge-Kutta Methods,” 2023,
  [DOI 10.3390/fractalfract7030245](https://doi.org/10.3390/fractalfract7030245).
- Richman and Moorman, “Physiological Time-Series Analysis Using Approximate
  Entropy and Sample Entropy,” 2000,
  [DOI 10.1152/ajpheart.2000.278.6.H2039](https://doi.org/10.1152/ajpheart.2000.278.6.H2039).
- Grassberger and Procaccia, “Characterization of Strange Attractors,” 1983,
  [DOI 10.1103/PhysRevLett.50.346](https://doi.org/10.1103/PhysRevLett.50.346).
- Bandt and Pompe, “Permutation Entropy,” 2002,
  [DOI 10.1103/PhysRevLett.88.174102](https://doi.org/10.1103/PhysRevLett.88.174102).
- Peng et al., “Mosaic Organization of DNA Nucleotides,” 1994,
  [DOI 10.1103/PhysRevE.49.1685](https://doi.org/10.1103/PhysRevE.49.1685).
- Higuchi, “Approach to an Irregular Time Series on the Basis of the Fractal
  Theory,” 1988,
  [DOI 10.1016/0167-2789(88)90081-4](https://doi.org/10.1016/0167-2789(88)90081-4).

See also the [plotting function catalog](plot_function_catalog.md), where each
public graph function has a direct call and a reproducible example image.
