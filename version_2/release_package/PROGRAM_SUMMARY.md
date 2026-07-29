# Program summary

Program title: `hidden-attractors-fo`

Version: 1.1.0

Repository: <https://github.com/Xerkkun/Hidden-Attractors-Localization>

Archive DOI: <https://doi.org/10.17605/OSF.IO/ZGK74>

License: MIT

Programming language: Python, with optional native C backends.

Dependencies: NumPy, SciPy, Matplotlib, Numba, and PyYAML. Optional analysis and
documentation dependencies are declared as installation extras.

Nature of problem: reproducible localization, numerical integration,
sampled-neighborhood verification, and conservative characterization of
integer-order and commensurate Caputo fractional-order Lur'e-compatible
systems. The package also characterizes trajectories and scalar time series
independently of the localization workflow.

Solution method: scalar Lur'e formulation, describing-function and Nyquist seed
construction, integer or Caputo continuation, ABM/EFORK integration, finite-time
diagnostics, all-equilibrium neighborhood tests, structured manifests, and
reproducible figure export. Independent characterization includes Lyapunov,
spectral, Poincare, boundedness, bifurcation, and 0-1 diagnostics. Version
1.1.0 fully integrates Rosenstein/Eckmann Lyapunov reconstruction and
Kaplan--Yorke dimension for uniformly sampled scalar time series.

Comprehensive software test: the integer Chua Lur'e reference executes search,
continuation, final integration, sampled-neighborhood controls, and
deterministic JSON/CSV output. The quick release control is recorded in
`sample_output/comprehensive_sample_summary.json`.

Restrictions: the localization workflow assumes scalar Lur'e-compatible systems
and commensurate order, with Caputo contracts for `0 < q <= 1`. Numerical
diagnostics and sampled-neighborhood results do not constitute a global proof of
hiddenness.

Running time: metadata and smoke checks complete in seconds. Numerical cost
depends on trajectory length, fractional-memory history, continuation grid, and
sampled-neighborhood size. The recorded quick comprehensive control took 2.455
seconds on its stated Windows environment; this value is provenance, not a
cross-platform benchmark.
