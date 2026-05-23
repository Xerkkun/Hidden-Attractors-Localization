# Integer Chua Verification Matrix

This matrix declares only checks already supported by stored artifacts. It
keeps the integer baseline separate from later fractional-case scripts and
from the independent EFORK method benchmark.

| Stage | Result currently supported | Validated against | Evidence |
|-------|----------------------------|-------------------|----------|
| `01_algebra` | Lur'e representation and symbolic transfer/canonical derivation | Supplied Wolfram Language/Mathematica derivation executed through `wolframscript`; theoretical report | `01_algebra/algebra_validation_summary.json`; `08_literature_comparison/sources/wolfram_run/wolfram_symbolic_output.txt` |
| `02_lure_df` | Selected low-frequency branch, gain, amplitude, harmonic seed, and closure residual | Guan--Xie Example 6 displayed values; locally executed MATLAB script; report | `08_literature_comparison/paper_numeric_comparison.csv`; `08_literature_comparison/matlab_numeric_comparison.csv` |
| `03_integrators` | Corrected EFORK-3 `q=1` stage ordering and regenerated continuation | Ghoreishi--Ghaffari--Saad table reproduction; Dr. de la Fraga supplied script; regenerated library run | `03_integrators/integrator_validation_summary.json`; `../efork3_ghoreishi_ghaffari/03_integrators/benchmark_results.csv` |
| `05_dynamic_analysis` | Regenerated Lyapunov spectrum plus FFT/PSD frequency diagnostics | Corrected Python/native EFORK-3 `q=1` run | `05_dynamic_analysis/dynamic_analysis_summary.json`; `fig11a_fft_x.png`; `fig11d_psd_x.png` |
| `06_hiddenness` | Regenerated `504` finite equilibrium-neighborhood probes with `TARGET=0` | Corrected Python/native EFORK-3 `q=1` run | `06_hiddenness/hiddenness_validation_summary.json`; `summary_by_radius_integer.csv` |
| `08_literature_comparison` | Source roles and externally comparable numerical values registered | Guan--Xie (2025), MATLAB R2025b, Wolfram/Mathematica symbolic source, attached report | `08_literature_comparison/` |

## Reading Rule

MATLAB reproduces numerical harmonic and canonical-transform values; the
Wolfram/Mathematica source validates symbolic algebra only. Guan--Xie supplies
rounded published values for the same Chua parameter set. Integration-dependent
artifacts from the earlier reversed `K3` implementation were deleted and
regenerated using the published `a31*K1 + a32*K2` order. The report remains a
source for theory and seed construction; its earlier trajectory-level values
are superseded by the corrected run. Hiddenness remains a finite numerical
candidate classification, not a global proof.
