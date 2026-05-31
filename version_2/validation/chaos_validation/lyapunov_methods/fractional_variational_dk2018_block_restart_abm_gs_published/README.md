# DK2018 block-restart ABM-GS published reproduction

This directory is the official promotion target for quantitative reproduction
of Danca and Kuznetsov (2018), DOI `10.1142/S0218127418500670`.

The method identifier is `fractional_variational_dk2018_block_restart_abm_gs`.
It integrates the extended system in renormalization blocks and applies
Gram-Schmidt between blocks, matching the supplied `FO_Lyapunov.m` contract.

The tracked state is
`published_benchmarks_pending_reproduced_discrepancy`. Numeric CSV files are
added only by
`validation/python/update_lyapunov_method_validation_status.py` after the RF
and Lorenz native C benchmarks both pass their published tolerances.

Compatibility with Garrappa `fde12.m` behavior remains an explicit gate. An
independent MATLAB R2025b oracle run using both the historical 2012 revision
and the supplied 2025 revision matched the native C RF result but did not
recover the RF vector printed in the 2018 article. See
`validation/matlab/README.md`. A native ABM-GS run must still match the
published RF and Lorenz values before promotion; MATLAB parity alone is
insufficient. In the RF run, `lambda_1` and `lambda_2` are within absolute
tolerance `0.05`; only `lambda_3` fails (`0.2525353` absolute difference).

The explicit long native run recorded on 2026-05-31 produced:

| Case | Long-run verdict | Failing components |
|---|---|---|
| Lorenz `q=0.985` | `published_benchmark_passed_quantitative` | none |
| RF `q=0.999` | `published_benchmark_failed` | `lambda_3` |

Fast CI executes native smoke cases only. Published quantitative tests are
marked `slow`, `published`, and `native`; they run only when
`RUN_PUBLISHED_LYAPUNOV=1`.

Passing this lane does not validate `fractional_variational_abm_qr`, does not
certify chaos, and does not certify hiddenness.
