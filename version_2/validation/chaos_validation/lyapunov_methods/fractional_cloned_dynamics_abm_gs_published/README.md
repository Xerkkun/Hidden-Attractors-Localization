# Fischer 2020 cloned-dynamics GS lane

This directory records the published-algorithm reproduction lane for
`fractional_cloned_dynamics_abm_gs_published`.

Current status: `published_benchmarks_not_run`.

The method uses ABM predictor-corrector integration with
`memory_protocol: published_block_restart` and modified Gram-Schmidt
orthonormalization. Results are finite-time local Lyapunov indicators. They do
not certify chaos or hiddenness, and they are not a full-memory Caputo-aware
claim.
