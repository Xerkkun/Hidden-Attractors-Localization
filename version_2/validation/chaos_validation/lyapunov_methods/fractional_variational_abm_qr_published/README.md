# Fixed-lower-limit full-history ABM-QR validation

This directory tracks the local `fractional_variational_abm_qr` method.
Its fixed-lower-limit, history-aware QR contract is intentionally distinct
from the DK2018 block-restart reproduction lane.

The native C backend implements this contract with direct and FFT-block
convolution modes. Short direct-versus-FFT parity tests are available.
Published quantitative validation remains pending.

DK2018 RF/Lorenz results cannot promote this lane because they use
`dk2018_block_restart_abm_gs`, not `fixed_lower_limit_full_history_qr`.
Promotion requires separate quantitative benchmarks designed for the local
full-history QR contract.

No result in this directory certifies chaos or hiddenness.
