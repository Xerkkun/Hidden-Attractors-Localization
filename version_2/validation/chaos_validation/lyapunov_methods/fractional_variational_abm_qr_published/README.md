# Fixed-lower-limit full-history ABM-QR validation

This directory tracks the local `fractional_variational_abm_qr` method.
Its fixed-lower-limit, history-aware QR contract is intentionally distinct
from the DK2018 block-restart reproduction lane.

The native C backend implements this contract with direct and FFT-block
convolution modes. Short direct-versus-FFT parity tests are available.
Published quantitative validation remains pending.

No result in this directory certifies chaos or hiddenness.
