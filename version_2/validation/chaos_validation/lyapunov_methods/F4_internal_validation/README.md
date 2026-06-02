# F4 internal Lyapunov validation

F4 records controlled internal consistency evidence for each implemented Lyapunov family.
It reuses existing published-reference artifacts and does not run long sweeps by default.
The closure state is `f4_complete_with_documented_discrepancies` when every method has a control, a sensitivity reference, and a bibliographic or internal reference.
This state does not certify chaos, hiddenness, or fractional method validity.
