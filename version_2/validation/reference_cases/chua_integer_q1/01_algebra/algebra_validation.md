# Algebra Validation

The theoretical report and regenerated hiddenness summary use the same
piecewise-linear Chua parameters at `q=1`. The report and Python seed artifact
record zero residual for the implemented Lur'e representation. The equilibria
below are copied from the regenerated hiddenness result and are the centers used
for the finite neighborhood probes.

The supplied `chua_entero_algebraico_sin_numericos.wl` source was executed
through `wolframscript` during promotion. It evaluates the symbolic Lur'e
split, transfer function, canonical-transform identities, and
describing-function construction. It intentionally contains no numerical
parameter evaluation, so it validates algebraic structure rather than the
numeric trajectory.
