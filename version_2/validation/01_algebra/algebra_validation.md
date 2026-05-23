# Algebra Validation

The non-smooth fractional Chua model at `q=0.9998` reproduces Danca's parameter set and the MATLAB validation values. Python returns the same three equilibria, zero vector-field residuals to floating-point precision, central-difference agreement with the analytic regional Jacobians, and the same inner/outer spectra exported by MATLAB and Wolfram. Matignon classification is stable at `E0` and unstable at `E+` and `E-`.

The supplied Wolfram source verifies the symbolic identities after renaming its protected local symbol `Tr` to `Treal`.

## Alcance de esta etapa / Scope of this stage

Esta etapa valida formalmente la consistencia de los equilibrios, Jacobianos, autovalores y el criterio de estabilidad de Matignon. 
Esta etapa **no** valida la integración de trayectoria fraccionaria, la existencia de atractores (caóticos u otros), las propiedades de ocultamiento (hiddenness) ni la robustez del sistema bajo perturbaciones o variaciones paramétricas. Dichas propiedades y validaciones numéricas se delegan y demuestran formalmente en las etapas posteriores del contrato de validación.

