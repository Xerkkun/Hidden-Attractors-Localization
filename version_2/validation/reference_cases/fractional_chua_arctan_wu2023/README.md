# Fractional Chua Arctan Wu2023 Evidence Package

Este directorio pertenece exclusivamente al caso suave
`fractional_chua_arctan_wu2023`. No contiene ni reutiliza resultados
numéricos del Chua no suave.

## Estado

| Etapa | Artefacto | Estado |
| --- | --- | --- |
| Álgebra y Matignon | `01_algebra/chua_arctan_wu2023_algebra.json` | ejecutada |
| Semillas Lur'e clásicas centradas | `02_lure_df/centered_seeds.json` | ejecutada |
| Condiciones iniciales reportadas | `03_reported_initial_conditions/` y `validation_summary.json` | ejecutada con ADM local publicado |
| Robustez y vecindades de equilibrios | por generar | pendiente |

La corrida fresca `q=0.99`, `h=0.01`, `N=10000` usa `ADM_WU2023`,
`adm_order=4`, backend `adm_local_reproduction` y
`memory_policy=none_local_adm`. Esta reproduccion sigue la recurrencia local
del articulo de Wu y no es equivalente a una integracion ABM con memoria
Caputo completa. Bajo ese contrato, ambas condiciones iniciales reportadas se
clasifican como `regular_periodic_rejected` despues del transitorio. Este
resultado no valida caos.

Las semillas Lur'e centradas se generan con la transferencia entera publicada
`W_pub(j*omega)=r^T(j*omega I-P)^(-1)b`, con `q_seed=1.0` y
`transfer_exponent_applied=false`. El modo `fractional_spectral` queda
separado como opcion experimental configurable.

`hidden_verified` permanece en `false` hasta ejecutar controles de cuenca
alrededor de `E0`, `E+` y `E-` con la referencia dinámica robusta definida
por la configuración.
