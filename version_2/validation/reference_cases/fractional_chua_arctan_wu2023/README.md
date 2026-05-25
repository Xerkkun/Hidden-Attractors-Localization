# Fractional Chua Arctan Wu2023 Evidence Package

Este directorio pertenece exclusivamente al caso suave
`fractional_chua_arctan_wu2023`. No contiene ni reutiliza resultados
numéricos del Chua no suave.

## Estado

| Etapa | Artefacto | Estado |
| --- | --- | --- |
| Álgebra y Matignon | `01_algebra/chua_arctan_wu2023_algebra.json` | ejecutada |
| Semillas Lur'e clásicas centradas | `02_lure_df/centered_seeds.json` | ejecutada |
| Condiciones iniciales reportadas | `03_reported_initial_conditions/` y `validation_summary.json` | ejecutada con historia completa efectiva |
| Robustez y vecindades de equilibrios | por generar | pendiente |

La corrida fresca `q=0.99`, `h=0.01`, `N=10000` con historia completa
efectiva (`memory_length=100.01`) clasifica ambas condiciones iniciales
reportadas como `regular_periodic_rejected` después del transitorio, con
periodicidad confirmada por `x`, `y` y `z`. Este resultado no valida caos.

`hidden_verified` permanece en `false` hasta ejecutar controles de cuenca
alrededor de `E0`, `E+` y `E-` con la referencia dinámica robusta definida
por la configuración.
