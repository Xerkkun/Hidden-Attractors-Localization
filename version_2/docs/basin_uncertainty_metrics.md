# Entropía de cuencas e incertidumbre de estado final

Estado: `experimental`. Estas funciones cuantifican una clasificación finita;
no certifican por sí solas fractalidad, frontera Wada, caos ni hiddenness global.

## Alcance entero y fraccionario

`hidden_attractors.analysis.basin_uncertainty` recibe etiquetas de destino ya
calculadas. Por ello el mismo código sirve para una malla de condiciones
iniciales de orden entero o para resultados de un solver fraccionario. En el
segundo caso, el experimento debe conservar además:

- definición de derivada y orden;
- terminal inferior y prehistoria;
- método, paso, tolerancias y política de memoria;
- clasificador de destino, horizonte y tasa no clasificada.

Compartir la métrica no vuelve Markoviano el modelo fraccionario. Dos puntos con
el mismo estado proyectado pueden representar historias diferentes.

## Entropía de cuencas

Para una caja no vacía `i` con proporciones de etiquetas `p_ij`, HAFO calcula

\[
S_i=-\sum_j p_{ij}\log p_{ij},\qquad
S_b=\frac{1}{N}\sum_{i=1}^{N}S_i.
\]

La entropía de las cajas de frontera promedia sólo cajas con al menos dos
destinos:

\[
S_{bb}=\frac{1}{N_b}\sum_{i\in\mathrm{frontera}}S_i.
\]

La implementación Numba divide una malla bidimensional regular en cajas no
solapadas ancladas en el origen. Por defecto exige divisibilidad para promediar
cajas de igual área. `partial_boxes="drop"` elimina franjas incompletas y
`"include_equal"` conserva cajas parciales con peso igual, variante que queda
registrada en el resultado. Las etiquetas de `ignored_labels` no forman
probabilidades; una caja que contenga sólo dichas etiquetas se reporta vacía.

```python
from hidden_attractors.analysis.basin_uncertainty import basin_entropy

result = basin_entropy(class_ids, box_size=(8, 8), ignored_labels=(-1,))
print(result.basin_entropy, result.boundary_basin_entropy)
```

El resultado registra forma de caja, política parcial, número de cuencas
observadas, muestras ignoradas o descartadas, cajas vacías y fracción de
frontera. La base debe ser mayor que uno.
También informa el margen y una tolerancia numérica al comparar `S_bb` con
`log(2)` en la base elegida. `boundary_entropy_defined` distingue el caso sin
cajas de frontera. El booleano `boundary_entropy_above_log_two` sólo se activa
si `log_two_criterion_applicable` es verdadero: al menos tres cuencas observadas,
cajas comparables y ninguna muestra ignorada que altere sus probabilidades.
`log_two_criterion_reason` conserva la causa cuando esas precondiciones finitas
no se cumplen. Esta comparación implementa el criterio suficiente de Daza et al.
bajo sus hipótesis; no sustituye un estudio de convergencia en escala ni una
prueba Wada.

## Fracción de incertidumbre

Para pares de condiciones de referencia y perturbadas separados por una escala
declarada `epsilon`, la fracción es

\[
f(\epsilon)=\frac{\#\{k:c_k\ne c'_k\}}{N_\mathrm{válido}}.
\]

```python
from hidden_attractors.analysis.basin_uncertainty import uncertainty_fraction

estimate = uncertainty_fraction(
    labels_reference,
    labels_perturbed,
    ignored_labels=(-1,),
    confidence=0.95,
    perturbation_scale=1e-4,
    scale_units="dimensionless_state",
    perturbation_norm="euclidean",
    perturbation_direction=(1.0, 0.0),
)
```

HAFO conserva escala, unidades, norma y dirección declaradas, y devuelve el
intervalo binomial de Wilson bajo la hipótesis de pares
Bernoulli independientes. En una rejilla espacial correlacionada es sólo un
intervalo descriptivo; no incluye error de integración, clasificación o memoria.

Si varias escalas muestran un régimen aproximadamente potencial,
`estimate_uncertainty_exponent` ajusta

\[
\log f(\epsilon)=\alpha\log\epsilon+\log C.
\]

Devuelve `alpha`, `C`, error estándar ordinario, `R²` y todas las escalas usadas.
Cuando se proporciona `sampling_space_dimension=D` —dimensión de la malla o
sección, no necesariamente del espacio de fases completo— también reporta
`D-alpha` y si cae en `[0,D]`. Una respuesta constante produce `R²=NaN` y estado
degenerado. Incluso un `R²` alto no demuestra régimen asintótico.

## Protocolo mínimo recomendado

1. Repetir varias resoluciones de malla y tamaños/desplazamientos de caja.
2. Reportar muestras ignoradas o no clasificadas, sin convertirlas en un destino.
3. Construir perturbaciones isotrópicas o declarar su dirección y norma.
4. Repetir el ajuste en subintervalos de escala y con nuevas muestras.
5. En orden fraccionario, repetir políticas de prehistoria/memoria relevantes.
6. Mantener estos resultados como soporte de cuenca; hiddenness requiere además
   controles desde vecindades de todos los equilibrios pertinentes.

## Referencias

- A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin y M. A. F. Sanjuán,
  “Basin entropy: a new tool to analyze uncertainty in dynamical systems”,
  *Scientific Reports* 6, 31416 (2016),
  [DOI 10.1038/srep31416](https://doi.org/10.1038/srep31416).
- S. W. McDonald, C. Grebogi, E. Ott y J. A. Yorke, “Structure and crises of
  fractal basin boundaries”, *Physics Letters A* 107, 51–54 (1985),
  [DOI 10.1016/0375-9601(85)90193-8](https://doi.org/10.1016/0375-9601(85)90193-8).
