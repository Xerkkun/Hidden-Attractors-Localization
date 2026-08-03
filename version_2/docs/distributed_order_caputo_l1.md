# Solver Caputo de orden distribuido L1

Estado: `experimental`.

HAFO separa dos objetos que no deben confundirse:

- `distributed_order_gl_derivative` es un operador sobre muestras que permite
  bases GL/RL/Caputo desplazada, pesos firmados y memoria finita;
- `integrate_distributed_order_caputo_l1` es un solver de trayectoria para una
  ecuación Caputo de orden distribuido con medida discreta no negativa e
  historia completa.

La separación evita presentar una combinación algebraica GL como si fuera un
problema inicial Caputo bien planteado.

## Modelo continuo y cuadratura en el orden

El modelo soportado es

\[
 \int_{(0,1]} \mu(\alpha),{}_a^CD_t^\alpha x(t)\,d\alpha
 = f(t,x(t)),\qquad x(a)=x_0,
\]

aproximado por una regla explícita

\[
 \sum_{r=1}^{R}\Omega_r,{}_a^CD_t^{\alpha_r}x(t)
 = f(t,x(t)),
 \qquad 0<\alpha_r\le 1,\quad \Omega_r\ge0.
\]

La API no estima una densidad ni selecciona nodos automáticamente. El usuario
debe declarar una de estas semánticas:

- `nonnegative_mass`: `order_weights` ya contiene las masas efectivas
  \(\Omega_r\);
- `nonnegative_quadrature_density`: `order_weights` contiene los pesos de la
  regla y `density_values` contiene \(\mu(\alpha_r)\), por lo que
  \(\Omega_r=w_r\mu(\alpha_r)\).

`normalization="none"` es el valor predeterminado porque normalizar cambia el
modelo y posiblemente sus unidades. `unit_mass` sólo se aplica si se solicita y
el resultado conserva la masa cruda, la masa efectiva y sus normas L1.

Las medidas firmadas permanecen disponibles en el operador muestral, pero este
solver las rechaza. Con masas no negativas, el coeficiente del incremento
corriente es positivo; con cancelación firmada haría falta otro contrato de
existencia, estabilidad e invertibilidad.

## Discretización temporal L1

Para \(0<\alpha_r<1\), sobre \(t_n=a+nh\), HAFO usa

\[
 {}_a^CD_t^{\alpha_r}x(t_n)
 \approx
 \frac{h^{-\alpha_r}}{\Gamma(2-\alpha_r)}
 \sum_{k=0}^{n-1} b_k^{(r)}
 \bigl(x_{n-k}-x_{n-k-1}\bigr),
\]

\[
 b_k^{(r)}=(k+1)^{1-\alpha_r}-k^{1-\alpha_r}.
\]

Para historiales largos, los pesos se evalúan mediante `expm1` y `log1p` cuando
la resta directa pierde condicionamiento. No se usa `fastmath`.

Un nodo exactamente en \(\alpha=1\) activa una rama declarada:

\[
 b_0=1,\qquad b_k=0\;(k\ge1),\qquad
 \rho=\Omega/h.
\]

Por tanto, una masa única \((\alpha,\Omega)=(1,1)\) reduce al método de Euler
implícito. Esta interoperabilidad con orden entero es una extensión HAFO; no se
presenta como una tasa L1 fraccionaria.

## Kernel combinado

Defina

\[
 \rho_r=\frac{\Omega_r h^{-\alpha_r}}
                   {\Gamma(2-\alpha_r)},
 \qquad
 K_k=\sum_{r=1}^{R}\rho_r b_k^{(r)}.
\]

Entonces el problema discreto es

\[
 \sum_{k=0}^{n-1}K_k
 \bigl(x_{n-k}-x_{n-k-1}\bigr)=f(t_n,x_n).
\]

Como \(K_0=A=\sum_r\rho_r>0\), se aísla

\[
 x_n=B_n+A^{-1}f(t_n,x_n),
\]

\[
 B_n=x_{n-1}-A^{-1}
 \sum_{k=1}^{n-1}K_k
 \bigl(x_{n-k}-x_{n-k-1}\bigr).
\]

HAFO precomputa una sola secuencia \(K_k\). El costo estructural pasa de la
implementación ingenua `O(R*N^2*d)` a

\[
 O(RN+N^2d),
\]

más las evaluaciones de `f` y las iteraciones del corrector. El almacenamiento
es `O(N*d+N+R)` y no se materializa un tensor
`orden × tiempo × estado`.

## Corrector implícito y fallos

El predictor usa el estado anterior:

\[
 x_n^{(0)}=B_n+A^{-1}f(t_n,x_{n-1}),
\]

y Picard itera

\[
 x_n^{(m+1)}=B_n+A^{-1}f(t_n,x_n^{(m)}).
\]

Una condición suficiente local de contracción sería \(L_f/A<1\), pero HAFO no
estima \(L_f\) ni infiere contractividad. Cada salida reporta iteraciones y el
residuo algebraico

\[
 \left\|A(x_n-B_n)-f(t_n,x_n)\right\|.
\]

`on_nonconvergence="raise"` produce `DistributedOrderCorrectorError`;
`"return"` entrega sólo el prefijo aceptado y usa
`status="corrector_nonconvergence"`. Un estado no finito fallido no se incluye
ni se cuenta como paso completado. Los metadatos distinguen Numba solicitado,
intentado, realmente usado y cualquier fallback.

## Compatibilidad en el terminal inferior

Si todos los nodos efectivos satisfacen \(\alpha_r<1\) y se exige una solución
`C1`, cada término Caputo vale cero en \(t=a\); entonces debe cumplirse
\(f(a,x_0)=0\). HAFO sólo emite una advertencia cuando se declara
`initial_regularity="smooth"` y el residuo supera la tolerancia.

Los arranques débilmente singulares se conservan mediante
`initial_regularity="nonsmooth"` (`"weak"` es alias), sin heredar tasas de error
para soluciones suaves. Si existe masa efectiva en \(\alpha=1\), el término
clásico contiene \(x'(a)\) y el chequeo cero deja de ser aplicable.

## API directa

```python
import numpy as np
from scipy.special import gamma

from hidden_attractors.fractional import (
    integrate_distributed_order_caputo_l1,
)

nodes = np.array([0.35, 0.80])
masses = np.array([0.40, 0.60])


def rhs(time, state):
    del state
    value = sum(
        mass * gamma(3.0) / gamma(3.0 - alpha)
        * time ** (2.0 - alpha)
        for alpha, mass in zip(nodes, masses, strict=True)
    )
    return np.array([value])


result = integrate_distributed_order_caputo_l1(
    rhs,
    initial_state=np.array([1.0]),
    order_nodes=nodes,
    order_weights=masses,
    step=0.01,
    n_steps=100,
    initial_regularity="smooth",
)
```

El problema manufacturado tiene solución continua \(x(t)=1+t^2\). El error de
una malla finita es una comprobación numérica, no un teorema de convergencia.

## `FractionalProblem` y Toolbox Chaos

En esta definición, `orders` contiene los nodos de la medida y es independiente
de la dimensión del estado. No es un orden nominal ignorado:

```python
from hidden_attractors.fractional import (
    FractionalProblem,
    solve_fractional_problem,
)

problem = FractionalProblem(
    derivative="caputo_distributed_order",
    method="distributed_order_caputo_l1",
    orders=[0.35, 0.80],
    initial_state=[1.0],
    step=0.01,
    t_span=(0.0, 1.0),
    memory_policy="full_history",
    kernel_parameters={
        "order_weights": [0.40, 0.60],
        "weight_semantics": "nonnegative_mass",
        "normalization": "none",
        "order_quadrature_name": "two_atom_demo",
    },
    method_options={
        "initial_regularity": "smooth",
        "on_nonconvergence": "raise",
    },
    allow_experimental=True,
)

trajectory = solve_fractional_problem(problem, rhs)
```

`order_mode="distributed"` impide confundir el número de nodos con el número
de componentes. `solve_fractional_system` reutiliza el mismo contrato para un
sistema registrado o definido por expresiones en Toolbox Chaos.

## Pruebas y oráculo Wolfram

`tests/test_distributed_order_caputo_solver.py` contiene 86 casos focales:
solución manufacturada, refinamiento temporal con cuadratura fija, reducción
monoorden, comparación con Type III constante, límite backward Euler, sistema
lineal acoplado independiente, kernel combinado, paridad Numba--Python,
densidades/normalización, validación estricta, corrector, compatibilidad,
divergencia, no finitud y despacho por `FractionalProblem`.

El caso
`validation/wolfram/cases/distributed_order_caputo_l1.wl` no lee HAFO. Deriva
cada contribución mediante `Integrate` sobre el kernel Caputo, agrega tres nodos
y resuelve directamente una recurrencia lineal afín. La corrida focal de
`tests/test_distributed_order_caputo_wolfram.py` aprobó 6 pruebas. El artefacto
retuvo:

- residuo simbólico del peso: `0`;
- residuo máximo de la recurrencia: `0.0`;
- error máximo de la solución afín: `0.0`;
- diferencia máxima Wolfram--HAFO: `1.5543122344752192e-15`;
- tolerancia del comparador: `8e-12`.

La exactitud del caso afín es una identidad de esta discretización y una
comprobación cruzada finita. No demuestra una tasa para soluciones generales.

## Backend y alcance

- `use_acceleration=True`: Numba construye el kernel y evalúa las sumas
  históricas; el RHS y Picard siguen en Python.
- `allow_python_fallback=True`: un fallo Numba cambia a la referencia NumPy y se
  registra.
- sólo `full_history` está implementado;
- no hay estimador adaptativo de error temporal ni de cuadratura en el orden;
- no hay malla graduada, SOE, CQ corregida ni prueba automática de estabilidad;
- una trayectoria finita no prueba caos, atracción ni ocultedad.

## Fuentes y frontera de atribución

- [Caputo (2001)](https://www.math.bas.bg/complan/fcaa/volume4/index.html)
  define el marco de orden distribuido; no se le atribuye DOI.
- [Diethelm y Ford (2009)](https://doi.org/10.1016/j.cam.2008.07.018)
  fundamentan la reducción por cuadratura en el orden y su análisis numérico,
  principalmente para problemas lineales.
- [Hu, Liu, Anh y Turner (2014)](https://doi.org/10.21914/ANZIAMJ.V55I0.7888)
  combinan una regla de orden no negativa con una discretización L1 implícita en
  un problema de difusión.
- [Lin y Xu (2007)](https://doi.org/10.1016/j.jcp.2007.02.001) sustentan el
  análisis L1 bajo regularidad suficiente.
- [Yin et al. (2021)](https://doi.org/10.3934/dcdsb.2020168) muestran por qué la
  regularidad y las correcciones de arranque importan en CQ de orden distribuido;
  se conserva como frontera futura, no como validación del Picard HAFO.

La cuadratura explícita, L1 y la recurrencia agregada son consistentes con esas
familias publicadas. El corrector vectorial Picard, la rama exacta
`alpha=1` y la optimización del kernel combinado son adaptaciones declaradas de
HAFO y no heredan automáticamente sus teoremas de estabilidad o convergencia.
