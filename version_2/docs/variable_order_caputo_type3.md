# Caputo de orden variable Tipo III: especificación L1/PECE para HAFO

Fecha de la especificación: 3 de agosto de 2026. Estado de implementación:
**`implemented`, ejecutable y validada de forma enfocada**. La estabilidad de
la API continúa siendo **`experimental`**. HAFO dispone de la
función pública `integrate_variable_order_caputo_type3_l1` y del despacho por
`FractionalProblem`. Este estado acredita el contrato de software y casos
numéricos finitos; no transfiere automáticamente a sistemas ODE caóticos no
lineales las propiedades demostradas para ecuaciones de difusión.

SciSpace se utilizó como índice de descubrimiento. Las fórmulas que gobiernan
esta especificación se contrastaron con artículos primarios, manuscritos de
autor y páginas editoriales. Un resumen generado por el índice no sustituye la
ecuación publicada ni autoriza a trasladar pesos entre definiciones distintas
de orden variable.

## Decisión de alcance

La implementación actual es un solver de trayectoria con estas restricciones
explícitas:

- derivada izquierda de Caputo de orden variable **Tipo III** de
  Tavares--Almeida--Torres;
- orden escalar conmensurado aplicado a todos los componentes;
- perfiles prescritos con firma `alpha(t)`, `alpha(t, initial_state)` o
  `alpha(t, initial_state, parameters)`;
- el argumento de estado siempre es una copia fija del estado inicial: ninguna
  de esas firmas convierte el orden en `alpha(t, current_state)`;
- `0 < alpha_min <= alpha(t) <= alpha_max < 1`;
- terminal inferior igual al inicio de la integración;
- malla física uniforme e historia completa;
- historia L1 directa como método implementado y oráculo numérico;
- corrector Picard iterado hasta tolerancia como ruta reproducible;
- acumulación determinista en `float64` y Numba con `fastmath=False`.

Los tokens ejecutables son `derivative="caputo_variable_type3"` y
`method="vo_caputo_type3_l1"`. El registro genérico
`variable_order_caputo`/`variable_order_pece` permanece planificado y no es un
alias ejecutable: así se evita ocultar la convención Tipo III.

No se incluyen órdenes por componente, dependencia respecto al estado actual,
pasos adaptativos, memoria truncada ni el extremo entero `alpha=1`. Esas
extensiones cambian el contrato o requieren análisis separado.

## Definiciones Tipo I, II y III

Sea `a` el terminal inferior, `x` una función suficientemente regular y
`0 < alpha(t) < 1`. Tavares, Almeida y Torres distinguen tres operadores que
coinciden cuando `alpha` es constante, pero no cuando `alpha'(t)` es distinta
de cero.

### Tipo I

$$
{}^{C,I}_{a}D_t^{\alpha(t)}x(t)
=\frac{1}{\Gamma(1-\alpha(t))}\frac{d}{dt}
\int_a^t (t-s)^{-\alpha(t)}[x(s)-x(a)]\,ds.
$$

La derivada exterior actúa sobre la integral, pero no sobre el factor
`1/Gamma(1-alpha(t))`.

### Tipo II

$$
{}^{C,II}_{a}D_t^{\alpha(t)}x(t)
=\frac{d}{dt}\left[
\frac{1}{\Gamma(1-\alpha(t))}
\int_a^t (t-s)^{-\alpha(t)}[x(s)-x(a)]\,ds
\right].
$$

Aquí la derivada exterior también actúa sobre el factor Gamma dependiente del
tiempo.

### Tipo III

$$
{}^{C,III}_{a}D_t^{\alpha(t)}x(t)
=\frac{1}{\Gamma(1-\alpha(t))}
\int_a^t (t-s)^{-\alpha(t)}x'(s)\,ds.
$$

Esta es la definición seleccionada para HAFO. En cada tiempo de salida `t`, el
valor actual `alpha(t)` se congela en **todo** el núcleo que integra la historia.
No es el operador alternativo que emplea `alpha(s)` dentro de la integral. Para
`alpha'(t) != 0`, las relaciones entre los Tipos I, II y III contienen términos
adicionales con `alpha'(t)`, logaritmos y, según la forma, la función digamma;
por eso sus discretizaciones no son intercambiables.

La fuente normativa para estas tres definiciones, su diferencia y las fórmulas
exactas sobre potencias es Tavares--Almeida--Torres,
[DOI 10.1016/j.cnsns.2015.10.027](https://doi.org/10.1016/j.cnsns.2015.10.027),
con [manuscrito primario accesible](https://arxiv.org/pdf/1511.02017).

## Problema de valor inicial objetivo

Para un estado `u(t)` en `R^d`, la primera versión resolverá, componente a
componente con un único orden escalar,

$$
{}^{C,III}_{a}D_t^{\alpha(t)}u(t)=f(t,u(t)),
\qquad t\in(a,T],
\qquad u(a)=u_0.
$$

La ecuación se impone para `t>a`. Esta precisión importa cuando la solución
tiene una singularidad débil en el origen y el límite del lado derecho no es
cero.

## Derivación L1 en malla uniforme

Sea

$$
t_n=a+nh,\qquad U_n\approx u(t_n),\qquad
\alpha_n=\alpha(t_n).
$$

En cada intervalo `[t_j,t_{j+1}]`, L1 sustituye `u'(s)` por la pendiente
lineal

$$
u'(s)\approx\frac{U_{j+1}-U_j}{h}.
$$

Al integrar exactamente el núcleo con `alpha_n` congelado se obtiene

$$
\begin{aligned}
{}^{C,III}_{a}D_{t_n}^{\alpha_n}u(t_n)
&\approx
\frac{1}{\Gamma(1-\alpha_n)}
\sum_{j=0}^{n-1}\frac{U_{j+1}-U_j}{h}
\int_{t_j}^{t_{j+1}}(t_n-s)^{-\alpha_n}\,ds \\
&=
\frac{h^{-\alpha_n}}{\Gamma(2-\alpha_n)}
\sum_{j=0}^{n-1}\omega_{n-j-1}^{(n)}
(U_{j+1}-U_j),
\end{aligned}
$$

donde

$$
\omega_k^{(n)}
=(k+1)^{1-\alpha_n}-k^{1-\alpha_n},
\qquad k\geq 0.
$$

Así, el operador discreto de referencia es

$$
D_{L1,n}^{\alpha_n}U=
\frac{h^{-\alpha_n}}{\Gamma(2-\alpha_n)}
\sum_{j=0}^{n-1}\omega_{n-j-1}^{(n)}
(U_{j+1}-U_j).
$$

Los pesos dependen de `n` mediante `alpha_n`; la historia no es una convolución
estacionaria. No se puede reutilizar sin prueba una FFT de orden constante ni
una caché única de pesos.

La fórmula anterior coincide con la forma L1 de orden evaluado en el tiempo de
salida usada por Fang--Sun--Wang,
[DOI 10.1016/j.camwa.2020.07.009](https://doi.org/10.1016/j.camwa.2020.07.009),
y por Zaky et al. en un problema no lineal de difusión-reacción,
[DOI 10.1016/j.cam.2022.114832](https://doi.org/10.1016/j.cam.2022.114832),
con [manuscrito de autor abierto](https://backoffice.biblio.ugent.be/download/01GP0NDAYERJH7A88PNDXJGKBK/01GP0NPQ7Y324TKAH15P6J80DX).

### Separación de la incógnita actual

Como `omega_0^(n)=1`, se define

$$
H_n=
\sum_{j=0}^{n-2}\omega_{n-j-1}^{(n)}(U_{j+1}-U_j),
$$

con suma vacía para `n=1`, y

$$
s_n=\Gamma(2-\alpha_n)h^{\alpha_n},
\qquad B_n=U_{n-1}-H_n.
$$

La ecuación L1 no lineal queda

$$
U_n=B_n+s_n f(t_n,U_n).
$$

Esta forma es el núcleo implementable: la historia `H_n` se calcula una sola
vez por paso y el corrector sólo resuelve una ecuación algebraica de dimensión
`d`.

## PECE como adaptación HAFO de la ecuación L1

La ruta implementada no presupone una identidad de inversión de orden variable.
Aplica predictor--evaluador--corrector--evaluador directamente al punto fijo de
la ecuación L1.

Para el paso `n`:

1. **P**, predictor determinista:

   $$
   Z_n^{(0)}=B_n+s_n f(t_n,U_{n-1}).
   $$

2. **E**, evaluar `f(t_n,Z_n^(m))`.
3. **C**, corregir:

   $$
   Z_n^{(m+1)}=B_n+s_n f(t_n,Z_n^{(m)}).
   $$

4. **E**, evaluar el residuo del nuevo estado:

   $$
   R_n^{(m+1)}=
   Z_n^{(m+1)}-B_n-s_n f(t_n,Z_n^{(m+1)}).
   $$

5. Repetir C--E hasta que

   $$
   \lVert R_n^{(m+1)}\rVert
   \leq \mathrm{atol}+\mathrm{rtol}\lVert Z_n^{(m+1)}\rVert.
   $$

Una sola corrección no equivale a resolver exactamente la ecuación L1
implícita. La ruta implementada usa `PEC^mE` hasta tolerancia y registra por
paso el número de iteraciones y el residuo final.

Si `f(t,.)` es Lipschitz con constante `L`, una condición suficiente para que
Picard sea contractivo en la norma elegida es

$$
s_nL<1.
$$

Esta condición y el envoltorio Picard son una **adaptación HAFO**, no un
resultado atribuido a los artículos L1. La política implementada es explícita:

- `on_nonconvergence="raise"` lanza `VariableOrderCorrectorError`;
- `on_nonconvergence="return"` devuelve la trayectoria aceptada hasta el paso
  anterior, `status="corrector_nonconvergence"` y el índice fallido en
  `solver_info["nonconverged_step"]`.

Newton y la reducción local del paso no están implementados en esta ruta.
Reducir `h` exige reiniciar la integración uniforme; un rechazo local requerirá
primero un contrato de malla no uniforme y pesos coherentes.

Zerari--Odibat--Shawagfeh,
[DOI 10.1002/mma.9613](https://doi.org/10.1002/mma.9613), anuncian una fórmula
de inversión equivalente y un predictor--corrector para derivadas
Liouville--Caputo de orden variable. El material accesible durante esta revisión
no permitió comprobar ecuación por ecuación que su operador sea exactamente el
Tipo III anterior. Sus pesos no son normativos para este núcleo hasta completar
esa verificación.

## Pesos y acumulación numéricamente estables

La resta directa

```python
(k + 1.0) ** p - k ** p
```

pierde cifras para `k` grande o `p=1-alpha_n` pequeño. La función pública
`variable_order_l1_weight` y el kernel Numba usan una estrategia híbrida:
conservan la fórmula directa para retardos de hasta 1024 y cambian después a

```python
def stable_large_lag_weight(k: int, alpha_n: float) -> float:
    if k == 0:
        return 1.0
    p = 1.0 - alpha_n
    return k**p * expm1(p * log1p(1.0 / k))
```

El factor de paso actual se evalúa como

```python
s_n = gamma(2.0 - alpha_n) * h**alpha_n
```

Una variante en espacio logarítmico permanece como posible robustecimiento
para escalas extremas; no describe el código actual.

Contrato de reproducibilidad implementado:

- bucle histórico con orden fijo de acumulación;
- `float64` y arreglos C-contiguos;
- `fastmath=False` en el núcleo Numba;
- no paralelizar internamente una reducción si cambia el orden de suma;
- validar `alpha_n` y finitud antes de calcular Gamma o pesos;
- no usar la fórmula límite `alpha=1` sin una rama entera explícita.

`_history_sum_numba` recorre la historia de forma serial, calcula cada peso en
el bucle y no materializa una matriz triangular. Numba compila esta suma sin
`fastmath`; el RHS y el corrector Picard permanecen en la capa Python. La suma
compensada y el paralelismo entre trayectorias son extensiones futuras, no
propiedades atribuidas al backend actual.

## Regularidad y compatibilidad inicial

Tavares--Almeida--Torres prueban que, si `u` pertenece a `C^1`, las tres
derivadas Caputo de orden variable consideradas tienden a cero cuando
`t -> a+`. Por tanto, una solución clásica que haga continua la ecuación hasta
el terminal necesita

$$
f(a,u_0)=0.
$$

Esta es una condición de compatibilidad para la clase suave, no una razón para
rechazar toda semilla con `f(a,u0) != 0`. Como en problemas Caputo clásicos,
puede existir una solución continua con arranque débilmente singular,
`u(t)-u0 = O((t-a)^sigma)`, y la ecuación se impone para `t>a`.

La API distingue:

- `initial_regularity="smooth"`: el usuario afirma regularidad suficiente; si
  `||f(a,u0)||` excede `compatibility_tolerance`, se emite
  `VariableOrderInitialCompatibilityWarning` sin rechazar automáticamente la
  integración;
- `initial_regularity="nonsmooth"`: se admite arranque singular y no se promete
  el orden suave en malla uniforme;
- `initial_regularity="weak"`: alias aceptado que se normaliza y registra como
  `nonsmooth`;
- `initial_regularity="unknown"`: se conserva el resultado y se registra la
  compatibilidad observada, sin inferir regularidad.

Para `u` suficientemente suave, en particular bajo la hipótesis `C^2` usada en
los análisis L1 citados, el error de consistencia de la derivada es

$$
O(h^{2-\alpha_n}).
$$

Sobre todo el intervalo, la tasa conservadora asociada a esta cota local es
`2-alpha_max`. Esto es una afirmación sobre la aproximación L1 de la derivada,
no un teorema ya verificado de error global para el solver HAFO en ODE caóticas
no lineales. Los arranques singulares pueden reducir el orden observado y
motivan una futura malla graduada.

El trabajo de Wang--Zheng,
[DOI 10.1007/s10444-019-09690-0](https://doi.org/10.1007/s10444-019-09690-0),
analiza buena formulación, regularidad débil y mallas graduadas para una
ecuación no lineal de orden variable. No se traslada aquí su teorema: un
manuscrito relacionado accesible de Zheng,
[arXiv:2110.04707](https://arxiv.org/abs/2110.04707), usa `alpha(s)` dentro de la
historia. Antes de usar cualquier tasa de esa línea debe verificarse que la
definición del artículo concreto coincide con el Tipo III de tiempo actual.

La ausencia general de ley de semigrupo para integrales de orden variable y la
necesidad de definir con cuidado la solución del IVP se discuten en
[*On definition of solution of initial value problem for fractional differential
equation of variable order*](https://www.aimspress.com/article/doi/10.3934/math.2021401),
[DOI 10.3934/math.2021401](https://doi.org/10.3934/math.2021401). Esta es la
razón para resolver directamente la ecuación L1 y no asumir sin demostración
`I^(alpha(t)) D^(alpha(t)) u = u-u0`.

## Complejidad y ruta de optimización

Para `N` pasos y dimensión `d`, el L1 directo de historia completa requiere:

- `O(N^2)` evaluaciones escalares de pesos/historia;
- trabajo vectorial `O(N^2 d)` al acumular los `d` componentes;
- almacenamiento `O(Nd)` para la trayectoria y sus incrementos;
- `O(sum_n m_n C_f)` adicional para `m_n` evaluaciones del RHS por corrector;
- memoria auxiliar `O(d)` para la suma histórica en flujo.

La historia `H_n` no cambia durante las iteraciones del corrector y se calcula
una sola vez por paso. `use_acceleration=True` selecciona la suma histórica L1
directa compilada con Numba; no selecciona un algoritmo de memoria rápida ni
cambia las complejidades anteriores. Si Numba falla y
`allow_python_fallback=True`, se registra el error y se continúa con la suma
Python/NumPy; con `False`, el fallo se propaga.

Fang--Sun--Wang publican una aproximación rápida mediante bloques binarios
desplazados y polinomios que reduce, para su algoritmo, el trabajo directo a
aproximadamente `O(r N log N)` y la memoria auxiliar a `O(r log N)`. Esa ruta
debe añadirse después como backend distinto, con tolerancia y grado `r`
registrados, y compararse contra el oráculo L1 directo. El resultado publicado
para una derivada y ecuaciones de difusión no demuestra por sí solo estabilidad
de trayectorias caóticas HAFO.

Memoria finita, reinicios de bloques o *short memory* cambian la definición
causal efectiva y no forman parte de la implementación actual.

## API implementada

### Función directa

La función pública acepta el perfil mediante `order_function`. Las tres firmas
reconocidas son:

```python
def alpha(time): ...
def alpha(time, initial_state): ...
def alpha(time, initial_state, parameters): ...
```

La tercera firma se selecciona cuando se proporcionan `parameters`. En las dos
firmas con estado, la función recibe en cada evaluación una copia separada del
**estado inicial**, nunca el estado que evoluciona. El perfil continúa siendo
prescrito en el tiempo.

```python
from hidden_attractors.fractional import (
    integrate_variable_order_caputo_type3_l1,
)


def alpha(time, initial_state, parameters):
    del initial_state
    return parameters["alpha0"] + parameters["slope"] * time


result = integrate_variable_order_caputo_type3_l1(
    rhs=rhs,
    initial_state=u0,
    parameters=params,
    step=1.0e-3,
    n_steps=100_000,
    lower_terminal=0.0,
    order_function=alpha,
    order_function_name="affine-alpha",
    corrector_atol=1.0e-12,
    corrector_rtol=1.0e-10,
    corrector_max_iterations=50,
    on_nonconvergence="raise",       # "raise" o "return"
    initial_regularity="unknown",    # "smooth", "nonsmooth" o "unknown"
    compatibility_tolerance=1.0e-10,
    use_acceleration=True,            # suma L1 directa con Numba
    allow_python_fallback=True,
    divergence_norm=120.0,
)
```

`weak` se acepta como alias de `nonsmooth`. La política de memoria no es un
argumento seleccionable: esta función implementa únicamente `full_history`.
El resultado `VariableOrderCaputoType3Result` contiene tiempos, estados, órdenes
evaluados, iteraciones y residuos del corrector, backend, estado de terminación,
metadatos del solver y alcance
`scope="finite_numerical_trajectory_only"`.

Los estados de terminación actualmente posibles son `ok`,
`corrector_nonconvergence`, `nonfinite_solution` y `diverged`. El segundo sólo
se devuelve con `on_nonconvergence="return"`; la política `raise` lanza la
excepción antes de construir el resultado.

### Despacho por `FractionalProblem`

La misma ruta está integrada en el contrato general:

```python
from hidden_attractors.fractional import FractionalProblem, solve_fractional_problem

alpha_at_a = alpha(0.0, u0, params)

problem = FractionalProblem(
    derivative="caputo_variable_type3",
    method="vo_caputo_type3_l1",
    orders=alpha_at_a,
    initial_state=u0,
    step=1.0e-3,
    t_span=(0.0, 100.0),
    lower_terminal=0.0,
    memory_policy="full_history",
    kernel_parameters={
        "order_function": alpha,
        "order_function_name": "affine-alpha",
    },
    method_options={
        "corrector_atol": 1.0e-12,
        "corrector_rtol": 1.0e-10,
        "corrector_max_iterations": 50,
        "on_nonconvergence": "raise",
        "initial_regularity": "unknown",
        "compatibility_tolerance": 1.0e-10,
    },
    allow_experimental=True,
)

dispatched = solve_fractional_problem(
    problem,
    rhs,
    parameters=params,
    use_acceleration=True,
    allow_python_fallback=True,
    divergence_norm=120.0,
)
```

El campo `orders` conserva un valor nominal conmensurado requerido por el
esquema actual de `FractionalProblem` y debe coincidir con
`alpha(lower_terminal)`; el despacho rechaza una discrepancia. No genera la
secuencia variable ni sustituye al callback: `order_function` gobierna todos
los valores realmente usados y el resultado registra su mínimo y máximo
observados.

Los metadatos efectivos incluyen:

- `definition="tavares_type_iii_current_time"`;
- `discretization="uniform_l1"` y `corrector="picard"`;
- `history_complexity="O(N^2)"`,
  `history_component_work="O(N^2*d)"` y
  `history_storage="O(N*d)"`;
- `order_min`, `order_max`, nombre y firma reconocida del callback;
- terminal, paso, muestras solicitadas/devueltas y política `full_history`;
- backend y si la historia usó Numba o realizó *fallback*;
- tolerancias, iteraciones, residuos y diagnósticos completos del eventual paso
  no convergido;
- regularidad y residuo de compatibilidad inicial;
- validación puntual de rango sobre toda la agenda solicitada, sin inferir
  regularidad de `alpha(t)` ni contractividad de Picard;
- `published_scope="type_iii_definition_and_l1_discretization"` y
  `hafo_adaptation="implicit_system_picard_corrector"`.

Una trayectoria obtenida con este solver no prueba caos, atracción ni
ocultedad. Los diagnósticos y el protocolo de vecindades siguen siendo etapas
separadas.

## Estado de implementación y pruebas enfocadas

El código ejecutable reside en
`hidden_attractors/fractional/variable_order_caputo_type3.py`, se exporta desde
`hidden_attractors.fractional` y se despacha en
`hidden_attractors/fractional/problem.py`. La batería específica está en
`tests/test_variable_order_caputo_type3_solver.py`.

La corrida enfocada reportada para esta integración terminó con **64 pruebas
directas aprobadas**. Otras **6 pruebas** cubren el artefacto/comparador Wolfram,
incluido el caso vivo disponible en el host. Cubre el contrato de software y
casos numéricos finitos, incluida la paridad Python--Numba, divergencia inicial
sin evaluar el RHS, exclusión de estados no finitos y fallback parcial; no constituye una
prueba de estabilidad o convergencia global para cualquier RHS ni evidencia de
dinámica caótica u ocultedad.

### Oráculo independiente Wolfram

El caso
`validation/wolfram/cases/variable_order_caputo_type3_l1.wl` no importa HAFO ni
lee el reporte. Deriva los pesos L1 mediante `Integrate`, verifica la identidad
de potencia tipo III, reconstruye una recurrencia manufacturada y comprueba la
reducción a L1 de orden constante. El comparador público reportó residuo
simbólico cero, diferencia máxima Wolfram--HAFO de
`1.5543122344752192e-15` y diferencia de trayectoria de
`2.220446049250313e-16`. Los errores finitos del operador (`2.0245592e-2`) y de
la recurrencia (`1.4766213e-2`) se conservan como diagnósticos de esa malla; no
son una tasa de convergencia ni evidencia de estabilidad o dinámica caótica.

## Pruebas manufacturadas y de regresión

### 1. Oráculo exacto para potencias

Para `p>0` y

$$
u_*(t)=u_0+c(t-a)^p,
$$

el Tipo III satisface exactamente

$$
{}^{C,III}_{a}D_t^{\alpha(t)}u_*(t)
=c\frac{\Gamma(p+1)}{\Gamma(p+1-\alpha(t))}
(t-a)^{p-\alpha(t)}.
$$

Este oráculo debe evaluarse en espacio logarítmico lejos del terminal. En
`t=a` se define el límite analítico correspondiente, sin evaluar `log(0)`.

### 2. Convergencia suave y RHS no lineal

Elegir un perfil suave y acotado, por ejemplo

$$
\alpha(t)=\alpha_0+\delta\frac{t-a}{T-a},
$$

con rango estrictamente contenido en `(0,1)`, y `p>=2`. Para un vector `c` y una
matriz `A`, fabricar

$$
f(t,u)=D_*^{III}(t)+A[u-u_*(t)].
$$

La solución exacta sigue siendo `u_*`. Debe escogerse `A` de modo que
`s_n ||A|| < 1` en la malla de prueba. Sobre muestras exactas, el error del
operador L1 debe exhibir la tendencia de consistencia compatible con
`2-alpha_max`. Para el error de estado del solver se registra la pendiente
observada; no se codifica como teorema un orden global aún no demostrado para
este contrato ODE.

### 3. Arranque débilmente singular

Usar

$$
u_*(t)=u_0+c(t-a)^\sigma,
\qquad 0<\sigma<2,
$$

incluido `sigma=alpha(a)`. La prueba debe aceptar una solución continua no `C^1`,
mostrar la degradación de la malla uniforme y comprobar que la compatibilidad
suave no se aplica como rechazo universal.

### 4. Semántica Tipo III

Con `alpha'(t) != 0`, comparar el operador contra el oráculo de potencia
anterior. El resultado Tipo III no contiene términos explícitos con
`alpha'(t)`, logaritmos o digamma. Esta prueba detecta la implementación
accidental de Tipo I o Tipo II.

### 5. Reducción a orden constante

Con `alpha(t) = alpha0`, pesos, trayectoria y residuos deben coincidir, salvo
redondeo, con el L1 constante que resuelve la misma ecuación discreta. No se
exige igualdad muestra a muestra con ABM/PECE, porque es otro método; ambos deben
converger al mismo problema continuo bajo refinamiento.

### 6. Perfiles y extremos permitidos

Cubrir perfiles constante, lineal y oscilatorio suave, además de valores
cercanos a `alpha_min` y `alpha_max`. Verificar en cada paso rango, finitud,
positividad y decrecimiento de los pesos. Un valor fuera de `(0,1)` debe fallar
antes de actualizar la historia.

### 7. Corrector no lineal

- caso contractivo con convergencia y residuo bajo tolerancia;
- caso deliberado sin convergencia antes de `max_iterations`;
- política `raise` reproducible;
- Newton sólo si se proporciona un Jacobiano consistente;
- una corrección etiquetada como aproximación PECE, no como L1 implícito
  convergido.

### 8. Estabilidad de pesos y paridad de backend

- comparar `expm1/log1p` con un oráculo de alta precisión para `k` grande;
- verificar continuidad al aproximarse a los límites admitidos sin incluir
  `alpha=1`;
- exigir paridad Python--Numba dentro de tolerancia declarada;
- repetir con reducción determinista y comprobar metadatos;
- validar que no se materialice una matriz triangular de pesos.

### 9. Puerta científica posterior

Sólo después de superar convergencia manufacturada, reducción constante,
paridad de backend y residuos del corrector se deben ejecutar Chua u otros
sistemas caóticos. Una órbita visualmente compleja no promociona el método ni
constituye evidencia de atractor oculto.

## Frontera de evidencia

### Respaldado por publicaciones primarias

- las definiciones Tipo I, II y III y su no equivalencia para orden variable;
- la fórmula exacta Tipo III para potencias;
- el límite cero en el terminal para funciones `C^1`;
- la fórmula L1 con el orden congelado en el tiempo de salida;
- la consistencia `O(h^(2-alpha_n))` bajo regularidad suave;
- algoritmos rápidos por bloques para la evaluación L1 en problemas de orden
  variable;
- la falta general de una ley de semigrupo que permita una inversión ingenua.

### Adaptación de diseño propia de HAFO

- escribir la ecuación implícita como `U_n = B_n + s_n f(t_n,U_n)`;
- usar PECE/Picard directamente sobre esa ecuación;
- exigir residuo, número máximo de iteraciones y política de fallo;
- usar pesos híbridos directos/`expm1`--`log1p` y una suma Numba determinista
  sin `fastmath`;
- aceptar callbacks de orden con contexto fijo del estado inicial sin convertir
  el problema en uno de orden dependiente de la trayectoria;
- los nombres de API, metadatos y advertencias de regularidad;
- el RHS manufacturado no lineal y las puertas de validación.

### No establecido todavía

- convergencia global del solver para un RHS caótico general bajo el Tipo III;
- estabilidad a tiempos largos para Chua u otros sistemas no suaves;
- equivalencia de los pesos de Zerari et al. con esta definición exacta;
- validez de teoremas basados en operadores con `alpha(s)` para este operador
  con `alpha(t_n)`;
- precisión de exponentes de Lyapunov, cuencas o clasificación de ocultedad;
- seguridad de una memoria truncada, reinicios o una aceleración rápida sin
  comparación contra el oráculo directo.

## Trazabilidad SciSpace y fuentes contrastadas

| ID SciSpace | Fuente localizada | Uso y límite en esta especificación |
|---|---|---|
| `hqcx7g1ci4j4` | Tavares--Almeida--Torres, [DOI 10.1016/j.cnsns.2015.10.027](https://doi.org/10.1016/j.cnsns.2015.10.027) | Fuente normativa para Tipos I/II/III, límite inicial y potencias. |
| `2dv0mwnhst` | Fang--Sun--Wang, [DOI 10.1016/j.camwa.2020.07.009](https://doi.org/10.1016/j.camwa.2020.07.009) | L1 de orden actual y aceleración por bloques; sus resultados PDE no se convierten en teoremas ODE caóticos. |
| `4fvmy874p2` | Zerari--Odibat--Shawagfeh, [DOI 10.1002/mma.9613](https://doi.org/10.1002/mma.9613) | Descubrimiento de PECE; equivalencia exacta con Tipo III pendiente. |
| `20uusysv6h` | Wang--Zheng, [DOI 10.1007/s10444-019-09690-0](https://doi.org/10.1007/s10444-019-09690-0) | Regularidad y mallas graduadas como lectura relacionada; convención exacta por verificar antes de reutilizar tasas. |
| `5a14pubj1f` | Zheng, [arXiv:2110.04707](https://arxiv.org/abs/2110.04707) | Evidencia explícita de una convención alternativa con `alpha(s)`; no normativa para Tipo III. |
| `36jlwwxhcy` | [DOI 10.3934/math.2021401](https://doi.org/10.3934/math.2021401) | Definición de solución y cautela por ausencia de semigrupo en orden variable. |

La búsqueda exacta no devolvió en SciSpace el artículo de Zaky et al. usado para
contrastar L1. Su DOI y su manuscrito se verificaron por la editorial y el
repositorio institucional. Esta ausencia se conserva como límite del índice,
no como ausencia de la publicación.

## Criterio de promoción

La puerta de implementación ejecutable a `implemented` ya se cumplió con:

1. implementación Python de referencia;
2. suma histórica Numba con paridad enfocada;
3. pruebas de pesos, potencia manufacturada, reducción a orden constante y
   arranque no suave;
4. residuos Picard auditables y fallo determinista;
5. metadatos que congelan definición, perfil de orden, malla y backend;
6. integración en `FractionalProblem` y exportación pública;
7. 64 pruebas directas enfocadas aprobadas en la corrida reportada;
8. oráculo Wolfram independiente y 6 pruebas de artefacto/comparación aprobadas.

Para promover la estabilidad de la API más allá de `experimental` todavía se
requiere un estudio de convergencia global específico para el IVP Tipo III, regularidad
inicial más amplia, horizontes largos, sistemas no suaves y comparaciones
independientes sobre mallas/familias adicionales. El algoritmo rápido de Fang
debe implementarse como backend
separado y superar una comparación frente al L1 directo en perfiles constantes,
suaves y cercanos a los extremos permitidos. El backend Numba actual acelera la
misma suma directa `O(N^2)` y no satisface por sí solo esa puerta de memoria
rápida.
