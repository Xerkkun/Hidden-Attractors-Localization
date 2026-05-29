# Reproducción de Casos Publicados (`published_case_reproduction`)

Este directorio contiene las configuraciones y herramientas necesarias para validar y reproducir tres sistemas caóticos con atractores ocultos publicados en la literatura científica:

1. **Kuznetsov et al. 2017**: Chua entero con no linealidad de saturación (piecewise lineal).
2. **Danca 2017**: Chua fraccionario no suave con no linealidad de saturación.
3. **Wu et al. 2023**: Chua fraccionario con no linealidad de tipo arcotangente.

---

## 1. Diferencia entre Reproducción Publicada y Extensión Fraccionaria

Para el estudio de sistemas caóticos de orden fraccionario con estructura tipo Lur'e, existen dos metodologías para la generación de la semilla inicial basada en balance armónico:

### A. Modo Reproducción Publicada (`published_integer_laplace`)
Es el método clásico utilizado por los autores en los artículos publicados. Consiste en utilizar la función de transferencia del sistema entero asociado ($q=1$) para calcular la frecuencia de cruce $\omega_0$, la ganancia límite $k$, la amplitud de la función descriptiva $a_0$, y el estado de la semilla inicial $X_{seed}$, incluso si el sistema dinámico a integrar posteriormente es fraccionario.

Bajo este modo, se evalúa:
$$W_{pub}(j\omega) = r^T (j\omega I - P)^{-1} b$$

Los parámetros de la semilla ($\omega_0, k, a_0, S, X_{seed}$) **no dependen** del orden fraccionario $q$.

### B. Modo Extensión Fraccionaria (`fractional_spectral`)
Es la extensión matemática y conceptual propia de la librería para sistemas fraccionarios. En este modo se evalúa la función de transferencia espectral directamente sobre el orden fraccionario $q$:
$$W_q(j\omega) = r^T ((j\omega)^q I - P)^{-1} b$$
donde $z = (j\omega)^q = \omega^q e^{j q \pi / 2}$.

Bajo este modo, los parámetros de la semilla ($\omega_0, k, a_0, S, X_{seed}$) **sí dependen** directamente del orden fraccionario $q$.

---

## 2. El Rol de $q$ en la Dinámica de Caputo

Para los sistemas fraccionarios analizados en esta validación (Danca 2017 y Wu 2023):
- La **fase de semilla** se calcula usando la transferencia clásica de Laplace entera (modo `published_integer_laplace`).
- El orden fraccionario $q$ actúa exclusivamente sobre la **dinámica temporal** en la integración numérica de la derivada de Caputo:
$${}^C D_t^q X = P X + b \psi(r^T X)$$

---

## 3. Alcance de esta Validación: No Certifica `hidden_verified` por sí sola

> [!WARNING]
> La correcta reproducción de la fórmula de transferencia, de la semilla armónica, y de las trayectorias publicadas en esta fase de validación **no certifica** la propiedad de atractor oculto (`hidden_verified`).

La verificación rigurosa de que un atractor sea oculto (según la definición de Leonov--Kuznetsov) requiere:
1. Simulación dinámica del sistema desde vecindades (probes) de **todos** los puntos de equilibrio inestables (por ejemplo, en Chua, evaluando vecindades con radio $\delta$ alrededor de $E_0$, $E_+$ y $E_-$).
2. Verificación de que las trayectorias que nacen en dichas vecindades diverjan a infinito o se sientan atraídas por otros estados estables, y que ninguna de ellas converja al atractor candidato bajo el contrato numérico establecido.
3. Análisis sistemático del mapa de cuencas de atracción.
