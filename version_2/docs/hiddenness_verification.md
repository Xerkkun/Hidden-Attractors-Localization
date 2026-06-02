# Verificación Estricta de Ocultedad Operacional (Strict Operational Hiddenness Verification)

Este módulo establece un protocolo estricto para validar la condición operacional de ocultedad en atractores caóticos de orden entero y fraccionario. 

En la literatura científica, un atractor $\mathcal{A}$ se clasifica como **oculto** (hidden attractor) si su cuenca de atracción $\mathcal{B}(\mathcal{A})$ no interseca la vecindad de ningún punto de equilibrio del sistema:

$$\mathcal{B}(\mathcal{A}) \cap \mathcal{U}_{\epsilon}(X_i^*) = \emptyset \quad \forall X_i^*$$

Si la cuenca interseca alguna vecindad equilibrio, el atractor es **auto-excitado** (self-excited).

---

## Limitación de Métodos Analíticos y Heurísticos

Métodos como el balance armónico (función descriptiva o DF), el criterio de Nyquist, la continuación de homoclínicas/bifurcaciones, o la simulación acotada de un solo seed **no son suficientes** para certificar un sistema como `hidden_verified`. 

* **Función Descriptiva / Nyquist:** Son aproximaciones lineales equivalentes locales (tipo Weyl) que sirven únicamente para **generar candidatos/semillas** (`seed_found`). No demuestran la no-intersección de trayectorias globales.
* **Continuación:** Demuestra la persistencia de una estructura oscilatoria al variar un parámetro de control $\eta$, pero no descarta contactos transitorios en la vecindad del equilibrio para el sistema final.
* **Diagnósticos de Caos:** Validan la naturaleza caótica (Lyapunov positivo, prueba 0-1) pero no su topología de cuencas de atracción.

Por lo tanto, la biblioteca restringe la etiqueta `hidden_verified` exclusivamente a los sistemas que aprueban el protocolo operacional completo de muestreo en vecindades de control.

---

## Protocolo de Muestreo de Vecindades (Sphere Probes)

Para verificar numéricamente la condición de ocultedad, se realiza un barrido de trayectorias integradas hacia atrás y adelante en esferas concéntricas alrededor de **todos** los puntos de equilibrio $X_i^*$.

### 1. Requisitos de Cobertura de Radios
Se exige por contrato ensayar un conjunto estricto de radios $\epsilon$ decrecientes para descartar cuencas delgadas:
$$\epsilon \in \{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$$

Cualquier omisión de estos radios restringe el veredicto a `HIDDEN_COMPATIBLE` (nunca `HIDDEN_VERIFIED`).

### 2. Condición de Dirección Unitaria Uniforme
Para cada muestra $j$ en el radio $\epsilon$, el punto inicial $x_0$ se genera en la esfera mediante una dirección unitaria $v_j$:
$$x_0 = X_i^* + \epsilon \cdot v_j$$

La biblioteca ejecuta una auditoría de tolerancia estricta sobre la norma de la dirección:
$$\|v_j\| = \left\| \frac{x_0 - X_i^*}{\epsilon} \right\| = 1.0 \pm 10^{-10}$$

Si alguna muestra excede esta tolerancia, se levanta un error de protocolo inmediato.

### 3. Criterio Local de Evaluación (criterio_local)
Para cada par $(X_i^*, \epsilon)$, se clasifica la vecindad localmente como:
* **PASS:** Cero trayectorias iniciadas en la esfera alcanzan la vecindad del atractor caótico objetivo.
* **FAIL:** Al menos una trayectoria interseca el atractor (contacto detectado, implicando que el equilibrio pertenece a la cuenca de atracción del atractor).
* **INCOMPLETE:** Ocurrió un fallo de integración numérica o no se completó el número requerido de muestras.

---

## Estados del Contrato de Ocultedad (HiddennessVerificationStatus)

El contrato formal evalúa la matriz de resultados y asigna uno de los siguientes estados jerárquicos:

1. **HIDDEN_VERIFIED:** 
   * Protocolo de vecindades completado para **todos** los equilibrios del sistema.
   * Todos los 4 radios exigidos fueron ensayados.
   * Cero contactos con el atractor objetivo (criterio local `PASS` en todas las muestras).
   * Cero fallos numéricos detectados (o permitidos bajo configuración explícita).
2. **HIDDEN_COMPATIBLE:** 
   * No se detectaron contactos con el atractor caótico.
   * Sin embargo, el protocolo está incompleto (ej. se ensayó solo un subconjunto de radios o solo equilibrios inestables).
3. **SELF_EXCITED_CONTACT_DETECTED:** 
   * Se detectó al menos un contacto entre las vecindades de los equilibrios y el atractor. El atractor no es oculto.
4. **NUMERICAL_FAILURE:** 
   * Ocurrieron fallos en el integrador (divergencia, NaN) durante el barrido de esferas y `allow_numerical_failures` está en `False`.
5. **CANDIDATE_NOT_AVAILABLE / SEED_NOT_AVAILABLE:**
   * La simulación del seed o la continuación fallaron en localizar el atractor de referencia.

---

## Nota Metodológica Importante (Danca Context)

> [!WARNING]
> La declaración de `hidden_verified` es estrictamente operacional. La ausencia de contactos numéricos bajo las tolerancias y radios especificados **no constituye una prueba matemática global de ocultedad**. Representa una verificación computacional exhaustiva bajo un contrato de discretización finito. Las dinámicas de orden fraccionario (especialmente con memoria larga) pueden presentar dinámicas transitorias extremadamente lentas que requieran tiempos de integración superiores a los límites computacionales del protocolo.
