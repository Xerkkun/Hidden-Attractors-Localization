# Resumen: Búsqueda de Atractores mediante Función Descriptiva Sesgada

Este informe resume los hallazgos clave de la búsqueda sistemática de semillas asimétricas (sesgadas) en el sistema de Chua fraccionario no suave con saturación ($q = 0.9998$).

## Hallazgos Clave

1. **Localización de Semillas Sesgadas:** La función descriptiva sesgada (BDF) logró localizar de manera exitosa semillas con sesgos DC significativos ($c \approx \pm 2.0$ a $\pm 2.8$ y amplitudes $A \approx 3.0$ a $4.6$).
2. **Supervivencia y Estabilidad:**
   * Las semillas sesgadas con sesgo no nulo **solo sobrevivieron** la continuación Caputo para $m_1 = -1.1468$.
   * Para $m_1 \le -1.20$, todas las órbitas sesgadas resultaron numéricamente inestables y divergen rápidamente durante la continuación Caputo, sugiriendo la ausencia de atractores sesgados estables en esta región.
3. **Clasificación Dinámica:** Ninguna de las ramas sesgadas genuinas convergió a un atractor caótico o no periódico. Todas las simulaciones estables post-transientes para estas ramas terminaron en **ciclos límite periódicos regulares** (`biased_regular_periodic_rejected`).
4. **Comparación con Atractores Centrados:**
   * Se identificó un caso de **coincidencia de atractor** en $m_1 = -1.1468, m_0 = -0.20$ donde la semilla sesgada convergió exactamente al mismo ciclo límite centrado (distancia de centroides = 0.02).
   * En otros casos estables ($m_0 = -0.1768$ y $m_0 = -0.24$), las semillas sesgadas convergieron a ciclos límite asimétricos distintos de la órbita centrada de referencia.

## Tabla Resumen de Candidatos Sesgados ($|c| > 0.05$)

| Parámetros ($m_1$, $m_0$) | Semilla ($A, c, \omega$) | Estatus de Continuación | Estatus de Simulación | Clasificación Dinámica | ¿Mismo Atractor que el Centrado? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **(-1.1468, -0.1768)** | $A=4.578, c=-2.776, \omega=2.040$ | `ok` | `ok` | Ciclo Límite Periódico | No (distancia = 3.646) |
| **(-1.1468, -0.20)** | $A=4.470, c=-2.705, \omega=2.040$ | `ok` | `ok` | Ciclo Límite Periódico | Sí (distancia = 0.020) |
| **(-1.1468, -0.24)** | $A=4.284, c=-2.581, \omega=2.040$ | `ok` | `ok` | Ciclo Límite Periódico | No (distancia = 3.076) |
| **(-1.20, -0.1768)** | $A=3.636, c=2.338, \omega=2.040$ | `failed` | - | Divergencia | n/a |
| **(-1.20, -0.20)** | $A=3.556, c=2.278, \omega=2.040$ | `failed` | - | Divergencia | n/a |
| **(-1.20, -0.24)** | $A=3.418, c=-2.176, \omega=2.040$ | `failed` | - | Divergencia | n/a |
| **(-1.25, -0.1768)** | $A=3.109, c=-2.021, \omega=2.040$ | `failed` | - | Divergencia | n/a |
| **(-1.25, -0.20)** | $A=3.044, c=1.970, \omega=2.040$ | `failed` | - | Divergencia | n/a |
| **(-1.25, -0.24)** | $A=2.934, c=-1.882, \omega=2.040$ | `failed` | - | Divergencia | n/a |

> [!IMPORTANT]
> **Conclusión Metodológica:**
> Las semillas asimétricas/sesgadas localizadas teóricamente mediante BDF no revelan nuevos candidatos a atractores caóticos (ocultos o autoexcitados) en esta grilla. El comportamiento caótico observado se limita exclusivamente a las ramas centradas clásicas ($c \approx 0.0$).
>
> **Nota de Seguridad:** Estos análisis no constituyen pruebas de ocultedad para ningún atractor.
