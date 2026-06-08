# Resumen: Búsqueda Corregida de Atractores mediante Función Descriptiva Sesgada

Este informe resume los hallazgos clave de la búsqueda sistemática corregida de semillas asimétricas (sesgadas) en el sistema de Chua fraccionario no suave con saturación ($q = 0.9998$, $h = 0.01$), implementando la auditoría de signo armónico y la homotopía afín preservadora del bias DC.

> [!WARNING]
> **ADVERTENCIA CRÍTICA:**
> Estos resultados **no prueban ocultedad**. La detección de una semilla o su supervivencia en la continuación numérica no garantiza que el atractor final sea oculto. Se requieren pruebas rigurosas de vecindades de equilibrios para clasificar formalmente un atractor como oculto.

---

## 1. Modificaciones Metodológicas Clave

1. **Corrección de Signo Armónico:**
   Se auditó la función de transferencia lineal oficial de la librería, estableciendo que la condición armónica correcta es $1 + W_q(\omega) N_1(A,c) = 0$ (convención *plus*), con $W_q$ calculado como $r^T ( (j\omega)^q I - P )^{-1} b$. La convención anterior utilizaba un signo negativo ($1 - W_q N_1 = 0$) debido a una inversión de signo en el cálculo manual de la respuesta lineal.
2. **Homotopía Afín de Weyl-Caputo:**
   Para resolver la inconsistencia DC en la continuación de Caputo, se reemplazó la homotopía lineal estándar por una homotopía afín que mantiene el estado DC $x_{bar}$ y la ganancia efectiva del primer armónico $k_{eff} = N_1$ invariantes para $\eta = 0$:
   $$f_\eta(X) = P_{aff} X + const_{aff} + \eta b \left[ \psi(r^T X) - \psi_0(A, c) - k_{eff}(r^T X - c) \right]$$
   donde:
   * $P_{aff} = P + k_{eff} b r^T$
   * $const_{aff} = b \left[ \psi_0 - k_{eff} c \right]$
   
   Esta formulación simplifica exactamente al campo vectorial de Chua original para $\eta = 1$ (error de identidad algebraico $< 10^{-14}$).

---

## 2. Hallazgos Clave

1. **Impacto en la Continuación Caputo:**
   - **Caso $m_1 = -1.20, m_0 = -0.1768$ (Rama 0):** Con la homotopía afín corregida, la semilla centrada **logró converger exitosamente** (`ok`), recuperando una órbita periódica que antes fallaba por inconsistencia en la deformación.
   - **Caso $m_1 = -1.20, m_0 = -0.1768$ (Rama 1):** La órbita asimétrica sesgada ($c = 2.338$) **divergió correctamente** (`failed`), mostrando que la convergencia del run anterior era un artefacto debido a que la homotopía incompleta anuló incorrectamente el bias al inicio del camino.
2. **Inestabilidad del Sesgo en Alta Pendiente ($m_1 \le -1.20$):**
   Todas las semillas genuinamente sesgadas ($c \approx \pm 2.0$) divergen bajo la continuación afín de Caputo para regímenes de alta disipación exterior ($m_1 \le -1.20$).
3. **Persistencia de Órbitas Caóticas:**
   Las únicas órbitas caóticas estables corresponden a las ramas centradas clásicas ($c \approx 0$) para $m_1 = -1.20, m_0 = -0.20$ y $m_1 = -1.20, m_0 = -0.24$.

---

## 3. Tabla de Candidatos Sesgados ($|c| > 0.05$) y Comparación

| Parámetros ($m_1$, $m_0$) | Semilla ($A, c, \omega$) | Estatus Anterior | Estatus Corregido | Clasificación Anterior | Clasificación Corregida | ¿Cambio por Homotopía? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **(-1.1468, -0.1768)** | $A=4.578, c=2.776, \omega=2.040$ | `ok` | `ok` | Ciclo Periódico | Ciclo Periódico | Sí (Homotopía afín) |
| **(-1.1468, -0.2000)** | $A=4.470, c=-2.705, \omega=2.040$ | `ok` | `ok` | Ciclo Periódico | Ciclo Periódico | Sí (Homotopía afín) |
| **(-1.1468, -0.2400)** | $A=4.284, c=-2.581, \omega=2.040$ | `ok` | `ok` | Ciclo Periódico | Ciclo Periódico | Sí (Homotopía afín) |
| **(-1.2000, -0.1768)** | $A=3.636, c=2.338, \omega=2.040$ | `ok` | `failed` | Ciclo Periódico | Divergencia | **Sí (Divergió correctamente)** |
| **(-1.2000, -0.2000)** | $A=3.556, c=2.278, \omega=2.040$ | `failed` | `failed` | Divergencia | Divergencia | No |
| **(-1.2000, -0.2400)** | $A=3.418, c=-2.176, \omega=2.040$ | `failed` | `failed` | Divergencia | Divergencia | No |
| **(-1.2500, -0.1768)** | $A=3.109, c=\pm 2.021, \omega=2.040$ | `failed` | `failed` | Divergencia | Divergencia | No |
| **(-1.2500, -0.2000)** | $A=3.044, c=\pm 1.970, \omega=2.040$ | `failed` | `failed` | Divergencia | Divergencia | No |
| **(-1.2500, -0.2400)** | $A=2.934, c=1.882, \omega=2.040$ | `failed` | `failed` | Divergencia | Divergencia | No |

---

## 4. Conclusión Metodológica

La introducción de la homotopía afín y la estandarización del signo armónico corrigen los artefactos de convergencia espuria y las fallas artificiales en la continuación Caputo de bias. Las semillas sesgadas asimétricas teóricas no dan origen a atractores estables en la región $m_1 \le -1.20$, confirmando la robustez y dominancia de la dinámica caótica centrada clásica.
