# Reproducción de Casos Publicados (`published_case_reproduction`)

Este directorio contiene las configuraciones y herramientas para validar y reproducir tres sistemas caóticos clásicos de la literatura científica:

---

## 1. Tabla de Casos y Datos Bibliográficos Oficiales

| Identificador del Caso | Autores | Título | Revista / Referencia | Año | Volumen / Art. No. | DOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `kuznetsov2017_chua_integer` | N. V. Kuznetsov, O. A. Kuznetsova, G. A. Leonov, T. N. Mokaev, N. V. Stankevich | Localization of hidden Chua attractors by the describing function method | *IFAC-PapersOnLine* (arXiv:1705.02311) | 2017 | 50 (1), 11956-11961 | [10.1016/j.ifacol.2017.08.1633](https://doi.org/10.1016/j.ifacol.2017.08.1633) |
| `danca2017_chua_fractional_saturation` | Marius-F. Danca | Hidden chaotic attractors in fractional-order systems | *Nonlinear Dynamics* | 2017 | 89, 577–586 | [10.1007/s11071-017-3472-7](https://doi.org/10.1007/s11071-017-3472-7) |
| `wu2023_chua_fractional_arctan` | Xianming Wu, Longxiang Fu, Shaobo He, Zhao Yao, Huihai Wang, Jiayu Han | Hidden attractors in a new fractional-order Chua system with arctan nonlinearity and its DSP implementation | *Results in Physics* | 2023 | 52, 106866 | [10.1016/j.rinp.2023.106866](https://doi.org/10.1016/j.rinp.2023.106866) |

---

## 2. Diferencia entre Reproducción Publicada y Extensión Fraccionaria

Existen dos maneras en que la función de transferencia lineal interviene en el balance armónico (describing function):

### A. Modo Reproducción Publicada (`published_integer_laplace`)
Es el método usado por los artículos originales. Consiste en emplear la transferencia clásica de Laplace ($q=1$) para localizar $\omega_0, k, a_0$ y el estado de la semilla inicial, incluso cuando la dinámica temporal final sea fraccionaria:
$$W_{pub}(j\omega) = r^T (j\omega I - P)^{-1} b$$
En este modo:
- $z = j\omega$
- $q_{seed} = 1.0$
- Los coeficientes y el estado inicial de la semilla **no dependen** de la dinámica fraccionaria $q$.
- La semilla se construye usando la **fórmula cerrada entera**, sin autovectores.

### B. Modo Extensión Espectral Fraccionaria (`fractional_spectral`)
Es una extensión propia de esta librería. Aquí, el balance armónico se evalúa directamente sobre la transferencia generalizada en orden fraccionario $q$:
$$W_q(j\omega) = r^T ((j\omega)^q I - P)^{-1} b$$
En este modo:
- $z = (j\omega)^q = \omega^q e^{j q \pi / 2}$
- Los coeficientes y el estado inicial de la semilla **sí dependen** de $q$.

---

## 3. Rol de $q$ en la Dinámica de Caputo

Para Danca 2017 ($q=0.9998$) y Wu 2023 ($q=0.99$):
- La fase de balance armónico usa la transferencia de Laplace clásica (modo `published_integer_laplace`).
- El orden $q$ interviene exclusivamente en la integración causal Caputo:
$${}^C D_t^q X = P X + b \psi(r^T X)$$

---

## 4. Lista de Valores Faltantes en Artículos (`missing_values`)

Si un artículo no reporta un dato, este se registra como `null` en la configuración y se declara como faltante, resultando en una reproducción parcial honesta:

* **Danca 2017**:
  - `omega0` (no reportado en el texto)
  - `k` (no reportado)
  - `a0` (no reportado)
  - `seed_plus` y `seed_minus` (no reportados)
  - `initial_conditions_from_paper` (la condición inicial exacta para el atractor oculto de la Fig. 3 no se publica).
* **Wu 2023**:
  - `omega0` (no reportado)
  - `k` (no reportado)
  - `a0` (no reportado)
  - `seed_plus` y `seed_minus` (no reportados)

---

## 5. Control de Integración Dinámica (`run_dynamics`)

Por defecto, la validación se enfoca en la consistencia algebraica de transferencia y semillas para evitar simulaciones de Caputo excesivamente pesadas durante pruebas rápidas.

* **Ejecución básica (sin simulación pesada):**
  ```bash
  python validation/python/run_published_reproduction.py --all
  ```
  *(Las trayectorias se omitirán de forma segura, y `dynamics_reproduction.json` indicará `status: "skipped"`).*

* **Ejecución completa (incluyendo trayectorias dinámicas):**
  ```bash
  python validation/python/run_published_reproduction.py --all --run-dynamics
  ```

---

## 6. Advertencia Metodológica de Ocultedad

> [!WARNING]
> Esta fase de reproducción **no certifica** la propiedad de atractor oculto (`hidden_verified` no aparece en los outputs). Para clasificar un atractor como verificado oculto se requiere evaluar vecindades de tamaño $\delta$ alrededor de todos los equilibrios e investigar las cuencas de atracción bajo el protocolo dinámico robusto.
