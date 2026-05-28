# Reproducción ADM — Chua fraccionario arctan, Wu et al. 2023

**Ruta B del repositorio.**  Este documento describe el integrador ADM local
implementado en `src/integrators/adm_wu2023.py` y el modo de simulación directa
`simulate_attractor_only`.

---

## 1. Sistema objetivo

Para orden Caputo conmensurable **q = 0.99**:

```
D^q x = alpha*(y - x) - alpha*f(x)
D^q y = x - y + z
D^q z = -beta*y - gamma*z

f(x) = m*x + (n - m)*arctan(x)
```

En forma expandida:

```
D^q x = -alpha*(1+m)*x + alpha*y - alpha*(n-m)*arctan(x)
D^q y =  x - y + z
D^q z = -beta*y - gamma*z
```

---

## 2. Parámetros oficiales (Wu et al. 2023, Tabla 1)

| Parámetro | Valor |
|-----------|------:|
| `alpha`   | 8.4562 |
| `beta`    | 12.0732 |
| `gamma`   | 0.0052 |
| `m`       | 0.4 |
| `n`       | −1.1585 |
| `n − m`   | −1.5585 |
| `q`       | 0.99 |
| `h`       | 0.01 |
| `N`       | 10 000 |

---

## 3. Condiciones iniciales reportadas

| Etiqueta | Valor |
|----------|-------|
| `x0_plus`  | [13.8, 0.7093, −19.8768] |
| `x0_minus` | [−13.8, −0.7093, 19.8768] |
| `x0_fig`   | [13.0, 0.7, −19.0] (Fig. 5 del artículo, variante) |

---

## 4. Método ADM — actualización local de 4to orden

El artículo utiliza el **Método de Descomposición de Adomian (ADM)** para
integrar el sistema Caputo.  La solución se expresa como serie de potencias
en el paso de tiempo:

```
x_{n+1} = sum_{k=0}^{4}  C1^k * h^{kq} / Gamma(kq + 1)
y_{n+1} = sum_{k=0}^{4}  C2^k * h^{kq} / Gamma(kq + 1)
z_{n+1} = sum_{k=0}^{4}  C3^k * h^{kq} / Gamma(kq + 1)
```

### 4.1 Coeficientes (orden 0 y 1)

```
C1^0 = x_n,   C2^0 = y_n,   C3^0 = z_n

C1^1 = -alpha*(1+m)*x_n + alpha*y_n - alpha*(n-m)*arctan(x_n)
C2^1 = x_n - y_n + z_n
C3^1 = -beta*y_n - gamma*z_n
```

### 4.2 Derivadas de arctan en x_n (polinomios de Adomian)

Sea `s = x_n^2 + 1`:

```
g0 = 1/s                       (primera derivada de arctan)
g1_raw = -2*x_n / s^2          (segunda derivada, sin factor 1/2!)
g2_raw = 8*x_n^2/s^3 - 2/s^2  (tercera derivada, sin factor 1/3!)
```

### 4.3 Coeficiente orden 2

```
C1^2 = -alpha*(1+m)*C1^1 + alpha*C2^1 - alpha*(n-m)*g0*C1^1
C2^2 = C1^1 - C2^1 + C3^1
C3^2 = -beta*C2^1 - gamma*C3^1
```

### 4.4 Coeficiente orden 3

```
r_A2 = Gamma(2q+1) / Gamma(2q+2)

A2 = g0*C1^2 + (1/2)*g1_raw*(C1^1)^2 * r_A2

C1^3 = -alpha*(1+m)*C1^2 + alpha*C2^2 - alpha*(n-m)*A2
C2^3 = C1^2 - C2^2 + C3^2
C3^3 = -beta*C2^2 - gamma*C3^2
```

### 4.5 Coeficiente orden 4

```
r_A3a = Gamma(3q+1) / (Gamma(q+1)*Gamma(2q+1))
r_A3b = Gamma(3q+1) / Gamma(3q+3)

A3 = g0*C1^3
     + g1_raw*C1^1*C1^2 * r_A3a
     + (1/6)*g2_raw*(C1^1)^3 * r_A3b

C1^4 = -alpha*(1+m)*C1^3 + alpha*C2^3 - alpha*(n-m)*A3
C2^4 = C1^3 - C2^3 + C3^3
C3^4 = -beta*C2^3 - gamma*C3^3
```

---

## 5. Diferencia fundamental: ADM local vs. Caputo con memoria completa

| Aspecto | ADM local (Wu 2023) | ABM / EFORK-3 (Caputo) |
|---------|---------------------|------------------------|
| Memoria | Sólo el estado actual `X_n` | Historia completa desde t=0 |
| Convolu­ción Caputo | No | Sí (kernel `(t-s)^{q-1}/Gamma(q)`) |
| Complejidad por paso | O(1) | O(n) (crece con el tiempo) |
| Equivalencia matemática | **No** | Sí, con la definición Caputo |

> **El ADM local no es una aproximación del operador Caputo con historia
> completa.**  Es una aproximación del operador de Grünwald-Letnikov local
> (sólo un paso atrás).  Los resultados numéricos pueden diferir
> significativamente para t grande.

---

## 6. Comandos de ejecución

### 6.1 Simulación directa con ADM (Ruta B)

```bash
python -m src.cli.run_workflow \
  --config configs/examples/chua_arctan_wu2023_adm_attractor_only.yaml
```

o directamente:

```bash
python -m src.cli.simulate_attractor \
  --config configs/examples/chua_arctan_wu2023_adm_attractor_only.yaml
```

### 6.2 Comparación ADM vs ABM vs EFORK

```bash
# ADM (reproducción del paper)
python -m src.cli.run_workflow \
  --config configs/examples/chua_arctan_wu2023_compare_integrators.yaml \
  --integrator adm_wu2023 \
  --output-dir outputs/chua_arctan_wu2023_compare/adm_wu2023

# ABM (Caputo memoria completa)
python -m src.cli.run_workflow \
  --config configs/examples/chua_arctan_wu2023_compare_integrators.yaml \
  --integrator abm \
  --output-dir outputs/chua_arctan_wu2023_compare/abm

# EFORK-3 (Caputo memoria completa)
python -m src.cli.run_workflow \
  --config configs/examples/chua_arctan_wu2023_compare_integrators.yaml \
  --integrator efork \
  --output-dir outputs/chua_arctan_wu2023_compare/efork
```

---

## 7. Salidas esperadas

Para cada condición inicial (`x0_plus`, `x0_minus`, `x0_fig`) el workflow genera:

```
outputs/chua_arctan_wu2023_adm_attractor_only/
├── summary.json
├── x0_plus_timeseries.csv
├── x0_plus_attractor.csv
├── x0_minus_timeseries.csv
├── x0_minus_attractor.csv
├── x0_fig_timeseries.csv
├── x0_fig_attractor.csv
└── figures/
    ├── x0_plus_3d.png
    ├── x0_plus_xy.png
    ├── x0_plus_xz.png
    ├── x0_plus_yz.png
    └── (ídem para x0_minus y x0_fig)
```

El `summary.json` incluye siempre:

```json
{
  "integrator_class": "adm_local_reproduction",
  "hidden_verified": false,
  "scientific_label": "ADM reproduction of Wu et al. 2023..."
}
```

---

## 8. Interpretación de resultados

### Si ADM produce caos pero ABM/EFORK producen trayectoria periódica

Esto es consistente con la hipótesis de que el artículo usa ADM local
(sin memoria fraccionaria acumulada) y puede generar trayectorias que
difieren de la integración Caputo rigurosa.  Las posibles causas son:

1. **Diferencia de método**: El ADM local no conserva la historia Caputo.
2. **Error de truncamiento**: El polinomio de 4to orden introduce errores
   acumulativos que pueden desestabilizar trayectorias periódicas.
3. **Diferencia genuina en la dinámica**: Posible (pero requiere verificación
   adicional con métodos Caputo rigurosos en ventanas de tiempo más largas).

### Para verificar ocultedad bajo Caputo

La etiqueta `hidden_verified = true` **sólo** puede asignarse después de:

1. Integrar con ABM o EFORK-3 (memoria Caputo completa).
2. Probar vecindades de **todos** los equilibrios (E0, E+, E−) con radios
   declarados explícitamente.
3. Confirmar que ninguna trayectoria vecina converge al atractor desde
   las vecindades de los equilibrios.

**La Ruta B nunca asigna `hidden_verified = true`.**

---

## 9. Advertencia científica

> ⚠️ La reproducción ADM local no equivale a verificación de ocultedad.
> Los resultados de ADM y ABM/EFORK pueden diferir por razones puramente
> numéricas.  No concluir automáticamente que hay un atractor oculto de
> Caputo basándose únicamente en resultados de ADM.

---

## 10. Referencias

- Wu et al. (2023). *Hidden attractors in a new fractional-order Chua system
  with arctan nonlinearity and its DSP implementation.*
- Diethelm et al. (2002). *A predictor-corrector approach for the numerical
  solution of fractional differential equations.*  (ABM method)
- Ghoreishi, Ghaffari & Saad (2023). *EFORK-3 explicit three-stage method.*
- Adomian (1994). *Solving Frontier Problems of Physics: The Decomposition Method.*
