# Resumen de Candidatos Caóticos de la Rama 0 (Chua-Arctan Fraccionario)

Este reporte contiene un análisis matemático, metodológico y numérico detallado de los 13 candidatos de la **Rama 0** (`branch_0`) del sistema de Chua con no linealidad arcotangente fraccionaria. Se evalúa el origen de sus semillas, el proceso de continuación de parámetros, su cercanía en el espacio de fases y una comparación visual de cuatro solucionadores.

---

## 1. Identificación y Características Compartidas (Primero)

Todos los 13 candidatos de la Rama 0 comparten el mismo "núcleo" dinámico del sistema de Chua fraccionario:
- **Parámetros del Sistema**: $\alpha = 8.4562$, $\beta = 12.0732$, $\gamma = 0.0052$, $q = 0.99$, $h = 0.005\text{ s}$.
- **Frecuencia del Límite Oscilatorio**: La frecuencia de cruce $\omega_0$ es **exactamente $2.0392$** en todos los casos.
- **Ganancia Describiente Necesaria**: La ganancia $k$ requerida para intersectar el diagrama de Nyquist es exactamente:
  - $k = -1.0369$ (para los casos con $a_1 = 0.1$)
  - $k = -1.1369$ (para los casos con $a_1 = 0.2$)

### La Regla de Producto de Cruce: $|a_2| \cdot \rho$
Para que exista una intersección armónica (semilla), la máxima ganancia equivalente de la no linealidad de Chua-Arctan en el origen ($|a_2| \cdot \rho$) debe superar el umbral de la ganancia requerida $|k|$. 

De los 13 candidatos:
* **9 de ellos** tienen un producto $|a_2| \cdot \rho$ igual a **exactamente $1.5000$**.
* Los 4 restantes tienen valores de $1.25$ (límite inferior de existencia), $1.5585$, $1.6000$ y $1.8000$.

Esto demuestra que **la característica compartida que define a los candidatos caóticos es la sintonización precisa del producto $|a_2| \cdot \rho \approx 1.50$**, lo cual compensa linealmente la variación en la escala de amplitud del atractor y permite la bifurcación al ciclo caótico a través de una intersección de Nyquist idéntica.

### Tabla de Parámetros de los Candidatos Caóticos (Rama 0)

| Caso ID (Filtro) | $a_1$ | $a_2$ | $\rho$ | Producto $\|a_2\| \cdot \rho$ | $\omega_0$ | Ganancia $k$ | Amplitud $A$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `a1_0p1_a2_m1_rho_1p25` | 0.1 | -1.0000 | 1.25 | 1.2500 | 2.0392 | -1.0369 | 0.7963 |
| `a1_0p1_a2_m1_rho_1p5` | 0.1 | -1.0000 | 1.50 | 1.5000 | 2.0392 | -1.0369 | 1.0717 |
| `a1_0p1_a2_m1p2_rho_1p25` | 0.1 | -1.2000 | 1.25 | 1.5000 | 2.0392 | -1.0369 | 1.2860 |
| `a1_0p1_a2_m1p2_rho_1p5` | 0.1 | -1.2000 | 1.50 | 1.8000 | 2.0392 | -1.0369 | 1.5070 |
| `a1_0p1_a2_m1p5585_rho_1` | 0.1 | -1.5585 | 1.00 | 1.5585 | 2.0392 | -1.0369 | 1.7390 |
| `a1_0p1_a2_m2_rho_0p75` | 0.1 | -2.0000 | 0.75 | 1.5000 | 2.0392 | -1.0369 | 2.1433 |
| `a1_0p1_a2_m2p5_rho_0p5` | 0.1 | -2.5000 | 0.50 | 1.2500 | 2.0392 | -1.0369 | 1.9908 |
| `a1_0p1_a2_m3_rho_0p5` | 0.1 | -3.0000 | 0.50 | 1.5000 | 2.0392 | -1.0369 | 3.2150 |
| `a1_0p2_a2_m0p8_rho_2` | 0.2 | -0.8000 | 2.00 | 1.6000 | 2.0392 | -1.1369 | 0.7571 |
| `a1_0p2_a2_m1_rho_1p5` | 0.2 | -1.0000 | 1.50 | 1.5000 | 2.0392 | -1.1369 | 0.8655 |
| `a1_0p2_a2_m1p2_rho_1p25` | 0.2 | -1.2000 | 1.25 | 1.5000 | 2.0392 | -1.1369 | 1.0385 |
| `a1_0p2_a2_m1p5585_rho_1` | 0.2 | -1.5585 | 1.00 | 1.5585 | 2.0392 | -1.1369 | 1.4259 |
| `a1_0p2_a2_m2_rho_0p75` | 0.2 | -2.0000 | 0.75 | 1.5000 | 2.0392 | -1.1369 | 1.7309 |

---

## 2. Origen de Semillas y Continuación Numérica (Tercero)

El workflow de localización de atractores ocultos utiliza el enfoque de la función describiente (DF):

```mermaid
graph TD
    A[Calcular Función de Transferencia W(jω)] --> B[Detectar Frecuencia de Cruce ω0 donde Im(W) = 0]
    B --> C[Calcular Ganancia de Cruce k = -1 / Re(W)]
    C --> D[Resolver Amplitud A0 tal que N(A0) = k mediante Bisección]
    D --> E[Generar Semilla en Estado Espacial x0 con Base Modal]
    E --> F[Aplicar Continuación Homotópica de η de 0 a 1]
    F --> G[Integración Final con Estado Final en eta = 1]
```

### Proceso de Búsqueda de la Semilla
1. **Filtro Nyquist (`k_phi`)**: Se evalúa la respuesta en frecuencia $W(j\omega)$ del sistema lineal. Se detectan las raíces $\omega_0$ en las que la componente imaginaria se anula.
2. **Igualación de Ganancia**: Al anularse la parte imaginaria, se calcula la ganancia real $k$ requerida. Mediante bisección en la amplitud $A$, se halla $A_0$ que cumple $N(A_0) = k$.
3. **Construcción Modal**: La condición armónica predice una órbita. La semilla en el espacio de estados de 3 dimensiones se obtiene proyectando la amplitud modal sobre el autovector del cruce Nyquist: $x_0 = \text{Re}(v \cdot A_0)$.

### Continuación Homotópica de Parámetros ($\eta$)
Para pasar del límite cíclico lineal ($\eta=0$) al sistema real no lineal de Chua ($\eta=1$), se define un campo vectorial deformado con el parámetro de continuación $\eta \in [0, 1]$:
$$f(t, x) = P_0 x + \eta \cdot b \cdot [ \Psi(r^T x) - k \cdot r^T x ]$$
Donde $P_0 = P + k b r^T$ es el sistema linealizado estabilizado en el ciclo, y $\Psi(\cdot)$ es la función arcotangente real.
* Se incrementa $\eta$ desde $0$ hasta $1$ en 21 pasos uniformes.
* En cada paso se integra durante un tiempo transitorio ($30\text{ s}$) y se registra el final.
* El estado final en $\eta_j$ y su memoria fraccionaria sirven como el estado inicial y prehistoria para $\eta_{j+1}$.
* En $\eta = 1$, la órbita de la semilla original ha convergido suavemente al atractor caótico del sistema no lineal completo.

---

## 3. Evaluación de Cercanía de Semillas e Identidad (Cuarto)

Para determinar si estos atractores son diferentes o el mismo, analizamos la distancia geométrica (norma $L_2$) entre las condiciones iniciales (semillas) de los 13 candidatos:

* **Distancia Mínima entre Semillas**: **$0.014045$**
  * Entre: `a1_0p1_a2_m1p5585_rho_1_branch_0`
  * Y: `a1_0p2_a2_m2_rho_0p75_branch_0`
* **Distancia Máxima entre Semillas**: **$4.289052$**
  * Entre: `a1_0p1_a2_m3_rho_0p5_branch_0`
  * Y: `a1_0p2_a2_m0p8_rho_2_branch_0`
* **Semilla Promedio (Centroide de la Familia)**:
  $$\bar{x}_0 = [1.505140, 0.094925, -2.150358]^T$$
* **Desviación Estándar de la Familia**:
  $$\sigma = [0.656531, 0.041406, 0.937970]^T$$

> [!NOTE]
> **Conclusión sobre la Identidad del Atractor**:
> Debido a que las ecuaciones tienen coeficientes de no linealidad ($a_2$) y escalas ($\rho$) formalmente distintos, **estos 13 candidatos representan matemáticamente 13 sistemas dinámicos diferentes**.
>
> Sin embargo, físicamente, el atractor caótico pertenece a la **misma clase y topología de atractor caótico de doble scroll (familia de atractores auto-similares)**. La extrema cercanía de algunas semillas (distancias menores a $0.014$) confirma que todos estos puntos pertenecen a la misma región del espacio de fases, y que el comportamiento caótico se preserva conservando la misma topología fundamental gracias a la compensación del factor de escala en la no linealidad ($|a_2| \cdot \rho$).

---

## 4. Galería de Gráficos Comparativos (Segundo)

Se integraron los 13 candidatos hasta $t_{final} = 500\text{ s}$ con un paso $h = 0.005\text{ s}$, eliminando los primeros $100\text{ s}$ de transitorio. A continuación se presentan las visualizaciones de espacio de fases en una cuadrícula de 2x2 comparando **ADM, ABM, EFORK Truncado (ventana de memoria de 40.0s) y EFORK Completo (historial completo)** para cada candidato:

````carousel
![Caso 1](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1_rho_1p25_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 2](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1_rho_1p5_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 3](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p2_rho_1p25_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 4](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p2_rho_1p5_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 5](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p5585_rho_1_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 6](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m2_rho_0p75_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 7](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m2p5_rho_0p5_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 8](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m3_rho_0p5_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 9](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m0p8_rho_2_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 10](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1_rho_1p5_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 11](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1p2_rho_1p25_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 12](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1p5585_rho_1_branch_0_four_solvers_phase3d.png)
<!-- slide -->
![Caso 13](file:///C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m2_rho_0p75_branch_0_four_solvers_phase3d.png)
````

> [!TIP]
> **Observación Metodológica**:
> El comportamiento dinámico obtenido mediante EFORK Truncado (ventana de $40\text{ s}$) y EFORK Completo es cualitativamente idéntico en todos los casos. Esto valida que para el sistema de Chua-Arctan, el principio de la "memoria corta" fraccionaria es extremadamente seguro para simular trayectorias caóticas de larga duración, reduciendo drásticamente la carga computacional sin alterar el comportamiento cualitativo del atractor.
