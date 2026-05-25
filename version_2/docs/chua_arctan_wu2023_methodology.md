# Metodología Chua Fraccionario Arctan Wu2023

Esta página especifica el caso `fractional_chua_arctan_wu2023`. Sus
artefactos se almacenan de forma independiente al caso Chua no suave; este
último sólo sirve como plantilla de etapas del pipeline.

## Sistema Oficial

Para un orden conmensurable Caputo `q=0.99`:

```text
^C D_t^q x = alpha*(y - x - f(x))
^C D_t^q y = x - y + z
^C D_t^q z = -beta*y - gamma*z
f(x) = a1*x + a2*atan(rho*x)
```

| Parámetro | Valor |
| --- | ---: |
| `alpha` | `8.4562` |
| `beta` | `12.0732` |
| `gamma` | `0.0052` |
| `m = a1` | `0.4` |
| `n` | `-1.1585` |
| `a2 = n-m` | `-1.5585` |
| `rho` | `1.0` |
| `q` | `0.99` |
| `h`, `N` | `0.01`, `10000` |

Las condiciones iniciales reproducibles del artículo se registran como
`x0_plus=(13.8, 0.7093, -19.8768)` y
`x0_minus=(-13.8, -0.7093, 19.8768)`.

## Forma Lur'e Manual

Con `sigma=r^T X=x`, la dinámica se escribe
`^C D_t^q X = P X + b psi(sigma)` mediante:

```text
P = [[-alpha*(1+a1), alpha, 0],
     [1, -1, 1],
     [0, -beta, -gamma]]
b = [-alpha, 0, 0]^T
r = [1, 0, 0]^T
psi(sigma) = a2*atan(rho*sigma)
```

Esta separación es manual y auditable; el paquete no infiere una forma Lur'e
desde el campo vectorial.

## Equilibrios

De la segunda y tercera ecuación en equilibrio:

```text
y* = gamma*x*/(beta+gamma)
z* = -beta*x*/(beta+gamma)
```

La primera ecuación reduce a:

```text
(a1 + beta/(beta+gamma))*x* + a2*atan(rho*x*) = 0.
```

Sus raíces producen:

| Equilibrio | Estado |
| --- | --- |
| `E0` | `(0, 0, 0)` |
| `E+` | `(0.60967911698, 0.00026247941849, -0.60941663756)` |
| `E-` | `(-0.60967911698, -0.00026247941849, 0.60941663756)` |

`hidden_attractors.validation.chua_arctan_wu2023` sustituye cada punto en el
campo vectorial y guarda el residual `||F(E*)||`.

## Jacobiano Y Matignon

La derivada de la característica suave es:

```text
dphi/dx = a1 + a2*rho/(1 + (rho*x)^2)
```

y por tanto:

```text
J(X) = [[-alpha*(1+dphi/dx), alpha, 0],
        [1, -1, 1],
        [0, -beta, -gamma]]
```

Para un sistema Caputo conmensurable, el criterio local de Matignon exige:

```text
|arg(lambda_i(J))| > q*pi/2
```

para cada autovalor. A `q=0.99`, la validación algebraica clasifica los tres
equilibrios como inestables: `E0` posee un autovalor real positivo y
`E+`, `E-` poseen un par complejo en el sector inestable.

## Transferencia Fraccionaria Y Función Descriptiva

La evaluación armónica usa la rama principal:

```text
lambda = (j*omega)^q = omega^q * exp(j*q*pi/2)
W_q(j*omega) = r^T*(lambda*I - P)^(-1)*b
```

No se sustituye por `W(j*omega)` de orden entero. Para
`psi(sigma)=a2*atan(rho*sigma)`, la función descriptiva clásica es:

```text
N(A) = 2*a2*(sqrt(1+(rho*A)^2)-1)/(rho*A^2).
```

Como `a2<0`, `N(A)<0` para `A>0`. Con el vector `b` anterior, la ecuación
modal de la linealización `P+b*N(A)*r^T` es:

```text
1 - W_q(j*omega)*N(A) = 0.
```

Esto es equivalente a la convención histórica del código
`r^T*(P-lambda*I)^(-1)*b` con cierre `1+W_code*N=0`. En cambio, imponer
`W_q*N=-1` manteniendo `b=(-alpha,0,0)^T` y la misma `psi` produce ganancias
de signo opuesto a `N(A)` y no constituye una rama Wu2023 admisible.

El JSON de semillas guarda `omega`, `k=N(A)`, `A`, el eigenvector normalizado,
`lambda=(j*omega)^q`, el estado semilla y ambos residuos de signo.

## Weyl-Caputo Y Continuación

La construcción armónica representa una respuesta estacionaria tipo Weyl y
sirve sólo para localizar semillas. La comprobación causal debe reintegrar el
problema de valor inicial Caputo desde una condición inicial declarada. El
puente metodológico es:

1. generar la semilla mediante balance armónico fraccionario;
2. continuar la no linealidad hasta el sistema objetivo `eta=1`;
3. integrar Caputo con política de memoria registrada;
4. descartar transitorio y decidir periodicidad/robustez;
5. sólo entonces ejecutar controles de cuenca.

La configuración permite `full_history` y `finite_memory`. La variante
`finite_memory` parte de una ventana de `40.0`; los cambios de ventana son
pruebas de robustez, no una sustitución silenciosa del problema Caputo.

## Protocolo De Ocultedad Y Descarte Periódico

Una semilla no se excluye por su forma armónica. En el sistema objetivo, tras
descartar el transitorio, `hidden_attractors.diagnostics.periodicity` calcula
FFT/PSD por componente, razón de pico dominante, entropía espectral, deriva
de frecuencia por ventanas y rango. Se requiere coincidencia de al menos dos
componentes para etiquetar periodicidad.

Las etiquetas dinámicas son:

| Etiqueta | Uso |
| --- | --- |
| `regular_periodic_rejected` | Órbita periódica no admisible para caos. |
| `thin_periodic_rejected` | Traza casi cerrada y delgada; rechazada. |
| `nonperiodic_candidate` | No periódica en el filtro; falta robustez. |
| `chaotic_candidate_pending_robustness` | Banda ancha compatible con caos; falta robustez y cuenca. |

Si una órbita regular no tiene contacto desde las vecindades probadas de los
equilibrios, puede documentarse como `regular_hidden_like_not_chaotic`; no se
usa para validar un atractor caótico. La etiqueta `hidden_verified` sólo es
legal después de probar vecindades de `E0`, `E+` y `E-` para los radios
declarados, bajo una referencia dinámica robusta.

## Extensión Machado

Machado no forma parte de la validación principal Wu2023. Una extensión
exploratoria futura para `arctan` debe especificar una rama compleja:

```text
N_mu(A) = exp(mu*Log_branch(N(A)))
```

y producir artefactos marcados explícitamente como experimentales, separados
de la reproducción del artículo.

## Artefactos

- Configuración: `configs/chua_arctan_wu2023_caputo.json`.
- Ejemplo: `examples/chua_arctan_wu2023/`.
- Algebra JSON: `validation/reference_cases/fractional_chua_arctan_wu2023/01_algebra/`.
- Semillas JSON: `validation/reference_cases/fractional_chua_arctan_wu2023/02_lure_df/`.
