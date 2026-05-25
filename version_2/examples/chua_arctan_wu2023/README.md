# Fractional Chua Arctan Wu2023

Este ejemplo implementa, de forma separada del Chua no suave, el sistema
fraccionario con no linealidad `arctan` reportado por Wu et al. (2023):

```text
^C D_t^q x = alpha * (y - x - f(x))
^C D_t^q y = x - y + z
^C D_t^q z = -beta * y - gamma * z
f(x) = a1*x + a2*atan(rho*x)
```

con `alpha=8.4562`, `beta=12.0732`, `gamma=0.0052`, `a1=m=0.4`,
`a2=n-m=-1.5585`, `rho=1`, `q=0.99`, `h=0.01` y `N=10000`.

## Equilibrios Y Matignon

La ecuación escalar de equilibrio produce:

| Punto | Estado aproximado |
| --- | --- |
| `E0` | `(0, 0, 0)` |
| `E+` | `(0.60967911698, 2.6247941849e-4, -0.60941663756)` |
| `E-` | `(-0.60967911698, -2.6247941849e-4, 0.60941663756)` |

El criterio de Matignon usa `|arg(lambda_i)| > q*pi/2`. Para `q=0.99`,
`E0`, `E+` y `E-` resultan localmente inestables; esto no prueba ni descarta
por sí solo un atractor oculto.

## Forma Lur'e

La separación manual usada para generar semillas es:

```text
P = [[-alpha*(1+a1), alpha, 0],
     [1, -1, 1],
     [0, -beta, -gamma]]
b = [-alpha, 0, 0]^T
r = [1, 0, 0]^T
psi(sigma) = a2*atan(rho*sigma)
```

La transferencia fraccionaria se evalúa con `lambda=(j*omega)^q`, nunca
con la transferencia entera:

```text
W_q(j*omega) = r^T * (lambda*I - P)^(-1) * b
N(A) = 2*a2*(sqrt(1+(rho*A)^2)-1)/(rho*A^2)
```

Con el signo de `b` y `psi` anterior, el cierre consistente es:

```text
1 - W_q(j*omega)*N(A) = 0
```

La expresión `1 + W_q*N(A)=0` con exactamente el mismo `b` y `a2<0` invierte
el lazo y no produce ramas centradas admisibles. La salida JSON conserva esta
auditoría de signo para impedir una reproducción algebraicamente inconsistente.

## Condiciones Iniciales Del Paper

La configuración carga explícitamente:

```text
x0_plus  = [13.8000,  0.7093, -19.8768]
x0_minus = [-13.8000, -0.7093,  19.8768]
```

Se integran sólo cuando se solicita la fase dinámica; no se confunden con
semillas Lur'e ni con resultados previos del Chua no suave.

## Ejecución

Desde la raíz `version_2`:

```powershell
python examples/chua_arctan_wu2023/run_seed_generation.py
python examples/chua_arctan_wu2023/run_validation.py
python examples/chua_arctan_wu2023/run_validation.py --run-trajectories
python examples/chua_arctan_wu2023/run_validation.py --run-trajectories --full-history
python examples/chua_arctan_wu2023/plot_basins.py
```

La configuración principal está en `configs/chua_arctan_wu2023_caputo.json`.
`full_history` reproduce el historial Caputo completo para la corrida corta
del paper; `finite_memory` inicia en `memory_length=40.0` como variante
robusta y escalable.

## Protocolo De Ocultedad

Una semilla armónica no se rechaza por ser armónica. El filtro periódico se
ejecuta únicamente después de integrar el sistema objetivo, descartar el
transitorio y analizar FFT/PSD, entropía espectral, deriva de frecuencia y
rango en al menos dos componentes.

Ninguna salida de este ejemplo se etiqueta `hidden_verified` sin muestrear
vecindades de `E0`, `E+` y `E-` para todos los radios configurados y sin una
referencia dinámica robusta. Una órbita periódica cuya cuenca no toque los
equilibrios puede archivarse como `regular_hidden_like_not_chaotic`, nunca
como validación de atractor caótico oculto.

Machado queda fuera de la reproducción principal de Wu2023. Una futura
extensión arctan debe usar una rama compleja explícita
`N_mu(A)=exp(mu*Log_branch(N(A)))` y etiquetarse como experimental.
