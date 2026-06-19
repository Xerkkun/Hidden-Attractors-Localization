# Systems / Sistemas

The package exposes a registry for chaotic systems. Built-in workflows use the same registry that user-defined systems can use.
El paquete expone un registro para sistemas caóticos. Los flujos de trabajo integrados utilizan el mismo registro que pueden utilizar los sistemas definidos por el usuario.

## Inspect Registered Systems / Inspección de Sistemas Registrados

```bash
hidden-attractors-systems
hidden-attractors-systems --system chua-nonsmooth --equilibria --state 0,0,0
```

The built-in Chua systems are `chua-nonsmooth` and `chua-arctan`. The Chua non-smooth system advertises the official protocol interface:
Los sistemas Chua integrados son `chua-nonsmooth` y `chua-arctan`. El sistema Chua no suave anuncia la interfaz de protocolo oficial:

```bash
hidden-attractors-protocol --help
```

---

## Define a New System / Definición de un Nuevo Sistema

User-defined systems register a vector field, parameters, optional equilibria, and optional workflow commands:
Los sistemas definidos por el usuario registran un campo vectorial, parámetros, equilibrios opcionales y comandos de flujo de trabajo opcionales:

```python
from typing import Any, Mapping

import numpy as np

from hidden_attractors.systems import ChaoticSystem, register_system


def rhs(state: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    x, y, z = state
    sigma = float(p["sigma"])
    rho = float(p["rho"])
    beta = float(p["beta"])
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


register_system(
    ChaoticSystem(
        name="lorenz63",
        dimension=3,
        rhs=rhs,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        description="Classic Lorenz 63 system.",
    ),
    replace=True,
)
```

See `examples/custom_system_definition.py` for a runnable example.
See also `examples/new_system_workflow_spec.py` for the next layer: writing a `WorkflowInputSpec` that records solver, classifier, target-reference, basin, and refinement inputs before launching reusable workflows.
Consulte `examples/custom_system_definition.py` para obtener un ejemplo ejecutable. Consulte también `examples/new_system_workflow_spec.py` para la siguiente capa: escribir una `WorkflowInputSpec` que registre las entradas del resolvedor, el clasificador, la referencia de objetivo, la cuenca y el refinamiento antes de lanzar flujos de trabajo reutilizables.

---

## Workflow Contract / Contrato de Flujo de Trabajo

A system definition is not automatically a full hidden-attractor workflow. A complete workflow follows the fixed official order and still needs:
La definición de un sistema no es automáticamente un flujo de trabajo completo de atractores ocultos. Un flujo de trabajo completo sigue el orden oficial fijo y aún necesita:

- a manual Lur'e form when the DF/Nyquist route is used: `D^q x = A x + b psi(c^T x)`;
- a classical describing function `N(A)` for the scalar nonlinearity;
- a Machado describing-function branch `N_mu(A, mu)` when Machado seeds or sweeps are requested;
- a `ContinuationPlan(lambda_values=...)` map from the auxiliary harmonic or smoothed system (`lambda=0`) to the original nonlinear system (`lambda=1`);
- equilibrium or target-neighborhood checks;
- hiddenness controls sampled inside balls centered at every equilibrium;
- basin classification or a documented replacement criterion;
- trajectory diagnostics and report outputs;
- an explicit numerical contract: model, parameters, time horizon, step size, memory length if fractional, burn-in, backend, and thresholds.
- 
- una forma de Lur'e manual cuando se utiliza la ruta DF/Nyquist: `D^q x = A x + b psi(c^T x)`;
- una función descriptiva clásica `N(A)` para la no-linealidad escalar;
- una rama de función descriptiva de Machado `N_mu(A, mu)` cuando se solicitan semillas o barridos de Machado;
- un mapa de `ContinuationPlan(lambda_values=...)` desde el sistema armónico o suavizado auxiliar (`lambda=0`) al sistema no lineal original (`lambda=1`);
- comprobaciones de equilibrio o vecindad de objetivos;
- controles de ocultedad muestreados dentro de bolas centradas en cada equilibrio;
- clasificación de cuencas o un criterio de reemplazo documentado;
- diagnósticos de trayectoria y salidas de reportes;
- un contrato numérico explícito: modelo, parámetros, horizonte de tiempo, tamaño de paso, longitud de memoria si es fraccionario, transitorio (burn-in), backend y umbrales.

For heavy numerical work, add a C backend or an adapter to a proven external solver before exposing the workflow as stable.
Para trabajos numéricos pesados, agregue un backend de C o un adaptador a un resolvedor externo probado antes de exponer el flujo de trabajo como estable.

The package-level readiness checker makes this distinction explicit:
El comprobador de preparación a nivel de paquete hace explícita esta distinción:

```bash
hidden-attractors-workflow-requirements --workflow basin --system chua-nonsmooth
hidden-attractors-workflow-requirements --workflow strict-refinement --system chua-nonsmooth
```

If the checker reports missing `integrator`, `target-reference`, or `basin-slice`, add those fields to a `WorkflowInputSpec`; do not add them to the vector-field definition itself.
Si el comprobador informa de la falta de `integrator`, `target-reference` o `basin-slice`, agregue esos campos a una `WorkflowInputSpec`; no los agregue a la definición del campo vectorial en sí.

---

## Lur'e Requirement / Requisito de Lur'e

For the full route used by the Chua examples, the user must provide more than the equations. The system must be entered both as a vector field and as a Lur'e split:
Para la ruta completa utilizada por los ejemplos de Chua, el usuario debe proporcionar algo más que las ecuaciones. El sistema debe introducirse tanto como un campo vectorial como una división de Lur'e:

```text
D^q x = A x + b psi(c^T x)
```

The package does not infer that split automatically. The user must provide:
El paquete no infiere esa división automáticamente. El usuario debe proporcionar:

- `A`, `b`, and `c`;
- the scalar nonlinearity `psi(sigma)`;
- the classical describing function `N(A)`;
- the Machado branch `N_mu(A, mu)`;
- amplitude/gain compatibility rules when closed-form bounds are known;
- equilibria for hiddenness tests;
- a Jacobian if Lyapunov exponents should be robust and fast.
- 
- `A`, `b` y `c`;
- la no-linealidad escalar `psi(sigma)`;
- la función descriptiva clásica `N(A)`;
- la rama de Machado `N_mu(A, mu)`;
- reglas de compatibilidad de amplitud/ganancia cuando se conocen límites de forma cerrada;
- equilibrios para pruebas de ocultedad;
- un Jacobiano si los exponentes de Lyapunov deben ser robustos y rápidos.

The generic Machado branch currently expects a real-valued branch. It extends the seed space only and never constitutes hiddenness evidence. The standard admitted form is:
La rama genérica de Machado espera actualmente una rama de valor real. Extiende el espacio de semillas únicamente y nunca constituye evidencia de ocultedad. La forma estándar admitida es:

```text
N_mu(A, mu) = N(A)^mu,     mu > 0,     N(A) > 0
```

If a system has a complex, sign-changing, or multi-branch describing function, the user must define the branch manually through `machado_describing_function` and document the convention. The package should not silently guess it.
Si un sistema tiene una función descriptiva compleja, que cambia de signo o de múltiples ramas, el usuario debe definir la rama manualmente a través de `machado_describing_function` y documentar la convención. El paquete no debe adivinarla silenciosamente.

---

## Integer-Order Systems / Sistemas de Orden Entero

Systems may be integer-order systems. For order one, the reusable API is under `hidden_attractors.workflows.integer_lure`:
Los sistemas pueden ser sistemas de orden entero. Para orden uno, la API reutilizable se encuentra bajo `hidden_attractors.workflows.integer_lure`:

```python
from hidden_attractors import get_system
from hidden_attractors.workflows.integer_lure import (
    integer_lure_seed,
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    run_integer_lure_hiddenness_controls,
)

from hidden_attractors import ContinuationPlan

system = get_system("chua-nonsmooth")
seed = integer_lure_seed(system)
steps = continue_integer_lure_seed(
    system,
    seed,
    plan=ContinuationPlan.uniform(9, internal_parameter="epsilon"),
)
target_seed, trajectory, status = final_integer_lure_attractor(system, steps[-1].x_out)
probes = run_integer_lure_hiddenness_controls(system, trajectory)
```

`examples/integer_lure_chua_protocol.py` is the small runnable Chua integer example. The regenerated corrected Chua integer run in `validation/reference_cases/chua_integer_q1/` is the promoted reference artifact set for what an integer-order workflow should be able to reproduce or adapt:
`examples/integer_lure_chua_protocol.py` es el pequeño ejemplo ejecutable del caso de Chua de orden entero. La ejecución corregida regenerada de Chua entero en `validation/reference_cases/chua_integer_q1/` es el conjunto de artefactos de referencia promocionado para lo que un flujo de trabajo de orden entero debería ser capaz de reproducir o adaptar:

- `fig01`: Nyquist/describing-function and real/imaginary transfer-component closure;
- `fig02`: continuation (its archived filename predates the public `lambda` vocabulary);
- `fig03`: final attractor and linearized-versus-original comparison;
- `fig04` and `fig05`: reference section and hiddenness controls;
- `fig06`, `fig10`, and `fig12`: basin cuts and 3D basin summaries;
- `fig08` and `fig09`: bifurcation sweeps;
- `fig11`: FFT and PSD spectral diagnostics;
- `fig13`: Lyapunov convergence.
- 
- `fig01`: Nyquist/función descriptiva y cierre de componentes de transferencia reales/imaginarios;
- `fig02`: continuación (su nombre de archivo archivado es anterior al vocabulario público `lambda`);
- `fig03`: atractor final y comparación linealizado-versus-original;
- `fig04` y `fig05`: sección de referencia y controles de ocultedad;
- `fig06`, `fig10` y `fig12`: cortes de cuencas y resúmenes de cuencas en 3D;
- `fig08` y `fig09`: barridos de bifurcación;
- `fig11`: diagnósticos espectrales de FFT y PSD;
- `fig13`: convergencia de Lyapunov.

The generic library now exposes the reusable pieces for seed generation, continuation, final trajectories, hiddenness controls, plotting, and integer-order Lyapunov estimates. Basin C backends and fractional C backends still require a system-specific native implementation or adapter.
La biblioteca genérica expone ahora las piezas reutilizables para la generación de semillas, continuación, trayectorias finales, controles de ocultedad, trazado de gráficos y estimaciones de Lyapunov de orden entero. Los backends de cuencas en C y los backends fraccionarios en C aún requieren una implementación nativa o adaptador específico del sistema.

The audited parameter set, numerical results, attached theoretical report, and evidence-source status are collected in [Integer Chua q=1 Reference](integer_chua_reference.md). Promoted validation artifacts live under `validation/reference_cases/chua_integer_q1/`; they are kept separate from the fractional candidate validation tree.
El conjunto de parámetros auditado, los resultados numéricos, el informe teórico adjunto y el estado de la fuente de evidencia se recopilan en [Referencia de Chua Entero q=1](integer_chua_reference.md). Los artefactos de validación promocionados viven bajo `validation/reference_cases/chua_integer_q1/`; se mantienen separados del árbol de validación de candidatos fraccionarios.

The EFORK-3 stage formula used for this regenerated run is checked separately against the published manufactured-solution benchmarks in [EFORK-3 Published Validation](efork3_validation.md).
La fórmula de la etapa EFORK-3 utilizada para esta ejecución regenerada se verifica por separado frente a los puntos de referencia de soluciones manufacturadas publicados en [Validación Publicada de EFORK-3](efork3_validation.md).
