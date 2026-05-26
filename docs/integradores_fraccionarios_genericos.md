# Integradores Fraccionarios Genéricos en C

Este documento describe la capa nativa y genérica de integración fraccionaria para ecuaciones diferenciales fraccionarias (FDE) del tipo Caputo:

$$ \sideset{^C}{_t^q} D x(t) = f(t, x(t); \text{params}) $$

con $x \in \mathbb{R}^n$ y $0 < q \leq 1$.

---

## 1. Diferencia entre Memoria Completa y Truncada

El operador de Caputo es de carácter **no local**. Para integrar un paso en el tiempo $t_n$, se requiere evaluar la historia completa del sistema desde $t = 0$:

- **Memoria Completa (Full Memory)**: Utiliza todo el historial de integración disponible. Esto es matemáticamente exacto para la definición clásica de Caputo, pero tiene una complejidad computacional de $\mathcal{O}(N^2)$ en tiempo y $\mathcal{O}(N)$ en almacenamiento.
- **Memoria Truncada / Ventana (Windowed Memory)**: Implementa la aproximación de "memoria corta". Asume que el comportamiento pasado lejano tiene una influencia despreciable. Solo conserva la historia en una ventana deslizante de longitud fija $M$ pasos ($L_m = M \cdot h$ en tiempo).

> [!WARNING]
> La memoria truncada es una **aproximación heurística** y no es matemáticamente idéntica al Caputo completo. Su uso debe documentarse debidamente en cualquier publicación científica.

---

## 2. API Unificada en Python

La función principal en Python es `fractional_integrate`, ubicada en [fractional_c.py](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/src/integrators/fractional_c.py):

```python
def fractional_integrate(
    rhs,
    x0,
    q,
    h,
    t_final,
    method: str,
    memory_mode: str,
    memory_window_length: int | None = None,
    history_times=None,
    history_states=None,
    system=None,
    params=None,
    use_c_backend=True,
    divergence_norm=120.0,
    return_history=False,
    allow_python_fallback=False
) -> tuple[np.ndarray, np.ndarray, str, dict]:
```

### Argumentos
- `rhs`: Campo vectorial. Puede ser `rhs(t, x)` o `rhs(x)`.
- `x0`: Condición inicial.
- `q`: Orden fraccionario ($0 < q \leq 1$).
- `h`: Paso temporal.
- `t_final`: Tiempo de integración.
- `method`: `"abm"` (Adams-Bashforth-Moulton) o `"efork"` (Enhanced Fractional Order Runge-Kutta).
- `memory_mode`: `"full"` o `"window"`.
- `memory_window_length`: Número de pasos en la ventana.
- `history_times` y `history_states`: Historia preexistente.
- `return_history`: Si es `True`, el resultado retornado incluye la historia previa de entrada.
- `allow_python_fallback`: Si es `False`, cualquier error al compilar o cargar la capa C lanzará una excepción explícita.

---

## 3. Registro de Sistemas de Alto Rendimiento en C

Para evitar el costo computacional de llamar a callbacks de Python desde el bucle C a través de ctypes, se pueden registrar campos vectoriales en C puro en [rhs_registry.py](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/src/native/rhs_registry.py).

### Cómo registrar un nuevo sistema
1. Define la estructura de parámetros y la función RHS en [fractional_integrators.c](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/src/native/csrc/fractional_integrators.c):
   ```c
   typedef struct {
       double param_a;
   } MySystemParams;

   void my_system_rhs(double t, const double *x, double *dx, int n, void *params) {
       MySystemParams *p = (MySystemParams *)params;
       dx[0] = p->param_a * x[0];
   }
   
   API_EXPORT void *get_my_system_rhs(void) {
       return (void *)my_system_rhs;
   }
   ```
2. Registra el sistema en Python en `rhs_registry.py`:
   ```python
   class MySystemParamsStruct(ctypes.Structure):
       _fields_ = [("param_a", ctypes.c_double)]

   def build_my_system_params(system):
       return MySystemParamsStruct(param_a=float(system.a))

   register_c_rhs("my_system_id", "get_my_system_rhs", build_my_system_params)
   ```

Si el sistema no está registrado, `fractional_integrate` construirá automáticamente un callback ctypes seguro para evaluar la función definida en Python a alto rendimiento.

---

## 4. Historia y Continuación Fraccionaria

Durante la continuación por deformación de Lur'e, el parámetro $\eta$ cambia de manera discreta. Al reiniciar la integración para cada etapa, **no se debe borrar la memoria Caputo**, ya que esto introduciría transitorios artificiales que destruyen la coherencia del estado fraccionario.

El workflow en [continuation_fractional.py](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/src/continuation/continuation_fractional.py) propaga el historial completo `(history_times, history_states)` de una etapa a otra de manera transparente y eficiente.
