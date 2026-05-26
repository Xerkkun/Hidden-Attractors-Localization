import numpy as np
from typing import Callable, Tuple, Optional, Any

from .abm import caputo_abm_integrate
from .efork import efork_integrate
from hidden_attractors.solvers.efork_published import efork3_caputo_integrate

def integrate_general(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    integrator: str = "efork",
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    divergence_norm: Optional[float] = 120.0,
    system: Optional[Any] = None,
    use_c_backend: bool = True
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Solucionador general unificado (facade) para integrar cualquier sistema fraccionario o entero.
    Soporta los esquemas numéricos ABM y EFORK, y realiza fallbacks transparentes a Python si 
    el sistema es arbitrario (fuera de la familia de Chua nativa en C).
    
    Parámetros:
        rhs: Función que calcula la derivada. Puede ser de la forma rhs(t, x) o rhs(x).
        x0: Vector de estado inicial.
        q: Orden de la derivada (q = 1.0 es entero, 0.0 < q < 1.0 es fraccionario).
        h: Tamaño del paso temporal.
        t_final: Tiempo final de integración.
        integrator: Integrador a utilizar ("efork" o "abm").
        memory_mode: Memoria para Caputo ("full" o "window").
        memory_window_length: Longitud de ventana si memory_mode es "window".
        divergence_norm: Norma a partir de la cual se considera divergencia (si se supera, se frena).
        system: Instancia de sistema Chua opcional (para habilitar aceleración C).
        use_c_backend: Si es True, intenta usar el motor compilado en C nativo.
        
    Retorna:
        Tuple[t_arr, x_arr, status]:
            t_arr: Grilla temporal de integración.
            x_arr: Matriz de estados resultantes (filas = tiempo, columnas = variables).
            status: Estado del solver ("ok", "diverged", "solver_exception:...").
    """
    # 1. Cast inputs to standard formats
    x0_arr = np.asarray(x0, dtype=float)
    dim = x0_arr.size
    
    # 2. Normalize rhs signature to rhs_time_dep(t, x)
    # Check if rhs accepts 1 or 2 arguments by wrapping it
    def rhs_t(t: float, x: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(rhs(t, x), dtype=float)
        except TypeError:
            # Fallback to rhs(x) if it only accepts 1 parameter
            return np.asarray(rhs(x), dtype=float)
            
    # 3. If a native Chua system is passed and use_c_backend is True, try utilizing accelerated specialized C backends
    is_saturation = system is not None and "saturation" in getattr(system, "system_id", "")
    is_arctan = system is not None and "arctan" in getattr(system, "system_id", "")
    
    if use_c_backend and (is_saturation or is_arctan):
        try:
            if integrator == "efork":
                t_arr, x_arr, status = efork_integrate(
                    system, x0_arr, q=q, h=h, t_final=t_final,
                    memory_mode=memory_mode, memory_window_length=memory_window_length,
                    use_c_backend=True
                )
                return t_arr, x_arr, status
            elif integrator == "abm" and is_saturation:
                t_arr, x_arr, status = caputo_abm_integrate(
                    system.evaluate_rhs, x0_arr, q=q, h=h, t_final=t_final,
                    divergence_norm=divergence_norm, memory_mode=memory_mode,
                    memory_window_length=memory_window_length, system=system,
                    use_c_backend=True
                )
                return t_arr, x_arr, status
        except Exception:
            pass
            
    # 3.5 General FDE C-solver with Python callbacks for arbitrary/general systems
    if use_c_backend and 0.0 < q < 1.0:
        if integrator == "efork" and memory_mode == "window":
            # EFORK doesn't support window memory in C
            pass
        else:
            try:
                from hidden_attractors.native.backends import GeneralFDEBackend
                backend = GeneralFDEBackend.build()
                t_arr, x_arr, status = backend.integrate(
                    rhs=rhs_t, x0=x0_arr, q=q, h=h, t_final=t_final,
                    divergence_norm=divergence_norm if divergence_norm is not None else 0.0,
                    integrator=integrator
                )
                return t_arr, x_arr, status
            except Exception:
                # Transparent fallback to general Python solvers
                pass
            
    # 4. Fallback/Standard Python Solvers
    # A. Integer Order Solver (q == 1.0)
    if q == 1.0:
        n_steps = int(round(t_final / h))
        x_arr = np.zeros((n_steps + 1, dim))
        t_arr = np.zeros(n_steps + 1)
        x_arr[0] = x0_arr
        t_arr[0] = 0.0
        
        x = x0_arr.copy()
        status = "ok"
        last_idx = 0
        
        # Predictor-Corrector of order 2 (Euler-Heun PECE)
        for n in range(n_steps):
            t_curr = n * h
            t_next = (n + 1) * h
            try:
                # Predictor
                f_curr = rhs_t(t_curr, x)
                x_pred = x + h * f_curr
                # Corrector
                f_next = rhs_t(t_next, x_pred)
                x_next = x + 0.5 * h * (f_curr + f_next)
            except Exception as exc:
                status = f"solver_exception:{exc}"
                break
                
            if divergence_norm is not None and np.linalg.norm(x_next) > divergence_norm:
                status = "diverged"
                x_arr[n + 1] = x_next
                t_arr[n + 1] = t_next
                last_idx = n + 1
                break
                
            x = x_next
            x_arr[n + 1] = x
            t_arr[n + 1] = t_next
            last_idx = n + 1
            
        return t_arr[:last_idx + 1], x_arr[:last_idx + 1], status
        
    # B. Fractional Order Solver (0 < q < 1.0)
    if integrator == "abm":
        # General Python ABM solver
        # Note: caputo_abm_integrate has a Python reference _python_abm_integrate
        t_arr, x_arr, status = caputo_abm_integrate(
            lambda x: rhs_t(0.0, x), x0_arr, q=q, h=h, t_final=t_final,
            divergence_norm=divergence_norm, memory_mode=memory_mode,
            memory_window_length=memory_window_length, use_c_backend=False
        )
        return t_arr, x_arr, status
        
    else: # EFORK
        # Published general Python EFORK-3 solver
        if memory_mode == "window":
            raise ValueError("EFORK no disponible para esta combinación")
            
        try:
            t_arr, x_arr = efork3_caputo_integrate(
                rhs_t, x0_arr, alpha=q, h=h, t_final=t_final
            )
            
            # Check for divergence
            norms = np.linalg.norm(x_arr, axis=1)
            status = "ok"
            last_idx = len(t_arr)
            if divergence_norm is not None:
                div_indices = np.where(norms > divergence_norm)[0]
                if len(div_indices) > 0:
                    status = "diverged"
                    last_idx = div_indices[0] + 1
                    
            return t_arr[:last_idx], x_arr[:last_idx], status
        except Exception as exc:
            return np.array([0.0]), np.array([x0_arr]), f"solver_exception:{exc}"
