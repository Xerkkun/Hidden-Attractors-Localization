import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import gamma
from scipy.optimize import brentq

# ============================================================
# 0) CONFIGURACION GENERAL
# ============================================================

real_dtype = np.float64
complex_dtype = np.complex128


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return real_dtype(default)
    try:
        value = real_dtype(float(raw))
    except ValueError as exc:
        raise ValueError(f"{name} debe ser numerico.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} debe ser finito.")
    return value


def _env_positive_float(name, default):
    value = _env_float(name, default)
    if float(value) <= 0.0:
        raise ValueError(f"{name} debe ser positivo.")
    return value


def _env_nonnegative_float(name, default):
    value = _env_float(name, default)
    if float(value) < 0.0:
        raise ValueError(f"{name} debe ser no negativo.")
    return value

# ------------------------------------------------------------
# Sistema fraccionario de Chua no suave con saturacion
# q = 0.9998
# ------------------------------------------------------------
PARAMS = {
    "model":  os.environ.get(
        "HIDDEN_ATTRACTORS_MODEL",
        "arctan" if os.environ.get("HIDDEN_ATTRACTORS_MODEL_ARCTAN", "").strip().lower() in {"1", "true", "yes", "on"} else "piecewise",
    ),
    "alpha": _env_positive_float("HIDDEN_ATTRACTORS_ALPHA_CHUA", 8.4562),
    "beta":  _env_positive_float("HIDDEN_ATTRACTORS_BETA", 12.0732),
    "gamma": _env_float("HIDDEN_ATTRACTORS_GAMMA_CHUA", 0.0052),
    "m0":    _env_float("HIDDEN_ATTRACTORS_M0", -0.1768),
    "m1":    _env_float("HIDDEN_ATTRACTORS_M1", -1.1468),
    "a1":    _env_float("HIDDEN_ATTRACTORS_A1", 0.4),
    "a2":    _env_float("HIDDEN_ATTRACTORS_A2", -1.5585),
    "rho":   _env_positive_float("HIDDEN_ATTRACTORS_RHO", 1.0),
}
QORD = _env_float("HIDDEN_ATTRACTORS_FRAC_ORDER", 0.9998)

# ------------------------------------------------------------
# Parametros numericos
# ------------------------------------------------------------
H = _env_positive_float("HIDDEN_ATTRACTORS_H", 0.01)
LM = _env_positive_float("HIDDEN_ATTRACTORS_LM", 8.0)          # memoria truncada EFORK
T_TRANSIENT = _env_nonnegative_float("HIDDEN_ATTRACTORS_T_TRANSIENT", 100.0)
T_KEEP = _env_positive_float("HIDDEN_ATTRACTORS_T_KEEP", 70.0)

# Continuacion en epsilon
EPS_VALUES = np.linspace(0.1, 1.0, 10)

# Escaneo para localizar omega0
WMIN = 1e-4
WMAX = 10.0
NSCAN = 20000

# Elegir rama:
# 0 -> rama de menor k (recomendada)
# 1 -> rama de mayor k
BRANCH_INDEX = 0


# ============================================================
# 1) UTILIDADES BASICAS
# ============================================================

def sat_scalar(x):
    if x < -1.0:
        return real_dtype(-1.0)
    if x > 1.0:
        return real_dtype(1.0)
    return real_dtype(x)

def sat_vec(x):
    x = np.asarray(x, dtype=real_dtype)
    return np.clip(x, -1.0, 1.0)

def validate_fractional_order(qord):
    qord = real_dtype(qord)
    if not np.isfinite(qord) or not (0.0 < float(qord) <= 1.0):
        raise ValueError("El orden fraccionario q debe cumplir 0 < q <= 1.")
    return qord


QORD = validate_fractional_order(QORD)


def cpower_iw_q(omega, qord):
    """
    (i*omega)^q en la rama principal:
    (i*omega)^q = omega^q * exp(i*pi*q/2), omega > 0
    """
    omega = real_dtype(omega)
    qord = validate_fractional_order(qord)
    return complex_dtype((omega ** qord) * np.exp(1j * np.pi * qord / 2.0))


# ============================================================
# 2) CHUA FRACCIONARIO NO SUAVE
#    ^C D_t^q x = P x + qvec * psi(r^T x)
# ============================================================

def chua_model(p):
    raw = p.get("model", "piecewise")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"arctan", "atan", "smooth"}:
            return "arctan"
        return "piecewise"
    return "arctan" if int(raw) == 1 else "piecewise"

def chua_gain_A(p):
    return real_dtype(p["m0"] - p["m1"])

def chua_matrices(p):
    a = real_dtype(p["alpha"])
    b = real_dtype(p["beta"])
    g = real_dtype(p["gamma"])
    if chua_model(p) == "arctan":
        base_slope = real_dtype(p["a1"])
    else:
        base_slope = real_dtype(p["m1"])

    P = np.array([
        [-a * (1.0 + base_slope), a,   0.0],
        [1.0,                     -1.0, 1.0],
        [0.0,                     -b,  -g]
    ], dtype=real_dtype)

    qvec = np.array([-a, 0.0, 0.0], dtype=real_dtype)
    r    = np.array([ 1.0, 0.0, 0.0], dtype=real_dtype)

    return P, qvec, r

def psi_sigma(sigma, p):
    if chua_model(p) == "arctan":
        return real_dtype(p["a2"]) * real_dtype(np.arctan(real_dtype(p["rho"]) * real_dtype(sigma)))
    A = chua_gain_A(p)
    return A * sat_scalar(sigma)

def delta_sigma(sigma, p, k):
    return psi_sigma(sigma, p) - real_dtype(k) * real_dtype(sigma)

def rhs_original(x, p):
    P, qvec, r = chua_matrices(p)
    sigma = real_dtype(r @ x)
    return P @ x + qvec * psi_sigma(sigma, p)

def build_P0(p, k):
    P, qvec, r = chua_matrices(p)
    return P + real_dtype(k) * np.outer(qvec, r)

def rhs_epsilon_family(x, p, k, eps):
    P0 = build_P0(p, k)
    _, qvec, r = chua_matrices(p)
    sigma = real_dtype(r @ x)
    return P0 @ x + real_dtype(eps) * qvec * delta_sigma(sigma, p, k)


# ============================================================
# 3) FUNCION DE TRANSFERENCIA FRACCIONARIA Y BALANCE ARMONICO
# ============================================================

def W_frac(omega, qord, p):
    """
    W_q(i*omega) = r^T ( P - lambda I )^{-1} qvec,
    lambda = (i*omega)^q = omega^q exp(i q pi/2).

    Convencion de signo:
    W_code(i omega) = r^T (P - (i omega)^q I)^(-1) q_v.
    Si se usa W_report(i omega) = r^T ((i omega)^q I - P)^(-1) q_v,
    entonces W_report = -W_code; por eso aqui k = -1 / Re(W_code).
    """
    P, qvec, r = chua_matrices(p)
    P = P.astype(complex_dtype)
    qvec = qvec.astype(complex_dtype).reshape(-1, 1)
    r = r.astype(complex_dtype).reshape(1, -1)

    lam = cpower_iw_q(omega, qord)
    M = P - lam * np.eye(3, dtype=complex_dtype)
    return (r @ np.linalg.inv(M) @ qvec)[0, 0]

def imag_W(omega, qord, p):
    return float(np.imag(W_frac(omega, qord, p)))

def find_omega_k_pairs(qord, p, wmin=WMIN, wmax=WMAX, nscan=NSCAN):
    """
    Busca todas las raíces de Im(W_q(i*omega)) = 0 en [wmin, wmax]
    y devuelve pares (omega0, k) con k = -1/Re(W_q(i*omega0)).
    """
    qord = validate_fractional_order(qord)
    ws = np.linspace(wmin, wmax, nscan)
    vals = np.array([imag_W(w, qord, p) for w in ws], dtype=float)

    roots = []
    for i in range(len(ws) - 1):
        f1, f2 = vals[i], vals[i + 1]
        if np.isnan(f1) or np.isnan(f2):
            continue

        if f1 == 0.0:
            roots.append(ws[i])
        elif f1 * f2 < 0.0:
            try:
                root = brentq(lambda w: imag_W(w, qord, p), ws[i], ws[i + 1], maxiter=500)
                roots.append(root)
            except ValueError:
                pass

    # eliminar duplicados numericos
    roots_sorted = []
    for r0 in sorted(roots):
        if not roots_sorted or abs(r0 - roots_sorted[-1]) > 1e-7:
            roots_sorted.append(r0)

    pairs = []
    for omega0 in roots_sorted:
        W0 = W_frac(omega0, qord, p)
        reW = np.real(W0)
        if abs(reW) < 1e-12:
            continue
        k = -1.0 / reW
        if is_describing_gain_compatible(k, p):
            pairs.append((float(omega0), float(k)))

    # ordenar por k creciente
    pairs.sort(key=lambda z: z[1])
    return pairs


def find_omega_k_candidates(qord, p, wmin=WMIN, wmax=WMAX, nscan=NSCAN):
    """
    Busca las raices de Im(W_q(i*omega)) = 0 y devuelve los pares crudos
    (omega0, k). No filtra por una funcion descriptiva especifica.

    Esto permite reutilizar las mismas frecuencias para la DF clasica y para
    la extension tipo Machado, donde la compatibilidad de k depende de mu.
    """
    qord = validate_fractional_order(qord)
    ws = np.linspace(wmin, wmax, nscan)
    vals = np.array([imag_W(w, qord, p) for w in ws], dtype=float)

    roots = []
    for i in range(len(ws) - 1):
        f1, f2 = vals[i], vals[i + 1]
        if np.isnan(f1) or np.isnan(f2):
            continue

        if f1 == 0.0:
            roots.append(ws[i])
        elif f1 * f2 < 0.0:
            try:
                root = brentq(lambda w: imag_W(w, qord, p), ws[i], ws[i + 1], maxiter=500)
                roots.append(root)
            except ValueError:
                pass

    roots_sorted = []
    for r0 in sorted(roots):
        if not roots_sorted or abs(r0 - roots_sorted[-1]) > 1e-7:
            roots_sorted.append(r0)

    pairs = []
    for omega0 in roots_sorted:
        W0 = W_frac(omega0, qord, p)
        reW = np.real(W0)
        if abs(reW) < 1e-12:
            continue
        pairs.append((float(omega0), float(-1.0 / reW)))

    pairs.sort(key=lambda z: z[1])
    return pairs


# ============================================================
# 4) DESCRIBING FUNCTION DE LA NO LINEALIDAD
# ============================================================

def N_sat(a, p):
    """
    Funcion descriptiva de psi(sigma).

    - piecewise: psi=A sat(sigma), A=m0-m1.
    - arctan:   psi=a2 arctan(rho sigma).
    """
    a = float(a)

    if a <= 0:
        raise ValueError("La amplitud a debe ser positiva.")

    if chua_model(p) == "arctan":
        a2 = float(p["a2"])
        rho = float(p["rho"])
        return a2 * 2.0 * (np.sqrt(1.0 + (rho * a) ** 2) - 1.0) / (rho * a * a)

    A = float(chua_gain_A(p))
    if a <= 1.0:
        return A

    return (2.0 * A / np.pi) * (np.arcsin(1.0 / a) + np.sqrt(a*a - 1.0) / (a*a))


def fourier_coefficients_psi(A, sigma0, p, K=10, n_quad=4096):
    """
    Coeficientes de Fourier de y(theta)=psi(sigma0 + A cos(theta)).

    Convencion:
        a_k = (1/pi) int_0^{2pi} y(theta) cos(k theta) dtheta
        b_k = (1/pi) int_0^{2pi} y(theta) sin(k theta) dtheta
        Y_k = a_k - i b_k

    La cuadratura es numerica y no usa derivadas, por lo que sirve para la
    saturacion no suave. El coeficiente DC se reporta como y_mean aparte.
    """
    A = float(A)
    sigma0 = float(sigma0)
    K = int(K)
    n_quad = int(n_quad)
    if A <= 0.0 or not np.isfinite(A):
        raise ValueError("A debe ser positivo y finito.")
    if K < 1:
        raise ValueError("K debe ser al menos 1.")
    if n_quad < max(64, 8 * K):
        raise ValueError("n_quad debe ser suficientemente grande para K.")

    theta = np.linspace(0.0, 2.0 * np.pi, n_quad, endpoint=False, dtype=real_dtype)
    y = np.array([psi_sigma(sigma0 + A * np.cos(th), p) for th in theta], dtype=real_dtype)
    coeffs = {}
    for k in range(1, K + 1):
        c = np.cos(k * theta)
        s = np.sin(k * theta)
        ak = 2.0 * float(np.mean(y * c))
        bk = 2.0 * float(np.mean(y * s))
        coeffs[k] = {
            "a": ak,
            "b": bk,
            "Y": complex_dtype(ak - 1j * bk),
        }
    return {
        "A": A,
        "sigma0": sigma0,
        "K": K,
        "n_quad": n_quad,
        "y_mean": float(np.mean(y)),
        "coefficients": coeffs,
    }


def N_biased(A, sigma0, p, K=10, n_quad=4096):
    """
    Funcion descriptiva sesgada N(A,sigma0)=Y_1/A.
    """
    data = fourier_coefficients_psi(A, sigma0, p, K=max(1, int(K)), n_quad=n_quad)
    return complex_dtype(data["coefficients"][1]["Y"] / float(A))


def machado_complex_power(N, mu, branch=0, eps=1e-12):
    """
    Potencia compleja para la familia tipo Machado:
        N_mu = exp(mu Log_branch(N)).

    branch=0 es la rama principal, con argumento en (-pi, pi]. Si |N| es muy
    pequeno se rechaza para evitar elevar ruido numerico a potencias
    fraccionarias.
    """
    z = complex(N)
    mu = float(mu)
    branch = int(branch)
    if abs(z) < float(eps):
        raise ValueError("N esta demasiado cerca de cero para la potencia Machado.")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu debe ser positivo.")
    log_branch = np.log(abs(z)) + 1j * (np.angle(z) + 2.0 * np.pi * branch)
    return complex_dtype(np.exp(mu * log_branch))


def reconstruct_biased_lure_seed(qord, p, A, sigma0, omega, theta=0.0, K=10, n_quad=4096):
    """
    Reconstruccion minima de semilla sesgada consistente con sigma=r^T X.

    Se usa la convencion fraccionaria de frecuencia lambda=(i omega)^q. Para
    el termino DC se resuelve por minimos cuadrados:
        P Xbar + b y_mean = 0,   r^T Xbar = sigma0.
    Para el primer armonico se resuelve:
        (lambda I - P) V = b Y_1,   r^T V = A.

    La restriccion r^T X evita asumir que sigma0 coincide con una coordenada
    salvo que r=e1, aunque en este Chua no suave efectivamente r=e1.
    """
    qord = validate_fractional_order(qord)
    fourier = fourier_coefficients_psi(A, sigma0, p, K=max(1, int(K)), n_quad=n_quad)
    Y1 = complex(fourier["coefficients"][1]["Y"])
    y_mean = float(fourier["y_mean"])
    P, qvec, r = chua_matrices(p)

    lhs_dc = np.vstack([P, r.reshape(1, -1)]).astype(real_dtype)
    rhs_dc = np.concatenate([-qvec * y_mean, np.array([float(sigma0)], dtype=real_dtype)])
    xbar, *_ = np.linalg.lstsq(lhs_dc, rhs_dc, rcond=None)

    lam = cpower_iw_q(omega, qord)
    lhs_h = np.vstack([
        (lam * np.eye(3, dtype=complex_dtype) - P.astype(complex_dtype)),
        r.astype(complex_dtype).reshape(1, -1),
    ])
    rhs_h = np.concatenate([
        qvec.astype(complex_dtype) * Y1,
        np.array([complex(A, 0.0)], dtype=complex_dtype),
    ])
    V, *_ = np.linalg.lstsq(lhs_h, rhs_h, rcond=None)
    seed = np.asarray(xbar, dtype=real_dtype) + np.real(V * np.exp(1j * float(theta)))
    return seed.astype(real_dtype), np.asarray(xbar, dtype=real_dtype), V.astype(complex_dtype), fourier

def is_describing_gain_compatible(k, p):
    k = float(k)
    if chua_model(p) == "arctan":
        a2 = float(p["a2"])
        rho = float(p["rho"])
        return np.sign(k) == np.sign(a2) and 0.0 < abs(k) < abs(a2) * rho
    A = float(chua_gain_A(p))
    return np.sign(k) == np.sign(A) and 0.0 < abs(k) <= abs(A) + 1e-10


def N_sat_machado(a, p, mu):
    """
    Funcion descriptiva fraccionaria auxiliar tipo Machado.

    Para el Chua no suave descrito en docs/reporte_unificado_chua_fraccionario.tex se tiene
    N_psi(a)>0 y se usa N_{psi,mu}(a) = N_psi(a)^mu, mu>0.
    """
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("El orden descriptivo auxiliar mu debe ser positivo.")
    if chua_model(p) != "piecewise":
        raise ValueError("La extension tipo Machado implementada aqui aplica al modelo piecewise/no suave.")
    base = float(N_sat(a, p))
    if base <= 0.0:
        raise ValueError("N_psi(a) debe ser positivo para usar la rama real de Machado.")
    return float(base ** mu)


def is_machado_gain_compatible(k, p, mu):
    if chua_model(p) != "piecewise":
        return False
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        return False
    k = float(k)
    A = float(chua_gain_A(p))
    if A <= 0.0:
        return False
    return 0.0 < k <= A ** mu + 1e-10


def solve_amplitude_from_k(k, p, amin=1.0 + 1e-9, amax=100.0, nscan=20000):
    """
    Resuelve N(a0)=k.
    """
    k = float(k)

    if chua_model(p) == "arctan":
        if not is_describing_gain_compatible(k, p):
            raise RuntimeError("La ganancia k no es compatible con la funcion descriptiva arctan.")
        a2 = float(p["a2"])
        rho = float(p["rho"])
        a2_sq = 4.0 * a2 * (a2 * rho - k) / (k * k * rho)
        if a2_sq <= 0.0:
            raise RuntimeError("La amplitud arctan calculada no es real positiva.")
        return float(np.sqrt(a2_sq))

    # Caso a <= 1: N(a)=A. Si k ~ A, devolvemos a cercano a 1.
    A = float(chua_gain_A(p))
    if abs(k - A) < 1e-10:
        return 1.0

    def f(a):
        return N_sat(a, p) - k

    grid = np.linspace(amin, amax, nscan)
    vals = np.array([f(a) for a in grid], dtype=float)

    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            return float(grid[i])
        if vals[i] * vals[i + 1] < 0.0:
            return float(brentq(f, grid[i], grid[i + 1], maxiter=500))

    raise RuntimeError("No se encontró amplitud a0 tal que N(a0)=k.")


def solve_machado_amplitude_from_k(k, p, mu, amin=1.0 + 1e-9, amax=100.0, nscan=20000):
    """
    Resuelve N_psi(a)^mu = k para la extension tipo Machado.

    Equivalentemente, N_psi(a) = k^(1/mu). La ganancia k viene de la misma
    condicion de Nyquist; mu solo desplaza la amplitud candidata.
    """
    if not is_machado_gain_compatible(k, p, mu):
        raise RuntimeError("La ganancia k no es compatible con la funcion descriptiva tipo Machado.")

    mu = float(mu)
    k = float(k)
    A = float(chua_gain_A(p))
    target = float(k ** (1.0 / mu))

    if abs(target - A) < 1e-10:
        return 1.0
    if not (0.0 < target < A):
        raise RuntimeError("La amplitud Machado no tiene solucion real en la rama saturada.")

    def f(a):
        return N_sat(a, p) - target

    grid = np.linspace(amin, amax, nscan)
    vals = np.array([f(a) for a in grid], dtype=float)

    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            return float(grid[i])
        if vals[i] * vals[i + 1] < 0.0:
            return float(brentq(f, grid[i], grid[i + 1], maxiter=500))

    raise RuntimeError("No se encontro amplitud a_mu tal que N_psi(a_mu)^mu=k.")


def solve_amplitude_for_describing_method(k, p, method="classic", mu=1.0):
    method = str(method).strip().lower()
    if method in {"classic", "clasica", "classical"}:
        return solve_amplitude_from_k(k, p)
    if method in {"machado", "fdf", "fractional"}:
        return solve_machado_amplitude_from_k(k, p, mu)
    raise ValueError("method debe ser 'classic' o 'machado'.")


# ============================================================
# 5) SEMILLA ARMONICA FRACCIONARIA
# ============================================================

def build_fractional_seed(qord, p, omega0, k, a0, theta=0.0):
    """
    Construye la semilla armónica aproximada para el sistema fraccionario.

    Se resuelve:
        (P0 - (i*omega0)^q I) v = 0
    y se normaliza con r^T v = 1.

    Luego:
        x_seed(theta) = a0 * Re(v * exp(i*theta))
    """
    qord = validate_fractional_order(qord)
    P0 = build_P0(p, k).astype(complex_dtype)
    lam0 = cpower_iw_q(omega0, qord)

    eigvals, eigvecs = np.linalg.eig(P0)
    idx = np.argmin(np.abs(eigvals - lam0))
    v = eigvecs[:, idx]

    if abs(v[0]) < 1e-14:
        raise RuntimeError("No se pudo normalizar el autovector con r^T v = 1.")

    v = v / v[0]
    x_seed = float(a0) * np.real(v * np.exp(1j * float(theta)))

    return x_seed.astype(real_dtype), v, eigvals[idx]


# ============================================================
# 6) EFORK (ADAPTADO A CHUA)
# ============================================================

def memory_fractional_component(k, t, arr, vtn, h, alpha, nu):
    start = max(0, k - nu)
    sum_ = real_dtype(0.0)
    gamma_term = gamma(2.0 - float(alpha))

    for j in range(start, k):
        t0 = vtn[j]
        t1 = vtn[j + 1]
        v1 = (t - t0) ** (1.0 - float(alpha))
        v2 = (t - t1) ** (1.0 - float(alpha))
        diff = (arr[j + 1] - arr[j])
        sum_ += diff * (v1 - v2)

    return real_dtype(sum_ / (float(h) * gamma_term))

def extract_memory_window(traj, Lm, h):
    """
    Extrae una ventana de prehistoria compatible con EFORK.

    Propósito matemático:
    Transportar la memoria truncada de una integración fraccionaria de Caputo
    entre etapas de continuación. En EFORK con longitud de memoria Lm, la
    dinámica en el siguiente tramo no queda determinada solo por el último
    punto, sino por una ventana discreta de historia reciente.

    Ecuaciones usadas:
    No resuelve una ecuación nueva; selecciona los últimos nu = ceil(Lm/h)
    subintervalos de una trayectoria [t, x, y, z] para que el término de
    memoria usado por memory_fractional_component pueda evaluarse con esos
    incrementos históricos.

    Parámetros de entrada:
    traj:
        Arreglo con columnas [t, x, y, z].
    Lm:
        Longitud de memoria truncada usada por EFORK.
    h:
        Paso temporal.

    Salida:
    Arreglo [t, x, y, z] con tiempos desplazados para terminar en t=0.

    Advertencias sobre validez:
    Esta es una aproximación de memoria corta. No convierte la continuación en
    una prueba de existencia de órbitas periódicas de Caputo; solo evita el
    reinicio artificial de la memoria causal en cada tramo.
    """
    traj = np.asarray(traj, dtype=real_dtype)
    if traj.ndim != 2 or traj.shape[1] != 4 or traj.shape[0] < 1:
        raise ValueError("traj debe tener forma (N, 4) con columnas [t, x, y, z].")
    if float(h) <= 0.0:
        raise ValueError("El paso h debe ser positivo.")
    if float(Lm) <= 0.0:
        raise ValueError("La longitud de memoria Lm debe ser positiva.")

    nu = max(1, int(np.ceil(float(Lm) / float(h))))
    window = traj[-min(traj.shape[0], nu + 1):].copy()
    window[:, 0] -= window[-1, 0]
    return window.astype(real_dtype)


def _prepare_efork_history(x0, history, h, Lm):
    x0 = np.asarray(x0, dtype=real_dtype)
    if x0.shape != (3,):
        raise ValueError("x0 debe ser un vector de dimensión 3.")
    if float(h) <= 0.0:
        raise ValueError("El paso h debe ser positivo.")
    if float(Lm) <= 0.0:
        raise ValueError("La longitud de memoria Lm debe ser positiva.")

    if history is None:
        return np.array([[0.0, x0[0], x0[1], x0[2]]], dtype=real_dtype)

    hist = np.asarray(history, dtype=real_dtype)
    if hist.ndim != 2 or hist.shape[0] < 1 or hist.shape[1] not in (3, 4):
        raise ValueError("history debe tener forma (N, 3) o (N, 4).")

    if hist.shape[1] == 3:
        states = hist.copy()
        times = (np.arange(hist.shape[0], dtype=real_dtype) - (hist.shape[0] - 1)) * float(h)
    else:
        times = hist[:, 0].copy()
        states = hist[:, 1:4].copy()
        if hist.shape[0] > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("Los tiempos de history deben ser estrictamente crecientes.")
        times -= times[-1]

    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(times)):
        raise ValueError("history contiene valores no finitos.")
    if times.shape[0] > 1 and not np.allclose(np.diff(times), float(h), rtol=1e-7, atol=1e-10):
        raise ValueError("history debe estar muestreada con el mismo paso h.")

    nu = max(1, int(np.ceil(float(Lm) / float(h))))
    if times.shape[0] > nu + 1:
        times = times[-(nu + 1):]
        states = states[-(nu + 1):]
        times -= times[-1]
    times[-1] = 0.0

    return np.column_stack((times, states)).astype(real_dtype)


def efork3_integrate(rhs, x0, qord, h, Lm, t_f, history=None, return_full_history=False):
    """
    EFORK explícito con memoria truncada para el sistema tridimensional.

    Propósito matemático:
    Integrar una ecuación fraccionaria causal tipo Caputo usando una
    formulación EFORK con memoria finita. Si se entrega history, el tramo
    nuevo arranca desde una prehistoria discreta, no desde un único punto.

    Ecuaciones usadas:
    El término de memoria se aproxima por memory_fractional_component, que
    suma incrementos históricos en la ventana nu = ceil(Lm/h). Los coeficientes
    EFORK dependen de Gamma(1+q), Gamma(1+2q) y Gamma(1+3q).

    Parámetros de entrada:
    rhs:
        Campo vectorial f(x) del sistema continuado u original.
    x0:
        Estado inicial si no se proporciona history.
    qord:
        Orden fraccionario, 0 < qord <= 1.
    h, Lm, t_f:
        Paso, longitud de memoria truncada y tiempo final del nuevo tramo.
    history:
        Ventana opcional con columnas [t, x, y, z] terminando en el estado
        inicial. Los tiempos se desplazan internamente para terminar en t=0.
    return_full_history:
        Si es True, devuelve también la prehistoria con tiempos negativos.

    Salida:
    Arreglo con columnas [t, x, y, z]. Por defecto contiene solo el tramo nuevo
    desde t=0; con return_full_history=True incluye la prehistoria.

    Advertencias sobre validez:
    La memoria se trunca a Lm. Al transportar history en continuación se hace
    una homotopía numérica con memoria corta, no una equivalencia exacta entre
    sistemas de Caputo con parámetros distintos.
    """
    qord = validate_fractional_order(qord)
    if float(t_f) < 0.0:
        raise ValueError("t_f debe ser no negativo.")

    hist = _prepare_efork_history(x0, history, h, Lm)
    N1 = int(np.ceil(float(t_f / h)))
    nu = max(1, int(np.ceil(float(Lm / h))))
    ha = float(h) ** float(qord)

    gamma1 = gamma(1.0 + float(qord))
    gamma2 = gamma(1.0 + 2.0 * float(qord))
    gamma3 = gamma(1.0 + 3.0 * float(qord))

    c2 = (1.0 / (2.0 * gamma1)) ** (1.0 / float(qord))
    c3 = (1.0 / (4.0 * gamma1)) ** (1.0 / float(qord))
    a21 = 1.0 / (2.0 * gamma1 * gamma1)
    a31 = (gamma1**2 * gamma(2.0 * float(qord) + 1.0)
           + 2.0 * gamma(2.0 * float(qord) + 1.0)**2
           - gamma(3.0 * float(qord) + 1.0)) / (
           4.0 * gamma1**2 * (2.0 * gamma(2.0 * float(qord) + 1.0)**2
           - gamma(3.0 * float(qord) + 1.0)))
    a32 = - gamma(2.0 * float(qord) + 1.0) / (
           4.0 * (2.0 * gamma(2.0 * float(qord) + 1.0)**2
           - gamma(3.0 * float(qord) + 1.0)))

    w1 = (8.0 * gamma1**3 * gamma2**2
          - 6.0 * gamma1**3 * gamma3
          + gamma2 * gamma3) / (gamma1 * gamma2 * gamma3)
    w2 = 2.0 * gamma1**2 * (4.0 * gamma2**2 - gamma3) / (gamma2 * gamma3)
    w3 = -8.0 * gamma1**2 * (2.0 * gamma2**2 - gamma3) / (gamma2 * gamma3)

    hist_len = hist.shape[0]
    start_idx = hist_len - 1
    total_len = hist_len + N1
    vtn = np.zeros(total_len, dtype=real_dtype)
    vxn = np.zeros(total_len, dtype=real_dtype)
    vyn = np.zeros(total_len, dtype=real_dtype)
    vzn = np.zeros(total_len, dtype=real_dtype)

    vtn[:hist_len] = hist[:, 0]
    vxn[:hist_len] = hist[:, 1]
    vyn[:hist_len] = hist[:, 2]
    vzn[:hist_len] = hist[:, 3]

    x_n, y_n, z_n = vxn[start_idx], vyn[start_idx], vzn[start_idx]

    for n in range(start_idx, start_idx + N1):
        tn = vtn[n]

        if n == 0:
            mem_x = mem_y = mem_z = real_dtype(0.0)
        else:
            mem_x = memory_fractional_component(n, tn, vxn, vtn, h, qord, nu)
            mem_y = memory_fractional_component(n, tn, vyn, vtn, h, qord, nu)
            mem_z = memory_fractional_component(n, tn, vzn, vtn, h, qord, nu)

        f1 = rhs(np.array([x_n, y_n, z_n], dtype=real_dtype))
        f1x, f1y, f1z = f1[0] - mem_x, f1[1] - mem_y, f1[2] - mem_z
        K1x, K1y, K1z = ha * f1x, ha * f1y, ha * f1z

        f2 = rhs(np.array([x_n + a21 * K1x,
                           y_n + a21 * K1y,
                           z_n + a21 * K1z], dtype=real_dtype))
        t2 = tn + c2 * float(h)
        mem2_x = memory_fractional_component(n, t2, vxn, vtn, h, qord, nu) if n > 0 else real_dtype(0.0)
        mem2_y = memory_fractional_component(n, t2, vyn, vtn, h, qord, nu) if n > 0 else real_dtype(0.0)
        mem2_z = memory_fractional_component(n, t2, vzn, vtn, h, qord, nu) if n > 0 else real_dtype(0.0)
        K2x, K2y, K2z = ha * (f2[0] - mem2_x), ha * (f2[1] - mem2_y), ha * (f2[2] - mem2_z)

        f3 = rhs(np.array([x_n + a31 * K1x + a32 * K2x,
                           y_n + a31 * K1y + a32 * K2y,
                           z_n + a31 * K1z + a32 * K2z], dtype=real_dtype))
        t3 = tn + c3 * float(h)
        mem3_x = memory_fractional_component(n, t3, vxn, vtn, h, qord, nu) if n > 0 else real_dtype(0.0)
        mem3_y = memory_fractional_component(n, t3, vyn, vtn, h, qord, nu) if n > 0 else real_dtype(0.0)
        mem3_z = memory_fractional_component(n, t3, vzn, vtn, h, qord, nu) if n > 0 else real_dtype(0.0)
        K3x, K3y, K3z = ha * (f3[0] - mem3_x), ha * (f3[1] - mem3_y), ha * (f3[2] - mem3_z)

        x_n1 = x_n + w1 * K1x + w2 * K2x + w3 * K3x
        y_n1 = y_n + w1 * K1y + w2 * K2y + w3 * K3y
        z_n1 = z_n + w1 * K1z + w2 * K2z + w3 * K3z

        vtn[n + 1], vxn[n + 1], vyn[n + 1], vzn[n + 1] = (
            vtn[n] + float(h), x_n1, y_n1, z_n1
        )
        x_n, y_n, z_n = x_n1, y_n1, z_n1

    full = np.column_stack((vtn, vxn, vyn, vzn)).astype(np.float64)
    if return_full_history:
        return full

    segment = full[start_idx:start_idx + N1 + 1].copy()
    segment[:, 0] -= segment[0, 0]
    return segment.astype(np.float64)


# ============================================================
# 7) CONTINUACION NUMERICA EN EPSILON
# ============================================================

def continuation_in_epsilon(p, qord, k, x0_seed, eps_values,
                            h=H, Lm=LM, t_transient=T_TRANSIENT, t_keep=T_KEEP,
                            memory_mode="window", memory_update_source="transient"):
    """
    Continúa la familia en epsilon transportando una semilla fraccionaria.

    Propósito matemático:
    Llevar una semilla armónica de la familia linealizada/no lineal parcial
    hacia el sistema objetivo epsilon=1. En modo "window" se transporta una
    ventana de memoria truncada para respetar que la dinámica de Caputo no
    está determinada solo por el último estado.

    Ecuaciones usadas:
    Integra ^C D_t^q x = P0 x + epsilon qvec delta(r^T x) con EFORK. Para cada
    epsilon se usa un tramo transitorio y luego un tramo de observación.

    Parámetros de entrada:
    p, qord, k:
        Parámetros del sistema, orden fraccionario y ganancia de cierre DF.
    x0_seed:
        Semilla inicial producida por balance armónico/Weyl.
    eps_values:
        Secuencia de valores de epsilon para la homotopía.
    h, Lm, t_transient, t_keep:
        Paso, memoria truncada y tiempos de integración.
    memory_mode:
        "point" usa solo x_out como dato inicial. "window" transporta los
        últimos ceil(Lm/h)+1 puntos.
    memory_update_source:
        "transient" pasa al siguiente epsilon la salida del transitorio.
        "observed" pasa la ventana al final del tramo de observación t_keep.
        La continuación antigua se reproduce con memory_mode="point" y
        memory_update_source="transient".

    Salida:
    Lista de diccionarios por epsilon con estado de entrada, estado posterior
    al transitorio y trayectoria de observación.

    Advertencias sobre validez:
    La continuación en epsilon es una herramienta heurística para producir
    semillas. La existencia del atractor en Caputo debe validarse después con
    integración causal y análisis de cuencas.
    """
    qord = validate_fractional_order(qord)
    memory_mode = str(memory_mode).strip().lower()
    if memory_mode not in {"point", "window"}:
        raise ValueError("memory_mode debe ser 'point' o 'window'.")
    memory_update_source = str(memory_update_source).strip().lower()
    if memory_update_source not in {"transient", "observed"}:
        raise ValueError("memory_update_source debe ser 'transient' u 'observed'.")

    x_in = np.asarray(x0_seed, dtype=real_dtype).copy()
    memory_history = None
    results = []

    for eps in eps_values:
        rhs = lambda x, ee=eps: rhs_epsilon_family(x, p, k=k, eps=ee)

        use_window = memory_mode == "window" and memory_history is not None
        xt_full = efork3_integrate(
            rhs,
            x_in,
            qord=qord,
            h=h,
            Lm=Lm,
            t_f=t_transient,
            history=memory_history if use_window else None,
            return_full_history=use_window,
        )
        if use_window:
            xt = xt_full[xt_full[:, 0] >= -1e-12].copy()
            xt[:, 0] -= xt[0, 0]
            memory_out = extract_memory_window(xt_full, Lm=Lm, h=h)
        else:
            xt = xt_full
            memory_out = extract_memory_window(xt, Lm=Lm, h=h) if memory_mode == "window" else None

        x_transient_out = xt[-1, 1:4].astype(real_dtype)
        keep_history = memory_out if memory_mode == "window" else None
        if keep_history is None:
            xa = efork3_integrate(rhs, x_transient_out, qord=qord, h=h, Lm=Lm, t_f=t_keep)
            memory_keep = None
        else:
            xa_full = efork3_integrate(
                rhs,
                x_transient_out,
                qord=qord,
                h=h,
                Lm=Lm,
                t_f=t_keep,
                history=keep_history,
                return_full_history=True,
            )
            xa = xa_full[xa_full[:, 0] >= -1e-12].copy()
            xa[:, 0] -= xa[0, 0]
            memory_keep = extract_memory_window(xa_full, Lm=Lm, h=h)

        if memory_update_source == "observed":
            x_out = xa[-1, 1:4].astype(real_dtype)
            next_memory = memory_keep if memory_keep is not None else (
                extract_memory_window(xa, Lm=Lm, h=h) if memory_mode == "window" else None
            )
        else:
            x_out = x_transient_out
            next_memory = memory_out

        results.append({
            "eps": float(eps),
            "x_in": x_in.astype(np.float64).copy(),
            "x_transient_out": x_transient_out.astype(np.float64).copy(),
            "x_out": x_out.astype(np.float64).copy(),
            "traj": xa.copy(),
            "memory_mode": memory_mode,
            "memory_update_source": memory_update_source,
            "history_points_in": int(0 if memory_history is None else memory_history.shape[0]),
            "history_points_out": int(0 if next_memory is None else next_memory.shape[0]),
        })

        x_in = x_out.copy()
        if memory_mode == "window":
            memory_history = next_memory.copy()

    return results


# ============================================================
# 8) GRAFICAS
# ============================================================

def plot_initial_condition_progress(results):
    eps = np.array([r["eps"] for r in results], dtype=float)
    Xin = np.array([r["x_in"] for r in results], dtype=float)
    Xout = np.array([r["x_out"] for r in results], dtype=float)

    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    labels = ["x", "y", "z"]

    for i in range(3):
        axs[i].plot(eps, Xin[:, i], "o-", label="CI del paso")
        axs[i].plot(eps, Xout[:, i], "x--", label="Estado final del paso")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()

    axs[-1].set_xlabel(r"$\epsilon$")
    fig.suptitle("Avance de las condiciones iniciales en la continuación numérica")
    plt.tight_layout()
    plt.show()

def plot_selected_attractors_same_figure(results):
    if len(results) == 0:
        return

    idxs = [0]
    if len(results) > 1:
        idxs.append(1)
    if len(results) > 2:
        idxs.append(len(results) - 1)

    idxs = list(dict.fromkeys(idxs))
    labels = ["Primer atractor", "Segundo atractor", "Último atractor"]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label, idx in zip(labels, idxs):
        T = results[idx]["traj"]
        eps = results[idx]["eps"]
        xin = results[idx]["x_in"]

        ax.plot(T[:, 1], T[:, 2], T[:, 3], lw=0.9, label=f"{label} ($\\epsilon$={eps:.2f})")
        ax.scatter(xin[0], xin[1], xin[2], s=30, marker="o")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Primer, segundo y último atractor de la continuación")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_last_attractor_only(results):
    if len(results) == 0:
        return

    T = results[-1]["traj"]
    eps_last = results[-1]["eps"]
    x_in_last = results[-1]["x_in"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(T[:, 1], T[:, 2], T[:, 3], lw=0.9, label=f"Último atractor ($\\epsilon$={eps_last:.2f})")
    ax.scatter(x_in_last[0], x_in_last[1], x_in_last[2], s=35, marker="o", label="CI del último paso")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Último atractor encontrado")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 9) REPORTE EN CONSOLA
# ============================================================

def print_harmonic_report(pairs, p, qord):
    print("\n=== BALANCE ARMONICO FRACCIONARIO ===")
    print(f"q = {qord:.7f}")
    print("Ramas encontradas (omega0, k, a0):")
    for i, (omega0, k) in enumerate(pairs):
        a0 = solve_amplitude_from_k(k, p)
        print(f"  rama {i}: omega0 = {omega0:.10f}, k = {k:.10f}, a0 = {a0:.10f}")

def print_continuation_summary(results):
    print("\n=== CONTINUACION EN EPSILON ===")
    print("eps\t\t x_in\t\t\t\t\t x_out")
    for r in results:
        xi = r["x_in"]
        xo = r["x_out"]
        print(
            f"{r['eps']:.2f}\t"
            f"[{xi[0]: .6f}, {xi[1]: .6f}, {xi[2]: .6f}]\t"
            f"[{xo[0]: .6f}, {xo[1]: .6f}, {xo[2]: .6f}]"
        )


# ============================================================
# 10) MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # A) localizar (omega0, k) del sistema fraccionario
    # --------------------------------------------------------
    pairs = find_omega_k_pairs(QORD, PARAMS, wmin=WMIN, wmax=WMAX, nscan=NSCAN)
    if len(pairs) == 0:
        raise RuntimeError("No se encontraron raíces para Im(W_q(i*omega))=0.")

    print_harmonic_report(pairs, PARAMS, QORD)

    if BRANCH_INDEX < 0 or BRANCH_INDEX >= len(pairs):
        raise ValueError("BRANCH_INDEX fuera de rango.")

    omega0, k0 = pairs[BRANCH_INDEX]
    a0 = solve_amplitude_from_k(k0, PARAMS)

    # --------------------------------------------------------
    # B) semilla armónica fraccionaria
    # --------------------------------------------------------
    xseed, v, eig_match = build_fractional_seed(QORD, PARAMS, omega0, k0, a0)

    print("\n=== SEMILLA FRACCIONARIA ===")
    print(f"Rama elegida   : {BRANCH_INDEX}")
    print(f"omega0         : {omega0:.12f}")
    print(f"k0             : {k0:.12f}")
    print(f"a0             : {a0:.12f}")
    print(f"autovalor match: {eig_match}")
    print(f"autovector v   : {v}")
    print(f"xseed          : {xseed}")

    # --------------------------------------------------------
    # C) continuación en epsilon
    # --------------------------------------------------------
    results = continuation_in_epsilon(
        p=PARAMS,
        qord=QORD,
        k=k0,
        x0_seed=xseed,
        eps_values=EPS_VALUES,
        h=H,
        Lm=LM,
        t_transient=T_TRANSIENT,
        t_keep=T_KEEP,
        memory_mode="window",
        memory_update_source="observed"
    )

    print_continuation_summary(results)

    # --------------------------------------------------------
    # D) gráficas solicitadas
    # --------------------------------------------------------
    plot_initial_condition_progress(results)
    plot_selected_attractors_same_figure(results)
    plot_last_attractor_only(results)
