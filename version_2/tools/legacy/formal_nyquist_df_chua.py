#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formal_nyquist_df_chua.py

Análisis formal en frecuencia para el sistema de Chua fraccionario en forma de Lur'e:

    ^C D_t^q X = P X + qvec * psi(r^T X),   0 < q < 1,

con la no linealidad saturación que aparece en la versión no suave de Chua.

Objetivo
--------
1) Calcular equilibrios y su estabilidad local mediante el criterio angular fraccionario.
2) Construir la transferencia fraccionaria W_q(iω).
3) Aplicar función descriptiva a la saturación y localizar intersecciones Nyquist
   / -1/N(a), que son candidatos de balance armónico.
4) Construir una semilla oscilatoria para usarla después en continuación numérica.

Referencias matemáticas implementadas
-------------------------------------
[1] M.-F. Danca, "Hidden chaotic attractors in fractional-order systems",
    Nonlinear Dynamics, 2018. Ecuaciones del Chua fraccionario no suave,
    estabilidad local de equilibrios e índice de inestabilidad.
[2] P. Vigué, C. Vergez, B. Lombard, B. Cochelin,
    "Continuation of periodic solutions for systems with fractional derivatives",
    Nonlinear Dynamics, 2019. Uso de derivada de Weyl para periodicidad exacta.
[3] P.-E. Haacker et al.,
    "Hill-Type Stability Analysis of Periodic Solutions of Fractional-Order
    Differential Equations", arXiv:2509.24639. Puente Weyl-Caputo y límites del
    ansatz exponencial del lado estable.
[4] A. Gelb, W.E. Vander Velde,
    "Multiple-Input Describing Functions and Nonlinear System Design", McGraw-Hill.
[5] R.S. Barbosa, J.A.T. Machado,
    trabajos sobre función descriptiva y sistemas fraccionarios.

Observación importante
----------------------
Este script NO prueba la existencia de un ciclo límite exacto del sistema autónomo
con derivada de Caputo. Lo que produce es una oscilación dominante candidata en el
marco de balance armónico/Weyl, útil como semilla para continuación e integración
numérica del sistema real de Caputo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# 0) CONFIGURACIÓN
# ============================================================
CONFIG = {
    "params": {
        # Set I de Chua oculto entero/fraccionario usado en Danca 2018
        "alpha_chua": 8.4562,
        "beta": 12.0732,
        "gamma_chua": 0.0052,
        "m0": -0.1768,
        "m1": -1.1468,
    },
    "frac_order": 0.985,
    "omega_scan": {
        "wmin": 1e-4,
        "wmax": 10.0,
        "nscan": 25000,
    },
    "amplitude_scan": {
        "amin": 1.0 + 1e-9,
        "amax": 100.0,
        "nscan": 25000,
    },
    "continuation": {
        "eps_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "branch_index": 0,
    },
    "outputs": {
        "json": "nyquist_df_summary.json",
        "png_nyquist": "nyquist_df_chua.png",
        "png_seed": "seed_family_chua.png",
    },
}

real_t = np.float64
complex_t = np.complex128


# ============================================================
# 1) SISTEMA DE CHUA FRACCIONARIO NO SUAVE
# ============================================================
def sat_scalar(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def sat_vec(x):
    x = np.asarray(x, dtype=real_t)
    return np.clip(x, -1.0, 1.0)


def f_chua(x: float, p: dict) -> float:
    """No linealidad piecewise-linear de Chua."""
    m0 = float(p["m0"])
    m1 = float(p["m1"])
    return m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))


def chua_equilibria(p: dict) -> dict[str, np.ndarray]:
    """
    Equilibrios del Chua no suave de Danca.
    Se implementa directamente desde la ecuación escalar de equilibrio.
    """
    a = float(p["alpha_chua"])
    beta = float(p["beta"])
    gamma = float(p["gamma_chua"])
    m0 = float(p["m0"])
    m1 = float(p["m1"])

    eqs: dict[str, np.ndarray] = {"E0": np.array([0.0, 0.0, 0.0], dtype=real_t)}

    s = -beta / (beta + gamma)
    den = m1 - s
    if abs(den) < 1e-14:
        return eqs

    # x* resuelve h(x*) + beta/(beta+gamma) x* = 0 en la rama |x|>1
    x_plus = -(m0 - m1) / den
    x_minus = (m0 - m1) / den

    for lab, xx in (("E+", x_plus), ("E-", x_minus)):
        if abs(xx) > 1.0:
            fx = f_chua(xx, p)
            yy = xx + fx
            zz = fx
            eqs[lab] = np.array([xx, yy, zz], dtype=real_t)

    return eqs


def jacobian_at_equilibrium(eq: np.ndarray, p: dict) -> np.ndarray:
    """
    Jacobiano por tramos:
    - si |x*|<1, pendiente m0;
    - si |x*|>1, pendiente m1.
    Fórmula consistente con Danca 2018.
    """
    a = float(p["alpha_chua"])
    beta = float(p["beta"])
    gamma = float(p["gamma_chua"])
    m0 = float(p["m0"])
    m1 = float(p["m1"])

    x = float(eq[0])
    slope = m0 if abs(x) < 1.0 else m1
    return np.array([
        [-a - a * slope, a, 0.0],
        [1.0, -1.0, 1.0],
        [0.0, -beta, -gamma],
    ], dtype=real_t)


def instability_measure(eigs: np.ndarray, qord: float) -> float:
    """
    iota = q - 2*alpha_min/pi, tal como se usa en Danca 2018.
    """
    alpha_min = np.min(np.abs(np.angle(eigs)))
    return float(qord - 2.0 * alpha_min / np.pi)


# ============================================================
# 2) FORMA DE LUR'E Y TRANSFERENCIA FRACCIONARIA
# ============================================================
def chua_lure_matrices(p: dict):
    """
    Forma de Lur'e con pendiente externa m1:

        ^C D_t^q X = P X + qvec * psi(r^T X),

    psi(sigma) = (m0 - m1) sat(sigma).
    """
    a = float(p["alpha_chua"])
    beta = float(p["beta"])
    gamma = float(p["gamma_chua"])
    m1 = float(p["m1"])

    P = np.array([
        [-a * (1.0 + m1), a, 0.0],
        [1.0, -1.0, 1.0],
        [0.0, -beta, -gamma],
    ], dtype=real_t)

    qvec = np.array([-a, 0.0, 0.0], dtype=real_t)
    r = np.array([1.0, 0.0, 0.0], dtype=real_t)
    return P, qvec, r


def chua_gain_A(p: dict) -> float:
    return float(p["m0"] - p["m1"])


def validate_fractional_order(qord: float) -> float:
    qord = float(qord)
    if not np.isfinite(qord) or not (0.0 < qord <= 1.0):
        raise ValueError("El orden fraccionario q debe cumplir 0 < q <= 1.")
    return qord


def cpower_iw_q(omega: float, qord: float) -> complex:
    """(iω)^q en la rama principal."""
    qord = validate_fractional_order(qord)
    return complex((omega ** qord) * np.exp(1j * np.pi * qord / 2.0))


def W_frac(omega: float, qord: float, p: dict) -> complex:
    """
    W_q(iω) = r^T (P - (iω)^q I)^(-1) qvec

    Nota: esta convención de signo es consistente con el archivo chua_initial_cond.py
    adjunto por la usuaria. Si se usa la convención alternativa, la ecuación de cierre
    cambia de signo, pero no el procedimiento.
    """
    P, qvec, r = chua_lure_matrices(p)
    P = P.astype(complex_t)
    qvec = qvec.astype(complex_t).reshape(-1, 1)
    r = r.astype(complex_t).reshape(1, -1)
    lam = cpower_iw_q(float(omega), float(qord))
    M = P - lam * np.eye(3, dtype=complex_t)
    return (r @ np.linalg.inv(M) @ qvec)[0, 0]


# ============================================================
# 3) FUNCIÓN DESCRIPTIVA DE LA SATURACIÓN
# ============================================================
def N_sat(a: float, p: dict) -> float:
    """
    Función descriptiva de psi(sigma)=A sat(sigma), con A = m0-m1.

    Para 0<a<=1: N(a)=A.
    Para a>1:
        N(a)=(2A/pi) * [ asin(1/a) + sqrt(a^2-1)/a^2 ].
    """
    A = chua_gain_A(p)
    a = float(a)
    if a <= 0:
        raise ValueError("La amplitud debe ser positiva.")
    if a <= 1.0:
        return A
    return (2.0 * A / np.pi) * (np.arcsin(1.0 / a) + np.sqrt(a * a - 1.0) / (a * a))


def solve_amplitude_from_k(k: float, p: dict, amin: float, amax: float, nscan: int) -> float:
    """
    Resuelve N(a)=k para a>1. Si k≈A y a<=1 es admisible, devuelve a=1.
    """
    A = chua_gain_A(p)
    if abs(k - A) < 1e-10:
        return 1.0

    def f(a):
        return N_sat(a, p) - k

    grid = np.linspace(amin, amax, int(nscan))
    vals = np.array([f(x) for x in grid], dtype=float)

    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            return float(grid[i])
        if vals[i] * vals[i + 1] < 0.0:
            return float(brentq(f, grid[i], grid[i + 1], maxiter=500))

    raise RuntimeError("No se encontró amplitud a tal que N(a)=k.")


# ============================================================
# 4) NYQUIST + BALANCE ARMÓNICO
# ============================================================
def imag_W(omega: float, qord: float, p: dict) -> float:
    return float(np.imag(W_frac(omega, qord, p)))


def find_omega_k_pairs(qord: float, p: dict, wmin: float, wmax: float, nscan: int):
    """
    Localiza raíces de Im W_q(iω)=0 y calcula k=-1/Re W_q(iω).
    Interpreta k como el valor requerido por la ecuación de cierre del balance armónico.
    """
    qord = validate_fractional_order(qord)
    ws = np.linspace(float(wmin), float(wmax), int(nscan))
    vals = np.array([imag_W(w, qord, p) for w in ws], dtype=float)
    roots: list[float] = []

    for i in range(len(ws) - 1):
        f1, f2 = vals[i], vals[i + 1]
        if np.isnan(f1) or np.isnan(f2):
            continue
        if f1 == 0.0:
            roots.append(float(ws[i]))
        elif f1 * f2 < 0.0:
            try:
                root = brentq(lambda w: imag_W(w, qord, p), float(ws[i]), float(ws[i + 1]), maxiter=500)
                roots.append(float(root))
            except ValueError:
                pass

    roots = sorted(roots)
    roots_unique: list[float] = []
    for rr in roots:
        if (not roots_unique) or abs(rr - roots_unique[-1]) > 1e-7:
            roots_unique.append(rr)

    pairs = []
    for omega0 in roots_unique:
        W0 = W_frac(omega0, qord, p)
        reW = np.real(W0)
        if abs(reW) < 1e-12:
            continue
        k = -1.0 / reW
        if k > 0:
            pairs.append((float(omega0), float(k), complex(W0)))
    pairs.sort(key=lambda z: z[1])
    return pairs


def build_fractional_seed(qord: float, p: dict, omega0: float, k: float, a0: float):
    """
    Construcción de semilla armónica:
        (P0 - (iω0)^q I) v = 0,
    normalizando con r^T v = 1, y luego x_seed = a0 Re(v).

    Esto NO produce una órbita exacta de Caputo; produce una semilla útil
    para continuación e integración causal posterior.
    """
    qord = validate_fractional_order(qord)
    P, qvec, r = chua_lure_matrices(p)
    P0 = P + float(k) * np.outer(qvec, r)
    P0 = P0.astype(complex_t)
    lam0 = cpower_iw_q(float(omega0), float(qord))

    eigvals, eigvecs = np.linalg.eig(P0)
    idx = int(np.argmin(np.abs(eigvals - lam0)))
    v = eigvecs[:, idx]
    if abs(v[0]) < 1e-14:
        raise RuntimeError("No se pudo normalizar el autovector con r^T v = 1.")
    v = v / v[0]
    x_seed = float(a0) * np.real(v)
    return x_seed.astype(real_t), v, eigvals[idx]


def plot_nyquist_and_df(p: dict, qord: float, pairs, out_png: str):
    omg = np.logspace(np.log10(CONFIG["omega_scan"]["wmin"]), np.log10(CONFIG["omega_scan"]["wmax"]), 4000)
    W = np.array([W_frac(w, qord, p) for w in omg], dtype=complex_t)

    # Locus real de -1/N(a)
    aa = np.linspace(1.0 + 1e-6, CONFIG["amplitude_scan"]["amax"], 2500)
    minus_invN = -1.0 / np.array([N_sat(a, p) for a in aa], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.plot(W.real, W.imag, lw=1.8, label=r"$W_q(i\omega)$")
    ax.plot(minus_invN, np.zeros_like(minus_invN), lw=1.6, color="tab:orange", label=r"$-1/N(a)$")

    for j, (omega0, k, W0) in enumerate(pairs):
        a0 = solve_amplitude_from_k(k, p,
                                    CONFIG["amplitude_scan"]["amin"],
                                    CONFIG["amplitude_scan"]["amax"],
                                    CONFIG["amplitude_scan"]["nscan"])
        ax.scatter(W0.real, W0.imag, s=42, zorder=4)
        ax.annotate(f"rama {j}\n$\\omega_0$={omega0:.4f}\n$a_0$={a0:.3f}",
                    (W0.real, W0.imag), xytext=(8, 8), textcoords="offset points", fontsize=8)

    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"Re")
    ax.set_ylabel(r"Im")
    ax.set_title(f"Nyquist fraccionario + función descriptiva (q={qord:.4f})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seed_family(seed: np.ndarray, eigvec: np.ndarray, qord: float, omega0: float, out_png: str):
    t = np.linspace(0.0, 2.0 * np.pi / max(omega0, 1e-9), 600)
    X = np.real(np.exp(1j * omega0 * t)[:, None] * eigvec[None, :]) * float(seed[0])

    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=1.5)
    ax.scatter(seed[0], seed[1], seed[2], s=40, c="tab:red", label="semilla t=0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Curva armónica candidata (q={qord:.4f}, ω0={omega0:.4f})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5) MAIN
# ============================================================
def main():
    p = CONFIG["params"]
    qord = validate_fractional_order(CONFIG["frac_order"])

    # Equilibrios y estabilidad local
    eqs = chua_equilibria(p)
    eq_report = {}
    for name, eq in eqs.items():
        J = jacobian_at_equilibrium(eq, p)
        eigs = np.linalg.eigvals(J)
        iota = instability_measure(eigs, qord)
        eq_report[name] = {
            "point": eq.tolist(),
            "eigs": [[float(np.real(z)), float(np.imag(z))] for z in eigs],
            "iota": float(iota),
            "stable": bool(iota < 0.0),
        }

    # Nyquist / DF
    pairs = find_omega_k_pairs(qord, p,
                               CONFIG["omega_scan"]["wmin"],
                               CONFIG["omega_scan"]["wmax"],
                               CONFIG["omega_scan"]["nscan"])
    if not pairs:
        raise RuntimeError("No se encontraron ramas ω0-k del balance armónico.")

    branches = []
    for j, (omega0, k, W0) in enumerate(pairs):
        a0 = solve_amplitude_from_k(k, p,
                                    CONFIG["amplitude_scan"]["amin"],
                                    CONFIG["amplitude_scan"]["amax"],
                                    CONFIG["amplitude_scan"]["nscan"])
        x_seed, v, eig_match = build_fractional_seed(qord, p, omega0, k, a0)
        branches.append({
            "branch_index": j,
            "omega0": float(omega0),
            "k": float(k),
            "W0": [float(np.real(W0)), float(np.imag(W0))],
            "a0": float(a0),
            "seed": x_seed.tolist(),
            "eig_match": [float(np.real(eig_match)), float(np.imag(eig_match))],
            "eigvec_real": np.real(v).astype(float).tolist(),
            "eigvec_imag": np.imag(v).astype(float).tolist(),
        })

    chosen = CONFIG["continuation"]["branch_index"]
    if chosen < 0 or chosen >= len(branches):
        raise ValueError("branch_index fuera de rango.")
    chosen_branch = branches[chosen]

    plot_nyquist_and_df(p, qord, pairs, CONFIG["outputs"]["png_nyquist"])
    plot_seed_family(np.array(chosen_branch["seed"], dtype=float),
                     np.array(chosen_branch["eigvec_real"], dtype=float) + 1j * np.array(chosen_branch["eigvec_imag"], dtype=float),
                     qord, chosen_branch["omega0"],
                     CONFIG["outputs"]["png_seed"])

    summary = {
        "params": p,
        "frac_order": qord,
        "equilibria": eq_report,
        "branches": branches,
        "chosen_branch": chosen_branch,
        "notes": [
            "Las intersecciones Nyquist / -1/N(a) son candidatos de balance armónico.",
            "La semilla construida es para el problema armónico/Weyl, no una órbita exacta de Caputo.",
            "La validación final de atractor oculto debe hacerse por integración causal y cuencas/vecindades de equilibrio."
        ]
    }

    with open(CONFIG["outputs"]["json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Resumen guardado en:", CONFIG["outputs"]["json"])
    print("Figura Nyquist guardada en:", CONFIG["outputs"]["png_nyquist"])
    print("Figura semilla guardada en:", CONFIG["outputs"]["png_seed"])
    print("\nEquilibrios y estabilidad local:")
    for k, v in eq_report.items():
        print(f"{k}: punto={v['point']}, iota={v['iota']:.6f}, estable={v['stable']}")
    print("\nRamas candidatas:")
    for br in branches:
        print(f"rama {br['branch_index']}: omega0={br['omega0']:.8f}, k={br['k']:.8f}, a0={br['a0']:.8f}")


if __name__ == "__main__":
    main()
