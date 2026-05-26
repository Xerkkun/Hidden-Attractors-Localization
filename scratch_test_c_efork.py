import sys
from pathlib import Path
import numpy as np
import yaml
import os

# Ensure the root of the project is in python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.workflows.centered_lure_df_workflow import run_centered_lure_df_workflow
from src.workflows.configs import load_and_validate_config
from src.integrators.general import integrate_general

def main():
    print("=" * 90)
    print(" EJECUCIÓN RÁPIDA DE LOS TRES SISTEMAS DE CHUA CON EL MOTOR C EFORK ")
    print("=" * 90)
    print("Nota: Limitaremos el tiempo a t_final = 30.0 y desactivaremos las pruebas de vecindarios")
    print("para obtener resultados de localización inmediatos en menos de 5 segundos.\n")

    presets = [
        "configs/examples/chua_integer_centered_lure_df.yaml",
        "configs/examples/chua_fractional_centered_lure_df.yaml",
        "configs/examples/chua_arctan_fractional_centered_lure_df.yaml"
    ]

    summaries = []

    for preset_path in presets:
        print(f"\n---> Cargando configuración desde: {preset_path}")
        config = load_and_validate_config(preset_path)
        
        config["t_final"] = 30.0
        config["t_burn"] = 10.0
        config["run_hiddenness_tests"] = False
        config["run_basin_slices"] = False
        config["plot_enabled"] = False
        config["save_figures"] = False
        
        print(f"Ejecutando workflow para: {config['system_id']} (Integrador: {config['integrator']})...")
        try:
            summary = run_centered_lure_df_workflow(config)
            summaries.append(summary)
        except Exception as e:
            print(f"ERROR al ejecutar {config['system_id']}: {e}")

    print("\n" + "=" * 90)
    print(" RESUMEN UNIFICADO DE LOCALIZACIÓN DE CHUA (INTEGRACIÓN EN C EN SEGUNDOS) ")
    print("=" * 90)
    print(f"| {'System ID':<30} | {'q':<8} | {'Integrator':<10} | {'omega0':<10} | {'Amp A0':<10} | {'Gain k':<10} |")
    print("-" * 90)
    for s in summaries:
        q_str = f"{s['q']:.4f}"
        w0_str = f"{s['omega0']:.4f}" if s['omega0'] == s['omega0'] else "N/A"
        a0_str = f"{s['amplitude_a0']:.4f}" if s['amplitude_a0'] == s['amplitude_a0'] else "N/A"
        k_str = f"{s['k']:.4f}" if s['k'] == s['k'] else "N/A"
        print(f"| {s['system_id']:<30} | {q_str:<8} | {s['integrator']:<10} | {w0_str:<10} | {a0_str:<10} | {k_str:<10} |")
    print("=" * 90)
    print("Todos los integradores se ejecutaron por defecto utilizando los backends en C nativos.")
    print("=" * 90 + "\n")

    print("=" * 90)
    print(" DEMOSTRACIÓN EXTRA: SOLUCIONADOR FDE GENERAL EN C (LORENZ FRACCIONARIO) ")
    print("=" * 90)
    print("Integraremos un sistema caótico de Lorenz de orden fraccionario (q = 0.90) usando")
    print("nuestro solucionador general compilado en C nativo con callbacks de Python para las derivadas.")
    
    # Definir el campo vectorial para el Lorenz fraccionario
    def lorenz_deriv(t, x):
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        dxdt = sigma * (x[1] - x[0])
        dydt = x[0] * (rho - x[2]) - x[1]
        dzdt = x[0] * x[1] - beta * x[2]
        return np.array([dxdt, dydt, dzdt])
        
    x0 = np.array([1.0, 1.0, 1.0])
    
    print("\nEjecutando Lorenz Fraccionario (q = 0.90) con EFORK general en C...")
    t_ef, x_ef, status_ef = integrate_general(
        lorenz_deriv, x0, q=0.90, h=0.01, t_final=5.0, integrator="efork", use_c_backend=True
    )
    print(f"EFORK C - Estado: {status_ef}, Pasos calculados: {len(t_ef)}")
    print(f"Estado final EFORK C: x={x_ef[-1, 0]:.4f}, y={x_ef[-1, 1]:.4f}, z={x_ef[-1, 2]:.4f}")
    
    print("\nEjecutando Lorenz Fraccionario (q = 0.90) con ABM general en C...")
    t_ab, x_ab, status_ab = integrate_general(
        lorenz_deriv, x0, q=0.90, h=0.01, t_final=5.0, integrator="abm", use_c_backend=True
    )
    print(f"ABM C - Estado: {status_ab}, Pasos calculados: {len(t_ab)}")
    print(f"Estado final ABM C: x={x_ab[-1, 0]:.4f}, y={x_ab[-1, 1]:.4f}, z={x_ab[-1, 2]:.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
