"""Grafica los ceros de G para los tres campos de Chua, sin integrar EDO/EFD."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.optimize import brentq


RAIZ = Path(__file__).resolve().parent
FUENTE_PREDETERMINADA = RAIZ / "report_sections" / "02_modelos_y_fundamentos.tex"


def bloque_parametros(texto: str, etiqueta: str) -> str:
    posicion = texto.index("\\label{" + etiqueta + "}")
    inicio = texto.rfind("\\begin{equation}", 0, posicion)
    if inicio < 0:
        raise ValueError(f"No se encontró la ecuación de {etiqueta}")
    return texto[inicio:posicion]


def escalar(bloque: str, simbolo: str) -> float:
    patron = re.escape(simbolo) + r"\s*&?\s*=\s*([-+]?\d+(?:\.\d+)?)"
    coincidencia = re.search(patron, bloque)
    if coincidencia is None:
        raise ValueError(f"No se encontró el parámetro {simbolo}")
    return float(coincidencia.group(1))


def leer_parametros(fuente: Path) -> list[dict]:
    texto = fuente.read_text(encoding="utf-8")
    pwl = bloque_parametros(texto, "eq:parametros_enteros")
    vector = re.search(r"=\(([^)]+)\)", pwl)
    if vector is None:
        raise ValueError("No se encontró el vector nominal PWL")
    alfa, beta, gamma, m0, m1 = map(float, vector.group(1).split(","))
    casos = [{"nombre": "PWL nominal", "tipo": "pwl", "alpha": alfa,
              "beta": beta, "gamma": gamma, "m0": m0, "m1": m1,
              "etiqueta_fuente": "eq:parametros_enteros"}]
    for nombre, etiqueta in [("Wu: arctangente", "eq:parametros_wu"),
                             ("c590: arctangente", "eq:parametros_c590")]:
        bloque = bloque_parametros(texto, etiqueta)
        if nombre.startswith("Wu"):
            vector = re.search(r"=\(([^)]+)\)", bloque)
            if vector is None:
                raise ValueError("No se encontró el vector de Wu")
            alfa, beta, gamma = map(float, vector.group(1).split(","))
        else:
            alfa, beta, gamma = (escalar(bloque, s)
                                  for s in (r"\alpha", r"\beta", r"\gamma"))
        casos.append({"nombre": nombre, "tipo": "arctan", "alpha": alfa,
                      "beta": beta, "gamma": gamma,
                      "a1": escalar(bloque, "a_1"),
                      "a2": escalar(bloque, "a_2"),
                      "rho": escalar(bloque, r"\rho"),
                      "etiqueta_fuente": etiqueta})
    return casos


def caracteristica(x, caso: dict):
    x = np.asarray(x)
    if caso["tipo"] == "pwl":
        return caso["m1"] * x + (caso["m0"] - caso["m1"]) * np.clip(x, -1, 1)
    return caso["a1"] * x + caso["a2"] * np.arctan(caso["rho"] * x)


def reduccion(x, caso: dict):
    return caracteristica(x, caso) + caso["beta"] * x / (caso["beta"] + caso["gamma"])


def raiz_positiva(caso: dict) -> tuple[float, tuple[float, float]]:
    # El origen se excluye del intervalo; no se usa una raíz tabulada como dato.
    malla = np.geomspace(1e-5, 100.0, 2000)
    valores = reduccion(malla, caso)
    cambios = np.flatnonzero(valores[:-1] * valores[1:] < 0)
    if len(cambios) != 1:
        raise ValueError(f"Se esperó un único cero positivo en {caso['nombre']}")
    k = int(cambios[0])
    intervalo = (float(malla[k]), float(malla[k + 1]))
    raiz = brentq(lambda x: float(reduccion(x, caso)), *intervalo,
                  xtol=1e-13, rtol=1e-13)
    return float(raiz), intervalo


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuente", type=Path, default=FUENTE_PREDETERMINADA,
                        help="Fuente LaTeX que contiene los parámetros de los tres casos")
    parser.add_argument("--salida", type=Path, default=RAIZ / "figures" / "fundamentos",
                        help="Carpeta de salida de la figura y sus datos de cálculo")
    parser.add_argument("--png", action="store_true", help="Generar también una vista PNG")
    args = parser.parse_args()
    fuente = args.fuente.resolve(strict=True)
    salida = args.salida.resolve()
    salida.mkdir(parents=True, exist_ok=True)
    casos = leer_parametros(fuente)

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 9.3, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.3, "ytick.labelsize": 8.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    fig, ejes = plt.subplots(1, 3, figsize=(15 / 2.54, 7.25 / 2.54))
    azul, rojo, gris = "#185680", "#AC351D", "#68747D"
    registros = []
    for i, (eje, caso) in enumerate(zip(ejes, casos, strict=True)):
        raiz, intervalo = raiz_positiva(caso)
        extremo = 1.28 * raiz
        x = np.linspace(-extremo, extremo, 1601)
        if caso["tipo"] == "pwl":
            x = np.unique(np.r_[x, -1, 0, 1, -raiz, raiz])
        else:
            x = np.unique(np.r_[x, 0, -raiz, raiz])
        g = reduccion(x, caso)
        limite_y = 1.17 * float(np.max(np.abs(g)))
        eje.axhline(0, color=gris, linewidth=0.75, zorder=1)
        eje.axvline(0, color="#BCC4CA", linewidth=0.55, zorder=0)
        if caso["tipo"] == "pwl":
            for quiebre in (-1, 1):
                eje.axvline(quiebre, color="#A6ADB3", linewidth=0.65,
                            linestyle=(0, (3, 3)), zorder=0)
                eje.text(quiebre, (0.86 if quiebre < 0 else -0.91) * limite_y,
                         "$-1$" if quiebre < 0 else "$1$",
                         ha="right" if quiebre < 0 else "left", va="bottom",
                         color=gris, fontsize=7.5)
        eje.plot(x, g, color=azul, linewidth=1.6, zorder=2)
        raices = np.array([-raiz, 0.0, raiz])
        residuos = reduccion(raices, caso)
        eje.scatter(raices, np.zeros(3), color=rojo, edgecolor="white",
                    linewidth=0.6, s=24, zorder=4)
        eje.set_xlim(-extremo, extremo)
        eje.set_ylim(-limite_y, limite_y)
        eje.set_xticks(raices)
        eje.set_xticklabels([f"−{raiz:.4f}", "0", f"{raiz:.4f}"])
        eje.set_xlabel("$x$", labelpad=3)
        eje.set_ylabel("$G(x)$", labelpad=2)
        eje.set_title(f"({chr(97+i)}) {caso['nombre']}", pad=8)
        eje.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        eje.grid(axis="y", color="#E5E9EC", linewidth=0.45, zorder=-2)
        eje.tick_params(axis="both", direction="out", length=3, pad=3)
        registros.append({"caso": caso["nombre"], "parametros": caso,
                          "raices": raices.tolist(),
                          "residuos_G": residuos.tolist(),
                          "intervalo_brentq_positivo": list(intervalo),
                          "limites_x": list(eje.get_xlim()),
                          "limites_y": list(eje.get_ylim())})
    fig.suptitle(r"$G(x)=h(x)+\dfrac{\beta}{\beta+\gamma}\,x$", y=0.98, fontsize=11)
    claves = [Line2D([0], [0], color=azul, lw=1.6, label="Reducción escalar"),
              Line2D([0], [0], marker="o", linestyle="none", color=rojo,
                     markersize=4.5, label=r"Ceros de $G$: $0$ y $\pm x_*$")]
    fig.legend(handles=claves, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               frameon=False, ncol=2, fontsize=8.7, handlelength=1.8,
               columnspacing=2)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.24, top=0.75, wspace=0.48)
    pdf = salida / "equilibrios_reduccion_escalar.pdf"
    png = salida / "equilibrios_reduccion_escalar.png"
    fig.savefig(pdf, metadata={"Title": "Reducción escalar de los equilibrios de Chua"})
    salidas = [pdf]
    if args.png:
        fig.savefig(png, dpi=360)
        salidas.append(png)
    plt.close(fig)
    manifiesto = {
        "descripcion": "Evaluación de funciones escalares; no se integraron trayectorias.",
        "dimensiones_cm": [15, 7.25],
        "fuente_parametros": fuente.name, "fuente_sha256": sha256(fuente),
        "script_sha256": sha256(Path(__file__)),
        "metodo_raices": "scipy.optimize.brentq", "casos": registros,
        "salidas": [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                    for p in salidas],
    }
    (salida / "equilibrios_reduccion_escalar.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifiesto, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
