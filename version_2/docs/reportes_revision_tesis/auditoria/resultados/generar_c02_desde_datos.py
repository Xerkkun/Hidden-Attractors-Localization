"""Regenerar figuras del control C02 desde datos conservados, sin simular."""
import hashlib
import json
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_c02_restart(source: Path, destination: Path) -> dict:
    """Mostrar x(t) y proyección (x,y) del reinicio causal C02.

    Entrada: NPZ con columnas t,x,y,z; salida: figura PDF y hashes.
    Se muestran t=280..300 para legibilidad. Sin interpolar ni recalcular
    estados. La forma recurrente no demuestra periodicidad exacta ni caos.
    """
    with np.load(source) as archive:
        trajectory = archive["trajectory"]
    if trajectory.shape != (30001, 4) or not np.isfinite(trajectory).all():
        raise ValueError("Trayectoria C02 inválida")
    if destination.exists():
        raise FileExistsError("La figura ya existe; revisar antes de sobrescribir")
    post = trajectory[trajectory[:, 0] >= 280]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
    axes[0].plot(post[:, 0], post[:, 1], color="#004080", linewidth=1.1)
    axes[0].set(xlabel="t (unidades de tiempo)", ylabel="x", xlim=(280, 300))
    axes[1].plot(post[:, 1], post[:, 2], color="#004080", linewidth=1.1)
    axes[1].set(xlabel="x", ylabel="y")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={"Title":"Control C02: reinicio causal", "Author":""})
    plt.close(fig)
    return {"data":str(source),"data_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "figure":str(destination),"figure_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
            "display_interval":[280,300],"columns":["t","x","y","z"],
            "scope":"Proyección del control conservado; no certificación de periodicidad o caos"}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=here / "C02_independent_restart.npz")
    parser.add_argument("--output", type=Path, default=here.parents[1] / "figures/revision/c02_reinicio_validado.pdf")
    args = parser.parse_args()
    record = plot_c02_restart(args.source, args.output)
    print(json.dumps(record, indent=2, ensure_ascii=False))
