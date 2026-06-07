import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2")))

from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems.builtins import _chua_lure_system
from types import SimpleNamespace

def main():
    params = ChuaParameters(
        model="nonsmooth",
        alpha=8.4562,
        beta=12.0732,
        gamma=0.0052,
        m0=-0.1768,
        m1=-1.1468
    )
    system = SimpleNamespace(parameters={"model": "nonsmooth", "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052, "m0": -0.1768, "m1": -1.1468}, lure=_chua_lure_system({"model": "nonsmooth", "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052, "m0": -0.1768, "m1": -1.1468}))
    
    # Candidate 1 seed x0 after continuation (as computed in scratch_saturation_test.py)
    x0 = np.array([5.12814798, -0.42504425, -8.03860148])
    
    print("Integrating Candidate 1 with ABM (full memory)...")
    times, states, status, info = fractional_integrate(
        rhs=lambda t, val: system.lure.matrix @ val + system.lure.input_vector * system.lure.nonlinearity(system.lure.output_vector @ val),
        x0=x0,
        q=0.9998,
        h=0.01,
        t_final=300.0,
        method="abm",
        memory_mode="full",
        use_c_backend=True,
        allow_python_fallback=True
    )
    print(f"Integration status: {status}")
    
    # Plot tail to avoid transient
    n_burn = int(100.0 / 0.01)
    tail = states[n_burn:]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[:, 0], tail[:, 1], tail[:, 2], lw=0.5, color="blue")
    ax.set_title("Candidate 1: nonsmooth Chua (q=0.9998, ABM)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    out_img = Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\outputs\saturation_candidate1_phase3d.png")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"Saved figure to {out_img}")

if __name__ == "__main__":
    main()
