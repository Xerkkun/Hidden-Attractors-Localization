import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "version_2"))

from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import ChuaParameters, rhs_nonsmooth

def main():
    alpha = 8.4562
    beta = 12.0732
    gamma = 0.0052
    m0 = -0.1768
    m1 = -1.1468
    q = 0.9998
    h = 0.01
    t_final = 300.0
    
    params = ChuaParameters(
        model="nonsmooth",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        m0=m0,
        m1=m1,
    )
    
    # Try the seed
    x0 = np.array([3.039383584794975, -0.2416862069577155, -6.873467365218827], dtype=float)
    
    def rhs_time(t, state):
        return rhs_nonsmooth(state, params)
        
    print("Integrating Danca reference...")
    times, states, status, info = fractional_integrate(
        rhs_time,
        x0,
        q=q,
        h=h,
        t_final=t_final,
        method="abm",
        memory_mode="full",
        use_c_backend=True,
        allow_python_fallback=True,
        divergence_norm=120.0,
    )
    print(f"Status: {status}")
    print(f"Final state: {states[-1] if len(states) else 'None'}")
    print(f"Traj length: {len(states)}")
    
    # Save the trajectory if ok
    if status == "ok":
        out_path = ROOT / "danca_ref_trajectory.csv"
        np.savetxt(out_path, np.column_stack((times, states)), delimiter=",", header="t,x,y,z", comments="")
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
