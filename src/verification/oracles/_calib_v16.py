"""Scratch calibration for V1.6 (production box, storage-energy demo, figure trajectories). Not committed."""
import numpy as np
from scipy.integrate import solve_ivp
import regulator_stability as rs

sp = rs.StabilityParams()
K_grid = np.linspace(0.1, 6.0, 80)
print(f"naive  K_hopf={rs.hopf_gain(sp,'naive'):.3f}  passive K_hopf={rs.hopf_gain(sp,'passive'):.3f}")

# production gain box: K in [1.7,2.6], d0 in [1.5,1.8] — naive must tremble somewhere, passive clean
Kbox = np.linspace(1.7, 2.6, 7); dbox = np.linspace(1.5, 1.8, 4)
n_naive_osc = n_pass_osc = 0
for d in dbox:
    spd = rs.StabilityParams(d0=d)
    for K in Kbox:
        if rs.limit_cycle_amplitude(spd, K, "naive") > 0.02: n_naive_osc += 1
        if rs.limit_cycle_amplitude(spd, K, "passive") > 0.02: n_pass_osc += 1
tot = len(Kbox) * len(dbox)
print(f"production box ({tot} pts): naive oscillates at {n_naive_osc}/{tot}; passive at {n_pass_osc}/{tot}")

# storage-energy demo at production gain K=2.0 (naive limit cycle vs passive decay)
Kp = 2.0
for ctrl in ("naive", "passive"):
    P0, x0, _ = rs.healthy_state(sp, Kp)
    sol = solve_ivp(rs.rhs, [0, 60], [P0 - 0.05, x0, 0.0], args=(sp, Kp, ctrl),
                    rtol=1e-8, atol=1e-10, method="LSODA", max_step=0.1,
                    t_eval=np.linspace(0, 60, 1200))
    E = np.array([rs.storage_energy(sol.y[:, k], sp, Kp) for k in range(sol.y.shape[1])])
    E_end = E[-200:].mean()
    print(f"  {ctrl:8s}: storage energy start={E[5]:.4f} end(mean last)={E_end:.4f}  "
          f"{'sustained (limit cycle)' if E_end > 0.01 else 'decayed to FP'}")

# 2-D area over the figure box
Kg = np.linspace(0.5, 5.0, 30); dg = np.linspace(0.5, 2.5, 26)
print(f"2-D stable area: naive={rs.region_area(sp,'naive',Kg,dg):.3f}  passive={rs.region_area(sp,'passive',Kg,dg):.3f}")
