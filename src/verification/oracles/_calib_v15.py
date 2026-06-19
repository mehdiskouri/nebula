"""Scratch calibration for V1.5 (coupled slow-reserve death, positive-feedback diagnostic,
3-D resting equilibrium, basin areas for the figure). NOT committed."""
import numpy as np
from scipy.integrate import solve_ivp
import regulator as rg

p = rg.RegulatorParams()
rc = rg.r_critical(p)
print(f"r_crit = {rc:.3f}")

# ---- 3-D resting equilibrium of the coupled (P,x,r) system ----
sol = solve_ivp(lambda t, s: rg.rhs_full(s, p), [0, 2000],
                [rg.healthy_fp(p, 1.0)["P"], 0.394, 1.0], rtol=1e-9, atol=1e-11, method="LSODA")
P_rest, x_rest, r_rest = sol.y[:, -1]
print(f"3-D resting equilibrium: P*={P_rest:.3f}  x*={x_rest:.3f}  r*={r_rest:.3f}  (> r_crit={rc:.3f}: {r_rest>rc})")

# ---- coupled death cascade: a recoverable vs a fatal hemorrhage (sustained reserve sink) ----
def run_bleed(sink, t_bleed=(20, 60), tmax=300):
    def f(t, s):
        d = rg.rhs_full(s, p).copy()
        if t_bleed[0] <= t <= t_bleed[1]:
            d[2] -= sink                       # hemorrhage: an extra reserve drain while bleeding
        return d
    s0 = [P_rest, x_rest, r_rest]
    sol = solve_ivp(f, [0, tmax], s0, rtol=1e-8, atol=1e-10, method="LSODA",
                    max_step=0.5, t_eval=np.linspace(0, tmax, 1600))   # max_step: see the bleed window
    return sol

for sink in (0.02, 0.04, 0.05, 0.06):
    sol = run_bleed(sink)
    Pend, rend = sol.y[0, -1], sol.y[2, -1]; rmin = sol.y[2].min()
    dead = Pend < 0.1
    print(f"  bleed sink={sink:.3f}: r_min={rmin:.3f} (r_crit={rc:.3f}) end P={Pend:.3f} -> {'DEAD' if dead else 'recovered'}")

# ---- positive-feedback diagnostic: loop gain dPdot/dP in the collapse region ----
def dPdot_dP(P, x, eps=1e-5):
    f = lambda PP: rg.pump(PP, p) * (1 + p.beta * x) - p.gamma * PP
    return (f(P + eps) - f(P - eps)) / (2 * eps)
s = rg.saddle_fp(p, 1.0)
print(f"\npositive-feedback: saddle eig max = {s['eig'].real.max():.3f} (>0)")
xs = s["x"]
gains = [dPdot_dP(P, xs) for P in np.linspace(0.05, s["P"] * 0.95, 8)]
print(f"  loop-gain dPdot/dP just below the saddle (x=x_saddle): max={max(gains):.3f}  "
      f"(>0 ⇒ autocatalytic) any-positive={any(g > 0 for g in gains)}")

# ---- basin areas at 3 reserve levels (for the figure / extra B evidence) ----
Pg = np.linspace(0, 2.4, 30); Xg = np.linspace(0, p.a_cap, 20)
print("\nbasin fraction & critical-bleed by reserve:")
for r in (1.0, 0.6, 0.3):
    area = rg.basin_area(p, r, Pg, Xg)
    print(f"  r={r}: basin_area={area:.3f}  crit_bleed={rg.critical_bleed(p, r):.3f}")
