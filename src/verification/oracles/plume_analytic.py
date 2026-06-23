"""
Plume oracle for V3.1 (Tier 3) — the Morton–Taylor–Turner (MTT) integral plume model.

This is the INDEPENDENT oracle for the buoyant flame solver (operators/flow.py). It is
obtained "a different way" in the strongest sense: a 1-D ODE *integral* model derived from
the turbulent-entrainment hypothesis, with no shared code or discretization with the 3-D
Boussinesq PDE solver it judges. If the solver truly produces a buoyant plume, its
height-resolved fluxes must match this model's SCALING and its conserved invariant.

The model (Morton, Taylor & Turner 1956; Boussinesq, uniform unstratified environment,
top-hat profiles). With plume radius b(z), vertical velocity w(z) and reduced gravity
g'(z) = g (rho_amb - rho)/rho_ref, define the (per-pi) fluxes

    Q = b^2 w            (volume flux)
    M = b^2 w^2          (specific momentum flux)
    F = b^2 w g'         (buoyancy flux)

and the entrainment closure (inflow speed = alpha * w):

    dQ/dz = 2 alpha M^{1/2}
    dM/dz = 2 F Q / M
    dF/dz = 0            (uniform environment: buoyancy flux is CONSERVED with height)

The pure-plume self-similar solution (the analytic target) has

    Q ∝ z^{5/3},  M ∝ z^{4/3},  b ∝ z,  w ∝ z^{-1/3},  g' (and ΔT) ∝ z^{-5/3},

with F(z) = F0 constant. The two load-bearing, solver-agnostic facts a real buoyant plume
must reproduce: (i) buoyancy flux is conserved with height; (ii) the centerline buoyancy
decays as z^{-5/3} while the velocity decays as z^{-1/3} and the width grows linearly.
"""
import numpy as np
from scipy.integrate import solve_ivp

# pure-plume self-similar exponents (the analytic target; independent of constants).
EXPONENTS = {"Q": 5.0 / 3.0, "M": 4.0 / 3.0, "b": 1.0, "w": -1.0 / 3.0, "gp": -5.0 / 3.0}


def pure_plume_constants(F0, alpha):
    """Closed-form similarity prefactors c_M, c_Q for Q=c_Q z^{5/3}, M=c_M z^{4/3} (F=F0)."""
    c_M = (9.0 * alpha * F0 / 5.0) ** (2.0 / 3.0)
    c_Q = (6.0 * alpha / 5.0) * (9.0 * alpha * F0 / 5.0) ** (1.0 / 3.0)
    return c_M, c_Q


def pure_plume(z, F0, alpha):
    """Analytic pure-plume profiles at height(s) z>0: dict with Q, M, F, b, w, gp."""
    z = np.asarray(z, float)
    c_M, c_Q = pure_plume_constants(F0, alpha)
    Q = c_Q * z ** (5.0 / 3.0)
    M = c_M * z ** (4.0 / 3.0)
    b = (c_Q / np.sqrt(c_M)) * z
    w = (c_M / c_Q) * z ** (-1.0 / 3.0)
    gp = (F0 / c_Q) * z ** (-5.0 / 3.0)
    return {"Q": Q, "M": M, "F": np.full_like(z, F0), "b": b, "w": w, "gp": gp}


def mtt_integrate(F0, alpha, z0, z1, n=400):
    """Integrate the MTT ODEs from z0 to z1 (IC = pure-plume similarity at z0).

    Returns (z, dict of Q,M,F,b,w,gp). A numeric reference obtained by a 1-D ODE solve —
    a wholly different numerical method from the 3-D PDE solver under test.
    """
    ic = pure_plume(z0, F0, alpha)
    y0 = [float(ic["Q"]), float(ic["M"]), float(ic["F"])]

    def rhs(z, y):
        Q, M, F = y
        M = max(M, 1e-30)
        return [2.0 * alpha * np.sqrt(M), 2.0 * F * Q / M, 0.0]

    zs = np.linspace(z0, z1, n)
    sol = solve_ivp(rhs, (z0, z1), y0, t_eval=zs, rtol=1e-9, atol=1e-12, method="RK45")
    Q, M, F = sol.y
    b = np.sqrt(np.clip(Q * Q / np.maximum(M, 1e-30), 0, None))
    w = M / np.maximum(Q, 1e-30)
    gp = F / np.maximum(Q, 1e-30)
    return zs, {"Q": Q, "M": M, "F": F, "b": b, "w": w, "gp": gp}


def loglog_slope(z, y, zlo=None, zhi=None):
    """Least-squares slope of log y vs log z over [zlo, zhi] (the self-similar window)."""
    z = np.asarray(z, float); y = np.asarray(y, float)
    m = (z > 0) & (y > 0)
    if zlo is not None:
        m &= z >= zlo
    if zhi is not None:
        m &= z <= zhi
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(z[m]), np.log(y[m]), 1)[0])


def buoyancy_flux_variation(z, F, zlo=None, zhi=None):
    """Relative spread of F(z) over the window — the conserved-invariant check (→0 ideal)."""
    z = np.asarray(z, float); F = np.asarray(F, float)
    m = np.ones_like(z, bool)
    if zlo is not None:
        m &= z >= zlo
    if zhi is not None:
        m &= z <= zhi
    Fm = F[m]
    return float((Fm.max() - Fm.min()) / (np.abs(Fm).mean() + 1e-30))


if __name__ == "__main__":
    F0, alpha = 1.0, 0.1
    # 1) the analytic similarity slopes are exactly the MTT exponents.
    z = np.linspace(1.0, 50.0, 200)
    pp = pure_plume(z, F0, alpha)
    print("1) pure-plume log-log slopes vs MTT exponents:")
    ok = True
    for k, target in EXPONENTS.items():
        s = loglog_slope(z, pp[k])
        ok &= abs(s - target) < 1e-6
        print(f"   {k:>2}: slope {s:+.4f}  target {target:+.4f}")
    assert ok

    # 2) the ODE integrator reproduces the same self-similar scaling from a pure-plume IC.
    zs, sol = mtt_integrate(F0, alpha, z0=1.0, z1=50.0)
    print("2) MTT ODE-integrated slopes (self-similar window z∈[10,45]):")
    for k, target in EXPONENTS.items():
        s = loglog_slope(zs, sol[k], 10, 45)
        print(f"   {k:>2}: slope {s:+.4f}  target {target:+.4f}")
        assert abs(s - target) < 0.03, k

    # 3) buoyancy flux is conserved with height.
    var = buoyancy_flux_variation(zs, sol["F"], 5, 45)
    print(f"3) buoyancy-flux variation over height = {var:.2e}  (conserved → ~0)")
    assert var < 1e-6
    print("\nplume_analytic oracle self-checks passed.")
