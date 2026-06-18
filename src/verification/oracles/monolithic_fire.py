"""
Monolithic coupled fire integrator — the independent trajectory oracle (protocol §7).

Solves the SAME coupled fire system as the operators in fire_operators.py, but
*without operator splitting*: every term is evaluated together and advanced with an
adaptive sub-stepped RK4. Adaptive sub-stepping (not an implicit solve) is what
resolves the stiff thermal runaway and keeps it self-limiting — as O2 depletes within
a sub-step the combustion rate falls smoothly, so no reactant is ever over-subscribed
and nothing must be clamped. That is precisely the contrast with the split runtime:
the monolithic reference conserves cleanly through the impulse, proving the split's
conservation-residual spike is a splitting artifact, not physics.

(An implicit backward-Euler solve is the textbook alternative; for a latency-insensitive
reference oracle, adaptive RK4 is simpler, obviously correct, and validated below against
scipy's stiff solver on the 0-D coupled ODE.)

Shared by V0.3 and V1.2. Pure numpy (fast at N<=24).
"""
import numpy as np

import fire_operators as fo
from bus_runtime import FIELDS


def _rk4(st, p, h):
    def add(a, b, s):
        return {f: a[f] + s * b[f] for f in FIELDS}
    k1 = fo.coupled_rhs(st, p)
    k2 = fo.coupled_rhs(add(st, k1, h / 2), p)
    k3 = fo.coupled_rhs(add(st, k2, h / 2), p)
    k4 = fo.coupled_rhs(add(st, k3, h), p)
    return {f: st[f] + (h / 6.0) * (k1[f] + 2 * k2[f] + 2 * k3[f] + k4[f]) for f in FIELDS}


def _stable_h(st, p, remaining, safety=0.2):
    """Largest sub-step that cannot over-deplete a reactant or over-diffuse."""
    r_py = fo.pyrolysis_rate(st["T"], st["m_s"], p)
    r_cb = fo.combustion_rate(st["T"], st["gas"], st["o2"], p)
    eps = 1e-30
    limits = [
        safety * float(np.min(st["m_s"] / (r_py + eps))),
        safety * float(np.min(st["gas"] / (r_cb + eps))),
        safety * float(np.min(st["o2"] / (p.s_o2 * r_cb + eps))),
        safety * p.dx * p.dx / (6.0 * max(p.k_wood, p.k_char)),
    ]
    h = min([remaining] + [x for x in limits if np.isfinite(x) and x > 0])
    return max(h, remaining * 1e-6)


def step_monolithic(st, p, dt, lightning=None, ledger=None):
    """Advance the coupled system by dt with adaptive RK4 sub-stepping.

    Deposits the optional lightning impulse first (same as the split runtime), then
    integrates. Accumulates a trapezoidal ledger of boundary/reaction transfers for
    budget-closure checks. Returns (new_state, ledger).
    """
    st = {f: st[f].copy() for f in FIELDS}
    ledger = {} if ledger is None else ledger
    if lightning is not None:
        m = lightning["mask"]; ncell = max(int(m.sum()), 1)
        st["T"][m] += lightning.get("energy", 0.0) / p.C_V / ncell
        st["q"][m] += lightning.get("charge", 0.0) / ncell
        ledger["lightning_energy"] = ledger.get("lightning_energy", 0.0) + lightning.get("energy", 0.0)

    done = 0.0
    while done < dt - 1e-15:
        h = _stable_h(st, p, dt - done)
        r_py = fo.pyrolysis_rate(st["T"], st["m_s"], p)
        r_cb = fo.combustion_rate(st["T"], st["gas"], st["o2"], p)
        _, bloss = fo.conduction_energy(st["T"], st["char"], st["m_s"], p)
        _, o2in = fo.o2_boundary_influx(st["o2"], p)
        ledger["combustion_exo"] = ledger.get("combustion_exo", 0.0) + float((p.dH_cb * r_cb).sum()) * h
        ledger["pyrolysis_endo"] = ledger.get("pyrolysis_endo", 0.0) - float((p.dH_py * r_py).sum()) * h
        ledger["boundary_heat_loss"] = ledger.get("boundary_heat_loss", 0.0) + bloss * h
        ledger["exhaust_vent"] = ledger.get("exhaust_vent", 0.0) + float(((1.0 + p.s_o2) * r_cb).sum()) * h
        ledger["o2_influx"] = ledger.get("o2_influx", 0.0) + o2in * h
        st = _rk4(st, p, h)
        done += h
    return st, ledger


def run(st, p, dt, nsteps, lightning_at=None, lightning=None):
    """Convenience: integrate nsteps, optionally firing a lightning impulse at a step."""
    traj = [{f: st[f].copy() for f in FIELDS}]
    led = {}
    for n in range(nsteps):
        lg = lightning if (lightning_at is not None and n == lightning_at) else None
        st, led = step_monolithic(st, p, dt, lightning=lg, ledger=led)
        traj.append({f: st[f].copy() for f in FIELDS})
    return st, traj, led


if __name__ == "__main__":
    from scipy.integrate import solve_ivp
    np.set_printoptions(precision=5, suppress=True)
    p = fo.FireParams()

    # 1) 0-D coupled ODE: monolithic RK4 sub-stepper vs scipy stiff solver (Radau).
    #    Single cell (N=1) -> coupled_rhs has no interior conduction, only boundary terms.
    st0 = {"T": np.array([[[800.0]]]), "m_s": np.array([[[1.0]]]),
           "gas": np.array([[[0.3]]]), "o2": np.array([[[0.5]]]),
           "char": np.array([[[0.0]]]), "q": np.array([[[0.2]]])}
    T_end = 0.02
    mine, _ = step_monolithic({k: v.copy() for k, v in st0.items()}, p, T_end)

    def rhs(t, y):
        s = {f: np.array([[[y[i]]]]) for i, f in enumerate(FIELDS)}
        F = fo.coupled_rhs(s, p)
        return [F[f].item() for f in FIELDS]
    y0 = [st0[f].item() for f in FIELDS]
    sol = solve_ivp(rhs, [0, T_end], y0, method="Radau", rtol=1e-9, atol=1e-12)
    yf = sol.y[:, -1]
    print("1) 0-D monolithic vs scipy Radau (coupled ODE):")
    for i, f in enumerate(FIELDS):
        print(f"   {f:4s} mine={mine[f].item():.6f}  scipy={yf[i]:.6f}  "
              f"|d|={abs(mine[f].item()-yf[i]):.2e}")

    # 2) mass-conservation invariant (combustion OFF -> m_s+gas+char exactly conserved).
    p2 = fo.FireParams(A_cb=0.0, h_loss=0.0, o2_influx=0.0)
    N = 8
    st = {"T": np.full((N, N, N), 750.0), "m_s": np.ones((N, N, N)),
          "gas": np.full((N, N, N), 0.05), "o2": np.full((N, N, N), 0.2),
          "char": np.zeros((N, N, N)), "q": np.zeros((N, N, N))}
    M0 = (st["m_s"] + st["gas"] + st["char"]).sum()
    stf, _ = step_monolithic(st, p2, 0.01)
    M1 = (stf["m_s"] + stf["gas"] + stf["char"]).sum()
    print(f"\n2) pyrolysis-only solid mass conservation: |dM|/M = {abs(M1-M0)/M0:.2e}")
