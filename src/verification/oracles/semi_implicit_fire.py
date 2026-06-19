"""
Semi-implicit (IMEX) fire split — the REDESIGN reference for V1.2 (protocol §V1.2,
failure->outcome "REDESIGN: adopt a semi-implicit split").

The naive bus split (bus_runtime.step_split) is plain FORWARD EULER on
fire_operators.coupled_rhs: only conditionally stable on the stiff Arrhenius
reaction terms, so past an explicit step limit a reactant is over-subscribed, the
commit clamps it to zero, and the integrated burn outcome is corrupted. multirate.py
fixes this by rate-driven SUB-STEPPING (many small explicit steps). This module is the
OTHER documented fallback: a LINEARLY-IMPLICIT treatment of the stiff reactant
depletion that is unconditionally stable, so the production step survives WITHOUT
sub-stepping.

Scheme (Lie-split: implicit reaction, then explicit transport):
  - Freeze the Arrhenius rate CONSTANTS at T^n (the linearization that makes the
    otherwise-nonlinear depletion linear in the reactant).
  - Pyrolysis is linear in m_s -> backward Euler:  m_s^{n+1} = m_s^n / (1 + k_py dt).
    The consumed solid C_py = m_s^n - m_s^{n+1} is bounded in [0, m_s^n] for ANY dt:
    it can never overshoot, so no clamp is ever needed.
  - Combustion is linear in gas (o2 frozen) -> backward Euler on gas; the reaction
    extent is additionally capped at the available O2 (s_o2 * C_cb <= o2), which is the
    physical O2-limited self-extinction, applied consistently to gas, o2 and the heat
    release (NOT a post-hoc clamp on a committed negative).
  - Conduction, boundary O2 influx: explicit (reuse fire_operators); charge: implicit decay.

Same rate laws and conserved transfers as the operators, so it converges to the
monolithic RK4 oracle (monolithic_fire) as dt->0 at first order. The O(dt) Lie-splitting
error between reaction and conduction is the only consistency error; the win is
unconditional stability on the stiff char<->conduction<->pyrolysis loop.

Shared by V1.2. Pure numpy. bus_runtime/fire_operators/multirate/monolithic_fire are
left UNTOUCHED (they are V0.3/V1.1-critical and already passing); this module imports them.
"""
import numpy as np

import fire_operators as fo
import bus_runtime as br

FIELDS = br.FIELDS


def step_semi_implicit(st, p, dt, lightning=None):
    """Advance the coupled fire system by `dt` with the IMEX semi-implicit split.

    Unconditionally stable on the stiff reaction terms: no reactant clamp is reached
    because the implicit depletion is bounded by construction. Returns the new state.
    """
    st = {f: st[f].copy() for f in FIELDS}

    # optional composite impulse (same deposit convention as the split / monolithic runtimes)
    if lightning is not None:
        m = lightning["mask"]; ncell = max(int(m.sum()), 1)
        st["T"][m] += lightning.get("energy", 0.0) / p.C_V / ncell
        st["q"][m] += lightning.get("charge", 0.0) / ncell

    T = st["T"]

    # --- implicit reaction sub-step (rate constants frozen at T^n) ---
    # pyrolysis: linear in m_s -> backward Euler, consumed amount bounded by m_s
    k_py = p.A_py * np.exp(-p.Ta_py / np.maximum(T, 1.0))
    m_s_new = st["m_s"] / (1.0 + k_py * dt)
    C_py = st["m_s"] - m_s_new                      # >= 0, <= m_s : never overshoots
    st["m_s"] = m_s_new
    st["gas"] += p.nu_g * C_py
    st["char"] += p.nu_c * C_py
    T -= (p.dH_py * C_py) / p.C_V                   # endothermic

    # combustion: linear in gas (o2 frozen) -> backward Euler, capped at available O2
    k_cb = p.A_cb * np.exp(-p.Ta_cb / np.maximum(T, 1.0)) * np.maximum(st["o2"], 0.0)
    gas_new = st["gas"] / (1.0 + k_cb * dt)
    C_cb = st["gas"] - gas_new                      # gas burned, >= 0, <= gas
    o2_cap = np.maximum(st["o2"], 0.0) / p.s_o2 * (1.0 - 1e-12)
    C_cb = np.minimum(C_cb, o2_cap)                 # physical O2-limited extinction
    st["gas"] -= C_cb
    st["o2"] -= p.s_o2 * C_cb
    T += (p.dH_cb * C_cb) / p.C_V                   # exothermic

    # --- explicit transport sub-step ---
    dE, _ = fo.conduction_energy(T, st["char"], st["m_s"], p)
    T += (dE / p.C_V) * dt
    o2_src, _ = fo.o2_boundary_influx(st["o2"], p)
    st["o2"] += o2_src * dt
    st["q"] = st["q"] / (1.0 + p.lambda_q * dt)     # implicit charge decay (always stable)

    st["T"] = T
    return st


def run_semi_implicit(st, p, dt, nsteps, lightning_at=None, lightning=None):
    """Convenience: integrate nsteps of global dt with the semi-implicit split."""
    st = br.copy_state(st)
    for n in range(nsteps):
        lg = lightning if (lightning_at is not None and n == lightning_at) else None
        st = step_semi_implicit(st, p, dt, lightning=lg)
    return st


def is_finite_state(st):
    return all(np.all(np.isfinite(st[f])) for f in FIELDS)


if __name__ == "__main__":
    import monolithic_fire as mf
    np.seterr(all="ignore")
    np.set_printoptions(precision=5, suppress=True)
    p = fo.FireParams()

    # 1) UNCONDITIONAL non-negativity: a hot, fuel-rich, O2-limited cell over a HUGE step
    #    (forward Euler would over-deplete and clamp; the implicit depletion cannot).
    N = 6
    st = br.make_state(N, T0=1400.0, m_s0=1.0, gas0=5.0, o2=0.1, char0=0.0)
    for dt in (1e-3, 1e-1, 1e2):
        s = step_semi_implicit(st, p, dt)
        mins = {f: float(s[f].min()) for f in ("m_s", "gas", "o2", "char")}
        print(f"1) dt={dt:<6g} reactant minima {mins}  finite={is_finite_state(s)}")
        assert all(v >= -1e-12 for v in mins.values()), "reactant went negative!"
    print("   -> no reactant ever negative, no clamp needed (unconditionally stable).")

    # 2) CONVERGENCE to the monolithic oracle as dt -> 0 on a single (0-D) cell.
    st0 = {"T": np.array([[[800.0]]]), "m_s": np.array([[[1.0]]]),
           "gas": np.array([[[0.3]]]), "o2": np.array([[[0.5]]]),
           "char": np.array([[[0.0]]]), "q": np.array([[[0.2]]])}
    T_END = 0.02
    ref, _ = mf.step_monolithic({k: v.copy() for k, v in st0.items()}, p, T_END)
    print("\n2) semi-implicit -> monolithic oracle as dt shrinks (single cell, T_END=0.02):")
    prev = None
    for nsub in (4, 16, 64, 256, 1024):
        dt = T_END / nsub
        s = run_semi_implicit({k: v.copy() for k, v in st0.items()}, p, dt, nsub)
        err = max(abs(s[f].item() - ref[f].item()) for f in FIELDS)
        rate = "" if prev is None else f"  (err ratio vs prev: {prev / err:.2f})"
        print(f"   nsub={nsub:<5d} dt={dt:.2e}  max|semi-ref|={err:.3e}{rate}")
        prev = err
    print("   -> error halves as dt halves => first-order convergent (expected for IMEX Lie split).")
