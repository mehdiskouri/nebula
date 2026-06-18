"""
Fire operators — the four constitutive transfer laws of the "tree on fire"
(ARCHITECTURE.md §III.3; Decision #13: package the transfer operator, not the
phenomenon). Shared by V0.3 (conservation + composite-OOD), V1.2 (split stability),
V1.3 (Jensen sub-cell variance).

Each operator READS shared state and emits only:
  - contributions: additive {field -> per-cell delta-rate} staged into conserved buses
    (order-independent: combustion + pyrolysis + conduction compose freely);
  - ledger: {name -> scalar} the boundary/source-sink transfers the audit balances against;
  - an envelope() self-check on its OWN input marginals (the per-operator validity box
    that the composite-OOD test shows is necessary-but-insufficient).

The phenomenon "fire" is the fixed point of these coupled through T, gas (volatiles),
O2, char (ARCHITECTURE §III.3). Kinetic constants are documented PLAUSIBILITY-ENGINE
values (Part VII), tuned only for a stable in-distribution burn + an impulse runaway.

State fields (each an (N,N,N) array): T [K], m_s solid wood, gas volatiles, o2 oxygen,
char, q charge. Heat capacity C_V folds T<->energy. Rates are per unit time.
"""
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-300


@dataclass
class FireParams:
    dx: float = 1.0
    C_V: float = 1.0              # volumetric heat capacity (energy = C_V * T per cell)
    # pyrolysis: solid -> nu_g gas + nu_c char, endothermic
    A_py: float = 3.0e6
    Ta_py: float = 9000.0
    nu_g: float = 0.7
    nu_c: float = 0.3
    dH_py: float = 4.0           # endothermic enthalpy per unit solid pyrolyzed
    # combustion: gas + s_o2 O2 -> exhaust + heat
    A_cb: float = 4.0e5
    Ta_cb: float = 7000.0
    s_o2: float = 1.0            # O2 consumed per unit gas burned
    dH_cb: float = 60.0          # heat released per unit gas burned (strongly exothermic)
    # conduction: k(chi) char insulates
    k_wood: float = 0.08
    k_char: float = 0.008
    chi_max: float = 1.0
    # boundary
    h_loss: float = 0.02         # surface heat-loss coeff to ambient
    T_amb: float = 300.0
    o2_influx: float = 0.05      # surface O2 replenishment coeff toward o2_amb
    o2_amb: float = 0.23
    # charge (lightning bus): simple dissipation
    lambda_q: float = 0.5
    S0: float = 1.0              # baseline structural strength (char-weakening transition)
    # per-operator validity envelopes: the temperature ranges over which each rate law
    # is calibrated (the marginal each operator self-checks). A composite event can keep
    # every operator's T inside its box while the JOINT state (fuel-rich + O2-poor + rate)
    # is off-distribution -> per-operator envelopes miss it, the conservation audit catches it.
    py_T: tuple = (300.0, 900.0)
    cb_T: tuple = (400.0, 1400.0)


# ---------- rate laws (shared by split runtime and monolithic oracle) ----------

def pyrolysis_rate(T, m_s, p):
    """Arrhenius pyrolysis rate of solid wood [mass/time], >= 0."""
    return p.A_py * np.exp(-p.Ta_py / np.maximum(T, 1.0)) * np.maximum(m_s, 0.0)


def combustion_rate(T, gas, o2, p):
    """Arrhenius, bilinear in gas*O2 [gas-mass/time]; self-limits as o2 -> 0."""
    return (p.A_cb * np.exp(-p.Ta_cb / np.maximum(T, 1.0))
            * np.maximum(gas, 0.0) * np.maximum(o2, 0.0))


def _k_chi(char, m_s, p):
    chi = char / (char + m_s + 1e-12)
    return p.k_wood * (1.0 - chi) + p.k_char * chi


def conduction_energy(T, char, m_s, p):
    """Interior-conserving 7-point Fourier flux + ledgered boundary loss.

    Returns (dE_cell, boundary_loss_total): dE_cell sums to boundary_loss_total over
    the domain (interior face fluxes cancel exactly, by antisymmetric accumulation).
    """
    k = _k_chi(char, m_s, p)
    dE = np.zeros_like(T)
    inv_dx2 = 1.0 / (p.dx * p.dx)
    for ax in range(3):
        kf = 0.5 * (np.take(k, range(0, T.shape[ax] - 1), axis=ax)
                    + np.take(k, range(1, T.shape[ax]), axis=ax))
        lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[ax] = slice(0, T.shape[ax] - 1); hi[ax] = slice(1, T.shape[ax])
        flux = kf * (T[tuple(hi)] - T[tuple(lo)]) * inv_dx2   # from lo into hi
        dE[tuple(lo)] += flux
        dE[tuple(hi)] -= flux
    # boundary heat loss on all 6 outer faces -> ambient (ledgered)
    bloss = np.zeros_like(T)
    face = p.h_loss * inv_dx2
    for ax in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[ax] = 0; s1[ax] = T.shape[ax] - 1
        bloss[tuple(s0)] += face * (p.T_amb - T[tuple(s0)])
        bloss[tuple(s1)] += face * (p.T_amb - T[tuple(s1)])
    dE += bloss
    return dE, float(bloss.sum())


def o2_boundary_influx(o2, p):
    """O2 replenishment on the 6 outer faces toward o2_amb. Returns (do2_cell, total)."""
    src = np.zeros_like(o2)
    for ax in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[ax] = 0; s1[ax] = o2.shape[ax] - 1
        src[tuple(s0)] += p.o2_influx * (p.o2_amb - o2[tuple(s0)])
        src[tuple(s1)] += p.o2_influx * (p.o2_amb - o2[tuple(s1)])
    return src, float(src.sum())


# ---------- operators: contributions + ledger + envelope (the split interface) ----------

def op_pyrolysis(st, p):
    r = pyrolysis_rate(st["T"], st["m_s"], p)
    deltas = {
        "m_s": -r,
        "gas": +p.nu_g * r,
        "char": +p.nu_c * r,
        "T":   -(p.dH_py * r) / p.C_V,       # endothermic
    }
    ledger = {"pyrolysis_endo": float(-(p.dH_py * r).sum())}   # energy removed (<0)
    return deltas, ledger


def op_combustion(st, p):
    r = combustion_rate(st["T"], st["gas"], st["o2"], p)
    deltas = {
        "gas": -r,
        "o2":  -p.s_o2 * r,
        "T":   +(p.dH_cb * r) / p.C_V,       # exothermic
    }
    # exhaust vented (gas + O2 consumed leave as products) -> ledgered mass sink
    ledger = {"combustion_exo": float((p.dH_cb * r).sum()),
              "exhaust_vent": float(((1.0 + p.s_o2) * r).sum())}
    return deltas, ledger


def op_conduction(st, p):
    dE, bloss = conduction_energy(st["T"], st["char"], st["m_s"], p)
    deltas = {"T": dE / p.C_V}
    ledger = {"boundary_heat_loss": bloss}
    return deltas, ledger


def op_o2_supply(st, p):
    src, tot = o2_boundary_influx(st["o2"], p)
    return {"o2": src}, {"o2_influx": tot}


def op_charge_dissipation(st, p):
    d = -p.lambda_q * st["q"]
    return {"q": d}, {"charge_dissipated": float(d.sum())}


def op_char_weakening(st, p):
    """TRANSITION (not a conserved-bus contribution): strength S = S0 (1 - chi)."""
    chi = st["char"] / (st["char"] + st["m_s"] + 1e-12)
    return {"S": p.S0 * (1.0 - chi / p.chi_max)}


# the additive contribution operators (transitions handled separately by the runtime)
CONTRIBUTION_OPS = {
    "pyrolysis": op_pyrolysis,
    "combustion": op_combustion,
    "conduction": op_conduction,
    "o2_supply": op_o2_supply,
    "charge_dissipation": op_charge_dissipation,
}


# ---------- per-operator validity envelopes (marginal input checks) ----------

def envelopes(st, p):
    """Per-operator 'in distribution?' on each operator's OWN input marginals.

    Returns {op_name: bool array} — True where the operator considers its local
    inputs within its calibrated range. These are necessary-but-insufficient: a
    composite event can leave every marginal in-range while the joint state is OOD.
    """
    T = st["T"]
    return {
        "pyrolysis":  (T >= p.py_T[0]) & (T <= p.py_T[1]),
        "combustion": (T >= p.cb_T[0]) & (T <= p.cb_T[1]),
        "conduction": np.ones_like(T, dtype=bool),
    }


# ---------- the coupled monolithic RHS (used by the oracle + governing residual) ----------

def coupled_rhs(st, p):
    """F_coupled(U): the fully-coupled time-derivative of every field at state st.

    Same rate laws as the operators, summed WITHOUT splitting. Used by the monolithic
    implicit integrator and to evaluate the governing-equation residual of a committed
    split state.
    """
    r_py = pyrolysis_rate(st["T"], st["m_s"], p)
    r_cb = combustion_rate(st["T"], st["gas"], st["o2"], p)
    dE_cond, _ = conduction_energy(st["T"], st["char"], st["m_s"], p)
    o2_src, _ = o2_boundary_influx(st["o2"], p)
    return {
        "T":   (p.dH_cb * r_cb - p.dH_py * r_py) / p.C_V + dE_cond / p.C_V,
        "m_s": -r_py,
        "gas": p.nu_g * r_py - r_cb,
        "o2":  -p.s_o2 * r_cb + o2_src,
        "char": p.nu_c * r_py,
        "q":   -p.lambda_q * st["q"],
    }


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    p = FireParams()
    N = 6
    st = {
        "T": np.full((N, N, N), 700.0),
        "m_s": np.ones((N, N, N)),
        "gas": np.full((N, N, N), 0.1),
        "o2": np.full((N, N, N), 0.2),
        "char": np.zeros((N, N, N)),
        "q": np.zeros((N, N, N)),
    }
    print("1) rates at T=700:")
    print("   pyrolysis  =", pyrolysis_rate(st["T"], st["m_s"], p).mean())
    print("   combustion =", combustion_rate(st["T"], st["gas"], st["o2"], p).mean())

    print("2) conduction interior conservation (uniform T -> ~0 interior flux):")
    st["T"][3, 3, 3] = 1200.0
    dE, bloss = conduction_energy(st["T"], st["char"], st["m_s"], p)
    print(f"   sum(dE) = {dE.sum():.3e}   boundary_loss = {bloss:.3e}   "
          f"(sum(dE) must equal boundary_loss)")

    print("3) contribution-op mass balance (pyrolysis: dm_s = -(dgas+dchar)):")
    d, _ = op_pyrolysis(st, p)
    resid = (d["m_s"] + d["gas"] + d["char"])
    print(f"   max |dm_s + dgas + dchar| = {np.abs(resid).max():.3e}")

    print("4) envelopes at T=700 (in-range):",
          {k: bool(v.all()) for k, v in envelopes(st, p).items()})
