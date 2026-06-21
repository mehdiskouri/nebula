"""
Fire operators — the four constitutive transfer laws of the "tree on fire"
(ARCHITECTURE §III.3; Decision #13: package the transfer operator, not the phenomenon).
Verified by V0.3 (conservation + composite-OOD), V1.2 (split stability), V1.3 (Jensen).

Each operator READS shared state and emits only additive contributions into conserved
buses (combustion + pyrolysis + conduction compose freely) plus a ledger of boundary /
source-sink transfers the audit balances against. The phenomenon "fire" is the fixed
point of these coupled through T, gas (volatiles), O2, char. Kinetic constants are
documented PLAUSIBILITY-ENGINE values (ARCHITECTURE Part VII), tuned only for a stable
in-distribution burn + an impulse runaway.

State fields (each an (N,N,N) array): T [K], m_s solid wood, gas volatiles, o2 oxygen,
char, q charge. Heat capacity C_V folds T<->energy. Rates are per unit time.

Ported verbatim-in-behaviour from src/verification/oracles/{fire_operators,bus_runtime}.py
(frozen oracles) and re-expressed as a core.buses.Domain (the field-agnostic runtime).
"""
from dataclasses import dataclass

import numpy as np

from ..core.buses import Domain
from ..core.schema import Field, Operator

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
    # per-operator validity envelopes (the marginal each operator self-checks; V0.3 shows
    # they are necessary-but-insufficient -- the conservation audit is the primary monitor).
    py_T: tuple = (300.0, 900.0)
    cb_T: tuple = (400.0, 1400.0)


# ---------- rate laws ----------

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

    Returns (dE_cell, boundary_loss_total): dE_cell sums to boundary_loss_total over the
    domain (interior face fluxes cancel exactly, by antisymmetric accumulation).
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


# ---------- operators: contributions + ledger (the split interface) ----------

def op_pyrolysis(st, p):
    r = pyrolysis_rate(st["T"], st["m_s"], p)
    deltas = {"m_s": -r, "gas": +p.nu_g * r, "char": +p.nu_c * r, "T": -(p.dH_py * r) / p.C_V}
    ledger = {"pyrolysis_endo": float(-(p.dH_py * r).sum())}
    return deltas, ledger


def op_combustion(st, p):
    r = combustion_rate(st["T"], st["gas"], st["o2"], p)
    deltas = {"gas": -r, "o2": -p.s_o2 * r, "T": +(p.dH_cb * r) / p.C_V}
    ledger = {"combustion_exo": float((p.dH_cb * r).sum()),
              "exhaust_vent": float(((1.0 + p.s_o2) * r).sum())}
    return deltas, ledger


def op_conduction(st, p):
    dE, bloss = conduction_energy(st["T"], st["char"], st["m_s"], p)
    return {"T": dE / p.C_V}, {"boundary_heat_loss": bloss}


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


def coupled_rhs(st, p):
    """F_coupled(U): the fully-coupled time-derivative of every field (no splitting).

    The monolithic RHS used to evaluate the governing-equation residual of a committed
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


def envelopes(st, p):
    """Per-operator 'in distribution?' on each operator's OWN input marginals (necessary
    but insufficient -- a composite event can keep every marginal in-range while the joint
    state is OOD; V0.3)."""
    T = st["T"]
    return {
        "pyrolysis":  (T >= p.py_T[0]) & (T <= p.py_T[1]),
        "combustion": (T >= p.cb_T[0]) & (T <= p.cb_T[1]),
        "conduction": np.ones_like(T, dtype=bool),
    }


# ---------- the bespoke fire conservation audit (ported from bus_runtime._audit) ----------

def fire_audit(old, new, staged, ledger, p, dt):
    """Per-bus relative conservation residual.

    residual = |committed Delta(total) - ledgered/intended net transfer| / gross flow,
    where gross flow = total absolute staged movement that step. ~0 in-distribution; a
    spike is the composite-OOD symptom per-operator envelopes miss (V0.3, the PRIMARY monitor).
    """
    res = {}
    dE = (new["T"].sum() - old["T"].sum()) * p.C_V
    led_E = (ledger.get("combustion_exo", 0.0) + ledger.get("pyrolysis_endo", 0.0)
             + ledger.get("boundary_heat_loss", 0.0) + ledger.get("lightning_energy", 0.0))
    tp_E = float(np.abs(staged["T"] * dt).sum()) * p.C_V + abs(ledger.get("lightning_energy", 0.0))
    res["energy"] = abs(dE - led_E) / (tp_E + 1e-30)

    dM = ((new["m_s"] + new["gas"] + new["char"] + new["o2"]).sum()
          - (old["m_s"] + old["gas"] + old["char"] + old["o2"]).sum())
    led_M = ledger.get("o2_influx", 0.0) - ledger.get("exhaust_vent", 0.0)
    tp_M = float(sum(np.abs(staged[f] * dt).sum() for f in ("m_s", "gas", "char", "o2")))
    res["mass"] = abs(dM - led_M) / (tp_M + 1e-30)

    do2 = new["o2"].sum() - old["o2"].sum()
    intended_do2 = (staged["o2"] * dt).sum()
    tp_o2 = float(np.abs(staged["o2"] * dt).sum())
    res["o2"] = abs(do2 - intended_do2) / (tp_o2 + 1e-30)

    dq = new["q"].sum() - old["q"].sum()
    intended_dq = (staged["q"] * dt).sum()
    tp_q = float(np.abs(staged["q"] * dt).sum()) + abs(ledger.get("lightning_charge", 0.0))
    res["charge"] = abs(dq - intended_dq) / (tp_q + 1e-30)
    return res


# ---------- state + domain + ignition source ----------

def make_state(N, T0=320.0, m_s0=1.0, gas0=0.02, o2=0.23, char0=0.0):
    return {
        "T":   np.full((N, N, N), float(T0)),
        "m_s": np.full((N, N, N), float(m_s0)),
        "gas": np.full((N, N, N), float(gas0)),
        "o2":  np.full((N, N, N), float(o2)),
        "char": np.full((N, N, N), float(char0)),
        "q":   np.zeros((N, N, N)),
    }


# canonical contribution-op order matching the frozen oracle (for bit-exact regression).
ORACLE_OP_ORDER = ("pyrolysis", "combustion", "conduction", "o2_supply", "charge_dissipation")

FIRE_FIELDS = (
    Field("T", bus="energy"),
    Field("m_s", bus="mass", clamp_min=0.0),
    Field("gas", bus="mass", clamp_min=0.0),
    Field("char", bus="mass", clamp_min=0.0),
    Field("o2", bus="mass", clamp_min=0.0),
    Field("q", bus="charge", clamp_min=0.0),
)


def fire_domain(params=None) -> Domain:
    """Assemble the fire law-domain: the 4 transfer operators + supply/dissipation + the
    char-weakening transition, with the bespoke fire conservation audit."""
    p = params if params is not None else FireParams()
    ops = (
        Operator("pyrolysis", reads=("T", "m_s"), contribute=op_pyrolysis),
        Operator("combustion", reads=("T", "gas", "o2"), contribute=op_combustion),
        Operator("conduction", reads=("T", "char", "m_s"), contribute=op_conduction),
        Operator("o2_supply", reads=("o2",), contribute=op_o2_supply),
        Operator("charge_dissipation", reads=("q",), contribute=op_charge_dissipation),
        Operator("char_weakening", reads=("char", "m_s"), transition=op_char_weakening, priority=0),
    )
    return Domain("fire", FIRE_FIELDS, ops, p, audit_fn=fire_audit, coupled_rhs=coupled_rhs)


def ignition(mask, energy, charge=0.0):
    """Build a `sources` callable depositing a composite heat (+charge) impulse over `mask`.

    Deposited as a rate so that *dt yields exactly `energy`/`charge` (the oracle's lightning
    convention); ledgered as 'lightning_energy'/'lightning_charge' for the fire audit.
    """
    def src(staged, st, p, dt):
        ncell = max(int(np.count_nonzero(mask)), 1)
        staged["T"][mask] += (energy / p.C_V) / ncell / dt
        if charge:
            staged["q"][mask] += charge / ncell / dt
        return {"lightning_energy": float(energy), "lightning_charge": float(charge)}
    return src


if __name__ == "__main__":
    from ..core import buses
    np.set_printoptions(precision=4, suppress=True)
    dom = fire_domain()
    N = 8
    st = make_state(N)

    # 1) in-distribution single step: all bus residuals ~0 (no clamp at gentle rates).
    _, led, audit, gov = buses.step(dom, st, dt=1e-4, op_order=ORACLE_OP_ORDER)
    print("1) in-distribution audit:", {k: f"{v:.1e}" for k, v in audit.items()}, f" gov={gov:.2e}")
    assert max(audit.values()) < 1e-6

    # 2) order-independence of additive contributions (divergence below solver tol).
    import itertools
    base, *_ = buses.step(dom, st, dt=1e-4)
    worst = 0.0
    for order in list(itertools.permutations(ORACLE_OP_ORDER))[:12]:
        alt, *_ = buses.step(dom, st, dt=1e-4, op_order=order)
        worst = max(worst, max(float(np.abs(alt[f] - base[f]).max()) for f in dom.field_names))
    print(f"2) contribution order-independence: max divergence over orders = {worst:.2e}")
    assert worst < 1e-9

    # 3) contention: hot, fuel-rich, O2-poor cell over a big step -> clamp -> audit spike.
    st2 = make_state(N, T0=1500.0, gas0=5.0, o2=1e-3)
    _, _, audit2, _ = buses.step(dom, st2, dt=1e-1, op_order=ORACLE_OP_ORDER)
    print("3) contention audit:", {k: f"{v:.1e}" for k, v in audit2.items()}, " (mass/o2 should spike)")
    print("\nfire operators self-checks passed.")
