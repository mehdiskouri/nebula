"""
Reacting buoyant flow — the FLAME (ARCHITECTURE §III.3; the V3.2 mechanism).

Phase-0 combusted in place inside the wood voxels, so there was no flame: the volatile gas
never went anywhere and "fire" was orange spheres on the bark. This module couples the
buoyant transport (operators.flow, verified V3.1) with gas-phase combustion (the V0.3
`fire.combustion_rate` kinetics, reused) so the reaction happens where it physically does —
in the rising gas ABOVE the fuel, where the fuel-rich plume entrains oxidizer and reaches
the stoichiometric surface (Burke–Schumann). That standoff, and a flame height that grows
with fuel supply, are what make it read as a flame.

The cycle each step (gather→stage→reduce→commit, transport = conserved flux):
  1. INJECT  : the pyrolysis source adds fuel-rich gas (Z=1) + a little heat at the base —
               fuel-rich/oxidizer-poor, so it does NOT burn at the source.
  2. ENTRAIN : fresh oxidizer is replenished toward ambient at the open side boundaries.
  3. TRANSPORT: flow.step advects {T, gas, o2, soot, Z} on the buoyancy field (T drives it).
  4. REACT   : gas-phase combustion where gas+O2+T meet — semi-implicit (V1.2 discipline) so
               a reactant can never overshoot; releases heat (→ more buoyancy → taller flame)
               and yields soot (the V3.3 smoke/emission source).
  Z (mixture fraction) is a CONSERVED passive scalar — never consumed by reaction — so the
  Z=Z_st iso-surface is the in-sim Burke–Schumann flame-sheet reference (V3.2 oracle).

Deterministic (numpy fixed-order; V0.5). Reuses flow + fire unedited.
"""
from dataclasses import dataclass, field

import numpy as np

from . import flow
from . import fire as fo


@dataclass
class ReactingParams:
    fp: flow.FlowParams = field(default_factory=lambda: flow.FlowParams(dx=1.0, g=9.81,
                                beta=1.0 / 300.0, T_ref=300.0, cfl=0.8))
    # PHYSICAL flame kinetics (causal calibration, not a render hack): the Tier-0 `dH_cb=60` was
    # calibrated for the 0-D burn's CHAR FRACTION and is ~50× too small to reach a flame TEMPERATURE
    # (heat-of-combustion × gas fraction ≈ 12 K rise → a cool near-extinction flame). A physical heat
    # of combustion (dH_cb≈2500 in these units → adiabatic flame temp in the real wood-flame
    # 1300–1900 K range) with a reachable hot branch (Ta_cb≈4500) makes the gas flame SELF-SUSTAIN
    # HOTTER THAN ITS FUEL SOURCE — i.e. behave like a flame. The unedited Tier-0/1 fire keeps dH_cb=60
    # (this is a separate FireParams instance, so Tier-0/1 parity is untouched).
    firep: fo.FireParams = field(default_factory=lambda: fo.FireParams(Ta_cb=4500.0, dH_cb=2500.0))
    o2_amb: float = 0.23          # ambient oxidizer fraction
    o2_entrain: float = 1.0       # boundary entrainment rate toward o2_amb (fresh air in)
    fuel_rate: float = 2.0        # gas injected per unit time (flame size/temperature scale with it)
    T_inject: float = 700.0       # injected-gas temperature (physical pyrolysis temp)
    soot_yield: float = 0.06      # soot produced per unit gas burned
    soot_oxid: float = 0.4        # soot burnt out by O2 (rate coeff)
    T_pilot: float = 1400.0       # initial pilot kernel that lights the flame


def make_state(shape, p):
    """Quiescent ambient air + zero velocity. Scalars: T, gas, o2, soot, Z (mixture frac)."""
    T = np.full(shape, p.fp.T_ref)
    sc = {"T": T, "gas": np.zeros(shape), "o2": np.full(shape, p.o2_amb),
          "soot": np.zeros(shape), "Z": np.zeros(shape)}
    u, v, w = flow.zero_velocity(shape)
    return sc, (u, v, w)


def pilot(sc, mask):
    """Light the flame: a hot kernel so the first gas that entrains O2 ignites and self-sustains."""
    sc["T"][mask] = np.maximum(sc["T"][mask], 1400.0)


def _entrain_o2(o2, p, dt):
    """Replenish oxidizer toward ambient on the open (side + top) boundaries — fresh-air inflow."""
    for ax in range(3):
        for end in (0, -1):
            if ax == 2 and end == 0:
                continue                       # bottom is the (closed) ground plate
            s = [slice(None)] * 3; s[ax] = end
            o2[tuple(s)] += p.o2_entrain * (p.o2_amb - o2[tuple(s)]) * dt
    return o2


def react(sc, p, dt):
    """Gas-phase combustion (semi-implicit, capped) + soot formation/oxidation. Returns heat
    release rate field rr (for diagnostics). Z is untouched (conserved mixture fraction)."""
    T, gas, o2 = sc["T"], sc["gas"], sc["o2"]
    fp = p.firep
    rr = fo.combustion_rate(T, gas, o2, fp)                    # for diagnostics (pre-update)
    # semi-implicit depletion of gas (linear), capped by available O2 (V1.2 stability)
    k_cb = fp.A_cb * np.exp(-fp.Ta_cb / np.maximum(T, 1.0)) * np.maximum(o2, 0.0)
    burned = gas - gas / (1.0 + k_cb * dt)
    burned = np.minimum(burned, np.maximum(o2, 0.0) / fp.s_o2 * (1.0 - 1e-9))
    sc["gas"] = gas - burned
    sc["o2"] = o2 - fp.s_o2 * burned
    sc["T"] = T + (fp.dH_cb * burned) / fp.C_V
    sc["soot"] = np.clip(sc["soot"] + p.soot_yield * burned
                         - p.soot_oxid * sc["soot"] * np.maximum(sc["o2"], 0.0) * dt, 0.0, None)
    return rr


def step(sc, vel, p, dt, source=None):
    """One reacting-flow step over global `dt` (internally CFL-substepped by flow.step).

    source: boolean mask of fuel-injection cells (the burning wood surface). At those cells
    gas is added (Z driven to 1) and T held at the injection temperature.
    """
    u, v, w = vel
    if source is not None:
        sc["gas"][source] += p.fuel_rate * dt
        sc["Z"][source] = 1.0
        sc["T"][source] = np.maximum(sc["T"][source], p.T_inject)
    sc["o2"] = _entrain_o2(sc["o2"], p, dt)
    u, v, w, sc, info = flow.step(u, v, w, sc, p.fp, dt)     # buoyant transport of all scalars
    rr = react(sc, p, dt)                                      # gas-phase flame
    return sc, (u, v, w), info, rr


# ----------------------------------------------------------------------- flame diagnostics
def reaction_field(sc, p):
    """Heat-release rate field (W/cell) = dH_cb · combustion_rate."""
    return p.firep.dH_cb * fo.combustion_rate(sc["T"], sc["gas"], sc["o2"], p.firep)


def flame_metrics(sc, p, origin_z=0.0, dx=1.0, src_z_top=0.0, rr=None):
    """Standoff + height of the reaction zone.

    Returns dict: q_total (∫ heat release), z_react (HRR-weighted mean height), z_tip (top of
    the reaction zone), standoff (z_react − src_z_top). All heights in world z.
    """
    if rr is None:
        rr = reaction_field(sc, p)
    nz = rr.shape[2]
    z = origin_z + (np.arange(nz) + 0.5) * dx
    hrr_z = rr.sum(axis=(0, 1))                                # heat release per height
    q_total = float(hrr_z.sum())
    if q_total <= 1e-30:
        return {"q_total": 0.0, "z_react": origin_z, "z_tip": origin_z, "standoff": 0.0}
    z_react = float((hrr_z * z).sum() / q_total)
    thr = 0.02 * hrr_z.max()
    z_tip = float(z[hrr_z > thr].max()) if (hrr_z > thr).any() else z_react
    return {"q_total": q_total, "z_react": z_react, "z_tip": z_tip,
            "standoff": z_react - src_z_top}


def simulate(shape, p, source, n_steps, dt=0.5, pilot_mask=None, light_until=4,
             collect_from=None):
    """Run the reacting flow; return (sc, vel, history). history time-averages flame metrics
    over the last (n_steps − collect_from) steps for a steady reading."""
    sc, vel = make_state(shape, p)
    collect_from = collect_from if collect_from is not None else int(0.7 * n_steps)
    src_z_top = (np.argwhere(source)[:, 2].max() + 0.5) * p.fp.dx
    mets = []
    for n in range(n_steps):
        if pilot_mask is not None and n < light_until:
            pilot(sc, pilot_mask)
        sc, vel, info, rr = step(sc, vel, p, dt, source=source)
        if n >= collect_from:
            mets.append(flame_metrics(sc, p, dx=p.fp.dx, src_z_top=src_z_top, rr=rr))
    avg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]} if mets else {}
    return sc, vel, {"avg": avg, "src_z_top": src_z_top}


if __name__ == "__main__":
    np.seterr(all="ignore")
    nx, nz = 28, 64
    shape = (nx, nx, nz)
    p = ReactingParams()
    cx = nx // 2
    ix, iy = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    src = np.zeros(shape, bool)
    src[((ix - cx) ** 2 + (iy - cx) ** 2) <= 9, 0:2] = True     # fuel disk at the base
    pilot_mask = np.zeros(shape, bool)
    pilot_mask[((ix - cx) ** 2 + (iy - cx) ** 2) <= 9, 5:11] = True   # pilot just above source

    sc, vel, hist = simulate(shape, p, src, n_steps=70, dt=0.5, pilot_mask=pilot_mask)
    a = hist["avg"]
    print(f"1) flame: total HRR {a['q_total']:.1f}; reaction-zone z {a['z_react']:.2f}; "
          f"tip z {a['z_tip']:.2f}; source top z {hist['src_z_top']:.2f}")
    print(f"2) STANDOFF (z_react − src_top) = {a['standoff']:.2f}  (>0 ⇒ flame is above the fuel)")
    assert a["q_total"] > 0 and a["standoff"] > 0.5, "flame did not stand off the fuel"

    # 2b) THERMAL realism (the causal calibration): the gas flame self-sustains HOTTER than its fuel
    # source and reaches a physical wood-flame temperature — it behaves like a flame, not a warm blob.
    flame_T = float(sc["T"].max()); src_T = float(sc["T"][src].max())
    print(f"2b) flame peak {flame_T:.0f}K vs source {src_T:.0f}K; hotter-than-source {flame_T > src_T + 50}; "
          f"physical (1100–2100K) {1100 <= flame_T <= 2100}")
    assert flame_T > src_T + 50 and 1100 <= flame_T <= 2100, "flame is not a physical hot flame"

    # 3) extinction: starve oxidizer (no entrainment, low ambient) -> flame dies.
    p2 = ReactingParams(o2_entrain=0.0, o2_amb=0.0)
    sc2, _, h2 = simulate(shape, p2, src, n_steps=80, dt=0.5, pilot_mask=pilot_mask)
    print(f"3) extinction (no O2): total HRR {h2['avg']['q_total']:.3e} (≈0)")
    assert h2["avg"]["q_total"] < 0.01 * a["q_total"]
    print("\ngas_combustion (flame) self-checks passed.")
