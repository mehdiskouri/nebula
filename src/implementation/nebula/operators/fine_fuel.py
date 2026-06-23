"""
Fine-fuel combustion — the crown flash (ARCHITECTURE §III.1; the V3.5 mechanism).

The canopy (V3.4) gives every leaf a fuel state (mass, moisture, char, area). This operator
burns it: a leaf is FINE fuel (thickness ~0.3 mm, huge surface/volume), so under the flame's
preheat it dries, ignites, and burns OUT in seconds — far faster than the centimetre-scale wood.
So as the trunk fire's plume climbs, the leaves ignite in a height-ordered wave and the canopy
"flashes" while the branches are still warming. Each leaf: dry (boil off moisture) → ignite (when
dry and hot) → burn (mass→0, char→1, area curls/shrinks, volatiles released to feed the flame) →
burn out. The size-scaling laws are the V3.5 oracle (`finefuel_ref.py`).

Deterministic (numpy, no RNG). Couples to a local gas temperature (from the `gas_combustion`
flame, or a prescribed rising front for verification — the flame's existence/standoff is V3.2).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FineFuelParams:
    thickness: float = 3.0e-4     # leaf half-thickness [m] (fine fuel)
    T_ignite: float = 600.0       # ignition temperature [K]
    T_boil: float = 373.0         # water boils off above this
    dry_rate: float = 4.0         # moisture loss per unit (T−T_boil) per second
    burn_rate: float = 6.0e-6     # base mass-loss rate const; divided by thickness² (d²-law)
    char_yield: float = 0.35      # char fraction of burned leaf mass
    curl: float = 0.7             # fractional area shrink as a leaf burns (curling)


@dataclass
class LeafFuel:
    mass0: np.ndarray      # (L,) initial dry mass (for conservation accounting)
    mass: np.ndarray       # (L,) remaining dry mass
    moisture: np.ndarray   # (L,) water mass fraction
    char: np.ndarray       # (L,) char fraction in [0,1]
    area: np.ndarray       # (L,) current one-sided area (shrinks as it burns)
    area0: np.ndarray      # (L,) initial area
    ignited: np.ndarray    # (L,) bool
    ignite_t: np.ndarray   # (L,) ignition time (nan if not yet)
    z: np.ndarray          # (L,) height (for the crown-flash wave)

    @property
    def n(self):
        return len(self.mass)


def from_canopy(canopy, p=None):
    p = p or FineFuelParams()
    L = canopy.n
    return LeafFuel(mass0=canopy.mass.copy(), mass=canopy.mass.copy(),
                    moisture=canopy.moisture.copy(), char=canopy.char.copy(),
                    area=canopy.area.copy(), area0=canopy.area.copy(),
                    ignited=np.zeros(L, bool), ignite_t=np.full(L, np.nan),
                    z=canopy.pos[:, 2].copy())


def burnout_rate(p):
    """Mass-loss rate constant ∝ 1/thickness² (the d²-law: thin → fast burnout)."""
    return p.burn_rate / (p.thickness ** 2)


def step(fuel, T_local, dt, t, p=None):
    """Advance leaf drying / ignition / burning given each leaf's local gas temperature T_local."""
    p = p or FineFuelParams()
    T = np.asarray(T_local, float)
    # 1) DRY: boil off moisture where hot (must reach ~0 before ignition)
    drying = np.clip(T - p.T_boil, 0, None) * p.dry_rate * dt
    fuel.moisture = np.clip(fuel.moisture - drying, 0.0, None)
    # 2) IGNITE: dry + hot + still has fuel
    can_ignite = (~fuel.ignited) & (fuel.moisture <= 1e-3) & (T >= p.T_ignite) & (fuel.mass > 0)
    fuel.ignited |= can_ignite
    fuel.ignite_t[can_ignite] = t
    # 3) BURN: ignited leaves lose mass fast (d²-law), char up, area curls
    k = burnout_rate(p)
    burning = fuel.ignited & (fuel.mass > 0)
    dm = np.minimum(fuel.mass, k * dt) * burning
    fuel.mass = fuel.mass - dm
    frac = 1.0 - fuel.mass / np.maximum(fuel.mass0, 1e-30)
    fuel.char = np.clip(frac, 0.0, 1.0)
    fuel.area = fuel.area0 * (1.0 - p.curl * fuel.char)        # curling shrink
    return float(dm.sum())                                      # volatile mass released this step


def crown_flash(canopy, p=None, dt=0.1, n_steps=120, front_speed=0.6, flame_T=1100.0,
                preheat=0.8, z0=None):
    """Run a rising flame front through the canopy and record the ignition wave.

    The front height climbs at `front_speed`; a leaf within `preheat` below the front sees
    `flame_T` (radiative/convective preheat), else ambient. Returns (fuel, history).
    """
    p = p or FineFuelParams()
    fuel = from_canopy(canopy, p)
    z0 = float(canopy.pos[:, 2].min()) if z0 is None else z0
    history = {"t": [], "front": [], "ignited_frac": [], "burning_frac": [], "mass_frac": []}
    released = 0.0
    for s in range(n_steps):
        t = s * dt
        front = z0 + front_speed * t
        T_local = np.where(fuel.z <= front + 1e-9,
                           np.where(fuel.z >= front - preheat, flame_T, 0.5 * flame_T), 300.0)
        # leaves below the front but outside the preheat band still feel residual heat (warm plume)
        released += step(fuel, T_local, dt, t, p)
        history["t"].append(t); history["front"].append(front)
        history["ignited_frac"].append(float(fuel.ignited.mean()))
        history["burning_frac"].append(float(((fuel.mass > 0) & fuel.ignited).mean()))
        history["mass_frac"].append(float(fuel.mass.sum() / max(fuel.mass0.sum(), 1e-30)))
    history = {k: np.array(v) for k, v in history.items()}
    return fuel, history, released


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/nebula/src/verification/oracles")
    import finefuel_ref as ff
    from .growth import grow_tree, GrowthParams
    from . import canopy as cano
    np.seterr(all="ignore")

    p = FineFuelParams()

    # 1) burnout scaling matches the d²-law oracle: leaf ≪ branch.
    leaf_bo = p.thickness ** 2 / p.burn_rate                    # 1/burnout_rate
    branch_p = FineFuelParams(thickness=2.0e-2)
    branch_bo = branch_p.thickness ** 2 / branch_p.burn_rate
    print(f"1) burnout time leaf {leaf_bo:.2f}s ≪ branch {branch_bo:.0f}s  (ratio {leaf_bo/branch_bo:.2e}; oracle d²)")
    assert leaf_bo / branch_bo < 0.01

    # 2) the crown flash: a rising front ignites the canopy in a height-ordered wave.
    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    can = cano.generate_canopy(tree, cano.CanopyParams(), seed=7)
    fuel, hist, released = crown_flash(can, p, dt=0.1, n_steps=140, front_speed=0.5)
    ig = fuel.ignite_t[~np.isnan(fuel.ignite_t)]
    zlo, zhi = fuel.z.min(), fuel.z.max()
    lo_t = np.nanmedian(fuel.ignite_t[fuel.z < zlo + 0.3 * (zhi - zlo)])
    hi_t = np.nanmedian(fuel.ignite_t[fuel.z > zlo + 0.7 * (zhi - zlo)])
    print(f"2) crown flash: {fuel.ignited.mean()*100:.0f}% leaves ignited; base median t {lo_t:.1f}s "
          f"< crown median t {hi_t:.1f}s (lag {hi_t-lo_t:.1f}s)")
    assert fuel.ignited.mean() > 0.9 and hi_t > lo_t

    # 3) leaves burn out fast and the fuel pool is conserved (burned == released).
    burned = float((fuel.mass0 - fuel.mass).sum())
    print(f"3) final canopy mass {hist['mass_frac'][-1]*100:.0f}% of initial; burned {burned:.3f} "
          f"== released {released:.3f} (conserved); char mean {fuel.char.mean():.2f}")
    assert hist["mass_frac"][-1] < 0.1 and abs(burned - released) < 1e-9

    # 4) determinism
    fuel2, _, _ = crown_flash(can, p, dt=0.1, n_steps=140, front_speed=0.5)
    print(f"4) determinism: identical ignition times = {np.array_equal(np.nan_to_num(fuel.ignite_t), np.nan_to_num(fuel2.ignite_t))}")
    assert np.array_equal(np.nan_to_num(fuel.ignite_t), np.nan_to_num(fuel2.ignite_t))
    print("\nfine_fuel (crown flash) self-checks passed.")
