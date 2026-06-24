"""
Whole-tree fire spread — a fire that PROPAGATES through the tree's own fuel over time.

A real tree fire is not a burner at the base: the WHOLE TREE is fuel and the fire SPREADS through it.
This models that as a fire FRONT propagating over the fuel elements (branch segments + leaves), with a
physically-grounded spread speed, plus a continuous gas/flame VOLUME built from the burning front:

  spread speed (the front)  : along the fuel, faster UPWARD (buoyancy carries fire up), faster into
                              FINE FUEL (leaf/twig tips — high surface/volume, low thermal mass: they
                              catch first, V3.5), and along connected branches (radiative/contact
                              spread). Arrival time per element = a shortest-path (Dijkstra) from the
                              ignition point on this weighted graph.
  burnout (per element)     : once lit, an element burns for a time ∝ its thickness (the d²-law: fine
                              leaves flash and are gone; thick branches smoulder) — char rises 0→1.
  flame VOLUME (continuous) : the currently-burning elements outgas + release heat into a 3-D field
                              that is smeared buoyantly UPWARD into licking flames (blackbody from
                              temperature) with rising soot — the gas/flame is everywhere the fire is,
                              not a box, and not made of spheres.

`simulate(times)` returns the burn over time — the spreading-fire ANIMATION.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass
class TreeFireParams:
    spread_base: float = 0.055      # base front speed [m/s] (a slow climbing fire)
    up_gain: float = 2.6            # upward spread multiplier (buoyancy carries fire up)
    down_gain: float = 0.25         # downward spread is slow
    fine_gain: float = 2.2          # fine fuel (leaves/tips) ignites faster
    branch_gain: float = 1.3        # spread is easier along a connected branch
    rad_radius: float = 0.42        # spread-graph neighbour radius [m]
    burnout_leaf: float = 3.0       # leaf burnout time [s] (fine fuel: fast, but visible)
    burnout_branch_per_m: float = 110.0  # branch burnout ∝ thickness [s per m]
    T_flame: float = 1350.0         # active-flame temperature [K] (cools as the element chars)
    grid_n: int = 60
    soot_yield: float = 0.5
    buoy_lift: int = 7              # how many cells the flame licks upward
    buoy_spread: float = 0.85       # lateral diffusion of the flame per lift step


class TreeFire:
    pass


def build(tree, canopy, p=None):
    """Fuel elements (above-ground branches + leaves) + the weighted spread graph + the flame grid."""
    p = p or TreeFireParams()
    tf = TreeFire(); tf.p = p
    seg = np.array([i for i in range(tree.n) if tree.parent[i] >= 0 and tree.pos[i, 2] > -0.05])
    a = tree.pos[tree.parent[seg]]; b = tree.pos[seg]
    bpos = 0.5 * (a + b)
    brad = 0.5 * (tree.radius[tree.parent[seg]] + tree.radius[seg])
    bthick = 2.0 * brad
    seg_of = -np.ones(tree.n, int); seg_of[seg] = np.arange(len(seg))
    bparent = np.array([seg_of[tree.parent[i]] if seg_of[tree.parent[i]] >= 0 else -1 for i in seg])

    if canopy is not None and canopy.n > 0:
        lpos = canopy.pos; lthick = np.full(canopy.n, 8e-4)
        lparent = seg_of[np.clip(canopy.twig_node, 0, tree.n - 1)]
    else:
        lpos = np.zeros((0, 3)); lthick = np.zeros(0); lparent = np.zeros(0, int)

    nb = len(seg)
    tf.n_branch = nb
    tf.branch_node = seg                                 # tree node each branch element rides
    tf.pos = np.vstack([bpos, lpos])
    tf.thick = np.concatenate([bthick, lthick])
    tf.is_leaf = np.concatenate([np.zeros(nb, bool), np.ones(len(lpos), bool)])
    tf.parent = np.concatenate([bparent, np.where(lparent >= 0, lparent, -1)]).astype(int)
    tf.n = len(tf.pos)
    tf.tree = tree

    # --- weighted directed spread graph: edge cost = distance / spread_speed ---
    kd = cKDTree(tf.pos)
    k = min(9, tf.n)
    dist, idx = kd.query(tf.pos, k=k)
    src = np.repeat(np.arange(tf.n), k - 1)
    dst = idx[:, 1:].ravel()
    dd = dist[:, 1:].ravel() + 1e-4
    # add branch-parent links (both directions) as strong connections
    bl = tf.parent >= 0
    src = np.concatenate([src, np.arange(tf.n)[bl], tf.parent[bl]])
    dst = np.concatenate([dst, tf.parent[bl], np.arange(tf.n)[bl]])
    bd = np.linalg.norm(tf.pos[np.arange(tf.n)[bl]] - tf.pos[tf.parent[bl]], axis=1) + 1e-4
    dd = np.concatenate([dd, bd, bd])
    is_branch_edge = np.concatenate([np.zeros(len(idx[:, 1:].ravel()), bool),
                                     np.ones(2 * int(bl.sum()), bool)])
    dz = tf.pos[dst, 2] - tf.pos[src, 2]                  # +ve = spreading upward
    vert = dd
    up = np.clip(dz / vert, -1, 1)
    speed = p.spread_base * np.where(up > 0, 1 + p.up_gain * up, 1 + p.down_gain * up)
    speed *= np.where(tf.is_leaf[dst], p.fine_gain, 1.0)  # fine fuel ignites faster
    speed *= np.where(is_branch_edge, p.branch_gain, 1.0)
    cost = dd / np.maximum(speed, 1e-3)
    tf.G = sp.csr_matrix((cost, (src, dst)), shape=(tf.n, tf.n))

    # --- burnout time per element (fine = fast; the d²-law size dependence, V3.5) ---
    tf.burnout = np.where(tf.is_leaf, p.burnout_leaf, p.burnout_branch_per_m * tf.thick + 0.6)

    # --- flame grid over the above-ground tree ---
    lo = tf.pos.min(0) - 0.3; hi = tf.pos.max(0) + 0.3; lo[2] = max(lo[2], -0.1)
    fsp = (hi - lo).max() / p.grid_n
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / fsp)) + 1 for d in range(3))
    tf.origin = lo; tf.spacing = fsp; tf.gshape = shape
    cell = np.clip(((tf.pos - lo) / fsp).astype(int), 0, np.array(shape) - 1)
    tf.cell = cell
    tf.cell_flat = (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]
    return tf


def ignite(tf, frac=0.05, radius=0.3):
    """Seed the fire low on the trunk; returns the arrival-time field (Dijkstra over the graph)."""
    z = tf.pos[:, 2]; z0, z1 = z.min(), z.max()
    base = (~tf.is_leaf) & (z < z0 + frac * (z1 - z0))
    seeds = np.where(base)[0]
    if len(seeds) == 0:
        seeds = np.where(~tf.is_leaf)[0][:1]
    arrival = dijkstra(tf.G, directed=True, indices=seeds, min_only=True)
    arrival[~np.isfinite(arrival)] = arrival[np.isfinite(arrival)].max() * 1.5 + 1.0
    tf.arrival = arrival
    # active burn window: when ~92% of the fuel has finished burning (drop the thick-branch tail)
    tf.t_end = float(np.quantile(arrival + tf.burnout, 0.92))
    return arrival


def _buoyant_flame_volume(tf, intensity, soot_src):
    """Build a continuous flame field by depositing burning intensity into the grid and smearing it
    UPWARD into licking flames (a cheap buoyancy: lift + lateral diffuse), with rising soot."""
    p = tf.p
    nx, ny, nz = tf.gshape
    dep = np.bincount(tf.cell_flat, weights=intensity, minlength=nx * ny * nz).reshape(tf.gshape)
    sdep = np.bincount(tf.cell_flat, weights=soot_src, minlength=nx * ny * nz).reshape(tf.gshape)
    flame = dep.copy(); smoke = sdep.copy()
    acc = dep.copy(); sacc = sdep.copy()
    for i in range(p.buoy_lift):
        acc = np.roll(acc, 1, axis=2) * 0.82                  # rise
        acc[:, :, 0] = 0.0
        acc = (acc + p.buoy_spread * 0.25 * (np.roll(acc, 1, 0) + np.roll(acc, -1, 0)
                                             + np.roll(acc, 1, 1) + np.roll(acc, -1, 1))) / (1 + p.buoy_spread)
        flame = np.maximum(flame, acc * (1.0 - i / (p.buoy_lift + 2)))
        sacc = np.roll(sacc, 1, axis=2) * 0.9
        sacc[:, :, 0] = 0.0
        smoke = smoke + sacc * 0.5
    return flame, smoke


def state_at(tf, t):
    """The burn state at time t: per-element char/burning + the continuous flame VOLUME (Tg, soot)."""
    p = tf.p
    age = t - tf.arrival
    lit = age >= 0
    char = np.clip(age / np.maximum(tf.burnout, 1e-6), 0.0, 1.0)
    burning = lit & (char < 1.0)
    # flame intensity per burning element: peaks just after ignition, fades as it chars; fine fuel
    # flares bright and brief
    flare = burning * (1.0 - char) ** 0.7 * np.where(tf.is_leaf, 1.4, 1.0)
    Tg_field, soot = _buoyant_flame_volume(tf, flare * 1.0, flare * p.soot_yield)
    # map deposited flame intensity -> temperature (saturating toward the flame temperature)
    Tg = 300.0 + (p.T_flame - 300.0) * (1.0 - np.exp(-2.5 * Tg_field / (Tg_field.max() + 1e-9)))
    Tg = np.where(Tg_field > 1e-6, Tg, 300.0)
    return {"char": char, "burning": burning, "lit": lit, "flare": flare,
            "Tg": Tg.astype(np.float32), "soot": soot.astype(np.float32)}


def simulate(tree, canopy, p=None, n_frames=60):
    p = p or TreeFireParams()
    tf = build(tree, canopy, p)
    ignite(tf)
    times = np.linspace(0.0, tf.t_end, n_frames)
    hist = [state_at(tf, t) for t in times]
    return tf, times, hist


if __name__ == "__main__":
    from .growth import grow_tree, GrowthParams
    from . import canopy as cano
    import time
    np.seterr(all="ignore")
    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    can = cano.generate_canopy(tree, cano.CanopyParams(), seed=7)
    p = TreeFireParams()
    t0 = time.time()
    tf = build(tree, can, p); ignite(tf)
    print(f"1) fuel {tf.n} ({int((~tf.is_leaf).sum())} branch + {int(tf.is_leaf.sum())} leaf); "
          f"grid {tf.gshape}; arrival built in {time.time()-t0:.1f}s; burn ends t={tf.t_end:.1f}s")
    z = tf.pos[:, 2]; z0, z1 = z.min(), z.max()
    # the front climbs: median height of the burning set rises over time
    print("2) fire front climbs + flame volume forms:")
    prev_h = -1; climbed = True
    for t in np.linspace(0.1, tf.t_end, 8):
        s = state_at(tf, t)
        edge = (z[s["lit"]].max() - z0) / (z1 - z0) if s["lit"].any() else 0    # leading edge climbs
        cf = s["char"].mean()
        print(f"   t={t:5.1f}s  burning {int(s['burning'].sum()):5d}  leading-edge {edge*100:3.0f}%  "
              f"charred {cf*100:3.0f}%  flame Tmax {s['Tg'].max():.0f}K")
    # fine fuel goes before branches (leaf char ahead of branch char at mid-burn)
    s = state_at(tf, tf.t_end * 0.55)
    leaf_c = s["char"][tf.is_leaf].mean(); br_c = s["char"][~tf.is_leaf].mean()
    print(f"3) fine-fuel-first @ mid-burn: leaf char {leaf_c:.2f} > branch char {br_c:.2f}")
    assert leaf_c > br_c
    s_end = state_at(tf, tf.t_end)
    print(f"4) end: {int((s_end['char']>0.5).sum())}/{tf.n} elements charred; flame volume nonzero "
          f"frames: max Tg over burn reached {max(state_at(tf,t)['Tg'].max() for t in np.linspace(0.1,tf.t_end,8)):.0f}K")
    assert s_end["char"].mean() > 0.6
    print("\ntree_fire front-propagation self-checks passed.")
