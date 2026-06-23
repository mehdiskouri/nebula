"""
Canopy — leaves as derived, combustible fine fuel (ARCHITECTURE §III.1; the V3.4/V3.5 mechanism).

Phase-0 produced a bare skeleton (blocker #4): no foliage, so a "tree on fire" had nothing to
flash and read as a winter stick. A tree's dominant visual mass IS its canopy. This operator
deposits leaves on the slender canopy twigs the growth front already produced, arranged by
SPIRAL PHYLLOTAXIS at the golden angle ψ=137.5° (the arrangement that packs leaves evenly for
light capture — `phyllotaxis_ref.py` oracle, V3.4). Each leaf is also a FINE-FUEL element
(small mass, high surface/volume, live moisture) so it ignites and burns out far faster than
wood — the physics of the crown flash (V3.5). Leaves are grown, not authored: their positions,
density, and fuel state derive from the skeleton + a seeded phyllotactic rule.

Deterministic in (tree, params, seed): leaf jitter is hashed from the twig lineage (V1.8
discipline), never the global draw order.
"""
from dataclasses import dataclass

import numpy as np

from ..core.determinism import rng_from_key

GOLDEN_ANGLE_DEG = 137.50776405003785


@dataclass
class CanopyParams:
    twig_radius_max: float = 0.05    # nodes thinner than this (outer twigs) bear leaves
    min_order: int = 1               # don't put leaves on the trunk (order 0)
    leaves_per_node: int = 6         # leaves per twig whorl
    petiole: float = 0.05            # radial standoff of the leaf from the twig axis [m]
    leaf_area: float = 0.0035        # one-sided leaf area [m^2]
    leaf_thickness: float = 3.0e-4   # leaf half-thickness [m] (sets the fine-fuel burnout time)
    leaf_dry_mass: float = 0.0007    # dry mass per leaf [kg]
    leaf_moisture: float = 0.6       # live-fuel moisture (water mass fraction)
    angle_deg: float = GOLDEN_ANGLE_DEG   # phyllotactic divergence (control overrides to break it)
    jitter: float = 0.12             # small deterministic positional jitter (natural look)


@dataclass
class LeafCanopy:
    pos: np.ndarray        # (L,3) leaf centroids
    normal: np.ndarray     # (L,3) outward leaf normals
    azimuth: np.ndarray    # (L,) phyllotactic azimuth used (radians) — for the V3.4 check
    twig_node: np.ndarray  # (L,) the skeleton node each leaf rides (topple skinning + char sampling)
    area: np.ndarray       # (L,) one-sided area
    mass: np.ndarray       # (L,) dry fuel mass (depletes as it burns)
    moisture: np.ndarray   # (L,) water mass fraction (must boil off before ignition)
    char: np.ndarray       # (L,) char fraction in [0,1]

    @property
    def n(self):
        return len(self.pos)


def _frame(axis):
    """A stable orthonormal (u,v) perpendicular to `axis`."""
    d = axis / (np.linalg.norm(axis) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(d, u)
    return d, u, v


def leaf_bearing_nodes(tree, cp):
    """Indices of the slender outer canopy nodes that bear leaves (twigs, above ground)."""
    up = tree.pos[:, 2]
    return np.where((tree.radius < cp.twig_radius_max) & (up > 0.0)
                    & (tree.order >= cp.min_order) & (tree.parent >= 0))[0]


def generate_canopy(tree, cp=None, seed=0):
    """Grow the canopy: a LeafCanopy of leaves on the twigs, spiral-phyllotaxis arranged.

    The phyllotactic index winds CONTINUOUSLY along each shoot (cumulative over the leaf-bearing
    nodes in birth-generation order), so consecutive leaves differ in azimuth by the golden angle.
    """
    cp = cp or CanopyParams()
    angle = np.deg2rad(cp.angle_deg)
    nodes = leaf_bearing_nodes(tree, cp)
    # cumulative phyllotactic index: order leaf nodes by generation (a proxy for shoot arc length)
    order_by_gen = nodes[np.argsort(tree.gen[nodes], kind="stable")]
    cum = {int(n): k for k, n in enumerate(order_by_gen)}

    pos, normal, azim, tnode = [], [], [], []
    for i in nodes:
        j = int(tree.parent[i])
        axis = tree.pos[i] - tree.pos[j]
        if np.linalg.norm(axis) < 1e-9:
            continue
        d, u, v = _frame(axis)
        rng = rng_from_key("leaf", seed, int(i))
        base = cum[int(i)] * cp.leaves_per_node
        for m in range(cp.leaves_per_node):
            az = (base + m) * angle + cp.jitter * (rng.random() - 0.5)
            frac = (m + 0.5) / cp.leaves_per_node + cp.jitter * (rng.random() - 0.5)
            radial = np.cos(az) * u + np.sin(az) * v
            # leaf droops slightly outward+down: outward radial, with a touch of the twig axis
            outward = radial * (tree.radius[i] + cp.petiole)
            along = d * (np.clip(frac, 0.0, 1.0) * np.linalg.norm(axis))
            pos.append(tree.pos[j] + along + outward)
            normal.append(radial)
            azim.append(az)
            tnode.append(int(i))
    if not pos:
        z = np.zeros((0, 3))
        return LeafCanopy(z, z, np.zeros(0), np.zeros(0, int), *(np.zeros(0) for _ in range(3)))

    L = len(pos)
    area = np.full(L, cp.leaf_area)
    mass = np.full(L, cp.leaf_dry_mass)
    moisture = np.full(L, cp.leaf_moisture)
    char = np.zeros(L)
    return LeafCanopy(np.array(pos), np.array(normal), np.array(azim),
                      np.array(tnode, int), area, mass, moisture, char)


# --------------------------------------------------------------------- canopy diagnostics
def divergence_angles_deg(canopy):
    """Consecutive within-twig leaf azimuth increments, folded to [0,180]° — the phyllotaxis."""
    out = []
    for tn in np.unique(canopy.twig_node):
        a = np.sort(canopy.azimuth[canopy.twig_node == tn])
        if len(a) < 2:
            continue
        d = np.rad2deg(np.diff(a)) % 360.0
        out.extend(np.minimum(d, 360.0 - d).tolist())
    return np.array(out)


def angular_uniformity(canopy):
    """Mean over twigs of the leaf azimuths' angular-gap CV (lower = more even = golden-like).

    Golden-angle whorls spread azimuths evenly (small gap variance); a rational control angle
    overlaps leaves into clumps (large gap variance). This is the falsifiable canopy-quality test.
    """
    cvs = []
    for tn in np.unique(canopy.twig_node):
        a = np.sort(canopy.azimuth[canopy.twig_node == tn] % (2 * np.pi))
        if len(a) < 3:
            continue
        gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
        cvs.append(gaps.std() / (gaps.mean() + 1e-12))
    return float(np.mean(cvs)) if cvs else np.nan


def leaf_area_index(canopy, tree):
    """LAI = total one-sided leaf area / crown ground footprint (xy bbox of the canopy)."""
    up = tree.pos[:, 2] > 0
    xy = tree.pos[up, :2]
    foot = max(float(np.ptp(xy[:, 0])) * float(np.ptp(xy[:, 1])), 1e-6)
    return float(canopy.area.sum() / foot)


def crown_fill(canopy, tree, bands=8):
    """Fraction of canopy height-bands that contain leaves (spatial spread, not a single clump)."""
    up = tree.pos[:, 2] > 0
    zlo, zhi = tree.pos[up, 2].min(), tree.pos[:, 2].max()
    if zhi <= zlo or canopy.n == 0:
        return 0.0
    b = np.clip(((canopy.pos[:, 2] - zlo) / (zhi - zlo) * bands).astype(int), 0, bands - 1)
    return float(len(np.unique(b)) / bands)


if __name__ == "__main__":
    from .growth import grow_tree, GrowthParams
    np.seterr(all="ignore")
    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))

    cp = CanopyParams()
    can = generate_canopy(tree, cp, seed=7)
    print(f"1) canopy: {can.n} leaves on {len(np.unique(can.twig_node))} twigs "
          f"(tree {tree.n} nodes)")
    assert can.n > 200

    div = divergence_angles_deg(can)
    print(f"2) divergence angle: median {np.median(div):.2f}°  (golden {137.5076 % 360:.2f}° "
          f"folded {min(137.5076, 360-137.5076):.2f}°)")
    assert abs(np.median(div) - 137.5076) < 1.0

    u_gold = angular_uniformity(can)
    can_ctrl = generate_canopy(tree, CanopyParams(angle_deg=90.0), seed=7)   # rational control
    u_ctrl = angular_uniformity(can_ctrl)
    print(f"3) angular-gap CV: golden {u_gold:.3f}  vs 90° control {u_ctrl:.3f}  (golden lower = even)")
    assert u_gold < u_ctrl

    lai = leaf_area_index(can, tree); fill = crown_fill(can, tree)
    print(f"4) LAI {lai:.2f} (broadleaf 2–8); crown fill {fill:.2f} (spread through the crown)")
    assert 1.5 <= lai <= 9.0 and fill >= 0.6

    can2 = generate_canopy(tree, cp, seed=7)
    print(f"5) determinism: identical regen = {np.array_equal(can.pos, can2.pos)}")
    assert np.array_equal(can.pos, can2.pos)
    print("\ncanopy self-checks passed.")
