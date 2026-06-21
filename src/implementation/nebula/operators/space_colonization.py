"""
Continuous-space growth: the architecture's actual decision rule (ARCHITECTURE §III.1).

§III.1 names the growth decision rule as "a seeded, field-biased L-system (branching topology)
COUPLED TO space-colonization (venation/limbs pulled toward light/open space)", sensing light /
moisture / hormone fields, with the tree front = "cambium sheet + apical meristems + ROOT TIPS".
The frozen V1.8 oracle (operators/growth.grow) implemented only a lattice L-system with axis-
aligned ±1 drift and no space colonization -> homogeneous, sharp-angled, rootless, blobby. V1.8
only ever tested determinism + write-back heal, never morphology, which is why that passed.

This module implements the missing half: the **space-colonization** front (Runions et al.) in
CONTINUOUS 3-D, with:
  - a canopy attractor cloud (light / open space) and a root attractor cloud (moisture);
  - apical meristems that grow toward the mean direction of nearby attractors (natural, irregular,
    space-filling branching) with **gravitropism / phototropism** (trunk pulled up) and
    **root gravitropism** (roots pulled down + out) -- so directions are continuous, not lattice;
  - **apical dominance** via the attractor kill radius (a tip consumes nearby attractors);
  - **pipe-model radii** (da Vinci / Murray: r_parent = (Σ r_child^p)^(1/p)) -> thick tapering
    trunk + root flare + slender twigs;
  - **root tips** as the same algorithm one domain over (roots-toward-moisture, §III.7).

Deterministic (attractors + tie-breaks seeded by determinism.rng_from_key). Produces a TreeModel
(operators.growth.TreeModel) consumed by the SDF / fire / restriction / XPBD / render. The verified
grow()/trace/write-back machinery in growth.py is left untouched (V1.8 regression intact).
"""
import numpy as np
from scipy.spatial import cKDTree

from ..core.determinism import rng_from_key


def _sample_ellipsoid(rng, n, center, radii, z_stretch=1.0, zmin=None):
    """n points roughly uniform in an axis-aligned ellipsoid (optionally clipped below zmin)."""
    pts = []
    c = np.asarray(center, float); rad = np.asarray(radii, float)
    while len(pts) < n:
        u = rng.uniform(-1, 1, size=(2 * n, 3))
        u = u[(u ** 2).sum(1) <= 1.0]
        q = c[None, :] + u * rad[None, :]
        q[:, 2] = c[2] + (q[:, 2] - c[2]) * z_stretch
        if zmin is not None:
            q = q[q[:, 2] >= zmin]
        pts.extend(q.tolist())
    return np.array(pts[:n])


def _colonize(start, attractors, tropism, gp, max_nodes):
    """Space-colonization from `start` toward `attractors`, biased by `tropism`. Returns
    (pos (N,3), parent (N,), gen (N,)). Each iteration: assign attractors to the nearest node
    within the influence radius; each such node grows one child toward the mean attractor
    direction + tropism; attractors within the kill radius of the new front are removed."""
    H = gp.sc_height
    D = gp.sc_step * H
    d_inf = gp.sc_influence * D
    d_kill = gp.sc_kill * D
    trop = np.asarray(tropism, float)

    pos = [np.asarray(start, float)]
    parent = [-1]
    gen = [0]
    attr = np.asarray(attractors, float).copy()

    for it in range(gp.sc_max_iter):
        if len(attr) == 0 or len(pos) >= max_nodes:
            break
        P = np.asarray(pos)
        kd = cKDTree(P)
        dist, idx = kd.query(attr)                       # nearest existing node per attractor
        within = dist < d_inf
        if not within.any():
            break
        # group attractors by the node they pull
        groups = {}
        for ai in np.where(within)[0]:
            groups.setdefault(int(idx[ai]), []).append(ai)
        new_pos, new_parent, new_gen = [], [], []
        for node_i, alist in groups.items():
            base = P[node_i]
            d = attr[alist] - base
            nrm = np.linalg.norm(d, axis=1, keepdims=True)
            d = d / np.maximum(nrm, 1e-9)
            g = d.sum(0)
            g = g / (np.linalg.norm(g) + 1e-9) + trop    # + gravitropism / phototropism
            g = g / (np.linalg.norm(g) + 1e-9)
            new_pos.append(base + D * g)
            new_parent.append(node_i)
            new_gen.append(it + 1)
        if not new_pos:
            break
        base_n = len(pos)
        pos.extend(new_pos); parent.extend(new_parent); gen.extend(new_gen)
        # kill attractors reached by any node (apical dominance / consumption)
        kd2 = cKDTree(np.asarray(pos))
        dk, _ = kd2.query(attr)
        attr = attr[dk > d_kill]
    return np.asarray(pos), np.asarray(parent, np.int64), np.asarray(gen, np.int64)


def _assign_orders(pos, parent):
    """Branch order via apical dominance: each node's straightest child continues the order; the
    others increment it (so trunk=0, main limbs=1, ...)."""
    n = len(pos)
    children = [[] for _ in range(n)]
    for i in range(n):
        if parent[i] >= 0:
            children[parent[i]].append(i)
    order = np.zeros(n, np.int64)
    # incoming direction per node
    indir = np.zeros((n, 3))
    for i in range(n):
        if parent[i] >= 0:
            v = pos[i] - pos[parent[i]]
            indir[i] = v / (np.linalg.norm(v) + 1e-9)
    from collections import deque
    dq = deque([0])
    while dq:
        u = dq.popleft()
        kids = children[u]
        if kids:
            ud = indir[u] if parent[u] >= 0 else np.array([0, 0, 1.0])
            algn = []
            for c in kids:
                v = pos[c] - pos[u]; v = v / (np.linalg.norm(v) + 1e-9)
                algn.append(float(v @ ud))
            main = kids[int(np.argmax(algn))]
            for c in kids:
                order[c] = order[u] + (0 if c == main else 1)
                dq.append(c)
    return order


def _pipe_radii(pos, parent, gp):
    """Pipe-model radii (postorder): leaves get r_leaf, r_parent = (Σ r_child^p)^(1/p). Gives a
    thick tapering trunk + root flare (node 0 supports both canopy and roots) and slender twigs."""
    n = len(pos)
    children = [[] for _ in range(n)]
    for i in range(n):
        if parent[i] >= 0:
            children[parent[i]].append(i)
    order_by_depth = np.argsort(-_depth(pos, parent, children))  # deepest first ~ postorder
    r = np.full(n, gp.r_leaf, float)
    p = gp.pipe_exp
    for i in order_by_depth:
        if children[i]:
            r[i] = (sum(r[c] ** p for c in children[i])) ** (1.0 / p)
    return np.maximum(r, gp.r_leaf)


def _depth(pos, parent, children):
    """Topological depth from root (BFS) -- used to order the pipe-model postorder."""
    n = len(pos)
    dep = np.zeros(n, np.int64)
    from collections import deque
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for c in children[u]:
            dep[c] = dep[u] + 1; dq.append(c)
    return dep


def grow_tree_sc(seed=0, age=None, gp=None):
    """Grow a realistic tree by space colonization -> a growth.TreeModel.

    Deterministic in (seed, params, age). `age` (<= gp.max_gen) scales the tree size by limiting
    colonization iterations (a younger, smaller tree); default = full size.
    """
    from .growth import GrowthParams, TreeModel, op_reaction_wood
    gp = gp or GrowthParams()
    H = gp.sc_height
    frac = 1.0 if age is None else float(np.clip(age / gp.max_gen, 0.25, 1.0))

    rng = rng_from_key("sc", seed, gp.sc_n_canopy, gp.sc_n_roots, round(frac, 4))
    crown_c = np.array([0.0, 0.0, gp.sc_crown_cz * H * (0.7 + 0.3 * frac)])
    crown_r = gp.sc_crown_r * H * (0.6 + 0.4 * frac)
    canopy = _sample_ellipsoid(rng, gp.sc_n_canopy, crown_c,
                               [crown_r, crown_r, crown_r], z_stretch=gp.sc_crown_stretch,
                               zmin=gp.sc_trunk_h * H * 0.6)
    roots = _sample_ellipsoid(rng, gp.sc_n_roots,
                              [0.0, 0.0, -gp.sc_root_depth * H * 0.5],
                              [gp.sc_root_r * H, gp.sc_root_r * H, gp.sc_root_depth * H * 0.5],
                              z_stretch=1.0, zmin=None)
    roots = roots[roots[:, 2] < -1e-3]

    gp_age = gp if age is None else _scaled(gp, frac)
    cpos, cpar, cgen = _colonize(np.zeros(3), canopy, [0, 0, gp.sc_up_tropism], gp_age,
                                 gp.sc_max_nodes)
    # roots: separate colonization from origin, pulled down (and outward by the attractor spread)
    rpos, rpar, rgen = _colonize(np.zeros(3), roots, [0, 0, -gp.sc_root_tropism], gp_age,
                                 gp.sc_max_nodes // 3)

    # merge (origin shared at index 0); offset root indices, remap root-of-roots parent to origin
    off = len(cpos)
    rpar2 = rpar.copy()
    keep = np.arange(1, len(rpos))                       # drop the duplicate origin
    remap = {0: 0}
    for new_i, old_i in enumerate(keep, start=off):
        remap[old_i] = new_i
    rpar_merged = np.array([0 if rpar[o] == 0 else remap[rpar[o]] for o in keep], np.int64)
    pos = np.vstack([cpos, rpos[keep]])
    parent = np.concatenate([cpar, rpar_merged])
    gen = np.concatenate([cgen, rgen[keep] + cgen.max() + 1])  # roots are "later" generations

    order = _assign_orders(pos, parent)
    radius = _pipe_radii(pos, parent, gp)
    r_bark = np.maximum(radius - gp.bark_thickness, 0.25 * radius)
    # heartwood: the oldest wood (low gen, near trunk) is heartwood; tips are sapwood.
    gmax = max(int(gen.max()), 1)
    heart = gp.heart_frac * np.clip(1.0 - gen / gmax - 0.15, 0.0, 1.0)
    r_heart = np.minimum(radius * heart, r_bark)
    reaction, _ = op_reaction_wood(pos, parent, gen, gp)  # Wolff stress proxy (material tag only)

    return TreeModel(seed=seed, age=(gp.max_gen if age is None else age), params=gp,
                     pos=pos, parent=parent, order=order, gen=gen, radius=radius,
                     r_heart=r_heart, r_bark=r_bark, reaction=reaction)


def _scaled(gp, frac):
    """A copy of gp with the iteration budget scaled by `frac` (younger tree)."""
    import dataclasses
    return dataclasses.replace(gp, sc_max_iter=max(int(gp.sc_max_iter * frac), 30))


if __name__ == "__main__":
    from .growth import GrowthParams
    t = grow_tree_sc(seed=7, gp=GrowthParams())
    H = float(np.ptp(t.pos[:, 2]))
    canopy = (t.pos[:, 2] > 0).sum(); roots = (t.pos[:, 2] < 0).sum()
    print(f"space-colonization tree: {t.n} nodes ({canopy} canopy, {roots} root), height {H:.2f}")
    print(f"  trunk radius {t.radius[0]:.3f}, tip {t.radius.min():.3f}, max {t.radius.max():.3f}")
    print(f"  branch orders 0..{t.order.max()}; segments {len(t.segments())}")
    assert t.n > 200 and canopy > 100 and roots > 20 and t.order.max() >= 3
    assert t.radius.argmax() == 0 or t.radius[0] > t.radius.mean() * 3   # trunk base thickest
    print("space colonization self-checks passed.")
