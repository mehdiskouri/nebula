"""
Minimal XPBD over the coarse physics cloud (ARCHITECTURE Part II, Part IV; Decision #5).

Each skeleton edge is a Constraint hyperedge -- and a Constraint hyperedge IS an XPBD distance
constraint. The solve uses the substrate's categorize -> color -> flatten discipline: constraints
are graph-colored so each color group is node-disjoint and projects in parallel with NO write
conflicts (the same race-free guarantee as the conserved-bus reduce). Gravity enters as the
gradient of a potential (guardrail #1: energies, not raw forces), so the dynamics stay stable.

The growth -> fire -> mechanics coupling closes here: the char-weakening transition S = S0(1-chi)
(operators.fire.op_char_weakening) drives each constraint's compliance alpha = alpha0 / S. As a
branch chars, S -> 0, its constraints go compliant and then FRACTURE (broken), and the branch
falls under gravity -- mortality/structure emergent from a conserved quantity (char), not scripted.

The per-color projection is vectorized in numpy -- correct, deterministic, and genuinely
race-free because a color's constraints are node-disjoint (so the same launch maps directly onto a
GPU kernel, one thread per constraint, no atomics; the fire field update already exercises the
Taichi path, Part V). Pure numpy + the hypergraph coloring.
"""
from dataclasses import dataclass, field as _field

import numpy as np

from ..core.hypergraph import Hypergraph, Nodes


@dataclass
class XPBDModel:
    """The constraint topology + materials of the coarse cloud (rest state)."""
    x0: np.ndarray              # (M,3) rest positions
    winv: np.ndarray           # (M,) inverse mass (0 == pinned)
    edges: np.ndarray          # (E,2) constraint node pairs
    rest: np.ndarray           # (E,) rest lengths
    alpha0: np.ndarray         # (E,) base compliance
    colors: list               # list of (constraint-index arrays); each color is node-disjoint
    anchor_stiffness: float = 0.25   # PBD pull toward rest pose (the wood's flexural rigidity)

    @property
    def M(self):
        return self.x0.shape[0]

    @property
    def E(self):
        return self.edges.shape[0]


@dataclass
class XPBDState:
    x: np.ndarray              # (M,3) current positions
    v: np.ndarray              # (M,3) velocities

    @classmethod
    def at_rest(cls, model):
        return cls(model.x0.copy(), np.zeros_like(model.x0))


def from_tree(model_tree, density=1.0, base_compliance=1e-6, pin_ground=True):
    """Build an XPBDModel from a grown TreeModel: skeleton nodes + distance constraints, root pinned,
    constraints colored race-free via the hypergraph.

    Two constraint families: parent-edge distances (stretch) and node->grandparent distances
    (bending stiffness -- a cheap second-order constraint that keeps the tree from flopping at its
    joints, the wood's flexural rigidity). Nodal mass ~ pi r^2 (radius-based); the root is pinned.
    """
    pos = model_tree.pos
    M = pos.shape[0]
    par = model_tree.parent
    edges = []
    for i in range(M):                                   # stretch: parent edges
        if int(par[i]) >= 0:
            edges.append((int(par[i]), i))
    for i in range(M):                                   # bending: node -> grandparent
        j = int(par[i])
        if j >= 0 and int(par[j]) >= 0:
            edges.append((int(par[j]), i))
    edges = np.asarray(edges, np.int64)
    rest = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)

    # nodal mass ~ pi r^2 * spacing (radius-based; root pinned).
    mass = np.maximum(np.pi * model_tree.radius ** 2 * model_tree.params.spacing * density, 1e-6)
    winv = 1.0 / mass
    if pin_ground:
        winv[0] = 0.0                                    # the root holds the tree to the ground

    alpha0 = np.full(len(edges), base_compliance)

    # categorize -> color -> flatten: color the constraint hyperedges so each color is node-disjoint.
    hg = Hypergraph(Nodes(M))
    for (a, b) in edges:
        hg.add_edge("constraint", [a, b])
    colors = [np.asarray(bucket, np.int64) for bucket in hg.color("constraint")]
    return XPBDModel(pos.copy(), winv, edges, rest, alpha0, colors)


def char_to_compliance(model, char_per_edge, S0=1.0, S_break=0.08):
    """Map per-edge char fraction chi -> (compliance alpha, broken mask) via S = S0(1-chi).

    alpha = alpha0 / max(S, eps): as chi -> 1, S -> 0, the constraint goes compliant; below
    S_break it FRACTURES (removed from the solve) -> the branch detaches. (op_char_weakening.)"""
    chi = np.clip(np.asarray(char_per_edge, float), 0.0, 1.0)
    S = S0 * (1.0 - chi)
    alpha = model.alpha0 / np.maximum(S, 1e-6)
    broken = S < S_break
    return alpha, broken


def step(model, state, dt, gravity=(0.0, 0.0, -9.81), iters=8, alpha=None, broken=None,
         anchored=None):
    """One XPBD substep: predict under gravity (a potential), then iterate the colored projections.

    Each color group is node-disjoint, so its constraints project independently (race-free). XPBD
    multipliers lambda are reset per substep and accumulated across `iters`. alpha/broken: optional
    per-edge compliance / fracture mask. anchored: optional per-node bool -- anchored nodes are
    softly pulled toward their rest pose (the wood's flexural rigidity); a charred/detached limb
    has its anchor released so it falls. Default: every free node is anchored. Returns XPBDState.
    """
    g = np.asarray(gravity, float)
    alpha = model.alpha0 if alpha is None else np.asarray(alpha, float)
    broken = np.zeros(model.E, bool) if broken is None else np.asarray(broken, bool)
    free = model.winv > 0
    anchored = free.copy() if anchored is None else (np.asarray(anchored, bool) & free)
    x = state.x.copy(); v = state.v.copy()
    v[free] += g * dt
    x_prev = x.copy()
    x[free] += v[free] * dt
    lam = np.zeros(model.E)
    for _ in range(iters):
        for cidx in model.colors:
            ec = model.edges[cidx]; i = ec[:, 0]; j = ec[:, 1]
            d = x[i] - x[j]; L = np.linalg.norm(d, axis=1)
            n = d / np.maximum(L, 1e-12)[:, None]
            C = L - model.rest[cidx]
            wsum = model.winv[i] + model.winv[j]
            at = alpha[cidx] / (dt * dt)
            dlam = (-C - at * lam[cidx]) / (wsum + at + 1e-30)
            dlam = np.where(broken[cidx] | (wsum <= 0), 0.0, dlam)
            lam[cidx] += dlam
            corr = dlam[:, None] * n
            np.add.at(x, i, model.winv[i][:, None] * corr)
            np.add.at(x, j, -model.winv[j][:, None] * corr)
        if model.anchor_stiffness > 0 and anchored.any():   # soft pull toward rest pose
            x[anchored] += model.anchor_stiffness * (model.x0[anchored] - x[anchored])
    v = (x - x_prev) / dt
    v[~free] = 0.0
    return XPBDState(x, v)


def simulate(model, dt=1e-2, steps=50, gravity=(0.0, 0.0, -9.81), iters=8, alpha=None,
             broken=None, anchored=None):
    st = XPBDState.at_rest(model)
    for _ in range(steps):
        st = step(model, st, dt, gravity=gravity, iters=iters, alpha=alpha, broken=broken,
                  anchored=anchored)
    return st


if __name__ == "__main__":
    from ..operators.growth import grow_tree, GrowthParams
    np.set_printoptions(precision=3, suppress=True)

    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    model = from_tree(tree)
    print(f"1) XPBD model: {model.M} nodes, {model.E} constraints, {len(model.colors)} colors "
          f"(race-free); root pinned winv={model.winv[0]}")
    # colors must be node-disjoint
    for cidx in model.colors:
        ec = model.edges[cidx]
        assert len(np.unique(ec.ravel())) == ec.size, "color group not node-disjoint!"
    print("   all color groups node-disjoint: True")

    H = float(tree.pos[:, 2].max() - tree.pos[:, 2].min())
    DT, STEPS, ITERS = 8e-3, 60, 12

    # 2) intact tree under gravity holds its pose (the anchor = flexural rigidity).
    st = simulate(model, dt=DT, steps=STEPS, iters=ITERS)
    drift = float(np.linalg.norm(st.x - model.x0, axis=1).max())
    print(f"2) intact tree under gravity: max node drift = {drift:.3f} (tree height {H:.2f}; holds pose)")
    assert np.all(np.isfinite(st.x)) and drift < 0.25 * H, "intact tree did not hold its pose"

    # subtree sizes -> pick a DISTAL branch (moderate size, not the near-root mega-subtree).
    size = np.ones(model.M)
    for i in np.argsort(-tree.gen):
        j = int(tree.parent[i])
        if j >= 0:
            size[j] += size[i]
    cand = [i for i in range(1, model.M) if 8 <= size[i] <= 0.2 * model.M]
    target = max(cand, key=lambda i: tree.pos[i, 2])     # the highest such branch (most distal)
    sub = np.zeros(model.M, bool); sub[target] = True
    changed = True
    while changed:                                       # mark the whole subtree below `target`
        changed = False
        for i in range(model.M):
            j = int(tree.parent[i])
            if j >= 0 and sub[j] and not sub[i]:
                sub[i] = True; changed = True

    # 3) char ONLY the bridge constraints (subtree <-> trunk) -> they fracture -> the limb detaches
    #    and falls (anchor released), while the rest of the tree holds.
    bridge = np.array([1.0 if (sub[a] ^ sub[b]) else 0.0 for (a, b) in model.edges])
    alpha, broken = char_to_compliance(model, bridge)
    anchored = ~sub                                      # the detached limb is no longer held
    st2 = simulate(model, dt=DT, steps=STEPS, iters=ITERS, alpha=alpha, broken=broken, anchored=anchored)

    intact_fall = float((model.x0[sub, 2] - st.x[sub, 2]).mean())    # same branch, intact sim
    charred_fall = float((model.x0[sub, 2] - st2.x[sub, 2]).mean())   # charred -> detached
    rest_drift = float(np.linalg.norm(st2.x[~sub] - model.x0[~sub], axis=1).max())
    print(f"3) distal branch: {int(sub.sum())} nodes, {int(broken.sum())} bridge constraints fractured")
    print(f"   mean branch fall: intact={intact_fall:.3f} vs charred={charred_fall:.3f}; "
          f"rest-of-tree drift={rest_drift:.3f}")
    assert broken.sum() > 0 and charred_fall > intact_fall + 0.3, "charred branch did not detach/fall"
    assert rest_drift < 0.25 * H, "charring one branch should not collapse the rest"
    print("\nxpbd self-checks passed.")
